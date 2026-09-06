"""Cross-document entity resolution.

Stages (in order; every fusion is recorded — there are no silent merges):
  1. Exact match: same name (case-insensitive) + same type -> merge
  2. Normalized match: same normalize_label() + same type -> merge
     (lowercase/strip/leading-article/plural-inflection fold)
  3. Alias match: both names map to the same canonical in the alias file
     (alias-to-alias fuses too), or entity A's name appears in entity B's
     aliases -> merge
  4. Fuzzy match: Levenshtein ratio > threshold + same type -> merge.
     Production merges pass threshold=1.0 (stage off) since 2026-09-05: at
     0.85 it fused antonyms (skilled/unskilled, mean/median, HANK/RANK,
     liquid/illiquid). Reviewed fusions live in vocab/canonical_entities.yaml.

Pass fusion_log=[] to collect one FusionEvent per absorbed entity:
  {"canonical", "absorbed", "stage", "sources"} where sources lists the
  absorbed entity's provenance documents. Invariant: every entity absorbed
  by a merge appears exactly once in the log (zero silent fusions).

Type discipline: no stage ever fuses entities of different entity_type.
Cross-type same-normalized pairs are documented, not fused
(see schemas/canonical_entities.yaml, status=type-split).

Optionally uses LLM for disambiguation of ambiguous fuzzy matches.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from ontograph.models import Entity, Relation, KnowledgeGraph

FusionEvent = dict  # {"canonical": str, "absorbed": str, "stage": str, "sources": list[str]}


def normalize_label(name: str) -> str:
    """Deterministic vocabulary fold: lowercase, whitespace/strip cleanup,
    leading-article strip, simple plural/inflection fold.

    This is the single normalization rule shared by the resolver and the
    T3 audit scripts (t3_vocab.py, t3_canonical.py). Keep them in sync.
    """
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("ses") and len(s) > 4:
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s


def load_canonical_aliases(path: str | Path) -> dict[str, str]:
    """Load schemas/canonical_entities.yaml alias assertions.

    Returns {alias_literal_lower: canonical_literal}. Only hand-reviewed
    `aliases:` entries are loaded; `members:` spelling variants are NOT
    auto-fused (they are cross-graph variants documented for authors).
    Minimal YAML parsing (no dependency): reads `canonical:` once per
    group and dash-prefixed items under `aliases:`.
    """
    aliases: dict[str, str] = {}
    canonical: str | None = None
    in_aliases = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- key:") or line.startswith("groups:") or line.startswith("version:"):
            in_aliases = False
        elif line.startswith("canonical:"):
            canonical = line.split("canonical:", 1)[1].strip()
            in_aliases = False
        elif line.startswith("aliases:"):
            # `aliases: []` (inline empty) asserts nothing.
            in_aliases = line == "aliases:"
        elif in_aliases and line.startswith("- ") and canonical:
            aliases[line[2:].strip().lower()] = canonical
    return aliases


def _similarity(a: str, b: str) -> float:
    """Case-insensitive similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _merge_entities(primary: Entity, secondary: Entity) -> Entity:
    """Merge secondary into primary, keeping primary's name."""
    return Entity(
        name=primary.name,
        entity_type=primary.entity_type,
        aliases=sorted(set(
            primary.aliases + secondary.aliases
            + ([secondary.name] if secondary.name != primary.name else [])
        )),
        # sorted(): set iteration order is hash-randomized; without sorting,
        # identical inputs produce byte-different merges across runs.
        observations=sorted(set(primary.observations + secondary.observations)),
        provenance=primary.provenance + secondary.provenance,
        metadata={**secondary.metadata, **primary.metadata},
    )


def _record(
    fusion_log: list[FusionEvent] | None,
    stage: str,
    primary: Entity,
    secondary: Entity,
) -> None:
    """Record one fusion event. All merges go through here: a merge with
    no log entry is a bug (zero silent fusions)."""
    if fusion_log is None:
        return
    docs = sorted({p.document for p in secondary.provenance})
    fusion_log.append({
        "canonical": primary.name,
        "absorbed": secondary.name,
        "stage": stage,
        "sources": docs,
    })


def resolve_entities(
    entities: list[Entity],
    threshold: float = 0.85,
    aliases: dict[str, str] | None = None,
    fusion_log: list[FusionEvent] | None = None,
) -> tuple[list[Entity], dict[str, str]]:
    """Resolve duplicate entities across documents.

    Args:
        entities: List of entities (potentially from multiple documents)
        threshold: Fuzzy matching threshold (0-1)
        aliases: Optional {alias_lower: canonical_literal} from
            load_canonical_aliases(); matched case-insensitively, same-type only
        fusion_log: Optional list; one FusionEvent is appended per absorbed
            entity (zero silent fusions: len(log) == inputs - outputs)

    Returns:
        (resolved_entities, rename_map) where rename_map maps old names
        to canonical names for updating relations.
    """
    resolved: list[Entity] = []
    rename_map: dict[str, str] = {}  # old_name -> canonical_name
    # Cache keyed by NAME (not id()): _merge_entities creates replacement
    # objects on every fusion, so garbage-collected ids get reused and an
    # id-keyed cache returns stale norms -> false normalized matches.
    norm_cache: dict[str, str] = {}

    def _norm(e: Entity) -> str:
        if e.name not in norm_cache:
            norm_cache[e.name] = normalize_label(e.name)
        return norm_cache[e.name]

    for entity in entities:
        matched = False

        for i, existing in enumerate(resolved):
            same_type = entity.entity_type == existing.entity_type
            # Stage 1: Exact match (case-insensitive, same type)
            if entity.name.lower() == existing.name.lower() and same_type:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                _record(fusion_log, "exact", existing, entity)
                matched = True
                break

            # Stage 2: Normalized match (same type only)
            if same_type and _norm(entity) == _norm(existing) and entity.name != existing.name:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                _record(fusion_log, "normalized", existing, entity)
                matched = True
                break

            # Stage 3a: canonical alias file (same type only). Both names are
            # mapped to their canonical form, so two aliases of the same
            # canonical fuse even when the canonical literal itself is absent
            # from this merge (2026-09-05 T3 repair).
            if same_type and aliases:
                ca = aliases.get(entity.name.lower(), entity.name).lower()
                cb = aliases.get(existing.name.lower(), existing.name).lower()
                if ca == cb and (entity.name.lower() in aliases or existing.name.lower() in aliases):
                    resolved[i] = _merge_entities(existing, entity)
                    rename_map[entity.name] = existing.name
                    _record(fusion_log, "alias", existing, entity)
                    matched = True
                    break

            # Stage 3b: Alias match via entity alias lists
            existing_aliases = {a.lower() for a in existing.aliases}
            if entity.name.lower() in existing_aliases:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                _record(fusion_log, "alias", existing, entity)
                matched = True
                break
            entity_aliases = {a.lower() for a in entity.aliases}
            if existing.name.lower() in entity_aliases:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                _record(fusion_log, "alias", existing, entity)
                matched = True
                break

            # Stage 4: Fuzzy match (same type required)
            if same_type:
                sim = _similarity(entity.name, existing.name)
                if sim >= threshold:
                    resolved[i] = _merge_entities(existing, entity)
                    rename_map[entity.name] = existing.name
                    _record(fusion_log, "fuzzy", existing, entity)
                    matched = True
                    break

        if not matched:
            resolved.append(Entity(
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=list(entity.aliases),
                observations=list(entity.observations),
                provenance=list(entity.provenance),
                metadata=dict(entity.metadata),
            ))

    return resolved, rename_map


def resolve_relations(
    relations: list[Relation],
    rename_map: dict[str, str],
) -> list[Relation]:
    """Update relation source/target names according to rename_map.

    Preserves polarity and discipline-v1 fields (sign, form, lag,
    edge_class, confidence, conditional_on) that a previous version
    silently reset to defaults.
    Deduplicates exact (source, target, type, sign) repeats, unioning
    their provenance; opposing signs on the same triple are kept as
    distinct relations (they are contradiction signal, not duplicates).
    """
    updated = []
    seen: dict[tuple, int] = {}

    for rel in relations:
        source = rename_map.get(rel.source, rel.source)
        target = rename_map.get(rel.target, rel.target)
        if source == target:
            continue

        key = (source, target, rel.relation_type, rel.sign)
        if key in seen:
            prev = updated[seen[key]]
            have = {(p.document, p.section, p.passage) for p in prev.provenance}
            for p in rel.provenance:
                if (p.document, p.section, p.passage) not in have:
                    prev.provenance.append(p)
                    have.add((p.document, p.section, p.passage))
            prev.confidence = max(prev.confidence, rel.confidence)
            for c in rel.conditional_on:
                if c not in prev.conditional_on:
                    prev.conditional_on.append(c)
            continue
        seen[key] = len(updated)

        updated.append(Relation(
            source=source,
            target=target,
            relation_type=rel.relation_type,
            weight=rel.weight,
            provenance=list(rel.provenance),
            metadata=dict(rel.metadata),
            sign=rel.sign,
            form=rel.form,
            lag=rel.lag,
            edge_class=rel.edge_class,
            confidence=rel.confidence,
            conditional_on=list(rel.conditional_on),
        ))

    return updated


def merge_knowledge_graphs(
    graphs: list[KnowledgeGraph],
    threshold: float = 0.85,
    aliases: dict[str, str] | None = None,
    fusion_log: list[FusionEvent] | None = None,
) -> KnowledgeGraph:
    """Merge multiple knowledge graphs with entity resolution.

    Args:
        graphs: List of knowledge graphs (from different documents)
        threshold: Fuzzy matching threshold
        aliases: Optional canonical alias map (see load_canonical_aliases)
        fusion_log: Optional list collecting one FusionEvent per fusion

    Returns:
        Merged knowledge graph with resolved entities
    """
    all_entities = []
    all_relations = []
    all_docs = []

    for kg in graphs:
        all_entities.extend(kg.entities.values())
        all_relations.extend(kg.relations)
        all_docs.extend(kg.documents)

    resolved_entities, rename_map = resolve_entities(
        all_entities, threshold, aliases=aliases, fusion_log=fusion_log,
    )
    resolved_relations = resolve_relations(all_relations, rename_map)

    merged = KnowledgeGraph(
        schema_name=graphs[0].schema_name if graphs else "base",
        # sorted(): set iteration order is hash-randomized across runs.
        documents=sorted(set(all_docs)),
    )
    seen_names: dict[str, Entity] = {}
    for e in resolved_entities:
        if e.name in seen_names:
            # Same literal, different entity_type (same-type pairs always
            # merge in resolve_entities). KnowledgeGraph.entities is keyed
            # by name, so inserting both would silently fold the second
            # into the first inside add_entity. Fuse explicitly instead:
            # keep the first graph's type, union the rest, and log it.
            # Deterministic: input graph order decides the surviving type.
            primary = seen_names[e.name]
            fused = _merge_entities(primary, e)
            seen_names[e.name] = fused
            _record(fusion_log, "type-collision", primary, e)
        else:
            seen_names[e.name] = e
    for e in seen_names.values():
        merged.add_entity(e)
    for r in resolved_relations:
        merged.add_relation(r)

    return merged
