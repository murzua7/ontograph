"""Phase D — LLM × AST agreement-set merger (spec §4.1.3).

Distinct from `ontograph/resolver.py`, which deduplicates entities within a
single extraction. This module reconciles *two* extractions — prose-from-LLM
and code-from-AST — and surfaces the three disciplinary buckets:

    agreement : both sides saw it                 → enters the default KG
    llm_only  : prose said yes, code said no      → audit target
    ast_only  : code said yes, prose said no      → audit target

A `sign_conflicts` list pairs up mechanism edges whose signs disagree on
`+` vs `-` (unknown never conflicts with anything).

Abstract LLM entities that map to concrete AST entities (same name, different
type) are bound via an emitted `implements` relation; both entities are kept
(they denote different things — theory vs. implementation) but the pairing
itself counts as agreement.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .models import Entity, KnowledgeGraph, Relation


# ── Default fuzzy name matcher ──────────────────────────────────────────

def default_name_matcher(a: str, b: str) -> float:
    """Lowercase + space/underscore-normalised similarity in [0, 1]."""
    def _norm(s: str) -> str:
        return s.lower().replace("_", " ").strip()
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ── Report ──────────────────────────────────────────────────────────────

@dataclass
class MergeReport:
    agreement_entities: list[Entity] = field(default_factory=list)
    llm_only_entities: list[Entity] = field(default_factory=list)
    ast_only_entities: list[Entity] = field(default_factory=list)
    agreement_relations: list[Relation] = field(default_factory=list)
    llm_only_relations: list[Relation] = field(default_factory=list)
    ast_only_relations: list[Relation] = field(default_factory=list)
    sign_conflicts: list[tuple[Relation, Relation]] = field(default_factory=list)

    def to_json(self) -> str:
        def _ent_list(items: list[Entity]) -> list[dict]:
            return [asdict(e) for e in items]

        def _rel_list(items: list[Relation]) -> list[dict]:
            return [asdict(r) for r in items]

        payload = {
            "agreement_entities": _ent_list(self.agreement_entities),
            "llm_only_entities": _ent_list(self.llm_only_entities),
            "ast_only_entities": _ent_list(self.ast_only_entities),
            "agreement_relations": _rel_list(self.agreement_relations),
            "llm_only_relations": _rel_list(self.llm_only_relations),
            "ast_only_relations": _rel_list(self.ast_only_relations),
            "sign_conflicts": [
                [asdict(pair[0]), asdict(pair[1])] for pair in self.sign_conflicts
            ],
        }
        return json.dumps(payload, indent=2, default=str)


# ── Entity merge helper ─────────────────────────────────────────────────

def _merge_entity_pair(primary: Entity, secondary: Entity) -> Entity:
    """Combine a matched pair into a single canonical entity.

    `primary` supplies identity (name, type); `secondary` contributes
    aliases, anchors, observations, provenance. When the AST half of the
    pair is concrete and the LLM half abstract, call with primary=ast.
    """
    merged = Entity(
        name=primary.name,
        entity_type=primary.entity_type,
        aliases=list(primary.aliases),
        observations=list(primary.observations),
        provenance=list(primary.provenance),
        metadata=dict(primary.metadata),
        code_anchors=list(primary.code_anchors),
        citation_anchors=list(primary.citation_anchors),
        abstract=primary.abstract and secondary.abstract,
        abstract_rationale=primary.abstract_rationale or secondary.abstract_rationale,
        confidence=max(primary.confidence, secondary.confidence),
    )
    # Union secondary contributions, deduplicating by identity tuples.
    if secondary.name not in merged.aliases and secondary.name != merged.name:
        merged.aliases.append(secondary.name)
    for alias in secondary.aliases:
        if alias not in merged.aliases and alias != merged.name:
            merged.aliases.append(alias)
    for obs in secondary.observations:
        if obs not in merged.observations:
            merged.observations.append(obs)
    merged.provenance.extend(secondary.provenance)

    existing_codes = {(a.repo, a.path, a.line, a.symbol) for a in merged.code_anchors}
    for a in secondary.code_anchors:
        if (a.repo, a.path, a.line, a.symbol) not in existing_codes:
            merged.code_anchors.append(a)

    existing_cites = {(c.key, c.pages) for c in merged.citation_anchors}
    for c in secondary.citation_anchors:
        if (c.key, c.pages) not in existing_cites:
            merged.citation_anchors.append(c)

    return merged


# ── Entity matcher ──────────────────────────────────────────────────────

def _find_match(
    candidate: Entity,
    others: list[Entity],
    name_matcher: Callable[[str, str], float],
    threshold: float,
    used: set[int],
    *,
    allow_abstract_cross_type: bool,
) -> int | None:
    """Return an index into `others` that best matches `candidate`, or None.

    `allow_abstract_cross_type=True` relaxes the type-equality rule when
    `candidate.abstract` is True (spec §4.1.3: theory↔implementation binding).
    """
    best_idx: int | None = None
    best_score = threshold
    for i, other in enumerate(others):
        if i in used:
            continue
        score = name_matcher(candidate.name, other.name)
        if score < best_score:
            continue
        type_match = candidate.entity_type == other.entity_type
        if not type_match and not (allow_abstract_cross_type and candidate.abstract):
            continue
        if score > best_score or best_idx is None:
            best_idx = i
            best_score = score
    return best_idx


# ── Relation key ────────────────────────────────────────────────────────

_PAIRED_SIGNS = {"+", "-"}


def _rel_key(rel: Relation, name_map: dict[str, str]) -> tuple[str, str, str]:
    src = name_map.get(rel.source, rel.source)
    tgt = name_map.get(rel.target, rel.target)
    return (src, tgt, rel.relation_type)


# ── Public entry point ──────────────────────────────────────────────────

def merge_extractions(
    llm_entities: list[Entity],
    llm_relations: list[Relation],
    ast_entities: list[Entity],
    ast_relations: list[Relation],
    *,
    name_matcher: Callable[[str, str], float] = default_name_matcher,
    threshold: float = 0.85,
) -> tuple[KnowledgeGraph, MergeReport]:
    """Merge LLM and AST extractions into an agreement-gated KnowledgeGraph.

    Returns (kg, report). The KG contains only agreement entities + agreement
    relations; the report preserves all disagreements for audit.
    """
    report = MergeReport()

    # ── Entity reconciliation ────────────────────────────────────────
    # name_map: LLM-name → canonical name in merged KG (so relation endpoints
    # can be re-keyed after agreement).
    name_map_llm: dict[str, str] = {}
    name_map_ast: dict[str, str] = {}
    used_ast: set[int] = set()

    # Pass 1: same-type agreement (dedup).
    for llm_ent in llm_entities:
        idx = _find_match(
            llm_ent, ast_entities, name_matcher, threshold, used_ast,
            allow_abstract_cross_type=False,
        )
        if idx is None:
            continue
        ast_ent = ast_entities[idx]
        used_ast.add(idx)
        # AST half carries code grounding → use it as primary.
        canonical = _merge_entity_pair(ast_ent, llm_ent)
        report.agreement_entities.append(canonical)
        name_map_llm[llm_ent.name] = canonical.name
        name_map_ast[ast_ent.name] = canonical.name

    # Pass 2: abstract-LLM ↔ concrete-AST binding (no dedup; `implements`
    # relation emitted). Only LLM entities not already matched in pass 1.
    matched_llm_names = set(name_map_llm.keys())
    for llm_ent in llm_entities:
        if llm_ent.name in matched_llm_names or not llm_ent.abstract:
            continue
        idx = _find_match(
            llm_ent, ast_entities, name_matcher, threshold, used_ast,
            allow_abstract_cross_type=True,
        )
        if idx is None:
            continue
        ast_ent = ast_entities[idx]
        used_ast.add(idx)
        # Both entities enter the agreement bucket as-is.
        report.agreement_entities.append(llm_ent)
        report.agreement_entities.append(ast_ent)
        name_map_llm[llm_ent.name] = llm_ent.name
        name_map_ast[ast_ent.name] = ast_ent.name
        # Implements relation: theory → mechanism binding.
        report.agreement_relations.append(Relation(
            source=llm_ent.name,
            target=ast_ent.name,
            relation_type="implements",
            edge_class="identity",
            sign="unknown",
            lag="unknown",
        ))

    # Disagreement buckets.
    matched_llm = set(name_map_llm.keys())
    for llm_ent in llm_entities:
        if llm_ent.name not in matched_llm:
            report.llm_only_entities.append(llm_ent)
    for i, ast_ent in enumerate(ast_entities):
        if i not in used_ast:
            report.ast_only_entities.append(ast_ent)

    # ── Relation reconciliation ──────────────────────────────────────
    llm_rel_index: dict[tuple[str, str, str], list[Relation]] = {}
    for rel in llm_relations:
        llm_rel_index.setdefault(_rel_key(rel, name_map_llm), []).append(rel)

    ast_rel_index: dict[tuple[str, str, str], list[Relation]] = {}
    for rel in ast_relations:
        ast_rel_index.setdefault(_rel_key(rel, name_map_ast), []).append(rel)

    agreement_entity_names = {e.name for e in report.agreement_entities}

    consumed_llm_keys: set[tuple[str, str, str]] = set()
    consumed_ast_keys: set[tuple[str, str, str]] = set()

    for key, llm_rels in llm_rel_index.items():
        if key not in ast_rel_index:
            continue
        src, tgt, _ = key
        if src not in agreement_entity_names or tgt not in agreement_entity_names:
            # Endpoints must themselves be agreed — otherwise the "agreement"
            # relation would dangle outside the emitted KG.
            continue
        ast_rels = ast_rel_index[key]
        consumed_llm_keys.add(key)
        consumed_ast_keys.add(key)
        # Take the first LLM relation as the agreement relation; record
        # sign conflicts across any pair of paired-sign disagreements.
        canonical = llm_rels[0]
        canonical = Relation(
            source=src, target=tgt, relation_type=canonical.relation_type,
            weight=canonical.weight,
            provenance=list(canonical.provenance),
            metadata=dict(canonical.metadata),
            sign=canonical.sign,
            lag=canonical.lag,
            form=canonical.form,
            conditional_on=list(canonical.conditional_on),
            edge_class=canonical.edge_class,
            confidence=canonical.confidence,
        )
        report.agreement_relations.append(canonical)
        for llm_r in llm_rels:
            for ast_r in ast_rels:
                if (llm_r.sign in _PAIRED_SIGNS and ast_r.sign in _PAIRED_SIGNS
                        and llm_r.sign != ast_r.sign):
                    report.sign_conflicts.append((llm_r, ast_r))

    for key, rels in llm_rel_index.items():
        if key in consumed_llm_keys:
            continue
        report.llm_only_relations.extend(rels)
    for key, rels in ast_rel_index.items():
        if key in consumed_ast_keys:
            continue
        report.ast_only_relations.extend(rels)

    # ── Build output KG ──────────────────────────────────────────────
    kg = KnowledgeGraph()
    for ent in report.agreement_entities:
        kg.add_entity(ent)
    for rel in report.agreement_relations:
        kg.add_relation(rel)

    return kg, report
