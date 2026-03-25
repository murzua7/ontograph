"""Cross-document entity resolution.

Three stages:
  1. Exact match: same name + same type -> merge
  2. Fuzzy match: Levenshtein ratio > threshold + same type -> merge
  3. Alias match: entity A's name appears in entity B's aliases -> merge

Optionally uses LLM for disambiguation of ambiguous fuzzy matches.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ontograph.models import Entity, Relation, KnowledgeGraph


def _similarity(a: str, b: str) -> float:
    """Case-insensitive similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _merge_entities(primary: Entity, secondary: Entity) -> Entity:
    """Merge secondary into primary, keeping primary's name."""
    return Entity(
        name=primary.name,
        entity_type=primary.entity_type,
        aliases=list(set(
            primary.aliases + secondary.aliases
            + ([secondary.name] if secondary.name != primary.name else [])
        )),
        observations=list(set(primary.observations + secondary.observations)),
        provenance=primary.provenance + secondary.provenance,
        metadata={**secondary.metadata, **primary.metadata},
    )


def resolve_entities(
    entities: list[Entity],
    threshold: float = 0.85,
) -> tuple[list[Entity], dict[str, str]]:
    """Resolve duplicate entities across documents.

    Args:
        entities: List of entities (potentially from multiple documents)
        threshold: Fuzzy matching threshold (0-1)

    Returns:
        (resolved_entities, rename_map) where rename_map maps old names
        to canonical names for updating relations.
    """
    resolved: list[Entity] = []
    rename_map: dict[str, str] = {}  # old_name -> canonical_name

    for entity in entities:
        matched = False

        for i, existing in enumerate(resolved):
            # Stage 1: Exact match (case-insensitive)
            if entity.name.lower() == existing.name.lower() and entity.entity_type == existing.entity_type:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                matched = True
                break

            # Stage 2: Alias match
            existing_aliases = {a.lower() for a in existing.aliases}
            if entity.name.lower() in existing_aliases:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                matched = True
                break
            entity_aliases = {a.lower() for a in entity.aliases}
            if existing.name.lower() in entity_aliases:
                resolved[i] = _merge_entities(existing, entity)
                rename_map[entity.name] = existing.name
                matched = True
                break

            # Stage 3: Fuzzy match (same type required)
            if entity.entity_type == existing.entity_type:
                sim = _similarity(entity.name, existing.name)
                if sim >= threshold:
                    resolved[i] = _merge_entities(existing, entity)
                    rename_map[entity.name] = existing.name
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

    Also deduplicates relations with same (source, target, type).
    """
    updated = []
    seen = set()

    for rel in relations:
        source = rename_map.get(rel.source, rel.source)
        target = rename_map.get(rel.target, rel.target)
        if source == target:
            continue

        key = (source, target, rel.relation_type)
        if key in seen:
            continue
        seen.add(key)

        updated.append(Relation(
            source=source,
            target=target,
            relation_type=rel.relation_type,
            weight=rel.weight,
            provenance=list(rel.provenance),
            metadata=dict(rel.metadata),
        ))

    return updated


def merge_knowledge_graphs(
    graphs: list[KnowledgeGraph],
    threshold: float = 0.85,
) -> KnowledgeGraph:
    """Merge multiple knowledge graphs with entity resolution.

    Args:
        graphs: List of knowledge graphs (from different documents)
        threshold: Fuzzy matching threshold

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

    resolved_entities, rename_map = resolve_entities(all_entities, threshold)
    resolved_relations = resolve_relations(all_relations, rename_map)

    merged = KnowledgeGraph(
        schema_name=graphs[0].schema_name if graphs else "base",
        documents=list(set(all_docs)),
    )
    for e in resolved_entities:
        merged.add_entity(e)
    for r in resolved_relations:
        merged.add_relation(r)

    return merged
