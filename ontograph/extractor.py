"""Heuristic entity and relation extraction from parsed documents.

Pattern-based extraction that works without LLM calls. Catches ~30-50%
of entities and ~10-20% of relations. Serves as baseline and pre-seeder
for LLM extraction.
"""

from __future__ import annotations

import re
from ontograph.models import Entity, Relation, Provenance
from ontograph.schema import OntologySchema
from ontograph.parsers import ParsedDocument, Section


# Patterns for entity detection
ENTITY_PATTERNS = [
    # "the X model/agent/mechanism/variable/market"
    (r"the\s+(\w[\w\s]{1,40}?)\s+(model|agent|mechanism|variable|market|rule|constraint|parameter|shock|sector|framework|theory)",
     lambda m: (m.group(1).strip(), m.group(2).strip())),
    # "X is defined as"
    (r"(\w[\w\s]{1,40}?)\s+(?:is|are)\s+defined\s+as",
     lambda m: (m.group(1).strip(), "concept")),
    # "we model X as"
    (r"we\s+(?:model|define|represent|treat)\s+(\w[\w\s]{1,40}?)\s+as",
     lambda m: (m.group(1).strip(), "concept")),
]

# Patterns for relation detection
RELATION_PATTERNS = [
    # "X causes/affects/leads to Y"
    (r"(\w[\w\s]{1,30}?)\s+(?:causes?|affects?|leads?\s+to|results?\s+in)\s+(\w[\w\s]{1,30}?)(?:\.|,|;|\s+and\s+|\s+which)",
     "causes"),
    # "X feeds into Y"
    (r"(\w[\w\s]{1,30}?)\s+feeds?\s+into\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "feeds_into"),
    # "X is constrained by Y" / "X constrains Y"
    (r"(\w[\w\s]{1,30}?)\s+(?:is\s+)?constrained\s+by\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "constrains"),
    # "X depends on Y"
    (r"(\w[\w\s]{1,30}?)\s+depends?\s+on\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "feeds_into"),
    # "X propagates to Y"
    (r"(\w[\w\s]{1,30}?)\s+propagates?\s+(?:to|through|via)\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "propagates_losses_to"),
    # "X amplifies Y"
    (r"(\w[\w\s]{1,30}?)\s+amplif(?:y|ies)\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "amplifies"),
    # "X triggers Y"
    (r"(\w[\w\s]{1,30}?)\s+triggers?\s+(\w[\w\s]{1,30}?)(?:\.|,|;)",
     "triggers"),
]


def _clean_name(name: str) -> str:
    """Clean extracted entity name."""
    name = re.sub(r"\s+", " ", name).strip()
    # Remove leading articles
    name = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", name, flags=re.IGNORECASE)
    # Remove trailing verbs/prepositions
    name = re.sub(r"\s+(is|are|was|were|has|have|of|in|on|at|to|for|by|with)$", "", name, flags=re.IGNORECASE)
    return name.strip()


def _map_type_to_schema(raw_type: str, schema: OntologySchema) -> str:
    """Map a raw extracted type label to the closest schema entity type."""
    raw = raw_type.lower()

    # Direct match
    if raw in schema.entity_types:
        return raw

    # Check subtypes
    for etype, edef in schema.entity_types.items():
        if raw in [s.lower() for s in edef.subtypes]:
            return etype
        if raw in edef.description.lower():
            return etype

    # Keyword mapping
    type_map = {
        "model": "theory", "framework": "theory", "theory": "theory",
        "rule": "mechanism", "process": "mechanism", "mechanism": "mechanism",
        "variable": "state_variable", "indicator": "state_variable",
        "market": "market", "sector": "market",
        "agent": "agent", "actor": "agent",
        "constraint": "constraint", "identity": "constraint",
        "parameter": "parameter", "coefficient": "parameter",
        "shock": "shock", "disturbance": "shock",
    }
    for keyword, mapped_type in type_map.items():
        if keyword in raw and mapped_type in schema.entity_types:
            return mapped_type

    # Fallback to first entity type in schema
    return schema.entity_type_names[0] if schema.entity_type_names else "concept"


def extract_entities(doc: ParsedDocument, schema: OntologySchema) -> list[Entity]:
    """Extract entities from a parsed document using regex patterns."""
    entities: dict[str, Entity] = {}

    for section in doc.sections:
        text = section.text
        for pattern, extractor in ENTITY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name, raw_type = extractor(match)
                name = _clean_name(name)
                if len(name) < 2 or len(name) > 50:
                    continue

                entity_type = _map_type_to_schema(raw_type, schema)

                prov = Provenance(
                    document=doc.source_path,
                    section=section.heading,
                    page=section.page,
                    passage=match.group(0)[:200],
                    char_offset=match.start(),
                )

                if name in entities:
                    entities[name].provenance.append(prov)
                else:
                    entities[name] = Entity(
                        name=name,
                        entity_type=entity_type,
                        provenance=[prov],
                    )

    return list(entities.values())


def extract_relations(
    doc: ParsedDocument,
    entities: list[Entity],
    schema: OntologySchema,
) -> list[Relation]:
    """Extract relations between known entities using regex patterns."""
    entity_names = {e.name.lower(): e.name for e in entities}
    relations = []

    for section in doc.sections:
        text = section.text
        for pattern, rel_type in RELATION_PATTERNS:
            if rel_type not in schema.relation_types:
                rel_type = "relates_to" if "relates_to" in schema.relation_types else schema.relation_type_names[0]

            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_raw = _clean_name(match.group(1))
                target_raw = _clean_name(match.group(2))

                # Match to known entities (case-insensitive)
                source = entity_names.get(source_raw.lower())
                target = entity_names.get(target_raw.lower())

                if source and target and source != target:
                    prov = Provenance(
                        document=doc.source_path,
                        section=section.heading,
                        page=section.page,
                        passage=match.group(0)[:200],
                        char_offset=match.start(),
                    )
                    relations.append(Relation(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        weight=0.5,  # heuristic confidence
                        provenance=[prov],
                    ))

    return relations


def extract_from_document(
    doc: ParsedDocument,
    schema: OntologySchema,
) -> tuple[list[Entity], list[Relation]]:
    """Full heuristic extraction pipeline."""
    entities = extract_entities(doc, schema)
    relations = extract_relations(doc, entities, schema)
    return entities, relations
