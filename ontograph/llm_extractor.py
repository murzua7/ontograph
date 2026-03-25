"""LLM-powered entity and relation extraction.

Two-pass architecture:
  Pass 1: Extract entities from each section (grounded on schema types)
  Pass 2: Extract relations between known entities (grounded on schema relations)

This produces ~3-5x more entities and ~10x more relations than heuristic
extraction, with provenance and confidence scores.
"""

from __future__ import annotations

import json
from ontograph.models import Entity, Relation, Provenance
from ontograph.schema import OntologySchema
from ontograph.parsers import ParsedDocument, Section
from ontograph.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ENTITY_SYSTEM = """You are an expert knowledge graph builder. You extract structured entities from academic text according to a given ontology schema. Always respond with valid JSON only."""

ENTITY_PROMPT = """Given this ontology schema with the following entity types:

{schema_types}

Extract all entities from the text below. For each entity, provide:
- "name": canonical name (short, specific, no articles)
- "type": one of the entity types listed above
- "aliases": other names or abbreviations used for this entity in the text
- "observations": 2-3 factual statements about this entity from the text
- "passage": the exact sentence or phrase that mentions this entity

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON in this format:
{{"entities": [{{"name": "...", "type": "...", "aliases": ["..."], "observations": ["..."], "passage": "..."}}]}}"""


RELATION_SYSTEM = """You are an expert knowledge graph builder. You identify typed relationships between known entities from academic text according to a given ontology schema. Always respond with valid JSON only."""

RELATION_PROMPT = """Given these entities already extracted from the document:

{entity_list}

And this ontology schema for relations:

{schema_relations}

Extract all relationships between these entities from the text below.
For each relation, provide:
- "source": exact entity name from the list above
- "target": exact entity name from the list above
- "type": one of the relation types listed above
- "passage": the exact sentence that justifies this relation
- "confidence": 0.0 to 1.0 (how certain you are)

IMPORTANT: source and target MUST be exact names from the entity list.

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON in this format:
{{"relations": [{{"source": "...", "target": "...", "type": "...", "passage": "...", "confidence": 0.8}}]}}"""


# ---------------------------------------------------------------------------
# Schema formatting helpers
# ---------------------------------------------------------------------------

def _format_entity_types(schema: OntologySchema) -> str:
    lines = []
    for name, edef in schema.entity_types.items():
        subtypes = f" (subtypes: {', '.join(edef.subtypes)})" if edef.subtypes else ""
        lines.append(f"- {name}: {edef.description}{subtypes}")
    return "\n".join(lines)


def _format_relation_types(schema: OntologySchema) -> str:
    lines = []
    for name, rdef in schema.relation_types.items():
        domain = f" [from: {', '.join(rdef.domain)}]" if rdef.domain else ""
        rng = f" [to: {', '.join(rdef.range)}]" if rdef.range else ""
        lines.append(f"- {name}: {rdef.description}{domain}{rng}")
    return "\n".join(lines)


def _format_entity_list(entities: list[Entity]) -> str:
    lines = []
    for e in entities:
        lines.append(f"- {e.name} (type: {e.entity_type})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 2500, overlap: int = 500) -> list[str]:
    """Split text into overlapping chunks for LLM context window."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 1
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Pass 1: Entity extraction
# ---------------------------------------------------------------------------

def _extract_entities_from_section(
    client: LLMClient,
    section: Section,
    doc_name: str,
    schema: OntologySchema,
) -> list[Entity]:
    """Extract entities from a single section using LLM."""
    entities = []
    chunks = _chunk_text(section.text)

    for chunk in chunks:
        prompt = ENTITY_PROMPT.format(
            schema_types=_format_entity_types(schema),
            document_name=doc_name,
            section_heading=section.heading,
            section_text=chunk,
        )

        messages = [
            {"role": "system", "content": ENTITY_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            result = client.chat_json(messages)
        except (ValueError, Exception) as e:
            continue  # skip on parse error

        for edata in result.get("entities", []):
            name = edata.get("name", "").strip()
            etype = edata.get("type", "").strip()
            if not name or not etype:
                continue
            # Validate type against schema
            if not schema.validate_entity_type(etype):
                # Try to find closest match
                for st in schema.entity_type_names:
                    if etype.lower() in st.lower() or st.lower() in etype.lower():
                        etype = st
                        break
                else:
                    etype = schema.entity_type_names[0]

            prov = Provenance(
                document=doc_name,
                section=section.heading,
                page=section.page,
                passage=edata.get("passage", "")[:300],
            )

            entities.append(Entity(
                name=name,
                entity_type=etype,
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=[prov],
            ))

    return entities


# ---------------------------------------------------------------------------
# Pass 2: Relation extraction (grounded on known entities)
# ---------------------------------------------------------------------------

def _extract_relations_from_section(
    client: LLMClient,
    section: Section,
    doc_name: str,
    entities: list[Entity],
    schema: OntologySchema,
) -> list[Relation]:
    """Extract relations between known entities from a section."""
    if not entities:
        return []

    relations = []
    entity_names = {e.name for e in entities}
    chunks = _chunk_text(section.text)

    for chunk in chunks:
        prompt = RELATION_PROMPT.format(
            entity_list=_format_entity_list(entities),
            schema_relations=_format_relation_types(schema),
            document_name=doc_name,
            section_heading=section.heading,
            section_text=chunk,
        )

        messages = [
            {"role": "system", "content": RELATION_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            result = client.chat_json(messages)
        except (ValueError, Exception):
            continue

        for rdata in result.get("relations", []):
            source = rdata.get("source", "").strip()
            target = rdata.get("target", "").strip()
            rtype = rdata.get("type", "").strip()

            # Validate: source and target must be known entities
            if source not in entity_names or target not in entity_names:
                continue
            if source == target:
                continue
            if rtype not in schema.relation_types:
                # Find closest
                for rt in schema.relation_type_names:
                    if rtype.lower() in rt.lower() or rt.lower() in rtype.lower():
                        rtype = rt
                        break
                else:
                    rtype = schema.relation_type_names[0]

            prov = Provenance(
                document=doc_name,
                section=section.heading,
                page=section.page,
                passage=rdata.get("passage", "")[:300],
            )

            relations.append(Relation(
                source=source,
                target=target,
                relation_type=rtype,
                weight=float(rdata.get("confidence", 0.8)),
                provenance=[prov],
            ))

    return relations


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_entities(entities: list[Entity]) -> list[Entity]:
    """Merge entities with the same name (case-insensitive)."""
    merged: dict[str, Entity] = {}
    for e in entities:
        key = e.name.lower()
        if key in merged:
            existing = merged[key]
            existing.aliases = list(set(existing.aliases + e.aliases))
            existing.observations = list(set(existing.observations + e.observations))
            existing.provenance.extend(e.provenance)
        else:
            merged[key] = Entity(
                name=e.name,
                entity_type=e.entity_type,
                aliases=list(e.aliases),
                observations=list(e.observations),
                provenance=list(e.provenance),
                metadata=dict(e.metadata),
            )
    return list(merged.values())


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def llm_extract_from_document(
    doc: ParsedDocument,
    schema: OntologySchema,
    client: LLMClient | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Two-pass LLM extraction: entities first, then grounded relations.

    Args:
        doc: Parsed document with sections
        schema: Ontology schema defining entity/relation types
        client: LLM client (defaults to Anthropic if not provided)

    Returns:
        (entities, relations) with provenance
    """
    if client is None:
        client = LLMClient(mode="anthropic")

    # Pass 1: Extract entities from each section
    all_entities: list[Entity] = []
    for section in doc.sections:
        if len(section.text.strip()) < 20:
            continue
        section_entities = _extract_entities_from_section(
            client, section, doc.source_path, schema,
        )
        all_entities.extend(section_entities)

    # Deduplicate across sections
    entities = _dedup_entities(all_entities)

    # Pass 2: Extract relations grounded on known entities
    all_relations: list[Relation] = []
    for section in doc.sections:
        if len(section.text.strip()) < 20:
            continue
        section_relations = _extract_relations_from_section(
            client, section, doc.source_path, entities, schema,
        )
        all_relations.extend(section_relations)

    return entities, all_relations
