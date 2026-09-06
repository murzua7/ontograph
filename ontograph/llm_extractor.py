"""LLM-powered entity and relation extraction.

Schema-grounded mode (default):
  Pass 1: Extract entities from each section (grounded on schema types)
  Pass 2: Extract relations between known entities (grounded on schema relations)

Schema-free mode (--schema-free):
  Pass 1: Open entity extraction — LLM invents its own type labels
  Pass 2: Open relation extraction — LLM invents its own relation verb phrases
  Pass 3: Type consolidation — one LLM call merges synonymous labels across
          the whole corpus into a minimal canonical vocabulary
"""

from __future__ import annotations

import json
from collections import Counter
from ontograph.models import DocAnchor, Entity, Relation, Provenance
from ontograph.schema import OntologySchema
from ontograph.parsers import ParsedDocument, Section
from ontograph.llm_client import LLMClient


def _locate_passage(section: Section, passage: str) -> tuple[int, int, int | None]:
    """Locate `passage` inside `section.text`, returning (char_start, char_end, line).

    Used to build `DocAnchor`s: a found passage is a concrete, reproducible
    pointer into the source document. Returns (-1, -1, None) when the
    passage cannot be located (LLM paraphrased, or the chunk boundary
    truncated the quote).

    The returned char offsets are document-level (section.char_offset +
    local offset), so the anchor works against ParsedDocument.full_text.
    """
    if not passage:
        return (-1, -1, None)
    idx = section.text.find(passage)
    if idx < 0:
        # Try first 80 chars — LLM often returns a slightly-trimmed quote.
        head = passage[:80]
        if len(head) < 20:
            return (-1, -1, None)
        idx = section.text.find(head)
        if idx < 0:
            return (-1, -1, None)
    base = section.char_offset if section.char_offset >= 0 else -1
    if base < 0:
        return (-1, -1, None)
    char_start = base + idx
    char_end = char_start + len(passage)
    # Line count: section start line + newlines before `idx` in section.text.
    line: int | None = None
    if section.line is not None:
        line = section.line + section.text.count("\n", 0, idx)
    return (char_start, char_end, line)


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
- "sign": the marginal effect of source on target. One of:
    "+"       : increase in source drives increase in target (monotone positive)
    "-"       : increase in source drives decrease in target (monotone negative)
    "±"       : effect is context-dependent or bidirectional
    "0"       : definitionally zero / no causal effect
    "unknown" : sign cannot be inferred from the passage
- "lag": temporal delay. One of:
    "within_step" : same simulation step / contemporaneous
    "across_step" : next step or later
    "Q+k"         : integer k ≥ 1 quarters delay
    "unknown"     : not specified
- "form": functional form of the dependence. One of:
    "linear"    : additive / proportional
    "monotone"  : same-sign but nonlinear
    "threshold" : switch-like (activates past a cutoff)
    "identity"  : accounting identity / stock-flow constraint
    "unknown"   : not specified
- "edge_class": category of the edge. One of:
    "identity"   : accounting identity or conservation law
    "mechanism"  : causal / behavioural mechanism
    "parameter"  : parameter → variable influence
    "structural" : architectural / compositional wiring
    "abstract"   : high-level conceptual link
- "conditional_on": list of entity names whose state modulates this relation; [] if none

IMPORTANT: source and target MUST be exact names from the entity list.
If the passage does not determine a field, use "unknown" (never guess).

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON in this format:
{{"relations": [{{"source": "...", "target": "...", "type": "...", "passage": "...", "confidence": 0.8, "sign": "+", "lag": "within_step", "form": "linear", "edge_class": "mechanism", "conditional_on": []}}]}}"""


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

            raw_passage = edata.get("passage", "")
            passage = raw_passage[:300]
            prov = Provenance(
                document=doc_name,
                section=section.heading,
                page=section.page,
                passage=passage,
            )
            doc_anchors: list[DocAnchor] = []
            cs, ce, ln = _locate_passage(section, raw_passage)
            if cs >= 0:
                doc_anchors.append(DocAnchor(
                    document=doc_name,
                    char_start=cs,
                    char_end=ce,
                    line=ln,
                    section=section.heading,
                    passage=passage,
                ))

            entities.append(Entity(
                name=name,
                entity_type=etype,
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=[prov],
                doc_anchors=doc_anchors,
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
                sign=_coerce_sign(rdata.get("sign")),
                lag=_coerce_lag(rdata.get("lag")),
                form=_coerce_form(rdata.get("form")),
                edge_class=_coerce_edge_class(rdata.get("edge_class")),
                conditional_on=_coerce_conditional_on(rdata.get("conditional_on"), entity_names),
                confidence=float(rdata.get("confidence", 1.0)),
            ))

    return relations


_VALID_SIGN = {"+", "-", "±", "0", "unknown"}
_VALID_FORM = {"linear", "monotone", "threshold", "identity", "unknown"}
_VALID_EDGE_CLASS = {"identity", "mechanism", "parameter", "structural", "abstract"}


def _coerce_sign(raw) -> str:
    if isinstance(raw, str):
        s = raw.strip()
        if s in _VALID_SIGN:
            return s
        # Normalise common variants
        low = s.lower()
        if low in {"pos", "positive", "plus"}:
            return "+"
        if low in {"neg", "negative", "minus"}:
            return "-"
        if low in {"mixed", "both", "ambiguous", "bidirectional"}:
            return "±"
        if low in {"zero", "none"}:
            return "0"
    return "unknown"


def _coerce_lag(raw) -> str:
    if isinstance(raw, str):
        s = raw.strip()
        if s in {"within_step", "across_step", "unknown"}:
            return s
        # Accept Q+k notation
        if s.startswith("Q+") and s[2:].isdigit():
            return s
        low = s.lower()
        if low in {"contemporaneous", "same_step", "immediate"}:
            return "within_step"
        if low in {"next_step", "one_step", "lagged"}:
            return "across_step"
    return "unknown"


def _coerce_form(raw) -> str:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _VALID_FORM:
            return s
    return "unknown"


def _coerce_edge_class(raw) -> str:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _VALID_EDGE_CLASS:
            return s
    return "mechanism"


def _coerce_conditional_on(raw, entity_names: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, str) and x in entity_names]


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
            existing.aliases = sorted(set(existing.aliases + e.aliases))
            existing.observations = sorted(set(existing.observations + e.observations))
            existing.provenance.extend(e.provenance)
            # Dedupe doc_anchors by (document, char_start, char_end, line)
            seen = {(d.document, d.char_start, d.char_end, d.line)
                    for d in existing.doc_anchors}
            for d in e.doc_anchors:
                key_d = (d.document, d.char_start, d.char_end, d.line)
                if key_d not in seen:
                    existing.doc_anchors.append(d)
                    seen.add(key_d)
        else:
            merged[key] = Entity(
                name=e.name,
                entity_type=e.entity_type,
                aliases=list(e.aliases),
                observations=list(e.observations),
                provenance=list(e.provenance),
                metadata=dict(e.metadata),
                doc_anchors=list(e.doc_anchors),
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


# ---------------------------------------------------------------------------
# Schema-free prompts
# ---------------------------------------------------------------------------

SCHEMA_FREE_ENTITY_SYSTEM = """You are an expert knowledge graph builder. Extract structured entities from text and assign each a descriptive snake_case type label that captures what kind of thing it is. Invent precise, reusable labels — do NOT constrain yourself to any fixed vocabulary. Always respond with valid JSON only."""

SCHEMA_FREE_ENTITY_PROMPT = """Extract all important entities from the text below.

For each entity provide:
- "name": canonical name (short, specific, no leading articles)
- "type": a snake_case label describing what kind of thing this is
  Good examples: state_variable, training_method, agent, implementation_bug,
  evaluation_metric, market_mechanism, accounting_identity, neural_component
  Bad examples: thing, item, concept, other, misc
- "aliases": other names or abbreviations used in the text
- "observations": 2–3 factual statements grounded in the text
- "passage": the exact sentence or phrase where this entity appears

Consistency rule: use the SAME type label for similar things across the document.

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON:
{{"entities": [{{"name": "...", "type": "...", "aliases": ["..."], "observations": ["..."], "passage": "..."}}]}}"""


SCHEMA_FREE_RELATION_SYSTEM = """You are an expert knowledge graph builder. Identify relationships between known entities using precise snake_case verb-phrase type labels. Always respond with valid JSON only."""

SCHEMA_FREE_RELATION_PROMPT = """Given these entities:

{entity_list}

Extract all relationships between them from the text below.
For each relation:
- "source": exact entity name from the list above
- "target": exact entity name from the list above
- "type": a snake_case verb phrase (e.g. causes, constrains, feeds_into,
  breaks, is_applied_to, evaluated_by, implements, propagates_to, measured_by)
- "passage": the exact sentence that justifies this relation
- "confidence": 0.0–1.0
- "sign": "+" (source↑→target↑), "-" (source↑→target↓), "±" (mixed/bidirectional),
  "0" (no causal effect), or "unknown" (passage does not determine it)
- "lag": "within_step" (contemporaneous), "across_step" (next step), "Q+k" (k quarters),
  or "unknown"
- "form": "linear", "monotone", "threshold", "identity", or "unknown"
- "edge_class": "identity" (accounting), "mechanism" (causal), "parameter",
  "structural" (wiring), or "abstract"
- "conditional_on": list of entity names (from the list above) whose state
  modulates this relation; [] if none

IMPORTANT: source and target MUST be exact names from the entity list above.
If the passage does not determine a field, use "unknown" (never guess).

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON:
{{"relations": [{{"source": "...", "target": "...", "type": "...", "passage": "...", "confidence": 0.8, "sign": "+", "lag": "within_step", "form": "linear", "edge_class": "mechanism", "conditional_on": []}}]}}"""


# ---------------------------------------------------------------------------
# Consolidation prompts
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM = """You consolidate raw type labels extracted by an LLM into a clean, minimal ontology vocabulary. Eliminate redundancy and noise while preserving meaningful distinctions. Always respond with valid JSON only."""

CONSOLIDATION_ENTITY_PROMPT = """Below are entity type labels discovered by an LLM reading a corpus.
Each label shows: frequency (how many entities use it) and example entity names of that type.

Your task: consolidate into {target_min}–{target_max} canonical entity types.

Rules (in priority order):
1. MERGE true synonyms and near-synonyms → pick the clearer, more general name
2. MERGE overly-specific labels that are just instances of a broader type
3. MERGE any label appearing ≤2 times — almost always noise or over-specificity
4. KEEP SEPARATE genuinely distinct concepts even if related
   (e.g. "agent" ≠ "mechanism", "parameter" ≠ "state_variable")
5. Every input label MUST appear in the mapping — either mapped to itself or to a canonical type

Raw type labels:
{type_list}

Respond ONLY with JSON:
{{
  "reasoning": "2–3 sentences explaining the key merge decisions",
  "mapping": {{"raw_label": "canonical_type", ...}},
  "vocabulary": [
    {{"type": "canonical_type", "description": "one-sentence definition", "example_members": ["entity1", "entity2", "entity3"]}}
  ]
}}"""

CONSOLIDATION_RELATION_PROMPT = """Below are relation type labels (verb phrases) discovered by an LLM reading a corpus.
Each label shows: frequency and example (source → target) pairs.

Your task: consolidate into {target_min}–{target_max} canonical relation types.

Rules (in priority order):
1. MERGE synonymous verb phrases (e.g. "causes" and "leads_to" → "causes")
2. MERGE any label appearing ≤2 times
3. KEEP SEPARATE relations with genuinely different causal or structural meaning
4. Every input label MUST appear in the mapping

Raw relation labels:
{type_list}

Respond ONLY with JSON:
{{
  "reasoning": "2–3 sentences explaining the key merge decisions",
  "mapping": {{"raw_label": "canonical_relation", ...}},
  "vocabulary": [
    {{"type": "canonical_relation", "description": "one-sentence definition", "example_pairs": ["A → B", "C → D"]}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Schema-free extraction helpers
# ---------------------------------------------------------------------------

def _normalize_type(raw: str) -> str:
    """Normalise a free-form type label to snake_case."""
    return raw.lower().strip().replace(" ", "_").replace("-", "_")


def _schema_free_extract_entities(
    client: LLMClient,
    section: Section,
    doc_name: str,
) -> list[Entity]:
    entities = []
    for chunk in _chunk_text(section.text):
        prompt = SCHEMA_FREE_ENTITY_PROMPT.format(
            document_name=doc_name,
            section_heading=section.heading,
            section_text=chunk,
        )
        try:
            result = client.chat_json([
                {"role": "system", "content": SCHEMA_FREE_ENTITY_SYSTEM},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            continue

        for edata in result.get("entities", []):
            name = edata.get("name", "").strip()
            etype = _normalize_type(edata.get("type", "concept"))
            if not name or not etype:
                continue
            raw_passage = edata.get("passage", "")
            passage = raw_passage[:300]
            prov = Provenance(
                document=doc_name,
                section=section.heading,
                page=section.page,
                passage=passage,
            )
            doc_anchors: list[DocAnchor] = []
            cs, ce, ln = _locate_passage(section, raw_passage)
            if cs >= 0:
                doc_anchors.append(DocAnchor(
                    document=doc_name,
                    char_start=cs,
                    char_end=ce,
                    line=ln,
                    section=section.heading,
                    passage=passage,
                ))
            entities.append(Entity(
                name=name,
                entity_type=etype,
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=[prov],
                doc_anchors=doc_anchors,
            ))
    return entities


def _schema_free_extract_relations(
    client: LLMClient,
    section: Section,
    doc_name: str,
    entities: list[Entity],
) -> list[Relation]:
    if not entities:
        return []
    entity_names = {e.name for e in entities}
    relations = []
    for chunk in _chunk_text(section.text):
        prompt = SCHEMA_FREE_RELATION_PROMPT.format(
            entity_list=_format_entity_list(entities),
            document_name=doc_name,
            section_heading=section.heading,
            section_text=chunk,
        )
        try:
            result = client.chat_json([
                {"role": "system", "content": SCHEMA_FREE_RELATION_SYSTEM},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            continue

        for rdata in result.get("relations", []):
            source = rdata.get("source", "").strip()
            target = rdata.get("target", "").strip()
            rtype = _normalize_type(rdata.get("type", "relates_to"))
            if source not in entity_names or target not in entity_names or source == target:
                continue
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
                sign=_coerce_sign(rdata.get("sign")),
                lag=_coerce_lag(rdata.get("lag")),
                form=_coerce_form(rdata.get("form")),
                edge_class=_coerce_edge_class(rdata.get("edge_class")),
                conditional_on=_coerce_conditional_on(rdata.get("conditional_on"), entity_names),
                confidence=float(rdata.get("confidence", 1.0)),
            ))
    return relations


# ---------------------------------------------------------------------------
# Pass 3: Type consolidation
# ---------------------------------------------------------------------------

def _build_type_list(type_counts: Counter, type_examples: dict[str, list[str]]) -> str:
    """Format type stats for the consolidation prompt."""
    lines = []
    for t, count in type_counts.most_common():
        examples = type_examples.get(t, [])
        ex_str = ", ".join(f'"{x}"' for x in examples[:3])
        rare = "  [RARE — likely noise]" if count <= 2 else ""
        lines.append(f'- "{t}" (count={count}, examples: {ex_str}){rare}')
    return "\n".join(lines)


def consolidate_types(
    entities: list[Entity],
    relations: list[Relation],
    client: LLMClient,
    console=None,
) -> tuple[dict[str, str], dict[str, str], list[dict], list[dict]]:
    """Consolidate raw type labels into a canonical vocabulary.

    Two separate LLM calls: one for entity types, one for relation types.
    Each call sees type labels WITH frequency counts AND example names so the
    LLM has enough semantic signal to make correct merge/keep decisions.

    Returns:
        entity_map:      raw_type -> canonical_type
        relation_map:    raw_type -> canonical_type
        entity_vocab:    list of {type, description, example_members}
        relation_vocab:  list of {type, description, example_pairs}
    """
    # --- Collect entity type stats ---
    entity_type_counts: Counter = Counter(e.entity_type for e in entities)
    entity_type_examples: dict[str, list[str]] = {}
    for e in entities:
        bucket = entity_type_examples.setdefault(e.entity_type, [])
        if len(bucket) < 4:
            bucket.append(e.name)

    n_entity_types = len(entity_type_counts)
    e_target_min = max(4, min(8, n_entity_types // 3))
    e_target_max = max(8, min(15, n_entity_types // 2))

    if console:
        console.print(
            f"  Consolidating {n_entity_types} entity type labels "
            f"→ {e_target_min}–{e_target_max} canonical types..."
        )

    e_result = client.chat_json([
        {"role": "system", "content": CONSOLIDATION_SYSTEM},
        {"role": "user", "content": CONSOLIDATION_ENTITY_PROMPT.format(
            target_min=e_target_min,
            target_max=e_target_max,
            type_list=_build_type_list(entity_type_counts, entity_type_examples),
        )},
    ])

    entity_map: dict[str, str] = e_result.get("mapping", {})
    entity_vocab: list[dict] = e_result.get("vocabulary", [])
    if console and e_result.get("reasoning"):
        console.print(f"  [dim]{e_result['reasoning']}[/dim]")

    # Safety: ensure every raw type is mapped (identity fallback)
    for t in entity_type_counts:
        if t not in entity_map:
            entity_map[t] = t

    # --- Collect relation type stats ---
    relation_type_counts: Counter = Counter(r.relation_type for r in relations)
    relation_type_examples: dict[str, list[str]] = {}
    for r in relations:
        bucket = relation_type_examples.setdefault(r.relation_type, [])
        if len(bucket) < 4:
            bucket.append(f"{r.source} → {r.target}")

    n_relation_types = len(relation_type_counts)
    r_target_min = max(3, min(6, n_relation_types // 3))
    r_target_max = max(6, min(12, n_relation_types // 2))

    if console:
        console.print(
            f"  Consolidating {n_relation_types} relation type labels "
            f"→ {r_target_min}–{r_target_max} canonical types..."
        )

    r_result = client.chat_json([
        {"role": "system", "content": CONSOLIDATION_SYSTEM},
        {"role": "user", "content": CONSOLIDATION_RELATION_PROMPT.format(
            target_min=r_target_min,
            target_max=r_target_max,
            type_list=_build_type_list(relation_type_counts, relation_type_examples),
        )},
    ])

    relation_map: dict[str, str] = r_result.get("mapping", {})
    relation_vocab: list[dict] = r_result.get("vocabulary", [])
    if console and r_result.get("reasoning"):
        console.print(f"  [dim]{r_result['reasoning']}[/dim]")

    for t in relation_type_counts:
        if t not in relation_map:
            relation_map[t] = t

    return entity_map, relation_map, entity_vocab, relation_vocab


def apply_type_consolidation(
    entities: list[Entity],
    relations: list[Relation],
    entity_map: dict[str, str],
    relation_map: dict[str, str],
) -> None:
    """Apply consolidation mapping in-place."""
    for e in entities:
        e.entity_type = entity_map.get(e.entity_type, e.entity_type)
    for r in relations:
        r.relation_type = relation_map.get(r.relation_type, r.relation_type)


# ---------------------------------------------------------------------------
# Schema-free extraction pipeline
# ---------------------------------------------------------------------------

def schema_free_extract_from_document(
    doc: ParsedDocument,
    client: LLMClient,
) -> tuple[list[Entity], list[Relation]]:
    """Passes 1+2: open extraction without schema constraints.

    Types are raw LLM-generated labels. Run consolidate_types() over the
    full corpus afterwards, then apply_type_consolidation() before building
    the graph.
    """
    all_entities: list[Entity] = []
    for section in doc.sections:
        if len(section.text.strip()) < 20:
            continue
        all_entities.extend(
            _schema_free_extract_entities(client, section, doc.source_path)
        )

    entities = _dedup_entities(all_entities)

    all_relations: list[Relation] = []
    for section in doc.sections:
        if len(section.text.strip()) < 20:
            continue
        all_relations.extend(
            _schema_free_extract_relations(client, section, doc.source_path, entities)
        )

    return entities, all_relations
