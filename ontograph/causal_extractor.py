"""LLM-powered causal claim extraction.

Stage A: Identify causal claims — decompose compound statements into
         atomic (source, mechanism, target) triples
Stage B: Assess evidence quality — score mechanism specificity,
         evidence adequacy, counterfactual presence
"""

from __future__ import annotations

from ontograph.models import Provenance
from ontograph.causal_models import (
    CausalClaim,
    CausalMechanism,
    CausalCondition,
    CausalGraph,
    EVIDENCE_TAXONOMY,
    make_claim_id,
)
from ontograph.causal_scoring import compute_confidence
from ontograph.llm_extractor import _chunk_text
from ontograph.llm_client import LLMClient
from ontograph.parsers import ParsedDocument, Section


# ---------------------------------------------------------------------------
# Stage A: Causal claim identification
# ---------------------------------------------------------------------------

CAUSAL_CLAIM_SYSTEM = """You are an expert at identifying causal claims in economic and financial text. You decompose compound causal statements into atomic triples and classify each claim's evidence type and assertiveness. Always respond with valid JSON only."""

CAUSAL_CLAIM_PROMPT = """Identify all causal claims in the text below. For EACH causal claim:

1. DECOMPOSE compound statements into atomic triples: (cause_entity, mechanism, effect_entity)
   Example: "The oil embargo raised inflation, depressed output, and triggered a banking crisis"
   = THREE separate claims: oil→inflation, oil→output, oil→banking_crisis

2. CLASSIFY the evidence type (pick the most specific that applies):
   accounting_identity, institutional_mechanism, meta_analysis, rct,
   natural_experiment, synthetic_control, iv, did, rdd, event_study,
   historical_precedent, panel_fixed_effects, structural_estimation,
   reduced_form_regression, granger_causality, model_simulation,
   market_implied, narrative

3. ASSESS assertiveness from the language:
   - "definitional": "must", "by definition", "is equal to"
   - "strong": "causes", "leads to", "has been shown to"
   - "moderate": "tends to", "contributes to", "is likely to"
   - "hedged": "may", "might", "is associated with"

4. IDENTIFY the mechanism: HOW does the cause produce the effect?
   Through what channel, process, or institution?

5. DETERMINE the direction: "positive", "negative", "ambiguous"

6. NOTE any conditions: Under what circumstances does this hold?

7. NOTE any time lag if mentioned.

Document: {document_name}
Section: {section_heading}

Text:
{section_text}

Respond ONLY with JSON:
{{"claims": [
  {{
    "source": "cause entity name",
    "target": "effect entity name",
    "mechanism_name": "short_snake_case_channel_name",
    "mechanism_description": "how the cause produces the effect",
    "direction": "positive|negative|ambiguous",
    "evidence_type": "from the list above",
    "evidence_directness": "primary|cited|secondary_citation",
    "assertiveness": "definitional|strong|moderate|hedged",
    "conditions": [{{"variable": "...", "operator": "...", "threshold": "...", "description": "..."}}],
    "time_lag_min": "e.g. 1 month or null",
    "time_lag_max": "e.g. 6 months or null",
    "claim_text": "verbatim sentence making the assertion",
    "counterfactual_stated": true/false
  }}
]}}"""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_claims_from_section(
    client: LLMClient,
    section: Section,
    doc_name: str,
) -> list[dict]:
    """Stage A: extract raw causal claims from a single section."""
    raw_claims = []
    for chunk in _chunk_text(section.text):
        prompt = CAUSAL_CLAIM_PROMPT.format(
            document_name=doc_name,
            section_heading=section.heading,
            section_text=chunk,
        )
        try:
            result = client.chat_json([
                {"role": "system", "content": CAUSAL_CLAIM_SYSTEM},
                {"role": "user", "content": prompt},
            ])
        except (ValueError, Exception):
            continue

        for cdata in result.get("claims", []):
            source = cdata.get("source", "").strip()
            target = cdata.get("target", "").strip()
            if not source or not target or source == target:
                continue
            cdata["_document"] = doc_name
            cdata["_section"] = section.heading
            cdata["_page"] = section.page
            raw_claims.append(cdata)

    return raw_claims


def _build_causal_claim(raw: dict, doc_name: str) -> CausalClaim:
    """Construct a CausalClaim from raw LLM extraction output.

    Applies defaults for missing fields and validates evidence_type.
    """
    source = raw.get("source", "Unknown").strip()
    target = raw.get("target", "Unknown").strip()
    claim_text = raw.get("claim_text", "")

    # Validate evidence type
    evidence_type = raw.get("evidence_type", "narrative")
    if evidence_type not in EVIDENCE_TAXONOMY:
        evidence_type = "narrative"

    # Build mechanism
    conditions = []
    for cond_data in raw.get("conditions", []):
        if isinstance(cond_data, dict) and cond_data.get("variable"):
            conditions.append(CausalCondition(
                variable=cond_data["variable"],
                operator=cond_data.get("operator", "regime_is"),
                threshold=str(cond_data.get("threshold", "")),
                description=cond_data.get("description", ""),
            ))

    mechanism = CausalMechanism(
        name=raw.get("mechanism_name", "unspecified"),
        description=raw.get("mechanism_description", ""),
        direction=raw.get("direction", "ambiguous"),
        time_lag_min=raw.get("time_lag_min"),
        time_lag_max=raw.get("time_lag_max"),
        conditions=conditions,
    )

    # Build provenance
    prov = Provenance(
        document=raw.get("_document", doc_name),
        section=raw.get("_section", ""),
        page=raw.get("_page"),
        passage=claim_text[:300],
    )

    assertiveness = raw.get("assertiveness", "moderate")
    if assertiveness not in ("definitional", "strong", "moderate", "hedged"):
        assertiveness = "moderate"

    directness = raw.get("evidence_directness", "cited")
    if directness not in ("primary", "cited", "secondary_citation"):
        directness = "cited"

    direction = raw.get("direction", "ambiguous")
    if direction not in ("positive", "negative", "ambiguous", "state_dependent"):
        direction = "ambiguous"

    claim = CausalClaim(
        id=make_claim_id(source, target, evidence_type, claim_text),
        source=source,
        target=target,
        mechanisms=[mechanism],
        evidence_type=evidence_type,
        evidence_count=1,
        evidence_directness=directness,
        claim_assertiveness=assertiveness,
        net_direction=direction,
        counterfactual_stated=bool(raw.get("counterfactual_stated", False)),
        sources=[prov],
        claim_text=claim_text,
        extracted_by="llm",
    )

    return claim


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def causal_extract_from_document(
    doc: ParsedDocument,
    client: LLMClient | None = None,
    target_context: dict[str, str] | None = None,
) -> CausalGraph:
    """Extract causal claims from a document and build a CausalGraph.

    Args:
        doc: Parsed document with sections
        client: LLM client (defaults to Anthropic)
        target_context: Geographic/temporal context for confidence scoring

    Returns:
        CausalGraph with extracted and scored claims
    """
    if client is None:
        client = LLMClient(mode="anthropic")

    # Stage A: Extract raw claims from each section
    all_raw: list[dict] = []
    for section in doc.sections:
        if len(section.text.strip()) < 20:
            continue
        section_claims = _extract_claims_from_section(client, section, doc.source_path)
        all_raw.extend(section_claims)

    # Build CausalClaim objects and compute confidence
    cg = CausalGraph(
        source_knowledge_graph=doc.source_path,
    )

    for raw in all_raw:
        claim = _build_causal_claim(raw, doc.source_path)
        compute_confidence(claim, target_context)
        cg.add_claim(claim)

    return cg
