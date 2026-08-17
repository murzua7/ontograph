"""Falsifier-as-training-signal re-extraction (#49).

When the simulator-in-the-loop falsifier's `flipped` bucket says
``KG sign + vs simulator sign -`` for an edge, that disagreement is a
*prior* — the edge is suspicious enough to merit a targeted LLM
re-prompt. The passage remains ground truth (spec §2); the simulator
prior only *selects which edges to re-examine*.

Public API:
- ``build_prior_reprompt_messages(relation, sim_sign, sim_lag)`` — messages
  for one edge, showing both the KG sign and the simulator's contrary
  sign so the LLM can arbitrate against the passage.
- ``apply_falsifier_prior(kg, report, client, min_confidence=0.6)`` —
  walks the `flipped` bucket of a `FalsifyReport`, re-prompts once per
  edge, mutates `kg` in place when the LLM responds confidently.

This module is a sibling of `reprompt.py` (which handles `unknown`
signs without a simulator prior). Both share the same invariants:
anchored disambiguation only — no passage, no re-prompt.
"""
from __future__ import annotations

from typing import Any

from ontograph.falsify import FalsifyReport
from ontograph.llm_client import LLMClient
from ontograph.models import KnowledgeGraph, Relation

_VALID_SIGNS = {"+", "-", "±", "0", "unknown"}

_SYSTEM = (
    "You re-evaluate a single causal relation's sign given three inputs: "
    "(a) the source passage, (b) the sign originally extracted from that "
    "passage, and (c) a prior from independent simulator evidence that "
    "disagrees with (b). The passage is ground truth — the simulator prior "
    "only flags edges worth re-examining. Use the passage to decide whether "
    "to keep the original sign, adopt the simulator's sign, switch to "
    "'±' (ambiguous), '0' (no effect), or 'unknown' (passage truly silent). "
    "Respond with strict JSON: "
    '{"sign": "+" | "-" | "±" | "0" | "unknown", '
    '"confidence": <float 0..1>, "rationale": "<short>"}.'
)

_USER_TEMPLATE = (
    "Relation: `{source}` --[{rel_type}]--> `{target}`\n"
    "Original extracted sign: {kg_sign}\n"
    "Simulator prior: {sim_sign} (lag: {sim_lag})\n"
    "Passage:\n---\n{passage}\n---\n"
    "Answer with the JSON object described in the system prompt."
)


def build_prior_reprompt_messages(
    relation: Relation,
    *,
    sim_sign: str,
    sim_lag: str,
) -> list[dict[str, str]]:
    """Build the per-edge LLM message list for simulator-prior re-prompting."""
    passage = ""
    if relation.provenance:
        passage = (relation.provenance[0].passage or "").strip()
    user = _USER_TEMPLATE.format(
        source=relation.source,
        rel_type=relation.relation_type,
        target=relation.target,
        kg_sign=relation.sign,
        sim_sign=sim_sign,
        sim_lag=sim_lag,
        passage=passage,
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _coerce_sign(raw: Any) -> str:
    if isinstance(raw, str) and raw in _VALID_SIGNS:
        return raw
    return "unknown"


def _find_relation(
    kg: KnowledgeGraph,
    source: str,
    target: str,
    relation_type: str,
) -> Relation | None:
    for r in kg.relations:
        if (r.source == source and r.target == target
                and r.relation_type == relation_type):
            return r
    return None


def apply_falsifier_prior(
    kg: KnowledgeGraph,
    report: FalsifyReport,
    client: LLMClient,
    *,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Walk `report.flipped` and re-prompt each contested edge.

    Mutates `kg` in place when the LLM returns a confidently-different
    sign. Returns a structured report:

        n_candidates : edges re-prompted (had provenance + were in KG)
        n_updated    : sign was revised
        n_defended   : LLM kept the original KG sign with high confidence
        updates      : list of {source, target, new_sign, confidence, rationale}
        skipped      : list of {source, target, reason}

    Only the `flipped` bucket is processed. `confirmed` edges are by
    definition not suspicious; `unknown` edges are handled by
    `reprompt.targeted_reprompt_unknown_signs`.
    """
    updates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    n_candidates = 0
    n_defended = 0

    for entry in report.flipped:
        source = entry["source"]
        target = entry["target"]
        rel_type = entry["relation_type"]
        sim_sign = entry["sim_sign"]
        sim_lag = entry.get("sim_lag", "unknown")

        r = _find_relation(kg, source, target, rel_type)
        if r is None:
            skipped.append({
                "source": source, "target": target,
                "relation_type": rel_type,
                "reason": "not_in_kg",
            })
            continue

        passage = ""
        if r.provenance:
            passage = (r.provenance[0].passage or "").strip()
        if not passage:
            skipped.append({
                "source": source, "target": target,
                "relation_type": rel_type,
                "reason": "no_provenance",
            })
            continue

        n_candidates += 1
        original_sign = r.sign
        messages = build_prior_reprompt_messages(
            r, sim_sign=sim_sign, sim_lag=sim_lag,
        )
        try:
            payload = client.chat_json(messages)
        except Exception as exc:
            skipped.append({
                "source": source, "target": target,
                "relation_type": rel_type,
                "reason": f"llm_error: {type(exc).__name__}",
            })
            continue

        if not isinstance(payload, dict):
            skipped.append({
                "source": source, "target": target,
                "relation_type": rel_type,
                "reason": "malformed_response",
            })
            continue

        new_sign = _coerce_sign(payload.get("sign"))
        try:
            conf = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        if new_sign == "unknown" or conf < min_confidence:
            skipped.append({
                "source": source, "target": target,
                "relation_type": rel_type,
                "reason": f"low_confidence: sign={new_sign} conf={conf:.2f}",
            })
            continue

        if new_sign == original_sign:
            # LLM defended the original sign against the simulator prior —
            # a meaningful outcome: passage trumps simulator.
            n_defended += 1
            continue

        r.sign = new_sign
        updates.append({
            "source": source, "target": target,
            "relation_type": rel_type,
            "original_sign": original_sign,
            "sim_sign": sim_sign,
            "new_sign": new_sign,
            "confidence": conf,
            "rationale": (payload.get("rationale") or "")[:200],
        })

    return {
        "n_candidates": n_candidates,
        "n_updated": len(updates),
        "n_defended": n_defended,
        "updates": updates,
        "skipped": skipped,
    }
