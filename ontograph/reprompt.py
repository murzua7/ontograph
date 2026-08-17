"""Targeted sign re-prompt for unknown-sign relations (#45).

Motivation: the bulk-extraction pass emits `sign="unknown"` whenever the
passage-level signal isn't strong enough during schema-free relation
extraction. A focused second pass — one prompt per unknown-sign edge,
anchored on the relation's own provenance passage — recovers many of
those signs without rerunning the whole extractor.

Public API:
- `build_sign_reprompt_messages(relation)` — LLM messages for one relation.
- `targeted_reprompt_unknown_signs(kg, client, *, min_confidence=0.6)` —
  walks the KG, prompts once per unknown-sign relation with provenance,
  mutates the KG in place, returns a structured report.

This module intentionally never paid-API-routes: the caller supplies the
`LLMClient`, which in production is a `claude-code` client hitting the
subscription.
"""
from __future__ import annotations

from typing import Any

from ontograph.llm_client import LLMClient
from ontograph.models import KnowledgeGraph, Relation

# Mirrors `_VALID_SIGN` in llm_extractor but kept local to avoid leaking
# an import of the extractor just to reach the constant.
_VALID_SIGNS = {"+", "-", "±", "0", "unknown"}

_SYSTEM = (
    "You are a careful analyst. Given a single causal relation and the "
    "passage it was extracted from, determine the sign of the effect. "
    "Respond with a strict JSON object of the form: "
    '{"sign": "+" | "-" | "±" | "0" | "unknown", '
    '"confidence": <float 0..1>, "rationale": "<short>"}. '
    "Use '+' for monotonic increase, '-' for monotonic decrease, "
    "'±' for mixed/non-monotone, '0' for zero/no-effect, "
    "'unknown' when the passage genuinely does not disambiguate."
)

_USER_TEMPLATE = (
    "Relation: `{source}` --[{rel_type}]--> `{target}`\n"
    "Passage:\n---\n{passage}\n---\n"
    "Answer with the JSON object described in the system prompt."
)


def build_sign_reprompt_messages(relation: Relation) -> list[dict[str, str]]:
    """Build the per-relation LLM message list for the sign-reprompt pass."""
    passage = ""
    if relation.provenance:
        passage = (relation.provenance[0].passage or "").strip()
    user = _USER_TEMPLATE.format(
        source=relation.source,
        rel_type=relation.relation_type,
        target=relation.target,
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


def targeted_reprompt_unknown_signs(
    kg: KnowledgeGraph,
    client: LLMClient,
    *,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Walk `kg`, re-prompting the LLM for each unknown-sign relation.

    Mutates `kg` in place (confident answers only). Returns a report
    listing every update and every skip with its reason. Relations
    without provenance are skipped with `reason="no_provenance"` because
    the whole point of this pass is *anchored* disambiguation — without
    a passage the re-prompt is just the original extraction, retried.
    """
    updates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    n_candidates = 0

    for r in kg.relations:
        if r.sign != "unknown":
            continue
        passage = ""
        if r.provenance:
            passage = (r.provenance[0].passage or "").strip()
        if not passage:
            skipped.append({
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "reason": "no_provenance",
            })
            continue
        n_candidates += 1

        messages = build_sign_reprompt_messages(r)
        try:
            payload = client.chat_json(messages)
        except Exception as exc:
            skipped.append({
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "reason": f"llm_error: {type(exc).__name__}",
            })
            continue

        if not isinstance(payload, dict):
            skipped.append({
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
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
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "reason": f"low_confidence: sign={new_sign} conf={conf:.2f}",
            })
            continue

        r.sign = new_sign
        updates.append({
            "source": r.source,
            "target": r.target,
            "relation_type": r.relation_type,
            "new_sign": new_sign,
            "confidence": conf,
            "rationale": (payload.get("rationale") or "")[:200],
        })

    return {
        "n_candidates": n_candidates,
        "n_updated": len(updates),
        "n_unchanged": n_candidates - len(updates),
        "updates": updates,
        "skipped": skipped,
    }
