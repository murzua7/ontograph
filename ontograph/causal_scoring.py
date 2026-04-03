"""Confidence scoring for causal claims.

Composite score: 0.30 * evidence + 0.25 * external_validity
               + 0.20 * mechanism + 0.15 * evidence_diversity
               + 0.10 * assertiveness

Plus a temporal precedence gate (hard constraint, not a weight).
"""

from __future__ import annotations

from math import log
from typing import Any

from ontograph.causal_models import (
    CausalClaim,
    EVIDENCE_TAXONOMY,
    ASSERTIVENESS_SCORES,
)

# Weights — recalibratable via backtesting (spec Section 4.1)
W_EVIDENCE = 0.30
W_EXTERNAL_VALIDITY = 0.25
W_MECHANISM = 0.20
W_EVIDENCE_DIVERSITY = 0.15
W_ASSERTIVENESS = 0.10


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def evidence_base_score(evidence_type: str) -> float:
    """Lookup base score from the evidence taxonomy."""
    return EVIDENCE_TAXONOMY.get(evidence_type, 0.15)


def external_validity_score(
    claim: CausalClaim,
    target_context: dict[str, str] | None = None,
) -> float:
    """Score how well the evidence context matches the application context.

    V1 heuristic: compares geographic_scope strings.
    Full version (spec Section 4.3) requires structured regime metadata.
    """
    if target_context is None:
        return 0.50  # neutral when no target context provided

    target_country = target_context.get("country", "")
    target_decade = target_context.get("decade", "")
    target_economy_type = target_context.get("economy_type", "")

    scope = claim.geographic_scope.lower()
    geo_entities = [g.lower() for g in claim.geographic_entities]

    # Same country match
    if target_country and target_country.lower() in geo_entities:
        score = 1.0 if target_decade else 0.80
    elif scope == "global" or scope == "oecd":
        # Broad scope — decent applicability if same economy type
        if target_economy_type and target_economy_type.lower() in scope:
            score = 0.70
        else:
            score = 0.50
    elif target_economy_type:
        score = 0.40
    else:
        score = 0.30

    return min(1.0, score)


def mechanism_score(claim: CausalClaim) -> float:
    """Score mechanism specificity (spec Section 4.5).

    5-tier: quantified (1.0), articulated (0.8), vague (0.5),
    implied (0.3), none (0.0).
    """
    if not claim.mechanisms:
        return 0.0

    scores = []
    for mech in claim.mechanisms:
        if mech.elasticity_range is not None or mech.elasticity_source:
            scores.append(1.0)  # quantified
        elif len(mech.description) > 30:
            scores.append(0.8)  # articulated
        elif mech.description:
            scores.append(0.5)  # vague
        elif mech.name:
            scores.append(0.3)  # implied
        else:
            scores.append(0.0)

    return max(scores)  # best mechanism determines score


def evidence_diversity_score(claim: CausalClaim) -> float:
    """Score based on independent evidence count (fallback formula).

    Full version (spec Section 4.4) uses method/dataset diversity.
    V1 uses: min(1.0, log(1 + n) / log(6)), saturates at ~5.
    """
    n = max(1, claim.evidence_count)
    return min(1.0, log(1 + n) / log(6))


def assertiveness_score(claim: CausalClaim) -> float:
    """Lookup score from claim assertiveness level."""
    return ASSERTIVENESS_SCORES.get(claim.claim_assertiveness, 0.20)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_confidence(
    claim: CausalClaim,
    target_context: dict[str, str] | None = None,
) -> float:
    """Compute composite confidence score for a causal claim.

    Returns a float in [0, 1]. Also sets claim.confidence in-place.
    """
    e = evidence_base_score(claim.evidence_type)
    v = external_validity_score(claim, target_context)
    m = mechanism_score(claim)
    d = evidence_diversity_score(claim)
    a = assertiveness_score(claim)

    score = (
        W_EVIDENCE * e
        + W_EXTERNAL_VALIDITY * v
        + W_MECHANISM * m
        + W_EVIDENCE_DIVERSITY * d
        + W_ASSERTIVENESS * a
    )
    claim.confidence = round(score, 4)
    claim.strength = classify_strength(claim.confidence)
    return claim.confidence


# ---------------------------------------------------------------------------
# Temporal gate
# ---------------------------------------------------------------------------

def apply_temporal_gate(
    claim: CausalClaim,
    temporal_precedence: str = "unknown",
) -> CausalClaim:
    """Cap strength at 'correlation' if temporal precedence is violated.

    temporal_precedence: "established", "unknown", "reversed"
    """
    if temporal_precedence == "reversed":
        claim.strength = "correlation"
        claim.confidence = min(claim.confidence, 0.33)
    elif temporal_precedence == "unknown" and claim.causal_type == "macro_statistical":
        claim.confidence *= 0.7
        claim.strength = classify_strength(claim.confidence)
    return claim


# ---------------------------------------------------------------------------
# Strength classification
# ---------------------------------------------------------------------------

def classify_strength(
    confidence: float,
    thresholds: tuple[float, float] = (0.35, 0.65),
) -> str:
    """Bin continuous confidence into three strength categories."""
    if confidence < thresholds[0]:
        return "correlation"
    if confidence < thresholds[1]:
        return "weak_causal"
    return "strong_causal"
