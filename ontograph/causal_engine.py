"""Cascade simulation engine for causal graphs.

Max-confidence priority traversal with:
- Direction propagation (sign algebra)
- Condition checking against regime context
- Sign conflict detection
- Time-stepped feedback mode
"""

from __future__ import annotations

import heapq
import re
from dataclasses import field

from ontograph.causal_models import (
    CausalGraph,
    CausalClaim,
    CausalMechanism,
    CausalCondition,
    CascadeEffect,
    Shock,
)


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

# Approximate conversion to days for comparison
_TIME_UNITS: dict[str, float] = {
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 30.0,
    "months": 30.0,
    "quarter": 90.0,
    "quarters": 90.0,
    "year": 365.0,
    "years": 365.0,
}

_TIME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)")


def parse_time_duration(s: str | None) -> float:
    """Convert a human-readable duration string to approximate days.

    "3 months" → 90.0, "1 week" → 7.0, "immediate" → 0.0
    Returns 0.0 for None, "immediate", or unparseable strings.
    """
    if s is None or s.strip().lower() in ("", "immediate", "unknown"):
        return 0.0
    m = _TIME_PATTERN.search(s.lower())
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        return n * _TIME_UNITS.get(unit, 30.0)
    return 0.0


def min_time(a: str, b: str) -> str:
    """Return the earlier of two time horizon strings."""
    da, db = parse_time_duration(a), parse_time_duration(b)
    return a if da <= db else b


def max_time(a: str, b: str) -> str:
    """Return the later of two time horizon strings."""
    da, db = parse_time_duration(a), parse_time_duration(b)
    return a if da >= db else b


def add_time(base: str, lag: str) -> str:
    """Add a time lag to a base time horizon. Returns a descriptive string."""
    db = parse_time_duration(base)
    dl = parse_time_duration(lag)
    total = db + dl
    if total == 0:
        return "immediate"
    # Convert back to human-readable
    if total <= 7:
        return f"{int(total)} days" if total > 1 else "1 day"
    if total <= 60:
        weeks = round(total / 7)
        return f"{weeks} weeks" if weeks > 1 else "1 week"
    if total <= 300:
        months = round(total / 30)
        return f"{months} months" if months > 1 else "1 month"
    if total <= 540:
        quarters = round(total / 90)
        return f"{quarters} quarters" if quarters > 1 else "1 quarter"
    years = round(total / 365, 1)
    if years == int(years):
        years = int(years)
    return f"{years} years" if years != 1 else "1 year"


# ---------------------------------------------------------------------------
# Direction propagation
# ---------------------------------------------------------------------------

def propagate_direction(
    incoming: str,
    mechanism_direction: str,
) -> str:
    """Propagate directional sign through a mechanism.

    Same sign → positive compound; opposite → negative.
    Ambiguous or state_dependent contaminates the result.
    """
    if incoming == "ambiguous" or mechanism_direction == "ambiguous":
        return "ambiguous"
    if incoming == "state_dependent" or mechanism_direction == "state_dependent":
        return "state_dependent"
    if incoming == mechanism_direction:
        return "positive"
    return "negative"


# ---------------------------------------------------------------------------
# Condition checking
# ---------------------------------------------------------------------------

def _evaluate_condition(
    condition: CausalCondition,
    regime_context: dict[str, str],
) -> bool:
    """Evaluate a single condition against regime context.

    If the condition's variable is not in regime_context, returns True
    (permissive: cascade runs with partial information).
    """
    value_str = regime_context.get(condition.variable)
    if value_str is None:
        return True  # permissive default

    op = condition.operator
    threshold = condition.threshold

    if op == "regime_is":
        return value_str.lower() == threshold.lower()

    if op == "==":
        return value_str == threshold

    # Parse the value as numeric (needed for >, <, >=, <=, in_range)
    try:
        value = float(value_str.rstrip("%"))
    except (ValueError, TypeError):
        return value_str == threshold

    if op == "in_range":
        parts = threshold.split("-")
        if len(parts) == 2:
            try:
                lo, hi = float(parts[0].rstrip("%")), float(parts[1].rstrip("%"))
                return lo <= value <= hi
            except ValueError:
                pass
        return True

    # Scalar comparison operators
    try:
        thresh = float(threshold.rstrip("%"))
    except (ValueError, TypeError):
        return True

    if op == ">":
        return value > thresh
    if op == "<":
        return value < thresh
    if op == ">=":
        return value >= thresh
    if op == "<=":
        return value <= thresh

    return True  # unknown operator → permissive


def all_conditions_met(
    conditions: list[CausalCondition],
    regime_context: dict[str, str],
) -> bool:
    """Check if all conditions are met given the current regime context."""
    return all(_evaluate_condition(c, regime_context) for c in conditions)


# ---------------------------------------------------------------------------
# Cascade engine
# ---------------------------------------------------------------------------

def run_cascade(
    graph: CausalGraph,
    shock: Shock,
    min_confidence: float = 0.15,
    max_depth: int = 6,
) -> dict[str, CascadeEffect]:
    """Max-confidence priority traversal.

    Explores edges in order of descending path confidence. Multiple
    paths to the same node are aggregated: confidence = max, direction
    = aggregated across channels, n_incoming_channels counts independent paths.
    """
    effects: dict[str, CascadeEffect] = {}
    regime = {**graph.regime_context, **shock.regime_context}

    # Priority queue: (-confidence, counter, entity, path, direction, t_min, t_max)
    # Counter breaks ties to avoid comparing non-comparable types
    counter = 0
    pq: list[tuple] = [(-1.0, counter, shock.entity, [shock.entity],
                         shock.direction, "immediate", "immediate")]

    while pq:
        neg_conf, _, current, path, direction, t_min, t_max = heapq.heappop(pq)
        current_confidence = -neg_conf

        if len(path) > max_depth + 1:
            continue

        # Update or create effect record
        if current not in effects:
            effects[current] = CascadeEffect(
                entity=current,
                direction=direction,
                confidence=current_confidence,
                time_horizon_min=t_min,
                time_horizon_max=t_max,
                causal_path=list(path),
                all_paths=[list(path)],
                n_incoming_channels=1,
                incoming_directions={direction: 1},
            )
        else:
            eff = effects[current]
            eff.all_paths.append(list(path))
            eff.n_incoming_channels += 1
            eff.incoming_directions[direction] = (
                eff.incoming_directions.get(direction, 0) + 1
            )
            if current_confidence > eff.confidence:
                eff.confidence = current_confidence
                eff.causal_path = list(path)
            # Detect direction conflict
            dirs = set(eff.incoming_directions.keys()) - {"ambiguous", "state_dependent"}
            if len(dirs) > 1:
                eff.direction = "conflicted"
            # Widen time horizon
            eff.time_horizon_min = min_time(eff.time_horizon_min, t_min)
            eff.time_horizon_max = max_time(eff.time_horizon_max, t_max)

        # Explore outgoing causal edges
        for claim in graph.outgoing_claims(current):
            # Check claim-level conditions (from each mechanism)
            for mechanism in claim.mechanisms:
                if not all_conditions_met(mechanism.conditions, regime):
                    continue

                edge_confidence = current_confidence * claim.confidence
                if edge_confidence < min_confidence:
                    continue

                next_direction = propagate_direction(direction, mechanism.direction)
                next_t_min = add_time(t_min, mechanism.time_lag_min or "immediate")
                next_t_max = add_time(t_max, mechanism.time_lag_max or t_max)

                if claim.target not in path:
                    counter += 1
                    new_path = path + [claim.target]
                    heapq.heappush(pq, (
                        -edge_confidence,
                        counter,
                        claim.target,
                        new_path,
                        next_direction,
                        next_t_min,
                        next_t_max,
                    ))

    return effects


# ---------------------------------------------------------------------------
# Feedback mode
# ---------------------------------------------------------------------------

def _merge_effects(existing: CascadeEffect, new: CascadeEffect) -> None:
    """Merge a new effect into an existing one."""
    existing.all_paths.extend(new.all_paths)
    existing.n_incoming_channels += new.n_incoming_channels
    for d, count in new.incoming_directions.items():
        existing.incoming_directions[d] = (
            existing.incoming_directions.get(d, 0) + count
        )
    if new.confidence > existing.confidence:
        existing.confidence = new.confidence
        existing.causal_path = new.causal_path
    dirs = set(existing.incoming_directions.keys()) - {"ambiguous", "state_dependent"}
    if len(dirs) > 1:
        existing.direction = "conflicted"
    existing.time_horizon_min = min_time(existing.time_horizon_min, new.time_horizon_min)
    existing.time_horizon_max = max_time(existing.time_horizon_max, new.time_horizon_max)


def run_cascade_with_feedback(
    graph: CausalGraph,
    shock: Shock,
    n_iterations: int = 3,
    min_confidence: float = 0.15,
    max_depth: int = 6,
    feedback_decay: float = 0.7,
) -> list[dict[str, CascadeEffect]]:
    """Multi-pass cascade for feedback loops.

    Iteration 0: Direct shock propagation
    Iteration k>0: Effects from iteration k-1 become new shocks,
    with confidence multiplied by feedback_decay^k.

    Returns a list of dicts, one per iteration.
    """
    all_iterations: list[dict[str, CascadeEffect]] = []
    current_shocks = [shock]

    for iteration in range(n_iterations):
        iteration_effects: dict[str, CascadeEffect] = {}

        for s in current_shocks:
            effects = run_cascade(
                graph, s, min_confidence=min_confidence, max_depth=max_depth,
            )
            for entity, eff in effects.items():
                if iteration > 0:
                    eff.confidence *= feedback_decay ** iteration
                if entity in iteration_effects:
                    _merge_effects(iteration_effects[entity], eff)
                else:
                    iteration_effects[entity] = eff

        all_iterations.append(iteration_effects)

        # Convert significant effects into new shocks for next iteration
        current_shocks = [
            Shock(
                entity=entity,
                shock_type="cascade_feedback",
                description=f"Feedback from {eff.causal_path}",
                direction=eff.direction if eff.direction in ("positive", "negative") else "negative",
                magnitude_qualitative="derived",
                counterfactual="vs no-feedback baseline",
                regime_context=shock.regime_context,
            )
            for entity, eff in iteration_effects.items()
            if eff.confidence >= min_confidence * 2
            and eff.direction not in ("ambiguous", "conflicted", "state_dependent")
        ]

        if not current_shocks:
            break  # no significant effects to propagate

    return all_iterations
