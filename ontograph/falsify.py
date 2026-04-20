"""Phase E — simulator-in-the-loop falsifier (spec §4.3).

Takes a knowledge graph of hypothesised causal edges (the KG) and a
`SimulatorAdapter` that knows how to query empirical ground truth from a
running simulator (finite-difference Jacobians, gradient-norm probes, …).
Partitions the KG's mechanism edges into three buckets:

    confirmed : KG sign == simulator sign              (hypothesis holds)
    flipped   : KG sign == +, simulator == − (or reverse) — audit target
    unknown   : simulator could not resolve a sign      (silence ≠ disagreement)

`unknown` in either direction is silence, not contradiction (spec §2
principle 2). `±` ("ambiguous / matches-nothing") never contradicts; a KG
sign of `unknown` is skipped entirely because there is nothing to falsify.

The SimulatorAdapter protocol is the narrow seam between ontograph and
the simulator. A DABS-side implementation lives in the DABS repo; this
module only depends on the protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import KnowledgeGraph


# ── Adapter protocol ────────────────────────────────────────────────────

class SimulatorAdapter(Protocol):
    """Narrow contract a simulator must satisfy to be falsifiable.

    Both methods may return `"unknown"` when the simulator cannot resolve
    the quantity (no perturbation support, response below numerical
    noise, etc.). This is silence, and the falsifier treats it as such.
    """

    def edge_sign(self, source: str, target: str) -> str:
        """Return the empirical sign of d(target)/d(source).

        Expected return values: `"+"`, `"-"`, `"±"`, `"0"`, or `"unknown"`.
        """

    def edge_lag(self, source: str, target: str) -> str:
        """Return the lag classification for the source→target perturbation."""


# ── Report ──────────────────────────────────────────────────────────────

@dataclass
class FalsifyReport:
    """Result of a falsification run.

    Each bucket is a list of dicts so the JSON payload reads cleanly
    without a consumer-side decoder.
    """
    confirmed: list[dict[str, Any]] = field(default_factory=list)
    flipped: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    examined: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "confirmed": self.confirmed,
                "flipped": self.flipped,
                "unknown": self.unknown,
                "total": self.total,
                "examined": self.examined,
            },
            indent=2,
            default=str,
        )


# ── Core entry point ────────────────────────────────────────────────────

_PAIRED_SIGNS = {"+", "-"}


def _entry(rel, sim_sign: str, sim_lag: str) -> dict[str, Any]:
    return {
        "source": rel.source,
        "target": rel.target,
        "relation_type": rel.relation_type,
        "edge_class": rel.edge_class,
        "kg_sign": rel.sign,
        "sim_sign": sim_sign,
        "kg_lag": rel.lag,
        "sim_lag": sim_lag,
    }


def falsify(
    kg: KnowledgeGraph,
    adapter: SimulatorAdapter,
    *,
    subset: str = "mechanism",
) -> FalsifyReport:
    """Partition KG edges into confirmed / flipped / unknown buckets.

    `subset="mechanism"` (default) examines only mechanism-class edges —
    identity / structural / parameter / abstract edges are out of scope
    for causal falsification. `subset="all"` examines every edge (useful
    for debugging the adapter itself).

    KG relations with `sign == "unknown"` are silently skipped: there's
    no hypothesis to falsify. `±` ("ambiguous / matches-nothing") is
    recorded as examined but can never be classified as flipped — it is
    bucketed as `confirmed` when the simulator returns a paired sign
    (because `±` subsumes both) or `unknown` when the simulator is
    silent.
    """
    if subset not in {"mechanism", "all"}:
        raise ValueError(f"subset must be 'mechanism' or 'all'; got {subset!r}")

    report = FalsifyReport(total=len(kg.relations))

    for rel in kg.relations:
        if subset == "mechanism" and rel.edge_class != "mechanism":
            continue
        # Nothing to falsify when the KG itself is silent.
        if rel.sign == "unknown":
            continue

        report.examined += 1
        sim_sign = adapter.edge_sign(rel.source, rel.target)
        sim_lag = adapter.edge_lag(rel.source, rel.target)

        if sim_sign == "unknown":
            report.unknown.append(_entry(rel, sim_sign, sim_lag))
            continue

        # `±` never conflicts with anything: it matches-nothing.
        if rel.sign == "±":
            report.confirmed.append(_entry(rel, sim_sign, sim_lag))
            continue

        # Paired-sign case: `+` vs `-` is the only real conflict. All
        # other combinations (e.g. `0` vs `+`, `±` vs `+`) are treated as
        # confirmations or non-conflicts — spec §2 keeps the conflict
        # predicate narrow on purpose.
        if (rel.sign in _PAIRED_SIGNS and sim_sign in _PAIRED_SIGNS
                and rel.sign != sim_sign):
            report.flipped.append(_entry(rel, sim_sign, sim_lag))
        else:
            report.confirmed.append(_entry(rel, sim_sign, sim_lag))

    return report
