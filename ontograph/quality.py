"""Phase D — quality gate: six metrics + pass/fail verdict.

Computes coverage, groundedness, signedness, orphan_rate, consistency, and
cycle_count against the `validation:` block of the active schema (discipline-v1).

Gate keys consumed (all optional; absent keys ⇒ metric is not gated):
    require_grounding_for : list[str]     — entity types that must be grounded
    allow_abstract        : list[str]     — entity types that may be abstract
    groundedness_target   : float in [0,1]
    signedness_target     : float in [0,1]
    coverage_target       : float in [0,1]
    cycle_policy          : {within_step: forbid|allow, across_step: forbid|allow}

Design notes (advisor-driven, 2026-04-20):
    - Sign conflict = `+` vs `-` only. `unknown` is silence, not disagreement;
      `±` is matches-nothing, so it never conflicts.
    - Coverage without `ast_entities` ⇒ ratio None ⇒ gate skipped.
    - Groundedness is type-scoped when `require_grounding_for` is present; the
      global fallback (no type filter) kicks in only when that key is missing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx

from .merge import default_name_matcher
from .models import KnowledgeGraph, Entity
from .schema import OntologySchema

# Matcher threshold for the coverage metric. Kept aligned with the default in
# `ontograph.merge.merge_extractions` so coverage's "KG has this entity" is
# the same predicate as merge's "these are the same entity". Passing a stricter
# (e.g. 1.0) or looser matcher is available via `compute_quality_report`'s
# keyword arg.
_DEFAULT_COVERAGE_THRESHOLD = 0.85


# ── Report ──────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Result of a quality-gate run.

    Each top-level dict is self-describing so the JSON payload is readable
    without referring back to this module.
    """
    coverage: dict[str, Any] = field(default_factory=dict)
    groundedness: dict[str, Any] = field(default_factory=dict)
    signedness: dict[str, Any] = field(default_factory=dict)
    orphan_rate: dict[str, Any] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    cycle_count: dict[str, Any] = field(default_factory=dict)
    gates_passed: bool = True
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "coverage": self.coverage,
                "groundedness": self.groundedness,
                "signedness": self.signedness,
                "orphan_rate": self.orphan_rate,
                "consistency": self.consistency,
                "cycle_count": self.cycle_count,
                "gates_passed": self.gates_passed,
                "failures": list(self.failures),
            },
            indent=2,
            default=str,
        )


# ── Groundedness ────────────────────────────────────────────────────────

def _groundedness(kg: KnowledgeGraph, validation: dict[str, Any]) -> dict[str, Any]:
    require_types = set(validation.get("require_grounding_for") or [])
    allow_abstract = set(validation.get("allow_abstract") or [])

    by_type_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))  # (grounded, total)
    total = 0
    grounded = 0
    for ent in kg.entities.values():
        if require_types and ent.entity_type not in require_types:
            continue
        if ent.abstract and ent.entity_type in allow_abstract:
            continue
        g, t = by_type_counts[ent.entity_type]
        is_g = ent.is_grounded()
        by_type_counts[ent.entity_type] = (g + (1 if is_g else 0), t + 1)
        total += 1
        if is_g:
            grounded += 1

    ratio = (grounded / total) if total else 1.0
    by_type = {
        t: (g / c) if c else 1.0
        for t, (g, c) in by_type_counts.items()
    }
    return {"ratio": ratio, "grounded": grounded, "total": total, "by_type": by_type}


# ── Signedness ──────────────────────────────────────────────────────────

_SIGNED_VALUES = {"+", "-", "±", "0"}


def _signedness(kg: KnowledgeGraph) -> dict[str, Any]:
    total = 0
    signed = 0
    for rel in kg.relations:
        if rel.edge_class != "mechanism":
            continue
        total += 1
        if rel.sign in _SIGNED_VALUES:
            signed += 1
    ratio = (signed / total) if total else 1.0
    return {"ratio": ratio, "signed": signed, "total": total}


# ── Coverage ────────────────────────────────────────────────────────────

def _coverage(
    kg: KnowledgeGraph,
    ast_entities: list[Entity] | None,
    *,
    name_matcher: Callable[[str, str], float] = default_name_matcher,
    threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    if ast_entities is None:
        return {"ratio": None, "matched": 0, "total": 0}
    kg_names = list(kg.entities.keys())
    total = len(ast_entities)
    matched = 0
    for a in ast_entities:
        if a.name in kg.entities:
            matched += 1
            continue
        if any(name_matcher(a.name, kn) >= threshold for kn in kg_names):
            matched += 1
    ratio = (matched / total) if total else 1.0
    return {"ratio": ratio, "matched": matched, "total": total}


# ── Orphan rate ─────────────────────────────────────────────────────────

def _orphan_rate(kg: KnowledgeGraph) -> dict[str, Any]:
    incident: set[str] = set()
    for rel in kg.relations:
        incident.add(rel.source)
        incident.add(rel.target)
    total = 0
    orphans = 0
    orphan_names: list[str] = []
    for name, ent in kg.entities.items():
        if ent.abstract:
            continue
        total += 1
        if name not in incident:
            orphans += 1
            orphan_names.append(name)
    ratio = (orphans / total) if total else 0.0
    return {
        "ratio": ratio,
        "orphans": orphans,
        "total": total,
        "orphan_names": orphan_names,
    }


# ── Consistency (sign conflicts) ────────────────────────────────────────

def _consistency(kg: KnowledgeGraph) -> dict[str, Any]:
    """Detect `+` vs `-` conflicts on the same (source, target) mechanism edge.

    `unknown` is silence, `±` is matches-nothing: neither triggers a conflict.
    Denominator is the number of pairs with ≥2 determined-sign edges (i.e.
    pairs where a conflict check was actually performed).
    """
    pair_signs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in kg.relations:
        if rel.edge_class != "mechanism":
            continue
        if rel.sign in _SIGNED_VALUES:
            pair_signs[(rel.source, rel.target)].append(rel.sign)

    conflicts: list[dict[str, Any]] = []
    examined = 0
    for pair, signs in pair_signs.items():
        if len(signs) < 2:
            continue
        examined += 1
        if "+" in signs and "-" in signs:
            conflicts.append({
                "source": pair[0],
                "target": pair[1],
                "signs": sorted(set(signs)),
            })

    ratio = 1.0 if examined == 0 else 1.0 - len(conflicts) / examined
    return {"ratio": ratio, "conflicts": conflicts, "examined": examined}


# ── Cycle count (decomposed by lag) ─────────────────────────────────────

def _cycle_count(kg: KnowledgeGraph) -> dict[str, Any]:
    def _count_cycles_in(edges: list[tuple[str, str]]) -> int:
        if not edges:
            return 0
        g = nx.DiGraph()
        g.add_edges_from(edges)
        # Count non-trivial SCCs (≥2 nodes) + self-loop nodes. This is robust
        # and cheap; nx.simple_cycles can blow up on dense graphs.
        count = 0
        for scc in nx.strongly_connected_components(g):
            if len(scc) > 1:
                count += 1
            else:
                n = next(iter(scc))
                if g.has_edge(n, n):
                    count += 1
        return count

    within_edges = [(r.source, r.target) for r in kg.relations if r.lag == "within_step"]
    across_edges = [(r.source, r.target) for r in kg.relations if r.lag == "across_step"]
    return {
        "within_step": _count_cycles_in(within_edges),
        "across_step": _count_cycles_in(across_edges),
    }


# ── Public entry point ──────────────────────────────────────────────────

def compute_quality_report(
    kg: KnowledgeGraph,
    schema: OntologySchema,
    *,
    ast_entities: list[Entity] | None = None,
    coverage_name_matcher: Callable[[str, str], float] = default_name_matcher,
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> QualityReport:
    """Run all six metrics and decide gate pass/fail from schema.validation."""
    v = dict(schema.validation or {})
    report = QualityReport(
        coverage=_coverage(
            kg, ast_entities,
            name_matcher=coverage_name_matcher,
            threshold=coverage_threshold,
        ),
        groundedness=_groundedness(kg, v),
        signedness=_signedness(kg),
        orphan_rate=_orphan_rate(kg),
        consistency=_consistency(kg),
        cycle_count=_cycle_count(kg),
    )

    failures: list[str] = []

    # Groundedness gate
    g_target = v.get("groundedness_target")
    if g_target is not None and report.groundedness["total"] > 0:
        if report.groundedness["ratio"] < g_target:
            failures.append(
                f"groundedness {report.groundedness['ratio']:.3f} < target {g_target}"
            )

    # Signedness gate
    s_target = v.get("signedness_target")
    if s_target is not None and report.signedness["total"] > 0:
        if report.signedness["ratio"] < s_target:
            failures.append(
                f"signedness {report.signedness['ratio']:.3f} < target {s_target}"
            )

    # Coverage gate (skip if ratio is None — no AST input)
    c_target = v.get("coverage_target")
    c_ratio = report.coverage["ratio"]
    if c_target is not None and c_ratio is not None:
        if c_ratio < c_target:
            failures.append(f"coverage {c_ratio:.3f} < target {c_target}")

    # Cycle policy gate.
    #
    # Vocabulary: `forbid` → any cycle at that lag fails the gate.
    # Any other value (including `allow`) is *permissive* — cycles at that lag
    # are neither required nor forbidden. In particular, spec §1.2 criterion 4
    # ("≥1 across_step cycle expected for interesting dynamics") is NOT
    # enforced here; that would need an extended vocabulary
    # ({forbid, allow, require}) and is deliberately deferred.
    cycle_policy = v.get("cycle_policy") or {}
    if cycle_policy.get("within_step") == "forbid" and report.cycle_count["within_step"] > 0:
        failures.append(
            f"cycle-within_step: {report.cycle_count['within_step']} cycle(s) forbidden"
        )
    if cycle_policy.get("across_step") == "forbid" and report.cycle_count["across_step"] > 0:
        failures.append(
            f"cycle-across_step: {report.cycle_count['across_step']} cycle(s) forbidden"
        )

    report.failures = failures
    report.gates_passed = not failures
    return report
