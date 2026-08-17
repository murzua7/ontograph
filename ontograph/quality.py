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
from pathlib import Path
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
    resolution: dict[str, Any] = field(default_factory=dict)
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
                "resolution": self.resolution,
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


# ── Anchor resolution (flag-gated; default OFF) ─────────────────────────
#
# Groundedness checks anchor SHAPE; nothing in the gate ever opened a file, so
# a fabricated anchor scored the same as a real one (content-blindness probe,
# knowledge-vault outputs/probes/2026-08-16-ontograph-gate-content-blindness).
# This pass resolves every code anchor against a caller-supplied mapping of
# repo name to filesystem root: root known, file exists, line within the file,
# and, when the anchor names a symbol, the symbol text present in the file (a
# containment check, not a parse). Resolution is existence, never content: a
# real anchor whose passage is about something else still resolves, because
# claim-evidence congruence belongs to the judged content axis, not here.
# Citation anchors are counted but not checked; resolving them needs a
# bibliography, which the gate does not have.

def _find_document(name: str, doc_roots: list[Path]) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.is_file():
        return p
    for root in doc_roots:
        cand = root / name
        if cand.is_file():
            return cand
    return None


def _resolve_doc(a, doc_roots: list[Path]) -> tuple[str | None, bool]:
    """Resolve one document anchor. Returns (reason, drifted).

    reason None means resolvable. Drifted means the passage is present in the
    document but not at the recorded span, which resolves (the content
    exists) and is reported, because span rot is metadata decay rather than
    fabrication. Only anchors of grounding grade (a char offset or a line)
    are ever passed here, matching what is_grounded credits.
    """
    doc = _find_document(a.document, doc_roots)
    if doc is None:
        return "document not found", False
    try:
        text = doc.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "document unreadable", False
    if a.passage:
        if 0 <= a.char_start <= a.char_end <= len(text) \
                and text[a.char_start:a.char_end] == a.passage:
            return None, False
        if a.passage in text:
            return None, True
        return "passage not found in document", False
    if a.char_start >= 0:
        if a.char_end <= len(text):
            return None, False
        return f"span {a.char_start}-{a.char_end} out of range " \
               f"(document has {len(text)} characters)", False
    n_lines = text.count("\n") + 1
    if a.line is not None and 1 <= a.line <= n_lines:
        return None, False
    return f"line {a.line} out of range (document has {n_lines} lines)", False


def _resolution(kg: KnowledgeGraph,
                anchor_roots: dict[str, str] | None,
                doc_roots: list[str] | None = None) -> dict[str, Any]:
    roots = {k: Path(v) for k, v in (anchor_roots or {}).items()}
    droots = [Path(d) for d in (doc_roots or [])]
    checked = 0
    citation_unchecked = 0
    drifted: list[dict[str, Any]] = []
    unresolvable: list[dict[str, Any]] = []
    for ent in kg.entities.values():
        citation_unchecked += len(ent.citation_anchors)
        for a in getattr(ent, "doc_anchors", []):
            if a.char_start < 0 and a.line is None:
                continue  # not grounding-grade; is_grounded does not credit it
            checked += 1
            reason, drift = _resolve_doc(a, droots)
            entry = {"entity": ent.name, "document": a.document,
                     "char_start": a.char_start, "char_end": a.char_end,
                     "line": a.line}
            if reason is not None:
                unresolvable.append(dict(entry, reason=reason))
            elif drift:
                drifted.append(entry)
        for a in ent.code_anchors:
            checked += 1
            reason = None
            root = roots.get(a.repo)
            if root is None:
                reason = "unknown repo root"
            else:
                f = root / a.path
                if not f.is_file():
                    reason = "file missing"
                else:
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        reason = "file unreadable"
                    else:
                        n_lines = text.count("\n") + 1
                        if a.line < 1 or a.line > n_lines:
                            reason = f"line {a.line} out of range (file has {n_lines})"
                        elif a.symbol and a.symbol not in text:
                            reason = f"symbol '{a.symbol}' not found"
            if reason is not None:
                unresolvable.append({"entity": ent.name, "repo": a.repo,
                                     "path": a.path, "line": a.line,
                                     "symbol": a.symbol, "reason": reason})
    resolved = checked - len(unresolvable)
    return {"enabled": True, "checked": checked, "resolved": resolved,
            "ratio": (resolved / checked) if checked else 1.0,
            "unresolvable": unresolvable, "drifted": drifted,
            "citation_anchors_unchecked": citation_unchecked}


# ── Signedness ──────────────────────────────────────────────────────────

_SIGNED_VALUES = {"+", "-", "±", "0"}

# Weighted signedness (advisor-driven, 2026-04-21): `+/-` carry a clear
# directional claim (full credit); `±/0` are explicit-ambiguous annotations
# — useful, but carry half the information of a signed mechanism; `unknown`
# is extractor silence and gets zero credit. Prevents a gate from being
# gamed by blanket-labelling edges `±`.
_SIGN_WEIGHT: dict[str, float] = {
    "+": 1.0, "-": 1.0,
    "±": 0.5, "0": 0.5,
}


def _signedness(kg: KnowledgeGraph) -> dict[str, Any]:
    total = 0
    signed = 0
    weighted = 0.0
    by_sign: dict[str, int] = defaultdict(int)
    for rel in kg.relations:
        if rel.edge_class != "mechanism":
            continue
        total += 1
        by_sign[rel.sign or "unknown"] += 1
        if rel.sign in _SIGNED_VALUES:
            signed += 1
        weighted += _SIGN_WEIGHT.get(rel.sign or "", 0.0)
    ratio = (signed / total) if total else 1.0
    weighted_ratio = (weighted / total) if total else 1.0
    return {
        "ratio": ratio,
        "signed": signed,
        "total": total,
        "weighted_ratio": weighted_ratio,
        "weighted_signed": weighted,
        "by_sign": dict(by_sign),
    }


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
    resolve_anchors: bool = False,
    anchor_roots: dict[str, str] | None = None,
    doc_roots: list[str] | None = None,
) -> QualityReport:
    """Run all six metrics and decide gate pass/fail from schema.validation.

    `resolve_anchors` (default OFF, so existing callers are unaffected) adds
    the anchor-resolution pass: every code anchor must resolve against
    `anchor_roots` (repo name -> filesystem root), and every grounding-grade
    document anchor must resolve against `doc_roots` (directories searched
    for the anchor's document), or the gate fails. A passage found in its
    document but not at the recorded span resolves and is reported as
    drifted."""
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
        resolution=_resolution(kg, anchor_roots, doc_roots) if resolve_anchors
        else {"enabled": False, "checked": 0},
    )

    failures: list[str] = []

    # Anchor-resolution gate (only when the pass ran). Any unresolvable code
    # anchor fails: a fabricated anchor is a provenance defect per se, not a
    # ratio to trade off.
    if report.resolution.get("enabled") and report.resolution["unresolvable"]:
        failures.append(
            f"anchor_resolution: {len(report.resolution['unresolvable'])} "
            "unresolvable anchor(s)"
        )

    # Groundedness gate
    g_target = v.get("groundedness_target")
    if g_target is not None and report.groundedness["total"] > 0:
        if report.groundedness["ratio"] < g_target:
            failures.append(
                f"groundedness {report.groundedness['ratio']:.3f} < target {g_target}"
            )

    # Signedness gate (binary: any sign ∈ {+,-,±,0} counts)
    s_target = v.get("signedness_target")
    if s_target is not None and report.signedness["total"] > 0:
        if report.signedness["ratio"] < s_target:
            failures.append(
                f"signedness {report.signedness['ratio']:.3f} < target {s_target}"
            )

    # Weighted signedness gate (+/- → 1.0, ±/0 → 0.5, unknown → 0.0).
    # Independent of the binary gate; schemas can gate on either or both.
    ws_target = v.get("weighted_signedness_target")
    if ws_target is not None and report.signedness["total"] > 0:
        if report.signedness["weighted_ratio"] < ws_target:
            failures.append(
                f"weighted_signedness {report.signedness['weighted_ratio']:.3f} "
                f"< target {ws_target}"
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
