"""Intentional-ambiguity extraction benchmark (#48).

Purpose: measure whether an extractor *correctly identifies the epistemic
kind of claim the source passage supports* — not merely whether it gets
edges "mostly right".

Four gold classes, one per way a passage can speak about a directed pair:

    decisive_positive  — the passage commits to a positive effect
    decisive_negative  — the passage commits to a negative effect
    ambiguous          — the passage explicitly flags the effect as
                         regime-dependent / context-dependent / sign-unclear
    silent             — the passage mentions the entities but does not
                         make a causal claim

The headline metric is ``overconfidence_rate``: of the gold
`ambiguous`+`silent` cases, what fraction did the extractor pin down as a
decisive sign? An extractor that maximises raw coverage by always emitting
`+` will score near zero on this metric; a calibrated extractor will
reserve `+/-` for passages that actually commit.

Mapping extractor output → predicted class:

    edge.sign == '+'                         → decisive_positive
    edge.sign == '-'                         → decisive_negative
    edge.sign in {'±', '0'}                  → ambiguous
    edge.sign == 'unknown' or edge missing   → silent

`0` ("explicitly no effect") is lumped with `±` because for this benchmark
we care about *epistemic confidence*: both are "the passage didn't let me
commit to a direction". If the future requires separating them, split the
class here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from ontograph.models import KnowledgeGraph


ExpectedClass = Literal[
    "decisive_positive",
    "decisive_negative",
    "ambiguous",
    "silent",
]

_VALID_CLASSES: frozenset[str] = frozenset({
    "decisive_positive",
    "decisive_negative",
    "ambiguous",
    "silent",
})

_DECISIVE_CLASSES: frozenset[str] = frozenset({
    "decisive_positive",
    "decisive_negative",
})


@dataclass
class AmbiguityCase:
    """One gold-labelled (source, target) pair with supporting passage.

    `passage` is kept on the case so a benchmark fixture is self-documenting
    — a failing case can be re-read without chasing doc-line anchors.
    """
    source: str
    target: str
    expected_class: str
    passage: str

    def __post_init__(self) -> None:
        if self.expected_class not in _VALID_CLASSES:
            raise ValueError(
                f"expected_class={self.expected_class!r} not in "
                f"{sorted(_VALID_CLASSES)}"
            )


@dataclass
class AmbiguityBenchmark:
    cases: list[AmbiguityCase] = field(default_factory=list)


def _predict_class(sign: str | None) -> str:
    """Map an extracted edge's sign to a predicted class label."""
    if sign == "+":
        return "decisive_positive"
    if sign == "-":
        return "decisive_negative"
    if sign in {"±", "0"}:
        return "ambiguous"
    # `unknown`, `None`, or anything else ⇒ extractor silence.
    return "silent"


def _edges_by_pair(kg: KnowledgeGraph) -> dict[tuple[str, str], list[str]]:
    """Group extracted signs by (source, target). Multiple edges per pair
    are possible — `_select_sign` picks the representative."""
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in kg.relations:
        if r.edge_class != "mechanism":
            continue
        by_pair[(r.source, r.target)].append(r.sign or "unknown")
    return by_pair


def _select_sign(signs: list[str]) -> str:
    """Pick the most-informative sign when a pair has multiple edges.

    Preference order: decisive `+/-` > explicit-ambiguous `±/0` > `unknown`.
    If both `+` and `-` are present, the pair is in conflict — we surface
    that as `±` (explicit-ambiguous) rather than picking a side arbitrarily.
    """
    has_pos = "+" in signs
    has_neg = "-" in signs
    if has_pos and has_neg:
        return "±"
    if has_pos:
        return "+"
    if has_neg:
        return "-"
    if "±" in signs:
        return "±"
    if "0" in signs:
        return "0"
    return "unknown"


@dataclass
class AmbiguityScore:
    overall_accuracy: float
    overconfidence_rate: float
    decisive_recall: float
    silence_recall: float
    decisive_sign_errors: int
    by_class: dict[str, dict[str, Any]]
    confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "overconfidence_rate": self.overconfidence_rate,
            "decisive_recall": self.decisive_recall,
            "silence_recall": self.silence_recall,
            "decisive_sign_errors": self.decisive_sign_errors,
            "by_class": dict(self.by_class),
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
        }


def score_ambiguity(
    kg: KnowledgeGraph,
    benchmark: AmbiguityBenchmark,
) -> AmbiguityScore:
    """Score an extracted KG against a gold ambiguity benchmark.

    Only edges with ``edge_class == "mechanism"`` are considered — accounting
    and definitional edges encode structure, not epistemic commitment, so
    they don't participate in this score. If a caller's extractor emits those
    classes exclusively for a gold pair, the pair is treated as `silent`.
    """
    by_pair = _edges_by_pair(kg)

    # Confusion matrix: confusion[gold_class][predicted_class] = count.
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in _VALID_CLASSES} for g in _VALID_CLASSES
    }
    per_class_totals: dict[str, int] = {g: 0 for g in _VALID_CLASSES}

    decisive_sign_errors = 0
    correct = 0
    total = len(benchmark.cases)

    for case in benchmark.cases:
        gold = case.expected_class
        signs = by_pair.get((case.source, case.target))
        if signs is None:
            predicted = "silent"
        else:
            predicted = _predict_class(_select_sign(signs))
        confusion[gold][predicted] += 1
        per_class_totals[gold] += 1
        if predicted == gold:
            correct += 1
        elif gold in _DECISIVE_CLASSES and predicted in _DECISIVE_CLASSES:
            # Both decisive but opposite sign → sign flip, not a class error.
            decisive_sign_errors += 1

    # Overconfidence: of gold non-decisive cases, how many were predicted
    # as decisive? Upper-bounded by 1.0; 0.0 is the calibrated ideal.
    non_decisive_total = (
        per_class_totals["ambiguous"] + per_class_totals["silent"]
    )
    overconfident = 0
    for gold in ("ambiguous", "silent"):
        for pred in _DECISIVE_CLASSES:
            overconfident += confusion[gold][pred]
    overconfidence_rate = (
        overconfident / non_decisive_total if non_decisive_total else 0.0
    )

    # Decisive recall: of gold decisive cases, how many were recovered with
    # the correct decisive label?
    decisive_total = sum(per_class_totals[g] for g in _DECISIVE_CLASSES)
    decisive_hits = sum(confusion[g][g] for g in _DECISIVE_CLASSES)
    decisive_recall = (
        decisive_hits / decisive_total if decisive_total else 1.0
    )

    # Silence recall: of gold silent cases, how many did the extractor also
    # treat as silent? (Reward for correctly NOT fabricating claims.)
    silent_total = per_class_totals["silent"]
    silence_recall = (
        confusion["silent"]["silent"] / silent_total if silent_total else 1.0
    )

    by_class: dict[str, dict[str, Any]] = {}
    for g in _VALID_CLASSES:
        t = per_class_totals[g]
        hit = confusion[g][g]
        by_class[g] = {
            "total": t,
            "correct": hit,
            "recall": (hit / t) if t else 1.0,
        }

    return AmbiguityScore(
        overall_accuracy=(correct / total) if total else 1.0,
        overconfidence_rate=overconfidence_rate,
        decisive_recall=decisive_recall,
        silence_recall=silence_recall,
        decisive_sign_errors=decisive_sign_errors,
        by_class=by_class,
        confusion=confusion,
    )
