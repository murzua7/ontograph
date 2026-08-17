"""Tests for the intentional-ambiguity extraction benchmark (#48).

The idea: every edge in the gold set is labelled by the *kind of claim the
source passage actually supports* — decisive (`+` / `-`), explicit-ambiguous
(`±` / `0`), or silent (the passage doesn't let you call it either way).

An extractor that blindly labels every edge `+` looks accurate on any
aggregate sign metric, but is epistemically overconfident. This benchmark
surfaces that failure mode directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ontograph.benchmark_ambiguity import (
    AmbiguityBenchmark,
    AmbiguityCase,
    score_ambiguity,
)
from ontograph.models import Entity, KnowledgeGraph, Relation

GOLD_FIXTURE = Path(__file__).parent / "fixtures" / "ambiguity_gold.json"


def _kg_with(edges: list[tuple[str, str, str]]) -> KnowledgeGraph:
    """Build a KG with (source, target, sign) triples."""
    kg = KnowledgeGraph()
    names: set[str] = set()
    for s, t, _ in edges:
        names.update([s, t])
    for n in names:
        kg.add_entity(Entity(name=n, entity_type="variable"))
    for s, t, sign in edges:
        kg.add_relation(Relation(
            source=s, target=t, relation_type="r",
            sign=sign, lag="within_step", edge_class="mechanism",
        ))
    return kg


class TestAmbiguityCase:
    def test_case_records_expected_class(self):
        c = AmbiguityCase(
            source="A", target="B",
            expected_class="decisive_positive",
            passage="A raises B.",
        )
        assert c.expected_class == "decisive_positive"

    def test_rejects_unknown_expected_class(self):
        with pytest.raises(ValueError):
            AmbiguityCase(
                source="A", target="B",
                expected_class="definitely_positive",  # typo
                passage="irrelevant",
            )


class TestScoreAmbiguity:
    def _bench(self) -> AmbiguityBenchmark:
        """Four cases — one per class — so every score bucket gets exercised."""
        return AmbiguityBenchmark(cases=[
            AmbiguityCase("A", "B", "decisive_positive",
                          "An increase in A causes an increase in B."),
            AmbiguityCase("A", "C", "decisive_negative",
                          "Higher A depresses C."),
            AmbiguityCase("D", "E", "ambiguous",
                          "The effect of D on E depends on regime."),
            AmbiguityCase("F", "G", "silent",
                          "F and G co-occur in the dataset."),
        ])

    def test_perfect_extractor_scores_one(self):
        """Extractor that exactly matches the gold labels."""
        bench = self._bench()
        kg = _kg_with([
            ("A", "B", "+"),
            ("A", "C", "-"),
            ("D", "E", "±"),
            # silent → no edge extracted (explicit silence is the right call)
        ])
        score = score_ambiguity(kg, bench)
        assert score.overall_accuracy == pytest.approx(1.0)
        assert score.overconfidence_rate == pytest.approx(0.0)
        assert score.silence_recall == pytest.approx(1.0)

    def test_overconfident_extractor_penalised(self):
        """Extractor that labels truly-ambiguous edges as `+` — overconfident."""
        bench = self._bench()
        kg = _kg_with([
            ("A", "B", "+"),
            ("A", "C", "-"),
            ("D", "E", "+"),   # overconfident: gold = ambiguous
            ("F", "G", "+"),   # overconfident: gold = silent
        ])
        score = score_ambiguity(kg, bench)
        # 2 of the 2 non-decisive gold cases got a decisive label.
        assert score.overconfidence_rate == pytest.approx(1.0)
        # Overall accuracy drops because 2/4 cases are wrong.
        assert score.overall_accuracy == pytest.approx(0.5)

    def test_silent_extractor_on_decisive_passage_penalised(self):
        """Extractor that omits decisive edges is under-confident — penalised
        via decisive_recall but NOT via overconfidence_rate."""
        bench = self._bench()
        kg = _kg_with([
            # No A→B or A→C (extractor missed decisive claims).
            ("D", "E", "±"),
            # F→G correctly silent.
        ])
        score = score_ambiguity(kg, bench)
        assert score.decisive_recall == pytest.approx(0.0)
        assert score.overconfidence_rate == pytest.approx(0.0)

    def test_sign_flip_counts_as_decisive_error(self):
        """Gold says `+`, extractor says `-` — decisive sign error, not
        ambiguity error."""
        bench = AmbiguityBenchmark(cases=[
            AmbiguityCase("A", "B", "decisive_positive", "A lifts B."),
        ])
        kg = _kg_with([("A", "B", "-")])
        score = score_ambiguity(kg, bench)
        assert score.decisive_sign_errors == 1
        assert score.overall_accuracy == pytest.approx(0.0)

    def test_report_shape_is_serialisable(self):
        """The score object must round-trip to JSON for report writing."""
        bench = self._bench()
        kg = _kg_with([("A", "B", "+")])
        score = score_ambiguity(kg, bench)
        d = score.to_dict()
        assert set(d.keys()) >= {
            "overall_accuracy",
            "overconfidence_rate",
            "decisive_recall",
            "silence_recall",
            "by_class",
            "confusion",
        }
        # Confusion matrix rows and columns span the four expected classes.
        expected = {"decisive_positive", "decisive_negative",
                    "ambiguous", "silent"}
        assert set(d["confusion"].keys()) == expected

    def test_silent_gold_with_ambiguous_prediction(self):
        """Gold says `silent`, extractor says `±` — hedging on a silent
        passage fails silence_recall but is NOT overconfidence."""
        bench = AmbiguityBenchmark(cases=[
            AmbiguityCase("F", "G", "silent", "F and G co-occur."),
        ])
        kg = _kg_with([("F", "G", "±")])
        score = score_ambiguity(kg, bench)
        assert score.silence_recall == pytest.approx(0.0)
        assert score.overconfidence_rate == pytest.approx(0.0)

    def test_extractor_with_unknown_sign_counted_as_silent(self):
        """`unknown` on an extracted edge is extractor silence, equivalent to
        omitting the edge for the purposes of this benchmark."""
        bench = AmbiguityBenchmark(cases=[
            AmbiguityCase("F", "G", "silent",
                          "F and G appear together but no claim is made."),
        ])
        # Extractor extracted the edge but declared no sign — treat as silence.
        kg = _kg_with([("F", "G", "unknown")])
        score = score_ambiguity(kg, bench)
        assert score.silence_recall == pytest.approx(1.0)


class TestReferenceGoldFixture:
    """Validate the shipped reference gold set is well-formed and loadable.

    Without this check, a typo in the JSON ships silently — users see an
    opaque KeyError the first time they run `ontograph ambiguity-benchmark`.
    """

    def test_fixture_exists_and_parses(self):
        assert GOLD_FIXTURE.exists(), f"{GOLD_FIXTURE} missing"
        records = json.loads(GOLD_FIXTURE.read_text(encoding="utf-8"))
        assert isinstance(records, list)
        assert len(records) >= 10, "benchmark should exercise enough cases"

    def test_fixture_constructs_all_cases(self):
        records = json.loads(GOLD_FIXTURE.read_text(encoding="utf-8"))
        cases = [AmbiguityCase(**r) for r in records]
        # All four classes must be present — otherwise the benchmark
        # can't surface the failure modes it claims to surface.
        classes = {c.expected_class for c in cases}
        assert classes == {
            "decisive_positive", "decisive_negative",
            "ambiguous", "silent",
        }
