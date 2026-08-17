"""Tests for `ontograph ambiguity-benchmark` (#48)."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from ontograph import cli
from ontograph.models import Entity, KnowledgeGraph, Relation


def _kg_with(edges):
    kg = KnowledgeGraph()
    names = {n for s, t, _ in edges for n in (s, t)}
    for n in names:
        kg.add_entity(Entity(name=n, entity_type="variable"))
    for s, t, sign in edges:
        kg.add_relation(Relation(
            source=s, target=t, relation_type="r",
            sign=sign, lag="within_step", edge_class="mechanism",
        ))
    return kg


@pytest.fixture
def kg_and_gold(tmp_path):
    kg = _kg_with([
        ("A", "B", "+"),    # matches gold decisive_positive
        ("A", "C", "+"),    # overconfident vs gold ambiguous
    ])
    gold = [
        {"source": "A", "target": "B", "expected_class": "decisive_positive",
         "passage": "A lifts B."},
        {"source": "A", "target": "C", "expected_class": "ambiguous",
         "passage": "A's effect on C depends on regime."},
    ]
    kg_path = tmp_path / "kg.json"
    gold_path = tmp_path / "gold.json"
    kg_path.write_text(kg.to_json(), encoding="utf-8")
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    return kg_path, gold_path


def _args(kg_path, gold_path, **overrides):
    base = {"kg": str(kg_path), "gold": str(gold_path), "report": None}
    base.update(overrides)
    return Namespace(**base)


class TestCmdAmbiguityBenchmark:
    def test_returns_zero_and_writes_report(self, kg_and_gold):
        kg_p, gold_p = kg_and_gold
        rc = cli.cmd_ambiguity_benchmark(_args(kg_p, gold_p))
        assert rc == 0
        default_report = kg_p.with_suffix(".ambiguity.json")
        assert default_report.exists()
        payload = json.loads(default_report.read_text(encoding="utf-8"))
        assert payload["overall_accuracy"] == 0.5
        assert payload["overconfidence_rate"] == 1.0  # 1 of 1 ambiguous → decisive

    def test_bad_gold_returns_nonzero(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        kg_p.write_text(_kg_with([("A", "B", "+")]).to_json(), encoding="utf-8")
        gold_p = tmp_path / "bad.json"
        gold_p.write_text(json.dumps([{"source": "A"}]), encoding="utf-8")  # missing keys
        rc = cli.cmd_ambiguity_benchmark(_args(kg_p, gold_p))
        assert rc == 1

    def test_custom_report_path(self, kg_and_gold, tmp_path):
        kg_p, gold_p = kg_and_gold
        out = tmp_path / "custom.json"
        rc = cli.cmd_ambiguity_benchmark(_args(kg_p, gold_p, report=str(out)))
        assert rc == 0
        assert out.exists()
