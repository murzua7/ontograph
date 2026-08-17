"""Tests for `ontograph cycle-report` (#43)."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from ontograph.cli import cmd_cycle_report
from ontograph.models import Entity, KnowledgeGraph, Relation


def _kg_with_cycle(*, lag: str, signs: tuple[str, str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for n in ("a", "b"):
        kg.add_entity(Entity(name=n, entity_type="variable"))
    s1, s2 = signs
    kg.add_relation(Relation(
        source="a", target="b", relation_type="affects",
        sign=s1, lag=lag, edge_class="mechanism",
    ))
    kg.add_relation(Relation(
        source="b", target="a", relation_type="affects",
        sign=s2, lag=lag, edge_class="mechanism",
    ))
    return kg


@pytest.fixture
def two_cycle_kg(tmp_path):
    kg = _kg_with_cycle(lag="within_step", signs=("+", "-"))
    p = tmp_path / "kg.json"
    p.write_text(kg.to_json(), encoding="utf-8")
    return p


def _args(input_path, **overrides):
    base = dict(input=str(input_path), output=None, max_len=8, max_cycles=50)
    base.update(overrides)
    return Namespace(**base)


class TestCmdCycleReport:
    def test_returns_zero_and_writes_default_report(self, two_cycle_kg):
        rc = cmd_cycle_report(_args(two_cycle_kg))
        assert rc == 0
        out = two_cycle_kg.with_suffix(".cycles.json")
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["n_cycles"] == 1
        assert payload["cycles"][0]["lag"] == "within_step"
        assert sorted(payload["cycles"][0]["nodes"]) == ["a", "b"]

    def test_sign_product_embedded(self, two_cycle_kg):
        cmd_cycle_report(_args(two_cycle_kg))
        out = two_cycle_kg.with_suffix(".cycles.json")
        payload = json.loads(out.read_text(encoding="utf-8"))
        # + * - = -
        assert payload["cycles"][0]["product"] == "-"

    def test_explicit_output_path_used(self, two_cycle_kg, tmp_path):
        custom = tmp_path / "elsewhere.json"
        cmd_cycle_report(_args(two_cycle_kg, output=str(custom)))
        assert custom.exists()
        # Default location was NOT written.
        assert not two_cycle_kg.with_suffix(".cycles.json").exists()

    def test_lags_split_in_report(self, tmp_path):
        # One cycle per lag bucket; report should distinguish them.
        kg = KnowledgeGraph()
        for n in ("a", "b", "c", "d"):
            kg.add_entity(Entity(name=n, entity_type="variable"))
        for src, tgt, sgn, lag in [
            ("a", "b", "+", "within_step"), ("b", "a", "+", "within_step"),
            ("c", "d", "-", "across_step"), ("d", "c", "+", "across_step"),
        ]:
            kg.add_relation(Relation(
                source=src, target=tgt, relation_type="r",
                sign=sgn, lag=lag, edge_class="mechanism",
            ))
        p = tmp_path / "kg.json"
        p.write_text(kg.to_json(), encoding="utf-8")

        cmd_cycle_report(_args(p))
        payload = json.loads(p.with_suffix(".cycles.json").read_text(encoding="utf-8"))
        assert payload["n_cycles"] == 2
        lags = sorted(c["lag"] for c in payload["cycles"])
        assert lags == ["across_step", "within_step"]

    def test_max_len_caps_enumeration(self, tmp_path):
        # 3-cycle a→b→c→a; cap at length 2 → no cycle reported.
        kg = KnowledgeGraph()
        for n in ("a", "b", "c"):
            kg.add_entity(Entity(name=n, entity_type="variable"))
        for src, tgt in [("a", "b"), ("b", "c"), ("c", "a")]:
            kg.add_relation(Relation(
                source=src, target=tgt, relation_type="r",
                sign="+", lag="within_step", edge_class="mechanism",
            ))
        p = tmp_path / "kg.json"
        p.write_text(kg.to_json(), encoding="utf-8")
        cmd_cycle_report(_args(p, max_len=2))
        payload = json.loads(p.with_suffix(".cycles.json").read_text(encoding="utf-8"))
        assert payload["n_cycles"] == 0

    def test_no_cycles_emits_empty_list(self, tmp_path):
        kg = KnowledgeGraph()
        for n in ("a", "b"):
            kg.add_entity(Entity(name=n, entity_type="variable"))
        kg.add_relation(Relation(
            source="a", target="b", relation_type="r",
            sign="+", lag="within_step", edge_class="mechanism",
        ))
        p = tmp_path / "kg.json"
        p.write_text(kg.to_json(), encoding="utf-8")
        cmd_cycle_report(_args(p))
        payload = json.loads(p.with_suffix(".cycles.json").read_text(encoding="utf-8"))
        assert payload["n_cycles"] == 0
        assert payload["cycles"] == []

    def test_report_carries_caps(self, two_cycle_kg):
        cmd_cycle_report(_args(two_cycle_kg, max_len=5, max_cycles=10))
        payload = json.loads(two_cycle_kg.with_suffix(".cycles.json").read_text(encoding="utf-8"))
        assert payload["max_len"] == 5
        assert payload["max_cycles_per_lag"] == 10
