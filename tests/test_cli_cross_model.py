"""Tests for `ontograph cross-model` (#46)."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from ontograph import cli
from ontograph.models import Entity, KnowledgeGraph, Relation


def _kg(entities, rels):
    kg = KnowledgeGraph()
    for n, t in entities:
        kg.add_entity(Entity(name=n, entity_type=t))
    for r in rels:
        kg.add_relation(Relation(**r))
    return kg


@pytest.fixture
def two_kg_paths(tmp_path):
    ents = [("a", "variable"), ("b", "variable")]
    a_rels = [{"source": "a", "target": "b", "relation_type": "r",
               "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
    b_rels = [{"source": "a", "target": "b", "relation_type": "r",
               "sign": "-", "lag": "within_step", "edge_class": "mechanism"}]
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(_kg(ents, a_rels).to_json(), encoding="utf-8")
    pb.write_text(_kg(ents + [("c", "variable")], b_rels).to_json(), encoding="utf-8")
    return pa, pb


def _args(a_path, b_path, **overrides):
    base = dict(kg_a=str(a_path), kg_b=str(b_path), output=None, report=None)
    base.update(overrides)
    return Namespace(**base)


class TestCmdCrossModel:
    def test_returns_zero_and_writes_agreement_kg(self, two_kg_paths):
        pa, pb = two_kg_paths
        rc = cli.cmd_cross_model(_args(pa, pb))
        assert rc == 0
        out = pa.with_name(pa.stem + ".cross_model.json")
        assert out.exists()
        kg = KnowledgeGraph.from_json(out.read_text(encoding="utf-8"))
        # `c` is b-only → excluded; `a` and `b` are shared.
        assert set(kg.entities) == {"a", "b"}
        # Sign conflict → agreement edge sign is 'unknown'.
        assert kg.relations[0].sign == "unknown"

    def test_report_records_sign_conflict(self, two_kg_paths):
        pa, pb = two_kg_paths
        cli.cmd_cross_model(_args(pa, pb))
        report_path = pa.with_name(pa.stem + ".cross_model_report.json")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(payload["relations"]["sign_conflicts"]) == 1
        assert payload["summary"]["n_entities_a"] == 2
        assert payload["summary"]["n_entities_b"] == 3

    def test_explicit_output_paths_respected(self, two_kg_paths, tmp_path):
        pa, pb = two_kg_paths
        out_kg = tmp_path / "custom_kg.json"
        out_rep = tmp_path / "custom_rep.json"
        cli.cmd_cross_model(_args(pa, pb, output=str(out_kg), report=str(out_rep)))
        assert out_kg.exists()
        assert out_rep.exists()
