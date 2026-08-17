"""Tests for `ontograph falsifier-reprompt` (#49)."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from ontograph import cli
from ontograph.llm_client import LLMClient
from ontograph.models import Entity, KnowledgeGraph, Provenance, Relation


def _kg_with_flipped(tmp_path):
    kg = KnowledgeGraph()
    for n in ("A", "B", "C", "D"):
        kg.add_entity(Entity(name=n, entity_type="variable"))
    kg.add_relation(Relation(
        source="A", target="B", relation_type="r", sign="+",
        lag="within_step", edge_class="mechanism",
        provenance=[Provenance(document="d", passage="A unambiguously depresses B.")],
    ))
    kg.add_relation(Relation(
        source="C", target="D", relation_type="r", sign="+",
        lag="within_step", edge_class="mechanism",
        provenance=[Provenance(document="d", passage="C raises D.")],
    ))
    kg_path = tmp_path / "kg.json"
    kg_path.write_text(kg.to_json(), encoding="utf-8")
    return kg_path


def _falsify_report(tmp_path):
    """One flipped edge (A→B) and one confirmed (C→D)."""
    payload = {
        "confirmed": [{
            "source": "C", "target": "D", "relation_type": "r",
            "edge_class": "mechanism",
            "kg_sign": "+", "sim_sign": "+",
            "kg_lag": "within_step", "sim_lag": "within_step",
        }],
        "flipped": [{
            "source": "A", "target": "B", "relation_type": "r",
            "edge_class": "mechanism",
            "kg_sign": "+", "sim_sign": "-",
            "kg_lag": "within_step", "sim_lag": "within_step",
        }],
        "unknown": [],
        "total": 2,
        "examined": 2,
    }
    p = tmp_path / "falsify.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _args(kg_path, report_path, **overrides):
    base = dict(
        kg=str(kg_path),
        falsify_report=str(report_path),
        output=None, report=None,
        min_confidence=0.6,
        llm_backend="mock", llm_model=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestCmdFalsifierReprompt:
    def test_returns_zero_and_updates_flipped_edge(self, tmp_path, monkeypatch):
        kg_p = _kg_with_flipped(tmp_path)
        rep_p = _falsify_report(tmp_path)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses([
                '{"sign": "-", "confidence": 0.9, "rationale": "passage says depresses"}',
            ])
            return c

        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)
        rc = cli.cmd_falsifier_reprompt(_args(kg_p, rep_p))
        assert rc == 0

        out_kg_path = kg_p.with_suffix(".falsifier_reprompted.json")
        assert out_kg_path.exists()
        kg = KnowledgeGraph.from_json(out_kg_path.read_text(encoding="utf-8"))
        signs = {(r.source, r.target): r.sign for r in kg.relations}
        assert signs[("A", "B")] == "-"      # flipped + revised
        assert signs[("C", "D")] == "+"      # confirmed, untouched

    def test_report_file_written(self, tmp_path, monkeypatch):
        kg_p = _kg_with_flipped(tmp_path)
        rep_p = _falsify_report(tmp_path)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses([
                '{"sign": "-", "confidence": 0.9, "rationale": "x"}',
            ])
            return c

        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)
        rc = cli.cmd_falsifier_reprompt(_args(kg_p, rep_p))
        assert rc == 0
        rep_out = kg_p.with_suffix(".falsifier_reprompt_report.json")
        assert rep_out.exists()
        d = json.loads(rep_out.read_text(encoding="utf-8"))
        assert d["n_updated"] == 1
        assert d["n_candidates"] == 1

    def test_missing_report_returns_nonzero(self, tmp_path):
        kg_p = _kg_with_flipped(tmp_path)
        bad = tmp_path / "nope.json"
        rc = cli.cmd_falsifier_reprompt(_args(kg_p, bad))
        assert rc == 1

    def test_malformed_falsify_report_returns_nonzero(self, tmp_path):
        kg_p = _kg_with_flipped(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("{not even json", encoding="utf-8")
        rc = cli.cmd_falsifier_reprompt(_args(kg_p, bad))
        assert rc == 1
