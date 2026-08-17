"""Tests for `ontograph reprompt-signs` (#45)."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from ontograph import cli
from ontograph.llm_client import LLMClient
from ontograph.models import Entity, KnowledgeGraph, Provenance, Relation


def _kg_with_unknowns_file(tmp_path, n: int = 2):
    kg = KnowledgeGraph()
    for i in range(n):
        a, b = f"a{i}", f"b{i}"
        kg.add_entity(Entity(name=a, entity_type="variable"))
        kg.add_entity(Entity(name=b, entity_type="variable"))
        kg.add_relation(Relation(
            source=a, target=b, relation_type="r",
            sign="unknown", lag="within_step", edge_class="mechanism",
            provenance=[Provenance(document="d", passage=f"passage {i}")],
        ))
    p = tmp_path / "kg.json"
    p.write_text(kg.to_json(), encoding="utf-8")
    return p


def _args(input_path, **overrides):
    base = dict(
        input=str(input_path), output=None, report=None,
        min_confidence=0.6,
        llm_backend="mock", llm_model=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestCmdReprompt:
    def test_returns_zero_and_writes_updated_kg(self, tmp_path, monkeypatch):
        p = _kg_with_unknowns_file(tmp_path, n=2)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses([
                '{"sign": "+", "confidence": 0.9}',
                '{"sign": "-", "confidence": 0.8}',
            ])
            return c

        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)

        rc = cli.cmd_reprompt_signs(_args(p))
        assert rc == 0

        out_kg = p.with_suffix(".reprompted.json")
        assert out_kg.exists()
        kg = KnowledgeGraph.from_json(out_kg.read_text(encoding="utf-8"))
        signs = sorted(r.sign for r in kg.relations)
        assert signs == ["+", "-"]

    def test_report_written_next_to_output(self, tmp_path, monkeypatch):
        p = _kg_with_unknowns_file(tmp_path, n=1)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses(['{"sign": "+", "confidence": 0.9}'])
            return c
        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)

        cli.cmd_reprompt_signs(_args(p))
        report_path = p.with_suffix(".reprompt_report.json")
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["n_candidates"] == 1
        assert report["n_updated"] == 1

    def test_explicit_output_paths_respected(self, tmp_path, monkeypatch):
        p = _kg_with_unknowns_file(tmp_path, n=1)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses(['{"sign": "+", "confidence": 0.9}'])
            return c
        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)

        custom_out = tmp_path / "new_kg.json"
        custom_rep = tmp_path / "new_report.json"
        cli.cmd_reprompt_signs(_args(
            p, output=str(custom_out), report=str(custom_rep),
        ))
        assert custom_out.exists()
        assert custom_rep.exists()
        # Default location should NOT be written when an explicit path is given.
        assert not p.with_suffix(".reprompted.json").exists()

    def test_low_confidence_leaves_sign_unchanged(self, tmp_path, monkeypatch):
        p = _kg_with_unknowns_file(tmp_path, n=1)

        def _stub_client(args):
            c = LLMClient(mode="mock")
            c.set_mock_responses(['{"sign": "+", "confidence": 0.3}'])
            return c
        monkeypatch.setattr(cli, "_make_llm_client", _stub_client, raising=False)

        cli.cmd_reprompt_signs(_args(p))
        out_kg = p.with_suffix(".reprompted.json")
        kg = KnowledgeGraph.from_json(out_kg.read_text(encoding="utf-8"))
        assert kg.relations[0].sign == "unknown"
