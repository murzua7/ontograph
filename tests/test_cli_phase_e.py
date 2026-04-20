"""Phase E — CLI wiring tests (TDD).

Covers `ontograph falsify --kg <path> --simulator module:Class
[--subset {mechanism,all}] [--report <path>]`.
"""

from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from ontograph.models import Entity, KnowledgeGraph, Relation


def _write_kg(path: Path, relations) -> None:
    kg = KnowledgeGraph()
    names = {r.source for r in relations} | {r.target for r in relations}
    for n in names:
        kg.add_entity(Entity(name=n, entity_type="mechanism"))
    for r in relations:
        kg.add_relation(r)
    path.write_text(kg.to_json(), encoding="utf-8")


def _rel(src, tgt, *, sign="+", edge_class="mechanism"):
    return Relation(
        source=src, target=tgt, relation_type="feeds_into",
        sign=sign, lag="within_step", edge_class=edge_class,
    )


def _install_stub_adapter_module(monkeypatch, *, signs):
    """Install a synthetic module exposing a `StubAdapter` the CLI can import."""

    class StubAdapter:
        def __init__(self, *args, **kwargs):
            self._signs = signs

        def edge_sign(self, source, target):
            return self._signs.get((source, target), "unknown")

        def edge_lag(self, source, target):
            return "within_step"

    module = types.ModuleType("tests._stub_adapter")
    module.StubAdapter = StubAdapter
    monkeypatch.setitem(sys.modules, "tests._stub_adapter", module)
    return "tests._stub_adapter:StubAdapter"


class TestCmdFalsify:
    def test_writes_report_and_buckets_edges(self, tmp_path, monkeypatch):
        from ontograph.cli import cmd_falsify

        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, [
            _rel("a", "b", sign="+"),   # confirmed
            _rel("c", "d", sign="+"),   # flipped
        ])
        simulator = _install_stub_adapter_module(monkeypatch, signs={
            ("a", "b"): "+",
            ("c", "d"): "-",
        })

        out_p = tmp_path / "falsify.json"
        args = Namespace(
            kg=str(kg_p),
            simulator=simulator,
            subset="mechanism",
            report=str(out_p),
        )
        rc = cmd_falsify(args)
        assert rc == 0
        assert out_p.exists()

        payload = json.loads(out_p.read_text(encoding="utf-8"))
        assert len(payload["confirmed"]) == 1
        assert len(payload["flipped"]) == 1
        assert payload["examined"] == 2

    def test_invalid_simulator_spec_rc_nonzero(self, tmp_path):
        from ontograph.cli import cmd_falsify
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, [_rel("a", "b")])
        args = Namespace(
            kg=str(kg_p),
            simulator="not_a_module:NoClass",
            subset="mechanism",
            report=str(tmp_path / "out.json"),
        )
        rc = cmd_falsify(args)
        assert rc != 0

    def test_subset_all_examines_non_mechanism_edges(self, tmp_path, monkeypatch):
        from ontograph.cli import cmd_falsify

        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, [
            _rel("a", "b", sign="+", edge_class="mechanism"),
            _rel("a", "b", sign="+", edge_class="identity"),
        ])
        simulator = _install_stub_adapter_module(monkeypatch, signs={
            ("a", "b"): "+",
        })
        out_p = tmp_path / "falsify.json"
        args = Namespace(
            kg=str(kg_p),
            simulator=simulator,
            subset="all",
            report=str(out_p),
        )
        cmd_falsify(args)
        payload = json.loads(out_p.read_text(encoding="utf-8"))
        assert payload["examined"] == 2
