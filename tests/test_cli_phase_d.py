"""Phase D — CLI wiring tests (TDD).

Covers `ontograph quality` (runs gate + emits report) and `ontograph
cross-check` (dual-extract LLM+AST, merge, emit KG + merge report).

LLM calls are stubbed so these tests stay offline and deterministic.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from ontograph.models import CodeAnchor, Entity, KnowledgeGraph, Relation


# ── helpers ────────────────────────────────────────────────────────────

def _write_schema(tmp_path: Path, validation: dict | None = None) -> Path:
    body = {
        "name": "cli_sch",
        "entity_types": {
            "phase": {"description": "p"},
            "mechanism": {"description": "m"},
            "state_variable": {"description": "sv"},
            "constraint": {"description": "c"},
        },
        "relation_types": {
            "feeds_into": {"description": "f", "default_edge_class": "mechanism"},
        },
        "extraction_hints": {
            "ast_patterns": [
                {"pattern": "def _phase_*", "entity_type": "phase"},
            ],
        },
    }
    if validation is not None:
        body["validation"] = validation
    p = tmp_path / "sch.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def _write_kg(path: Path, *, entities, relations=()) -> None:
    kg = KnowledgeGraph()
    for e in entities:
        kg.add_entity(e)
    for r in relations:
        kg.add_relation(r)
    path.write_text(kg.to_json(), encoding="utf-8")


def _ent(name, etype, *, grounded=False, abstract=False):
    anchors = [CodeAnchor(repo="r", path="p.py", line=1, symbol=name)] if grounded else []
    return Entity(name=name, entity_type=etype, code_anchors=anchors, abstract=abstract)


# ── 1. `ontograph quality` ─────────────────────────────────────────────

class TestCmdQuality:
    def test_writes_report_and_returns_zero_on_pass(self, tmp_path, capsys):
        from ontograph.cli import cmd_quality
        schema_p = _write_schema(tmp_path, validation={"groundedness_target": 0.5})
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[_ent("m1", "mechanism", grounded=True)])

        args = Namespace(
            input=str(kg_p),
            schema=str(schema_p),
            ast_repo=None,
            output=None,
            allow_gate_failure=False,
        )
        rc = cmd_quality(args)
        assert rc == 0
        # report written next to the kg
        report_p = kg_p.with_suffix(".quality.json")
        assert report_p.exists()
        payload = json.loads(report_p.read_text(encoding="utf-8"))
        assert payload["gates_passed"] is True

    def test_returns_nonzero_on_gate_failure(self, tmp_path):
        from ontograph.cli import cmd_quality
        schema_p = _write_schema(tmp_path, validation={
            "require_grounding_for": ["mechanism"],
            "groundedness_target": 1.0,
        })
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[_ent("m1", "mechanism", grounded=False)])
        args = Namespace(
            input=str(kg_p),
            schema=str(schema_p),
            ast_repo=None,
            output=None,
            allow_gate_failure=False,
        )
        rc = cmd_quality(args)
        assert rc != 0

    def test_allow_gate_failure_suppresses_nonzero_exit(self, tmp_path):
        from ontograph.cli import cmd_quality
        schema_p = _write_schema(tmp_path, validation={
            "require_grounding_for": ["mechanism"],
            "groundedness_target": 1.0,
        })
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[_ent("m1", "mechanism", grounded=False)])
        args = Namespace(
            input=str(kg_p),
            schema=str(schema_p),
            ast_repo=None,
            output=None,
            allow_gate_failure=True,
        )
        rc = cmd_quality(args)
        assert rc == 0

    def test_ast_repo_feeds_coverage(self, tmp_path):
        """With --ast-repo, coverage is computed."""
        from ontograph.cli import cmd_quality
        schema_p = _write_schema(tmp_path, validation={"coverage_target": 0.5})
        # Make a minimal repo with two discoverable phases.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "simulation.py").write_text(
            "def _phase_one():\n    pass\n\ndef _phase_two():\n    pass\n",
            encoding="utf-8",
        )
        kg_p = tmp_path / "kg.json"
        # KG covers only one of the two.
        _write_kg(kg_p, entities=[_ent("_phase_one", "phase", grounded=True)])
        args = Namespace(
            input=str(kg_p),
            schema=str(schema_p),
            ast_repo=str(repo),
            output=None,
            allow_gate_failure=False,
        )
        rc = cmd_quality(args)
        report_p = kg_p.with_suffix(".quality.json")
        payload = json.loads(report_p.read_text(encoding="utf-8"))
        # 1 of 2 matched → coverage 0.5
        assert payload["coverage"]["ratio"] == pytest.approx(0.5)
        assert payload["coverage"]["total"] == 2


# ── 2. `ontograph cross-check` ─────────────────────────────────────────

class TestCmdCrossCheck:
    def test_cross_check_emits_kg_and_report(self, tmp_path, monkeypatch):
        """Cross-check merges LLM + AST extractions and writes both outputs.

        LLM extractor is stubbed to return a hand-built extraction that
        agrees with what the AST extractor will produce on a tiny repo.
        """
        from ontograph import cli

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "simulation.py").write_text(
            "def _phase_one():\n    pass\n", encoding="utf-8",
        )
        # A prose doc (its content is irrelevant — the stub ignores it).
        doc = tmp_path / "doc.md"
        doc.write_text("# doc\n", encoding="utf-8")

        def _stub_llm_extract(path_or_input, schema, *, client=None):
            return (
                [Entity(name="_phase_one", entity_type="phase")],
                [],
            )
        # monkeypatch the LLM pipeline that cmd_cross_check delegates to
        monkeypatch.setattr(cli, "_cross_check_llm_extract", _stub_llm_extract,
                            raising=False)

        out_kg = tmp_path / "merged.json"
        out_report = tmp_path / "merge_report.json"
        args = Namespace(
            input=str(doc),
            ast_repo=str(repo),
            schema=str(schema_p),
            output=str(out_kg),
            report=str(out_report),
            llm_backend="anthropic",
            llm_model=None,
        )
        rc = cli.cmd_cross_check(args)
        assert rc == 0
        assert out_kg.exists()
        assert out_report.exists()
        payload = json.loads(out_report.read_text(encoding="utf-8"))
        names = {e["name"] for e in payload["agreement_entities"]}
        assert "_phase_one" in names
