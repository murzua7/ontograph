"""Phase D — CLI wiring tests (TDD).

Covers the five Phase-D CLI verbs / extensions:

    * `ontograph quality`        — run gate + emit report on an existing KG
    * `ontograph cross-check`    — dual-extract LLM+AST, merge, emit KG + report
    * `ontograph ingest` extras  — --ast-repo / --dual-extract / --quality-report
    * `ontograph ground`         — post-hoc grounding pass (add anchors)
    * `ontograph diff`           — compare two KGs as markdown

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


# ── 3. `ontograph ingest` extensions ───────────────────────────────────

def _ingest_args(**overrides):
    """Build the Namespace cmd_ingest expects; defaults to heuristic mode.

    Every attribute cmd_ingest reads via getattr is supplied so tests don't
    collide with `Namespace`'s AttributeError-on-missing behaviour.
    """
    base = dict(
        input="",
        schema="base",
        output="graph.json",
        llm=False,
        llm_backend="anthropic",
        llm_model=None,
        batch_out=None,
        batch_in=None,
        schema_free=False,
        save_schema=None,
        ast_repo=None,
        dual_extract=False,
        quality_report=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestCmdIngestAstRepo:
    def test_ast_repo_adds_grounding_to_heuristic_extraction(self, tmp_path):
        """--ast-repo runs the AST extractor and unions grounded entities into the KG."""
        from ontograph.cli import cmd_ingest

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text(
            "def _phase_alpha():\n    pass\n",
            encoding="utf-8",
        )
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nNothing extractable here.\n", encoding="utf-8")

        out_kg = tmp_path / "kg.json"
        args = _ingest_args(
            input=str(doc),
            schema=str(schema_p),
            output=str(out_kg),
            ast_repo=str(repo),
        )
        cmd_ingest(args)

        assert out_kg.exists()
        payload = json.loads(out_kg.read_text(encoding="utf-8"))
        # `entities` is a dict keyed by name in the KG JSON schema.
        assert "_phase_alpha" in payload["entities"]

    def test_quality_report_written_alongside_kg(self, tmp_path):
        """--quality-report emits the six-metric JSON next to the KG."""
        from ontograph.cli import cmd_ingest

        schema_p = _write_schema(tmp_path, validation={"groundedness_target": 0.0})
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text("def _phase_alpha():\n    pass\n", encoding="utf-8")
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n", encoding="utf-8")

        out_kg = tmp_path / "kg.json"
        out_report = tmp_path / "kg.quality.json"
        args = _ingest_args(
            input=str(doc),
            schema=str(schema_p),
            output=str(out_kg),
            ast_repo=str(repo),
            quality_report=str(out_report),
        )
        cmd_ingest(args)

        assert out_report.exists()
        payload = json.loads(out_report.read_text(encoding="utf-8"))
        assert "groundedness" in payload
        assert "coverage" in payload

    def test_dual_extract_requires_ast_repo(self, tmp_path):
        """--dual-extract without --ast-repo is a usage error (nonzero rc)."""
        from ontograph.cli import cmd_ingest

        schema_p = _write_schema(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n", encoding="utf-8")
        args = _ingest_args(
            input=str(doc),
            schema=str(schema_p),
            output=str(tmp_path / "kg.json"),
            dual_extract=True,
        )
        rc = cmd_ingest(args)
        assert rc != 0

    def test_dual_extract_produces_agreement_only_kg(self, tmp_path, monkeypatch):
        """--dual-extract runs LLM+AST, merges, writes only agreement bucket to KG."""
        from ontograph import cli

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text("def _phase_alpha():\n    pass\n", encoding="utf-8")
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n", encoding="utf-8")

        # LLM sees both `_phase_alpha` (agreement) and `lonely_concept`
        # (llm-only). Only the agreement entity should end up in the KG.
        def _stub_llm_extract(doc_obj, schema, *, client=None):
            return (
                [
                    Entity(name="_phase_alpha", entity_type="phase"),
                    Entity(name="lonely_concept", entity_type="phase"),
                ],
                [],
            )
        monkeypatch.setattr(cli, "_ingest_llm_extract", _stub_llm_extract, raising=False)

        out_kg = tmp_path / "kg.json"
        args = _ingest_args(
            input=str(doc),
            schema=str(schema_p),
            output=str(out_kg),
            ast_repo=str(repo),
            dual_extract=True,
            llm=True,
        )
        rc = cli.cmd_ingest(args)
        assert rc == 0

        payload = json.loads(out_kg.read_text(encoding="utf-8"))
        assert "_phase_alpha" in payload["entities"]
        assert "lonely_concept" not in payload["entities"]


# ── 4. `ontograph ground` ──────────────────────────────────────────────

class TestCmdGround:
    def test_adds_code_anchor_to_ungrounded_entity(self, tmp_path):
        """Ungrounded KG entity whose name matches an AST symbol gets anchored."""
        from ontograph.cli import cmd_ground

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text(
            "def _phase_alpha():\n    pass\n", encoding="utf-8",
        )
        # KG entity has no anchors → not grounded
        ungrounded = Entity(name="_phase_alpha", entity_type="phase")
        assert not ungrounded.is_grounded()

        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[ungrounded])

        out_p = tmp_path / "kg.grounded.json"
        args = Namespace(
            kg=str(kg_p),
            ast_repo=str(repo),
            citation_bib=None,
            in_place=False,
            output=str(out_p),
            schema=str(schema_p),
        )
        rc = cmd_ground(args)
        assert rc == 0
        assert out_p.exists()
        payload = json.loads(out_p.read_text(encoding="utf-8"))
        assert "_phase_alpha" in payload["entities"]
        anchors = payload["entities"]["_phase_alpha"]["code_anchors"]
        assert len(anchors) >= 1
        assert anchors[0]["symbol"] == "_phase_alpha"

    def test_in_place_overwrites_input_file(self, tmp_path):
        from ontograph.cli import cmd_ground

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text("def _phase_alpha():\n    pass\n", encoding="utf-8")

        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[Entity(name="_phase_alpha", entity_type="phase")])
        args = Namespace(
            kg=str(kg_p),
            ast_repo=str(repo),
            citation_bib=None,
            in_place=True,
            output=None,
            schema=str(schema_p),
        )
        cmd_ground(args)
        payload = json.loads(kg_p.read_text(encoding="utf-8"))
        e = payload["entities"]["_phase_alpha"]
        assert len(e["code_anchors"]) >= 1

    def test_does_not_duplicate_existing_anchors(self, tmp_path):
        """Running ground twice is idempotent — anchor set doesn't grow."""
        from ontograph.cli import cmd_ground

        schema_p = _write_schema(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sim.py").write_text("def _phase_alpha():\n    pass\n", encoding="utf-8")

        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, entities=[Entity(name="_phase_alpha", entity_type="phase")])
        args = Namespace(
            kg=str(kg_p),
            ast_repo=str(repo),
            citation_bib=None,
            in_place=True,
            output=None,
            schema=str(schema_p),
        )
        cmd_ground(args)
        first = json.loads(kg_p.read_text(encoding="utf-8"))
        cmd_ground(args)
        second = json.loads(kg_p.read_text(encoding="utf-8"))
        e1 = first["entities"]["_phase_alpha"]
        e2 = second["entities"]["_phase_alpha"]
        assert len(e1["code_anchors"]) == len(e2["code_anchors"])


# ── 5. `ontograph diff` ────────────────────────────────────────────────

class TestCmdDiff:
    def test_diff_reports_added_and_removed_entities(self, tmp_path):
        from ontograph.cli import cmd_diff
        from ontograph.models import Relation

        old_p = tmp_path / "old.json"
        new_p = tmp_path / "new.json"
        _write_kg(old_p, entities=[
            _ent("a", "phase", grounded=True),
            _ent("b", "phase", grounded=True),
        ])
        _write_kg(new_p, entities=[
            _ent("a", "phase", grounded=True),
            _ent("c", "phase", grounded=True),
        ])
        out_p = tmp_path / "diff.md"
        args = Namespace(old=str(old_p), new=str(new_p), report=str(out_p))
        rc = cmd_diff(args)
        assert rc == 0
        md = out_p.read_text(encoding="utf-8")
        assert "b" in md  # removed
        assert "c" in md  # added
        # and a marker so consumers can parse sections
        assert "removed" in md.lower()
        assert "added" in md.lower()

    def test_diff_reports_added_relations(self, tmp_path):
        from ontograph.cli import cmd_diff
        from ontograph.models import Relation

        old_p = tmp_path / "old.json"
        new_p = tmp_path / "new.json"
        _write_kg(
            old_p,
            entities=[_ent("a", "phase", grounded=True), _ent("b", "phase", grounded=True)],
        )
        _write_kg(
            new_p,
            entities=[_ent("a", "phase", grounded=True), _ent("b", "phase", grounded=True)],
            relations=[Relation(
                source="a", target="b",
                relation_type="feeds_into",
                edge_class="mechanism",
            )],
        )
        out_p = tmp_path / "diff.md"
        args = Namespace(old=str(old_p), new=str(new_p), report=str(out_p))
        cmd_diff(args)
        md = out_p.read_text(encoding="utf-8")
        assert "a" in md and "b" in md and "feeds_into" in md

    def test_diff_no_changes_emits_sentinel(self, tmp_path):
        from ontograph.cli import cmd_diff

        old_p = tmp_path / "old.json"
        new_p = tmp_path / "new.json"
        ents = [_ent("a", "phase", grounded=True)]
        _write_kg(old_p, entities=ents)
        _write_kg(new_p, entities=ents)
        out_p = tmp_path / "diff.md"
        args = Namespace(old=str(old_p), new=str(new_p), report=str(out_p))
        cmd_diff(args)
        md = out_p.read_text(encoding="utf-8").lower()
        assert "no changes" in md or "identical" in md
