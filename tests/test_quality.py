"""Phase D — quality gate tests (TDD).

Covers `ontograph.quality.compute_quality_report(kg, schema, *, ast_entities=None)`.

Design decisions (advisor-driven, 2026-04-20):
- Gate keys are DABS-schema live: `groundedness_target`, `signedness_target`,
  `coverage_target`, `cycle_policy: {within_step, across_step}`,
  `require_grounding_for`, `allow_abstract`.
- Coverage without AST → None; skip the gate (don't fail on missing input).
- Groundedness is type-scoped via `require_grounding_for` + `allow_abstract`.
- Sign conflict = `+` vs `-` only. `unknown ↔ anything` is NOT a conflict
  (spec §2 principle 2). `±` matches-nothing (safer).
- `cycle_policy.within_step: forbid` is enforced; `across_step: allow` is
  permissive (not required-present). Spec §1.2 criterion 4's "≥1 across_step"
  is not enforced until dabs.yaml vocabulary extends to `{forbid, allow, require}`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ── helpers ────────────────────────────────────────────────────────────

def _schema(tmp_path: Path, validation: dict | None = None,
            extra_entity_types: list[str] | None = None,
            extra_relation_types: dict | None = None) -> "OntologySchema":
    """Write a minimal schema with the given validation block and return it loaded."""
    from ontograph.schema import load_schema
    etypes = {
        "phase": {"description": "p"},
        "mechanism": {"description": "m"},
        "constraint": {"description": "c"},
        "theory": {"description": "t"},
        "concept": {"description": "c"},
        "state_variable": {"description": "sv"},
    }
    for t in (extra_entity_types or []):
        etypes[t] = {"description": t}
    rtypes = {
        "feeds_into": {"description": "f", "default_edge_class": "mechanism"},
        "asserts": {"description": "a", "default_edge_class": "identity"},
        "executes_after": {"description": "e", "default_edge_class": "structural"},
    }
    for name, defn in (extra_relation_types or {}).items():
        rtypes[name] = defn
    body = {
        "name": "q",
        "entity_types": etypes,
        "relation_types": rtypes,
    }
    if validation is not None:
        body["validation"] = validation
    p = tmp_path / "sch.yaml"
    p.write_text(yaml.safe_dump(body))
    return load_schema(str(p))


def _kg(entities: list, relations: list) -> "KnowledgeGraph":
    from ontograph.models import KnowledgeGraph
    kg = KnowledgeGraph()
    for e in entities:
        kg.add_entity(e)
    for r in relations:
        kg.add_relation(r)
    return kg


def _mk_entity(name, etype, *, grounded=False, abstract=False, rationale=""):
    from ontograph.models import Entity, CodeAnchor
    anchors = [CodeAnchor(repo="r", path="p.py", line=1, symbol=name)] if grounded else []
    return Entity(
        name=name, entity_type=etype,
        code_anchors=anchors, abstract=abstract,
        abstract_rationale=rationale,
    )


def _mk_rel(src, tgt, rtype="feeds_into", *, sign="unknown", lag="unknown",
            edge_class="mechanism"):
    from ontograph.models import Relation
    return Relation(
        source=src, target=tgt, relation_type=rtype,
        sign=sign, lag=lag, edge_class=edge_class,
    )


# ── 1. Groundedness ────────────────────────────────────────────────────

class TestGroundedness:
    def test_type_scoped_via_require_grounding_for(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "require_grounding_for": ["mechanism", "phase"],
            "allow_abstract": ["theory", "concept"],
            "groundedness_target": 0.8,
        })
        kg = _kg([
            _mk_entity("m1", "mechanism", grounded=True),
            _mk_entity("m2", "mechanism", grounded=True),
            _mk_entity("m3", "mechanism", grounded=False),   # ungrounded, required
            _mk_entity("th1", "theory", abstract=True, rationale="pure concept"),
            _mk_entity("sv1", "state_variable", grounded=False),  # not required
        ], [])
        rep = compute_quality_report(kg, schema)
        # Denominator = mechanisms that are not abstract (3 of them). Numerator = 2 grounded.
        assert rep.groundedness["total"] == 3
        assert rep.groundedness["grounded"] == 2
        assert rep.groundedness["ratio"] == pytest.approx(2 / 3)
        # 0.666 < 0.8 → gate fails
        assert rep.gates_passed is False
        assert any("groundedness" in f.lower() for f in rep.failures)

    def test_abstract_excluded_from_denominator_if_allowed(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "require_grounding_for": ["theory"],
            "allow_abstract": ["theory"],
            "groundedness_target": 1.0,
        })
        kg = _kg([
            _mk_entity("th1", "theory", abstract=True, rationale="pure"),
            _mk_entity("th2", "theory", grounded=True),
        ], [])
        rep = compute_quality_report(kg, schema)
        # th1 is abstract+allowed → excluded; th2 is concrete+grounded → included, grounded.
        assert rep.groundedness["total"] == 1
        assert rep.groundedness["grounded"] == 1
        assert rep.groundedness["ratio"] == 1.0

    def test_global_fallback_when_no_require_grounding_for(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"groundedness_target": 0.5})
        kg = _kg([
            _mk_entity("a", "mechanism", grounded=True),
            _mk_entity("b", "mechanism", grounded=False),
        ], [])
        rep = compute_quality_report(kg, schema)
        assert rep.groundedness["total"] == 2
        assert rep.groundedness["grounded"] == 1
        assert rep.groundedness["ratio"] == 0.5

    def test_by_type_breakdown_present(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "require_grounding_for": ["mechanism", "phase"],
        })
        kg = _kg([
            _mk_entity("m1", "mechanism", grounded=True),
            _mk_entity("m2", "mechanism", grounded=False),
            _mk_entity("p1", "phase", grounded=True),
        ], [])
        rep = compute_quality_report(kg, schema)
        by_type = rep.groundedness["by_type"]
        assert by_type["mechanism"] == pytest.approx(0.5)
        assert by_type["phase"] == 1.0


# ── 2. Signedness ──────────────────────────────────────────────────────

class TestSignedness:
    def test_mechanism_edges_with_sign_counted(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"signedness_target": 0.5})
        rels = [
            _mk_rel("a", "b", sign="+", edge_class="mechanism"),
            _mk_rel("a", "c", sign="-", edge_class="mechanism"),
            _mk_rel("a", "d", sign="unknown", edge_class="mechanism"),
            _mk_rel("a", "e", sign="unknown", edge_class="structural"),  # not mechanism
        ]
        kg = _kg([_mk_entity(n, "state_variable") for n in ["a", "b", "c", "d", "e"]], rels)
        rep = compute_quality_report(kg, schema)
        # Mechanism edges: 3 (to b/c/d). Signed: 2 (+, -). unknown excluded.
        assert rep.signedness["total"] == 3
        assert rep.signedness["signed"] == 2
        assert rep.signedness["ratio"] == pytest.approx(2 / 3)
        # 0.666 >= 0.5 → pass
        assert rep.gates_passed is True

    def test_no_mechanism_edges_is_vacuous_pass(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"signedness_target": 0.7})
        kg = _kg([_mk_entity("a", "state_variable")], [])
        rep = compute_quality_report(kg, schema)
        assert rep.signedness["total"] == 0
        assert rep.signedness["ratio"] == 1.0

    def test_ambiguous_and_zero_count_as_signed(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"signedness_target": 1.0})
        rels = [
            _mk_rel("a", "b", sign="±", edge_class="mechanism"),
            _mk_rel("a", "c", sign="0", edge_class="mechanism"),
        ]
        kg = _kg([_mk_entity(n, "state_variable") for n in ["a", "b", "c"]], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.signedness["signed"] == 2


# ── 3. Coverage ────────────────────────────────────────────────────────

class TestCoverage:
    def test_coverage_none_when_no_ast_entities(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"coverage_target": 0.85})
        kg = _kg([_mk_entity("m1", "mechanism", grounded=True)], [])
        rep = compute_quality_report(kg, schema)  # no ast_entities
        assert rep.coverage["ratio"] is None
        assert not any("coverage" in f.lower() for f in rep.failures)

    def test_coverage_computed_when_ast_entities_passed(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"coverage_target": 0.5})
        ast_ents = [
            _mk_entity("m1", "mechanism", grounded=True),
            _mk_entity("m2", "mechanism", grounded=True),
            _mk_entity("p1", "phase", grounded=True),
            _mk_entity("p2", "phase", grounded=True),
        ]
        # KG covers 2 of 4
        kg = _kg([
            _mk_entity("m1", "mechanism", grounded=True),
            _mk_entity("p1", "phase", grounded=True),
        ], [])
        rep = compute_quality_report(kg, schema, ast_entities=ast_ents)
        assert rep.coverage["matched"] == 2
        assert rep.coverage["total"] == 4
        assert rep.coverage["ratio"] == 0.5

    def test_coverage_fails_gate_when_below_target(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"coverage_target": 0.85})
        ast_ents = [_mk_entity(f"m{i}", "mechanism", grounded=True) for i in range(10)]
        kg = _kg([_mk_entity("m0", "mechanism", grounded=True)], [])
        rep = compute_quality_report(kg, schema, ast_entities=ast_ents)
        assert rep.coverage["ratio"] == 0.1
        assert rep.gates_passed is False
        assert any("coverage" in f.lower() for f in rep.failures)

    def test_coverage_uses_fuzzy_match_consistent_with_merge(self, tmp_path):
        """Coverage must use the same fuzzy predicate as merge_extractions.

        Regression guard (advisor 2026-04-20): on real DABS regen the LLM
        will emit names like 'alpha phase' while the AST emits '_phase_alpha'.
        Those two *would* merge via `default_name_matcher`; coverage must
        therefore count this as a hit, not a miss.
        """
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"coverage_target": 0.99})
        ast_ents = [_mk_entity("alpha_phase", "phase", grounded=True)]
        kg = _kg([_mk_entity("alpha phase", "phase", grounded=True)], [])
        rep = compute_quality_report(kg, schema, ast_entities=ast_ents)
        assert rep.coverage["matched"] == 1
        assert rep.coverage["ratio"] == 1.0
        assert rep.gates_passed is True


# ── 4. Orphan rate ─────────────────────────────────────────────────────

class TestOrphanRate:
    def test_orphan_is_entity_with_no_incident_edge(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={})
        rels = [_mk_rel("a", "b", edge_class="mechanism")]
        kg = _kg([
            _mk_entity("a", "mechanism", grounded=True),
            _mk_entity("b", "state_variable"),
            _mk_entity("c", "mechanism", grounded=True),  # orphan
        ], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.orphan_rate["orphans"] == 1
        assert rep.orphan_rate["total"] == 3
        assert rep.orphan_rate["ratio"] == pytest.approx(1 / 3)

    def test_abstract_entities_excluded_from_orphan_count(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={})
        kg = _kg([
            _mk_entity("th", "theory", abstract=True, rationale="pure"),
        ], [])
        rep = compute_quality_report(kg, schema)
        assert rep.orphan_rate["orphans"] == 0


# ── 5. Consistency (sign conflicts) ────────────────────────────────────

class TestConsistency:
    def test_plus_vs_minus_is_conflict(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={})
        rels = [
            _mk_rel("a", "b", sign="+", edge_class="mechanism"),
            _mk_rel("a", "b", sign="-", edge_class="mechanism"),
        ]
        kg = _kg([_mk_entity(n, "state_variable") for n in ["a", "b"]], rels)
        rep = compute_quality_report(kg, schema)
        assert len(rep.consistency["conflicts"]) == 1
        # consistency = 1 - 1/1 = 0
        assert rep.consistency["ratio"] == 0.0

    def test_unknown_vs_plus_is_not_conflict(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={})
        rels = [
            _mk_rel("a", "b", sign="+", edge_class="mechanism"),
            _mk_rel("a", "b", sign="unknown", edge_class="mechanism"),
        ]
        kg = _kg([_mk_entity(n, "state_variable") for n in ["a", "b"]], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.consistency["conflicts"] == []
        assert rep.consistency["ratio"] == 1.0

    def test_ambiguous_is_not_conflict_with_plus(self, tmp_path):
        """±  matches-nothing convention: not a conflict against + (safer)."""
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={})
        rels = [
            _mk_rel("a", "b", sign="+", edge_class="mechanism"),
            _mk_rel("a", "b", sign="±", edge_class="mechanism"),
        ]
        kg = _kg([_mk_entity(n, "state_variable") for n in ["a", "b"]], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.consistency["conflicts"] == []


# ── 6. Cycle count (decomposed by lag) ─────────────────────────────────

class TestCycleCount:
    def test_within_step_cycle_flagged(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "cycle_policy": {"within_step": "forbid", "across_step": "allow"},
        })
        rels = [
            _mk_rel("a", "b", lag="within_step"),
            _mk_rel("b", "a", lag="within_step"),
        ]
        kg = _kg([_mk_entity(n, "mechanism", grounded=True) for n in ["a", "b"]], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.cycle_count["within_step"] >= 1
        assert rep.gates_passed is False
        assert any("cycle" in f.lower() and "within" in f.lower() for f in rep.failures)

    def test_across_step_cycle_allowed(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "cycle_policy": {"within_step": "forbid", "across_step": "allow"},
            "groundedness_target": 0.0,
            "signedness_target": 0.0,
        })
        rels = [
            _mk_rel("a", "b", lag="across_step"),
            _mk_rel("b", "a", lag="across_step"),
        ]
        kg = _kg([_mk_entity(n, "mechanism", grounded=True) for n in ["a", "b"]], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.cycle_count["across_step"] >= 1
        assert rep.cycle_count["within_step"] == 0
        assert rep.gates_passed is True


# ── 7. Full gate integration ───────────────────────────────────────────

class TestGateIntegration:
    def test_all_gates_pass(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={
            "require_grounding_for": ["mechanism"],
            "allow_abstract": ["theory"],
            "groundedness_target": 0.8,
            "signedness_target": 0.5,
            "coverage_target": 0.5,
            "cycle_policy": {"within_step": "forbid", "across_step": "allow"},
        })
        rels = [
            _mk_rel("a", "b", sign="+", edge_class="mechanism"),
            _mk_rel("b", "c", sign="-", edge_class="mechanism"),
        ]
        kg = _kg([
            _mk_entity("a", "mechanism", grounded=True),
            _mk_entity("b", "mechanism", grounded=True),
            _mk_entity("c", "state_variable"),
        ], rels)
        rep = compute_quality_report(kg, schema)
        assert rep.gates_passed is True
        assert rep.failures == []

    def test_empty_validation_block_passes_vacuously(self, tmp_path):
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation=None)
        kg = _kg([_mk_entity("a", "mechanism")], [])
        rep = compute_quality_report(kg, schema)
        assert rep.gates_passed is True

    def test_to_json_roundtrips_all_fields(self, tmp_path):
        """QualityReport must serialise to JSON for `<kg>.quality.json` emission."""
        import json
        from ontograph.quality import compute_quality_report
        schema = _schema(tmp_path, validation={"groundedness_target": 0.8})
        kg = _kg([_mk_entity("a", "mechanism", grounded=True)], [])
        rep = compute_quality_report(kg, schema)
        payload = json.loads(rep.to_json())
        assert "groundedness" in payload
        assert "signedness" in payload
        assert "coverage" in payload
        assert "orphan_rate" in payload
        assert "consistency" in payload
        assert "cycle_count" in payload
        assert "gates_passed" in payload
        assert "failures" in payload
