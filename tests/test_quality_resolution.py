"""Anchor resolution (flag-gated) - regression fixtures from the
content-blindness probe of 2026-08-16 (knowledge-vault
outputs/probes/2026-08-16-ontograph-gate-content-blindness).

The probe's four claims, rebuilt at HEAD with code anchors (the probe's A and
B used document anchors, which exist only on the rescued 2026-04-21 branch;
their classes are represented here with code anchors so the fixtures bind at
HEAD):
  A  supported                anchor into a real file, real line, real symbol
  B  grounded-but-unrelated   anchor into a REAL file whose content has nothing
                              to do with the claim - resolution must PASS it,
                              because content belongs to the judged axis
  C  baseless                 no anchor (groundedness catches it, unchanged)
  D  grounded-by-phantom      the probe's exact fabricated anchor - resolution
                              must FAIL it, which is the class this pass exists
                              to kill
"""
import textwrap

import pytest
import yaml

from ontograph.models import KnowledgeGraph, Entity, Relation, CodeAnchor
from ontograph.schema import load_schema
from ontograph.quality import compute_quality_report

REAL_CODE = textwrap.dedent(
    """\
    def inflation_target():
        return 0.02

    def football_result():
        return "3-1"
    """
)


@pytest.fixture()
def schema(tmp_path):
    body = {
        "name": "probe",
        "entity_types": {"state_variable": {"description": "sv"}},
        "relation_types": {"feeds_into": {"description": "f",
                                          "default_edge_class": "mechanism"}},
        "validation": {
            "require_grounding_for": ["state_variable"],
            "groundedness_target": 0.9,
        },
    }
    p = tmp_path / "sch.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_schema(str(p))


@pytest.fixture()
def repo_root(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "econ.py").write_text(REAL_CODE, encoding="utf-8")
    return tmp_path


def _ent(name, code=None):
    return Entity(name=name, entity_type="state_variable",
                  code_anchors=code or [])


def _build(entities):
    kg = KnowledgeGraph()
    for e in entities:
        kg.add_entity(e)
    names = [e.name for e in entities]
    for a, b in zip(names, names[1:]):
        kg.add_relation(Relation(source=a, target=b, relation_type="feeds_into",
                                 sign="+", lag="unknown", edge_class="mechanism"))
    return kg


def _fixtures():
    a = _ent("inflation_target",
             code=[CodeAnchor(repo="dabs", path="real/econ.py", line=1,
                              symbol="inflation_target")])
    b = _ent("mortgage_demand",
             code=[CodeAnchor(repo="dabs", path="real/econ.py", line=4,
                              symbol="football_result")])
    c = _ent("phantom_channel")
    d = _ent("credit_spread",
             code=[CodeAnchor(repo="dabs", path="no/such/file_anywhere.py",
                              line=7, symbol="gpu_memory_pool")])
    return a, b, c, d


def test_flag_off_preserves_the_measured_blindness(schema, repo_root):
    """Default OFF: the probe's run-1 result reproduces at HEAD - the phantom
    anchor passes every gate. The blindness is preserved, not silently fixed,
    because existing callers must see identical verdicts."""
    a, b, _, d = _fixtures()
    report = compute_quality_report(_build([a, b, d]), schema)
    assert report.gates_passed, report.failures
    assert report.resolution == {"enabled": False, "checked": 0}
    assert report.groundedness["grounded"] == 3


def test_flag_on_kills_the_fabricated_anchor_class(schema, repo_root):
    a, b, _, d = _fixtures()
    report = compute_quality_report(
        _build([a, b, d]), schema,
        resolve_anchors=True, anchor_roots={"dabs": str(repo_root)})
    assert not report.gates_passed
    assert any("anchor_resolution" in f for f in report.failures)
    bad = report.resolution["unresolvable"]
    assert [u["entity"] for u in bad] == ["credit_spread"]
    assert bad[0]["reason"] == "file missing"
    assert report.resolution["checked"] == 3
    assert report.resolution["resolved"] == 2


def test_resolution_is_existence_never_content(schema, repo_root):
    """The B fixture: a real anchor about football under a mortgage claim
    RESOLVES. Claim-evidence congruence stays the judged content axis's job,
    per the probe's division of labour."""
    _, b, _, _ = _fixtures()
    report = compute_quality_report(
        _build([b]), schema,
        resolve_anchors=True, anchor_roots={"dabs": str(repo_root)})
    assert report.gates_passed, report.failures
    assert report.resolution["unresolvable"] == []


def test_anchorless_entity_stays_groundedness_business(schema, repo_root):
    """The C fixture is caught by groundedness exactly as in probe run 2; the
    resolution pass has nothing to check on it."""
    a, b, c, d = _fixtures()
    report = compute_quality_report(
        _build([a, b, c, d]), schema,
        resolve_anchors=True, anchor_roots={"dabs": str(repo_root)})
    assert not report.gates_passed
    assert any("groundedness" in f for f in report.failures)
    assert any("anchor_resolution" in f for f in report.failures)


def test_line_out_of_range_and_missing_symbol_and_unknown_root(schema, repo_root):
    over = _ent("over", code=[CodeAnchor(repo="dabs", path="real/econ.py",
                                         line=999)])
    nosym = _ent("nosym", code=[CodeAnchor(repo="dabs", path="real/econ.py",
                                           line=1, symbol="not_here_at_all")])
    noroot = _ent("noroot", code=[CodeAnchor(repo="elsewhere", path="x.py",
                                             line=1)])
    report = compute_quality_report(
        _build([over, nosym, noroot]), schema,
        resolve_anchors=True, anchor_roots={"dabs": str(repo_root)})
    reasons = {u["entity"]: u["reason"] for u in report.resolution["unresolvable"]}
    assert reasons["over"].startswith("line 999 out of range")
    assert reasons["nosym"] == "symbol 'not_here_at_all' not found"
    assert reasons["noroot"] == "unknown repo root"


def test_report_json_carries_the_resolution_block(schema, repo_root):
    a, *_ = _fixtures()
    report = compute_quality_report(
        _build([a]), schema,
        resolve_anchors=True, anchor_roots={"dabs": str(repo_root)})
    assert '"resolution"' in report.to_json()
