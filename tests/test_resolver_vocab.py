"""T3 repair tests: vocabulary normalization stage, canonical alias file,
and the zero-silent-fusions invariant (every absorbed entity logged once)."""
from __future__ import annotations

from pathlib import Path

from ontograph.models import Entity, KnowledgeGraph, Provenance
from ontograph.resolver import (
    load_canonical_aliases,
    merge_knowledge_graphs,
    normalize_label,
    resolve_entities,
)

SCHEMAS = Path(__file__).resolve().parent.parent / "vocab"


def _ent(name, etype="state_variable", doc="d.pdf"):
    return Entity(
        name=name, entity_type=etype,
        provenance=[Provenance(document=doc, section="p.1", passage="x" * 20)],
    )


def test_normalize_label_cases():
    assert normalize_label("Commercial Banks") == "commercial bank"
    assert normalize_label("  The  Phillips   Curve ") == "phillips curve"
    assert normalize_label("A Central Bank") == "central bank"
    assert normalize_label("An Economy") == "economy"
    assert normalize_label("Inflation Expectations") == "inflation expectation"
    assert normalize_label("Federal Funds Rate") == "federal funds rate"
    assert normalize_label("Glass") == "glass"  # -ss guard
    assert normalize_label("Tax") == "tax"  # short-word guard


def test_normalized_stage_merges_plural_variants():
    log = []
    out, ren = resolve_entities(
        [_ent("Corporate Bond Spread"), _ent("Corporate bond spreads", doc="e.pdf")],
        fusion_log=log,
    )
    assert len(out) == 1
    assert len(log) == 1
    assert log[0]["stage"] == "normalized"
    assert log[0]["absorbed"] == "Corporate bond spreads"
    assert log[0]["sources"] == ["e.pdf"]
    assert ren["Corporate bond spreads"] == "Corporate Bond Spread"


def test_exact_stage_still_first_and_logged():
    log = []
    out, _ = resolve_entities(
        [_ent("Output gap"), _ent("OUTPUT GAP", doc="e.pdf")], fusion_log=log,
    )
    assert len(out) == 1 and log[0]["stage"] == "exact"


def test_cross_type_never_fuses():
    log = []
    out, _ = resolve_entities(
        [_ent("Government spending", "mechanism"), _ent("government spending", "state_variable")],
        fusion_log=log,
    )
    assert len(out) == 2 and log == []


def test_fuzzy_stage_logged():
    log = []
    out, _ = resolve_entities(
        [_ent("Phillips curve slope"), _ent("Phillips-curve slope", doc="e.pdf")], fusion_log=log,
    )
    assert len(out) == 1
    assert log[0]["stage"] in ("normalized", "fuzzy")
    assert log[0]["canonical"] == "Phillips curve slope"


def test_zero_silent_fusions_invariant():
    ents = [
        _ent("Output gap"), _ent("OUTPUT GAP", doc="b.pdf"),
        _ent("Corporate Bond Spread"), _ent("Corporate bond spreads", doc="b.pdf"),
        _ent("Phillips curve"), _ent("Philips curve", doc="c.pdf"),  # fuzzy typo
        _ent("Unique entity"),
        _ent("Government spending", "mechanism"),
        _ent("government spending", "state_variable"),  # cross-type: kept
    ]
    log = []
    out, _ = resolve_entities(ents, fusion_log=log)
    assert len(out) + len(log) == len(ents)
    assert len(out) == 6
    assert sorted(e["stage"] for e in log) == ["exact", "fuzzy", "normalized"]
    assert {e["absorbed"] for e in log} == {
        "OUTPUT GAP", "Corporate bond spreads", "Philips curve"}


def test_canonical_yaml_loads_reviewed_aliases():
    """2026-09-05: the vocabulary carries hand-reviewed assertions; every alias
    maps to a non-empty canonical, no alias is its own canonical, and the
    exact stage still runs before the alias stage."""
    aliases = load_canonical_aliases(SCHEMAS / "canonical_entities.yaml")
    assert len(aliases) >= 60
    assert all(isinstance(v, str) and v for v in aliases.values())
    assert all(k != v.lower() for k, v in aliases.items())
    # antonym pairs must never be asserted
    for a, b in [("skilled employment", "unskilled employment"), ("mean net worth", "median net worth"),
                 ("hank model", "rank model"), ("liquid wealth", "illiquid wealth")]:
        assert aliases.get(a, a).lower() != aliases.get(b, b).lower()
    log = []
    out, _ = resolve_entities(
        [_ent("Central Bank"), _ent("Central bank", doc="e.pdf")],
        aliases=aliases, fusion_log=log,
    )
    assert len(out) == 1 and log[0]["stage"] == "exact"


def test_hand_alias_fuses_and_logs():
    log = []
    out, ren = resolve_entities(
        [_ent("Real GDP"), _ent("Real output", doc="e.pdf")],
        aliases={"real output": "Real GDP"}, fusion_log=log,
    )
    assert len(out) == 1 and log[0]["stage"] == "alias"
    assert ren["Real output"] == "Real GDP"


def test_merge_passthrough_collects_log():
    def _kg(doc, name):
        kg = KnowledgeGraph(schema_name="economics", documents=[doc])
        kg.add_entity(_ent(name, doc=doc))
        return kg

    log = []
    merged = merge_knowledge_graphs(
        [_kg("a.pdf", "Output gap"), _kg("b.pdf", "Output gaps")], fusion_log=log)
    assert len(merged.entities) == 1 and len(log) == 1
    assert log[0] == {
        "canonical": "Output gap", "absorbed": "Output gaps",
        "stage": "normalized", "sources": ["b.pdf"],
    }


def test_no_false_normalized_fusion_after_merge_churn():
    # Regression: norm cache keyed by id() returned stale norms once merged
    # objects were garbage-collected and their ids reused, fusing e.g.
    # 'precautionary saving' into 'target buffer-stock wealth'. Churn many
    # merges first, then verify two distinct same-type labels stay separate.
    ents = [_ent(f"Churn entity {i}", doc=f"c{i}.pdf") for i in range(30)]
    ents += [_ent("Churn entity 0", doc="dup.pdf")] * 1
    ents += [_ent("precautionary saving"), _ent("target buffer-stock wealth")]
    log = []
    out, _ = resolve_entities(ents, fusion_log=log)
    names = {e.name for e in out}
    assert "precautionary saving" in names
    assert "target buffer-stock wealth" in names
    assert len(out) + len(log) == len(ents)


def test_type_collision_logged_not_silent():    # Same literal, different types: resolve_entities keeps both (type
    # discipline); the merge must fuse them explicitly with a log entry
    # instead of letting KnowledgeGraph.add_entity fold them silently.
    def _kg(doc, name, etype):
        kg = KnowledgeGraph(schema_name="economics", documents=[doc])
        kg.add_entity(_ent(name, etype, doc))
        return kg

    log = []
    merged = merge_knowledge_graphs(
        [_kg("s.pdf", "DebtRank", "theory"), _kg("p.pdf", "DebtRank", "mechanism")],
        fusion_log=log,
    )
    assert len(merged.entities) == 1
    assert len(log) == 1
    assert log[0]["stage"] == "type-collision"
    assert log[0]["canonical"] == "DebtRank"
    assert log[0]["sources"] == ["p.pdf"]
    # content preserved: first graph's type wins, provenance unioned
    assert merged.entities["DebtRank"].entity_type == "theory"
    assert len(merged.entities["DebtRank"].provenance) == 2


def test_alias_to_alias_fuses_without_canonical_present():
    """T3 repair: two aliases of one canonical fuse even if the canonical
    literal is absent from the merge."""
    aliases = {"natural interest rate": "Natural rate of interest",
               "natural rate of interest": "Natural rate of interest"}
    log = []
    out, ren = resolve_entities(
        [_ent("Natural Interest Rate"), _ent("natural rate of interest", doc="e.pdf")],
        threshold=1.0, aliases=aliases, fusion_log=log,
    )
    assert len(out) == 1 and log[0]["stage"] == "alias"


def test_alias_stage_respects_type():
    aliases = {"banks": "Commercial banks"}
    out, _ = resolve_entities(
        [_ent("Commercial banks", "agent"), _ent("Banks", "market", doc="e.pdf")],
        threshold=1.0, aliases=aliases,
    )
    assert len(out) == 2


def test_threshold_one_disables_fuzzy_antonym_fusion():
    log = []
    out, _ = resolve_entities(
        [_ent("Skilled employment"), _ent("Unskilled employment", doc="e.pdf")],
        threshold=1.0, fusion_log=log,
    )
    assert len(out) == 2 and log == []
