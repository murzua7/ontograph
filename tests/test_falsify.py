"""Phase E — simulator-in-the-loop falsifier tests (TDD).

Covers `ontograph.falsify.falsify(kg, adapter, *, subset) -> FalsifyReport`.

Spec §4.3 contract:
- SimulatorAdapter protocol exposes `edge_sign(src, tgt) -> Sign` and
  `edge_lag(src, tgt) -> Lag`.
- Three buckets: `confirmed` (kg sign matches sim sign), `flipped` (kg
  sign contradicts sim sign on paired `+`/`-`), `unknown` (simulator
  returned `unknown` — silence is not disagreement; no ruling).
- Default `subset="mechanism"` — only mechanism edges are falsified;
  identity/structural/parameter/abstract edges are out of scope.
- KG relations whose sign is itself `unknown` are silently skipped
  (nothing to falsify).
"""

from __future__ import annotations

import json

from ontograph.models import Entity, KnowledgeGraph, Relation


# ── helpers ────────────────────────────────────────────────────────────

def _kg(relations, entities=()):
    kg = KnowledgeGraph()
    for e in entities:
        kg.add_entity(e)
    else:
        # Ensure endpoints exist so `from_json` / serialization stays sane.
        names = {r.source for r in relations} | {r.target for r in relations}
        for n in names:
            if n not in kg.entities:
                kg.add_entity(Entity(name=n, entity_type="mechanism"))
    for r in relations:
        kg.add_relation(r)
    return kg


def _rel(src, tgt, *, sign="+", lag="within_step", edge_class="mechanism"):
    return Relation(
        source=src, target=tgt, relation_type="feeds_into",
        sign=sign, lag=lag, edge_class=edge_class,
    )


class _DictAdapter:
    """Simple adapter backed by hand-specified (src, tgt) -> (sign, lag) dicts."""

    def __init__(self, signs=None, lags=None):
        self._signs = signs or {}
        self._lags = lags or {}

    def edge_sign(self, source, target):
        return self._signs.get((source, target), "unknown")

    def edge_lag(self, source, target):
        return self._lags.get((source, target), "unknown")


# ── 1. Core bucketing ───────────────────────────────────────────────────

class TestFalsifyBuckets:
    def test_matching_sign_is_confirmed(self):
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="+")])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter)
        assert len(report.confirmed) == 1
        assert report.flipped == []
        assert report.unknown == []

    def test_opposite_sign_is_flipped(self):
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="+")])
        adapter = _DictAdapter(signs={("a", "b"): "-"})
        report = falsify(kg, adapter)
        assert len(report.flipped) == 1
        assert report.confirmed == []
        entry = report.flipped[0]
        assert entry["source"] == "a"
        assert entry["target"] == "b"
        assert entry["kg_sign"] == "+"
        assert entry["sim_sign"] == "-"

    def test_sim_unknown_goes_to_unknown_bucket_not_flipped(self):
        """Spec §2 principle 2: silence is not disagreement."""
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="+")])
        adapter = _DictAdapter()  # everything "unknown"
        report = falsify(kg, adapter)
        assert report.confirmed == []
        assert report.flipped == []
        assert len(report.unknown) == 1

    def test_kg_sign_unknown_is_skipped(self):
        """Nothing to falsify if the KG itself is silent."""
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="unknown")])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter)
        assert report.confirmed == []
        assert report.flipped == []
        assert report.unknown == []

    def test_ambiguous_sign_is_not_flipped(self):
        """`±` matches-nothing — never contradicts the simulator."""
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="±")])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter)
        # Spec: only paired signs `+`/`-` can conflict. `±` is matches-nothing.
        assert report.flipped == []


# ── 2. Subset filter ────────────────────────────────────────────────────

class TestSubset:
    def test_default_subset_mechanism_filters_non_mechanism_edges(self):
        from ontograph.falsify import falsify
        kg = _kg([
            _rel("a", "b", sign="+", edge_class="mechanism"),
            _rel("a", "b", sign="+", edge_class="identity"),
        ])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter)
        # Only the mechanism edge is examined under the default subset.
        assert len(report.confirmed) == 1

    def test_subset_all_falsifies_every_edge(self):
        from ontograph.falsify import falsify
        kg = _kg([
            _rel("a", "b", sign="+", edge_class="mechanism"),
            _rel("a", "b", sign="+", edge_class="structural"),
        ])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter, subset="all")
        assert len(report.confirmed) == 2


# ── 3. Multi-edge realistic case ────────────────────────────────────────

class TestMixedReport:
    def test_mixed_buckets_populate_correctly(self):
        from ontograph.falsify import falsify
        kg = _kg([
            _rel("a", "b", sign="+"),   # confirmed
            _rel("c", "d", sign="+"),   # flipped
            _rel("e", "f", sign="-"),   # unknown (sim silent)
        ])
        adapter = _DictAdapter(signs={
            ("a", "b"): "+",
            ("c", "d"): "-",
        })
        report = falsify(kg, adapter)
        assert {(x["source"], x["target"]) for x in report.confirmed} == {("a", "b")}
        assert {(x["source"], x["target"]) for x in report.flipped}   == {("c", "d")}
        assert {(x["source"], x["target"]) for x in report.unknown}   == {("e", "f")}
        # Summary counters make scripts easy to assert against.
        assert report.total == 3
        assert report.examined == 3


# ── 4. Lag reporting ────────────────────────────────────────────────────

class TestLagReport:
    def test_adapter_lag_recorded_on_confirmed_entries(self):
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="+", lag="within_step")])
        adapter = _DictAdapter(
            signs={("a", "b"): "+"},
            lags={("a", "b"): "Q+2"},
        )
        report = falsify(kg, adapter)
        entry = report.confirmed[0]
        assert entry["kg_lag"] == "within_step"
        assert entry["sim_lag"] == "Q+2"


# ── 5. Serialisation ────────────────────────────────────────────────────

class TestFalsifyReportSerialisation:
    def test_to_json_roundtrips(self):
        from ontograph.falsify import falsify
        kg = _kg([_rel("a", "b", sign="+")])
        adapter = _DictAdapter(signs={("a", "b"): "+"})
        report = falsify(kg, adapter)
        payload = json.loads(report.to_json())
        assert "confirmed" in payload
        assert "flipped" in payload
        assert "unknown" in payload
        assert payload["total"] == 1
        assert payload["examined"] == 1
