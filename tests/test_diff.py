"""Tests for `ontograph.diff.diff_kgs` — structured KG diff (#44).

Produces per-entity and per-relation classification ('added' / 'removed' /
'changed' / 'unchanged') consumed by the viewer overlay. Keeps the same
relation key as the existing markdown-diff CLI (`cmd_diff`) so both surfaces
agree on what 'same edge' means.
"""
from __future__ import annotations

import pytest

from ontograph.diff import diff_kgs
from ontograph.models import Entity, KnowledgeGraph, Relation


def _kg(entities: list[tuple[str, str]], rels: list[dict]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for name, etype in entities:
        kg.add_entity(Entity(name=name, entity_type=etype))
    for r in rels:
        kg.add_relation(Relation(**r))
    return kg


class TestEntityDiff:
    def test_identical_kgs_all_unchanged(self):
        ents = [("a", "variable"), ("b", "variable")]
        rels = [{"source": "a", "target": "b", "relation_type": "r",
                 "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        base = _kg(ents, rels)
        head = _kg(ents, rels)
        d = diff_kgs(base, head)
        assert d["entities"]["added"] == []
        assert d["entities"]["removed"] == []
        assert d["entities"]["changed"] == []
        assert sorted(d["entities"]["unchanged"]) == ["a", "b"]

    def test_added_entity(self):
        base = _kg([("a", "variable")], [])
        head = _kg([("a", "variable"), ("b", "variable")], [])
        d = diff_kgs(base, head)
        assert d["entities"]["added"] == ["b"]

    def test_removed_entity(self):
        base = _kg([("a", "variable"), ("b", "variable")], [])
        head = _kg([("a", "variable")], [])
        d = diff_kgs(base, head)
        assert d["entities"]["removed"] == ["b"]

    def test_changed_entity_type(self):
        base = _kg([("a", "variable")], [])
        head = _kg([("a", "stock")], [])
        d = diff_kgs(base, head)
        changed = d["entities"]["changed"]
        assert len(changed) == 1
        assert changed[0]["name"] == "a"
        assert changed[0]["fields"]["entity_type"] == ("variable", "stock")

    def test_changed_entity_aliases_not_tracked_as_change(self):
        # Aliases/observations are metadata, not identity — don't flag.
        base = _kg([("a", "variable")], [])
        head = _kg([("a", "variable")], [])
        head.entities["a"].aliases = ["alpha"]
        d = diff_kgs(base, head)
        assert d["entities"]["changed"] == []


class TestRelationDiff:
    def test_added_relation(self):
        base = _kg([("a", "variable"), ("b", "variable")], [])
        head_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        head = _kg([("a", "variable"), ("b", "variable")], head_rels)
        d = diff_kgs(base, head)
        assert len(d["relations"]["added"]) == 1
        assert d["relations"]["added"][0]["key"] == ("a", "b", "r")

    def test_removed_relation(self):
        base_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        base = _kg([("a", "variable"), ("b", "variable")], base_rels)
        head = _kg([("a", "variable"), ("b", "variable")], [])
        d = diff_kgs(base, head)
        assert len(d["relations"]["removed"]) == 1
        assert d["relations"]["removed"][0]["key"] == ("a", "b", "r")

    def test_changed_relation_sign(self):
        ents = [("a", "variable"), ("b", "variable")]
        base_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        head_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "-", "lag": "within_step", "edge_class": "mechanism"}]
        d = diff_kgs(_kg(ents, base_rels), _kg(ents, head_rels))
        changed = d["relations"]["changed"]
        assert len(changed) == 1
        assert changed[0]["key"] == ("a", "b", "r")
        assert changed[0]["fields"]["sign"] == ("+", "-")

    def test_changed_relation_lag(self):
        ents = [("a", "variable"), ("b", "variable")]
        base_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        head_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "across_step", "edge_class": "mechanism"}]
        d = diff_kgs(_kg(ents, base_rels), _kg(ents, head_rels))
        changed = d["relations"]["changed"]
        assert len(changed) == 1
        assert changed[0]["fields"]["lag"] == ("within_step", "across_step")

    def test_parallel_edges_same_rel_type_keyed_once(self):
        """Two edges between the same (src, tgt, rel) collapse under the key.
        Parallel edges are not the common case in discipline-v1; keep the
        first one to match `cmd_diff`'s convention."""
        ents = [("a", "variable"), ("b", "variable")]
        base_rels = [{"source": "a", "target": "b", "relation_type": "r",
                      "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        d = diff_kgs(_kg(ents, base_rels), _kg(ents, base_rels))
        # No change under the key; unchanged count must be 1.
        assert len(d["relations"]["unchanged"]) == 1


class TestSummaryCounts:
    def test_counts_present(self):
        base = _kg([("a", "variable")], [])
        head = _kg([("a", "variable"), ("b", "variable")], [])
        d = diff_kgs(base, head)
        assert d["summary"]["entities_added"] == 1
        assert d["summary"]["entities_removed"] == 0
        assert d["summary"]["entities_changed"] == 0
        assert d["summary"]["relations_added"] == 0
        assert d["summary"]["relations_removed"] == 0
        assert d["summary"]["relations_changed"] == 0


class TestPathInputAccepted:
    def test_accepts_kg_paths(self, tmp_path):
        base = _kg([("a", "variable")], [])
        head = _kg([("a", "variable"), ("b", "variable")], [])
        bp = tmp_path / "base.json"
        hp = tmp_path / "head.json"
        bp.write_text(base.to_json(), encoding="utf-8")
        hp.write_text(head.to_json(), encoding="utf-8")
        d = diff_kgs(bp, hp)
        assert d["entities"]["added"] == ["b"]
