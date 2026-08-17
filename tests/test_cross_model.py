"""Tests for cross-model extraction agreement (#46)."""
from __future__ import annotations

import pytest

from ontograph.cross_model import cross_model_agreement
from ontograph.models import Entity, KnowledgeGraph, Relation


def _kg(entities: list[tuple[str, str]], rels: list[dict]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for name, etype in entities:
        kg.add_entity(Entity(name=name, entity_type=etype))
    for r in rels:
        kg.add_relation(Relation(**r))
    return kg


class TestEntityAgreement:
    def test_identical_kgs_full_agreement(self):
        ents = [("a", "variable"), ("b", "variable")]
        a = _kg(ents, [])
        b = _kg(ents, [])
        result = cross_model_agreement(a, b)
        assert sorted(result["entities"]["agreement"]) == ["a", "b"]
        assert result["entities"]["a_only"] == []
        assert result["entities"]["b_only"] == []
        assert result["entities"]["type_conflicts"] == []

    def test_a_only_and_b_only(self):
        a = _kg([("x", "variable"), ("shared", "variable")], [])
        b = _kg([("y", "variable"), ("shared", "variable")], [])
        result = cross_model_agreement(a, b)
        assert result["entities"]["a_only"] == ["x"]
        assert result["entities"]["b_only"] == ["y"]
        assert result["entities"]["agreement"] == ["shared"]

    def test_type_conflict_listed_separately(self):
        a = _kg([("p", "variable")], [])
        b = _kg([("p", "policy_tool")], [])
        result = cross_model_agreement(a, b)
        # Same name, disagreeing type → type_conflict, NOT agreement.
        assert result["entities"]["agreement"] == []
        assert len(result["entities"]["type_conflicts"]) == 1
        conflict = result["entities"]["type_conflicts"][0]
        assert conflict["name"] == "p"
        assert conflict["a"] == "variable"
        assert conflict["b"] == "policy_tool"


class TestRelationAgreement:
    def test_agreeing_relations_merged_with_consensus_sign(self):
        ents = [("a", "variable"), ("b", "variable")]
        rels = [{"source": "a", "target": "b", "relation_type": "r",
                 "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        result = cross_model_agreement(_kg(ents, rels), _kg(ents, rels))
        agreement = result["relations"]["agreement"]
        assert len(agreement) == 1
        rel = agreement[0]
        assert rel["sign"] == "+"
        assert rel["lag"] == "within_step"

    def test_sign_conflict_marked_unknown_in_agreement(self):
        ents = [("a", "variable"), ("b", "variable")]
        a_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        b_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "-", "lag": "within_step", "edge_class": "mechanism"}]
        result = cross_model_agreement(_kg(ents, a_rels), _kg(ents, b_rels))
        assert len(result["relations"]["agreement"]) == 1
        # Shape matches: both saw it (structure agreement) but sign is contested.
        assert result["relations"]["agreement"][0]["sign"] == "unknown"
        assert len(result["relations"]["sign_conflicts"]) == 1

    def test_unknown_sign_on_one_side_yields_other_sign(self):
        """If one model returned unknown and the other a concrete sign, the
        concrete one wins in the agreement — `unknown` carries no signal."""
        ents = [("a", "variable"), ("b", "variable")]
        a_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        b_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "unknown", "lag": "within_step", "edge_class": "mechanism"}]
        result = cross_model_agreement(_kg(ents, a_rels), _kg(ents, b_rels))
        assert result["relations"]["agreement"][0]["sign"] == "+"
        assert result["relations"]["sign_conflicts"] == []

    def test_a_only_and_b_only_relations(self):
        ents = [("a", "variable"), ("b", "variable")]
        a_rels = [{"source": "a", "target": "b", "relation_type": "r1",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        b_rels = [{"source": "a", "target": "b", "relation_type": "r2",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        result = cross_model_agreement(_kg(ents, a_rels), _kg(ents, b_rels))
        assert len(result["relations"]["agreement"]) == 0
        assert len(result["relations"]["a_only"]) == 1
        assert len(result["relations"]["b_only"]) == 1


class TestSummary:
    def test_agreement_rates_reported(self):
        a = _kg([("x", "variable"), ("shared", "variable")], [])
        b = _kg([("y", "variable"), ("shared", "variable")], [])
        result = cross_model_agreement(a, b)
        # 1 agreement out of 3 unique entities → 1/3 ≈ 0.333
        assert 0.30 < result["summary"]["entity_agreement_rate"] < 0.40

    def test_sign_agreement_rate_only_counts_known_signs(self):
        ents = [("a", "variable"), ("b", "variable")]
        a_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        b_rels = [{"source": "a", "target": "b", "relation_type": "r",
                   "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        result = cross_model_agreement(_kg(ents, a_rels), _kg(ents, b_rels))
        assert result["summary"]["sign_agreement_rate"] == 1.0


class TestAgreementKG:
    def test_build_agreement_kg_contains_only_shared(self):
        from ontograph.cross_model import build_agreement_kg
        ents = [("a", "variable"), ("b", "variable")]
        rels = [{"source": "a", "target": "b", "relation_type": "r",
                 "sign": "+", "lag": "within_step", "edge_class": "mechanism"}]
        kg_a = _kg(ents, rels)
        kg_b = _kg(ents + [("extra", "variable")], rels)
        result = cross_model_agreement(kg_a, kg_b)
        kg = build_agreement_kg(result, kg_a, kg_b)
        assert set(kg.entities) == {"a", "b"}
        assert len(kg.relations) == 1
        assert kg.relations[0].sign == "+"
