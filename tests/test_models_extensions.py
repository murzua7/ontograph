"""Phase A — Extended entity/relation data model for ontology-extraction-discipline-v1.

Tests are written before implementation (TDD).

Covers:
- CodeAnchor / CitationAnchor dataclasses
- Entity: code_anchors, citation_anchors, abstract, abstract_rationale, confidence
- Entity.is_grounded()
- Relation: sign, lag, form, conditional_on, edge_class, confidence
- Backward compatibility: pre-existing KG JSON must still round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── CodeAnchor / CitationAnchor ──

class TestAnchors:
    def test_code_anchor_fields(self):
        from ontograph.models import CodeAnchor
        a = CodeAnchor(repo="DiffMABM", path="diffmabm/model.py", line=120, symbol="_phase_firms")
        assert a.repo == "DiffMABM"
        assert a.path == "diffmabm/model.py"
        assert a.line == 120
        assert a.symbol == "_phase_firms"

    def test_code_anchor_defaults(self):
        from ontograph.models import CodeAnchor
        a = CodeAnchor(repo="r", path="p.py", line=1)
        assert a.symbol == ""

    def test_citation_anchor_fields(self):
        from ontograph.models import CitationAnchor
        c = CitationAnchor(key="wiese2024", pages="12-14")
        assert c.key == "wiese2024"
        assert c.pages == "12-14"

    def test_citation_anchor_defaults(self):
        from ontograph.models import CitationAnchor
        c = CitationAnchor(key="bernanke1999")
        assert c.pages == ""


# ── Entity extensions ──

class TestEntityExtensions:
    def test_new_fields_default_empty(self):
        from ontograph.models import Entity
        e = Entity(name="Firm", entity_type="agent")
        assert e.code_anchors == []
        assert e.citation_anchors == []
        assert e.abstract is False
        assert e.abstract_rationale == ""
        assert e.confidence == 1.0

    def test_abstract_entity(self):
        from ontograph.models import Entity
        e = Entity(
            name="Heterogeneity",
            entity_type="theory",
            abstract=True,
            abstract_rationale="No bindable code path; purely theoretical scaffolding",
        )
        assert e.abstract is True
        assert "bindable" in e.abstract_rationale

    def test_is_grounded_with_code_anchor(self):
        from ontograph.models import Entity, CodeAnchor
        e = Entity(
            name="firms_produce",
            entity_type="mechanism",
            code_anchors=[CodeAnchor(repo="DiffMABM", path="diffmabm/model.py", line=120, symbol="_phase_firms")],
        )
        assert e.is_grounded() is True

    def test_is_grounded_with_citation(self):
        from ontograph.models import Entity, CitationAnchor
        e = Entity(
            name="DMP matching",
            entity_type="theory",
            citation_anchors=[CitationAnchor(key="pissarides2000", pages="1-10")],
        )
        assert e.is_grounded() is True

    def test_is_grounded_false_when_empty(self):
        from ontograph.models import Entity
        e = Entity(name="Floating concept", entity_type="concept")
        assert e.is_grounded() is False

    def test_is_grounded_abstract_allowed_ungrounded(self):
        """Abstract entities are ontologically allowed to lack anchors — they still return False,
        but the quality gate treats abstract==True as a deliberate choice, not a missing anchor.
        is_grounded() itself reports *raw* anchor presence."""
        from ontograph.models import Entity
        e = Entity(name="Complexity theory", entity_type="theory", abstract=True,
                   abstract_rationale="meta-theoretical framing")
        assert e.is_grounded() is False  # no anchors present; abstract flag is orthogonal

    def test_is_grounded_requires_nonempty_path_on_code_anchor(self):
        """A CodeAnchor with empty path is not grounding."""
        from ontograph.models import Entity, CodeAnchor
        e = Entity(name="x", entity_type="mechanism",
                   code_anchors=[CodeAnchor(repo="", path="", line=0)])
        assert e.is_grounded() is False

    def test_is_grounded_requires_nonempty_key_on_citation(self):
        from ontograph.models import Entity, CitationAnchor
        e = Entity(name="x", entity_type="theory",
                   citation_anchors=[CitationAnchor(key="")])
        assert e.is_grounded() is False


# ── Relation extensions ──

class TestRelationExtensions:
    def test_new_fields_defaults(self):
        from ontograph.models import Relation
        r = Relation(source="A", target="B", relation_type="feeds_into")
        assert r.sign == "unknown"
        assert r.lag == "unknown"
        assert r.form == "unknown"
        assert r.conditional_on == []
        assert r.edge_class == "mechanism"
        assert r.confidence == 1.0

    def test_signed_relation(self):
        from ontograph.models import Relation
        r = Relation(
            source="credit_spread",
            target="investment_demand",
            relation_type="dampens",
            sign="-",
            lag="within_step",
            form="linear",
            edge_class="mechanism",
            confidence=0.9,
        )
        assert r.sign == "-"
        assert r.lag == "within_step"
        assert r.form == "linear"
        assert r.edge_class == "mechanism"
        assert r.confidence == 0.9

    def test_conditional_on_supported(self):
        from ontograph.models import Relation
        r = Relation(
            source="policy_rate",
            target="aggregate_demand",
            relation_type="causes",
            sign="-",
            lag="Q+1",
            conditional_on=["inflation > target", "zero_lower_bound_inactive"],
        )
        assert "inflation > target" in r.conditional_on
        assert r.lag == "Q+1"

    def test_edge_class_values_allowed(self):
        from ontograph.models import Relation
        for cls in ["identity", "mechanism", "parameter", "structural", "abstract"]:
            r = Relation(source="a", target="b", relation_type="relates_to", edge_class=cls)
            assert r.edge_class == cls


# ── Backward compatibility: old JSON must still round-trip ──

class TestBackwardCompatibility:
    def _legacy_json(self) -> str:
        """A KG serialized by the old pre-extension schema — no new fields present."""
        return json.dumps({
            "schema_name": "economics",
            "documents": ["docs/foo.md"],
            "entities": {
                "Firm": {
                    "name": "Firm",
                    "entity_type": "agent",
                    "aliases": ["firms"],
                    "observations": ["produces goods"],
                    "provenance": [{"document": "docs/foo.md", "section": "Agents",
                                    "page": None, "passage": "", "char_offset": 0}],
                    "metadata": {},
                },
                "Bank": {
                    "name": "Bank",
                    "entity_type": "agent",
                    "aliases": [],
                    "observations": [],
                    "provenance": [],
                    "metadata": {},
                },
            },
            "relations": [
                {
                    "source": "Firm",
                    "target": "Bank",
                    "relation_type": "feeds_into",
                    "weight": 1.0,
                    "provenance": [],
                    "metadata": {},
                }
            ],
        })

    def test_old_json_loads_without_error(self):
        from ontograph.models import KnowledgeGraph
        kg = KnowledgeGraph.from_json(self._legacy_json())
        assert "Firm" in kg.entities
        assert len(kg.relations) == 1

    def test_old_json_defaults_new_entity_fields(self):
        from ontograph.models import KnowledgeGraph
        kg = KnowledgeGraph.from_json(self._legacy_json())
        firm = kg.entities["Firm"]
        assert firm.code_anchors == []
        assert firm.citation_anchors == []
        assert firm.abstract is False
        assert firm.abstract_rationale == ""
        assert firm.confidence == 1.0

    def test_old_json_defaults_new_relation_fields(self):
        from ontograph.models import KnowledgeGraph
        kg = KnowledgeGraph.from_json(self._legacy_json())
        r = kg.relations[0]
        assert r.sign == "unknown"
        assert r.lag == "unknown"
        assert r.form == "unknown"
        assert r.conditional_on == []
        assert r.edge_class == "mechanism"
        assert r.confidence == 1.0

    def test_roundtrip_preserves_new_fields(self):
        from ontograph.models import (
            KnowledgeGraph, Entity, Relation, CodeAnchor, CitationAnchor,
        )
        kg = KnowledgeGraph(schema_name="dabs")
        kg.add_entity(Entity(
            name="firms_produce",
            entity_type="mechanism",
            code_anchors=[CodeAnchor(repo="DiffMABM", path="diffmabm/model.py",
                                     line=120, symbol="_phase_firms")],
            citation_anchors=[CitationAnchor(key="pissarides2000", pages="1-10")],
            confidence=0.8,
        ))
        kg.add_entity(Entity(
            name="Heterogeneity",
            entity_type="theory",
            abstract=True,
            abstract_rationale="meta-theoretical scaffold",
        ))
        kg.add_relation(Relation(
            source="firms_produce",
            target="Heterogeneity",
            relation_type="implements",
            sign="+",
            lag="within_step",
            form="monotone",
            conditional_on=["regime==normal"],
            edge_class="identity",
            confidence=0.7,
        ))
        kg2 = KnowledgeGraph.from_json(kg.to_json())
        firm = kg2.entities["firms_produce"]
        assert len(firm.code_anchors) == 1
        assert firm.code_anchors[0].symbol == "_phase_firms"
        assert firm.citation_anchors[0].key == "pissarides2000"
        assert firm.confidence == 0.8
        theory = kg2.entities["Heterogeneity"]
        assert theory.abstract is True
        r = kg2.relations[0]
        assert r.sign == "+"
        assert r.lag == "within_step"
        assert r.form == "monotone"
        assert r.conditional_on == ["regime==normal"]
        assert r.edge_class == "identity"
        assert r.confidence == 0.7
