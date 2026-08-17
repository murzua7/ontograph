"""Tests for ontograph.provenance — extraction-provenance hashing.

A KG is only reproducible if it carries enough metadata to pin down the
extractor. These tests lock in:
  (a) schema_hash is deterministic, stable under cosmetic edits, changes
      under semantic edits;
  (b) prompt_hash is deterministic and changes between modes;
  (c) `extraction` field survives JSON round-trip via KnowledgeGraph.
"""
from __future__ import annotations

from ontograph.models import KnowledgeGraph
from ontograph.provenance import (
    canonical_schema_dict,
    extraction_provenance,
    prompt_hash_for_mode,
    schema_hash,
)
from ontograph.schema import EntityTypeDef, OntologySchema, RelationTypeDef


def _small_schema(name="test") -> OntologySchema:
    s = OntologySchema(name=name, version="1.0", description="")
    s.entity_types["agent"] = EntityTypeDef(
        name="agent", description="an actor", subtypes=["bank", "firm"], color="#4CAF50"
    )
    s.entity_types["mechanism"] = EntityTypeDef(
        name="mechanism", description="a process", color="#2196F3"
    )
    s.relation_types["causes"] = RelationTypeDef(
        name="causes", description="causal link",
        default_sign="+", default_lag="within_step",
        default_form="linear", default_edge_class="mechanism",
    )
    return s


class TestSchemaHash:
    def test_is_deterministic(self):
        a = _small_schema()
        b = _small_schema()
        assert schema_hash(a) == schema_hash(b)

    def test_color_change_does_not_change_hash(self):
        """Display-only edits must not perturb the schema hash."""
        a = _small_schema()
        b = _small_schema()
        b.entity_types["agent"].color = "#FF0000"
        assert schema_hash(a) == schema_hash(b)

    def test_description_change_changes_hash(self):
        """Description is part of the prompt the LLM sees → semantic."""
        a = _small_schema()
        b = _small_schema()
        b.entity_types["agent"].description = "a different kind of actor"
        assert schema_hash(a) != schema_hash(b)

    def test_new_entity_type_changes_hash(self):
        a = _small_schema()
        b = _small_schema()
        b.entity_types["market"] = EntityTypeDef(name="market", description="a market")
        assert schema_hash(a) != schema_hash(b)

    def test_default_sign_change_changes_hash(self):
        """Default-sign flows into extraction — must affect hash."""
        a = _small_schema()
        b = _small_schema()
        b.relation_types["causes"].default_sign = "-"
        assert schema_hash(a) != schema_hash(b)

    def test_none_schema_returns_none(self):
        assert schema_hash(None) is None

    def test_canonical_dict_is_sorted(self):
        canon = canonical_schema_dict(_small_schema())
        ekeys = list(canon["entity_types"].keys())
        assert ekeys == sorted(ekeys)


class TestPromptHash:
    def test_llm_mode_stable(self):
        a = prompt_hash_for_mode("llm")
        b = prompt_hash_for_mode("llm")
        assert a == b and len(a) == 64

    def test_schema_free_differs_from_llm(self):
        assert prompt_hash_for_mode("llm") != prompt_hash_for_mode("schema-free")

    def test_heuristic_returns_empty(self):
        assert prompt_hash_for_mode("heuristic") == ""


class TestExtractionProvenance:
    def test_keys_present(self):
        prov = extraction_provenance(
            mode="llm", schema=_small_schema(),
            model="claude-sonnet-4-5", backend="anthropic",
        )
        for k in (
            "mode", "ontograph_version", "extracted_at",
            "schema_hash", "prompt_hash", "model", "backend",
        ):
            assert k in prov
        assert prov["mode"] == "llm"
        assert prov["model"] == "claude-sonnet-4-5"

    def test_schema_free_has_null_schema_hash(self):
        prov = extraction_provenance(
            mode="schema-free", schema=None, model=None, backend="anthropic",
        )
        assert prov["schema_hash"] is None
        assert prov["prompt_hash"] != ""  # schema-free still has prompts

    def test_round_trip_through_kg_json(self):
        schema = _small_schema()
        kg = KnowledgeGraph(schema_name="test")
        kg.extraction = extraction_provenance(
            mode="llm", schema=schema,
            model="claude-sonnet-4-5", backend="anthropic",
        )
        payload = kg.to_json()
        kg2 = KnowledgeGraph.from_json(payload)
        assert kg2.extraction == kg.extraction
        assert kg2.extraction["schema_hash"] == schema_hash(schema)

    def test_old_kg_json_without_extraction_defaults_to_empty(self):
        """Backwards compat: KGs from before 2026-04-21 have no extraction key."""
        import json
        payload = json.dumps({
            "entities": {},
            "relations": [],
            "schema_name": "base",
            "documents": [],
        })
        kg = KnowledgeGraph.from_json(payload)
        assert kg.extraction == {}

    def test_extra_dict_merges(self):
        prov = extraction_provenance(
            mode="llm", schema=_small_schema(),
            model="x", backend="y",
            extra={"doc_hash": "deadbeef", "run_id": "42"},
        )
        assert prov["doc_hash"] == "deadbeef"
        assert prov["run_id"] == "42"
