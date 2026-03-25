"""Tests for Phase 2: LLM extraction, entity resolution, merge."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ── LLM Client ──

class TestLLMClient:
    def test_mock_mode(self):
        from ontograph.llm_client import LLMClient
        client = LLMClient(mode="mock")
        client.set_mock_responses(['{"entities": [{"name": "Firm", "type": "agent"}]}'])
        resp = client.chat([{"role": "user", "content": "test"}])
        assert "Firm" in resp.content

    def test_mock_chat_json(self):
        from ontograph.llm_client import LLMClient
        client = LLMClient(mode="mock")
        client.set_mock_responses(['{"entities": [{"name": "Bank", "type": "agent"}]}'])
        result = client.chat_json([{"role": "user", "content": "test"}])
        assert result["entities"][0]["name"] == "Bank"

    def test_mock_exhausted_returns_empty(self):
        from ontograph.llm_client import LLMClient
        client = LLMClient(mode="mock")
        client.set_mock_responses([])
        result = client.chat_json([{"role": "user", "content": "test"}])
        assert result == {"entities": [], "relations": []}

    def test_extract_json_from_markdown_block(self):
        from ontograph.llm_client import _extract_json
        text = 'Some preamble\n```json\n{"key": "value"}\n```\nMore text'
        assert _extract_json(text) == {"key": "value"}

    def test_extract_json_from_bare(self):
        from ontograph.llm_client import _extract_json
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_from_surrounded(self):
        from ontograph.llm_client import _extract_json
        text = 'Here is the result: {"entities": []} and that is all.'
        assert _extract_json(text) == {"entities": []}


# ── LLM Extractor ──

class TestLLMExtractor:
    def _mock_entity_response(self):
        return json.dumps({"entities": [
            {"name": "Central Bank", "type": "agent", "aliases": ["CB", "monetary authority"],
             "observations": ["Sets the policy rate", "Uses Taylor rule"],
             "passage": "The central bank sets the policy rate via the Taylor rule."},
            {"name": "Taylor Rule", "type": "mechanism", "aliases": [],
             "observations": ["Determines the interest rate based on inflation and output gap"],
             "passage": "The Taylor rule mechanism constrains monetary policy."},
            {"name": "Inflation", "type": "state_variable", "aliases": ["CPI inflation"],
             "observations": ["Key target of monetary policy"],
             "passage": "Inflation is the key target."},
        ]})

    def _mock_relation_response(self):
        return json.dumps({"relations": [
            {"source": "Central Bank", "target": "Taylor Rule", "type": "implements",
             "passage": "The central bank implements the Taylor rule.", "confidence": 0.95},
            {"source": "Taylor Rule", "target": "Inflation", "type": "feeds_into",
             "passage": "The Taylor rule responds to inflation.", "confidence": 0.9},
        ]})

    def test_llm_extract_entities(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.llm_client import LLMClient
        from ontograph.llm_extractor import llm_extract_from_document

        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")

        client = LLMClient(mode="mock")
        # One entity response per section, one relation response per section
        n_sections = len([s for s in doc.sections if len(s.text.strip()) >= 20])
        responses = ([self._mock_entity_response()] * n_sections
                     + [self._mock_relation_response()] * n_sections)
        client.set_mock_responses(responses)

        entities, relations = llm_extract_from_document(doc, schema, client=client)
        assert len(entities) >= 1
        names = [e.name for e in entities]
        assert "Central Bank" in names

    def test_llm_extract_relations_grounded(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.llm_client import LLMClient
        from ontograph.llm_extractor import llm_extract_from_document

        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")

        client = LLMClient(mode="mock")
        n_sections = len([s for s in doc.sections if len(s.text.strip()) >= 20])
        responses = ([self._mock_entity_response()] * n_sections
                     + [self._mock_relation_response()] * n_sections)
        client.set_mock_responses(responses)

        entities, relations = llm_extract_from_document(doc, schema, client=client)
        # Relations should only reference known entities
        entity_names = {e.name for e in entities}
        for r in relations:
            assert r.source in entity_names
            assert r.target in entity_names

    def test_llm_extract_provenance(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.llm_client import LLMClient
        from ontograph.llm_extractor import llm_extract_from_document

        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")

        client = LLMClient(mode="mock")
        n_sections = len([s for s in doc.sections if len(s.text.strip()) >= 20])
        responses = ([self._mock_entity_response()] * n_sections
                     + [self._mock_relation_response()] * n_sections)
        client.set_mock_responses(responses)

        entities, relations = llm_extract_from_document(doc, schema, client=client)
        for e in entities:
            assert len(e.provenance) >= 1
            assert e.provenance[0].document != ""

    def test_dedup_across_sections(self):
        from ontograph.llm_extractor import _dedup_entities
        from ontograph.models import Entity, Provenance

        e1 = Entity(name="Firm", entity_type="agent", observations=["produces goods"],
                     provenance=[Provenance(document="a.md", section="Intro")])
        e2 = Entity(name="firm", entity_type="agent", observations=["sets prices"],
                     provenance=[Provenance(document="a.md", section="Model")])
        result = _dedup_entities([e1, e2])
        assert len(result) == 1
        assert len(result[0].observations) == 2
        assert len(result[0].provenance) == 2

    def test_chunking(self):
        from ontograph.llm_extractor import _chunk_text
        short = "Hello world."
        assert _chunk_text(short) == [short]

        long = "A" * 5000
        chunks = _chunk_text(long, max_chars=2500, overlap=500)
        assert len(chunks) >= 2
        # Overlap: end of chunk 1 should appear in start of chunk 2
        assert chunks[0][-500:] == chunks[1][:500]


# ── Resolver ──

class TestResolver:
    def test_exact_match(self):
        from ontograph.models import Entity
        from ontograph.resolver import resolve_entities

        e1 = Entity(name="Firm", entity_type="agent", observations=["a"])
        e2 = Entity(name="Firm", entity_type="agent", observations=["b"])
        resolved, rename = resolve_entities([e1, e2])
        assert len(resolved) == 1
        assert len(resolved[0].observations) == 2

    def test_case_insensitive_match(self):
        from ontograph.models import Entity
        from ontograph.resolver import resolve_entities

        e1 = Entity(name="Central Bank", entity_type="agent")
        e2 = Entity(name="central bank", entity_type="agent")
        resolved, rename = resolve_entities([e1, e2])
        assert len(resolved) == 1
        assert "central bank" in rename

    def test_alias_match(self):
        from ontograph.models import Entity
        from ontograph.resolver import resolve_entities

        e1 = Entity(name="Central Bank", entity_type="agent", aliases=["CB", "monetary authority"])
        e2 = Entity(name="CB", entity_type="agent")
        resolved, rename = resolve_entities([e1, e2])
        assert len(resolved) == 1
        assert rename["CB"] == "Central Bank"

    def test_fuzzy_match(self):
        from ontograph.models import Entity
        from ontograph.resolver import resolve_entities

        e1 = Entity(name="Taylor Rule", entity_type="mechanism")
        e2 = Entity(name="Taylor rule", entity_type="mechanism")
        resolved, rename = resolve_entities([e1, e2], threshold=0.8)
        assert len(resolved) == 1

    def test_no_cross_type_merge(self):
        from ontograph.models import Entity
        from ontograph.resolver import resolve_entities

        e1 = Entity(name="Bank", entity_type="agent")
        e2 = Entity(name="Bank", entity_type="market")  # different type
        resolved, rename = resolve_entities([e1, e2])
        # Same name but different types should NOT merge via fuzzy
        # (exact match requires same type)
        assert len(resolved) == 2

    def test_resolve_relations(self):
        from ontograph.models import Relation
        from ontograph.resolver import resolve_relations

        rename = {"CB": "Central Bank", "inflation rate": "Inflation"}
        rels = [
            Relation(source="CB", target="inflation rate", relation_type="feeds_into"),
            Relation(source="Firm", target="Central Bank", relation_type="feeds_into"),
        ]
        resolved = resolve_relations(rels, rename)
        assert resolved[0].source == "Central Bank"
        assert resolved[0].target == "Inflation"

    def test_merge_knowledge_graphs(self):
        from ontograph.models import KnowledgeGraph, Entity, Relation
        from ontograph.resolver import merge_knowledge_graphs

        kg1 = KnowledgeGraph(schema_name="economics", documents=["paper1.md"])
        kg1.add_entity(Entity(name="Firm", entity_type="agent", observations=["produces"]))
        kg1.add_entity(Entity(name="Bank", entity_type="agent"))
        kg1.add_relation(Relation(source="Firm", target="Bank", relation_type="feeds_into"))

        kg2 = KnowledgeGraph(schema_name="economics", documents=["paper2.md"])
        kg2.add_entity(Entity(name="Firm", entity_type="agent", observations=["sets prices"]))
        kg2.add_entity(Entity(name="Household", entity_type="agent"))
        kg2.add_relation(Relation(source="Firm", target="Household", relation_type="feeds_into"))

        merged = merge_knowledge_graphs([kg1, kg2])
        assert len(merged.entities) == 3  # Firm merged, Bank + Household unique
        assert len(merged.relations) == 2
        assert len(merged.entities["Firm"].observations) == 2
        assert len(merged.documents) == 2
