"""Core tests for ontograph: models, schema, parsers, extractor, graph, export."""

from pathlib import Path
import json

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ── Models ──

class TestModels:
    def test_entity_id_deterministic(self):
        from ontograph.models import Entity
        e = Entity(name="Taylor Rule", entity_type="mechanism")
        assert e.id == e.id  # deterministic
        assert "taylor_rule" in e.id

    def test_entity_id_unique(self):
        from ontograph.models import Entity
        a = Entity(name="Firm", entity_type="agent")
        b = Entity(name="Bank", entity_type="agent")
        assert a.id != b.id

    def test_kg_add_entity_dedup(self):
        from ontograph.models import KnowledgeGraph, Entity, Provenance
        kg = KnowledgeGraph()
        e1 = Entity(name="Firm", entity_type="agent", observations=["produces goods"])
        e2 = Entity(name="Firm", entity_type="agent", observations=["sets prices"],
                     provenance=[Provenance(document="paper2.md")])
        kg.add_entity(e1)
        kg.add_entity(e2)
        assert len(kg.entities) == 1
        assert "produces goods" in kg.entities["Firm"].observations
        assert "sets prices" in kg.entities["Firm"].observations
        assert len(kg.entities["Firm"].provenance) == 1

    def test_kg_json_roundtrip(self):
        from ontograph.models import KnowledgeGraph, Entity, Relation
        kg = KnowledgeGraph(schema_name="test")
        kg.add_entity(Entity(name="A", entity_type="agent"))
        kg.add_entity(Entity(name="B", entity_type="mechanism"))
        kg.add_relation(Relation(source="A", target="B", relation_type="causes"))
        j = kg.to_json()
        kg2 = KnowledgeGraph.from_json(j)
        assert len(kg2.entities) == 2
        assert len(kg2.relations) == 1
        assert kg2.relations[0].relation_type == "causes"


# ── Schema ──

class TestSchema:
    def test_load_base(self):
        from ontograph.schema import load_schema
        s = load_schema("base")
        assert s.name == "base"
        assert "concept" in s.entity_types
        assert "causes" in s.relation_types

    def test_load_economics(self):
        from ontograph.schema import load_schema
        s = load_schema("economics")
        assert "agent" in s.entity_types
        assert "mechanism" in s.entity_types
        assert "feeds_into" in s.relation_types
        assert "propagates_losses_to" in s.relation_types

    def test_entity_type_validation(self):
        from ontograph.schema import load_schema
        s = load_schema("economics")
        assert s.validate_entity_type("agent")
        assert not s.validate_entity_type("foobar")

    def test_extraction_hints(self):
        from ontograph.schema import load_schema
        s = load_schema("economics")
        assert len(s.extraction_hints["entity_signals"]) > 0
        assert len(s.extraction_hints["relation_signals"]) > 0

    def test_missing_schema_raises(self):
        from ontograph.schema import load_schema
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_schema_xyz")


# ── Parsers ──

class TestParsers:
    def test_parse_markdown(self):
        from ontograph.parsers import parse_document
        doc = parse_document(FIXTURES / "sample_economics.md")
        assert doc.title == "A Simple Macro-Financial Model"
        assert len(doc.sections) >= 4
        assert any("Agents" in s.heading for s in doc.sections)

    def test_parse_unknown_ext_as_text(self):
        import tempfile
        from ontograph.parsers import parse_document
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("Hello world. The banking agent provides loans.")
            f.flush()
            doc = parse_document(f.name)
        assert "Hello world" in doc.full_text


# ── Extractor ──

class TestExtractor:
    def test_extract_entities_from_sample(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.extractor import extract_entities
        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")
        entities = extract_entities(doc, schema)
        names = [e.name.lower() for e in entities]
        # Should find at least some key entities
        assert any("household" in n for n in names) or any("firm" in n for n in names)
        assert len(entities) >= 3

    def test_extract_relations_from_sample(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.extractor import extract_from_document
        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")
        entities, relations = extract_from_document(doc, schema)
        # Relations require matching entity names, so may be fewer
        assert isinstance(relations, list)

    def test_provenance_tracking(self):
        from ontograph.parsers import parse_document
        from ontograph.schema import load_schema
        from ontograph.extractor import extract_entities
        doc = parse_document(FIXTURES / "sample_economics.md")
        schema = load_schema("economics")
        entities = extract_entities(doc, schema)
        for e in entities:
            assert len(e.provenance) >= 1
            assert e.provenance[0].document == str(FIXTURES / "sample_economics.md")


# ── Graph ──

class TestGraph:
    def _make_graph(self):
        from ontograph.models import Entity, Relation
        from ontograph.graph import OntologyGraph
        g = OntologyGraph()
        g.add_entity(Entity(name="Firm", entity_type="agent"))
        g.add_entity(Entity(name="Bank", entity_type="agent"))
        g.add_entity(Entity(name="Credit Market", entity_type="market"))
        g.add_relation(Relation(source="Firm", target="Credit Market", relation_type="feeds_into"))
        g.add_relation(Relation(source="Credit Market", target="Bank", relation_type="feeds_into"))
        g.add_relation(Relation(source="Bank", target="Firm", relation_type="propagates_losses_to"))
        return g

    def test_add_entities(self):
        g = self._make_graph()
        assert g.n_entities == 3
        assert g.n_relations == 3

    def test_find_cycles(self):
        g = self._make_graph()
        cycles = g.find_cycles()
        assert len(cycles) >= 1
        # The cycle: Firm -> Credit Market -> Bank -> Firm
        flat = [n for c in cycles for n in c]
        assert "Firm" in flat

    def test_cascade_depth(self):
        g = self._make_graph()
        depth = g.cascade_depth("propagates_losses_to")
        assert depth >= 1

    def test_degree_centrality(self):
        g = self._make_graph()
        cent = g.degree_centrality()
        assert len(cent) == 3
        assert all(0 <= v <= 1 for v in cent.values())

    def test_summary(self):
        g = self._make_graph()
        s = g.summary()
        assert s["n_entities"] == 3
        assert s["n_relations"] == 3
        assert s["n_cycles"] >= 1

    def test_kg_roundtrip(self):
        g = self._make_graph()
        kg = g.to_kg()
        from ontograph.graph import OntologyGraph
        g2 = OntologyGraph.from_kg(kg)
        assert g2.n_entities == 3
        assert g2.n_relations == 3


# ── Export ──

class TestExport:
    def test_export_mermaid(self):
        from ontograph.models import Entity, Relation
        from ontograph.graph import OntologyGraph
        from ontograph.export import export_mermaid
        g = OntologyGraph()
        g.add_entity(Entity(name="Firm", entity_type="agent"))
        g.add_entity(Entity(name="Bank", entity_type="agent"))
        g.add_relation(Relation(source="Firm", target="Bank", relation_type="feeds_into"))
        mermaid = export_mermaid(g)
        assert "graph LR" in mermaid
        assert "Firm" in mermaid
        assert "feeds_into" in mermaid

    def test_export_json_roundtrip(self, tmp_path):
        from ontograph.models import Entity, Relation, KnowledgeGraph
        from ontograph.graph import OntologyGraph
        from ontograph.export import export_json
        g = OntologyGraph()
        g.add_entity(Entity(name="A", entity_type="agent"))
        g.add_entity(Entity(name="B", entity_type="mechanism"))
        g.add_relation(Relation(source="A", target="B", relation_type="causes"))
        out = tmp_path / "test.json"
        export_json(g, out)
        data = json.loads(out.read_text())
        assert "entities" in data
        assert "relations" in data
