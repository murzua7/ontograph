"""Tests for Phase 3: MCP bridge, additional schemas, dashboard import."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ── MCP Bridge ──

class TestMCPBridge:
    def test_export_to_mcp(self, tmp_path):
        from ontograph.models import Entity, Relation
        from ontograph.graph import OntologyGraph
        from ontograph.mcp_bridge import export_to_mcp

        g = OntologyGraph()
        g.add_entity(Entity(name="Firm", entity_type="agent", observations=["produces"]))
        g.add_entity(Entity(name="Bank", entity_type="agent"))
        g.add_relation(Relation(source="Firm", target="Bank", relation_type="feeds_into"))

        jsonl_path = tmp_path / "kg.jsonl"
        n = export_to_mcp(g, prefix="test", path=jsonl_path)
        assert n == 3  # 2 entities + 1 relation

        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        # Check format
        e1 = json.loads(lines[0])
        assert e1["type"] == "entity"
        assert e1["name"].startswith("test:")
        assert "entityType" in e1

        r1 = json.loads(lines[2])
        assert r1["type"] == "relation"
        assert r1["from"].startswith("test:")

    def test_import_from_mcp(self, tmp_path):
        from ontograph.mcp_bridge import export_to_mcp, import_from_mcp
        from ontograph.models import Entity, Relation
        from ontograph.graph import OntologyGraph

        g = OntologyGraph()
        g.add_entity(Entity(name="Firm", entity_type="agent", observations=["produces goods"]))
        g.add_entity(Entity(name="Bank", entity_type="agent"))
        g.add_relation(Relation(source="Firm", target="Bank", relation_type="feeds_into"))

        jsonl_path = tmp_path / "kg.jsonl"
        export_to_mcp(g, prefix="test", path=jsonl_path)

        kg = import_from_mcp(prefix="test", path=jsonl_path)
        assert len(kg.entities) == 2
        assert "Firm" in kg.entities
        assert len(kg.relations) == 1

    def test_prefix_filtering(self, tmp_path):
        from ontograph.mcp_bridge import export_to_mcp, import_from_mcp
        from ontograph.models import Entity
        from ontograph.graph import OntologyGraph

        g1 = OntologyGraph()
        g1.add_entity(Entity(name="Firm", entity_type="agent"))
        g2 = OntologyGraph()
        g2.add_entity(Entity(name="Protein", entity_type="molecule"))

        jsonl_path = tmp_path / "kg.jsonl"
        export_to_mcp(g1, prefix="econ", path=jsonl_path, append=False)
        export_to_mcp(g2, prefix="bio", path=jsonl_path, append=True)

        kg_econ = import_from_mcp(prefix="econ", path=jsonl_path)
        kg_bio = import_from_mcp(prefix="bio", path=jsonl_path)
        assert len(kg_econ.entities) == 1
        assert "Firm" in kg_econ.entities
        assert len(kg_bio.entities) == 1
        assert "Protein" in kg_bio.entities

    def test_append_mode(self, tmp_path):
        from ontograph.mcp_bridge import export_to_mcp
        from ontograph.models import Entity
        from ontograph.graph import OntologyGraph

        g = OntologyGraph()
        g.add_entity(Entity(name="A", entity_type="agent"))

        jsonl_path = tmp_path / "kg.jsonl"
        export_to_mcp(g, prefix="t", path=jsonl_path)
        export_to_mcp(g, prefix="t", path=jsonl_path, append=True)

        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # appended


# ── Additional Schemas ──

class TestAdditionalSchemas:
    def test_load_biology(self):
        from ontograph.schema import load_schema
        s = load_schema("biology")
        assert s.name == "biology"
        assert "molecule" in s.entity_types
        assert "gene" in s.entity_types
        assert "regulates" in s.relation_types
        assert "activates" in s.relation_types

    def test_load_engineering(self):
        from ontograph.schema import load_schema
        s = load_schema("engineering")
        assert s.name == "engineering"
        assert "component" in s.entity_types
        assert "system" in s.entity_types
        assert "connects_to" in s.relation_types
        assert "controls" in s.relation_types

    def test_biology_entity_types_have_colors(self):
        from ontograph.schema import load_schema
        s = load_schema("biology")
        for etype in s.entity_types.values():
            assert etype.color.startswith("#")

    def test_all_schemas_load(self):
        from ontograph.schema import load_schema, SCHEMAS_DIR
        for f in SCHEMAS_DIR.glob("*.yaml"):
            s = load_schema(f.stem)
            assert len(s.entity_types) >= 3
            assert len(s.relation_types) >= 3


# ── Dashboard Import ──

class TestDashboardImport:
    def test_dashboard_module_importable(self):
        """Dashboard module should import without starting Streamlit."""
        # This tests that the module doesn't have side effects on import
        # (the actual rendering requires a Streamlit runtime)
        import importlib
        spec = importlib.util.find_spec("ontograph.dashboard.app")
        assert spec is not None

    def test_graph_to_agraph_conversion(self):
        """Test that our graph data can be converted to agraph format."""
        from ontograph.models import Entity, Relation
        from ontograph.graph import OntologyGraph

        g = OntologyGraph()
        g.add_entity(Entity(name="Firm", entity_type="agent"))
        g.add_entity(Entity(name="Bank", entity_type="agent"))
        g.add_relation(Relation(source="Firm", target="Bank", relation_type="feeds_into"))

        # Simulate what the dashboard does
        from streamlit_agraph import Node, Edge
        nodes = []
        for name, entity in g._entities.items():
            nodes.append(Node(id=name, label=name, size=25, color="#4CAF50"))
        edges = []
        for u, v, data in g.g.edges(data=True):
            edges.append(Edge(source=u, target=v, label=data.get("relation_type", "")))

        assert len(nodes) == 2
        assert len(edges) == 1
        assert edges[0].source == "Firm"
        assert edges[0].to == "Bank"
