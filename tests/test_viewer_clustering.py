"""Tests for clustered-mode rendering in the 3D viewer (#47)."""
from __future__ import annotations

import json
import re

import pytest

from ontograph.dashboard.force3d import build_3d_html
from ontograph.models import Entity, KnowledgeGraph, Relation


def _extract_data_json(html: str) -> dict:
    """Pull the `const DATA = {...};` JSON blob out of the rendered HTML."""
    marker = "const DATA = "
    i = html.index(marker) + len(marker)
    depth = 0
    j = i
    while j < len(html):
        c = html[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[i:j + 1])
        j += 1
    raise AssertionError("Could not find data blob")


def _write_kg(path, n_per=6):
    """Two-clique KG bridged by a single edge; good for clustering tests."""
    kg = KnowledgeGraph()
    for i in range(n_per):
        kg.add_entity(Entity(name=f"L{i}", entity_type="variable"))
        kg.add_entity(Entity(name=f"R{i}", entity_type="variable"))
    for i in range(n_per):
        for j in range(n_per):
            if i != j:
                for (a, b) in [(f"L{i}", f"L{j}"), (f"R{i}", f"R{j}")]:
                    kg.add_relation(Relation(
                        source=a, target=b, relation_type="r",
                        sign="+", lag="within_step", edge_class="mechanism",
                    ))
    kg.add_relation(Relation(
        source="L0", target="R0", relation_type="bridge",
        sign="+", lag="within_step", edge_class="mechanism",
    ))
    path.write_text(kg.to_json(), encoding="utf-8")


class TestClusterMode:
    def test_disabled_by_default_on_small_kg(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=3)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        # Individual nodes, not supernodes.
        node_ids = sorted(n["id"] for n in data["nodes"])
        assert "L0" in node_ids and "R0" in node_ids
        # No cluster metadata on any node.
        assert all(n.get("is_cluster") is not True for n in data["nodes"])

    def test_cluster_mode_forced_emits_supernodes(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode=True)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        # Supernodes: fewer than 12 nodes, and every node is a cluster.
        assert all(n.get("is_cluster") is True for n in data["nodes"])
        assert len(data["nodes"]) < 12
        # The two-clique topology should produce two supernodes.
        assert len(data["nodes"]) == 2

    def test_supernode_carries_size_and_members(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode=True)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        for n in data["nodes"]:
            assert n["size"] == 6
            assert len(n["members"]) == 6

    def test_cluster_edge_carries_count(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode=True)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        # One bridge edge between the two clusters.
        assert len(data["links"]) == 1
        assert data["links"][0]["count"] == 1

    def test_auto_triggers_when_threshold_exceeded(self, tmp_path):
        """cluster_mode='auto' + small threshold → clustering activates."""
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode="auto", cluster_threshold=5)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        assert all(n.get("is_cluster") is True for n in data["nodes"])

    def test_auto_leaves_small_kg_uncoarsened(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=3)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode="auto", cluster_threshold=100)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        assert all(n.get("is_cluster") is not True for n in data["nodes"])

    def test_legend_reports_cluster_mode(self, tmp_path):
        kg_p = tmp_path / "kg.json"
        _write_kg(kg_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, cluster_mode=True)
        html = out.read_text(encoding="utf-8")
        # A clustered-mode indicator so the user isn't confused about the
        # "why am I seeing fewer nodes?" question.
        assert "Clustered" in html or "cluster-mode" in html

    def test_diff_mode_suppresses_clustering(self, tmp_path):
        """Diff overlays + cluster supernodes don't compose; diff wins."""
        kg_p = tmp_path / "kg.json"
        base_p = tmp_path / "base.json"
        _write_kg(kg_p, n_per=6)
        _write_kg(base_p, n_per=6)
        out = tmp_path / "viewer.html"
        build_3d_html(kg_p, out, base_kg_path=base_p, cluster_mode=True)
        data = _extract_data_json(out.read_text(encoding="utf-8"))
        # Even with cluster_mode=True, diff mode must keep individual nodes
        # so the +/-/~ overlay is interpretable.
        assert all(n.get("is_cluster") is not True for n in data["nodes"])
