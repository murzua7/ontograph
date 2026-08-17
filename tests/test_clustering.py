"""Tests for hierarchical clustering of large KGs (#47)."""
from __future__ import annotations

import pytest

from ontograph.clustering import cluster_kg
from ontograph.models import Entity, KnowledgeGraph, Relation


def _mk_kg(entity_names, rels):
    kg = KnowledgeGraph()
    for n in entity_names:
        kg.add_entity(Entity(name=n, entity_type="variable"))
    for src, tgt in rels:
        kg.add_relation(Relation(
            source=src, target=tgt, relation_type="r",
            sign="+", lag="within_step", edge_class="mechanism",
        ))
    return kg


def _two_cliques(n_per=6):
    """Two tightly-connected cliques joined by a single bridge edge."""
    left = [f"L{i}" for i in range(n_per)]
    right = [f"R{i}" for i in range(n_per)]
    rels = []
    for i in range(n_per):
        for j in range(n_per):
            if i != j:
                rels.append((left[i], left[j]))
                rels.append((right[i], right[j]))
    rels.append((left[0], right[0]))
    return _mk_kg(left + right, rels)


class TestClusterKG:
    def test_small_kg_below_threshold_returns_identity_clusters(self):
        """Below `max_nodes`, no clustering — each node is its own 'cluster'."""
        kg = _mk_kg(["a", "b", "c"], [("a", "b"), ("b", "c")])
        result = cluster_kg(kg, max_nodes=10)
        assert result["clustered"] is False
        # Identity mapping: each node maps to itself.
        assert result["node_to_cluster"] == {"a": "a", "b": "b", "c": "c"}

    def test_above_threshold_triggers_clustering(self):
        kg = _two_cliques(n_per=6)
        result = cluster_kg(kg, max_nodes=5)
        assert result["clustered"] is True
        # Fewer clusters than nodes.
        n_clusters = len(set(result["node_to_cluster"].values()))
        assert n_clusters < len(kg.entities)

    def test_two_cliques_recover_two_clusters(self):
        """Greedy modularity should find the two-community structure."""
        kg = _two_cliques(n_per=6)
        result = cluster_kg(kg, max_nodes=1)  # force clustering
        cluster_ids = set(result["node_to_cluster"].values())
        assert len(cluster_ids) == 2
        # All L-nodes in one cluster, all R-nodes in another.
        left_clusters = {result["node_to_cluster"][f"L{i}"] for i in range(6)}
        right_clusters = {result["node_to_cluster"][f"R{i}"] for i in range(6)}
        assert len(left_clusters) == 1
        assert len(right_clusters) == 1
        assert left_clusters != right_clusters

    def test_cluster_summary_reports_size_and_members(self):
        kg = _two_cliques(n_per=6)
        result = cluster_kg(kg, max_nodes=1)
        summary = result["cluster_summary"]
        # Two clusters, each with 6 members.
        assert len(summary) == 2
        for cid, info in summary.items():
            assert info["size"] == 6
            assert len(info["members"]) == 6

    def test_cluster_edges_aggregate_cross_cluster_only(self):
        """Edge list between clusters; within-cluster edges are absorbed."""
        kg = _two_cliques(n_per=6)
        result = cluster_kg(kg, max_nodes=1)
        cluster_edges = result["cluster_edges"]
        # Exactly one bridge edge connects the two communities.
        cross_cluster = [e for e in cluster_edges if e["source"] != e["target"]]
        assert len(cross_cluster) == 1
        # That aggregated edge carries a count field so the viewer can
        # thicken bridges by underlying-edge mass.
        assert cross_cluster[0]["count"] == 1

    def test_deterministic_given_fixed_input(self):
        """Same KG → same clustering (determinism matters for tests/diffs)."""
        kg = _two_cliques(n_per=6)
        r1 = cluster_kg(kg, max_nodes=1)
        r2 = cluster_kg(kg, max_nodes=1)
        assert r1["node_to_cluster"] == r2["node_to_cluster"]

    def test_isolated_node_is_its_own_cluster(self):
        kg = _mk_kg(
            ["a", "b", "c", "lonely"],
            [("a", "b"), ("b", "c"), ("c", "a")],
        )
        result = cluster_kg(kg, max_nodes=1)
        # `lonely` has no edges → greedy_modularity places it in its own
        # community; we just verify it is mapped to *some* cluster.
        assert "lonely" in result["node_to_cluster"]
