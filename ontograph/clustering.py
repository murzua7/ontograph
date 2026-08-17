"""Hierarchical clustering for large KG visualisation (#47).

The 3D viewer renders every node as a WebGL particle. Somewhere north of
~5k nodes and ~10k edges, layout stabilisation goes from a couple of
seconds to minutes and fps collapses. The remedy is coarsening: compute
a community partition, render a supernode per community, and aggregate
cross-community edges into weighted bridges.

Two design choices worth recording:

- **Greedy modularity**, not Louvain/Leiden. NetworkX ships it natively,
  it is deterministic given a fixed node iteration order (so tests and
  inter-session diffs are stable), and for our regime (V ≤ 50k, E sparse)
  the O((V+E) log V) runtime is comfortably under a second.

- **Undirected for partitioning only.** Communities describe *topological
  neighbourhoods*, not causal flow. The underlying KG is directed and we
  preserve that directionality in the aggregated cluster-edge list — two
  clusters can be connected by bridges in both directions and they are
  surfaced as two separate entries.

Public surface:
- `cluster_kg(kg, *, max_nodes=5000)` — always returns a structure the
  viewer can consume; sets `clustered=False` and an identity mapping when
  the graph is below threshold, so callers can treat the result
  uniformly.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms import community as nx_community

from ontograph.models import KnowledgeGraph


def _identity_result(kg: KnowledgeGraph) -> dict[str, Any]:
    """Below-threshold pass-through: every node is 'its own cluster'."""
    node_to_cluster = {n: n for n in kg.entities}
    cluster_summary = {
        n: {"size": 1, "members": [n]} for n in kg.entities
    }
    # Cluster-edges in identity mode are just the original edges with
    # count=1; we still collapse parallel relations (same src,tgt pair
    # but different relation_types) so the viewer sees at most one
    # supernode-supernode link per ordered pair.
    seen: dict[tuple[str, str], int] = {}
    for r in kg.relations:
        key = (r.source, r.target)
        seen[key] = seen.get(key, 0) + 1
    cluster_edges = [
        {"source": s, "target": t, "count": c}
        for (s, t), c in seen.items()
    ]
    return {
        "clustered": False,
        "node_to_cluster": node_to_cluster,
        "cluster_summary": cluster_summary,
        "cluster_edges": cluster_edges,
    }


def cluster_kg(
    kg: KnowledgeGraph,
    *,
    max_nodes: int = 5000,
) -> dict[str, Any]:
    """Partition a KG into communities for coarsened visualisation.

    Returns the same shape whether or not clustering is triggered so the
    viewer can branch once on `clustered` and otherwise consume the
    result uniformly:

    - `clustered`: bool — True if the node count exceeds `max_nodes`.
    - `node_to_cluster`: dict[entity_name, cluster_id].
    - `cluster_summary`: dict[cluster_id, {size, members}].
    - `cluster_edges`: list of `{source, target, count}` — aggregated
      relation multiplicities between cluster pairs (direction-preserving).
    """
    if len(kg.entities) <= max_nodes:
        return _identity_result(kg)

    # Build an undirected graph for community detection. Isolated
    # entities must still appear as nodes so greedy_modularity places
    # them in singleton communities rather than dropping them.
    g = nx.Graph()
    g.add_nodes_from(kg.entities.keys())
    for r in kg.relations:
        g.add_edge(r.source, r.target)

    communities = list(nx_community.greedy_modularity_communities(g))

    # Deterministic, human-readable cluster ids: pick the
    # lexicographically-smallest member as the cluster label. Avoids
    # the "cluster_3" renumbering problem when the KG is re-extracted
    # with one entity renamed.
    node_to_cluster: dict[str, str] = {}
    cluster_summary: dict[str, dict[str, Any]] = {}
    for comm in communities:
        members = sorted(comm)
        cid = members[0]
        cluster_summary[cid] = {"size": len(members), "members": members}
        for n in members:
            node_to_cluster[n] = cid

    # Aggregate cross-cluster edge multiplicities. We keep direction —
    # a cluster pair connected by A→B edges and B→A edges surfaces as
    # two distinct entries, matching the directed semantics of the
    # underlying KG.
    pair_counts: dict[tuple[str, str], int] = {}
    for r in kg.relations:
        s_cluster = node_to_cluster[r.source]
        t_cluster = node_to_cluster[r.target]
        if s_cluster == t_cluster:
            continue
        key = (s_cluster, t_cluster)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    cluster_edges = [
        {"source": s, "target": t, "count": c}
        for (s, t), c in sorted(pair_counts.items())
    ]

    return {
        "clustered": True,
        "node_to_cluster": node_to_cluster,
        "cluster_summary": cluster_summary,
        "cluster_edges": cluster_edges,
    }
