"""Cycle enumeration over discipline-v1 relations.

Public surface used by the 3D viewer (cycle-explorer panel) and the
`ontograph cycle-report` CLI subcommand. Cycles are split by lag so the
two temporal regimes (within-step / across-step) are reported separately —
mixing them would produce phantom cycles that don't actually exist in
either subgraph.
"""
from __future__ import annotations

import networkx as nx

from ontograph.paths import sign_product

# Default caps. Bounded so dense graphs don't blow up downstream consumers:
# `simple_cycles` enumerates everything up to `MAX_CYCLE_LEN` and we keep the
# first `MAX_CYCLES` per lag bucket after sorting by length.
MAX_CYCLE_LEN = 8
MAX_CYCLES = 50


def enumerate_cycles(
    relations: list[dict],
    *,
    max_len: int = MAX_CYCLE_LEN,
    max_cycles: int = MAX_CYCLES,
) -> list[dict]:
    """Bounded simple-cycle enumeration, split by lag bucket.

    For each cycle we emit the ordered node list, the link indices it
    traverses (indexing into `relations`), the sign list, and the composed
    sign product. Indices are stable for the lifetime of `relations` —
    the viewer uses them to highlight cycle edges on click.

    Note: parallel edges between the same `(u, v)` pair at the same lag
    are resolved by picking the first matching relation in `relations`
    order. If two parallel edges carry different signs, the reported
    cycle product depends on that order. Deferred fix — enumerate both
    sign-parametrisations when it matters.
    """
    by_pair: dict[tuple[str, str], list[int]] = {}
    for idx, r in enumerate(relations):
        by_pair.setdefault((r["source"], r["target"]), []).append(idx)

    out: list[dict] = []
    for lag in ("within_step", "across_step"):
        edges = [(r["source"], r["target"]) for r in relations if r.get("lag") == lag]
        if not edges:
            continue
        g = nx.DiGraph()
        g.add_edges_from(edges)
        enumerated: list[list[str]] = []
        try:
            for cyc in nx.simple_cycles(g, length_bound=max_len):
                if len(cyc) <= 1:
                    continue  # skip self-loops — they're reported separately below
                enumerated.append(cyc)
                if len(enumerated) >= max_cycles * 4:
                    # Over-collect so sorting-by-length picks the shortest.
                    break
        except TypeError:
            # Older NetworkX: no length_bound kwarg. Fall back.
            for cyc in nx.simple_cycles(g):
                if 2 <= len(cyc) <= max_len:
                    enumerated.append(cyc)
                if len(enumerated) >= max_cycles * 4:
                    break
        # Self-loops: simple_cycles treats them as 1-cycles.
        for n in g.nodes():
            if g.has_edge(n, n):
                enumerated.append([n])
        enumerated.sort(key=lambda c: (len(c), ",".join(c)))
        enumerated = enumerated[:max_cycles]

        for cyc in enumerated:
            # Reconstruct the edge traversal for this cycle (closing edge last).
            node_seq = list(cyc) + [cyc[0]]
            edge_idxs: list[int] = []
            signs: list[str] = []
            broken = False
            for u, v in zip(node_seq[:-1], node_seq[1:]):
                candidates = [
                    i for i in by_pair.get((u, v), [])
                    if relations[i].get("lag") == lag
                ]
                if not candidates:
                    broken = True
                    break
                i = candidates[0]
                edge_idxs.append(i)
                signs.append(relations[i].get("sign") or "unknown")
            if broken:
                continue
            out.append({
                "lag": lag,
                "nodes": list(cyc),
                "edge_idxs": edge_idxs,
                "signs": signs,
                "product": sign_product(signs),
            })

    # Stable ordering across lag buckets: within_step first, then by length.
    out.sort(key=lambda c: (0 if c["lag"] == "within_step" else 1,
                            len(c["nodes"]), ",".join(c["nodes"])))
    return out
