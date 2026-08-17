"""Tests for cycle enumeration used by the viewer and the cycle-report CLI.

Covers `ontograph.cycles.enumerate_cycles` — bounded simple-cycle enumeration
split by lag, with embedded sign-product.
"""
from __future__ import annotations

import pytest

from ontograph.cycles import enumerate_cycles


def _rel(src, tgt, *, sign="unknown", lag="within_step"):
    return {
        "source": src, "target": tgt, "relation_type": "r",
        "sign": sign, "lag": lag, "edge_class": "mechanism",
    }


class TestBasicEnumeration:
    def test_no_cycles_empty_result(self):
        rels = [
            _rel("a", "b", sign="+"),
            _rel("b", "c", sign="+"),
            _rel("c", "d", sign="+"),
        ]
        assert enumerate_cycles(rels) == []

    def test_two_cycle(self):
        rels = [
            _rel("a", "b", sign="+"),
            _rel("b", "a", sign="-"),
        ]
        cycles = enumerate_cycles(rels)
        assert len(cycles) == 1
        c = cycles[0]
        assert sorted(c["nodes"]) == ["a", "b"]
        assert c["lag"] == "within_step"
        assert c["product"] == "-"  # + * - = -

    def test_three_cycle_sign_product(self):
        rels = [
            _rel("a", "b", sign="+"),
            _rel("b", "c", sign="-"),
            _rel("c", "a", sign="-"),
        ]
        cycles = enumerate_cycles(rels)
        assert len(cycles) == 1
        # + * - * - = +
        assert cycles[0]["product"] == "+"

    def test_self_loop_reported_as_length_one_cycle(self):
        rels = [_rel("x", "x", sign="-")]
        cycles = enumerate_cycles(rels)
        assert len(cycles) == 1
        assert cycles[0]["nodes"] == ["x"]
        assert cycles[0]["product"] == "-"


class TestLagSeparation:
    def test_cycles_split_by_lag(self):
        """Cycles that span both lag types should be enumerated only within
        each lag's subgraph — not across lags."""
        rels = [
            _rel("a", "b", sign="+", lag="within_step"),
            _rel("b", "a", sign="+", lag="across_step"),  # different lag
        ]
        cycles = enumerate_cycles(rels)
        # No cycle: the two edges are in different lag buckets.
        assert cycles == []

    def test_two_cycles_different_lags(self):
        rels = [
            _rel("a", "b", sign="+", lag="within_step"),
            _rel("b", "a", sign="-", lag="within_step"),
            _rel("c", "d", sign="+", lag="across_step"),
            _rel("d", "c", sign="-", lag="across_step"),
        ]
        cycles = enumerate_cycles(rels)
        lags = {c["lag"] for c in cycles}
        assert lags == {"within_step", "across_step"}
        assert len(cycles) == 2

    def test_within_step_sorted_before_across_step(self):
        rels = [
            _rel("c", "d", lag="across_step"),
            _rel("d", "c", lag="across_step"),
            _rel("a", "b", lag="within_step"),
            _rel("b", "a", lag="within_step"),
        ]
        cycles = enumerate_cycles(rels)
        # within_step first (stable ordering contract).
        assert cycles[0]["lag"] == "within_step"
        assert cycles[1]["lag"] == "across_step"


class TestEdgeIndices:
    def test_edge_idxs_refer_to_original_relations(self):
        rels = [
            _rel("a", "b", sign="+"),   # idx 0
            _rel("b", "a", sign="-"),   # idx 1
            _rel("x", "y", sign="+"),   # idx 2 (not in a cycle)
        ]
        cycles = enumerate_cycles(rels)
        assert len(cycles) == 1
        # edge_idxs must index back into the original relations list.
        for i in cycles[0]["edge_idxs"]:
            assert 0 <= i < len(rels)
            assert rels[i]["source"] in cycles[0]["nodes"]
            assert rels[i]["target"] in cycles[0]["nodes"]


class TestBounding:
    def test_max_len_filters_long_cycles(self):
        # Build a 5-node cycle; cap at length 3 → no cycles reported.
        rels = [
            _rel("a", "b"), _rel("b", "c"), _rel("c", "d"),
            _rel("d", "e"), _rel("e", "a"),
        ]
        cycles = enumerate_cycles(rels, max_len=3)
        assert cycles == []

    def test_max_len_accepts_short_cycles(self):
        rels = [_rel("a", "b"), _rel("b", "a")]
        cycles = enumerate_cycles(rels, max_len=3)
        assert len(cycles) == 1


class TestUnknownSign:
    def test_cycle_with_unknown_sign_gets_unknown_product(self):
        rels = [
            _rel("a", "b", sign="+"),
            _rel("b", "a", sign="unknown"),
        ]
        cycles = enumerate_cycles(rels)
        assert len(cycles) == 1
        assert cycles[0]["product"] == "unknown"
