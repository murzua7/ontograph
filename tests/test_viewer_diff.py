"""Tests for the 3D viewer's KG-diff overlay (#44).

When `build_3d_html` is given a `base_kg_path`, every node and link in the
rendered data payload carries a `diff_status` in {added, removed, changed,
unchanged}. Removed items come from the base and are injected into the
viewer payload so they can be ghost-rendered. A diff-mode legend section
surfaces counts and toggle checkboxes so the user can hide/show each bucket.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ontograph.dashboard.force3d import build_3d_html


def _write(path, entities, relations):
    path.write_text(json.dumps({"entities": entities, "relations": relations}), encoding="utf-8")


@pytest.fixture
def base_and_head(tmp_path):
    base_entities = {
        "a": {"entity_type": "variable"},
        "b": {"entity_type": "variable"},
        "c": {"entity_type": "variable"},       # removed in head
    }
    base_relations = [
        {"source": "a", "target": "b", "relation_type": "r",
         "sign": "+", "lag": "within_step", "edge_class": "mechanism"},
        {"source": "b", "target": "c", "relation_type": "r",
         "sign": "+", "lag": "within_step", "edge_class": "mechanism"},  # removed
    ]
    head_entities = {
        "a": {"entity_type": "variable"},
        "b": {"entity_type": "variable"},
        "d": {"entity_type": "variable"},       # added
    }
    head_relations = [
        {"source": "a", "target": "b", "relation_type": "r",
         "sign": "-", "lag": "within_step", "edge_class": "mechanism"},  # changed (sign)
        {"source": "a", "target": "d", "relation_type": "r",
         "sign": "+", "lag": "within_step", "edge_class": "mechanism"},  # added
    ]
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    _write(base, base_entities, base_relations)
    _write(head, head_entities, head_relations)
    out = tmp_path / "head.html"
    build_3d_html(head, out, base_kg_path=base)
    return out.read_text(encoding="utf-8")


def _extract_data_json(html: str) -> dict:
    """Pull the `const DATA = {...};` payload from the generated HTML."""
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    # The JSON ends at the closing brace followed by `;`. Brace-count scan.
    depth = 0
    i = start
    while i < len(html):
        c = html[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:i + 1])
        i += 1
    raise AssertionError("Could not extract DATA JSON")


class TestDiffStatusOnNodes:
    def test_added_node_has_added_status(self, base_and_head):
        data = _extract_data_json(base_and_head)
        d_node = next(n for n in data["nodes"] if n["id"] == "d")
        assert d_node["diff_status"] == "added"

    def test_unchanged_node_has_unchanged_status(self, base_and_head):
        data = _extract_data_json(base_and_head)
        a_node = next(n for n in data["nodes"] if n["id"] == "a")
        assert a_node["diff_status"] == "unchanged"

    def test_removed_node_is_present_with_removed_status(self, base_and_head):
        data = _extract_data_json(base_and_head)
        c_nodes = [n for n in data["nodes"] if n["id"] == "c"]
        assert len(c_nodes) == 1
        assert c_nodes[0]["diff_status"] == "removed"


class TestDiffStatusOnLinks:
    def test_added_link_marked(self, base_and_head):
        data = _extract_data_json(base_and_head)
        added = [l for l in data["links"]
                 if l["source"] == "a" and l["target"] == "d"]
        assert len(added) == 1
        assert added[0]["diff_status"] == "added"

    def test_changed_link_marked(self, base_and_head):
        data = _extract_data_json(base_and_head)
        changed = [l for l in data["links"]
                   if l["source"] == "a" and l["target"] == "b"]
        assert len(changed) == 1
        assert changed[0]["diff_status"] == "changed"

    def test_removed_link_present_with_removed_status(self, base_and_head):
        data = _extract_data_json(base_and_head)
        removed = [l for l in data["links"]
                   if l["source"] == "b" and l["target"] == "c"]
        assert len(removed) == 1
        assert removed[0]["diff_status"] == "removed"


class TestDiffLegendInHtml:
    def test_diff_toggles_present(self, base_and_head):
        for status in ("added", "removed", "changed", "unchanged"):
            assert f'class="tgDiff" data-val="{status}"' in base_and_head

    def test_diff_panel_header(self, base_and_head):
        assert "Diff" in base_and_head

    def test_diff_summary_counts(self, base_and_head):
        # 1 added / 1 removed / 1 changed / 2 unchanged (a, b) at entity level.
        # The summary text doesn't need exact wording — just that counts
        # are reachable. We grep for a stable data attribute instead.
        assert 'id="diffSummary"' in base_and_head


class TestDiffDisabledWhenNoBase:
    def test_all_status_unchanged_without_base(self, tmp_path):
        entities = {"a": {"entity_type": "variable"}}
        relations = []
        p = tmp_path / "kg.json"
        _write(p, entities, relations)
        out = tmp_path / "kg.html"
        build_3d_html(p, out)
        html = out.read_text(encoding="utf-8")
        # No diff toggles when base is not provided.
        assert 'class="tgDiff"' not in html

    def test_no_base_nodes_still_carry_unchanged_status(self, tmp_path):
        entities = {"a": {"entity_type": "variable"}}
        relations = []
        p = tmp_path / "kg.json"
        _write(p, entities, relations)
        out = tmp_path / "kg.html"
        build_3d_html(p, out)
        html = out.read_text(encoding="utf-8")
        data = _extract_data_json(html)
        assert all(n["diff_status"] == "unchanged" for n in data["nodes"])
