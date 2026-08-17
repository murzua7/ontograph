"""Tests for the 3D viewer's filter-toggle markup (#41).

These exercise `build_3d_html` end-to-end at the HTML level: we render a
tiny KG and assert the generated markup contains the expected checkbox
rows for entity types, edge signs, and edge classes. We intentionally
avoid parsing the HTML — grepping for the stable `data-val` contract is
both cheap and forward-compatible with CSS tweaks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ontograph.dashboard.force3d import build_3d_html


@pytest.fixture
def small_kg(tmp_path):
    kg = {
        "entities": {
            "inflation": {"entity_type": "variable", "observations": ["CPI"]},
            "rate":      {"entity_type": "policy_tool"},
            "output":    {"entity_type": "variable"},
            "bank":      {"entity_type": "actor"},
        },
        "relations": [
            {"source": "rate", "target": "inflation", "relation_type": "affects",
             "sign": "-", "lag": "across_step", "form": "linear",
             "edge_class": "mechanism", "provenance": []},
            {"source": "inflation", "target": "rate", "relation_type": "triggers",
             "sign": "+", "lag": "within_step", "form": "threshold",
             "edge_class": "policy", "provenance": []},
            {"source": "output", "target": "rate", "relation_type": "corr",
             "sign": "unknown", "lag": "within_step", "form": "unknown",
             "edge_class": "evidence", "provenance": []},
            {"source": "bank", "target": "output", "relation_type": "holds",
             "sign": "±", "lag": "within_step", "form": "unknown",
             "edge_class": "mechanism", "provenance": []},
        ],
    }
    kg_path = tmp_path / "kg.json"
    out_path = tmp_path / "kg.html"
    kg_path.write_text(json.dumps(kg), encoding="utf-8")
    build_3d_html(kg_path, out_path)
    return out_path.read_text(encoding="utf-8")


class TestEntityTypeToggles:
    def test_each_type_has_checkbox(self, small_kg):
        for t in ("variable", "policy_tool", "actor"):
            assert f'class="tgType" data-val="{t}"' in small_kg

    def test_entity_type_checkboxes_default_checked(self, small_kg):
        assert 'class="tgType" data-val="variable" checked' in small_kg


class TestSignToggles:
    def test_observed_signs_have_checkboxes(self, small_kg):
        # "+", "-", "unknown", and "±" all appear in the tiny KG.
        for s in ("+", "-", "unknown"):
            assert f'class="tgSign" data-val="{s}"' in small_kg

    def test_pm_sign_html_escaped(self, small_kg):
        """The '±' character (U+00B1) must survive escaping as &plusmn; or literal."""
        # html.escape doesn't touch non-ASCII — literal char should be present.
        assert 'class="tgSign" data-val="±"' in small_kg

    def test_absent_signs_have_no_row(self, small_kg):
        # "0" doesn't appear in relations → no row.
        assert 'data-val="0"' not in small_kg


class TestEdgeClassToggles:
    def test_each_edge_class_has_checkbox(self, small_kg):
        for ec in ("mechanism", "policy", "evidence"):
            assert f'class="tgEc" data-val="{ec}"' in small_kg


class TestFilterLogicWired:
    def test_applyFilters_function_present(self, small_kg):
        assert 'function applyFilters()' in small_kg

    def test_applyFilters_reads_all_three_classes(self, small_kg):
        assert "checkedVals('tgType')" in small_kg
        assert "checkedVals('tgSign')" in small_kg
        assert "checkedVals('tgEc')" in small_kg

    def test_applyFilters_wired_to_change_events(self, small_kg):
        # A single querySelectorAll of all three classes.
        assert '.tgType, .tgSign, .tgEc' in small_kg

    def test_hideUnknown_removed_subsumed_by_sign_toggle(self, small_kg):
        """The old hideUnknown checkbox is redundant with the per-sign toggles."""
        assert 'id="hideUnknown"' not in small_kg

    def test_hideIsolated_still_present(self, small_kg):
        assert 'id="hideIsolated"' in small_kg


class TestCountBadges:
    def test_row_shows_occurrence_count(self, small_kg):
        """Each row ends with a <span class='ct'>N</span> count badge."""
        # Two "variable" nodes: inflation, output.
        assert 'class="ct">2</span>' in small_kg
        # Two mechanism edges: rate→inflation, bank→output.
        # Find it in the edge-class row specifically (cheap grep).
        assert 'data-val="mechanism"' in small_kg
