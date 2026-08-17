"""Tests for the 3D viewer's subgraph export feature (#42).

Scope: assert the export UI is wired into the legend panel, that the
underlying JS state contract (`origIdx` per link, `src_entities` and
`src_relations` in the data payload) is preserved, and that both
download functions (JSON, CSV) are present with the expected MIME types
and column ordering.

We don't execute the JS — these are markup-level contracts that catch
regressions cheaply. End-to-end Selenium-style verification is deferred
because the value of catching contract drift outweighs the value of
testing browser behavior we don't control.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ontograph.dashboard.force3d import build_3d_html


@pytest.fixture
def rendered_html(tmp_path):
    kg = {
        "entities": {
            "inflation": {
                "entity_type": "variable",
                "observations": ["CPI grows"],
                "aliases": ["π"],
                "code_anchors": [{"path": "doc.md", "line": 1}],
            },
            "rate": {"entity_type": "policy_tool"},
            "output": {"entity_type": "variable"},
        },
        "relations": [
            {"source": "rate", "target": "inflation", "relation_type": "affects",
             "sign": "-", "lag": "across_step", "form": "linear",
             "edge_class": "mechanism",
             "provenance": [{"passage": "tighten", "document": "doc.md"}],
             "confidence": 0.9, "conditional_on": ["regime=normal"]},
            {"source": "inflation", "target": "rate", "relation_type": "triggers",
             "sign": "+", "lag": "within_step", "form": "threshold",
             "edge_class": "policy", "provenance": [], "confidence": 0.7},
            {"source": "output", "target": "rate", "relation_type": "corr",
             "sign": "unknown", "lag": "within_step", "form": "unknown",
             "edge_class": "evidence", "provenance": []},
        ],
    }
    kg_path = tmp_path / "kg.json"
    out_path = tmp_path / "kg.html"
    kg_path.write_text(json.dumps(kg), encoding="utf-8")
    build_3d_html(kg_path, out_path)
    return out_path.read_text(encoding="utf-8")


class TestExportUI:
    def test_export_section_present(self, rendered_html):
        assert "<h3>Export" in rendered_html
        assert 'id="exportCtl"' in rendered_html

    def test_json_button_present(self, rendered_html):
        assert 'id="btnExportJSON"' in rendered_html

    def test_csv_button_present(self, rendered_html):
        assert 'id="btnExportCSV"' in rendered_html

    def test_export_count_badge_present(self, rendered_html):
        assert 'id="exportCt"' in rendered_html


class TestExportDataContract:
    def test_origIdx_propagated_to_every_link(self, rendered_html):
        """Each link's `origIdx` is what lets `visibleSlice` re-fetch
        the original Relation dict (with full provenance) from
        `src_relations`. Drop it and exports become lossy."""
        # Three relations → expect origIdx 0, 1, 2 to appear.
        for i in range(3):
            assert f'"origIdx": {i}' in rendered_html or f'"origIdx":{i}' in rendered_html

    def test_src_entities_embedded(self, rendered_html):
        assert '"src_entities":' in rendered_html or '"src_entities" :' in rendered_html

    def test_src_relations_embedded(self, rendered_html):
        assert '"src_relations":' in rendered_html or '"src_relations" :' in rendered_html

    def test_src_entities_carries_full_fields(self, rendered_html):
        """Round-trip lossless = aliases + observations preserved."""
        # The π alias and CPI observation should both be in the data payload.
        assert "CPI grows" in rendered_html
        assert "\\u03c0" in rendered_html or "π" in rendered_html

    def test_src_relations_carries_provenance(self, rendered_html):
        """Provenance survives so the exported KG re-renders with passages."""
        assert "tighten" in rendered_html
        assert "regime=normal" in rendered_html


class TestExportBehavior:
    def test_downloadBlob_helper_present(self, rendered_html):
        assert "function downloadBlob(" in rendered_html

    def test_visibleSlice_helper_present(self, rendered_html):
        """visibleSlice is the load-bearing function — it's what makes
        the export respect the active filter pipeline."""
        assert "function visibleSlice()" in rendered_html

    def test_json_export_uses_application_json_mime(self, rendered_html):
        assert "'application/json'" in rendered_html

    def test_csv_export_uses_text_csv_mime(self, rendered_html):
        assert "'text/csv'" in rendered_html

    def test_csv_columns_in_documented_order(self, rendered_html):
        """The CSV header is part of the public contract — pandas users
        depend on the column ordering. Don't reorder without a release note."""
        assert "source,target,sign,lag,edge_class,relation_type" in rendered_html

    def test_export_count_refreshes_on_filter_change(self, rendered_html):
        """The badge listens to the same `change` events as `applyFilters`."""
        assert "refreshExportCount" in rendered_html
        # And it must be wired to the same toggle classes.
        assert re.search(
            r"querySelectorAll\('\.tgType, \.tgSign, \.tgEc.*'\)\s*\.forEach\([^)]*refreshExportCount",
            rendered_html, flags=re.DOTALL,
        )

    def test_csv_escaping_present(self, rendered_html):
        """A free-text source/target name with commas must not corrupt the CSV.
        We don't run it, but we assert the escaping branch exists."""
        # Look for the escape regex `[",\n]`.
        assert "/[\",\\n]/" in rendered_html or '/[\",\\\\n]/' in rendered_html
