"""Tests for DocAnchor grounding.

A DocAnchor is a concrete, locatable pointer into a source document:
`document + (char_start, char_end) + optional line`. This test suite
locks in three invariants:

1. A DocAnchor with a located span counts toward `Entity.is_grounded()`.
2. A DocAnchor carrying only document+passage (no offset, no line) does
   NOT count — that's no stronger than Provenance.
3. DocAnchor survives KG JSON round-trip.
4. Markdown parser emits section-level `char_offset` / `line` metadata
   that an LLM extractor can consume to locate passages.
"""
from __future__ import annotations

from pathlib import Path

from ontograph.models import (
    CitationAnchor,
    CodeAnchor,
    DocAnchor,
    Entity,
    KnowledgeGraph,
)
from ontograph.parsers import parse_document


class TestIsGrounded:
    def test_located_doc_anchor_counts(self):
        e = Entity(
            name="policy_rate",
            entity_type="parameter",
            doc_anchors=[DocAnchor(
                document="paper.md", char_start=412, char_end=465,
                line=18, passage="The central bank sets policy_rate.",
            )],
        )
        assert e.is_grounded() is True

    def test_line_only_doc_anchor_counts(self):
        """Line numbers alone are a reproducible pointer."""
        e = Entity(
            name="policy_rate", entity_type="parameter",
            doc_anchors=[DocAnchor(document="paper.md", line=18)],
        )
        assert e.is_grounded() is True

    def test_document_only_doc_anchor_does_not_count(self):
        """Without an offset or line, a DocAnchor is no stronger than Provenance."""
        e = Entity(
            name="policy_rate", entity_type="parameter",
            doc_anchors=[DocAnchor(document="paper.md", passage="...")],
        )
        assert e.is_grounded() is False

    def test_empty_document_does_not_count(self):
        e = Entity(
            name="policy_rate", entity_type="parameter",
            doc_anchors=[DocAnchor(document="", char_start=10, char_end=20)],
        )
        assert e.is_grounded() is False

    def test_existing_anchors_still_count(self):
        """DocAnchor addition must not regress existing grounding paths."""
        e1 = Entity(
            name="x", entity_type="parameter",
            code_anchors=[CodeAnchor(repo="r", path="p", line=1, symbol="s")],
        )
        e2 = Entity(
            name="x", entity_type="parameter",
            citation_anchors=[CitationAnchor(key="wiese2024")],
        )
        assert e1.is_grounded() is True
        assert e2.is_grounded() is True


class TestKgRoundTrip:
    def test_doc_anchors_survive_json(self):
        kg = KnowledgeGraph(schema_name="base")
        kg.entities["policy_rate"] = Entity(
            name="policy_rate", entity_type="parameter",
            doc_anchors=[
                DocAnchor(document="paper.md", char_start=10, char_end=42,
                          line=5, section="Monetary policy",
                          passage="policy_rate is set quarterly."),
                DocAnchor(document="paper2.md", char_start=100, char_end=150,
                          line=8, section="Intro", passage="see also."),
            ],
        )
        kg2 = KnowledgeGraph.from_json(kg.to_json())
        before = kg.entities["policy_rate"].doc_anchors
        after = kg2.entities["policy_rate"].doc_anchors
        assert len(after) == 2
        assert after[0].document == "paper.md"
        assert after[0].char_start == 10 and after[0].char_end == 42
        assert after[0].line == 5
        assert after[0].section == "Monetary policy"
        assert after[1].document == "paper2.md"

    def test_missing_doc_anchors_in_old_json_defaults_to_empty(self):
        """Backwards-compat: KGs written before the DocAnchor addition."""
        import json
        payload = json.dumps({
            "entities": {"x": {
                "name": "x", "entity_type": "parameter",
                "aliases": [], "observations": [], "provenance": [],
                "metadata": {}, "code_anchors": [], "citation_anchors": [],
                "abstract": False, "abstract_rationale": "", "confidence": 1.0,
            }},
            "relations": [],
            "schema_name": "base",
            "documents": [],
        })
        kg = KnowledgeGraph.from_json(payload)
        assert kg.entities["x"].doc_anchors == []


class TestMarkdownParserLineTracking:
    def test_parser_populates_char_offset_and_line(self, tmp_path: Path):
        md = "# Intro\n\nfirst body line.\n\n## Sub\n\nsecond body line.\n"
        p = tmp_path / "sample.md"
        p.write_text(md, encoding="utf-8")
        doc = parse_document(p)
        assert len(doc.sections) == 2
        intro = doc.sections[0]
        sub = doc.sections[1]
        # Both should carry reproducible offsets.
        assert intro.char_offset >= 0
        assert sub.char_offset >= 0
        assert intro.char_offset < sub.char_offset
        # Line tracking is 1-indexed.
        assert intro.line is not None and intro.line >= 1
        assert sub.line is not None and sub.line > intro.line
        # The recorded char_offset should point at a location whose prefix
        # matches the section body (modulo .strip() on the body).
        assert doc.full_text[intro.char_offset:].startswith("first body line.")
        assert doc.full_text[sub.char_offset:].startswith("second body line.")

    def test_passage_is_locatable(self, tmp_path: Path):
        """Round-trip: parse MD → locate a passage → recover it exactly."""
        from ontograph.llm_extractor import _locate_passage

        md = ("# Monetary policy\n"
              "\n"
              "The central bank sets the policy_rate at quarterly frequency.\n"
              "Transmission to deposit_rate is via a markup.\n")
        p = tmp_path / "mp.md"
        p.write_text(md, encoding="utf-8")
        doc = parse_document(p)
        section = doc.sections[0]
        passage = "The central bank sets the policy_rate at quarterly frequency."
        cs, ce, ln = _locate_passage(section, passage)
        assert cs >= 0
        assert doc.full_text[cs:ce] == passage
        assert ln is not None and ln >= section.line

    def test_missing_passage_returns_sentinel(self, tmp_path: Path):
        from ontograph.llm_extractor import _locate_passage

        md = "# Intro\n\nfirst body line.\n"
        p = tmp_path / "m.md"
        p.write_text(md, encoding="utf-8")
        doc = parse_document(p)
        cs, ce, ln = _locate_passage(doc.sections[0], "not present in text")
        assert cs == -1 and ce == -1 and ln is None


class TestGroundednessMetricRecognisesDocAnchors:
    """The quality gate should treat DocAnchor-grounded entities as grounded."""

    def test_doc_anchor_grounding_passes_gate(self):
        from ontograph.quality import compute_quality_report
        from ontograph.schema import load_schema

        kg = KnowledgeGraph(schema_name="base")
        # Two entities: one DocAnchor-grounded, one ungrounded.
        kg.entities["grounded"] = Entity(
            name="grounded", entity_type="concept",
            doc_anchors=[DocAnchor(document="d.md", char_start=0, char_end=5)],
        )
        kg.entities["ungrounded"] = Entity(
            name="ungrounded", entity_type="concept",
        )

        schema = load_schema("base")
        # Force the gate: require 50% grounded; we have 50% exactly.
        schema.validation = {"groundedness_target": 0.5}
        report = compute_quality_report(kg, schema)
        assert report.groundedness["ratio"] == 0.5
        assert report.gates_passed
