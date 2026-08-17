"""Tests for the targeted sign re-prompt pass (#45).

Walks unknown-sign relations and asks the LLM a single focused question
per relation: "given this passage, is the effect +/-/±/0/unknown?"
Updates the KG in place when the new answer is confident.
"""
from __future__ import annotations

import json

import pytest

from ontograph.llm_client import LLMClient
from ontograph.models import Entity, KnowledgeGraph, Provenance, Relation
from ontograph.reprompt import (
    build_sign_reprompt_messages,
    targeted_reprompt_unknown_signs,
)


def _kg_with_unknowns(*signs: str) -> KnowledgeGraph:
    """Build a KG with one relation per `signs` entry, each between dedicated nodes.
    Every relation carries a non-empty provenance passage."""
    kg = KnowledgeGraph()
    for i, _ in enumerate(signs):
        a, b = f"a{i}", f"b{i}"
        kg.add_entity(Entity(name=a, entity_type="variable"))
        kg.add_entity(Entity(name=b, entity_type="variable"))
    for i, s in enumerate(signs):
        kg.add_relation(Relation(
            source=f"a{i}", target=f"b{i}", relation_type="r",
            sign=s, lag="within_step", edge_class="mechanism",
            provenance=[Provenance(
                document="doc.md", section="body",
                passage=f"Passage {i}: a{i} increases b{i} monotonically.",
            )],
        ))
    return kg


class TestBuildPromptMessages:
    def test_passage_and_relation_appear_in_prompt(self):
        r = Relation(
            source="rate", target="inflation", relation_type="affects",
            sign="unknown", lag="across_step", edge_class="mechanism",
            provenance=[Provenance(document="doc", passage="Higher rates reduce CPI.")],
        )
        msgs = build_sign_reprompt_messages(r)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        user = msgs[1]["content"]
        assert "rate" in user and "inflation" in user
        assert "affects" in user
        assert "Higher rates reduce CPI." in user


class TestTargetedReprompt:
    def test_updates_confident_unknown_signs(self):
        kg = _kg_with_unknowns("unknown", "unknown")
        client = LLMClient(mode="mock")
        client.set_mock_responses([
            '{"sign": "+", "confidence": 0.9, "rationale": "clearly positive"}',
            '{"sign": "-", "confidence": 0.8, "rationale": "clearly negative"}',
        ])
        report = targeted_reprompt_unknown_signs(kg, client)
        assert report["n_candidates"] == 2
        assert report["n_updated"] == 2
        assert kg.relations[0].sign == "+"
        assert kg.relations[1].sign == "-"

    def test_skips_low_confidence_answers(self):
        kg = _kg_with_unknowns("unknown")
        client = LLMClient(mode="mock")
        client.set_mock_responses([
            '{"sign": "+", "confidence": 0.3, "rationale": "not sure"}',
        ])
        report = targeted_reprompt_unknown_signs(kg, client, min_confidence=0.6)
        assert report["n_updated"] == 0
        assert kg.relations[0].sign == "unknown"

    def test_skips_relations_without_provenance(self):
        kg = KnowledgeGraph()
        kg.add_entity(Entity(name="a", entity_type="variable"))
        kg.add_entity(Entity(name="b", entity_type="variable"))
        kg.add_relation(Relation(
            source="a", target="b", relation_type="r",
            sign="unknown", lag="within_step", edge_class="mechanism",
            provenance=[],  # no passage
        ))
        client = LLMClient(mode="mock")
        client.set_mock_responses(['{"sign": "+", "confidence": 0.9}'])
        report = targeted_reprompt_unknown_signs(kg, client)
        assert report["n_candidates"] == 0
        assert len(report["skipped"]) == 1
        assert report["skipped"][0]["reason"] == "no_provenance"
        assert kg.relations[0].sign == "unknown"

    def test_leaves_known_signs_untouched(self):
        kg = _kg_with_unknowns("+", "unknown")
        client = LLMClient(mode="mock")
        client.set_mock_responses([
            '{"sign": "-", "confidence": 0.9}',
        ])
        report = targeted_reprompt_unknown_signs(kg, client)
        assert report["n_candidates"] == 1  # only the unknown
        assert kg.relations[0].sign == "+"  # untouched
        assert kg.relations[1].sign == "-"  # updated

    def test_invalid_llm_response_kept_as_unknown(self):
        """Malformed / unparseable LLM answer must NOT corrupt the KG."""
        kg = _kg_with_unknowns("unknown")
        client = LLMClient(mode="mock")
        client.set_mock_responses(["not valid json at all"])
        report = targeted_reprompt_unknown_signs(kg, client)
        assert report["n_updated"] == 0
        assert kg.relations[0].sign == "unknown"
        assert len(report["skipped"]) == 1

    def test_update_records_new_sign_and_confidence(self):
        kg = _kg_with_unknowns("unknown")
        client = LLMClient(mode="mock")
        client.set_mock_responses([
            '{"sign": "±", "confidence": 0.7, "rationale": "mixed evidence"}',
        ])
        report = targeted_reprompt_unknown_signs(kg, client)
        assert len(report["updates"]) == 1
        u = report["updates"][0]
        assert u["new_sign"] == "±"
        assert u["confidence"] == 0.7
        assert u["source"] == "a0" and u["target"] == "b0"
