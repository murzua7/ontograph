"""Tests for falsifier-as-training-signal re-extraction (#49).

When the simulator's empirical sign disagrees with the KG's extracted sign
(the `flipped` bucket of `falsify()`), that disagreement is a *prior*: the
edge is suspicious enough to warrant a targeted re-prompt. The LLM is
shown the passage, the original sign, and the simulator's opposing sign,
and asked to re-rule. Passage remains ground truth; the simulator prior
only *selects which edges to re-examine*.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ontograph.falsifier_prior import (
    apply_falsifier_prior,
    build_prior_reprompt_messages,
)
from ontograph.falsify import FalsifyReport
from ontograph.models import Entity, KnowledgeGraph, Provenance, Relation


class _FakeClient:
    """Record messages + return canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat_json(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("no more responses")
        return self._responses.pop(0)


def _rel(source, target, sign, passage="A causes B."):
    return Relation(
        source=source, target=target,
        relation_type="r", sign=sign, lag="within_step",
        edge_class="mechanism",
        provenance=[Provenance(document="d.md", passage=passage)],
    )


def _kg_with(relations):
    kg = KnowledgeGraph()
    names = {r.source for r in relations} | {r.target for r in relations}
    for n in names:
        kg.add_entity(Entity(name=n, entity_type="variable"))
    for r in relations:
        kg.add_relation(r)
    return kg


def _flipped_entry(source, target, kg_sign, sim_sign):
    return {
        "source": source, "target": target,
        "relation_type": "r", "edge_class": "mechanism",
        "kg_sign": kg_sign, "sim_sign": sim_sign,
        "kg_lag": "within_step", "sim_lag": "within_step",
    }


class TestBuildPriorReprompt:
    def test_message_includes_kg_sign_sim_sign_and_passage(self):
        r = _rel("A", "B", "+", passage="A raises B.")
        msgs = build_prior_reprompt_messages(
            r, sim_sign="-", sim_lag="within_step",
        )
        # Two messages: system + user.
        assert len(msgs) == 2
        user = msgs[1]["content"]
        assert "A raises B." in user
        assert "A" in user and "B" in user
        # Both signs surfaced so the LLM can arbitrate.
        assert "+" in user
        assert "-" in user

    def test_system_prompt_pins_passage_as_ground_truth(self):
        r = _rel("A", "B", "+")
        msgs = build_prior_reprompt_messages(r, sim_sign="-", sim_lag="within_step")
        sys = msgs[0]["content"].lower()
        # Spec §2: the passage is authoritative; the simulator only selects.
        assert "passage" in sys
        assert "simulator" in sys or "prior" in sys


class TestApplyFalsifierPrior:
    def test_flipped_edge_updated_when_llm_confident(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([
            {"sign": "-", "confidence": 0.9, "rationale": "passage says depresses"},
        ])
        out = apply_falsifier_prior(kg, report, client)
        assert kg.relations[0].sign == "-"
        assert out["n_updated"] == 1
        assert out["updates"][0]["new_sign"] == "-"

    def test_flipped_edge_preserved_when_llm_defends_original(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([
            {"sign": "+", "confidence": 0.95,
             "rationale": "simulator wrong, passage unambiguously says +"},
        ])
        out = apply_falsifier_prior(kg, report, client)
        # Sign identical to original — not counted as an update.
        assert kg.relations[0].sign == "+"
        assert out["n_updated"] == 0
        assert out["n_defended"] == 1

    def test_flipped_edge_demoted_to_ambiguous(self):
        """LLM may decide the passage actually is mixed — valid outcome."""
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([
            {"sign": "±", "confidence": 0.8,
             "rationale": "passage hedges on regime"},
        ])
        out = apply_falsifier_prior(kg, report, client)
        assert kg.relations[0].sign == "±"
        assert out["n_updated"] == 1

    def test_low_confidence_response_skipped(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([
            {"sign": "-", "confidence": 0.3, "rationale": "uncertain"},
        ])
        out = apply_falsifier_prior(kg, report, client, min_confidence=0.6)
        assert kg.relations[0].sign == "+"  # unchanged
        assert out["n_updated"] == 0
        assert any("low_confidence" in s["reason"] for s in out["skipped"])

    def test_only_flipped_edges_reprompted_not_confirmed(self):
        r_flipped = _rel("A", "B", "+")
        r_confirmed = _rel("C", "D", "+")
        kg = _kg_with([r_flipped, r_confirmed])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            confirmed=[_flipped_entry("C", "D", "+", "+")],
            total=2, examined=2,
        )
        client = _FakeClient([
            {"sign": "-", "confidence": 0.9, "rationale": "x"},
        ])
        apply_falsifier_prior(kg, report, client)
        # Exactly one LLM call: only the flipped edge was re-prompted.
        assert len(client.calls) == 1

    def test_missing_provenance_skipped(self):
        r = Relation(
            source="A", target="B", relation_type="r",
            sign="+", lag="within_step", edge_class="mechanism",
            # No provenance — anchored re-extraction requires a passage.
        )
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([])
        out = apply_falsifier_prior(kg, report, client)
        assert len(client.calls) == 0
        assert any("no_provenance" in s["reason"] for s in out["skipped"])

    def test_malformed_llm_response_skipped(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient(["not a dict"])
        out = apply_falsifier_prior(kg, report, client)
        assert kg.relations[0].sign == "+"
        assert any("malformed" in s["reason"] for s in out["skipped"])

    def test_llm_exception_skipped_not_raised(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )

        class _BoomClient:
            calls: list = []
            def chat_json(self, messages):
                self.calls.append(messages)
                raise RuntimeError("network is on fire")

        out = apply_falsifier_prior(kg, report, _BoomClient())
        assert kg.relations[0].sign == "+"
        assert any("llm_error" in s["reason"] for s in out["skipped"])

    def test_report_shape(self):
        r = _rel("A", "B", "+")
        kg = _kg_with([r])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([
            {"sign": "-", "confidence": 0.9, "rationale": "x"},
        ])
        out = apply_falsifier_prior(kg, report, client)
        assert set(out.keys()) >= {
            "n_candidates", "n_updated", "n_defended",
            "updates", "skipped",
        }

    def test_edge_missing_from_kg_skipped(self):
        """Falsify report might reference an edge pruned from the KG since."""
        kg = _kg_with([_rel("X", "Y", "+")])
        report = FalsifyReport(
            flipped=[_flipped_entry("A", "B", "+", "-")],
            total=1, examined=1,
        )
        client = _FakeClient([])
        out = apply_falsifier_prior(kg, report, client)
        assert len(client.calls) == 0
        assert any("not_in_kg" in s["reason"] for s in out["skipped"])
