"""Golden-path round-trip test for discipline-v1 fields.

Was motivated by the 2026-04-20 file-mtime bug: the LLM extractor wrote
correct sign/lag/form/edge_class/conditional_on into each `Relation`, but
`OntologyGraph.to_kg()` reconstructed relations from the NetworkX edge's
`relation_type`/`weight` only, dropping the discipline-v1 fields silently.
Every downstream quality metric then defaulted, and signedness stuck at
0.0 despite the extractor working correctly.

This test asserts that every discipline-v1 field survives the full
pipeline:  OntologyGraph.add_* → to_kg → to_json → from_json → from_kg
→ to_kg. If any link in that chain drops a field, the test fails at the
exact field, not at an aggregate metric three levels downstream.
"""
from __future__ import annotations

import json

from ontograph.graph import OntologyGraph
from ontograph.models import (
    CitationAnchor,
    CodeAnchor,
    Entity,
    KnowledgeGraph,
    Provenance,
    Relation,
)


# --- Fixture builders ---------------------------------------------------


def _golden_entity(name: str = "policy_rate", etype: str = "parameter") -> Entity:
    return Entity(
        name=name,
        entity_type=etype,
        aliases=["rate", "r_t"],
        observations=[
            "Central bank policy rate, set at quarterly frequency.",
            "Transmits to deposit_rate via a bank markup.",
        ],
        provenance=[
            Provenance(
                document="wiki/architecture/dabs.md",
                section="Monetary policy",
                page=3,
                passage="The central bank sets policy_rate at quarterly frequency.",
                char_offset=412,
            )
        ],
        metadata={"subtype": "exogenous_policy"},
        code_anchors=[
            CodeAnchor(repo="DiffMABM", path="dabs/policy.py", line=42, symbol="policy_rate")
        ],
        citation_anchors=[CitationAnchor(key="wiese2024", pages="12-14")],
        abstract=False,
        abstract_rationale="",
        confidence=0.92,
    )


def _golden_relation(
    source: str = "policy_rate",
    target: str = "deposit_rate",
    sign: str = "+",
) -> Relation:
    return Relation(
        source=source,
        target=target,
        relation_type="transmits_to",
        weight=0.8,
        provenance=[
            Provenance(
                document="wiki/architecture/dabs.md",
                section="Monetary policy",
                page=3,
                passage="policy_rate transmits to deposit_rate via a markup.",
                char_offset=488,
            )
        ],
        metadata={"derivation": "identity"},
        sign=sign,
        lag="within_step",
        form="linear",
        conditional_on=["inflation_regime", "central_bank_active"],
        edge_class="mechanism",
        confidence=0.88,
    )


def _populated_graph() -> OntologyGraph:
    g = OntologyGraph()
    g.add_entity(_golden_entity("policy_rate", "parameter"))
    g.add_entity(_golden_entity("deposit_rate", "state_variable"))
    g.add_entity(_golden_entity("hpi", "state_variable"))
    g.add_entity(_golden_entity("rental_index", "state_variable"))
    g.add_relation(_golden_relation("policy_rate", "deposit_rate", "+"))
    g.add_relation(_golden_relation("hpi", "rental_index", "+"))
    # Cover the rarer sign/lag/form values too.
    g.add_relation(
        Relation(
            source="deposit_rate",
            target="hpi",
            relation_type="dampens",
            weight=0.4,
            sign="-",
            lag="Q+2",
            form="threshold",
            conditional_on=[],
            edge_class="parameter",
            confidence=0.55,
            provenance=[
                Provenance(document="d.md", passage="higher deposit rates slow hpi.")
            ],
            metadata={},
        )
    )
    g.add_relation(
        Relation(
            source="hpi",
            target="policy_rate",
            relation_type="informs",
            weight=0.3,
            sign="±",
            lag="across_step",
            form="monotone",
            conditional_on=["regime"],
            edge_class="abstract",
            confidence=0.7,
        )
    )
    g.add_relation(
        Relation(
            source="rental_index",
            target="hpi",
            relation_type="is_component_of",
            weight=1.0,
            sign="0",
            lag="within_step",
            form="identity",
            conditional_on=[],
            edge_class="identity",
            confidence=1.0,
        )
    )
    return g


# --- Tests --------------------------------------------------------------


class TestDisciplineV1RoundTrip:
    """Every discipline-v1 field must survive the full serialization chain."""

    def test_entity_fields_survive_json_round_trip(self):
        kg_before = _populated_graph().to_kg()
        kg_after = KnowledgeGraph.from_json(kg_before.to_json())

        e_before = kg_before.entities["policy_rate"]
        e_after = kg_after.entities["policy_rate"]

        assert e_after.name == e_before.name
        assert e_after.entity_type == e_before.entity_type
        assert e_after.aliases == e_before.aliases
        assert e_after.observations == e_before.observations
        assert len(e_after.provenance) == len(e_before.provenance)
        assert e_after.provenance[0].document == e_before.provenance[0].document
        assert e_after.provenance[0].section == e_before.provenance[0].section
        assert e_after.provenance[0].page == e_before.provenance[0].page
        assert e_after.provenance[0].passage == e_before.provenance[0].passage
        assert e_after.provenance[0].char_offset == e_before.provenance[0].char_offset
        assert e_after.metadata == e_before.metadata
        assert len(e_after.code_anchors) == len(e_before.code_anchors)
        assert e_after.code_anchors[0].repo == e_before.code_anchors[0].repo
        assert e_after.code_anchors[0].path == e_before.code_anchors[0].path
        assert e_after.code_anchors[0].line == e_before.code_anchors[0].line
        assert e_after.code_anchors[0].symbol == e_before.code_anchors[0].symbol
        assert len(e_after.citation_anchors) == len(e_before.citation_anchors)
        assert e_after.citation_anchors[0].key == e_before.citation_anchors[0].key
        assert e_after.citation_anchors[0].pages == e_before.citation_anchors[0].pages
        assert e_after.abstract == e_before.abstract
        assert e_after.abstract_rationale == e_before.abstract_rationale
        assert e_after.confidence == e_before.confidence

    def test_relation_discipline_v1_fields_survive_json_round_trip(self):
        """THE regression test. Every discipline-v1 edge field must round-trip."""
        kg_before = _populated_graph().to_kg()
        kg_after = KnowledgeGraph.from_json(kg_before.to_json())

        # Index by (source, target, relation_type) for field-level comparison.
        def _key(r):
            return (r.source, r.target, r.relation_type)

        before = {_key(r): r for r in kg_before.relations}
        after = {_key(r): r for r in kg_after.relations}
        assert set(before) == set(after), "relation set changed on round-trip"

        for k, r_before in before.items():
            r_after = after[k]
            assert r_after.sign == r_before.sign, f"sign dropped for {k}: {r_before.sign} -> {r_after.sign}"
            assert r_after.lag == r_before.lag, f"lag dropped for {k}: {r_before.lag} -> {r_after.lag}"
            assert r_after.form == r_before.form, f"form dropped for {k}: {r_before.form} -> {r_after.form}"
            assert r_after.edge_class == r_before.edge_class, f"edge_class dropped for {k}"
            assert r_after.conditional_on == r_before.conditional_on, f"conditional_on dropped for {k}"
            assert r_after.confidence == r_before.confidence, f"confidence dropped for {k}"
            assert r_after.weight == r_before.weight
            assert r_after.metadata == r_before.metadata
            assert len(r_after.provenance) == len(r_before.provenance)
            if r_before.provenance:
                assert r_after.provenance[0].document == r_before.provenance[0].document
                assert r_after.provenance[0].passage == r_before.provenance[0].passage

    def test_graph_round_trip_preserves_relation_fields(self):
        """OntologyGraph.from_kg(to_kg(g)).to_kg() must preserve fields too.

        This is the specific path that broke in the 2026-04-20 regression:
        `add_relation` was stashing only (relation_type, weight) on the
        NetworkX edge, so `to_kg()` reconstructed every relation with
        schema defaults. The fix was to stash the full Relation under
        `_relation`. This test locks that in.
        """
        g1 = _populated_graph()
        kg1 = g1.to_kg()
        g2 = OntologyGraph.from_kg(kg1)
        kg2 = g2.to_kg()

        def _key(r):
            return (r.source, r.target, r.relation_type)

        before = {_key(r): r for r in kg1.relations}
        after = {_key(r): r for r in kg2.relations}
        assert set(before) == set(after)
        for k in before:
            rb, ra = before[k], after[k]
            assert ra.sign == rb.sign, f"graph round-trip dropped sign on {k}"
            assert ra.lag == rb.lag, f"graph round-trip dropped lag on {k}"
            assert ra.form == rb.form
            assert ra.edge_class == rb.edge_class
            assert ra.conditional_on == rb.conditional_on
            assert ra.confidence == rb.confidence

    def test_full_chain_graph_to_json_to_graph_to_kg(self):
        """The full chain: graph → kg → json → kg → graph → kg."""
        g1 = _populated_graph()
        kg1 = g1.to_kg()
        js = kg1.to_json()
        kg2 = KnowledgeGraph.from_json(js)
        g2 = OntologyGraph.from_kg(kg2)
        kg3 = g2.to_kg()

        def _key(r):
            return (r.source, r.target, r.relation_type)

        signs_before = {_key(r): r.sign for r in kg1.relations}
        signs_after  = {_key(r): r.sign for r in kg3.relations}
        assert signs_before == signs_after

        lags_before = {_key(r): r.lag for r in kg1.relations}
        lags_after  = {_key(r): r.lag for r in kg3.relations}
        assert lags_before == lags_after

        classes_before = {_key(r): r.edge_class for r in kg1.relations}
        classes_after  = {_key(r): r.edge_class for r in kg3.relations}
        assert classes_before == classes_after

    def test_sign_distribution_survives(self):
        """Catches the exact failure mode: would-be 'signedness stuck at 0'.

        If any serialization layer silently defaulted sign → 'unknown',
        the sign distribution would collapse and this would fail.
        """
        g = _populated_graph()
        kg1 = g.to_kg()
        kg2 = KnowledgeGraph.from_json(kg1.to_json())

        def _sign_counts(kg):
            c = {}
            for r in kg.relations:
                c[r.sign] = c.get(r.sign, 0) + 1
            return c

        assert _sign_counts(kg1) == _sign_counts(kg2)
        # Sanity: the fixture actually covers the sign alphabet.
        assert {"+", "-", "±", "0"} <= set(_sign_counts(kg1))

    def test_json_payload_actually_contains_the_fields(self):
        """Guard against a serializer that silently drops fields at the dict
        layer (e.g. a custom to_json that filters keys)."""
        kg = _populated_graph().to_kg()
        payload = json.loads(kg.to_json())
        rel = payload["relations"][0]
        for field_name in ("sign", "lag", "form", "edge_class", "conditional_on", "confidence"):
            assert field_name in rel, f"{field_name} missing from JSON payload"
        ent = next(iter(payload["entities"].values()))
        for field_name in (
            "aliases", "observations", "provenance", "metadata",
            "code_anchors", "citation_anchors", "abstract",
            "abstract_rationale", "confidence",
        ):
            assert field_name in ent, f"{field_name} missing from JSON payload"
