"""Tests for the causal graph layer: models, scoring, extraction, cascade engine."""

import json
import pytest


# ── Models ──

class TestCausalModels:
    def test_causal_condition_defaults(self):
        from ontograph.causal_models import CausalCondition
        c = CausalCondition(variable="leverage", operator=">", threshold="80%")
        assert c.description == ""
        assert c.currently_met is None

    def test_causal_mechanism_defaults(self):
        from ontograph.causal_models import CausalMechanism
        m = CausalMechanism(name="credit_channel", description="Through bank lending")
        assert m.direction == "ambiguous"
        assert m.nonlinear is False
        assert m.conditions == []
        assert m.elasticity_range is None

    def test_causal_mechanism_with_conditions(self):
        from ontograph.causal_models import CausalMechanism, CausalCondition
        cond = CausalCondition(
            variable="capacity_utilization", operator=">", threshold="85%",
            description="Near full capacity",
        )
        m = CausalMechanism(
            name="cost_push", description="Energy cost passthrough",
            direction="positive", conditions=[cond],
        )
        assert len(m.conditions) == 1
        assert m.conditions[0].variable == "capacity_utilization"

    def test_causal_claim_defaults(self):
        from ontograph.causal_models import CausalClaim
        c = CausalClaim(id="test_001", source="Oil Price", target="Inflation")
        assert c.causal_type == "macro_statistical"
        assert c.evidence_type == "narrative"
        assert c.confidence == 0.0
        assert c.strength == "correlation"
        assert c.net_direction == "ambiguous"
        assert c.extracted_by == "llm"
        assert c.contradicted_by == []

    def test_claim_id_deterministic(self):
        from ontograph.causal_models import make_claim_id
        id1 = make_claim_id("Oil", "Inflation", "narrative", "oil causes inflation")
        id2 = make_claim_id("Oil", "Inflation", "narrative", "oil causes inflation")
        assert id1 == id2

    def test_claim_id_unique_for_different_inputs(self):
        from ontograph.causal_models import make_claim_id
        id1 = make_claim_id("Oil", "Inflation", "narrative")
        id2 = make_claim_id("Oil", "GDP", "narrative")
        assert id1 != id2

    def test_causal_graph_add_claim(self):
        from ontograph.causal_models import CausalGraph, CausalClaim
        cg = CausalGraph()
        claim = CausalClaim(id="c1", source="A", target="B", net_direction="positive")
        cg.add_claim(claim)
        assert "c1" in cg.claims
        assert "A" in cg.entities
        assert "B" in cg.entities

    def test_causal_graph_outgoing_claims(self):
        from ontograph.causal_models import CausalGraph, CausalClaim
        cg = CausalGraph()
        cg.add_claim(CausalClaim(id="c1", source="A", target="B"))
        cg.add_claim(CausalClaim(id="c2", source="A", target="C"))
        cg.add_claim(CausalClaim(id="c3", source="B", target="C"))
        out = cg.outgoing_claims("A")
        assert len(out) == 2
        assert all(c.source == "A" for c in out)
        assert len(cg.outgoing_claims("B")) == 1
        assert len(cg.outgoing_claims("C")) == 0

    def test_shock_defaults(self):
        from ontograph.causal_models import Shock
        s = Shock(entity="Hormuz", shock_type="supply_disruption",
                  description="Full closure")
        assert s.direction == "negative"
        assert s.magnitude_qualitative == "moderate"
        assert s.regime_context == {}

    def test_cascade_effect_defaults(self):
        from ontograph.causal_models import CascadeEffect
        e = CascadeEffect(entity="GDP")
        assert e.confidence == 0.0
        assert e.time_horizon_min == "immediate"
        assert e.n_incoming_channels == 1


class TestCausalSerialization:
    def _make_graph(self):
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism, CausalCondition,
        )
        from ontograph.models import Provenance

        cg = CausalGraph(
            schema_name="causal",
            build_date="2026-04-03",
            regime_context={"leverage": "high"},
        )
        cond = CausalCondition(
            variable="capacity_utilization", operator=">", threshold="85%",
            description="Near full capacity",
        )
        mech = CausalMechanism(
            name="energy_cost_passthrough",
            description="Higher energy costs passed to consumer prices",
            direction="positive",
            nonlinear=True,
            elasticity_range=(0.3, 0.7),
            elasticity_horizon="short_run",
            time_lag_min="1 month",
            time_lag_max="6 months",
            conditions=[cond],
        )
        claim = CausalClaim(
            id="oil_inflation_001",
            source="Oil Price",
            target="Inflation",
            mechanisms=[mech],
            causal_type="macro_statistical",
            evidence_type="meta_analysis",
            evidence_count=5,
            evidence_directness="cited",
            claim_assertiveness="strong",
            confidence=0.72,
            strength="strong_causal",
            net_direction="positive",
            geographic_scope="OECD",
            geographic_entities=["FR", "DE"],
            sources=[Provenance(document="survey.pdf", section="Ch.3")],
            claim_text="Oil price shocks raise consumer prices",
        )
        cg.add_claim(claim)
        return cg

    def test_json_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        assert len(cg2.claims) == 1
        claim = cg2.claims["oil_inflation_001"]
        assert claim.source == "Oil Price"
        assert claim.target == "Inflation"
        assert claim.confidence == 0.72
        assert claim.evidence_type == "meta_analysis"

    def test_mechanism_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        claim = cg2.claims["oil_inflation_001"]
        assert len(claim.mechanisms) == 1
        mech = claim.mechanisms[0]
        assert mech.name == "energy_cost_passthrough"
        assert mech.direction == "positive"
        assert mech.nonlinear is True
        assert mech.elasticity_range == (0.3, 0.7)
        assert mech.time_lag_min == "1 month"

    def test_condition_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        mech = cg2.claims["oil_inflation_001"].mechanisms[0]
        assert len(mech.conditions) == 1
        cond = mech.conditions[0]
        assert cond.variable == "capacity_utilization"
        assert cond.operator == ">"
        assert cond.threshold == "85%"

    def test_entities_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        assert "Oil Price" in cg2.entities
        assert "Inflation" in cg2.entities

    def test_provenance_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        sources = cg2.claims["oil_inflation_001"].sources
        assert len(sources) == 1
        assert sources[0].document == "survey.pdf"

    def test_regime_context_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = self._make_graph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        assert cg2.regime_context == {"leverage": "high"}

    def test_empty_graph_roundtrip(self):
        from ontograph.causal_models import CausalGraph
        cg = CausalGraph()
        j = cg.to_json()
        cg2 = CausalGraph.from_json(j)
        assert len(cg2.claims) == 0
        assert len(cg2.entities) == 0


class TestEvidenceTaxonomy:
    def test_all_18_types_present(self):
        from ontograph.causal_models import EVIDENCE_TAXONOMY
        assert len(EVIDENCE_TAXONOMY) == 18

    def test_scores_in_range(self):
        from ontograph.causal_models import EVIDENCE_TAXONOMY
        for etype, score in EVIDENCE_TAXONOMY.items():
            assert 0.0 <= score <= 1.0, f"{etype} score {score} out of range"

    def test_accounting_identity_highest(self):
        from ontograph.causal_models import EVIDENCE_TAXONOMY
        assert EVIDENCE_TAXONOMY["accounting_identity"] == 1.0

    def test_narrative_lowest(self):
        from ontograph.causal_models import EVIDENCE_TAXONOMY
        assert EVIDENCE_TAXONOMY["narrative"] == min(EVIDENCE_TAXONOMY.values())

    def test_ordering_rct_above_narrative(self):
        from ontograph.causal_models import EVIDENCE_TAXONOMY
        assert EVIDENCE_TAXONOMY["rct"] > EVIDENCE_TAXONOMY["narrative"]
        assert EVIDENCE_TAXONOMY["natural_experiment"] > EVIDENCE_TAXONOMY["granger_causality"]
        assert EVIDENCE_TAXONOMY["meta_analysis"] >= EVIDENCE_TAXONOMY["rct"]

    def test_assertiveness_scores(self):
        from ontograph.causal_models import ASSERTIVENESS_SCORES
        assert ASSERTIVENESS_SCORES["definitional"] == 1.0
        assert ASSERTIVENESS_SCORES["hedged"] == 0.2
        assert len(ASSERTIVENESS_SCORES) == 4

    def test_mechanism_scores(self):
        from ontograph.causal_models import MECHANISM_SCORES
        assert MECHANISM_SCORES["quantified"] == 1.0
        assert MECHANISM_SCORES["none"] == 0.0
        assert len(MECHANISM_SCORES) == 5


# ── Scoring ──

class TestConfidenceScoring:
    def _make_claim(self, **overrides):
        from ontograph.causal_models import CausalClaim, CausalMechanism
        defaults = dict(
            id="test_001",
            source="Oil Price",
            target="Inflation",
            mechanisms=[CausalMechanism(
                name="energy_cost_passthrough",
                description="Higher energy costs passed to consumer prices via CPI basket",
                direction="positive",
            )],
            evidence_type="meta_analysis",
            evidence_count=3,
            claim_assertiveness="strong",
            geographic_scope="OECD",
        )
        defaults.update(overrides)
        return CausalClaim(**defaults)

    def test_evidence_base_score_known_type(self):
        from ontograph.causal_scoring import evidence_base_score
        assert evidence_base_score("meta_analysis") == 0.90
        assert evidence_base_score("narrative") == 0.15
        assert evidence_base_score("accounting_identity") == 1.00

    def test_evidence_base_score_unknown_type(self):
        from ontograph.causal_scoring import evidence_base_score
        assert evidence_base_score("unknown_type") == 0.15  # default

    def test_mechanism_score_quantified(self):
        from ontograph.causal_scoring import mechanism_score
        from ontograph.causal_models import CausalClaim, CausalMechanism
        claim = CausalClaim(
            id="q1", source="A", target="B",
            mechanisms=[CausalMechanism(
                name="test", description="detailed channel",
                elasticity_range=(0.3, 0.7),
            )],
        )
        assert mechanism_score(claim) == 1.0

    def test_mechanism_score_articulated(self):
        from ontograph.causal_scoring import mechanism_score
        claim = self._make_claim()
        # description > 30 chars, no elasticity
        assert mechanism_score(claim) == 0.8

    def test_mechanism_score_none(self):
        from ontograph.causal_scoring import mechanism_score
        from ontograph.causal_models import CausalClaim
        claim = CausalClaim(id="n1", source="A", target="B", mechanisms=[])
        assert mechanism_score(claim) == 0.0

    def test_evidence_diversity_score(self):
        from ontograph.causal_scoring import evidence_diversity_score
        claim1 = self._make_claim(evidence_count=1)
        claim5 = self._make_claim(evidence_count=5)
        s1 = evidence_diversity_score(claim1)
        s5 = evidence_diversity_score(claim5)
        assert 0 < s1 < s5 <= 1.0

    def test_assertiveness_score(self):
        from ontograph.causal_scoring import assertiveness_score
        strong = self._make_claim(claim_assertiveness="strong")
        hedged = self._make_claim(claim_assertiveness="hedged")
        assert assertiveness_score(strong) == 0.80
        assert assertiveness_score(hedged) == 0.20

    def test_compute_confidence_range(self):
        from ontograph.causal_scoring import compute_confidence
        claim = self._make_claim()
        score = compute_confidence(claim)
        assert 0.0 <= score <= 1.0
        assert claim.confidence == score
        assert claim.strength in ("correlation", "weak_causal", "strong_causal")

    def test_compute_confidence_high_evidence(self):
        from ontograph.causal_scoring import compute_confidence
        claim = self._make_claim(
            evidence_type="accounting_identity",
            evidence_count=5,
            claim_assertiveness="definitional",
        )
        score = compute_confidence(claim)
        # Accounting identity + definitional + multiple sources → high
        assert score > 0.6

    def test_compute_confidence_weak_claim(self):
        from ontograph.causal_scoring import compute_confidence
        from ontograph.causal_models import CausalClaim
        claim = CausalClaim(
            id="w1", source="A", target="B",
            evidence_type="narrative",
            evidence_count=1,
            claim_assertiveness="hedged",
        )
        score = compute_confidence(claim)
        assert score < 0.35

    def test_external_validity_with_context(self):
        from ontograph.causal_scoring import external_validity_score
        claim = self._make_claim(
            geographic_scope="OECD",
            geographic_entities=["FR", "DE"],
        )
        ctx = {"country": "FR", "decade": "2020s"}
        score = external_validity_score(claim, ctx)
        assert score == 1.0

    def test_external_validity_no_context(self):
        from ontograph.causal_scoring import external_validity_score
        claim = self._make_claim()
        assert external_validity_score(claim, None) == 0.50

    def test_classify_strength(self):
        from ontograph.causal_scoring import classify_strength
        assert classify_strength(0.10) == "correlation"
        assert classify_strength(0.34) == "correlation"
        assert classify_strength(0.35) == "weak_causal"
        assert classify_strength(0.50) == "weak_causal"
        assert classify_strength(0.65) == "strong_causal"
        assert classify_strength(0.90) == "strong_causal"

    def test_temporal_gate_reversed(self):
        from ontograph.causal_scoring import compute_confidence, apply_temporal_gate
        claim = self._make_claim()
        compute_confidence(claim)
        original = claim.confidence
        apply_temporal_gate(claim, temporal_precedence="reversed")
        assert claim.strength == "correlation"
        assert claim.confidence <= 0.33

    def test_temporal_gate_unknown_macro(self):
        from ontograph.causal_scoring import compute_confidence, apply_temporal_gate
        claim = self._make_claim(causal_type="macro_statistical")
        compute_confidence(claim)
        original = claim.confidence
        apply_temporal_gate(claim, temporal_precedence="unknown")
        assert claim.confidence == pytest.approx(original * 0.7, abs=0.01)

    def test_temporal_gate_established(self):
        from ontograph.causal_scoring import compute_confidence, apply_temporal_gate
        claim = self._make_claim()
        compute_confidence(claim)
        original = claim.confidence
        apply_temporal_gate(claim, temporal_precedence="established")
        assert claim.confidence == original  # no change


# ── Time utilities ──

class TestTimeUtils:
    def test_parse_immediate(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("immediate") == 0.0
        assert parse_time_duration(None) == 0.0
        assert parse_time_duration("") == 0.0

    def test_parse_days(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("1 day") == 1.0
        assert parse_time_duration("5 days") == 5.0

    def test_parse_months(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("3 months") == 90.0
        assert parse_time_duration("1 month") == 30.0

    def test_parse_quarters(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("2 quarters") == 180.0

    def test_parse_years(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("1 year") == 365.0

    def test_parse_unknown_string(self):
        from ontograph.causal_engine import parse_time_duration
        assert parse_time_duration("unknown") == 0.0
        assert parse_time_duration("not a duration") == 0.0

    def test_min_time(self):
        from ontograph.causal_engine import min_time
        assert min_time("1 month", "3 months") == "1 month"
        assert min_time("immediate", "1 year") == "immediate"

    def test_max_time(self):
        from ontograph.causal_engine import max_time
        assert max_time("1 month", "3 months") == "3 months"
        assert max_time("immediate", "1 year") == "1 year"

    def test_add_time(self):
        from ontograph.causal_engine import add_time
        result = add_time("immediate", "3 months")
        assert "3 months" == result
        result2 = add_time("1 month", "2 months")
        assert "3 months" == result2

    def test_add_time_immediate(self):
        from ontograph.causal_engine import add_time
        assert add_time("immediate", "immediate") == "immediate"


# ── Cascade engine ──

class TestCascadeEngine:
    def _make_linear_graph(self):
        """A → B → C, all positive, high confidence."""
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism,
        )
        cg = CausalGraph()
        cg.add_claim(CausalClaim(
            id="ab", source="A", target="B",
            confidence=0.90,
            net_direction="positive",
            mechanisms=[CausalMechanism(
                name="direct", description="A directly causes B",
                direction="positive", time_lag_min="1 month", time_lag_max="3 months",
            )],
        ))
        cg.add_claim(CausalClaim(
            id="bc", source="B", target="C",
            confidence=0.80,
            net_direction="positive",
            mechanisms=[CausalMechanism(
                name="indirect", description="B flows to C via market",
                direction="positive", time_lag_min="2 months", time_lag_max="6 months",
            )],
        ))
        return cg

    def _make_conflict_graph(self):
        """A → B via two paths with conflicting directions."""
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism,
        )
        cg = CausalGraph()
        # Direct path: A → B positive
        cg.add_claim(CausalClaim(
            id="ab_pos", source="A", target="B",
            confidence=0.80,
            mechanisms=[CausalMechanism(
                name="cost_push", description="Higher costs pass through",
                direction="positive",
            )],
        ))
        # Indirect path: A → C → B negative
        cg.add_claim(CausalClaim(
            id="ac", source="A", target="C",
            confidence=0.70,
            mechanisms=[CausalMechanism(
                name="demand_destruction", description="Kills demand",
                direction="negative",
            )],
        ))
        cg.add_claim(CausalClaim(
            id="cb", source="C", target="B",
            confidence=0.70,
            mechanisms=[CausalMechanism(
                name="spillover", description="Demand drop spills over",
                direction="positive",  # positive: same direction as incoming
            )],
        ))
        return cg

    def test_basic_propagation(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test shock",
                      direction="positive")
        effects = run_cascade(cg, shock)
        assert "A" in effects
        assert "B" in effects
        assert "C" in effects
        # B confidence = 1.0 * 0.90 = 0.90
        assert effects["B"].confidence == pytest.approx(0.90, abs=0.01)
        # C confidence = 0.90 * 0.80 = 0.72
        assert effects["C"].confidence == pytest.approx(0.72, abs=0.01)

    def test_direction_propagation_positive(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        effects = run_cascade(cg, shock)
        # positive × positive = positive
        assert effects["B"].direction == "positive"
        assert effects["C"].direction == "positive"

    def test_direction_propagation_negative(self):
        from ontograph.causal_engine import propagate_direction
        assert propagate_direction("positive", "negative") == "negative"
        assert propagate_direction("negative", "negative") == "positive"
        assert propagate_direction("positive", "ambiguous") == "ambiguous"

    def test_min_confidence_filter(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        # High threshold should filter out C (0.72 < 0.80)
        effects = run_cascade(cg, shock, min_confidence=0.80)
        assert "B" in effects
        assert "C" not in effects

    def test_sign_conflict_detection(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import Shock
        cg = self._make_conflict_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        effects = run_cascade(cg, shock, min_confidence=0.10)
        # B receives positive from direct path and negative from A→C→B
        assert effects["B"].direction == "conflicted"
        assert effects["B"].n_incoming_channels >= 2

    def test_condition_checking_blocks_edge(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism, CausalCondition, Shock,
        )
        cg = CausalGraph()
        cg.add_claim(CausalClaim(
            id="ab_cond", source="A", target="B",
            confidence=0.90,
            mechanisms=[CausalMechanism(
                name="conditional", description="Only when high leverage",
                direction="positive",
                conditions=[CausalCondition(
                    variable="leverage", operator="regime_is", threshold="high",
                )],
            )],
        ))
        # Shock without matching regime → edge blocked
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive",
                      regime_context={"leverage": "low"})
        effects = run_cascade(cg, shock)
        assert "B" not in effects  # condition not met

        # Shock with matching regime → edge passes
        shock2 = Shock(entity="A", shock_type="test", description="Test",
                       direction="positive",
                       regime_context={"leverage": "high"})
        effects2 = run_cascade(cg, shock2)
        assert "B" in effects2

    def test_condition_permissive_missing_variable(self):
        from ontograph.causal_engine import _evaluate_condition
        from ontograph.causal_models import CausalCondition
        cond = CausalCondition(variable="missing_var", operator=">", threshold="50")
        assert _evaluate_condition(cond, {}) is True  # permissive

    def test_max_depth_limit(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism, Shock,
        )
        # Build a long chain: A → B → C → D → E → F
        cg = CausalGraph()
        nodes = list("ABCDEF")
        for i in range(len(nodes) - 1):
            cg.add_claim(CausalClaim(
                id=f"{nodes[i]}{nodes[i+1]}",
                source=nodes[i], target=nodes[i+1],
                confidence=0.90,
                mechanisms=[CausalMechanism(
                    name="chain", description="Sequential",
                    direction="positive",
                )],
            ))
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        effects = run_cascade(cg, shock, max_depth=3)
        # Should reach A, B, C, D (3 hops) but not E, F
        assert "D" in effects
        assert "E" not in effects

    def test_no_revisit_in_same_path(self):
        """Cycle A → B → A should not create infinite loop."""
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import (
            CausalGraph, CausalClaim, CausalMechanism, Shock,
        )
        cg = CausalGraph()
        cg.add_claim(CausalClaim(
            id="ab", source="A", target="B", confidence=0.90,
            mechanisms=[CausalMechanism(name="x", description="x", direction="positive")],
        ))
        cg.add_claim(CausalClaim(
            id="ba", source="B", target="A", confidence=0.90,
            mechanisms=[CausalMechanism(name="y", description="y", direction="positive")],
        ))
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        effects = run_cascade(cg, shock)
        # Should terminate without infinite loop
        assert "A" in effects
        assert "B" in effects

    def test_time_horizon_propagation(self):
        from ontograph.causal_engine import run_cascade
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        effects = run_cascade(cg, shock)
        # B: 1-3 months from A
        assert effects["B"].time_horizon_min != "immediate"
        # C: cumulative lag from A→B→C
        assert effects["C"].time_horizon_min != "immediate"

    def test_feedback_mode_basic(self):
        from ontograph.causal_engine import run_cascade_with_feedback
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        iterations = run_cascade_with_feedback(cg, shock, n_iterations=2)
        assert len(iterations) >= 1
        # Iteration 0 should have effects
        assert len(iterations[0]) > 0

    def test_feedback_decay(self):
        from ontograph.causal_engine import run_cascade_with_feedback
        from ontograph.causal_models import Shock
        cg = self._make_linear_graph()
        shock = Shock(entity="A", shock_type="test", description="Test",
                      direction="positive")
        iterations = run_cascade_with_feedback(
            cg, shock, n_iterations=3, feedback_decay=0.7,
        )
        if len(iterations) >= 2:
            # Later iterations should have lower max confidence
            max_conf_0 = max(e.confidence for e in iterations[0].values())
            max_conf_1 = max(
                (e.confidence for e in iterations[1].values()),
                default=0,
            )
            assert max_conf_1 <= max_conf_0

    def test_numeric_condition_operators(self):
        from ontograph.causal_engine import _evaluate_condition
        from ontograph.causal_models import CausalCondition
        ctx = {"rate": "5.0", "util": "90%"}
        assert _evaluate_condition(
            CausalCondition(variable="rate", operator=">", threshold="3.0"), ctx
        ) is True
        assert _evaluate_condition(
            CausalCondition(variable="rate", operator="<", threshold="3.0"), ctx
        ) is False
        assert _evaluate_condition(
            CausalCondition(variable="util", operator=">=", threshold="85%"), ctx
        ) is True

    def test_in_range_condition(self):
        from ontograph.causal_engine import _evaluate_condition
        from ontograph.causal_models import CausalCondition
        ctx = {"rate": "5.0"}
        assert _evaluate_condition(
            CausalCondition(variable="rate", operator="in_range", threshold="3-7"), ctx
        ) is True
        assert _evaluate_condition(
            CausalCondition(variable="rate", operator="in_range", threshold="6-10"), ctx
        ) is False


# ── Extraction ──

class TestCausalExtraction:
    def test_build_causal_claim_basic(self):
        from ontograph.causal_extractor import _build_causal_claim
        raw = {
            "source": "Oil Price",
            "target": "Inflation",
            "mechanism_name": "energy_cost_passthrough",
            "mechanism_description": "Higher energy costs passed to consumer prices",
            "direction": "positive",
            "evidence_type": "meta_analysis",
            "evidence_directness": "cited",
            "assertiveness": "strong",
            "claim_text": "Oil price shocks raise consumer prices",
            "conditions": [],
            "time_lag_min": "1 month",
            "time_lag_max": "6 months",
            "_document": "survey.pdf",
            "_section": "Ch.3",
        }
        claim = _build_causal_claim(raw, "survey.pdf")
        assert claim.source == "Oil Price"
        assert claim.target == "Inflation"
        assert claim.evidence_type == "meta_analysis"
        assert claim.claim_assertiveness == "strong"
        assert claim.net_direction == "positive"
        assert len(claim.mechanisms) == 1
        assert claim.mechanisms[0].name == "energy_cost_passthrough"
        assert claim.mechanisms[0].time_lag_min == "1 month"
        assert len(claim.sources) == 1

    def test_build_causal_claim_defaults(self):
        from ontograph.causal_extractor import _build_causal_claim
        raw = {"source": "A", "target": "B"}
        claim = _build_causal_claim(raw, "doc.md")
        assert claim.evidence_type == "narrative"
        assert claim.claim_assertiveness == "moderate"
        assert claim.net_direction == "ambiguous"
        assert claim.extracted_by == "llm"

    def test_build_claim_invalid_evidence_type(self):
        from ontograph.causal_extractor import _build_causal_claim
        raw = {"source": "A", "target": "B", "evidence_type": "INVALID_TYPE"}
        claim = _build_causal_claim(raw, "doc.md")
        assert claim.evidence_type == "narrative"  # falls back

    def test_build_claim_with_conditions(self):
        from ontograph.causal_extractor import _build_causal_claim
        raw = {
            "source": "A", "target": "B",
            "conditions": [
                {"variable": "leverage", "operator": "regime_is",
                 "threshold": "high", "description": "High leverage regime"},
            ],
        }
        claim = _build_causal_claim(raw, "doc.md")
        assert len(claim.mechanisms[0].conditions) == 1
        assert claim.mechanisms[0].conditions[0].variable == "leverage"

    def test_build_claim_id_deterministic(self):
        from ontograph.causal_extractor import _build_causal_claim
        raw = {"source": "Oil", "target": "CPI", "evidence_type": "narrative",
               "claim_text": "oil raises CPI"}
        c1 = _build_causal_claim(raw, "doc.md")
        c2 = _build_causal_claim(raw, "doc.md")
        assert c1.id == c2.id

    def test_causal_extract_from_document_mock(self):
        """Test full pipeline with mock LLM client."""
        from ontograph.causal_extractor import causal_extract_from_document
        from ontograph.llm_client import LLMClient
        from ontograph.parsers import ParsedDocument, Section

        # Mock client that returns a fixed extraction
        client = LLMClient(mode="mock")

        doc = ParsedDocument(
            title="Test Document",
            sections=[Section(
                heading="Introduction",
                text="Higher interest rates reduce investment through "
                     "the cost of capital channel. This relationship has "
                     "been documented in multiple meta-analyses.",
                level=1,
            )],
            full_text="...",
            source_path="test_doc.md",
        )

        # Mock client will return empty results, which is fine for testing
        # the pipeline doesn't crash
        cg = causal_extract_from_document(doc, client=client)
        assert cg is not None
        assert cg.source_knowledge_graph == "test_doc.md"

    def test_claim_confidence_is_computed(self):
        from ontograph.causal_extractor import _build_causal_claim
        from ontograph.causal_scoring import compute_confidence
        raw = {
            "source": "Policy Rate", "target": "Investment",
            "mechanism_name": "cost_of_capital",
            "mechanism_description": "Higher rates increase cost of capital, reducing NPV of projects",
            "direction": "negative",
            "evidence_type": "meta_analysis",
            "assertiveness": "strong",
        }
        claim = _build_causal_claim(raw, "doc.md")
        score = compute_confidence(claim)
        assert 0 < score <= 1.0
        assert claim.confidence == score
        assert claim.strength in ("correlation", "weak_causal", "strong_causal")
