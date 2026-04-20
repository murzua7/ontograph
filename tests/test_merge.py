"""Phase D — merge + agreement-set tests (TDD).

Covers `ontograph.merge.merge_extractions(llm_entities, llm_relations,
ast_entities, ast_relations, *, name_matcher, threshold) -> (KG, MergeReport)`.

Spec §4.1.3 contract:
- Entities match when name_matcher(a, b) >= threshold AND entity_type matches
  (with the abstract-LLM ↔ concrete-AST special case producing an `implements`
  agreement relation instead of an entity merge).
- MergeReport has three entity buckets (agreement / llm_only / ast_only),
  three relation buckets (same), plus `sign_conflicts` for paired mechanism
  edges whose signs disagree on `+` vs `-`.
- Only the agreement bucket enters the returned KnowledgeGraph.
"""

from __future__ import annotations


def _ent(name, etype, *, abstract=False, rationale="", grounded=False):
    from ontograph.models import CodeAnchor, Entity
    anchors = [CodeAnchor(repo="r", path="p.py", line=1, symbol=name)] if grounded else []
    return Entity(
        name=name, entity_type=etype,
        code_anchors=anchors, abstract=abstract, abstract_rationale=rationale,
    )


def _rel(src, tgt, rtype="feeds_into", *, sign="unknown", lag="unknown",
         edge_class="mechanism"):
    from ontograph.models import Relation
    return Relation(
        source=src, target=tgt, relation_type=rtype,
        sign=sign, lag=lag, edge_class=edge_class,
    )


# ── 1. Entity agreement ────────────────────────────────────────────────

class TestEntityAgreement:
    def test_exact_name_and_type_match_is_agreement(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("alpha", "mechanism")]
        ast = [_ent("alpha", "mechanism", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        assert len(report.agreement_entities) == 1
        assert report.llm_only_entities == []
        assert report.ast_only_entities == []
        # merged entity keeps AST grounding
        merged = report.agreement_entities[0]
        assert merged.is_grounded()

    def test_llm_only_entity_goes_to_llm_bucket(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("alpha", "concept")]
        kg, report = merge_extractions(llm, [], [], [])
        assert [e.name for e in report.llm_only_entities] == ["alpha"]
        assert report.agreement_entities == []

    def test_ast_only_entity_goes_to_ast_bucket(self):
        from ontograph.merge import merge_extractions
        ast = [_ent("beta", "phase", grounded=True)]
        kg, report = merge_extractions([], [], ast, [])
        assert [e.name for e in report.ast_only_entities] == ["beta"]
        assert report.agreement_entities == []

    def test_type_mismatch_prevents_agreement_for_concrete_pair(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("alpha", "mechanism")]
        ast = [_ent("alpha", "state_variable", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        assert report.agreement_entities == []
        assert len(report.llm_only_entities) == 1
        assert len(report.ast_only_entities) == 1


# ── 2. Fuzzy matching via default name_matcher ─────────────────────────

class TestFuzzyMatching:
    def test_normalised_name_match(self):
        """'alpha_phase' vs 'alpha phase' should match under default matcher."""
        from ontograph.merge import merge_extractions
        llm = [_ent("alpha phase", "phase")]
        ast = [_ent("alpha_phase", "phase", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        assert len(report.agreement_entities) == 1

    def test_threshold_cutoff(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("apple", "mechanism")]
        ast = [_ent("orange", "mechanism", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [], threshold=0.85)
        assert report.agreement_entities == []
        assert len(report.llm_only_entities) == 1
        assert len(report.ast_only_entities) == 1

    def test_custom_name_matcher_respected(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("X", "mechanism")]
        ast = [_ent("Y", "mechanism", grounded=True)]
        # Pathological matcher that calls everything a match.
        kg, report = merge_extractions(
            llm, [], ast, [],
            name_matcher=lambda a, b: 1.0, threshold=0.5,
        )
        assert len(report.agreement_entities) == 1


# ── 3. Abstract LLM ↔ concrete AST ─────────────────────────────────────

class TestAbstractToConcreteBinding:
    def test_abstract_llm_matches_concrete_ast_via_implements(self):
        """Spec §4.1.3: abstract=True LLM entity binds to concrete AST
        entity via an `implements` agreement relation — both entities stay
        in the KG, but the binding is recorded as agreement, not dedup."""
        from ontograph.merge import merge_extractions
        llm = [_ent("tobin q", "theory", abstract=True, rationale="pure theory")]
        ast = [_ent("tobin q", "mechanism", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        # Both entities survive in agreement bucket — they are *not* merged
        # into one node because their types differ.
        names = {e.name for e in report.agreement_entities}
        types = {e.entity_type for e in report.agreement_entities}
        assert "tobin q" in names
        assert types == {"theory", "mechanism"}
        # An implements relation ties the two together.
        impls = [r for r in report.agreement_relations
                 if r.relation_type == "implements"]
        assert len(impls) == 1


# ── 4. Relation agreement / disagreement ───────────────────────────────

class TestRelationAgreement:
    def test_matched_edge_on_matched_entities_is_agreement(self):
        from ontograph.merge import merge_extractions
        llm_e = [_ent("a", "mechanism"), _ent("b", "state_variable")]
        ast_e = [_ent("a", "mechanism", grounded=True),
                 _ent("b", "state_variable", grounded=True)]
        llm_r = [_rel("a", "b", sign="+", edge_class="mechanism")]
        ast_r = [_rel("a", "b", sign="+", edge_class="mechanism")]
        kg, report = merge_extractions(llm_e, llm_r, ast_e, ast_r)
        assert len(report.agreement_relations) == 1
        assert report.sign_conflicts == []

    def test_llm_only_relation_when_ast_silent(self):
        from ontograph.merge import merge_extractions
        llm_e = [_ent("a", "mechanism"), _ent("b", "state_variable")]
        ast_e = [_ent("a", "mechanism", grounded=True),
                 _ent("b", "state_variable", grounded=True)]
        llm_r = [_rel("a", "b", edge_class="mechanism")]
        kg, report = merge_extractions(llm_e, llm_r, ast_e, [])
        assert len(report.llm_only_relations) == 1
        assert report.ast_only_relations == []
        assert report.agreement_relations == []

    def test_relation_on_unmatched_entity_stays_in_source_bucket(self):
        """An LLM relation whose endpoints only LLM saw stays llm_only."""
        from ontograph.merge import merge_extractions
        llm_e = [_ent("x", "mechanism"), _ent("y", "state_variable")]
        ast_e = []
        llm_r = [_rel("x", "y", edge_class="mechanism")]
        kg, report = merge_extractions(llm_e, llm_r, ast_e, [])
        assert len(report.llm_only_relations) == 1

    def test_sign_conflict_recorded_when_plus_vs_minus(self):
        from ontograph.merge import merge_extractions
        llm_e = [_ent("a", "mechanism"), _ent("b", "state_variable")]
        ast_e = [_ent("a", "mechanism", grounded=True),
                 _ent("b", "state_variable", grounded=True)]
        llm_r = [_rel("a", "b", sign="+", edge_class="mechanism")]
        ast_r = [_rel("a", "b", sign="-", edge_class="mechanism")]
        kg, report = merge_extractions(llm_e, llm_r, ast_e, ast_r)
        assert len(report.sign_conflicts) == 1
        llm_rel, ast_rel = report.sign_conflicts[0]
        assert {llm_rel.sign, ast_rel.sign} == {"+", "-"}

    def test_unknown_vs_plus_is_not_a_sign_conflict(self):
        """Silence ≠ disagreement (spec §2 principle 2)."""
        from ontograph.merge import merge_extractions
        llm_e = [_ent("a", "mechanism"), _ent("b", "state_variable")]
        ast_e = [_ent("a", "mechanism", grounded=True),
                 _ent("b", "state_variable", grounded=True)]
        llm_r = [_rel("a", "b", sign="+", edge_class="mechanism")]
        ast_r = [_rel("a", "b", sign="unknown", edge_class="mechanism")]
        kg, report = merge_extractions(llm_e, llm_r, ast_e, ast_r)
        assert report.sign_conflicts == []


# ── 5. KnowledgeGraph output ───────────────────────────────────────────

class TestMergedKnowledgeGraph:
    def test_only_agreement_enters_kg(self):
        from ontograph.merge import merge_extractions
        llm = [_ent("alpha", "mechanism"), _ent("lonely", "concept")]
        ast = [_ent("alpha", "mechanism", grounded=True),
               _ent("standalone", "phase", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        assert "alpha" in kg.entities
        assert "lonely" not in kg.entities
        assert "standalone" not in kg.entities


# ── 6. Report serialisation ────────────────────────────────────────────

class TestReportSerialisation:
    def test_to_json_roundtrips(self):
        import json
        from ontograph.merge import merge_extractions
        llm = [_ent("a", "mechanism")]
        ast = [_ent("a", "mechanism", grounded=True)]
        kg, report = merge_extractions(llm, [], ast, [])
        payload = json.loads(report.to_json())
        assert "agreement_entities" in payload
        assert "llm_only_entities" in payload
        assert "ast_only_entities" in payload
        assert "agreement_relations" in payload
        assert "llm_only_relations" in payload
        assert "ast_only_relations" in payload
        assert "sign_conflicts" in payload
