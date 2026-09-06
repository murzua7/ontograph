"""Regression test (T1 repair): merge must preserve relation polarity fields.

`resolve_relations` previously rebuilt each Relation with only
source/target/type/weight/provenance/metadata, silently resetting
sign/form/lag/edge_class/confidence/conditional_on. All 1606 merged
relations ended `sign: unknown` while per-doc inputs were 94% signed.
"""
from __future__ import annotations

from ontograph.models import Entity, KnowledgeGraph, Provenance, Relation
from ontograph.resolver import merge_knowledge_graphs


def _kg(doc, rels):
    kg = KnowledgeGraph(schema_name="economics", documents=[doc])
    names = set()
    for r in rels:
        for n in (r.source, r.target):
            if n not in names:
                kg.add_entity(Entity(name=n, entity_type="state_variable"))
                names.add(n)
    for r in rels:
        kg.add_relation(r)
    return kg


def _rel(src, tgt, rtype="causes", sign="+", form="monotone", lag="within_step",
         edge_class="mechanism", confidence=0.9, conditional_on=(), doc="d.pdf"):
    return Relation(
        source=src, target=tgt, relation_type=rtype, sign=sign, form=form,
        lag=lag, edge_class=edge_class, confidence=confidence,
        conditional_on=list(conditional_on),
        provenance=[Provenance(document=doc, section="p.1", passage="q")],
    )


class TestMergePreservesPolarity:
    def test_signed_fields_survive_merge(self):
        a = _kg("a.pdf", [_rel("X", "Y", sign="+", form="linear",
                                lag="across_step", edge_class="structural",
                                confidence=0.7, conditional_on=["Z"])])
        m = merge_knowledge_graphs([a])
        assert len(m.relations) == 1
        r = m.relations[0]
        assert (r.sign, r.form, r.lag, r.edge_class) == ("+", "linear", "across_step", "structural")
        assert r.confidence == 0.7
        assert r.conditional_on == ["Z"]

    def test_opposing_signs_both_survive(self):
        a = _kg("a.pdf", [_rel("X", "Y", sign="+")])
        b = _kg("b.pdf", [_rel("X", "Y", sign="-")])
        m = merge_knowledge_graphs([a, b])
        assert sorted(r.sign for r in m.relations) == ["+", "-"]

    def test_exact_duplicates_still_dedupe(self):
        a = _kg("a.pdf", [_rel("X", "Y", sign="+", doc="a.pdf")])
        b = _kg("b.pdf", [_rel("X", "Y", sign="+", doc="b.pdf")])
        m = merge_knowledge_graphs([a, b])
        assert len(m.relations) == 1
        assert m.relations[0].sign == "+"
        assert len(m.relations[0].provenance) == 2
