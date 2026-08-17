"""Cross-model extraction agreement (#46).

Compares two independent LLM extractions of the same document — e.g. one
pass by Claude and another by GPT-4, or two Claude versions — and reports
where they agree. Distinct from `ontograph.merge`, which reconciles a
single LLM pass with an AST pass: there, the AST side is structural
ground-truth and is weighted accordingly. Here, both sides are
equally-trusted prose extractions; agreement is symmetric.

Public surface:
- `cross_model_agreement(kg_a, kg_b)` — structured report.
- `build_agreement_kg(report, kg_a, kg_b)` — materialise the intersection.

Relation key: `(source, target, relation_type)`, matching `ontograph.diff`
and `cmd_diff` so users don't have to reason about which surface uses
which identity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from ontograph.models import Entity, KnowledgeGraph, Relation


def _load(kg: Union[KnowledgeGraph, str, Path]) -> KnowledgeGraph:
    if isinstance(kg, KnowledgeGraph):
        return kg
    return KnowledgeGraph.from_json(Path(kg).read_text(encoding="utf-8"))


def _rel_key(r: Relation) -> tuple[str, str, str]:
    return (r.source, r.target, r.relation_type)


def _reconcile_sign(a: str, b: str) -> tuple[str, bool]:
    """Pick the consensus sign for an agreement relation.

    Returns (reconciled_sign, is_conflict). Rules:
    - Both equal → that sign, no conflict.
    - One 'unknown' → the other (unknown carries no signal), no conflict.
    - Both known but disagree → 'unknown', conflict.
    """
    if a == b:
        return a, False
    if a == "unknown":
        return b, False
    if b == "unknown":
        return a, False
    return "unknown", True


def cross_model_agreement(
    kg_a: Union[KnowledgeGraph, str, Path],
    kg_b: Union[KnowledgeGraph, str, Path],
) -> dict[str, Any]:
    """Classify entities and relations by cross-model agreement.

    Entities:
    - `agreement`: same name AND same entity_type.
    - `a_only` / `b_only`: name appears only on that side.
    - `type_conflicts`: same name, different entity_type — listed separately
      so the user doesn't silently lose one model's disagreement.

    Relations: keyed by `(source, target, relation_type)`. Sign, lag, and
    edge_class are reconciled per-field; conflicts are surfaced even when
    the structure agrees.
    """
    a = _load(kg_a)
    b = _load(kg_b)

    a_names = set(a.entities)
    b_names = set(b.entities)

    entity_agreement: list[str] = []
    type_conflicts: list[dict[str, Any]] = []
    for n in sorted(a_names & b_names):
        ta = a.entities[n].entity_type
        tb = b.entities[n].entity_type
        if ta == tb:
            entity_agreement.append(n)
        else:
            type_conflicts.append({"name": n, "a": ta, "b": tb})

    a_only_entities = sorted(a_names - b_names)
    b_only_entities = sorted(b_names - a_names)

    a_rels = {_rel_key(r): r for r in a.relations}
    b_rels = {_rel_key(r): r for r in b.relations}
    shared_keys = sorted(a_rels.keys() & b_rels.keys())

    agreement_rels: list[dict[str, Any]] = []
    sign_conflicts: list[dict[str, Any]] = []
    lag_conflicts: list[dict[str, Any]] = []
    for k in shared_keys:
        ra, rb = a_rels[k], b_rels[k]
        # Only emit agreement if BOTH endpoints are in entity_agreement —
        # otherwise one model is making a claim on an entity the other
        # doesn't recognise, and the edge cannot honestly be called agreed.
        if ra.source not in entity_agreement or ra.target not in entity_agreement:
            continue
        sign, sign_conflict = _reconcile_sign(ra.sign, rb.sign)
        if sign_conflict:
            sign_conflicts.append({
                "key": k, "a_sign": ra.sign, "b_sign": rb.sign,
            })
        lag_conflict = ra.lag != rb.lag and ra.lag != "unknown" and rb.lag != "unknown"
        if lag_conflict:
            lag_conflicts.append({
                "key": k, "a_lag": ra.lag, "b_lag": rb.lag,
            })
        lag = ra.lag if ra.lag == rb.lag else (
            rb.lag if ra.lag == "unknown" else (
                ra.lag if rb.lag == "unknown" else "unknown"
            )
        )
        agreement_rels.append({
            "key": k,
            "source": ra.source, "target": ra.target,
            "relation_type": ra.relation_type,
            "sign": sign, "lag": lag,
            "edge_class": ra.edge_class if ra.edge_class == rb.edge_class else "mechanism",
        })

    a_only_rel_keys = sorted(a_rels.keys() - b_rels.keys())
    b_only_rel_keys = sorted(b_rels.keys() - a_rels.keys())
    a_only_relations = [
        {"key": k, "source": k[0], "target": k[1], "relation_type": k[2],
         "sign": a_rels[k].sign, "lag": a_rels[k].lag}
        for k in a_only_rel_keys
    ]
    b_only_relations = [
        {"key": k, "source": k[0], "target": k[1], "relation_type": k[2],
         "sign": b_rels[k].sign, "lag": b_rels[k].lag}
        for k in b_only_rel_keys
    ]

    # Summary rates: treat the union as the denominator so "model A only" and
    # "model B only" both reduce the agreement rate the same way.
    total_entities = len(a_names | b_names)
    entity_rate = len(entity_agreement) / total_entities if total_entities else 1.0
    total_rel_keys = len(set(a_rels) | set(b_rels))
    rel_rate = len(agreement_rels) / total_rel_keys if total_rel_keys else 1.0

    known_sign_pairs = sum(
        1 for k in shared_keys
        if a_rels[k].sign != "unknown" and b_rels[k].sign != "unknown"
    )
    sign_agreements = sum(
        1 for k in shared_keys
        if a_rels[k].sign != "unknown" and b_rels[k].sign != "unknown"
        and a_rels[k].sign == b_rels[k].sign
    )
    sign_rate = sign_agreements / known_sign_pairs if known_sign_pairs else 1.0

    return {
        "entities": {
            "agreement": entity_agreement,
            "a_only": a_only_entities,
            "b_only": b_only_entities,
            "type_conflicts": type_conflicts,
        },
        "relations": {
            "agreement": agreement_rels,
            "a_only": a_only_relations,
            "b_only": b_only_relations,
            "sign_conflicts": sign_conflicts,
            "lag_conflicts": lag_conflicts,
        },
        "summary": {
            "entity_agreement_rate": entity_rate,
            "relation_agreement_rate": rel_rate,
            "sign_agreement_rate": sign_rate,
            "n_entities_a": len(a_names),
            "n_entities_b": len(b_names),
            "n_relations_a": len(a_rels),
            "n_relations_b": len(b_rels),
        },
    }


def build_agreement_kg(
    report: dict[str, Any],
    kg_a: Union[KnowledgeGraph, str, Path],
    kg_b: Union[KnowledgeGraph, str, Path],
) -> KnowledgeGraph:
    """Materialise the agreement intersection as a standalone KG.

    Entities are cloned from `kg_a` (arbitrary but deterministic; for
    agreement entities, both sides agree on identity anyway). Relations
    use the reconciled sign/lag/edge_class from the report.
    """
    a = _load(kg_a)
    _ = _load(kg_b)  # validated; not used further
    out = KnowledgeGraph()
    for name in report["entities"]["agreement"]:
        src = a.entities[name]
        out.add_entity(Entity(name=src.name, entity_type=src.entity_type,
                              aliases=list(src.aliases),
                              observations=list(src.observations)))
    for r in report["relations"]["agreement"]:
        out.add_relation(Relation(
            source=r["source"], target=r["target"],
            relation_type=r["relation_type"],
            sign=r["sign"], lag=r["lag"],
            edge_class=r["edge_class"],
        ))
    return out
