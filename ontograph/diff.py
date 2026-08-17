"""Structured KG diff for viewer overlay and programmatic consumers (#44).

Public surface: `diff_kgs(base, head) -> dict`. Classifies entities and
relations into added / removed / changed / unchanged. Shares the
`(source, target, relation_type)` relation-key convention with the existing
markdown-diff CLI so both surfaces agree on identity.

Fields watched for 'changed':
- Entity: `entity_type` (identity-adjacent; alias/observation drift is
  bookkeeping and would produce noisy diffs).
- Relation: `sign`, `lag`, `edge_class`, `form`, `weight`, `confidence` —
  the discipline-v1 semantic facets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from ontograph.models import KnowledgeGraph

_RELATION_FIELDS_WATCHED = ("sign", "lag", "edge_class", "form", "weight", "confidence")


def _load(kg: Union[KnowledgeGraph, str, Path]) -> KnowledgeGraph:
    if isinstance(kg, KnowledgeGraph):
        return kg
    return KnowledgeGraph.from_json(Path(kg).read_text(encoding="utf-8"))


def _rel_key(r) -> tuple[str, str, str]:
    return (r.source, r.target, r.relation_type)


def diff_kgs(
    base: Union[KnowledgeGraph, str, Path],
    head: Union[KnowledgeGraph, str, Path],
) -> dict[str, Any]:
    """Classify entity and relation changes between two KGs.

    Returns a dict with `entities`, `relations`, and `summary` sections.
    Relation keys appear in the result as 3-tuples `(source, target, rel)`.
    """
    base_kg = _load(base)
    head_kg = _load(head)

    base_names = set(base_kg.entities)
    head_names = set(head_kg.entities)
    added_e = sorted(head_names - base_names)
    removed_e = sorted(base_names - head_names)
    kept_e = sorted(base_names & head_names)

    changed_e: list[dict[str, Any]] = []
    unchanged_e: list[str] = []
    for name in kept_e:
        b = base_kg.entities[name]
        h = head_kg.entities[name]
        if b.entity_type != h.entity_type:
            changed_e.append({
                "name": name,
                "fields": {"entity_type": (b.entity_type, h.entity_type)},
            })
        else:
            unchanged_e.append(name)

    # Collapse parallel edges under the first relation with that key — matches
    # `cmd_diff` behaviour. If two edges share (src, tgt, rel), only the first
    # participates in the diff.
    base_rels: dict[tuple[str, str, str], Any] = {}
    for r in base_kg.relations:
        base_rels.setdefault(_rel_key(r), r)
    head_rels: dict[tuple[str, str, str], Any] = {}
    for r in head_kg.relations:
        head_rels.setdefault(_rel_key(r), r)

    added_r = [{"key": k, "relation": head_rels[k]} for k in sorted(head_rels.keys() - base_rels.keys())]
    removed_r = [{"key": k, "relation": base_rels[k]} for k in sorted(base_rels.keys() - head_rels.keys())]

    changed_r: list[dict[str, Any]] = []
    unchanged_r: list[dict[str, Any]] = []
    for k in sorted(base_rels.keys() & head_rels.keys()):
        b = base_rels[k]
        h = head_rels[k]
        field_deltas: dict[str, tuple] = {}
        for f in _RELATION_FIELDS_WATCHED:
            bv = getattr(b, f, None)
            hv = getattr(h, f, None)
            if bv != hv:
                field_deltas[f] = (bv, hv)
        if field_deltas:
            changed_r.append({"key": k, "fields": field_deltas,
                              "base": b, "head": h})
        else:
            unchanged_r.append({"key": k, "relation": h})

    return {
        "entities": {
            "added": added_e,
            "removed": removed_e,
            "changed": changed_e,
            "unchanged": unchanged_e,
        },
        "relations": {
            "added": added_r,
            "removed": removed_r,
            "changed": changed_r,
            "unchanged": unchanged_r,
        },
        "summary": {
            "entities_added": len(added_e),
            "entities_removed": len(removed_e),
            "entities_changed": len(changed_e),
            "relations_added": len(added_r),
            "relations_removed": len(removed_r),
            "relations_changed": len(changed_r),
        },
    }
