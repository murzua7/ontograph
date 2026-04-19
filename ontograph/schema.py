"""Ontology schema loader and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


@dataclass
class EntityTypeDef:
    name: str
    description: str = ""
    subtypes: list[str] = field(default_factory=list)
    color: str = "#607D8B"


@dataclass
class RelationTypeDef:
    name: str
    description: str = ""
    domain: list[str] = field(default_factory=list)  # allowed source types
    range: list[str] = field(default_factory=list)   # allowed target types
    symmetric: bool = False
    # Edge-semantic defaults used by extractors when the evidence is silent.
    default_sign: str = "unknown"
    default_lag: str = "unknown"
    default_form: str = "unknown"
    default_edge_class: str = "mechanism"


@dataclass
class OntologySchema:
    name: str
    version: str = "1.0"
    description: str = ""
    entity_types: dict[str, EntityTypeDef] = field(default_factory=dict)
    relation_types: dict[str, RelationTypeDef] = field(default_factory=dict)
    extraction_hints: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_type_names(self) -> list[str]:
        return list(self.entity_types.keys())

    @property
    def relation_type_names(self) -> list[str]:
        return list(self.relation_types.keys())

    def validate_entity_type(self, t: str) -> bool:
        return t in self.entity_types

    def validate_relation(self, source_type: str, rel_type: str, target_type: str) -> bool:
        if rel_type not in self.relation_types:
            return False
        rdef = self.relation_types[rel_type]
        if rdef.domain and source_type not in rdef.domain:
            return False
        if rdef.range and target_type not in rdef.range:
            return False
        return True


def _resolve_schema_path(name_or_path: str) -> Path:
    """Resolve a schema reference to an absolute path.

    Order: existing path on disk → SCHEMAS_DIR/<name>.yaml. Raises FileNotFoundError otherwise.
    """
    path = Path(name_or_path)
    if path.exists():
        return path.resolve()
    preset = SCHEMAS_DIR / f"{name_or_path}.yaml"
    if preset.exists():
        return preset.resolve()
    raise FileNotFoundError(f"Schema not found: {name_or_path} (tried {path}, {preset})")


def _merge_raw(parent: dict, child: dict) -> dict:
    """Merge a child raw YAML dict over a parent raw YAML dict.

    Rules:
    - Scalars (`name`, `version`, `description`): child overrides parent.
    - `entity_types`, `relation_types`: shallow dict merge; child entry for a given
      name replaces parent entry entirely (no deep merge of subtypes).
    - `validation`: shallow dict merge by key; child key overrides parent on collision.
    - `extraction_hints`: per-key list concatenation, de-duplicated preserving order.
    """
    out: dict = dict(parent)
    for key, val in child.items():
        if key in ("entity_types", "relation_types"):
            merged = dict(parent.get(key) or {})
            for name, defn in (val or {}).items():
                merged[name] = defn
            out[key] = merged
        elif key == "validation":
            merged = dict(parent.get("validation") or {})
            for vk, vv in (val or {}).items():
                merged[vk] = vv
            out[key] = merged
        elif key == "extraction_hints":
            parent_hints = parent.get("extraction_hints") or {}
            child_hints = val or {}
            merged_hints: dict[str, list] = {}
            for hk in set(parent_hints) | set(child_hints):
                lst = list(parent_hints.get(hk, []) or [])
                for item in child_hints.get(hk, []) or []:
                    if item not in lst:
                        lst.append(item)
                merged_hints[hk] = lst
            out[key] = merged_hints
        else:
            out[key] = val
    return out


def _load_raw(path: Path, visited: frozenset[Path]) -> dict:
    """Load a YAML schema file and recursively merge all parents reachable via `extends:`.

    Cycle detection: triggered *before* YAML read of the offending file.
    """
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Cycle detected in schema extends chain: {path} already visited")
    visited = visited | {path}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    if isinstance(parent_ref, list):
        # Multi-inheritance: merge parents left-to-right, then child overrides.
        merged_parents: dict = {}
        for ref in parent_ref:
            parent_path = _resolve_schema_path(str(ref))
            merged_parents = _merge_raw(merged_parents, _load_raw(parent_path, visited))
        return _merge_raw(merged_parents, raw)
    parent_path = _resolve_schema_path(str(parent_ref))
    parent_raw = _load_raw(parent_path, visited)
    return _merge_raw(parent_raw, raw)


def load_schema(name_or_path: str) -> OntologySchema:
    """Load an ontology schema from a preset name or file path.

    Supports `extends:` for hybrid base-plus-domain composition.
    """
    path = _resolve_schema_path(name_or_path)
    raw = _load_raw(path, frozenset())

    schema = OntologySchema(
        name=raw.get("name", path.stem),
        version=raw.get("version", "1.0"),
        description=raw.get("description", ""),
        validation=dict(raw.get("validation") or {}),
    )

    for etype, edef in (raw.get("entity_types") or {}).items():
        if isinstance(edef, str):
            edef = {"description": edef}
        schema.entity_types[etype] = EntityTypeDef(
            name=etype,
            description=edef.get("description", ""),
            subtypes=edef.get("subtypes", []),
            color=edef.get("color", "#607D8B"),
        )

    for rtype, rdef in (raw.get("relation_types") or {}).items():
        if isinstance(rdef, str):
            rdef = {"description": rdef}
        schema.relation_types[rtype] = RelationTypeDef(
            name=rtype,
            description=rdef.get("description", ""),
            domain=rdef.get("domain", []),
            range=rdef.get("range", []),
            symmetric=rdef.get("symmetric", False),
            default_sign=rdef.get("default_sign", "unknown"),
            default_lag=rdef.get("default_lag", "unknown"),
            default_form=rdef.get("default_form", "unknown"),
            default_edge_class=rdef.get("default_edge_class", "mechanism"),
        )

    hints = raw.get("extraction_hints") or {}
    schema.extraction_hints = {k: list(v or []) for k, v in hints.items()}
    schema.extraction_hints.setdefault("entity_signals", [])
    schema.extraction_hints.setdefault("relation_signals", [])

    return schema
