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


@dataclass
class OntologySchema:
    name: str
    version: str = "1.0"
    description: str = ""
    entity_types: dict[str, EntityTypeDef] = field(default_factory=dict)
    relation_types: dict[str, RelationTypeDef] = field(default_factory=dict)
    extraction_hints: dict[str, list[str]] = field(default_factory=dict)

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


def load_schema(name_or_path: str) -> OntologySchema:
    """Load an ontology schema from a preset name or file path."""
    path = Path(name_or_path)
    if not path.exists():
        path = SCHEMAS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {name_or_path} (tried {path})")

    with open(path) as f:
        raw = yaml.safe_load(f)

    schema = OntologySchema(
        name=raw.get("name", path.stem),
        version=raw.get("version", "1.0"),
        description=raw.get("description", ""),
    )

    for etype, edef in raw.get("entity_types", {}).items():
        if isinstance(edef, str):
            edef = {"description": edef}
        schema.entity_types[etype] = EntityTypeDef(
            name=etype,
            description=edef.get("description", ""),
            subtypes=edef.get("subtypes", []),
            color=edef.get("color", "#607D8B"),
        )

    for rtype, rdef in raw.get("relation_types", {}).items():
        if isinstance(rdef, str):
            rdef = {"description": rdef}
        schema.relation_types[rtype] = RelationTypeDef(
            name=rtype,
            description=rdef.get("description", ""),
            domain=rdef.get("domain", []),
            range=rdef.get("range", []),
            symmetric=rdef.get("symmetric", False),
        )

    hints = raw.get("extraction_hints", {})
    schema.extraction_hints = {
        "entity_signals": hints.get("entity_signals", []),
        "relation_signals": hints.get("relation_signals", []),
    }

    return schema
