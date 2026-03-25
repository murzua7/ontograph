"""Core data models for ontograph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Provenance:
    """Source location for an extracted entity or relation."""
    document: str
    section: str = ""
    page: int | None = None
    passage: str = ""
    char_offset: int = 0


@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        slug = self.name.lower().replace(" ", "_")
        h = hashlib.sha256(self.name.encode()).hexdigest()[:8]
        return f"{slug}_{h}"


@dataclass
class Relation:
    """A typed edge in the knowledge graph."""
    source: str  # entity name
    target: str  # entity name
    relation_type: str
    weight: float = 1.0
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Serializable knowledge graph."""
    entities: dict[str, Entity] = field(default_factory=dict)  # name -> Entity
    relations: list[Relation] = field(default_factory=list)
    schema_name: str = "base"
    documents: list[str] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        if entity.name in self.entities:
            existing = self.entities[entity.name]
            existing.aliases = list(set(existing.aliases + entity.aliases))
            existing.observations = list(set(existing.observations + entity.observations))
            existing.provenance.extend(entity.provenance)
        else:
            self.entities[entity.name] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def to_json(self) -> str:
        def _ser(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)
        return json.dumps(asdict(self), default=_ser, indent=2)

    @classmethod
    def from_json(cls, data: str) -> KnowledgeGraph:
        raw = json.loads(data)
        kg = cls(schema_name=raw.get("schema_name", "base"),
                 documents=raw.get("documents", []))
        for name, edata in raw.get("entities", {}).items():
            prov = [Provenance(**p) for p in edata.get("provenance", [])]
            kg.entities[name] = Entity(
                name=edata["name"],
                entity_type=edata["entity_type"],
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=prov,
                metadata=edata.get("metadata", {}),
            )
        for rdata in raw.get("relations", []):
            prov = [Provenance(**p) for p in rdata.get("provenance", [])]
            kg.relations.append(Relation(
                source=rdata["source"],
                target=rdata["target"],
                relation_type=rdata["relation_type"],
                weight=rdata.get("weight", 1.0),
                provenance=prov,
                metadata=rdata.get("metadata", {}),
            ))
        return kg
