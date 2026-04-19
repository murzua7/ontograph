"""Core data models for ontograph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# Edge-semantic literals used by Relation. Stored as strings so JSON round-trips
# cleanly; the Literal annotations are documentation + static-check hints.
Sign = Literal["+", "-", "±", "0", "unknown"]
Lag = str  # "within_step", "across_step", or "Q+k" for integer k, or "unknown"
Form = Literal["linear", "monotone", "threshold", "identity", "unknown"]
EdgeClass = Literal["identity", "mechanism", "parameter", "structural", "abstract"]


@dataclass
class Provenance:
    """Source location for an extracted entity or relation."""
    document: str
    section: str = ""
    page: int | None = None
    passage: str = ""
    char_offset: int = 0


@dataclass
class CodeAnchor:
    """Grounding anchor: a concrete code location."""
    repo: str
    path: str
    line: int
    symbol: str = ""


@dataclass
class CitationAnchor:
    """Grounding anchor: a bibliographic reference."""
    key: str
    pages: str = ""


@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Grounding — discipline-v1 additions
    code_anchors: list[CodeAnchor] = field(default_factory=list)
    citation_anchors: list[CitationAnchor] = field(default_factory=list)
    abstract: bool = False
    abstract_rationale: str = ""
    confidence: float = 1.0

    @property
    def id(self) -> str:
        slug = self.name.lower().replace(" ", "_")
        h = hashlib.sha256(self.name.encode()).hexdigest()[:8]
        return f"{slug}_{h}"

    def is_grounded(self) -> bool:
        """True iff the entity carries at least one non-empty code or citation anchor.

        `abstract` is orthogonal: an abstract entity that deliberately has no anchors
        still returns False here. The quality gate is responsible for combining
        is_grounded() with the abstract flag to decide whether ungroundedness is a defect.
        """
        for a in self.code_anchors:
            if a.path and a.repo:
                return True
        for c in self.citation_anchors:
            if c.key:
                return True
        return False


@dataclass
class Relation:
    """A typed edge in the knowledge graph."""
    source: str  # entity name
    target: str  # entity name
    relation_type: str
    weight: float = 1.0
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Edge semantics — discipline-v1 additions
    sign: Sign = "unknown"
    lag: Lag = "unknown"
    form: Form = "unknown"
    conditional_on: list[str] = field(default_factory=list)
    edge_class: EdgeClass = "mechanism"
    confidence: float = 1.0


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
            # Merge anchors by identity tuple to avoid duplicates.
            existing_codes = {(a.repo, a.path, a.line, a.symbol) for a in existing.code_anchors}
            for a in entity.code_anchors:
                if (a.repo, a.path, a.line, a.symbol) not in existing_codes:
                    existing.code_anchors.append(a)
            existing_cites = {(c.key, c.pages) for c in existing.citation_anchors}
            for c in entity.citation_anchors:
                if (c.key, c.pages) not in existing_cites:
                    existing.citation_anchors.append(c)
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
            code_anchors = [CodeAnchor(**a) for a in edata.get("code_anchors", [])]
            citation_anchors = [CitationAnchor(**c) for c in edata.get("citation_anchors", [])]
            kg.entities[name] = Entity(
                name=edata["name"],
                entity_type=edata["entity_type"],
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=prov,
                metadata=edata.get("metadata", {}),
                code_anchors=code_anchors,
                citation_anchors=citation_anchors,
                abstract=edata.get("abstract", False),
                abstract_rationale=edata.get("abstract_rationale", ""),
                confidence=edata.get("confidence", 1.0),
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
                sign=rdata.get("sign", "unknown"),
                lag=rdata.get("lag", "unknown"),
                form=rdata.get("form", "unknown"),
                conditional_on=rdata.get("conditional_on", []),
                edge_class=rdata.get("edge_class", "mechanism"),
                confidence=rdata.get("confidence", 1.0),
            ))
        return kg
