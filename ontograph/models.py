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
class DocAnchor:
    """Grounding anchor: a concrete, reproducible pointer into the source doc.

    A DocAnchor upgrades a vague `Provenance.passage` string into a locatable
    span. The canonical pointer is (document, char_start, char_end) against
    the parsed document's full_text. `line` is an optional markdown line
    number (1-indexed) for human-readable navigation; `section` mirrors
    `Provenance.section` so DocAnchors are self-contained.

    Unlike `Provenance` (which is narrative — "this is where the claim came
    from"), a DocAnchor is a *grounding claim*: the entity is represented
    verbatim at this exact span. That is why DocAnchor counts toward
    `Entity.is_grounded()` and Provenance does not.
    """
    document: str
    char_start: int = -1
    char_end: int = -1
    line: int | None = None
    section: str = ""
    passage: str = ""


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
    doc_anchors: list[DocAnchor] = field(default_factory=list)
    abstract: bool = False
    abstract_rationale: str = ""
    confidence: float = 1.0

    @property
    def id(self) -> str:
        slug = self.name.lower().replace(" ", "_")
        h = hashlib.sha256(self.name.encode()).hexdigest()[:8]
        return f"{slug}_{h}"

    def is_grounded(self) -> bool:
        """True iff the entity carries at least one concrete anchor.

        Three grounding families, any one of which is sufficient:
          • `code_anchors` — concrete code reference (repo+path)
          • `citation_anchors` — bibliographic reference (key)
          • `doc_anchors` — reproducible document span (document + located
            char range or line)

        A DocAnchor with char_start ≥ 0 (or an explicit `line`) counts —
        both give a reader a reproducible way to re-find the passage. A
        DocAnchor carrying only document+passage (no offset) does NOT
        count: that is no stronger than ordinary Provenance.

        `abstract` is orthogonal: an abstract entity that deliberately
        has no anchors still returns False here. The quality gate is
        responsible for combining `is_grounded()` with the abstract flag.
        """
        for a in self.code_anchors:
            if a.path and a.repo:
                return True
        for c in self.citation_anchors:
            if c.key:
                return True
        for d in self.doc_anchors:
            if d.document and (d.char_start >= 0 or d.line is not None):
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
    # Extraction provenance: prompt_hash, schema_hash, model, backend,
    # ontograph_version, extracted_at, mode. Populated by `ontograph ingest`
    # via `ontograph.provenance.extraction_provenance`. Empty for KGs
    # created programmatically or loaded from older JSON.
    extraction: dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: Entity) -> None:
        if entity.name in self.entities:
            existing = self.entities[entity.name]
            # sorted(): set iteration order is hash-randomized across runs.
            existing.aliases = sorted(set(existing.aliases + entity.aliases))
            existing.observations = sorted(set(existing.observations + entity.observations))
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
            existing_docs = {(d.document, d.char_start, d.char_end, d.line)
                             for d in existing.doc_anchors}
            for d in entity.doc_anchors:
                if (d.document, d.char_start, d.char_end, d.line) not in existing_docs:
                    existing.doc_anchors.append(d)
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
                 documents=raw.get("documents", []),
                 extraction=raw.get("extraction", {}))
        for name, edata in raw.get("entities", {}).items():
            prov = [Provenance(**p) for p in edata.get("provenance", [])]
            code_anchors = [CodeAnchor(**a) for a in edata.get("code_anchors", [])]
            citation_anchors = [CitationAnchor(**c) for c in edata.get("citation_anchors", [])]
            doc_anchors = [DocAnchor(**d) for d in edata.get("doc_anchors", [])]
            kg.entities[name] = Entity(
                name=edata["name"],
                entity_type=edata["entity_type"],
                aliases=edata.get("aliases", []),
                observations=edata.get("observations", []),
                provenance=prov,
                metadata=edata.get("metadata", {}),
                code_anchors=code_anchors,
                citation_anchors=citation_anchors,
                doc_anchors=doc_anchors,
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
