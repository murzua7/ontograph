"""NetworkX-backed ontology graph with analysis methods."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ontograph.models import Entity, Relation, KnowledgeGraph
from ontograph.schema import OntologySchema


class OntologyGraph:
    """Interactive ontology graph backed by NetworkX."""

    def __init__(self, schema: OntologySchema | None = None):
        self.g = nx.DiGraph()
        self.schema = schema
        self._entities: dict[str, Entity] = {}

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.name] = entity
        self.g.add_node(
            entity.name,
            entity_type=entity.entity_type,
            observations=entity.observations,
            aliases=entity.aliases,
        )

    def add_relation(self, relation: Relation) -> None:
        self.g.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            weight=relation.weight,
        )

    def get_entity(self, name: str) -> Entity | None:
        return self._entities.get(name)

    @property
    def n_entities(self) -> int:
        return self.g.number_of_nodes()

    @property
    def n_relations(self) -> int:
        return self.g.number_of_edges()

    def entities_by_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def neighbors(self, name: str, direction: str = "both") -> list[str]:
        if direction == "out":
            return list(self.g.successors(name))
        elif direction == "in":
            return list(self.g.predecessors(name))
        return list(set(self.g.successors(name)) | set(self.g.predecessors(name)))

    # --- Analysis ---

    def find_cycles(self) -> list[list[str]]:
        """Find all simple cycles in the graph."""
        try:
            return list(nx.simple_cycles(self.g))
        except nx.NetworkXError:
            return []

    def find_chains(self, min_length: int = 3) -> list[list[str]]:
        """Find longest paths (chains without cycles)."""
        chains = []
        for source in self.g.nodes():
            for target in self.g.nodes():
                if source != target:
                    try:
                        for path in nx.all_simple_paths(self.g, source, target, cutoff=10):
                            if len(path) >= min_length:
                                chains.append(path)
                    except nx.NetworkXError:
                        continue
        # Deduplicate and sort by length
        seen = set()
        unique = []
        for chain in sorted(chains, key=len, reverse=True):
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                unique.append(chain)
        return unique[:20]  # top 20

    def cascade_depth(self, relation_type: str = "propagates_losses_to") -> int:
        """Longest path through edges of a given relation type."""
        subg = nx.DiGraph()
        for u, v, data in self.g.edges(data=True):
            if data.get("relation_type") == relation_type:
                subg.add_edge(u, v)
        if subg.number_of_edges() == 0:
            return 0
        try:
            return nx.dag_longest_path_length(subg)
        except nx.NetworkXUnfeasible:
            # Has cycles — return cycle length as proxy
            cycles = list(nx.simple_cycles(subg))
            return max(len(c) for c in cycles) if cycles else 0

    def degree_centrality(self) -> dict[str, float]:
        return nx.degree_centrality(self.g)

    def connected_components(self) -> list[set[str]]:
        return [c for c in nx.weakly_connected_components(self.g)]

    def subgraph(self, names: list[str]) -> OntologyGraph:
        sub = OntologyGraph(schema=self.schema)
        for name in names:
            if name in self._entities:
                sub.add_entity(self._entities[name])
        for u, v, data in self.g.edges(data=True):
            if u in names and v in names:
                sub.g.add_edge(u, v, **data)
        return sub

    # --- Serialization ---

    def to_kg(self) -> KnowledgeGraph:
        kg = KnowledgeGraph(schema_name=self.schema.name if self.schema else "base")
        kg.entities = dict(self._entities)
        for u, v, data in self.g.edges(data=True):
            source_entity = self._entities.get(u)
            target_entity = self._entities.get(v)
            if source_entity and target_entity:
                kg.relations.append(Relation(
                    source=u, target=v,
                    relation_type=data.get("relation_type", "relates_to"),
                    weight=data.get("weight", 1.0),
                ))
        return kg

    @classmethod
    def from_kg(cls, kg: KnowledgeGraph, schema: OntologySchema | None = None) -> OntologyGraph:
        og = cls(schema=schema)
        for entity in kg.entities.values():
            og.add_entity(entity)
        for relation in kg.relations:
            og.add_relation(relation)
        return og

    def summary(self) -> dict[str, Any]:
        """Quick summary statistics."""
        type_counts = {}
        for e in self._entities.values():
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

        rel_type_counts = {}
        for _, _, data in self.g.edges(data=True):
            rt = data.get("relation_type", "unknown")
            rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1

        cycles = self.find_cycles()

        return {
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "entity_types": type_counts,
            "relation_types": rel_type_counts,
            "n_cycles": len(cycles),
            "n_components": len(self.connected_components()),
        }
