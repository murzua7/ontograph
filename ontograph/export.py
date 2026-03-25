"""Export knowledge graphs to various formats."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ontograph.graph import OntologyGraph
from ontograph.models import KnowledgeGraph


def export_json(graph: OntologyGraph, path: str | Path) -> None:
    """Export as JSON (canonical portable format)."""
    kg = graph.to_kg()
    Path(path).write_text(kg.to_json(), encoding="utf-8")


def export_graphml(graph: OntologyGraph, path: str | Path) -> None:
    """Export as GraphML (compatible with Gephi, yEd, Cytoscape desktop)."""
    nx.write_graphml(graph.g, str(path))


def export_mermaid(graph: OntologyGraph) -> str:
    """Export as Mermaid diagram text."""
    lines = ["graph LR"]
    # Node definitions with type labels
    for name, data in graph.g.nodes(data=True):
        etype = data.get("entity_type", "concept")
        safe = name.replace(" ", "_").replace('"', "'")
        lines.append(f'    {safe}["{name}<br/><i>{etype}</i>"]')

    # Edge definitions
    for u, v, data in graph.g.edges(data=True):
        rtype = data.get("relation_type", "relates_to")
        safe_u = u.replace(" ", "_").replace('"', "'")
        safe_v = v.replace(" ", "_").replace('"', "'")
        lines.append(f'    {safe_u} -->|{rtype}| {safe_v}')

    return "\n".join(lines)


def export_jsonld(graph: OntologyGraph) -> str:
    """Export as JSON-LD (linked data format)."""
    context = {
        "@context": {
            "ontograph": "https://github.com/murzua7/ontograph/schema#",
            "name": "ontograph:name",
            "type": "ontograph:entityType",
            "source": "ontograph:source",
            "target": "ontograph:target",
            "relationType": "ontograph:relationType",
        }
    }

    entities = []
    for name, entity in graph._entities.items():
        entities.append({
            "@id": f"ontograph:entity/{entity.id}",
            "name": name,
            "type": entity.entity_type,
            "observations": entity.observations,
        })

    relations = []
    for u, v, data in graph.g.edges(data=True):
        relations.append({
            "source": u,
            "target": v,
            "relationType": data.get("relation_type", "relates_to"),
        })

    doc = {
        **context,
        "entities": entities,
        "relations": relations,
    }
    return json.dumps(doc, indent=2)
