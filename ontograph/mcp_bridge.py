"""Bridge between ontograph and MCP memory knowledge graph.

Exports entities/relations to the MCP memory JSONL format used by
Claude Code's memory server (~/.claude/memory/knowledge_graph.jsonl).
"""

from __future__ import annotations

import json
from pathlib import Path

from ontograph.graph import OntologyGraph
from ontograph.models import KnowledgeGraph

DEFAULT_JSONL_PATH = Path.home() / ".claude" / "memory" / "knowledge_graph.jsonl"


def export_to_mcp(
    graph: OntologyGraph,
    prefix: str = "ontograph",
    path: Path | str = DEFAULT_JSONL_PATH,
    append: bool = True,
) -> int:
    """Export graph entities and relations to MCP memory JSONL format.

    Args:
        graph: OntologyGraph to export
        prefix: Namespace prefix for entity names (avoids collisions)
        path: Path to knowledge_graph.jsonl
        append: If True, append to existing file; if False, overwrite

    Returns:
        Number of lines written
    """
    path = Path(path)
    lines = []

    for name, entity in graph._entities.items():
        prefixed_name = f"{prefix}:{name}" if prefix else name
        lines.append(json.dumps({
            "type": "entity",
            "name": prefixed_name,
            "entityType": entity.entity_type,
            "observations": entity.observations[:10],
        }))

    for u, v, data in graph.g.edges(data=True):
        source = f"{prefix}:{u}" if prefix else u
        target = f"{prefix}:{v}" if prefix else v
        lines.append(json.dumps({
            "type": "relation",
            "from": source,
            "to": target,
            "relationType": data.get("relation_type", "relates_to"),
        }))

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return len(lines)


def import_from_mcp(
    prefix: str = "ontograph",
    path: Path | str = DEFAULT_JSONL_PATH,
) -> KnowledgeGraph:
    """Import entities/relations from MCP memory JSONL, filtered by prefix.

    Args:
        prefix: Only import entities/relations with this namespace prefix
        path: Path to knowledge_graph.jsonl

    Returns:
        KnowledgeGraph with imported data
    """
    from ontograph.models import Entity, Relation

    path = Path(path)
    if not path.exists():
        return KnowledgeGraph()

    kg = KnowledgeGraph()
    prefix_pattern = f"{prefix}:" if prefix else ""

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") == "entity":
                name = record.get("name", "")
                if prefix_pattern and not name.startswith(prefix_pattern):
                    continue
                clean_name = name[len(prefix_pattern):] if prefix_pattern else name
                kg.add_entity(Entity(
                    name=clean_name,
                    entity_type=record.get("entityType", "concept"),
                    observations=record.get("observations", []),
                ))

            elif record.get("type") == "relation":
                source = record.get("from", "")
                target = record.get("to", "")
                if prefix_pattern:
                    if not source.startswith(prefix_pattern) or not target.startswith(prefix_pattern):
                        continue
                    source = source[len(prefix_pattern):]
                    target = target[len(prefix_pattern):]
                kg.add_relation(Relation(
                    source=source,
                    target=target,
                    relation_type=record.get("relationType", "relates_to"),
                ))

    return kg
