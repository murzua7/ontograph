"""CLI entry point for ontograph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _make_llm_client(args):
    """Build LLM client from CLI args."""
    from ontograph.llm_client import LLMClient
    return LLMClient(
        mode=getattr(args, "llm_backend", "anthropic"),
        model=getattr(args, "llm_model", None),
    )


def cmd_ingest(args):
    """Ingest a document and extract a knowledge graph."""
    from ontograph.parsers import parse_document
    from ontograph.schema import load_schema
    from ontograph.extractor import extract_from_document
    from ontograph.graph import OntologyGraph

    schema = load_schema(args.schema)
    path = Path(args.input)

    if path.is_dir():
        files = list(path.glob("**/*.md")) + list(path.glob("**/*.html")) + list(path.glob("**/*.pdf")) + list(path.glob("**/*.txt"))
    else:
        files = [path]

    mode_label = f"llm ({args.llm_backend})" if args.llm else "heuristic"
    console.print(f"[bold]ontograph[/bold] | schema={schema.name} | mode={mode_label} | {len(files)} file(s)")

    graph = OntologyGraph(schema=schema)

    for f in files:
        console.print(f"  Parsing: {f.name}")
        doc = parse_document(f)

        if args.llm:
            from ontograph.llm_extractor import llm_extract_from_document
            client = _make_llm_client(args)
            entities, relations = llm_extract_from_document(doc, schema, client=client)
        else:
            entities, relations = extract_from_document(doc, schema)

        for e in entities:
            graph.add_entity(e)
        for r in relations:
            graph.add_relation(r)

        console.print(f"    -> {len(entities)} entities, {len(relations)} relations")

    # Output
    output = Path(args.output)
    kg = graph.to_kg()
    kg.documents = [str(f) for f in files]
    output.write_text(kg.to_json(), encoding="utf-8")
    console.print(f"\n[green]Graph saved to {output}[/green]")

    _print_summary(graph)


def cmd_merge(args):
    """Merge multiple knowledge graphs with entity resolution."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph
    from ontograph.resolver import merge_knowledge_graphs

    graphs = []
    for path in args.inputs:
        kg = KnowledgeGraph.from_json(Path(path).read_text(encoding="utf-8"))
        graphs.append(kg)
        console.print(f"  Loaded: {path} ({len(kg.entities)} entities, {len(kg.relations)} relations)")

    merged = merge_knowledge_graphs(graphs, threshold=args.threshold)
    graph = OntologyGraph.from_kg(merged)

    total_before = sum(len(g.entities) for g in graphs)
    console.print(f"\n  Entities before: {total_before} -> after: {len(merged.entities)} (resolved {total_before - len(merged.entities)})")

    output = Path(args.output)
    output.write_text(merged.to_json(), encoding="utf-8")
    console.print(f"[green]Merged graph saved to {output}[/green]")

    _print_summary(graph)


def cmd_analyze(args):
    """Analyze a knowledge graph."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    console.print(f"[bold]ontograph analyze[/bold] | {graph.n_entities} entities, {graph.n_relations} relations\n")

    if args.cycles or args.all:
        cycles = graph.find_cycles()
        console.print(f"[bold]Cycles ({len(cycles)}):[/bold]")
        for cycle in cycles[:10]:
            console.print(f"  {' -> '.join(cycle)} -> {cycle[0]}")

    if args.centrality or args.all:
        centrality = graph.degree_centrality()
        top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
        table = Table(title="Degree Centrality (top 15)")
        table.add_column("Entity")
        table.add_column("Centrality", justify="right")
        for name, val in top:
            table.add_row(name, f"{val:.3f}")
        console.print(table)

    if args.cascade_depth or args.all:
        depth = graph.cascade_depth()
        console.print(f"\n[bold]Max cascade depth:[/bold] {depth}")


def cmd_export(args):
    """Export a knowledge graph to another format."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph
    from ontograph.export import export_graphml, export_mermaid, export_jsonld

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    fmt = args.format
    output = Path(args.output) if args.output else Path(args.input).with_suffix(f".{fmt}")

    if fmt == "graphml":
        export_graphml(graph, output)
    elif fmt == "mermaid":
        output.write_text(export_mermaid(graph), encoding="utf-8")
    elif fmt == "jsonld":
        output.write_text(export_jsonld(graph), encoding="utf-8")
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        return

    console.print(f"[green]Exported to {output}[/green]")


def cmd_mcp_sync(args):
    """Sync a knowledge graph to MCP memory."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph
    from ontograph.mcp_bridge import export_to_mcp

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    n = export_to_mcp(graph, prefix=args.prefix, append=not args.overwrite)
    console.print(f"[green]Synced {n} records to MCP memory (prefix={args.prefix})[/green]")


def cmd_serve(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(args.port)])


def _print_summary(graph):
    summary = graph.summary()
    table = Table(title="Extraction Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Entities", str(summary["n_entities"]))
    table.add_row("Relations", str(summary["n_relations"]))
    table.add_row("Cycles", str(summary["n_cycles"]))
    table.add_row("Components", str(summary["n_components"]))
    for etype, count in summary["entity_types"].items():
        table.add_row(f"  {etype}", str(count))
    for rtype, count in summary.get("relation_types", {}).items():
        table.add_row(f"  [{rtype}]", str(count))
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        prog="ontograph",
        description="Extract ontological knowledge graphs from documents",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Extract knowledge graph from document(s)")
    p_ingest.add_argument("input", help="Document or directory path")
    p_ingest.add_argument("--schema", default="base", help="Ontology schema name or path")
    p_ingest.add_argument("--output", "-o", default="graph.json", help="Output JSON path")
    p_ingest.add_argument("--llm", action="store_true", help="Use LLM extraction")
    p_ingest.add_argument("--llm-backend", default="anthropic",
                          choices=["anthropic", "openai", "ollama"],
                          help="LLM backend (default: anthropic)")
    p_ingest.add_argument("--llm-model", default=None, help="Override model name")

    # merge
    p_merge = sub.add_parser("merge", help="Merge multiple knowledge graphs")
    p_merge.add_argument("inputs", nargs="+", help="Graph JSON paths to merge")
    p_merge.add_argument("--output", "-o", default="merged.json", help="Output path")
    p_merge.add_argument("--threshold", type=float, default=0.85,
                         help="Fuzzy matching threshold (0-1, default: 0.85)")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a knowledge graph")
    p_analyze.add_argument("input", help="Graph JSON path")
    p_analyze.add_argument("--cycles", action="store_true")
    p_analyze.add_argument("--centrality", action="store_true")
    p_analyze.add_argument("--cascade-depth", action="store_true")
    p_analyze.add_argument("--all", action="store_true")

    # export
    p_export = sub.add_parser("export", help="Export graph to another format")
    p_export.add_argument("input", help="Graph JSON path")
    p_export.add_argument("--format", "-f", choices=["graphml", "mermaid", "jsonld"], required=True)
    p_export.add_argument("--output", "-o", help="Output path")

    # mcp-sync
    p_mcp = sub.add_parser("mcp-sync", help="Sync graph to MCP memory")
    p_mcp.add_argument("input", help="Graph JSON path")
    p_mcp.add_argument("--prefix", default="ontograph", help="Namespace prefix")
    p_mcp.add_argument("--overwrite", action="store_true", help="Overwrite instead of append")

    # serve
    p_serve = sub.add_parser("serve", help="Launch Streamlit dashboard")
    p_serve.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    commands = {
        "ingest": cmd_ingest,
        "merge": cmd_merge,
        "analyze": cmd_analyze,
        "export": cmd_export,
        "mcp-sync": cmd_mcp_sync,
        "serve": cmd_serve,
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
