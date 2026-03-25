"""CLI entry point for ontograph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_ingest(args):
    """Ingest a document and extract a knowledge graph."""
    from ontograph.parsers import parse_document
    from ontograph.schema import load_schema
    from ontograph.extractor import extract_from_document
    from ontograph.graph import OntologyGraph
    from ontograph.export import export_json

    schema = load_schema(args.schema)
    path = Path(args.input)

    if path.is_dir():
        files = list(path.glob("**/*.md")) + list(path.glob("**/*.html")) + list(path.glob("**/*.pdf")) + list(path.glob("**/*.txt"))
    else:
        files = [path]

    console.print(f"[bold]ontograph[/bold] | schema={schema.name} | {len(files)} file(s)")

    graph = OntologyGraph(schema=schema)

    for f in files:
        console.print(f"  Parsing: {f.name}")
        doc = parse_document(f)

        if args.llm:
            try:
                from ontograph.llm_extractor import llm_extract_from_document
                entities, relations = llm_extract_from_document(doc, schema)
            except ImportError:
                console.print("  [yellow]LLM extractor not available, falling back to heuristic[/yellow]")
                entities, relations = extract_from_document(doc, schema)
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

    # Summary
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
    console.print(table)


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


def cmd_serve(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(args.port)])


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
    p_ingest.add_argument("--llm", action="store_true", help="Use LLM extraction (requires llm_extractor)")

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

    # serve
    p_serve = sub.add_parser("serve", help="Launch Streamlit dashboard")
    p_serve.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
