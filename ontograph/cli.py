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


def _generate_batch_prompts(files, schema, batch_out_path):
    """Generate JSONL file of extraction prompts for offline LLM processing."""
    from ontograph.parsers import parse_document
    from ontograph.llm_extractor import (
        ENTITY_SYSTEM, ENTITY_PROMPT, RELATION_SYSTEM, RELATION_PROMPT,
        _format_entity_types, _format_relation_types, _chunk_text,
    )

    prompts = []
    prompt_id = 0

    for f in files:
        console.print(f"  Parsing: {f.name}")
        doc = parse_document(f)

        for section in doc.sections:
            if len(section.text.strip()) < 20:
                continue
            chunks = _chunk_text(section.text)
            for ci, chunk in enumerate(chunks):
                prompt_text = ENTITY_PROMPT.format(
                    schema_types=_format_entity_types(schema),
                    document_name=doc.source_path,
                    section_heading=section.heading,
                    section_text=chunk,
                )
                prompts.append({
                    "id": f"{f.stem}_{prompt_id}_chunk{ci}",
                    "prompt": f"{ENTITY_SYSTEM}\n\n{prompt_text}",
                    "section": section.heading,
                    "document": str(f),
                })
                prompt_id += 1

    out = Path(batch_out_path)
    with out.open("w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    console.print(f"\n[green]Generated {len(prompts)} prompts -> {out}[/green]")


def _import_batch_responses(batch_in_path, schema):
    """Import JSONL responses and build entities/relations with provenance."""
    from ontograph.models import Entity, Relation, Provenance
    from ontograph.llm_extractor import _dedup_entities

    entities = []
    relations = []

    with open(batch_in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rid = record.get("id", "unknown")
            response = record.get("response", {})
            document = record.get("document", rid)
            section = record.get("section", "")

            # Process entities
            for edata in response.get("entities", []):
                name = edata.get("name", "").strip()
                etype = edata.get("type", "").strip()
                if not name or not etype:
                    continue
                # Validate type against schema
                if not schema.validate_entity_type(etype):
                    for st in schema.entity_type_names:
                        if etype.lower() in st.lower() or st.lower() in etype.lower():
                            etype = st
                            break
                    else:
                        etype = schema.entity_type_names[0]

                prov = Provenance(
                    document=document,
                    section=section,
                    passage=edata.get("passage", "")[:300],
                )
                entities.append(Entity(
                    name=name,
                    entity_type=etype,
                    aliases=edata.get("aliases", []),
                    observations=edata.get("observations", []),
                    provenance=[prov],
                ))

            # Process relations
            entity_names = {e.name for e in entities}
            for rdata in response.get("relations", []):
                source = rdata.get("source", "").strip()
                target = rdata.get("target", "").strip()
                rtype = rdata.get("type", "").strip()
                if source not in entity_names or target not in entity_names:
                    continue
                if source == target:
                    continue
                if rtype not in schema.relation_types:
                    for rt in schema.relation_type_names:
                        if rtype.lower() in rt.lower() or rt.lower() in rtype.lower():
                            rtype = rt
                            break
                    else:
                        rtype = schema.relation_type_names[0]

                prov = Provenance(
                    document=document,
                    section=section,
                    passage=rdata.get("passage", "")[:300],
                )
                relations.append(Relation(
                    source=source,
                    target=target,
                    relation_type=rtype,
                    weight=float(rdata.get("confidence", 0.8)),
                    provenance=[prov],
                ))

    entities = _dedup_entities(entities)
    return entities, relations


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

    # --- Batch-out: generate prompts JSONL and exit ---
    if getattr(args, "batch_out", None):
        console.print(f"[bold]ontograph[/bold] | schema={schema.name} | mode=batch-out | {len(files)} file(s)")
        _generate_batch_prompts(files, schema, args.batch_out)
        return

    # --- Batch-in: import responses JSONL and build graph ---
    if getattr(args, "batch_in", None):
        console.print(f"[bold]ontograph[/bold] | schema={schema.name} | mode=batch-in | importing {args.batch_in}")
        entities, relations = _import_batch_responses(args.batch_in, schema)
        graph = OntologyGraph(schema=schema)
        for e in entities:
            graph.add_entity(e)
        for r in relations:
            graph.add_relation(r)
        console.print(f"  -> {len(entities)} entities, {len(relations)} relations")

        output = Path(args.output)
        kg = graph.to_kg()
        kg.documents = [str(f) for f in files]
        output.write_text(kg.to_json(), encoding="utf-8")
        console.print(f"\n[green]Graph saved to {output}[/green]")
        _print_summary(graph)
        return

    # --- Schema-free mode ---
    if getattr(args, "schema_free", False):
        from ontograph.llm_extractor import (
            schema_free_extract_from_document,
            consolidate_types,
            apply_type_consolidation,
            _dedup_entities,
        )
        client = _make_llm_client(args)
        console.print(
            f"[bold]ontograph[/bold] | mode=schema-free | {len(files)} file(s)\n"
            "  Pass 1+2: open extraction (no vocabulary constraints)"
        )

        all_raw_entities = []
        all_raw_relations = []

        for f in files:
            console.print(f"  Parsing: {f.name}")
            doc = parse_document(f)
            entities, relations = schema_free_extract_from_document(doc, client)
            all_raw_entities.extend(entities)
            all_raw_relations.extend(relations)
            console.print(f"    -> {len(entities)} entities, {len(relations)} relations (raw)")

        console.print(
            f"\n[bold]Pass 3: Type consolidation[/bold] "
            f"({len(set(e.entity_type for e in all_raw_entities))} entity types, "
            f"{len(set(r.relation_type for r in all_raw_relations))} relation types)"
        )

        entity_map, relation_map, entity_vocab, relation_vocab = consolidate_types(
            all_raw_entities, all_raw_relations, client, console=console,
        )

        apply_type_consolidation(all_raw_entities, all_raw_relations, entity_map, relation_map)
        entities = _dedup_entities(all_raw_entities)

        n_etypes = len(set(e.entity_type for e in entities))
        n_rtypes = len(set(r.relation_type for r in all_raw_relations))
        console.print(f"  Consolidated → {n_etypes} entity types, {n_rtypes} relation types")

        # Build graph from consolidated entities/relations
        graph = OntologyGraph(schema=None)
        for e in entities:
            graph.add_entity(e)
        for r in all_raw_relations:
            graph.add_relation(r)

        # Save induced schema if requested
        save_schema_path = getattr(args, "save_schema", None)
        if save_schema_path:
            from ontograph.export import export_induced_schema
            schema_stem = Path(save_schema_path).stem
            schema_yaml = export_induced_schema(entity_vocab, relation_vocab, name=schema_stem)
            Path(save_schema_path).write_text(schema_yaml, encoding="utf-8")
            console.print(f"[green]Induced schema saved to {save_schema_path}[/green]")
            console.print(f"  Reuse with: ontograph ingest ... --schema {save_schema_path}")

        output = Path(args.output)
        kg = graph.to_kg()
        kg.schema_name = "induced"
        kg.documents = [str(f) for f in files]
        output.write_text(kg.to_json(), encoding="utf-8")
        console.print(f"\n[green]Graph saved to {output}[/green]")
        _print_summary(graph)
        return

    # --- Normal mode ---
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
    from ontograph.export import export_graphml, export_mermaid, export_jsonld, export_obsidian_canvas

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    fmt = args.format
    ext = "canvas" if fmt == "canvas" else fmt
    output = Path(args.output) if args.output else Path(args.input).with_suffix(f".{ext}")

    if fmt == "graphml":
        export_graphml(graph, output)
    elif fmt == "mermaid":
        output.write_text(export_mermaid(graph), encoding="utf-8")
    elif fmt == "jsonld":
        output.write_text(export_jsonld(graph), encoding="utf-8")
    elif fmt == "canvas":
        output.write_text(export_obsidian_canvas(graph), encoding="utf-8")
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        return

    console.print(f"[green]Exported to {output}[/green]")


def cmd_extract(args):
    """Extract a neighborhood subgraph around a root entity."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    try:
        sub = graph.subgraph_neighborhood(args.root, depth=args.depth)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    output = Path(args.output)
    sub_kg = sub.to_kg()
    sub_kg.schema_name = kg.schema_name
    sub_kg.documents = kg.documents
    output.write_text(sub_kg.to_json(), encoding="utf-8")
    console.print(
        f"[green]Subgraph ({sub.n_entities} entities, {sub.n_relations} relations) "
        f"saved to {output}[/green]"
    )
    _print_summary(sub)


def cmd_mcp_sync(args):
    """Sync a knowledge graph to MCP memory."""
    from ontograph.models import KnowledgeGraph
    from ontograph.graph import OntologyGraph
    from ontograph.mcp_bridge import export_to_mcp

    kg = KnowledgeGraph.from_json(Path(args.input).read_text(encoding="utf-8"))
    graph = OntologyGraph.from_kg(kg)

    n = export_to_mcp(graph, prefix=args.prefix, append=not args.overwrite)
    console.print(f"[green]Synced {n} records to MCP memory (prefix={args.prefix})[/green]")


def cmd_view(args):
    """Generate a self-contained Cytoscape.js HTML viewer and open in browser."""
    import webbrowser

    graph_path = Path(args.input)
    graph_data = graph_path.read_text(encoding="utf-8")

    template_path = Path(__file__).parent / "dashboard" / "cytoscape_viewer.html"
    template = template_path.read_text(encoding="utf-8")

    # Inject graph data into the HTML template
    html = template.replace(
        '<script id="graph-data" type="application/json"></script>',
        f'<script id="graph-data" type="application/json">{graph_data}</script>',
    )

    output = Path(args.output) if args.output else graph_path.with_suffix(".html")
    output.write_text(html, encoding="utf-8")
    console.print(f"[green]Viewer saved to {output}[/green]")

    if not args.no_open:
        webbrowser.open(str(output.resolve()))


def cmd_serve(args):
    """Launch the Streamlit dashboard."""
    import os
    import subprocess

    if getattr(args, "graph", None):
        os.environ["ONTOGRAPH_GRAPH"] = str(Path(args.graph).resolve())
    if getattr(args, "schema", None):
        os.environ["ONTOGRAPH_SCHEMA"] = args.schema

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(args.port)])


def cmd_causal_ingest(args):
    """Extract causal claims from document(s) using LLM."""
    from ontograph.parsers import parse_document
    from ontograph.causal_extractor import causal_extract_from_document
    from ontograph.causal_models import CausalGraph

    path = Path(args.input)
    if path.is_dir():
        files = (list(path.glob("**/*.md")) + list(path.glob("**/*.html"))
                 + list(path.glob("**/*.pdf")) + list(path.glob("**/*.txt")))
    else:
        files = [path]

    client = _make_llm_client(args)
    console.print(
        f"[bold]ontograph causal ingest[/bold] | backend={args.llm_backend} | {len(files)} file(s)"
    )

    all_graphs = []
    for f in files:
        console.print(f"  Parsing: {f.name}")
        doc = parse_document(f)
        cg = causal_extract_from_document(doc, client=client)
        all_graphs.append(cg)
        console.print(f"    -> {len(cg.claims)} causal claims")

    # Merge all into one graph
    merged = CausalGraph()
    for cg in all_graphs:
        for claim in cg.claims.values():
            merged.add_claim(claim)

    output = Path(args.output)
    output.write_text(merged.to_json(), encoding="utf-8")
    console.print(f"\n[green]Causal graph saved to {output}[/green]")
    _print_causal_summary(merged)


def cmd_causal_add(args):
    """Manually add a causal claim to an existing causal graph."""
    from ontograph.causal_models import (
        CausalGraph, CausalClaim, CausalMechanism, make_claim_id,
    )
    from ontograph.causal_scoring import compute_confidence

    output = Path(args.output)

    # Load existing graph or create new
    if output.exists():
        cg = CausalGraph.from_json(output.read_text(encoding="utf-8"))
    else:
        cg = CausalGraph()

    mechanism = CausalMechanism(
        name=args.mechanism,
        description=args.note or "",
        direction=args.direction,
    )

    claim_id = make_claim_id(args.source, args.target, args.evidence_type, args.note or "")
    claim = CausalClaim(
        id=claim_id,
        source=args.source,
        target=args.target,
        mechanisms=[mechanism],
        evidence_type=args.evidence_type,
        claim_assertiveness=args.assertiveness,
        net_direction=args.direction,
        extracted_by="manual",
        claim_text=args.note or "",
    )
    compute_confidence(claim)
    cg.add_claim(claim)

    output.write_text(cg.to_json(), encoding="utf-8")
    console.print(
        f"[green]Added claim: {args.source} -> {args.target} "
        f"({args.direction}, {claim.strength}, conf={claim.confidence:.2f})[/green]"
    )
    console.print(f"  Saved to {output} ({len(cg.claims)} total claims)")


def _print_causal_summary(cg):
    from ontograph.causal_models import CausalGraph
    table = Table(title="Causal Graph Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Claims", str(len(cg.claims)))
    table.add_row("Entities", str(len(cg.entities)))

    # Strength distribution
    strength_counts = {}
    for claim in cg.claims.values():
        strength_counts[claim.strength] = strength_counts.get(claim.strength, 0) + 1
    for s, count in sorted(strength_counts.items()):
        table.add_row(f"  {s}", str(count))

    # Evidence type distribution
    etype_counts = {}
    for claim in cg.claims.values():
        etype_counts[claim.evidence_type] = etype_counts.get(claim.evidence_type, 0) + 1
    for et, count in sorted(etype_counts.items(), key=lambda x: -x[1]):
        table.add_row(f"  [{et}]", str(count))

    console.print(table)


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
                          choices=["anthropic", "openai", "ollama", "claude-code"],
                          help="LLM backend (default: anthropic). claude-code uses Claude Code CLI pipe mode (Max subscription, no API credits)")
    p_ingest.add_argument("--llm-model", default=None, help="Override model name")
    p_ingest.add_argument("--batch-out", default=None, metavar="PATH",
                          help="Generate extraction prompts JSONL (offline batch mode)")
    p_ingest.add_argument("--batch-in", default=None, metavar="PATH",
                          help="Import extraction responses JSONL (offline batch mode)")
    p_ingest.add_argument("--schema-free", action="store_true",
                          help="Schema-free mode: LLM discovers types inductively, "
                               "then consolidates into a minimal vocabulary (ignores --schema)")
    p_ingest.add_argument("--save-schema", default=None, metavar="PATH",
                          help="Save the induced schema to a YAML file for reuse (schema-free mode only)")

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
    p_export.add_argument("--format", "-f", choices=["graphml", "mermaid", "jsonld", "canvas"], required=True)
    p_export.add_argument("--output", "-o", help="Output path")

    # mcp-sync
    p_mcp = sub.add_parser("mcp-sync", help="Sync graph to MCP memory")
    p_mcp.add_argument("input", help="Graph JSON path")
    p_mcp.add_argument("--prefix", default="ontograph", help="Namespace prefix")
    p_mcp.add_argument("--overwrite", action="store_true", help="Overwrite instead of append")

    # view (Cytoscape.js HTML)
    p_view = sub.add_parser("view", help="Generate interactive Cytoscape.js HTML viewer")
    p_view.add_argument("input", help="Graph JSON path")
    p_view.add_argument("--output", "-o", help="Output HTML path (default: same name as input)")
    p_view.add_argument("--no-open", action="store_true", help="Don't open in browser")

    # extract
    p_extract = sub.add_parser("extract", help="Extract neighborhood subgraph around a root entity")
    p_extract.add_argument("input", help="Graph JSON path")
    p_extract.add_argument("--root", "-r", required=True, help="Root entity name")
    p_extract.add_argument("--depth", "-d", type=int, default=2, help="Hop depth (default: 2)")
    p_extract.add_argument("--output", "-o", default="subgraph.json", help="Output JSON path")

    # serve
    p_serve = sub.add_parser("serve", help="Launch Streamlit dashboard")
    p_serve.add_argument("--port", type=int, default=8501)
    p_serve.add_argument("--graph", default=None, help="Auto-load a graph JSON file on startup")
    p_serve.add_argument("--schema", default=None, help="Schema to use with --graph (default: base)")

    # causal (nested subparser)
    p_causal = sub.add_parser("causal", help="Causal graph extraction and analysis")
    causal_sub = p_causal.add_subparsers(dest="causal_command")

    # causal ingest
    p_ci = causal_sub.add_parser("ingest", help="Extract causal claims from document(s)")
    p_ci.add_argument("input", help="Document or directory path")
    p_ci.add_argument("--output", "-o", default="causal_graph.json", help="Output JSON path")
    p_ci.add_argument("--llm-backend", default="anthropic",
                      choices=["anthropic", "openai", "ollama", "claude-code"],
                      help="LLM backend (default: anthropic)")
    p_ci.add_argument("--llm-model", default=None, help="Override model name")

    # causal add
    p_ca = causal_sub.add_parser("add", help="Manually add a causal claim")
    p_ca.add_argument("--source", required=True, help="Cause entity name")
    p_ca.add_argument("--target", required=True, help="Effect entity name")
    p_ca.add_argument("--mechanism", required=True, help="Mechanism name (snake_case)")
    p_ca.add_argument("--evidence-type", default="narrative",
                      help="Evidence type (e.g. accounting_identity, meta_analysis, narrative)")
    p_ca.add_argument("--direction", default="positive",
                      choices=["positive", "negative", "ambiguous"],
                      help="Direction of effect (default: positive)")
    p_ca.add_argument("--assertiveness", default="moderate",
                      choices=["definitional", "strong", "moderate", "hedged"],
                      help="Claim assertiveness (default: moderate)")
    p_ca.add_argument("--note", default=None, help="Description or verbatim claim text")
    p_ca.add_argument("--output", "-o", default="causal_graph.json", help="Output JSON path")

    args = parser.parse_args()
    commands = {
        "ingest": cmd_ingest,
        "merge": cmd_merge,
        "analyze": cmd_analyze,
        "export": cmd_export,
        "extract": cmd_extract,
        "view": cmd_view,
        "mcp-sync": cmd_mcp_sync,
        "serve": cmd_serve,
    }

    if args.command == "causal":
        causal_commands = {
            "ingest": cmd_causal_ingest,
            "add": cmd_causal_add,
        }
        if args.causal_command in causal_commands:
            causal_commands[args.causal_command](args)
        else:
            p_causal.print_help()
    elif args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
