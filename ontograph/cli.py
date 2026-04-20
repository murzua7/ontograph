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
    # --dual-extract requires --ast-repo — surface the mis-config fast.
    if getattr(args, "dual_extract", False) and not getattr(args, "ast_repo", None):
        console.print("[red]--dual-extract requires --ast-repo[/red]")
        return 1

    mode_label = f"llm ({args.llm_backend})" if args.llm else "heuristic"
    ast_label = " + ast" if getattr(args, "ast_repo", None) else ""
    dual_label = " (dual-extract)" if getattr(args, "dual_extract", False) else ""
    console.print(
        f"[bold]ontograph[/bold] | schema={schema.name} | "
        f"mode={mode_label}{ast_label}{dual_label} | {len(files)} file(s)"
    )

    # Accumulate text-side (LLM or heuristic) extractions across all files.
    text_entities: list = []
    text_relations: list = []
    for f in files:
        console.print(f"  Parsing: {f.name}")
        doc = parse_document(f)

        if args.llm:
            client = _make_llm_client(args)
            entities, relations = _ingest_llm_extract(doc, schema, client=client)
        else:
            entities, relations = extract_from_document(doc, schema)

        text_entities.extend(entities)
        text_relations.extend(relations)
        console.print(f"    -> {len(entities)} entities, {len(relations)} relations")

    # Optional AST-side extraction (Phase D / spec §5).
    ast_entities: list = []
    ast_relations: list = []
    if getattr(args, "ast_repo", None):
        from ontograph.ast_extractor import extract_from_repo
        ast_entities, ast_relations = extract_from_repo(Path(args.ast_repo), schema)
        console.print(f"  AST: {len(ast_entities)} entities, {len(ast_relations)} relations")

    graph = OntologyGraph(schema=schema)

    if getattr(args, "dual_extract", False):
        # Agreement-only KG: both sides had to see the entity/relation.
        from ontograph.merge import merge_extractions
        merged_kg, merge_report = merge_extractions(
            text_entities, text_relations,
            ast_entities, ast_relations,
        )
        for e in merged_kg.entities.values():
            graph.add_entity(e)
        for r in merged_kg.relations:
            graph.add_relation(r)
        console.print(
            f"  merge: agreement={len(merge_report.agreement_entities)} entities / "
            f"{len(merge_report.agreement_relations)} relations"
        )
    else:
        # Union mode — text side supplies names/semantics, AST side adds grounding.
        for e in text_entities:
            graph.add_entity(e)
        for r in text_relations:
            graph.add_relation(r)
        for e in ast_entities:
            graph.add_entity(e)
        for r in ast_relations:
            graph.add_relation(r)

    # Output the KG.
    output = Path(args.output)
    kg = graph.to_kg()
    kg.documents = [str(f) for f in files]
    output.write_text(kg.to_json(), encoding="utf-8")
    console.print(f"\n[green]Graph saved to {output}[/green]")

    # Optional quality report (spec §5 `--quality-report`).
    report_flag = getattr(args, "quality_report", None)
    if report_flag is not None or getattr(args, "ast_repo", None):
        from ontograph.quality import compute_quality_report
        report = compute_quality_report(
            kg, schema,
            ast_entities=ast_entities or None,
        )
        # Only persist when the flag is explicitly set; ast_repo alone just
        # computes the six metrics for the console summary.
        if report_flag is not None:
            report_path = Path(report_flag)
            report_path.write_text(report.to_json(), encoding="utf-8")
            console.print(f"[green]Quality report saved to {report_path}[/green]")
        if not report.gates_passed:
            console.print("[yellow]quality gate failed[/yellow]")
            for f in report.failures:
                console.print(f"  [yellow]·[/yellow] {f}")

    _print_summary(graph)
    return 0


def _ingest_llm_extract(doc, schema, *, client=None):
    """LLM-extraction seam used by `cmd_ingest` in dual-extract mode.

    Kept as a module-level indirection so test suites can monkeypatch it
    without standing up a real LLM backend.
    """
    from ontograph.llm_extractor import llm_extract_from_document
    return llm_extract_from_document(doc, schema, client=client)


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


def _cross_check_llm_extract(doc_path, schema, *, client=None):
    """Run the LLM extractor over a single document and return (entities, relations).

    Separated from cmd_cross_check so tests can monkeypatch this seam
    without standing up a real LLM backend.
    """
    from ontograph.llm_extractor import llm_extract_from_document
    from ontograph.parsers import parse_document

    doc = parse_document(Path(doc_path))
    return llm_extract_from_document(doc, schema, client=client)


def cmd_quality(args) -> int:
    """Run the quality gate on a knowledge graph.

    Returns an exit code: 0 on pass (or --allow-gate-failure), 1 on fail.
    Also writes a JSON report alongside the input graph (or to --output).
    """
    from ontograph.ast_extractor import extract_from_repo
    from ontograph.models import KnowledgeGraph
    from ontograph.quality import compute_quality_report
    from ontograph.schema import load_schema

    schema = load_schema(args.schema)
    kg_path = Path(args.input)
    kg = KnowledgeGraph.from_json(kg_path.read_text(encoding="utf-8"))

    ast_entities = None
    if getattr(args, "ast_repo", None):
        ast_ents, _ = extract_from_repo(Path(args.ast_repo), schema)
        ast_entities = ast_ents

    report = compute_quality_report(kg, schema, ast_entities=ast_entities)

    out_path = Path(args.output) if getattr(args, "output", None) else kg_path.with_suffix(".quality.json")
    out_path.write_text(report.to_json(), encoding="utf-8")

    # Console table for at-a-glance reading.
    table = Table(title="Quality Gate")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Status")

    def _fmt_ratio(r):
        return "n/a" if r is None else f"{r:.3f}"

    def _status(metric: str) -> str:
        hits = [f for f in report.failures if metric in f.lower()]
        return "[red]FAIL[/red]" if hits else "[green]pass[/green]"

    table.add_row("coverage",     _fmt_ratio(report.coverage["ratio"]),     _status("coverage"))
    table.add_row("groundedness", _fmt_ratio(report.groundedness["ratio"]), _status("groundedness"))
    table.add_row("signedness",   _fmt_ratio(report.signedness["ratio"]),   _status("signedness"))
    table.add_row("orphan_rate",  _fmt_ratio(report.orphan_rate["ratio"]),  "—")
    table.add_row("consistency",  _fmt_ratio(report.consistency["ratio"]),  "—")
    table.add_row("cycles (within)", str(report.cycle_count["within_step"]), _status("cycle-within_step"))
    table.add_row("cycles (across)", str(report.cycle_count["across_step"]), _status("cycle-across_step"))
    console.print(table)
    console.print(f"Report → {out_path}")

    if not report.gates_passed and not getattr(args, "allow_gate_failure", False):
        for f in report.failures:
            console.print(f"  [red]·[/red] {f}")
        return 1
    if not report.gates_passed:
        console.print("[yellow]gates failed (--allow-gate-failure)[/yellow]")
    return 0


def cmd_cross_check(args) -> int:
    """Dual-extract (LLM + AST) and emit agreement-gated KG + merge report."""
    from ontograph.ast_extractor import extract_from_repo
    from ontograph.merge import merge_extractions
    from ontograph.schema import load_schema

    schema = load_schema(args.schema)

    client = _make_llm_client(args)
    llm_entities, llm_relations = _cross_check_llm_extract(
        args.input, schema, client=client,
    )
    console.print(f"  LLM: {len(llm_entities)} entities, {len(llm_relations)} relations")

    ast_entities, ast_relations = extract_from_repo(Path(args.ast_repo), schema)
    console.print(f"  AST: {len(ast_entities)} entities, {len(ast_relations)} relations")

    kg, report = merge_extractions(
        llm_entities, llm_relations,
        ast_entities, ast_relations,
    )

    out_kg = Path(args.output)
    out_kg.write_text(kg.to_json(), encoding="utf-8")
    console.print(f"[green]Agreement KG → {out_kg}[/green] "
                  f"({len(kg.entities)} entities, {len(kg.relations)} relations)")

    out_report = Path(args.report) if getattr(args, "report", None) else out_kg.with_suffix(".merge_report.json")
    out_report.write_text(report.to_json(), encoding="utf-8")

    # Bucket sizes summary.
    table = Table(title="Merge Buckets")
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    table.add_row("agreement entities", str(len(report.agreement_entities)))
    table.add_row("llm-only entities",  str(len(report.llm_only_entities)))
    table.add_row("ast-only entities",  str(len(report.ast_only_entities)))
    table.add_row("agreement relations", str(len(report.agreement_relations)))
    table.add_row("llm-only relations",  str(len(report.llm_only_relations)))
    table.add_row("ast-only relations",  str(len(report.ast_only_relations)))
    table.add_row("sign conflicts",      str(len(report.sign_conflicts)))
    console.print(table)
    console.print(f"Report → {out_report}")

    return 0


def cmd_ground(args) -> int:
    """Post-hoc grounding pass (spec §5).

    Walks an existing KG and attaches `CodeAnchor` entries from `--ast-repo`
    (and, optionally, `CitationAnchor` entries from `--citation-bib`) to
    entities that currently lack grounding. Fuzzy-matches entity names to
    AST symbols using `ontograph.merge.default_name_matcher` + 0.85 threshold
    so the grounding predicate stays aligned with merge/coverage.
    """
    from ontograph.ast_extractor import extract_from_repo
    from ontograph.merge import default_name_matcher
    from ontograph.models import CitationAnchor, KnowledgeGraph
    from ontograph.schema import load_schema

    schema = load_schema(args.schema)
    kg_path = Path(args.kg)
    kg = KnowledgeGraph.from_json(kg_path.read_text(encoding="utf-8"))

    ast_entities, _ = extract_from_repo(Path(args.ast_repo), schema)
    # Group AST anchors by symbol — one AST entity may emit multiple anchors
    # (e.g. for an overloaded name in different files).
    ast_by_name: dict[str, list] = {}
    for ae in ast_entities:
        ast_by_name.setdefault(ae.name, []).extend(ae.code_anchors)

    # Optional bib: accepts a simple JSON mapping of {entity_name: {"key":..., "pages":...}}
    # or a list of `{"name":..., "key":..., "pages":...}` records. Richer .bib
    # parsing is deferred — the JSON form covers the pilot case.
    citation_map: dict[str, list[CitationAnchor]] = {}
    bib_path = getattr(args, "citation_bib", None)
    if bib_path:
        raw = json.loads(Path(bib_path).read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else [
            {"name": n, **(v if isinstance(v, dict) else {"key": v})}
            for n, v in raw.items()
        ]
        for e in entries:
            citation_map.setdefault(e["name"], []).append(
                CitationAnchor(key=e["key"], pages=e.get("pages", ""))
            )

    threshold = 0.85
    code_added = 0
    cite_added = 0
    for ent in kg.entities.values():
        existing_codes = {(a.repo, a.path, a.line, a.symbol) for a in ent.code_anchors}
        # Exact match first; fall back to fuzzy if no exact hit.
        candidates = ast_by_name.get(ent.name, [])
        if not candidates:
            for sym, anchors in ast_by_name.items():
                if default_name_matcher(ent.name, sym) >= threshold:
                    candidates = anchors
                    break
        for a in candidates:
            if (a.repo, a.path, a.line, a.symbol) not in existing_codes:
                ent.code_anchors.append(a)
                existing_codes.add((a.repo, a.path, a.line, a.symbol))
                code_added += 1

        existing_cites = {(c.key, c.pages) for c in ent.citation_anchors}
        for c in citation_map.get(ent.name, []):
            if (c.key, c.pages) not in existing_cites:
                ent.citation_anchors.append(c)
                existing_cites.add((c.key, c.pages))
                cite_added += 1

    out_path = kg_path if getattr(args, "in_place", False) else Path(
        getattr(args, "output", None) or kg_path.with_suffix(".grounded.json")
    )
    out_path.write_text(kg.to_json(), encoding="utf-8")
    console.print(
        f"[green]Grounded → {out_path}[/green] "
        f"(+{code_added} code anchors, +{cite_added} citation anchors)"
    )
    return 0


def cmd_diff(args) -> int:
    """Compare two KGs and emit a markdown report of added/removed/changed (spec §5)."""
    from ontograph.models import KnowledgeGraph

    old_kg = KnowledgeGraph.from_json(Path(args.old).read_text(encoding="utf-8"))
    new_kg = KnowledgeGraph.from_json(Path(args.new).read_text(encoding="utf-8"))

    old_names = set(old_kg.entities)
    new_names = set(new_kg.entities)
    added_entities = sorted(new_names - old_names)
    removed_entities = sorted(old_names - new_names)
    kept_entities = sorted(old_names & new_names)

    def _rel_key(r) -> tuple[str, str, str]:
        return (r.source, r.target, r.relation_type)

    old_rels = {_rel_key(r): r for r in old_kg.relations}
    new_rels = {_rel_key(r): r for r in new_kg.relations}
    added_rels = [new_rels[k] for k in sorted(new_rels.keys() - old_rels.keys())]
    removed_rels = [old_rels[k] for k in sorted(old_rels.keys() - new_rels.keys())]

    changed_type: list[tuple[str, str, str]] = []
    for name in kept_entities:
        o = old_kg.entities[name].entity_type
        n = new_kg.entities[name].entity_type
        if o != n:
            changed_type.append((name, o, n))

    lines: list[str] = []
    lines.append(f"# KG diff: `{Path(args.old).name}` → `{Path(args.new).name}`\n")

    if not (added_entities or removed_entities or added_rels or removed_rels or changed_type):
        lines.append("_No changes — the two knowledge graphs are identical at the "
                     "(name, entity_type) × (source, target, relation_type) level._\n")
    else:
        lines.append("## Entities\n")
        lines.append(f"- added: {len(added_entities)}\n")
        for n in added_entities:
            lines.append(f"  - `{n}` ({new_kg.entities[n].entity_type})\n")
        lines.append(f"- removed: {len(removed_entities)}\n")
        for n in removed_entities:
            lines.append(f"  - `{n}` ({old_kg.entities[n].entity_type})\n")
        if changed_type:
            lines.append(f"- changed type: {len(changed_type)}\n")
            for n, o, nn in changed_type:
                lines.append(f"  - `{n}`: {o} → {nn}\n")

        lines.append("\n## Relations\n")
        lines.append(f"- added: {len(added_rels)}\n")
        for r in added_rels:
            lines.append(f"  - `{r.source}` --[{r.relation_type}]--> `{r.target}`\n")
        lines.append(f"- removed: {len(removed_rels)}\n")
        for r in removed_rels:
            lines.append(f"  - `{r.source}` --[{r.relation_type}]--> `{r.target}`\n")

    Path(args.report).write_text("".join(lines), encoding="utf-8")
    console.print(
        f"[green]Diff → {args.report}[/green] "
        f"(entities: +{len(added_entities)}/-{len(removed_entities)} · "
        f"relations: +{len(added_rels)}/-{len(removed_rels)})"
    )
    return 0


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
    # Phase D additions (spec §5)
    p_ingest.add_argument("--ast-repo", default=None, metavar="PATH",
                          help="Repository root to AST-extract alongside text extraction")
    p_ingest.add_argument("--dual-extract", action="store_true",
                          help="Emit only the LLM×AST agreement set (requires --ast-repo)")
    p_ingest.add_argument("--quality-report", default=None, metavar="PATH",
                          help="Write the six-metric quality report to this path")

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

    # quality (Phase D gate)
    p_quality = sub.add_parser("quality", help="Run the discipline-v1 quality gate on a KG")
    p_quality.add_argument("input", help="Graph JSON path")
    p_quality.add_argument("--schema", default="base", help="Schema name or YAML path")
    p_quality.add_argument("--ast-repo", default=None,
                           help="Repository root to AST-extract for coverage metric")
    p_quality.add_argument("--output", "-o", default=None,
                           help="Report JSON path (default: <input>.quality.json)")
    p_quality.add_argument("--allow-gate-failure", action="store_true",
                           help="Exit 0 even if gates fail (report still written)")

    # ground (Phase D post-hoc grounding pass)
    p_ground = sub.add_parser("ground",
                              help="Post-hoc grounding: attach code/citation anchors to a KG")
    p_ground.add_argument("--kg", required=True, help="Input KG JSON path")
    p_ground.add_argument("--ast-repo", required=True, help="Repository root for AST extraction")
    p_ground.add_argument("--citation-bib", default=None, metavar="PATH",
                          help="JSON file mapping entity names to citation keys")
    p_ground.add_argument("--in-place", action="store_true",
                          help="Overwrite the input KG with the grounded version")
    p_ground.add_argument("--output", "-o", default=None,
                          help="Output JSON path (default: <kg>.grounded.json; ignored with --in-place)")
    p_ground.add_argument("--schema", default="base", help="Schema name or YAML path")

    # diff (Phase D KG comparison)
    p_diff = sub.add_parser("diff",
                            help="Compare two knowledge graphs and emit a markdown report")
    p_diff.add_argument("--old", required=True, help="Baseline KG JSON path")
    p_diff.add_argument("--new", required=True, help="Current KG JSON path")
    p_diff.add_argument("--report", required=True, help="Output markdown path")

    # cross-check (Phase D LLM × AST agreement set)
    p_xcheck = sub.add_parser("cross-check",
                              help="Dual-extract LLM+AST and emit agreement-gated KG + report")
    p_xcheck.add_argument("input", help="Document or directory path (prose input for LLM)")
    p_xcheck.add_argument("--ast-repo", required=True, help="Repository root for AST extraction")
    p_xcheck.add_argument("--schema", default="base", help="Schema name or YAML path")
    p_xcheck.add_argument("--output", "-o", default="agreement.json",
                          help="Agreement KG JSON path")
    p_xcheck.add_argument("--report", default=None,
                          help="Merge report JSON path (default: <output>.merge_report.json)")
    p_xcheck.add_argument("--llm-backend", default="anthropic",
                          choices=["anthropic", "openai", "ollama", "claude-code"])
    p_xcheck.add_argument("--llm-model", default=None)

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
        "quality": cmd_quality,
        "cross-check": cmd_cross_check,
        "ground": cmd_ground,
        "diff": cmd_diff,
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
        rc = commands[args.command](args)
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
