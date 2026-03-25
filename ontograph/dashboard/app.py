"""Streamlit dashboard for ontograph knowledge graphs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure ontograph is importable when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ontograph.models import KnowledgeGraph, Entity
from ontograph.graph import OntologyGraph
from ontograph.schema import load_schema, OntologySchema, SCHEMAS_DIR
from ontograph.export import export_mermaid, export_jsonld

st.set_page_config(page_title="ontograph", page_icon="🕸", layout="wide")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_graph() -> OntologyGraph | None:
    return st.session_state.get("graph")


def _get_schema() -> OntologySchema | None:
    return st.session_state.get("schema")


def _get_kg() -> KnowledgeGraph | None:
    return st.session_state.get("kg")


def _load_graph_from_json(data: str, schema: OntologySchema) -> None:
    kg = KnowledgeGraph.from_json(data)
    graph = OntologyGraph.from_kg(kg, schema=schema)
    st.session_state["kg"] = kg
    st.session_state["graph"] = graph
    st.session_state["schema"] = schema


# ---------------------------------------------------------------------------
# Sidebar: Upload & Schema
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("ontograph")
    st.sidebar.caption("Knowledge graph extraction from documents")

    # Schema selector
    schema_files = sorted(SCHEMAS_DIR.glob("*.yaml"))
    schema_names = [f.stem for f in schema_files]
    selected = st.sidebar.selectbox("Ontology Schema", schema_names, index=0)

    if selected:
        schema = load_schema(selected)
        st.session_state["schema"] = schema

    st.sidebar.divider()

    # File upload: JSON graph
    st.sidebar.subheader("Load Graph")
    uploaded = st.sidebar.file_uploader("Upload graph.json", type=["json"])
    if uploaded:
        data = uploaded.read().decode("utf-8")
        schema = st.session_state.get("schema") or load_schema("base")
        _load_graph_from_json(data, schema)
        st.sidebar.success(f"Loaded {_get_graph().n_entities} entities, {_get_graph().n_relations} relations")

    st.sidebar.divider()

    # Document upload for extraction
    st.sidebar.subheader("Extract from Document")
    doc_file = st.sidebar.file_uploader("Upload document", type=["md", "txt", "html"])
    if doc_file and st.sidebar.button("Extract (heuristic)"):
        _extract_from_upload(doc_file)

    # Graph stats
    graph = _get_graph()
    if graph:
        st.sidebar.divider()
        st.sidebar.subheader("Graph Stats")
        summary = graph.summary()
        st.sidebar.metric("Entities", summary["n_entities"])
        st.sidebar.metric("Relations", summary["n_relations"])
        st.sidebar.metric("Cycles", summary["n_cycles"])
        st.sidebar.metric("Components", summary["n_components"])


def _extract_from_upload(doc_file):
    """Extract entities/relations from uploaded document."""
    import tempfile
    from ontograph.parsers import parse_document
    from ontograph.extractor import extract_from_document

    schema = st.session_state.get("schema") or load_schema("base")

    # Write to temp file to parse
    suffix = "." + (doc_file.name.split(".")[-1] if "." in doc_file.name else "txt")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as f:
        f.write(doc_file.read())
        tmp_path = f.name

    doc = parse_document(tmp_path)
    entities, relations = extract_from_document(doc, schema)

    graph = _get_graph() or OntologyGraph(schema=schema)
    for e in entities:
        graph.add_entity(e)
    for r in relations:
        graph.add_relation(r)

    kg = graph.to_kg()
    kg.documents.append(doc_file.name)

    st.session_state["graph"] = graph
    st.session_state["kg"] = kg
    st.sidebar.success(f"Extracted {len(entities)} entities, {len(relations)} relations")


# ---------------------------------------------------------------------------
# Tab 1: Graph Explorer
# ---------------------------------------------------------------------------

def render_graph_explorer():
    graph = _get_graph()
    schema = _get_schema()
    if not graph or graph.n_entities == 0:
        st.info("Upload a graph.json or extract from a document to get started.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_types = sorted(set(e.entity_type for e in graph._entities.values()))
        selected_types = st.multiselect("Filter entity types", all_types, default=all_types)
    with col2:
        all_rel_types = sorted(set(
            d.get("relation_type", "unknown") for _, _, d in graph.g.edges(data=True)
        ))
        selected_rels = st.multiselect("Filter relation types", all_rel_types, default=all_rel_types)

    # Build agraph nodes and edges
    from streamlit_agraph import agraph, Node, Edge, Config

    # Color map from schema
    color_map = {}
    if schema:
        for etype, edef in schema.entity_types.items():
            color_map[etype] = edef.color

    nodes = []
    for name, entity in graph._entities.items():
        if entity.entity_type not in selected_types:
            continue
        color = color_map.get(entity.entity_type, "#607D8B")
        nodes.append(Node(
            id=name,
            label=name,
            size=20 + 5 * graph.g.degree(name),
            color=color,
            title=f"{entity.entity_type}\n" + "\n".join(entity.observations[:3]),
        ))

    node_ids = {n.id for n in nodes}
    edges = []
    for u, v, data in graph.g.edges(data=True):
        rtype = data.get("relation_type", "unknown")
        if rtype not in selected_rels:
            continue
        if u not in node_ids or v not in node_ids:
            continue
        edges.append(Edge(
            source=u,
            target=v,
            label=rtype,
            color="#888888",
        ))

    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
    )

    if nodes:
        selected_node = agraph(nodes=nodes, edges=edges, config=config)

        # Detail panel for selected node
        if selected_node and selected_node in graph._entities:
            entity = graph._entities[selected_node]
            st.divider()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.subheader(entity.name)
                st.caption(f"Type: {entity.entity_type}")
                if entity.aliases:
                    st.write(f"**Aliases:** {', '.join(entity.aliases)}")
                st.write("**Observations:**")
                for obs in entity.observations:
                    st.write(f"- {obs}")
            with col_b:
                st.write("**Provenance:**")
                for prov in entity.provenance:
                    st.write(f"- *{prov.document}* [{prov.section}]")
                    if prov.passage:
                        st.caption(f'"{prov.passage[:200]}"')

                # Connected entities
                neighbors = graph.neighbors(entity.name)
                if neighbors:
                    st.write(f"**Connected to:** {', '.join(neighbors[:10])}")
    else:
        st.warning("No entities match the current filters.")


# ---------------------------------------------------------------------------
# Tab 2: Entity Table
# ---------------------------------------------------------------------------

def render_entity_table():
    graph = _get_graph()
    if not graph or graph.n_entities == 0:
        st.info("No graph loaded.")
        return

    import pandas as pd
    rows = []
    for name, entity in graph._entities.items():
        rows.append({
            "Name": name,
            "Type": entity.entity_type,
            "Aliases": ", ".join(entity.aliases) if entity.aliases else "",
            "Observations": len(entity.observations),
            "Sources": len(entity.provenance),
            "Degree": graph.g.degree(name),
        })

    df = pd.DataFrame(rows)

    # Filters
    types = st.multiselect("Filter by type", sorted(df["Type"].unique()), default=sorted(df["Type"].unique()))
    df = df[df["Type"].isin(types)]

    search = st.text_input("Search entities")
    if search:
        mask = df["Name"].str.contains(search, case=False, na=False)
        df = df[mask]

    st.dataframe(df.sort_values("Degree", ascending=False), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Relation Table
# ---------------------------------------------------------------------------

def render_relation_table():
    graph = _get_graph()
    if not graph or graph.n_relations == 0:
        st.info("No relations in graph.")
        return

    import pandas as pd
    rows = []
    for u, v, data in graph.g.edges(data=True):
        rows.append({
            "Source": u,
            "Target": v,
            "Type": data.get("relation_type", "unknown"),
            "Weight": data.get("weight", 1.0),
        })

    df = pd.DataFrame(rows)
    types = st.multiselect("Filter by relation type", sorted(df["Type"].unique()), default=sorted(df["Type"].unique()), key="rel_filter")
    df = df[df["Type"].isin(types)]

    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: Analysis
# ---------------------------------------------------------------------------

def render_analysis():
    graph = _get_graph()
    if not graph or graph.n_entities == 0:
        st.info("No graph loaded.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Cycles")
        cycles = graph.find_cycles()
        if cycles:
            for i, cycle in enumerate(cycles[:10]):
                st.write(f"{i+1}. {' -> '.join(cycle)} -> {cycle[0]}")
        else:
            st.write("No cycles detected.")

    with col2:
        st.subheader("Centrality (top 10)")
        centrality = graph.degree_centrality()
        top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, val in top:
            st.write(f"**{name}**: {val:.3f}")

    with col3:
        st.subheader("Components")
        components = graph.connected_components()
        st.metric("Connected Components", len(components))
        depth = graph.cascade_depth()
        st.metric("Max Cascade Depth", depth)

    # Entity type distribution chart
    st.divider()
    st.subheader("Entity Type Distribution")
    import plotly.express as px
    import pandas as pd

    summary = graph.summary()
    type_data = pd.DataFrame([
        {"Type": k, "Count": v} for k, v in summary["entity_types"].items()
    ])
    if not type_data.empty:
        schema = _get_schema()
        colors = {}
        if schema:
            colors = {k: v.color for k, v in schema.entity_types.items()}
        fig = px.bar(type_data, x="Type", y="Count", color="Type",
                     color_discrete_map=colors if colors else None)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    if summary.get("relation_types"):
        st.subheader("Relation Type Distribution")
        rel_data = pd.DataFrame([
            {"Type": k, "Count": v} for k, v in summary["relation_types"].items()
        ])
        fig2 = px.bar(rel_data, x="Type", y="Count")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Export
# ---------------------------------------------------------------------------

def render_export():
    graph = _get_graph()
    kg = _get_kg()
    if not graph or graph.n_entities == 0:
        st.info("No graph loaded.")
        return

    st.subheader("Download Graph")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if kg:
            st.download_button(
                "JSON", kg.to_json(), file_name="ontograph.json", mime="application/json"
            )

    with col2:
        mermaid = export_mermaid(graph)
        st.download_button("Mermaid", mermaid, file_name="ontograph.mmd", mime="text/plain")

    with col3:
        jsonld = export_jsonld(graph)
        st.download_button("JSON-LD", jsonld, file_name="ontograph.jsonld", mime="application/ld+json")

    with col4:
        import tempfile
        from ontograph.export import export_graphml
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            export_graphml(graph, f.name)
            graphml_data = Path(f.name).read_text(encoding="utf-8")
        st.download_button("GraphML", graphml_data, file_name="ontograph.graphml", mime="application/xml")

    # Mermaid preview
    st.divider()
    st.subheader("Mermaid Preview")
    st.code(mermaid, language="mermaid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    tab_graph, tab_entities, tab_relations, tab_analysis, tab_export = st.tabs([
        "Graph Explorer", "Entities", "Relations", "Analysis", "Export"
    ])

    with tab_graph:
        render_graph_explorer()
    with tab_entities:
        render_entity_table()
    with tab_relations:
        render_relation_table()
    with tab_analysis:
        render_analysis()
    with tab_export:
        render_export()


if __name__ == "__main__":
    main()
else:
    main()
