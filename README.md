# ontograph

**Extract ontological knowledge graphs from academic papers and documents.**

Feed ontograph your research papers. It reads them, identifies the concepts, agents, mechanisms, and constraints described in the text, maps the relationships between them, and produces an interactive knowledge graph you can explore, query, and export.

```
Papers (PDF/MD/HTML)  -->  Entity Extraction  -->  Knowledge Graph  -->  Dashboard
                               |                       |
                        Ontology Schema          Cycles, centrality,
                        (economics, biology,     cascade depth,
                         engineering, custom)     export to GraphML/Mermaid/JSON-LD
```

---

## Why

- **Literature reviews are slow.** Reading 50 papers and mentally mapping how models relate to each other takes weeks. ontograph does the structural mapping in minutes.
- **No tool does this.** Microsoft GraphRAG produces community summaries but no typed ontology. Palantir Ontology is proprietary. SciKGTeX requires author annotations. ontograph extracts typed knowledge graphs from unstructured text, for any domain.
- **Provenance matters.** Every entity and relation links back to the exact paper, section, and passage it was extracted from. No hallucinated structure.

---

## Quick Start

```bash
# Install
git clone https://github.com/murzua7/ontograph.git
cd ontograph
uv sync

# Extract a knowledge graph (heuristic mode -- instant, no API needed)
uv run ontograph ingest paper.md --schema economics -o graph.json

# Extract with LLM (much higher quality)
uv run ontograph ingest paper.pdf --schema economics --llm -o graph.json

# Analyze the graph
uv run ontograph analyze graph.json --all

# Launch interactive dashboard
uv run ontograph serve
```

---

## Features

### Domain-Agnostic Ontology Schemas

Define entity types and relation types in YAML. Ships with four presets:

| Schema | Entity Types | Relation Types | For |
|--------|-------------|----------------|-----|
| `economics` | agent, mechanism, state_variable, market, constraint, parameter, theory, shock | feeds_into, constrains, propagates_losses_to, creates_feedback_with, causes, triggers, amplifies, dampens | Macro-financial models, policy analysis |
| `biology` | molecule, cell, process, gene, organism, disease, phenotype, technique | encodes, regulates, activates, inhibits, binds_to, expressed_in, treats | Molecular biology, drug discovery |
| `engineering` | component, system, signal, property, requirement, failure_mode, material, standard | contains, connects_to, sends_signal, controls, causes_failure, complies_with | Systems engineering, control theory |
| `base` | concept, agent, process, property, constraint | relates_to, causes, part_of, constrains, measures | Any domain |

Custom schemas are just YAML files:

```yaml
name: my_domain
entity_types:
  concept:
    description: "A key concept"
    color: "#4CAF50"
relation_types:
  causes:
    description: "Causal relationship"
```

```bash
uv run ontograph ingest paper.md --schema my_schema.yaml
```

### Two Extraction Modes

**Heuristic** (default) -- Regex-based pattern matching. Instant, no API calls, catches ~30-50% of entities. Good for quick exploration.

**LLM-powered** (`--llm`) -- Two-pass architecture for high precision:
1. **Pass 1**: Extract entities from each section, grounded on schema types
2. **Pass 2**: Extract relations between *known entities*, avoiding hallucinated references

Supports Anthropic (Claude), OpenAI, and Ollama (local, free):

```bash
uv run ontograph ingest paper.pdf --schema economics --llm                          # Anthropic
uv run ontograph ingest paper.pdf --schema biology --llm --llm-backend ollama       # Local
uv run ontograph ingest paper.pdf --schema engineering --llm --llm-backend openai   # OpenAI
```

### Multi-Document Merge with Entity Resolution

Feed multiple papers, then merge with automatic deduplication:

```bash
uv run ontograph ingest paper1.pdf --schema economics -o g1.json
uv run ontograph ingest paper2.pdf --schema economics -o g2.json
uv run ontograph ingest paper3.pdf --schema economics -o g3.json

uv run ontograph merge g1.json g2.json g3.json -o merged.json --threshold 0.85
```

Entity resolution works in three stages:
1. **Exact match** -- same name + same type
2. **Alias match** -- entity name appears in another's alias list
3. **Fuzzy match** -- Levenshtein similarity > threshold, same type

### Graph Analysis

```bash
uv run ontograph analyze merged.json --all
```

- **Cycle detection** -- feedback loops (e.g., firm default -> bank loss -> credit contraction -> firm default)
- **Cascade depth** -- longest contagion path through `propagates_losses_to` edges
- **Degree centrality** -- most connected concepts
- **Connected components** -- isolated subgraphs

### Interactive Dashboard

```bash
uv run ontograph serve
```

Five tabs:
- **Graph Explorer** -- Force-directed visualization (Cytoscape.js). Nodes colored by entity type. Click to see observations, provenance, and connections.
- **Entities** -- Searchable, sortable table with type filters.
- **Relations** -- Filterable relation table.
- **Analysis** -- Cycle detection, centrality ranking, type distribution charts.
- **Export** -- One-click download as JSON, GraphML, Mermaid, or JSON-LD.

### Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| JSON | Portable, re-importable | `ontograph export graph.json -f json` |
| GraphML | Gephi, yEd, Cytoscape desktop | `ontograph export graph.json -f graphml` |
| Mermaid | Embed in Markdown/docs | `ontograph export graph.json -f mermaid` |
| JSON-LD | Linked data / semantic web | `ontograph export graph.json -f jsonld` |

### MCP Memory Bridge

Sync extracted knowledge into Claude Code's memory for conversational querying:

```bash
uv run ontograph mcp-sync graph.json --prefix thesis
```

Then in Claude Code: *"What feeds into bank default in my thesis graph?"*

---

## Architecture

```
ontograph/
  models.py          # Entity, Relation, KnowledgeGraph, Provenance dataclasses
  schema.py          # YAML ontology loader + validation
  parsers/           # Markdown, HTML (BeautifulSoup), PDF (pdfplumber)
  extractor.py       # Heuristic extraction (regex patterns)
  llm_client.py      # Multi-backend LLM (Anthropic, OpenAI, Ollama, mock)
  llm_extractor.py   # Two-pass LLM extraction with schema grounding
  resolver.py        # Entity resolution (exact + alias + fuzzy)
  graph.py           # NetworkX wrapper with analysis methods
  export.py          # JSON, GraphML, Mermaid, JSON-LD exporters
  mcp_bridge.py      # Claude Code MCP memory sync
  dashboard/app.py   # Streamlit interactive visualization
  cli.py             # CLI: ingest, merge, analyze, export, mcp-sync, serve
schemas/
  economics.yaml     # 8 entity types, 12 relation types
  biology.yaml       # 8 entity types, 11 relation types
  engineering.yaml   # 8 entity types, 9 relation types
  base.yaml          # 5 entity types, 5 relation types
```

---

## Full CLI Reference

```
ontograph ingest <file_or_dir> --schema <name> [--llm] [--llm-backend anthropic|openai|ollama] [-o graph.json]
ontograph merge <g1.json> <g2.json> ... [-o merged.json] [--threshold 0.85]
ontograph analyze <graph.json> [--cycles] [--centrality] [--cascade-depth] [--all]
ontograph export <graph.json> -f graphml|mermaid|jsonld [-o output]
ontograph mcp-sync <graph.json> [--prefix ontograph] [--overwrite]
ontograph serve [--port 8501]
```

---

## Install

```bash
git clone https://github.com/murzua7/ontograph.git
cd ontograph
uv sync                          # core
uv sync --extra pdf              # + PDF parsing
uv sync --extra llm              # + LLM backends
uv sync --extra all              # everything
```

Requires Python 3.10+.

---

## License

MIT
