# Spec: Native Causal Graph Support for view/analyze/export

**Date**: 2026-04-04
**Status**: planned
**Motivation**: `view`, `analyze`, and `export` only accept KnowledgeGraph JSON (`{entities, relations}`). CausalGraph JSON (`{claims}`) requires manual conversion. This spec adds native support.

---

## Architecture

**Conversion layer** approach: a new `causal_view.py` module converts `CausalGraph -> KnowledgeGraph` with all causal metadata preserved in entity/relation `metadata` dicts. The existing HTML viewer gets a causal mode branch keyed off `schema_name === "causal"`. No template duplication.

---

## Phase 1: Conversion Layer (`ontograph/causal_view.py` -- new file)

### `detect_graph_type(json_str: str) -> Literal["knowledge", "causal"]`
- Checks for `"claims"` key with nested `source`/`target`/`mechanisms` vs `"entities"` key
- Returns the type string

### `causal_graph_to_knowledge_graph(cg: CausalGraph) -> KnowledgeGraph`

**Entity construction** (from unique entity names in `cg._entity_list`):
- `entity_type = "causal_entity"`
- `metadata.causal_role`: computed from graph structure
  - source-only = `"cause"`
  - target-only = `"effect"`
  - both = `"mediator"`
- `metadata.out_degree`, `metadata.in_degree`: claim counts
- `metadata.in_feedback_loop`: bool, from NetworkX cycle detection on the directed claim graph

**Relation construction** (from each `CausalClaim`):
- `source = claim.source`
- `target = claim.target`
- `relation_type = claim.net_direction` (e.g., "positive", "negative", "ambiguous")
- `weight = claim.confidence`
- `metadata` dict containing: `confidence`, `strength`, `evidence_type`, `net_direction`, `causal_type`, `mechanisms` (serialized list of dicts), `claim_assertiveness`, `direction_contested`, `reversibility`, `claim_text`, `claim_id`

Set `kg.schema_name = "causal"` -- the discriminator for auto-detection and viewer mode switching.

### `_load_graph_auto(path: Path) -> tuple[str, KnowledgeGraph | CausalGraph]`
- Reads JSON, calls `detect_graph_type`, returns `("knowledge", KG)` or `("causal", CG)`

---

## Phase 2: CLI Wiring (`ontograph/cli.py`)

### Auto-detection in existing commands

Modify `cmd_view`, `cmd_analyze`, `cmd_export` to detect CausalGraph JSON and route accordingly. If causal, convert via `causal_graph_to_knowledge_graph()` (for view) or delegate to `cmd_causal_*` (for analyze/export).

### New subcommands under `causal`

**`causal view`** -- same args as `view` (input, --output, --no-open)
- Loads CausalGraph, converts to KG, injects into viewer template
- Identical to auto-detected path but explicit

**`causal analyze`** -- args: input, --cycles, --centrality, --cascade-depth, --conflicts, --all
- `--cycles`: Find feedback loops via `OntologyGraph.find_cycles()`, annotate with direction per edge
- `--centrality`: Degree centrality with causal_role column
- `--cascade-depth`: Longest path (DAG) or longest cycle (cyclic)
- `--conflicts`: Claims with `direction_contested=True` or non-empty `contradicted_by`

**`causal export`** -- args: input, --format (graphml|mermaid|canvas), --output
- Delegates to format-specific functions in `export.py`

### Dispatch wiring

Add to `causal_commands` dict:
```python
causal_commands = {
    "ingest": cmd_causal_ingest,
    "add": cmd_causal_add,
    "view": cmd_causal_view,
    "analyze": cmd_causal_analyze,
    "export": cmd_causal_export,
}
```

---

## Phase 3: Causal Exports (`ontograph/export.py`)

### `export_causal_mermaid(cg: CausalGraph) -> str`
- Direction-specific arrows: `-->` positive, `-.->` ambiguous, `--x` negative
- Edge labels: `|mechanism_name (conf=0.74)|`
- Node styling via `classDef` for cause/effect/mediator roles

### `export_causal_graphml(cg: CausalGraph, path: Path) -> None`
- NetworkX DiGraph with node attributes: `causal_role`, `in_feedback_loop`
- Edge attributes: `confidence`, `evidence_type`, `net_direction`, `strength`, `mechanisms` (JSON string), `causal_type`

### `export_causal_canvas(cg: CausalGraph) -> str`
- Obsidian Canvas JSON with direction indicators in node text
- Edge labels: direction + evidence_type

---

## Phase 4: HTML Viewer Causal Mode (`ontograph/dashboard/cytoscape_viewer.html`)

Detect causal mode via `data.schema_name === "causal"` in `buildCytoscapeElements()`.

### Visual encoding (causal mode only)

**Edges:**
| Property | Mapping |
|----------|---------|
| Color | positive=#4CAF50, negative=#F44336, ambiguous=#8b949e, conflicted=#FF9800 |
| Opacity | `0.3 + 0.7 * confidence` |
| Line style | accounting_identity/rct/meta_analysis/natural_experiment = solid; iv/did/rdd/event_study/panel = dashed; narrative/model_simulation/granger = dotted |
| Width | correlation=1px, weak_causal=2px, strong_causal=4px |

**Nodes:**
| Property | Mapping |
|----------|---------|
| Color | cause=#2196F3, effect=#FF9800, mediator=#9C27B0 |
| Glow | Pulsing ring if `in_feedback_loop === true` |

### UI changes (causal mode only)
- Stats panel: show "Claims" not "Relations", add "Avg Confidence"
- Detail panel: show mechanisms list, evidence type, confidence bar, direction, claim text
- Legend: causal-specific (direction colors, line styles, confidence scale)

---

## Phase 5: Tests (`tests/test_causal_view.py` -- new file)

1. `test_causal_to_kg_entities` -- all entities appear with correct causal_role
2. `test_causal_to_kg_relations` -- each claim becomes a Relation with full metadata
3. `test_causal_to_kg_metadata_fields` -- confidence, evidence_type, mechanisms, direction present
4. `test_detect_graph_type_causal` -- returns "causal" for CausalGraph JSON
5. `test_detect_graph_type_knowledge` -- returns "knowledge" for KG JSON
6. `test_feedback_loop_detection` -- cycle entities get `in_feedback_loop=True`
7. `test_causal_mermaid_export` -- direction-specific arrows present
8. `test_causal_graphml_preserves_metadata` -- attributes in output
9. `test_auto_detect_routes_to_causal` -- `cmd_view` with causal JSON uses conversion

---

## Implementation Order

1. `ontograph/causal_view.py` (new) -- foundation
2. `ontograph/cli.py` -- subcommands + auto-detection
3. `ontograph/export.py` -- causal export functions
4. `ontograph/dashboard/cytoscape_viewer.html` -- causal visual encoding
5. `tests/test_causal_view.py` (new) -- test coverage

---

## Design Decisions

**Why conversion layer, not second viewer template**: The HTML viewer is ~2000 lines. Duplicating = maintenance burden. Instead, the conversion maps causal semantics into `{entities, relations}`, and the viewer conditionally branches on `schema_name` for styling (~50-80 lines of JS).

**Why `schema_name = "causal"` as discriminator**: Both `KnowledgeGraph` and viewer already read `schema_name`. Zero new fields needed.

**Why metadata dicts**: `Entity.metadata` and `Relation.metadata` already exist as `dict[str, Any]`. The viewer already passes `metadata` into Cytoscape data objects, so causal fields are immediately accessible in JS.

**Backwards compatibility**: All changes are additive. Existing commands work unchanged with KG JSON. Auto-detection only triggers when input has `claims` at top level.
