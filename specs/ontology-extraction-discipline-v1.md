# Ontograph — Ontology Extraction Discipline, Spec V1

**Author:** Matias Urzua
**Date:** 2026-04-19
**Status:** Draft
**Motivation:** `knowledge-vault/wiki/methods/ontology-extraction-methodology.md`
**Companion (does not supersede):** `specs/causal-graph-v2.md`

---

## 1. Problem statement

Ontograph currently extracts typed entities and relations from prose documents using an LLM pipeline. The result is a knowledge graph with schema-validated labels but no **grounding** (entities are not verified against source code), no **semantic load** on edges (no sign, lag, functional form, conditionality), no **temporal stratification** (within-step vs cross-step edges are indistinguishable), no **epistemic stratification** (identity vs mechanism vs parameter), and no **quality metrics** (coverage, groundedness, consistency, orphan rate are not reported).

For the two target downstream uses — structural priors for differentiable simulators, and causal-path scaffolds for potential-outcome analysis — the current output is insufficient.

This spec defines the upstream discipline: **how to extract an ontology that is fit for those downstream purposes**. It sits below `specs/causal-graph-v2.md`, which defines the causal overlay on top of the base graph this spec hardens.

### 1.1 Scope

**In scope:**
- Dual extraction pipeline (LLM-from-prose + AST-from-code) with cross-check
- Grounding: every entity cites `file:line` or citation key or `abstract: true`
- Typed edges: `sign`, `lag`, `form`, `conditional_on`, `class`, `confidence`
- Epistemic stratification: `identity` / `mechanism` / `parameter` / `structural` / `abstract`
- Quality metrics reported per extraction run
- Quality gates as run-failure conditions
- Schema `extends:` support
- Simulator-in-the-loop falsifier for sign-claimed edges

**Out of scope (handled elsewhere):**
- Pearl-style do-calculus over the mechanism subgraph — `specs/causal-graph-v2.md`
- Regime-conditional causal inference — `specs/causal-graph-v2.md`
- Dashboard / visualisation changes (separate spec)
- MCP bridge changes (separate spec)

### 1.2 Success criteria

A re-extraction of the DABS ontology using this spec must produce a graph that passes these gates:

1. `groundedness >= 0.8` (≥80% of entities have `file:line` or citation or `abstract: true`)
2. `signedness >= 0.7` (≥70% of mechanism edges carry a sign)
3. `coverage >= 0.85` (≥85% of discoverable mechanisms in `DiffMABM/dabs/` appear as nodes)
4. `cycle_count.across_step > 0` (at least one feedback loop, since DABS is dynamical)
5. All gates pass with documented provenance for each failure exception.

---

## 2. Design principles

1. **Grounding before synthesis.** No entity enters the graph without an anchor. Prose-only entities are rejected unless explicitly marked `abstract: true` with rationale.
2. **Typed edges are the default.** Untyped edges (sign=unknown, lag=unknown) are allowed but counted against the `signedness` metric and reported as extraction debt.
3. **Dual extraction with agreement set.** AST and LLM pipelines run independently; agreement is promoted, disagreement is logged.
4. **Quality metrics are artefacts of every run.** `extraction_report.json` sits alongside the KG. Downstream tools may refuse to load KGs without passing reports.
5. **Backward compatibility for one release.** Existing KGs produced by v0.x load without error; new fields default to `unknown`. Deprecation warning surfaces at load time.

---

## 3. Data model changes

### 3.1 Extend `models.Entity`

```python
@dataclass
class CodeAnchor:
    """Grounding anchor pointing into source code."""
    repo: str           # "DiffMABM"
    path: str           # "dabs/approaches/batched/simulation.py"
    line: int           # 401
    symbol: str = ""    # "_step"

@dataclass
class CitationAnchor:
    """Grounding anchor into the literature."""
    key: str            # "jorgenson1963"
    pages: str = ""     # "247-259"

@dataclass
class Entity:
    name: str
    entity_type: str
    # existing fields preserved: aliases, observations, provenance, metadata
    aliases: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # NEW:
    code_anchors: list[CodeAnchor] = field(default_factory=list)
    citation_anchors: list[CitationAnchor] = field(default_factory=list)
    abstract: bool = False        # pure-concept node, no grounding required
    abstract_rationale: str = ""  # required iff abstract=True
    confidence: float = 1.0       # extraction confidence ∈ [0, 1]

    def is_grounded(self) -> bool:
        return bool(self.code_anchors) or bool(self.citation_anchors) or self.abstract
```

**Validation rule** (`models.Entity.__post_init__`): if `abstract` is True, `abstract_rationale` must be non-empty. Otherwise, at least one of `code_anchors` / `citation_anchors` must be present at commit time (the *extractor* may emit ungrounded entities; the *commit* step to a final KG rejects them or routes them to a `pending_grounding` bucket).

### 3.2 Extend `models.Relation`

```python
Sign = Literal["+", "-", "±", "0", "unknown"]
Lag = Literal["within_step", "across_step", "unknown"] | str  # "Q+1", "Q+4"
Form = Literal["linear", "monotone", "threshold", "identity", "unknown"]
EdgeClass = Literal["identity", "mechanism", "parameter", "structural", "abstract"]

@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    # existing fields preserved
    weight: float = 1.0
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # NEW:
    sign: Sign = "unknown"
    lag: Lag = "unknown"
    form: Form = "unknown"
    conditional_on: list[str] = field(default_factory=list)  # entity names that must hold
    edge_class: EdgeClass = "structural"
    confidence: float = 1.0
```

Default values are chosen so that v0.x KGs load identically: untyped edges become `sign=unknown, lag=unknown, form=unknown, edge_class=structural`.

### 3.3 Extend schema YAML

Current `schemas/*.yaml` supports `entity_types`, `relation_types`, `extraction_hints`. Extensions:

```yaml
# schemas/dabs.yaml (illustrative diff)
extends: economics                    # NEW — flat-merge base schema below this one

entity_types:
  phase:
    description: "A phase in the step function"
    color: "#3F51B5"
    default_edge_class: structural    # NEW — default class for edges into/out of this type

relation_types:
  credit_spread_to_default_rate:
    description: "BGG-type accelerator: rising spreads → higher defaults"
    domain: [mechanism]
    range: [state_variable]
    default_sign: "+"                 # NEW — extractor fills this unless prose overrides
    default_edge_class: mechanism     # NEW
    default_form: monotone            # NEW

  executes_after:
    description: "Temporal ordering within a single step"
    domain: [phase]
    range: [phase]
    default_lag: within_step          # NEW
    default_edge_class: structural

validation:                           # NEW — quality-gate thresholds
  groundedness_min: 0.8
  signedness_min: 0.7
  coverage_min: 0.85
  orphan_rate_max: 0.15
  require_across_step_cycles: true    # for dynamical systems
```

### 3.4 Extend `OntologySchema` loader

`schema.py::load_schema` gains `extends` support via BFS-merge:

```python
def load_schema(name_or_path: str, _seen: set[str] | None = None) -> OntologySchema:
    _seen = _seen or set()
    if name_or_path in _seen:
        raise ValueError(f"Schema extension cycle: {name_or_path}")
    _seen.add(name_or_path)

    raw = _read_yaml(_resolve(name_or_path))
    base: OntologySchema | None = None
    if "extends" in raw:
        base = load_schema(raw["extends"], _seen=_seen)

    # merge base fields under current overrides
    schema = _construct_from_raw(raw)
    if base:
        schema = _merge_schemas(base, schema)   # current overrides base
    return schema
```

The merge semantics: child schema's entity_types / relation_types union with parent's; on name collision, child wins; `validation` block is dict-merged with child override.

---

## 4. Pipeline architecture

### 4.1 Dual extraction

Current pipeline (simplified):

```
parse_document → llm_extract → dedup → schema_validate → write KG
```

New pipeline:

```
┌─ parse_document  ─┐
│                   ├─ llm_extract ──┐
└─ parse_codebase ──┤                ├─ merge + agreement_set → dedup → schema_validate
                    └─ ast_extract ──┘                                          │
                                                                                ↓
                                                                quality_gate → write KG
                                                                                +
                                                                extraction_report.json
```

#### 4.1.1 AST extractor (new)

New module: `ontograph/ast_extractor.py`

```python
def extract_from_repo(
    repo_root: Path,
    schema: OntologySchema,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """
    Walk Python AST (via `ast` or `libcst`) and emit candidate entities/relations.

    Entity candidates:
      - Every `class` declaration  → entity_type inferred from schema hints
                                     (e.g., `nn.Module` subclass → learned_operator)
      - Every `def _phase_*`       → phase entity
      - Every module-level name matching `smooth_*` → smooth_operator
      - Every `_assert_*` method   → assertion / invariant
    Relation candidates:
      - Function call sites        → "uses" / "invokes" relations
      - Method-call within `_phase_*` that writes to ctx[k] → (phase) writes (state)
      - Import graph               → structural dependency
    """
```

**Schema-driven inference.** The AST extractor uses `schema.extraction_hints` to map code patterns to entity types:

```yaml
extraction_hints:
  ast_patterns:
    - pattern: "def _phase_*"
      entity_type: phase
    - pattern: "class * (nn.Module)"
      entity_type: learned_operator
    - pattern: "smooth_* = "
      entity_type: smooth_operator
    - pattern: "def _assert_*"
      entity_type: assertion
```

Output entities carry `code_anchors` populated from the AST node's file+line. This is where grounding is built in by construction.

#### 4.1.2 LLM extractor (existing, lightly modified)

Only changes:

1. Prompts ask for `sign`, `lag`, `form` when the prose contains signal words (`increase`, `decrease`, `lag`, `quarter`, `threshold`, `if`, `when`). A new prompt scaffold lives in `llm_extractor.py::TYPED_RELATION_PROMPT`.
2. Prompts ask the model to cite a paragraph excerpt verbatim; that excerpt becomes `Provenance.passage`, and the model's claimed sign/lag/form is only accepted if the excerpt contains the signal words.
3. Prompts instruct the model to mark `abstract: true` for pure-concept entities (e.g., "Tobin-Q theory" vs. `agents/firms.py::tobin_q_investment`).

#### 4.1.3 Merge + agreement set

New module: `ontograph/merge.py`

```python
@dataclass
class MergeReport:
    agreement_entities:    list[Entity]        # in both P₁ and P₂
    llm_only_entities:     list[Entity]        # prose says yes, code says no
    ast_only_entities:     list[Entity]        # code says yes, prose says no
    agreement_relations:   list[Relation]
    llm_only_relations:    list[Relation]
    ast_only_relations:    list[Relation]
    sign_conflicts:        list[tuple[Relation, Relation]]  # same edge, different signs

def merge_extractions(
    llm_entities, llm_relations,
    ast_entities, ast_relations,
    *,
    name_matcher: Callable[[str, str], float] = default_name_matcher,
    threshold: float = 0.85,
) -> tuple[KnowledgeGraph, MergeReport]:
    ...
```

Agreement rule: entities match if `name_matcher(a.name, b.name) >= threshold` AND `entity_type` matches (allowing `abstract=True` LLM entity to match a concrete AST entity — the LLM-named theory binds to its AST-named implementation via an `implements` relation).

The merge preserves all three buckets (`agreement`, `llm_only`, `ast_only`). Only `agreement` enters the default KG; `llm_only` and `ast_only` are emitted as `extraction_report.disagreements` for human audit. This matches the methodology rule: agreement is high-confidence, disagreement is an audit target.

### 4.2 Quality-gate stage

New module: `ontograph/quality.py`

```python
@dataclass
class QualityReport:
    coverage:      dict[str, Any]
    groundedness:  dict[str, Any]
    signedness:    dict[str, Any]
    orphan_rate:   dict[str, Any]
    consistency:   dict[str, Any]
    cycle_count:   dict[str, Any]
    gates_passed:  bool
    failures:      list[str]       # human-readable

def compute_quality_report(
    kg: KnowledgeGraph,
    schema: OntologySchema,
    *,
    ast_entities: list[Entity] | None = None,  # for coverage computation
) -> QualityReport:
    ...
```

Gate enforcement:

- If `schema.validation` is present and any threshold is violated, the CLI exits non-zero unless `--allow-gate-failure` is passed.
- The report is always written to `<kg-path>.quality.json` alongside the KG.

#### 4.2.1 Metric definitions

- **Coverage** = `|entities_in_kg matching AST candidates| / |AST candidate entities|`. Computed only when an AST extraction is available.
- **Groundedness** = `|entities with code_anchor | citation_anchor | abstract| / |entities|`.
- **Signedness** = `|mechanism edges with sign != "unknown"| / |mechanism edges|`.
- **Orphan rate** = `|entities with no incident edges| / |entities|`. Excludes `abstract: true` nodes by default (pure-concept labels have no expected edges).
- **Consistency** = `1 − |sign_conflicts| / |distinct_edges|`. Sign conflict defined as two relations with same `(source, target, relation_type)` and contradictory signs.
- **Cycle count** is decomposed by `lag`: `within_step` cycles are reported (acyclicity within a step is typically expected), `across_step` cycles are reported and counted against `require_across_step_cycles` for dynamical-system schemas.

### 4.3 Simulator-in-the-loop falsifier (optional command)

New command: `ontograph falsify --kg <path> --simulator <adapter> [--subset mechanism]`

Adapter protocol (users implement for their simulator):

```python
class SimulatorAdapter(Protocol):
    def edge_sign(self, source: str, target: str) -> Sign:
        """Query the simulator for the empirical sign of d(target)/d(source)."""
    def edge_lag(self, source: str, target: str) -> Lag:
        """Quarters of lag between source perturbation and target response."""
```

For DABS, the adapter wraps `mvm/gradnorm.py` (for magnitudes) and finite-difference Jacobians (for sign). Output:

```
{
  "confirmed":  [(source, target, kg_sign, sim_sign)...],   # match
  "flipped":    [(source, target, kg_sign, sim_sign)...],   # contradicts — audit
  "unknown":    [(source, target)...],                       # simulator couldn't resolve
}
```

This is the **direct simulator-in-the-loop falsification mechanism** described in §4.7 of the companion methodology report. It is the last line of defence against LLM-hallucinated signs.

---

## 5. CLI additions

```
ontograph ingest              # existing — extended to support --ast-repo and --dual-extract
  --ast-repo <path>           # NEW — run AST extraction over this repo
  --dual-extract              # NEW — require both LLM and AST pipelines to run
  --quality-report <path>     # NEW — write quality report (default: <kg>.quality.json)

ontograph ground              # NEW — post-hoc grounding pass
  --kg <path>                 # input KG to ground
  --ast-repo <path>           # repo to search for anchors
  --citation-bib <path>       # BibTeX / Zotero JSON of known citations
  --in-place                  # overwrite KG with grounded version

ontograph quality             # NEW — recompute quality report on an existing KG
  --kg <path>                 # input
  --ast-repo <path>           # optional, for coverage
  --schema <name|path>

ontograph diff                # NEW — compare two KGs produced at different times
  --old <path>                # baseline
  --new <path>                # current
  --report <path>             # output markdown of added/removed/changed edges

ontograph falsify             # NEW — run simulator-in-the-loop falsifier
  --kg <path>
  --simulator <adapter_module>
  --subset {mechanism,all}
  --report <path>
```

All new commands respect existing `--schema` and `--llm-backend` flags.

---

## 6. Migration plan

### Phase A — Data-model extension (non-breaking)

1. Add `CodeAnchor`, `CitationAnchor`, `abstract`, `abstract_rationale`, `confidence` to `Entity`. All optional with defaults.
2. Add `sign`, `lag`, `form`, `conditional_on`, `edge_class`, `confidence` to `Relation`. All optional with defaults.
3. Update `to_json` / `from_json` to round-trip new fields. **Critical:** `from_json` accepts v0.x payloads without the new fields and populates defaults.
4. Release as `ontograph 0.(next).0` with changelog note: "New fields on Entity/Relation are additive; v0.x KGs load unchanged."

### Phase B — Schema `extends` + `validation` block

1. Implement `load_schema` BFS-merge.
2. Implement `schema.validation` parsing.
3. Rewrite `schemas/dabs.yaml` to `extends: economics` — reduces duplication introduced during the pilot.
4. Add `validation:` block to `dabs.yaml`, `economics.yaml` with conservative defaults.

### Phase C — AST extractor

1. `ast_extractor.py` with schema-driven pattern matching.
2. Unit tests against a frozen snapshot of `DiffMABM/dabs/` (`tests/fixtures/dabs-snapshot/`).
3. Coverage measurement: run against DABS, report metrics.

### Phase D — Merge + quality gate

1. `merge.py` with `MergeReport`.
2. `quality.py` with `QualityReport` and all six metrics.
3. Wire into `cli.py::ingest` with gate enforcement.
4. Regenerate DABS KG using the full pipeline. Compare against pilot KG (145 entities / 67 relations); expect coverage ↑, groundedness ↑, signedness ↑.

### Phase E — Falsifier (optional, depends on DABS MVM availability)

1. `falsify.py` with `SimulatorAdapter` protocol.
2. DABS adapter at `DiffMABM/dabs/mvm/ontograph_adapter.py` (implemented in DABS repo, not here).
3. End-to-end test: run falsifier against DABS KG, confirm the ≥3 edges we already have empirical Jacobians for (from `project_jacobian_diagnostic_result`).

---

## 7. Test plan

### 7.1 Unit tests

- `test_models_backward_compat.py` — v0.x JSON loads into v1 data model without loss.
- `test_schema_extends.py` — `extends: economics` merges correctly; cycle detection works; child override on collision.
- `test_ast_extractor.py` — fixture repo produces expected entities; pattern matching correct; anchors populated.
- `test_merge.py` — agreement / llm-only / ast-only buckets populated correctly; sign-conflict detection works.
- `test_quality.py` — each metric computed correctly on hand-built KGs; gates trigger on threshold violation.
- `test_relation_sign.py` — Relation.sign round-trips; unknown signs don't count toward signedness.

### 7.2 Integration tests

- `test_dabs_end_to_end.py` — ingest `wiki/architecture/dabs-architectural-ontology.md` with `--ast-repo DiffMABM`, assert quality gates pass.
- `test_falsify_mock.py` — mock SimulatorAdapter, verify flipped/confirmed/unknown buckets populate correctly.

### 7.3 Property tests (`hypothesis`)

- For any `Relation` with sign ∈ {+, -} and another Relation reversing it on same endpoints, the consistency metric drops by `1/n`.
- For any Entity with no anchors and `abstract=False`, quality gate fails.
- For any schema with `extends` cycle, `load_schema` raises `ValueError`.

### 7.4 Regression test: the DABS pilot itself

- Pin the current 145-entity / 67-relation pilot KG as `tests/fixtures/dabs-pilot-kg-v0.json`.
- After Phase A migration: load this file, assert entity/relation counts unchanged, assert all v1 fields default correctly.
- After Phase D: re-extract using dual pipeline, assert coverage improves (expected ≥ 0.85), groundedness improves (expected ≥ 0.8), orphan rate drops (expected < 0.15 once doc edits from the pilot are in place).

---

## 8. Deliverables and phasing

| Phase | Deliverable | Estimated effort | Depends on |
|-------|-------------|------------------|------------|
| A | Data-model extension + serialisation round-trip | 1-2 days | — |
| B | Schema `extends` + `validation` block | 0.5 day | A |
| C | AST extractor + schema-driven patterns | 2-3 days | B |
| D | Merge + quality gate + CLI wiring | 2-3 days | C |
| E | Falsifier + DABS adapter (in DABS repo) | 3-4 days | D, MVM |

**Total:** ~2 working weeks to MVP through phase D; phase E deferrable.

---

## 9. Relationship to existing specs

- `specs/causal-graph-v2.md` defines the Pearl-style causal overlay. That overlay reads `edge_class=mechanism` edges from the base graph produced by *this* spec and layers `CausalMechanism` / `CausalCondition` on top. Nothing in v2 breaks; v2 now has typed base edges to build on, which it previously assumed existed only in the "attached" causal layer.
- `specs/KnowledgeGraph.tla` (TLA+ invariants) will need one clause added: grounding — `∀ e ∈ Entities : IsGrounded(e) ∨ e.abstract`. The existing uniqueness and relation-validity invariants carry over.
- `specs/EntityResolver.tla` is unaffected by this spec (resolution is about deduplication, orthogonal to grounding / typing).

---

## 10. Open questions

- **Partial grounding.** Should an entity with a citation but no code anchor count as fully grounded for DABS? Proposed: yes for `theory` and `abstract` entity types; no for `mechanism`, `phase`, `learned_operator` (these must have code anchors).
- **Sign granularity.** Is `±` (ambiguous / context-dependent) enough, or do we need `positive_in_regime_X, negative_in_regime_Y`? Proposed: use `conditional_on` + split into two relations with opposite signs; avoid multi-valued `sign` to keep do-calculus semantics clean.
- **LLM prompt for typed edges.** Current prompt gets 60-80% of sign / lag right in our calibration set. Target: ≥90% before wiring dual-extract into the default pipeline. Until then: default to `unknown` rather than guess.
- **Cycle detection semantics.** Within-step cycles typically indicate a modelling error (fixed-point solver implicit). Across-step cycles are desired for dynamical systems. Should the metric weight these differently in the consistency score? Proposed: separate metrics, no composite score.

---

## 11. Non-goals

- **Not** a replacement for the existing LLM pipeline. LLM-from-prose remains the primary route for prose-heavy sources without an adjoining codebase.
- **Not** a formal verification system. TLA+ specs remain at the data-model level; this spec does not aim to prove extraction correctness.
- **Not** an ontology editor. Graph edits happen at the source-document layer (per the methodology rule "fix the doc, not the KG"); this spec enforces that discipline, does not replace it.

---

## See also

- `knowledge-vault/wiki/methods/ontology-extraction-methodology.md` — the methodology this spec implements
- `specs/causal-graph-v2.md` — Pearl-style causal overlay (upstream consumer of this spec's output)
- `knowledge-vault/wiki/architecture/dabs-architectural-ontology.md` — the primary source document for the DABS ontology
