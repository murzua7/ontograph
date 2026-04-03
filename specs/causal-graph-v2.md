# Ontograph Causal Graph — Spec V2

**Author:** Matias Urzua  
**Date:** 2026-04-03  
**Status:** Draft  
**Supersedes:** Spec V1 (inline conversation, 2026-04-03)  
**V1 review:** `specs/causal-graph-v1-review.md`

---

## 1. Problem Statement

Standard knowledge graphs encode structural relationships: X *feeds_into* Y, X *constrains* Y. These are topological facts. Causal claims are strictly stronger: they assert that intervening on X changes Y, through some mechanism, with some lag, conditional on some regime, with some quality of evidence behind the assertion.

Conflating structural co-occurrence with causal mechanism is the central epistemological error in applied economics. This spec designs a system that maintains the distinction rigorously from the data model up.

### 1.1 Two End Uses

**Use A — DABS structural prior.** The causal graph constrains what the DABS differentiable ABM is allowed to learn. Instead of discovering causal structure from 17 quarterly observations (hopeless), the GNN receives a prior over which macro variables influence which, with what sign, at what lag. The GNN learns residuals on top of this prior.

**Use B — Global macro scenario engine.** Given a geopolitical or economic shock, trace causal cascades through known transmission channels. Time-layer the effects. Surface consequences that sophisticated market participants may not have priced, especially cross-market and lagged effects.

### 1.2 Design Principles

1. **Epistemic honesty.** The system must never present uncertain estimates as precise. Propagate direction and confidence; defer magnitude estimation to structural models (DABS) where possible.
2. **Regime awareness.** Causal relationships in macroeconomics are state-dependent. A static graph with fixed edge weights cannot represent this. Regime conditioning is a first-class requirement.
3. **Falsifiability.** The system must be backtestable against historical episodes. Without evaluation, it is unfalsifiable and therefore useless.
4. **Composability.** The causal graph is a layer on top of the ontograph knowledge graph. It reuses entity names and adds causal semantics. It is not a replacement.

---

## 2. Data Model

### 2.1 Causal Mechanism

A single causal claim may operate through multiple simultaneous transmission channels. Each channel has its own conditions, time horizon, and effect characteristics.

```python
@dataclass
class CausalMechanism:
    """A single transmission channel within a causal claim."""
    name: str                       # "energy_cost_passthrough", "credit_channel"
    description: str                # human-readable explanation of how the channel works

    # Effect characteristics for this channel
    direction: Literal["positive", "negative", "ambiguous", "state_dependent"]
    nonlinear: bool = False         # does the effect depend on magnitude/regime?

    # Elasticity (optional, often unavailable)
    elasticity_range: tuple[float, float] | None = None  # (low_estimate, high_estimate)
    elasticity_horizon: Literal["short_run", "long_run"] | None = None
    elasticity_source: str | None = None  # citation

    # Temporal
    time_lag_min: str | None = None   # "1 day", "2 weeks", "3 months"
    time_lag_max: str | None = None   # "1 week", "3 months", "2 years"

    # Conditions under which this channel is active
    conditions: list[CausalCondition] = field(default_factory=list)
```

### 2.2 Causal Condition

Conditions are machine-evaluable, not free-text strings. The cascade engine checks conditions at each edge before traversal.

```python
@dataclass
class CausalCondition:
    """A regime condition that gates a causal channel."""
    variable: str                   # "capacity_utilization", "policy_rate", "leverage_ratio"
    operator: Literal[">", "<", ">=", "<=", "==", "in_range", "regime_is"]
    threshold: str                  # "85%", "0%", "high_leverage"
    description: str                # "holds when economy near full capacity"
    currently_met: bool | None = None  # populated at scenario-evaluation time
```

### 2.3 Causal Claim

The primary data object. A container for one or more mechanisms connecting a cause to an effect.

```python
@dataclass
class CausalClaim:
    id: str
    source: str                     # cause entity name
    target: str                     # effect entity name
    mechanisms: list[CausalMechanism]  # one or more transmission channels

    # ── Causal type ───────────────────────────────────────────────
    causal_type: Literal[
        "micro_mechanical",         # balance-sheet identity, institutional rule
        "macro_statistical",        # aggregate empirical relationship
    ]

    # ── Epistemology ──────────────────────────────────────────────
    evidence_type: str              # from the expanded taxonomy (Section 3)
    evidence_count: int             # number of independent sources
    evidence_directness: Literal["primary", "cited", "secondary_citation"]
    claim_assertiveness: Literal["hedged", "moderate", "strong", "definitional"]

    # ── Confidence (computed, not extracted) ───────────────────────
    confidence: float               # composite score [0, 1] — Section 4
    strength: Literal["correlation", "weak_causal", "strong_causal"]

    # ── Aggregate direction (across all mechanisms) ───────────────
    net_direction: Literal["positive", "negative", "ambiguous", "conflicted"]
    direction_contested: bool = False  # does the literature disagree on direction?

    # ── Scope ─────────────────────────────────────────────────────
    geographic_scope: str           # "global", "OECD", "France", "emerging_markets"
    geographic_entities: list[str] = field(default_factory=list)  # ["DE", "JP", "KR"]
    temporal_scope: str | None = None  # "1970-present", "post-GFC"

    # ── Reversibility ─────────────────────────────────────────────
    reversibility: Literal["permanent", "temporary", "unknown"] = "unknown"

    # ── Provenance ────────────────────────────────────────────────
    sources: list[Provenance] = field(default_factory=list)
    claim_text: str = ""            # verbatim sentence(s) making the causal assertion
    counterfactual_stated: bool = False  # did the source articulate the counterfactual?
    extracted_by: Literal["llm", "manual", "heuristic"] = "llm"

    # ── Contestation ──────────────────────────────────────────────
    contradicted_by: list[str] = field(default_factory=list)  # claim IDs
    superseded_by: str | None = None  # newer claim ID if updated

    # ── Trading metadata (optional, populated by analyst) ─────────
    asset_classes: list[str] = field(default_factory=list)     # ["commodities", "fx", "rates"]
    sector_codes: list[str] = field(default_factory=list)      # GICS codes
```

### 2.4 Causal Graph

Stored separately from the knowledge graph but references the same entity names.

```python
@dataclass
class CausalGraph:
    claims: dict[str, CausalClaim] = field(default_factory=dict)  # id -> claim
    entities: set[str] = field(default_factory=set)  # entity names referenced
    schema_name: str = "causal"
    source_knowledge_graph: str | None = None  # path to the KG this was derived from
    build_date: str = ""
    regime_context: dict[str, str] = field(default_factory=dict)  # current regime state
```

**Storage:** Serialised as `causal_graph.json`, separate from the main `graph.json`. Both reference the same entity names. The viewer loads both files and overlays the causal layer.

---

## 3. Evidence Taxonomy

The V1 taxonomy was biased toward medical/social-science methods. The revised taxonomy covers the full range of evidence types relevant to macroeconomics and finance.

### 3.1 Evidence Types and Base Scores

| Evidence Type | Base Score | Description | Example |
|---|---|---|---|
| `accounting_identity` | **1.00** | Definitionally true. If oil supply drops and demand is constant, price rises. | GDP = C + I + G + NX |
| `institutional_mechanism` | **0.90** | Near-certain given institutional structure exists. | "ECB mandate requires response to inflation >2%" |
| `meta_analysis` | **0.90** | Synthesis of multiple independent studies | Havranek & Rusnak (2013) monetary transmission meta-analysis |
| `rct` | **0.90** | Randomised controlled trial | Rare in macro; common in development |
| `natural_experiment` | **0.80** | Exogenous variation, researcher-identified | German reunification, Romer & Romer monetary shocks |
| `synthetic_control` | **0.75** | Abadie-style counterfactual construction | Effect of EMU on trade |
| `iv` | **0.70** | Instrumental variable identification | Bartik instruments for labour shocks |
| `did` | **0.70** | Difference-in-differences | Policy change affecting some states but not others |
| `rdd` | **0.70** | Regression discontinuity | Threshold-based policy eligibility |
| `event_study` | **0.65** | High-frequency study around a discrete event | Stock price response to Fed announcements |
| `historical_precedent` | **0.60** | Repeated occurrence across multiple episodes | "Oil spiked in 5/5 prior Hormuz threats" |
| `panel_fixed_effects` | **0.50** | Fixed-effects regression, no clean identification | Cross-country growth regressions |
| `structural_estimation` | **0.50** | Estimated structural model (DSGE, IO) | Smets & Wouters (2007) |
| `reduced_form_regression` | **0.40** | OLS/GLS without identification strategy | Time-series regression with controls |
| `granger_causality` | **0.30** | Statistical temporal precedence test | VAR-based Granger tests |
| `model_simulation` | **0.25** | Unestimated model simulation | Calibrated DSGE comparative statics |
| `market_implied` | **0.20** | Derived from market prices (options, forwards) | Options-implied probability of rate hike |
| `narrative` | **0.15** | Author assertion without empirical support | "We believe X causes Y" |

### 3.2 Key Differences from V1

1. **`accounting_identity` at 1.0.** Physical and accounting constraints are definitionally true. "If Hormuz closes, 20% of global oil transit is disrupted" is not a probabilistic claim — it is mechanical.

2. **`historical_precedent` at 0.60** (was bucketed as "narrative" at 0.15). If oil spiked in every prior Hormuz threat, that is frequentist evidence. Not an experiment, but much stronger than narrative.

3. **`institutional_mechanism` at 0.90.** "Pipeline X has capacity Y barrels/day" is a physical fact. "The ECB is mandated to maintain price stability" is an institutional fact. These are near-certain given current institutional architecture.

4. **`market_implied` added at 0.20.** Low base score because market beliefs can be wrong, but useful for the trading application (comparing cascade-implied to market-implied).

---

## 4. Causal Strength Scoring — Revised

### 4.1 Composite Score

```
confidence = 0.30 * evidence_score
           + 0.25 * external_validity_score    [NEW — was missing]
           + 0.20 * mechanism_score
           + 0.15 * evidence_diversity_score   [REVISED — was raw count]
           + 0.10 * assertiveness_score        [NEW — replaces temporal precedence]
```

### 4.2 Temporal Precedence — now a gate, not a weight

Temporal precedence is a **necessary condition** for causation, not a degree-of-support factor:

```python
def apply_temporal_gate(claim: CausalClaim) -> CausalClaim:
    """Cap strength at 'correlation' if temporal precedence is violated."""
    if claim.temporal_precedence == "reversed":
        claim.strength = "correlation"
        claim.confidence = min(claim.confidence, 0.33)
    elif claim.temporal_precedence == "unknown" and claim.causal_type == "macro_statistical":
        claim.confidence *= 0.7  # penalise but don't cap
    # For equilibrium relationships and accounting identities: no temporal gate
    return claim
```

### 4.3 External Validity Score (NEW)

Measures how well the evidence context matches the application context:

| Context Match | Score |
|---|---|
| Same country, same decade, similar institutional setting | 1.00 |
| Same economy type (OECD-OECD), same decade | 0.80 |
| Same economy type, different decade (but similar institutions) | 0.60 |
| Different economy type, same decade | 0.40 |
| Different economy type, different decade | 0.20 |
| Lab/micro experiment applied to macro context | 0.10 |

The cascade engine must know the **target context** (which country, which year, which institutional regime) to compute this score. This is set in `CausalGraph.regime_context`.

### 4.4 Evidence Diversity Score (revised from raw count)

Raw evidence count is misleading because studies share datasets and methods. The revised score rewards **methodological diversity**:

```python
def evidence_diversity_score(claim: CausalClaim) -> float:
    """Score based on how many distinct methods/datasets support the claim."""
    methods_used = set(s.method for s in claim.supporting_studies)
    datasets_used = set(s.dataset for s in claim.supporting_studies)

    method_diversity = min(1.0, len(methods_used) / 4)   # 4 distinct methods = 1.0
    dataset_diversity = min(1.0, len(datasets_used) / 3)  # 3 distinct datasets = 1.0

    return 0.6 * method_diversity + 0.4 * dataset_diversity
```

When detailed study metadata is unavailable (common), fall back to:
```python
fallback = min(1.0, log(1 + n_independent) / log(6))  # saturates at ~5
```

### 4.5 Mechanism Score (refined from binary)

| Mechanism Description | Score |
|---|---|
| Named channel + quantified (elasticity or coefficient from data) | 1.00 |
| Named channel + plausible economic logic articulated | 0.80 |
| Named channel, vague description ("through financial channels") | 0.50 |
| Mechanism implied but not stated | 0.30 |
| No mechanism identified | 0.00 |

### 4.6 Claim Assertiveness Score (NEW — replaces temporal precedence as weight)

Captures how strongly the source asserts the causal claim:

| Assertiveness | Score | Linguistic markers |
|---|---|---|
| `definitional` | 1.00 | "by definition", "must", "is equal to" |
| `strong` | 0.80 | "causes", "has been shown to", "demonstrates" |
| `moderate` | 0.50 | "is likely to", "tends to", "contributes to" |
| `hedged` | 0.20 | "may", "might", "is associated with", "could" |

### 4.7 Strength Classification

The continuous score is the **primary representation**. The three-bin classification is for display and filtering only. Thresholds are configurable:

```python
def classify_strength(confidence: float, thresholds: tuple = (0.35, 0.65)) -> str:
    if confidence < thresholds[0]: return "correlation"
    if confidence < thresholds[1]: return "weak_causal"
    return "strong_causal"
```

Default thresholds (0.35, 0.65) should be calibrated against backtesting results (Section 10) and adjusted per domain.

---

## 5. Extraction Pipeline

### 5.1 Architecture

```
For each document:
  Stage A: Identify all causal claims in the text
           → atomic (source, mechanism, target) triples
           → classify evidence type and assertiveness

  Stage B: Assess each claim's evidence quality
           → score mechanism specificity, evidence count, context match
           → compute composite confidence

After all documents:
  Stage C: Aggregate across documents
           → merge duplicate claims (same source-target pair)
           → flag contradictions
           → Bayesian update of confidence from multiple sources
```

### 5.2 Stage A — Causal Claim Identification

The extraction prompt must handle several difficult cases:

**Compound claims must be decomposed.**
> "The oil embargo raised inflation, depressed industrial output, and triggered a banking crisis"

= three atomic claims: oil→inflation, oil→output, oil→banking_crisis. The prompt explicitly instructs decomposition.

**Implicit causal language must be recognised.**
> "The effect of monetary tightening on credit supply"

= causal claim (monetary_tightening → credit_supply, negative) even though "causes" doesn't appear.

**Hedged claims must be captured with assertiveness tag.**
> "Higher rates may reduce investment"

= causal claim with assertiveness = "hedged".

**First-person vs cited evidence must be distinguished.**
> "We find that X causes Y" (primary)
> "Romer and Romer (2004) show that X causes Y" (cited)
> "According to standard theory, X causes Y" (secondary/textbook)

**Extraction prompt:**

```
You are identifying causal claims in economic text. For EACH causal claim:

1. DECOMPOSE compound statements into atomic triples:
   (cause_entity, mechanism, effect_entity)

2. CLASSIFY the claim type:
   - "accounting_identity": definitionally true (GDP = C+I+G+NX)
   - "institutional_mechanism": follows from institutional rules
   - "empirical_claim": asserted based on data/evidence
   - "theoretical_claim": follows from a model or theory

3. ASSESS assertiveness from the language:
   - "definitional": "must", "by definition", "is equal to"
   - "strong": "causes", "leads to", "has been shown to"
   - "moderate": "tends to", "contributes to", "is likely to"
   - "hedged": "may", "might", "is associated with"

4. IDENTIFY the evidence type cited (if any):
   - Does the text reference a specific study? What method did it use?
   - Is this the author's own empirical result or a cited result?
   - Is this a theoretical prediction or an empirical finding?

5. EXTRACT the mechanism:
   - HOW does the cause produce the effect?
   - Through what channel, process, or institution?
   - Are there multiple channels? List each separately.

6. NOTE any conditions:
   - Under what circumstances does this relationship hold?
   - "When inflation is high", "in fixed exchange rate regimes", etc.

7. NOTE any quantitative effect:
   - Magnitude, elasticity, or regression coefficient if stated
   - Time horizon if stated

For each claim provide:
- "source": cause entity name
- "target": effect entity name
- "mechanisms": [{"name": "...", "description": "...", "direction": "positive/negative/ambiguous"}]
- "evidence_type": from the taxonomy
- "evidence_directness": "primary" / "cited" / "secondary_citation"
- "assertiveness": "definitional" / "strong" / "moderate" / "hedged"
- "conditions": [{"variable": "...", "operator": "...", "threshold": "..."}]
- "magnitude": quantitative effect if stated, else null
- "time_horizon": {"min": "...", "max": "..."} if stated
- "claim_text": verbatim sentence making the assertion
- "passage": surrounding context
- "confidence_note": anything relevant to assessing reliability
```

### 5.3 Stage B — Evidence Quality Assessment

A second prompt that takes each extracted claim and:
1. Scores mechanism specificity (Section 4.5)
2. Assesses whether the evidence cited is adequate for the strength of claim
3. Identifies known confounders not controlled for
4. Checks if a counterfactual is stated
5. Computes the composite confidence score

This stage can be merged with Stage A for shorter texts. For long documents (books, survey papers), the two-stage approach prevents context overload.

### 5.4 Stage C — Cross-Document Aggregation

When the same causal claim appears in multiple documents:

**Agreement:** Two studies using different methods find the same effect → increase confidence (evidence diversity score rises).

**Contradiction:** Source A says X→Y is positive, Source B says X→Y is negative → flag as `direction_contested = True`, store both claims, present both to the user.

**Supersession:** A newer study with better identification overrides an older one → populate `superseded_by` field on the older claim.

The aggregation is not automatic for contradictions — it flags them for human review (Section 9).

---

## 6. Cascade Simulation Engine

### 6.1 Design Principles

1. **V1 propagates direction + confidence only.** Magnitude propagation requires structural models and is deferred. The cascade engine answers "what is affected and how confident are we?" not "by how much?"
2. **Max-confidence priority traversal, not BFS.** Strongest paths are found first.
3. **Multiple paths to the same target compound, not compete.** Parallel channels reinforce.
4. **Cycles are handled via time-stepped unrolling.** Not deferred.
5. **Conditions are checked at every edge.** Machine-evaluable conditions gate traversal.

### 6.2 Shock Specification

```python
@dataclass
class Shock:
    entity: str                    # "Strait of Hormuz"
    shock_type: str                # "supply_disruption", "policy_change", "price_shock"
    description: str               # "Full closure of Strait, blocking ~20% of global oil transit"
    direction: Literal["positive", "negative"]
    magnitude_qualitative: str     # "severe", "moderate", "minor"
    counterfactual: str            # "vs pre-shock baseline"
    date: str | None = None
    regime_context: dict[str, str] = field(default_factory=dict)
    # e.g. {"capacity_utilization": "high", "policy_rate_regime": "normal", "leverage": "moderate"}
```

The `regime_context` is evaluated against each edge's `CausalCondition` objects to determine which channels are active.

### 6.3 Propagation Algorithm — Max-Confidence Priority Queue

```python
import heapq

@dataclass
class CascadeEffect:
    entity: str
    direction: Literal["positive", "negative", "ambiguous", "conflicted"]
    confidence: float              # product of edge confidences along best path
    time_horizon_min: str          # earliest this effect materialises
    time_horizon_max: str          # latest
    causal_path: list[str]         # entities in the chain: [A, B, C, D]
    all_paths: list[list[str]]     # all paths reaching this entity
    n_incoming_channels: int       # number of independent paths (for reinforcement)
    incoming_directions: dict      # {"positive": 3, "negative": 1} — for conflict detection

def run_cascade(
    graph: CausalGraph,
    shock: Shock,
    min_confidence: float = 0.15,
    max_depth: int = 6,
) -> dict[str, CascadeEffect]:
    """
    Max-confidence priority traversal.

    Explores edges in order of descending path confidence (strongest
    paths first). Multiple paths to the same node are aggregated:
    - Confidence: max across all paths (best single chain)
    - Direction: aggregated across all incoming channels
    - Reinforcement: n_incoming_channels used for supplementary scoring
    """
    effects: dict[str, CascadeEffect] = {}
    # Priority queue: (-confidence, entity_id, path, direction, time_min, time_max)
    # Negative confidence because heapq is a min-heap
    pq = [(-1.0, shock.entity, [shock.entity], shock.direction, "immediate", "immediate")]

    while pq:
        neg_conf, current, path, direction, t_min, t_max = heapq.heappop(pq)
        current_confidence = -neg_conf

        if len(path) > max_depth + 1:
            continue

        # Update or create effect record for current node
        if current not in effects:
            effects[current] = CascadeEffect(
                entity=current,
                direction=direction,
                confidence=current_confidence,
                time_horizon_min=t_min,
                time_horizon_max=t_max,
                causal_path=path,
                all_paths=[path],
                n_incoming_channels=1,
                incoming_directions={direction: 1},
            )
        else:
            eff = effects[current]
            eff.all_paths.append(path)
            eff.n_incoming_channels += 1
            eff.incoming_directions[direction] = eff.incoming_directions.get(direction, 0) + 1
            if current_confidence > eff.confidence:
                eff.confidence = current_confidence
                eff.causal_path = path
            # Detect direction conflict
            if len(set(eff.incoming_directions.keys()) - {"ambiguous"}) > 1:
                eff.direction = "conflicted"
            # Update time horizon to widest range
            eff.time_horizon_min = min_time(eff.time_horizon_min, t_min)
            eff.time_horizon_max = max_time(eff.time_horizon_max, t_max)

        # Explore outgoing causal edges
        for claim in graph.outgoing_claims(current):
            # Check conditions against current regime context
            if not all_conditions_met(claim, shock.regime_context):
                continue

            for mechanism in claim.mechanisms:
                if not all_conditions_met_mechanism(mechanism, shock.regime_context):
                    continue

                # Compute edge confidence (no depth decay — edge confidences compound naturally)
                edge_confidence = current_confidence * claim.confidence
                if edge_confidence < min_confidence:
                    continue

                # Propagate direction through this mechanism
                next_direction = propagate_direction(direction, mechanism.direction)

                # Propagate time horizon
                next_t_min = add_time(t_min, mechanism.time_lag_min or "immediate")
                next_t_max = add_time(t_max, mechanism.time_lag_max or t_max)

                # Avoid revisiting the same node in the same path (prevents infinite loops)
                if claim.target not in path:
                    new_path = path + [claim.target]
                    heapq.heappush(pq, (
                        -edge_confidence,
                        claim.target,
                        new_path,
                        next_direction,
                        next_t_min,
                        next_t_max,
                    ))

    return effects
```

### 6.4 No Depth Decay

V1 had `decay(d) = 0.85^d`. This is removed. The individual edge confidences already compound multiplicatively and provide the correct discount:

```
3-hop chain with edges (0.90, 0.90, 0.90) → 0.73
3-hop chain with edges (0.60, 0.60, 0.60) → 0.22
```

The edge confidences themselves capture epistemic uncertainty. Adding an extra depth penalty double-counts and artificially suppresses well-grounded long chains.

### 6.5 Direction Propagation

```python
def propagate_direction(
    incoming: str,
    mechanism_direction: str,
) -> str:
    """Propagate directional sign through a mechanism."""
    if incoming == "ambiguous" or mechanism_direction == "ambiguous":
        return "ambiguous"
    if incoming == "state_dependent" or mechanism_direction == "state_dependent":
        return "state_dependent"
    if incoming == mechanism_direction:
        return "positive"   # same sign → positive compound
    else:
        return "negative"   # opposite signs → negative compound
```

### 6.6 Sign Conflict Resolution

When multiple paths reach the same target with different directions, the effect is marked `"conflicted"`. The cascade output reports both sides:

```
ECB Policy Rate: CONFLICTED
  Channel 1: Oil → Inflation → Rate hike [positive, confidence=0.72]
  Channel 2: Oil → Recession → Rate cut [negative, confidence=0.55]
  Resolution: depends on which channel dominates (regime-dependent)
```

The cascade engine does NOT resolve conflicts automatically. It surfaces them for the analyst. Resolution requires either:
- Structural model (DABS) that can simulate the net effect
- Domain expertise about which channel dominates in the current regime
- Historical precedent (what happened last time?)

### 6.7 Cycle Handling — Time-Stepped Unrolling

For feedback loops, the cascade allows the same entity to appear at different time steps but not in the same path (the `if claim.target not in path` check prevents infinite loops within a single traversal).

To capture feedback dynamics, the cascade engine supports an optional iterative mode:

```python
def run_cascade_with_feedback(
    graph: CausalGraph,
    shock: Shock,
    n_iterations: int = 3,
    min_confidence: float = 0.15,
) -> list[dict[str, CascadeEffect]]:
    """
    Multi-pass cascade for feedback loops.

    Iteration 0: Direct shock propagation
    Iteration 1: Second-order effects (using iteration-0 effects as new shocks)
    Iteration 2: Third-order effects (feedback from iteration 1)
    ...

    Effects are time-layered: iteration k represents effects at time
    horizon k steps after the initial shock.

    Dampening: each iteration multiplies all confidences by 0.7 to
    reflect increasing uncertainty in feedback dynamics.
    """
    all_iterations = []
    current_shocks = [shock]
    feedback_decay = 0.7  # per-iteration decay for feedback uncertainty

    for iteration in range(n_iterations):
        iteration_effects = {}
        for s in current_shocks:
            effects = run_cascade(graph, s, min_confidence=min_confidence)
            for entity, eff in effects.items():
                if iteration > 0:
                    eff.confidence *= feedback_decay ** iteration
                if entity in iteration_effects:
                    # Merge with existing
                    merge_effects(iteration_effects[entity], eff)
                else:
                    iteration_effects[entity] = eff

        all_iterations.append(iteration_effects)

        # Convert significant effects into new shocks for next iteration
        current_shocks = [
            Shock(
                entity=entity,
                shock_type="cascade_feedback",
                description=f"Feedback from {eff.causal_path}",
                direction=eff.direction,
                magnitude_qualitative="derived",
                counterfactual="vs no-feedback baseline",
                regime_context=shock.regime_context,
            )
            for entity, eff in iteration_effects.items()
            if eff.confidence >= min_confidence * 2
            and eff.direction not in ("ambiguous", "conflicted")
        ]

    return all_iterations
```

The feedback mode runs 2-3 iterations. The 0.7 per-iteration decay ensures feedback effects are progressively discounted, preventing runaway amplification. This is analogous to how DABS's step function propagates effects forward in time, making the cascade engine a simplified preview of what DABS computes more rigorously.

---

## 7. DABS Integration

### 7.1 Abstraction Mapping

The causal graph operates on **macro concepts** (GDP, inflation, credit, policy rate). DABS operates on **agents** (360 firms, 2882 households, 5 banks). These are different levels of abstraction.

The integration operates at the **macro state variable level**, not the agent level:

| Causal graph edge | DABS mapping |
|---|---|
| Monetary Policy → Credit Allocation | Constrains which aggregate variables in `_phase_credit` can influence which |
| Oil Price → Firm Production | Constrains how the energy cost input in `_phase_production` couples to output |
| Bank Capital → Lending | Constrains the capital adequacy check in `_phase_credit` |

The causal graph does NOT constrain agent-to-agent GNN message passing. It constrains the **macro coupling equations** — which aggregate state variables can influence which other aggregate state variables in the 14-phase step function.

### 7.2 Structural Prior on Macro Coupling

Each causal edge provides a prior belief about whether a coupling term should exist in the step function:

```python
# In DABS training setup
for claim in causal_graph.claims.values():
    source_var = map_entity_to_state_variable(claim.source)
    target_var = map_entity_to_state_variable(claim.target)

    if source_var and target_var:
        # Initialise coupling weight from causal confidence
        coupling_prior[source_var][target_var] = claim.confidence

        # Set sign constraint from net direction
        if claim.net_direction == "positive":
            sign_constraint[source_var][target_var] = "positive"
        elif claim.net_direction == "negative":
            sign_constraint[source_var][target_var] = "negative"
        # "ambiguous" and "conflicted": no sign constraint, GNN learns freely
```

### 7.3 Regularisation

Three tiers of regularisation based on causal strength:

```python
for source, target, strength in causal_edges:
    w = coupling_weight[source][target]

    if strength == "strong_causal":
        # Penalty for weight being too small — don't ignore known causal channels
        loss += lambda_strong * relu(w_min - abs(w))

    elif strength == "correlation":
        # L1 penalty pushes toward zero — let the data decide
        loss += lambda_corr * abs(w)

    # weak_causal: no special regularisation, standard L2 decay only
```

### 7.4 Temporal Lag in Coupling

If a causal channel has a known lag (e.g., "monetary policy → output: 2-3 quarters"), the DABS step function should couple the *lagged* source variable to the current target:

```python
# Instead of: output_t = f(policy_t, ...)
# Use:        output_t = f(policy_{t-2}, policy_{t-3}, ...)
```

This requires storing lag information in the causal claim and mapping it to DABS's quarterly time steps.

---

## 8. Trading Application

### 8.1 What the Cascade Engine Provides

For a given shock, the cascade engine outputs a time-layered set of affected entities with:
- **Direction** (positive/negative/conflicted)
- **Confidence** (continuous score, product of edge confidences)
- **Time horizon** (when the effect materialises)
- **Number of independent channels** reaching this entity
- **Sign conflict flag** (if channels disagree)
- **Causal path** (for interpretability)

It does **NOT** provide magnitude estimates. Magnitude requires structural models or domain expertise.

### 8.2 What the Cascade Engine Does NOT Provide

- **What the market has already priced.** Without market data integration (forward curves, implied vol, CDS spreads), the cascade engine cannot compute tradeable alpha. The cascade output is a scenario analysis, not a trade recommendation.
- **Position sizing.** Even if an opportunity is identified, confidence levels of 0.4-0.6 at depth 3 mean positions should be small (Kelly criterion).
- **Instrument mapping.** Entity names must be manually mapped to tradeable instruments (ETFs, futures, options). This is analyst work.

### 8.3 Opportunity Identification

The cascade engine ranks effects by a composite "non-obviousness" score:

```python
def opportunity_score(effect: CascadeEffect, shock: Shock) -> float:
    """
    High score = high-confidence, non-obvious, tradeable.

    Non-obviousness is proxied by:
    1. Cross-market: effect is in a different asset class than the shock
    2. Time lag: longer lag = less likely priced immediately
    3. Path complexity: more hops through less-liquid intermediaries
    4. Sign conflict: if only one of the conflicting channels is salient
    """
    cross_market = 1.0 if effect.asset_class != shock.asset_class else 0.3
    time_lag_score = time_horizon_to_float(effect.time_horizon_min)  # 0-1, longer = higher
    path_complexity = min(1.0, (len(effect.causal_path) - 1) / 4)
    reinforcement = min(1.0, effect.n_incoming_channels / 3)  # multiple channels = more robust

    non_obviousness = 0.35 * cross_market + 0.30 * time_lag_score + 0.20 * path_complexity + 0.15 * reinforcement

    return effect.confidence * non_obviousness
```

### 8.4 Adversarial Considerations

The cascade engine's output is **necessary but not sufficient** for trading. Sophisticated macro funds (Bridgewater, Citadel, Brevan Howard) run similar scenario analysis with:
- Larger teams of domain experts
- Proprietary market data feeds
- Structural macro models (their version of DABS)
- Real-time market pricing integration

The edge of this system is NOT in discovering cascades that nobody else can see. The edge, if any, is in:
1. **Systematic coverage** — no human analyst traces all possible cascades; a computational tool does
2. **Speed** — running the cascade in seconds after a shock, before the market has fully priced second-order effects
3. **Cross-market** — connecting effects across asset classes that desk-specific analysts miss
4. **Regime-conditional** — surfacing effects that only activate under specific conditions

The system should be evaluated against "what would a skilled macro analyst produce in 30 minutes?" not against "what does no one know?"

---

## 9. Human-in-the-Loop

### 9.1 Review Interface

Extracted causal claims are presented for human review before entering the graph:

```
CLAIM: "Higher interest rates reduce investment"
Source: Mankiw Principles Ch.11 (cited, secondary)
Type: empirical_claim | Assertiveness: strong
Evidence: narrative | Directness: secondary_citation
Mechanism: cost_of_capital → lower NPV of projects → reduced capex
Direction: negative | Confidence: 0.38 (weak_causal)

[ACCEPT] [REJECT] [MODIFY] [FLAG AS CONTESTED]
```

### 9.2 Manual Claims

Domain experts can add claims not found in any document:

```bash
ontograph causal add \
  --source "Strait of Hormuz" --target "Qatar LNG Supply" \
  --mechanism "physical_transit_blockage" \
  --evidence-type accounting_identity \
  --direction negative \
  --assertiveness definitional \
  --note "~40% of Qatar LNG transits Strait"
```

### 9.3 Audit Trail

Every claim records:
- Who added/modified it (user or LLM extraction)
- When
- From which document (if extracted)
- Review status: `pending_review`, `accepted`, `rejected`, `contested`

---

## 10. Evaluation and Backtesting

### 10.1 Historical Episode Protocol

Select 3-5 well-documented historical episodes where the shock and consequences are known:

| Episode | Shock | Key consequences (ground truth) |
|---|---|---|
| 1973 Oil Embargo | OPEC embargo → oil +300% | Stagflation, recession, auto industry decline, petrochemical surge, dollar devaluation |
| 2008 GFC | Lehman default → credit freeze | Bank failures, recession, QE, sovereign debt crisis, unemployment surge |
| 2020 COVID | Lockdowns → demand collapse | Oil crash, supply chain disruption, fiscal expansion, healthcare surge, remote work shift |
| 2022 Russia-Ukraine | Energy supply disruption | EU gas crisis, inflation surge, ECB rate hikes, German industrial recession |

### 10.2 Backtesting Procedure

1. Build causal graph from literature available **before** the shock date
2. Define the shock in the cascade engine
3. Run cascade simulation
4. Compare predicted effects against actual observed outcomes

### 10.3 Evaluation Metrics

| Metric | Measures | How |
|---|---|---|
| **Directional hit rate** | Did the cascade correctly predict the sign of the effect? | % of affected entities where predicted direction matches actual |
| **Coverage** | Did the cascade identify the actual consequences? | % of known consequences that appear in cascade output |
| **False positive rate** | Did the cascade predict effects that didn't materialise? | % of cascade effects not observed in actuals |
| **Confidence calibration** | Are confidence scores well-calibrated? | Bin effects by confidence; check that 80%-confidence effects actually occur ~80% of the time |
| **Timing accuracy** | Did the time horizon predictions match? | Correlation between predicted and actual time-to-effect |
| **Baseline comparison** | Does the cascade engine beat naive alternatives? | Compare against: (a) random, (b) "everything goes down in a crisis", (c) human analyst with 30 minutes |

### 10.4 Calibration

Backtest results feed back into the system:
- If confidence scores are systematically too high → adjust thresholds
- If certain evidence types produce unreliable claims → adjust base scores
- If regime conditioning is important → prioritise structured conditions
- If sign conflicts are common → adjust the conflict resolution heuristic

---

## 11. Regime Conditioning

### 11.1 Regime Variables

A regime is defined by a set of macro state variables:

```python
REGIME_VARIABLES = {
    "capacity_utilization":   ["low", "normal", "high"],
    "policy_rate_regime":     ["zero_lower_bound", "normal", "tightening"],
    "leverage":               ["low", "moderate", "high"],
    "exchange_rate_regime":   ["fixed", "managed", "floating"],
    "inflation_regime":       ["deflation", "low", "moderate", "high"],
    "fiscal_stance":          ["austerity", "neutral", "expansionary"],
    "financial_stress":       ["calm", "elevated", "crisis"],
}
```

### 11.2 Regime-Tagged Causal Edges

Each causal mechanism can have regime-specific behaviour:

```
Edge: Oil Price → Inflation
  Mechanism: energy_cost_passthrough
    Regime {capacity_utilization: high} → direction: positive, confidence: 0.85
    Regime {capacity_utilization: low}  → direction: positive, confidence: 0.50
    (same direction but weaker pass-through when slack is large)

Edge: Monetary Policy → Output
  Mechanism: credit_channel
    Regime {policy_rate_regime: normal}            → direction: negative, confidence: 0.80
    Regime {policy_rate_regime: zero_lower_bound}  → direction: ambiguous, confidence: 0.30
    (monetary transmission breaks at the ZLB)
```

### 11.3 Cascade Engine Regime Awareness

When the cascade engine reaches an edge, it:
1. Checks the `regime_context` from the shock specification
2. Selects the regime-specific variant of each mechanism (best match)
3. If no regime match, uses the default (regime-agnostic) variant
4. If conditions are explicitly not met, skips the edge

This makes the cascade engine's output contingent on "what regime are we in right now?" — different regimes produce different cascades from the same shock.

---

## 12. Implementation Sequence

| Step | What | Effort | Dependencies |
|------|------|--------|-------------|
| 1 | Data model: `CausalClaim`, `CausalMechanism`, `CausalCondition`, `CausalGraph` | Low | None |
| 2 | Evidence taxonomy: constants + base scores | Low | None |
| 3 | Confidence scoring function | Low | Step 1, 2 |
| 4 | Causal extraction prompts (Stage A + B) | Medium | Step 1 |
| 5 | CLI: `ontograph causal ingest`, `ontograph causal add` | Low | Step 1, 4 |
| 6 | Cascade engine: max-confidence priority traversal | Medium | Step 1 |
| 7 | Cascade engine: condition checking | Medium | Step 6 |
| 8 | Cascade engine: sign conflict detection + time-stepped feedback | Medium | Step 6 |
| 9 | Viewer: causal layer toggle + cascade mode | Medium | Step 1, 6 |
| 10 | Backtesting framework: historical episode evaluation | Medium | Step 6 |
| 11 | Human review interface (CLI or viewer-based) | Medium | Step 5 |
| 12 | Cross-document aggregation + contradiction detection | Medium | Step 4 |
| 13 | Regime conditioning: structured conditions + regime-tagged edges | High | Step 7 |
| 14 | DABS integration: macro coupling prior from causal graph | High | Step 1, DABS refactor |
| 15 | Trading: opportunity scoring + scenario panel in viewer | Medium | Step 6, 9 |

**V1 milestone (steps 1-8):** A working causal extraction + cascade engine with direction-only propagation, condition checking, sign conflict detection, and feedback loops. No viewer, no DABS integration, no trading features.

**V2 milestone (steps 9-12):** Viewer integration, backtesting, human review, cross-document aggregation.

**V3 milestone (steps 13-15):** Regime conditioning, DABS integration, trading application.

---

## 13. Open Questions (Deferred)

These are genuinely unresolved and deferred to future work:

1. **Magnitude propagation.** V1 propagates direction only. When should magnitude be added? Only after backtesting validates directional accuracy?

2. **General equilibrium vs partial equilibrium.** The cascade engine is sequential (partial equilibrium). When multiple feedback loops interact, the net effect may differ from what sequential propagation predicts. Is the time-stepped feedback mode sufficient, or do we need DABS itself for this?

3. **Market pricing integration.** For trading, we need to compare cascade-implied to market-implied effects. What data feeds are needed? Forward curves, implied vol, CDS? How to formalise "the market hasn't priced this"?

4. **Source quality weighting.** A DiD from the *AER* is not equivalent to a DiD from a working paper. Do we track journal quality? Citation count? This risks academic gatekeeping but ignoring it risks garbage-in.

5. **Causal graph as formal SCM.** The proper mathematical framework is Pearl's structural causal model. Mapping `CausalClaim` to a formal SCM enables `do()`-calculus interventions and d-separation checks. Is the formalism worth the complexity?
