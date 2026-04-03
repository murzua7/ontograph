"""Causal graph data models and evidence taxonomy.

The causal graph is a layer on top of the knowledge graph that adds
causal semantics: evidence quality, mechanism specificity, regime
conditioning, and direction of effect. Stored separately as
causal_graph.json, referencing the same entity names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Evidence taxonomy — base scores for 18 evidence types
# ---------------------------------------------------------------------------

EVIDENCE_TAXONOMY: dict[str, float] = {
    "accounting_identity": 1.00,
    "institutional_mechanism": 0.90,
    "meta_analysis": 0.90,
    "rct": 0.90,
    "natural_experiment": 0.80,
    "synthetic_control": 0.75,
    "iv": 0.70,
    "did": 0.70,
    "rdd": 0.70,
    "event_study": 0.65,
    "historical_precedent": 0.60,
    "panel_fixed_effects": 0.50,
    "structural_estimation": 0.50,
    "reduced_form_regression": 0.40,
    "granger_causality": 0.30,
    "model_simulation": 0.25,
    "market_implied": 0.20,
    "narrative": 0.15,
}

ASSERTIVENESS_SCORES: dict[str, float] = {
    "definitional": 1.00,
    "strong": 0.80,
    "moderate": 0.50,
    "hedged": 0.20,
}

MECHANISM_SCORES: dict[str, float] = {
    "quantified": 1.00,       # named channel + elasticity/coefficient from data
    "articulated": 0.80,      # named channel + plausible economic logic
    "vague": 0.50,            # named channel, vague description
    "implied": 0.30,          # mechanism implied but not stated
    "none": 0.00,             # no mechanism identified
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CausalCondition:
    """A regime condition that gates a causal channel."""
    variable: str
    operator: Literal[">", "<", ">=", "<=", "==", "in_range", "regime_is"]
    threshold: str
    description: str = ""
    currently_met: bool | None = None


@dataclass
class CausalMechanism:
    """A single transmission channel within a causal claim."""
    name: str
    description: str

    # Effect characteristics
    direction: Literal["positive", "negative", "ambiguous", "state_dependent"] = "ambiguous"
    nonlinear: bool = False

    # Elasticity (optional)
    elasticity_range: tuple[float, float] | None = None
    elasticity_horizon: Literal["short_run", "long_run"] | None = None
    elasticity_source: str | None = None

    # Temporal
    time_lag_min: str | None = None
    time_lag_max: str | None = None

    # Conditions under which this channel is active
    conditions: list[CausalCondition] = field(default_factory=list)


@dataclass
class CausalClaim:
    """A causal assertion connecting a cause entity to an effect entity."""
    id: str
    source: str
    target: str
    mechanisms: list[CausalMechanism] = field(default_factory=list)

    # Causal type
    causal_type: Literal["micro_mechanical", "macro_statistical"] = "macro_statistical"

    # Epistemology
    evidence_type: str = "narrative"
    evidence_count: int = 1
    evidence_directness: Literal["primary", "cited", "secondary_citation"] = "cited"
    claim_assertiveness: Literal["hedged", "moderate", "strong", "definitional"] = "moderate"

    # Confidence (computed)
    confidence: float = 0.0
    strength: Literal["correlation", "weak_causal", "strong_causal"] = "correlation"

    # Aggregate direction
    net_direction: Literal["positive", "negative", "ambiguous", "conflicted"] = "ambiguous"
    direction_contested: bool = False

    # Scope
    geographic_scope: str = "global"
    geographic_entities: list[str] = field(default_factory=list)
    temporal_scope: str | None = None

    # Reversibility
    reversibility: Literal["permanent", "temporary", "unknown"] = "unknown"

    # Provenance
    sources: list[Any] = field(default_factory=list)  # list[Provenance]
    claim_text: str = ""
    counterfactual_stated: bool = False
    extracted_by: Literal["llm", "manual", "heuristic"] = "llm"

    # Contestation
    contradicted_by: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    # Trading metadata (optional)
    asset_classes: list[str] = field(default_factory=list)
    sector_codes: list[str] = field(default_factory=list)


@dataclass
class CausalGraph:
    """Container for causal claims, stored separately from the knowledge graph."""
    claims: dict[str, CausalClaim] = field(default_factory=dict)
    _entity_list: list[str] = field(default_factory=list)
    schema_name: str = "causal"
    source_knowledge_graph: str | None = None
    build_date: str = ""
    regime_context: dict[str, str] = field(default_factory=dict)

    # Lazy index for outgoing_claims()
    _source_index: dict[str, list[str]] | None = field(
        default=None, repr=False, compare=False,
    )

    @property
    def entities(self) -> set[str]:
        return set(self._entity_list)

    @entities.setter
    def entities(self, value: set[str] | list[str]) -> None:
        self._entity_list = sorted(value)

    def add_claim(self, claim: CausalClaim) -> None:
        self.claims[claim.id] = claim
        self._source_index = None  # invalidate
        # Track referenced entities
        for name in (claim.source, claim.target):
            if name not in self._entity_list:
                self._entity_list.append(name)
                self._entity_list.sort()

    def _build_source_index(self) -> None:
        self._source_index = {}
        for cid, claim in self.claims.items():
            self._source_index.setdefault(claim.source, []).append(cid)

    def outgoing_claims(self, entity: str) -> list[CausalClaim]:
        """All claims where source == entity."""
        if self._source_index is None:
            self._build_source_index()
        return [self.claims[cid] for cid in self._source_index.get(entity, [])]

    def to_json(self) -> str:
        def _ser(obj: Any) -> Any:
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        data = {
            "claims": {cid: asdict(c) for cid, c in self.claims.items()},
            "entities": sorted(self._entity_list),
            "schema_name": self.schema_name,
            "source_knowledge_graph": self.source_knowledge_graph,
            "build_date": self.build_date,
            "regime_context": self.regime_context,
        }
        return json.dumps(data, default=_ser, indent=2)

    @classmethod
    def from_json(cls, data: str) -> CausalGraph:
        raw = json.loads(data)
        cg = cls(
            schema_name=raw.get("schema_name", "causal"),
            source_knowledge_graph=raw.get("source_knowledge_graph"),
            build_date=raw.get("build_date", ""),
            regime_context=raw.get("regime_context", {}),
        )
        cg._entity_list = raw.get("entities", [])

        for cid, cdata in raw.get("claims", {}).items():
            mechanisms = []
            for mdata in cdata.get("mechanisms", []):
                conditions = [
                    CausalCondition(**cond)
                    for cond in mdata.pop("conditions", [])
                ]
                # Handle elasticity_range: list -> tuple
                er = mdata.pop("elasticity_range", None)
                if er is not None and isinstance(er, list):
                    er = tuple(er)
                mechanisms.append(CausalMechanism(
                    **{k: v for k, v in mdata.items()
                       if k in CausalMechanism.__dataclass_fields__},
                    conditions=conditions,
                    elasticity_range=er,
                ))

            # Reconstruct provenance objects
            from ontograph.models import Provenance
            sources = [
                Provenance(**s) if isinstance(s, dict) else s
                for s in cdata.get("sources", [])
            ]

            # Build claim, filtering to known fields
            claim_fields = {
                k: v for k, v in cdata.items()
                if k in CausalClaim.__dataclass_fields__
                and k not in ("mechanisms", "sources")
            }
            claim = CausalClaim(
                **claim_fields,
                mechanisms=mechanisms,
                sources=sources,
            )
            cg.claims[cid] = claim

        return cg


@dataclass
class Shock:
    """Input specification for cascade simulation."""
    entity: str
    shock_type: str
    description: str
    direction: Literal["positive", "negative"] = "negative"
    magnitude_qualitative: str = "moderate"
    counterfactual: str = "vs pre-shock baseline"
    date: str | None = None
    regime_context: dict[str, str] = field(default_factory=dict)


@dataclass
class CascadeEffect:
    """Cascade output: the effect on a single entity."""
    entity: str
    direction: Literal["positive", "negative", "ambiguous", "conflicted"] = "ambiguous"
    confidence: float = 0.0
    time_horizon_min: str = "immediate"
    time_horizon_max: str = "immediate"
    causal_path: list[str] = field(default_factory=list)
    all_paths: list[list[str]] = field(default_factory=list)
    n_incoming_channels: int = 1
    incoming_directions: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim_id(source: str, target: str, evidence_type: str, claim_text: str = "") -> str:
    """Generate a deterministic claim ID from key fields."""
    raw = f"{source}|{target}|{evidence_type}|{claim_text}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    slug = f"{source}_{target}".lower().replace(" ", "_")[:40]
    return f"{slug}_{h}"
