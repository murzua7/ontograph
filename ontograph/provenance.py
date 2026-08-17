"""Extraction-provenance metadata for knowledge graphs.

Every KG should carry enough metadata to be *reproducible*: which prompts
were used, which schema, which model, which ontograph version, when.
Without this, a KG is an artefact of unknown lineage — if the extractor
prompt changes or the schema is edited, you cannot tell whether two graphs
were produced by the same pipeline.

`extraction_provenance()` produces a canonical dict of hashes and version
strings that gets attached to `KnowledgeGraph.extraction`.

Hashing conventions
-------------------
- `prompt_hash` : sha256 of the concatenated extractor prompt templates
  (system + user prompt). Includes schema-free and consolidation prompts
  when the relevant mode is active.
- `schema_hash` : sha256 of the schema's canonical JSON representation
  (entity types, relation types, validation rules — sorted keys). `None`
  when running schema-free before consolidation.
- `model`       : model identifier (e.g. "claude-sonnet-4-5").
- `backend`     : backend label ("anthropic", "openai", "ollama",
  "claude-code", "heuristic", "batch").
- `ontograph_version` : `ontograph.__version__`.
- `extracted_at`      : ISO-8601 UTC timestamp.
- `mode`              : extraction mode ("heuristic", "llm", "schema-free",
  "dual-extract", "batch-in", "batch-out").

The schema_hash is stable across whitespace / comment changes in the YAML,
because we hash the canonical dataclass view, not the source text.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from ontograph import __version__ as ONTOGRAPH_VERSION


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_schema_dict(schema) -> dict[str, Any]:
    """Return a canonical, sort-stable dict view of an OntologySchema.

    Excludes display-only fields (color) so that cosmetic edits do not
    change the schema hash. Includes default_sign/lag/form/edge_class —
    those directly influence extraction behaviour.
    """
    if schema is None:
        return {}
    entity_types = {}
    for name, edef in sorted(schema.entity_types.items()):
        entity_types[name] = {
            "description": edef.description,
            "subtypes": list(edef.subtypes),
        }
    relation_types = {}
    for name, rdef in sorted(schema.relation_types.items()):
        relation_types[name] = {
            "description": rdef.description,
            "domain": list(rdef.domain),
            "range": list(rdef.range),
            "symmetric": rdef.symmetric,
            "default_sign": rdef.default_sign,
            "default_lag": rdef.default_lag,
            "default_form": rdef.default_form,
            "default_edge_class": rdef.default_edge_class,
        }
    return {
        "name": schema.name,
        "version": schema.version,
        "description": schema.description,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "validation": dict(sorted((schema.validation or {}).items())),
    }


def schema_hash(schema) -> str | None:
    """sha256 of the canonical schema dict. None if schema is None."""
    if schema is None:
        return None
    canon = canonical_schema_dict(schema)
    return _sha256_hex(json.dumps(canon, sort_keys=True, ensure_ascii=False))


def prompt_hash_for_mode(mode: str) -> str:
    """sha256 of the concatenated extractor prompt templates for `mode`.

    Imports are deferred so this module doesn't pull in LLM plumbing for
    callers that only want schema hashing.
    """
    from ontograph.llm_extractor import (
        ENTITY_SYSTEM, ENTITY_PROMPT,
        RELATION_SYSTEM, RELATION_PROMPT,
        SCHEMA_FREE_ENTITY_SYSTEM, SCHEMA_FREE_ENTITY_PROMPT,
        SCHEMA_FREE_RELATION_SYSTEM, SCHEMA_FREE_RELATION_PROMPT,
        CONSOLIDATION_SYSTEM,
        CONSOLIDATION_ENTITY_PROMPT, CONSOLIDATION_RELATION_PROMPT,
    )
    if mode == "schema-free":
        parts = [
            SCHEMA_FREE_ENTITY_SYSTEM, SCHEMA_FREE_ENTITY_PROMPT,
            SCHEMA_FREE_RELATION_SYSTEM, SCHEMA_FREE_RELATION_PROMPT,
            CONSOLIDATION_SYSTEM,
            CONSOLIDATION_ENTITY_PROMPT, CONSOLIDATION_RELATION_PROMPT,
        ]
    elif mode in {"llm", "batch-in", "batch-out", "dual-extract"}:
        parts = [ENTITY_SYSTEM, ENTITY_PROMPT, RELATION_SYSTEM, RELATION_PROMPT]
    else:
        # heuristic / unknown: empty prompts
        return ""
    return _sha256_hex("\n---\n".join(parts))


def extraction_provenance(
    *,
    mode: str,
    schema=None,
    model: str | None = None,
    backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical extraction-provenance dict.

    Args:
        mode:    one of "heuristic", "llm", "schema-free", "dual-extract",
                 "batch-in", "batch-out".
        schema:  OntologySchema instance (or None for schema-free/heuristic).
        model:   LLM model id (None for heuristic mode).
        backend: LLM backend label (None for heuristic mode).
        extra:   extra keys to merge into the returned dict (e.g. `doc_hash`).
    """
    out: dict[str, Any] = {
        "mode": mode,
        "ontograph_version": ONTOGRAPH_VERSION,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_hash": schema_hash(schema),
        "prompt_hash": prompt_hash_for_mode(mode),
        "model": model,
        "backend": backend,
    }
    if extra:
        out.update(extra)
    return out
