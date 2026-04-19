"""Phase C — AST-based entity extractor.

Walks `.py` files under a repository root, parses them with the stdlib `ast`
module, and emits grounded `Entity` objects for nodes matching patterns
declared in `schema.extraction_hints.ast_patterns`.

Scope (discipline-v1 §6 Phase C): entities only. Relations from the AST are
deferred to Phase D because schema-valid DABS relations (`applied_during`,
`executes_after`, …) encode ordering and domain/range constraints that AST
structure alone does not carry.

Pattern grammar (in `ast_patterns[i].pattern`):

    def NAME           — FunctionDef with NAME glob-match
    class NAME         — ClassDef with NAME glob-match (any or no base)
    class NAME(BASE)   — ClassDef with NAME AND first-base glob-match
    NAME =             — Assign with a single Name target glob-match

Whitespace around `=`, `(`, `)` is tolerated. Globs are `fnmatch`-style.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import CodeAnchor, Entity, Relation
from .schema import OntologySchema


# ── Pattern parsing ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _ParsedPattern:
    kind: str                       # "func" | "class" | "assign"
    name_glob: str                  # glob against symbol name
    base_glob: str | None = None    # glob against first base, class-with-base only
    entity_type: str = ""
    subtype: str | None = None      # optional spec §4.1.1 subtype tag


_FUNC_RE = re.compile(r"^def\s+(\S+)\s*$")
_CLASS_BASE_RE = re.compile(r"^class\s+(\S+?)\s*\(\s*(.+?)\s*\)\s*$")
_CLASS_RE = re.compile(r"^class\s+(\S+)\s*$")
_ASSIGN_RE = re.compile(r"^(\S+)\s*=\s*$")


def _parse_pattern(raw: str, entity_type: str, subtype: str | None = None) -> _ParsedPattern | None:
    s = raw.strip()
    m = _FUNC_RE.match(s)
    if m:
        return _ParsedPattern("func", m.group(1), None, entity_type, subtype)
    m = _CLASS_BASE_RE.match(s)
    if m:
        return _ParsedPattern("class", m.group(1), m.group(2), entity_type, subtype)
    m = _CLASS_RE.match(s)
    if m:
        return _ParsedPattern("class", m.group(1), None, entity_type, subtype)
    m = _ASSIGN_RE.match(s)
    if m:
        return _ParsedPattern("assign", m.group(1), None, entity_type, subtype)
    return None


def _compile_patterns(schema: OntologySchema) -> list[_ParsedPattern]:
    raw_list = schema.extraction_hints.get("ast_patterns", []) or []
    out: list[_ParsedPattern] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        pat = item.get("pattern")
        etype = item.get("entity_type")
        if not pat or not etype:
            continue
        subtype_raw = item.get("subtype")
        subtype = str(subtype_raw) if subtype_raw else None
        parsed = _parse_pattern(str(pat), str(etype), subtype)
        if parsed is not None:
            out.append(parsed)
    return out


# ── Repo traversal ─────────────────────────────────────────────────────

def _iter_py_files(
    repo_root: Path,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
) -> Iterable[Path]:
    for p in sorted(repo_root.rglob("*.py")):
        rel = p.relative_to(repo_root).as_posix()
        if include_paths and not any(
            rel == inc or rel.startswith(inc.rstrip("/") + "/")
            for inc in include_paths
        ):
            continue
        if exclude_paths and any(
            rel == exc or rel.startswith(exc.rstrip("/") + "/")
            for exc in exclude_paths
        ):
            continue
        yield p


# ── Node → pattern matching ────────────────────────────────────────────

def _first_base_name(cls: ast.ClassDef) -> str | None:
    """Render the first base of a ClassDef to a string using ast.unparse.

    Returns None if the class has no bases. Handles `nn.Module`, `MyClass`,
    `module.Sub.Base`, etc. Keyword args (metaclass=…) are ignored.
    """
    if not cls.bases:
        return None
    try:
        return ast.unparse(cls.bases[0])
    except Exception:
        return None


def _match_functiondef(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    patterns: list[_ParsedPattern],
) -> list[_ParsedPattern]:
    return [p for p in patterns if p.kind == "func" and fnmatch.fnmatchcase(node.name, p.name_glob)]


def _match_classdef(node: ast.ClassDef, patterns: list[_ParsedPattern]) -> list[_ParsedPattern]:
    matched: list[_ParsedPattern] = []
    base = _first_base_name(node)
    for p in patterns:
        if p.kind != "class":
            continue
        if not fnmatch.fnmatchcase(node.name, p.name_glob):
            continue
        if p.base_glob is None:
            matched.append(p)
            continue
        if base is None:
            continue
        if fnmatch.fnmatchcase(base, p.base_glob):
            matched.append(p)
    return matched


def _match_assign(node: ast.Assign, patterns: list[_ParsedPattern]) -> list[tuple[_ParsedPattern, str]]:
    """Return (pattern, symbol) pairs — assignments can have multiple Name targets."""
    out: list[tuple[_ParsedPattern, str]] = []
    for tgt in node.targets:
        if not isinstance(tgt, ast.Name):
            continue
        for p in patterns:
            if p.kind == "assign" and fnmatch.fnmatchcase(tgt.id, p.name_glob):
                out.append((p, tgt.id))
    return out


# ── Entity assembly ────────────────────────────────────────────────────

def _make_entity(
    name: str,
    entity_type: str,
    repo_root: Path,
    file_path: Path,
    line: int,
    subtype: str | None = None,
) -> Entity:
    rel_posix = file_path.relative_to(repo_root).as_posix()
    anchor = CodeAnchor(repo=repo_root.name, path=rel_posix, line=line, symbol=name)
    metadata: dict[str, object] = {}
    if subtype:
        metadata["subtype"] = subtype
    return Entity(
        name=name,
        entity_type=entity_type,
        code_anchors=[anchor],
        abstract=False,
        metadata=metadata,
    )


# ── Public entry point ─────────────────────────────────────────────────

def extract_from_repo(
    repo_root: Path,
    schema: OntologySchema,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Walk `repo_root`, parse each `.py` file, emit grounded Entity objects
    for AST nodes matching `schema.extraction_hints.ast_patterns`.

    Phase C emits no relations — Phase D merges entities with text-extracted
    relations and applies schema defaults.
    """
    repo_root = Path(repo_root).resolve()
    patterns = _compile_patterns(schema)
    entities: list[Entity] = []

    if not patterns:
        return entities, []

    for py in _iter_py_files(repo_root, include_paths, exclude_paths):
        try:
            tree = ast.parse(py.read_text(), filename=str(py))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for p in _match_functiondef(node, patterns):
                    entities.append(_make_entity(node.name, p.entity_type, repo_root, py, node.lineno, p.subtype))
            elif isinstance(node, ast.ClassDef):
                for p in _match_classdef(node, patterns):
                    entities.append(_make_entity(node.name, p.entity_type, repo_root, py, node.lineno, p.subtype))
            elif isinstance(node, ast.Assign):
                for p, sym in _match_assign(node, patterns):
                    entities.append(_make_entity(sym, p.entity_type, repo_root, py, node.lineno, p.subtype))

    return entities, []
