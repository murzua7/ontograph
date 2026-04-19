"""Phase C — AST extractor tests (TDD red).

Covers the entities-only contract from discipline-v1 §4.1.1 and §6 (Phase C):

    extract_from_repo(
        repo_root: Path,
        schema: OntologySchema,
        *,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> tuple[list[Entity], list[Relation]]

Relations are deferred to Phase D merge per advisor scope. Patterns come from
`schema.extraction_hints.ast_patterns` (a list of `{pattern, entity_type}` dicts).

Pattern grammar (4 kinds + fnmatch globs on NAME/BASE):
    def NAME           → FunctionDef with matching name
    class NAME         → ClassDef with matching name, any base
    class NAME (BASE)  → ClassDef with matching name AND matching first base
    NAME =             → Assign with single Name target matching
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


FIXTURES = Path(__file__).parent / "fixtures"


# ── helpers ────────────────────────────────────────────────────────────

def _write_schema_with_patterns(tmp_path: Path, patterns: list[dict]) -> Path:
    """Write a minimal schema with the given ast_patterns list and return its path."""
    p = tmp_path / "sch.yaml"
    p.write_text(yaml.safe_dump({
        "name": "sch",
        "entity_types": {
            "phase": {"description": "p"},
            "learned_operator": {"description": "lo"},
            "smooth_operator": {"description": "so"},
            "assertion": {"description": "a"},
            "mechanism": {"description": "m"},
        },
        "extraction_hints": {"ast_patterns": patterns},
    }))
    return p


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a synthetic repo under tmp_path / 'repo' with the given files."""
    root = tmp_path / "repo"
    root.mkdir()
    for relpath, content in files.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    return root


# ── 1. Function-def patterns (def NAME) ────────────────────────────────

class TestFunctionDefPatterns:
    def test_phase_function_extracted_as_phase_entity(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": (
                "def _phase_production(state):\n"
                "    return state\n"
                "\n"
                "def _phase_clearing(state):\n"
                "    return state\n"
                "\n"
                "def unrelated():\n"
                "    pass\n"
            ),
        })
        entities, relations = extract_from_repo(repo, schema)
        names = sorted(e.name for e in entities)
        assert names == ["_phase_clearing", "_phase_production"]
        assert all(e.entity_type == "phase" for e in entities)

    def test_assertion_function_extracted(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _assert_*", "entity_type": "assertion"},
        ])))
        repo = _make_repo(tmp_path, {
            "checks.py": (
                "def _assert_balance_sheet(state):\n"
                "    pass\n"
                "\n"
                "def _assert_positive(x):\n"
                "    pass\n"
            ),
        })
        entities, _ = extract_from_repo(repo, schema)
        assert {e.name for e in entities} == {"_assert_balance_sheet", "_assert_positive"}

    def test_unrelated_function_not_extracted(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": "def helper():\n    return 1\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        assert entities == []


# ── 2. Class patterns with and without base ────────────────────────────

class TestClassPatterns:
    def test_class_with_base_matches_base(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "class *(nn.Module)", "entity_type": "learned_operator"},
        ])))
        repo = _make_repo(tmp_path, {
            "ops.py": (
                "import torch.nn as nn\n"
                "\n"
                "class Adapter(nn.Module):\n"
                "    pass\n"
                "\n"
                "class PlainClass:\n"
                "    pass\n"
                "\n"
                "class OtherBase(SomeOther):\n"
                "    pass\n"
            ),
        })
        entities, _ = extract_from_repo(repo, schema)
        names = {e.name for e in entities}
        assert names == {"Adapter"}
        assert next(iter(entities)).entity_type == "learned_operator"

    def test_class_without_base_in_pattern_matches_any(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "class *Adapter", "entity_type": "learned_operator"},
        ])))
        repo = _make_repo(tmp_path, {
            "ops.py": (
                "class InitialAdapter(nn.Module):\n"
                "    pass\n"
                "\n"
                "class FinalAdapter:\n"
                "    pass\n"
                "\n"
                "class Other:\n"
                "    pass\n"
            ),
        })
        entities, _ = extract_from_repo(repo, schema)
        names = {e.name for e in entities}
        assert names == {"InitialAdapter", "FinalAdapter"}


# ── 3. Assignment patterns (NAME =) ────────────────────────────────────

class TestAssignmentPatterns:
    def test_smooth_assignment_extracted(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "smooth_* =", "entity_type": "smooth_operator"},
        ])))
        repo = _make_repo(tmp_path, {
            "relax.py": (
                "smooth_relu = lambda x: x\n"
                "smooth_max = lambda a, b: a\n"
                "hard_step = lambda x: 1 if x > 0 else 0\n"
            ),
        })
        entities, _ = extract_from_repo(repo, schema)
        names = {e.name for e in entities}
        assert names == {"smooth_relu", "smooth_max"}
        assert all(e.entity_type == "smooth_operator" for e in entities)


# ── 4. CodeAnchor population ───────────────────────────────────────────

class TestCodeAnchorGrounding:
    def test_every_extracted_entity_has_code_anchor(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": "def _phase_a(): pass\n\ndef _phase_b(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        for e in entities:
            assert e.is_grounded()
            assert len(e.code_anchors) == 1
            anchor = e.code_anchors[0]
            assert anchor.repo == repo.name
            assert anchor.path == "sim.py"
            assert anchor.line >= 1
            assert anchor.symbol == e.name

    def test_code_anchor_path_is_posix_relative(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "pkg/sub/mod.py": "def _phase_x(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        assert len(entities) == 1
        # Always forward slashes, regardless of host OS — makes anchors
        # platform-portable across repo checkouts.
        assert entities[0].code_anchors[0].path == "pkg/sub/mod.py"

    def test_code_anchor_line_matches_def_line(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": (
                "# blank line 1\n"
                "# blank line 2\n"
                "def _phase_a():\n"          # line 3
                "    pass\n"
                "\n"
                "def _phase_b():\n"          # line 6
                "    pass\n"
            ),
        })
        entities, _ = extract_from_repo(repo, schema)
        by_name = {e.name: e for e in entities}
        assert by_name["_phase_a"].code_anchors[0].line == 3
        assert by_name["_phase_b"].code_anchors[0].line == 6


# ── 5. Path filtering ──────────────────────────────────────────────────

class TestPathFiltering:
    def test_include_paths_limits_scan(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim/core.py": "def _phase_a(): pass\n",
            "tests/test_core.py": "def _phase_b(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema, include_paths=["sim"])
        assert {e.name for e in entities} == {"_phase_a"}

    def test_exclude_paths_skips_dir(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim/core.py": "def _phase_a(): pass\n",
            "tests/test_core.py": "def _phase_b(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema, exclude_paths=["tests"])
        assert {e.name for e in entities} == {"_phase_a"}

    def test_only_py_files_scanned(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": "def _phase_a(): pass\n",
            "README.md": "def _phase_fake(): pass\n",
            "config.yaml": "def _phase_yaml(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        assert {e.name for e in entities} == {"_phase_a"}


# ── 6. Empty / missing hints ───────────────────────────────────────────

class TestEmptyHints:
    def test_no_ast_patterns_yields_no_entities(self, tmp_path):
        """A schema without `ast_patterns` is not a failure; just nothing to extract."""
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema("base")   # base has no ast_patterns
        repo = _make_repo(tmp_path, {"sim.py": "def _phase_a(): pass\n"})
        entities, relations = extract_from_repo(repo, schema)
        assert entities == []
        assert relations == []

    def test_empty_ast_patterns_list_is_noop(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [])))
        repo = _make_repo(tmp_path, {"sim.py": "def _phase_a(): pass\n"})
        entities, _ = extract_from_repo(repo, schema)
        assert entities == []


# ── 7. Relations deferred ──────────────────────────────────────────────

class TestRelationsDeferredToPhaseD:
    """Phase C is entities-only per advisor scope.

    `applied_during` is mechanism→phase (not the phase→submech direction AST
    would suggest); `executes_after` needs phase ordering AST cannot provide.
    Relations come from Phase D merge with text-extraction + schema defaults."""

    def test_no_relations_emitted(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
            {"pattern": "class *(nn.Module)", "entity_type": "learned_operator"},
        ])))
        repo = _make_repo(tmp_path, {
            "sim.py": (
                "class Adapter(nn.Module):\n"
                "    pass\n"
                "\n"
                "def _phase_a(state, adapter):\n"
                "    return adapter(state)\n"
            ),
        })
        _, relations = extract_from_repo(repo, schema)
        assert relations == []


# ── 8. Entity provenance / grounding attributes ────────────────────────

class TestEntityAttributes:
    def test_extracted_entity_not_abstract(self, tmp_path):
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {"sim.py": "def _phase_a(): pass\n"})
        entities, _ = extract_from_repo(repo, schema)
        assert len(entities) == 1
        assert entities[0].abstract is False

    def test_duplicate_definitions_across_files_produce_two_anchors(self, tmp_path):
        """Same symbol defined in two files — extractor returns two entities;
        downstream KnowledgeGraph.add_entity merges anchors by identity tuple."""
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "a.py": "def _phase_dup(): pass\n",
            "b.py": "def _phase_dup(): pass\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        assert len(entities) == 2
        assert {e.code_anchors[0].path for e in entities} == {"a.py", "b.py"}


# ── 9. Syntax-error tolerance ──────────────────────────────────────────

class TestSyntaxErrorTolerance:
    def test_broken_file_skipped_not_fatal(self, tmp_path):
        """A single unparseable .py file must not abort the whole scan."""
        from ontograph.ast_extractor import extract_from_repo
        from ontograph.schema import load_schema
        schema = load_schema(str(_write_schema_with_patterns(tmp_path, [
            {"pattern": "def _phase_*", "entity_type": "phase"},
        ])))
        repo = _make_repo(tmp_path, {
            "good.py": "def _phase_a(): pass\n",
            "broken.py": "def _phase_b(:  # unclosed paren\n",
        })
        entities, _ = extract_from_repo(repo, schema)
        assert {e.name for e in entities} == {"_phase_a"}
