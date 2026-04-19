"""Phase B — Schema extensions for ontology-extraction-discipline-v1.

Tests written before implementation (TDD).

Covers:
- `extends:` BFS merge (child overrides parent; multi-level chains)
- Cycle detection in extends chain
- `validation:` block on OntologySchema
- `default_sign`, `default_lag`, `default_edge_class`, `default_form` on RelationTypeDef
- Backward compatibility: schemas without extends still load flat
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


# ── Existing schemas unchanged (regression) ──

class TestFlatSchemasStillWork:
    def test_base_still_loads(self):
        from ontograph.schema import load_schema
        s = load_schema("base")
        assert s.name == "base"
        assert "concept" in s.entity_types

    def test_economics_still_loads(self):
        from ontograph.schema import load_schema
        s = load_schema("economics")
        assert "agent" in s.entity_types
        assert "feeds_into" in s.relation_types

    def test_load_schema_no_extends_is_no_op(self):
        """A schema without `extends:` loads exactly as before."""
        from ontograph.schema import load_schema
        s = load_schema("base")
        # base.yaml has 5 entity types and 5 relation types
        assert len(s.entity_types) == 5
        assert len(s.relation_types) == 5


# ── New extends: behaviour ──

class TestExtendsMerge:
    def _write_yaml(self, tmp_path: Path, name: str, body: dict) -> Path:
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(body))
        return p

    def test_single_extends_inherits_parent(self, tmp_path):
        from ontograph.schema import load_schema
        parent = self._write_yaml(tmp_path, "parent", {
            "name": "parent",
            "entity_types": {
                "foo": {"description": "parent-foo", "color": "#111111"},
            },
            "relation_types": {
                "rel_a": {"description": "parent rel"},
            },
        })
        child = self._write_yaml(tmp_path, "child", {
            "name": "child",
            "extends": str(parent),
            "entity_types": {
                "bar": {"description": "child-bar"},
            },
        })
        s = load_schema(str(child))
        assert "foo" in s.entity_types      # inherited
        assert "bar" in s.entity_types      # own
        assert "rel_a" in s.relation_types  # inherited
        assert s.name == "child"

    def test_child_overrides_parent_field(self, tmp_path):
        """Child's entity_type definition replaces parent's when names collide."""
        from ontograph.schema import load_schema
        parent = self._write_yaml(tmp_path, "parent", {
            "name": "parent",
            "entity_types": {"foo": {"description": "parent", "color": "#111111"}},
        })
        child = self._write_yaml(tmp_path, "child", {
            "name": "child",
            "extends": str(parent),
            "entity_types": {"foo": {"description": "child override", "color": "#222222"}},
        })
        s = load_schema(str(child))
        assert s.entity_types["foo"].description == "child override"
        assert s.entity_types["foo"].color == "#222222"

    def test_extends_chain_resolves(self, tmp_path):
        """A -> B -> C where A extends B and B extends C: loading A picks up C's types."""
        from ontograph.schema import load_schema
        grand = self._write_yaml(tmp_path, "grand", {
            "name": "grand",
            "entity_types": {"g_ent": {"description": "grand"}},
        })
        mid = self._write_yaml(tmp_path, "mid", {
            "name": "mid",
            "extends": str(grand),
            "entity_types": {"m_ent": {"description": "mid"}},
        })
        leaf = self._write_yaml(tmp_path, "leaf", {
            "name": "leaf",
            "extends": str(mid),
            "entity_types": {"l_ent": {"description": "leaf"}},
        })
        s = load_schema(str(leaf))
        assert "g_ent" in s.entity_types
        assert "m_ent" in s.entity_types
        assert "l_ent" in s.entity_types

    def test_extends_cycle_detected(self, tmp_path):
        """A -> B -> A must raise before attempting infinite recursion."""
        from ontograph.schema import load_schema
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(yaml.safe_dump({
            "name": "a", "extends": str(b),
            "entity_types": {"x": {"description": "x"}},
        }))
        b.write_text(yaml.safe_dump({
            "name": "b", "extends": str(a),
            "entity_types": {"y": {"description": "y"}},
        }))
        with pytest.raises(ValueError, match="[Cc]ycle"):
            load_schema(str(a))

    def test_extends_preset_name_resolution(self, tmp_path):
        """`extends: economics` (bare name) resolves via SCHEMAS_DIR, like load_schema does."""
        from ontograph.schema import load_schema
        child = tmp_path / "child.yaml"
        child.write_text(yaml.safe_dump({
            "name": "child",
            "extends": "economics",
            "entity_types": {"my_custom": {"description": "custom"}},
        }))
        s = load_schema(str(child))
        assert "agent" in s.entity_types   # from economics
        assert "my_custom" in s.entity_types


# ── New validation block ──

class TestValidationBlock:
    def test_default_empty_validation(self):
        from ontograph.schema import load_schema
        s = load_schema("base")
        assert hasattr(s, "validation")
        assert s.validation == {}

    def test_validation_loads_from_yaml(self, tmp_path):
        from ontograph.schema import load_schema
        p = tmp_path / "v.yaml"
        p.write_text(yaml.safe_dump({
            "name": "v",
            "validation": {
                "require_grounding_for": ["mechanism", "parameter"],
                "allow_abstract": ["theory"],
                "cycle_policy": {"within_step": "forbid", "across_step": "allow"},
            },
            "entity_types": {"mechanism": {"description": "m"}},
        }))
        s = load_schema(str(p))
        assert s.validation["require_grounding_for"] == ["mechanism", "parameter"]
        assert s.validation["allow_abstract"] == ["theory"]
        assert s.validation["cycle_policy"]["across_step"] == "allow"

    def test_validation_inherits_via_extends(self, tmp_path):
        from ontograph.schema import load_schema
        parent = tmp_path / "p.yaml"
        parent.write_text(yaml.safe_dump({
            "name": "p",
            "validation": {"require_grounding_for": ["mechanism"]},
        }))
        child = tmp_path / "c.yaml"
        child.write_text(yaml.safe_dump({
            "name": "c",
            "extends": str(parent),
            "validation": {"allow_abstract": ["theory"]},
        }))
        s = load_schema(str(child))
        # Both parent and child keys present; child does not blow away parent unless same key.
        assert s.validation["require_grounding_for"] == ["mechanism"]
        assert s.validation["allow_abstract"] == ["theory"]


# ── New default_* on RelationTypeDef ──

class TestRelationTypeDefaults:
    def test_defaults_are_unknown_when_absent(self):
        from ontograph.schema import load_schema
        s = load_schema("economics")
        r = s.relation_types["feeds_into"]
        assert r.default_sign == "unknown"
        assert r.default_lag == "unknown"
        assert r.default_form == "unknown"
        assert r.default_edge_class == "mechanism"

    def test_defaults_loaded_from_yaml(self, tmp_path):
        from ontograph.schema import load_schema
        p = tmp_path / "r.yaml"
        p.write_text(yaml.safe_dump({
            "name": "r",
            "relation_types": {
                "dampens": {
                    "description": "d",
                    "default_sign": "-",
                    "default_lag": "within_step",
                    "default_form": "monotone",
                    "default_edge_class": "mechanism",
                },
                "equals": {
                    "description": "identity",
                    "default_edge_class": "identity",
                    "default_form": "identity",
                    "default_sign": "+",
                    "default_lag": "within_step",
                },
            },
        }))
        s = load_schema(str(p))
        d = s.relation_types["dampens"]
        assert d.default_sign == "-"
        assert d.default_lag == "within_step"
        assert d.default_form == "monotone"
        eq = s.relation_types["equals"]
        assert eq.default_edge_class == "identity"
        assert eq.default_form == "identity"


# ── DABS schema equivalence: extends-based vs self-contained ──

class TestDabsExtendsEquivalence:
    """After the dabs.yaml refactor to `extends: economics`, the loaded schema
    must contain every entity type and relation type the old self-contained file had.

    This is a characterization test (Feathers): it pins behaviour before further change."""

    def test_dabs_inherits_economics_types(self):
        from ontograph.schema import load_schema
        dabs = load_schema("dabs")
        economics = load_schema("economics")
        for ename in economics.entity_types:
            assert ename in dabs.entity_types, f"dabs lost economics entity '{ename}'"
        for rname in economics.relation_types:
            assert rname in dabs.relation_types, f"dabs lost economics relation '{rname}'"

    def test_dabs_keeps_its_own_additions(self):
        from ontograph.schema import load_schema
        dabs = load_schema("dabs")
        # DABS-specific entity additions (see dabs.yaml §entity_types DABS-specific additions)
        for name in ("data_source", "bug_fix", "phase"):
            assert name in dabs.entity_types
        # DABS-specific relation additions
        for name in ("relaxes", "asserts", "implements_as", "supersedes",
                     "applied_during", "executes_after", "sourced_from",
                     "embeds", "distils_into"):
            assert name in dabs.relation_types

    def test_dabs_child_override_preserved(self):
        """DABS extends economics's `mechanism` with extra subtypes (phase, learned_operator, …).
        After extends-merge, dabs.entity_types['mechanism'].subtypes must include them."""
        from ontograph.schema import load_schema
        dabs = load_schema("dabs")
        mech = dabs.entity_types["mechanism"]
        for st in ("phase", "learned_operator", "smooth_operator",
                   "optimizer", "training_regime", "accounting_operator"):
            assert st in mech.subtypes, f"dabs mechanism.subtypes missing '{st}'"
