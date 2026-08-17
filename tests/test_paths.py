"""Tests for signed-path algebra (`ontograph.paths.sign_product`)."""
from __future__ import annotations

import pytest

from ontograph.paths import sign_product


class TestSignProductBasics:
    def test_empty_is_plus(self):
        """Empty product is identity — zero-length path preserves sign."""
        assert sign_product([]) == "+"

    def test_all_plus(self):
        assert sign_product(["+", "+", "+"]) == "+"

    def test_single_minus_flips(self):
        assert sign_product(["-"]) == "-"

    def test_minus_minus_is_plus(self):
        assert sign_product(["-", "-"]) == "+"

    def test_plus_minus_plus_is_minus(self):
        assert sign_product(["+", "-", "+"]) == "-"

    def test_three_minuses_is_minus(self):
        assert sign_product(["-", "-", "-"]) == "-"


class TestAnnihilators:
    def test_zero_anywhere_returns_zero(self):
        assert sign_product(["+", "0", "-"]) == "0"
        assert sign_product(["0"]) == "0"
        assert sign_product(["-", "0"]) == "0"

    def test_zero_beats_unknown(self):
        """Once the product is zero, no later hop (even unknown) matters."""
        assert sign_product(["+", "0", "unknown"]) == "0"

    def test_unknown_poisons_if_before_zero(self):
        """But an unknown encountered BEFORE the zero still poisons."""
        assert sign_product(["+", "unknown", "0"]) == "unknown"


class TestAmbiguous:
    def test_ambiguous_propagates(self):
        assert sign_product(["+", "±", "+"]) == "±"

    def test_ambiguous_flips_with_minus(self):
        """`±` is still a sign — minus still acts on the overall parity."""
        # The carry goes: + → (±, still +) → (-, flipped to -) → still ambiguous
        # so the result is ±.
        assert sign_product(["±", "-"]) == "±"

    def test_ambiguous_beaten_by_zero(self):
        assert sign_product(["±", "0"]) == "0"


class TestUnknown:
    def test_unknown_poisons(self):
        assert sign_product(["unknown"]) == "unknown"
        assert sign_product(["+", "unknown", "-"]) == "unknown"

    def test_empty_sign_treated_as_unknown(self):
        """Defensive: a missing/empty sign also poisons."""
        assert sign_product([""]) == "unknown"

    def test_unrecognised_sign_poisons(self):
        """Unknown vocabulary items get the same treatment as unknown."""
        assert sign_product(["+", "XYZ"]) == "unknown"


class TestDocumentedExamples:
    """Examples used in the viewer comment block and external docs."""

    @pytest.mark.parametrize("signs,expected", [
        (["+", "+"], "+"),
        (["+", "-"], "-"),
        (["-", "-"], "+"),
        (["+", "±", "-"], "±"),
        (["+", "0", "+"], "0"),
        (["unknown", "+"], "unknown"),
    ])
    def test_examples(self, signs, expected):
        assert sign_product(signs) == expected
