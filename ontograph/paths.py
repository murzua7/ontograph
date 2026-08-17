"""Signed-path algebra.

Canonical specification of how edge signs compose along a directed path.
The 3D viewer (`ontograph.dashboard.force3d`) and the CLI cycle-report share
these conventions; keep the two implementations in sync.

Rules:
    +       identity (leaves carry unchanged)
    -       flips the carry (+ ↔ -)
    0       annihilator: once any hop is 0, the product is 0
    ±       marks the carry as ambiguous-but-propagating
    unknown poisons: any unknown hop makes the whole product "unknown"

Rationale:
    `0` says "this hop has no effect" — the composed path inherits that.
    `±` says "the hop's sign is genuinely ambiguous in the evidence"; it
    still propagates a parity-preserving operation, just flagged as ambiguous.
    `unknown` is extractor silence, not a claim; composing silence with
    anything yields a path whose net sign is undefined.
"""
from __future__ import annotations

from typing import Iterable

_VALID = {"+", "-", "0", "±"}


def sign_product(signs: Iterable[str]) -> str:
    """Fold a sequence of edge signs into a single net sign.

    Returns one of: "+", "-", "0", "±", "unknown". Empty input is "+"
    (identity for zero-length paths).
    """
    current = "+"
    ambiguous = False
    for s in signs:
        if s not in _VALID:
            return "unknown"
        if s == "0":
            return "0"
        if s == "±":
            ambiguous = True
        elif s == "-":
            current = "-" if current == "+" else "+"
        # "+" leaves current unchanged
    return "±" if ambiguous else current
