"""Thin CLI wrapper around `ontograph.dashboard.force3d.build_3d_html`.

Kept for backward compatibility; the canonical entry point is now
`ontograph view <graph.json>`, which uses the 3D viewer by default.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from ontograph.dashboard.force3d import build_3d_html


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/render_3d.py <graph.json> [out.html] [--no-open]")
        sys.exit(2)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if (len(sys.argv) > 2 and not sys.argv[2].startswith("--")) \
          else inp.with_suffix(".3d.html")
    path = build_3d_html(inp, out)
    print(f"Wrote {path}  ({path.stat().st_size/1024:.0f} KB)")
    if "--no-open" not in sys.argv:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
