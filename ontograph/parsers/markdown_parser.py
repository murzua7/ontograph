"""Markdown document parser."""

from __future__ import annotations

import re
from pathlib import Path

from ontograph.parsers import ParsedDocument, Section


def parse_markdown(path: Path) -> ParsedDocument:
    """Parse a Markdown file into sections.

    Tracks two extra fields on each Section for DocAnchor use:
      • `char_offset` — byte offset (in chars) of the section body's first
        character in the raw `full_text`.
      • `line` — 1-indexed source line where the section begins.

    Both are computed from the raw text, not the `\\n`-joined rebuild, so
    they're stable reproducible pointers.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    # Precompute the char offset at the start of each line (1-indexed by line).
    # line_offsets[i] = char offset where line i+1 begins in `text`.
    line_offsets: list[int] = [0]
    running = 0
    for ln in lines:
        running += len(ln) + 1  # +1 for the stripped "\n"
        line_offsets.append(running)

    sections: list[Section] = []
    current_heading = "Introduction"
    current_level = 1
    current_body_lines: list[str] = []
    current_body_start_line: int = 1  # 1-indexed

    title = path.stem

    def _flush(end_line: int) -> None:
        body = "\n".join(current_body_lines).strip()
        if not body:
            return
        # Find the char offset of the first non-blank body line — that's
        # where `body` actually begins (strip() skipped leading blanks).
        start_line = current_body_start_line
        for offset, raw in enumerate(current_body_lines):
            if raw.strip():
                start_line = current_body_start_line + offset
                break
        char_offset = line_offsets[start_line - 1] if start_line - 1 < len(line_offsets) else -1
        sections.append(Section(
            heading=current_heading,
            text=body,
            level=current_level,
            char_offset=char_offset,
            line=start_line,
        ))

    for idx, line in enumerate(lines, start=1):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            _flush(idx - 1)
            current_heading = heading_match.group(2).strip()
            current_level = len(heading_match.group(1))
            current_body_lines = []
            current_body_start_line = idx + 1

            if current_level == 1 and title == path.stem:
                title = current_heading
        else:
            current_body_lines.append(line)

    _flush(len(lines))

    return ParsedDocument(
        title=title,
        sections=sections,
        full_text=text,
        source_path=str(path),
    )
