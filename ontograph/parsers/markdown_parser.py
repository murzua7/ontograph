"""Markdown document parser."""

from __future__ import annotations

import re
from pathlib import Path

from ontograph.parsers import ParsedDocument, Section


def parse_markdown(path: Path) -> ParsedDocument:
    """Parse a Markdown file into sections."""
    text = path.read_text(encoding="utf-8", errors="replace")

    sections = []
    current_heading = "Introduction"
    current_level = 1
    current_lines = []

    title = path.stem

    for line in text.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(Section(
                        heading=current_heading,
                        text=body,
                        level=current_level,
                    ))
            current_heading = heading_match.group(2).strip()
            current_level = len(heading_match.group(1))
            current_lines = []

            # Use first H1 as title
            if current_level == 1 and title == path.stem:
                title = current_heading
        else:
            current_lines.append(line)

    # Final section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(Section(heading=current_heading, text=body, level=current_level))

    return ParsedDocument(
        title=title,
        sections=sections,
        full_text=text,
        source_path=str(path),
    )
