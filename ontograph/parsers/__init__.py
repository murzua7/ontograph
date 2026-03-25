"""Document parsers: dispatch by file extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    heading: str
    text: str
    level: int = 1
    page: int | None = None


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section]
    full_text: str
    source_path: str
    metadata: dict = field(default_factory=dict)


def parse_document(path: str | Path) -> ParsedDocument:
    """Parse a document by file extension."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".md", ".markdown", ".txt"):
        from ontograph.parsers.markdown_parser import parse_markdown
        return parse_markdown(path)
    elif ext in (".html", ".htm"):
        from ontograph.parsers.html_parser import parse_html
        return parse_html(path)
    elif ext == ".pdf":
        from ontograph.parsers.pdf_parser import parse_pdf
        return parse_pdf(path)
    else:
        # Fallback: treat as plain text
        text = path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(
            title=path.stem,
            sections=[Section(heading="Full Text", text=text)],
            full_text=text,
            source_path=str(path),
        )
