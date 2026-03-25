"""PDF document parser (optional dependency: pdfplumber)."""

from __future__ import annotations

from pathlib import Path

from ontograph.parsers import ParsedDocument, Section


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse a PDF file into sections.

    Uses pdfplumber for text extraction. Falls back to plain page-by-page
    extraction if section headings cannot be detected.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "PDF parsing requires pdfplumber. Install with: uv add pdfplumber"
        )

    sections = []
    full_text_parts = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            full_text_parts.append(text)
            if text.strip():
                sections.append(Section(
                    heading=f"Page {i + 1}",
                    text=text.strip(),
                    page=i + 1,
                ))

    full_text = "\n\n".join(full_text_parts)
    title = path.stem

    # Try to extract title from first page
    if sections:
        first_lines = sections[0].text.split("\n")
        if first_lines:
            title = first_lines[0][:100].strip()

    return ParsedDocument(
        title=title,
        sections=sections,
        full_text=full_text,
        source_path=str(path),
    )
