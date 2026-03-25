"""HTML document parser."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ontograph.parsers import ParsedDocument, Section


def parse_html(path: Path) -> ParsedDocument:
    """Parse an HTML file into sections."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag
             else h1_tag.get_text(strip=True) if h1_tag
             else path.stem)

    # Extract sections by headings
    sections = []
    heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    if not heading_tags:
        # No headings: treat entire body as one section
        body = soup.get_text(separator="\n", strip=True)
        sections.append(Section(heading="Full Text", text=body))
    else:
        for i, tag in enumerate(heading_tags):
            level = int(tag.name[1])
            heading = tag.get_text(strip=True)

            # Collect text until next heading
            text_parts = []
            for sibling in tag.next_siblings:
                if sibling.name and sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    break
                if hasattr(sibling, "get_text"):
                    t = sibling.get_text(strip=True)
                    if t:
                        text_parts.append(t)

            body = "\n".join(text_parts)
            if body:
                sections.append(Section(heading=heading, text=body, level=level))

    full_text = soup.get_text(separator="\n", strip=True)

    return ParsedDocument(
        title=title,
        sections=sections,
        full_text=full_text,
        source_path=str(path),
    )
