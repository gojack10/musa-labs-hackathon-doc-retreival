"""Generic markdown chunker. Splits any well-structured .md file into hierarchical chunks."""

import re


def chunk_markdown(path: str) -> list[dict]:
    """Split a markdown file into chunks by headings.

    Uses heading hierarchy (h1 > h2 > h3 etc.) to derive parent relationships.
    Returns list of dicts with keys: title, level, text, parent_title.
    """
    with open(path) as f:
        text = f.read()

    header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_re.finditer(text))

    if not matches:
        return []

    # Track most recent heading at each level for parent resolution
    heading_stack: dict[int, str] = {}
    chunks: list[dict] = []

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        # Body = text between this header and the next
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Parent = most recent heading one level up
        parent_title = heading_stack.get(level - 1)

        # Update stack: set this level, clear all deeper levels
        heading_stack[level] = title
        for deeper in range(level + 1, 7):
            heading_stack.pop(deeper, None)

        chunks.append({
            "title": title,
            "level": level,
            "text": body,
            "parent_title": parent_title,
        })

    return chunks
