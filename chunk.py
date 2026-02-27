"""PDF chunker for EU AI Act. Extracts structured markdown and splits into chunks."""

import re

import fitz


def pdf_to_markdown(path: str, output: str = "eu_ai_act.md") -> str:
    """Convert PDF to structured markdown file. Returns output path."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    raw = "\n".join(pages)

    # CRITICAL: normalize form feed chars before any regex
    # pymupdf uses \x0c for page breaks; re.MULTILINE ^ doesn't match after \x0c
    raw = raw.replace("\x0c", "\n")

    # Clean up excessive whitespace runs but preserve paragraph breaks
    raw = re.sub(r"\n{4,}", "\n\n\n", raw)

    lines = raw.split("\n")
    md_lines: list[str] = []
    i = 0

    # Patterns
    chapter_re = re.compile(r"^(CHAPTER\s+[IVXLC]+)$")
    article_re = re.compile(r"^(Article\s+\d+)$")
    annex_re = re.compile(r"^(ANNEX\s+[IVXLC]+)$")

    while i < len(lines):
        line = lines[i].strip()

        # Chapter heading: "CHAPTER I" on its own line, title on next line
        m = chapter_re.match(line)
        if m:
            heading = m.group(1)
            # Peek at next non-empty line for chapter title
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title_line = lines[j].strip()
                # Chapter titles are usually short uppercase lines
                if len(title_line) < 120 and not article_re.match(title_line):
                    heading = f"{heading}: {title_line}"
                    i = j  # skip title line
            md_lines.append(f"\n# {heading}\n")
            i += 1
            continue

        # Article heading: "Article N" alone on line
        m = article_re.match(line)
        if m:
            heading = m.group(1)
            # Peek at next non-empty line for article title
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title_line = lines[j].strip()
                if len(title_line) < 120 and not article_re.match(title_line) and not chapter_re.match(title_line):
                    heading = f"{heading}: {title_line}"
                    i = j
            md_lines.append(f"\n## {heading}\n")
            i += 1
            continue

        # Annex heading: "ANNEX I" etc
        m = annex_re.match(line)
        if m:
            heading = m.group(1)
            # Peek for annex title
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title_line = lines[j].strip()
                if len(title_line) < 200 and not annex_re.match(title_line):
                    heading = f"{heading}: {title_line}"
                    i = j
            md_lines.append(f"\n# {heading}\n")
            i += 1
            continue

        # Regular text line
        md_lines.append(line)
        i += 1

    md_text = "\n".join(md_lines)

    # Inject Recitals heading before the "Whereas:" preamble if not already present
    # The recitals start with "THE EUROPEAN PARLIAMENT..." then "Whereas:" then numbered paragraphs
    whereas_match = re.search(r"^Whereas:", md_text, re.MULTILINE)
    if whereas_match:
        md_text = md_text[: whereas_match.start()] + "# Recitals\n\n" + md_text[whereas_match.start() :]

    with open(output, "w") as f:
        f.write(md_text)

    return output


def chunk_markdown(path: str) -> list[dict]:
    """Split markdown file into chunks by headers. Returns list of chunk dicts."""
    with open(path) as f:
        text = f.read()

    # Split on markdown headers
    header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    chunks: list[dict] = []
    current_chapter: str | None = None

    matches = list(header_re.finditer(text))
    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        # Get body text between this header and the next
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Track current chapter for parent_title
        if level == 1 and title.upper().startswith("CHAPTER"):
            current_chapter = title

        # Determine parent_title
        parent_title: str | None = None
        if level == 1:
            parent_title = None  # top-level
        elif level == 2 and title.startswith("Recitals"):
            parent_title = "Recitals"
        elif level == 2:
            parent_title = current_chapter  # articles belong to their chapter

        chunk = {"title": title, "level": level, "text": body, "parent_title": parent_title}

        # Special handling: split Recitals into sub-chunks
        if title == "Recitals" and level == 1:
            # Create the parent node with empty text
            chunk["text"] = ""
            chunks.append(chunk)

            # Split recital body into sub-chunks by numbered paragraphs
            para_re = re.compile(r"^\(\d+\)", re.MULTILINE)
            para_matches = list(para_re.finditer(body))

            if para_matches:
                group_size = 20
                for gi in range(0, len(para_matches), group_size):
                    group_end = min(gi + group_size, len(para_matches))
                    start_pos = para_matches[gi].start()
                    end_pos = para_matches[group_end].start() if group_end < len(para_matches) else len(body)
                    sub_text = body[start_pos:end_pos].strip()

                    sub_title = f"Recitals {gi + 1}-{group_end}"
                    chunks.append(
                        {"title": sub_title, "level": 2, "text": sub_text, "parent_title": "Recitals"}
                    )
            continue

        chunks.append(chunk)

    return chunks


def chunk_pdf(path: str) -> list[dict]:
    """Convenience: PDF -> markdown -> chunks."""
    md_path = pdf_to_markdown(path)
    return chunk_markdown(md_path)
