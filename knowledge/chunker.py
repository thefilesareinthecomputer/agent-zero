"""Section-based chunking for knowledge base files.

Splits markdown files on heading boundaries for indexing and token-aware
retrieval. Detects the highest heading level present and splits on that.
Handles recursive splitting for oversized sections and hard splits when
no sub-headings exist.
"""

import re
from pathlib import Path

from knowledge.knowledge_store import _parse_frontmatter
from knowledge.tokenizer import count_tokens, truncate_to_tokens


def _detect_heading_level(body: str) -> int | None:
    """Find the highest (lowest number) heading level in the body.

    Scans for ## (2), ### (3), #### (4). Returns None if no headings found.
    """
    for level in (2, 3, 4):
        pattern = rf"^{'#' * level}\s+\S"
        if re.search(pattern, body, re.MULTILINE):
            return level
    return None


def _split_on_level(body: str, level: int) -> list[tuple[str, str]]:
    """Split body into (heading, content) pairs at the given heading level.

    Content before the first heading at this level is captured with
    heading="(preamble)". The heading prefix (###) is stripped from
    the heading text. Divider lines (---) immediately before headings
    are excluded from the preceding section's content.
    """
    prefix = "#" * level
    pattern = rf"^{prefix}\s+(.+)$"
    sections: list[tuple[str, str]] = []
    current_heading = "(preamble)"
    current_lines: list[str] = []

    for line in body.split("\n"):
        match = re.match(pattern, line)
        if match:
            # Close previous section, stripping trailing dividers
            content = "\n".join(current_lines).rstrip()
            if content.endswith("---"):
                content = content[:-3].rstrip()
            if current_lines or sections:
                sections.append((current_heading, content))
            current_heading = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Close final section
    content = "\n".join(current_lines).rstrip()
    sections.append((current_heading, content))

    return sections


def _hard_split(text: str, max_tokens: int, overlap: int = 50) -> list[str]:
    """Split text into token-bounded chunks with overlap.

    Used as a last resort when no sub-headings exist and the section
    exceeds the token budget.
    """
    from knowledge.tokenizer import _get_encoder
    enc = _get_encoder()
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start = end - overlap if end < len(tokens) else end
    return chunks


def _chunk_sections(
    sections: list[tuple[str, str]],
    level: int,
    max_tokens: int | None,
) -> list[dict]:
    """Convert (heading, content) pairs into chunk dicts.

    If max_tokens is set and a section exceeds it, try to split on the
    next heading level down. If that fails, hard-split at token boundaries.
    """
    chunks = []

    for heading, content in sections:
        # Skip empty preamble sections
        if heading == "(preamble)" and not content.strip():
            continue

        full_text = content if heading == "(preamble)" else f"{'#' * level} {heading}\n\n{content}"
        tok_count = count_tokens(full_text)

        if max_tokens is None or tok_count <= max_tokens:
            chunks.append({
                "heading": heading,
                "content": full_text,
                "token_count": tok_count,
            })
        else:
            # Try splitting on next heading level
            next_level = level + 1
            sub_level = _detect_heading_level(content)
            if sub_level and sub_level > level:
                sub_sections = _split_on_level(content, sub_level)
                sub_chunks = _chunk_sections(sub_sections, sub_level, max_tokens)
                # Prefix sub-chunk headings with parent heading for context
                for sc in sub_chunks:
                    if sc["heading"] != "(preamble)":
                        sc["heading"] = f"{heading} > {sc['heading']}"
                    else:
                        sc["heading"] = heading
                chunks.extend(sub_chunks)
            else:
                # No sub-headings -- hard split
                parts = _hard_split(full_text, max_tokens)
                for i, part in enumerate(parts):
                    suffix = f" (part {i + 1}/{len(parts)})" if len(parts) > 1 else ""
                    chunks.append({
                        "heading": f"{heading}{suffix}",
                        "content": part,
                        "token_count": count_tokens(part),
                    })

    return chunks


def chunk_file(
    text: str,
    filename: str,
    *,
    max_tokens: int | None = None,
) -> list[dict]:
    """Chunk a knowledge file into sections for indexing.

    Args:
        text: Raw file content (including frontmatter).
        filename: The filename (used as fallback heading).
        max_tokens: Optional per-chunk token limit. Sections exceeding
            this are recursively split on sub-headings, then hard-split
            at token boundaries if no sub-headings exist.

    Returns list of dicts, each with:
        heading: str -- section heading (or filename for headingless files)
        content: str -- section text
        chunk_index: int -- 0-based position in the file
        token_count: int -- token count of the content
    """
    _, body = _parse_frontmatter(text)
    body = body.strip()

    if not body:
        return []

    level = _detect_heading_level(body)

    if level is None:
        # No headings -- single chunk
        tok_count = count_tokens(body)
        if max_tokens and tok_count > max_tokens:
            parts = _hard_split(body, max_tokens)
            return [
                {
                    "heading": Path(filename).stem,
                    "content": part,
                    "chunk_index": i,
                    "token_count": count_tokens(part),
                }
                for i, part in enumerate(parts)
            ]
        return [{
            "heading": Path(filename).stem,
            "content": body,
            "chunk_index": 0,
            "token_count": tok_count,
        }]

    sections = _split_on_level(body, level)
    chunks = _chunk_sections(sections, level, max_tokens)

    # Assign chunk indices
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks
