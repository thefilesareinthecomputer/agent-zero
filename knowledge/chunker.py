"""Section-based chunking for knowledge base files.

Splits markdown files on heading boundaries for indexing and token-aware
retrieval. Detects the highest heading level present (H1-H5) and splits
on that. Handles recursive splitting for oversized sections and hard
splits when no sub-headings exist.

Also provides heading tree construction -- a hierarchical view of all
headings with cumulative token counts per subtree. Used by read_knowledge
to give the agent a structural map of oversized files.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from knowledge.knowledge_store import _parse_frontmatter
from knowledge.tokenizer import count_tokens, truncate_to_tokens


@dataclass
class HeadingNode:
    """A node in the heading tree. Represents one markdown heading."""
    level: int                                  # 1-5 (or 0 for root)
    heading: str                                # heading text without # prefix
    own_tokens: int = 0                         # tokens in this node's direct content
    subtree_tokens: int = 0                     # own_tokens + all children's subtree_tokens
    children: list["HeadingNode"] = field(default_factory=list)
    summary: str = ""                           # LLM-generated summary (populated from index)


def _detect_heading_level(body: str) -> int | None:
    """Find the highest (lowest number) heading level in the body.

    Scans for H1 through H5. Returns None if no headings found.
    """
    for level in range(1, 6):
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


def build_heading_tree(text: str, filename: str) -> HeadingNode:
    """Build a hierarchical heading tree from a markdown file.

    Parses all H1-H5 headings and organizes them into a tree where each
    node's subtree_tokens includes its own content plus all descendants.
    The root node (level 0) represents the file itself.

    Args:
        text: Raw file content (including frontmatter).
        filename: Used as the root node heading.

    Returns:
        HeadingNode tree with computed subtree_tokens at every level.
    """
    _, body = _parse_frontmatter(text)
    body = body.strip()

    root = HeadingNode(level=0, heading=Path(filename).stem)

    if not body:
        return root

    # Parse all headings and their content spans
    heading_pattern = re.compile(r"^(#{1,5})\s+(.+)$", re.MULTILINE)
    entries: list[tuple[int, str, str]] = []  # (level, heading, content)

    matches = list(heading_pattern.finditer(body))

    if not matches:
        # No headings -- all content belongs to root
        root.own_tokens = count_tokens(body)
        root.subtree_tokens = root.own_tokens
        return root

    # Content before first heading = root's own content
    preamble = body[:matches[0].start()].strip()
    if preamble:
        root.own_tokens = count_tokens(preamble)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[content_start:content_end].strip()
        # Strip trailing dividers
        if content.endswith("---"):
            content = content[:-3].rstrip()
        entries.append((level, heading, content))

    # Build tree using a stack-based approach
    stack: list[HeadingNode] = [root]

    for level, heading, content in entries:
        node = HeadingNode(
            level=level,
            heading=heading,
            own_tokens=count_tokens(content) if content else 0,
        )
        # Pop stack until we find a parent with a lower level
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)

    # Compute subtree_tokens bottom-up
    _compute_subtree_tokens(root)

    return root


def _compute_subtree_tokens(node: HeadingNode) -> int:
    """Recursively compute subtree_tokens for a node and all descendants."""
    child_total = sum(_compute_subtree_tokens(c) for c in node.children)
    node.subtree_tokens = node.own_tokens + child_total
    return node.subtree_tokens


def format_heading_tree(root: HeadingNode, indent: int = 0) -> str:
    """Render a heading tree as indented text for LLM consumption.

    Output looks like:
        dim-modeling-guide (2,450 tokens)
          ## Overview (320 tokens) -- Introduces dimensional modeling concepts
          ## Star Schema (1,200 tokens) -- Covers star schema design patterns
            ### Fact Tables (500 tokens) -- Fact table grain and measures
            ### Dimension Tables (700 tokens) -- Conformed dimension design
          ## Snowflake Schema (930 tokens) -- Normalized variant of star schema
    """
    lines: list[str] = []
    prefix = "  " * indent

    if root.level == 0:
        # Root node -- show filename
        root_line = f"{root.heading} ({root.subtree_tokens:,} tokens)"
        if root.summary:
            root_line += f" -- {root.summary}"
        lines.append(root_line)
        for child in root.children:
            lines.append(format_heading_tree(child, indent + 1))
    else:
        hashes = "#" * root.level
        tok = root.subtree_tokens if root.children else root.own_tokens
        node_line = f"{prefix}{hashes} {root.heading} ({tok:,} tokens)"
        if root.summary:
            node_line += f" -- {root.summary}"
        lines.append(node_line)
        if root.children:
            for child in root.children:
                lines.append(format_heading_tree(child, indent + 1))

    return "\n".join(lines)


def enrich_tree_summaries(
    node: HeadingNode,
    summaries: dict[str, str],
) -> None:
    """Attach summaries from the KB index to heading tree nodes.

    Walks the tree and sets each node's summary field from the
    heading -> summary mapping. Matching is case-insensitive.
    Modifies the tree in place.
    """
    # Build lowercase lookup for case-insensitive matching
    lower_map = {k.lower(): v for k, v in summaries.items()}

    def _walk(n: HeadingNode) -> None:
        key = n.heading.lower()
        if key in lower_map:
            n.summary = lower_map[key]
        for child in n.children:
            _walk(child)

    _walk(node)


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
    """Chunk a knowledge file into H1-level concept sections for indexing.

    H1 headings are first-class concept boundaries -- each H1 block becomes
    one chunk.  H2-H5 headings are children of their parent H1 and remain
    inside that chunk.  When max_tokens is set and an H1 chunk exceeds it,
    the chunk is recursively split at H2 (then H3, etc.), then hard-split
    at token boundaries if no sub-headings exist.

    Files with no H1 fall back to H2 as the primary split level, then H3, etc.
    Files with no headings at all produce a single chunk.

    Args:
        text: Raw file content (including frontmatter).
        filename: The filename (used as fallback heading for headingless files).
        max_tokens: Optional per-chunk token limit.

    Returns list of dicts, each with:
        heading: str -- H1 section heading (or composite path for sub-splits)
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
