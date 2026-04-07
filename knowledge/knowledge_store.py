"""Knowledge base store -- manages a folder of Obsidian-compatible markdown files.

Each file has:
- YAML frontmatter (tags, created, last-modified)
- H1 heading matching the filename
- Auto-generated table of contents linking to H2 sections
- H2 sections separated by --- dividers

The agent provides section content. This module handles the structure.
"""

import re
from datetime import date
from pathlib import Path

import yaml

from agent.config import KNOWLEDGE_PATH

KNOWLEDGE_DIR = Path(KNOWLEDGE_PATH)


def _sanitize_filename(name: str) -> str:
    """Normalize a filename to lowercase-hyphenated with .md extension."""
    name = name.strip().lower()
    name = re.sub(r"\.md$", "", name)
    name = re.sub(r"[_\s]+", "-", name)
    name = re.sub(r"[^a-z0-9-]", "", name)
    return name + ".md"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a file into frontmatter dict and remaining content.

    Returns (metadata, body). If no valid frontmatter, returns ({}, full text).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text

    return meta, parts[2]


def _extract_sections(body: str) -> str:
    """Extract the H2 section content from a file body.

    Strips the H1 heading and TOC, returning everything from the first
    H2 onward. This is what the agent edits and passes back to save_file.
    """
    # Find the first ## heading
    match = re.search(r"^---\s*\n##\s+", body, re.MULTILINE)
    if match:
        return body[match.start():].strip()

    # Fallback: find first ## without preceding ---
    match = re.search(r"^##\s+", body, re.MULTILINE)
    if match:
        return body[match.start():].strip()

    # No sections found -- return body stripped of H1 and TOC
    lines = body.strip().split("\n")
    content_lines = []
    past_toc = False
    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("[") and "](#" in line:
            continue
        if line.strip() == "---" and not past_toc:
            past_toc = True
            continue
        if past_toc or (not line.startswith("[") and line.strip()):
            past_toc = True
            content_lines.append(line)

    return "\n".join(content_lines).strip()


def _build_toc(content: str) -> str:
    """Generate a table of contents from H2 headings in the content."""
    headings = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)
    if not headings:
        return ""
    lines = []
    for heading in headings:
        anchor = heading.strip().lower()
        anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
        anchor = re.sub(r"\s+", "-", anchor)
        lines.append(f"[{heading.strip()}](#{anchor})")
    return "\n".join(lines)


def _ensure_section_dividers(content: str) -> str:
    """Make sure each H2 section is preceded by a --- divider."""
    lines = content.strip().split("\n")
    result = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            # Add divider before H2 if not already there
            if result and result[-1].strip() != "---":
                result.append("")
                result.append("---")
        result.append(line)
    return "\n".join(result)


def _build_file(filename: str, content: str, tags: list[str],
                created: str | None = None) -> str:
    """Assemble a complete knowledge file from parts.

    Args:
        filename: The .md filename (used for H1 heading).
        content: H2 section content from the agent.
        tags: List of tag strings.
        created: ISO date string. Defaults to today if None.
    """
    today = date.today().isoformat()
    stem = filename.replace(".md", "")

    meta = {
        "tags": tags,
        "created": created or today,
        "last-modified": today,
    }

    # Build frontmatter
    fm_lines = ["---"]
    fm_lines.append("tags:")
    for tag in meta["tags"]:
        fm_lines.append(f"  - {tag}")
    fm_lines.append(f"created: {meta['created']}")
    fm_lines.append(f"last-modified: {meta['last-modified']}")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines)

    # Ensure dividers between sections
    content = _ensure_section_dividers(content)

    # Build TOC
    toc = _build_toc(content)

    # Assemble
    parts = [frontmatter, f"# {stem}"]
    if toc:
        parts.append(toc)
    parts.append("")
    parts.append(content)
    parts.append("")

    return "\n".join(parts)


def list_files() -> list[dict]:
    """List all knowledge base .md files with metadata.

    Returns list of dicts: {filename, tags, created, last_modified, path}.
    Sorted by last_modified descending.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for path in KNOWLEDGE_DIR.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        meta, _ = _parse_frontmatter(text)
        results.append({
            "filename": path.name,
            "tags": meta.get("tags", []),
            "created": str(meta.get("created", "")),
            "last_modified": str(meta.get("last-modified", "")),
            "path": str(path),
        })

    results.sort(key=lambda r: r["last_modified"], reverse=True)
    return results


def read_file(filename: str) -> str | None:
    """Read a knowledge file and return the editable section content.

    Returns the H2 sections (everything after frontmatter, H1, and TOC).
    This is what the agent modifies and passes back to save_file.
    Returns None if the file does not exist.
    """
    filename = _sanitize_filename(filename)
    path = KNOWLEDGE_DIR / filename

    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    _, body = _parse_frontmatter(text)
    return _extract_sections(body)


def save_file(filename: str, content: str, tags: list[str]) -> str:
    """Create or update a knowledge file.

    The agent provides content as H2 sections with body text.
    This function wraps it with frontmatter, H1, and auto-generated TOC.

    If the file already exists, the original created date is preserved.

    Args:
        filename: Name for the file (sanitized to lowercase-hyphens.md).
        content: H2 section content from the agent.
        tags: List of tag strings.

    Returns the path of the written file.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_filename(filename)
    path = KNOWLEDGE_DIR / filename

    # Preserve created date from existing file
    created = None
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
            meta, _ = _parse_frontmatter(existing)
            created = str(meta.get("created", ""))
        except (OSError, UnicodeDecodeError):
            pass

    file_content = _build_file(filename, content, tags, created or None)
    path.write_text(file_content, encoding="utf-8")
    return str(path)


def search_files(query: str) -> list[dict]:
    """Search all knowledge files for a keyword or phrase.

    Case-insensitive substring search across file content (not frontmatter).
    Returns list of {filename, matching_lines}. Capped at 10 results.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    query_lower = query.lower()
    results = []

    for path in KNOWLEDGE_DIR.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        _, body = _parse_frontmatter(text)
        matches = []
        for line in body.split("\n"):
            if query_lower in line.lower():
                stripped = line.strip()
                if stripped and not stripped.startswith("[") and stripped != "---":
                    matches.append(stripped)

        if matches:
            results.append({
                "filename": path.name,
                "matching_lines": matches[:5],
            })

        if len(results) >= 10:
            break

    return results
