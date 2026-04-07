"""Knowledge base store -- manages folders of markdown files, optionally
Obsidian-compatible.

Supports two modes:
- Agent-managed files (knowledge/): full read-write with frontmatter, TOC,
  and section structure. The agent creates and edits these.
- Drop-in files: any .md file placed in the directory (or subdirectories)
  is discoverable via list, read, and search. Files without frontmatter
  or H2 structure are returned as-is.

All functions accept an optional base_dir parameter. This allows the same
operations to work against knowledge/ (read-write) and knowledge_canon/
(read-only) without code duplication.

Recursive: all operations scan subdirectories (rglob). Filenames are
returned as relative paths from the base directory (e.g. "projects/api.md").
"""

import re
from datetime import date
from pathlib import Path

import yaml

from agent.config import KNOWLEDGE_PATH

KNOWLEDGE_DIR = Path(KNOWLEDGE_PATH)


def _sanitize_path(name: str) -> str:
    """Normalize a file path for saving. Sanitizes each component separately
    to allow subdirectories (e.g. "projects/my file" -> "projects/my-file.md").
    """
    parts = Path(name).parts
    sanitized = []
    for i, part in enumerate(parts):
        p = part.strip().lower()
        if i == len(parts) - 1:
            # Last component is the filename -- ensure .md extension
            p = re.sub(r"\.md$", "", p)
            p = re.sub(r"[_\s]+", "-", p)
            p = re.sub(r"[^a-z0-9-]", "", p)
            p = p + ".md"
        else:
            # Directory component
            p = re.sub(r"[_\s]+", "-", p)
            p = re.sub(r"[^a-z0-9-]", "", p)
        if p:
            sanitized.append(p)

    return str(Path(*sanitized)) if sanitized else "untitled.md"


def _relative_name(path: Path, base_dir: Path) -> str:
    """Return the path relative to base_dir as a string."""
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return path.name


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
    """Extract readable content from a file body.

    For structured files (with H2 sections): returns everything from the
    first H2 onward, stripping H1 and TOC.
    For plain files (no H2 sections): returns the full body stripped of
    any H1 heading.
    """
    # Find the first ## heading preceded by a divider
    match = re.search(r"^---\s*\n##\s+", body, re.MULTILINE)
    if match:
        return body[match.start():].strip()

    # Fallback: find first ## without preceding ---
    match = re.search(r"^##\s+", body, re.MULTILINE)
    if match:
        return body[match.start():].strip()

    # No H2 sections -- return body stripped of H1 and TOC lines
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

    result = "\n".join(content_lines).strip()

    # If stripping removed everything, return the raw body
    if not result:
        return body.strip()

    return result


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
    # Use just the stem of the last path component for the H1
    stem = Path(filename).stem

    meta = {
        "tags": tags,
        "created": created or today,
        "last-modified": today,
    }

    fm_lines = ["---"]
    fm_lines.append("tags:")
    for tag in meta["tags"]:
        fm_lines.append(f"  - {tag}")
    fm_lines.append(f"created: {meta['created']}")
    fm_lines.append(f"last-modified: {meta['last-modified']}")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines)

    content = _ensure_section_dividers(content)
    toc = _build_toc(content)

    parts = [frontmatter, f"# {stem}"]
    if toc:
        parts.append(toc)
    parts.append("")
    parts.append(content)
    parts.append("")

    return "\n".join(parts)


def _iter_md_files(base_dir: Path) -> list[Path]:
    """Recursively find all .md files under base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return sorted(base_dir.rglob("*.md"))


def list_files(*, filter_tags: list[str] | None = None,
               exclude_tags: list[str] | None = None,
               base_dir: Path | None = None) -> list[dict]:
    """List .md files with metadata, optionally filtered by tags.

    Scans recursively. Filenames are relative paths from base_dir.

    Args:
        filter_tags: If provided, only include files with at least one match.
        exclude_tags: If provided, exclude files with any of these tags.
        base_dir: Directory to scan. Defaults to KNOWLEDGE_DIR.

    Returns list of dicts: {filename, tags, created, last_modified, path}.
    Sorted by last_modified descending.
    """
    base = base_dir or KNOWLEDGE_DIR
    results = []

    for path in _iter_md_files(base):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        meta, _ = _parse_frontmatter(text)
        tags = meta.get("tags", []) or []

        if exclude_tags and any(t in tags for t in exclude_tags):
            continue

        if filter_tags and not any(t in tags for t in filter_tags):
            continue

        results.append({
            "filename": _relative_name(path, base),
            "tags": tags,
            "created": str(meta.get("created", "")),
            "last_modified": str(meta.get("last-modified", "")),
            "path": str(path),
        })

    results.sort(key=lambda r: r["last_modified"], reverse=True)
    return results


def read_file(filename: str, *, base_dir: Path | None = None) -> str | None:
    """Read a knowledge file and return its content.

    For structured files: returns H2 sections (what the agent edits).
    For plain files: returns the full body content.
    Does NOT sanitize the filename -- reads it as-is to support drop-in files.
    Returns None if the file does not exist.
    """
    base = base_dir or KNOWLEDGE_DIR
    path = base / filename

    if not path.exists():
        return None

    # Security: ensure the resolved path is under the base directory
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    _, body = _parse_frontmatter(text)
    return _extract_sections(body)


def save_file(filename: str, content: str, tags: list[str],
              *, base_dir: Path | None = None) -> str:
    """Create or update a knowledge file.

    The agent provides content as H2 sections with body text.
    This function wraps it with frontmatter, H1, and auto-generated TOC.
    Sanitizes the filename. Creates subdirectories as needed.

    If the file already exists, the original created date is preserved.

    Returns the path of the written file.
    """
    base = base_dir or KNOWLEDGE_DIR
    base.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_path(filename)
    path = base / filename

    # Create parent directories for nested paths
    path.parent.mkdir(parents=True, exist_ok=True)

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


def search_files(query: str, *, base_dir: Path | None = None) -> list[dict]:
    """Search all .md files for a keyword or phrase.

    Scans recursively. Case-insensitive substring search across content
    (not frontmatter). Returns list of {filename, matching_lines}.
    Capped at 10 results.
    """
    base = base_dir or KNOWLEDGE_DIR
    query_lower = query.lower()
    results = []

    for path in _iter_md_files(base):
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
                "filename": _relative_name(path, base),
                "matching_lines": matches[:5],
            })

        if len(results) >= 10:
            break

    return results
