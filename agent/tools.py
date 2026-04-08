"""Agent Zero tools -- @tool decorated functions for the ReAct agent."""

import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_shell_command(command: str) -> str:
    """Run a shell command and return its output. Use for system tasks like
    listing files, checking processes, getting system info, etc."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\nReturn code: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file and return them as a string."""
    try:
        path = Path(file_path).expanduser()
        if not path.exists():
            return f"File not found: {file_path}"
        if not path.is_file():
            return f"Not a file: {file_path}"
        content = path.read_text(encoding="utf-8")
        if len(content) > 10_000:
            return content[:10_000] + f"\n\n[truncated — {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if they don't exist."""
    try:
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


# -- Knowledge base tools --

from agent.config import KNOWLEDGE_CANON_PATH
from knowledge.knowledge_store import (
    list_files as _kb_list,
    read_file as _kb_read,
    save_file as _kb_save,
    search_files as _kb_search,
)
from bridge.claude_md import write_claude_md as _write_claude_md

_CANON_DIR = Path(KNOWLEDGE_CANON_PATH)


def _is_canon_file(filename: str) -> bool:
    """Check if a filename exists in the canon directory."""
    return (_CANON_DIR / filename).exists()


def _merged_list(**kwargs) -> list[dict]:
    """List files from both knowledge/ and knowledge_canon/, merged.

    Canon files are marked with source="canon". Knowledge files with
    source="knowledge". Sorted by last_modified descending.
    """
    kb_files = _kb_list(**kwargs)
    for f in kb_files:
        f["source"] = "knowledge"

    canon_files = _kb_list(base_dir=_CANON_DIR, **kwargs)
    for f in canon_files:
        f["source"] = "canon"

    merged = kb_files + canon_files
    merged.sort(key=lambda r: r["last_modified"], reverse=True)
    return merged


@tool
def list_knowledge() -> str:
    """List all knowledge base files with their tags and last-modified dates.
    Includes both editable files and read-only canon files."""
    files = _merged_list()
    if not files:
        return "Knowledge base is empty."
    lines = []
    for f in files:
        tags = ", ".join(f["tags"]) if f["tags"] else "no tags"
        source = " [canon]" if f["source"] == "canon" else ""
        lines.append(
            f"{f['filename']}{source}  [{tags}]  modified: {f['last_modified']}"
        )
    return "\n".join(lines)


@tool
def read_knowledge(filename: str) -> str:
    """Read a knowledge base file by name. Returns the section content.
    Searches both editable and canon files. Use list_knowledge to see
    available files."""
    # Try knowledge/ first, then canon/
    content = _kb_read(filename)
    if content is None:
        content = _kb_read(filename, base_dir=_CANON_DIR)
    if content is None:
        return f"File not found: {filename}"
    if not content:
        return "(file exists but has no section content)"
    return content


@tool
def search_knowledge(query: str) -> str:
    """Search all knowledge base files for a keyword or phrase. Searches
    both editable and canon files. Returns matching filenames and lines."""
    kb_results = _kb_search(query)
    canon_results = _kb_search(query, base_dir=_CANON_DIR)

    # Tag canon results
    for r in canon_results:
        r["filename"] = r["filename"] + " [canon]"

    results = kb_results + canon_results
    if not results:
        return f"No matches for '{query}' in knowledge base."
    lines = []
    for r in results:
        lines.append(f"-- {r['filename']}")
        for ml in r["matching_lines"]:
            lines.append(f"   {ml}")
    return "\n".join(lines)


@tool
def save_knowledge(filename: str, content: str, tags: str, project: str = "") -> str:
    """Create or update a knowledge base file. Content should be organized as
    ## sections with text underneath. To edit an existing file, read it first
    with read_knowledge, make changes, then save the full updated content.
    Tags are comma-separated simple words (e.g. 'ai, memory, agent-zero') -- no colons.
    Project is an optional project name written as a top-level frontmatter field.
    Cannot overwrite canon (read-only) files."""
    # Block writes to files that exist in canon
    if _is_canon_file(filename):
        return f"Cannot save: {filename} is a canon file (read-only)."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        path = _kb_save(filename, content, tag_list, project=project or None)
        return f"Saved: {path}"
    except Exception as e:
        return f"Error saving knowledge file: {e}"


# -- CLAUDE.md bridge tools --


@tool
def update_project_context(project_path: str, project_name: str) -> str:
    """Update the CLAUDE.md file in a project directory. Reads knowledge
    files tagged with project:<project_name> from both editable and canon
    knowledge, assembles a structured CLAUDE.md, and writes it to
    project_path. Claude Code reads this file automatically at session start.
    Files tagged private or secret are excluded.

    project_path can be an absolute path or a name/relative path. Relative
    paths resolve under project_outputs/ in the agent root -- do not pass
    bare names like 'desktop' expecting system paths."""
    try:
        path = _write_claude_md(project_path, project_name)
        return f"Updated CLAUDE.md at {path}"
    except Exception as e:
        return f"Error updating project context: {e}"
