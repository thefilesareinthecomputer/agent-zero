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

from knowledge.knowledge_store import (
    list_files as _kb_list,
    read_file as _kb_read,
    save_file as _kb_save,
    search_files as _kb_search,
)


@tool
def list_knowledge() -> str:
    """List all knowledge base files with their tags and last-modified dates."""
    files = _kb_list()
    if not files:
        return "Knowledge base is empty."
    lines = []
    for f in files:
        tags = ", ".join(f["tags"]) if f["tags"] else "no tags"
        lines.append(f"{f['filename']}  [{tags}]  modified: {f['last_modified']}")
    return "\n".join(lines)


@tool
def read_knowledge(filename: str) -> str:
    """Read a knowledge base file by name. Returns the section content for
    viewing or editing. Use list_knowledge to see available files."""
    content = _kb_read(filename)
    if content is None:
        return f"File not found: {filename}"
    if not content:
        return "(file exists but has no section content)"
    return content


@tool
def search_knowledge(query: str) -> str:
    """Search all knowledge base files for a keyword or phrase. Returns
    matching filenames and lines."""
    results = _kb_search(query)
    if not results:
        return f"No matches for '{query}' in knowledge base."
    lines = []
    for r in results:
        lines.append(f"-- {r['filename']}")
        for ml in r["matching_lines"]:
            lines.append(f"   {ml}")
    return "\n".join(lines)


@tool
def save_knowledge(filename: str, content: str, tags: str) -> str:
    """Create or update a knowledge base file. Content should be organized as
    ## sections with text underneath. To edit an existing file, read it first
    with read_knowledge, make changes, then save the full updated content.
    Tags are comma-separated (e.g. 'preferences, food')."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        path = _kb_save(filename, content, tag_list)
        return f"Saved: {path}"
    except Exception as e:
        return f"Error saving knowledge file: {e}"
