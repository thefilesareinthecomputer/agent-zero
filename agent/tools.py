"""Agent Zero tools -- @tool decorated functions for the ReAct agent."""

import re
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


# -- Retrieval budget state (written by _build_prompt in agent.py each step) --
_current_available_tokens: int = 999999
_current_kb_loads: int = 0
_KB_MAX_LOADS_PER_RESPONSE: int = 3

# -- Snapshot state (written by _build_prompt each step) --
_last_agent_response: str = ""


# -- Knowledge base tools --

from agent.config import KB_FILE_MAX_TOKENS, KNOWLEDGE_CANON_PATH, KNOWLEDGE_PATH
from knowledge.knowledge_store import (
    list_files as _kb_list,
    read_file as _kb_read,
    save_file as _kb_save,
    search_files as _kb_search,
)
from knowledge.chunker import (
    build_heading_tree as _build_heading_tree,
    chunk_file as _chunk_file,
    enrich_tree_summaries as _enrich_tree_summaries,
    format_heading_tree as _format_heading_tree,
)
from knowledge.tokenizer import (
    count_tokens as _count_tokens,
    estimate_gemma_tokens as _estimate_gemma_tokens,
    truncate_to_tokens as _truncate,
)
from knowledge.kb_index import (
    get_summaries as _get_summaries,
    index_file as _index_file,
    search_kb as _search_kb,
)
from bridge.claude_md import write_claude_md as _write_claude_md
from memory.entity_registry import (
    add_alias as _add_alias,
    list_entities as _list_entities,
    register_entity as _register_entity,
    resolve_entity as _resolve_entity,
    search_entities as _search_entities,
    update_summary as _update_summary,
)

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
    """Returns the heading tree for a knowledge file -- H1-H5 structure
    with token counts per subtree and section summaries. Never loads
    full content. Use read_knowledge_section to load the specific
    sections you need. This lets you shop for content and control
    exactly how many tokens you consume."""
    # Try knowledge/ first, then canon/
    content = _kb_read(filename)
    source = "knowledge"
    if content is None:
        content = _kb_read(filename, base_dir=_CANON_DIR)
        source = "canon"
    if content is None:
        return f"File not found: {filename}"
    if not content:
        return "(file exists but has no section content)"

    tok_count = _estimate_gemma_tokens(content)

    # Build heading tree
    _kb_dir = Path(KNOWLEDGE_PATH)
    file_path = _CANON_DIR / filename if source == "canon" else _kb_dir / filename
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raw = ""

    tree = _build_heading_tree(raw, filename) if raw else None
    if tree:
        # Enrich tree nodes with LLM summaries from the KB index
        try:
            summaries = _get_summaries(filename, source)
            if summaries:
                _enrich_tree_summaries(tree, summaries)
        except Exception:
            pass  # Index unavailable -- tree still works without summaries
    tree_text = _format_heading_tree(tree) if tree else None

    if tree_text:
        tree_cost = _estimate_gemma_tokens(tree_text)
        lines = [
            f"[{tok_count:,} tokens total]",
            "",
            tree_text,
            "",
            f"[Tree cost: ~{tree_cost:,} tokens. "
            f"Each read_knowledge_section call costs additional tokens -- "
            f"check subtree sizes above before loading.]",
        ]
        return "\n".join(lines)

    # No headings -- single-section file, report size
    stem = Path(filename).stem
    return (
        f"[{tok_count:,} tokens, no headings]\n"
        f"Use read_knowledge_section(\"{filename}\", \"{stem}\") to load."
    )


@tool
def read_knowledge_section(filename: str, section: str) -> str:
    """Load a specific section from a knowledge base file. Use after
    read_knowledge to load a section you chose from the heading tree.

    H1 headings are top-level concept boundaries -- passing an H1 heading
    loads the full concept block (including all H2-H5 children).  Passing
    an H2-H5 heading extracts just that subsection from within its parent H1.
    The section parameter is a case-insensitive substring match against
    any heading level.

    This tool enforces retrieval limits. It will refuse to load if:
    - You have already loaded 3 sections this response, or
    - Available context is below 5,000 tokens."""
    # -- Hard enforcement: retrieval limit and context cap --
    if _current_kb_loads >= _KB_MAX_LOADS_PER_RESPONSE:
        return (
            f"REFUSED: Retrieval limit reached ({_KB_MAX_LOADS_PER_RESPONSE} "
            f"sections loaded this response). Respond with what you have."
        )
    if _current_available_tokens < 5000:
        return "REFUSED: Context budget too low to load more content. Respond with what you have."

    # Find the file
    _kb_dir = Path(KNOWLEDGE_PATH)
    path = _kb_dir / filename
    if not path.exists():
        path = _CANON_DIR / filename
    if not path.exists():
        return f"File not found: {filename}"

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"Error reading file: {filename}"

    chunks = _chunk_file(raw, filename)
    section_lower = section.lower()

    # Pass 1: H1 concept heading match (case-insensitive substring)
    for chunk in chunks:
        if section_lower in chunk["heading"].lower():
            if chunk["token_count"] > KB_FILE_MAX_TOKENS:
                truncated, _ = _truncate(chunk["content"], KB_FILE_MAX_TOKENS)
                tok = _estimate_gemma_tokens(truncated)
                return truncated + f"\n\n[Loaded ~{tok:,} tokens (truncated)]"
            tok = _estimate_gemma_tokens(chunk["content"])
            return chunk["content"] + f"\n\n[Loaded ~{tok:,} tokens]"

    # Pass 2: H2-H5 subsection extraction. Scans within each H1 chunk for a
    # markdown heading matching the query and extracts from that heading to the
    # next heading at the same or higher level.
    heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    for chunk in chunks:
        lines = chunk["content"].split("\n")
        for i, line in enumerate(lines):
            m = heading_re.match(line)
            if not m or section_lower not in m.group(2).lower():
                continue
            level = len(m.group(1))
            extracted = [line]
            for j in range(i + 1, len(lines)):
                nm = re.match(r"^(#{1,6})\s", lines[j])
                if nm and len(nm.group(1)) <= level:
                    break
                extracted.append(lines[j])
            result = "\n".join(extracted).strip()
            tok = _count_tokens(result)
            if tok > KB_FILE_MAX_TOKENS:
                result, _ = _truncate(result, KB_FILE_MAX_TOKENS)
                est = _estimate_gemma_tokens(result)
                return result + f"\n\n[Loaded ~{est:,} tokens (truncated)]"
            est = _estimate_gemma_tokens(result)
            return result + f"\n\n[Loaded ~{est:,} tokens]"

    available = [c["heading"] for c in chunks]
    return (
        f"Section '{section}' not found in {filename}.\n"
        f"Available H1 concepts: {', '.join(available)}\n"
        f"Pick one of these, or respond with what you already have. "
        f"Do not guess section names."
    )


@tool
def search_knowledge(query: str) -> str:
    """Search knowledge base by topic. Returns files grouped by relevance
    with section outlines and matched sections. Use read_knowledge to get
    a file's full heading tree, or read_knowledge_section to load content
    directly if you already know the section name."""
    # Semantic search via KB vector index (grouped by file)
    hits = _search_kb(query, top_k=10)

    if hits:
        lines = []
        for file_hit in hits:
            src = " [canon]" if file_hit["source"] == "canon" else ""
            ftok = file_hit.get("file_tokens", 0)
            outline = file_hit.get("file_outline", "")
            lines.append(f"-- {file_hit['filename']}{src} ({ftok:,} tokens)")
            if outline:
                lines.append(f"   sections: {outline}")
            for h in file_hit["hits"]:
                lines.append(
                    f"   > {h['heading']} -- {h.get('summary', '')}"
                )
        lines.append(
            f"\n[{len(hits)} files matched. Use read_knowledge to browse "
            f"heading trees before loading sections.]"
        )
        return "\n".join(lines)

    # Fallback to substring search if KB index is empty (first run)
    kb_results = _kb_search(query)
    canon_results = _kb_search(query, base_dir=_CANON_DIR)
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
        # Re-index in the KB vector store
        try:
            _index_file(filename, source="knowledge")
        except Exception:
            pass  # Index failure should not block save
        return f"Saved: {path}"
    except Exception as e:
        return f"Error saving knowledge file: {e}"


@tool
async def draft_knowledge(
    filename: str,
    draft_content: str,
    tags: str,
    instructions: str,
    project: str = "",
) -> str:
    """Draft a knowledge base file for refinement by the heavy model (26B).
    Write your rough draft as draft_content with ## section headings, and put
    your recommended improvements in instructions. The heavy model will refine
    the draft and save the final version. Use this instead of save_knowledge
    when the content would benefit from deeper analysis and restructuring.
    Tags are comma-separated simple words. Cannot overwrite canon files."""
    if _is_canon_file(filename):
        return f"Cannot save: {filename} is a canon file (read-only)."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        import asyncio
        try:
            asyncio.get_running_loop()
            is_async = True
        except RuntimeError:
            is_async = False

        if is_async:
            from agent.kb_refine import refine_kb_draft
            result = await refine_kb_draft(
                filename, draft_content, instructions, tag_list,
                project=project or None,
            )
        else:
            from agent.kb_refine import refine_kb_draft_sync
            result = refine_kb_draft_sync(
                filename, draft_content, instructions, tag_list,
                project=project or None,
            )
        return result
    except Exception as e:
        return f"Error drafting knowledge file: {e}"


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filename-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:max_len]


@tool
def snapshot_to_knowledge(topic: str = "") -> str:
    """Save your last response as a knowledge base file. Call this when the
    user says 'snapshot', 'save that', 'save this', or asks to save your
    last output. You do NOT need to reproduce the content -- it is captured
    automatically. Just provide an optional short topic name."""
    if not _last_agent_response or not _last_agent_response.strip():
        return "Nothing to snapshot -- no prior response found."

    content = _last_agent_response

    # Generate filename
    if topic:
        filename = f"{_slugify(topic)}.md"
    else:
        m = re.search(r'^#{1,3}\s+(.+)', content, re.MULTILINE)
        if m:
            filename = f"{_slugify(m.group(1))}.md"
        else:
            filename = f"snapshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"

    # Generate tags from topic or headings
    if topic:
        words = re.findall(r'[a-z]{3,}', topic.lower())
        tags = [w for w in words if w not in ('the', 'and', 'for', 'with', 'from')][:5]
    else:
        headings = re.findall(r'^#{1,3}\s+(.+)', content, re.MULTILINE)
        words = re.findall(r'[a-z]{3,}', ' '.join(headings).lower())
        tags = [w for w in words if w not in ('the', 'and', 'for', 'with', 'from')][:5]
    if not tags:
        tags = ["snapshot"]

    # Check for existing file with same name
    if _is_canon_file(filename):
        return f"Cannot save: {filename} is a canon file (read-only)."

    try:
        path = _kb_save(filename, content, tags)
        try:
            _index_file(filename, source="knowledge")
        except Exception:
            pass
        tok = _estimate_gemma_tokens(content)
        return f"Saved snapshot: {path} (~{tok:,} tokens, tags: {', '.join(tags)})"
    except Exception as e:
        return f"Error saving snapshot: {e}"


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


# -- Entity registry tools --


@tool
def lookup_entity(name: str) -> str:
    """Look up a named entity (person, place, project, concept, etc.) by name
    or alias. Returns the entity's canonical name, type, aliases, summary,
    and mention count. Use this to check if you already know about someone
    or something before asking the user."""
    entity = _resolve_entity(name)
    if not entity:
        # Try substring search as fallback
        matches = _search_entities(name)
        if matches:
            lines = [f"No exact match for '{name}'. Similar entities:"]
            for m in matches[:5]:
                aliases = ", ".join(m["aliases"]) if m["aliases"] else "none"
                lines.append(
                    f"- {m['name']} ({m['entity_type']}) -- "
                    f"aliases: {aliases} -- {m['summary'] or 'no summary'}"
                )
            return "\n".join(lines)
        return f"No entity found for '{name}'."

    aliases = ", ".join(entity["aliases"]) if entity["aliases"] else "none"
    return (
        f"Name: {entity['name']}\n"
        f"Type: {entity['entity_type']}\n"
        f"Aliases: {aliases}\n"
        f"Summary: {entity['summary'] or '(no summary yet)'}\n"
        f"Mentions: {entity['mention_count']}\n"
        f"First seen: {entity['first_seen']}\n"
        f"Last seen: {entity['last_seen']}"
    )


@tool
def manage_entity(
    name: str,
    action: str,
    value: str = "",
) -> str:
    """Manage the entity registry. Actions:
    - 'add_alias': add an alias for an existing entity (value = the alias)
    - 'update_summary': set an entity's summary (value = new summary)
    - 'register': create a new entity (value = entity type: person/place/project/concept/organization/thing)
    - 'list': list all entities (name is ignored, value = optional type filter)

    Entities are automatically extracted from conversations. Use this tool
    to manually correct, merge, or enrich entity records."""
    action = action.lower().strip()

    if action == "list":
        entities = _list_entities(entity_type=value if value else None, limit=30)
        if not entities:
            return "No entities registered yet."
        lines = []
        for e in entities:
            aliases = ", ".join(e["aliases"]) if e["aliases"] else ""
            alias_str = f" (aka {aliases})" if aliases else ""
            lines.append(
                f"- {e['name']}{alias_str} [{e['entity_type']}] "
                f"x{e['mention_count']} -- {e['summary'] or 'no summary'}"
            )
        return "\n".join(lines)

    if action == "register":
        entity = _register_entity(name, entity_type=value or "thing")
        return f"Registered: {entity['name']} ({entity['entity_type']})"

    # For add_alias and update_summary, resolve the entity first
    entity = _resolve_entity(name)
    if not entity:
        return f"Entity '{name}' not found. Register it first."

    if action == "add_alias":
        if not value:
            return "Provide the alias as the value parameter."
        updated = _add_alias(entity["id"], value)
        return f"Added alias '{value}' to {updated['name']}. Aliases: {', '.join(updated['aliases'])}"

    if action == "update_summary":
        if not value:
            return "Provide the summary as the value parameter."
        updated = _update_summary(entity["id"], value)
        return f"Updated summary for {updated['name']}."

    return f"Unknown action: {action}. Use add_alias, update_summary, register, or list."
