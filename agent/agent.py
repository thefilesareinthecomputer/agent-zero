"""Agent Zero -- LangGraph ReAct agent with SQLite checkpointing and memory."""

import sqlite3
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from agent.config import (
    AGENT_DB_PATH, MAIN_MODEL, NUM_CTX, NUM_PREDICT,
)
from agent.llm import make_chat_ollama
from knowledge.tokenizer import estimate_gemma_tokens as _estimate_gemma_tokens
from agent.tools import (
    get_current_time, read_file, run_shell_command, write_file,
    draft_knowledge, list_knowledge, read_knowledge,
    read_knowledge_section, search_knowledge, save_knowledge,
    snapshot_to_knowledge,
    update_project_context,
    lookup_entity, manage_entity,
)
from knowledge.kb_index import search_kb as _search_kb, sync_kb_index
from memory.memory_manager import (
    get_relevant_context_compact, memory_count,
)


# -- Core prompt (always present) --
_CORE_PROMPT = (
    "You are Agent Zero, local AI on Apple Silicon. Personality: seasoned "
    "senior data engineer. Dry, witty, direct. Opinions welcome. Warm but "
    "never bubbly. Deadpan humor. No emojis, no unicode decorations, no "
    "filler ('Great question!', 'Sure thing!'). Just answer.\n\n"
    "KB is YOUR workspace -- your files, your notes, not the user's.\n\n"
    "Memory: conversation memory (automatic), knowledge base (markdown "
    "files you manage). Entity registry tracks people/places/projects -- "
    "lookup_entity before asking user.\n\n"
    "Tools: time, shell, file read/write, KB, entities, project context. "
    "Use when helpful. No preamble. No post-summaries.\n\n"
    "When user says 'snapshot', 'save that', or 'save this' -- call "
    "snapshot_to_knowledge. Content captured automatically, no need to "
    "reproduce it."
)

# -- KB block (conditional on KB relevance) --
_KB_PROMPT = (
    "\n\n--- KB ACTIVE ---\n"
    "search_knowledge → file summaries. read_knowledge → heading tree "
    "(H1-H5, token costs, no content). read_knowledge_section → load "
    "section. Check cost vs budget before each load.\n\n"
    "Shop trees, load only what needed. Search before creating new files. "
    "Edit: read tree → load sections → save full updated content. "
    "draft_knowledge for big writes (26B refines). save_knowledge for "
    "quick edits. Save valuable synthesis as KB file.\n\n"
    "[canon] = read-only, authoritative. "
    "update_project_context = CLAUDE.md assembly (set project= on save). "
    "private/secret tags excluded.\n\n"
    "RULES:\n"
    "1. PLAN before first KB call.\n"
    "2. Max 3 section loads/response. Tool refuses after.\n"
    "3. Check cost vs budget before each load.\n"
    "4. 'Not found' → pick from tree or respond with what you have.\n"
    "5. Can answer → respond. No loading 'just in case.'\n"
    "6. Budget directives (>> ... <<) override all."
)

TOOLS = [
    get_current_time, run_shell_command, read_file, write_file,
    draft_knowledge, list_knowledge, read_knowledge,
    read_knowledge_section, search_knowledge, save_knowledge,
    snapshot_to_knowledge,
    update_project_context,
    lookup_entity, manage_entity,
]


_KB_MAX_LOADS_PER_RESPONSE = 3


def _count_kb_loads(messages: list) -> tuple[int, int]:
    """Count KB section loads and total KB retrieval tokens since last human message.

    Returns (section_loads, kb_tokens).
    """
    kb_tool_names = {"read_knowledge_section", "read_knowledge", "search_knowledge"}
    kb_tokens = 0
    kb_loads = 0
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            break
        if (hasattr(msg, "type") and msg.type == "tool"
                and hasattr(msg, "name") and msg.name in kb_tool_names
                and isinstance(getattr(msg, "content", None), str)):
            kb_tokens += _estimate_gemma_tokens(msg.content)
            if msg.name == "read_knowledge_section":
                kb_loads += 1
    return kb_loads, kb_tokens


def _build_prompt(state: dict) -> list:
    """Build system prompt with conditional KB block, memories, and budget.

    Core prompt (~350 tokens) is always present. KB workflow + retrieval
    rules (~300 tokens) are injected only when KB content is relevant to
    the query. Conversational messages get ~65% smaller system prompts.
    """
    messages = state.get("messages", [])

    # Find the latest user message for memory retrieval
    query = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # Always start with the slim core prompt
    system_content = _CORE_PROMPT

    # -- Memory injection (always, when available) --
    if query and memory_count() > 0:
        memories = get_relevant_context_compact(query, top_k=3)
        if memories:
            mem_block = "\n".join(f"- {m}" for m in memories)
            system_content += "\n\n[Memory]\n" + mem_block

    # -- KB relevance detection --
    kb_hits = []
    if query:
        try:
            kb_hits = _search_kb(query, top_k=5)
        except Exception:
            pass  # KB index not available

    try:
        kb_loads, kb_tokens = _count_kb_loads(messages)
    except Exception:
        kb_loads, kb_tokens = 0, 0

    kb_relevant = bool(kb_hits) or kb_loads > 0

    # Capture last AI response for snapshot_to_knowledge tool.
    # Walk backward past the current human message, grab first AI with content.
    last_ai = ""
    past_human = False
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            past_human = True
            continue
        if past_human and hasattr(msg, "type") and msg.type == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                last_ai = content
                break

    # Push state to tools module (always, for safety)
    try:
        import agent.tools as _tools_module
        _tools_module._current_kb_loads = kb_loads
        _tools_module._last_agent_response = last_ai
    except Exception:
        pass

    # -- Conditional KB block --
    if kb_relevant:
        system_content += _KB_PROMPT

        # Inject relevant KB file summaries
        if kb_hits:
            lines = ["\n\n[Knowledge base]"]
            for file_hit in kb_hits[:3]:
                fn = file_hit["filename"]
                src = " [canon]" if file_hit["source"] == "canon" else ""
                ftok = file_hit.get("file_tokens", 0)
                outline = file_hit.get("file_outline", "")
                lines.append(
                    f"- {fn}{src} ({ftok:,} tokens, "
                    f"sections: {outline})"
                )
                for h in file_hit["hits"][:2]:
                    lines.append(
                        f"    matched: {h['heading']} -- {h['summary']}"
                    )
            system_content += "\n".join(lines)

        # -- Context budget and retrieval directives --
        all_messages = [SystemMessage(content=system_content)] + list(messages)
        used_tokens = sum(
            _estimate_gemma_tokens(m.content)
            for m in all_messages
            if hasattr(m, "content") and isinstance(m.content, str)
        )
        remaining = NUM_CTX - used_tokens
        generation_reserve = NUM_PREDICT
        available = max(0, remaining - generation_reserve)
        pct_used = int((used_tokens / NUM_CTX) * 100)

        try:
            _tools_module._current_available_tokens = available
        except Exception:
            pass

        # Escalating directives -- visual markers (>> <<) act as attention
        # anchors for small models.
        directive = ""
        if kb_loads >= _KB_MAX_LOADS_PER_RESPONSE:
            directive = (
                "\n\n>> RETRIEVAL LIMIT REACHED: You have loaded "
                f"{kb_loads} sections this response. Do NOT call "
                "read_knowledge_section again. Respond now with the "
                "content you have. <<"
            )
        elif available < 5_000:
            directive = (
                "\n\n>> CONTEXT FULL: Do NOT call any knowledge tools. "
                "Respond immediately with what you have. <<"
            )
        elif available < 10_000:
            directive = (
                "\n\n>> LOW CONTEXT: You may load at most ONE more section. "
                "Pick the single most relevant one, then respond. <<"
            )
        elif available < 20_000:
            directive = (
                "\n\n>> Context getting tight. Be selective with further loads. <<"
            )

        budget_line = (
            f"\n\n[Context budget: ~{used_tokens:,} / {NUM_CTX:,} tokens used "
            f"({pct_used}%) | ~{available:,} available | "
            f"reserve {generation_reserve:,} for response | "
            f"KB: {kb_loads}/{_KB_MAX_LOADS_PER_RESPONSE} sections loaded "
            f"(~{kb_tokens:,} tokens)]"
        )
        all_messages[0] = SystemMessage(
            content=system_content + directive + budget_line
        )
    else:
        # No KB context -- reset tool budget to safe defaults
        try:
            _tools_module._current_available_tokens = 999_999
        except Exception:
            pass
        all_messages = [SystemMessage(content=system_content)] + list(messages)

    return all_messages


def create_agent(
    model: str | None = None,
    db_path: str | None = None,
    skip_kb_index: bool = False,
) -> tuple:
    """Create the LangGraph ReAct agent with SQLite checkpointing.

    Returns (agent, checkpointer). Caller should close checkpointer.conn
    when done. Pass skip_kb_index=True when the caller handles KB indexing
    externally (e.g. in a background thread).
    """
    model = model or MAIN_MODEL
    db_path = db_path or AGENT_DB_PATH

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Sync KB vector index with files on disk
    if not skip_kb_index:
        try:
            sync_kb_index()
        except Exception:
            pass  # KB index failure should not block agent startup

    llm = make_chat_ollama(model=model, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=_build_prompt,
        checkpointer=checkpointer,
    )

    return agent, checkpointer


async def create_async_agent(
    model: str | None = None,
    db_path: str | None = None,
    skip_kb_index: bool = False,
    checkpointer=None,
) -> tuple:
    """Create the LangGraph ReAct agent with async SQLite checkpointing.

    For use in the FastAPI web server. Uses AsyncSqliteSaver so
    agent.astream() works natively in async handlers.

    Returns (agent, checkpointer). Caller should await checkpointer.conn.close()
    when done. Pass skip_kb_index=True when the caller handles KB indexing
    externally (e.g. run once before creating multiple agents).

    Pass checkpointer to reuse an existing AsyncSqliteSaver (e.g. when
    recreating an agent after a provider toggle, preserving conversation state).
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    model = model or MAIN_MODEL
    db_path = db_path or AGENT_DB_PATH

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Sync KB vector index with files on disk
    import asyncio
    if not skip_kb_index:
        try:
            await asyncio.to_thread(sync_kb_index)
        except Exception:
            pass  # KB index failure should not block agent startup

    llm = make_chat_ollama(model=model, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)

    if checkpointer is None:
        conn = await aiosqlite.connect(db_path)
        checkpointer = AsyncSqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=_build_prompt,
        checkpointer=checkpointer,
    )

    return agent, checkpointer
