"""Agent Zero -- LangGraph ReAct agent with SQLite checkpointing and memory."""

import sqlite3
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from agent.config import (
    AGENT_DB_PATH, MAIN_MODEL, NUM_CTX, NUM_PREDICT, OLLAMA_BASE_URL,
)
from agent.tools import (
    get_current_time, read_file, run_shell_command, write_file,
    list_knowledge, read_knowledge, read_knowledge_section,
    search_knowledge, save_knowledge, update_project_context,
)
from knowledge.kb_index import search_kb as _search_kb, sync_kb_index
from memory.memory_manager import get_relevant_context, memory_count


SYSTEM_PROMPT = (
    "You are Agent Zero, a local AI assistant running on Apple Silicon.\n\n"
    "You have two memory systems:\n"
    "- Conversation memory: automatic. Past exchanges are retrieved and shown "
    "to you when relevant. You do not need to manage this.\n"
    "- Knowledge base: a folder of markdown files you manage. Use "
    "search_knowledge to find relevant files by topic (returns summaries, "
    "not full content). Use read_knowledge to load the full content when "
    "you need to work with a file. For large files, read_knowledge will "
    "list available sections -- use read_knowledge_section to load a "
    "specific section. Before creating a new file, search to see if the "
    "topic is already covered. To edit an existing file, read it first, "
    "then save the full updated content. Prefer updating existing files "
    "over creating new ones for the same topic.\n\n"
    "When you produce a valuable synthesis in conversation -- a comparison, "
    "analysis, how-to, or research summary -- consider saving it as a "
    "knowledge file so it is available in future sessions rather than lost "
    "in chat history.\n\n"
    "You can also update CLAUDE.md files in project directories using "
    "update_project_context. This assembles relevant knowledge into a file "
    "that Claude Code reads automatically. When saving a knowledge file with "
    "save_knowledge, set the project parameter (e.g. project='agent-zero') "
    "to associate it with a specific project. Only files with a matching "
    "project value are included in CLAUDE.md -- files without one are not. "
    "Files tagged private or secret are never included in CLAUDE.md.\n\n"
    "Some knowledge files are marked [canon] -- these are read-only reference "
    "files maintained by the user. You cannot edit or delete them. Treat canon "
    "content as authoritative. They appear alongside regular files in list, "
    "read, and search results.\n\n"
    "You have tools for: checking time, running shell commands, reading/writing "
    "files, managing your knowledge base, and updating project context. "
    "Use tools when they help. Be direct and concise."
)

TOOLS = [
    get_current_time, run_shell_command, read_file, write_file,
    list_knowledge, read_knowledge, read_knowledge_section,
    search_knowledge, save_knowledge, update_project_context,
]


def _build_prompt(state: dict) -> list:
    """Build system prompt with relevant memories injected."""
    messages = state.get("messages", [])

    # Find the latest user message for memory retrieval
    query = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # Base system prompt, optionally enriched with memories and knowledge
    system_content = SYSTEM_PROMPT

    if query:
        # Inject relevant conversation memories
        if memory_count() > 0:
            memories = get_relevant_context(query, top_k=3)
            if memories:
                system_content += (
                    "\n\nRelevant context from past conversations:\n"
                    + "\n---\n".join(memories)
                )

        # Inject relevant KB summaries for topic awareness
        try:
            kb_hits = _search_kb(query, top_k=3)
            if kb_hits:
                system_content += (
                    "\n\nRelevant knowledge base files:\n"
                    + "\n".join(
                        f"- {h['filename']}: {h['heading']} -- {h['summary']}"
                        for h in kb_hits
                    )
                )
        except Exception:
            pass  # KB index not available -- skip injection

    return [SystemMessage(content=system_content)] + list(messages)


def create_agent(
    model: str | None = None,
    db_path: str | None = None,
) -> tuple:
    """Create the LangGraph ReAct agent with SQLite checkpointing.

    Returns (agent, checkpointer). Caller should close checkpointer.conn
    when done.
    """
    model = model or MAIN_MODEL
    db_path = db_path or AGENT_DB_PATH

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Sync KB vector index with files on disk
    try:
        sync_kb_index()
    except Exception:
        pass  # KB index failure should not block agent startup

    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
    )

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
) -> tuple:
    """Create the LangGraph ReAct agent with async SQLite checkpointing.

    For use in the FastAPI web server. Uses AsyncSqliteSaver so
    agent.astream() works natively in async handlers.

    Returns (agent, checkpointer). Caller should await checkpointer.conn.close()
    when done.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    model = model or MAIN_MODEL
    db_path = db_path or AGENT_DB_PATH

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Sync KB vector index with files on disk
    import asyncio
    try:
        await asyncio.to_thread(sync_kb_index)
    except Exception:
        pass  # KB index failure should not block agent startup

    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
    )

    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=_build_prompt,
        checkpointer=checkpointer,
    )

    return agent, checkpointer
