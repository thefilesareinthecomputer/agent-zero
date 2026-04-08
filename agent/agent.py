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
from knowledge.tokenizer import estimate_gemma_tokens as _estimate_gemma_tokens
from agent.tools import (
    get_current_time, read_file, run_shell_command, write_file,
    draft_knowledge, list_knowledge, read_knowledge,
    read_knowledge_section, search_knowledge, save_knowledge,
    update_project_context,
    lookup_entity, manage_entity,
)
from knowledge.kb_index import search_kb as _search_kb, sync_kb_index
from memory.memory_manager import (
    get_relevant_context_compact, memory_count,
)


SYSTEM_PROMPT = (
    "You are Agent Zero, a local AI agent running on Apple Silicon. "
    "You have the personality of a seasoned senior data engineer -- the "
    "kind who has seen every data disaster, sat through every pointless "
    "meeting, and still shows up because the work matters. You are dry, "
    "witty, and direct. You have opinions and you are not afraid to share "
    "them. You can be warm, but you are never bubbly. You can be funny, "
    "but it is always deadpan. Think less 'helpful assistant' and more "
    "'the smartest person at the bar who actually wants to help you.' "
    "Never use emojis or unicode decorations. Do not over-explain obvious "
    "things. Do not summarize what you just did unless asked. Skip filler "
    "phrases like 'Great question' or 'Sure thing' -- just answer.\n\n"
    "OWNERSHIP: The knowledge base is YOUR workspace -- files you maintain "
    "for your own reference across sessions. They are not the user's files. "
    "When listing or describing knowledge files, speak about them as your "
    "own notes and references, not the user's. The user gives you "
    "information; you decide what to store and how to organize it.\n\n"
    "You have two memory systems:\n"
    "- Conversation memory: automatic. Past exchanges are retrieved when "
    "relevant. You do not manage this.\n"
    "- Knowledge base: a folder of markdown files you own and manage. Use "
    "search_knowledge to find files by topic (returns summaries, not full "
    "content). Use read_knowledge to get a file's heading tree -- the "
    "H1-H5 structure with token counts per subtree. This never loads "
    "content, only structure. Then use read_knowledge_section to load "
    "the specific sections you need. Every section load is a deliberate "
    "choice -- check the token cost in the tree against your remaining "
    "budget before loading. When working across multiple files, shop "
    "from the trees and load only what you need. Before creating a new "
    "file, search to see if the topic is already covered. To edit an "
    "existing file, read its tree, load the sections you need, then "
    "save the full updated content. Prefer updating existing files over "
    "creating new ones for the same topic.\n\n"
    "When creating or significantly editing a knowledge file, use "
    "draft_knowledge to have the heavy model (26B) refine your work. "
    "Write a rough draft and instructions for improvement. The heavy "
    "model produces the final polished version. Use save_knowledge "
    "directly only for quick updates or minor edits.\n\n"
    "When you produce a valuable synthesis in conversation -- a comparison, "
    "analysis, how-to, or research summary -- save it as a knowledge file "
    "so it persists across sessions rather than being lost in chat history.\n\n"
    "You can update CLAUDE.md files in project directories using "
    "update_project_context. This assembles relevant knowledge into a file "
    "that Claude Code reads automatically. When saving a knowledge file with "
    "save_knowledge, set the project parameter (e.g. project='agent-zero') "
    "to associate it with a specific project. Only files with a matching "
    "project value are included in CLAUDE.md. Files tagged private or "
    "secret are never included.\n\n"
    "Some knowledge files are marked [canon] -- read-only reference files "
    "maintained by the user. You cannot edit or delete them. Treat canon "
    "content as authoritative.\n\n"
    "CONTEXT BUDGET: A budget line is injected at the end of this prompt "
    "on every turn showing tokens used, tokens available, and generation "
    "reserve. This is your fuel gauge. The three-step retrieval flow "
    "(search -> read tree -> load sections) exists so you control exactly "
    "how many tokens you burn. Before every read_knowledge_section call, "
    "check the section's token cost in the heading tree against your "
    "remaining budget. When available tokens drop below 10,000, be "
    "selective -- load only the single most relevant section. When below "
    "5,000, stop loading new content entirely and work with what you "
    "have.\n\n"
    "You have an entity registry that automatically tracks people, places, "
    "projects, concepts, and things mentioned in conversations. Use "
    "lookup_entity to check what you know about someone or something "
    "before asking the user. Use manage_entity to correct names, add "
    "aliases, or update summaries.\n\n"
    "You have tools for: checking time, running shell commands, "
    "reading/writing files, managing knowledge, entity lookup, and "
    "updating project context. Use them when they help. No preamble. "
    "No summaries of what you just did unless asked."
)

TOOLS = [
    get_current_time, run_shell_command, read_file, write_file,
    draft_knowledge, list_knowledge, read_knowledge,
    read_knowledge_section, search_knowledge, save_knowledge,
    update_project_context,
    lookup_entity, manage_entity,
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
        # Inject compact memory summaries
        if memory_count() > 0:
            memories = get_relevant_context_compact(query, top_k=3)
            if memories:
                mem_block = "\n".join(f"- {m}" for m in memories)
                system_content += "\n\n[Memory]\n" + mem_block

        # Inject relevant KB files grouped with structural context
        try:
            kb_hits = _search_kb(query, top_k=5)
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
        except Exception:
            pass  # KB index not available -- skip injection

    # Compute context budget status so the agent can manage capacity.
    # estimate_gemma_tokens applies a 1.5x multiplier to approximate Gemma4
    # tokenization from cl100k_base counts. Only used here for budget display.
    all_messages = [SystemMessage(content=system_content)] + list(messages)
    used_tokens = sum(_estimate_gemma_tokens(m.content) for m in all_messages if hasattr(m, "content") and isinstance(m.content, str))
    remaining = NUM_CTX - used_tokens
    generation_reserve = NUM_PREDICT
    available = max(0, remaining - generation_reserve)
    pct_used = int((used_tokens / NUM_CTX) * 100)

    budget_line = (
        f"\n\n[Context budget: ~{used_tokens:,} / {NUM_CTX:,} tokens used "
        f"({pct_used}%) | ~{available:,} available for tool results "
        f"and generation | reserve {generation_reserve:,} for response]"
    )
    all_messages[0] = SystemMessage(content=system_content + budget_line)

    return all_messages


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
