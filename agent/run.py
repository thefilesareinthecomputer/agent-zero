"""Agent Zero CLI -- interactive loop with streaming responses and memory."""

import sys
import threading
import uuid

from pathlib import Path

from agent.agent import create_agent
from agent.config import EFFECTIVE_CHAT_MODEL, EFFECTIVE_KB_REFINE_MODEL, KNOWLEDGE_CANON_PATH
from knowledge.knowledge_store import list_files as kb_list_files

_CANON_DIR = Path(KNOWLEDGE_CANON_PATH)
from memory.entity_registry import init_db as init_entity_db, entity_count
from memory.memory_manager import (
    MAX_MEMORIES,
    forget_all,
    forget_last,
    list_memories,
    memory_count,
    prune,
    store_exchange,
)


def print_banner() -> None:
    print()
    print("  Agent Zero -- local AI agent")
    print("  Commands: 'new' = new thread, 'quit' = exit")
    print("  Memory:   'memories' = list recent, 'forget last' = delete last")
    print("            'forget all' = wipe memory")
    print("  Knowledge: 'knowledge' = list knowledge base files")
    print("  Provider: 'local' / 'cloud' = toggle inference provider, 'provider' = show current")
    print()


def _handle_command(user_input: str) -> bool:
    """Handle CLI commands. Returns True if input was a command."""
    cmd = user_input.lower().strip()

    if cmd == "memories":
        entries = list_memories(limit=10)
        if not entries:
            print("  (no memories)")
        else:
            print(f"  {memory_count()} total memories. Most recent:")
            for i, entry in enumerate(entries, 1):
                meta = entry.get("metadata", {})
                tag = f"{meta.get('category', '?')}/{meta.get('subcategory', '?')}"
                preview = entry["text"][:70].replace("\n", " ")
                if len(entry["text"]) > 70:
                    preview += "..."
                print(f"  {i}. [{tag}] {preview}")
        return True

    if cmd == "forget last":
        deleted = forget_last()
        if deleted:
            preview = deleted[:80].replace("\n", " ")
            print(f"  Forgot: {preview}")
        else:
            print("  Nothing to forget.")
        return True

    if cmd == "forget all":
        n = forget_all()
        print(f"  Wiped {n} memories.")
        return True

    if cmd == "knowledge":
        kb_files = kb_list_files()
        canon_files = kb_list_files(base_dir=_CANON_DIR)

        for f in kb_files:
            f["source"] = ""
        for f in canon_files:
            f["source"] = " [canon]"

        files = kb_files + canon_files
        files.sort(key=lambda r: r["last_modified"], reverse=True)

        if not files:
            print("  (no knowledge files)")
        else:
            print(f"  {len(files)} knowledge file(s):")
            for i, f in enumerate(files, 1):
                tags = ", ".join(f["tags"]) if f["tags"] else "no tags"
                name = f["filename"].replace(".md", "")
                source = f["source"]
                print(f"  {i}. {name}{source}  [{tags}]  modified: {f['last_modified']}")
        return True

    if cmd in ("cloud", "local"):
        from agent.runtime_config import get_provider, set_provider
        set_provider(cmd)
        print(f"  Provider: {get_provider()}")
        return True

    if cmd == "provider":
        from agent.runtime_config import get_provider
        print(f"  Provider: {get_provider()}")
        return True

    return False


def _run_kb_index_bg() -> None:
    """Run KB indexing in a background thread. Prints result if anything changed."""
    try:
        from knowledge.kb_index import sync_kb_index
        result = sync_kb_index()
        if result.get("indexed") or result.get("removed"):
            print(
                f"\r  [KB index: {result['indexed']} indexed, "
                f"{result['removed']} removed]"
            )
    except Exception:
        pass


def run_cli() -> None:
    print_banner()

    use_heavy = "--heavy" in sys.argv
    model = EFFECTIVE_KB_REFINE_MODEL if use_heavy else EFFECTIVE_CHAT_MODEL
    print(f"Loading agent ({model})...")

    # Run KB indexing in a background thread so the CLI prompt appears immediately.
    threading.Thread(target=_run_kb_index_bg, daemon=True).start()

    agent, checkpointer = create_agent(model=model, skip_kb_index=True)

    # Enforce memory capacity cap on startup
    pruned = prune()
    if pruned > 0:
        print(f"Pruned {pruned} overflow memories (capacity cap: {MAX_MEMORIES}).")

    # Initialize entity registry
    init_entity_db()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    label = "26B" if use_heavy else "4B"
    n_entities = entity_count()
    print(f"Ready ({label}). Thread: {thread_id[:8]} | Memories: {memory_count()} | Entities: {n_entities}")
    print("-" * 40)

    try:
        while True:
            try:
                raw = input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            user_input = raw.strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye.")
                break

            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                print(f"New thread: {thread_id[:8]}")
                continue

            # CLI commands (memory, knowledge, etc.)
            if _handle_command(user_input):
                continue

            messages = [{"role": "user", "content": user_input}]
            agent_response = ""

            try:
                for chunk in agent.stream(
                    {"messages": messages},
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in chunk.items():
                        if "messages" not in node_output:
                            continue
                        for msg in node_output["messages"]:
                            if msg.type == "ai":
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(f"  [calling {tc['name']}...]")
                                elif msg.content:
                                    agent_response = msg.content
                                    print(f"\nAgent Zero: {msg.content}")
                            elif msg.type == "tool":
                                content = msg.content[:200]
                                if len(msg.content) > 200:
                                    content += "..."
                                print(f"  [{msg.name}] {content}")

                # Store the exchange in memory
                if agent_response:
                    store_exchange(user_input, agent_response, thread_id)

                    # Log turn to training data JSONL for fine-tuning pipeline.
                    try:
                        from agent.runtime_config import get_provider
                        from fine_tuning.capture import log_turn
                        log_turn(
                            thread_id,
                            get_provider(),
                            model,
                            "heavy" if use_heavy else "fast",
                            [
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": agent_response},
                            ],
                        )
                    except Exception:
                        pass  # training log failure must not break chat

            except KeyboardInterrupt:
                print("\n(interrupted)")
            except Exception as e:
                print(f"\nError: {e}")

    finally:
        if hasattr(checkpointer, "conn"):
            checkpointer.conn.close()


if __name__ == "__main__":
    run_cli()
