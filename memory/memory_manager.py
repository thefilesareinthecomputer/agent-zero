"""Unified memory interface — stores, retrieves, deduplicates, and prunes memories.

Memory management pipeline:
1. Noise filter — short banter (< 3 words) is never stored
2. Tagging — e2b classifies into category/subcategory + update/addition intent
3. Updates — same subcategory + similar = replace (contradiction detection)
4. Additions — same subcategory + existing memories = LLM novelty check
   (asks e2b: "does this add new info?") — solves the threshold problem
   where cosine distance can't distinguish "pizza yay" from "pepperoni with hot honey"
5. Dedup — near-identical messages (any intent) just refresh timestamp
"""

import time

from memory import vector_store
from memory.entity_registry import init_db as _init_entity_db, process_entities
from memory.tagger import check_novelty, tag_message

# Similarity thresholds (cosine distance — lower = more similar)
DEDUP_THRESHOLD = 0.15       # near-identical → refresh timestamp
CONTRADICT_THRESHOLD = 0.50  # for updates: same subtopic → replace

# Noise filter
MIN_WORDS_TO_STORE = 3

# Capacity cap (no age-based pruning -- memories persist indefinitely)
MAX_MEMORIES = 2000

# Recency tiebreaker weight -- additive penalty on distance.
# Only breaks ties between semantically similar results. A more relevant
# but older memory always outranks a less relevant but newer one.
# At 0.0001: 30-day penalty = 0.072, 1-year = 0.876, 1-hour = 0.0001
RECENCY_WEIGHT = 0.0001


def _generate_memory_summary(user_msg: str, agent_msg: str) -> str:
    """Generate a compact 1-sentence summary of an exchange using e2b.

    Falls back to the first 200 chars of the user message if the LLM
    is unavailable or returns garbage.
    """
    try:
        from langchain_ollama import ChatOllama
        from agent.config import FAST_MODEL, OLLAMA_BASE_URL

        llm = ChatOllama(
            model=FAST_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_ctx=2048,
            num_predict=64,
        )
        prompt = (
            "Summarize this exchange in one sentence. "
            "Be specific. Return only the summary.\n\n"
            f"User: {user_msg[:500]}\n"
            f"Agent: {agent_msg[:500]}"
        )
        response = llm.invoke(prompt)
        summary = response.content.strip()
        if summary:
            return summary[:200]
    except Exception:
        pass
    return user_msg[:200]


def _store_and_extract(text: str, metadata: dict, user_msg: str) -> str:
    """Store a memory and extract entities from the user message.

    Entity extraction is best-effort -- failures never block storage.
    """
    doc_id = vector_store.store(text, metadata)

    # Extract and register named entities from the user message
    try:
        process_entities(user_msg)
    except Exception:
        pass  # Entity extraction failure must not block memory storage

    return doc_id


def store_exchange(user_msg: str, agent_msg: str, thread_id: str) -> str | None:
    """Store a user/agent exchange with smart dedup, contradiction, and novelty handling.

    Pipeline:
    1. Noise filter (word count)
    2. Tag (category/subcategory/intent via e2b)
    3. Generate compact summary via e2b
    4. Dedup check (any intent — near-identical messages refresh, not stack)
    5. If update → contradiction replacement within subcategory
    6. If addition + existing memories in subcategory → novelty check via e2b
    7. If addition + no existing memories → store directly
    8. Extract named entities and register in entity registry

    Returns the document ID, or None if skipped/deduped.
    """
    # 1. Noise filter
    if len(user_msg.split()) < MIN_WORDS_TO_STORE:
        return None

    # 2. Tag
    tags = tag_message(user_msg)
    category = tags["category"]
    subcategory = tags["subcategory"]
    intent = tags.get("intent", "addition")

    # 3. Generate compact summary
    summary = _generate_memory_summary(user_msg, agent_msg)

    text = f"User: {user_msg}\nAgent: {agent_msg}"
    metadata = {
        "thread_id": thread_id,
        "timestamp": time.time(),
        "type": "exchange",
        "category": category,
        "subcategory": subcategory,
        "intent": intent,
        "summary": summary,
    }

    # Skip smart checks for general/untagged — too broad to compare
    if category == "general" or subcategory == "untagged":
        return _store_and_extract(text, metadata, user_msg)

    # Find similar memories in the same subcategory
    if vector_store.count() == 0:
        return _store_and_extract(text, metadata, user_msg)

    where_filter = {
        "$and": [
            {"category": category},
            {"subcategory": subcategory},
        ]
    }
    similar = vector_store.search(user_msg, top_k=5, where=where_filter)

    if not similar:
        # Nothing in this subcategory yet — store directly
        return _store_and_extract(text, metadata, user_msg)

    # 4. Dedup check — applies to any intent
    closest = similar[0]
    if closest["distance"] < DEDUP_THRESHOLD:
        old_meta = closest["metadata"]
        old_meta["timestamp"] = time.time()
        vector_store.update_metadata(closest["id"], old_meta)
        return None

    # 5. Update → contradiction replacement
    if intent == "update":
        for match in similar:
            if match["distance"] < CONTRADICT_THRESHOLD:
                vector_store.delete([match["id"]])
                break
        return _store_and_extract(text, metadata, user_msg)

    # 6. Addition + existing memories → novelty check
    existing_texts = [m["text"] for m in similar]
    if check_novelty(user_msg, existing_texts):
        return _store_and_extract(text, metadata, user_msg)

    # Not novel — refresh the closest match's timestamp instead
    old_meta = closest["metadata"]
    old_meta["timestamp"] = time.time()
    vector_store.update_metadata(closest["id"], old_meta)
    return None


def get_relevant_context(query: str, top_k: int = 5) -> list[str]:
    """Retrieve past exchanges relevant to the query.

    Ranked by semantic distance (primary) with a tiny recency tiebreaker
    (additive penalty). Age never filters or overrides relevance -- a
    highly relevant year-old memory still outranks a weakly relevant
    recent one.
    """
    results = vector_store.search(query, top_k=top_k)
    if not results:
        return []

    now = time.time()
    for r in results:
        age_hours = (now - r["metadata"].get("timestamp", now)) / 3600
        r["adjusted_distance"] = r["distance"] + (age_hours * RECENCY_WEIGHT)

    results.sort(key=lambda r: r["adjusted_distance"])
    return [r["text"] for r in results]


def get_relevant_context_compact(query: str, top_k: int = 3) -> list[str]:
    """Retrieve compact summaries of relevant memories.

    Returns summary metadata when available (generated at write time).
    Falls back to raw exchange text for legacy docs that predate
    the summary field. Ranked by semantic distance with recency
    tiebreaker -- age never overrides relevance.
    """
    results = vector_store.search(query, top_k=top_k)
    if not results:
        return []

    now = time.time()
    for r in results:
        age_hours = (now - r["metadata"].get("timestamp", now)) / 3600
        r["adjusted_distance"] = r["distance"] + (age_hours * RECENCY_WEIGHT)

    results.sort(key=lambda r: r["adjusted_distance"])
    return [
        r["metadata"].get("summary") or r["text"]
        for r in results
    ]


def forget_all() -> int:
    """Delete all memories. Returns the count that was deleted."""
    n = vector_store.count()
    if n > 0:
        vector_store.delete_all()
    return n


def forget_last() -> str | None:
    """Delete the most recent memory. Returns the deleted text, or None."""
    recent = vector_store.get_all(limit=1)
    if not recent:
        return None
    vector_store.delete([recent[0]["id"]])
    return recent[0]["text"]


def list_memories(limit: int = 10) -> list[dict]:
    """Return the most recent memories with their metadata."""
    return vector_store.get_all(limit=limit)


def prune() -> int:
    """Enforce capacity cap on stored memories. Returns the number pruned.

    No age-based deletion -- memories persist indefinitely. Only removes
    the oldest entries when total count exceeds MAX_MEMORIES. Ground
    truths and old but relevant memories are never destroyed by time alone.
    """
    all_docs = vector_store.get_all()
    if not all_docs or len(all_docs) <= MAX_MEMORIES:
        return 0

    # all_docs is sorted newest-first by get_all() -- drop the tail
    overflow = all_docs[MAX_MEMORIES:]
    to_delete = [d["id"] for d in overflow]

    if to_delete:
        vector_store.delete(to_delete)

    return len(to_delete)


def memory_count() -> int:
    """Return total number of stored memories."""
    return vector_store.count()
