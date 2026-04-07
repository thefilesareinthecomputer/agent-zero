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
from memory.tagger import check_novelty, tag_message

# Similarity thresholds (cosine distance — lower = more similar)
DEDUP_THRESHOLD = 0.15       # near-identical → refresh timestamp
CONTRADICT_THRESHOLD = 0.50  # for updates: same subtopic → replace

# Noise filter
MIN_WORDS_TO_STORE = 3

# Pruning
MAX_AGE_HOURS = 720  # 30 days
MAX_MEMORIES = 500   # hard cap


def store_exchange(user_msg: str, agent_msg: str, thread_id: str) -> str | None:
    """Store a user/agent exchange with smart dedup, contradiction, and novelty handling.

    Pipeline:
    1. Noise filter (word count)
    2. Tag (category/subcategory/intent via e2b)
    3. Dedup check (any intent — near-identical messages refresh, not stack)
    4. If update → contradiction replacement within subcategory
    5. If addition + existing memories in subcategory → novelty check via e2b
    6. If addition + no existing memories → store directly

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

    text = f"User: {user_msg}\nAgent: {agent_msg}"
    metadata = {
        "thread_id": thread_id,
        "timestamp": time.time(),
        "type": "exchange",
        "category": category,
        "subcategory": subcategory,
        "intent": intent,
    }

    # Skip smart checks for general/untagged — too broad to compare
    if category == "general" or subcategory == "untagged":
        return vector_store.store(text, metadata)

    # Find similar memories in the same subcategory
    if vector_store.count() == 0:
        return vector_store.store(text, metadata)

    where_filter = {
        "$and": [
            {"category": category},
            {"subcategory": subcategory},
        ]
    }
    similar = vector_store.search(user_msg, top_k=5, where=where_filter)

    if not similar:
        # Nothing in this subcategory yet — store directly
        return vector_store.store(text, metadata)

    # 3. Dedup check — applies to any intent
    closest = similar[0]
    if closest["distance"] < DEDUP_THRESHOLD:
        old_meta = closest["metadata"]
        old_meta["timestamp"] = time.time()
        vector_store.update_metadata(closest["id"], old_meta)
        return None

    # 4. Update → contradiction replacement
    if intent == "update":
        for match in similar:
            if match["distance"] < CONTRADICT_THRESHOLD:
                vector_store.delete([match["id"]])
                break
        return vector_store.store(text, metadata)

    # 5. Addition + existing memories → novelty check
    existing_texts = [m["text"] for m in similar]
    if check_novelty(user_msg, existing_texts):
        return vector_store.store(text, metadata)

    # Not novel — refresh the closest match's timestamp instead
    old_meta = closest["metadata"]
    old_meta["timestamp"] = time.time()
    vector_store.update_metadata(closest["id"], old_meta)
    return None


def get_relevant_context(query: str, top_k: int = 5) -> list[str]:
    """Retrieve past exchanges relevant to the query, ranked by recency-weighted similarity."""
    results = vector_store.search(query, top_k=top_k)
    if not results:
        return []

    now = time.time()
    for r in results:
        age_hours = (now - r["metadata"].get("timestamp", now)) / 3600
        recency_boost = 1.0 / (1.0 + age_hours * 0.1)
        r["score"] = (1.0 - r["distance"]) * recency_boost

    results.sort(key=lambda r: r["score"], reverse=True)
    return [r["text"] for r in results]


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
    """Remove old and low-value memories. Returns the number pruned."""
    all_docs = vector_store.get_all()
    if not all_docs:
        return 0

    now = time.time()
    to_delete = []

    for doc in all_docs:
        age_hours = (now - doc["metadata"].get("timestamp", now)) / 3600
        if age_hours > MAX_AGE_HOURS:
            to_delete.append(doc["id"])

    remaining = [d for d in all_docs if d["id"] not in set(to_delete)]
    if len(remaining) > MAX_MEMORIES:
        overflow = remaining[MAX_MEMORIES:]
        to_delete.extend(d["id"] for d in overflow)

    if to_delete:
        vector_store.delete(to_delete)

    return len(to_delete)


def memory_count() -> int:
    """Return total number of stored memories."""
    return vector_store.count()
