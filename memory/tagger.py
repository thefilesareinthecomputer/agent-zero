"""Memory tagger and novelty checker — uses the fast model (e2b) for:
1. Classifying messages into category/subcategory + update/addition intent
2. Checking whether a new addition contains genuinely new information
"""

import re

import ollama

from agent.config import FAST_MODEL, OLLAMA_BASE_URL

_client: ollama.Client | None = None

CATEGORIES = {
    "user-preference": "favorites, likes, dislikes, tastes, preferences",
    "personal": "identity, biography, goals, journal entries, credentials",
    "project": "work, builds, code, architecture, technical decisions",
    "opinion": "beliefs, worldview, judgments, values",
    "task": "action items, reminders, to-dos, scheduled work",
    "knowledge-base": "facts, quotes, code examples, research, documentation",
    "general": "casual conversation, greetings, anything that doesn't fit elsewhere",
}

_TAG_PROMPT = """Classify this user message. Return TWO lines, nothing else.

Line 1: category/subcategory
Line 2: "update" or "addition"

Categories:
- user-preference (favorites, likes, dislikes — e.g., favorite-color, favorite-food, favorite-music, favorite-movie, favorite-comedy)
- personal (about the user — e.g., identity, goals, journal, credentials, biography)
- project (work/build related — e.g., task-list, architecture, decisions, bugs)
- opinion (beliefs, views — e.g., world-model, tech-opinions, philosophy)
- task (action items — e.g., to-do, reminders, scheduled)
- knowledge-base (info to retain — e.g., quotes, code-examples, research, documentation)
- general (casual chat, greetings, doesn't fit elsewhere)

update vs addition:
- "update" = REPLACES previous info on this topic (e.g., "my favorite color is now green" replaces old color)
- "addition" = ADDS to existing info, can coexist (e.g., "I also like sushi" adds to food list)

Important:
- "Monty Python" is a comedy show, NOT a programming language
- Use the most specific subcategory that fits the MEANING, not individual words
- When in doubt, choose "addition" — it's safer to keep memories than lose them

Examples:
"my favorite color is blue" → user-preference/favorite-color, update
"I also love sushi" → user-preference/favorite-food, addition
"my name is Alex" → personal/identity, update
"I like Monty Python" → user-preference/favorite-comedy, addition
"actually my favorite band is Radiohead" → user-preference/favorite-music, update
"remind me to buy groceries" → task/to-do, addition

User message: "{user_msg}"
"""

_NOVELTY_PROMPT = """You are a memory manager. Decide if a new message contains information worth storing separately from existing memories.

Existing memories on this topic:
{existing_memories}

New message: "{new_msg}"

Does the new message contain meaningful NEW information not already captured above?
- New specific details, facts, or preferences = YES
- Restating, rephrasing, or reacting to what's already known = NO

Answer with ONLY "YES" or "NO".
"""


def _get_client() -> ollama.Client:
    """Lazy-init Ollama client."""
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_BASE_URL)
    return _client


def _normalize(s: str) -> str:
    """Normalize a tag component to lowercase-hyphenated."""
    s = s.strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s


def tag_message(user_msg: str) -> dict[str, str]:
    """Classify a user message into category/subcategory and intent.

    Returns {"category": str, "subcategory": str, "intent": "update" | "addition"}.
    Falls back to general/untagged/addition on any error.
    """
    fallback = {"category": "general", "subcategory": "untagged", "intent": "addition"}

    try:
        client = _get_client()
        prompt = _TAG_PROMPT.format(user_msg=user_msg)
        response = client.chat(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 100},
        )
        raw = response.message.content.strip().lower()

        # Parse category/subcategory
        tag_match = re.search(r"([a-z-]+)/([a-z0-9-]+)", raw)
        if not tag_match:
            return fallback

        category = _normalize(tag_match.group(1))
        subcategory = _normalize(tag_match.group(2))

        if category not in CATEGORIES:
            return fallback

        # Parse intent — default to addition if unclear
        intent = "addition"
        lines = raw.strip().split("\n")
        if len(lines) >= 2:
            second_line = lines[-1].strip()
            if "update" in second_line:
                intent = "update"
            elif "addition" in second_line:
                intent = "addition"

        return {"category": category, "subcategory": subcategory, "intent": intent}

    except Exception:
        return fallback


def check_novelty(new_msg: str, existing_memories: list[str]) -> bool:
    """Ask the fast model whether a new message adds genuinely new information
    beyond what's already in existing memories.

    Returns True if the message is novel and should be stored.
    Returns True on any error (safer to store than lose).
    """
    if not existing_memories:
        return True

    try:
        client = _get_client()
        memories_text = "\n".join(
            f"- {mem[:200]}" for mem in existing_memories
        )
        prompt = _NOVELTY_PROMPT.format(
            existing_memories=memories_text,
            new_msg=new_msg,
        )
        response = client.chat(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 20},
        )
        answer = response.message.content.strip().upper()
        return "YES" in answer

    except Exception:
        return True  # on error, store rather than lose
