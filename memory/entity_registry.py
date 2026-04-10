"""Named entity registry -- SQLite-backed store for people, places, things,
projects, and concepts that the agent encounters over time.

Entities have a canonical name, type, aliases (for dedup), and a summary
that the agent maintains. The registry is consulted during memory tagging
to resolve nicknames/aliases to canonical names and avoid duplicate entries.

Schema:
    entities(
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,   -- canonical name (case-preserved)
        entity_type TEXT NOT NULL,          -- person, place, project, concept, thing
        aliases     TEXT DEFAULT '',        -- pipe-separated aliases, lowercase
        summary     TEXT DEFAULT '',        -- agent-maintained description
        first_seen  REAL NOT NULL,          -- unix timestamp
        last_seen   REAL NOT NULL,          -- unix timestamp (updated on every mention)
        mention_count INTEGER DEFAULT 1,    -- how often this entity is referenced
    )
"""

import logging
import sqlite3
import time
import uuid
from pathlib import Path

from agent.config import EFFECTIVE_FAST_MODEL

log = logging.getLogger(__name__)

# Default DB path -- same data/ directory as ChromaDB and SQLite agent memory
_DB_PATH: str | None = None

ENTITY_TYPES = {"person", "place", "project", "concept", "thing", "organization"}


def _get_db_path() -> str:
    """Resolve the entity registry DB path lazily."""
    global _DB_PATH
    if _DB_PATH is None:
        from agent.config import CHROMA_DB_PATH
        _DB_PATH = str(Path(CHROMA_DB_PATH).parent / "entity_registry.db")
    return _DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Open a connection to the entity registry SQLite DB."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create the entities table if it doesn't exist.

    Optionally accepts a custom db_path for testing.
    """
    global _DB_PATH
    if db_path is not None:
        _DB_PATH = db_path

    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL UNIQUE,
                entity_type   TEXT NOT NULL,
                aliases       TEXT DEFAULT '',
                summary       TEXT DEFAULT '',
                first_seen    REAL NOT NULL,
                last_seen     REAL NOT NULL,
                mention_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name_lower
            ON entities (name COLLATE NOCASE)
        """)
        conn.commit()
    finally:
        conn.close()


def resolve_entity(name: str) -> dict | None:
    """Look up an entity by canonical name or alias.

    Case-insensitive. Checks canonical name first, then aliases.
    Returns entity dict or None.
    """
    conn = _get_conn()
    try:
        name_lower = name.lower().strip()

        # Check canonical name
        row = conn.execute(
            "SELECT * FROM entities WHERE LOWER(name) = ?", (name_lower,)
        ).fetchone()
        if row:
            return _row_to_dict(row)

        # Check aliases (pipe-separated, stored lowercase)
        rows = conn.execute("SELECT * FROM entities WHERE aliases != ''").fetchall()
        for row in rows:
            aliases = row["aliases"].split("|")
            if name_lower in aliases:
                return _row_to_dict(row)

        return None
    finally:
        conn.close()


def register_entity(
    name: str,
    entity_type: str,
    aliases: list[str] | None = None,
    summary: str = "",
) -> dict:
    """Register a new entity or update an existing one.

    If an entity with this name (or alias) already exists, updates
    last_seen, increments mention_count, and merges any new aliases.
    Otherwise creates a new entry.

    Returns the entity dict.
    """
    entity_type = entity_type.lower()
    if entity_type not in ENTITY_TYPES:
        entity_type = "thing"

    # Check if entity already exists (by name or alias)
    existing = resolve_entity(name)
    if existing:
        return touch_entity(existing["id"], new_aliases=aliases, summary=summary)

    # Also check if any of the provided aliases match an existing entity
    for alias in (aliases or []):
        existing = resolve_entity(alias)
        if existing:
            # Add the new name as an alias too
            all_aliases = (aliases or []) + [name]
            return touch_entity(existing["id"], new_aliases=all_aliases, summary=summary)

    now = time.time()
    entity_id = str(uuid.uuid4())
    alias_str = "|".join(a.lower().strip() for a in (aliases or []) if a.strip())

    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO entities (id, name, entity_type, aliases, summary,
               first_seen, last_seen, mention_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
            (entity_id, name.strip(), entity_type, alias_str, summary, now, now),
        )
        conn.commit()
        return get_entity(entity_id)
    finally:
        conn.close()


def touch_entity(
    entity_id: str,
    new_aliases: list[str] | None = None,
    summary: str = "",
) -> dict:
    """Update last_seen, increment mention_count, merge aliases.

    If summary is non-empty, updates the stored summary.
    Returns updated entity dict.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Entity {entity_id} not found")

        # Merge aliases
        existing_aliases = set(row["aliases"].split("|")) if row["aliases"] else set()
        existing_aliases.discard("")
        if new_aliases:
            for a in new_aliases:
                a_lower = a.lower().strip()
                if a_lower and a_lower != row["name"].lower():
                    existing_aliases.add(a_lower)
        alias_str = "|".join(sorted(existing_aliases))

        # Update summary only if a new one is provided
        new_summary = summary if summary else row["summary"]

        conn.execute(
            """UPDATE entities SET
                last_seen = ?,
                mention_count = mention_count + 1,
                aliases = ?,
                summary = ?
               WHERE id = ?""",
            (time.time(), alias_str, new_summary, entity_id),
        )
        conn.commit()
        return get_entity(entity_id)
    finally:
        conn.close()


def get_entity(entity_id: str) -> dict | None:
    """Fetch a single entity by ID."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_entities(
    entity_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List entities, optionally filtered by type.

    Sorted by mention_count descending (most referenced first).
    """
    conn = _get_conn()
    try:
        if entity_type:
            rows = conn.execute(
                """SELECT * FROM entities WHERE entity_type = ?
                   ORDER BY mention_count DESC LIMIT ?""",
                (entity_type.lower(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def search_entities(query: str) -> list[dict]:
    """Search entities by name, alias, or summary substring.

    Case-insensitive. Returns all matches sorted by mention_count.
    """
    conn = _get_conn()
    try:
        q = f"%{query.lower()}%"
        rows = conn.execute(
            """SELECT * FROM entities
               WHERE LOWER(name) LIKE ?
                  OR aliases LIKE ?
                  OR LOWER(summary) LIKE ?
               ORDER BY mention_count DESC""",
            (q, q, q),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def add_alias(entity_id: str, alias: str) -> dict:
    """Add a single alias to an existing entity."""
    return touch_entity(entity_id, new_aliases=[alias])


def update_summary(entity_id: str, summary: str) -> dict:
    """Replace an entity's summary."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE entities SET summary = ? WHERE id = ?",
            (summary, entity_id),
        )
        conn.commit()
        return get_entity(entity_id)
    finally:
        conn.close()


def delete_entity(entity_id: str) -> bool:
    """Delete an entity by ID. Returns True if deleted."""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def entity_count() -> int:
    """Return total number of registered entities."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return row[0]
    finally:
        conn.close()


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text using the fast model (e2b).

    Returns list of {"name": str, "type": str} dicts.
    Falls back to empty list on any error.
    """
    try:
        from agent.llm import make_ollama_client
        client = make_ollama_client()

        prompt = (
            "Extract named entities from this text. For each entity, return "
            "one line in the format: NAME|TYPE\n\n"
            "Types: person, place, project, concept, organization, thing\n\n"
            "Only extract specific, proper entities (not generic nouns). "
            "Return ONLY the entity lines, nothing else. "
            "If no entities found, return NONE.\n\n"
            f"Text: {text[:1000]}"
        )

        response = client.chat(
            model=EFFECTIVE_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200},
        )

        raw = response.message.content.strip()
        if not raw or raw.upper() == "NONE":
            return []

        entities = []
        for line in raw.split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                name = parts[0].strip()
                etype = parts[1].strip().lower()
                if name and etype in ENTITY_TYPES:
                    entities.append({"name": name, "type": etype})

        return entities

    except Exception as e:
        log.warning("Entity extraction failed: %s", e)
        return []


def process_entities(text: str) -> list[dict]:
    """Extract entities from text and register/update them in the registry.

    This is the main entry point called from the memory pipeline.
    Extracts entities via LLM, resolves against existing registry
    (dedup by alias), registers new ones, and touches existing ones.

    Returns list of entity dicts that were registered or updated.
    """
    extracted = extract_entities(text)
    if not extracted:
        return []

    results = []
    for ent in extracted:
        entity = register_entity(
            name=ent["name"],
            entity_type=ent["type"],
        )
        if entity:
            results.append(entity)

    return results


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict with parsed aliases."""
    d = dict(row)
    # Parse aliases from pipe-separated string to list
    d["aliases"] = [a for a in d["aliases"].split("|") if a] if d["aliases"] else []
    return d
