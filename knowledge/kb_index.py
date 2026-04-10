"""Knowledge base vector index -- ChromaDB collection for semantic search.

Maintains a separate "knowledge" collection alongside the conversation
memory collection. Each knowledge file is chunked by section and indexed
with LLM-generated summaries for discovery.

Sync is mtime-based: only files that changed since last index are
re-processed. LLM summaries are generated at index time using the fast
model (e2b) via ChatOllama.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

import chromadb

from agent.config import (
    CHROMA_DB_PATH, EMBED_MAX_TOKENS, EFFECTIVE_FAST_MODEL,
    KNOWLEDGE_CANON_PATH, KNOWLEDGE_PATH,
)
from knowledge.chunker import chunk_file
from knowledge.knowledge_store import _parse_frontmatter
from memory.embeddings import OllamaEmbedding

log = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

_KNOWLEDGE_DIR = Path(KNOWLEDGE_PATH)
_CANON_DIR = Path(KNOWLEDGE_CANON_PATH)
_SKIP_FILES = {"index.md", "log.md"}
_MANIFEST_PATH = Path(CHROMA_DB_PATH).parent / "kb_manifest.json"


def _load_manifest() -> dict[str, int]:
    """Load mtime manifest from disk. Returns empty dict if missing."""
    if _MANIFEST_PATH.exists():
        try:
            return json.loads(_MANIFEST_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_manifest(manifest: dict[str, int]) -> None:
    """Write mtime manifest to disk."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _get_kb_collection() -> chromadb.Collection:
    """Lazy-init the knowledge ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=OllamaEmbedding(),
        )
    return _collection


def _generate_summary(content: str, heading: str) -> str:
    """Generate a 1-2 sentence summary of a chunk using the fast model.

    Falls back to mechanical first-sentence extraction if Ollama is
    unreachable or the call fails.
    """
    fallback = _mechanical_summary(content)

    try:
        from agent.llm import make_chat_ollama
        llm = make_chat_ollama(model=EFFECTIVE_FAST_MODEL, num_ctx=16384, num_predict=128)
        prompt = (
            f"Summarize this section titled '{heading}' in 1-2 sentences. "
            "Be specific about what it covers. Return only the summary, "
            "no preamble.\n\n"
            f"{content[:3000]}"
        )
        response = llm.invoke(prompt)
        summary = response.content.strip()
        if summary:
            return summary[:500]
    except Exception as e:
        log.warning("KB index: LLM summary failed for '%s': %s", heading, e)

    return fallback


def _mechanical_summary(content: str, max_chars: int = 200) -> str:
    """Extract the first meaningful line as a fallback summary."""
    for line in content.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped == "---":
            continue
        if stripped.startswith("[") and "](#" in stripped:
            continue
        if stripped.startswith("|"):
            continue
        if len(stripped) > max_chars:
            return stripped[:max_chars] + "..."
        return stripped
    return heading if "heading" in dir() else "(no summary)"


def index_file(
    filename: str,
    source: str,
    base_dir: Path | None = None,
) -> int:
    """Index a single knowledge file into the vector store.

    Reads the file, chunks it, generates LLM summaries, and upserts
    into ChromaDB. Existing chunks for this filename+source are deleted
    first to avoid duplicates.

    Returns the number of chunks indexed.
    """
    base = base_dir or (_CANON_DIR if source == "canon" else _KNOWLEDGE_DIR)
    path = base / filename

    if not path.exists():
        return 0

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        log.warning("KB index: could not read %s", path)
        return 0

    mtime = int(path.stat().st_mtime)

    # Extract frontmatter metadata before chunking
    frontmatter, _ = _parse_frontmatter(text)
    tags = frontmatter.get("tags", [])
    file_tags = ", ".join(str(t) for t in tags) if tags else ""
    file_project = str(frontmatter.get("project", "")) if frontmatter.get("project") else ""
    date_created = str(frontmatter.get("date-created", ""))
    last_modified = str(frontmatter.get("last-modified", ""))

    chunks = chunk_file(text, filename, max_tokens=EMBED_MAX_TOKENS)

    if not chunks:
        return 0

    collection = _get_kb_collection()

    # Compute file-level stats once for all chunks
    file_tokens = sum(c["token_count"] for c in chunks)
    section_count = len(chunks)

    # Build compact outline: unique top-level section names only.
    # Strip nested paths (e.g. "Parent > Child > Leaf" -> "Parent").
    # Dedup and cap at 500 chars to avoid bloating the system prompt.
    seen = []
    for c in chunks:
        top = c["heading"].split(" > ")[0].strip()
        if top not in seen and top != "(preamble)":
            seen.append(top)
    file_outline = " | ".join(seen)
    if len(file_outline) > 500:
        file_outline = file_outline[:497] + "..."

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        summary = _generate_summary(chunk["content"], chunk["heading"])
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)
        documents.append(chunk["content"])
        metadatas.append({
            "filename": filename,
            "source": source,
            "heading": chunk["heading"],
            "chunk_index": chunk["chunk_index"],
            "summary": summary,
            "token_count": chunk["token_count"],
            "mtime": mtime,
            "file_tokens": file_tokens,
            "section_count": section_count,
            "file_outline": file_outline,
            "tags": file_tags,
            "project": file_project,
            "date_created": date_created,
            "last_modified": last_modified,
        })

    # Remove old chunks only immediately before adding new ones.
    # If collection.add() fails (e.g. embedding error on a single chunk),
    # catch and log -- do NOT let one bad file crash the entire sync.
    try:
        remove_file(filename, source)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as e:
        log.warning("KB index: failed to index %s [%s]: %s", filename, source, e)
        return 0

    log.info("KB index: indexed %d chunks from %s [%s]", len(ids), filename, source)
    return len(ids)


def remove_file(filename: str, source: str) -> int:
    """Remove all chunks for a given filename+source from the index.

    Returns the number of chunks removed.
    """
    collection = _get_kb_collection()

    # Query for existing chunks
    results = collection.get(
        where={"$and": [{"filename": filename}, {"source": source}]},
    )
    doc_ids = results["ids"]
    if doc_ids:
        collection.delete(ids=doc_ids)
    return len(doc_ids)


def get_summaries(filename: str, source: str) -> dict[str, str]:
    """Fetch heading -> summary mapping for a file from the index.

    Returns a dict mapping heading names (case-preserved) to their
    LLM-generated summaries. Used to enrich heading trees with context.
    Returns empty dict if file is not indexed.
    """
    collection = _get_kb_collection()
    total = collection.count()
    if total == 0:
        return {}

    try:
        results = collection.get(
            where={"$and": [{"filename": filename}, {"source": source}]},
            include=["metadatas"],
        )
    except Exception:
        return {}

    summaries: dict[str, str] = {}
    for meta in results["metadatas"]:
        heading = meta.get("heading", "")
        summary = meta.get("summary", "")
        if heading and summary:
            summaries[heading] = summary
    return summaries


def search_kb(query: str, top_k: int = 10) -> list[dict]:
    """Semantic search across all KB chunks, grouped by file.

    Returns list of dicts, one per file, ordered by best hit distance:
        filename, source, file_tokens, section_count, file_outline,
        hits: [{heading, summary, chunk_index, distance}]
    """
    collection = _get_kb_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, total),
    )

    # Group raw hits by filename
    by_file: dict[str, dict] = {}
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        fn = meta["filename"]
        if fn not in by_file:
            by_file[fn] = {
                "filename": fn,
                "source": meta["source"],
                "file_tokens": meta.get("file_tokens", 0),
                "section_count": meta.get("section_count", 0),
                "file_outline": meta.get("file_outline", ""),
                "hits": [],
            }
        by_file[fn]["hits"].append({
            "heading": meta["heading"],
            "summary": meta.get("summary", ""),
            "chunk_index": meta["chunk_index"],
            "distance": results["distances"][0][i],
        })

    # Sort files by their best (lowest distance) hit
    grouped = list(by_file.values())
    grouped.sort(key=lambda f: min(h["distance"] for h in f["hits"]))
    return grouped


def sync_kb_index() -> dict[str, int]:
    """Sync the KB vector index with files on disk.

    Uses a local JSON manifest (data/kb_manifest.json) for mtime tracking
    instead of querying ChromaDB metadata. Faster, reliable across ChromaDB
    versions. Re-indexes files whose mtime changed. Removes entries for
    deleted files.

    Returns {"indexed": N, "removed": M}.
    """
    indexed = 0
    removed = 0
    manifest = _load_manifest()
    collection = _get_kb_collection()
    collection_empty = collection.count() == 0

    for source, base_dir in [("knowledge", _KNOWLEDGE_DIR), ("canon", _CANON_DIR)]:
        if not base_dir.exists():
            continue

        # Collect current files on disk
        disk_files: dict[str, float] = {}
        for path in sorted(base_dir.rglob("*.md")):
            rel = str(path.relative_to(base_dir))
            if rel in _SKIP_FILES:
                continue
            disk_files[rel] = path.stat().st_mtime

        # Index new or changed files (mtime check against manifest)
        for filename, mtime in disk_files.items():
            key = f"{source}/{filename}"
            manifest_mtime = manifest.get(key)
            disk_mtime = int(mtime)

            if collection_empty or manifest_mtime is None or disk_mtime != manifest_mtime:
                count = index_file(filename, source, base_dir=base_dir)
                indexed += count
                if count > 0:
                    manifest[key] = disk_mtime

        # Remove entries for deleted files
        prefix = f"{source}/"
        for key in list(manifest.keys()):
            if key.startswith(prefix):
                filename = key[len(prefix):]
                if filename not in disk_files:
                    remove_file(filename, source)
                    removed += 1
                    del manifest[key]

    _save_manifest(manifest)

    # Unload e2b after batch indexing to free VRAM
    if indexed > 0:
        try:
            from bridge.models import sync_unload_model
            from agent.config import TAGGER_MODEL
            sync_unload_model(TAGGER_MODEL)
        except Exception:
            pass

    result = {"indexed": indexed, "removed": removed}
    log.info("KB index sync: %s", result)
    return result
