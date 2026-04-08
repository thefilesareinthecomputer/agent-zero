"""Knowledge base vector index -- ChromaDB collection for semantic search.

Maintains a separate "knowledge" collection alongside the conversation
memory collection. Each knowledge file is chunked by section and indexed
with LLM-generated summaries for discovery.

Sync is mtime-based: only files that changed since last index are
re-processed. LLM summaries are generated at index time using the fast
model (e2b) via ChatOllama.
"""

import logging
import os
import time
import uuid
from pathlib import Path

import chromadb

from agent.config import (
    CHROMA_DB_PATH, FAST_MODEL, KNOWLEDGE_CANON_PATH, KNOWLEDGE_PATH,
    OLLAMA_BASE_URL,
)
from knowledge.chunker import chunk_file

log = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

_KNOWLEDGE_DIR = Path(KNOWLEDGE_PATH)
_CANON_DIR = Path(KNOWLEDGE_CANON_PATH)
_SKIP_FILES = {"index.md", "log.md"}


def _get_kb_collection() -> chromadb.Collection:
    """Lazy-init the knowledge ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _generate_summary(content: str, heading: str) -> str:
    """Generate a 1-2 sentence summary of a chunk using the fast model.

    Falls back to mechanical first-sentence extraction if Ollama is
    unreachable or the call fails.
    """
    fallback = _mechanical_summary(content)

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=FAST_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            num_predict=128,
        )
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

    mtime = path.stat().st_mtime
    chunks = chunk_file(text, filename)

    if not chunks:
        return 0

    # Remove existing chunks for this file
    remove_file(filename, source)

    collection = _get_kb_collection()
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
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
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


def search_kb(query: str, top_k: int = 10) -> list[dict]:
    """Semantic search across all KB chunks.

    Returns list of dicts with discovery-level info (no full content):
        filename, source, heading, summary, chunk_index, distance
    """
    collection = _get_kb_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, total),
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "filename": meta["filename"],
            "source": meta["source"],
            "heading": meta["heading"],
            "summary": meta.get("summary", ""),
            "chunk_index": meta["chunk_index"],
            "distance": results["distances"][0][i],
        })
    return hits


def sync_kb_index() -> dict[str, int]:
    """Sync the KB vector index with files on disk.

    Scans both knowledge/ and knowledge_canon/ directories. Re-indexes
    files whose mtime has changed. Removes index entries for deleted files.

    Returns {"indexed": N, "removed": M}.
    """
    indexed = 0
    removed = 0

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

        # Get indexed files from ChromaDB
        collection = _get_kb_collection()
        total = collection.count()
        if total > 0:
            all_docs = collection.get(
                where={"source": source},
                include=["metadatas"],
            )
            # Build map of filename -> max mtime from indexed chunks
            indexed_mtimes: dict[str, float] = {}
            indexed_ids_by_file: dict[str, list[str]] = {}
            for i, doc_id in enumerate(all_docs["ids"]):
                meta = all_docs["metadatas"][i]
                fn = meta["filename"]
                mt = meta.get("mtime", 0.0)
                indexed_mtimes[fn] = max(indexed_mtimes.get(fn, 0.0), mt)
                indexed_ids_by_file.setdefault(fn, []).append(doc_id)
        else:
            indexed_mtimes = {}
            indexed_ids_by_file = {}

        # Index new or changed files
        for filename, mtime in disk_files.items():
            existing_mtime = indexed_mtimes.get(filename)
            if existing_mtime is None or abs(mtime - existing_mtime) > 0.5:
                count = index_file(filename, source, base_dir=base_dir)
                indexed += count

        # Remove entries for deleted files
        for filename in indexed_ids_by_file:
            if filename not in disk_files:
                count = remove_file(filename, source)
                removed += count

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
