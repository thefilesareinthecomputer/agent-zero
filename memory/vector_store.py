"""ChromaDB vector store for conversation memory."""

import time
import uuid

import chromadb

from agent.config import CHROMA_DB_PATH
from memory.embeddings import OllamaEmbedding

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Lazy-init persistent ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
            embedding_function=OllamaEmbedding(),
        )
    return _collection


def store(text: str, metadata: dict | None = None) -> str:
    """Embed and store a text document. Returns the document ID."""
    collection = _get_collection()
    doc_id = str(uuid.uuid4())
    meta = metadata or {}
    meta["timestamp"] = meta.get("timestamp", time.time())
    collection.add(
        documents=[text],
        metadatas=[meta],
        ids=[doc_id],
    )
    return doc_id


def search(
    query: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Search for similar documents. Returns list of {id, text, metadata, distance}.

    Optional `where` filter for ChromaDB metadata queries, e.g.:
        {"$and": [{"category": "user-preference"}, {"subcategory": "favorite-color"}]}
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    kwargs = {
        "query_texts": [query],
        "n_results": min(top_k, total),
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    memories = []
    for i in range(len(results["ids"][0])):
        memories.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return memories


def get_all(limit: int = 0) -> list[dict]:
    """Return all documents, sorted by timestamp descending (newest first).

    If limit > 0, return only the most recent `limit` entries.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    for i in range(len(results["ids"])):
        docs.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    docs.sort(key=lambda d: d["metadata"].get("timestamp", 0), reverse=True)
    if limit > 0:
        docs = docs[:limit]
    return docs


def delete(doc_ids: list[str]) -> None:
    """Delete documents by ID."""
    if not doc_ids:
        return
    collection = _get_collection()
    collection.delete(ids=doc_ids)


def delete_all() -> None:
    """Delete all documents in the collection."""
    global _collection
    _get_collection()  # ensure client is initialized
    if _client is not None:
        _client.delete_collection("conversations")
        # Re-create using the same client -- do NOT call _get_collection(),
        # which would reinitialize _client from config when _collection is None
        _collection = _client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
            embedding_function=OllamaEmbedding(),
        )


def update_metadata(doc_id: str, metadata: dict) -> None:
    """Update metadata on an existing document."""
    collection = _get_collection()
    collection.update(ids=[doc_id], metadatas=[metadata])


def count() -> int:
    """Return total number of stored documents."""
    return _get_collection().count()
