#!/usr/bin/env python3
"""Re-embed existing ChromaDB collections with the new embedding model.

Reads all documents and metadata from both the conversations and knowledge
collections, deletes them, recreates with OllamaEmbedding, and re-adds
the data. The new embedding function re-embeds automatically on add.

For the knowledge collection, it's cleaner to just re-index from files
since that also regenerates summaries. This script preserves conversation
memories that would otherwise be lost.

Usage:
    python scripts/reembed.py [--kb-only] [--memory-only]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import chromadb

from agent.config import CHROMA_DB_PATH
from memory.embeddings import OllamaEmbedding


def reembed_conversations(client: chromadb.ClientAPI) -> int:
    """Re-embed the conversations collection. Returns doc count."""
    try:
        old = client.get_collection("conversations")
    except Exception:
        print("  No conversations collection found -- skipping.")
        return 0

    total = old.count()
    if total == 0:
        print("  Conversations collection is empty -- skipping.")
        return 0

    # Read all data
    data = old.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    print(f"  Read {len(ids)} conversation memories.")

    # Delete and recreate with new embedding function
    client.delete_collection("conversations")
    new = client.get_or_create_collection(
        name="conversations",
        metadata={"hnsw:space": "cosine"},
        embedding_function=OllamaEmbedding(),
    )

    # Re-add in batches (ChromaDB default batch limit is 5461)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        new.add(
            ids=ids[i:end],
            documents=docs[i:end],
            metadatas=metas[i:end],
        )
        print(f"  Re-embedded {end}/{len(ids)} memories...")

    print(f"  Done. {len(ids)} conversation memories re-embedded.")
    return len(ids)


def reembed_knowledge(client: chromadb.ClientAPI) -> int:
    """Re-index knowledge collection from files on disk."""
    try:
        client.delete_collection("knowledge")
        print("  Deleted old knowledge collection.")
    except Exception:
        print("  No existing knowledge collection.")

    # Recreate with new embedding function (sync_kb_index will populate)
    client.get_or_create_collection(
        name="knowledge",
        metadata={"hnsw:space": "cosine"},
        embedding_function=OllamaEmbedding(),
    )

    # Reset module-level state so sync picks up fresh collection
    import knowledge.kb_index as kb_mod
    kb_mod._client = None
    kb_mod._collection = None  # force re-init via _get_kb_collection()

    from knowledge.kb_index import sync_kb_index
    result = sync_kb_index()
    count = result["indexed"]
    print(f"  Re-indexed {count} KB chunks from files on disk.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Re-embed ChromaDB collections")
    parser.add_argument("--kb-only", action="store_true", help="Only re-embed knowledge")
    parser.add_argument("--memory-only", action="store_true", help="Only re-embed conversations")
    args = parser.parse_args()

    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if not args.kb_only:
        print("\n[Conversations]")
        reembed_conversations(client)

    if not args.memory_only:
        print("\n[Knowledge]")
        reembed_knowledge(client)

    print("\nDone. All collections now use OllamaEmbedding (nomic-embed-text).")


if __name__ == "__main__":
    main()
