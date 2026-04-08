"""Shared Ollama embedding function for all ChromaDB collections.

Provides a single OllamaEmbedding class that both the conversations and
knowledge collections use. This ensures consistent embeddings across the
entire system and makes swapping models a one-line config change.

IMPORTANT: Ollama ignores options.num_ctx for the /api/embed endpoint.
nomic-embed-text runs at its default 2048 token context regardless of
what you pass. EMBED_MAX_TOKENS (default 1500) accounts for this --
chunks are sized by the chunker to fit within 2048 with headroom.
To use the model's full 8192 context, create a custom Ollama model:
    echo 'FROM nomic-embed-text\nPARAMETER num_ctx 8192' | ollama create nomic-embed-8k
Then set EMBED_MODEL=nomic-embed-8k and EMBED_MAX_TOKENS=7500 in .env.
"""

from chromadb import Documents, EmbeddingFunction, Embeddings

from agent.config import EMBED_MODEL, OLLAMA_BASE_URL


class OllamaEmbedding(EmbeddingFunction):
    """ChromaDB-compatible embedding function using Ollama.

    Calls the Ollama embed endpoint for each document. Chunks must be
    sized to fit the embedding model's context window BEFORE reaching
    this function -- the chunker enforces EMBED_MAX_TOKENS at index time.
    """

    def __init__(self) -> None:
        pass

    def name(self) -> str:
        """Return a unique name for this embedding function."""
        return f"ollama-{EMBED_MODEL}"

    def get_config(self) -> dict:
        """Return config for serialization (ChromaDB future requirement)."""
        return {"model": EMBED_MODEL, "host": OLLAMA_BASE_URL}

    def __call__(self, input: Documents) -> Embeddings:
        import ollama as _ollama_client

        client = _ollama_client.Client(host=OLLAMA_BASE_URL)
        results = []
        for text in input:
            resp = client.embed(model=EMBED_MODEL, input=text)
            results.append(resp["embeddings"][0])
        return results
