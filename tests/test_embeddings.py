"""Tests for memory/embeddings.py -- OllamaEmbedding function."""

from unittest.mock import MagicMock, patch

from memory.embeddings import OllamaEmbedding


class TestOllamaEmbedding:
    def test_returns_embedding_per_document(self):
        """Each input document should produce an embedding vector."""
        mock_client = MagicMock()
        mock_client.embed.return_value = {
            "embeddings": [[0.1, 0.2, 0.3] * 256]
        }

        embed_fn = OllamaEmbedding()
        with patch("ollama.Client", return_value=mock_client):
            result = embed_fn(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768

    def test_handles_batch_inputs(self):
        """Multiple documents should each get their own embedding."""
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            {"embeddings": [[0.1] * 768]},
            {"embeddings": [[0.2] * 768]},
            {"embeddings": [[0.3] * 768]},
        ]

        embed_fn = OllamaEmbedding()
        with patch("ollama.Client", return_value=mock_client):
            result = embed_fn(["doc one", "doc two", "doc three"])

        assert len(result) == 3
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3

    def test_calls_ollama_with_correct_model(self):
        """Should use EMBED_MODEL from config."""
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.0] * 768]}

        embed_fn = OllamaEmbedding()
        with patch("ollama.Client", return_value=mock_client):
            embed_fn(["test"])

        from agent.config import EMBED_MODEL
        mock_client.embed.assert_called_once_with(
            model=EMBED_MODEL, input="test",
        )

    def test_uses_correct_host(self):
        """Should connect to OLLAMA_BASE_URL from config."""
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.0] * 768]}

        embed_fn = OllamaEmbedding()
        with patch("ollama.Client", return_value=mock_client) as mock_cls:
            embed_fn(["test"])

        from agent.config import OLLAMA_BASE_URL
        mock_cls.assert_called_once_with(host=OLLAMA_BASE_URL)
