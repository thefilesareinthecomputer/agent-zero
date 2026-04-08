"""Tests for memory summary generation and compact retrieval.

Tests _generate_memory_summary and get_relevant_context_compact from
memory/memory_manager.py. All LLM calls are mocked.
"""

import time
from unittest.mock import MagicMock, patch

import chromadb
import pytest

import memory.vector_store as vs
import memory.memory_manager as mm


@pytest.fixture(autouse=True)
def fresh_chroma(tmp_path, monkeypatch):
    """Isolated ChromaDB for each test."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="conversations",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(vs, "_client", client)
    monkeypatch.setattr(vs, "_collection", collection)


class TestGenerateMemorySummary:
    def test_returns_llm_summary(self):
        """When LLM succeeds, returns its output."""
        mock_response = MagicMock()
        mock_response.content = "User asked about pizza preferences."

        with patch("memory.memory_manager.ChatOllama", create=True) as MockLLM:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            # Patch the lazy import inside the function
            with patch.dict("sys.modules", {
                "langchain_ollama": MagicMock(ChatOllama=lambda **kwargs: mock_llm)
            }):
                result = mm._generate_memory_summary(
                    "I love pepperoni pizza", "Great choice!"
                )

        assert "pizza" in result.lower()

    def test_fallback_on_llm_failure(self):
        """When LLM raises, falls back to user message prefix."""
        with patch.dict("sys.modules", {
            "langchain_ollama": MagicMock(
                ChatOllama=MagicMock(side_effect=RuntimeError("no model"))
            )
        }):
            result = mm._generate_memory_summary(
                "I love pepperoni pizza with hot honey", "Sounds great!"
            )

        assert result == "I love pepperoni pizza with hot honey"

    def test_truncates_long_summary(self):
        """Summary output capped at 200 chars."""
        long_text = "A" * 500
        mock_response = MagicMock()
        mock_response.content = long_text

        with patch.dict("sys.modules", {
            "langchain_ollama": MagicMock(
                ChatOllama=lambda **kwargs: MagicMock(
                    invoke=MagicMock(return_value=mock_response)
                )
            )
        }):
            result = mm._generate_memory_summary("test", "test")

        assert len(result) <= 200

    def test_fallback_truncates_user_msg(self):
        """Fallback caps user message at 200 chars."""
        long_msg = "word " * 100  # 500 chars
        with patch.dict("sys.modules", {
            "langchain_ollama": MagicMock(
                ChatOllama=MagicMock(side_effect=Exception("fail"))
            )
        }):
            result = mm._generate_memory_summary(long_msg, "ok")

        assert len(result) <= 200


class TestGetRelevantContextCompact:
    def test_returns_summaries_when_present(self):
        """When metadata has summary field, return that instead of raw text."""
        vs.store("User: I like pizza\nAgent: Noted.", {
            "category": "pref", "subcategory": "food",
            "timestamp": time.time(),
            "summary": "User prefers pizza.",
        })
        results = mm.get_relevant_context_compact("pizza")
        assert len(results) > 0
        assert results[0] == "User prefers pizza."

    def test_falls_back_to_raw_text(self):
        """Legacy docs without summary field return raw exchange text."""
        vs.store("User: I like sushi\nAgent: Nice.", {
            "category": "pref", "subcategory": "food",
            "timestamp": time.time(),
            # No summary field
        })
        results = mm.get_relevant_context_compact("sushi")
        assert len(results) > 0
        assert "User: I like sushi" in results[0]

    def test_returns_empty_on_empty_store(self):
        results = mm.get_relevant_context_compact("anything")
        assert results == []

    def test_respects_top_k(self):
        """Should return at most top_k results."""
        for i in range(5):
            vs.store(f"User: message {i}\nAgent: ok.", {
                "category": "general", "subcategory": "chat",
                "timestamp": time.time() - i,
                "summary": f"Summary {i}",
            })
        results = mm.get_relevant_context_compact("message", top_k=2)
        assert len(results) <= 2

    def test_recency_tiebreaker_ordering(self):
        """When content is identical, recency tiebreaker should order newer first."""
        # Use IDENTICAL text so embedding distances are the same
        vs.store("User: I like blue\nAgent: ok.", {
            "category": "pref", "subcategory": "color",
            "timestamp": time.time() - 365 * 24 * 3600,  # 1 year old
            "summary": "Old: user likes blue.",
        })
        vs.store("User: I like blue\nAgent: ok.", {
            "category": "pref", "subcategory": "color",
            "timestamp": time.time(),  # just now
            "summary": "New: user likes blue.",
        })
        results = mm.get_relevant_context_compact("I like blue")
        assert len(results) == 2
        # With identical distances, the 1-year age penalty (0.876) should
        # push the old one below the new one
        assert "New" in results[0]

    def test_relevance_beats_recency(self):
        """A more relevant but older memory should outrank a less relevant but newer one."""
        vs.store("User: machine learning neural networks deep learning\nAgent: ok.", {
            "category": "knowledge", "subcategory": "ml",
            "timestamp": time.time() - 30 * 24 * 3600,  # 30 days old
            "summary": "Old but relevant: ML discussion.",
        })
        vs.store("User: I had pasta for dinner last night\nAgent: ok.", {
            "category": "pref", "subcategory": "food",
            "timestamp": time.time(),  # just now
            "summary": "New but irrelevant: dinner.",
        })
        results = mm.get_relevant_context_compact("neural networks machine learning")
        assert len(results) == 2
        # The ML memory should rank first despite being 30 days old
        assert "ML" in results[0]


class TestStoreExchangeWithSummary:
    def test_stores_summary_in_metadata(self):
        """store_exchange should include summary in stored metadata."""
        with patch("memory.memory_manager.tag_message", return_value={
            "category": "general", "subcategory": "untagged", "intent": "addition"
        }):
            with patch.object(
                mm, "_generate_memory_summary",
                return_value="User discussed test topics."
            ):
                doc_id = mm.store_exchange(
                    "this is a test message", "sure thing", "t1"
                )

        assert doc_id is not None
        docs = vs.get_all()
        assert len(docs) == 1
        assert docs[0]["metadata"]["summary"] == "User discussed test topics."
