"""Tests for memory/memory_manager.py -- pipeline logic, dedup, contradiction,
novelty, pruning, and forget operations.

Strategy:
- Each test gets a fresh ephemeral ChromaDB collection (monkeypatched global state).
- tag_message and check_novelty are mocked to return controlled values.
- vector_store.search is mocked where specific distances are needed to exercise
  the dedup/contradiction/novelty branches.
"""

import time
from unittest.mock import MagicMock, patch

import chromadb
import pytest

import memory.vector_store as vs
import memory.memory_manager as mm


# -- Fixtures --

@pytest.fixture(autouse=True)
def fresh_chroma(tmp_path, monkeypatch):
    """Give each test a fully isolated ChromaDB in its own temp directory.

    EphemeralClient shares in-process storage across instances, so counts
    accumulate across tests. PersistentClient with tmp_path is truly isolated:
    each test gets a fresh directory, fresh database, count starts at 0.
    """
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="conversations",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(vs, "_client", client)
    monkeypatch.setattr(vs, "_collection", collection)


def _make_tags(category: str = "user-preference",
               subcategory: str = "favorite-color",
               intent: str = "addition") -> dict:
    return {"category": category, "subcategory": subcategory, "intent": intent}


# -- Noise filter --

class TestNoiseFilter:
    def test_short_message_not_stored(self):
        with patch("memory.memory_manager.tag_message") as mock_tag:
            result = mm.store_exchange("hi", "Hello!", "t1")
        assert result is None
        mock_tag.assert_not_called()  # noise filter fires before tagging

    def test_two_word_message_not_stored(self):
        with patch("memory.memory_manager.tag_message"):
            result = mm.store_exchange("two words", "OK", "t1")
        assert result is None

    def test_three_word_message_stored(self):
        with patch("memory.memory_manager.tag_message", return_value=_make_tags()):
            with patch("memory.memory_manager.check_novelty", return_value=True):
                result = mm.store_exchange("three word message", "Sure.", "t1")
        assert result is not None


# -- General / untagged bypass --

class TestGeneralBypass:
    def test_general_category_skips_smart_checks(self):
        """general/untagged messages bypass dedup and novelty checks, stored directly."""
        tags = _make_tags(category="general", subcategory="untagged")
        with patch("memory.memory_manager.tag_message", return_value=tags):
            result = mm.store_exchange("just chatting here mate", "Sure.", "t1")
        assert result is not None
        assert vs.count() == 1

    def test_stores_multiple_general_messages(self):
        tags = _make_tags(category="general", subcategory="untagged")
        with patch("memory.memory_manager.tag_message", return_value=tags):
            mm.store_exchange("first casual message", "Response 1.", "t1")
            mm.store_exchange("second casual message", "Response 2.", "t1")
        assert vs.count() == 2


# -- Deduplication --

class TestDedup:
    def test_near_identical_refreshes_timestamp_not_stored(self):
        """Distance < DEDUP_THRESHOLD: update timestamp, return None."""
        tags = _make_tags()
        existing_id = vs.store("User: my favorite color is blue\nAgent: Got it.", {
            "category": "user-preference",
            "subcategory": "favorite-color",
            "intent": "addition",
            "timestamp": time.time() - 3600,
        })

        fake_similar = [{
            "id": existing_id,
            "text": "User: my favorite color is blue\nAgent: Got it.",
            "metadata": {
                "category": "user-preference",
                "subcategory": "favorite-color",
                "intent": "addition",
                "timestamp": time.time() - 3600,
            },
            "distance": 0.05,  # below DEDUP_THRESHOLD (0.15)
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                result = mm.store_exchange(
                    "my favorite color is blue", "Yes, blue.", "t1"
                )

        assert result is None
        assert vs.count() == 1  # no new document added

    def test_dedup_updates_timestamp(self):
        """After dedup, the existing document's timestamp should be refreshed."""
        original_ts = time.time() - 7200
        tags = _make_tags()
        existing_id = vs.store("User: my favorite color is blue\nAgent: Got it.", {
            "category": "user-preference",
            "subcategory": "favorite-color",
            "intent": "addition",
            "timestamp": original_ts,
        })

        fake_similar = [{
            "id": existing_id,
            "text": "User: my favorite color is blue\nAgent: Got it.",
            "metadata": {
                "category": "user-preference",
                "subcategory": "favorite-color",
                "intent": "addition",
                "timestamp": original_ts,
            },
            "distance": 0.05,
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                mm.store_exchange("my favorite color is blue", "Yes, blue.", "t1")

        # Fetch the doc's updated metadata
        docs = vs.get_all()
        assert len(docs) == 1
        updated_ts = docs[0]["metadata"]["timestamp"]
        assert updated_ts > original_ts


# -- Contradiction / update --

class TestContradiction:
    def test_update_intent_replaces_old_memory(self):
        """Update with distance < CONTRADICT_THRESHOLD: old doc deleted, new stored."""
        tags = _make_tags(intent="update")
        old_id = vs.store("User: my favorite color is blue\nAgent: Noted.", {
            "category": "user-preference",
            "subcategory": "favorite-color",
            "intent": "update",
            "timestamp": time.time() - 1000,
        })

        fake_similar = [{
            "id": old_id,
            "text": "User: my favorite color is blue\nAgent: Noted.",
            "metadata": {
                "category": "user-preference",
                "subcategory": "favorite-color",
                "intent": "update",
                "timestamp": time.time() - 1000,
            },
            "distance": 0.30,  # within CONTRADICT_THRESHOLD (0.50)
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                new_id = mm.store_exchange(
                    "my favorite color is now green", "Updated to green.", "t1"
                )

        assert new_id is not None
        all_docs = vs.get_all()
        assert len(all_docs) == 1  # old deleted, new stored
        assert all_docs[0]["id"] == new_id

    def test_update_beyond_threshold_adds_without_replacing(self):
        """Update with distance >= CONTRADICT_THRESHOLD: no replacement, just add."""
        tags = _make_tags(intent="update")
        old_id = vs.store("User: I like cats\nAgent: Cats are great.", {
            "category": "user-preference",
            "subcategory": "pets",
            "intent": "update",
            "timestamp": time.time() - 1000,
        })

        fake_similar = [{
            "id": old_id,
            "text": "User: I like cats\nAgent: Cats are great.",
            "metadata": {
                "category": "user-preference",
                "subcategory": "pets",
                "intent": "update",
                "timestamp": time.time() - 1000,
            },
            "distance": 0.75,  # beyond CONTRADICT_THRESHOLD (0.50)
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                new_id = mm.store_exchange(
                    "I actually prefer reptiles as pets", "Reptiles it is.", "t1"
                )

        assert new_id is not None
        assert vs.count() == 2  # both kept


# -- Novelty (addition) --

class TestNovelty:
    def test_novel_addition_stored(self):
        tags = _make_tags(intent="addition")
        old_id = vs.store("User: I like pizza\nAgent: Noted.", {
            "category": "user-preference",
            "subcategory": "food",
            "intent": "addition",
            "timestamp": time.time() - 100,
        })

        fake_similar = [{
            "id": old_id,
            "text": "User: I like pizza\nAgent: Noted.",
            "metadata": {"category": "user-preference", "subcategory": "food",
                         "intent": "addition", "timestamp": time.time() - 100},
            "distance": 0.30,  # not a dedup (>0.15), not an update
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                with patch("memory.memory_manager.check_novelty", return_value=True):
                    result = mm.store_exchange(
                        "I also love sushi with hot honey", "Great combo.", "t1"
                    )

        assert result is not None
        assert vs.count() == 2

    def test_non_novel_addition_refreshes_timestamp(self):
        tags = _make_tags(intent="addition")
        original_ts = time.time() - 3600
        old_id = vs.store("User: I like pizza\nAgent: Noted.", {
            "category": "user-preference",
            "subcategory": "food",
            "intent": "addition",
            "timestamp": original_ts,
        })

        fake_similar = [{
            "id": old_id,
            "text": "User: I like pizza\nAgent: Noted.",
            "metadata": {"category": "user-preference", "subcategory": "food",
                         "intent": "addition", "timestamp": original_ts},
            "distance": 0.30,
        }]

        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.vector_store.search", return_value=fake_similar):
                with patch("memory.memory_manager.check_novelty", return_value=False):
                    result = mm.store_exchange("I also enjoy pizza", "Yes!", "t1")

        assert result is None
        assert vs.count() == 1  # no new doc
        docs = vs.get_all()
        assert docs[0]["metadata"]["timestamp"] > original_ts

    def test_addition_with_no_existing_memories_stored_directly(self):
        """When the subcategory is empty, store without asking for novelty."""
        tags = _make_tags(intent="addition")
        with patch("memory.memory_manager.tag_message", return_value=tags):
            with patch("memory.memory_manager.check_novelty") as mock_novelty:
                with patch("memory.vector_store.search", return_value=[]):
                    result = mm.store_exchange(
                        "I love spicy ramen noodles a lot", "Noted.", "t1"
                    )
        mock_novelty.assert_not_called()
        assert result is not None


# -- get_relevant_context --

class TestGetRelevantContext:
    def test_returns_empty_on_empty_store(self):
        result = mm.get_relevant_context("anything")
        assert result == []

    def test_returns_text_strings(self):
        vs.store("User: hello world test\nAgent: Hi there.", {
            "category": "general",
            "subcategory": "chat",
            "timestamp": time.time(),
        })
        results = mm.get_relevant_context("hello world")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], str)


# -- forget_last --

class TestForgetLast:
    def test_removes_most_recent(self):
        vs.store("User: older message\nAgent: ok.", {
            "category": "general", "subcategory": "x", "timestamp": time.time() - 100
        })
        vs.store("User: newer message\nAgent: ok.", {
            "category": "general", "subcategory": "x", "timestamp": time.time()
        })
        deleted = mm.forget_last()
        assert deleted is not None
        assert vs.count() == 1

    def test_returns_none_on_empty_store(self):
        assert mm.forget_last() is None


# -- forget_all --

class TestForgetAll:
    def test_deletes_all_memories(self):
        for i in range(5):
            vs.store(f"User: message {i}\nAgent: ok.", {
                "category": "general", "subcategory": "x",
                "timestamp": time.time() - i * 10,
            })
        n = mm.forget_all()
        assert n == 5
        assert vs.count() == 0

    def test_returns_zero_on_empty(self):
        assert mm.forget_all() == 0


# -- prune --

class TestPrune:
    def test_old_memories_never_pruned_by_age(self):
        """Age-based pruning is gone. Old memories persist indefinitely."""
        very_old_ts = time.time() - 365 * 24 * 3600  # 1 year old
        vs.store("User: ground truth\nAgent: ok.", {
            "category": "personal", "subcategory": "identity", "timestamp": very_old_ts
        })
        vs.store("User: recent thing\nAgent: ok.", {
            "category": "general", "subcategory": "x", "timestamp": time.time()
        })
        pruned = mm.prune()
        assert pruned == 0
        assert vs.count() == 2

    def test_keeps_all_under_cap(self):
        for i in range(3):
            vs.store(f"User: recent {i}\nAgent: ok.", {
                "category": "general", "subcategory": "x",
                "timestamp": time.time() - i * 60,
            })
        pruned = mm.prune()
        assert pruned == 0
        assert vs.count() == 3

    def test_prune_empty_store_is_safe(self):
        assert mm.prune() == 0

    def test_enforces_max_memories_cap(self):
        """When count exceeds MAX_MEMORIES, oldest entries are pruned."""
        original_max = mm.MAX_MEMORIES
        mm.MAX_MEMORIES = 3
        try:
            for i in range(5):
                vs.store(f"User: message {i}\nAgent: ok.", {
                    "category": "general", "subcategory": "x",
                    "timestamp": time.time() - (5 - i) * 10,  # newest = highest ts
                })
            pruned = mm.prune()
            assert pruned == 2
            assert vs.count() == 3
        finally:
            mm.MAX_MEMORIES = original_max

    def test_cap_keeps_newest(self):
        """After capacity prune, the newest entries survive."""
        original_max = mm.MAX_MEMORIES
        mm.MAX_MEMORIES = 2
        try:
            vs.store("User: oldest\nAgent: ok.", {
                "category": "general", "subcategory": "x",
                "timestamp": time.time() - 300,
            })
            vs.store("User: middle\nAgent: ok.", {
                "category": "general", "subcategory": "x",
                "timestamp": time.time() - 100,
            })
            vs.store("User: newest\nAgent: ok.", {
                "category": "general", "subcategory": "x",
                "timestamp": time.time(),
            })
            mm.prune()
            docs = vs.get_all()
            texts = [d["text"] for d in docs]
            assert "User: newest\nAgent: ok." in texts
            assert "User: middle\nAgent: ok." in texts
            assert "User: oldest\nAgent: ok." not in texts
        finally:
            mm.MAX_MEMORIES = original_max


# -- memory_count --

class TestMemoryCount:
    def test_count_zero_on_fresh_store(self):
        assert mm.memory_count() == 0

    def test_count_reflects_stored_docs(self):
        vs.store("User: test\nAgent: ok.", {
            "category": "general", "subcategory": "x", "timestamp": time.time()
        })
        assert mm.memory_count() == 1
