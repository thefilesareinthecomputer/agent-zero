"""Tests for memory/entity_registry.py -- SQLite-backed named entity registry.

Each test gets a fresh SQLite DB in a temp directory via the fresh_db fixture.
LLM calls in extract_entities are mocked.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import memory.entity_registry as er


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Give each test a fresh entity registry DB."""
    db_path = str(tmp_path / "test_entities.db")
    monkeypatch.setattr(er, "_DB_PATH", db_path)
    er.init_db(db_path)


# -- Registration --

class TestRegisterEntity:
    def test_creates_new_entity(self):
        entity = er.register_entity("Alice", "person")
        assert entity["name"] == "Alice"
        assert entity["entity_type"] == "person"
        assert entity["mention_count"] == 1
        assert er.entity_count() == 1

    def test_returns_existing_on_duplicate_name(self):
        er.register_entity("Alice", "person")
        entity = er.register_entity("Alice", "person")
        assert entity["mention_count"] == 2
        assert er.entity_count() == 1  # no duplicate

    def test_case_insensitive_dedup(self):
        er.register_entity("Alice", "person")
        entity = er.register_entity("alice", "person")
        assert entity["mention_count"] == 2
        assert er.entity_count() == 1

    def test_dedup_by_alias(self):
        er.register_entity("Alice Smith", "person", aliases=["alice"])
        entity = er.register_entity("alice", "person")
        assert entity["name"] == "Alice Smith"  # canonical name preserved
        assert entity["mention_count"] == 2
        assert er.entity_count() == 1

    def test_merges_new_aliases_on_register(self):
        er.register_entity("Alice", "person", aliases=["al"])
        entity = er.register_entity("Alice", "person", aliases=["ally"])
        assert "al" in entity["aliases"]
        assert "ally" in entity["aliases"]

    def test_invalid_type_defaults_to_thing(self):
        entity = er.register_entity("Widget", "gadget")
        assert entity["entity_type"] == "thing"

    def test_stores_aliases_lowercase(self):
        entity = er.register_entity("Bob", "person", aliases=["Bobby", "ROBERT"])
        assert "bobby" in entity["aliases"]
        assert "robert" in entity["aliases"]

    def test_alias_match_adds_new_name_as_alias(self):
        """When a new name matches an existing entity's alias, the new name
        becomes an alias on the existing entity."""
        er.register_entity("Robert", "person", aliases=["bob"])
        entity = er.register_entity("Bob", "person")
        assert entity["name"] == "Robert"
        assert er.entity_count() == 1


# -- Resolution --

class TestResolveEntity:
    def test_resolve_by_canonical_name(self):
        er.register_entity("Alice", "person")
        entity = er.resolve_entity("Alice")
        assert entity is not None
        assert entity["name"] == "Alice"

    def test_resolve_by_alias(self):
        er.register_entity("Alice Smith", "person", aliases=["alice", "al"])
        entity = er.resolve_entity("al")
        assert entity is not None
        assert entity["name"] == "Alice Smith"

    def test_resolve_case_insensitive(self):
        er.register_entity("Alice", "person")
        assert er.resolve_entity("ALICE") is not None
        assert er.resolve_entity("alice") is not None

    def test_resolve_unknown_returns_none(self):
        assert er.resolve_entity("Nobody") is None


# -- Touch / Update --

class TestTouchEntity:
    def test_increments_mention_count(self):
        entity = er.register_entity("Alice", "person")
        updated = er.touch_entity(entity["id"])
        assert updated["mention_count"] == 2

    def test_updates_last_seen(self):
        entity = er.register_entity("Alice", "person")
        original_last_seen = entity["last_seen"]
        time.sleep(0.01)
        updated = er.touch_entity(entity["id"])
        assert updated["last_seen"] > original_last_seen

    def test_merges_aliases(self):
        entity = er.register_entity("Alice", "person", aliases=["al"])
        updated = er.touch_entity(entity["id"], new_aliases=["ally"])
        assert "al" in updated["aliases"]
        assert "ally" in updated["aliases"]

    def test_does_not_add_canonical_name_as_alias(self):
        entity = er.register_entity("Alice", "person")
        updated = er.touch_entity(entity["id"], new_aliases=["Alice"])
        assert "alice" not in updated["aliases"]

    def test_updates_summary_when_provided(self):
        entity = er.register_entity("Alice", "person")
        updated = er.touch_entity(entity["id"], summary="The user's friend")
        assert updated["summary"] == "The user's friend"

    def test_keeps_old_summary_when_empty(self):
        entity = er.register_entity("Alice", "person", summary="Original")
        # summary="" is treated as a non-update in register_entity path
        # but touch_entity directly: empty string = keep old
        updated = er.touch_entity(entity["id"], summary="")
        assert updated["summary"] == "Original"


# -- Search --

class TestSearchEntities:
    def test_search_by_name(self):
        er.register_entity("Alice Smith", "person")
        er.register_entity("Bob Jones", "person")
        results = er.search_entities("alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice Smith"

    def test_search_by_alias(self):
        er.register_entity("Alice", "person", aliases=["al"])
        results = er.search_entities("al")
        assert len(results) >= 1

    def test_search_by_summary(self):
        er.register_entity("Alice", "person", summary="Data engineer at Acme")
        results = er.search_entities("acme")
        assert len(results) == 1

    def test_search_returns_empty_on_no_match(self):
        er.register_entity("Alice", "person")
        results = er.search_entities("zzzzz")
        assert results == []


# -- List --

class TestListEntities:
    def test_list_all(self):
        er.register_entity("Alice", "person")
        er.register_entity("Acme Corp", "organization")
        results = er.list_entities()
        assert len(results) == 2

    def test_list_by_type(self):
        er.register_entity("Alice", "person")
        er.register_entity("Acme Corp", "organization")
        results = er.list_entities(entity_type="person")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_sorted_by_mention_count(self):
        er.register_entity("Rare", "thing")
        popular = er.register_entity("Popular", "thing")
        er.touch_entity(popular["id"])
        er.touch_entity(popular["id"])
        results = er.list_entities()
        assert results[0]["name"] == "Popular"

    def test_list_empty_registry(self):
        assert er.list_entities() == []


# -- Delete --

class TestDeleteEntity:
    def test_deletes_by_id(self):
        entity = er.register_entity("Alice", "person")
        assert er.delete_entity(entity["id"]) is True
        assert er.entity_count() == 0

    def test_delete_nonexistent_returns_false(self):
        assert er.delete_entity("nonexistent-id") is False


# -- Summary update --

class TestUpdateSummary:
    def test_replaces_summary(self):
        entity = er.register_entity("Alice", "person", summary="Old")
        updated = er.update_summary(entity["id"], "New and improved")
        assert updated["summary"] == "New and improved"


# -- Add alias --

class TestAddAlias:
    def test_adds_single_alias(self):
        entity = er.register_entity("Alice", "person")
        updated = er.add_alias(entity["id"], "Al")
        assert "al" in updated["aliases"]

    def test_alias_dedup(self):
        entity = er.register_entity("Alice", "person", aliases=["al"])
        updated = er.add_alias(entity["id"], "al")
        # Should not duplicate
        assert updated["aliases"].count("al") == 1


# -- Extract entities (mocked LLM) --

class TestExtractEntities:
    def test_extracts_from_text(self):
        mock_response = MagicMock()
        mock_response.message.content = "Alice Smith|person\nAcme Corp|organization"

        with patch("ollama.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_response
            MockClient.return_value = mock_client

            entities = er.extract_entities("I met Alice Smith at Acme Corp today")

        assert len(entities) == 2
        assert entities[0] == {"name": "Alice Smith", "type": "person"}
        assert entities[1] == {"name": "Acme Corp", "type": "organization"}

    def test_returns_empty_on_none_response(self):
        mock_response = MagicMock()
        mock_response.message.content = "NONE"

        with patch("ollama.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_response
            MockClient.return_value = mock_client

            entities = er.extract_entities("just chatting")

        assert entities == []

    def test_returns_empty_on_error(self):
        with patch("ollama.Client", side_effect=Exception("no ollama")):
            entities = er.extract_entities("anything")
        assert entities == []

    def test_filters_invalid_types(self):
        mock_response = MagicMock()
        mock_response.message.content = "Alice|person\nFoo|invalid_type"

        with patch("ollama.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_response
            MockClient.return_value = mock_client

            entities = er.extract_entities("test")

        assert len(entities) == 1
        assert entities[0]["name"] == "Alice"


# -- Process entities (integration) --

class TestProcessEntities:
    def test_registers_extracted_entities(self):
        mock_response = MagicMock()
        mock_response.message.content = "Alice|person\nAgent Zero|project"

        with patch("ollama.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_response
            MockClient.return_value = mock_client

            results = er.process_entities("Working on Agent Zero with Alice")

        assert len(results) == 2
        assert er.entity_count() == 2

    def test_deduplicates_on_reprocess(self):
        mock_response = MagicMock()
        mock_response.message.content = "Alice|person"

        with patch("ollama.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_response
            MockClient.return_value = mock_client

            er.process_entities("Alice said hello")
            er.process_entities("Alice said goodbye")

        assert er.entity_count() == 1
        entity = er.resolve_entity("Alice")
        assert entity["mention_count"] == 2

    def test_empty_extraction_returns_empty(self):
        with patch.object(er, "extract_entities", return_value=[]):
            results = er.process_entities("nothing here")
        assert results == []


# -- Entity count --

class TestEntityCount:
    def test_zero_on_fresh_db(self):
        assert er.entity_count() == 0

    def test_reflects_registrations(self):
        er.register_entity("A", "thing")
        er.register_entity("B", "thing")
        assert er.entity_count() == 2
