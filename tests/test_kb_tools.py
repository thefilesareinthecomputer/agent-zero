"""Tests for agent/tools.py -- KB tool functions that the agent actually calls.

Covers: read_knowledge_section (pass-1 H1, pass-2 H2-H5, refusal logic),
read_knowledge (heading tree output), search_knowledge, save_knowledge,
list_knowledge, entity tool wrappers, and retrieval budget enforcement.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.tools as tools_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MULTI_H1_FILE = textwrap.dedent("""\
    ---
    date-created: 2026-04-08
    last-modified: 2026-04-08
    tags:
      - test
    ---
    # Concept Alpha
    Alpha overview text.

    ## Alpha Details
    Detailed content about alpha.

    ### Alpha Sub-Detail
    Deeply nested content under alpha.

    # Concept Beta
    Beta overview text.

    ## Beta Details
    Detailed content about beta.
""")

_SINGLE_H1_FILE = textwrap.dedent("""\
    ---
    date-created: 2026-04-08
    last-modified: 2026-04-08
    tags:
      - test
    ---
    # Main Title
    Some intro text.

    ## Section One
    Content of section one.

    ### Subsection 1A
    Deep content under section one.

    ## Section Two
    Content of section two.

    ## Section Three
    Content of section three.
""")


@pytest.fixture
def kb_dir(tmp_path):
    """Set up a temp knowledge dir with test files and patch config paths."""
    kb = tmp_path / "knowledge"
    kb.mkdir()
    canon = tmp_path / "knowledge_canon"
    canon.mkdir()

    (kb / "multi-h1.md").write_text(_MULTI_H1_FILE, encoding="utf-8")
    (kb / "single-h1.md").write_text(_SINGLE_H1_FILE, encoding="utf-8")

    # Patch all knowledge_store imports so they resolve files from tmp dirs.
    original_kb_read = tools_mod._kb_read
    original_kb_list = tools_mod._kb_list
    original_kb_save = tools_mod._kb_save

    def _patched_kb_read(filename, *, base_dir=None):
        return original_kb_read(filename, base_dir=base_dir or kb)

    def _patched_kb_list(*, base_dir=None, **kwargs):
        return original_kb_list(base_dir=base_dir or kb, **kwargs)

    def _patched_kb_save(filename, content, tags, **kwargs):
        return original_kb_save(filename, content, tags, base_dir=kb, **kwargs)

    with (
        patch.object(tools_mod, "KNOWLEDGE_PATH", str(kb)),
        patch("agent.tools._CANON_DIR", canon),
        patch("agent.tools._kb_read", _patched_kb_read),
        patch("agent.tools._kb_list", _patched_kb_list),
        patch("agent.tools._kb_save", _patched_kb_save),
    ):
        yield kb


@pytest.fixture(autouse=True)
def reset_budget():
    """Reset retrieval budget state before each test."""
    tools_mod._current_kb_loads = 0
    tools_mod._current_available_tokens = 999_999
    yield
    tools_mod._current_kb_loads = 0
    tools_mod._current_available_tokens = 999_999


# ---------------------------------------------------------------------------
# read_knowledge_section -- Pass 1: H1 concept match
# ---------------------------------------------------------------------------

class TestReadKnowledgeSectionPass1:
    def test_loads_h1_by_exact_name(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "Alpha overview text" in result
        assert "Alpha Details" in result
        assert "[Loaded ~" in result

    def test_h1_match_is_case_insensitive(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "concept beta"}
        )
        assert "Beta overview text" in result

    def test_h1_match_is_substring(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Alpha"}
        )
        assert "Alpha overview text" in result

    def test_h1_loads_all_children(self, kb_dir):
        """Loading an H1 should include its H2 and H3 children."""
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "Alpha Details" in result
        assert "Alpha Sub-Detail" in result


# ---------------------------------------------------------------------------
# read_knowledge_section -- Pass 2: H2-H5 subsection extraction
# ---------------------------------------------------------------------------

class TestReadKnowledgeSectionPass2:
    def test_extracts_h2_from_single_h1_file(self, kb_dir):
        """The failure mode that caused the original agent stall."""
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "single-h1.md", "section": "Section One"}
        )
        assert "Content of section one" in result
        assert "Subsection 1A" in result
        # Should NOT include content from Section Two
        assert "Content of section two" not in result

    def test_extracts_h3_from_within_h1(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Alpha Sub-Detail"}
        )
        assert "Deeply nested content under alpha" in result

    def test_extracts_h2_from_multi_h1_file(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Beta Details"}
        )
        assert "Detailed content about beta" in result
        # Should not include content from Alpha
        assert "Alpha overview" not in result

    def test_h2_extraction_is_case_insensitive(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "single-h1.md", "section": "section two"}
        )
        assert "Content of section two" in result

    def test_h2_extraction_stops_at_next_same_level(self, kb_dir):
        """Section One extraction should stop before Section Two."""
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "single-h1.md", "section": "Section One"}
        )
        assert "Content of section one" in result
        assert "Content of section two" not in result

    def test_last_h2_captures_to_end(self, kb_dir):
        """Last section in file should capture everything to EOF."""
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "single-h1.md", "section": "Section Three"}
        )
        assert "Content of section three" in result


# ---------------------------------------------------------------------------
# read_knowledge_section -- error handling
# ---------------------------------------------------------------------------

class TestReadKnowledgeSectionErrors:
    def test_section_not_found_lists_available(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Nonexistent Section"}
        )
        assert "not found" in result.lower()
        assert "Concept Alpha" in result
        assert "Concept Beta" in result

    def test_file_not_found(self, kb_dir):
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "no-such-file.md", "section": "anything"}
        )
        assert "not found" in result.lower()

    def test_cost_receipt_included(self, kb_dir):
        """Every successful load must include a token cost receipt."""
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "[Loaded ~" in result
        assert "tokens]" in result


# ---------------------------------------------------------------------------
# read_knowledge_section -- retrieval refusal
# ---------------------------------------------------------------------------

class TestReadKnowledgeSectionRefusal:
    def test_refuses_at_load_limit(self, kb_dir):
        tools_mod._current_kb_loads = 3
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "REFUSED" in result
        assert "limit" in result.lower()

    def test_refuses_when_context_too_low(self, kb_dir):
        tools_mod._current_available_tokens = 1000
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "REFUSED" in result
        assert "budget" in result.lower() or "context" in result.lower()

    def test_allows_load_under_limit(self, kb_dir):
        tools_mod._current_kb_loads = 2
        tools_mod._current_available_tokens = 50_000
        result = tools_mod.read_knowledge_section.invoke(
            {"filename": "multi-h1.md", "section": "Concept Alpha"}
        )
        assert "REFUSED" not in result
        assert "Alpha overview text" in result


# ---------------------------------------------------------------------------
# read_knowledge -- heading tree output
# ---------------------------------------------------------------------------

class TestReadKnowledge:
    def test_returns_heading_tree(self, kb_dir):
        result = tools_mod.read_knowledge.invoke({"filename": "multi-h1.md"})
        assert "Concept Alpha" in result
        assert "Concept Beta" in result
        assert "tokens total" in result.lower() or "tokens" in result.lower()

    def test_includes_tree_cost_warning(self, kb_dir):
        result = tools_mod.read_knowledge.invoke({"filename": "multi-h1.md"})
        assert "Tree cost" in result or "cost" in result.lower()

    def test_never_loads_full_content(self, kb_dir):
        """read_knowledge should show structure, not dump the whole file."""
        result = tools_mod.read_knowledge.invoke({"filename": "single-h1.md"})
        # Should NOT contain the actual body text of sections
        assert "Deep content under section one" not in result

    def test_file_not_found(self, kb_dir):
        result = tools_mod.read_knowledge.invoke({"filename": "nope.md"})
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# search_knowledge
# ---------------------------------------------------------------------------

class TestSearchKnowledge:
    @patch("agent.tools._search_kb", return_value=[
        {"filename": "ai.md", "source": "knowledge", "file_tokens": 800,
         "file_outline": "Overview, Models",
         "hits": [{"heading": "Overview", "summary": "AI overview notes"}]},
    ])
    def test_returns_file_matches(self, _mock_search, kb_dir):
        result = tools_mod.search_knowledge.invoke({"query": "artificial intelligence"})
        assert "ai.md" in result
        assert "800" in result

    @patch("agent.tools._search_kb", return_value=[])
    @patch("agent.tools._kb_search", return_value=[])
    def test_no_matches_message(self, _fallback, _search, kb_dir):
        result = tools_mod.search_knowledge.invoke({"query": "zzz_nonexistent_zzz"})
        assert "no match" in result.lower()


# ---------------------------------------------------------------------------
# save_knowledge -- real filesystem writes
# ---------------------------------------------------------------------------

class TestSaveKnowledge:
    def test_creates_new_file(self, kb_dir):
        with patch("agent.tools._index_file"):  # skip vector indexing
            result = tools_mod.save_knowledge.invoke({
                "filename": "new-note.md",
                "content": "## Overview\nSome content.",
                "tags": "test, ai",
            })
        assert "Saved" in result
        saved = (kb_dir / "new-note.md").read_text(encoding="utf-8")
        assert "Some content" in saved

    def test_refuses_canon_overwrite(self, kb_dir):
        # Create a canon file
        canon_dir = kb_dir.parent / "knowledge_canon"
        (canon_dir / "protected.md").write_text("# Protected\nDo not edit.")
        with patch("agent.tools._is_canon_file", return_value=True):
            result = tools_mod.save_knowledge.invoke({
                "filename": "protected.md",
                "content": "## Hacked",
                "tags": "evil",
            })
        assert "canon" in result.lower() or "cannot" in result.lower()

    def test_tags_persisted_in_frontmatter(self, kb_dir):
        with patch("agent.tools._index_file"):
            tools_mod.save_knowledge.invoke({
                "filename": "tagged.md",
                "content": "## Data\nStuff.",
                "tags": "python, data",
            })
        content = (kb_dir / "tagged.md").read_text(encoding="utf-8")
        assert "python" in content
        assert "data" in content

    def test_project_field_saved(self, kb_dir):
        with patch("agent.tools._index_file"):
            tools_mod.save_knowledge.invoke({
                "filename": "proj-note.md",
                "content": "## Notes\nProject specific.",
                "tags": "notes",
                "project": "agent-zero",
            })
        content = (kb_dir / "proj-note.md").read_text(encoding="utf-8")
        assert "agent-zero" in content


# ---------------------------------------------------------------------------
# list_knowledge
# ---------------------------------------------------------------------------

class TestListKnowledge:
    def test_lists_existing_files(self, kb_dir):
        result = tools_mod.list_knowledge.invoke({})
        assert "multi-h1.md" in result
        assert "single-h1.md" in result

    def test_empty_kb_message(self, kb_dir):
        # Remove test files
        for f in kb_dir.iterdir():
            f.unlink()
        result = tools_mod.list_knowledge.invoke({})
        assert "empty" in result.lower()


# ---------------------------------------------------------------------------
# Entity tool wrappers -- these call real SQLite via entity_registry
# ---------------------------------------------------------------------------

class TestEntityToolWrappers:
    @pytest.fixture(autouse=True)
    def fresh_entity_db(self, tmp_path):
        """Point entity registry at a temp DB for isolation."""
        import memory.entity_registry as er
        db_path = str(tmp_path / "entities.db")
        er.init_db(db_path)
        yield
        # Reset so other tests aren't affected
        er._DB_PATH = None

    def test_register_and_lookup(self):
        # Register via tool
        result = tools_mod.manage_entity.invoke({
            "name": "Alice",
            "action": "register",
            "value": "person",
        })
        assert "Registered" in result
        assert "Alice" in result

        # Lookup via tool
        result = tools_mod.lookup_entity.invoke({"name": "Alice"})
        assert "Alice" in result
        assert "person" in result

    def test_add_alias_and_resolve(self):
        tools_mod.manage_entity.invoke({
            "name": "Bob Smith",
            "action": "register",
            "value": "person",
        })
        tools_mod.manage_entity.invoke({
            "name": "Bob Smith",
            "action": "add_alias",
            "value": "Bobby",
        })
        result = tools_mod.lookup_entity.invoke({"name": "Bobby"})
        assert "Bob Smith" in result

    def test_update_summary(self):
        tools_mod.manage_entity.invoke({
            "name": "Project X",
            "action": "register",
            "value": "project",
        })
        tools_mod.manage_entity.invoke({
            "name": "Project X",
            "action": "update_summary",
            "value": "Secret internal project for ML inference",
        })
        result = tools_mod.lookup_entity.invoke({"name": "Project X"})
        assert "ML inference" in result

    def test_list_entities(self):
        tools_mod.manage_entity.invoke({
            "name": "Entity A", "action": "register", "value": "concept",
        })
        tools_mod.manage_entity.invoke({
            "name": "Entity B", "action": "register", "value": "person",
        })
        result = tools_mod.manage_entity.invoke({
            "name": "", "action": "list", "value": "",
        })
        assert "Entity A" in result
        assert "Entity B" in result

    def test_list_filter_by_type(self):
        tools_mod.manage_entity.invoke({
            "name": "City X", "action": "register", "value": "place",
        })
        tools_mod.manage_entity.invoke({
            "name": "Person Y", "action": "register", "value": "person",
        })
        result = tools_mod.manage_entity.invoke({
            "name": "", "action": "list", "value": "place",
        })
        assert "City X" in result
        assert "Person Y" not in result

    def test_lookup_nonexistent(self):
        result = tools_mod.lookup_entity.invoke({"name": "Nobody"})
        assert "not found" in result.lower() or "no entity" in result.lower()

    def test_manage_nonexistent_alias(self):
        result = tools_mod.manage_entity.invoke({
            "name": "Ghost",
            "action": "add_alias",
            "value": "phantom",
        })
        assert "not found" in result.lower()

    def test_unknown_action(self):
        # Register first so it gets past the "not found" check
        tools_mod.manage_entity.invoke({
            "name": "TestEntity", "action": "register", "value": "thing",
        })
        result = tools_mod.manage_entity.invoke({
            "name": "TestEntity", "action": "delete", "value": "",
        })
        assert "unknown" in result.lower()


# ---------------------------------------------------------------------------
# snapshot_to_knowledge -- the fix for "save that" stalling
# ---------------------------------------------------------------------------

class TestSnapshotToKnowledge:
    """The actual failure: user says 'save that', e4b stalls because
    save_knowledge requires reproducing the full response as a tool arg.
    snapshot_to_knowledge captures content automatically."""

    def test_saves_with_topic(self, kb_dir):
        tools_mod._last_agent_response = (
            "## Dimensional Modeling Report\n"
            "Fact tables record events. Dimension tables provide context.\n"
            "Use surrogate keys. SCDs manage change.\n"
        )
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke(
                {"topic": "dim modeling"}
            )
        assert "Saved" in result
        saved = (kb_dir / "dim-modeling.md").read_text(encoding="utf-8")
        assert "Fact tables" in saved
        assert "surrogate keys" in saved.lower() or "Surrogate" in saved

    def test_saves_without_topic_uses_heading(self, kb_dir):
        tools_mod._last_agent_response = (
            "# API Design Patterns\n"
            "REST vs GraphQL comparison.\n"
        )
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke({})
        assert "Saved" in result
        assert "api-design-patterns" in result.lower() or "api" in result.lower()

    def test_saves_without_topic_or_heading_uses_timestamp(self, kb_dir):
        tools_mod._last_agent_response = "Just some plain text without headings."
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke({})
        assert "Saved" in result
        assert "snapshot-" in result

    def test_refuses_when_no_prior_response(self, kb_dir):
        tools_mod._last_agent_response = ""
        result = tools_mod.snapshot_to_knowledge.invoke({})
        assert "nothing" in result.lower() or "no prior" in result.lower()

    def test_refuses_canon_overwrite(self, kb_dir):
        tools_mod._last_agent_response = "# Protected\nContent."
        canon_dir = kb_dir.parent / "knowledge_canon"
        (canon_dir / "protected.md").write_text("# Canon\nDo not edit.")
        with patch("agent.tools._is_canon_file", return_value=True):
            result = tools_mod.snapshot_to_knowledge.invoke(
                {"topic": "protected"}
            )
        assert "canon" in result.lower()

    def test_generates_tags_from_topic(self, kb_dir):
        tools_mod._last_agent_response = "## Report\nContent."
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke(
                {"topic": "machine learning pipelines"}
            )
        assert "Saved" in result
        assert "machine" in result or "learning" in result or "pipelines" in result

    def test_generates_tags_from_headings(self, kb_dir):
        tools_mod._last_agent_response = (
            "## Database Performance Tuning\n"
            "Index everything.\n"
            "## Query Optimization\n"
            "Use EXPLAIN.\n"
        )
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke({})
        assert "Saved" in result
        # Should have extracted tags from headings
        assert "tags:" in result

    def test_includes_token_count(self, kb_dir):
        tools_mod._last_agent_response = "## Report\n" + ("word " * 500)
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke(
                {"topic": "big report"}
            )
        assert "tokens" in result

    def test_content_persisted_correctly(self, kb_dir):
        """The whole point: content is captured from _last_agent_response,
        not from the model reproducing it as a tool argument."""
        original = (
            "## Full Analysis\n"
            "This is a detailed analysis that would be impossible for e4b\n"
            "to reproduce as a save_knowledge content argument without\n"
            "stalling or truncating. It includes specific technical details,\n"
            "code examples, comparisons, and structured markdown that the\n"
            "model generated in its response but cannot reliably copy into\n"
            "a tool call parameter.\n\n"
            "### Key Findings\n"
            "1. Connection pooling reduces latency by 40%\n"
            "2. Index-only scans avoid heap fetches\n"
            "3. Batch inserts outperform row-by-row by 10x\n"
        )
        tools_mod._last_agent_response = original
        with patch("agent.tools._index_file"):
            result = tools_mod.snapshot_to_knowledge.invoke(
                {"topic": "performance analysis"}
            )
        assert "Saved" in result
        saved = (kb_dir / "performance-analysis.md").read_text(encoding="utf-8")
        assert "Connection pooling reduces latency by 40%" in saved
        assert "Batch inserts" in saved
