"""Tests for agent/agent.py -- prompt assembly, KB relevance detection,
retrieval accounting, budget enforcement, and escalating directives.

These test the code paths that actually run on every model invocation.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.agent import (
    _CORE_PROMPT,
    _KB_MAX_LOADS_PER_RESPONSE,
    _KB_PROMPT,
    _build_prompt,
    _count_kb_loads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human(text: str) -> MagicMock:
    msg = MagicMock()
    msg.type = "human"
    msg.content = text
    return msg


def _ai(text: str, tool_calls: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.type = "ai"
    msg.content = text
    msg.tool_calls = tool_calls or []
    return msg


def _tool(name: str, content: str) -> MagicMock:
    msg = MagicMock()
    msg.type = "tool"
    msg.name = name
    msg.content = content
    return msg


# ---------------------------------------------------------------------------
# _count_kb_loads
# ---------------------------------------------------------------------------

class TestCountKbLoads:
    def test_empty_messages(self):
        loads, tokens = _count_kb_loads([])
        assert loads == 0
        assert tokens == 0

    def test_no_kb_tools(self):
        msgs = [_human("hello"), _ai("hi there")]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 0
        assert tokens == 0

    def test_counts_read_knowledge_section_as_load(self):
        msgs = [
            _human("tell me about X"),
            _ai("", tool_calls=[{"name": "read_knowledge_section"}]),
            _tool("read_knowledge_section", "## X\nContent here.\n\n[Loaded ~50 tokens]"),
        ]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 1
        assert tokens > 0

    def test_read_knowledge_not_counted_as_load(self):
        """read_knowledge (tree view) should add tokens but not count as a section load."""
        msgs = [
            _human("what do you know about Y"),
            _ai(""),
            _tool("read_knowledge", "[500 tokens total]\n\n# Y (500 tokens)\n  ## Sub (200 tokens)"),
        ]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 0
        assert tokens > 0  # tokens are counted for budget

    def test_multiple_loads_counted(self):
        msgs = [
            _human("compare A, B, and C"),
            _ai(""),
            _tool("read_knowledge_section", "Section A content"),
            _ai(""),
            _tool("read_knowledge_section", "Section B content"),
            _ai(""),
            _tool("read_knowledge_section", "Section C content"),
        ]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 3

    def test_only_counts_since_last_human_message(self):
        """Loads from a prior user turn should not count against the current turn."""
        msgs = [
            _human("first question"),
            _ai(""),
            _tool("read_knowledge_section", "Old section content from prior turn"),
            _ai("Here is my answer."),
            _human("second question"),
            _ai(""),
            _tool("read_knowledge_section", "New section content"),
        ]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 1  # only the one after "second question"

    def test_search_knowledge_adds_tokens_not_loads(self):
        msgs = [
            _human("search for topic Z"),
            _ai(""),
            _tool("search_knowledge", "-- file1.md (500 tokens)\n-- file2.md (300 tokens)"),
        ]
        loads, tokens = _count_kb_loads(msgs)
        assert loads == 0
        assert tokens > 0


# ---------------------------------------------------------------------------
# _build_prompt -- conditional assembly
# ---------------------------------------------------------------------------

class TestBuildPromptConditional:
    """Core prompt is always present. KB prompt is conditional on relevance."""

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_conversational_message_gets_core_only(self, _mc, _sk):
        state = {"messages": [_human("what time is it?")]}
        result = _build_prompt(state)
        system = result[0].content
        assert _CORE_PROMPT in system
        assert "KB ACTIVE" not in system
        assert "RULES:" not in system
        assert "Context budget" not in system

    @patch("agent.agent._search_kb", return_value=[
        {"filename": "test.md", "source": "knowledge", "file_tokens": 500,
         "file_outline": "Overview", "hits": [{"heading": "Overview", "summary": "test"}]}
    ])
    @patch("agent.agent.memory_count", return_value=0)
    def test_kb_relevant_query_gets_full_prompt(self, _mc, _sk):
        state = {"messages": [_human("what do you know about machine learning?")]}
        result = _build_prompt(state)
        system = result[0].content
        assert _CORE_PROMPT in system
        assert "KB ACTIVE" in system
        assert "RULES:" in system
        assert "Context budget" in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_prior_kb_loads_trigger_kb_prompt(self, _mc, _sk):
        """If the model already loaded KB content this turn, keep showing KB prompt."""
        state = {"messages": [
            _human("tell me about X"),
            _ai(""),
            _tool("read_knowledge_section", "## X\nSome content."),
        ]}
        result = _build_prompt(state)
        system = result[0].content
        assert "KB ACTIVE" in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_core_only_is_smaller_than_full(self, _mc, _sk):
        """The whole point -- conversational prompts should be significantly smaller."""
        state_conv = {"messages": [_human("hey, how are you?")]}
        result_conv = _build_prompt(state_conv)
        conv_len = len(result_conv[0].content)

        # Force KB relevance via prior loads
        state_kb = {"messages": [
            _human("tell me about X"),
            _ai(""),
            _tool("read_knowledge_section", "content"),
        ]}
        result_kb = _build_prompt(state_kb)
        kb_len = len(result_kb[0].content)

        # KB prompt should be at least 40% larger than core-only
        assert kb_len > conv_len * 1.4, (
            f"KB prompt ({kb_len} chars) should be significantly larger "
            f"than core-only ({conv_len} chars)"
        )


# ---------------------------------------------------------------------------
# _build_prompt -- memory injection
# ---------------------------------------------------------------------------

class TestBuildPromptMemory:
    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.get_relevant_context_compact", return_value=["User likes Python"])
    @patch("agent.agent.memory_count", return_value=5)
    def test_memories_injected_when_available(self, _mc, _grc, _sk):
        state = {"messages": [_human("what programming language should I use?")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "[Memory]" in system
        assert "User likes Python" in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_no_memory_block_when_empty(self, _mc, _sk):
        state = {"messages": [_human("hello")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "[Memory]" not in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.get_relevant_context_compact", return_value=["remembered fact"])
    @patch("agent.agent.memory_count", return_value=1)
    def test_memories_present_even_without_kb(self, _mc, _grc, _sk):
        """Memory injection is independent of KB relevance."""
        state = {"messages": [_human("what's my favorite color?")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "remembered fact" in system
        assert "KB ACTIVE" not in system


# ---------------------------------------------------------------------------
# _build_prompt -- budget and directives
# ---------------------------------------------------------------------------

class TestBuildPromptBudget:
    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_no_budget_line_for_conversational(self, _mc, _sk):
        state = {"messages": [_human("tell me a joke")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "Context budget" not in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_budget_line_present_for_kb_query(self, _mc, _sk):
        state = {"messages": [
            _human("explain X"),
            _ai(""),
            _tool("read_knowledge_section", "X content"),
        ]}
        result = _build_prompt(state)
        system = result[0].content
        assert "Context budget" in system
        assert "KB: 1/3 sections loaded" in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_retrieval_limit_directive_at_max_loads(self, _mc, _sk):
        """After 3 section loads, the RETRIEVAL LIMIT REACHED directive must appear."""
        msgs = [_human("compare everything")]
        for _ in range(_KB_MAX_LOADS_PER_RESPONSE):
            msgs.append(_ai(""))
            msgs.append(_tool("read_knowledge_section", "section content " * 20))
        state = {"messages": msgs}
        result = _build_prompt(state)
        system = result[0].content
        assert "RETRIEVAL LIMIT REACHED" in system

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_no_limit_directive_under_max(self, _mc, _sk):
        msgs = [
            _human("explain X"),
            _ai(""),
            _tool("read_knowledge_section", "short"),
        ]
        state = {"messages": msgs}
        result = _build_prompt(state)
        system = result[0].content
        assert "RETRIEVAL LIMIT REACHED" not in system


# ---------------------------------------------------------------------------
# _build_prompt -- KB hit injection
# ---------------------------------------------------------------------------

class TestBuildPromptKbHits:
    @patch("agent.agent._search_kb", return_value=[
        {"filename": "ai-notes.md", "source": "knowledge", "file_tokens": 1200,
         "file_outline": "Overview, Architecture",
         "hits": [
             {"heading": "Overview", "summary": "High-level AI notes"},
             {"heading": "Architecture", "summary": "System design patterns"},
         ]},
    ])
    @patch("agent.agent.memory_count", return_value=0)
    def test_kb_hits_shown_in_prompt(self, _mc, _sk):
        state = {"messages": [_human("what are my AI notes?")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "ai-notes.md" in system
        assert "1,200 tokens" in system
        assert "High-level AI notes" in system

    @patch("agent.agent._search_kb", return_value=[
        {"filename": "canon-ref.md", "source": "canon", "file_tokens": 300,
         "file_outline": "Reference",
         "hits": [{"heading": "Reference", "summary": "Canonical ref"}]},
    ])
    @patch("agent.agent.memory_count", return_value=0)
    def test_canon_label_shown(self, _mc, _sk):
        state = {"messages": [_human("check canon reference")]}
        result = _build_prompt(state)
        system = result[0].content
        assert "[canon]" in system


# ---------------------------------------------------------------------------
# _build_prompt -- tools module state sync
# ---------------------------------------------------------------------------

class TestBuildPromptToolsSync:
    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_tools_module_reset_when_no_kb(self, _mc, _sk):
        """When KB is not relevant, tools module should have safe defaults."""
        import agent.tools as tools_mod
        # Force dirty state
        tools_mod._current_available_tokens = 100
        tools_mod._current_kb_loads = 5

        state = {"messages": [_human("hello")]}
        _build_prompt(state)

        assert tools_mod._current_kb_loads == 0
        assert tools_mod._current_available_tokens == 999_999

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_tools_module_updated_when_kb_active(self, _mc, _sk):
        """When KB is active, tools module should reflect actual load count."""
        import agent.tools as tools_mod

        state = {"messages": [
            _human("explain X"),
            _ai(""),
            _tool("read_knowledge_section", "content"),
        ]}
        _build_prompt(state)

        assert tools_mod._current_kb_loads == 1
        assert tools_mod._current_available_tokens < 999_999


# ---------------------------------------------------------------------------
# _build_prompt -- snapshot state capture
# ---------------------------------------------------------------------------

class TestBuildPromptSnapshot:
    """_build_prompt must capture the last AI response so snapshot_to_knowledge
    can save it without the model reproducing the content."""

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_captures_last_ai_response(self, _mc, _sk):
        import agent.tools as tools_mod
        state = {"messages": [
            _human("tell me about X"),
            _ai("Here is a detailed report about X with findings."),
            _human("save that"),
        ]}
        _build_prompt(state)
        assert tools_mod._last_agent_response == "Here is a detailed report about X with findings."

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_skips_empty_ai_messages(self, _mc, _sk):
        """AI messages that are just tool calls (empty content) should be skipped."""
        import agent.tools as tools_mod
        state = {"messages": [
            _human("research X"),
            _ai("The real answer after tools."),
            _human("snapshot"),
        ]}
        _build_prompt(state)
        assert tools_mod._last_agent_response == "The real answer after tools."

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_empty_when_no_prior_ai(self, _mc, _sk):
        import agent.tools as tools_mod
        state = {"messages": [_human("snapshot")]}
        _build_prompt(state)
        assert tools_mod._last_agent_response == ""

    @patch("agent.agent._search_kb", return_value=[])
    @patch("agent.agent.memory_count", return_value=0)
    def test_grabs_most_recent_ai_not_older(self, _mc, _sk):
        """Should capture the AI response from the turn immediately before
        the current user message, not from older turns."""
        import agent.tools as tools_mod
        state = {"messages": [
            _human("first question"),
            _ai("Old answer from first turn."),
            _human("second question"),
            _ai("New answer from second turn."),
            _human("save that"),
        ]}
        _build_prompt(state)
        assert tools_mod._last_agent_response == "New answer from second turn."
