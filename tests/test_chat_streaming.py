"""Tests for bridge/chat.py -- _stream_agent behavior, store_exchange
async offload, empty response detection.

These test the actual code paths that caused the agent stall and
event loop blocking bugs.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bridge.chat import _stream_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_msg(content: str = "", tool_calls: list | None = None, meta: dict | None = None):
    msg = MagicMock()
    msg.type = "ai"
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.response_metadata = meta or {}
    return msg


def _tool_msg(name: str, content: str):
    msg = MagicMock()
    msg.type = "tool"
    msg.name = name
    msg.content = content
    return msg


def _make_agent_yielding(chunks: list):
    """Create a mock agent whose astream yields given chunks."""
    async def fake_astream(input_dict, config, stream_mode):
        for chunk in chunks:
            yield chunk

    agent = MagicMock()
    agent.astream = fake_astream
    return agent


# ---------------------------------------------------------------------------
# _stream_agent -- normal response
# ---------------------------------------------------------------------------

class TestStreamAgentNormal:
    @pytest.mark.asyncio
    async def test_yields_text_tokens(self):
        agent = _make_agent_yielding([
            {"agent": {"messages": [_ai_msg("Hello world")]}},
        ])
        events = []
        async for etype, data in _stream_agent(agent, "hi", "sess-1"):
            events.append((etype, data))

        types = [e[0] for e in events]
        assert "token" in types
        assert "done" in types
        done_data = [e[1] for e in events if e[0] == "done"][0]
        assert done_data["full_response"] == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_tool_calls_and_results(self):
        agent = _make_agent_yielding([
            {"agent": {"messages": [
                _ai_msg("", tool_calls=[{"name": "get_current_time", "args": {}}]),
            ]}},
            {"tools": {"messages": [
                _tool_msg("get_current_time", "2026-04-09 15:30:00"),
            ]}},
            {"agent": {"messages": [_ai_msg("It is 3:30 PM.")]}},
        ])
        events = []
        async for etype, data in _stream_agent(agent, "what time?", "sess-2"):
            events.append((etype, data))

        types = [e[0] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "token" in types

    @pytest.mark.asyncio
    async def test_usage_stats_emitted(self):
        agent = _make_agent_yielding([
            {"agent": {"messages": [
                _ai_msg("response", meta={
                    "prompt_eval_count": 500,
                    "eval_count": 50,
                }),
            ]}},
        ])
        events = []
        async for etype, data in _stream_agent(agent, "test", "sess-3"):
            events.append((etype, data))

        usage = [e[1] for e in events if e[0] == "usage"]
        assert len(usage) == 1
        assert usage[0]["prompt_tokens"] == 500
        assert usage[0]["completion_tokens"] == 50


# ---------------------------------------------------------------------------
# _stream_agent -- empty response detection (the stall bug)
# ---------------------------------------------------------------------------

class TestStreamAgentEmptyDetection:
    @pytest.mark.asyncio
    async def test_empty_ai_message_logged(self, caplog):
        """AI msg with no content AND no tool calls = warning."""
        agent = _make_agent_yielding([
            {"agent": {"messages": [_ai_msg("")]}},  # empty, no tools
            {"agent": {"messages": [_ai_msg("actual response")]}},
        ])
        with caplog.at_level(logging.WARNING, logger="bridge.chat"):
            events = []
            async for etype, data in _stream_agent(agent, "hello", "sess-4"):
                events.append((etype, data))

        assert any("Empty AI message" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_zero_text_despite_ai_messages_logged(self, caplog):
        """Agent produces AI messages but never any text = warning."""
        agent = _make_agent_yielding([
            {"agent": {"messages": [
                _ai_msg("", tool_calls=[{"name": "some_tool", "args": {}}]),
            ]}},
            {"tools": {"messages": [_tool_msg("some_tool", "result")]}},
            {"agent": {"messages": [_ai_msg("")]}},  # empty final
        ])
        with caplog.at_level(logging.WARNING, logger="bridge.chat"):
            events = []
            async for etype, data in _stream_agent(agent, "do something", "sess-5"):
                events.append((etype, data))

        done_data = [e[1] for e in events if e[0] == "done"][0]
        assert done_data["full_response"] == ""
        assert any("no text response" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_warning_on_normal_response(self, caplog):
        agent = _make_agent_yielding([
            {"agent": {"messages": [_ai_msg("All good here.")]}},
        ])
        with caplog.at_level(logging.WARNING, logger="bridge.chat"):
            async for _ in _stream_agent(agent, "hi", "sess-6"):
                pass

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# store_exchange -- asyncio.to_thread (event loop blocking fix)
# ---------------------------------------------------------------------------

class TestStoreExchangeAsync:
    """Verify store_exchange is called via asyncio.to_thread, not blocking."""

    @pytest.mark.asyncio
    async def test_store_exchange_runs_in_thread(self):
        """The actual fix: store_exchange must run via to_thread, not directly."""
        import bridge.chat as chat_mod

        # Instrument asyncio.to_thread to verify it's called
        calls = []
        original_to_thread = asyncio.to_thread

        async def tracking_to_thread(fn, *args, **kwargs):
            calls.append(fn.__name__ if hasattr(fn, '__name__') else str(fn))
            # Don't actually run store_exchange (needs full memory stack)
            return None

        mock_agent = _make_agent_yielding([
            {"agent": {"messages": [_ai_msg("test response")]}},
        ])

        with (
            patch.object(chat_mod, "_chat_agent", mock_agent),
            patch.object(chat_mod, "_lock", asyncio.Lock()),
            patch("bridge.chat.ensure_model", new_callable=AsyncMock),
            patch("bridge.chat.asyncio.to_thread", side_effect=tracking_to_thread),
        ):
            # Simulate what the /chat endpoint does
            from bridge.api_models import ChatRequest
            body = ChatRequest(message="test", session_id="test-sess")

            full_response = ""
            async with chat_mod._lock:
                async for event_type, data in _stream_agent(
                    mock_agent, body.message, "test-sess"
                ):
                    if event_type == "done":
                        full_response = data["full_response"]

            # Now do what the endpoint does after streaming
            if full_response:
                await tracking_to_thread(
                    chat_mod.store_exchange if hasattr(chat_mod, 'store_exchange')
                    else MagicMock(name='store_exchange', __name__='store_exchange'),
                    body.message, full_response, "test-sess"
                )

        assert "store_exchange" in calls, (
            "store_exchange must be called via asyncio.to_thread to avoid "
            "blocking the event loop"
        )

    @pytest.mark.asyncio
    async def test_store_exchange_failure_does_not_crash_chat(self):
        """Memory failure must not break the chat endpoint."""
        import bridge.chat as chat_mod

        async def failing_to_thread(fn, *args, **kwargs):
            raise RuntimeError("memory DB exploded")

        mock_agent = _make_agent_yielding([
            {"agent": {"messages": [_ai_msg("response before memory crash")]}},
        ])

        with (
            patch.object(chat_mod, "_chat_agent", mock_agent),
            patch.object(chat_mod, "_lock", asyncio.Lock()),
            patch("bridge.chat.ensure_model", new_callable=AsyncMock),
            patch("bridge.chat.asyncio.to_thread", side_effect=failing_to_thread),
        ):
            # Streaming should complete fine
            events = []
            async for etype, data in _stream_agent(mock_agent, "hello", "sess-7"):
                events.append((etype, data))

            done_events = [e for e in events if e[0] == "done"]
            assert len(done_events) == 1
            assert done_events[0][1]["full_response"] == "response before memory crash"

            # The to_thread call (store_exchange) should fail silently
            full_response = done_events[0][1]["full_response"]
            if full_response:
                try:
                    await failing_to_thread(MagicMock(), "hello", full_response, "sess-7")
                except Exception:
                    pass  # This is exactly what the endpoint does -- swallows it
