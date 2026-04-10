"""Integration tests -- real Ollama, real agent, real tool calls.

These hit the actual model (e2b for speed) and verify the agent
produces responses, calls tools correctly, and doesn't stall.
Run with: pytest tests/test_integration.py -v

Requires Ollama running with gemma4:e2b loaded.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from agent.agent import create_agent
from agent.config import OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    """Check if Ollama HTTP API responds."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _model_available(model: str) -> bool:
    """Check if a specific model is available in Ollama."""
    import json
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False


_SKIP_REASON = "Ollama not running or gemma4:e2b not available"
_requires_ollama = pytest.mark.skipif(
    not (_ollama_reachable() and _model_available("gemma4:e2b")),
    reason=_SKIP_REASON,
)


@pytest.fixture
def agent_pair(tmp_path):
    """Create a real agent with e2b model and temp SQLite DB."""
    db_path = str(tmp_path / "test_agent.db")
    agent, checkpointer = create_agent(model="gemma4:e2b", db_path=db_path)
    yield agent, checkpointer
    checkpointer.conn.close()


def _run_agent(agent, message: str, thread_id: str = "test-thread") -> dict:
    """Run agent synchronously, collect all output.

    Returns dict with keys: text, tool_calls, tool_results, ai_messages.
    """
    config = {"configurable": {"thread_id": thread_id}}
    messages = [{"role": "user", "content": message}]

    text = ""
    tool_calls = []
    tool_results = []
    ai_messages = 0

    for chunk in agent.stream(
        {"messages": messages}, config=config, stream_mode="updates"
    ):
        for node_name, node_output in chunk.items():
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if msg.type == "ai":
                    ai_messages += 1
                    if msg.content:
                        text += msg.content
                    for tc in (msg.tool_calls or []):
                        tool_calls.append(tc["name"])
                elif msg.type == "tool":
                    tool_results.append(msg.name)

    return {
        "text": text,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "ai_messages": ai_messages,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_requires_ollama
class TestAgentResponds:
    """Agent must produce actual text output. The stall bug = zero text."""

    def test_simple_greeting(self, agent_pair):
        agent, _ = agent_pair
        result = _run_agent(agent, "Say hello in one sentence.")
        assert result["text"].strip(), "Agent produced no text output"
        assert result["ai_messages"] >= 1

    def test_factual_question(self, agent_pair):
        agent, _ = agent_pair
        result = _run_agent(agent, "What is 2 + 2? Just the number.")
        assert result["text"].strip(), "Agent produced no text output"
        assert "4" in result["text"]

    def test_tool_use_get_time(self, agent_pair):
        """Agent should call get_current_time and include the result."""
        agent, _ = agent_pair
        result = _run_agent(agent, "What is the current time? Use your get_current_time tool.")
        assert result["text"].strip(), "Agent produced no text after tool call"
        assert "get_current_time" in result["tool_calls"]


@_requires_ollama
class TestPromptSize:
    """Verify conversational messages actually get smaller prompts."""

    def test_conversational_prompt_has_no_kb_block(self, agent_pair):
        """A simple greeting should NOT trigger KB prompt injection."""
        from unittest.mock import patch
        from agent.agent import _build_prompt

        # Simulate what the agent sees
        from langchain_core.messages import HumanMessage
        state = {"messages": [HumanMessage(content="hello, how are you?")]}

        with patch("agent.agent._search_kb", return_value=[]):
            result = _build_prompt(state)

        system = result[0].content
        assert "KB ACTIVE" not in system
        assert "RULES:" not in system
        assert "Context budget" not in system


@_requires_ollama
class TestMultiTurn:
    """Agent maintains conversation across turns via checkpointer."""

    def test_remembers_prior_turn(self, agent_pair):
        agent, _ = agent_pair
        thread = "multi-turn-test"
        _run_agent(agent, "My favorite color is blue. Remember that.", thread_id=thread)
        result = _run_agent(agent, "What is my favorite color?", thread_id=thread)
        assert "blue" in result["text"].lower(), (
            f"Agent didn't recall 'blue' from prior turn: {result['text'][:200]}"
        )
