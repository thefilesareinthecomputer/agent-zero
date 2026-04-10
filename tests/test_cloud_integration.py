"""Cloud integration tests -- real Ollama cloud API calls.

Skipped unless OLLAMA_CLOUD_API_KEY is set and the cloud endpoint responds.

Run independently: pytest tests/test_cloud_integration.py -v
"""

import socket
import urllib.error
import urllib.request

import pytest

from agent.config import (
    EFFECTIVE_FAST_MODEL,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_URL,
)


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

def _cloud_configured() -> bool:
    return bool(OLLAMA_CLOUD_API_KEY)


def _cloud_reachable() -> bool:
    """Return True if cloud endpoint is reachable (regardless of auth result).

    Only returns False on network-level failures (timeout, DNS, connection
    refused). HTTP errors like 401/403 still mean the server is up.
    """
    if not OLLAMA_CLOUD_API_KEY:
        return False
    import socket
    import urllib.error
    try:
        req = urllib.request.Request(
            f"{OLLAMA_CLOUD_URL}/api/tags",
            headers={"Authorization": f"Bearer {OLLAMA_CLOUD_API_KEY}"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except urllib.error.HTTPError:
        return True  # server responded, even if auth failed
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


_requires_cloud = pytest.mark.skipif(
    not (_cloud_configured() and _cloud_reachable()),
    reason="OLLAMA_CLOUD_API_KEY not set or cloud endpoint not reachable",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_cloud_provider(monkeypatch):
    """All tests in this module run with provider=cloud."""
    import agent.runtime_config as rc
    monkeypatch.setattr(rc, "_provider", "cloud")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_requires_cloud
class TestCloudConnection:
    def test_ollama_client_lists_models(self):
        """make_ollama_client() in cloud mode returns a non-empty model list."""
        from agent.llm import make_ollama_client
        client = make_ollama_client()
        result = client.list()
        assert hasattr(result, "models"), f"Unexpected list() result: {result}"
        assert len(result.models) > 0, "Cloud returned empty model list"

    def test_model_names_are_strings(self):
        """Each model entry in the cloud list has a string name."""
        from agent.llm import make_ollama_client
        client = make_ollama_client()
        for m in client.list().models:
            assert isinstance(m.model, str) and m.model


@_requires_cloud
class TestCloudInference:
    def test_chat_returns_text(self):
        """Cloud model returns a non-empty text response."""
        from agent.llm import make_chat_ollama
        llm = make_chat_ollama(model=EFFECTIVE_FAST_MODEL, num_ctx=2048, num_predict=32)
        response = llm.invoke("Reply with the single word: hello")
        assert response.content.strip(), "Cloud model returned empty response"

    def test_factual_answer_correct(self):
        """Cloud model returns a correct factual answer."""
        from agent.llm import make_chat_ollama
        llm = make_chat_ollama(model=EFFECTIVE_FAST_MODEL, num_ctx=2048, num_predict=16)
        response = llm.invoke("What is 2 + 2? Reply with just the number.")
        assert "4" in response.content, f"Expected '4' in: {response.content!r}"

    def test_auth_header_used(self, monkeypatch):
        """Cloud LLM call fails with invalid key (verifies auth is being sent)."""
        import agent.runtime_config as rc
        monkeypatch.setattr(rc, "OLLAMA_CLOUD_API_KEY", "invalid-key-that-will-be-rejected")
        from agent.llm import make_chat_ollama
        llm = make_chat_ollama(model=EFFECTIVE_FAST_MODEL, num_ctx=2048, num_predict=8)
        with pytest.raises(Exception):
            llm.invoke("hi")
