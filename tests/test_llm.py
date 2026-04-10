"""Tests for agent/llm.py -- ChatOllama and ollama.Client factories."""

from unittest.mock import MagicMock, call, patch

import pytest

import agent.llm as llm_mod
import agent.runtime_config as rc


@pytest.fixture(autouse=True)
def _reset_provider(monkeypatch):
    """Ensure local provider between tests."""
    monkeypatch.setattr(rc, "_provider", "local")
    monkeypatch.setattr(rc, "_on_change_hooks", [])


class TestMakeChatOllama:
    def test_local_uses_base_url(self):
        from agent.config import OLLAMA_BASE_URL
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:e4b", num_ctx=4096)
            mock_cls.assert_called_once()
            _, kwargs = mock_cls.call_args
            assert kwargs["base_url"] == OLLAMA_BASE_URL

    def test_local_no_client_kwargs(self):
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:e4b")
            _, kwargs = mock_cls.call_args
            # client_kwargs should be empty or absent when no auth
            assert kwargs.get("client_kwargs", {}) == {}

    def test_cloud_uses_cloud_url(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        from agent.config import OLLAMA_CLOUD_URL
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:26b")
            _, kwargs = mock_cls.call_args
            assert kwargs["base_url"] == OLLAMA_CLOUD_URL

    def test_cloud_with_key_injects_auth_header(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        monkeypatch.setattr("agent.runtime_config.OLLAMA_CLOUD_API_KEY", "test-key-abc")
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:26b")
            _, kwargs = mock_cls.call_args
            assert kwargs["client_kwargs"]["headers"]["Authorization"] == "Bearer test-key-abc"

    def test_cloud_empty_key_no_headers(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        monkeypatch.setattr("agent.runtime_config.OLLAMA_CLOUD_API_KEY", "")
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:26b")
            _, kwargs = mock_cls.call_args
            assert kwargs.get("client_kwargs", {}) == {}

    def test_model_and_kwargs_passed_through(self):
        with patch("agent.llm.ChatOllama") as mock_cls:
            llm_mod.make_chat_ollama(model="gemma4:e4b", num_ctx=8192, num_predict=512)
            _, kwargs = mock_cls.call_args
            assert kwargs["model"] == "gemma4:e4b"
            assert kwargs["num_ctx"] == 8192
            assert kwargs["num_predict"] == 512


class TestMakeOllamaClient:
    def test_local_uses_base_url(self):
        from agent.config import OLLAMA_BASE_URL
        with patch("agent.llm.ollama.Client") as mock_cls:
            llm_mod.make_ollama_client()
            mock_cls.assert_called_once_with(host=OLLAMA_BASE_URL)

    def test_local_no_headers(self):
        with patch("agent.llm.ollama.Client") as mock_cls:
            llm_mod.make_ollama_client()
            _, kwargs = mock_cls.call_args
            assert "headers" not in kwargs

    def test_cloud_uses_cloud_url(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        from agent.config import OLLAMA_CLOUD_URL
        with patch("agent.llm.ollama.Client") as mock_cls:
            llm_mod.make_ollama_client()
            mock_cls.assert_called_once()
            _, kwargs = mock_cls.call_args
            assert kwargs["host"] == OLLAMA_CLOUD_URL

    def test_cloud_with_key_passes_auth_header(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        monkeypatch.setattr("agent.runtime_config.OLLAMA_CLOUD_API_KEY", "test-key-xyz")
        with patch("agent.llm.ollama.Client") as mock_cls:
            llm_mod.make_ollama_client()
            _, kwargs = mock_cls.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-key-xyz"

    def test_cloud_empty_key_no_headers(self, monkeypatch):
        monkeypatch.setattr(rc, "_provider", "cloud")
        monkeypatch.setattr("agent.runtime_config.OLLAMA_CLOUD_API_KEY", "")
        with patch("agent.llm.ollama.Client") as mock_cls:
            llm_mod.make_ollama_client()
            _, kwargs = mock_cls.call_args
            assert "headers" not in kwargs
