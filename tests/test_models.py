"""Tests for bridge/models.py -- model lifecycle management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bridge.models as models_mod


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset module state between tests."""
    monkeypatch.setattr(models_mod, "_active_model", None)
    # Replace the module-level lock so tests don't share lock state
    monkeypatch.setattr(models_mod, "_swap_lock", asyncio.Lock())


class TestEnsureModel:
    def test_sets_active_model(self):
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock):
            asyncio.run(models_mod.ensure_model("gemma4:e4b"))
        assert models_mod.get_active_model() == "gemma4:e4b"

    def test_same_model_no_unload(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock) as mock_unload:
            asyncio.run(models_mod.ensure_model("gemma4:e4b"))
            mock_unload.assert_not_called()

    def test_different_model_triggers_unload(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock) as mock_unload:
            asyncio.run(models_mod.ensure_model("gemma4:26b"))
            mock_unload.assert_called_once_with("gemma4:e4b")
        assert models_mod.get_active_model() == "gemma4:26b"


class TestUnloadModel:
    def test_clears_active_if_match(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock):
            asyncio.run(models_mod.unload_model("gemma4:e4b"))
        assert models_mod.get_active_model() is None

    def test_preserves_active_if_different(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock):
            asyncio.run(models_mod.unload_model("gemma4:26b"))
        assert models_mod.get_active_model() == "gemma4:e4b"


class TestSwapForKb:
    def test_swap_sets_refine_model(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock):
            asyncio.run(models_mod.swap_for_kb())
        assert models_mod.get_active_model() == models_mod.KB_REFINE_MODEL

    def test_swap_back_restores_chat_model(self):
        models_mod._active_model = models_mod.KB_REFINE_MODEL
        with patch.object(models_mod, "_async_unload", new_callable=AsyncMock):
            asyncio.run(models_mod.swap_back_from_kb())
        assert models_mod.get_active_model() == models_mod.CHAT_MODEL


class TestSyncUnload:
    def test_calls_generate_with_keep_alive_zero(self):
        mock_client = MagicMock()
        with patch("bridge.models._ollama_client.Client", return_value=mock_client):
            models_mod.sync_unload_model("gemma4:e2b")
        mock_client.generate.assert_called_once_with(
            model="gemma4:e2b", prompt="", keep_alive=0
        )


class TestSyncSwap:
    def test_sync_swap_for_kb(self):
        models_mod._active_model = "gemma4:e4b"
        with patch.object(models_mod, "sync_unload_model"):
            models_mod.sync_swap_for_kb()
        assert models_mod.get_active_model() == models_mod.KB_REFINE_MODEL

    def test_sync_swap_back(self):
        models_mod._active_model = models_mod.KB_REFINE_MODEL
        with patch.object(models_mod, "sync_unload_model"):
            models_mod.sync_swap_back_from_kb()
        assert models_mod.get_active_model() == models_mod.CHAT_MODEL
