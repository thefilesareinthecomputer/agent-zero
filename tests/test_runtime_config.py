"""Tests for agent/runtime_config.py -- mutable provider state."""

import pytest

import agent.runtime_config as rc


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset runtime config state between tests."""
    from agent.config import IS_CLOUD
    monkeypatch.setattr(rc, "_provider", "cloud" if IS_CLOUD else "local")
    monkeypatch.setattr(rc, "_on_change_hooks", [])


class TestGetProvider:
    def test_default_matches_env(self):
        from agent.config import IS_CLOUD
        expected = "cloud" if IS_CLOUD else "local"
        assert rc.get_provider() == expected

    def test_is_cloud_matches_provider(self):
        rc._provider = "cloud"
        assert rc.is_cloud() is True
        rc._provider = "local"
        assert rc.is_cloud() is False


class TestSetProvider:
    def test_set_to_cloud(self):
        rc.set_provider("cloud")
        assert rc.get_provider() == "cloud"
        assert rc.is_cloud() is True

    def test_set_to_local(self):
        rc._provider = "cloud"
        rc.set_provider("local")
        assert rc.get_provider() == "local"
        assert rc.is_cloud() is False

    def test_toggle_roundtrip(self):
        rc.set_provider("cloud")
        rc.set_provider("local")
        assert rc.get_provider() == "local"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="must be 'local' or 'cloud'"):
            rc.set_provider("unknown")

    def test_invalid_empty_string_raises(self):
        with pytest.raises(ValueError):
            rc.set_provider("")


class TestOnChangeHook:
    def test_hook_fired_on_set_provider(self):
        calls = []
        rc.register_on_change(lambda: calls.append(1))
        rc.set_provider("cloud")
        assert calls == [1]

    def test_hook_fired_multiple_times(self):
        calls = []
        rc.register_on_change(lambda: calls.append(1))
        rc.set_provider("cloud")
        rc.set_provider("local")
        assert len(calls) == 2

    def test_multiple_hooks_all_called(self):
        results = []
        rc.register_on_change(lambda: results.append("a"))
        rc.register_on_change(lambda: results.append("b"))
        rc.set_provider("cloud")
        assert sorted(results) == ["a", "b"]

    def test_no_hook_no_error(self):
        # Should not raise even with no hooks registered
        rc.set_provider("cloud")


class TestGetBaseUrl:
    def test_local_returns_ollama_base_url(self):
        from agent.config import OLLAMA_BASE_URL
        rc._provider = "local"
        assert rc.get_base_url() == OLLAMA_BASE_URL

    def test_cloud_returns_cloud_url(self):
        from agent.config import OLLAMA_CLOUD_URL
        rc._provider = "cloud"
        assert rc.get_base_url() == OLLAMA_CLOUD_URL


class TestGetApiKey:
    def test_local_returns_empty(self):
        rc._provider = "local"
        assert rc.get_api_key() == ""

    def test_cloud_returns_cloud_key(self, monkeypatch):
        import agent.runtime_config as rc_mod
        monkeypatch.setattr("agent.runtime_config._provider", "cloud")
        # The key comes from agent.config.OLLAMA_CLOUD_API_KEY
        from agent.config import OLLAMA_CLOUD_API_KEY
        assert rc_mod.get_api_key() == OLLAMA_CLOUD_API_KEY


class TestGetEffectiveModel:
    def test_local_chat_returns_fast_text_model(self):
        from agent.config import FAST_TEXT_MODEL
        rc._provider = "local"
        assert rc.get_effective_model("chat") == FAST_TEXT_MODEL

    def test_local_kb_refine_returns_main_model(self):
        from agent.config import MAIN_MODEL
        rc._provider = "local"
        assert rc.get_effective_model("kb_refine") == MAIN_MODEL

    def test_local_fast_returns_fast_model(self):
        from agent.config import FAST_MODEL
        rc._provider = "local"
        assert rc.get_effective_model("fast") == FAST_MODEL

    def test_local_voice_returns_voice_model(self):
        from agent.config import VOICE_MODEL
        rc._provider = "local"
        assert rc.get_effective_model("voice") == VOICE_MODEL

    def test_cloud_chat_returns_cloud_chat_model(self, monkeypatch):
        from agent.config import CLOUD_CHAT_MODEL, FAST_TEXT_MODEL
        monkeypatch.setattr(rc, "_provider", "cloud")
        expected = CLOUD_CHAT_MODEL or FAST_TEXT_MODEL
        assert rc.get_effective_model("chat") == expected

    def test_cloud_kb_refine_returns_cloud_kb_model(self, monkeypatch):
        from agent.config import CLOUD_KB_REFINE_MODEL, MAIN_MODEL
        monkeypatch.setattr(rc, "_provider", "cloud")
        expected = CLOUD_KB_REFINE_MODEL or MAIN_MODEL
        assert rc.get_effective_model("kb_refine") == expected

    def test_cloud_voice_returns_cloud_voice_model(self, monkeypatch):
        from agent.config import CLOUD_VOICE_MODEL, VOICE_MODEL
        monkeypatch.setattr(rc, "_provider", "cloud")
        expected = CLOUD_VOICE_MODEL or VOICE_MODEL
        assert rc.get_effective_model("voice") == expected

    def test_unknown_role_returns_fast_text_model(self):
        from agent.config import FAST_TEXT_MODEL
        rc._provider = "local"
        assert rc.get_effective_model("nonexistent_role") == FAST_TEXT_MODEL

    def test_unknown_role_cloud_returns_fast_text_model(self, monkeypatch):
        from agent.config import FAST_TEXT_MODEL
        monkeypatch.setattr(rc, "_provider", "cloud")
        assert rc.get_effective_model("nonexistent_role") == FAST_TEXT_MODEL

    def test_reads_current_provider_not_startup_value(self, monkeypatch):
        """Toggling provider mid-session changes what get_effective_model returns."""
        from agent.config import FAST_TEXT_MODEL, CLOUD_CHAT_MODEL
        monkeypatch.setattr(rc, "_provider", "local")
        local_result = rc.get_effective_model("chat")
        assert local_result == FAST_TEXT_MODEL

        monkeypatch.setattr(rc, "_provider", "cloud")
        cloud_result = rc.get_effective_model("chat")
        cloud_expected = CLOUD_CHAT_MODEL or FAST_TEXT_MODEL
        assert cloud_result == cloud_expected

        # They differ if CLOUD_CHAT_MODEL is set
        if CLOUD_CHAT_MODEL:
            assert local_result != cloud_result
