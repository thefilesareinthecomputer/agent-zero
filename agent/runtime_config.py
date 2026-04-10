"""Mutable runtime configuration. Initialized from env defaults at startup.

Holds provider state so the app can toggle between local and cloud Ollama
at runtime without a restart. All LLM factories and VRAM management functions
read from this module rather than from the startup-time constants in config.py.
"""
from agent.config import (
    IS_CLOUD, OLLAMA_BASE_URL, OLLAMA_CLOUD_API_KEY, OLLAMA_CLOUD_URL,
    FAST_TEXT_MODEL, MAIN_MODEL, FAST_MODEL, VOICE_MODEL,
    CLOUD_CHAT_MODEL, CLOUD_KB_REFINE_MODEL, CLOUD_FAST_MODEL, CLOUD_VOICE_MODEL,
)

_provider: str = "cloud" if IS_CLOUD else "local"
_on_change_hooks: list = []


def get_provider() -> str:
    """Return the active provider: 'local' or 'cloud'."""
    return _provider


def is_cloud() -> bool:
    """Return True when the active provider is cloud."""
    return _provider == "cloud"


def get_base_url() -> str:
    """Return the Ollama base URL for the active provider."""
    return OLLAMA_CLOUD_URL if is_cloud() else OLLAMA_BASE_URL


def get_api_key() -> str:
    """Return the API key for the active provider (empty string for local)."""
    return OLLAMA_CLOUD_API_KEY if is_cloud() else ""


def set_provider(provider: str) -> None:
    """Switch the active provider. Fires all registered on-change hooks."""
    global _provider
    if provider not in ("local", "cloud"):
        raise ValueError(f"provider must be 'local' or 'cloud', got {provider!r}")
    _provider = provider
    for hook in _on_change_hooks:
        hook()


def register_on_change(fn) -> None:
    """Register a callable invoked whenever set_provider() is called.

    Used by modules with cached clients (e.g. memory/tagger.py) to
    invalidate their cached client when the provider changes.
    """
    _on_change_hooks.append(fn)


def get_effective_model(role: str) -> str:
    """Return the active model name for a role based on the current provider.

    Roles: 'chat', 'kb_refine', 'fast', 'voice'.
    Reads provider state at call time, so this is safe to call after a
    set_provider() toggle -- no restart needed.
    """
    if is_cloud():
        return {
            "chat": CLOUD_CHAT_MODEL or FAST_TEXT_MODEL,
            "kb_refine": CLOUD_KB_REFINE_MODEL or MAIN_MODEL,
            "fast": CLOUD_FAST_MODEL or FAST_MODEL,
            "voice": CLOUD_VOICE_MODEL or VOICE_MODEL,
        }.get(role, FAST_TEXT_MODEL)
    return {
        "chat": FAST_TEXT_MODEL,
        "kb_refine": MAIN_MODEL,
        "fast": FAST_MODEL,
        "voice": VOICE_MODEL,
    }.get(role, FAST_TEXT_MODEL)
