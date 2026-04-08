"""Model lifecycle management for Ollama.

Centralizes VRAM management: tracks which model is active, handles
load/unload via Ollama keep_alive=0, and provides swap helpers for
the KB refinement pipeline (e4b <-> 26b).

VRAM rules:
- One big model at a time
- e4b + e2b can coexist (~5GB total)
- When 26b loads, e4b unloads; when 26b finishes, e4b reloads
"""

import asyncio
import logging

import ollama as _ollama_client

from agent.config import CHAT_MODEL, KB_REFINE_MODEL, OLLAMA_BASE_URL

log = logging.getLogger(__name__)

_active_model: str | None = None
_swap_lock = asyncio.Lock()


def get_active_model() -> str | None:
    """Return the currently active chat model name."""
    return _active_model


async def ensure_model(model_name: str) -> None:
    """Ensure the given model is the active chat model in VRAM.

    If a different model is currently active, unload it first via
    Ollama keep_alive=0. Safe to call repeatedly with the same model.
    """
    global _active_model
    async with _swap_lock:
        if _active_model and _active_model != model_name:
            await _async_unload(_active_model)
        _active_model = model_name


async def unload_model(model_name: str) -> None:
    """Unload a specific model from Ollama VRAM. Best-effort."""
    global _active_model
    async with _swap_lock:
        await _async_unload(model_name)
        if _active_model == model_name:
            _active_model = None


async def swap_for_kb() -> None:
    """Swap from chat model (e4b) to KB refinement model (26b).

    Unloads e4b, sets 26b as active. Called before 26b inference.
    """
    global _active_model
    async with _swap_lock:
        if _active_model and _active_model != KB_REFINE_MODEL:
            await _async_unload(_active_model)
        _active_model = KB_REFINE_MODEL
    log.info("Model swap: loaded %s for KB refinement", KB_REFINE_MODEL)


async def swap_back_from_kb() -> None:
    """Swap from KB refinement model (26b) back to chat model (e4b).

    Unloads 26b, sets e4b as active. Called after 26b inference completes.
    """
    global _active_model
    async with _swap_lock:
        await _async_unload(KB_REFINE_MODEL)
        _active_model = CHAT_MODEL
    log.info("Model swap: unloaded %s, restored %s", KB_REFINE_MODEL, CHAT_MODEL)


def sync_unload_model(model_name: str) -> None:
    """Sync variant of unload for use in non-async code (kb_index, tagger).

    Uses the sync Ollama client directly.
    """
    try:
        client = _ollama_client.Client(host=OLLAMA_BASE_URL)
        client.generate(model=model_name, prompt="", keep_alive=0)
        log.info("Model unload (sync): %s", model_name)
    except Exception:
        pass


def sync_swap_for_kb() -> None:
    """Sync variant of swap_for_kb for CLI context."""
    global _active_model
    if _active_model and _active_model != KB_REFINE_MODEL:
        sync_unload_model(_active_model)
    _active_model = KB_REFINE_MODEL
    log.info("Model swap (sync): loaded %s for KB refinement", KB_REFINE_MODEL)


def sync_swap_back_from_kb() -> None:
    """Sync variant of swap_back_from_kb for CLI context."""
    global _active_model
    sync_unload_model(KB_REFINE_MODEL)
    _active_model = CHAT_MODEL
    log.info("Model swap (sync): unloaded %s, restored %s", KB_REFINE_MODEL, CHAT_MODEL)


async def _async_unload(model_name: str) -> None:
    """Send keep_alive=0 to Ollama to flush a model from VRAM."""
    try:
        client = _ollama_client.Client(host=OLLAMA_BASE_URL)
        await asyncio.to_thread(
            client.generate, model=model_name, prompt="", keep_alive=0
        )
        log.info("Model unload: %s", model_name)
    except Exception:
        pass  # best-effort
