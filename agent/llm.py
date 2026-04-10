"""LangChain and Ollama client factories with local/cloud routing.

All code that needs a ChatOllama or ollama.Client should call these factories
rather than instantiating directly. The factories read the active provider from
agent/runtime_config.py so they automatically pick up runtime toggles.
"""
from __future__ import annotations

import ollama
from langchain_ollama import ChatOllama

from agent.runtime_config import get_api_key, get_base_url


def _auth_headers() -> dict[str, str]:
    key = get_api_key()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def make_chat_ollama(model: str, **kwargs) -> ChatOllama:
    """Return a ChatOllama instance routed to local or cloud.

    Callers pass num_ctx, num_predict, etc. as kwargs; this function
    handles base_url and auth header injection transparently.
    """
    client_kwargs: dict = {}
    headers = _auth_headers()
    if headers:
        client_kwargs["headers"] = headers
    return ChatOllama(
        model=model,
        base_url=get_base_url(),
        client_kwargs=client_kwargs,
        **kwargs,
    )


def make_ollama_client() -> ollama.Client:
    """Return an ollama.Client routed to local or cloud."""
    headers = _auth_headers()
    if headers:
        return ollama.Client(host=get_base_url(), headers=headers)
    return ollama.Client(host=get_base_url())
