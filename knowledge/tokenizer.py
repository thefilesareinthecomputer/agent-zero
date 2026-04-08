"""Token counting and truncation utilities for knowledge base context limits.

Uses tiktoken cl100k_base as a proxy for Gemma tokenization. cl100k_base
typically over-counts relative to Gemma, which is the safe direction -- the
agent stays within context limits rather than exceeding them.
"""

import tiktoken

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return the token count for a string."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within a token budget.

    Returns (text, was_truncated). Truncation happens at token boundaries,
    not character boundaries, so no partial tokens appear in output.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = enc.decode(tokens[:max_tokens])
    return truncated, True
