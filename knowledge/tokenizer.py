"""Token counting and truncation utilities for knowledge base context limits.

Uses tiktoken cl100k_base as a proxy for Gemma tokenization. cl100k_base
undercounts relative to Gemma4's tokenizer (~1.5x ratio observed in
practice). A safety multiplier is applied so all token counts --
heading trees, budget checks, per-file limits -- err on the conservative
side. The agent sees slightly inflated numbers and stays within limits.
"""

import math

import tiktoken

_encoder: tiktoken.Encoding | None = None

# cl100k_base produces ~1.5x fewer tokens than Gemma4's tokenizer.
# Apply this multiplier so all callers get conservative estimates.
_GEMMA_SAFETY_MULTIPLIER = 1.5


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return the estimated token count for a string.

    Applies a safety multiplier to approximate Gemma4 tokenization
    from cl100k_base counts.
    """
    if not text:
        return 0
    raw = len(_get_encoder().encode(text))
    return int(raw * _GEMMA_SAFETY_MULTIPLIER)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within a token budget.

    max_tokens is in Gemma-equivalent tokens. Internally converts to
    cl100k_base tokens for the actual truncation point.

    Returns (text, was_truncated). Truncation happens at token boundaries,
    not character boundaries, so no partial tokens appear in output.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    # Convert Gemma-equivalent limit to cl100k_base limit (ceil to avoid
    # truncating below the actual raw count on round-trip)
    raw_limit = math.ceil(max_tokens / _GEMMA_SAFETY_MULTIPLIER)
    if len(tokens) <= raw_limit:
        return text, False
    truncated = enc.decode(tokens[:raw_limit])
    return truncated, True
