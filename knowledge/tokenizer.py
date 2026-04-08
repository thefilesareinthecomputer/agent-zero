"""Token counting and truncation utilities.

Uses tiktoken cl100k_base as a general-purpose tokenizer. All counts are
raw cl100k_base tokens -- no multipliers, no adjustments. This gives
consistent, predictable numbers throughout the system: chunker, heading
trees, KB index metadata, embedding limits.

For Gemma4 context budget estimation (where cl100k_base undercounts by
~1.5x), use estimate_gemma_tokens() instead. That function is only used
in agent.py for context window management and in tools.py for token
counts shown to the agent.
"""

import tiktoken

_encoder: tiktoken.Encoding | None = None

# cl100k_base undercounts relative to Gemma4's tokenizer by ~1.5x.
# Only applied in estimate_gemma_tokens(), not in count_tokens().
_GEMMA_MULTIPLIER = 1.5


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for a string.

    Raw count, no multipliers. Used by the chunker, heading trees,
    KB index metadata, and embedding size limits.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def estimate_gemma_tokens(text: str) -> int:
    """Estimate Gemma4 token count from cl100k_base.

    Applies a 1.5x multiplier to approximate Gemma4 tokenization.
    Use this ONLY for context budget calculations shown to the agent
    (agent.py, tools.py). Everything else should use count_tokens().
    """
    if not text:
        return 0
    raw = len(_get_encoder().encode(text))
    return int(raw * _GEMMA_MULTIPLIER)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within a token budget.

    max_tokens is in cl100k_base tokens (raw count).

    Returns (text, was_truncated). Truncation happens at token boundaries,
    not character boundaries, so no partial tokens appear in output.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = enc.decode(tokens[:max_tokens])
    return truncated, True
