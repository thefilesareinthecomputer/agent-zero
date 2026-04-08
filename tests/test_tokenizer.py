"""Tests for knowledge/tokenizer.py -- token counting and truncation."""

from knowledge.tokenizer import count_tokens, truncate_to_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_nonempty_returns_positive(self):
        assert count_tokens("hello world") > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world this is a longer sentence with more tokens")
        assert long > short

    def test_single_word(self):
        result = count_tokens("test")
        assert isinstance(result, int)
        assert result >= 1


class TestTruncateToTokens:
    def test_under_limit_unchanged(self):
        text = "short text"
        result, was_truncated = truncate_to_tokens(text, 1000)
        assert result == text
        assert was_truncated is False

    def test_over_limit_truncated(self):
        text = "word " * 500  # ~500 tokens
        result, was_truncated = truncate_to_tokens(text, 10)
        assert was_truncated is True
        assert len(result) < len(text)
        # Truncated output should be decodable (no partial tokens)
        assert isinstance(result, str)

    def test_exact_limit(self):
        text = "hello"
        tokens = count_tokens(text)
        result, was_truncated = truncate_to_tokens(text, tokens)
        assert result == text
        assert was_truncated is False

    def test_truncation_produces_valid_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        result, was_truncated = truncate_to_tokens(text, 20)
        assert was_truncated is True
        # Result should be valid UTF-8 (no mid-byte cuts)
        result.encode("utf-8")
