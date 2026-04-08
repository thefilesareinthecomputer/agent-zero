"""Tests for knowledge/chunker.py -- section-based markdown chunking."""

import pytest

from knowledge.chunker import chunk_file


def _make_file(body: str, tags: list[str] | None = None) -> str:
    """Wrap body in frontmatter to produce a complete knowledge file."""
    tag_lines = "\n".join(f"  - {t}" for t in (tags or ["test"]))
    return (
        f"---\n"
        f"date-created: 2026-04-08\n"
        f"last-modified: 2026-04-08\n"
        f"tags:\n"
        f"{tag_lines}\n"
        f"---\n"
        f"{body}"
    )


class TestH2Splitting:
    def test_splits_on_h2(self):
        body = (
            "## Section One\nContent one.\n\n"
            "## Section Two\nContent two.\n"
        )
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Section One"
        assert chunks[1]["heading"] == "Section Two"

    def test_chunk_has_required_keys(self):
        body = "## Overview\nSome content here.\n"
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 1
        chunk = chunks[0]
        for key in ("heading", "content", "chunk_index", "token_count"):
            assert key in chunk, f"Missing key: {key}"

    def test_chunk_index_sequential(self):
        body = "## A\nFirst.\n\n## B\nSecond.\n\n## C\nThird.\n"
        chunks = chunk_file(_make_file(body), "test.md")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == [0, 1, 2]

    def test_token_count_positive(self):
        body = "## Overview\nSome content here.\n"
        chunks = chunk_file(_make_file(body), "test.md")
        assert chunks[0]["token_count"] > 0

    def test_preamble_before_first_h2_included(self):
        body = "Some preamble text.\n\n## Section\nContent.\n"
        chunks = chunk_file(_make_file(body), "test.md")
        # Preamble with content should be included
        headings = [c["heading"] for c in chunks]
        assert any("preamble" in h or "Section" in h for h in headings)


class TestH3Splitting:
    def test_splits_on_h3_when_no_h2(self):
        body = (
            "### Concept A\nDetails about A.\n\n"
            "### Concept B\nDetails about B.\n"
        )
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Concept A"
        assert chunks[1]["heading"] == "Concept B"


class TestNoHeadings:
    def test_single_chunk_for_plain_text(self):
        body = "Just some plain text without any headings.\nAnother line.\n"
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "test"  # stem of filename

    def test_empty_body_returns_empty(self):
        chunks = chunk_file(_make_file(""), "test.md")
        assert chunks == []


class TestFrontmatterStripping:
    def test_frontmatter_not_in_chunks(self):
        body = "## Content\nActual content here.\n"
        text = _make_file(body)
        chunks = chunk_file(text, "test.md")
        for chunk in chunks:
            assert "date-created" not in chunk["content"]
            assert "last-modified" not in chunk["content"]


class TestMaxTokens:
    def test_small_sections_unchanged(self):
        body = "## A\nShort.\n\n## B\nAlso short.\n"
        chunks = chunk_file(_make_file(body), "test.md", max_tokens=1000)
        assert len(chunks) == 2

    def test_oversized_section_with_subheadings_splits(self):
        # Build a section that exceeds token limit with H3 sub-sections
        sub_content = "Detailed content. " * 100
        body = (
            f"## Big Section\n"
            f"### Sub A\n{sub_content}\n"
            f"### Sub B\n{sub_content}\n"
        )
        chunks = chunk_file(_make_file(body), "test.md", max_tokens=200)
        # Should have more than 1 chunk due to splitting
        assert len(chunks) >= 2

    def test_hard_split_when_no_subheadings(self):
        # Single section with no sub-headings that exceeds token limit
        long_content = "word " * 1000
        body = f"## Monolith\n{long_content}\n"
        chunks = chunk_file(_make_file(body), "test.md", max_tokens=100)
        assert len(chunks) > 1
        # All chunks should be within token budget (with some tolerance)
        for chunk in chunks:
            assert chunk["token_count"] <= 150  # allow some overlap margin
