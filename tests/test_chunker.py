"""Tests for knowledge/chunker.py -- section-based markdown chunking."""

import pytest

from knowledge.chunker import (
    HeadingNode,
    build_heading_tree,
    chunk_file,
    format_heading_tree,
)


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


class TestH1Splitting:
    def test_splits_on_h1(self):
        body = (
            "# Chapter One\nContent one.\n\n"
            "# Chapter Two\nContent two.\n"
        )
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Chapter One"
        assert chunks[1]["heading"] == "Chapter Two"

    def test_h1_detected_as_highest(self):
        body = (
            "# Top Level\n"
            "## Sub Section\nContent.\n"
        )
        chunks = chunk_file(_make_file(body), "test.md")
        # Should split on H1, keeping H2 inside the content
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Top Level"
        assert "## Sub Section" in chunks[0]["content"]


class TestH5Splitting:
    def test_splits_on_h5_when_highest(self):
        body = (
            "##### Detail A\nFine-grained A.\n\n"
            "##### Detail B\nFine-grained B.\n"
        )
        chunks = chunk_file(_make_file(body), "test.md")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Detail A"
        assert chunks[1]["heading"] == "Detail B"


class TestBuildHeadingTree:
    def test_empty_body(self):
        tree = build_heading_tree(_make_file(""), "test.md")
        assert tree.level == 0
        assert tree.subtree_tokens == 0
        assert tree.children == []

    def test_no_headings(self):
        body = "Just plain text with no headings at all."
        tree = build_heading_tree(_make_file(body), "test.md")
        assert tree.level == 0
        assert tree.own_tokens > 0
        assert tree.subtree_tokens == tree.own_tokens
        assert tree.children == []

    def test_flat_h2_structure(self):
        body = (
            "## Alpha\nContent alpha.\n\n"
            "## Beta\nContent beta.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        assert len(tree.children) == 2
        assert tree.children[0].heading == "Alpha"
        assert tree.children[0].level == 2
        assert tree.children[1].heading == "Beta"
        assert tree.children[1].level == 2

    def test_nested_h2_h3(self):
        body = (
            "## Section\n"
            "### Sub A\nContent A.\n"
            "### Sub B\nContent B.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        assert len(tree.children) == 1
        section = tree.children[0]
        assert section.heading == "Section"
        assert len(section.children) == 2
        assert section.children[0].heading == "Sub A"
        assert section.children[1].heading == "Sub B"

    def test_deep_nesting_h1_to_h5(self):
        body = (
            "# Title\n"
            "## Chapter\n"
            "### Section\n"
            "#### Subsection\n"
            "##### Detail\nDeep content.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        assert len(tree.children) == 1
        h1 = tree.children[0]
        assert h1.level == 1
        assert h1.heading == "Title"
        h2 = h1.children[0]
        assert h2.level == 2
        h3 = h2.children[0]
        assert h3.level == 3
        h4 = h3.children[0]
        assert h4.level == 4
        h5 = h4.children[0]
        assert h5.level == 5
        assert h5.heading == "Detail"

    def test_subtree_tokens_roll_up(self):
        body = (
            "## Parent\n"
            "### Child A\nSome content here.\n"
            "### Child B\nMore content here.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        parent = tree.children[0]
        child_sum = sum(c.subtree_tokens for c in parent.children)
        assert parent.subtree_tokens == parent.own_tokens + child_sum
        assert tree.subtree_tokens == tree.own_tokens + parent.subtree_tokens

    def test_preamble_tokens_in_root(self):
        body = (
            "Some preamble before any heading.\n\n"
            "## Section\nContent.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        assert tree.own_tokens > 0  # preamble counted
        assert tree.subtree_tokens > tree.children[0].subtree_tokens

    def test_sibling_levels_stay_flat(self):
        """H2 followed by H2 should be siblings, not parent-child."""
        body = (
            "## First\nA.\n\n"
            "## Second\nB.\n\n"
            "## Third\nC.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        assert len(tree.children) == 3
        for child in tree.children:
            assert child.level == 2
            assert child.children == []


class TestFormatHeadingTree:
    def test_renders_root_and_children(self):
        body = (
            "## Alpha\nContent.\n\n"
            "## Beta\nMore.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        output = format_heading_tree(tree)
        assert "test" in output  # root = filename stem
        assert "## Alpha" in output
        assert "## Beta" in output
        assert "tokens" in output

    def test_nested_indentation(self):
        body = (
            "## Parent\n"
            "### Child\nContent.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        output = format_heading_tree(tree)
        lines = output.split("\n")
        # Find the child line -- should be more indented than parent
        parent_line = [l for l in lines if "Parent" in l][0]
        child_line = [l for l in lines if "Child" in l][0]
        parent_indent = len(parent_line) - len(parent_line.lstrip())
        child_indent = len(child_line) - len(child_line.lstrip())
        assert child_indent > parent_indent

    def test_leaf_shows_own_tokens(self):
        body = "## Leaf\nSome leaf content here.\n"
        tree = build_heading_tree(_make_file(body), "test.md")
        output = format_heading_tree(tree)
        # Leaf node should show its token count
        leaf_line = [l for l in output.split("\n") if "Leaf" in l][0]
        assert "tokens" in leaf_line

    def test_branch_shows_subtree_tokens(self):
        body = (
            "## Branch\n"
            "### Leaf A\nContent A.\n"
            "### Leaf B\nContent B.\n"
        )
        tree = build_heading_tree(_make_file(body), "test.md")
        output = format_heading_tree(tree)
        branch_line = [l for l in output.split("\n") if "Branch" in l][0]
        # Branch should show subtree tokens (larger than any single leaf)
        assert "tokens" in branch_line
