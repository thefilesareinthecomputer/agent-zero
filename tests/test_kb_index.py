"""Tests for knowledge/kb_index.py -- KB vector index with mocked Ollama.

All tests mock _generate_summary to avoid needing a running Ollama instance.
ChromaDB uses an ephemeral client via monkeypatching.
"""

from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

import knowledge.kb_index as kb_index_mod
from knowledge.kb_index import (
    get_summaries,
    index_file,
    remove_file,
    search_kb,
    sync_kb_index,
)


@pytest.fixture(autouse=True)
def _ephemeral_collection(monkeypatch, tmp_path):
    """Replace the KB collection with a fresh ephemeral ChromaDB for each test."""
    client = chromadb.EphemeralClient()
    # Delete existing collection if present, then create fresh
    try:
        client.delete_collection("knowledge")
    except Exception:
        pass
    collection = client.create_collection(
        name="knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(kb_index_mod, "_client", client)
    monkeypatch.setattr(kb_index_mod, "_collection", collection)
    yield collection


@pytest.fixture
def mock_summary():
    """Mock _generate_summary to return a deterministic summary."""
    with patch.object(
        kb_index_mod, "_generate_summary",
        side_effect=lambda content, heading: f"Summary of {heading}",
    ):
        yield


def _write_kb_file(directory: Path, filename: str, content: str) -> Path:
    """Write a knowledge file to a directory."""
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _structured_file(sections: list[tuple[str, str]]) -> str:
    """Build a file with frontmatter and H2 sections."""
    lines = [
        "---",
        "date-created: 2026-04-08",
        "last-modified: 2026-04-08",
        "tags:",
        "  - test",
        "---",
    ]
    for heading, body in sections:
        lines.append(f"\n## {heading}\n")
        lines.append(body)
    return "\n".join(lines)


class TestIndexFile:
    def test_creates_chunks(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([
            ("Intro", "Introduction to the topic."),
            ("Details", "Detailed information here."),
        ])
        _write_kb_file(tmp_path, "test.md", content)
        count = index_file("test.md", "knowledge", base_dir=tmp_path)
        assert count == 2
        assert _ephemeral_collection.count() == 2

    def test_replaces_on_reindex(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("A", "Content A.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)
        assert _ephemeral_collection.count() == 1

        # Re-index with different content
        content2 = _structured_file([("B", "Content B."), ("C", "Content C.")])
        _write_kb_file(tmp_path, "test.md", content2)
        index_file("test.md", "knowledge", base_dir=tmp_path)
        assert _ephemeral_collection.count() == 2

    def test_missing_file_returns_zero(self, tmp_path, mock_summary):
        count = index_file("nonexistent.md", "knowledge", base_dir=tmp_path)
        assert count == 0

    def test_metadata_fields(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("Topic", "Content.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = _ephemeral_collection.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert meta["filename"] == "test.md"
        assert meta["source"] == "knowledge"
        assert meta["heading"] == "Topic"
        assert "summary" in meta
        assert "mtime" in meta
        assert "token_count" in meta

    def test_file_level_metadata(self, tmp_path, mock_summary, _ephemeral_collection):
        """Each chunk should carry file-level stats."""
        content = _structured_file([
            ("Alpha", "Content about alpha."),
            ("Beta", "Content about beta."),
        ])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = _ephemeral_collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            assert "file_tokens" in meta
            assert meta["file_tokens"] > 0
            assert meta["section_count"] == 2
            assert "file_outline" in meta
            assert "Alpha" in meta["file_outline"]
            assert "Beta" in meta["file_outline"]

    def test_tags_and_project_in_metadata(self, tmp_path, mock_summary, _ephemeral_collection):
        """Chunks should carry the file's frontmatter tags and project."""
        tag_lines = "\n".join(f"  - {t}" for t in ["data-modeling", "sql"])
        content = (
            f"---\n"
            f"date-created: 2026-04-08\n"
            f"last-modified: 2026-04-08\n"
            f"tags:\n"
            f"{tag_lines}\n"
            f"project: analytics\n"
            f"---\n"
            f"## Topic\nContent.\n"
        )
        _write_kb_file(tmp_path, "tagged.md", content)
        index_file("tagged.md", "knowledge", base_dir=tmp_path)

        results = _ephemeral_collection.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert "tags" in meta
        assert "data-modeling" in meta["tags"]
        assert "sql" in meta["tags"]
        assert meta["project"] == "analytics"

    def test_no_tags_stored_as_empty(self, tmp_path, mock_summary, _ephemeral_collection):
        """Files without tags should have empty string, not missing key."""
        content = "---\ndate-created: 2026-04-08\nlast-modified: 2026-04-08\ntags:\n  - test\n---\n## A\nContent.\n"
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = _ephemeral_collection.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert "tags" in meta
        assert "project" in meta


class TestGetSummaries:
    def test_returns_heading_summary_map(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([
            ("Intro", "Introduction to the topic."),
            ("Details", "Detailed information here."),
        ])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        summaries = get_summaries("test.md", "knowledge")
        assert "Intro" in summaries
        assert "Details" in summaries
        assert summaries["Intro"] == "Summary of Intro"
        assert summaries["Details"] == "Summary of Details"

    def test_returns_empty_for_unindexed_file(self, _ephemeral_collection):
        summaries = get_summaries("nonexistent.md", "knowledge")
        assert summaries == {}

    def test_returns_empty_on_empty_collection(self, _ephemeral_collection):
        summaries = get_summaries("anything.md", "knowledge")
        assert summaries == {}


class TestRemoveFile:
    def test_removes_all_chunks(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("A", "Content A."), ("B", "Content B.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)
        assert _ephemeral_collection.count() == 2

        removed = remove_file("test.md", "knowledge")
        assert removed == 2
        assert _ephemeral_collection.count() == 0

    def test_remove_nonexistent_returns_zero(self):
        removed = remove_file("ghost.md", "knowledge")
        assert removed == 0


class TestSearchKb:
    def test_returns_results(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([
            ("Machine Learning", "Neural networks and deep learning."),
            ("Databases", "SQL and relational databases."),
        ])
        _write_kb_file(tmp_path, "topics.md", content)
        index_file("topics.md", "knowledge", base_dir=tmp_path)

        results = search_kb("neural networks", top_k=5)
        assert len(results) > 0

    def test_result_grouped_by_file(self, tmp_path, mock_summary, _ephemeral_collection):
        """search_kb should return results grouped by filename."""
        content = _structured_file([
            ("Topic A", "Content about topic A."),
            ("Topic B", "Content about topic B."),
        ])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        # Single file should produce single group
        assert len(results) == 1
        file_hit = results[0]
        assert file_hit["filename"] == "test.md"
        assert file_hit["source"] == "knowledge"
        assert len(file_hit["hits"]) == 2

    def test_grouped_result_has_file_metadata(self, tmp_path, mock_summary, _ephemeral_collection):
        """Grouped results should include file-level metadata."""
        content = _structured_file([("Topic", "Content.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        file_hit = results[0]
        for field in ("filename", "source", "file_tokens", "section_count", "file_outline", "hits"):
            assert field in file_hit, f"Missing field: {field}"
        assert file_hit["file_tokens"] > 0
        assert file_hit["section_count"] == 1

    def test_hit_has_discovery_fields(self, tmp_path, mock_summary, _ephemeral_collection):
        """Individual hits within a file group should have heading, summary, distance."""
        content = _structured_file([("Topic", "Content.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        hit = results[0]["hits"][0]
        for field in ("heading", "summary", "chunk_index", "distance"):
            assert field in hit, f"Missing field: {field}"

    def test_no_full_content_in_results(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("Topic", "Detailed content here.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        file_hit = results[0]
        # Should NOT contain the full content field at file or hit level
        assert "content" not in file_hit
        assert "content" not in file_hit["hits"][0]

    def test_empty_collection_returns_empty(self, _ephemeral_collection):
        results = search_kb("anything", top_k=5)
        assert results == []

    def test_multiple_files_grouped_separately(self, tmp_path, mock_summary, _ephemeral_collection):
        """Results from different files should be in different groups."""
        content_a = _structured_file([("Neural Nets", "Deep learning overview.")])
        content_b = _structured_file([("SQL Basics", "Relational database intro.")])
        _write_kb_file(tmp_path, "ml.md", content_a)
        _write_kb_file(tmp_path, "db.md", content_b)
        index_file("ml.md", "knowledge", base_dir=tmp_path)
        index_file("db.md", "knowledge", base_dir=tmp_path)

        results = search_kb("learning databases", top_k=5)
        filenames = {r["filename"] for r in results}
        assert len(filenames) == 2

    def test_files_sorted_by_best_hit(self, tmp_path, mock_summary, _ephemeral_collection):
        """Files should be ordered by their best (lowest distance) hit."""
        content_a = _structured_file([("Exact Match Topic", "This is about exact match topic.")])
        content_b = _structured_file([("Unrelated", "Something completely different.")])
        _write_kb_file(tmp_path, "relevant.md", content_a)
        _write_kb_file(tmp_path, "other.md", content_b)
        index_file("relevant.md", "knowledge", base_dir=tmp_path)
        index_file("other.md", "knowledge", base_dir=tmp_path)

        results = search_kb("exact match topic", top_k=5)
        # The more relevant file should come first
        assert results[0]["filename"] == "relevant.md"


class TestSyncKbIndex:
    def test_indexes_new_files(self, tmp_path, mock_summary, monkeypatch, _ephemeral_collection):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        canon_dir = tmp_path / "canon"
        canon_dir.mkdir()

        content = _structured_file([("New", "New content.")])
        _write_kb_file(kb_dir, "new.md", content)

        monkeypatch.setattr(kb_index_mod, "_KNOWLEDGE_DIR", kb_dir)
        monkeypatch.setattr(kb_index_mod, "_CANON_DIR", canon_dir)

        result = sync_kb_index()
        assert result["indexed"] > 0
        assert _ephemeral_collection.count() > 0

    def test_skips_unchanged_files(self, tmp_path, mock_summary, monkeypatch, _ephemeral_collection):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        canon_dir = tmp_path / "canon"
        canon_dir.mkdir()

        content = _structured_file([("Existing", "Content.")])
        _write_kb_file(kb_dir, "existing.md", content)

        monkeypatch.setattr(kb_index_mod, "_KNOWLEDGE_DIR", kb_dir)
        monkeypatch.setattr(kb_index_mod, "_CANON_DIR", canon_dir)

        # First sync indexes the file
        result1 = sync_kb_index()
        assert result1["indexed"] > 0

        # Second sync should skip it (mtime unchanged)
        result2 = sync_kb_index()
        assert result2["indexed"] == 0

    def test_removes_deleted_files(self, tmp_path, mock_summary, monkeypatch, _ephemeral_collection):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        canon_dir = tmp_path / "canon"
        canon_dir.mkdir()

        content = _structured_file([("Delete Me", "Temporary.")])
        path = _write_kb_file(kb_dir, "temp.md", content)

        monkeypatch.setattr(kb_index_mod, "_KNOWLEDGE_DIR", kb_dir)
        monkeypatch.setattr(kb_index_mod, "_CANON_DIR", canon_dir)

        sync_kb_index()
        assert _ephemeral_collection.count() > 0

        # Delete the file and re-sync
        path.unlink()
        result = sync_kb_index()
        assert result["removed"] > 0
        assert _ephemeral_collection.count() == 0

    def test_skips_index_and_log(self, tmp_path, mock_summary, monkeypatch, _ephemeral_collection):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        canon_dir = tmp_path / "canon"
        canon_dir.mkdir()

        _write_kb_file(kb_dir, "index.md", "# Index\n| File | Tags |\n")
        _write_kb_file(kb_dir, "log.md", "## [2026-04-08] save | test.md\n")
        _write_kb_file(kb_dir, "real.md", _structured_file([("Real", "Content.")]))

        monkeypatch.setattr(kb_index_mod, "_KNOWLEDGE_DIR", kb_dir)
        monkeypatch.setattr(kb_index_mod, "_CANON_DIR", canon_dir)

        sync_kb_index()
        # Only real.md should be indexed, not index.md or log.md
        results = _ephemeral_collection.get(include=["metadatas"])
        filenames = {m["filename"] for m in results["metadatas"]}
        assert "real.md" in filenames
        assert "index.md" not in filenames
        assert "log.md" not in filenames
