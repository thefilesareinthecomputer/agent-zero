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

    def test_result_has_discovery_fields(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("Topic", "Content.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        assert len(results) == 1
        hit = results[0]
        for field in ("filename", "source", "heading", "summary", "chunk_index", "distance"):
            assert field in hit, f"Missing field: {field}"

    def test_no_full_content_in_results(self, tmp_path, mock_summary, _ephemeral_collection):
        content = _structured_file([("Topic", "Detailed content here.")])
        _write_kb_file(tmp_path, "test.md", content)
        index_file("test.md", "knowledge", base_dir=tmp_path)

        results = search_kb("topic", top_k=5)
        hit = results[0]
        # Should NOT contain the full content field
        assert "content" not in hit

    def test_empty_collection_returns_empty(self, _ephemeral_collection):
        results = search_kb("anything", top_k=5)
        assert results == []


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
