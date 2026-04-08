"""Tests for knowledge/knowledge_store.py -- file ops, frontmatter, search, index, log."""

from pathlib import Path

import pytest

from knowledge.knowledge_store import (
    _sanitize_path,
    append_log,
    get_file_metadata,
    list_files,
    read_file,
    rebuild_index,
    save_file,
    search_files,
)


# -- Helpers --

def _write_raw(directory: Path, filename: str, content: str) -> Path:
    """Write a raw file into a directory (no processing)."""
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _structured_file(tags: list[str], content: str,
                     last_modified: str = "2026-04-07") -> str:
    """Return a well-formed knowledge file string.

    Built without textwrap.dedent: multi-line tag_lines substitutions break
    indent uniformity and dedent produces a string that doesn't start with ---,
    causing _parse_frontmatter to return empty tags.
    """
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    return (
        f"---\n"
        f"tags:\n"
        f"{tag_lines}\n"
        f"created: 2026-04-01\n"
        f"last-modified: {last_modified}\n"
        f"---\n"
        f"# test-file\n\n"
        f"## Overview\n\n"
        f"{content}\n"
    )


# -- _sanitize_path --

class TestSanitizePath:
    def test_basic_name(self):
        assert _sanitize_path("my-file.md") == "my-file.md"

    def test_spaces_become_hyphens(self):
        assert _sanitize_path("my file") == "my-file.md"

    def test_underscores_become_hyphens(self):
        assert _sanitize_path("my_file") == "my-file.md"

    def test_uppercase_lowercased(self):
        assert _sanitize_path("MyFile") == "myfile.md"

    def test_special_chars_stripped(self):
        result = _sanitize_path("file!@#$.md")
        assert result == "file.md"

    def test_extension_added_if_missing(self):
        assert _sanitize_path("notefile").endswith(".md")

    def test_duplicate_extension_not_doubled(self):
        result = _sanitize_path("file.md")
        assert result.endswith(".md")
        assert not result.endswith(".md.md")

    def test_subdirectory_preserved(self):
        result = _sanitize_path("projects/api-notes")
        assert result == "projects/api-notes.md"

    def test_subdirectory_sanitized(self):
        result = _sanitize_path("My Projects/API Notes")
        assert result == "my-projects/api-notes.md"


# -- list_files --

class TestListFiles:
    def test_empty_dir(self, tmp_path):
        assert list_files(base_dir=tmp_path) == []

    def test_lists_md_files(self, tmp_path):
        _write_raw(tmp_path, "a.md", _structured_file(["foo"], "content"))
        results = list_files(base_dir=tmp_path)
        assert len(results) == 1
        assert results[0]["filename"] == "a.md"

    def test_filter_tags_includes_match(self, tmp_path):
        _write_raw(tmp_path, "match.md", _structured_file(["topic:A"], "relevant"))
        _write_raw(tmp_path, "skip.md", _structured_file(["topic:B"], "irrelevant"))
        results = list_files(filter_tags=["topic:A"], base_dir=tmp_path)
        filenames = [r["filename"] for r in results]
        assert "match.md" in filenames
        assert "skip.md" not in filenames

    def test_exclude_tags_removes_match(self, tmp_path):
        _write_raw(tmp_path, "keep.md", _structured_file(["visible"], "show me"))
        _write_raw(tmp_path, "hide.md", _structured_file(["private"], "hide me"))
        results = list_files(exclude_tags=["private"], base_dir=tmp_path)
        filenames = [r["filename"] for r in results]
        assert "keep.md" in filenames
        assert "hide.md" not in filenames

    def test_result_has_expected_fields(self, tmp_path):
        _write_raw(tmp_path, "file.md", _structured_file(["test"], "content"))
        result = list_files(base_dir=tmp_path)[0]
        for field in ("filename", "tags", "created", "last_modified", "path"):
            assert field in result

    def test_sorted_by_last_modified_desc(self, tmp_path):
        _write_raw(tmp_path, "old.md", _structured_file(["x"], "old", "2026-01-01"))
        _write_raw(tmp_path, "new.md", _structured_file(["x"], "new", "2026-04-07"))
        results = list_files(base_dir=tmp_path)
        assert results[0]["filename"] == "new.md"

    def test_recursive_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_raw(sub, "nested.md", _structured_file(["nested"], "nested content"))
        results = list_files(base_dir=tmp_path)
        filenames = [r["filename"] for r in results]
        assert "sub/nested.md" in filenames

    def test_file_without_frontmatter(self, tmp_path):
        _write_raw(tmp_path, "plain.md", "# Plain file\n\nNo frontmatter here.\n")
        results = list_files(base_dir=tmp_path)
        # Should be listed even without frontmatter
        assert any(r["filename"] == "plain.md" for r in results)


# -- read_file --

class TestReadFile:
    def test_returns_section_content(self, tmp_path):
        _write_raw(tmp_path, "notes.md", _structured_file(["test"], "Some knowledge here."))
        content = read_file("notes.md", base_dir=tmp_path)
        assert content is not None
        assert "Some knowledge here" in content

    def test_returns_none_for_missing_file(self, tmp_path):
        result = read_file("nonexistent.md", base_dir=tmp_path)
        assert result is None

    def test_path_traversal_returns_none(self, tmp_path):
        result = read_file("../../../etc/passwd", base_dir=tmp_path)
        assert result is None

    def test_plain_file_content_returned(self, tmp_path):
        _write_raw(tmp_path, "plain.md", "Just some plain text.\n")
        content = read_file("plain.md", base_dir=tmp_path)
        assert content is not None
        assert len(content) > 0

    def test_nested_file_readable(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_raw(sub, "nested.md", _structured_file(["x"], "nested value"))
        content = read_file("sub/nested.md", base_dir=tmp_path)
        assert content is not None
        assert "nested value" in content


# -- save_file --

class TestSaveFile:
    def test_creates_file(self, tmp_path):
        save_file("new-note.md", "## Section\nContent here.", ["test"], base_dir=tmp_path)
        assert (tmp_path / "new-note.md").exists()

    def test_content_readable_after_save(self, tmp_path):
        save_file("saved.md", "## Overview\nMy content.", ["tag"], base_dir=tmp_path)
        content = read_file("saved.md", base_dir=tmp_path)
        assert "My content" in content

    def test_tags_in_frontmatter(self, tmp_path):
        save_file("tagged.md", "## Sec\nBody.", ["alpha", "beta"], base_dir=tmp_path)
        raw = (tmp_path / "tagged.md").read_text()
        assert "alpha" in raw
        assert "beta" in raw

    def test_preserves_created_date_on_update(self, tmp_path):
        save_file("existing.md", "## V1\nFirst version.", ["x"], base_dir=tmp_path)
        raw_before = (tmp_path / "existing.md").read_text()
        import re
        match = re.search(r"created: (\S+)", raw_before)
        original_created = match.group(1) if match else None

        save_file("existing.md", "## V2\nSecond version.", ["x"], base_dir=tmp_path)
        raw_after = (tmp_path / "existing.md").read_text()
        assert original_created and original_created in raw_after

    def test_sanitizes_filename(self, tmp_path):
        path = save_file("My New Note", "## X\nContent.", ["y"], base_dir=tmp_path)
        assert "my-new-note.md" in path

    def test_creates_subdirectory(self, tmp_path):
        save_file("projects/api.md", "## Sec\nContent.", ["z"], base_dir=tmp_path)
        assert (tmp_path / "projects" / "api.md").exists()

    def test_rebuild_index_called(self, tmp_path):
        save_file("indexed.md", "## X\nContent.", ["tag"], base_dir=tmp_path)
        assert (tmp_path / "index.md").exists()

    def test_log_appended(self, tmp_path):
        save_file("logged.md", "## X\nContent.", ["tag"], base_dir=tmp_path)
        log = (tmp_path / "log.md").read_text()
        assert "logged.md" in log
        assert "save" in log


# -- get_file_metadata --

class TestGetFileMetadata:
    def test_returns_metadata(self, tmp_path):
        _write_raw(tmp_path, "meta.md", _structured_file(["alpha", "beta"], "x"))
        meta = get_file_metadata("meta.md", base_dir=tmp_path)
        assert meta is not None
        assert "alpha" in meta["tags"]
        assert "beta" in meta["tags"]

    def test_returns_none_for_missing(self, tmp_path):
        assert get_file_metadata("ghost.md", base_dir=tmp_path) is None

    def test_path_traversal_returns_none(self, tmp_path):
        assert get_file_metadata("../../passwd", base_dir=tmp_path) is None


# -- search_files --

class TestSearchFiles:
    def test_finds_matching_content(self, tmp_path):
        _write_raw(tmp_path, "doc.md", _structured_file(["x"], "The quick brown fox."))
        results = search_files("quick", base_dir=tmp_path)
        filenames = [r["filename"] for r in results]
        assert "doc.md" in filenames

    def test_case_insensitive(self, tmp_path):
        _write_raw(tmp_path, "doc.md", _structured_file(["x"], "UPPERCASE CONTENT HERE."))
        results = search_files("uppercase", base_dir=tmp_path)
        assert any(r["filename"] == "doc.md" for r in results)

    def test_no_match_returns_empty(self, tmp_path):
        _write_raw(tmp_path, "doc.md", _structured_file(["x"], "Nothing special."))
        assert search_files("zzznomatch", base_dir=tmp_path) == []

    def test_matching_lines_returned(self, tmp_path):
        _write_raw(tmp_path, "doc.md", _structured_file(["x"], "Find this phrase here."))
        results = search_files("Find this phrase", base_dir=tmp_path)
        assert len(results) > 0
        assert any("Find this phrase" in line for line in results[0]["matching_lines"])

    def test_capped_at_ten_results(self, tmp_path):
        for i in range(15):
            _write_raw(
                tmp_path, f"file{i}.md",
                _structured_file(["x"], f"keyword content for file {i}"),
            )
        results = search_files("keyword content", base_dir=tmp_path)
        assert len(results) <= 10


# -- rebuild_index --

class TestRebuildIndex:
    def test_creates_index_md(self, tmp_path):
        _write_raw(tmp_path, "a.md", _structured_file(["x"], "content"))
        rebuild_index(base_dir=tmp_path)
        assert (tmp_path / "index.md").exists()

    def test_index_contains_filename(self, tmp_path):
        _write_raw(tmp_path, "my-doc.md", _structured_file(["x"], "content"))
        rebuild_index(base_dir=tmp_path)
        index = (tmp_path / "index.md").read_text()
        assert "my-doc.md" in index

    def test_index_excludes_itself(self, tmp_path):
        _write_raw(tmp_path, "a.md", _structured_file(["x"], "content"))
        rebuild_index(base_dir=tmp_path)
        index = (tmp_path / "index.md").read_text()
        # The index should not list index.md or log.md as entries
        lines = [l for l in index.split("\n") if "| index.md |" in l]
        assert len(lines) == 0

    def test_index_excludes_log(self, tmp_path):
        _write_raw(tmp_path, "a.md", _structured_file(["x"], "content"))
        rebuild_index(base_dir=tmp_path)
        index = (tmp_path / "index.md").read_text()
        lines = [l for l in index.split("\n") if "| log.md |" in l]
        assert len(lines) == 0


# -- append_log --

class TestAppendLog:
    def test_creates_log_md(self, tmp_path):
        append_log("save", "test.md", base_dir=tmp_path)
        assert (tmp_path / "log.md").exists()

    def test_entry_contains_action_and_target(self, tmp_path):
        append_log("claude-md", "project-path", base_dir=tmp_path)
        text = (tmp_path / "log.md").read_text()
        assert "claude-md" in text
        assert "project-path" in text

    def test_multiple_entries_accumulate(self, tmp_path):
        append_log("save", "file1.md", base_dir=tmp_path)
        append_log("save", "file2.md", base_dir=tmp_path)
        text = (tmp_path / "log.md").read_text()
        assert "file1.md" in text
        assert "file2.md" in text

    def test_tags_included_when_provided(self, tmp_path):
        append_log("save", "file.md", tags=["alpha", "beta"], base_dir=tmp_path)
        text = (tmp_path / "log.md").read_text()
        assert "alpha" in text

    def test_detail_included_when_provided(self, tmp_path):
        append_log("claude-md", "path", detail="project: myapp", base_dir=tmp_path)
        text = (tmp_path / "log.md").read_text()
        assert "project: myapp" in text
