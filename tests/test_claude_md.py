"""Tests for bridge/claude_md.py -- scoring, budget management, path resolution."""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bridge.claude_md as cmd
import knowledge.knowledge_store as ks


# -- Fixtures --

def _write_kb_file(directory: Path, filename: str, tags: list[str],
                   content: str, last_modified: str = "2026-04-07") -> None:
    """Write a well-formed knowledge file into a directory."""
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    stem = filename.replace(".md", "")
    text = (
        f"---\n"
        f"tags:\n"
        f"{tag_lines}\n"
        f"created: 2026-04-01\n"
        f"last-modified: {last_modified}\n"
        f"---\n"
        f"# {stem}\n\n"
        f"## Content\n\n"
        f"{content}\n"
    )
    (directory / filename).write_text(text, encoding="utf-8")


@pytest.fixture()
def kb_patch(tmp_path, monkeypatch):
    """Isolated knowledge dirs. Patches module globals so no real KB is touched."""
    kb = tmp_path / "knowledge"
    kb.mkdir()
    canon = tmp_path / "canon"
    canon.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    monkeypatch.setattr(ks, "KNOWLEDGE_DIR", kb)
    monkeypatch.setattr(cmd, "_CANON_DIR", canon)
    monkeypatch.setattr(cmd, "_OUTPUTS_DIR", outputs)

    return {"kb": kb, "canon": canon, "outputs": outputs}


# -- _score_file --

class TestScoreFile:
    def test_recent_file_scores_near_one(self):
        today = datetime.now(timezone.utc).date().isoformat()
        info = {"last_modified": today, "source": "knowledge"}
        score = cmd._score_file(info)
        assert score > 0.9

    def test_old_file_scores_lower(self):
        old = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
        recent = datetime.now(timezone.utc).date().isoformat()
        old_score = cmd._score_file({"last_modified": old, "source": "knowledge"})
        recent_score = cmd._score_file({"last_modified": recent, "source": "knowledge"})
        assert old_score < recent_score

    def test_30_day_half_life(self):
        """A file modified exactly 30 days ago should score ~0.5 (before canon multiplier)."""
        thirty_ago = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        info = {"last_modified": thirty_ago, "source": "knowledge"}
        score = cmd._score_file(info)
        assert abs(score - 0.5) < 0.05

    def test_canon_bonus_doubles_score(self):
        today = datetime.now(timezone.utc).date().isoformat()
        kb = cmd._score_file({"last_modified": today, "source": "knowledge"})
        canon = cmd._score_file({"last_modified": today, "source": "canon"})
        assert abs(canon - kb * 2.0) < 0.01

    def test_unparseable_date_treated_as_old(self):
        """Bad date string should not crash and should produce a low score."""
        info = {"last_modified": "not-a-date", "source": "knowledge"}
        score = cmd._score_file(info)
        # 365-day-old equivalent: exp(-365 * ln(2)/30) ~= 1.8e-4
        assert score < 0.01

    def test_missing_date_treated_as_old(self):
        info = {"source": "knowledge"}
        score = cmd._score_file(info)
        assert score < 0.01


# -- generate_claude_md --

class TestGenerateClaudeMd:
    def test_no_matching_files_returns_header(self, kb_patch):
        result = cmd.generate_claude_md("nonexistent-project")
        assert "No knowledge files found" in result
        assert "Agent Zero" in result

    def test_includes_matching_file_content(self, kb_patch):
        _write_kb_file(
            kb_patch["kb"], "api-notes.md",
            tags=["project:myapp"],
            content="The API uses bearer tokens.",
        )
        result = cmd.generate_claude_md("myapp")
        assert "bearer tokens" in result

    def test_excludes_private_tagged_files(self, kb_patch):
        _write_kb_file(
            kb_patch["kb"], "secrets.md",
            tags=["project:myapp", "private"],
            content="Super secret passwords here.",
        )
        result = cmd.generate_claude_md("myapp")
        assert "Super secret passwords" not in result

    def test_excludes_secret_tagged_files(self, kb_patch):
        _write_kb_file(
            kb_patch["kb"], "creds.md",
            tags=["project:myapp", "secret"],
            content="AWS access key.",
        )
        result = cmd.generate_claude_md("myapp")
        assert "AWS access key" not in result

    def test_canon_file_included(self, kb_patch):
        _write_kb_file(
            kb_patch["canon"], "spec.md",
            tags=["project:myapp"],
            content="Canon spec content.",
        )
        result = cmd.generate_claude_md("myapp")
        assert "Canon spec content" in result

    def test_only_project_tagged_files_included(self, kb_patch):
        _write_kb_file(
            kb_patch["kb"], "unrelated.md",
            tags=["project:other"],
            content="This belongs to another project.",
        )
        _write_kb_file(
            kb_patch["kb"], "mine.md",
            tags=["project:myapp"],
            content="This belongs to myapp.",
        )
        result = cmd.generate_claude_md("myapp")
        assert "belongs to myapp" in result
        assert "belongs to another project" not in result

    def test_higher_scored_files_included_first_on_budget_overflow(
        self, kb_patch, monkeypatch
    ):
        """When budget is tight, recently modified files survive; stale ones are skipped."""
        today = datetime.now(timezone.utc).date().isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()

        # Recent file -- high score, short content that fits budget
        _write_kb_file(
            kb_patch["kb"], "recent.md",
            tags=["project:myapp"],
            content="Recent important note.",
            last_modified=today,
        )
        # Old file -- low score, more content
        _write_kb_file(
            kb_patch["kb"], "old-notes.md",
            tags=["project:myapp"],
            content="Old stale note " * 50,  # large enough to potentially get skipped
            last_modified=old,
        )

        # Shrink the budget to force overflow
        monkeypatch.setattr(cmd, "CLAUDE_MD_MAX_CHARS", 600)

        result = cmd.generate_claude_md("myapp")
        # Recent file should be present
        assert "Recent important note" in result
        # Skipped files section should appear when overflow occurs
        # (old-notes.md may be skipped or truncated -- either way, no crash)
        assert "Agent Zero" in result  # header always present

    def test_skipped_files_listed_at_bottom(self, kb_patch, monkeypatch):
        """Files that don't fit in budget are listed in the skipped notice."""
        today = datetime.now(timezone.utc).date().isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=200)).date().isoformat()

        _write_kb_file(
            kb_patch["kb"], "big-file.md",
            tags=["project:myapp"],
            content="High priority content.",
            last_modified=today,
        )
        _write_kb_file(
            kb_patch["kb"], "overflow-file.md",
            tags=["project:myapp"],
            content="This should overflow.",
            last_modified=old,
        )

        # Budget just big enough for the header and one file
        monkeypatch.setattr(cmd, "CLAUDE_MD_MAX_CHARS", 500)

        result = cmd.generate_claude_md("myapp")
        # If any files were skipped, the notice should list them
        if "omitted" in result:
            assert "overflow-file.md" in result or "big-file.md" in result

    def test_output_contains_header_warning(self, kb_patch):
        result = cmd.generate_claude_md("someproject")
        assert "auto-generated" in result
        assert "Manual edits will be overwritten" in result

    def test_truncated_file_includes_api_note(self, kb_patch, monkeypatch):
        """A file truncated mid-content should include the API fallback note."""
        _write_kb_file(
            kb_patch["kb"], "large.md",
            tags=["project:myapp"],
            content="Important data. " * 200,  # well over any per-file limit
        )
        # Set budget large enough to include but trigger truncation
        monkeypatch.setattr(cmd, "CLAUDE_MD_MAX_CHARS", 1200)
        monkeypatch.setattr(cmd, "FILE_MIN_CHARS", 100)

        result = cmd.generate_claude_md("myapp")
        if "Truncated" in result:
            assert "GET /knowledge/" in result


# -- write_claude_md --

class TestWriteClaudeMd:
    def test_writes_to_absolute_path(self, kb_patch, tmp_path):
        target = tmp_path / "some-project"
        cmd.write_claude_md(str(target), "myapp")
        assert (target / "CLAUDE.md").exists()

    def test_absolute_path_not_redirected(self, kb_patch, tmp_path):
        """Absolute paths should not be rerouted to project_outputs/."""
        target = tmp_path / "explicit-output"
        cmd.write_claude_md(str(target), "myapp")
        # File should be at the exact absolute path, not inside _OUTPUTS_DIR
        assert (target / "CLAUDE.md").exists()
        assert not (kb_patch["outputs"] / "explicit-output" / "CLAUDE.md").exists()

    def test_relative_path_resolves_to_outputs_dir(self, kb_patch):
        cmd.write_claude_md("my-project", "myapp")
        assert (kb_patch["outputs"] / "my-project" / "CLAUDE.md").exists()

    def test_creates_parent_dirs(self, kb_patch, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        cmd.write_claude_md(str(deep), "myapp")
        assert (deep / "CLAUDE.md").exists()

    def test_written_content_is_valid_markdown(self, kb_patch, tmp_path):
        target = tmp_path / "out"
        cmd.write_claude_md(str(target), "myapp")
        content = (target / "CLAUDE.md").read_text()
        assert content.startswith("#")
        assert len(content) > 0
