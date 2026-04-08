"""Tests for agent/kb_refine.py -- 26b draft refinement pipeline."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import kb_refine


@pytest.fixture(autouse=True)
def _isolate_fs(tmp_path, monkeypatch):
    """Point knowledge store at a temp dir so tests don't touch real files."""
    monkeypatch.setattr(
        "knowledge.knowledge_store.KNOWLEDGE_PATH", str(tmp_path)
    )


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response object."""
    resp = MagicMock()
    resp.content = content
    return resp


class TestBuildPrompt:
    def test_contains_all_fields(self):
        prompt = kb_refine._build_prompt("test.md", "draft text", "fix headings")
        assert "test.md" in prompt
        assert "draft text" in prompt
        assert "fix headings" in prompt

    def test_template_structure(self):
        prompt = kb_refine._build_prompt("f.md", "d", "i")
        assert "## File:" in prompt
        assert "## Draft:" in prompt
        assert "## Instructions:" in prompt


class TestRefineKbDraft:
    def test_calls_swap_and_saves(self):
        """Async variant swaps to 26b, invokes LLM, saves, swaps back."""
        mock_resp = _mock_llm_response("# Refined\n\nPolished content.")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        with (
            patch("bridge.models.swap_for_kb", new_callable=AsyncMock) as mock_swap,
            patch("bridge.models.swap_back_from_kb", new_callable=AsyncMock) as mock_swap_back,
            patch("agent.kb_refine.ChatOllama", return_value=mock_llm),
            patch.object(kb_refine, "_kb_save", return_value="/tmp/test.md") as mock_save,
            patch.object(kb_refine, "_index_file") as mock_index,
        ):
            result = asyncio.run(kb_refine.refine_kb_draft(
                filename="test.md",
                draft_content="rough draft",
                instructions="polish it",
                tags=["tag1"],
                project="proj",
            ))

            mock_swap.assert_called_once()
            mock_swap_back.assert_called_once()
            mock_save.assert_called_once_with(
                "test.md", "# Refined\n\nPolished content.", ["tag1"], project="proj"
            )
            mock_index.assert_called_once_with("test.md", source="knowledge")
            assert "Saved final version" in result

    def test_fallback_on_empty_response(self):
        """If 26b returns empty, saves the draft as-is."""
        mock_resp = _mock_llm_response("")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        with (
            patch("bridge.models.swap_for_kb", new_callable=AsyncMock),
            patch("bridge.models.swap_back_from_kb", new_callable=AsyncMock),
            patch("agent.kb_refine.ChatOllama", return_value=mock_llm),
            patch.object(kb_refine, "_kb_save", return_value="/tmp/test.md") as mock_save,
            patch.object(kb_refine, "_index_file"),
        ):
            result = asyncio.run(kb_refine.refine_kb_draft(
                filename="test.md",
                draft_content="my draft",
                instructions="fix it",
                tags=[],
            ))

            # Should save the original draft when LLM returns empty
            mock_save.assert_called_once_with("test.md", "my draft", [], project=None)
            assert "Saved final version" in result

    def test_fallback_on_llm_error(self):
        """If LLM call raises, saves draft as-is and reports error."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("connection refused"))

        with (
            patch("bridge.models.swap_for_kb", new_callable=AsyncMock),
            patch("bridge.models.swap_back_from_kb", new_callable=AsyncMock) as mock_swap_back,
            patch("agent.kb_refine.ChatOllama", return_value=mock_llm),
            patch.object(kb_refine, "_kb_save", return_value="/tmp/test.md") as mock_save,
            patch.object(kb_refine, "_index_file"),
        ):
            result = asyncio.run(kb_refine.refine_kb_draft(
                filename="test.md",
                draft_content="draft fallback",
                instructions="fix",
                tags=["t"],
            ))

            mock_save.assert_called_once_with("test.md", "draft fallback", ["t"], project=None)
            assert "refinement failed" in result
            # swap_back must be called even on error (finally block)
            mock_swap_back.assert_called_once()


class TestRefineKbDraftSync:
    def test_sync_calls_swap_and_saves(self):
        """Sync variant uses sync swap functions and llm.invoke."""
        mock_resp = _mock_llm_response("# Sync Refined\n\nDone.")
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=mock_resp)

        with (
            patch("bridge.models.sync_swap_for_kb") as mock_swap,
            patch("bridge.models.sync_swap_back_from_kb") as mock_swap_back,
            patch("agent.kb_refine.ChatOllama", return_value=mock_llm),
            patch.object(kb_refine, "_kb_save", return_value="/tmp/test.md") as mock_save,
            patch.object(kb_refine, "_index_file"),
        ):
            result = kb_refine.refine_kb_draft_sync(
                filename="test.md",
                draft_content="sync draft",
                instructions="rewrite",
                tags=["a", "b"],
                project="myproj",
            )

            mock_swap.assert_called_once()
            mock_swap_back.assert_called_once()
            mock_save.assert_called_once_with(
                "test.md", "# Sync Refined\n\nDone.", ["a", "b"], project="myproj"
            )
            assert "Saved final version" in result

    def test_sync_fallback_on_error(self):
        """Sync variant saves draft on LLM failure, always swaps back."""
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(side_effect=RuntimeError("timeout"))

        with (
            patch("bridge.models.sync_swap_for_kb"),
            patch("bridge.models.sync_swap_back_from_kb") as mock_swap_back,
            patch("agent.kb_refine.ChatOllama", return_value=mock_llm),
            patch.object(kb_refine, "_kb_save", return_value="/tmp/test.md") as mock_save,
            patch.object(kb_refine, "_index_file"),
        ):
            result = kb_refine.refine_kb_draft_sync(
                filename="test.md",
                draft_content="fallback draft",
                instructions="fix",
                tags=[],
            )

            mock_save.assert_called_once_with("test.md", "fallback draft", [], project=None)
            assert "refinement failed" in result
            mock_swap_back.assert_called_once()


class TestSaveResult:
    def test_saves_and_indexes(self):
        """_save_result calls save_file and index_file."""
        with (
            patch.object(kb_refine, "_kb_save", return_value="/tmp/out.md") as mock_save,
            patch.object(kb_refine, "_index_file") as mock_index,
        ):
            path = kb_refine._save_result("out.md", "content", ["t1"], "proj")
            mock_save.assert_called_once_with("out.md", "content", ["t1"], project="proj")
            mock_index.assert_called_once_with("out.md", source="knowledge")
            assert path == "/tmp/out.md"

    def test_index_failure_does_not_raise(self):
        """If index_file fails, _save_result still returns the path."""
        with (
            patch.object(kb_refine, "_kb_save", return_value="/tmp/out.md"),
            patch.object(kb_refine, "_index_file", side_effect=RuntimeError("chroma down")),
        ):
            path = kb_refine._save_result("out.md", "content", [], None)
            assert path == "/tmp/out.md"
