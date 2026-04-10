"""KB draft refinement pipeline -- 26b edits drafts from e4b.

The e4b agent writes a rough draft and instructions. This module loads
the 26b model, runs a single-turn refinement pass, saves the final
version, and unloads 26b. Both async (web) and sync (CLI) variants.
"""

import logging

from agent.config import EFFECTIVE_KB_REFINE_MODEL, NUM_CTX
from agent.llm import make_chat_ollama
from knowledge.kb_index import index_file as _index_file
from knowledge.knowledge_store import save_file as _kb_save

log = logging.getLogger(__name__)

_REFINE_PROMPT = """\
You are a knowledge base editor. Produce the final polished version of
this document. Use clear markdown structure with ## headings. Be thorough
and precise. Do not include preamble or meta-commentary -- output only
the document content.

## File: {filename}

## Draft:
{draft_content}

## Instructions:
{instructions}

Write the complete final document. Include all sections.\
"""


def _build_prompt(filename: str, draft_content: str, instructions: str) -> str:
    return _REFINE_PROMPT.format(
        filename=filename,
        draft_content=draft_content,
        instructions=instructions,
    )


def _save_result(
    filename: str,
    content: str,
    tags: list[str],
    project: str | None,
) -> str:
    """Save the refined content and re-index."""
    path = _kb_save(filename, content, tags, project=project)
    try:
        _index_file(filename, source="knowledge")
    except Exception:
        pass
    return path


async def refine_kb_draft(
    filename: str,
    draft_content: str,
    instructions: str,
    tags: list[str],
    project: str | None = None,
) -> str:
    """Async refinement: swap to 26b, refine draft, save, swap back.

    Returns a summary string describing what was saved.
    """
    from bridge.models import swap_back_from_kb, swap_for_kb

    await swap_for_kb()

    try:
        llm = make_chat_ollama(model=EFFECTIVE_KB_REFINE_MODEL, num_ctx=NUM_CTX, num_predict=4096)
        prompt = _build_prompt(filename, draft_content, instructions)
        response = await llm.ainvoke(prompt)
        refined_content = response.content.strip()

        if not refined_content:
            log.warning("KB refine: 26b returned empty response for %s", filename)
            refined_content = draft_content

        path = _save_result(filename, refined_content, tags, project)
        log.info("KB refine: saved %s", path)
        return f"[26B refinement] Saved final version: {filename}"

    except Exception as e:
        log.error("KB refine failed for %s: %s", filename, e)
        # Fall back: save the draft as-is
        path = _save_result(filename, draft_content, tags, project)
        return f"[26B refinement failed: {e}] Saved draft as-is: {filename}"

    finally:
        await swap_back_from_kb()


def refine_kb_draft_sync(
    filename: str,
    draft_content: str,
    instructions: str,
    tags: list[str],
    project: str | None = None,
) -> str:
    """Sync refinement for CLI context: swap to 26b, refine, save, swap back."""
    from bridge.models import sync_swap_back_from_kb, sync_swap_for_kb

    sync_swap_for_kb()

    try:
        llm = make_chat_ollama(model=EFFECTIVE_KB_REFINE_MODEL, num_ctx=NUM_CTX, num_predict=4096)
        prompt = _build_prompt(filename, draft_content, instructions)
        response = llm.invoke(prompt)
        refined_content = response.content.strip()

        if not refined_content:
            log.warning("KB refine: 26b returned empty response for %s", filename)
            refined_content = draft_content

        path = _save_result(filename, refined_content, tags, project)
        log.info("KB refine (sync): saved %s", path)
        return f"[26B refinement] Saved final version: {filename}"

    except Exception as e:
        log.error("KB refine (sync) failed for %s: %s", filename, e)
        path = _save_result(filename, draft_content, tags, project)
        return f"[26B refinement failed: {e}] Saved draft as-is: {filename}"

    finally:
        sync_swap_back_from_kb()
