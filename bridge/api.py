"""Agent Zero HTTP API -- privacy-preserving access to the knowledge base.

FastAPI server on localhost (127.0.0.1). Bearer token auth. Exposes
knowledge base CRUD and CLAUDE.md generation. Filesystem only -- no
vector store, no agent execution, no sessions.

Inspirations: Karpathy's LLM Wiki (index/log patterns), MemPalace (MCP).
"""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from agent.config import API_TOKEN, KNOWLEDGE_CANON_PATH
from bridge.api_models import (
    ClaudeMdGenerateRequest,
    ClaudeMdGenerateResponse,
    ClaudeMdWriteRequest,
    ClaudeMdWriteResponse,
    FileContent,
    FileInfo,
    HealthResponse,
    SaveRequest,
    SaveResponse,
    SearchResult,
)
from bridge.claude_md import generate_claude_md, write_claude_md
from knowledge.knowledge_store import (
    append_log,
    get_file_metadata,
    list_files,
    read_file,
    save_file,
    search_files,
)

VERSION = "0.1.0"
_CANON_DIR = Path(KNOWLEDGE_CANON_PATH)
_PRIVACY_EXCLUDE = ["private", "secret"]
_MAX_BODY_BYTES = 1_048_576  # 1 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Validates config on startup."""
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN not set in .env -- refusing to start")
    if len(API_TOKEN) < 32:
        raise RuntimeError("API_TOKEN must be at least 32 characters")
    yield


app = FastAPI(
    title="Agent Zero API",
    version=VERSION,
    lifespan=lifespan,
)


# -- Auth --

async def verify_token(authorization: str = Header(...)) -> str:
    """Validate bearer token. Timing-safe comparison."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


# -- Helpers --

def _is_canon_file(filename: str) -> bool:
    """Check if a filename exists in the canon directory."""
    return (_CANON_DIR / filename).exists()


def _merged_list(**kwargs) -> list[dict]:
    """List files from both knowledge/ and knowledge_canon/, merged.

    Privacy-filtered, absolute paths stripped, source-tagged.
    """
    kwargs.setdefault("exclude_tags", _PRIVACY_EXCLUDE)

    kb_files = list_files(**kwargs)
    for f in kb_files:
        f["source"] = "knowledge"

    canon_files = list_files(base_dir=_CANON_DIR, **kwargs)
    for f in canon_files:
        f["source"] = "canon"

    merged = kb_files + canon_files
    merged.sort(key=lambda r: r["last_modified"], reverse=True)

    for f in merged:
        f.pop("path", None)

    return merged


def _private_filenames() -> set[str]:
    """Build the set of filenames tagged private or secret across both dirs."""
    private_kb = list_files(filter_tags=_PRIVACY_EXCLUDE)
    private_canon = list_files(filter_tags=_PRIVACY_EXCLUDE, base_dir=_CANON_DIR)
    return {f["filename"] for f in private_kb + private_canon}


def _is_path_traversal(filename: str) -> bool:
    """Defense-in-depth check for path traversal attempts."""
    return ".." in filename.split("/")


# -- Routes --

@app.get("/health", response_model=HealthResponse)
async def health():
    """Server status check. No auth required."""
    return HealthResponse(status="ok", version=VERSION)


@app.get("/knowledge", response_model=list[FileInfo])
async def list_knowledge(
    _: str = Depends(verify_token),
    filter_tags: str | None = Query(None, description="Comma-separated tags to include"),
):
    """List all knowledge files (merged knowledge/ + canon/, privacy-filtered)."""
    kwargs = {}
    if filter_tags:
        kwargs["filter_tags"] = [t.strip() for t in filter_tags.split(",") if t.strip()]

    files = _merged_list(**kwargs)
    return [FileInfo(**f) for f in files]


@app.get("/knowledge/search", response_model=list[SearchResult])
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Search query"),
    _: str = Depends(verify_token),
):
    """Search knowledge files for a keyword or phrase (merged, privacy-filtered)."""
    private = _private_filenames()

    kb_results = search_files(q)
    canon_results = search_files(q, base_dir=_CANON_DIR)

    results = []
    for r in kb_results:
        if r["filename"] not in private:
            results.append(SearchResult(
                filename=r["filename"],
                matching_lines=r["matching_lines"],
                source="knowledge",
            ))
    for r in canon_results:
        if r["filename"] not in private:
            results.append(SearchResult(
                filename=r["filename"],
                matching_lines=r["matching_lines"],
                source="canon",
            ))

    return results


@app.get("/knowledge/{filename:path}", response_model=FileContent)
async def read_knowledge(
    filename: str,
    _: str = Depends(verify_token),
):
    """Read a knowledge file by name. Tries knowledge/ first, then canon/."""
    if _is_path_traversal(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Privacy check -- private files return 404, not 403
    meta = get_file_metadata(filename)
    source = "knowledge"
    if meta is None:
        meta = get_file_metadata(filename, base_dir=_CANON_DIR)
        source = "canon"
    if meta is None:
        raise HTTPException(status_code=404, detail="File not found")

    tags = meta.get("tags", [])
    if any(t in tags for t in _PRIVACY_EXCLUDE):
        raise HTTPException(status_code=404, detail="File not found")

    # Read content
    content = read_file(filename) if source == "knowledge" else None
    if content is None and source == "knowledge":
        content = read_file(filename, base_dir=_CANON_DIR)
        source = "canon"
    if content is None:
        content = read_file(filename, base_dir=_CANON_DIR)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")

    return FileContent(filename=filename, content=content, source=source)


@app.post("/knowledge", response_model=SaveResponse)
async def save_knowledge(
    req: SaveRequest,
    _: str = Depends(verify_token),
):
    """Create or update a knowledge file. Blocks writes to canon files."""
    if _is_path_traversal(req.filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if _is_canon_file(req.filename):
        raise HTTPException(status_code=403, detail="Cannot save: canon file (read-only)")

    if len(req.content.encode("utf-8")) > _MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 1 MB limit")

    path = save_file(req.filename, req.content, req.tags)
    # save_file already calls rebuild_index and append_log internally
    sanitized = Path(path).name
    return SaveResponse(filename=sanitized, message="Saved")


@app.post("/bridge/claude-md/generate", response_model=ClaudeMdGenerateResponse)
async def generate_claude_md_endpoint(
    req: ClaudeMdGenerateRequest,
    _: str = Depends(verify_token),
):
    """Generate CLAUDE.md content for a project. Returns content as object
    for the caller to inspect or modify before writing."""
    content = generate_claude_md(req.project_name)
    return ClaudeMdGenerateResponse(
        project_name=req.project_name,
        content=content,
    )


@app.post("/bridge/claude-md/write", response_model=ClaudeMdWriteResponse)
async def write_claude_md_endpoint(
    req: ClaudeMdWriteRequest,
    _: str = Depends(verify_token),
):
    """Write CLAUDE.md to a project directory.

    Two modes:
    - project_name only: auto-generates from knowledge base
    - content only: writes the provided content directly (for iteration workflow)
    """
    target_dir = Path(req.project_path).expanduser()

    if req.project_name:
        # Auto-generate from knowledge base
        path = write_claude_md(req.project_path, req.project_name)
        append_log("claude-md", req.project_path, detail=f"project: {req.project_name}")
        return ClaudeMdWriteResponse(message=f"Generated and wrote CLAUDE.md to {path}")

    # Write caller-provided content directly
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "CLAUDE.md"
    target.write_text(req.content, encoding="utf-8")
    append_log("claude-md", req.project_path, detail="caller-provided content")
    return ClaudeMdWriteResponse(message=f"Wrote CLAUDE.md to {target}")
