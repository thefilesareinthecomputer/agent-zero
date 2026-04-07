"""Tests for the Agent Zero HTTP API.

Uses FastAPI TestClient with temporary knowledge directories so tests
never touch real data.
"""

import os
import textwrap

import pytest
from fastapi.testclient import TestClient

# Set API_TOKEN before importing the app (config reads .env at import time)
os.environ["API_TOKEN"] = "test-token-that-is-at-least-32-characters-long"

from bridge.api import app  # noqa: E402

TOKEN = os.environ["API_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def kb_dirs(tmp_path):
    """Create isolated knowledge/ and knowledge_canon/ directories with
    test files. Patches the module-level paths so the API reads from them.
    """
    kb = tmp_path / "knowledge"
    kb.mkdir()
    canon = tmp_path / "knowledge_canon"
    canon.mkdir()

    # Regular file
    (kb / "test-file.md").write_text(textwrap.dedent("""\
        ---
        tags:
          - testing
          - demo
        created: 2026-04-07
        last-modified: 2026-04-07
        ---
        # test-file
        ## Overview
        This is a test knowledge file for API testing.
    """))

    # Private file -- must never appear in API responses
    (kb / "secret-plans.md").write_text(textwrap.dedent("""\
        ---
        tags:
          - private
          - plans
        created: 2026-04-07
        last-modified: 2026-04-07
        ---
        # secret-plans
        ## Details
        This file is private and should never be returned.
    """))

    # Canon file (read-only)
    (canon / "reference.md").write_text(textwrap.dedent("""\
        ---
        tags:
          - docs
        created: 2026-04-01
        last-modified: 2026-04-01
        ---
        # reference
        ## Content
        Canon reference material. Read-only.
    """))

    # Patch the paths used by the API and knowledge_store
    import bridge.api as api_mod
    import knowledge.knowledge_store as ks_mod

    original_kb_dir = ks_mod.KNOWLEDGE_DIR
    original_canon_dir = api_mod._CANON_DIR

    ks_mod.KNOWLEDGE_DIR = kb
    api_mod._CANON_DIR = canon

    yield {"knowledge": kb, "canon": canon}

    ks_mod.KNOWLEDGE_DIR = original_kb_dir
    api_mod._CANON_DIR = original_canon_dir


# -- Auth tests --

class TestAuth:
    def test_missing_token(self, client):
        resp = client.get("/knowledge")
        assert resp.status_code == 422  # missing header

    def test_wrong_token(self, client):
        resp = client.get("/knowledge", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_malformed_header(self, client):
        resp = client.get("/knowledge", headers={"Authorization": "Basic abc"})
        assert resp.status_code == 401

    def test_valid_token(self, client, kb_dirs):
        resp = client.get("/knowledge", headers=AUTH)
        assert resp.status_code == 200


# -- Health --

class TestHealth:
    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# -- List --

class TestList:
    def test_list_excludes_private(self, client, kb_dirs):
        resp = client.get("/knowledge", headers=AUTH)
        assert resp.status_code == 200
        filenames = [f["filename"] for f in resp.json()]
        assert "test-file.md" in filenames
        assert "secret-plans.md" not in filenames

    def test_list_includes_canon(self, client, kb_dirs):
        resp = client.get("/knowledge", headers=AUTH)
        filenames = [f["filename"] for f in resp.json()]
        assert "reference.md" in filenames

    def test_list_has_source_field(self, client, kb_dirs):
        resp = client.get("/knowledge", headers=AUTH)
        sources = {f["filename"]: f["source"] for f in resp.json()}
        assert sources.get("test-file.md") == "knowledge"
        assert sources.get("reference.md") == "canon"

    def test_list_strips_path(self, client, kb_dirs):
        resp = client.get("/knowledge", headers=AUTH)
        for f in resp.json():
            assert "path" not in f


# -- Read --

class TestRead:
    def test_read_file(self, client, kb_dirs):
        resp = client.get("/knowledge/test-file.md", headers=AUTH)
        assert resp.status_code == 200
        assert "test knowledge file" in resp.json()["content"]

    def test_read_canon(self, client, kb_dirs):
        resp = client.get("/knowledge/reference.md", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["source"] == "canon"

    def test_read_private_returns_404(self, client, kb_dirs):
        """Private files return 404, not 403 -- prevents enumeration."""
        resp = client.get("/knowledge/secret-plans.md", headers=AUTH)
        assert resp.status_code == 404

    def test_read_nonexistent(self, client, kb_dirs):
        resp = client.get("/knowledge/nonexistent.md", headers=AUTH)
        assert resp.status_code == 404

    def test_path_traversal(self, client, kb_dirs):
        resp = client.get("/knowledge/../../.env", headers=AUTH)
        assert resp.status_code in (400, 404)  # Starlette normalizes path before route


# -- Search --

class TestSearch:
    def test_search(self, client, kb_dirs):
        resp = client.get("/knowledge/search", params={"q": "test"}, headers=AUTH)
        assert resp.status_code == 200
        filenames = [r["filename"] for r in resp.json()]
        assert "test-file.md" in filenames

    def test_search_excludes_private(self, client, kb_dirs):
        resp = client.get("/knowledge/search", params={"q": "private"}, headers=AUTH)
        filenames = [r["filename"] for r in resp.json()]
        assert "secret-plans.md" not in filenames

    def test_search_empty_query(self, client, kb_dirs):
        resp = client.get("/knowledge/search", params={"q": ""}, headers=AUTH)
        assert resp.status_code == 422

    def test_search_no_query(self, client, kb_dirs):
        resp = client.get("/knowledge/search", headers=AUTH)
        assert resp.status_code == 422


# -- Save --

class TestSave:
    def test_save_new_file(self, client, kb_dirs):
        resp = client.post("/knowledge", json={
            "filename": "new-file.md",
            "content": "## Section\nNew content here.",
            "tags": ["test"],
        }, headers=AUTH)
        assert resp.status_code == 200
        assert "Saved" in resp.json()["message"]

        # Verify file exists
        assert (kb_dirs["knowledge"] / "new-file.md").exists()

    def test_save_canon_blocked(self, client, kb_dirs):
        resp = client.post("/knowledge", json={
            "filename": "reference.md",
            "content": "## Override\nAttempting to overwrite canon.",
            "tags": ["test"],
        }, headers=AUTH)
        assert resp.status_code == 403

    def test_save_path_traversal(self, client, kb_dirs):
        resp = client.post("/knowledge", json={
            "filename": "../../etc/passwd",
            "content": "exploit",
            "tags": [],
        }, headers=AUTH)
        assert resp.status_code == 400

    def test_index_rebuilt_after_save(self, client, kb_dirs):
        client.post("/knowledge", json={
            "filename": "indexed-file.md",
            "content": "## Topic\nSome content for the index.",
            "tags": ["indextest"],
        }, headers=AUTH)

        index = kb_dirs["knowledge"] / "index.md"
        assert index.exists()
        text = index.read_text()
        assert "indexed-file.md" in text

    def test_log_appended_after_save(self, client, kb_dirs):
        client.post("/knowledge", json={
            "filename": "logged-file.md",
            "content": "## Data\nContent to log.",
            "tags": ["logtest"],
        }, headers=AUTH)

        log = kb_dirs["knowledge"] / "log.md"
        assert log.exists()
        text = log.read_text()
        assert "logged-file.md" in text
        assert "save" in text


# -- CLAUDE.md --

class TestClaudeMd:
    def test_generate(self, client, kb_dirs):
        resp = client.post("/bridge/claude-md/generate", json={
            "project_name": "test-project",
        }, headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_name"] == "test-project"
        assert isinstance(data["content"], str)

    def test_write_with_project_name(self, client, kb_dirs, tmp_path):
        target = tmp_path / "output"
        resp = client.post("/bridge/claude-md/write", json={
            "project_path": str(target),
            "project_name": "test-project",
        }, headers=AUTH)
        assert resp.status_code == 200
        assert (target / "CLAUDE.md").exists()

    def test_write_with_content(self, client, kb_dirs, tmp_path):
        target = tmp_path / "output2"
        resp = client.post("/bridge/claude-md/write", json={
            "project_path": str(target),
            "content": "# Custom CLAUDE.md\nCaller-provided content.",
        }, headers=AUTH)
        assert resp.status_code == 200
        written = (target / "CLAUDE.md").read_text()
        assert "Caller-provided content" in written

    def test_write_both_rejected(self, client, kb_dirs, tmp_path):
        resp = client.post("/bridge/claude-md/write", json={
            "project_path": str(tmp_path),
            "project_name": "test",
            "content": "also content",
        }, headers=AUTH)
        assert resp.status_code == 422

    def test_write_neither_rejected(self, client, kb_dirs, tmp_path):
        resp = client.post("/bridge/claude-md/write", json={
            "project_path": str(tmp_path),
        }, headers=AUTH)
        assert resp.status_code == 422
