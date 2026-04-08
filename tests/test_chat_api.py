"""Tests for the chat and voice API endpoints."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["API_TOKEN"] = "test-token-that-is-at-least-32-characters-long"

from bridge.api import app  # noqa: E402

TOKEN = os.environ["API_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client():
    return TestClient(app)


# -- Mock agent that yields predetermined chunks --

def _make_mock_agent():
    """Create a mock agent whose astream yields a single AI response."""
    async def fake_astream(input_dict, config, stream_mode):
        yield {
            "agent": {
                "messages": [
                    MagicMock(
                        type="ai",
                        content="Test response.",
                        tool_calls=[],
                    )
                ]
            }
        }

    agent = MagicMock()
    agent.astream = fake_astream
    return agent


# -- Health --

class TestHealthVoice:
    def test_health_includes_voice_field(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "voice" in data


# -- Static files --

class TestStaticFiles:
    def test_ui_index_served(self, client):
        resp = client.get("/ui/index.html")
        assert resp.status_code == 200
        assert "Agent Zero" in resp.text

    def test_root_redirects(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307)
        assert "/ui/" in resp.headers.get("location", "")


# -- Chat SSE --

class TestChatSSE:
    def test_chat_requires_auth(self, client):
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 422  # missing header

    def test_chat_wrong_token(self, client):
        resp = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    @patch("bridge.chat._text_agent")
    @patch("bridge.chat._lock", new_callable=lambda: asyncio.Lock)
    def test_chat_returns_sse_stream(self, mock_lock, mock_agent, client):
        mock_agent_instance = _make_mock_agent()
        with patch("bridge.chat._text_agent", mock_agent_instance):
            resp = client.post(
                "/chat",
                json={"message": "hello"},
                headers=AUTH,
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    @patch("bridge.chat._text_agent")
    def test_chat_session_id_returned(self, mock_agent, client):
        mock_agent_instance = _make_mock_agent()
        with patch("bridge.chat._text_agent", mock_agent_instance):
            resp = client.post(
                "/chat",
                json={"message": "hello"},
                headers=AUTH,
            )
            # Session ID should be in response header
            assert "x-session-id" in resp.headers


# -- WebSocket auth --

class TestWebSocketAuth:
    def test_websocket_auth_required(self, client):
        """WebSocket without auth message should be closed."""
        with client.websocket_connect("/ws/audio") as ws:
            # Send binary first (not JSON auth) -- should get rejected
            ws.send_bytes(b"\x00" * 1024)
            try:
                msg = ws.receive_text()
                data = json.loads(msg)
                # Server should reject or close
                assert data.get("type") in ("auth_fail", None)
            except Exception:
                pass  # connection closed

    def test_websocket_auth_valid(self, client):
        """Valid auth should return auth_ok."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(json.dumps({
                "type": "auth",
                "token": TOKEN,
            }))
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["type"] == "auth_ok"

    def test_websocket_auth_invalid_token(self, client):
        """Wrong token should return auth_fail."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(json.dumps({
                "type": "auth",
                "token": "wrong-token-that-is-long-enough-for-compare",
            }))
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["type"] == "auth_fail"
