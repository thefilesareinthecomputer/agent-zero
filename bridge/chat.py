"""Chat and voice endpoints for the Agent Zero web UI.

SSE text chat (POST /chat) and WebSocket voice pipeline (WS /ws/audio).
Both share the agent lock and session management. Both store exchanges
to memory after agent completion.
"""

import asyncio
import json
import secrets
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import ollama as _ollama_client

from agent.config import API_TOKEN, FAST_TEXT_MODEL, MAIN_MODEL, OLLAMA_BASE_URL
from bridge.api_models import ChatRequest
from memory.memory_manager import store_exchange
from voice.pipeline import VoiceHandler
from voice.tts import stream_tts

router = APIRouter()

# -- Shared state (initialized by lifespan in api.py) --

_text_agent = None      # gemma4:26b -- heavy tasks, KB, file ops
_fast_agent = None      # gemma4:e4b -- quick chat
_voice_agent = None     # gemma4:e4b -- voice pipeline
_checkpointer = None    # shared AsyncSqliteSaver
_lock = asyncio.Lock()  # serializes agent access
_voice_ready = False    # set True after Whisper/VAD loaded


def init_agents(text_agent, fast_agent, voice_agent, checkpointer):
    """Called from api.py lifespan to set shared agent instances."""
    global _text_agent, _fast_agent, _voice_agent, _checkpointer
    _text_agent = text_agent
    _fast_agent = fast_agent
    _voice_agent = voice_agent
    _checkpointer = checkpointer


def set_voice_ready(ready: bool):
    global _voice_ready
    _voice_ready = ready


def is_voice_ready() -> bool:
    return _voice_ready


# -- Auth --

async def verify_token(authorization: str = Header(...)) -> str:
    """Validate bearer token. Same logic as api.py."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


# -- Agent streaming --

async def _stream_agent(agent, user_msg: str, session_id: str):
    """Async generator yielding (event_type, data) tuples from agent.astream().

    Reuses the chunk processing pattern from agent/run.py:143-158.
    """
    messages = [{"role": "user", "content": user_msg}]
    config = {"configurable": {"thread_id": session_id}}

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in agent.astream(
        {"messages": messages}, config=config, stream_mode="updates"
    ):
        for node_name, node_output in chunk.items():
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if msg.type == "ai":
                    meta = getattr(msg, "response_metadata", {}) or {}
                    if "prompt_eval_count" in meta:
                        prompt_tokens = meta["prompt_eval_count"]
                    if "eval_count" in meta:
                        completion_tokens = meta["eval_count"]
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield "tool_call", {
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                            }
                    elif msg.content:
                        full_response += msg.content
                        yield "token", {"text": msg.content}
                elif msg.type == "tool":
                    content = msg.content[:500] if msg.content else ""
                    yield "tool_result", {
                        "name": msg.name,
                        "content": content,
                    }

    if prompt_tokens:
        yield "usage", {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    yield "done", {"full_response": full_response}


# -- Model management --

_active_model: str | None = None
_model_names = {"fast": FAST_TEXT_MODEL, "heavy": MAIN_MODEL}


async def _ensure_model(agent_key: str):
    """Unload the other text model before inference so only one sits in VRAM."""
    global _active_model
    wanted = _model_names.get(agent_key, FAST_TEXT_MODEL)
    if _active_model and _active_model != wanted:
        try:
            client = _ollama_client.Client(host=OLLAMA_BASE_URL)
            await asyncio.to_thread(client.generate, model=_active_model, prompt="", keep_alive=0)
        except Exception:
            pass  # best-effort unload
    _active_model = wanted


# -- POST /chat (SSE text) --

@router.post("/chat")
async def chat(
    body: ChatRequest,
    _: str = Depends(verify_token),
):
    """Text chat with SSE streaming response. Returns 429 if agent is busy."""
    session_id = body.session_id or str(uuid.uuid4())
    agent_key = body.agent if body.agent in ("fast", "heavy") else "fast"
    agent = _text_agent if agent_key == "heavy" else _fast_agent

    if _lock.locked():
        raise HTTPException(status_code=429, detail="Agent busy")

    async def event_stream():
        yield _sse_event("session", {
            "session_id": session_id,
            "model": _model_names.get(agent_key, FAST_TEXT_MODEL),
        })

        await _ensure_model(agent_key)

        async with _lock:
            full_response = ""
            try:
                async for event_type, data in _stream_agent(
                    agent, body.message, session_id
                ):
                    if event_type == "done":
                        full_response = data["full_response"]
                    yield _sse_event(event_type, data)
            except Exception as e:
                yield _sse_event("error", {"message": str(e)})
                return

        # Store exchange to memory (outside lock)
        if full_response:
            try:
                store_exchange(body.message, full_response, session_id)
            except Exception:
                pass  # memory failure should not break chat

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-Id": session_id,
        },
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# -- WebSocket /ws/audio (voice) --

@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """Voice pipeline via WebSocket.

    Protocol:
    1. Client sends JSON auth message first
    2. Client sends PCM16 binary frames (512 samples = 1024 bytes)
    3. Server runs VAD -> wake word -> STT -> agent -> TTS
    4. Server sends JSON status/response messages + binary TTS audio

    Uses gemma4:e4b (voice agent). Queues one pending utterance if busy.
    """
    await websocket.accept()

    # -- Auth: first message must be JSON text with token --
    try:
        raw = await asyncio.wait_for(websocket.receive(), timeout=10.0)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await websocket.close(code=1008, reason="Auth timeout")
        return

    if raw.get("type") == "websocket.disconnect":
        return

    if "text" not in raw:
        # Binary frame instead of JSON auth -- reject cleanly
        await _ws_send(websocket, "auth_fail", {})
        await websocket.close(code=1008, reason="Expected JSON auth message")
        return

    first_msg = raw["text"]

    try:
        auth_data = json.loads(first_msg)
    except (json.JSONDecodeError, TypeError):
        await _ws_send(websocket, "auth_fail", {})
        await websocket.close(code=1008, reason="Invalid auth message")
        return

    if auth_data.get("type") != "auth":
        await _ws_send(websocket, "auth_fail", {})
        await websocket.close(code=1008, reason="First message must be auth")
        return

    token = auth_data.get("token", "")
    if not secrets.compare_digest(token, API_TOKEN):
        await _ws_send(websocket, "auth_fail", {})
        await websocket.close(code=1008, reason="Invalid token")
        return

    await _ws_send(websocket, "auth_ok", {})

    session_id = auth_data.get("session_id") or str(uuid.uuid4())
    handler = VoiceHandler()
    pending_query: str | None = None

    await _ws_send(websocket, "state", {"state": "listening"})

    try:
        while True:
            msg = await websocket.receive()

            # Text message (JSON control messages)
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = data.get("type")

                if msg_type == "tts_done":
                    handler.set_tts_playing(False)
                    await _ws_send(websocket, "state", {"state": "listening"})
                elif msg_type == "config":
                    pass  # future: wake word toggle
                elif msg_type == "stop":
                    pass  # future: cancel support

                continue

            # Binary message (audio frame)
            if "bytes" in msg:
                frame_bytes = msg["bytes"]

                # Track VAD state for UI
                prev_state = handler.vad_state
                try:
                    query = await handler.handle_audio_frame(frame_bytes)
                except Exception as exc:
                    print(f"[voice] ERROR in handle_audio_frame: {exc!r}")
                    import traceback; traceback.print_exc()
                    continue
                curr_state = handler.vad_state

                if curr_state != prev_state:
                    print(f"[voice] VAD state: {prev_state} -> {curr_state}")
                    if curr_state == "speaking":
                        await _ws_send(websocket, "state", {"state": "speaking"})

                if query is None:
                    continue

                print(f"[voice] query: {query!r}")

                # Got a transcribed query
                await _ws_send(websocket, "state", {"state": "processing"})
                handler.set_processing(True)
                await _ws_send(websocket, "transcription", {"text": query})

                # Try to acquire lock, queue if busy
                if _lock.locked():
                    pending_query = query  # depth-1 queue
                    continue

                await _process_voice_query(
                    websocket, handler, query, session_id
                )

                # Process pending query if one was queued
                if pending_query:
                    q = pending_query
                    pending_query = None
                    await _process_voice_query(
                        websocket, handler, q, session_id
                    )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        handler.reset()


async def _process_voice_query(
    websocket: WebSocket,
    handler: VoiceHandler,
    query: str,
    session_id: str,
):
    """Run agent on voice query, stream response, then TTS."""
    full_response = ""

    async with _lock:
        try:
            async for event_type, data in _stream_agent(
                _voice_agent, query, session_id
            ):
                await _ws_send(websocket, event_type, data)
                if event_type == "done":
                    full_response = data["full_response"]
        except Exception as e:
            await _ws_send(websocket, "error", {"detail": str(e)})
            handler.set_processing(False)
            await _ws_send(websocket, "state", {"state": "listening"})
            return

    # Store exchange to memory
    if full_response:
        try:
            store_exchange(query, full_response, session_id)
        except Exception:
            pass

    # TTS: synthesize and stream audio
    if full_response:
        handler.set_tts_playing(True)
        await _ws_send(websocket, "tts_start", {
            "sample_rate": 16000,
            "channels": 1,
        })

        try:
            async for pcm_chunk in stream_tts(full_response):
                await websocket.send_bytes(pcm_chunk)
        except Exception:
            pass

        await _ws_send(websocket, "tts_end", {})

    handler.set_processing(False)
    # Don't send "listening" here -- wait for client's tts_done message
    # (echo cancellation: mic stays suppressed until TTS playback finishes)
    if not handler._tts_playing:
        await _ws_send(websocket, "state", {"state": "listening"})


async def _ws_send(websocket: WebSocket, msg_type: str, data: dict):
    """Send a typed JSON message over WebSocket."""
    try:
        await websocket.send_text(json.dumps({"type": msg_type, **data}))
    except Exception:
        pass  # connection may be closing
