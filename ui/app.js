/**
 * Agent Zero -- Web UI
 *
 * Orchestrates auth, text chat (SSE), voice chat (WebSocket),
 * and audio playback. No build tools, no frameworks.
 */

// -- DOM refs --
const tokenInput = document.getElementById('token-input');
const connectBtn = document.getElementById('connect-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const statusDot = document.getElementById('status-dot');
const inputBar = document.getElementById('input-bar');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const messagesEl = document.getElementById('messages');

// -- State --
let token = localStorage.getItem('az_token') || '';
let sessionId = '';  // fresh session on every page load
let connected = false;
let busy = false;
let micOn = false;
let audioCtx = null;
let ws = null;
let workletNode = null;
let mediaStream = null;
let ctxSize = 16384;

// -- Init --
document.addEventListener('DOMContentLoaded', () => {
  if (token) {
    tokenInput.value = '****';
    tryConnect();
  }

  connectBtn.addEventListener('click', onConnect);
  disconnectBtn.addEventListener('click', onDisconnect);
  sendBtn.addEventListener('click', onSend);
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
  });
  micBtn.addEventListener('click', toggleMic);
});

// -- Auth --

async function onConnect() {
  const val = tokenInput.value.trim();
  if (!val || val === '****') return;
  token = val;
  localStorage.setItem('az_token', token);
  await tryConnect();
}

async function tryConnect() {
  try {
    const resp = await fetch('/health');
    if (!resp.ok) throw new Error('Server unreachable');
    const data = await resp.json();
    if (data.ctx_size) ctxSize = data.ctx_size;
    setConnected(true);
    updateCtxBar(0);
  } catch {
    setConnected(false);
  }
}

function onDisconnect() {
  token = '';
  localStorage.removeItem('az_token');
  sessionStorage.removeItem('az_session');
  sessionId = '';
  if (micOn) toggleMic();
  setConnected(false);
  tokenInput.value = '';
  messagesEl.innerHTML = '';
}

function setConnected(state) {
  connected = state;
  const ctxBar = document.getElementById('ctx-bar');
  if (state) {
    tokenInput.classList.add('hidden');
    connectBtn.classList.add('hidden');
    disconnectBtn.classList.remove('hidden');
    inputBar.classList.remove('hidden');
    ctxBar.classList.remove('hidden');
    statusDot.className = 'dot dot-connected';
    statusDot.title = 'Connected';
  } else {
    tokenInput.classList.remove('hidden');
    connectBtn.classList.remove('hidden');
    disconnectBtn.classList.add('hidden');
    inputBar.classList.add('hidden');
    ctxBar.classList.add('hidden');
    statusDot.className = 'dot dot-disconnected';
    statusDot.title = 'Disconnected';
    updateCtxBar(0);
  }
}

function updateCtxBar(promptTokens) {
  const fill = document.getElementById('ctx-fill');
  const label = document.getElementById('ctx-label');
  const bar = document.getElementById('ctx-bar');
  const pct = ctxSize > 0 ? (promptTokens / ctxSize) * 100 : 0;

  fill.style.width = `${Math.min(pct, 100)}%`;
  label.textContent = `${promptTokens.toLocaleString()} / ${ctxSize.toLocaleString()} ctx`;

  bar.classList.remove('ctx-warn', 'ctx-danger');
  if (pct >= 90) bar.classList.add('ctx-danger');
  else if (pct >= 75) bar.classList.add('ctx-warn');
}

function setBusy(state) {
  busy = state;
  textInput.disabled = state;
  sendBtn.disabled = state;
  statusDot.className = state ? 'dot dot-busy' : 'dot dot-connected';
  statusDot.title = state ? 'Agent busy' : 'Connected';
}

// -- Text Chat (SSE) --

async function onSend() {
  const msg = textInput.value.trim();
  if (!msg || busy || !connected) return;
  textInput.value = '';

  appendMessage('user', msg);
  setBusy(true);

  const agentMsg = createAgentMessage();

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: msg,
        session_id: sessionId || null,
        agent: document.getElementById('agent-select').value,
      }),
    });

    if (resp.status === 429) {
      agentMsg.textContent = '[Agent busy -- try again shortly]';
      setBusy(false);
      return;
    }

    if (!resp.ok) {
      agentMsg.textContent = `[Error: ${resp.status}]`;
      setBusy(false);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ') && eventType) {
          const data = JSON.parse(line.slice(6));
          handleSSEEvent(eventType, data, agentMsg);
          eventType = null;
        }
      }
    }
  } catch (e) {
    agentMsg.textContent = `[Connection error: ${e.message}]`;
  }

  setBusy(false);
}

function handleSSEEvent(type, data, agentMsg) {
  switch (type) {
    case 'session':
      sessionId = data.session_id;
      if (data.model) agentMsg.querySelector('.message-label').textContent = `Agent Zero [${data.model}]`;
      break;
    case 'tool_call':
      appendTool(agentMsg, `[calling ${data.name}...]`);
      break;
    case 'tool_result':
      appendTool(agentMsg, `[${data.name}] ${data.content}`);
      break;
    case 'token':
      appendToken(agentMsg, data.text);
      break;
    case 'usage':
      updateCtxBar(data.prompt_tokens);
      break;
    case 'done':
      break;
    case 'error':
      appendToken(agentMsg, `\n[Error: ${data.message}]`);
      break;
  }
}

// -- Voice (WebSocket) --

async function toggleMic() {
  if (micOn) {
    stopMic();
  } else {
    await startMic();
  }
}

async function startMic() {
  if (!connected) return;

  try {
    // AudioContext at 16kHz -- browser handles resampling
    audioCtx = new AudioContext({ sampleRate: 16000 });

    // Don't constrain sampleRate in getUserMedia -- Bluetooth mics (AirPods)
    // can't always satisfy it and silently fail. Let the browser choose the
    // native rate; AudioContext resamples to 16kHz internally.
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Register AudioWorklet
    await audioCtx.audioWorklet.addModule('/ui/audio-worklet.js');
    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');

    // Connect WebSocket
    const wsUrl = `ws://${location.host}/ws/audio`;
    ws = new WebSocket(wsUrl);

    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'auth',
        token: token,
        session_id: sessionId || null,
      }));
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } else {
        playAudioChunk(event.data);
      }
    };

    ws.onclose = () => { if (micOn) stopMic(); };
    ws.onerror = (e) => { console.error('WebSocket error:', e); };

    // Worklet sends PCM16 buffers over WebSocket
    workletNode.port.onmessage = (event) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };

    // Connect source -> worklet. Use a silent gain node as dummy output
    // so the worklet graph runs without routing mic audio to speakers
    // (avoids AirPods switching to degraded HFP mode).
    const silentGain = audioCtx.createGain();
    silentGain.gain.value = 0;
    source.connect(workletNode);
    workletNode.connect(silentGain);
    silentGain.connect(audioCtx.destination);

    micOn = true;
    micBtn.className = 'mic-on';
  } catch (e) {
    console.error('Mic start failed:', e);
    stopMic();
  }
}

function stopMic() {
  micOn = false;
  micBtn.className = 'mic-off';

  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (ws) { ws.close(); ws = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'auth_ok':
      break;
    case 'auth_fail':
      stopMic();
      break;
    case 'state':
      if (msg.state === 'processing') setBusy(true);
      if (msg.state === 'listening') setBusy(false);
      break;
    case 'transcription':
      appendMessage('voice-user', msg.text);
      break;
    case 'wake_reject':
      break;
    case 'token': {
      let agentMsg = document.querySelector('.message-voice-agent.streaming');
      if (!agentMsg) agentMsg = createAgentMessage('voice');
      appendToken(agentMsg, msg.text);
      break;
    }
    case 'tool_call': {
      let agentMsg = document.querySelector('.message-voice-agent.streaming');
      if (!agentMsg) agentMsg = createAgentMessage('voice');
      appendTool(agentMsg, `[calling ${msg.name}...]`);
      break;
    }
    case 'tool_result': {
      let agentMsg = document.querySelector('.message-voice-agent.streaming');
      if (!agentMsg) agentMsg = createAgentMessage('voice');
      appendTool(agentMsg, `[${msg.name}] ${msg.content}`);
      break;
    }
    case 'done': {
      const el = document.querySelector('.message-voice-agent.streaming');
      if (el) el.classList.remove('streaming');
      break;
    }
    case 'tts_start':
      // Mute worklet output during TTS playback (echo cancellation)
      if (workletNode) workletNode.port.postMessage({ type: 'mute' });
      break;
    case 'tts_end':
      if (workletNode) workletNode.port.postMessage({ type: 'unmute' });
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'tts_done' }));
      }
      break;
    case 'error':
      appendMessage('agent', `[Error: ${msg.detail}]`);
      break;
  }
}

// -- Audio playback --

let playbackQueue = [];
let playing = false;

function playAudioChunk(arrayBuffer) {
  playbackQueue.push(arrayBuffer);
  if (!playing) drainPlayback();
}

async function drainPlayback() {
  playing = true;
  const ctx = audioCtx || new AudioContext({ sampleRate: 16000 });

  while (playbackQueue.length > 0) {
    const buf = playbackQueue.shift();
    const int16 = new Int16Array(buf);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const audioBuffer = ctx.createBuffer(1, float32.length, 16000);
    audioBuffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start();

    await new Promise(resolve => { source.onended = resolve; });
  }

  playing = false;
}

// -- DOM helpers --

function appendMessage(role, text) {
  const div = document.createElement('div');
  const cssClass = role === 'user' ? 'message-user'
    : role === 'voice-user' ? 'message-voice-user'
    : 'message-agent';
  div.className = `message ${cssClass}`;

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You'
    : role === 'voice-user' ? 'You (voice)'
    : 'Agent Zero';
  div.appendChild(label);

  const content = document.createElement('div');
  content.textContent = text;
  div.appendChild(content);

  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function createAgentMessage(source = 'text') {
  const div = document.createElement('div');
  div.className = source === 'voice'
    ? 'message message-voice-agent streaming'
    : 'message message-agent streaming';

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = 'Agent Zero';
  div.appendChild(label);

  const content = document.createElement('div');
  content.className = 'agent-content';
  div.appendChild(content);

  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function appendToken(agentMsg, text) {
  const content = agentMsg.querySelector('.agent-content');
  if (content) {
    content.textContent += text;
  } else {
    agentMsg.textContent += text;
  }
  scrollToBottom();
}

function appendTool(agentMsg, text) {
  const toolDiv = document.createElement('div');
  toolDiv.className = 'tool-block';
  toolDiv.textContent = text;

  const content = agentMsg.querySelector('.agent-content');
  if (content) {
    content.appendChild(toolDiv);
  } else {
    agentMsg.appendChild(toolDiv);
  }
  scrollToBottom();
}

function scrollToBottom() {
  const container = document.getElementById('chat-container');
  container.scrollTop = container.scrollHeight;
}
