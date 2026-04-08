# Agent Zero

A local AI agent built on LangGraph and Ollama. Persistent memory, a knowledge base, voice chat, and a web UI -- all running on your own hardware with no cloud dependencies.

Designed to run alongside Claude Code: Agent Zero maintains project context in `CLAUDE.md` files that Claude Code reads automatically at session start. The two systems share a knowledge layer without any shared process or SDK dependency.

**Created:** April 2026


---

## What it does

- **Text chat** via browser (SSE streaming) or terminal CLI
- **Voice chat** via WebSocket -- wake word detection, Whisper STT, macOS TTS
- **Long-term memory** -- every conversation is embedded in ChromaDB with smart deduplication, contradiction detection, and LLM-based novelty filtering
- **Knowledge base** -- Obsidian-compatible markdown files the agent reads and writes, tagged and searchable
- **CLAUDE.md bridge** -- agent assembles project context from the knowledge base and writes it to any project directory for Claude Code to pick up
- **REST API** -- localhost-only, bearer token auth, full CRUD on the knowledge base


---

## Quick start

```bash
git clone https://github.com/thefilesareinthecomputer/agent-zero
cd agent-zero

python3.12 -m venv venv-agent-zero
source venv-agent-zero/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set API_TOKEN and review model names

# Install and start Ollama, then pull models
ollama pull gemma4:26b    # main model
ollama pull gemma4:e2b    # memory tagger (lightweight)

# Run the CLI
python -m agent.run

# Or run the web UI + API server
python -m bridge.api_run
# Opens at http://127.0.0.1:8900
```

For voice chat, also pull the voice model and set up Whisper:
```bash
ollama pull gemma4:e4b
# Whisper-MLX downloads on first startup (Apple Silicon only)
```


---

## Hardware

### Minimum (text chat only)

| | Requirement |
|--|-------------|
| RAM | 8 GB system memory |
| Storage | 10 GB free (model weights) |
| OS | macOS, Linux, or Windows |
| CPU | Any modern multi-core |

Run a small model like `gemma2:2b` or `phi3:mini` on Ollama. Text agent and web UI work anywhere Ollama does.

### Recommended (comfortable daily use)

| | Requirement |
|--|-------------|
| RAM | 16--32 GB unified or system memory |
| Storage | 30 GB+ free |
| OS | macOS (Sonoma or later) for voice; any OS for text |

This range handles 7B--13B models with decent context windows and comfortable response times.

### Reference build (full feature set, multiple concurrent models)

| Component | Spec |
|-----------|------|
| Chip | Apple M2 Ultra or equivalent |
| Memory | 64 GB unified |
| Storage | 1 TB SSD |
| OS | macOS Tahoe |

This configuration runs the 26B MoE main model, 2B tagger, voice model, and Whisper simultaneously (~22 GB total). Swap out models to fit your hardware -- the system is model-agnostic.


---

## Platform support

| Feature | macOS (Apple Silicon) | macOS (Intel) | Linux | Windows |
|---------|:---------------------:|:-------------:|:-----:|:-------:|
| Text agent + web UI | yes | yes | yes | yes |
| REST API | yes | yes | yes | yes |
| Voice (Whisper-MLX) | yes | -- | -- | -- |
| Voice (standard Whisper) | yes | yes | yes | yes |
| macOS `say` TTS | yes | yes | -- | -- |
| `launchctl` Ollama env vars | yes | yes | -- | -- |

**Linux/Windows voice:** replace `lightning-whisper-mlx` with `openai-whisper`, swap `voice/tts.py` for `pyttsx3` or `piper`, and set Ollama env vars via the system environment instead of `launchctl`. PRs welcome.

**Ollama env vars on Linux/Windows:** export in your shell profile or set them as system environment variables -- `launchctl` is macOS-only.


---

## Architecture

```
                          ┌─────────┐
                          │   You   │
                          └────┬────┘
                          ┌────┴────┐
                 ┌────────┤         ├────────┐
                 │        └─────────┘        │
        ┌────────┴────────┐        ┌─────────┴─────────┐
        │   Agent Zero    │        │   Claude Code      │
        │  (always-on)    │        │   (user-driven)    │
        │                 │        │                    │
        │  LangGraph      │  ───►  │  Reads CLAUDE.md   │
        │  Ollama/Gemma4  │        │  written by        │
        │  Memory layer   │        │  Agent Zero        │
        │  Voice          │        │                    │
        └────────┬────────┘        └────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───┴───┐ ┌─────┴─────┐ ┌───┴────┐
│Ollama │ │  Memory   │ │ Tools  │
│Gemma4 │ │ SQLite    │ │ shell  │
│26B MoE│ │ ChromaDB  │ │ files  │
│+ E2B  │ │           │ │ KB     │
└───────┘ └───────────┘ └────────┘
```

Agent Zero runs persistently and remembers everything. Claude Code is a separate tool you use for coding -- it reads the `CLAUDE.md` files Agent Zero writes. Neither process runs inside the other.


---

## Models

| Role | Default | Size (Q4) | Notes |
|------|---------|-----------|-------|
| Main (text chat) | `gemma4:26b` | ~17 GB | MoE, 4B active params. Fast + capable. |
| Voice | `gemma4:e4b` | ~3 GB | Short answers, commands. Web voice chat. |
| Tagger | `gemma4:e2b` | ~2 GB | Memory classification, novelty checking. |
| Heavy | `gemma4:31b` | ~20 GB | Dense. Available on-demand for complex tasks. |
| Reasoning | `llama3.3:70b` | ~42 GB | Load on-demand, unload main first. |
| Code | `qwen3-coder:30b` | ~18 GB | Code-specific tasks. |
| Vision | `qwen3-vl:30b` | ~18 GB | Image/document understanding. |

**Running on less RAM:** swap the main model for something smaller. `gemma2:9b`, `mistral:7b`, or `phi3:medium` all work -- change `MAIN_MODEL` in `.env`. The architecture is model-agnostic.

**Memory headroom on 64 GB:** main (26B MoE) + tagger (E2B) + voice (E4B) + Whisper ≈ 22 GB. Set `num_ctx=16384` -- Ollama defaults to 2048--4096, not the model's full 256K capability. Unload the main model before loading 70B: `curl http://localhost:11434/api/generate -d '{"model": "gemma4:26b", "keep_alive": 0}'`


---

## Project layout

```
agent-zero/
├── .env.example                  # Copy to .env and fill in your values
├── .python-version               # 3.12
├── requirements.txt              # Pinned deps (pip freeze)
├── agent/
│   ├── agent.py                  # LangGraph ReAct agent, SQLite checkpointing, memory injection
│   ├── config.py                 # .env-driven config
│   ├── tools.py                  # @tool definitions: time, shell, files, knowledge base, bridge
│   └── run.py                    # CLI entry point with streaming and memory commands
├── bridge/
│   ├── api.py                    # FastAPI app -- knowledge CRUD, CLAUDE.md generation, chat
│   ├── api_models.py             # Pydantic schemas
│   ├── api_run.py                # Uvicorn entry point
│   ├── chat.py                   # SSE text chat + WebSocket voice endpoints
│   └── claude_md.py              # CLAUDE.md assembler -- scored, budget-aware
├── knowledge/
│   └── knowledge_store.py        # Markdown KB: list, read, save, search, tag filter
├── memory/
│   ├── vector_store.py           # ChromaDB wrapper
│   ├── tagger.py                 # LLM-based category/subcategory tagging + novelty check
│   └── memory_manager.py         # Pipeline: noise filter, dedup, contradiction, novelty, prune
├── voice/
│   ├── vad.py                    # Silero-VAD state machine
│   ├── stt.py                    # Whisper-MLX STT + wake word extraction
│   ├── tts.py                    # macOS say TTS, sentence-chunked PCM streaming
│   └── pipeline.py               # VAD -> STT -> wake word -> query, echo cancellation
├── ui/
│   ├── index.html                # Single-page dark theme UI
│   ├── style.css
│   ├── app.js                    # SSE chat, WebSocket voice, audio playback
│   ├── audio-worklet.js          # float32 -> PCM16, 512-sample buffering
│   └── sounds/ready.wav          # Wake word confirmation tone
├── scripts/
│   ├── az                        # CLI launcher (symlink to /usr/local/bin/az)
│   ├── az-api                    # API server launcher
│   └── setup_ollama.sh           # Ollama env var setup for Apple Silicon
├── project_outputs/              # Default output dir for generated CLAUDE.md files
├── tests/
│   ├── test_api.py               # Knowledge CRUD, auth, privacy, CLAUDE.md routes (28 tests)
│   ├── test_chat_api.py          # SSE auth, WebSocket auth, static serving (10 tests)
│   ├── test_claude_md.py         # Bridge scoring, budget, path resolution (18 tests)
│   ├── test_knowledge_store.py   # KB file ops, frontmatter, search, index, log (38 tests)
│   ├── test_memory_manager.py    # Memory pipeline, dedup, contradiction, pruning (23 tests)
│   └── test_voice.py             # VAD, wake word, TTS, echo suppression (14 tests)
├── optimization/                 # Phase 4 -- DSPy GEPA prompt evolution (planned)
├── fine_tuning/                  # Phase 5 -- MLX LoRA fine-tuning (planned)
└── data/                         # Auto-created, gitignored -- SQLite DB, ChromaDB
```


---

## Configuration

Copy `.env.example` to `.env` and fill in your values.

```bash
# Models -- swap for smaller ones on limited hardware
MAIN_MODEL=gemma4:26b
FAST_MODEL=gemma4:e2b
REASONING_MODEL=llama3.3:70b
CODE_MODEL=qwen3-coder:30b
VISION_MODEL=qwen3-vl:30b
FINETUNE_MODEL=gemma4:e4b

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Paths (relative to project root)
AGENT_DB_PATH=data/agent_memory.db
CHROMA_DB_PATH=data/chroma_db

# API -- generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_TOKEN=your_token_here_min_32_chars
API_PORT=8900

# Bridge output directory for generated CLAUDE.md files
PROJECT_OUTPUTS_PATH=project_outputs
CLAUDE_MD_MAX_CHARS=65536

# Voice (macOS only)
VOICE_MODEL=gemma4:e4b
WHISPER_MODEL=distil-large-v3
TTS_VOICE=Samantha
TTS_RATE=175
VAD_THRESHOLD=0.5
VAD_SILENCE_MS=1000
MAX_SPEECH_SECONDS=30
VOICE_INPUT_GAIN=1.0
```

### Ollama on Apple Silicon

Set these via `launchctl` (not `export`) so the Ollama launch agent picks them up:

```bash
launchctl setenv OLLAMA_MAX_LOADED_MODELS "2"
launchctl setenv OLLAMA_NUM_PARALLEL "4"
launchctl setenv OLLAMA_KEEP_ALIVE "1800"
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"
launchctl setenv OLLAMA_HOST "127.0.0.1:11434"
```

On Linux/Windows, set these as normal environment variables or in a systemd unit file.

### Telemetry note

`langsmith` is a transitive dependency of LangGraph. By default it does nothing -- tracing only activates if you set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. To be explicit, add `LANGCHAIN_TRACING_V2=false` to your `.env`.


---

## Launchers

```bash
# Symlink for convenient CLI access (optional)
ln -sf "$(pwd)/scripts/az" /usr/local/bin/az
ln -sf "$(pwd)/scripts/az-api" /usr/local/bin/az-api

az          # run the CLI agent
az-api      # run the web UI + API server
```

CLI memory commands: `memories`, `forget last`, `forget all`, `knowledge`

CLI thread commands: `new` (start fresh thread), `quit`


---

## Build status

| Phase | Status | Description |
|-------|--------|-------------|
| 1: Foundation | done | LangGraph ReAct agent, tools, CLI, SQLite persistence |
| 2a: Memory | done | ChromaDB, tagging, dedup, contradiction, novelty, pruning |
| KB: Knowledge base | done | Obsidian markdown, tag filtering, auto index/log |
| 3a: CLAUDE.md bridge | done | Project-tagged KB → CLAUDE.md, scored + budget-aware |
| 3b: HTTP API | done | FastAPI, auth, privacy filtering, CRUD, 150 tests passing |
| 6: Voice + Web UI | done | Whisper STT, macOS TTS, VAD, SSE chat, WebSocket voice |
| 2b: Memory (txtai) | planned | Graph relationships + semantic SQL if ChromaDB proves insufficient |
| 4: Self-improvement | planned | DSPy GEPA prompt evolution |
| 7: Web tools | planned | Crawl4AI, calendar, clipboard |
| 5: Fine-tuning | planned | MLX LoRA on interaction data |


---

## Technical decisions

**Why LLM novelty checking instead of cosine thresholds for memory**

Cosine distance tells you whether two texts are about the same topic -- it does not tell you whether one adds new information beyond the other. "My favorite color is blue" and "My favorite color is blue, specifically cobalt, not navy" land at very similar distances. A fixed threshold either discards real details or lets noise through. The solution is to ask the fast model (E2B) directly: "does this add new information?" That judgment requires language understanding, not vector arithmetic.

**Why category/subcategory tagging**

Without scoped comparisons, "my favorite color is green" collides with "I like green tea" -- both mention green, both land nearby. The two-level taxonomy (`user-preference/favorite-color`, `user-preference/favorite-food`) keeps comparisons semantically grounded. Update vs. addition intent prevents additions from triggering contradiction replacement -- "I also like sushi" should stack alongside "I love pizza," not replace it.

**Why `langchain-ollama` not `langchain-community`**

The community `ChatOllama` throws `NotImplementedError` on `bind_tools()`. `langchain-ollama` is the maintained package with proper tool calling support.

**Why `AsyncSqliteSaver` in the web server**

The CLI uses sync `SqliteSaver`. The web server uses `AsyncSqliteSaver` so `agent.astream()` works natively in async FastAPI handlers without a thread bridge. Do not mix the two.

**Why Ollama env vars via `launchctl` on macOS**

Ollama runs as a macOS launch agent. Environment variables set with `export` in your shell are not inherited by it. You have to use `launchctl setenv` and restart Ollama. This applies to all Ollama config: `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, model limits, etc.

**Why full regeneration on CLAUDE.md updates**

Merging or diffing against an existing CLAUDE.md would require parsing it back into structure and reconciling with the source knowledge files. Full regeneration is simpler, deterministic, and the file is explicitly marked as auto-generated -- manual edits are not preserved by design.

**Why ChromaDB**

Simplest local vector store -- Python-native, no server process, straightforward API. The memory pipeline's novelty checking compensates for its limitations around semantic discrimination. txtai (graph relationships + semantic SQL) is on the roadmap if ChromaDB proves insufficient after more real usage.


---

## Troubleshooting

**High memory usage or swapping to disk**

Ollama's default `num_ctx` is 2048--4096. The 256K figure on model cards is the model's architectural maximum, not what Ollama allocates. If you set `num_ctx=262144`, the KV cache alone can exceed available memory. Keep it at `num_ctx=16384` in `agent.py` for good headroom without waste. Also requires `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_FLASH_ATTENTION=1` via `launchctl`.

**Ollama env vars not taking effect**

Use `launchctl setenv VAR_NAME "value"` then `pkill -f ollama && open -a Ollama`. Regular `export` in your shell does not reach the Ollama launch agent.

**Tool calls returning raw JSON instead of executing**

Do not set `format="json"` on the `ChatOllama` instance. It breaks `create_react_agent`'s tool execution path.

**`bind_tools` raises `NotImplementedError`**

Wrong import. Use `from langchain_ollama import ChatOllama`, not `from langchain_community.chat_models import ChatOllama`.

**`stream_mode="messages"` hangs**

Known issue with `create_react_agent` + `ChatOllama`. Use `stream_mode="updates"`.

**Models not using GPU**

Check `ollama ps` -- PROCESSOR column should show 100% GPU. If not, restart Ollama after setting `launchctl` env vars.

**OOM loading 70B**

Unload the main model first: `curl http://localhost:11434/api/generate -d '{"model": "gemma4:26b", "keep_alive": 0}'`

**CLI and web server conflict**

They cannot run concurrently -- both write to ChromaDB and SQLite. Run one at a time.

**WebSocket voice audio silent**

Set `ws.binaryType = 'arraybuffer'` before TTS audio arrives (already set in `app.js`). If you're extending the client, this is the first thing to check.


---

## Stack

| Layer | Package | License |
|-------|---------|---------|
| Agent framework | LangGraph | MIT |
| LLM interface | langchain-ollama | MIT |
| Inference | Ollama | MIT |
| Vector memory | ChromaDB | Apache 2.0 |
| API | FastAPI + Uvicorn | MIT / BSD |
| STT | lightning-whisper-mlx | MIT |
| VAD | silero-vad | MIT |
| Wake word | openwakeword | Apache 2.0 |
| ML runtime | MLX (Apple Silicon) | Apache 2.0 |
| Neural nets | PyTorch | BSD |
| DB persistence | aiosqlite | MIT |
| Validation | Pydantic | MIT |
| Tests | pytest | MIT |

Everything here is open source with permissive licenses. No commercial dependencies.


---

## Acknowledgments

Index and log patterns from Andrej Karpathy's LLM Wiki. ChromaDB organization approach from MemPalace by Milla Jovovich.


---

## License

MIT. See `LICENSE`.
