# Agent Zero -- Local AI Agent on Apple Silicon

Autonomous, self-improving AI agent running on a Mac Studio M2 Ultra (64GB). Built with LangGraph, Ollama (MLX backend), and a layered memory system. Agent Zero is the persistent brain -- long-term memory, project context, decision history, general assistant. Claude Code Desktop is used alongside it for coding tasks -- Agent Zero maintains project context in `CLAUDE.md` files that Desktop reads automatically. Everything runs locally.

**Created:** April 2, 2026


---

## Hardware

| Component | Spec |
|-----------|------|
| Machine | Mac Studio |
| Chip | M2 Ultra (24-core CPU, 60-core GPU) |
| Memory | 64 GB unified (shared CPU/GPU, zero-copy with MLX) |
| Storage | 1 TB SSD |
| OS | macOS Tahoe 26.1 |


---

## Models

| Role | Model | Size (Q4) | Context | Notes |
|------|-------|-----------|---------|-------|
| **Main (daily)** | `gemma4:26b` | ~17 GB | 256K | MoE, 4B active params. Faster and subjectively smarter than 31B dense. Daily driver. |
| **Heavy** | `gemma4:31b` | ~20 GB | 256K | Dense, all 31B params active. Better benchmarks but too slow for interactive chat. Tool-heavy tasks. |
| **Fast / Tagger** | `gemma4:e2b` | ~2 GB | 128K | Effective 2B params. Memory tagging, novelty checking, lightweight classification. |
| **Reasoning** | `llama3.3:70b` | ~42 GB | 128K | Deep reasoning. Load on-demand, unload main first. |
| **Code** | `qwen3-coder:30b` | ~18 GB | 128K | Code-specific tasks. |
| **Vision** | `qwen3-vl:30b` | ~18 GB | 128K | Image/document understanding. |
| **Fine-tune target** | `gemma4:e4b` | ~3 GB | 128K | LoRA fine-tune locally via MLX. |

**Memory rules for 64 GB:** one large model at a time. Main (26B MoE) + fast (E2B) ≈ 19 GB, leaving 45 GB for OS and context. Set `num_ctx=16384` for adequate conversation headroom (Ollama defaults to 2048-4096, not the model's 256K max capability). Set `num_predict=2048` to prevent output truncation (Ollama defaults to 128 tokens). Unload main before loading reasoning: `curl http://localhost:11434/api/generate -d '{"model": "gemma4:26b", "keep_alive": 0}'`


---

## Architecture

```
                          ┌─────────┐
                          │   User  │
                          └────┬────┘
                          ┌────┴────┐
                 ┌────────┤         ├────────┐
                 │        └─────────┘        │
        ┌────────┴────────┐        ┌─────────┴─────────┐
        │   Agent Zero    │        │  Claude Code       │
        │  (always-on)    │        │  Desktop           │
        │                 │        │  (user-driven)     │
        │  LangGraph      │        │                    │
        │  Ollama/Gemma4  │  ───►  │  Reads CLAUDE.md   │
        │  Memory layer   │        │  written by        │
        │  Tools          │        │  Agent Zero        │
        │  Voice          │        │                    │
        └────────┬────────┘        └────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───┴───┐ ┌─────┴─────┐ ┌───┴────┐
│Ollama │ │  Memory   │ │ Tools  │
│Gemma4 │ │ SQLite    │ │ shell  │
│26B MoE│ │ ChromaDB  │ │ files  │
│+ E2B  │ │           │ │        │
└───────┘ └───────────┘ └────────┘
```

**How they work together:**
- Agent Zero is the persistent brain -- runs locally, remembers everything, handles general tasks
- Claude Code Desktop is a separate app you use for coding -- you drive it manually
- Agent Zero writes/updates `CLAUDE.md` files with project context, architecture decisions, and conventions
- Claude Code Desktop reads `CLAUDE.md` automatically at session start -- inherits Agent Zero's knowledge
- You can ask Agent Zero to summarize context, then paste/reference it in Claude Code Desktop sessions


---

## Core stack

| Layer | Tool | Status | Why |
|-------|------|--------|-----|
| Agent framework | LangGraph | done | Industry standard. Persistence, streaming, tool orchestration. |
| Claude Code bridge | `CLAUDE.md` files | done | Agent Zero writes project context that Claude Code Desktop reads automatically. No SDK, no CLI, no binary dependency. |
| HTTP API | FastAPI + uvicorn | done | Privacy-preserving REST API on localhost:8900. Bearer token auth, knowledge CRUD, CLAUDE.md generation. |
| KB infrastructure | index.md + log.md | done | Auto-maintained catalog and append-only audit trail (inspired by Karpathy LLM Wiki). |
| Inference | Ollama | done | Model management, OpenAI-compatible API. |
| Chat persistence | SQLite via `SqliteSaver` | done | Zero-infra, file-based. |
| Vector memory | ChromaDB | done | Lightweight, local, Python-native. Semantic search over past interactions. |
| Memory tagging | Ollama (e2b) | done | Category/subcategory classification + update/addition intent via lightweight LLM. |
| Novelty checking | Ollama (e2b) | done | LLM-based judgment for whether new info adds value beyond existing memories. |
| Semantic + graph | txtai | planned | Embeddings + graph + SQL in one package. Knowledge graph without a separate DB. |
| Prompt optimization | DSPy GEPA | planned | Evolves prompts via reflection on execution traces. No weight changes. |
| Agent optimization | Microsoft Agent Lightning | planned | Framework-agnostic. Prompt optimization + optional RL. Works with LangGraph. |
| Local fine-tuning | MLX + mlx-lm | planned | Native Apple Silicon LoRA/QLoRA. Up to ~30B on 64GB. |
| Cloud fine-tuning | Unsloth + Google Colab | planned | Free T4 GPU for bigger experiments. Export GGUF to Ollama. |
| Voice STT | Whisper-MLX (lightning-whisper-mlx) | planned | Runs on Metal. |
| Voice TTS | macOS `say` | planned | Zero deps. |
| Web browsing | Crawl4AI | planned | Local-first, async, LLM-friendly extraction. |


---

## Project layout

```
agent-zero/
├── .env                          # Model config, Ollama URL, DB paths, voice params
├── .python-version               # 3.12
├── requirements.txt              # Pinned Python deps
├── README.md                     # This file
├── CLAUDE.md                     # Project context for Claude Code Desktop
├── agent/
│   ├── __init__.py
│   ├── agent.py                  # LangGraph ReAct agent, SQLite checkpointing, memory injection
│   ├── config.py                 # .env-driven model routing
│   ├── tools.py                  # @tool definitions (time, shell, file r/w, knowledge, bridge)
│   └── run.py                    # CLI entry point, streaming, memory commands
├── bridge/
│   ├── __init__.py
│   ├── api.py                    # FastAPI app -- 7 routes, auth, privacy filtering
│   ├── api_models.py             # Pydantic request/response schemas
│   ├── api_run.py                # Uvicorn entry point (python -m bridge.api_run)
│   └── claude_md.py              # CLAUDE.md assembler -- project-tagged knowledge to markdown
├── knowledge/
│   ├── __init__.py
│   └── knowledge_store.py        # Obsidian markdown files -- list, read, save, search, tag filter
├── memory/
│   ├── __init__.py
│   ├── vector_store.py           # ChromaDB wrapper -- store, search, tag-filtered queries
│   ├── tagger.py                 # LLM-based category/subcategory tagging + novelty checker (e2b)
│   └── memory_manager.py         # Unified interface -- dedup, contradiction, novelty, pruning
├── voice/                        # Phase 6
│   ├── __init__.py
│   ├── stt.py                    # Whisper-MLX speech-to-text
│   └── tts.py                    # macOS say text-to-speech
├── optimization/                 # Phase 4
│   ├── __init__.py
│   ├── dspy_optimizer.py         # GEPA prompt evolution
│   └── agent_lightning.py        # Agent Lightning integration
├── fine_tuning/                  # Phase 5
│   ├── prepare_data.py           # Extract training data from interaction logs
│   └── train_mlx.py             # MLX LoRA fine-tuning script
├── data/                         # Auto-created, gitignored
│   ├── agent_memory.db           # SQLite conversation persistence
│   └── chroma_db/                # ChromaDB vector store
├── scripts/
│   ├── az                        # CLI launcher (symlink to /usr/local/bin/az)
│   ├── az-api                    # API server launcher (symlink to /usr/local/bin/az-api)
│   └── setup_ollama.sh           # Ollama env vars + model pulls for M2 Ultra
├── tests/
│   └── test_api.py               # pytest suite -- auth, privacy, canon, traversal, CLAUDE.md
└── claude/                       # Build state tracking, gitignored
    └── state.md
```


---

## Configuration

### .env

```bash
# Models (Ollama)
MAIN_MODEL=gemma4:26b
FAST_MODEL=gemma4:e2b
REASONING_MODEL=llama3.3:70b
CODE_MODEL=qwen3-coder:30b
VISION_MODEL=qwen3-vl:30b
FINETUNE_MODEL=gemma4:e4b

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Memory
AGENT_DB_PATH=data/agent_memory.db
CHROMA_DB_PATH=data/chroma_db

# API
API_TOKEN=<generated-via-secrets.token_urlsafe(32)>
API_PORT=8900

# Voice
VOICE_LANGUAGE=en
VOICE_MIN_RMS=0.01
VOICE_CHUNK_SECONDS=3
VOICE_INPUT_GAIN=1.0
```
```

### Ollama (M2 Ultra 64GB)

```bash
launchctl setenv OLLAMA_MAX_LOADED_MODELS "2"
launchctl setenv OLLAMA_NUM_PARALLEL "4"
launchctl setenv OLLAMA_KEEP_ALIVE "1800"
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"
launchctl setenv OLLAMA_HOST "127.0.0.1:11434"
```


---

## Build plan

Each phase produces working, testable code before the next. Agreed build order: 1 > 2a > 3 > 6 > 2b > 4 > 7 > 5.

### Phase 1: Foundation -- DONE

LangGraph ReAct agent with ChatOllama, four tools (time, shell, file read, file write), SQLite checkpointing, CLI with streaming and thread switching. Completed April 2, 2026.

### Phase 2a: Memory (ChromaDB + smart dedup) -- DONE

Vector memory with intelligent storage pipeline. Completed April 3, 2026.

What got built:
- `memory/vector_store.py` -- ChromaDB persistent store with cosine distance, tag-filtered search via `where` clauses
- `memory/tagger.py` -- e2b classifies every message into category/subcategory (e.g. `user-preference/favorite-color`) plus update/addition intent
- `memory/memory_manager.py` -- five-stage pipeline: noise filter (< 3 words skipped), tagging, dedup (distance < 0.15 refreshes timestamp), contradiction replacement (updates within same subcategory, distance < 0.50), LLM novelty check (additions ask e2b "does this add new info beyond what's already stored?")
- Wired into agent.py via callable prompt that retrieves top-5 memories per query
- CLI commands: `memories`, `forget last`, `forget all`
- Pruning: 30-day TTL, 500 memory hard cap, runs on startup

Key design decision: cosine distance alone cannot distinguish "noise about same topic" from "new details about same topic." Both land at similar distances. The distinction requires LLM judgment about information novelty, not threshold tuning. That is why additions go through `check_novelty()` via e2b rather than a distance cutoff.

### Knowledge Base -- DONE

Obsidian-compatible markdown files managed by the agent. Completed April 6, 2026.

What got built:
- `knowledge/knowledge_store.py` -- list, read, save, search with YAML frontmatter and auto-generated TOC
- Four agent tools: `list_knowledge`, `read_knowledge`, `search_knowledge`, `save_knowledge`
- Tag filtering: `filter_tags` and `exclude_tags` keyword-only params on `list_files()`
- Edit model: read-modify-write via `save_knowledge` (no delete tool)
- Obsidian format: YAML frontmatter (tags, created, last-modified), H1 title, TOC, H2 sections with --- dividers

### Phase 3a: CLAUDE.md Bridge -- DONE

Agent Zero assembles knowledge base content into CLAUDE.md files for target project directories. Completed April 6, 2026.

What got built:
- `bridge/claude_md.py` -- collects files by `project:<name>` tag, excludes private/secret, 16KB size cap
- `update_project_context` tool -- agent writes CLAUDE.md to any project directory
- Full regeneration on each call -- no merge, no diffing
- Only project-tagged files are included (no global/untagged files leak in)

### Phase 3b: HTTP API -- DONE

Privacy-preserving REST API for local tools to query Agent Zero's knowledge base. Completed April 7, 2026.

What got built:
- `bridge/api.py` -- FastAPI app with 7 routes: health, list, search, read, save, claude-md generate, claude-md write
- `bridge/api_models.py` -- Pydantic request/response schemas with strict validation
- `bridge/api_run.py` -- Uvicorn entry point, hardcoded to 127.0.0.1, token validation at startup
- Bearer token auth (timing-safe via `secrets.compare_digest`, minimum 32 chars)
- Privacy model: files tagged private/secret excluded from all responses, return 404 (not 403) to prevent enumeration
- Merged listing: knowledge/ + knowledge_canon/ combined in responses with source tagging
- CLAUDE.md iteration workflow: generate returns content as object, write accepts caller-modified content
- `knowledge/knowledge_store.py` additions: `get_file_metadata()`, `rebuild_index()`, `append_log()`
- `index.md` -- auto-maintained catalog rebuilt on every save (inspired by Karpathy LLM Wiki)
- `log.md` -- append-only audit trail of all write operations
- `scripts/az-api` -- launcher script (mirrors az pattern)
- `tests/test_api.py` -- 28 pytest tests covering auth, privacy, canon blocks, path traversal, search, CLAUDE.md workflow
- System prompt updated to reference index.md and encourage query-to-page persistence

Inspirations: Karpathy's LLM Wiki (index/log/lint patterns), MemPalace (MCP server pattern).

### Phase 6: Voice -- NEXT

Independent of other phases, high daily-use value.

1. `voice/stt.py` -- Whisper-MLX, proper VAD (webrtcvad or silero-vad) before STT
2. `voice/tts.py` -- macOS `say`, configurable voice
3. Wake word detection with start-of-utterance requirement
4. Press-to-talk hotkey bypass
5. Wire into `agent/run.py` as alternative input mode

### Phase 4: Self-improvement (~2-3 weeks)

Two parallel tracks.

**Track A -- DSPy GEPA (prompt evolution):**
1. Add to `requirements.txt`: `dspy-ai`, `gepa`
2. Define Agent Zero's task signatures as DSPy modules (tool selection, response quality, memory retrieval accuracy)
3. Build evaluation metrics with textual feedback
4. Run GEPA with gemma4:31b as both student and reflection LM
5. Store optimized prompts, swap them into Agent Zero's system prompt

**Track B -- Agent Lightning (agent optimization):**
1. Add to `requirements.txt`: `agentlightning`
2. Add `agl.emit()` trace collectors to LangGraph execution
3. Start with prompt optimization mode (no RL infra)
4. Define reward functions: task completion, tool accuracy, memory relevance
5. Let Agent Lightning analyze traces and propose improvements

### Phase 5: Fine-tuning (~1-2 weeks)

Train a custom model on Agent Zero's interaction data.

**Local (MLX):**
1. `pip install mlx mlx-lm`
2. Build `fine_tuning/prepare_data.py` -- extract interaction logs from SQLite to instruction/input/output JSONL
3. QLoRA fine-tune gemma4:e4b (~3 GB base, fits easily in 64 GB)
4. Export LoRA adapter to GGUF to Ollama Modelfile
5. A/B test fine-tuned model vs base

**Colab (Unsloth) -- for larger models:**
1. Open Unsloth notebook in Google Colab (free T4)
2. Upload training data, fine-tune with QLoRA, export GGUF
3. `ollama create agent-zero-custom -f Modelfile`

### Phase 2b: Memory (txtai / knowledge graph) -- IF NEEDED

Add txtai for graph relationships and semantic SQL if ChromaDB alone proves insufficient. Evaluate after more real usage.

### Phase 7: Web + expanded tools (~1-2 weeks)

1. Add to `requirements.txt`: `crawl4ai`
2. Add web search and page fetch as LangGraph tools
3. Feed browser-extracted content into memory pipeline (summarize to embed to store)
4. Expand tools: calendar, clipboard, Finder integration via macOS APIs


---

## Technical decisions

**Why gemma4:26b MoE over 31B Dense for daily use:** 26B MoE has 26B total params but only 4B active per token. In practice it is significantly faster than the 31B dense model while being subjectively smarter and more honest in conversation. The 31B dense model is still available for heavy tasks but too slow for interactive chat. 31B remains the better fine-tuning base since QLoRA is well-supported on dense architectures and MoE fine-tuning requires expert profiling with sparse gradient issues.

**Why LLM novelty checking over cosine thresholds:** During memory development, cosine distance alone kept failing. "I love pizza" and "I love pepperoni with hot honey on pizza" land at similar distances -- the vector can tell they are about the same topic but not whether the second adds new information. A fixed threshold either eats real details or lets noise through. The solution: ask e2b (the fast model) "does this add new info beyond what's already stored?" This handles the information/noise distinction that similarity scores fundamentally cannot.

**Why category/subcategory tagging:** Scoping memory comparisons to the same subcategory prevents cross-topic collisions. Without tags, "my favorite color is green" could collide with "I like green tea." The two-level taxonomy (e.g. `user-preference/favorite-color`) keeps comparisons meaningful. Update vs addition intent prevents additions from triggering contradiction replacement -- "I also like sushi" should not replace "I love pizza" even though both are `favorite-food`.

**Why Claude Code Desktop (not CLI, not Agent SDK):** The CLI and Agent SDK give Anthropic's binary headless execution on your machine with phone-home mechanisms. The Desktop app is sandboxed in its own GUI where you see everything it does. The `CLAUDE.md` bridge gives Claude Code Desktop project context from Agent Zero without any Anthropic binary running in your agent's process. Trade-off: no programmatic delegation, but full visibility and zero hidden binary execution.

**Why Ollama:** Model management, OpenAI-compatible API, one-command pulls. MLX backend switch (March 2026 preview) gives ~3x speedup on Apple Silicon. Speculative decoding coming but not merged yet.

**Why DSPy GEPA + Agent Lightning:** GEPA evolves prompts by reflecting on failures -- no weight changes, works with any model. Agent Lightning wraps around LangGraph with near-zero code changes, supports prompt optimization + optional RL. Complementary.

**Why ChromaDB:** Simplest vector store -- Python-native, local, no server. txtai may be added later for graph relationships and semantic SQL if needed, but ChromaDB with LLM-based novelty checking has been sufficient so far.

**Why `langchain-ollama` not `langchain-community`:** The community `ChatOllama` throws `NotImplementedError` on `bind_tools()`. Only `langchain-ollama` supports tool calling.

**Why MLX for fine-tuning:** Apple's framework, built for unified memory. Zero-copy CPU/GPU. LoRA/QLoRA up to ~30B fits in 64GB. Target MLP layers (`gate_proj`, `up_proj`, `down_proj`) -- research shows MLP-only LoRA matches MLP+attention while attention-only underperforms. Use rank 16-32.


---

## Fine-tuning reference

| Method | Where | Max model size | Speed | Cost |
|--------|-------|----------------|-------|------|
| MLX LoRA/QLoRA | Local (M2 Ultra 64GB) | ~30B (Q4 base) | 20-60 min for 7B | $0 |
| Unsloth + Colab | Google Colab (free T4) | ~10B | 30-60 min | $0 |
| Unsloth + Colab Pro | Google Colab (A100) | ~70B | 1-2 hrs | ~$10/mo |

Workflow: prepare JSONL to train LoRA to export GGUF to `ollama create agent-zero-custom -f Modelfile` to swap model in `.env`.


---

## Troubleshooting

**High memory usage or swapping to disk:** Ollama defaults num_ctx to 2048-4096 (the 256K figure is the model's max capability, not Ollama's default). If you override num_ctx too high (e.g. 262144), the KV cache alone can exceed 64GB. Fix: set `num_ctx=16384` in ChatOllama constructor for adequate headroom. Also set `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_FLASH_ATTENTION=1` via launchctl. After fix: ~30GB, 100% GPU.

**Ollama env vars not taking effect:** On macOS, Ollama runs as a launch agent. Regular `export` in your shell does not reach it. Use `launchctl setenv VAR_NAME "value"` then restart Ollama. This is required for `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_MAX_LOADED_MODELS`, etc.

**Tool calls returning JSON instead of executing:** Do NOT set `format="json"` on ChatOllama. It breaks `create_react_agent` tool execution.

**`bind_tools` not implemented:** Wrong import. Use `from langchain_ollama import ChatOllama`, not `langchain-community`.

**stream_mode="messages" hangs:** Known issue with `create_react_agent` + `ChatOllama` in langgraph-prebuilt 1.0.8. Use `stream_mode="updates"` instead.

**Models not using GPU:** Check `ollama ps` PROCESSOR column. Should show 100% GPU. Restart Ollama after launchctl env var changes.

**OOM with 70B:** Unload the main model first: `curl http://localhost:11434/api/generate -d '{"model": "gemma4:26b", "keep_alive": 0}'`. Keep `OLLAMA_KV_CACHE_TYPE=q8_0`.


---

## References

| Resource | URL |
|----------|-----|
| Gemma 4 (Ollama) | https://ollama.com/library/gemma4 |
| Gemma 4 (Google) | https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ |
| Gemma 4 local guide (Unsloth) | https://unsloth.ai/docs/models/gemma-4 |
| MLX on Apple Silicon | https://yage.ai/share/mlx-apple-silicon-en-20260331.html |
| MLX-LM fine-tuning | https://markaicode.com/run-fine-tune-llms-mac-mlx-lm/ |
| DSPy GEPA | https://dspy.ai/api/optimizers/GEPA/overview/ |
| GEPA GitHub | https://github.com/gepa-ai/gepa |
| Agent Lightning | https://microsoft.github.io/agent-lightning/latest/ |
| Agent Lightning GitHub | https://github.com/microsoft/agent-lightning |
| Unsloth | https://unsloth.ai/ |
| LangGraph docs | https://docs.langchain.com/oss/python/langgraph/overview |
| ChromaDB | https://www.trychroma.com/ |
| txtai | https://neuml.github.io/txtai/ |
| Crawl4AI | https://docs.crawl4ai.com/ |