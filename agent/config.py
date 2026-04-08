"""Agent Zero configuration — loads .env, exposes model names and paths."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# Models
MAIN_MODEL: str = os.getenv("MAIN_MODEL", "gemma4:26b")
FAST_MODEL: str = os.getenv("FAST_MODEL", "gemma4:e2b")
REASONING_MODEL: str = os.getenv("REASONING_MODEL", "llama3.3:70b")
CODE_MODEL: str = os.getenv("CODE_MODEL", "qwen3-coder:30b")
VISION_MODEL: str = os.getenv("VISION_MODEL", "qwen3-vl:30b")
FINETUNE_MODEL: str = os.getenv("FINETUNE_MODEL", "gemma4:e4b")

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Paths (relative to project root)
AGENT_DB_PATH: str = str(_project_root / os.getenv("AGENT_DB_PATH", "data/agent_memory.db"))
CHROMA_DB_PATH: str = str(_project_root / os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
TXTAI_DB_PATH: str = str(_project_root / os.getenv("TXTAI_DB_PATH", "data/txtai_db"))
KNOWLEDGE_PATH: str = str(_project_root / os.getenv("KNOWLEDGE_PATH", "knowledge"))
KNOWLEDGE_CANON_PATH: str = str(_project_root / os.getenv("KNOWLEDGE_CANON_PATH", "knowledge_canon"))

# Agent inference
NUM_CTX: int = int(os.getenv("NUM_CTX", "65536"))
NUM_PREDICT: int = int(os.getenv("NUM_PREDICT", "2048"))

# Knowledge base context limits
KB_FILE_MAX_RATIO: float = float(os.getenv("KB_FILE_MAX_RATIO", "0.4"))
KB_FILE_MAX_TOKENS: int = int(os.getenv("KB_FILE_MAX_TOKENS", str(int(NUM_CTX * KB_FILE_MAX_RATIO))))
FAST_TEXT_MODEL: str = os.getenv("FAST_TEXT_MODEL", "gemma4:e4b")

# Semantic model role aliases
CHAT_MODEL: str = FAST_TEXT_MODEL      # e4b -- day-to-day interaction
KB_REFINE_MODEL: str = MAIN_MODEL      # 26b -- KB file editing
TAGGER_MODEL: str = FAST_MODEL         # e2b -- classification, summaries

# API
API_TOKEN: str = os.getenv("API_TOKEN", "")
API_PORT: int = int(os.getenv("API_PORT", "8900"))

# Voice
VOICE_MODEL: str = os.getenv("VOICE_MODEL", "gemma4:e4b")
VOICE_LANGUAGE: str = os.getenv("VOICE_LANGUAGE", "en")
VOICE_MIN_RMS: float = float(os.getenv("VOICE_MIN_RMS", "0.01"))
VOICE_CHUNK_SECONDS: int = int(os.getenv("VOICE_CHUNK_SECONDS", "3"))
VOICE_INPUT_GAIN: float = float(os.getenv("VOICE_INPUT_GAIN", "1.0"))
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "distil-large-v3")
TTS_VOICE: str = os.getenv("TTS_VOICE", "Samantha")
TTS_RATE: int = int(os.getenv("TTS_RATE", "175"))
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS: int = int(os.getenv("VAD_SILENCE_MS", "1000"))
MAX_SPEECH_SECONDS: int = int(os.getenv("MAX_SPEECH_SECONDS", "30"))

# UI
UI_DIR: str = str(_project_root / "ui")

# Bridge outputs
PROJECT_OUTPUTS_PATH: str = str(_project_root / os.getenv("PROJECT_OUTPUTS_PATH", "project_outputs"))
CLAUDE_MD_MAX_CHARS: int = int(os.getenv("CLAUDE_MD_MAX_CHARS", str(64 * 1024)))  # 64KB default
