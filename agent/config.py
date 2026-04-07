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

# Voice
VOICE_LANGUAGE: str = os.getenv("VOICE_LANGUAGE", "en")
VOICE_MIN_RMS: float = float(os.getenv("VOICE_MIN_RMS", "0.01"))
VOICE_CHUNK_SECONDS: int = int(os.getenv("VOICE_CHUNK_SECONDS", "3"))
VOICE_INPUT_GAIN: float = float(os.getenv("VOICE_INPUT_GAIN", "1.0"))
