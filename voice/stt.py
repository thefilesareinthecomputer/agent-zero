"""Speech-to-Text -- Whisper-MLX transcription for Apple Silicon.

Wraps lightning-whisper-mlx for GPU-accelerated transcription via Metal.
Model is loaded once and warmed up during server lifespan startup.

Cross-platform note: lightning-whisper-mlx requires macOS with Apple
Silicon. On Linux/Windows, substitute with faster-whisper or
openai-whisper with a compatible backend.
"""

import re

import numpy as np

from agent.config import WHISPER_MODEL, VOICE_LANGUAGE

_whisper = None

WAKE_PHRASE = "hey zero"
# Patterns to match: "hey zero", "hey, zero", "hey zero,"
_WAKE_PATTERN = re.compile(
    r"hey[,\s]*zero[,\s]*",
    re.IGNORECASE,
)


def load_whisper(model_size: str | None = None) -> None:
    """Preload the Whisper model into MLX memory.

    Call during server lifespan startup. Subsequent calls are no-ops.
    """
    global _whisper
    if _whisper is not None:
        return

    from lightning_whisper_mlx import LightningWhisperMLX
    _whisper = LightningWhisperMLX(
        model=model_size or WHISPER_MODEL,
        batch_size=12,
        quant=None,
    )


def warm_up() -> None:
    """Run a dummy transcription to pre-compile MLX kernels.

    Eliminates 2-4s cold-start penalty on the first real request.
    Must call load_whisper() first.
    """
    if _whisper is None:
        raise RuntimeError("Call load_whisper() before warm_up()")

    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    _whisper.transcribe(silence, language=VOICE_LANGUAGE)


def transcribe(audio: np.ndarray, language: str | None = None) -> str:
    """Transcribe float32 audio at 16kHz to text.

    Returns the transcription string. This is CPU/GPU-bound and
    takes ~300ms per utterance on M2 Ultra with distil-large-v3.
    Call via asyncio.to_thread() from async handlers.
    """
    if _whisper is None:
        raise RuntimeError("Call load_whisper() before transcribe()")

    result = _whisper.transcribe(
        audio,
        language=language or VOICE_LANGUAGE,
    )
    return result.get("text", "").strip()


def extract_after_wake_word(text: str) -> str | None:
    """Strip the wake phrase from transcription, return the remainder.

    Returns the query text after "hey zero", or None if the wake
    phrase is not found.
    """
    match = _WAKE_PATTERN.search(text)
    if match is None:
        return None

    remainder = text[match.end():].strip()
    return remainder if remainder else None
