"""Text-to-Speech -- macOS `say` command wrapper with sentence chunking.

Synthesizes text to PCM16 audio at 16kHz via macOS `say`. Supports
sentence-level chunking for streaming: each sentence is synthesized
independently so the first sentence streams while later ones generate.

Cross-platform alternatives:
  - Linux: piper-tts (fast, local, ONNX-based)
  - Windows: pyttsx3 or Windows SAPI
  - Any: Coqui TTS (heavier, GPU-optional)
"""

import asyncio
import re
import struct
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from agent.config import TTS_VOICE, TTS_RATE

# WAV header is 44 bytes for standard PCM
_WAV_HEADER_SIZE = 44

# Split on sentence boundaries, keeping the delimiter with the preceding text
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries.

    Handles edge cases: abbreviations (Mr., Dr., etc.) are not perfect
    but acceptable for TTS chunking where minor splits are harmless.
    """
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


async def synthesize_sentence(
    text: str,
    voice: str | None = None,
    rate: int | None = None,
) -> bytes:
    """Synthesize a single text string to PCM16 bytes at 16kHz.

    Returns raw PCM16 signed little-endian mono bytes (no WAV header).
    Uses macOS `say` via subprocess.
    """
    voice = voice or TTS_VOICE
    rate = rate or TTS_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = tmp.name

    proc = await asyncio.create_subprocess_exec(
        "say",
        "-v", voice,
        "--rate", str(rate),
        "--data-format=LEI16@16000",
        "-o", tmp_path,
        text,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    try:
        wav_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if len(wav_bytes) <= _WAV_HEADER_SIZE:
        return b""

    # Strip WAV header, return raw PCM
    return wav_bytes[_WAV_HEADER_SIZE:]


async def stream_tts(
    full_text: str,
    voice: str | None = None,
    rate: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield PCM16 bytes per sentence for streaming playback.

    Each sentence is synthesized independently. The first sentence
    streams while subsequent ones are still being generated.
    Latency to first audio: ~200ms for short sentences.
    """
    sentences = split_sentences(full_text)
    if not sentences:
        return

    for sentence in sentences:
        pcm = await synthesize_sentence(sentence, voice=voice, rate=rate)
        if pcm:
            yield pcm
