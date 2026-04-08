"""Voice pipeline -- integration layer for WebSocket voice handling.

Manages the full audio pipeline for a single WebSocket connection:
receives PCM frames, runs VAD, checks wake word via openWakeWord,
transcribes via Whisper-MLX, and returns query text.

Echo cancellation: discards all frames while TTS is playing.
Backpressure: discards frames while processing (transcribing/executing).
"""

import asyncio
import struct

import numpy as np

from voice.vad import SileroVAD, COMPLETE, SPEAKING, FRAME_SAMPLES


class VoiceHandler:
    """Manages the full voice pipeline for a WebSocket connection.

    Usage:
        handler = VoiceHandler()
        while True:
            frame_bytes = await websocket.receive_bytes()
            result = await handler.handle_audio_frame(frame_bytes)
            if result is not None:
                # result is the transcribed query text
                pass
    """

    def __init__(self):
        self._vad = SileroVAD()
        self._tts_playing = False
        self._processing = False

    def set_tts_playing(self, playing: bool) -> None:
        """Set echo cancellation flag. Frames discarded while True."""
        self._tts_playing = playing

    def set_processing(self, processing: bool) -> None:
        """Set processing flag. Frames discarded while True."""
        self._processing = processing

    @property
    def vad_state(self) -> str:
        return self._vad.state

    async def handle_audio_frame(self, frame_bytes: bytes) -> str | None:
        """Process a 1024-byte PCM16 frame (512 samples at 16kHz).

        Returns transcribed query text when a complete utterance with
        wake word is detected, or None if still accumulating / discarded.

        Echo cancellation: returns None immediately if TTS is playing.
        Backpressure: returns None immediately if agent is processing.
        """
        # Echo cancellation
        if self._tts_playing:
            return None

        # Backpressure -- discard frames during processing
        if self._processing:
            return None

        # Convert PCM16 bytes to float32
        if len(frame_bytes) != FRAME_SAMPLES * 2:
            return None  # wrong frame size, skip

        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Feed VAD
        state = self._vad.process_frame(samples)

        if state != COMPLETE:
            return None

        # VAD detected end-of-speech -- check wake word and transcribe
        audio = self._vad.get_audio()
        self._vad.reset()

        if len(audio) == 0:
            return None

        print(f"[pipeline] VAD complete, {len(audio)} samples -- transcribing")

        # Transcribe with Whisper, then check for wake word via substring match.
        # openWakeWord requires a trained custom model for "hey zero" -- no
        # built-in model exists. Whisper + substring matching is the correct
        # approach for a personal single-user workstation.
        from voice.stt import transcribe, extract_after_wake_word
        text = await asyncio.to_thread(transcribe, audio)
        print(f"[pipeline] transcript: {text!r}")

        if not text:
            return None

        # Strip wake phrase from transcription
        query = extract_after_wake_word(text)
        return query

    def reset(self) -> None:
        """Reset pipeline state for next utterance."""
        self._vad.reset()
        self._tts_playing = False
        self._processing = False
