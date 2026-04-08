"""Voice Activity Detection -- Silero-VAD wrapper with state machine.

Processes 512-sample PCM frames (16kHz, 32ms each). Detects speech
segments and accumulates audio until trailing silence confirms
end-of-utterance.

Silero-VAD v5 requires exactly 512 samples at 16kHz per frame.
The AudioWorklet in the browser must buffer to this size before sending.
"""

import time

import numpy as np

from agent.config import VAD_THRESHOLD, VAD_SILENCE_MS, MAX_SPEECH_SECONDS

# States
IDLE = "idle"
SPEAKING = "speaking"
TRAILING_SILENCE = "trailing_silence"
COMPLETE = "complete"

# Frame size: 512 samples at 16kHz = 32ms
FRAME_SAMPLES = 512
SAMPLE_RATE = 16000
MIN_SPEECH_MS = 250


class SileroVAD:
    """Silero-VAD state machine for speech segmentation.

    Feed 512-sample float32 frames via process_frame(). When state
    transitions to COMPLETE, call get_audio() to retrieve the
    accumulated speech buffer, then reset().
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        silence_ms: int = VAD_SILENCE_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
        max_speech_s: int = MAX_SPEECH_SECONDS,
    ):
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_s = max_speech_s

        self._model = None
        self._state = IDLE
        self._audio_buffer: list[np.ndarray] = []
        self._speech_start: float | None = None
        self._silence_start: float | None = None

    def _load_model(self) -> None:
        """Lazy-load the Silero-VAD ONNX model."""
        if self._model is not None:
            return
        import silero_vad
        self._model = silero_vad.load_silero_vad()

    def process_frame(self, frame: np.ndarray) -> str:
        """Process a 512-sample float32 frame. Returns current state.

        States: "idle", "speaking", "trailing_silence", "complete".
        """
        self._load_model()

        if len(frame) != FRAME_SAMPLES:
            raise ValueError(
                f"Expected {FRAME_SAMPLES} samples, got {len(frame)}"
            )

        import torch
        tensor = torch.from_numpy(frame).float()
        confidence = self._model(tensor, SAMPLE_RATE).item()
        is_speech = confidence > self.threshold
        now = time.monotonic()

        if self._state == IDLE:
            if is_speech:
                self._state = SPEAKING
                self._speech_start = now
                self._silence_start = None
                self._audio_buffer.append(frame.copy())

        elif self._state == SPEAKING:
            self._audio_buffer.append(frame.copy())

            # Check max duration cap
            elapsed_s = now - self._speech_start
            if elapsed_s >= self.max_speech_s:
                self._state = COMPLETE
                return self._state

            if not is_speech:
                self._state = TRAILING_SILENCE
                self._silence_start = now

        elif self._state == TRAILING_SILENCE:
            self._audio_buffer.append(frame.copy())

            # Check max duration cap
            elapsed_s = now - self._speech_start
            if elapsed_s >= self.max_speech_s:
                self._state = COMPLETE
                return self._state

            if is_speech:
                # Speech resumed
                self._state = SPEAKING
                self._silence_start = None
            else:
                silence_elapsed_ms = (now - self._silence_start) * 1000
                if silence_elapsed_ms >= self.silence_ms:
                    # Check minimum speech duration
                    speech_duration_ms = (now - self._speech_start) * 1000
                    if speech_duration_ms >= self.min_speech_ms:
                        self._state = COMPLETE
                    else:
                        # Transient -- discard and reset
                        self._state = IDLE
                        self._audio_buffer.clear()
                        self._speech_start = None
                        self._silence_start = None

        return self._state

    def get_audio(self) -> np.ndarray:
        """Return accumulated speech audio as a single float32 array."""
        if not self._audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_buffer)

    def reset(self) -> None:
        """Reset state machine for next utterance."""
        self._state = IDLE
        self._audio_buffer.clear()
        self._speech_start = None
        self._silence_start = None

    @property
    def state(self) -> str:
        return self._state
