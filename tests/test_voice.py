"""Tests for the voice engine modules."""

import platform
import struct

import numpy as np
import pytest

from voice.vad import SileroVAD, IDLE, SPEAKING, COMPLETE, FRAME_SAMPLES
from voice.stt import extract_after_wake_word
from voice.tts import split_sentences


# -- PCM conversion --

class TestPCMConversion:
    def test_pcm16_float32_roundtrip(self):
        """Convert float32 -> PCM16 -> float32, verify approximate match."""
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm16 = (original * 32768).clip(-32768, 32767).astype(np.int16)
        recovered = pcm16.astype(np.float32) / 32768.0
        np.testing.assert_allclose(original, recovered, atol=1e-4)


# -- VAD --

class TestVAD:
    def test_silence_stays_idle(self):
        """Feeding silence should keep VAD in IDLE state."""
        vad = SileroVAD()
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for _ in range(50):
            state = vad.process_frame(silence)
        assert state == IDLE

    def test_wrong_frame_size_raises(self):
        """VAD should reject frames that are not exactly 512 samples."""
        vad = SileroVAD()
        with pytest.raises(ValueError, match="Expected 512"):
            vad.process_frame(np.zeros(128, dtype=np.float32))

    def test_reset_clears_state(self):
        """Reset should return VAD to IDLE with empty buffer."""
        vad = SileroVAD()
        vad._state = SPEAKING
        vad._audio_buffer = [np.zeros(512, dtype=np.float32)]
        vad.reset()
        assert vad.state == IDLE
        assert len(vad._audio_buffer) == 0

    def test_get_audio_empty(self):
        """get_audio on fresh VAD returns empty array."""
        vad = SileroVAD()
        audio = vad.get_audio()
        assert len(audio) == 0

    def test_max_duration_forces_complete(self):
        """Speech exceeding max_speech_s should force COMPLETE."""
        vad = SileroVAD(max_speech_s=1)
        # Manually set state as if speaking for >1 second
        vad._state = SPEAKING
        vad._speech_start = 0  # ancient timestamp
        vad._audio_buffer = [np.zeros(512, dtype=np.float32)]
        # Process another frame -- should trigger max duration
        import time
        # Since speech_start is 0 and now is >> 1, it should complete
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        state = vad.process_frame(frame)
        assert state == COMPLETE


# -- Wake word --

class TestWakeWord:
    def test_found(self):
        assert extract_after_wake_word("hey zero what time is it") == "what time is it"

    def test_found_comma(self):
        assert extract_after_wake_word("hey, zero what time") == "what time"

    def test_found_trailing_comma(self):
        result = extract_after_wake_word("hey zero, what time is it")
        assert result == "what time is it"

    def test_not_found(self):
        assert extract_after_wake_word("what time is it") is None

    def test_case_insensitive(self):
        assert extract_after_wake_word("Hey Zero tell me a joke") == "tell me a joke"

    def test_only_wake_word(self):
        """Wake word with no query after it returns None."""
        assert extract_after_wake_word("hey zero") is None

    def test_empty_string(self):
        assert extract_after_wake_word("") is None


# -- TTS sentence splitting --

class TestSentenceSplit:
    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = split_sentences("First. Second. Third.")
        assert result == ["First.", "Second.", "Third."]

    def test_question_and_exclamation(self):
        result = split_sentences("Really? Yes! Done.")
        assert result == ["Really?", "Yes!", "Done."]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_no_punctuation(self):
        assert split_sentences("no ending punctuation") == ["no ending punctuation"]


# -- TTS integration (macOS only) --

class TestTTSIntegration:
    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_synthesize_produces_bytes(self):
        import asyncio
        from voice.tts import synthesize_sentence
        pcm = asyncio.run(synthesize_sentence("Hello.", voice="Samantha", rate=200))
        assert isinstance(pcm, bytes)
        assert len(pcm) > 0


# -- Echo suppression --

class TestEchoSuppression:
    def test_frames_discarded_when_tts_playing(self):
        from voice.pipeline import VoiceHandler
        handler = VoiceHandler()
        handler.set_tts_playing(True)
        # Should return None immediately (frame discarded)
        import asyncio
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
        result = asyncio.run(handler.handle_audio_frame(frame))
        assert result is None
