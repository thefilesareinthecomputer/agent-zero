"""Tests for fine_tuning/capture.py -- training data JSONL writer."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from fine_tuning.capture import log_turn


_VALID_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]


class TestLogTurn:
    def test_writes_one_line_for_text_response(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", _VALID_MESSAGES)

        logs = list((tmp_path / "training_logs").glob("*.jsonl"))
        assert len(logs) == 1
        lines = logs[0].read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["thread_id"] == "thread-1"
        assert record["provider"] == "local"
        assert record["model"] == "gemma4:e4b"
        assert record["agent"] == "fast"
        assert record["messages"] == _VALID_MESSAGES

    def test_skips_empty_message_list(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", [])

        log_dir = tmp_path / "training_logs"
        assert not log_dir.exists() or not list(log_dir.glob("*.jsonl"))

    def test_skips_tool_call_turn(self, tmp_path):
        """Last message with list content (tool calls) should not be logged."""
        messages = [
            {"role": "user", "content": "run a command"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "run_shell_command"}]},
        ]
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", messages)

        log_dir = tmp_path / "training_logs"
        assert not log_dir.exists() or not list(log_dir.glob("*.jsonl"))

    def test_skips_empty_assistant_content(self, tmp_path):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "   "},
        ]
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", messages)

        log_dir = tmp_path / "training_logs"
        assert not log_dir.exists() or not list(log_dir.glob("*.jsonl"))

    def test_skips_non_assistant_last_message(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", messages)

        log_dir = tmp_path / "training_logs"
        assert not log_dir.exists() or not list(log_dir.glob("*.jsonl"))

    def test_date_partitioned(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "cloud", "gemma4:26b", "heavy", _VALID_MESSAGES)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / "training_logs" / f"{today}.jsonl"
        assert log_file.exists()

    def test_provider_field_written(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "cloud", "gemma4:26b", "heavy", _VALID_MESSAGES)

        logs = list((tmp_path / "training_logs").glob("*.jsonl"))
        record = json.loads(logs[0].read_text().strip())
        assert record["provider"] == "cloud"

    def test_timestamp_is_iso8601(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", _VALID_MESSAGES)

        logs = list((tmp_path / "training_logs").glob("*.jsonl"))
        record = json.loads(logs[0].read_text().strip())
        # Should parse without error
        datetime.fromisoformat(record["timestamp"])

    def test_multiple_turns_appended(self, tmp_path):
        with patch("fine_tuning.capture._DATA_DIR", tmp_path):
            log_turn("thread-1", "local", "gemma4:e4b", "fast", _VALID_MESSAGES)
            log_turn("thread-2", "cloud", "gemma4:26b", "heavy", _VALID_MESSAGES)

        logs = list((tmp_path / "training_logs").glob("*.jsonl"))
        lines = logs[0].read_text().strip().splitlines()
        assert len(lines) == 2

    def test_thread_safe_concurrent_writes(self, tmp_path):
        """Concurrent calls must not corrupt the JSONL file."""
        import fine_tuning.capture as capture_mod

        original = capture_mod._DATA_DIR
        capture_mod._DATA_DIR = tmp_path
        try:
            errors = []

            def write():
                try:
                    log_turn("t", "local", "m", "fast", _VALID_MESSAGES)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=write) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            logs = list((tmp_path / "training_logs").glob("*.jsonl"))
            lines = logs[0].read_text().strip().splitlines()
            assert len(lines) == 20
            for line in lines:
                json.loads(line)  # each line must be valid JSON
        finally:
            capture_mod._DATA_DIR = original
