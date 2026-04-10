"""Training data capture -- appends completed chat turns to date-partitioned JSONL.

Each record is one completed (user, assistant) exchange tagged with the active
provider (local or cloud). Cloud-tagged records serve as teacher outputs for
knowledge distillation in Phase 5 (fine-tuning).

File layout:
    data/training_logs/YYYY-MM-DD.jsonl  -- one JSON line per turn

Schema:
    {
        "timestamp": ISO8601,
        "thread_id": str,
        "provider": "local" | "cloud",
        "model": str,
        "agent": "fast" | "heavy",
        "messages": [
            {"role": "user", "content": str},
            {"role": "assistant", "content": str}
        ]
    }
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.config import AGENT_DB_PATH

_DATA_DIR = Path(AGENT_DB_PATH).parent
_write_lock = threading.Lock()


def log_turn(
    thread_id: str,
    provider: str,
    model: str,
    agent_mode: str,
    messages: list[dict],
) -> None:
    """Append one completed turn to today's JSONL file.

    Only writes when the last message is a non-empty assistant text response.
    Pure tool-call turns (where content is a list, not a string) are skipped
    -- they are not useful for instruction fine-tuning.
    """
    if not messages:
        return
    last = messages[-1]
    if last.get("role") != "assistant":
        return
    content = last.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "provider": provider,
        "model": model,
        "agent": agent_mode,
        "messages": messages,
    }

    log_dir = _DATA_DIR / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    with _write_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
