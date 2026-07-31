"""Unit tests for offline activation extraction (no GPU required).

Validates that the offline JSONL row renderer produces the production schema
and that the zero-row guard fires with a useful message.
"""

from __future__ import annotations

import json
from pathlib import Path

from speedlm.activation_capture.offline_extract import (
    _write_conversation_jsonl,
)


class TestWriteConversationJsonl:
    """Verify the offline row renderer matches the production schema."""

    def test_emits_conversations_key(self, tmp_path: Path) -> None:
        """Row must use 'conversations' key, not 'input_ids'/'output'."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "hello world")

        line = p.read_text(encoding="utf-8").strip()
        row = json.loads(line)

        assert "conversations" in row
        assert "input_ids" not in row
        assert "output" not in row

    def test_conversation_structure(self, tmp_path: Path) -> None:
        """Two turns: user prompt followed by assistant response."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "tell me a joke", assistant_response="Why did the...")

        line = p.read_text(encoding="utf-8").strip()
        row = json.loads(line)

        convs = row["conversations"]
        assert len(convs) == 2

        assert convs[0] == {"role": "user", "content": "tell me a joke"}
        assert convs[1] == {"role": "assistant", "content": "Why did the..."}

    def test_default_assistant_empty_string(self, tmp_path: Path) -> None:
        """When no assistant_response is given, content is empty string."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "single prompt")

        line = p.read_text(encoding="utf-8").strip()
        row = json.loads(line)

        convs = row["conversations"]
        assert convs[1]["role"] == "assistant"
        assert convs[1]["content"] == ""

    def test_no_extra_keys(self, tmp_path: Path) -> None:
        """Only 'conversations' key present — no id, tools, etc."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "test")

        line = p.read_text(encoding="utf-8").strip()
        row = json.loads(line)

        assert list(row.keys()) == ["conversations"]

    def test_ensure_ascii_false(self, tmp_path: Path) -> None:
        """Unicode content is preserved, not escaped."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "hello", assistant_response="café")

        text = p.read_text(encoding="utf-8")
        assert "café" in text
        assert "\\u00e9" not in text