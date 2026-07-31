"""Unit tests for offline activation extraction (no GPU required).

Validates that the offline JSONL row renderer produces the production schema,
that the zero-row guard fires with a useful message, and that the health-check
routines behave correctly with mocked subprocesses.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

from speedlm.activation_capture.offline_extract import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    _wait_for_health,
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
        """Only 'conversations' key present -- no id, tools, etc."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "test")

        line = p.read_text(encoding="utf-8").strip()
        row = json.loads(line)

        assert list(row.keys()) == ["conversations"]

    def test_ensure_ascii_false(self, tmp_path: Path) -> None:
        """Unicode content is preserved, not escaped."""
        p = tmp_path / "speculators-conversations.jsonl"
        _write_conversation_jsonl(p, "hello", assistant_response="cafe")

        text = p.read_text(encoding="utf-8")
        assert "cafe" in text
        assert "\\u00e9" not in text


class TestDefaultGpuMemoryUtilization:
    """Verify the GPU memory utilization constant is reasonable."""

    def test_default_is_in_range(self) -> None:
        assert 0 < DEFAULT_GPU_MEMORY_UTILIZATION <= 1

    def test_default_is_half(self) -> None:
        assert DEFAULT_GPU_MEMORY_UTILIZATION == 0.5


class TestWaitForHealth:
    """Verify _wait_for_health behaviour with mocked subprocesses."""

    def test_short_circuits_on_dead_process(self, tmp_path: Path) -> None:
        """A subprocess that has already exited should raise immediately,
        not burn the full timeout."""
        log_path = tmp_path / "vllm.log"
        log_path.write_text("engine started\nengine crashed\n", encoding="utf-8")

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = 1  # Already dead

        t0 = time.monotonic()
        with pytest.raises(RuntimeError) as exc_info:
            _wait_for_health(
                "http://127.0.0.1:9999/health",
                proc,
                timeout=999.0,
                log_path=log_path,
            )
        elapsed = time.monotonic() - t0

        # Should return essentially instantly (< 0.1s), not burn 999s.
        assert elapsed < 0.5

        # The error message should mention the exit code.
        assert "exited with code 1" in str(exc_info.value)

        # The log tail should be included.
        assert "engine crashed" in str(exc_info.value)

    def test_timeout_includes_log_tail(self, tmp_path: Path) -> None:
        """When the timeout is reached, the error includes the log tail."""
        log_path = tmp_path / "vllm.log"
        log_path.write_text(
            "line 1\nline 2\nCRITICAL: OOM\n", encoding="utf-8"
        )

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None  # Still alive

        t0 = time.monotonic()
        # Patch urllib so every attempt fails
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ), pytest.raises(TimeoutError) as exc_info:
            _wait_for_health(
                "http://127.0.0.1:9999/health",
                proc,
                timeout=0.3,
                poll=0.1,
                log_path=log_path,
            )
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.2  # Should have waited roughly the timeout
        assert "CRITICAL: OOM" in str(exc_info.value)
        assert "did not become ready" in str(exc_info.value)

    def test_succeeds_on_healthy_response(self, tmp_path: Path) -> None:
        """A 200 response returns without raising."""
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None

        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            _wait_for_health(
                "http://127.0.0.1:9999/health",
                proc,
                timeout=5.0,
                log_path=None,
            )

    def test_no_log_path_omits_tail(self, tmp_path: Path) -> None:
        """When log_path is None, the error does not reference a log file."""
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ), pytest.raises(TimeoutError) as exc_info:
            _wait_for_health(
                "http://127.0.0.1:9999/health",
                proc,
                timeout=0.3,
                poll=0.1,
                log_path=None,
            )

        # No "log" markers in the message
        msg = str(exc_info.value)
        assert "vLLM log" not in msg