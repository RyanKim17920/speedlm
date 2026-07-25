from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from speedlm.cli import main
from speedlm.config import RedactionConfig, SpeedLMConfig
from speedlm.gateway.capture import CaptureManager
from speedlm.gateway.sse import AssembledResponse
from speedlm.traces.redact import RedactionReport, Redactor
from speedlm.traces.store import (
    TraceRecord,
    TraceStore,
    format_redaction_summary,
    summarize_redactions,
)

SECRET = "sk-ant-api03-EXAMPLEFAKEKEY123456"


def _record(*, content: str = "ordinary request", rid: str = "trace-1") -> TraceRecord:
    return TraceRecord(
        id=rid,
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": content},
            {"role": "assistant", "content": "done"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=5,
        completion_tokens=1,
    )


def _response() -> AssembledResponse:
    return AssembledResponse(
        id="response-1",
        model="test-model",
        created=1_700_000_000.0,
        content="done",
        tool_calls=(),
        prompt_tokens=5,
        completion_tokens=1,
    )


async def _immediate_to_thread(function: Any, *args: Any) -> Any:
    return function(*args)


def test_gateway_capture_redacts_before_jsonl_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "to_thread", _immediate_to_thread)

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)
        capture.submit(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": f"use {SECRET}"}],
            },
            _response(),
            endpoint="/v1/chat/completions",
            timestamp=1_700_000_000.0,
        )
        await capture.drain()

        raw = store.path.read_bytes()
        assert SECRET.encode() not in raw
        assert b"<REDACTED:anthropic_key>" in raw

    asyncio.run(scenario())


def test_traces_import_redacts_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "home"))
    source = tmp_path / "import.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": f"use {SECRET}"}],
                "model": "test-model",
                "timestamp": 1_700_000_000.0,
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["traces", "import", str(source)]) == 0
    capsys.readouterr()
    stored = (tmp_path / "home" / "traces" / "traces.jsonl").read_bytes()
    assert SECRET.encode() not in stored
    assert b"<REDACTED:anthropic_key>" in stored


def test_redaction_failure_drops_capture_without_raising_or_writing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "to_thread", _immediate_to_thread)

    class FailingRedactor:
        def redact(self, value: Any) -> tuple[Any, RedactionReport]:
            del value
            raise RuntimeError("redactor unavailable")

    async def scenario() -> None:
        path = tmp_path / "traces.jsonl"
        store = TraceStore(path, redactor=FailingRedactor())
        capture = CaptureManager(store)

        # submit() is on the response-independent capture path and must not
        # propagate privacy-processing failures to the client request.
        capture.submit(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": SECRET}],
            },
            _response(),
            endpoint="/v1/chat/completions",
            timestamp=1_700_000_000.0,
        )
        await capture.drain()

        assert not path.exists()

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())
    assert "dropping trace after redaction failure (RuntimeError)" in caplog.text
    assert SECRET not in caplog.text


def test_redaction_can_be_disabled_via_config(tmp_path: Path) -> None:
    config = SpeedLMConfig(
        model="test-model",
        redaction=RedactionConfig(enabled=False),
    )
    store = TraceStore.from_config(
        tmp_path / "traces.jsonl",
        config.buffer,
        redaction=config.redaction,
    )

    report = store.append(_record(content=SECRET))

    assert report is not None
    assert report.total == 0
    assert SECRET.encode() in store.path.read_bytes()


def test_redaction_runs_exactly_once_per_record(tmp_path: Path) -> None:
    class CountingRedactor:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = Redactor()

        def redact(self, value: Any) -> tuple[Any, RedactionReport]:
            self.calls += 1
            return self.delegate.redact(value)

    redactor = CountingRedactor()
    store = TraceStore(tmp_path / "traces.jsonl", redactor=redactor)

    report = store.append(_record(content=SECRET))

    assert redactor.calls == 1
    assert report is not None
    assert report.counts == {"anthropic_key": 1}
    assert SECRET.encode() not in store.path.read_bytes()


def test_stats_and_import_summary_report_categories_without_values(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces.jsonl")
    reports = [
        store.append(
            _record(
                content=f"{SECRET} lives under /home/privacy-user/project",
                rid="redacted",
            )
        ),
        store.append(_record(rid="plain")),
    ]

    stats = store.stats()
    summary = summarize_redactions(reports)
    rendered = format_redaction_summary(summary)

    assert stats.redacted_records == 1
    assert stats.redaction_counts == {"anthropic_key": 1, "home_path": 1}
    assert summary.records == 1
    assert summary.counts == stats.redaction_counts
    assert rendered == (
        "redacted: 1 record(s), 2 replacement(s) "
        "[anthropic_key: 1, home_path: 1]"
    )
    assert SECRET not in rendered
    assert "privacy-user" not in rendered
