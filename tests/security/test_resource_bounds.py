"""Proofs for trace-disk and streaming-capture resource bound failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from speedlm.config import SamplingConfig
from speedlm.gateway import proxy as proxy_module
from speedlm.traces.normalize import normalize_record
from speedlm.traces.store import TraceRecord, TraceStore


def _record(rid: str, *, tokens: int = 8) -> TraceRecord:
    return TraceRecord(
        id=rid,
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=tokens,
        completion_tokens=0,
    )


@pytest.mark.xfail(
    strict=True,
    reason="TraceStore.append never invokes token/age pruning",
)
def test_append_must_enforce_configured_trace_token_budget(tmp_path: Path) -> None:
    store = TraceStore(
        tmp_path / "traces.jsonl",
        max_tokens=10,
        max_age_days=365,
    )

    assert store.append(_record("one")) is not None
    assert store.append(_record("two")) is not None

    assert store.stats().tokens <= 10


@pytest.mark.xfail(
    strict=True,
    reason="zero usage is trusted as measured and defeats prune accounting",
)
def test_zero_reported_usage_must_not_bypass_prune_budget(tmp_path: Path) -> None:
    large = "A" * 20_000
    record = normalize_record(
        {
            "id": "zero-usage",
            "model": "test-model",
            "messages": [
                {"role": "user", "content": large},
                {"role": "assistant", "content": large},
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
        defaults=SamplingConfig(),
    )
    store = TraceStore(
        tmp_path / "traces.jsonl",
        max_tokens=100,
        max_age_days=365,
    )

    assert store.append(record) is not None
    store.prune(now=record.timestamp)

    assert list(store.iter_records()) == []


@pytest.mark.xfail(
    strict=True,
    reason="SSE capture has no equivalent of the non-streaming byte ceiling",
)
def test_sse_capture_must_obey_capture_body_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_module, "_MAX_CAPTURE_BODY_BYTES", 256)
    observer = proxy_module._ResponseObserver(  # noqa: SLF001 - security boundary probe
        "/v1/chat/completions",
        "text/event-stream",
    )
    content = "A" * 1_024
    event = {
        "choices": [{"index": 0, "delta": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    observer.feed(f"data: {json.dumps(event)}\n\n".encode())
    assembled = observer.finish()

    assert assembled is None or assembled.content is None

