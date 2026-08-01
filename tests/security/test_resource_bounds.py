"""Proofs for trace-disk and streaming-capture resource bound failures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def test_append_must_enforce_configured_trace_token_budget(tmp_path: Path) -> None:
    store = TraceStore(
        tmp_path / "traces.jsonl",
        max_tokens=10,
        max_age_days=365,
    )

    assert store.append(_record("one")) is not None
    assert store.append(_record("two")) is not None

    assert store.stats().tokens <= 10


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


def _feed_all(observer: object, chunks: tuple[bytes, ...]) -> bytearray | None:
    async def drive() -> bytearray | None:
        for chunk in chunks:
            await observer.feed(chunk)  # type: ignore[attr-defined]
        return observer.finish()  # type: ignore[attr-defined,no-any-return]

    return asyncio.run(drive())


def test_sse_capture_obeys_capture_body_limit() -> None:
    """A streamed body over the ceiling is dropped, not buffered without bound.

    The observer is content-type agnostic on purpose: it copies bytes and lets
    ``CaptureManager`` parse later, so the same ceiling that bounds a
    non-streaming JSON body also bounds an unbounded ``text/event-stream``.
    """
    observer = proxy_module._ResponseObserver(limit=256)  # noqa: SLF001 - security boundary probe
    content = "A" * 1_024
    event = {
        "choices": [{"index": 0, "delta": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    chunks = tuple(
        f"data: {json.dumps(event)}\n\n".encode()[i : i + 64]
        for i in range(0, len(f"data: {json.dumps(event)}\n\n".encode()), 64)
    )

    captured = _feed_all(observer, chunks)

    assert captured is None
    assert observer.overflowed
    assert observer.size > observer.limit


def test_sse_capture_retains_a_stream_under_the_limit() -> None:
    """The ceiling must not cost ordinary streamed captures their body."""
    observer = proxy_module._ResponseObserver(limit=4_096)  # noqa: SLF001 - security boundary probe
    payload = b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'

    captured = _feed_all(observer, (payload,))

    assert captured == bytearray(payload)
    assert not observer.overflowed


def test_default_capture_body_limit_is_bounded() -> None:
    """The production default must be a finite ceiling, not "unlimited"."""
    assert 0 < proxy_module._MAX_CAPTURE_BODY_BYTES <= 64 * 1024 * 1024  # noqa: SLF001

