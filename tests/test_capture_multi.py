from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from speedlm.gateway.capture import CaptureManager
from speedlm.traces.store import TraceStore


@pytest.fixture(autouse=True)
def _immediate_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def immediate(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def test_batched_completion_choices_are_correlated_exactly(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)
        adapter = capture.match("POST", "/v1/completions")
        assert adapter is not None
        request = {
            "model": "model",
            "prompt": ["prompt-a", "prompt-b"],
            "n": 2,
        }
        response = {
            "id": "cmpl-batch",
            "model": "model",
            "choices": [
                {"index": 3, "text": "b-1", "finish_reason": "stop"},
                {"index": 0, "text": "a-0", "finish_reason": "stop"},
                {"index": 2, "text": "b-0", "finish_reason": "stop"},
                {"index": 1, "text": "a-1", "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 12},
        }
        capture.submit_exchange(
            _json_bytes(request),
            _json_bytes(response),
            adapter=adapter,
            request_path="/v1/completions",
            method="POST",
            content_type="application/json",
            content_encoding="",
            timestamp=1_700_000_000.0,
        )
        await capture.drain()

        records = list(store.iter_records())
        assert len(records) == 4
        pairs = {
            (record.messages[0]["content"], record.messages[-1]["content"])
            for record in records
        }
        assert pairs == {
            ("prompt-a", "a-0"),
            ("prompt-a", "a-1"),
            ("prompt-b", "b-0"),
            ("prompt-b", "b-1"),
        }
        assert {record.id for record in records} == {
            f"cmpl-batch:choice:{index}" for index in range(4)
        }
        assert all(record.token_count_source == "estimated" for record in records)
        assert store.stats().total_dropped == 0

    asyncio.run(scenario())


def test_interleaved_stream_choices_and_gzip_json_are_captured(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)
        chat = capture.match("POST", "/v1/chat/completions")
        assert chat is not None
        request = _json_bytes(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "question"}],
                "n": 2,
            }
        )
        sse = (
            b'data: {"id":"chat-stream","model":"model","choices":['
            b'{"index":1,"delta":{"content":"second-"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'
            b'data: {"choices":[{"index":1,"delta":{"content":"choice"},'
            b'"finish_reason":"stop"},{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        )
        capture.submit_exchange(
            request,
            sse,
            adapter=chat,
            request_path="/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
            content_encoding="",
            timestamp=1_700_000_000.0,
        )

        compressed_response = gzip.compress(
            _json_bytes(
                {
                    "id": "chat-gzip",
                    "model": "model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "zipped"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        )
        capture.submit_exchange(
            request,
            compressed_response,
            adapter=chat,
            request_path="/v1/chat/completions",
            method="POST",
            content_type="application/json",
            content_encoding="gzip",
            timestamp=1_700_000_001.0,
        )
        await capture.drain()

        records = list(store.iter_records())
        assert {
            record.messages[-1]["content"] for record in records
        } == {"first", "second-choice", "zipped"}
        assert store.stats().total_dropped == 0

    asyncio.run(scenario())


def test_concurrent_exchange_burst_has_no_missing_or_crossed_pairs(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)
        adapter = capture.match("POST", "/v1/chat/completions")
        assert adapter is not None

        for index in reversed(range(32)):
            capture.submit_exchange(
                _json_bytes(
                    {
                        "model": "model",
                        "messages": [
                            {"role": "user", "content": f"input-{index}"}
                        ],
                    }
                ),
                _json_bytes(
                    {
                        "id": f"response-{index}",
                        "model": "model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"output-{index}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ),
                adapter=adapter,
                request_path="/v1/chat/completions",
                method="POST",
                content_type="application/json",
                content_encoding="",
                timestamp=1_700_000_000.0 + index,
            )
        await capture.drain()

        records = list(store.iter_records())
        assert len(records) == 32
        assert {
            (
                record.id,
                record.messages[0]["content"],
                record.messages[-1]["content"],
            )
            for record in records
        } == {
            (f"response-{index}", f"input-{index}", f"output-{index}")
            for index in range(32)
        }
        assert store.stats().total_dropped == 0

    asyncio.run(scenario())
