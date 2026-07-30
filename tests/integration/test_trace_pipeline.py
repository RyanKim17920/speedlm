from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from speedlm.cli import main
from speedlm.gateway.app import create_app
from speedlm.traces.store import TraceStore
from speedlm.training.masking import MaskPolicy, require_trainable_window
from speedlm.training.rows import prepare_training_row, training_row_from_trace
from speedlm.training.templates.chatml import ChatMLTemplate

_ASGIMessage = MutableMapping[str, Any]
_ASGIApp = Callable[
    [
        MutableMapping[str, Any],
        Callable[[], Awaitable[_ASGIMessage]],
        Callable[[_ASGIMessage], Awaitable[None]],
    ],
    Awaitable[None],
]


class _StreamingResponseStream(httpx.AsyncByteStream):
    def __init__(
        self,
        queue: asyncio.Queue[bytes | BaseException | None],
        app_task: asyncio.Task[None],
        disconnected: asyncio.Event,
    ) -> None:
        self._queue = queue
        self._app_task = app_task
        self._disconnected = disconnected

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                await self._app_task
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aclose(self) -> None:
        self._disconnected.set()
        if self._app_task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._app_task), timeout=1.0)
        except TimeoutError:
            self._app_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._app_task


class _StreamingASGITransport(httpx.AsyncBaseTransport):
    def __init__(self, app: _ASGIApp) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        scope: MutableMapping[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(key.lower(), value) for key, value in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
        }
        request_chunks = request.stream.__aiter__()
        request_complete = False
        response_started = asyncio.Event()
        response_complete = asyncio.Event()
        disconnected = asyncio.Event()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []

        async def receive() -> _ASGIMessage:
            nonlocal request_complete
            if not request_complete:
                try:
                    chunk = await request_chunks.__anext__()
                except StopAsyncIteration:
                    request_complete = True
                    return {"type": "http.request", "body": b"", "more_body": False}
                return {"type": "http.request", "body": chunk, "more_body": True}
            complete_task = asyncio.create_task(response_complete.wait())
            disconnect_task = asyncio.create_task(disconnected.wait())
            _, pending = await asyncio.wait(
                {complete_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            return {"type": "http.disconnect"}

        async def send(message: _ASGIMessage) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = list(message.get("headers", []))
                response_started.set()
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body and request.method != "HEAD" and not disconnected.is_set():
                    await queue.put(body)
                if not message.get("more_body", False):
                    response_complete.set()
                    await queue.put(None)

        async def run_app() -> None:
            try:
                await self._app(scope, receive, send)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put(exc)
                response_complete.set()
                response_started.set()
            finally:
                if not response_started.is_set():
                    response_started.set()
                if not response_complete.is_set() and not disconnected.is_set():
                    response_complete.set()
                    await queue.put(None)

        app_task = asyncio.create_task(run_app())
        await response_started.wait()
        return httpx.Response(
            status_code,
            headers=response_headers,
            stream=_StreamingResponseStream(queue, app_task, disconnected),
            request=request,
        )


class _CharacterTokenizer:
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _upstream_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        assert body["messages"] == [{"role": "user", "content": "Explain seams"}]
        return JSONResponse(
            {
                "id": "captured-integration",
                "created": 1_780_000_000,
                "model": "real-upstream-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Components must agree at their boundaries.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 7},
            }
        )

    return app


def test_capture_store_stats_and_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))

    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    store = TraceStore(tmp_path / "traces" / "traces.jsonl")

    async def scenario() -> None:
        upstream = httpx.AsyncClient(
            transport=_StreamingASGITransport(_upstream_app()),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            trace_store=store,
            upstream_client=upstream,
        )
        client = httpx.AsyncClient(
            transport=_StreamingASGITransport(gateway),
            base_url="http://gateway",
        )
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "requested-model",
                    "messages": [{"role": "user", "content": "Explain seams"}],
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "seed": 17,
                },
            )
            assert response.status_code == 200
            capture = gateway.state.capture
            assert capture is not None
            await capture.drain()
        finally:
            await client.aclose()
            await upstream.aclose()

    asyncio.run(scenario())

    stats = store.stats()
    assert (stats.count, stats.tokens, stats.measured_tokens) == (1, 11, 11)
    record = next(store.iter_records())
    row = training_row_from_trace(record)
    prepared = prepare_training_row(
        row,
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.FINAL_SPAN,
    )

    assert prepared.assistant_spans
    assert any(prepared.loss_mask)
    summary = require_trainable_window([prepared], total_seq_len=prepared.seq_len)
    assert summary.row_count == 1
    assert summary.retained_supervised_tokens > 0


def test_trace_import_round_trip_across_openai_and_proxy_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    source = tmp_path / "external.jsonl"
    openai_response = {
        "id": "openai-shaped",
        "created": 1_780_000_001,
        "model": "openai-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "An imported OpenAI response.",
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }
    proxy_capture = {
        "event": {
            "incoming_payload": {
                "body": {
                    "model": "proxy-model",
                    "messages": [{"role": "user", "content": "Captured prompt"}],
                    "temperature": 0.1,
                }
            },
            "upstream_result": {
                "data": {
                    "id": "proxy-shaped",
                    "created": 1_780_000_002,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Captured response.",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                }
            },
        }
    }
    source.write_text(
        "\n".join(json.dumps(record) for record in (openai_response, proxy_capture))
        + "\n",
        encoding="utf-8",
    )

    assert main(["traces", "import", str(source)]) == 0
    store = TraceStore(tmp_path / "traces" / "traces.jsonl")
    records = list(store.iter_records())
    assert {record.id for record in records} == {"openai-shaped", "proxy-shaped"}
    assert store.stats().count == 2

    prepared_rows = [
        prepare_training_row(
                training_row_from_trace(
                    record,
                    trust_untagged_assistant_messages=True,
                ),
            template=ChatMLTemplate(),
            tokenizer=_CharacterTokenizer(),
            mask_policy=MaskPolicy.FINAL_SPAN,
        )
        for record in records
    ]
    assert all(row.assistant_spans and any(row.loss_mask) for row in prepared_rows)
    summary = require_trainable_window(
        prepared_rows,
        total_seq_len=max(row.seq_len for row in prepared_rows),
    )
    assert summary.row_count == 2
    assert summary.rows_without_retained_supervision == 0
