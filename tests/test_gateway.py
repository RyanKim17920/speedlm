from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from speedlm.gateway import proxy as proxy_module
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.app import create_app
from speedlm.gateway.capture import CaptureManager
from speedlm.gateway.process import (
    ProcessError,
    VLLMProcess,
    build_vllm_argv,
)
from speedlm.gateway.sse import AssembledResponse, SSEAssembler, parse_json_response
from speedlm.traces.store import TraceRecord, TraceStore

ASGIMessage = MutableMapping[str, Any]
ASGIApp = Callable[
    [
        MutableMapping[str, Any],
        Callable[[], Awaitable[ASGIMessage]],
        Callable[[ASGIMessage], Awaitable[None]],
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


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """ASGI test transport that does not combine response body chunks."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise TypeError("streaming ASGI transport requires an async request stream")
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

        async def receive() -> ASGIMessage:
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
            done, pending = await asyncio.wait(
                {complete_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            del done
            for task in pending:
                task.cancel()
            return {"type": "http.disconnect"}

        async def send(message: ASGIMessage) -> None:
            nonlocal status_code, response_headers
            message_type = message["type"]
            if message_type == "http.response.start":
                status_code = int(message["status"])
                response_headers = list(message.get("headers", []))
                response_started.set()
                return
            if message_type != "http.response.body":
                return
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
        stream = _StreamingResponseStream(queue, app_task, disconnected)
        return httpx.Response(
            status_code,
            headers=response_headers,
            stream=stream,
            request=request,
        )


def _fake_upstream(state: dict[str, Any]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/passthrough")
    async def passthrough() -> JSONResponse:
        return JSONResponse(
            {"answer": 42},
            status_code=201,
            headers={"x-upstream": "yes"},
        )

    @app.get("/v1/stream")
    async def stream() -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"early"
            await state["allow_stream_finish"].wait()
            state["stream_finished"].set()
            yield b"late"

        return StreamingResponse(chunks(), media_type="application/octet-stream")

    @app.get("/v1/concurrent")
    async def concurrent() -> dict[str, bool]:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        if state["active"] == 2:
            state["both_active"].set()
        await state["release_concurrent"].wait()
        state["active"] -= 1
        return {"ok": True}

    @app.post("/v1/upload")
    async def upload(request: Request) -> dict[str, int]:
        size = 0
        async for chunk in request.stream():
            if chunk:
                size += len(chunk)
                state["upload_started"].set()
        return {"size": size}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.json()
        if not body.get("stream"):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "hello",
                "tool_calls": [],
            }
            if isinstance(body.get("test_reasoning"), str):
                message["reasoning"] = body["test_reasoning"]
            payload = {
                "id": "chat-nonstream",
                "created": 1_700_000_000,
                "model": "upstream-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop",
                        "stop_reason": 128009,
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
            if body.get("omit_usage"):
                payload.pop("usage")
            return JSONResponse(
                payload
            )

        async def events() -> AsyncIterator[bytes]:
            reasoning = body.get("test_reasoning")
            reasoning_parts = (
                [reasoning[: len(reasoning) // 2], reasoning[len(reasoning) // 2 :]]
                if isinstance(reasoning, str)
                else [None, None]
            )
            chunks = [
                {
                    "id": "chat-stream",
                    "created": 1_700_000_001,
                    "model": "upstream-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "call ",
                                "reasoning": reasoning_parts[0],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "tool-1",
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "arguments": '{"city":"',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "done",
                                "reasoning": reasoning_parts[1],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'Paris"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                            "stop_reason": "end",
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 5},
                },
            ]
            for event in chunks:
                encoded = f"data: {json.dumps(event)}\n\n".encode()
                midpoint = len(encoded) // 2
                yield encoded[:midpoint]
                await asyncio.sleep(0)
                yield encoded[midpoint:]
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/disconnect")
    async def disconnect() -> StreamingResponse:
        async def forever() -> AsyncIterator[bytes]:
            try:
                yield b"started"
                while True:
                    await asyncio.sleep(0.01)
                    yield b"."
            finally:
                state["cancelled"].set()

        return StreamingResponse(forever(), media_type="application/octet-stream")

    @app.get("/metrics")
    async def metrics() -> dict[str, bool]:
        state["admin_called"] = True
        return {"unsafe": True}

    return app


async def _clients(
    *,
    store: TraceStore | None = None,
) -> tuple[httpx.AsyncClient, httpx.AsyncClient, dict[str, Any], FastAPI]:
    state: dict[str, Any] = {
        "allow_stream_finish": asyncio.Event(),
        "stream_finished": asyncio.Event(),
        "upload_started": asyncio.Event(),
        "both_active": asyncio.Event(),
        "release_concurrent": asyncio.Event(),
        "cancelled": asyncio.Event(),
        "active": 0,
        "max_active": 0,
        "admin_called": False,
    }
    upstream_app = _fake_upstream(state)
    upstream = httpx.AsyncClient(
        transport=StreamingASGITransport(upstream_app),
        base_url="http://upstream",
    )
    gateway_app = create_app(
        "http://upstream",
        trace_store=store,
        upstream_client=upstream,
    )
    client = httpx.AsyncClient(
        transport=StreamingASGITransport(gateway_app),
        base_url="http://gateway",
    )
    return client, upstream, state, gateway_app


async def _close_clients(
    client: httpx.AsyncClient,
    upstream: httpx.AsyncClient,
    gateway: FastAPI,
) -> None:
    capture = gateway.state.capture
    if capture is not None:
        await capture.drain()
    await client.aclose()
    await upstream.aclose()


def test_non_streaming_passthrough_preserves_body_and_status(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.get("/v1/passthrough")
            assert response.status_code == 201
            assert response.json() == {"answer": 42}
            assert response.headers["x-upstream"] == "yes"
        finally:
            await _close_clients(client, upstream, gateway)
        assert not store.path.exists()

    asyncio.run(scenario())


def test_streaming_passes_early_chunk_before_upstream_finishes() -> None:
    async def scenario() -> None:
        client, upstream, state, gateway = await _clients()
        try:
            async with client.stream("GET", "/v1/stream") as response:
                chunks = response.aiter_raw()
                assert await anext(chunks) == b"early"
                assert not state["stream_finished"].is_set()
                state["allow_stream_finish"].set()
                assert b"".join([chunk async for chunk in chunks]) == b"late"
            assert state["stream_finished"].is_set()
        finally:
            await _close_clients(client, upstream, gateway)

    asyncio.run(scenario())


def test_request_body_streams_upstream() -> None:
    async def scenario() -> None:
        client, upstream, state, gateway = await _clients()
        try:
            async def body() -> AsyncIterator[bytes]:
                yield b"first"
                await asyncio.wait_for(state["upload_started"].wait(), timeout=1.0)
                yield b"second"

            response = await client.post("/v1/upload", content=body())
            assert response.json() == {"size": 11}
        finally:
            await _close_clients(client, upstream, gateway)

    asyncio.run(scenario())


def test_concurrent_requests_are_not_serialized() -> None:
    async def scenario() -> None:
        client, upstream, state, gateway = await _clients()
        try:
            requests = [
                asyncio.create_task(client.get("/v1/concurrent"))
                for _ in range(2)
            ]
            await asyncio.wait_for(state["both_active"].wait(), timeout=1.0)
            state["release_concurrent"].set()
            responses = await asyncio.gather(*requests)
            assert all(response.status_code == 200 for response in responses)
            assert state["max_active"] == 2
        finally:
            await _close_clients(client, upstream, gateway)

    asyncio.run(scenario())


def test_sse_reconstruction_with_multichunk_tool_call() -> None:
    assembler = SSEAssembler("/v1/chat/completions")
    events = [
        b'data: {"id":"x","model":"m","choices":[{"index":0,"delta":',
        b'{"content":"hi","reasoning":"check ","tool_calls":'
        b'[{"index":0,"id":"tc","type":"function",',
        b'"function":{"name":"f","arguments":"{\\"x\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"reasoning":"this",'
        b'"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},'
        b'"finish_reason":"tool_calls","stop_reason":7}],"usage":{"prompt_tokens":4,',
        b'"completion_tokens":2}}\n\ndata: [DONE]\n\n',
    ]
    for chunk in events:
        assembler.feed(chunk)
    result = assembler.finish()
    assert result.content == "hi"
    assert result.tool_calls == (
        {
            "id": "tc",
            "type": "function",
            "function": {"name": "f", "arguments": '{"x":1}'},
        },
    )
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert result.reasoning_content == "check this"
    assert result.reasoning_field == "reasoning"
    assert result.finish_reason == "tool_calls"
    assert result.stop_reason == 7
    assert assembler.done


def test_sse_without_usage_keeps_token_counts_unknown() -> None:
    assembler = SSEAssembler("/v1/chat/completions")
    assembler.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    result = assembler.finish()

    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert assembler.valid


def test_json_response_without_usage_keeps_token_counts_unknown() -> None:
    result = parse_json_response(
        json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "thinking": "carefully",
                        },
                        "finish_reason": "stop",
                        "stop_reason": "eos",
                    }
                ]
            }
        ).encode(),
        "/v1/chat/completions",
    )

    assert result is not None
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.reasoning_content == "carefully"
    assert result.reasoning_field == "thinking"
    assert result.finish_reason == "stop"
    assert result.stop_reason == "eos"


def test_client_disconnect_cancels_upstream() -> None:
    async def scenario() -> None:
        client, upstream, state, gateway = await _clients()
        try:
            async with client.stream("GET", "/v1/disconnect") as response:
                assert await anext(response.aiter_raw()) == b"started"
            await asyncio.wait_for(state["cancelled"].wait(), timeout=1.0)
        finally:
            await _close_clients(client, upstream, gateway)

    asyncio.run(scenario())


def test_admin_routes_are_blocked() -> None:
    async def scenario() -> None:
        client, upstream, state, gateway = await _clients()
        try:
            for path in (
                "/metrics",
                "/docs",
                "/reset_prefix_cache",
                "/v1",
                "/v1/load_lora_adapter",
            ):
                assert (await client.get(path)).status_code == 404
            for path in (
                "/abort_requests",
                "/collective_rpc",
                "/pause",
                "/reset_mm_cache",
                "/scale_elastic_ep",
                "/start_profile",
                "/update_weights",
            ):
                assert (await client.post(path)).status_code == 404
            assert state["admin_called"] is False
        finally:
            await _close_clients(client, upstream, gateway)

    asyncio.run(scenario())


def test_capture_writes_exactly_one_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "seed": 7,
                    "stream": True,
                    "test_reasoning": "inspect the tool arguments",
                },
            )
            assert response.status_code == 200
        finally:
            await _close_clients(client, upstream, gateway)

        records = list(store.iter_records())
        assert len(records) == 1
        record = records[0]
        assert record.model == "upstream-model"
        assert record.messages[-1]["content"] == "call done"
        assert (
            record.messages[-1]["reasoning_content"]
            == "inspect the tool arguments"
        )
        assert record.messages[-1]["reasoning"] == "inspect the tool arguments"
        assert record.messages[-1]["tool_calls"] == list(record.tool_calls)
        assert record.tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'
        assert record.prompt_tokens == 8
        assert record.completion_tokens == 5
        assert record.finish_reason == "tool_calls"
        assert record.stop_reason == "end"
        assert (record.temperature, record.top_p, record.seed) == (0.2, 0.9, 7)
        assert record.exchange_id is not None
        manifests = list(gateway.state.exchange_ledger.iter_manifests())
        assert [manifest["exchange_id"] for manifest in manifests] == [
            record.exchange_id
        ]

    asyncio.run(scenario())


def test_capture_without_usage_estimates_instead_of_recording_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        secret = "api_key=EXAMPLEFAKESECRET123456"
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "omit_usage": True,
                    "test_reasoning": secret,
                },
            )
            assert response.status_code == 200
        finally:
            await _close_clients(client, upstream, gateway)

        records = list(store.iter_records())
        assert len(records) == 1
        record = records[0]
        assert secret.encode() not in store.path.read_bytes()
        assert record.messages[-1]["reasoning_content"] == (
            "api_key=<REDACTED:api_key>"
        )
        assert record.messages[-1]["reasoning"] == (
            "api_key=<REDACTED:api_key>"
        )
        assert record.finish_reason == "stop"
        assert record.stop_reason == 128009
        assert record.token_count_source == "estimated"
        assert record.prompt_tokens
        assert record.completion_tokens

    asyncio.run(scenario())


def test_capture_failure_does_not_fail_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)

    class FailingStore(TraceStore):
        def append(self, record: TraceRecord) -> None:
            del record
            raise OSError("disk unavailable")

    async def scenario() -> None:
        store = FailingStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "hello"
        finally:
            await _close_clients(client, upstream, gateway)
        assert not store.path.exists()
        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["capture_error"] == 1

    asyncio.run(scenario())


def test_request_body_overflow_is_counted_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(proxy_module, "_MAX_CAPTURE_BODY_BYTES", 512)

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "x" * 2_000}],
                },
            )
            assert response.status_code == 200
        finally:
            await _close_clients(client, upstream, gateway)

        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["body_overflow"] == 1

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())

    assert "request body overflow" in caplog.text
    assert "size=" in caplog.text
    assert "limit=512" in caplog.text


def test_counter_write_failure_does_not_fail_client_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(proxy_module, "_MAX_CAPTURE_BODY_BYTES", 512)

    def fail_counter_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("stats disk unavailable")

    monkeypatch.setattr(
        "speedlm.traces.store.atomic_write_json",
        fail_counter_write,
    )

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "x" * 2_000}],
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "hello"
        finally:
            await _close_clients(client, upstream, gateway)

        assert not store.stats_path.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("observer_method", ["feed", "finish"])
def test_response_observer_errors_are_counted(
    observer_method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)

    def fail_observer(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("observer failed")

    monkeypatch.setattr(
        proxy_module._ResponseObserver,
        observer_method,
        fail_observer,
    )

    async def scenario() -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        client, upstream, _, gateway = await _clients(store=store)
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "request-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
        finally:
            await _close_clients(client, upstream, gateway)

        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["stream_observer_error"] == 1

    asyncio.run(scenario())


def test_cancelled_capture_is_counted_as_shutdown_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        append_started = asyncio.Event()
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)

        async def blocked_worker(function: Any, *args: Any) -> Any:
            if getattr(function, "__name__", "") == "append":
                append_started.set()
                await asyncio.Event().wait()
            return function(*args)

        monkeypatch.setattr(capture, "_run_in_worker", blocked_worker)
        capture.submit(
            {
                "model": "request-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            AssembledResponse(
                id="response-id",
                model="response-model",
                created=1_700_000_000.0,
                content="hello",
                tool_calls=(),
                prompt_tokens=1,
                completion_tokens=1,
            ),
            endpoint="/v1/chat/completions",
            timestamp=1_700_000_000.0,
        )
        await append_started.wait()
        next(iter(capture._tasks)).cancel()
        await capture.drain()

        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["shutdown_pending"] == 1

    asyncio.run(scenario())


def test_activity_tracks_busy_and_idle() -> None:
    now = [10.0]
    activity = ActivityTracker(clock=lambda: now[0])
    now[0] = 12.0
    assert activity.idle_seconds() == 2.0
    activity.begin()
    now[0] = 20.0
    assert activity.in_flight == 1
    assert activity.idle_seconds() == 0.0
    activity.end()
    now[0] = 23.5
    assert activity.idle_seconds() == 3.5


def test_child_argv_forces_loopback_and_preserves_passthrough() -> None:
    argv = build_vllm_argv(
        "model",
        ["--tensor-parallel-size", "2", "--host=0.0.0.0", "--port", "9000"],
        host="127.0.0.1",
        port=8123,
    )
    assert argv == [
        "vllm",
        "serve",
        "model",
        "--tensor-parallel-size",
        "2",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
    ]


async def _start_real_child(
    script: str,
    *,
    health_url: str = "http://127.0.0.1:1/health",
    startup_timeout: float = 2.0,
) -> VLLMProcess:
    child_script = (
        "import signal\n"
        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
        f"{textwrap.dedent(script)}"
    )
    child = VLLMProcess(
        [sys.executable, "-c", child_script],
        health_url=health_url,
        startup_timeout=startup_timeout,
    )
    await child.start()
    return child


def test_readiness_allows_silent_descendant_cpu_progress_past_stall_window() -> None:
    async def scenario() -> None:
        child = await _start_real_child(
            """
            import os
            import time

            worker_pid = os.fork()
            if worker_pid == 0:
                deadline = time.perf_counter() + 0.6
                while time.perf_counter() < deadline:
                    pass
                os._exit(0)

            os.waitpid(worker_pid, 0)
            time.sleep(9999)
            """,
            startup_timeout=1.5,
        )
        started = time.monotonic()
        readiness = asyncio.create_task(
            child.wait_ready(stall_timeout=0.15, poll_interval=0.01)
        )
        try:
            await asyncio.sleep(0.4)
            assert not readiness.done()
            assert time.monotonic() - started > 0.15
            assert child.returncode is None
        finally:
            readiness.cancel()
            with contextlib.suppress(asyncio.CancelledError, ProcessError):
                await readiness
            await child.shutdown()

    asyncio.run(scenario())


def test_readiness_stalls_alive_silent_process_with_no_cpu_progress() -> None:
    async def scenario() -> None:
        child = await _start_real_child(
            """
            import time
            time.sleep(9999)
            """
        )
        with pytest.raises(ProcessError, match="startup stalled"):
            await child.wait_ready(stall_timeout=0.15, poll_interval=0.01)
        assert child.returncode is not None

    asyncio.run(scenario())


def test_readiness_reports_early_exit_code_without_waiting_for_stall() -> None:
    async def scenario() -> None:
        child = await _start_real_child(
            """
            import sys
            sys.exit(7)
            """
        )
        started = time.monotonic()
        with pytest.raises(ProcessError, match="exited before readiness with code 7"):
            await child.wait_ready(stall_timeout=1.0, poll_interval=0.01)
        assert time.monotonic() - started < 1.0
        assert child.returncode == 7

    asyncio.run(scenario())


def test_readiness_stall_message_includes_diagnostics_and_log_tail() -> None:
    async def scenario() -> None:
        child = await _start_real_child(
            """
            import time
            for index in range(25):
                print(f"child-line-{index:02d}")
            time.sleep(9999)
            """
        )
        with pytest.raises(ProcessError) as raised:
            await child.wait_ready(stall_timeout=0.15, poll_interval=0.01)
        message = str(raised.value)
        assert "startup stalled after" in message
        assert "log silence:" in message
        assert "process alive: yes" in message
        assert "CPU time advanced: no" in message
        assert "Last 20 lines of vLLM log:" in message
        assert "child-line-05" in message
        assert "child-line-24" in message
        assert "child-line-04" not in message
        assert child.returncode is not None

    asyncio.run(scenario())


def test_readiness_enforces_absolute_hard_ceiling_despite_cpu_progress() -> None:
    async def scenario() -> None:
        child = await _start_real_child(
            """
            while True:
                pass
            """,
            startup_timeout=0.35,
        )
        with pytest.raises(ProcessError, match="absolute startup hard ceiling"):
            await child.wait_ready(stall_timeout=0.2, poll_interval=0.01)
        assert child.returncode is not None

    asyncio.run(scenario())


def test_child_python_output_is_forced_unbuffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
        child = await _start_real_child(
            """
            import os
            import time
            print(f"unbuffered={os.environ.get('PYTHONUNBUFFERED')}")
            time.sleep(9999)
            """
        )
        with pytest.raises(ProcessError) as raised:
            await child.wait_ready(stall_timeout=0.15, poll_interval=0.01)
        assert "unbuffered=1" in str(raised.value)

    asyncio.run(scenario())
