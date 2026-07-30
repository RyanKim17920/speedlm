from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import Response

from speedlm.gateway.app import create_app
from speedlm.gateway.capture import CaptureAdapter
from speedlm.gateway.sse import AssembledResponse
from speedlm.traces.store import TraceStore

ZEPHYR_PATH = "/v1/zephyr"
REQUEST_BYTES = (
    b'{"model":"zephyr-model","input":"request-marker",'
    b'"temperature":0.4,"top_p":0.8,"seed":41}'
)
RESPONSE_BYTES = (
    b'{"id":"zephyr-response-id","model":"zephyr-model",'
    b'"output":"response-marker","input_tokens":7,"output_tokens":3}'
)


@pytest.fixture(autouse=True)
def immediate_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run)


def _zephyr_response(
    body: bytes | bytearray,
    content_type: str,
) -> AssembledResponse | None:
    assert content_type.startswith("application/vnd.zephyr+json")
    payload = json.loads(body)
    assert isinstance(payload, dict)
    return AssembledResponse(
        id=payload["id"],
        model=payload["model"],
        created=1_700_000_000.0,
        content=payload["output"],
        tool_calls=(),
        prompt_tokens=payload["input_tokens"],
        completion_tokens=payload["output_tokens"],
        finish_reason="stop",
    )


def _zephyr_record(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    return {
        "id": response.id,
        "timestamp": response.created if response.created is not None else timestamp,
        "model": response.model or request_data["model"],
        "messages": [
            {
                "role": "user",
                "content": request_data["input"],
                "provenance_tag": "client_supplied",
            },
            {
                "role": "assistant",
                "content": response.content,
                "provenance_tag": "generated",
            },
        ],
        "tool_calls": [],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        },
        "temperature": request_data["temperature"],
        "top_p": request_data["top_p"],
        "seed": request_data["seed"],
        "finish_reason": response.finish_reason,
        "stop_reason": response.stop_reason,
    }


def _adapter(
    *,
    matches: Any,
    decode_response: Any = _zephyr_response,
    build_record: Any = _zephyr_record,
) -> CaptureAdapter:
    return CaptureAdapter(
        name="zephyr",
        matches=matches,
        decode_response=decode_response,
        build_record=build_record,
    )


async def _close_gateway(
    gateway: FastAPI,
    upstream: httpx.AsyncClient,
) -> None:
    capture = gateway.state.capture
    if capture is not None:
        await asyncio.wait_for(capture.drain(), timeout=2.0)
    await asyncio.wait_for(gateway.state.proxy.aclose(), timeout=2.0)
    await asyncio.wait_for(upstream.aclose(), timeout=2.0)


async def _request_app(
    app: FastAPI,
    method: str,
    target: str,
    *,
    body: bytes,
    headers: list[tuple[bytes, bytes]],
) -> httpx.Response:
    """Exercise the proxy response iterator without buffering its stream."""
    parsed = urlsplit(target)
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": headers,
            "server": ("gateway", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
        },
        receive,
    )
    response = await app.state.proxy.handle(request)
    chunks = [chunk async for chunk in response.body_iterator]
    return httpx.Response(
        response.status_code,
        headers=response.raw_headers,
        content=b"".join(chunks),
    )


def test_registered_adapter_captures_invented_endpoint_without_changing_transport(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observed: dict[str, Any] = {}
        upstream_app = FastAPI()

        @upstream_app.post(ZEPHYR_PATH)
        async def zephyr(request: Request) -> Response:
            observed["body"] = await request.body()
            observed["query"] = request.url.query
            observed["request_header"] = request.headers["x-harness"]
            return Response(
                RESPONSE_BYTES,
                status_code=201,
                media_type="application/vnd.zephyr+json",
                headers={"x-zephyr": "preserved"},
            )

        adapter_calls: list[tuple[str, str]] = []

        def matches(method: str, path: str) -> bool:
            adapter_calls.append((method, path))
            return method == "POST" and path.rstrip("/") == ZEPHYR_PATH

        store = TraceStore(tmp_path / "traces.jsonl")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            trace_store=store,
            upstream_client=upstream,
            capture_adapters=[_adapter(matches=matches)],
        )
        try:
            response = await asyncio.wait_for(
                _request_app(
                    gateway,
                    "POST",
                    f"{ZEPHYR_PATH}?lane=blue",
                    body=REQUEST_BYTES,
                    headers=[
                        (b"content-type", b"application/json"),
                        (b"x-harness", b"registry-contract"),
                    ],
                ),
                timeout=2.0,
            )

            assert response.status_code == 201
            assert response.content == RESPONSE_BYTES
            assert response.headers["content-type"].startswith(
                "application/vnd.zephyr+json"
            )
            assert response.headers["x-zephyr"] == "preserved"
            assert observed == {
                "body": REQUEST_BYTES,
                "query": "lane=blue",
                "request_header": "registry-contract",
            }
            assert adapter_calls == [("POST", ZEPHYR_PATH)]

            await asyncio.wait_for(gateway.state.capture.drain(), timeout=2.0)
            records = list(store.iter_records())
            assert len(records) == 1
            record = records[0]
            assert record.id == "zephyr-response-id"
            assert record.model == "zephyr-model"
            assert record.messages[0]["content"] == "request-marker"
            assert record.messages[-1]["content"] == "response-marker"
            assert record.messages[0]["provenance_tag"] == "client_supplied"
            assert record.messages[-1]["provenance_tag"] == "generated"
            assert (record.prompt_tokens, record.completion_tokens) == (7, 3)
            assert (record.temperature, record.top_p, record.seed) == (0.4, 0.8, 41)
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())


def test_registry_falls_through_nonmatching_adapter_and_captures_once(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        upstream_app = FastAPI()

        @upstream_app.post(ZEPHYR_PATH)
        async def zephyr(request: Request) -> Response:
            assert await request.body() == REQUEST_BYTES
            return Response(
                RESPONSE_BYTES,
                media_type="application/vnd.zephyr+json",
            )

        calls: list[str] = []

        def unrelated_matches(method: str, path: str) -> bool:
            calls.append(f"unrelated:{method}:{path}")
            return False

        def unrelated_decode(
            _body: bytes | bytearray,
            _content_type: str,
        ) -> AssembledResponse | None:
            raise AssertionError("a nonmatching adapter must not decode the exchange")

        def zephyr_matches(method: str, path: str) -> bool:
            calls.append(f"zephyr:{method}:{path}")
            return method == "POST" and path == ZEPHYR_PATH

        store = TraceStore(tmp_path / "traces.jsonl")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            trace_store=store,
            upstream_client=upstream,
            capture_adapters=[
                _adapter(
                    matches=unrelated_matches,
                    decode_response=unrelated_decode,
                ),
                _adapter(matches=zephyr_matches),
            ],
        )
        try:
            response = await asyncio.wait_for(
                _request_app(
                    gateway,
                    "POST",
                    ZEPHYR_PATH,
                    body=REQUEST_BYTES,
                    headers=[(b"content-type", b"application/json")],
                ),
                timeout=2.0,
            )
            assert response.status_code == 200
            assert response.content == RESPONSE_BYTES

            await asyncio.wait_for(gateway.state.capture.drain(), timeout=2.0)
            assert calls == [
                f"unrelated:POST:{ZEPHYR_PATH}",
                f"zephyr:POST:{ZEPHYR_PATH}",
            ]
            records = list(store.iter_records())
            assert len(records) == 1
            assert records[0].messages[0]["content"] == "request-marker"
            assert records[0].messages[-1]["content"] == "response-marker"
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())
