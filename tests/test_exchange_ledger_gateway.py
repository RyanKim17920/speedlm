from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import Response

from speedlm.gateway.app import create_app
from speedlm.gateway.exchange import ExchangeLedger


def _body_for(
    ledger: ExchangeLedger,
    manifest: dict[str, Any],
    side: str,
) -> bytes:
    directory = ledger.root / manifest["exchange_id"]
    return (directory / manifest[side]["body_file"]).read_bytes()


def _query_for(
    ledger: ExchangeLedger,
    manifest: dict[str, Any],
) -> bytes:
    directory = ledger.root / manifest["exchange_id"]
    return (directory / manifest["query_file"]).read_bytes()


def _assert_complete_raw_pair(
    ledger: ExchangeLedger,
    manifest: dict[str, Any],
    *,
    request: bytes,
    response: bytes,
) -> None:
    assert manifest["state"] == "complete"
    assert manifest["failure_reason"] is None
    assert manifest["request"]["complete"] is True
    assert manifest["request"]["bytes"] == len(request)
    assert manifest["request"]["sha256"] == hashlib.sha256(request).hexdigest()
    assert manifest["response"]["complete"] is True
    assert manifest["response"]["bytes"] == len(response)
    assert manifest["response"]["sha256"] == hashlib.sha256(response).hexdigest()
    assert _body_for(ledger, manifest, "request") == request
    assert _body_for(ledger, manifest, "response") == response


async def _close_gateway(
    gateway: FastAPI,
    upstream: httpx.AsyncClient,
) -> None:
    capture = gateway.state.capture
    if capture is not None:
        await asyncio.wait_for(capture.drain(), timeout=2.0)
    await asyncio.wait_for(gateway.state.proxy.aclose(), timeout=2.0)
    await asyncio.wait_for(upstream.aclose(), timeout=2.0)
    ledger = gateway.state.exchange_ledger
    if ledger is not None:
        await asyncio.wait_for(ledger.aclose(), timeout=2.0)


async def _request_gateway(
    gateway: FastAPI,
    target: str,
    *,
    body: bytes,
    content_type: str,
) -> httpx.Response:
    """Drive the proxy directly while keeping the downstream connected."""
    parsed = urlsplit(target)
    request_sent = False
    response_consumed = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await response_consumed.wait()
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [(b"content-type", content_type.encode())],
            "server": ("gateway", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
        },
        receive,
    )
    try:
        response = await asyncio.wait_for(
            gateway.state.proxy.handle(request),
            timeout=10.0,
        )

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            chunks = [bytes(response.body)]
        else:
            async def consume() -> list[bytes]:
                return [chunk async for chunk in body_iterator]

            chunks = await asyncio.wait_for(consume(), timeout=10.0)
        return httpx.Response(
            response.status_code,
            headers=response.raw_headers,
            content=b"".join(chunks),
        )
    finally:
        response_consumed.set()


def test_ledger_admission_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        upstream_called = False
        upstream_app = FastAPI()

        @upstream_app.post("/v1/chat/completions")
        async def chat() -> Response:
            nonlocal upstream_called
            upstream_called = True
            return Response(b"must not be reached")

        ledger = ExchangeLedger(tmp_path / "exchanges")

        async def fail_admission(**_kwargs: Any) -> Any:
            raise OSError("simulated unavailable storage")

        monkeypatch.setattr(ledger, "astart", fail_admission)
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            upstream_client=upstream,
            exchange_ledger=ledger,
        )
        try:
            response = await _request_gateway(
                gateway,
                "/v1/chat/completions",
                body=b"{}",
                content_type="application/json",
            )
            assert response.status_code == 503
            assert response.content == b"Capture storage unavailable"
            assert upstream_called is False
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())


def test_unknown_non_v1_post_is_preserved_in_raw_ledger(tmp_path: Path) -> None:
    async def scenario() -> None:
        request_body = b"\x00future-request\xff"
        response_body = b"\x80future-response\x00"
        upstream_app = FastAPI()

        @upstream_app.post("/native/future-generate")
        async def future_generate(request: Request) -> Response:
            assert await request.body() == request_body
            assert request.url.query == "mode=opaque"
            return Response(
                response_body,
                status_code=207,
                media_type="application/octet-stream",
            )

        ledger = ExchangeLedger(tmp_path / "exchanges")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            upstream_client=upstream,
            exchange_ledger=ledger,
        )
        try:
            response = await _request_gateway(
                gateway,
                "/native/future-generate?mode=opaque",
                body=request_body,
                content_type="application/x-future",
            )

            assert response.status_code == 207
            assert response.content == response_body
            manifests = list(ledger.iter_manifests())
            assert len(manifests) == 1
            manifest = manifests[0]
            assert manifest["method"] == "POST"
            assert manifest["path"] == "/native/future-generate"
            assert _query_for(ledger, manifest) == b"mode=opaque"
            assert manifest["response"]["status"] == 207
            _assert_complete_raw_pair(
                ledger,
                manifest,
                request=request_body,
                response=response_body,
            )
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())


def test_non_2xx_and_malformed_outputs_are_preserved_raw(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream_app = FastAPI()
        cases = {
            "rejected": (
                422,
                b'{"error":"request rejected","partial":true}\x00',
            ),
            "malformed": (
                200,
                b'{"id":"unfinished","choices":[',
            ),
        }

        @upstream_app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            body = json.loads(await request.body())
            status, response_body = cases[body["case"]]
            return Response(
                response_body,
                status_code=status,
                media_type="application/json",
            )

        ledger = ExchangeLedger(tmp_path / "exchanges")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            upstream_client=upstream,
            exchange_ledger=ledger,
        )
        requests = {
            name: json.dumps(
                {
                    "case": name,
                    "model": "test-model",
                    "messages": [{"role": "user", "content": name}],
                },
                separators=(",", ":"),
            ).encode()
            for name in cases
        }
        try:
            responses = {
                name: await _request_gateway(
                    gateway,
                    "/v1/chat/completions",
                    body=request_body,
                    content_type="application/json",
                )
                for name, request_body in requests.items()
            }

            assert {
                name: (response.status_code, response.content)
                for name, response in responses.items()
            } == {
                name: expected
                for name, expected in cases.items()
            }
            manifests = list(ledger.iter_manifests())
            assert len(manifests) == 2
            by_request = {
                _body_for(ledger, manifest, "request"): manifest
                for manifest in manifests
            }
            for name, request_body in requests.items():
                manifest = by_request[request_body]
                status, response_body = cases[name]
                assert manifest["response"]["status"] == status
                _assert_complete_raw_pair(
                    ledger,
                    manifest,
                    request=request_body,
                    response=response_body,
                )

        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())


def test_concurrent_gateway_requests_keep_exact_raw_pairs(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream_app = FastAPI()

        @upstream_app.post("/native/concurrent")
        async def concurrent(request: Request) -> Response:
            request_body = await request.body()
            payload = json.loads(request_body)
            index = payload["index"]
            await asyncio.sleep((31 - index) % 7 * 0.001)
            return Response(
                f"output-{index}:{payload['marker']}".encode(),
                media_type="application/octet-stream",
            )

        ledger = ExchangeLedger(tmp_path / "exchanges")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            upstream_client=upstream,
            exchange_ledger=ledger,
        )
        request_bodies = {
            index: json.dumps(
                {"index": index, "marker": f"input-{index}"},
                separators=(",", ":"),
            ).encode()
            for index in range(32)
        }
        expected_pairs = {
            (
                request_body,
                f"output-{index}:input-{index}".encode(),
            )
            for index, request_body in request_bodies.items()
        }
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        _request_gateway(
                            gateway,
                            f"/native/concurrent?lane={index % 4}",
                            body=request_body,
                            content_type="application/json",
                        )
                        for index, request_body in request_bodies.items()
                    )
                ),
                timeout=30.0,
            )

            assert len(responses) == 32
            assert all(response.status_code == 200 for response in responses)
            manifests = list(ledger.iter_manifests())
            assert len(manifests) == 32
            actual_pairs = {
                (
                    _body_for(ledger, manifest, "request"),
                    _body_for(ledger, manifest, "response"),
                )
                for manifest in manifests
            }
            assert actual_pairs == expected_pairs
            assert all(manifest["state"] == "complete" for manifest in manifests)
            assert all(manifest["request"]["complete"] is True for manifest in manifests)
            assert all(manifest["response"]["complete"] is True for manifest in manifests)
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())


def test_projection_body_limit_does_not_truncate_raw_exchange(tmp_path: Path) -> None:
    async def scenario() -> None:
        request_body = json.dumps(
            {
                "model": "large-model",
                "messages": [{"role": "user", "content": "R" * 2048}],
            },
            separators=(",", ":"),
        ).encode()
        response_body = json.dumps(
            {
                "id": "large-response",
                "model": "large-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "S" * 3072,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2048, "completion_tokens": 3072},
            },
            separators=(",", ":"),
        ).encode()
        upstream_app = FastAPI()

        @upstream_app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            assert await request.body() == request_body
            return Response(
                response_body,
                media_type="application/json",
            )

        ledger = ExchangeLedger(tmp_path / "exchanges")
        upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app),
            base_url="http://upstream",
        )
        gateway = create_app(
            "http://upstream",
            upstream_client=upstream,
            exchange_ledger=ledger,
            capture_body_limit=64,
        )
        try:
            response = await _request_gateway(
                gateway,
                "/v1/chat/completions",
                body=request_body,
                content_type="application/json",
            )

            assert response.status_code == 200
            assert response.content == response_body
            manifests = list(ledger.iter_manifests())
            assert len(manifests) == 1
            _assert_complete_raw_pair(
                ledger,
                manifests[0],
                request=request_body,
                response=response_body,
            )
        finally:
            await _close_gateway(gateway, upstream)

    asyncio.run(scenario())
