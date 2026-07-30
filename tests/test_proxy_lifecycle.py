from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.requests import Request

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.proxy import GatewayProxy


@dataclass(slots=True)
class _UpstreamRequest:
    content: AsyncIterator[bytes]


class _HeaderGateClient:
    def __init__(self) -> None:
        self.body_read = asyncio.Event()
        self.release_headers = asyncio.Event()
        self.cancelled = asyncio.Event()

    def build_request(
        self,
        method: str,
        target: httpx.URL,
        **kwargs: Any,
    ) -> _UpstreamRequest:
        del method, target
        return _UpstreamRequest(kwargs["content"])

    async def send(
        self,
        request: _UpstreamRequest,
        *,
        stream: bool,
    ) -> httpx.Response:
        assert stream
        async for _ in request.content:
            pass
        self.body_read.set()
        try:
            await self.release_headers.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return httpx.Response(
            200,
            content=b"ok",
            request=httpx.Request("POST", "http://upstream/v1/slow"),
        )


def _request(*, disconnect_after_body: bool) -> Request:
    messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    messages.put_nowait(
        {"type": "http.request", "body": b'{"input":"x"}', "more_body": False}
    )
    if disconnect_after_body:
        messages.put_nowait({"type": "http.disconnect"})

    async def receive() -> dict[str, Any]:
        return await messages.get()

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/slow",
            "raw_path": b"/v1/slow",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("gateway", 80),
            "root_path": "",
        },
        receive,
    )


def test_disconnect_before_upstream_headers_cancels_without_leaking_activity() -> None:
    async def scenario() -> None:
        client = _HeaderGateClient()
        activity = ActivityTracker()
        proxy = GatewayProxy(
            client,  # type: ignore[arg-type]
            "http://upstream",
            activity=activity,
            capture=None,
        )

        response = await asyncio.wait_for(
            proxy.handle(_request(disconnect_after_body=True)),
            timeout=0.5,
        )

        assert response.status_code == 499
        assert client.body_read.is_set()
        assert client.cancelled.is_set()
        assert activity.in_flight == 0
        assert not proxy._relays

    asyncio.run(scenario())


def test_connected_request_remains_pending_until_upstream_headers_arrive() -> None:
    async def scenario() -> None:
        client = _HeaderGateClient()
        activity = ActivityTracker()
        proxy = GatewayProxy(
            client,  # type: ignore[arg-type]
            "http://upstream",
            activity=activity,
            capture=None,
        )
        request_task = asyncio.create_task(
            proxy.handle(_request(disconnect_after_body=False))
        )
        await asyncio.wait_for(client.body_read.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert not request_task.done()
        assert activity.in_flight == 1

        client.release_headers.set()
        response = await asyncio.wait_for(request_task, timeout=0.5)
        assert response.status_code == 200
        for _ in range(10):
            if activity.in_flight == 0:
                break
            await asyncio.sleep(0)
        assert activity.in_flight == 0

    asyncio.run(scenario())
