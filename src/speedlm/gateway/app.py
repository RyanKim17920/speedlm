from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

from speedlm.config import SamplingConfig
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.capture import CaptureManager
from speedlm.gateway.proxy import GatewayProxy
from speedlm.traces.store import TraceStore


def create_app(
    upstream_base_url: str,
    *,
    trace_store: TraceStore | None = None,
    sampling: SamplingConfig | None = None,
    activity: ActivityTracker | None = None,
    upstream_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Create a fail-closed gateway using exactly one long-lived HTTP client."""
    tracker = activity or ActivityTracker()
    capture = (
        CaptureManager(trace_store, defaults=sampling)
        if trace_store is not None
        else None
    )
    owns_client = upstream_client is None
    client = upstream_client or httpx.AsyncClient(
        timeout=httpx.Timeout(None),
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        follow_redirects=False,
        trust_env=False,
    )
    proxy = GatewayProxy(
        client,
        upstream_base_url,
        activity=tracker,
        capture=capture,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        try:
            yield
        finally:
            if capture is not None:
                await capture.drain()
            if owns_client:
                await client.aclose()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.activity = tracker
    app.state.capture = capture
    app.state.upstream_client = client

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def route_all(request: Request, path: str) -> Response:
        del path
        return await proxy.handle(request)

    return app
