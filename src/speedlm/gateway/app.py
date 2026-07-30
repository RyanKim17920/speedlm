from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

from speedlm.config import SamplingConfig
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.capture import CaptureAdapter, CaptureManager
from speedlm.gateway.control import AdmissionGate
from speedlm.gateway.exchange import ExchangeLedger
from speedlm.gateway.proxy import GatewayProxy
from speedlm.traces.store import TraceStore


def create_app(
    upstream_base_url: str,
    *,
    trace_store: TraceStore | None = None,
    sampling: SamplingConfig | None = None,
    activity: ActivityTracker | None = None,
    admission: AdmissionGate | None = None,
    upstream_client: httpx.AsyncClient | None = None,
    capture_adapters: Sequence[CaptureAdapter] | None = None,
    exchange_ledger: ExchangeLedger | None = None,
    capture_body_limit: int | None = None,
    relay_queue_chunks: int | None = None,
    detached_drain_timeout: float | None = None,
    before_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Create a fail-closed gateway using exactly one long-lived HTTP client."""
    tracker = activity or ActivityTracker()
    capture = (
        CaptureManager(
            trace_store,
            defaults=sampling,
            adapters=capture_adapters,
        )
        if trace_store is not None
        else None
    )
    ledger = (
        exchange_ledger
        if exchange_ledger is not None
        else (
            ExchangeLedger(trace_store.path.parent / "exchanges")
            if trace_store is not None
            else None
        )
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
        admission=admission,
        capture=capture,
        exchange_ledger=ledger,
        capture_body_limit=capture_body_limit,
        relay_queue_chunks=relay_queue_chunks,
        detached_drain_timeout=detached_drain_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        try:
            if ledger is not None:
                await ledger.arecover_incomplete()
            yield
        finally:
            try:
                if before_shutdown is not None:
                    await before_shutdown()
            finally:
                await proxy.aclose()
                if capture is not None:
                    await capture.aclose()
                if owns_client:
                    await client.aclose()
                if ledger is not None:
                    await ledger.aclose()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.activity = tracker
    app.state.admission = admission
    app.state.capture = capture
    app.state.exchange_ledger = ledger
    app.state.proxy = proxy
    app.state.upstream_client = client

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def route_all(request: Request, path: str) -> Response:
        del path
        return await proxy.handle(request)

    return app
