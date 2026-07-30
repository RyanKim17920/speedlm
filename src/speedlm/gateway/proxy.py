from __future__ import annotations

import asyncio
import logging
import posixpath
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.capture import CaptureManager, capture_timestamp
from speedlm.gateway.control import AdmissionGate
from speedlm.gateway.exchange import (
    ExchangeLedger,
    ExchangeRecorder,
    ExchangeWriteError,
)

logger = logging.getLogger(__name__)

_HOP_BY_HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
_BLOCKED_V1_PATHS = {
    "/collective_rpc",
    "/finish_weight_update",
    "/get_world_size",
    "/init_weight_transfer_engine",
    "/is_paused",
    "/is_sleeping",
    "/pause",
    "/reset_encoder_cache",
    "/reset_mm_cache",
    "/reset_prefix_cache",
    "/resume",
    "/server_info",
    "/sleep",
    "/start_weight_update",
    "/update_weights",
    "/wake_up",
    "/v1/load_lora_adapter",
    "/v1/unload_lora_adapter",
    "/v1/reset_prefix_cache",
    "/v1/sleep",
    "/v1/start_profile",
    "/v1/stop_profile",
    "/v1/wake_up",
}
_CONTROL_ACTION_PREFIXES = (
    "abort",
    "collective",
    "is_scaling",
    "load_lora",
    "pause",
    "profile",
    "reset",
    "resume",
    "scale",
    "sleep",
    "start_profile",
    "stop_profile",
    "unload_lora",
    "update_weight",
    "wake",
)
_MAX_CAPTURE_BODY_BYTES = 32 * 1024 * 1024
_RELAY_QUEUE_CHUNKS = 16
_DETACHED_DRAIN_TIMEOUT_SECONDS = 30.0
_RELAY_END = object()


class _BodyTee:
    def __init__(
        self,
        source: AsyncIterator[bytes],
        *,
        limit: int,
        recorder: ExchangeRecorder | None = None,
    ) -> None:
        self._source = source
        self._limit = limit
        self._captured = bytearray()
        self._overflowed = False
        self._size = 0
        self._recorder = recorder
        self.complete = asyncio.Event()

    async def stream(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._source:
                self._size += len(chunk)
                if self._recorder is not None:
                    await self._recorder.afeed_request(chunk)
                if chunk and not self._overflowed:
                    if len(self._captured) + len(chunk) <= self._limit:
                        self._captured.extend(chunk)
                    else:
                        self._overflowed = True
                        self._captured.clear()
                if chunk:
                    yield chunk
        finally:
            if self._recorder is not None:
                await self._recorder.afinish_request()
            self.complete.set()

    def take_body(self) -> bytearray | None:
        """Transfer the captured body without copying it on the response path."""
        if self._overflowed:
            return None
        body = self._captured
        self._captured = bytearray()
        return body

    @property
    def size(self) -> int:
        return self._size

    @property
    def limit(self) -> int:
        return self._limit


class _ResponseObserver:
    """Bounded raw response capture.

    Parsing deliberately happens later in ``CaptureManager``. The relay path
    only copies bytes, so malformed or unusually complex JSON/SSE can never
    delay bytes being delivered to the client.
    """

    def __init__(
        self,
        *,
        limit: int,
        recorder: ExchangeRecorder | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("response capture limit must be positive")
        self._limit = limit
        self._body = bytearray()
        self._overflowed = False
        self._size = 0
        self._recorder = recorder

    async def feed(self, chunk: bytes) -> None:
        self._size += len(chunk)
        if self._recorder is not None:
            await self._recorder.afeed_response(chunk)
        if self._overflowed:
            return
        if len(self._body) + len(chunk) <= self._limit:
            self._body.extend(chunk)
        else:
            self._overflowed = True
            self._body.clear()

    def finish(self) -> bytearray | None:
        if self._overflowed:
            return None
        body = self._body
        self._body = bytearray()
        return body

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def size(self) -> int:
        return self._size

    @property
    def limit(self) -> int:
        return self._limit


@dataclass(slots=True)
class _RelayFailure:
    error: Exception


class _ResponseRelay:
    """Pump one upstream response independently of downstream consumption.

    A client disconnect detaches the bounded delivery queue but does not cancel
    the upstream read. This lets a generation that vLLM has already completed
    be captured and correlated even when a harness closes its HTTP stream
    immediately after receiving a terminal event.
    """

    def __init__(
        self,
        upstream_response: httpx.Response,
        *,
        observer: _ResponseObserver | None,
        on_complete: Any,
        on_finish: Any,
        on_observer_error: Any,
        on_incomplete: Any,
        queue_chunks: int,
        detached_drain_timeout: float | None,
    ) -> None:
        self._upstream_response = upstream_response
        self._observer = observer
        self._on_complete = on_complete
        self._on_finish = on_finish
        self._on_observer_error = on_observer_error
        self._on_incomplete = on_incomplete
        self._detached_drain_timeout = detached_drain_timeout
        self._drain_timeout_handle: asyncio.TimerHandle | None = None
        self._queue: asyncio.Queue[bytes | _RelayFailure | object] = asyncio.Queue(
            maxsize=queue_chunks
        )
        self._detached = asyncio.Event()
        self.task = asyncio.create_task(
            self._pump(),
            name="speedlm-upstream-response-relay",
        )

    async def stream(self) -> AsyncIterator[bytes]:
        try:
            while True:
                item = await self._queue.get()
                if item is _RELAY_END:
                    return
                if isinstance(item, _RelayFailure):
                    raise item.error
                if not isinstance(item, bytes):
                    raise RuntimeError("invalid response relay item")
                yield item
        finally:
            self._detached.set()
            if self._observer is None and not self.task.done():
                self.task.cancel()
            elif (
                not self.task.done()
                and self._detached_drain_timeout is not None
            ):
                self._drain_timeout_handle = asyncio.get_running_loop().call_later(
                    self._detached_drain_timeout,
                    self.task.cancel,
                )

    async def _deliver(self, item: bytes | _RelayFailure | object) -> None:
        if self._detached.is_set():
            return
        try:
            self._queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass

        put_task = asyncio.create_task(self._queue.put(item))
        detached_task = asyncio.create_task(self._detached.wait())
        try:
            done, _ = await asyncio.wait(
                {put_task, detached_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if detached_task in done:
                put_task.cancel()
            else:
                detached_task.cancel()
        finally:
            await asyncio.gather(
                put_task,
                detached_task,
                return_exceptions=True,
            )

    async def _pump(self) -> None:
        completed = False
        incomplete_reason: str | None = None
        try:
            async for chunk in self._upstream_response.aiter_raw():
                if self._observer is not None:
                    try:
                        await self._observer.feed(chunk)
                    except ExchangeWriteError:
                        raise
                    except Exception as exc:
                        logger.warning("dropping failed response capture: %s", exc)
                        self._observer = None
                        self._on_observer_error()
                await self._deliver(chunk)
            completed = True
        except asyncio.CancelledError:
            incomplete_reason = "response_relay_cancelled"
            raise
        except Exception as exc:
            incomplete_reason = f"upstream_response_error:{type(exc).__name__}"
            await self._deliver(_RelayFailure(exc))
        finally:
            if self._drain_timeout_handle is not None:
                self._drain_timeout_handle.cancel()
            try:
                await self._upstream_response.aclose()
            finally:
                try:
                    if completed:
                        await self._on_complete(self._observer)
                    else:
                        await self._on_incomplete(
                            incomplete_reason or "response_relay_incomplete"
                        )
                finally:
                    self._on_finish()
                    await self._deliver(_RELAY_END)


class GatewayProxy:
    """A streaming reverse proxy backed by one shared ``AsyncClient``."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        upstream_base_url: str,
        *,
        activity: ActivityTracker,
        admission: AdmissionGate | None = None,
        capture: CaptureManager | None,
        exchange_ledger: ExchangeLedger | None = None,
        capture_body_limit: int | None = None,
        relay_queue_chunks: int | None = None,
        detached_drain_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._upstream = httpx.URL(upstream_base_url)
        self._activity = activity
        self._admission = admission
        self._capture = capture
        self._exchange_ledger = exchange_ledger
        self._capture_body_limit = (
            _MAX_CAPTURE_BODY_BYTES
            if capture_body_limit is None
            else capture_body_limit
        )
        self._relay_queue_chunks = (
            _RELAY_QUEUE_CHUNKS
            if relay_queue_chunks is None
            else relay_queue_chunks
        )
        self._detached_drain_timeout = (
            _DETACHED_DRAIN_TIMEOUT_SECONDS
            if detached_drain_timeout is None
            else detached_drain_timeout
        )
        if self._capture_body_limit <= 0:
            raise ValueError("capture body limit must be positive")
        if self._relay_queue_chunks <= 0:
            raise ValueError("relay queue chunks must be positive")
        if self._detached_drain_timeout <= 0:
            raise ValueError("detached drain timeout must be positive")
        self._relays: set[asyncio.Task[None]] = set()

    async def aclose(self) -> None:
        """Cancel detached upstream relays during application shutdown."""
        pending = tuple(self._relays)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._relays.difference_update(pending)

    async def handle(self, request: Request) -> Response:
        activity_started = False
        if self._admission is not None:
            if (
                not self._admission.try_begin()
                and not await self._admission.wait_to_begin()
            ):
                return PlainTextResponse(
                    "Gateway shutting down",
                    status_code=503,
                )
            activity_started = True
        path = request.url.path
        if not _is_allowed_path(path, method=request.method):
            if activity_started:
                self._activity.end()
            return PlainTextResponse("Not Found", status_code=404)

        started_at = capture_timestamp()
        try:
            exchange = await self._start_exchange(request, started_at=started_at)
        except (OSError, RuntimeError) as exc:
            if activity_started:
                self._activity.end()
            logger.error("raw exchange ledger admission failed: %s", exc)
            return PlainTextResponse(
                "Capture storage unavailable",
                status_code=503,
            )
        body_tee = _BodyTee(
            request.stream(),
            limit=self._capture_body_limit,
            recorder=exchange,
        )
        target = self._upstream.copy_with(
            path=path,
            query=request.scope.get("query_string", b""),
        )
        request_headers = _strip_hop_by_hop(request.headers.raw, strip_host=True)
        upstream_request = self._client.build_request(
            request.method,
            target,
            headers=request_headers,
            content=body_tee.stream(),
        )

        if not activity_started:
            self._activity.begin()
        upstream_send = asyncio.create_task(
            self._client.send(upstream_request, stream=True),
            name="speedlm-upstream-request",
        )
        # Let the HTTP client take ownership of the streaming body before the
        # disconnect/body-completion race below starts another ASGI receiver.
        await asyncio.sleep(0)
        body_complete = asyncio.create_task(body_tee.complete.wait())
        try:
            send_or_body_done, _ = await asyncio.wait(
                {upstream_send, body_complete},
                return_when=asyncio.FIRST_COMPLETED,
            )
            body_complete.cancel()
            await asyncio.gather(body_complete, return_exceptions=True)
            if upstream_send not in send_or_body_done:
                disconnect = asyncio.create_task(_wait_for_disconnect(request))
                try:
                    send_or_disconnect, _ = await asyncio.wait(
                        {upstream_send, disconnect},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if upstream_send not in send_or_disconnect:
                        upstream_send.cancel()
                        await asyncio.gather(upstream_send, return_exceptions=True)
                        if exchange is not None:
                            await exchange.aabort(
                                "downstream_disconnected_before_response"
                            )
                        self._activity.end()
                        return Response(status_code=499)
                finally:
                    disconnect.cancel()
                    await asyncio.gather(disconnect, return_exceptions=True)
            upstream_response = await upstream_send
        except httpx.HTTPError as exc:
            if exchange is not None:
                await exchange.aabort(
                    f"upstream_request_error:{type(exc).__name__}"
                )
            self._activity.end()
            logger.error("upstream request failed: %s", exc)
            return PlainTextResponse("Bad Gateway", status_code=502)
        except asyncio.CancelledError:
            if exchange is not None:
                await exchange.aabort("downstream_cancelled_before_response")
            body_complete.cancel()
            upstream_send.cancel()
            await asyncio.gather(
                body_complete,
                upstream_send,
                return_exceptions=True,
            )
            self._activity.end()
            raise
        except Exception:
            if exchange is not None:
                await exchange.aabort("gateway_error_before_response")
            body_complete.cancel()
            upstream_send.cancel()
            await asyncio.gather(
                body_complete,
                upstream_send,
                return_exceptions=True,
            )
            self._activity.end()
            raise

        response_headers = _strip_hop_by_hop(upstream_response.headers.raw)
        content_type = upstream_response.headers.get("content-type", "")
        content_encoding = upstream_response.headers.get("content-encoding", "")
        if exchange is not None:
            await exchange.aset_response(
                status=upstream_response.status_code,
                headers=upstream_response.headers.raw,
            )
        capture_adapter = (
            self._capture.match(request.method, path)
            if self._capture is not None
            else None
        )
        observer = (
            _ResponseObserver(
                limit=self._capture_body_limit,
                recorder=exchange,
            )
            if exchange is not None
            or (
                self._capture is not None
                and capture_adapter is not None
                and 200 <= upstream_response.status_code < 300
            )
            else None
        )

        async def capture_complete(
            completed_observer: _ResponseObserver | None,
        ) -> None:
            if exchange is not None:
                await exchange.acomplete()
            if (
                completed_observer is None
                or self._capture is None
                or capture_adapter is None
                or not 200 <= upstream_response.status_code < 300
            ):
                return
            try:
                request_body = body_tee.take_body()
                response_body = completed_observer.finish()
            except Exception as exc:
                logger.warning("dropping failed response capture: %s", exc)
                self._capture.record_drop("stream_observer_error")
                return
            if request_body is None:
                logger.warning(
                    "dropping trace after request body overflow: size=%d limit=%d",
                    body_tee.size,
                    body_tee.limit,
                )
                self._capture.record_drop("body_overflow")
                return
            if response_body is None:
                logger.warning(
                    "dropping trace after response body overflow: size=%d limit=%d",
                    completed_observer.size,
                    completed_observer.limit,
                )
                self._capture.record_drop("body_overflow")
                return
            self._capture.submit_exchange(
                request_body,
                response_body,
                adapter=capture_adapter,
                request_path=path,
                method=request.method,
                content_type=content_type,
                content_encoding=content_encoding,
                timestamp=started_at,
                exchange_id=exchange.id if exchange is not None else None,
            )

        async def capture_incomplete(reason: str) -> None:
            if exchange is not None:
                await exchange.aabort(reason)

        relay = _ResponseRelay(
            upstream_response,
            observer=observer,
            on_complete=capture_complete,
            on_finish=self._activity.end,
            on_observer_error=(
                lambda: self._capture.record_drop("stream_observer_error")
                if self._capture is not None
                else None
            ),
            on_incomplete=capture_incomplete,
            queue_chunks=self._relay_queue_chunks,
            detached_drain_timeout=(
                None if exchange is not None else self._detached_drain_timeout
            ),
        )
        self._relays.add(relay.task)
        relay.task.add_done_callback(self._relay_done)

        response = StreamingResponse(
            relay.stream(),
            status_code=upstream_response.status_code,
        )
        response.raw_headers = response_headers
        return response

    async def _start_exchange(
        self,
        request: Request,
        *,
        started_at: float,
    ) -> ExchangeRecorder | None:
        ledger = self._exchange_ledger
        if ledger is None:
            return None
        return await ledger.astart(
            method=request.method,
            path=request.url.path,
            query=request.scope.get("query_string", b""),
            request_headers=request.headers.raw,
            started_at=started_at,
        )

    def _relay_done(self, task: asyncio.Task[None]) -> None:
        self._relays.discard(task)
        if task.cancelled():
            if self._capture is not None:
                self._capture.record_drop("shutdown_pending")
            return
        error = task.exception()
        if error is not None:
            logger.warning("upstream response relay failed: %s", error)


def _is_allowed_path(path: str, *, method: str = "GET") -> bool:
    """Allow public vLLM APIs without enumerating inference endpoints.

    OpenAI-compatible ``/v1/*`` APIs remain available for every HTTP method.
    Other vLLM inference surfaces are accepted for POST, which covers native
    generate/tokenize/score-style harnesses without exposing observability or
    management GET routes.
    """
    if not path.startswith("/") or path.startswith("//"):
        return False
    normalized = posixpath.normpath(path)
    candidate = normalized.rstrip("/") or "/"
    if path.rstrip("/") != candidate:
        return False
    if candidate in _BLOCKED_V1_PATHS or candidate in {
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/reset_prefix_cache",
    }:
        return False
    if candidate == "/health":
        return True
    if candidate.startswith("/v1/"):
        return True
    first_segment = candidate.removeprefix("/").split("/", 1)[0].lower()
    if first_segment.startswith(_CONTROL_ACTION_PREFIXES):
        return False
    return method.upper() == "POST"


def _strip_hop_by_hop(
    headers: list[tuple[bytes, bytes]],
    *,
    strip_host: bool = False,
) -> list[tuple[bytes, bytes]]:
    blocked = set(_HOP_BY_HOP)
    for name, value in headers:
        if name.lower() == b"connection":
            blocked.update(
                token.strip().lower()
                for token in value.split(b",")
                if token.strip()
            )
    if strip_host:
        blocked.add(b"host")
    return [(name, value) for name, value in headers if name.lower() not in blocked]


async def _wait_for_disconnect(request: Request) -> None:
    """Wait for disconnect after the request body has one exclusive consumer."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return
