from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.capture import CaptureManager, capture_timestamp, decode_request_body
from speedlm.gateway.sse import AssembledResponse, SSEAssembler, parse_json_response

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
_CAPTURE_ENDPOINTS = {"/v1/chat/completions", "/v1/completions"}
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
_MAX_CAPTURE_BODY_BYTES = 32 * 1024 * 1024


class _BodyTee:
    def __init__(self, source: AsyncIterator[bytes], *, limit: int) -> None:
        self._source = source
        self._limit = limit
        self._captured = bytearray()
        self._overflowed = False

    async def stream(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            if chunk and not self._overflowed:
                if len(self._captured) + len(chunk) <= self._limit:
                    self._captured.extend(chunk)
                else:
                    self._overflowed = True
                    self._captured.clear()
            if chunk:
                yield chunk

    def body(self) -> bytes | None:
        return None if self._overflowed else bytes(self._captured)


class _ResponseObserver:
    def __init__(self, endpoint: str, content_type: str) -> None:
        self._endpoint = endpoint
        self._sse = (
            SSEAssembler(endpoint)
            if content_type.lower().startswith("text/event-stream")
            else None
        )
        self._body = bytearray()
        self._overflowed = False

    def feed(self, chunk: bytes) -> None:
        if self._sse is not None:
            self._sse.feed(chunk)
            return
        if self._overflowed:
            return
        if len(self._body) + len(chunk) <= _MAX_CAPTURE_BODY_BYTES:
            self._body.extend(chunk)
        else:
            self._overflowed = True
            self._body.clear()

    def finish(self) -> AssembledResponse | None:
        if self._sse is not None:
            result = self._sse.finish()
            return result if self._sse.valid else None
        if self._overflowed:
            return None
        return parse_json_response(bytes(self._body), self._endpoint)


class GatewayProxy:
    """A streaming reverse proxy backed by one shared ``AsyncClient``."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        upstream_base_url: str,
        *,
        activity: ActivityTracker,
        capture: CaptureManager | None,
    ) -> None:
        self._client = client
        self._upstream = httpx.URL(upstream_base_url)
        self._activity = activity
        self._capture = capture

    async def handle(self, request: Request) -> Response:
        path = request.url.path
        if not _is_allowed_path(path):
            return PlainTextResponse("Not Found", status_code=404)

        started_at = capture_timestamp()
        body_tee = _BodyTee(request.stream(), limit=_MAX_CAPTURE_BODY_BYTES)
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

        self._activity.begin()
        try:
            upstream_response = await self._client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            self._activity.end()
            logger.error("upstream request failed: %s", exc)
            return PlainTextResponse("Bad Gateway", status_code=502)
        except asyncio.CancelledError:
            self._activity.end()
            raise
        except Exception:
            self._activity.end()
            raise

        response_headers = _strip_hop_by_hop(upstream_response.headers.raw)
        content_type = upstream_response.headers.get("content-type", "")
        observer = (
            _ResponseObserver(path, content_type)
            if self._capture is not None
            and path in _CAPTURE_ENDPOINTS
            and 200 <= upstream_response.status_code < 300
            else None
        )

        async def response_stream() -> AsyncIterator[bytes]:
            nonlocal observer
            completed = False
            try:
                async for chunk in upstream_response.aiter_raw():
                    if observer is not None:
                        try:
                            observer.feed(chunk)
                        except Exception as exc:
                            logger.warning("dropping failed response capture: %s", exc)
                            observer = None
                    yield chunk
                completed = True
            finally:
                try:
                    await upstream_response.aclose()
                finally:
                    self._activity.end()
                    if (
                        completed
                        and observer is not None
                        and self._capture is not None
                    ):
                        try:
                            request_body = body_tee.body()
                            request_data = (
                                decode_request_body(request_body)
                                if request_body is not None
                                else None
                            )
                            assembled = observer.finish()
                            if request_data is not None and assembled is not None:
                                self._capture.submit(
                                    request_data,
                                    assembled,
                                    endpoint=path,
                                    timestamp=started_at,
                                )
                        except Exception as exc:
                            logger.warning("dropping failed response capture: %s", exc)

        response = StreamingResponse(
            response_stream(),
            status_code=upstream_response.status_code,
        )
        response.raw_headers = response_headers
        return response


def _is_allowed_path(path: str) -> bool:
    return (
        path == "/health"
        or path.startswith("/v1/")
        and path.rstrip("/") not in _BLOCKED_V1_PATHS
    )


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
