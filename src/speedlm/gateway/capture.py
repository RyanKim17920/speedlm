from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from speedlm.config import SamplingConfig
from speedlm.gateway.sse import AssembledResponse
from speedlm.traces.normalize import normalize_record
from speedlm.traces.store import TraceStore

logger = logging.getLogger(__name__)


class CaptureManager:
    """Validate and append completed captures without delaying proxy responses."""

    def __init__(
        self,
        store: TraceStore,
        *,
        defaults: SamplingConfig | None = None,
    ) -> None:
        self._store = store
        self._defaults = defaults or SamplingConfig()
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(
        self,
        request_data: Mapping[str, Any],
        response: AssembledResponse,
        *,
        endpoint: str,
        timestamp: float,
    ) -> None:
        """Schedule one capture; validation and I/O failures are logged and dropped."""
        task = asyncio.create_task(
            self._capture(request_data, response, endpoint=endpoint, timestamp=timestamp)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for already scheduled writes, primarily during application shutdown."""
        while self._tasks:
            pending = tuple(self._tasks)
            await asyncio.gather(*pending)
            self._tasks.difference_update(pending)

    async def _capture(
        self,
        request_data: Mapping[str, Any],
        response: AssembledResponse,
        *,
        endpoint: str,
        timestamp: float,
    ) -> None:
        try:
            raw_record = _build_raw_record(
                request_data,
                response,
                endpoint=endpoint,
                timestamp=timestamp,
            )
            default_model = request_data.get("model")
            record = normalize_record(
                raw_record,
                defaults=self._defaults,
                default_model=default_model if isinstance(default_model, str) else None,
            )
            await asyncio.to_thread(self._store.append, record)
        except Exception as exc:
            logger.warning("dropping failed trace capture: %s", exc)


def decode_request_body(body: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _build_raw_record(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    *,
    endpoint: str,
    timestamp: float,
) -> dict[str, Any]:
    if endpoint == "/v1/chat/completions":
        raw_messages = request_data.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("chat request has no messages list")
        messages = [
            dict(message) if isinstance(message, dict) else message
            for message in raw_messages
        ]
    elif endpoint == "/v1/completions":
        prompt = request_data.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise ValueError("completion request has no supported prompt")
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"unsupported capture endpoint: {endpoint}")

    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        assistant["tool_calls"] = [dict(call) for call in response.tool_calls]
    messages.append(assistant)

    result: dict[str, Any] = {
        "timestamp": response.created if response.created is not None else timestamp,
        "messages": messages,
        "tool_calls": [dict(call) for call in response.tool_calls],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        },
    }
    if response.id is not None:
        result["id"] = response.id
    response_model = response.model
    request_model = request_data.get("model")
    if response_model is not None:
        result["model"] = response_model
    elif isinstance(request_model, str):
        result["model"] = request_model

    for key in ("temperature", "top_p", "seed"):
        if key in request_data:
            result[key] = request_data[key]
    return result


def capture_timestamp() -> float:
    return time.time()
