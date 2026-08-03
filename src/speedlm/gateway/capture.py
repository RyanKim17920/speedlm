from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from speedlm.config import SamplingConfig
from speedlm.gateway.responses import (
    parse_responses_response,
    parse_responses_sse,
    response_status_and_id,
    responses_request_messages,
)
from speedlm.gateway.sse import (
    AssembledResponse,
    SSEAssembler,
    parse_json_response,
    parse_json_responses,
)
from speedlm.gateway.worker import await_worker
from speedlm.traces.normalize import (
    SAMPLING_PROVENANCE_KEY,
    SAMPLING_SOURCE_CLIENT,
    SAMPLING_SOURCE_SERVER_DEFAULT,
    is_prefill_continuation,
    merge_assistant_prefill,
    normalize_record,
)
from speedlm.traces.store import TraceStore

logger = logging.getLogger(__name__)

_REASONING_FIELDS = {"reasoning_content", "reasoning", "thinking"}

# ── Sampling capture ────────────────────────────────────────────────────────
#
# Every field below changes what the engine generated, so anything omitted from
# the capture is a silent rewrite of the row's own decoding story. The previous
# allowlist was {temperature, top_p, seed}, which dropped stop strings (a
# stop-string truncation then looked like a natural turn ending), penalties,
# top_k/min_p, length caps, and the chat-template switches that decide how the
# prompt was rendered.
_RECORDED_REQUEST_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "bad_words",
    "allowed_token_ids",
    "logit_bias",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "min_tokens",
    "truncate_prompt_tokens",
    "n",
    "best_of",
    "response_format",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "tool_choice",
    "parallel_tool_calls",
    "add_generation_prompt",
    "continue_final_message",
    "chat_template",
    "chat_template_kwargs",
    "echo",
    "suffix",
)

#: Request keys that are transport/routing or are already captured elsewhere on
#: the record (``messages``/``prompt`` become the conversation, ``tools`` becomes
#: ``TraceRecord.tools``, ``model`` becomes ``TraceRecord.model``). Anything on a
#: request that is in neither this set nor the recorded set is reported in
#: ``unrecorded_request_fields`` so the allowlist's blind spots are stated on the
#: row instead of being invisible.
_STRUCTURAL_REQUEST_FIELDS: frozenset[str] = frozenset({
    "messages",
    "prompt",
    "input",
    "instructions",
    "model",
    "tools",
    "stream",
    "stream_options",
    "user",
    "metadata",
    "store",
    "request_id",
    "priority",
    "cache_salt",
})

# vLLM 0.25.1 (the pinned serving engine, see pyproject.toml) declares every
# sampling field as ``Optional[...] = None`` on ChatCompletionRequest and only
# substitutes a real value inside ``to_sampling_params``:
#   vllm/entrypoints/openai/chat_completion/protocol.py
#     :218   temperature: float | None = None
#     :562   _DEFAULT_SAMPLING_PARAMS = {"repetition_penalty": 1.0,
#            "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
#     :599   temperature = default_sampling_params.get(
#                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
# So an omitted temperature means the engine sampled at 1.0 — NOT greedy. That
# is why an absent field must never be backfilled with 0.0.
VLLM_ENGINE_SAMPLING_DEFAULTS: Mapping[str, float | int] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}

#: Plain pydantic field defaults on the request model. Unlike the five above,
#: these are not reachable by a model's ``generation_config.json``.
VLLM_REQUEST_FIELD_DEFAULTS: Mapping[str, float | int] = {
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
}

_SERVER_DEFAULTS: Mapping[str, float | int] = {
    **VLLM_ENGINE_SAMPLING_DEFAULTS,
    **VLLM_REQUEST_FIELD_DEFAULTS,
}

SERVER_DEFAULTS_ORIGIN = "vllm-0.25.1-openai-chat-completions"


@dataclass(frozen=True, slots=True)
class CaptureAdapter:
    """One pluggable request matcher and exchange decoder."""

    name: str
    matches: Callable[[str, str], bool]
    decode_response: Callable[
        [bytes | bytearray, str],
        AssembledResponse | Sequence[AssembledResponse] | None,
    ]
    build_record: Callable[
        [Mapping[str, Any], AssembledResponse, float],
        dict[str, Any],
    ]
    background_correlation: bool = False


class CaptureManager:
    """Validate and append completed captures without delaying proxy responses."""

    def __init__(
        self,
        store: TraceStore,
        *,
        defaults: SamplingConfig | None = None,
        adapters: Sequence[CaptureAdapter] | None = None,
    ) -> None:
        self._store = store
        self._defaults = defaults or SamplingConfig()
        self._tasks: set[asyncio.Task[None]] = set()
        self._exchange_lock = asyncio.Lock()
        self._pending_responses: dict[str, dict[str, Any]] = {}
        self._adapters = tuple(adapters) if adapters is not None else default_adapters()
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="speedlm-trace-decoder",
        )
        self._closed = False

    def match(self, method: str, path: str) -> CaptureAdapter | None:
        """Return the first registered adapter matching this exchange."""
        for adapter in self._adapters:
            if adapter.matches(method, path):
                return adapter
        return None

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
        task.add_done_callback(self._task_done)

    def submit_exchange(
        self,
        request_body: bytes | bytearray,
        response_body: bytes | bytearray,
        *,
        adapter: CaptureAdapter,
        request_path: str,
        method: str,
        content_type: str,
        content_encoding: str,
        timestamp: float,
        exchange_id: str | None = None,
    ) -> None:
        """Schedule raw exchange parsing and persistence off the relay path."""
        task = asyncio.create_task(
            self._capture_exchange(
                request_body,
                response_body,
                adapter=adapter,
                request_path=request_path,
                method=method,
                content_type=content_type,
                content_encoding=content_encoding,
                timestamp=timestamp,
                exchange_id=exchange_id,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def record_drop(self, reason: str) -> None:
        """Schedule best-effort drop accounting off the response path."""
        task = asyncio.create_task(self._record_drop(reason))
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            return
        try:
            self._store.record_drop("shutdown_pending")
        except Exception as exc:
            logger.warning("failed to count pending trace at shutdown: %s", exc)

    async def drain(self) -> None:
        """Wait for already scheduled writes, primarily during application shutdown."""
        while self._tasks:
            pending = tuple(self._tasks)
            await asyncio.gather(*pending, return_exceptions=True)
            self._tasks.difference_update(pending)

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.drain()
        self._closed = True
        self._executor.shutdown(wait=True)

    async def _run_in_worker(
        self,
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        if self._closed:
            raise RuntimeError("capture manager is closed")
        return await await_worker(
            self._executor.submit(partial(function, *args))
        )

    async def _record_drop(self, reason: str) -> None:
        try:
            await self._run_in_worker(self._store.record_drop, reason)
        except Exception as exc:
            logger.warning("failed to count dropped trace (%s): %s", reason, exc)

    async def _capture_exchange(
        self,
        request_body: bytes | bytearray,
        response_body: bytes | bytearray,
        *,
        adapter: CaptureAdapter,
        request_path: str,
        method: str,
        content_type: str,
        content_encoding: str,
        timestamp: float,
        exchange_id: str | None,
    ) -> None:
        del request_path  # Reserved for endpoint-specific correlation diagnostics.
        # Preserve completion order for background Responses API POST->GET
        # correlation and serialize writes without touching the response relay.
        async with self._exchange_lock:
            try:
                decoded_response_body = await self._run_in_worker(
                    _decode_content_encoding,
                    response_body,
                    content_encoding,
                )
                request_data = await self._run_in_worker(
                    decode_request_body,
                    request_body,
                )
                decoded = await self._run_in_worker(
                    adapter.decode_response,
                    decoded_response_body,
                    content_type,
                )
                responses = (
                    (decoded,)
                    if isinstance(decoded, AssembledResponse)
                    else tuple(decoded or ())
                )
                response = responses[0] if responses else None

                if adapter.background_correlation:
                    response_id, status = await self._run_in_worker(
                        response_status_and_id,
                        decoded_response_body,
                        content_type,
                    )
                    if (
                        method.upper() == "POST"
                        and request_data is not None
                        and not responses
                        and response_id is not None
                        and status in {"queued", "in_progress"}
                    ):
                        self._pending_responses[response_id] = request_data
                        return
                    if request_data is None and response is not None:
                        request_data = self._pending_responses.pop(
                            response.id or response_id or "",
                            None,
                        )
                    elif response is not None and response.id is not None:
                        self._pending_responses.pop(response.id, None)

                if request_data is None:
                    raise ValueError("capture request body is not a JSON object")
                if not responses:
                    raise ValueError("capture response could not be reconstructed")
                for decoded_response in responses:
                    await self._capture(
                        request_data,
                        decoded_response,
                        endpoint=adapter.name,
                        timestamp=timestamp,
                        record_builder=adapter.build_record,
                        exchange_id=exchange_id,
                    )
            except Exception as exc:
                await self._record_drop("capture_error")
                logger.warning("dropping failed raw exchange capture: %s", exc)

    async def _capture(
        self,
        request_data: Mapping[str, Any],
        response: AssembledResponse,
        *,
        endpoint: str,
        timestamp: float,
        record_builder: Callable[
            [Mapping[str, Any], AssembledResponse, float],
            dict[str, Any],
        ]
        | None = None,
        exchange_id: str | None = None,
    ) -> None:
        try:
            raw_record = (
                record_builder(request_data, response, timestamp)
                if record_builder is not None
                else _build_raw_record(
                    request_data,
                    response,
                    endpoint=endpoint,
                    timestamp=timestamp,
                )
            )
            default_model = request_data.get("model")
            record = normalize_record(
                raw_record,
                defaults=self._defaults,
                default_model=default_model if isinstance(default_model, str) else None,
            )
            record = replace(
                record,
                finish_reason=response.finish_reason,
                stop_reason=response.stop_reason,
                exchange_id=exchange_id,
            )
            await self._run_in_worker(self._store.append, record)
        except Exception as exc:
            await self._record_drop("capture_error")
            logger.warning("dropping failed trace capture: %s", exc)


def decode_request_body(body: bytes | bytearray) -> dict[str, Any] | None:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sampling_provenance(
    request_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (effective sampling params, provenance block) for one request.

    Params carry the client's value when the request had one and the serving
    engine's documented default when it did not. The block says which, per key,
    so a consumer can separate a request that asked for greedy decoding from one
    that simply said nothing and was sampled at the engine default.
    """
    params: dict[str, Any] = {}
    sources: dict[str, str] = {}
    unspecified: list[str] = []
    for key in _RECORDED_REQUEST_FIELDS:
        if key in request_data:
            params[key] = request_data[key]
            sources[key] = SAMPLING_SOURCE_CLIENT
        elif key in _SERVER_DEFAULTS:
            params[key] = _SERVER_DEFAULTS[key]
            sources[key] = SAMPLING_SOURCE_SERVER_DEFAULT
        else:
            # Absent from the request and with no engine default we can state
            # (e.g. `seed`, whose absence means a fresh random seed we never
            # observed). Listed rather than guessed.
            unspecified.append(key)

    block: dict[str, Any] = {
        "sources": sources,
        "params": params,
        "unspecified": unspecified,
        "server_defaults_origin": SERVER_DEFAULTS_ORIGIN,
        # With `--generation-config auto` (vLLM's default) a served model's
        # generation_config.json overrides these five, so a `server_default`
        # label on them is an assumption about the deployment, not a reading.
        "server_defaults_model_overridable": sorted(VLLM_ENGINE_SAMPLING_DEFAULTS),
        "unrecorded_request_fields": sorted(
            key
            for key in request_data
            if key not in sources
            and key not in unspecified
            and key not in _STRUCTURAL_REQUEST_FIELDS
        ),
    }
    return params, block


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
            {**message, "provenance_tag": "client_supplied"}
            if isinstance(message, dict)
            else message
            for message in raw_messages
        ]
    elif endpoint == "/v1/completions":
        prompt = request_data.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise ValueError("completion request has no supported prompt")
        prompt = _completion_prompt_for_choice(
            prompt,
            response=response,
            requested_n=request_data.get("n", 1),
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
                "provenance_tag": "client_supplied",
            }
        ]
    elif endpoint == "/v1/responses":
        messages = responses_request_messages(request_data)
        if not messages:
            raise ValueError("responses request has no recoverable input")
    else:
        raise ValueError(f"unsupported capture endpoint: {endpoint}")

    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
        "provenance_tag": "generated",
    }
    if response.tool_calls:
        assistant["tool_calls"] = [dict(call) for call in response.tool_calls]
    if response.reasoning_content is not None:
        assistant["reasoning_content"] = response.reasoning_content
        if (
            response.reasoning_field in _REASONING_FIELDS
            and response.reasoning_field != "reasoning_content"
        ):
            assistant[response.reasoning_field] = response.reasoning_content

    sampling_params, provenance = sampling_provenance(request_data)
    assistant[SAMPLING_PROVENANCE_KEY] = provenance

    if is_prefill_continuation(request_data, messages):
        # The engine continued the client's final assistant message, so this is
        # one turn. Appending would invent a turn boundary the model never saw.
        messages[-1] = merge_assistant_prefill(messages[-1], assistant)
    else:
        messages.append(assistant)

    result: dict[str, Any] = {
        "timestamp": response.created if response.created is not None else timestamp,
        "messages": messages,
        "tool_calls": [dict(call) for call in response.tool_calls],
        "usage": _usage_for_response(response),
        "finish_reason": response.finish_reason,
        "stop_reason": response.stop_reason,
    }
    request_tools = request_data.get("tools")
    if isinstance(request_tools, list) and all(
        isinstance(tool, Mapping) for tool in request_tools
    ):
        result["tools"] = [dict(tool) for tool in request_tools]
    if response.id is not None:
        result["id"] = (
            f"{response.id}:choice:{response.choice_index}"
            if response.choice_count > 1
            else response.id
        )
    response_model = response.model
    request_model = request_data.get("model")
    if response_model is not None:
        result["model"] = response_model
    elif isinstance(request_model, str):
        result["model"] = request_model

    # TraceRecord has fields for exactly these three; the rest of the recorded
    # set survives on the generated message's provenance block. Values come from
    # `sampling_params`, not from `request_data`, so an omitted temperature is
    # stored as the 1.0 the engine actually applied rather than as a fabricated
    # 0.0 that would be byte-identical to a genuine greedy row.
    for key in ("temperature", "top_p", "seed"):
        if key in sampling_params:
            result[key] = sampling_params[key]
    return result


def capture_timestamp() -> float:
    return time.time()


def _completion_prompt_for_choice(
    prompt: str | list[Any],
    *,
    response: AssembledResponse,
    requested_n: Any,
) -> str | list[Any]:
    if isinstance(prompt, str) or not prompt:
        return prompt
    # A flat token-id list is one tokenized prompt, not a batch.
    if all(
        not isinstance(token, bool) and isinstance(token, int)
        for token in prompt
    ):
        return prompt
    n = (
        requested_n
        if not isinstance(requested_n, bool)
        and isinstance(requested_n, int)
        and requested_n > 0
        else 1
    )
    prompt_index = response.choice_index // n
    if prompt_index >= len(prompt):
        raise ValueError(
            f"completion choice {response.choice_index} has no matching prompt"
        )
    selected = prompt[prompt_index]
    if not isinstance(selected, (str, list)):
        raise ValueError("completion batch contains an unsupported prompt")
    return selected


def _usage_for_response(response: AssembledResponse) -> dict[str, int | None]:
    if response.choice_count > 1:
        # vLLM reports aggregate request usage for n>1/batches. Copying it to
        # every pair would claim false per-output measurements; normalization
        # will estimate each correlated pair instead.
        return {"prompt_tokens": None, "completion_tokens": None}
    return {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }


def _decode_response_body(
    body: bytes | bytearray,
    *,
    endpoint: str,
    content_type: str,
) -> AssembledResponse | None:
    is_sse = content_type.lower().startswith("text/event-stream")
    if endpoint == "/v1/responses":
        return parse_responses_sse(body) if is_sse else parse_responses_response(body)
    if is_sse:
        assembler = SSEAssembler(endpoint)
        assembler.feed(bytes(body))
        result = assembler.finish()
        return result if assembler.valid else None
    return parse_json_response(bytes(body), endpoint)


def _decode_content_encoding(
    body: bytes | bytearray,
    content_encoding: str,
) -> bytes | bytearray:
    encodings = [
        value.strip().lower()
        for value in content_encoding.split(",")
        if value.strip() and value.strip().lower() != "identity"
    ]
    decoded = bytes(body)
    for encoding in reversed(encodings):
        if encoding == "gzip":
            decoded = gzip.decompress(decoded)
        elif encoding == "deflate":
            try:
                decoded = zlib.decompress(decoded)
            except zlib.error:
                decoded = zlib.decompress(decoded, -zlib.MAX_WBITS)
        else:
            raise ValueError(f"unsupported capture content encoding: {encoding}")
    return decoded


def _decode_response_bodies(
    body: bytes | bytearray,
    *,
    endpoint: str,
    content_type: str,
) -> tuple[AssembledResponse, ...]:
    if content_type.lower().startswith("text/event-stream"):
        assembler = SSEAssembler(endpoint)
        assembler.feed(bytes(body))
        return assembler.finish_all()
    return parse_json_responses(bytes(body), endpoint)


def default_adapters() -> tuple[CaptureAdapter, ...]:
    """Built-in protocol adapters; callers may replace or extend this registry."""
    return (
        CaptureAdapter(
            name="/v1/chat/completions",
            matches=_match_chat_completions,
            decode_response=_chat_response_decoder,
            build_record=_chat_record_builder,
        ),
        CaptureAdapter(
            name="/v1/chat/completions/batch",
            matches=_match_chat_batch,
            decode_response=_chat_response_decoder,
            build_record=_chat_batch_record_builder,
        ),
        CaptureAdapter(
            name="/v1/completions",
            matches=_match_completions,
            decode_response=_completions_response_decoder,
            build_record=_completions_record_builder,
        ),
        CaptureAdapter(
            name="/v1/responses",
            matches=_match_responses,
            decode_response=_responses_response_decoder,
            build_record=_responses_record_builder,
            background_correlation=True,
        ),
        CaptureAdapter(
            name="openai-structural-fallback",
            matches=_match_structural_fallback,
            decode_response=_structural_response_decoder,
            build_record=_structural_record_builder,
        ),
    )


def _match_chat_completions(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/") == "/v1/chat/completions"


def _match_chat_batch(method: str, path: str) -> bool:
    return (
        method.upper() == "POST"
        and path.rstrip("/") == "/v1/chat/completions/batch"
    )


def _match_completions(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/") == "/v1/completions"


def _match_responses(method: str, path: str) -> bool:
    normalized = path.rstrip("/")
    if method.upper() == "POST":
        return normalized == "/v1/responses"
    return (
        method.upper() == "GET"
        and normalized.startswith("/v1/responses/")
        and not normalized.endswith("/cancel")
    )


def _match_structural_fallback(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.startswith("/v1/")


def _chat_response_decoder(
    body: bytes | bytearray,
    content_type: str,
) -> tuple[AssembledResponse, ...]:
    return _decode_response_bodies(
        body,
        endpoint="/v1/chat/completions",
        content_type=content_type,
    )


def _completions_response_decoder(
    body: bytes | bytearray,
    content_type: str,
) -> tuple[AssembledResponse, ...]:
    return _decode_response_bodies(
        body,
        endpoint="/v1/completions",
        content_type=content_type,
    )


def _responses_response_decoder(
    body: bytes | bytearray,
    content_type: str,
) -> tuple[AssembledResponse, ...]:
    response = _decode_response_body(
        body,
        endpoint="/v1/responses",
        content_type=content_type,
    )
    return (response,) if response is not None else ()


def _structural_response_decoder(
    body: bytes | bytearray,
    content_type: str,
) -> tuple[AssembledResponse, ...]:
    for endpoint in ("/v1/chat/completions", "/v1/completions"):
        responses = _decode_response_bodies(
            body,
            endpoint=endpoint,
            content_type=content_type,
        )
        if responses:
            return responses
    response = _responses_response_decoder(body, content_type)
    return response


def _chat_record_builder(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    return _build_raw_record(
        request_data,
        response,
        endpoint="/v1/chat/completions",
        timestamp=timestamp,
    )


def _chat_batch_record_builder(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    conversations = request_data.get("messages")
    if not isinstance(conversations, list):
        raise ValueError("chat batch request has no messages batch")
    if response.choice_index >= len(conversations):
        raise ValueError(
            f"chat batch choice {response.choice_index} has no matching input"
        )
    messages = conversations[response.choice_index]
    if not isinstance(messages, list):
        raise ValueError("chat batch input is not a conversation")
    single_request = {**request_data, "messages": messages, "n": 1}
    return _build_raw_record(
        single_request,
        response,
        endpoint="/v1/chat/completions",
        timestamp=timestamp,
    )


def _completions_record_builder(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    return _build_raw_record(
        request_data,
        response,
        endpoint="/v1/completions",
        timestamp=timestamp,
    )


def _responses_record_builder(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    return _build_raw_record(
        request_data,
        response,
        endpoint="/v1/responses",
        timestamp=timestamp,
    )


def _structural_record_builder(
    request_data: Mapping[str, Any],
    response: AssembledResponse,
    timestamp: float,
) -> dict[str, Any]:
    if isinstance(request_data.get("messages"), list):
        endpoint = "/v1/chat/completions"
    elif isinstance(request_data.get("prompt"), (str, list)):
        endpoint = "/v1/completions"
    elif "input" in request_data or "instructions" in request_data:
        endpoint = "/v1/responses"
    else:
        raise ValueError("structural fallback found no generative request input")
    return _build_raw_record(
        request_data,
        response,
        endpoint=endpoint,
        timestamp=timestamp,
    )
