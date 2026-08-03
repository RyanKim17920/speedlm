from __future__ import annotations

import datetime
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.config import SamplingConfig
from speedlm.traces.store import TraceRecord, estimate_message_tokens

# ── Exceptions ──────────────────────────────────────────────────────────────

class NormalizeError(ValueError):
    """Raised when external input cannot be normalized (fail closed)."""


# ── Sampling provenance ─────────────────────────────────────────────────────
#
# The gate scores the draft head on greedy (temperature 0) argmax agreement, so
# whether a row was produced greedily or by sampling decides whether it is a
# valid training target. A recorded ``temperature`` is therefore only usable
# together with a statement of where that number came from.
#
# The provenance rides on the *generated* message rather than on the record
# because ``TraceRecord`` has no field for it and ``TraceRecord.from_dict``
# rejects unknown top-level keys (traces/store.py) — a new top-level key would
# make every new trace unreadable by any reader that has not been upgraded in
# lockstep. Messages are free-form dicts that already carry per-turn provenance
# (``provenance_tag``), so this follows an established convention, round-trips
# through ``to_dict``/``from_dict`` untouched, and is simply ignored by older
# readers. Records written before this field existed carry no block at all;
# consumers must read a missing block as "provenance unknown", never as greedy.
SAMPLING_PROVENANCE_KEY = "sampling_provenance"

#: The request carried the field explicitly (observed at the gateway).
SAMPLING_SOURCE_CLIENT = "client_explicit"
#: The request omitted the field; the recorded value is the default the serving
#: engine applied. A default, not an observation.
SAMPLING_SOURCE_SERVER_DEFAULT = "server_default"
#: The value was present on an ingested external record. An observation, but
#: whether the client or some intermediary set it is not knowable from the file.
SAMPLING_SOURCE_RECORD = "record_explicit"
#: No value anywhere. The number stored on the record is a configured
#: placeholder and must not be read as a description of how the row was decoded.
SAMPLING_SOURCE_UNKNOWN = "unknown"

#: Message keys that describe the capture rather than the conversation. They are
#: excluded from token estimation so provenance bookkeeping cannot inflate the
#: token counts that the buffer and the trainer read.
_NON_CONTENT_MESSAGE_KEYS = frozenset({
    SAMPLING_PROVENANCE_KEY,
    "history_truncated",
    "prefill_prefix_chars",
})


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Rejection:
    """A single rejected line from an external JSONL file."""

    line: int  # 1-based line number
    reason: str
    raw: str  # truncated to 500 chars


@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Outcome of normalizing an external JSONL file."""

    accepted: tuple[TraceRecord, ...]
    rejected: tuple[Rejection, ...]
    detected_shapes: tuple[str, ...] = ()

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def shape_counts(self) -> dict[str, int]:
        """Accepted records grouped by their detected external shape."""
        return dict(Counter(self.detected_shapes))


# ── Helpers ─────────────────────────────────────────────────────────────────

def _generate_id(data: Mapping[str, Any]) -> str:
    """Deterministic ID: ``"tr-" + sha256(canonical_json)[:16]``."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"tr-{digest}"


def _parse_timestamp_to_epoch(value: Any, *, index: int) -> float:
    """Convert a timestamp value to epoch-seconds float.

    Accepts:
    - int or float (epoch seconds)
    - ISO-8601 string (e.g., "2026-07-25T04:00:00Z", "+00:00" offset, naive)

    Raises NormalizeError on malformed input.
    """
    if isinstance(value, bool):
        raise NormalizeError(
            f"record[{index}]: "
            f"'timestamp' must be a number or ISO-8601 string (got bool)"
        )
    if isinstance(value, (int, float)):
        #: Python ints are unbounded, so ``math.isfinite`` on a JSON integer
        #: wider than a double raises OverflowError rather than returning
        #: False. Convert first so an out-of-range epoch is a declared
        #: rejection instead of an escaping OverflowError.
        try:
            value = float(value)
        except OverflowError:
            raise NormalizeError(
                f"record[{index}]: 'timestamp' is too large to represent"
            ) from None
        if not math.isfinite(value):
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be a finite number"
            )
        if value < 0:
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be >= 0"
            )
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise NormalizeError(
                f"record[{index}]: "
                f"'timestamp' is not a valid ISO-8601 string: {exc}"
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        epoch = dt.timestamp()
        if not math.isfinite(epoch):
            raise NormalizeError(
                f"record[{index}]: 'timestamp' converted to a non-finite epoch"
            )
        if epoch < 0:
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be >= 0"
            )
        return epoch
    raise NormalizeError(
        f"record[{index}]: 'timestamp' must be a number or "
        f"ISO-8601 string (got {type(value).__name__})"
    )


def _detect_shape(data: Mapping[str, Any], *, index: int) -> str:
    """Structurally identify one supported external record envelope."""
    has_choices = "choices" in data
    reply_fields = [key for key in ("response", "completion") if key in data]
    has_pair_envelope = "request" in data or bool(reply_fields)

    if has_pair_envelope:
        return "request-response"
    if has_choices and "messages" in data:
        return "proxy-capture"
    if has_choices:
        return "openai-response"
    if "messages" in data:
        has_tokens = (
            "usage" in data
            or "prompt_tokens" in data
            or "completion_tokens" in data
        )
        return "internal" if has_tokens else "bare-conversation"
    if _has_recoverable_conversation(data):
        return "proxy-capture"
    raise NormalizeError(
        f"record[{index}]: cannot recover a conversation: no messages "
        "or assistant content found"
    )


def _choice_message(
    response: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NormalizeError(
            f"record[{index}]: openai-response 'choices' must be a non-empty list"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise NormalizeError(
            f"record[{index}]: choices[0] must be an object"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise NormalizeError(
            f"record[{index}]: choices[0].message must be an object"
        )
    result = dict(message)
    result.setdefault("role", "assistant")
    result.setdefault("content", None)
    return result


def _openai_response_to_internal(
    data: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    assistant = _choice_message(data, index=index)

    normalized = dict(data)
    normalized.pop("choices", None)
    # A bare OpenAI *response* object carries no request, so the prompt that
    # produced this reply is not in the record and cannot be recovered. The
    # single-message conversation below is therefore a fragment, not a
    # conversation; flag it so a consumer does not read it as a complete turn
    # and train on a reply whose context it never saw.
    assistant = {**assistant, "history_truncated": True}
    normalized["messages"] = [assistant]
    tool_calls = assistant.get("tool_calls")
    if tool_calls is not None:
        normalized["tool_calls"] = tool_calls
    return normalized


def is_prefill_continuation(
    request: Mapping[str, Any],
    messages: Sequence[Any],
) -> bool:
    """True when the generation continues the final assistant message.

    A trailing assistant message in a request is only a *prefill* when the
    server was told to continue it. With the OpenAI/vLLM default
    (``add_generation_prompt=True``) a trailing assistant message renders as a
    finished turn and a genuinely new assistant turn follows, so two
    consecutive assistant messages are correct there and must be left alone.
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, Mapping) or last.get("role") != "assistant":
        return False
    if request.get("continue_final_message"):
        return True
    return request.get("add_generation_prompt") is False


def merge_assistant_prefill(
    prefill: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    """Fold a client prefill and its continuation into one assistant turn.

    Appending the generated reply after a prefill would inject a turn boundary
    into what the engine rendered as one continuous generation, which is a
    prompt the model was never shown. ``prefill_prefix_chars`` records how much
    of the merged content was client-supplied so a consumer can avoid
    supervising the prefix as if the model had produced it.
    """
    merged: dict[str, Any] = {
        **prefill,
        **{key: value for key, value in generated.items() if key != "content"},
    }
    head = prefill.get("content")
    tail = generated.get("content")
    if isinstance(head, str) and isinstance(tail, str):
        merged["content"] = head + tail
        merged["prefill_prefix_chars"] = len(head)
    elif isinstance(head, list) and isinstance(tail, list):
        merged["content"] = [*head, *tail]
        merged["prefill_prefix_chars"] = None
    elif head is None or head == "":
        merged["content"] = tail
        merged["prefill_prefix_chars"] = 0
    elif tail is None or tail == "":
        merged["content"] = head
        merged["prefill_prefix_chars"] = len(head) if isinstance(head, str) else None
    else:
        # Mixed structured/plain content: keep both rather than silently
        # dropping one half of a single turn.
        merged["content"] = [head, tail]
        merged["prefill_prefix_chars"] = None
    return merged


def _append_or_merge_assistant(
    messages: list[Any],
    assistant: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> None:
    if is_prefill_continuation(request, messages):
        messages[-1] = merge_assistant_prefill(messages[-1], assistant)
    else:
        messages.append(dict(assistant))


def _assistant_reply(value: Any, *, field: str, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        return {"role": "assistant", "content": value}
    if not isinstance(value, dict):
        raise NormalizeError(
            f"record[{index}]: '{field}' must be a string or object"
        )

    raw_message = value.get("message")
    if raw_message is not None:
        if not isinstance(raw_message, dict):
            raise NormalizeError(
                f"record[{index}]: {field}.message must be an object"
            )
        message = dict(raw_message)
        message.setdefault("role", "assistant")
        message.setdefault("content", None)
        return message
    if "role" in value or "content" in value or "tool_calls" in value:
        message = dict(value)
        message.setdefault("role", "assistant")
        message.setdefault("content", None)
        return message
    if "text" in value:
        return {"role": "assistant", "content": value["text"]}
    raise NormalizeError(
        f"record[{index}]: '{field}' object must contain 'message', "
        "'content', or 'text'"
    )


def _request_response_to_internal(
    data: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    reply_fields = [key for key in ("response", "completion") if key in data]
    reply_field = reply_fields[0] if reply_fields else None

    request: Mapping[str, Any]
    request_raw = data.get("request")
    if request_raw is not None:
        if not isinstance(request_raw, dict):
            raise NormalizeError(
                f"record[{index}]: 'request' must be an object"
            )
        if "messages" in data:
            raise NormalizeError(
                f"record[{index}]: ambiguous request messages: both "
                "request.messages and top-level 'messages' are present"
            )
        request = request_raw
    else:
        request = data

    request_messages = request.get("messages")
    if not isinstance(request_messages, list) or not request_messages:
        raise NormalizeError(
            f"record[{index}]: request 'messages' must be a non-empty list"
        )

    reply_raw = data[reply_field] if reply_field is not None else None
    response_metadata: Mapping[str, Any] = {}
    assistant: dict[str, Any] | None = None
    if isinstance(reply_raw, dict) and "choices" in reply_raw:
        assistant = _choice_message(reply_raw, index=index)
        response_metadata = reply_raw
    elif reply_field is not None:
        assistant = _assistant_reply(reply_raw, field=reply_field, index=index)
        if isinstance(reply_raw, dict):
            response_metadata = reply_raw

    combined_messages: list[Any] = [
        dict(message) if isinstance(message, dict) else message
        for message in request_messages
    ]
    if assistant is not None:
        _append_or_merge_assistant(combined_messages, assistant, request=request)
    normalized: dict[str, Any] = {
        "messages": combined_messages,
    }

    # Response metadata wins over envelope metadata; request parameters supply
    # the final fallback for the values that belong to the original request.
    for key in (
        "id", "model", "timestamp", "created", "usage",
        "prompt_tokens", "completion_tokens",
    ):
        if key in response_metadata:
            normalized[key] = response_metadata[key]
        elif key in data and key not in {"request", reply_field}:
            normalized[key] = data[key]
        elif key in request:
            normalized[key] = request[key]
    # ``tools`` belongs to the request and changes the rendered prompt, so
    # dropping it here silently rewrote the prompt the row claims to describe.
    for key in ("temperature", "top_p", "seed", "tools"):
        if key in request:
            normalized[key] = request[key]
        elif key in data and key not in {"request", reply_field}:
            normalized[key] = data[key]
        elif key in response_metadata:
            normalized[key] = response_metadata[key]

    if assistant is not None:
        tool_calls = assistant.get("tool_calls")
        if tool_calls is not None:
            normalized["tool_calls"] = tool_calls
    return normalized


def _walk_mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _nested_messages(
    data: Mapping[str, Any],
) -> tuple[list[Any] | None, Mapping[str, Any] | None]:
    for mapping in _walk_mappings(data):
        messages = mapping.get("messages")
        if isinstance(messages, list) and messages:
            return messages, mapping
    return None, None


def _nested_assistant(
    data: Mapping[str, Any],
    *,
    excluded: set[int],
    index: int,
) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
    mappings = list(_walk_mappings(data))
    for mapping in mappings:
        if "choices" in mapping:
            try:
                return _choice_message(mapping, index=index), mapping
            except NormalizeError:
                continue
    for mapping in mappings:
        if id(mapping) in excluded or mapping.get("role") != "assistant":
            continue
        if "content" in mapping or "tool_calls" in mapping:
            assistant = dict(mapping)
            assistant.setdefault("content", None)
            return assistant, mapping
    for mapping in mappings:
        for key in ("assistant", "reply", "response", "completion", "output"):
            if key not in mapping:
                continue
            try:
                return _assistant_reply(mapping[key], field=key, index=index), mapping
            except NormalizeError:
                continue
    return None, None


def _has_recoverable_conversation(data: Mapping[str, Any]) -> bool:
    messages, _ = _nested_messages(data)
    if messages is not None:
        return True
    assistant, _ = _nested_assistant(data, excluded=set(), index=0)
    return assistant is not None


def _first_nested(
    mappings: list[Mapping[str, Any]],
    key: str,
) -> Any:
    for mapping in mappings:
        if key in mapping:
            return mapping[key]
    return None


def _proxy_capture_to_internal(
    data: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    request_messages, request_metadata = _nested_messages(data)
    excluded = {
        id(message)
        for message in request_messages or []
        if isinstance(message, Mapping)
    }
    assistant, response_metadata = _nested_assistant(
        data,
        excluded=excluded,
        index=index,
    )
    if request_messages is None and assistant is None:
        raise NormalizeError(
            f"record[{index}]: cannot recover a conversation: no messages "
            "or assistant content found"
        )

    messages: list[Any] = [
        dict(message) if isinstance(message, Mapping) else message
        for message in (request_messages or [])
    ]
    if assistant is not None:
        _append_or_merge_assistant(
            messages,
            assistant,
            request=request_metadata if request_metadata is not None else {},
        )
    normalized: dict[str, Any] = {"messages": messages}

    mappings = list(_walk_mappings(data))
    response_first = [
        mapping for mapping in (response_metadata, *mappings)
        if mapping is not None
    ]
    request_first = [
        mapping for mapping in (request_metadata, *mappings)
        if mapping is not None
    ]
    for key in (
        "id", "timestamp", "created", "usage",
        "prompt_tokens", "completion_tokens",
    ):
        value = _first_nested(response_first, key)
        if value is not None:
            normalized[key] = value
    for key in ("model", "temperature", "top_p", "seed"):
        value = _first_nested(request_first, key)
        if value is not None:
            normalized[key] = value
    # ``tools`` changes the rendered prompt, so it has to survive the proxy
    # shape too. This shape is recovered by heuristic walk, so only take a
    # value that is actually a tool-schema list; anything else would turn a
    # previously accepted record into a rejection.
    tools_value = _first_nested(request_first, "tools")
    if isinstance(tools_value, list) and all(
        isinstance(tool, Mapping) for tool in tools_value
    ):
        normalized["tools"] = tools_value
    if assistant is not None and "tool_calls" in assistant:
        normalized["tool_calls"] = assistant["tool_calls"]
    return normalized


def _to_internal_shape(
    data: Mapping[str, Any],
    *,
    shape: str,
    index: int,
) -> dict[str, Any]:
    if shape == "openai-response":
        return _openai_response_to_internal(data, index=index)
    if shape == "request-response":
        try:
            return _request_response_to_internal(data, index=index)
        except NormalizeError:
            if _has_recoverable_conversation(data):
                return _proxy_capture_to_internal(data, index=index)
            raise
    if shape == "proxy-capture":
        return _proxy_capture_to_internal(data, index=index)
    return dict(data)


# ── Public API ──────────────────────────────────────────────────────────────

def normalize_record(
    data: Mapping[str, Any],
    *,
    defaults: SamplingConfig,
    default_model: str | None = None,
    index: int = 0,
) -> TraceRecord:
    """Auto-detect, validate, and normalize one supported external record.

    Raises NormalizeError on any violation.

    Note: unknown top-level keys in external records are silently ignored
    because external OpenAI logs carry noise.
    """

    shape = _detect_shape(data, index=index)
    normalized = _to_internal_shape(data, shape=shape, index=index)

    # --- messages (required, non-empty list of {role, content}) ---
    messages = normalized.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise NormalizeError(
            f"record[{index}]: 'messages' must be a non-empty list"
        )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise NormalizeError(
                f"record[{index}]: messages[{i}] must be an object"
            )
        if "role" not in msg or not isinstance(msg["role"], str) or not msg["role"]:
            raise NormalizeError(
                f"record[{index}]: "
                f"messages[{i}] must have a non-empty string 'role'"
            )
        if "content" not in msg:
            raise NormalizeError(
                f"record[{index}]: messages[{i}] must have a 'content' key"
            )
    messages_tuple: tuple[dict[str, Any], ...] = tuple(dict(m) for m in messages)

    # --- model ---
    model = normalized.get("model")
    if isinstance(model, str) and model:
        pass
    elif default_model is not None:
        model = default_model
    else:
        model = "unknown"

    # --- id ---
    rid = normalized.get("id")
    if isinstance(rid, str) and rid:
        pass
    else:
        rid = _generate_id(data)

    # --- timestamp ---
    ts = normalized.get("timestamp")
    created = normalized.get("created")
    if ts is not None:
        ts = _parse_timestamp_to_epoch(ts, index=index)
    elif created is not None:
        ts = _parse_timestamp_to_epoch(created, index=index)
    else:
        ts = time.time()

    # --- sampling parameters ---
    temperature = normalized.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise NormalizeError(
                f"record[{index}]: 'temperature' must be a number"
            )
        if temperature < 0:
            raise NormalizeError(
                f"record[{index}]: 'temperature' must be >= 0"
            )
    else:
        # Deliberately still a number, because TraceRecord.temperature is a
        # required float — but the provenance block below records that nothing
        # observed it, so this value can never be mistaken for a genuine
        # greedy request.
        temperature = defaults.temperature

    top_p = normalized.get("top_p")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise NormalizeError(
                f"record[{index}]: 'top_p' must be a number"
            )
        if not (0 < top_p <= 1):
            raise NormalizeError(
                f"record[{index}]: 'top_p' must satisfy 0 < top_p <= 1"
            )
    else:
        top_p = defaults.top_p

    seed = normalized.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise NormalizeError(
                f"record[{index}]: 'seed' must be an int"
            )
        if seed < 0:
            raise NormalizeError(
                f"record[{index}]: 'seed' must be >= 0"
            )
    else:
        seed = defaults.seed

    messages_tuple = _attach_sampling_provenance(
        messages_tuple,
        observed={
            key: normalized.get(key) is not None
            for key in ("temperature", "top_p", "seed")
        },
    )

    # --- tool_calls ---
    tool_calls_raw = normalized.get("tool_calls")
    if tool_calls_raw is not None:
        if not isinstance(tool_calls_raw, list):
            raise NormalizeError(
                f"record[{index}]: 'tool_calls' must be a list"
            )
        for i, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                raise NormalizeError(
                    f"record[{index}]: tool_calls[{i}] must be an object"
                )
        tool_calls: tuple[dict[str, Any], ...] = tuple(
            dict(tc) for tc in tool_calls_raw
        )
    else:
        tool_calls = ()

    # --- request tool schemas ---
    tools_raw = normalized.get("tools")
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise NormalizeError(f"record[{index}]: 'tools' must be a list")
        for i, tool in enumerate(tools_raw):
            if not isinstance(tool, dict):
                raise NormalizeError(
                    f"record[{index}]: tools[{i}] must be an object"
                )
        tools: tuple[dict[str, Any], ...] = tuple(dict(tool) for tool in tools_raw)
    else:
        tools = ()

    # --- token counts ---
    prompt_tokens, completion_tokens, token_count_source = _extract_tokens(
        normalized,
        index,
        messages_tuple,
    )

    return TraceRecord(
        id=rid,
        timestamp=float(ts),
        model=str(model),
        messages=messages_tuple,
        tool_calls=tool_calls,
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools=tools,
        token_count_source=token_count_source,
    )


def _attach_sampling_provenance(
    messages: tuple[dict[str, Any], ...],
    *,
    observed: Mapping[str, bool],
) -> tuple[dict[str, Any], ...]:
    """Label where each stored sampling value came from.

    A label written at capture time is authoritative and is never overwritten
    here: the gateway saw the request, this function only sees the file. Keys
    the gateway did not label get ``record_explicit`` when the record carried a
    value and ``unknown`` when the value was substituted from configuration.
    """
    if not messages:
        return messages
    target = len(messages) - 1
    for position in range(len(messages) - 1, -1, -1):
        if messages[position].get("role") == "assistant":
            target = position
            break

    existing = messages[target].get(SAMPLING_PROVENANCE_KEY)
    block: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    raw_sources = block.get("sources")
    sources: dict[str, Any] = dict(raw_sources) if isinstance(raw_sources, Mapping) else {}
    for key, was_observed in observed.items():
        if key in sources:
            continue
        sources[key] = (
            SAMPLING_SOURCE_RECORD if was_observed else SAMPLING_SOURCE_UNKNOWN
        )
    block["sources"] = sources

    updated = list(messages)
    updated[target] = {**messages[target], SAMPLING_PROVENANCE_KEY: block}
    return tuple(updated)


def _content_only(message: Mapping[str, Any]) -> dict[str, Any]:
    """Drop capture bookkeeping so it cannot be counted as conversation text."""
    if _NON_CONTENT_MESSAGE_KEYS.isdisjoint(message):
        return dict(message)
    return {
        key: value
        for key, value in message.items()
        if key not in _NON_CONTENT_MESSAGE_KEYS
    }


def _extract_tokens(
    data: Mapping[str, Any],
    index: int,
    messages: tuple[dict[str, Any], ...],
) -> tuple[int, int, str]:
    """Use measured counts when complete; otherwise estimate without rejecting."""
    del index  # Token problems deliberately fall back to estimates.
    estimated_prompt, estimated_completion = estimate_message_tokens(
        [_content_only(message) for message in messages]
    )
    usage = data.get("usage")
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
    else:
        pt = data.get("prompt_tokens")
        ct = data.get("completion_tokens")

    measured_pt = _usable_token_count(pt)
    measured_ct = _usable_token_count(ct)
    if measured_pt is not None and measured_ct is not None:
        source = (
            "estimated"
            if data.get("token_count_source") == "estimated"
            else "measured"
        )
        return measured_pt, measured_ct, source
    return (
        measured_pt if measured_pt is not None else estimated_prompt,
        measured_ct if measured_ct is not None else estimated_completion,
        "estimated",
    )


def _usable_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def normalize_file(
    path: Path,
    *,
    defaults: SamplingConfig | None = None,
    default_model: str | None = None,
) -> NormalizeResult:
    """Normalize an external OpenAI-format JSONL file.

    Reads the file line-by-line directly (not via ``storage.read_jsonl``)
    so that malformed JSON becomes a ``Rejection`` rather than an exception.
    A missing file raises ``NormalizeError``.

    Unknown top-level keys in external records are silently ignored because
    external OpenAI logs carry noise.
    """
    if defaults is None:
        defaults = SamplingConfig()

    if not path.exists():
        raise NormalizeError(f"file not found: {path}")

    accepted: list[TraceRecord] = []
    detected_shapes: list[str] = []
    rejected: list[Rejection] = []

    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue  # skip blank lines
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason=f"malformed JSON: {e}",
                        raw=stripped[:500],
                    )
                )
                continue
            if not isinstance(obj, dict):
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason="line is not a JSON object",
                        raw=stripped[:500],
                    )
                )
                continue
            try:
                shape = _detect_shape(obj, index=line_no)
                record = normalize_record(
                    obj,
                    defaults=defaults,
                    default_model=default_model,
                    index=line_no,
                )
                accepted.append(record)
                detected_shapes.append(shape)
            except NormalizeError as e:
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason=str(e),
                        raw=stripped[:500],
                    )
                )

    return NormalizeResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        detected_shapes=tuple(detected_shapes),
    )
