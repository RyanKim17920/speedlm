from __future__ import annotations

import datetime
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.config import SamplingConfig
from speedlm.traces.store import TraceRecord, estimate_message_tokens

# ── Exceptions ──────────────────────────────────────────────────────────────

class NormalizeError(ValueError):
    """Raised when external input cannot be normalized (fail closed)."""


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
    normalized["messages"] = [assistant]
    tool_calls = assistant.get("tool_calls")
    if tool_calls is not None:
        normalized["tool_calls"] = tool_calls
    return normalized


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

    combined_messages = [
        dict(message) if isinstance(message, dict) else message
        for message in request_messages
    ]
    if assistant is not None:
        combined_messages.append(assistant)
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
    for key in ("temperature", "top_p", "seed"):
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

    messages = [
        dict(message) if isinstance(message, Mapping) else message
        for message in (request_messages or [])
    ]
    if assistant is not None:
        messages.append(assistant)
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


def _extract_tokens(
    data: Mapping[str, Any],
    index: int,
    messages: tuple[dict[str, Any], ...],
) -> tuple[int, int, str]:
    """Use measured counts when complete; otherwise estimate without rejecting."""
    del index  # Token problems deliberately fall back to estimates.
    estimated_prompt, estimated_completion = estimate_message_tokens(list(messages))
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
