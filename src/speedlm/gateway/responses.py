from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from speedlm.gateway.sse import AssembledResponse

_TERMINAL_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
}


def parse_responses_response(body: bytes | bytearray) -> AssembledResponse | None:
    """Parse one non-streaming OpenAI Responses API result."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return _parse_response_payload(payload)


def parse_responses_sse(body: bytes | bytearray) -> AssembledResponse | None:
    """Recover the terminal response object from a Responses API SSE stream."""
    terminal: Mapping[str, Any] | None = None
    for raw_event in _sse_data_events(body):
        if raw_event.strip() == b"[DONE]":
            continue
        try:
            payload = json.loads(raw_event)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        event_type = payload.get("type")
        response = payload.get("response")
        if event_type in _TERMINAL_EVENTS and isinstance(response, Mapping):
            terminal = response
    return _parse_response_payload(terminal)


def response_status_and_id(
    body: bytes | bytearray,
    content_type: str,
) -> tuple[str | None, str | None]:
    """Return response id/status even when a background response has no output."""
    payload: Any = None
    if content_type.lower().startswith("text/event-stream"):
        for raw_event in _sse_data_events(body):
            if raw_event.strip() == b"[DONE]":
                continue
            try:
                candidate = json.loads(raw_event)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None, None
            if isinstance(candidate, Mapping):
                nested = candidate.get("response")
                payload = nested if isinstance(nested, Mapping) else candidate
    else:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, None
    if not isinstance(payload, Mapping):
        return None, None
    response_id = payload.get("id")
    status = payload.get("status")
    return (
        response_id if isinstance(response_id, str) and response_id else None,
        status if isinstance(status, str) and status else None,
    )


def responses_request_messages(
    request_data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert Responses API input items to the trace conversation shape."""
    messages: list[dict[str, Any]] = []
    instructions = request_data.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append(
            {
                "role": "system",
                "content": instructions,
                "provenance_tag": "client_supplied",
            }
        )

    request_input = request_data.get("input")
    if isinstance(request_input, str):
        messages.append(
            {
                "role": "user",
                "content": request_input,
                "provenance_tag": "client_supplied",
            }
        )
        return messages
    if not isinstance(request_input, list):
        return messages

    for item in request_input:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type in (None, "message") and isinstance(role, str) and role:
            messages.append(
                {
                    **dict(item),
                    "role": role,
                    "content": item.get("content"),
                    "provenance_tag": "client_supplied",
                }
            )
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": item.get("output"),
                    "tool_call_id": item.get("call_id"),
                    "provenance_tag": "client_supplied",
                }
            )
            continue
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                    "provenance_tag": "client_supplied",
                }
            )
    return messages


def _parse_response_payload(payload: Any) -> AssembledResponse | None:
    if not isinstance(payload, Mapping):
        return None
    output = payload.get("output")
    if not isinstance(output, list) or not output:
        return None

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    saw_output = False
    for item in output:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, Mapping):
                    continue
                part_type = part.get("type")
                text = part.get("text")
                if part_type in {"output_text", "text"} and isinstance(text, str):
                    content_parts.append(text)
                    saw_output = True
                elif (
                    part_type in {"reasoning_text", "summary_text"}
                    and isinstance(text, str)
                ):
                    reasoning_parts.append(text)
                    saw_output = True
        elif item_type in {"function_call", "custom_tool_call"}:
            name = item.get("name")
            arguments = item.get("arguments")
            if isinstance(name, str) and isinstance(arguments, str):
                call: dict[str, Any] = {
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                call_id = item.get("call_id") or item.get("id")
                if isinstance(call_id, str) and call_id:
                    call["id"] = call_id
                tool_calls.append(call)
                saw_output = True
        elif item_type == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    if not isinstance(part, Mapping):
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        reasoning_parts.append(text)
                        saw_output = True
    if not saw_output:
        return None

    usage = payload.get("usage")
    prompt_tokens = _non_negative_int(usage, "input_tokens")
    completion_tokens = _non_negative_int(usage, "output_tokens")
    response_id = payload.get("id")
    model = payload.get("model")
    created = payload.get("created_at")
    status = payload.get("status")
    incomplete_details = payload.get("incomplete_details")
    stop_reason: str | None = None
    if isinstance(incomplete_details, Mapping):
        reason = incomplete_details.get("reason")
        if isinstance(reason, str):
            stop_reason = reason
    return AssembledResponse(
        id=response_id if isinstance(response_id, str) and response_id else None,
        model=model if isinstance(model, str) and model else None,
        created=(
            float(created)
            if not isinstance(created, bool)
            and isinstance(created, (int, float))
            and created >= 0
            else None
        ),
        content="".join(content_parts) if content_parts else None,
        tool_calls=tuple(tool_calls),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        reasoning_field="reasoning_content" if reasoning_parts else None,
        finish_reason=status if isinstance(status, str) else None,
        stop_reason=stop_reason,
    )


def _sse_data_events(body: bytes | bytearray) -> list[bytes]:
    events: list[bytes] = []
    data: list[bytes] = []
    for line in bytes(body).splitlines():
        if not line:
            if data:
                events.append(b"\n".join(data))
                data.clear()
            continue
        if line.startswith(b":"):
            continue
        field, separator, value = line.partition(b":")
        if field != b"data":
            continue
        if separator and value.startswith(b" "):
            value = value[1:]
        data.append(value)
    if data:
        events.append(b"\n".join(data))
    return events


def _non_negative_int(value: Any, key: str) -> int | None:
    if not isinstance(value, Mapping):
        return None
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        return None
    return result
