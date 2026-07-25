from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AssembledResponse:
    """The capture-relevant fields reconstructed from an OpenAI response."""

    id: str | None
    model: str | None
    created: float | None
    content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(slots=True)
class _ToolCall:
    id: str = ""
    type: str = ""
    name: str = ""
    arguments: str = ""

    def update(self, delta: dict[str, Any]) -> None:
        call_id = delta.get("id")
        if isinstance(call_id, str):
            self.id += call_id
        call_type = delta.get("type")
        if isinstance(call_type, str):
            self.type += call_type
        function = delta.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str):
            self.name += name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments += arguments

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type or "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }
        if self.id:
            result["id"] = self.id
        return result


class SSEAssembler:
    """Incrementally parse SSE while reconstructing the first OpenAI choice.

    ``feed`` never raises for malformed upstream data. The caller can therefore
    tee bytes through this parser without putting capture on the client path.
    """

    def __init__(self, endpoint: str) -> None:
        if endpoint not in {"/v1/chat/completions", "/v1/completions"}:
            raise ValueError(f"unsupported SSE endpoint: {endpoint}")
        self._endpoint = endpoint
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []
        self._content: list[str] = []
        self._tool_calls: dict[int, _ToolCall] = {}
        self._id: str | None = None
        self._model: str | None = None
        self._created: float | None = None
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._done = False
        self._valid = True
        self._saw_choice = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def valid(self) -> bool:
        return self._valid and self._saw_choice

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._line_buffer.extend(chunk)
        while True:
            newline = self._line_buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._line_buffer[:newline])
            del self._line_buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            self._consume_line(line)

    def finish(self) -> AssembledResponse:
        """Consume a final unterminated event and return the assembled fields."""
        if self._line_buffer:
            line = bytes(self._line_buffer)
            self._line_buffer.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            self._consume_line(line)
        if self._event_data:
            self._consume_event()
        return AssembledResponse(
            id=self._id,
            model=self._model,
            created=self._created,
            content="".join(self._content) if self._content else None,
            tool_calls=tuple(
                call.to_dict() for _, call in sorted(self._tool_calls.items())
            ),
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    def _consume_line(self, line: bytes) -> None:
        if not line:
            self._consume_event()
            return
        if line.startswith(b":"):
            return
        field, separator, value = line.partition(b":")
        if field != b"data":
            return
        if separator and value.startswith(b" "):
            value = value[1:]
        self._event_data.append(value)

    def _consume_event(self) -> None:
        if not self._event_data:
            return
        raw = b"\n".join(self._event_data)
        self._event_data.clear()
        if raw.strip() == b"[DONE]":
            self._done = True
            return
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._valid = False
            return
        if not isinstance(payload, dict):
            self._valid = False
            return
        self._consume_payload(payload)

    def _consume_payload(self, payload: dict[str, Any]) -> None:
        response_id = payload.get("id")
        if self._id is None and isinstance(response_id, str) and response_id:
            self._id = response_id
        model = payload.get("model")
        if self._model is None and isinstance(model, str) and model:
            self._model = model
        created = payload.get("created")
        if (
            self._created is None
            and not isinstance(created, bool)
            and isinstance(created, (int, float))
            and created >= 0
        ):
            self._created = float(created)

        self._consume_usage(payload.get("usage"))
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if self._endpoint == "/v1/completions":
                text = choice.get("text")
                if isinstance(text, str):
                    self._saw_choice = True
                    self._content.append(text)
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            self._saw_choice = True
            content = delta.get("content")
            if isinstance(content, str):
                self._content.append(content)
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for fallback_index, raw_call in enumerate(tool_calls):
                if not isinstance(raw_call, dict):
                    continue
                index = raw_call.get("index", fallback_index)
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    continue
                self._tool_calls.setdefault(index, _ToolCall()).update(raw_call)

    def _consume_usage(self, usage: Any) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            self._valid = False
            return
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is not None and not (
            not isinstance(prompt_tokens, bool)
            and isinstance(prompt_tokens, int)
            and prompt_tokens >= 0
        ):
            self._valid = False
        elif prompt_tokens is not None:
            self._prompt_tokens = prompt_tokens
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None and not (
            not isinstance(completion_tokens, bool)
            and isinstance(completion_tokens, int)
            and completion_tokens >= 0
        ):
            self._valid = False
        elif completion_tokens is not None:
            self._completion_tokens = completion_tokens


def parse_json_response(body: bytes, endpoint: str) -> AssembledResponse | None:
    """Parse a non-streaming OpenAI response, returning ``None`` fail-closed."""
    if endpoint not in {"/v1/chat/completions", "/v1/completions"}:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    choice: dict[str, Any] | None = None
    for candidate in choices:
        if isinstance(candidate, dict) and candidate.get("index", 0) == 0:
            choice = candidate
            break
    if choice is None:
        return None

    content: str | None
    tool_calls: tuple[dict[str, Any], ...] = ()
    if endpoint == "/v1/completions":
        text = choice.get("text")
        if not isinstance(text, str):
            return None
        content = text
    else:
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        raw_content = message.get("content")
        if raw_content is not None and not isinstance(raw_content, str):
            return None
        content = raw_content
        raw_tool_calls = message.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list) or not all(
            isinstance(call, dict) for call in raw_tool_calls
        ):
            return None
        tool_calls = tuple(dict(call) for call in raw_tool_calls)

    usage = payload.get("usage")
    prompt_tokens = _non_negative_int(usage, "prompt_tokens")
    completion_tokens = _non_negative_int(usage, "completion_tokens")
    response_id = payload.get("id")
    model = payload.get("model")
    created = payload.get("created")
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
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _non_negative_int(value: Any, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        return None
    return result
