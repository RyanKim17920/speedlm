from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_SUPPORTED_ENDPOINTS = {"/v1/chat/completions", "/v1/completions"}
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


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
    reasoning_content: str | None = None
    reasoning_field: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    choice_index: int = 0
    choice_count: int = 1


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


@dataclass(slots=True)
class _Choice:
    content: list[str] = field(default_factory=list)
    reasoning: dict[str, list[str]] = field(default_factory=dict)
    tool_calls: dict[int, _ToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    saw_choice: bool = False
    valid: bool = True


class SSEAssembler:
    """Incrementally parse SSE while reconstructing OpenAI choices.

    ``feed`` never raises for malformed upstream data. The caller can therefore
    tee bytes through this parser without putting capture on the client path.

    ``finish`` retains the original choice-zero behavior. New callers should use
    ``finish_all`` to recover every returned choice, including interleaved
    ``n > 1`` streams.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []
        self._choices: dict[int, _Choice] = {}
        self._id: str | None = None
        self._model: str | None = None
        self._created: float | None = None
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._done = False
        self._valid = endpoint in _SUPPORTED_ENDPOINTS

    @property
    def done(self) -> bool:
        return self._done

    @property
    def valid(self) -> bool:
        choice = self._choices.get(0)
        return (
            self._valid
            and choice is not None
            and choice.saw_choice
            and choice.valid
        )

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
        """Return choice zero, preserving the original single-choice API."""
        self._finish_events()
        observed_count = sum(
            choice.saw_choice and choice.valid for choice in self._choices.values()
        )
        return self._assemble_choice(
            0,
            self._choices.get(0, _Choice()),
            choice_count=observed_count,
        )

    def finish_all(self) -> tuple[AssembledResponse, ...]:
        """Consume the final event and return every observed choice by index."""
        self._finish_events()
        if not self._valid:
            return ()
        observed = [
            (index, choice)
            for index, choice in sorted(self._choices.items())
            if choice.saw_choice and choice.valid
        ]
        choice_count = len(observed)
        return tuple(
            self._assemble_choice(index, choice, choice_count=choice_count)
            for index, choice in observed
        )

    def _finish_events(self) -> None:
        if self._line_buffer:
            line = bytes(self._line_buffer)
            self._line_buffer.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            self._consume_line(line)
        if self._event_data:
            self._consume_event()

    def _assemble_choice(
        self,
        index: int,
        choice: _Choice,
        *,
        choice_count: int,
    ) -> AssembledResponse:
        reasoning_field = next(iter(choice.reasoning), None)
        reasoning_parts = (
            choice.reasoning[reasoning_field] if reasoning_field is not None else []
        )
        return AssembledResponse(
            id=self._id,
            model=self._model,
            created=self._created,
            content="".join(choice.content) if choice.content else None,
            tool_calls=tuple(
                call.to_dict() for _, call in sorted(choice.tool_calls.items())
            ),
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            reasoning_content=(
                "".join(reasoning_parts) if reasoning_parts else None
            ),
            reasoning_field=reasoning_field,
            finish_reason=choice.finish_reason,
            stop_reason=choice.stop_reason,
            choice_index=index,
            choice_count=choice_count,
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
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                continue
            assembled_choice = self._choices.setdefault(index, _Choice())
            self._consume_finish_details(choice, assembled_choice)
            if self._endpoint == "/v1/completions":
                text = choice.get("text")
                if isinstance(text, str):
                    assembled_choice.saw_choice = True
                    assembled_choice.content.append(text)
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            assembled_choice.saw_choice = True
            content = delta.get("content")
            if isinstance(content, str):
                assembled_choice.content.append(content)
            self._consume_reasoning(delta, assembled_choice)
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for fallback_index, raw_call in enumerate(tool_calls):
                if not isinstance(raw_call, dict):
                    continue
                index = raw_call.get("index", fallback_index)
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    continue
                assembled_choice.tool_calls.setdefault(index, _ToolCall()).update(
                    raw_call
                )

    def _consume_reasoning(
        self,
        delta: dict[str, Any],
        choice: _Choice,
    ) -> None:
        for field_name in _REASONING_FIELDS:
            value = delta.get(field_name)
            if isinstance(value, str):
                choice.reasoning.setdefault(field_name, []).append(value)

    def _consume_finish_details(
        self,
        payload: dict[str, Any],
        choice: _Choice,
    ) -> None:
        finish_reason = payload.get("finish_reason")
        if isinstance(finish_reason, str):
            choice.finish_reason = finish_reason
        elif finish_reason is not None:
            choice.valid = False

        stop_reason = payload.get("stop_reason")
        if (
            isinstance(stop_reason, str)
            or not isinstance(stop_reason, bool)
            and isinstance(stop_reason, int)
        ):
            choice.stop_reason = stop_reason
        elif stop_reason is not None:
            choice.valid = False

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


def parse_json_responses(
    body: bytes,
    endpoint: str,
) -> tuple[AssembledResponse, ...]:
    """Parse every choice in a non-streaming OpenAI response."""
    if endpoint not in _SUPPORTED_ENDPOINTS:
        return ()
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ()
    if not isinstance(payload, dict):
        return ()

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ()

    usage = payload.get("usage")
    prompt_tokens = _non_negative_int(usage, "prompt_tokens")
    completion_tokens = _non_negative_int(usage, "completion_tokens")
    response_id = payload.get("id")
    model = payload.get("model")
    created = payload.get("created")
    assembled: list[AssembledResponse] = []
    choice_count = sum(isinstance(candidate, dict) for candidate in choices)
    for candidate in choices:
        if not isinstance(candidate, dict):
            continue
        choice_index = candidate.get("index", 0)
        if (
            isinstance(choice_index, bool)
            or not isinstance(choice_index, int)
            or choice_index < 0
        ):
            continue

        content: str | None
        reasoning_content: str | None = None
        reasoning_field: str | None = None
        tool_calls: tuple[dict[str, Any], ...] = ()
        if endpoint == "/v1/completions":
            text = candidate.get("text")
            if not isinstance(text, str):
                continue
            content = text
        else:
            message = candidate.get("message")
            if not isinstance(message, dict):
                continue
            raw_content = message.get("content")
            if raw_content is not None and not isinstance(raw_content, str):
                continue
            content = raw_content
            malformed = False
            for field_name in _REASONING_FIELDS:
                raw_reasoning = message.get(field_name)
                if raw_reasoning is None:
                    continue
                if not isinstance(raw_reasoning, str):
                    malformed = True
                    break
                reasoning_content = raw_reasoning
                reasoning_field = field_name
                break
            if malformed:
                continue
            raw_tool_calls = message.get("tool_calls", [])
            if not isinstance(raw_tool_calls, list) or not all(
                isinstance(call, dict) for call in raw_tool_calls
            ):
                continue
            tool_calls = tuple(dict(call) for call in raw_tool_calls)

        finish_reason = candidate.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            continue
        stop_reason = candidate.get("stop_reason")
        if stop_reason is not None and not (
            isinstance(stop_reason, str)
            or not isinstance(stop_reason, bool)
            and isinstance(stop_reason, int)
        ):
            continue
        assembled.append(
            AssembledResponse(
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
                reasoning_content=reasoning_content,
                reasoning_field=reasoning_field,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                choice_index=choice_index,
                choice_count=choice_count,
            )
        )
    return tuple(sorted(assembled, key=lambda response: response.choice_index))


def parse_json_response(body: bytes, endpoint: str) -> AssembledResponse | None:
    """Parse choice zero, preserving the original single-choice API."""
    for response in parse_json_responses(body, endpoint):
        if response.choice_index == 0:
            return response
    return None


def _non_negative_int(value: Any, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        return None
    return result
