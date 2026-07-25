"""Harmony/gpt-oss rendering with channel-aware assistant spans."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from speedlm.training.templates.base import (
    AssistantSpan,
    Conversation,
    ToolSchemas,
)

_START = "<|start|>"
_MESSAGE = "<|message|>"
_END = "<|end|>"
_CALL = "<|call|>"
_CHANNEL = "<|channel|>"
_TOOL_SUFFIXES = (
    "<|channel|>analysis",
    "<|channel|>commentary",
    "<|channel|>json",
)


def _string(value: object, location: str, *, nullable: bool = False) -> str:
    if value is None and nullable:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string{' or null' if nullable else ''}")
    return value


def _tool_name(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    for suffix in _TOOL_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _arguments(value: object, location: str) -> dict[str, Any]:
    """Decode OpenAI wire strings before Harmony applies arguments|tojson."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location} must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{location} must encode a JSON object")
    return value


def _block(
    role: str,
    content: str,
    *,
    channel: str | None = None,
    recipient: str | None = None,
    terminal: str = _END,
) -> str:
    header = f"{_START}{role}"
    if channel is not None:
        header += f"{_CHANNEL}{channel}"
    if recipient is not None:
        header += f" to={recipient}"
    if terminal == _CALL:
        header += "<|constrain|>json"
    return f"{header}{_MESSAGE}{content}{terminal}"


class HarmonyTemplate:
    """Channel-aware renderer for GPT-OSS Harmony conversations."""

    name = "harmony"

    def render(
        self,
        conversation: Conversation,
        *,
        tools: ToolSchemas = (),
    ) -> str:
        parts: list[str] = []
        if tools:
            schemas = json.dumps(
                list(tools), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            parts.append(_block("system", f"Tools are available:\n{schemas}"))

        for index, message in enumerate(conversation):
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"conversation[{index}].role must be a non-empty string")
            content = _string(
                message.get("content", ""),
                f"conversation[{index}].content",
                nullable=role == "assistant",
            )
            if role == "assistant":
                self._render_assistant(parts, message, content, index)
            elif role == "tool":
                name = _tool_name(
                    message.get("name"), f"conversation[{index}].name"
                )
                channel = _string(
                    message.get("channel", "commentary"),
                    f"conversation[{index}].channel",
                )
                parts.append(
                    _block(
                        f"functions.{name}",
                        content,
                        channel=channel,
                        recipient="assistant",
                    )
                )
            else:
                parts.append(_block(role, content))
        return "".join(parts)

    def _render_assistant(
        self,
        parts: list[str],
        message: Mapping[str, Any],
        content: str,
        index: int,
    ) -> None:
        reasoning = message.get("reasoning_content", message.get("thinking", ""))
        reasoning_text = _string(
            reasoning,
            f"conversation[{index}].reasoning_content",
            nullable=True,
        )
        if reasoning_text:
            reasoning_channel = _string(
                message.get("reasoning_channel", "analysis"),
                f"conversation[{index}].reasoning_channel",
            )
            parts.append(
                _block("assistant", reasoning_text, channel=reasoning_channel)
            )

        calls = message.get("tool_calls", ())
        if calls is None:
            calls = ()
        if not isinstance(calls, (list, tuple)):
            raise ValueError(f"conversation[{index}].tool_calls must be a sequence")
        for call_index, call in enumerate(calls):
            location = f"conversation[{index}].tool_calls[{call_index}]"
            if not isinstance(call, Mapping):
                raise ValueError(f"{location} must be an object")
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError(f"{location}.function must be an object")
            name = _tool_name(function.get("name"), f"{location}.function.name")
            arguments = _arguments(
                function.get("arguments"), f"{location}.function.arguments"
            )
            payload = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            channel = _string(
                message.get("tool_channel", "commentary"),
                f"conversation[{index}].tool_channel",
            )
            parts.append(
                _block(
                    "assistant",
                    payload,
                    channel=channel,
                    recipient=f"functions.{name}",
                    terminal=_CALL,
                )
            )

        if content or not calls:
            channel = _string(
                message.get("channel", "final"),
                f"conversation[{index}].channel",
            )
            parts.append(_block("assistant", content, channel=channel))

    def assistant_spans(self, rendered: str) -> tuple[AssistantSpan, ...]:
        spans: list[AssistantSpan] = []
        cursor = 0
        turn = -1
        previous_was_assistant = False
        while True:
            block = rendered.find(_START, cursor)
            if block < 0:
                break
            header_end = rendered.find(_MESSAGE, block + len(_START))
            if header_end < 0:
                raise ValueError("unterminated Harmony header")
            header = rendered[block + len(_START) : header_end]
            payload_start = header_end + len(_MESSAGE)
            end_marker, payload_end = self._terminal(rendered, payload_start)
            role = header.split(_CHANNEL, 1)[0].split(" ", 1)[0]
            is_assistant = role == "assistant"
            if is_assistant:
                if not previous_was_assistant:
                    turn += 1
                channel = (
                    header.split(_CHANNEL, 1)[1].split(" ", 1)[0]
                    if _CHANNEL in header
                    else "final"
                )
                if payload_end > payload_start:
                    spans.append(
                        AssistantSpan(
                            payload_start,
                            payload_end,
                            turn,
                            channel,
                        )
                    )
            previous_was_assistant = is_assistant
            cursor = payload_end + len(end_marker)
        return tuple(spans)

    @staticmethod
    def _terminal(rendered: str, start: int) -> tuple[str, int]:
        end = rendered.find(_END, start)
        call = rendered.find(_CALL, start)
        positions = [
            (position, marker)
            for position, marker in ((end, _END), (call, _CALL))
            if position >= 0
        ]
        if not positions:
            raise ValueError("unterminated Harmony message")
        position, marker = min(positions)
        return marker, position
