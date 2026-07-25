"""ChatML/Qwen rendering and assistant-span detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from speedlm.training.templates.base import (
    AssistantSpan,
    Conversation,
    ToolSchemas,
)

_START = "<|im_start|>"
_END = "<|im_end|>"


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("ChatML message content must be a string or null")
    return value


class ChatMLTemplate:
    """Flat ChatML renderer used by Qwen-family tokenizers."""

    name = "chatml"

    def render(
        self,
        conversation: Conversation,
        *,
        tools: ToolSchemas = (),
    ) -> str:
        del tools  # Tool schemas are normally injected by Qwen's system prompt.
        parts: list[str] = []
        for index, message in enumerate(conversation):
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"conversation[{index}].role must be a non-empty string")
            content = _content(message)
            if role == "assistant":
                reasoning = message.get(
                    "reasoning_content", message.get("thinking", "")
                )
                if reasoning is not None and not isinstance(reasoning, str):
                    raise ValueError(
                        f"conversation[{index}].reasoning_content must be a string or null"
                    )
                if reasoning:
                    content = f"{reasoning}\n{content}" if content else reasoning
            parts.append(f"{_START}{role}\n{content}{_END}\n")
        return "".join(parts)

    def assistant_spans(self, rendered: str) -> tuple[AssistantSpan, ...]:
        marker = f"{_START}assistant\n"
        spans: list[AssistantSpan] = []
        cursor = 0
        turn = 0
        while True:
            header = rendered.find(marker, cursor)
            if header < 0:
                break
            start = header + len(marker)
            end = rendered.find(_END, start)
            if end < 0:
                raise ValueError("unterminated ChatML assistant message")
            if end > start:
                spans.append(AssistantSpan(start, end, turn, "assistant"))
            turn += 1
            cursor = end + len(_END)
        return tuple(spans)
