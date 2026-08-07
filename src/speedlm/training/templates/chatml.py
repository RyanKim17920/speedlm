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


class ChatMLStructureError(ValueError):
    """Rendered text does not use the ChatML block structure this module assumes.

    Kept a :class:`ValueError` so existing callers that catch the renderer's
    other structural complaints keep catching this one.
    """


class ChatMLTemplate:
    """Flat ChatML renderer used by Qwen-family tokenizers.

    The assistant-span scan below hardcodes *this* family's control markers.
    That is a deliberate, narrow contract, not a general chat-template reader:
    see :meth:`assistant_spans`, which validates the assumption rather than
    letting a foreign rendering pass through as "no assistant content".
    """

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
        """Return assistant payload spans, refusing text that is not ChatML.

        This scan is *literal*: it keys off one family's ChatML control markers
        and nothing else.  That assumption holds for text this class rendered,
        and for a tokenizer whose own chat template emits the same blocks (the
        Qwen family's does).  It does not hold for an arbitrary Hugging Face
        chat template, and the previous implementation failed silently there:
        finding no marker, it returned an empty tuple, which every caller reads
        as the legitimate answer "this conversation supervises nothing" rather
        than as "these spans are wrong".  An all-zero loss mask derived that way
        is indistinguishable from a conversation with no assistant content.

        So the structure is validated rather than assumed.  Every byte of
        *rendered* must be accounted for by a block of the form
        marker-plus-role, newline, content, end-marker, with only whitespace
        between blocks; anything else raises :class:`ChatMLStructureError`.
        """
        spans: list[AssistantSpan] = []
        cursor = 0
        turn = 0
        blocks = 0
        while True:
            header = rendered.find(_START, cursor)
            if header < 0:
                break
            gap = rendered[cursor:header]
            if gap.strip():
                raise ChatMLStructureError(
                    f"text between ChatML blocks must be whitespace, got {gap!r}"
                )
            role_end = rendered.find("\n", header + len(_START))
            if role_end < 0:
                raise ChatMLStructureError(
                    "ChatML block header is not terminated by a newline"
                )
            role = rendered[header + len(_START) : role_end]
            if not role or _START in role or _END in role:
                raise ChatMLStructureError(
                    f"ChatML block header carries no usable role, got {role!r}"
                )
            start = role_end + 1
            end = rendered.find(_END, start)
            if end < 0:
                raise ValueError(f"unterminated ChatML {role} message")
            blocks += 1
            if role == "assistant":
                if end > start:
                    spans.append(AssistantSpan(start, end, turn, "assistant"))
                turn += 1
            cursor = end + len(_END)
        if not blocks:
            raise ChatMLStructureError(
                "rendered text contains no ChatML block, so ChatMLTemplate "
                "cannot locate assistant spans in it. This class assumes the "
                "ChatML control markers the Qwen family uses; a model whose own "
                "chat template renders a different structure needs spans "
                "derived from that template, not from this one."
            )
        trailing = rendered[cursor:]
        if trailing.strip():
            raise ChatMLStructureError(
                f"trailing text after the final ChatML block, got {trailing!r}"
            )
        return tuple(spans)
