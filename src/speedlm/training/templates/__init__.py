"""Built-in chat-template backends."""

from speedlm.training.templates.base import AssistantSpan, ChatTemplate
from speedlm.training.templates.chatml import ChatMLTemplate
from speedlm.training.templates.harmony import HarmonyTemplate

__all__ = [
    "AssistantSpan",
    "ChatMLTemplate",
    "ChatTemplate",
    "HarmonyTemplate",
]
