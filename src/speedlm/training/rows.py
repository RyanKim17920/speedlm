"""Template-agnostic conversion from captured traces to training rows."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from speedlm.traces.store import TraceRecord
from speedlm.training.masking import MaskPolicy
from speedlm.training.templates.base import (
    AssistantSpan,
    ChatTemplate,
)

_HARMONY_TOOL_SUFFIXES = (
    "<|channel|>analysis",
    "<|channel|>commentary",
    "<|channel|>json",
)


class Tokenizer(Protocol):
    """Minimal fast-tokenizer surface needed for exact mask projection."""

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """Validated captured conversation before template rendering."""

    id: str
    conversation: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    model: str | None = None
    model_revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedTrainingRow:
    """Tokenized row with an explicit loss mask."""

    id: str
    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    seq_len: int
    rendered: str
    assistant_spans: tuple[AssistantSpan, ...]
    mask_policy: MaskPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "input_ids": list(self.input_ids),
            "loss_mask": list(self.loss_mask),
            "seq_len": self.seq_len,
            "rendered": self.rendered,
            "mask_policy": self.mask_policy.value,
        }


def _json_copy(value: object, location: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must be JSON serializable") from error


def _content(value: object, location: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{location} must be a string, null, or a non-empty text content-part list"
        )
    text: list[str] = []
    for index, part in enumerate(value):
        if (
            not isinstance(part, Mapping)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
        ):
            raise ValueError(f"{location}[{index}] must be an OpenAI text content part")
        text.append(part["text"])
    return "".join(text)


def _tools(value: object, location: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location} must be a sequence")
    result = _json_copy(value, location)
    names: set[str] = set()
    for index, tool in enumerate(result):
        tool_location = f"{location}[{index}]"
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"{tool_location} must be an OpenAI function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"{tool_location}.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{tool_location}.function.name must be a non-empty string")
        if name in names:
            raise ValueError(f"{location} has duplicate function name {name!r}")
        names.add(name)
        if not isinstance(function.get("description"), str):
            raise ValueError(f"{tool_location}.function.description must be a string")
        if not isinstance(function.get("parameters"), dict):
            raise ValueError(f"{tool_location}.function.parameters must be an object")
    return result


def _messages(
    value: object,
    *,
    row_id: str,
    tool_names: set[str],
    trust_untagged_assistant_messages: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"trace {row_id!r} messages must be a non-empty sequence")
    messages = _json_copy(value, f"trace {row_id!r} messages")
    known_calls: dict[str, str] = {}
    assistant_count = 0
    for index, message in enumerate(messages):
        location = f"trace {row_id!r} messages[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{location} must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{location}.role must be a non-empty string")
        message["content"] = _content(message.get("content"), f"{location}.content")
        if message["content"] is None and role != "assistant":
            raise ValueError(f"{location}.content may be null only for assistant turns")
        if role == "assistant":
            if (
                trust_untagged_assistant_messages
                and message.get("provenance_tag") is None
            ):
                message["provenance_tag"] = "generated"
            assistant_count += 1
            _validate_reasoning(message, location)
            _validate_calls(message, location, known_calls, tool_names)
        elif message.get("tool_calls") is not None:
            raise ValueError(f"{location}.tool_calls is valid only for assistant turns")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in known_calls:
                raise ValueError(
                    f"{location}.tool_call_id does not reference an earlier tool call"
                )
            result_name = message.get("name")
            if result_name is not None:
                if not isinstance(result_name, str):
                    raise ValueError(f"{location}.name must be a string")
                for suffix in _HARMONY_TOOL_SUFFIXES:
                    if result_name.endswith(suffix):
                        result_name = result_name[: -len(suffix)]
                        message["name"] = result_name
                        break
                if result_name != known_calls[call_id]:
                    raise ValueError(
                        f"{location}.name does not match its referenced tool call"
                    )
    if not assistant_count:
        raise ValueError(f"trace {row_id!r} has no assistant turn")
    return messages


def _validate_reasoning(message: Mapping[str, Any], location: str) -> None:
    for field_name in ("thinking", "reasoning_content"):
        value = message.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{location}.{field_name} must be a string or null")


def _validate_calls(
    message: dict[str, Any],
    location: str,
    known_calls: dict[str, str],
    tool_names: set[str],
) -> None:
    calls = message.get("tool_calls")
    if calls is None:
        return
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{location}.tool_calls must be a non-empty list")
    if not tool_names:
        raise ValueError(f"{location}.tool_calls requires captured tool schemas")
    for index, call in enumerate(calls):
        call_location = f"{location}.tool_calls[{index}]"
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError(f"{call_location} must be an OpenAI function call")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"{call_location}.id must be a non-empty string")
        if call_id in known_calls:
            raise ValueError(f"{call_location}.id is duplicated")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"{call_location}.function must be an object")
        name = function.get("name")
        if isinstance(name, str):
            for suffix in _HARMONY_TOOL_SUFFIXES:
                if name.endswith(suffix) and name[: -len(suffix)] in tool_names:
                    name = name[: -len(suffix)]
                    function["name"] = name
                    break
        if not isinstance(name, str) or name not in tool_names:
            raise ValueError(f"{call_location}.function.name has no matching tool schema")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError(
                f"{call_location}.function.arguments must remain an OpenAI JSON string"
            )
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{call_location}.function.arguments must be valid JSON"
            ) from error
        if not isinstance(decoded, dict):
            raise ValueError(f"{call_location}.function.arguments must encode an object")
        known_calls[call_id] = name


def training_row_from_trace(
    trace: TraceRecord | Mapping[str, Any],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    model_revision: str | None = None,
    trust_untagged_assistant_messages: bool = False,
) -> TrainingRow:
    """Convert a production ``TraceRecord`` or JSON-compatible trace mapping.

    Production captures must carry per-message provenance. Operators importing a
    trusted offline corpus that predates provenance tags may explicitly set
    ``trust_untagged_assistant_messages=True``. That opt-in relabels only untagged
    assistant messages as ``"generated"``; unknown tags still fail closed.
    """
    if not isinstance(trust_untagged_assistant_messages, bool):
        raise ValueError("trust_untagged_assistant_messages must be a boolean")
    if isinstance(trace, TraceRecord):
        raw: Mapping[str, Any] = trace.to_dict()
    else:
        raw = trace
    row_id = raw.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("trace id must be a non-empty string")

    raw_tools: object = tools if tools is not None else raw.get("tools")
    if raw_tools is None:
        metadata_value = raw.get("metadata")
        if isinstance(metadata_value, Mapping):
            request = metadata_value.get("request")
            if isinstance(request, Mapping):
                raw_tools = request.get("tools")
    validated_tools = _tools(raw_tools, f"trace {row_id!r} tools")
    messages = _messages(
        raw.get("messages"),
        row_id=row_id,
        tool_names={tool["function"]["name"] for tool in validated_tools},
        trust_untagged_assistant_messages=trust_untagged_assistant_messages,
    )
    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError(f"trace {row_id!r} model must be a non-empty string or null")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"trace {row_id!r} metadata must be an object")
    return TrainingRow(
        id=row_id,
        conversation=tuple(messages),
        tools=tuple(validated_tools),
        model=model,
        model_revision=model_revision,
        metadata=_json_copy(metadata, f"trace {row_id!r} metadata"),
    )


def _generated_assistant_spans(
    row: TrainingRow,
    *,
    template: ChatTemplate,
    rendered: str,
    spans: Sequence[AssistantSpan],
) -> tuple[AssistantSpan, ...]:
    """Bind rendered spans to source messages and retain generated authorship.

    Prefix rendering avoids trusting template turn numbers to distinguish adjacent
    assistant messages. If a template is not prefix-stable, no span beyond the
    unstable prefix is admitted for supervision.
    """
    if not spans:
        return ()

    prefix_length = 0
    generated: list[AssistantSpan] = []
    for message_index, message in enumerate(row.conversation):
        prefix = template.render(
            row.conversation[: message_index + 1],
            tools=row.tools,
        )
        if not rendered.startswith(prefix) or len(prefix) < prefix_length:
            break
        next_prefix_length = len(prefix)
        if (
            message.get("role") == "assistant"
            and message.get("provenance_tag") == "generated"
        ):
            generated.extend(
                span
                for span in spans
                if prefix_length <= span.start and span.end <= next_prefix_length
            )
        prefix_length = next_prefix_length
    return tuple(generated)


def _integer_sequence(value: object, field: str, row_id: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"training row {row_id!r} tokenizer returned no {field}")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"training row {row_id!r} tokenizer returned non-integer {field}"
            )
        result.append(item)
    if not result:
        raise ValueError(f"training row {row_id!r} tokenizer returned empty {field}")
    return tuple(result)


def _offset_sequence(value: object, row_id: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"training row {row_id!r} requires a fast tokenizer offset_mapping"
        )
    result: list[tuple[int, int]] = []
    for offset in value:
        if (
            not isinstance(offset, Sequence)
            or len(offset) != 2
            or not all(isinstance(point, int) for point in offset)
        ):
            raise ValueError(
                f"training row {row_id!r} tokenizer returned invalid offset_mapping"
            )
        result.append((int(offset[0]), int(offset[1])))
    return tuple(result)


def load_tokenizer_snapshot(snapshot: Path) -> Any:
    """Load an exact local Transformers snapshot without a network fallback."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("transformers is required to load a tokenizer snapshot") from error
    resolved = snapshot.resolve()
    if not (resolved / "tokenizer.json").is_file():
        raise RuntimeError(f"tokenizer.json is unavailable in snapshot {resolved}")
    try:
        return AutoTokenizer.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"tokenizer in snapshot {resolved} could not be loaded offline"
        ) from error


def token_ids_sha256(input_ids: Sequence[int]) -> str:
    """Stable digest used for exact prepared-token comparisons."""
    payload = json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def harmony_render_messages(row: TrainingRow) -> list[dict[str, Any]]:
    """Return a non-mutating render view with decoded tool arguments."""
    messages = copy.deepcopy([dict(message) for message in row.conversation])
    for message in messages:
        if message.get("role") == "assistant" and message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            call["function"]["arguments"] = json.loads(arguments)
    return messages
