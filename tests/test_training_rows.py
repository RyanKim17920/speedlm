"""Contract tests for trace-to-training-row conversion and mask binding.

The rows module decides which rendered characters may receive loss. A silent
label/mask misalignment here trains a plausible-looking draft head on the wrong
tokens, so every validator and every span-binding boundary is pinned exactly.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from speedlm.traces.store import TraceRecord
from speedlm.training.masking import MaskPolicy
from speedlm.training.rows import (
    PreparedTrainingRow,
    TrainingRow,
    _generated_assistant_spans,
    _integer_sequence,
    _offset_sequence,
    harmony_render_messages,
    load_tokenizer_snapshot,
    token_ids_sha256,
    training_row_from_trace,
)
from speedlm.training.templates.base import AssistantSpan, ChatTemplate

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _tool(
    name: str = "search",
    *,
    description: str = "Search the corpus",
    parameters: object = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object"} if parameters is None else parameters,
        },
    }


def _call(
    call_id: str = "call-1",
    *,
    name: str = "search",
    arguments: object = '{"q":"x"}',
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _trace(**overrides: object) -> dict[str, Any]:
    """Minimal valid trace mapping; override any key."""
    trace: dict[str, Any] = {
        "id": "trace-1",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "provenance_tag": "generated"},
        ],
    }
    trace.update(overrides)
    return trace


def _trace_record(**overrides: object) -> TraceRecord:
    fields: dict[str, Any] = {
        "id": "record-1",
        "timestamp": 1.0,
        "model": "gpt-oss-20b",
        "messages": (
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "provenance_tag": "generated"},
        ),
        "tool_calls": (),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "prompt_tokens": 2,
        "completion_tokens": 1,
    }
    fields.update(overrides)
    return TraceRecord(**fields)


class _ConcatTemplate:
    """Prefix-stable template: the render is the concatenation of contents.

    Real class rather than a mock so the protocol conformance and the exact
    prefix arithmetic that ``_generated_assistant_spans`` relies on are real.
    """

    name = "concat"

    def render(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        return "".join(str(message.get("content") or "") for message in conversation)

    def assistant_spans(self, rendered: str) -> tuple[AssistantSpan, ...]:  # pragma: no cover
        raise AssertionError("spans are supplied explicitly by these tests")


class _UnstableAtThirdTemplate(_ConcatTemplate):
    """Prefix-stable except when rendering exactly three messages."""

    name = "unstable-at-third"

    def render(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        rendered = super().render(conversation, tools=tools)
        if len(conversation) == 3:
            return f"!{rendered}"
        return rendered


class _ShrinkingAtThirdTemplate(_ConcatTemplate):
    """Still a prefix of the full render, but shorter than the previous prefix."""

    name = "shrinking-at-third"

    def render(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        if len(conversation) == 3:
            return super().render(conversation[:1], tools=tools)
        return super().render(conversation, tools=tools)


def _row(conversation: Sequence[Mapping[str, Any]]) -> TrainingRow:
    return TrainingRow(id="row", conversation=tuple(conversation))


# --------------------------------------------------------------------------
# PreparedTrainingRow
# --------------------------------------------------------------------------


def test_prepared_row_as_dict_is_json_ready_and_drops_the_span_objects() -> None:
    """The serialized form is what reaches disk; spans are deliberately absent."""
    span = AssistantSpan(start=2, end=4, turn=0, channel="final")
    prepared = PreparedTrainingRow(
        id="row",
        input_ids=(1, 2, 3),
        loss_mask=(False, True, True),
        seq_len=3,
        rendered="UUAA",
        assistant_spans=(span,),
        mask_policy=MaskPolicy.FINAL_SPAN,
    )

    assert prepared.as_dict() == {
        "id": "row",
        "input_ids": [1, 2, 3],
        "loss_mask": [False, True, True],
        "seq_len": 3,
        "rendered": "UUAA",
        "mask_policy": "final_span",
    }


# --------------------------------------------------------------------------
# training_row_from_trace: inputs and passthrough
# --------------------------------------------------------------------------


def test_trace_record_input_matches_the_equivalent_mapping() -> None:
    record = _trace_record()

    from_record = training_row_from_trace(record)
    from_mapping = training_row_from_trace(record.to_dict())

    assert from_record.id == "record-1"
    assert from_record.model == "gpt-oss-20b"
    assert from_record.conversation == from_mapping.conversation
    assert from_record.metadata == {}
    assert from_record.tools == ()


def test_conversation_is_a_tuple_of_plain_dicts_not_the_source_objects() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    row = training_row_from_trace(_trace(messages=messages))

    assert isinstance(row.conversation, tuple)
    messages[0]["content"] = "mutated"
    assert row.conversation[0]["content"] == "hi"


def test_explicit_tools_argument_overrides_the_trace_tools() -> None:
    row = training_row_from_trace(
        _trace(tools=[_tool("from_trace")]),
        tools=[_tool("explicit")],
    )

    assert [tool["function"]["name"] for tool in row.tools] == ["explicit"]


def test_empty_explicit_tools_argument_suppresses_the_metadata_fallback() -> None:
    trace = _trace(metadata={"request": {"tools": [_tool("from_metadata")]}})

    row = training_row_from_trace(trace, tools=[])

    assert row.tools == ()


def test_tools_fall_back_to_metadata_request_tools() -> None:
    trace = _trace(metadata={"request": {"tools": [_tool("from_metadata")]}})

    row = training_row_from_trace(trace)

    assert [tool["function"]["name"] for tool in row.tools] == ["from_metadata"]


def test_metadata_without_a_request_mapping_yields_no_tools() -> None:
    assert training_row_from_trace(_trace(metadata={"request": "not-a-mapping"})).tools == ()
    assert training_row_from_trace(_trace(metadata={"other": 1})).tools == ()
    assert training_row_from_trace(_trace(metadata={"request": {}})).tools == ()


def test_model_revision_is_passed_through_verbatim() -> None:
    row = training_row_from_trace(_trace(), model_revision="abc123")

    assert row.model_revision == "abc123"
    assert training_row_from_trace(_trace()).model_revision is None


def test_metadata_is_deep_copied_not_aliased() -> None:
    metadata: dict[str, Any] = {"request": {"headers": {"x": "1"}}, "tags": ["a"]}

    row = training_row_from_trace(_trace(metadata=metadata))
    metadata["request"]["headers"]["x"] = "2"
    metadata["tags"].append("b")
    metadata["new"] = True

    assert row.metadata == {"request": {"headers": {"x": "1"}}, "tags": ["a"]}


def test_non_bool_trust_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="trust_untagged_assistant_messages must be a boolean"):
        training_row_from_trace(_trace(), trust_untagged_assistant_messages=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("row_id", [None, "", 7, b"trace"])
def test_invalid_row_id_is_rejected(row_id: object) -> None:
    with pytest.raises(ValueError, match="trace id must be a non-empty string"):
        training_row_from_trace(_trace(id=row_id))


def test_missing_id_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="trace id must be a non-empty string"):
        training_row_from_trace({"messages": [{"role": "assistant", "content": "x"}]})


def test_non_mapping_metadata_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=re.escape("trace 'trace-1' metadata must be an object")
    ):
        training_row_from_trace(_trace(metadata=["not", "a", "mapping"]))


def test_model_may_be_null_but_not_empty() -> None:
    assert training_row_from_trace(_trace(model=None)).model is None
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' model must be a non-empty string or null"),
    ):
        training_row_from_trace(_trace(model=""))
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' model must be a non-empty string or null"),
    ):
        training_row_from_trace(_trace(model=42))


# --------------------------------------------------------------------------
# provenance semantics
# --------------------------------------------------------------------------


def test_untagged_assistant_is_relabelled_generated_only_under_the_opt_in() -> None:
    trace = _trace(messages=[{"role": "assistant", "content": "hello"}])

    trusted = training_row_from_trace(trace, trust_untagged_assistant_messages=True)
    untrusted = training_row_from_trace(trace)

    assert trusted.conversation[0]["provenance_tag"] == "generated"
    assert "provenance_tag" not in untrusted.conversation[0]


def test_an_already_tagged_assistant_is_never_relabelled() -> None:
    trace = _trace(
        messages=[
            {"role": "assistant", "content": "a", "provenance_tag": "client_supplied"},
            {"role": "assistant", "content": "b", "provenance_tag": "generated"},
        ]
    )

    row = training_row_from_trace(trace, trust_untagged_assistant_messages=True)

    assert [message["provenance_tag"] for message in row.conversation] == [
        "client_supplied",
        "generated",
    ]


def test_the_opt_in_never_tags_non_assistant_turns() -> None:
    trace = _trace(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )

    row = training_row_from_trace(trace, trust_untagged_assistant_messages=True)

    assert "provenance_tag" not in row.conversation[0]
    assert row.conversation[1]["provenance_tag"] == "generated"


def test_an_unknown_provenance_tag_is_carried_but_never_supervised() -> None:
    """rows.py does not validate tag values; the fail-closed point is the mask.

    ``TraceRecord`` rejects unknown tags at capture time, but a plain mapping
    bypasses that, so the only thing standing between an unknown tag and a loss
    mask is the exact ``== "generated"`` check in ``_generated_assistant_spans``.
    """
    trace = _trace(
        messages=[{"role": "assistant", "content": "hello", "provenance_tag": "unknown"}]
    )

    row = training_row_from_trace(trace, trust_untagged_assistant_messages=True)

    assert row.conversation[0]["provenance_tag"] == "unknown"
    spans = (AssistantSpan(start=0, end=5, turn=0, channel="final"),)
    assert _generated_assistant_spans(
        row,
        template=_ConcatTemplate(),
        rendered="hello",
        spans=spans,
    ) == ()

    with pytest.raises(Exception, match="provenance_tag"):
        _trace_record(
            messages=({"role": "assistant", "content": "x", "provenance_tag": "unknown"},)
        )


# --------------------------------------------------------------------------
# _content
# --------------------------------------------------------------------------


def test_string_content_survives_unchanged() -> None:
    row = training_row_from_trace(_trace(messages=[{"role": "assistant", "content": "hé"}]))

    assert row.conversation[0]["content"] == "hé"


def test_null_content_is_assistant_only() -> None:
    row = training_row_from_trace(_trace(messages=[{"role": "assistant", "content": None}]))
    assert row.conversation[0]["content"] is None

    for role in ("user", "system", "tool"):
        with pytest.raises(
            ValueError,
            match=re.escape("content may be null only for assistant turns"),
        ):
            training_row_from_trace(
                _trace(
                    messages=[
                        {"role": role, "content": None},
                        {"role": "assistant", "content": "x"},
                    ]
                )
            )


def test_text_content_parts_are_concatenated_in_order() -> None:
    row = training_row_from_trace(
        _trace(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "one "},
                        {"type": "text", "text": "two "},
                        {"type": "text", "text": "three"},
                    ],
                }
            ]
        )
    )

    assert row.conversation[0]["content"] == "one two three"


def test_empty_content_part_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("must be a string, null, or a non-empty text content-part list"),
    ):
        training_row_from_trace(_trace(messages=[{"role": "assistant", "content": []}]))


def test_non_list_non_string_content_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("must be a string, null, or a non-empty text content-part list"),
    ):
        training_row_from_trace(_trace(messages=[{"role": "assistant", "content": 5}]))


def test_a_non_text_content_part_is_rejected_with_its_index() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].content[1] must be an OpenAI text content part"),
    ):
        training_row_from_trace(
            _trace(
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {"type": "image_url", "image_url": {"url": "http://x"}},
                        ],
                    }
                ]
            )
        )


def test_a_content_part_whose_text_is_not_a_string_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].content[0] must be an OpenAI text content part"),
    ):
        training_row_from_trace(
            _trace(messages=[{"role": "assistant", "content": [{"type": "text", "text": 1}]}])
        )


def test_a_bare_string_content_part_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].content[0] must be an OpenAI text content part"),
    ):
        training_row_from_trace(_trace(messages=[{"role": "assistant", "content": ["hi"]}]))


# --------------------------------------------------------------------------
# _tools
# --------------------------------------------------------------------------


def test_absent_tools_produce_an_empty_tuple() -> None:
    assert training_row_from_trace(_trace()).tools == ()


def test_non_sequence_tools_are_rejected() -> None:
    with pytest.raises(ValueError, match=re.escape("trace 'trace-1' tools must be a sequence")):
        training_row_from_trace(_trace(tools={"type": "function"}))


def test_a_non_function_tool_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0] must be an OpenAI function tool"),
    ):
        training_row_from_trace(_trace(tools=[{"type": "retrieval"}]))


def test_a_non_object_tool_entry_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0] must be an OpenAI function tool"),
    ):
        training_row_from_trace(_trace(tools=["search"]))


def test_a_non_object_function_body_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0].function must be an object"),
    ):
        training_row_from_trace(_trace(tools=[{"type": "function", "function": "search"}]))


@pytest.mark.parametrize("name", [None, "", 3])
def test_an_invalid_tool_name_is_rejected(name: object) -> None:
    tool = _tool()
    if name is None:
        del tool["function"]["name"]
    else:
        tool["function"]["name"] = name
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0].function.name must be a non-empty string"),
    ):
        training_row_from_trace(_trace(tools=[tool]))


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools has duplicate function name 'search'"),
    ):
        training_row_from_trace(_trace(tools=[_tool("search"), _tool("search")]))


def test_a_missing_tool_description_is_rejected() -> None:
    tool = _tool()
    del tool["function"]["description"]
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0].function.description must be a string"),
    ):
        training_row_from_trace(_trace(tools=[tool]))


def test_non_object_tool_parameters_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools[0].function.parameters must be an object"),
    ):
        training_row_from_trace(_trace(tools=[_tool(parameters=["object"])]))


def test_non_json_serializable_tools_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' tools must be JSON serializable"),
    ):
        training_row_from_trace(_trace(tools=[_tool(parameters={"enum": {1, 2}})]))


def test_a_tuple_of_tools_is_accepted_and_normalized_to_dicts() -> None:
    row = training_row_from_trace(_trace(tools=(_tool("search"),)))

    assert isinstance(row.tools, tuple)
    assert isinstance(row.tools[0], dict)
    assert row.tools[0]["function"]["name"] == "search"


# --------------------------------------------------------------------------
# _messages / _validate_calls
# --------------------------------------------------------------------------


def test_an_empty_message_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' messages must be a non-empty sequence"),
    ):
        training_row_from_trace(_trace(messages=[]))


def test_missing_messages_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' messages must be a non-empty sequence"),
    ):
        training_row_from_trace({"id": "trace-1"})


def test_a_non_object_message_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' messages[0] must be an object"),
    ):
        training_row_from_trace(_trace(messages=["hello"]))


@pytest.mark.parametrize("role", [None, "", 1])
def test_an_invalid_role_is_rejected(role: object) -> None:
    message: dict[str, Any] = {"content": "x"}
    if role is not None:
        message["role"] = role
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' messages[0].role must be a non-empty string"),
    ):
        training_row_from_trace(_trace(messages=[message]))


def test_a_conversation_without_an_assistant_turn_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("trace 'trace-1' has no assistant turn"),
    ):
        training_row_from_trace(_trace(messages=[{"role": "user", "content": "hi"}]))


def test_tool_calls_on_a_non_assistant_turn_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls is valid only for assistant turns"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {"role": "user", "content": "hi", "tool_calls": [_call()]},
                    {"role": "assistant", "content": "x"},
                ],
            )
        )


def test_an_empty_tool_calls_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls must be a non-empty list"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[{"role": "assistant", "content": None, "tool_calls": []}],
            )
        )


def test_tool_calls_without_captured_tool_schemas_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls requires captured tool schemas"),
    ):
        training_row_from_trace(
            _trace(messages=[{"role": "assistant", "content": None, "tool_calls": [_call()]}])
        )


def test_a_non_function_call_shape_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0] must be an OpenAI function call"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "type": "custom"}],
                    }
                ],
            )
        )


@pytest.mark.parametrize("call_id", [None, "", 5])
def test_an_invalid_call_id_is_rejected(call_id: object) -> None:
    call = _call()
    if call_id is None:
        del call["id"]
    else:
        call["id"] = call_id
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].id must be a non-empty string"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[{"role": "assistant", "content": None, "tool_calls": [call]}],
            )
        )


def test_a_duplicate_call_id_across_turns_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[2].tool_calls[0].id is duplicated"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {"role": "assistant", "content": None, "tool_calls": [_call("dup")]},
                    {"role": "tool", "tool_call_id": "dup", "content": "ok"},
                    {"role": "assistant", "content": None, "tool_calls": [_call("dup")]},
                ],
            )
        )


def test_a_non_object_call_function_is_rejected() -> None:
    call = _call()
    call["function"] = "search"
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function must be an object"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[{"role": "assistant", "content": None, "tool_calls": [call]}],
            )
        )


def test_a_call_naming_an_uncaptured_tool_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function.name has no matching tool schema"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool("search")],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(name="write")],
                    }
                ],
            )
        )


def test_a_non_string_call_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function.name has no matching tool schema"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool("search")],
                messages=[
                    {"role": "assistant", "content": None, "tool_calls": [_call(name=None)]}
                ],
            )
        )


def test_non_string_call_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "messages[0].tool_calls[0].function.arguments must remain an OpenAI JSON string"
        ),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(arguments={"q": "x"})],
                    }
                ],
            )
        )


def test_unparseable_call_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function.arguments must be valid JSON"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(arguments="{not json")],
                    }
                ],
            )
        )


@pytest.mark.parametrize("arguments", ["[1,2]", '"x"', "null", "3"])
def test_call_arguments_that_do_not_encode_an_object_are_rejected(arguments: str) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function.arguments must encode an object"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(arguments=arguments)],
                    }
                ],
            )
        )


def test_a_valid_tool_call_round_trip_keeps_arguments_as_a_json_string() -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool()],
            messages=[
                {"role": "user", "content": "find"},
                {"role": "assistant", "content": None, "tool_calls": [_call()]},
                {"role": "tool", "tool_call_id": "call-1", "name": "search", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ],
        )
    )

    assert row.conversation[1]["tool_calls"][0]["function"]["arguments"] == '{"q":"x"}'
    assert row.conversation[2]["name"] == "search"


def test_a_tool_result_referencing_no_earlier_call_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_call_id does not reference an earlier tool call"),
    ):
        training_row_from_trace(
            _trace(
                messages=[
                    {"role": "tool", "tool_call_id": "missing", "content": "ok"},
                    {"role": "assistant", "content": "x"},
                ]
            )
        )


def test_a_tool_result_with_a_non_string_call_id_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_call_id does not reference an earlier tool call"),
    ):
        training_row_from_trace(
            _trace(
                messages=[
                    {"role": "tool", "tool_call_id": 7, "content": "ok"},
                    {"role": "assistant", "content": "x"},
                ]
            )
        )


def test_a_tool_result_ordered_before_its_call_is_rejected() -> None:
    """Forward references are not resolvable; known_calls is built in order."""
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_call_id does not reference an earlier tool call"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
                    {"role": "assistant", "content": None, "tool_calls": [_call()]},
                ],
            )
        )


def test_a_tool_result_name_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[2].name does not match its referenced tool call"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool("search"), _tool("write")],
                messages=[
                    {"role": "user", "content": "find"},
                    {"role": "assistant", "content": None, "tool_calls": [_call(name="search")]},
                    {"role": "tool", "tool_call_id": "call-1", "name": "write", "content": "ok"},
                ],
            )
        )


def test_a_non_string_tool_result_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("messages[2].name must be a string"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool()],
                messages=[
                    {"role": "user", "content": "find"},
                    {"role": "assistant", "content": None, "tool_calls": [_call()]},
                    {"role": "tool", "tool_call_id": "call-1", "name": 5, "content": "ok"},
                ],
            )
        )


def test_a_tool_result_may_omit_its_name() -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool()],
            messages=[
                {"role": "assistant", "content": None, "tool_calls": [_call()]},
                {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            ],
        )
    )

    assert "name" not in row.conversation[1]


# --------------------------------------------------------------------------
# Harmony suffix stripping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    ["<|channel|>analysis", "<|channel|>commentary", "<|channel|>json"],
)
def test_every_harmony_suffix_is_stripped_from_a_call_naming_a_known_tool(suffix: str) -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool("search")],
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call(name=f"search{suffix}")],
                }
            ],
        )
    )

    assert row.conversation[0]["tool_calls"][0]["function"]["name"] == "search"


def test_a_suffixed_call_whose_stripped_name_is_unknown_is_rejected_unstripped() -> None:
    """Stripping at rows.py:213 is conditional on the stripped name being known."""
    with pytest.raises(
        ValueError,
        match=re.escape("messages[0].tool_calls[0].function.name has no matching tool schema"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool("search")],
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(name="write<|channel|>analysis")],
                    }
                ],
            )
        )


def test_a_tool_literally_named_with_a_suffix_wins_over_stripping() -> None:
    """The stripped name is only adopted when it resolves; the literal name is preferred."""
    row = training_row_from_trace(
        _trace(
            tools=[_tool("search<|channel|>analysis")],
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call(name="search<|channel|>analysis")],
                }
            ],
        )
    )

    assert (
        row.conversation[0]["tool_calls"][0]["function"]["name"] == "search<|channel|>analysis"
    )


def test_a_suffixed_tool_result_name_is_stripped_before_comparison() -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool("search")],
            messages=[
                {"role": "assistant", "content": None, "tool_calls": [_call(name="search")]},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "search<|channel|>commentary",
                    "content": "ok",
                },
            ],
        )
    )

    assert row.conversation[1]["name"] == "search"


def test_unconditional_result_name_stripping_rejects_a_literally_suffixed_tool() -> None:
    """BUG (rows.py:164-172): the tool-result strip is unconditional.

    Unlike the call-name strip at rows.py:212-216, which only adopts the stripped
    name when it resolves to a captured tool, the tool-result branch strips any
    trailing Harmony suffix before comparing. A tool genuinely named
    ``search<|channel|>analysis`` therefore records ``known_calls['call-1'] ==
    'search<|channel|>analysis'`` but its result name is rewritten to ``'search'``,
    so a perfectly consistent conversation is rejected. This test pins CURRENT
    behaviour; it should become an accepted round trip once the strip is made
    conditional on the stripped name matching the referenced call.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("messages[2].name does not match its referenced tool call"),
    ):
        training_row_from_trace(
            _trace(
                tools=[_tool("search<|channel|>analysis")],
                messages=[
                    {"role": "user", "content": "find"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(name="search<|channel|>analysis")],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "search<|channel|>analysis",
                        "content": "ok",
                    },
                ],
            )
        )


def test_only_the_first_matching_suffix_is_stripped() -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool("search<|channel|>json")],
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call(name="search<|channel|>json<|channel|>analysis")],
                }
            ],
        )
    )

    assert (
        row.conversation[0]["tool_calls"][0]["function"]["name"] == "search<|channel|>json"
    )


# --------------------------------------------------------------------------
# _validate_reasoning
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["thinking", "reasoning_content"])
def test_reasoning_fields_accept_strings_and_null(field_name: str) -> None:
    for value in ("because", None):
        row = training_row_from_trace(
            _trace(messages=[{"role": "assistant", "content": "x", field_name: value}])
        )
        assert row.conversation[0][field_name] == value


@pytest.mark.parametrize("field_name", ["thinking", "reasoning_content"])
def test_non_string_reasoning_is_rejected(field_name: str) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"messages[0].{field_name} must be a string or null"),
    ):
        training_row_from_trace(
            _trace(messages=[{"role": "assistant", "content": "x", field_name: ["a"]}])
        )


# --------------------------------------------------------------------------
# _generated_assistant_spans — the masking core
# --------------------------------------------------------------------------


def test_no_spans_returns_an_empty_tuple() -> None:
    row = _row([{"role": "assistant", "content": "ab", "provenance_tag": "generated"}])

    assert _generated_assistant_spans(
        row, template=_ConcatTemplate(), rendered="ab", spans=()
    ) == ()


def test_a_generated_assistant_span_is_admitted() -> None:
    row = _row(
        [
            {"role": "user", "content": "UU"},
            {"role": "assistant", "content": "AAA", "provenance_tag": "generated"},
        ]
    )
    span = AssistantSpan(start=2, end=5, turn=0, channel="final")

    assert _generated_assistant_spans(
        row, template=_ConcatTemplate(), rendered="UUAAA", spans=(span,)
    ) == (span,)


def test_an_untagged_assistant_span_is_excluded() -> None:
    row = _row(
        [
            {"role": "user", "content": "UU"},
            {"role": "assistant", "content": "AAA"},
        ]
    )
    span = AssistantSpan(start=2, end=5, turn=0, channel="final")

    assert _generated_assistant_spans(
        row, template=_ConcatTemplate(), rendered="UUAAA", spans=(span,)
    ) == ()


def test_a_client_supplied_assistant_span_is_excluded() -> None:
    row = _row(
        [{"role": "assistant", "content": "AAA", "provenance_tag": "client_supplied"}]
    )
    span = AssistantSpan(start=0, end=3, turn=0, channel="final")

    assert _generated_assistant_spans(
        row, template=_ConcatTemplate(), rendered="AAA", spans=(span,)
    ) == ()


def test_spans_over_user_and_tool_turns_are_excluded() -> None:
    row = _row(
        [
            {"role": "user", "content": "UU"},
            {"role": "tool", "content": "TT", "provenance_tag": "generated"},
            {"role": "assistant", "content": "AA", "provenance_tag": "generated"},
        ]
    )
    user_span = AssistantSpan(start=0, end=2, turn=0, channel="final")
    tool_span = AssistantSpan(start=2, end=4, turn=1, channel="final")
    assistant_span = AssistantSpan(start=4, end=6, turn=2, channel="final")

    assert _generated_assistant_spans(
        row,
        template=_ConcatTemplate(),
        rendered="UUTTAA",
        spans=(user_span, tool_span, assistant_span),
    ) == (assistant_span,)


def test_only_the_generated_one_of_two_adjacent_assistant_turns_is_supervised() -> None:
    """The exact misalignment the prefix rendering exists to prevent.

    Two assistant messages sit back to back; only the second was produced by the
    provider. Binding by turn index alone would supervise the replayed one.
    """
    row = _row(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "AA", "provenance_tag": "client_supplied"},
            {"role": "assistant", "content": "BB", "provenance_tag": "generated"},
        ]
    )
    replayed = AssistantSpan(start=1, end=3, turn=0, channel="final")
    generated = AssistantSpan(start=3, end=5, turn=1, channel="final")

    assert _generated_assistant_spans(
        row,
        template=_ConcatTemplate(),
        rendered="UAABB",
        spans=(replayed, generated),
    ) == (generated,)


def test_a_span_straddling_a_turn_boundary_is_dropped() -> None:
    """Admission is [prefix_length, next_prefix_length] inclusive on both ends."""
    row = _row(
        [
            {"role": "user", "content": "UU"},
            {"role": "assistant", "content": "AA", "provenance_tag": "generated"},
        ]
    )
    straddling = AssistantSpan(start=1, end=3, turn=0, channel="final")
    exact = AssistantSpan(start=2, end=4, turn=0, channel="final")
    overrunning = AssistantSpan(start=2, end=5, turn=0, channel="final")

    admitted = _generated_assistant_spans(
        row,
        template=_ConcatTemplate(),
        rendered="UUAA",
        spans=(straddling, exact, overrunning),
    )

    assert admitted == (exact,)


def test_a_non_prefix_stable_template_admits_nothing_past_the_unstable_point() -> None:
    row = _row(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "AA", "provenance_tag": "generated"},
            {"role": "user", "content": "V"},
            {"role": "assistant", "content": "BB", "provenance_tag": "generated"},
        ]
    )
    early = AssistantSpan(start=1, end=3, turn=0, channel="final")
    late = AssistantSpan(start=4, end=6, turn=1, channel="final")

    assert _generated_assistant_spans(
        row,
        template=_UnstableAtThirdTemplate(),
        rendered="UAAVBB",
        spans=(early, late),
    ) == (early,)


def test_a_non_monotonic_prefix_length_stops_admission() -> None:
    """Line 312's ``len(prefix) < prefix_length`` guard, hit with a real prefix.

    The shrinking template still returns a genuine prefix of the full render, so
    only the length regression can stop the loop.
    """
    template = _ShrinkingAtThirdTemplate()
    row = _row(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "AA", "provenance_tag": "generated"},
            {"role": "user", "content": "V"},
            {"role": "assistant", "content": "BB", "provenance_tag": "generated"},
        ]
    )
    assert "UAAVBB".startswith(template.render(row.conversation[:3]))

    early = AssistantSpan(start=1, end=3, turn=0, channel="final")
    late = AssistantSpan(start=4, end=6, turn=1, channel="final")

    assert _generated_assistant_spans(
        row, template=template, rendered="UAAVBB", spans=(early, late)
    ) == (early,)


def test_the_fake_templates_satisfy_the_chat_template_protocol() -> None:
    assert isinstance(_ConcatTemplate(), ChatTemplate)
    assert isinstance(_UnstableAtThirdTemplate(), ChatTemplate)


# --------------------------------------------------------------------------
# _integer_sequence / _offset_sequence
#
# NOTE: as of this commit neither helper has a caller anywhere in src/ or
# tests/ -- the prepare path that will consume them is not written yet. They are
# exercised directly so the contract is pinned before that wiring lands.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, 5, {"a": 1}, "012", b"012"])
def test_integer_sequence_rejects_non_sequences_and_text(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned no input_ids"),
    ):
        _integer_sequence(value, "input_ids", "r")


def test_integer_sequence_accepts_lists_and_tuples() -> None:
    assert _integer_sequence([1, 2, 3], "input_ids", "r") == (1, 2, 3)
    assert _integer_sequence((0,), "input_ids", "r") == (0,)


@pytest.mark.parametrize("item", [True, False, 1.0, "1", None])
def test_integer_sequence_rejects_non_int_items_including_bools(item: object) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned non-integer input_ids"),
    ):
        _integer_sequence([1, item], "input_ids", "r")


def test_integer_sequence_rejects_an_empty_result() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned empty input_ids"),
    ):
        _integer_sequence([], "input_ids", "r")


def test_integer_sequence_names_the_field_in_its_error() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned empty attention_mask"),
    ):
        _integer_sequence((), "attention_mask", "r")


@pytest.mark.parametrize("value", [None, 5, "ab", b"ab"])
def test_offset_sequence_rejects_non_sequences_and_text(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' requires a fast tokenizer offset_mapping"),
    ):
        _offset_sequence(value, "r")


def test_offset_sequence_normalizes_pairs_to_tuples() -> None:
    assert _offset_sequence([[0, 2], (2, 5)], "r") == ((0, 2), (2, 5))


def test_offset_sequence_accepts_an_empty_mapping() -> None:
    assert _offset_sequence([], "r") == ()


@pytest.mark.parametrize("offset", [[0], [0, 1, 2], 3, None])
def test_offset_sequence_rejects_wrong_length_offsets(offset: object) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned invalid offset_mapping"),
    ):
        _offset_sequence([[0, 1], offset], "r")


@pytest.mark.parametrize("point", [1.5, "1", None])
def test_offset_sequence_rejects_non_int_points(point: object) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("training row 'r' tokenizer returned invalid offset_mapping"),
    ):
        _offset_sequence([[0, point]], "r")


def test_offset_sequence_admits_bool_points() -> None:
    """Documented gap: unlike _integer_sequence, offsets do not reject bools.

    ``isinstance(True, int)`` is True and there is no bool guard at rows.py:353,
    so ``[True, False]`` becomes the offset ``(1, 0)`` -- an inverted span that
    would silently contribute nothing to a loss mask.
    """
    assert _offset_sequence([[True, False]], "r") == ((1, 0),)


# --------------------------------------------------------------------------
# token_ids_sha256
# --------------------------------------------------------------------------


def test_token_digest_matches_a_pinned_constant() -> None:
    """Pins the exact serialization: compact JSON of the id list, ASCII encoded.

    A change to separators, to list-vs-tuple rendering, or to the hash would
    invalidate every stored prepared-token comparison, so the constant is the
    contract.
    """
    assert token_ids_sha256([1, 2, 3]) == (
        "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
    )
    assert token_ids_sha256([]) == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )


def test_token_digest_is_deterministic_and_container_agnostic() -> None:
    assert token_ids_sha256([1, 2, 3]) == token_ids_sha256([1, 2, 3])
    assert token_ids_sha256((1, 2, 3)) == token_ids_sha256([1, 2, 3])
    assert token_ids_sha256(range(1, 4)) == token_ids_sha256([1, 2, 3])


def test_token_digest_is_order_sensitive() -> None:
    assert token_ids_sha256([1, 2, 3]) != token_ids_sha256([3, 2, 1])
    assert token_ids_sha256([1, 2]) != token_ids_sha256([12])


# --------------------------------------------------------------------------
# harmony_render_messages
# --------------------------------------------------------------------------


def test_harmony_render_replaces_null_assistant_content_with_an_empty_string() -> None:
    row = training_row_from_trace(
        _trace(messages=[{"role": "assistant", "content": None}])
    )

    assert harmony_render_messages(row)[0]["content"] == ""
    assert row.conversation[0]["content"] is None


def test_harmony_render_decodes_call_arguments_without_mutating_the_row() -> None:
    row = training_row_from_trace(
        _trace(
            tools=[_tool()],
            messages=[
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call(arguments='{"q":"x","n":2}')],
                },
            ],
        )
    )

    rendered = harmony_render_messages(row)

    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"q": "x", "n": 2}
    assert row.conversation[1]["tool_calls"][0]["function"]["arguments"] == '{"q":"x","n":2}'
    assert row.conversation[1]["content"] is None


def test_harmony_render_leaves_a_message_without_tool_calls_alone() -> None:
    row = training_row_from_trace(
        _trace(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "thinking": "hm"},
            ]
        )
    )

    rendered = harmony_render_messages(row)

    assert rendered == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "thinking": "hm"},
    ]


def test_harmony_render_crashes_on_an_explicit_null_tool_calls_key() -> None:
    """BUG (rows.py:395): ``message.get("tool_calls", [])`` returns None, not [].

    ``_validate_calls`` returns early for ``tool_calls: None`` and ``_json_copy``
    preserves the key, so an assistant message that carries an explicit null
    ``tool_calls`` -- which several OpenAI-compatible servers emit -- survives
    validation and then raises ``TypeError: 'NoneType' object is not iterable``
    here. ``message.get("tool_calls") or []`` would fix it. Pinning current
    behaviour.
    """
    row = training_row_from_trace(
        _trace(messages=[{"role": "assistant", "content": "x", "tool_calls": None}])
    )
    assert row.conversation[0]["tool_calls"] is None

    with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
        harmony_render_messages(row)


# --------------------------------------------------------------------------
# load_tokenizer_snapshot
# --------------------------------------------------------------------------


def test_a_missing_transformers_install_raises_a_named_runtime_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed; the ImportError path is unreachable")

    with pytest.raises(
        RuntimeError, match="transformers is required to load a tokenizer snapshot"
    ):
        load_tokenizer_snapshot(tmp_path)


def test_a_snapshot_without_tokenizer_json_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(RuntimeError, match="tokenizer.json is unavailable in snapshot"):
        load_tokenizer_snapshot(snapshot)


def test_an_unloadable_tokenizer_json_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="could not be loaded offline"):
        load_tokenizer_snapshot(snapshot)
