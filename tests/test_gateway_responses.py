"""Tests for speedlm.gateway.responses — the OpenAI /v1/responses capture branch.

Every parser in this module returns ``None`` instead of raising, so a parsing
regression is silent and poisons the training corpus rather than failing loudly.
These tests therefore assert the captured field values, and assert ``is None``
explicitly wherever the production code chooses silence.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from speedlm.gateway.capture import _decode_response_body
from speedlm.gateway.responses import (
    parse_responses_response,
    parse_responses_sse,
    response_status_and_id,
    responses_request_messages,
)

# ── helpers ─────────────────────────────────────────────────────────────────


def _json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode()


def _event(event_type: str, response: Any) -> str:
    return json.dumps({"type": event_type, "response": response}, separators=(",", ":"))


def _message(*texts: str, part_type: str = "output_text") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": part_type, "text": text} for text in texts],
    }


def _response(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "resp_abc",
        "object": "response",
        "model": "gpt-4.1",
        "created_at": 1_700_000_000,
        "status": "completed",
        "output": [_message("hello")],
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
    base.update(overrides)
    return base


# ── parse_responses_response: well-formed ───────────────────────────────────


def test_message_output_text_populates_every_field() -> None:
    parsed = parse_responses_response(_json(_response()))

    assert parsed is not None
    assert parsed.id == "resp_abc"
    assert parsed.model == "gpt-4.1"
    assert parsed.created == 1_700_000_000.0
    assert parsed.content == "hello"
    assert parsed.tool_calls == ()
    assert parsed.prompt_tokens == 11
    assert parsed.completion_tokens == 4
    assert parsed.reasoning_content is None
    assert parsed.reasoning_field is None
    assert parsed.finish_reason == "completed"
    assert parsed.stop_reason is None
    assert parsed.choice_index == 0
    assert parsed.choice_count == 1


def test_multiple_output_text_parts_are_concatenated_in_order() -> None:
    body = _json(_response(output=[_message("a", "b"), _message("c")]))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == "abc"


def test_plain_text_part_type_is_treated_as_content() -> None:
    body = _json(_response(output=[_message("legacy", part_type="text")]))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == "legacy"


@pytest.mark.parametrize("part_type", ["reasoning_text", "summary_text"])
def test_reasoning_parts_populate_reasoning_content(part_type: str) -> None:
    body = _json(
        _response(output=[_message("think ", "harder", part_type=part_type)])
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content is None
    assert parsed.reasoning_content == "think harder"
    # The Responses API has no ``reasoning_content`` field; the parser
    # deliberately normalizes onto the chat-completions name so downstream
    # capture (capture.py:353) recognizes it.
    assert parsed.reasoning_field == "reasoning_content"


def test_reasoning_item_summary_list_is_captured() -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [
                        {"type": "summary_text", "text": "step one. "},
                        {"type": "summary_text", "text": "step two."},
                    ],
                }
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content is None
    assert parsed.reasoning_content == "step one. step two."
    assert parsed.reasoning_field == "reasoning_content"


@pytest.mark.parametrize("item_type", ["function_call", "custom_tool_call"])
def test_tool_call_items_are_captured_with_call_id(item_type: str) -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": item_type,
                    "call_id": "call_9",
                    "id": "fc_9",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content is None
    assert parsed.tool_calls == (
        {
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            "id": "call_9",
        },
    )


def test_tool_call_falls_back_to_id_when_call_id_missing() -> None:
    body = _json(
        _response(
            output=[
                {"type": "function_call", "id": "fc_1", "name": "f", "arguments": "{}"}
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.tool_calls[0]["id"] == "fc_1"


def test_tool_call_without_any_id_omits_the_id_key() -> None:
    body = _json(
        _response(output=[{"type": "function_call", "name": "f", "arguments": "{}"}])
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.tool_calls == (
        {"type": "function", "function": {"name": "f", "arguments": "{}"}},
    )
    assert "id" not in parsed.tool_calls[0]


@pytest.mark.parametrize("call_id", ["", None, False])
def test_tool_call_with_falsy_call_id_falls_back_to_id(call_id: Any) -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "id": "fc_fallback",
                    "name": "f",
                    "arguments": "{}",
                }
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.tool_calls[0]["id"] == "fc_fallback"


def test_a_truthy_non_string_call_id_still_falls_back_to_id() -> None:
    """Regression: ``call_id`` must be a *usable* string before it wins.

    ``call_id or id`` short-circuited on any truthy value, so a numeric call_id
    both won the fallback and then failed the str check, leaving the tool call
    with no id at all and silently breaking correlation with its matching
    ``function_call_output``.
    """
    body = _json(
        _response(
            output=[
                {
                    "type": "function_call",
                    "call_id": 17,
                    "id": "fc_fallback",
                    "name": "f",
                    "arguments": "{}",
                }
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.tool_calls[0]["id"] == "fc_fallback"


def test_incomplete_details_reason_becomes_stop_reason() -> None:
    body = _json(
        _response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.finish_reason == "incomplete"
    assert parsed.stop_reason == "max_output_tokens"


@pytest.mark.parametrize(
    "incomplete_details",
    [None, "max_output_tokens", {"reason": 4}, {}, []],
)
def test_unusable_incomplete_details_leave_stop_reason_none(
    incomplete_details: Any,
) -> None:
    body = _json(_response(incomplete_details=incomplete_details))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.stop_reason is None


def test_mixed_output_items_are_all_captured() -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "plan;"}],
                },
                _message("visible "),
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                },
                _message("tail", part_type="reasoning_text"),
                {
                    "type": "custom_tool_call",
                    "call_id": "call_2",
                    "name": "exec",
                    "arguments": "ls",
                },
                {"type": "web_search_call", "status": "completed"},
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == "visible "
    assert parsed.reasoning_content == "plan;tail"
    assert [call["id"] for call in parsed.tool_calls] == ["call_1", "call_2"]


def test_bytearray_body_is_accepted() -> None:
    parsed = parse_responses_response(bytearray(_json(_response())))

    assert parsed is not None
    assert parsed.content == "hello"


def test_large_output_assembles_without_truncation() -> None:
    parts = [f"{index}," for index in range(3000)]
    body = _json(_response(output=[_message(*parts)]))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == "".join(parts)
    assert len(parsed.content) == sum(len(part) for part in parts)


# ── parse_responses_response: usage, created_at, identity coercion ──────────


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 0, "output_tokens": 0}, (0, 0)),
        ({"input_tokens": -1, "output_tokens": 5}, (None, 5)),
        # isinstance(True, int) is True, so bools are rejected explicitly.
        ({"input_tokens": True, "output_tokens": 5}, (None, 5)),
        ({"input_tokens": 1.0, "output_tokens": 5}, (None, 5)),
        ({"input_tokens": "8", "output_tokens": 5}, (None, 5)),
        ({"output_tokens": 5}, (None, 5)),
        ({}, (None, None)),
        (None, (None, None)),
        ("usage", (None, None)),
        ([("input_tokens", 3)], (None, None)),
    ],
)
def test_usage_coercion(usage: Any, expected: tuple[int | None, int | None]) -> None:
    body = _json(_response(usage=usage))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert (parsed.prompt_tokens, parsed.completion_tokens) == expected


def test_missing_usage_key_yields_no_token_counts() -> None:
    payload = _response()
    del payload["usage"]

    parsed = parse_responses_response(_json(payload))

    assert parsed is not None
    assert parsed.prompt_tokens is None
    assert parsed.completion_tokens is None


@pytest.mark.parametrize(
    ("created_at", "expected"),
    [
        (0, 0.0),
        (1_700_000_000, 1_700_000_000.0),
        (1.5, 1.5),
        (-1, None),
        # bool is an int subclass; the guard at responses.py:227 excludes it.
        (True, None),
        (False, None),
        ("1700000000", None),
        (None, None),
    ],
)
def test_created_at_coercion(created_at: Any, expected: float | None) -> None:
    body = _json(_response(created_at=created_at))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.created == expected


@pytest.mark.parametrize("value", ["", None, 5, [], {}])
def test_non_string_or_empty_id_and_model_become_none(value: Any) -> None:
    body = _json(_response(id=value, model=value))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.id is None
    assert parsed.model is None


@pytest.mark.parametrize("status", [None, 5, [], {"state": "completed"}])
def test_non_string_status_becomes_no_finish_reason(status: Any) -> None:
    body = _json(_response(status=status))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.finish_reason is None


def test_empty_string_status_is_kept_as_finish_reason() -> None:
    # Unlike id/model, an empty status is not normalized away.
    parsed = parse_responses_response(_json(_response(status="")))

    assert parsed is not None
    assert parsed.finish_reason == ""


# ── parse_responses_response: malformed input ───────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"   ",
        b"\n\n",
        b"{",
        b"not json",
        b"[]",
        b'["resp"]',
        b'"resp"',
        b"7",
        b"null",
        b"true",
    ],
)
def test_unparseable_or_non_object_bodies_return_none(body: bytes) -> None:
    assert parse_responses_response(body) is None


def test_invalid_utf8_body_returns_none() -> None:
    # json.loads raises UnicodeDecodeError, not JSONDecodeError, for these bytes.
    assert parse_responses_response(b'{"id": "\xff\xfe"}') is None


@pytest.mark.parametrize(
    "output",
    [None, [], "hello", {}, {"0": "hello"}, 7],
)
def test_missing_or_empty_output_returns_none(output: Any) -> None:
    assert parse_responses_response(_json(_response(output=output))) is None


def test_output_key_absent_returns_none() -> None:
    payload = _response()
    del payload["output"]

    assert parse_responses_response(_json(payload)) is None


def test_output_items_that_are_not_mappings_return_none() -> None:
    body = _json(_response(output=["message", 7, None, ["message"]]))

    assert parse_responses_response(body) is None


@pytest.mark.parametrize("content", [None, "hello", 7, {"type": "output_text"}])
def test_message_item_with_non_list_content_returns_none(content: Any) -> None:
    body = _json(
        _response(output=[{"type": "message", "role": "assistant", "content": content}])
    )

    assert parse_responses_response(body) is None


def test_content_parts_that_are_not_mappings_are_skipped() -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": "message",
                    "content": [
                        "raw",
                        None,
                        ["nested"],
                        {"type": "output_text", "text": "kept"},
                    ],
                }
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == "kept"


@pytest.mark.parametrize("text", [None, 7, 1.5, True, ["hi"], {"value": "hi"}])
def test_output_text_part_with_non_string_text_returns_none(text: Any) -> None:
    body = _json(
        _response(output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    )

    assert parse_responses_response(body) is None


def test_unknown_content_part_types_are_ignored() -> None:
    body = _json(
        _response(
            output=[
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "no"},
                        {"type": "output_audio", "text": "spoken"},
                        {"text": "typeless"},
                    ],
                }
            ]
        )
    )

    assert parse_responses_response(body) is None


def test_function_call_with_non_string_arguments_is_dropped_with_a_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: the drop stays, the silence does not.

    ``arguments`` must be an OpenAI JSON *string*; a server that pre-decodes it
    leaves nothing that can be restored to the bytes the model emitted, so
    re-serializing would fabricate training text. The item is still dropped —
    and when it is the only output the exchange is still discarded — but the
    reason is now logged instead of surfacing downstream as an unexplained
    "capture response could not be reconstructed".
    """
    body = _json(
        _response(
            output=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": {"city": "Paris"},
                }
            ]
        )
    )

    with caplog.at_level(logging.WARNING, logger="speedlm.gateway.responses"):
        assert parse_responses_response(body) is None

    assert "dropping unusable /v1/responses function_call output item" in caplog.text
    assert "arguments type=dict" in caplog.text


def test_function_call_with_non_string_name_is_dropped_but_siblings_survive() -> None:
    body = _json(
        _response(
            output=[
                {"type": "function_call", "name": None, "arguments": "{}"},
                _message("kept"),
            ]
        )
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.tool_calls == ()
    assert parsed.content == "kept"


@pytest.mark.parametrize(
    "summary",
    [None, "plan", {}, ["plan"], [None], [{"text": 7}], [{"summary": "plan"}], []],
)
def test_reasoning_item_with_unusable_summary_returns_none(summary: Any) -> None:
    body = _json(_response(output=[{"type": "reasoning", "summary": summary}]))

    assert parse_responses_response(body) is None


def test_reasoning_summary_text_without_part_type_is_still_captured() -> None:
    # The reasoning branch checks only for a str ``text``, unlike message parts.
    body = _json(
        _response(output=[{"type": "reasoning", "summary": [{"text": "untyped"}]}])
    )

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.reasoning_content == "untyped"


def test_output_present_but_nothing_recognisable_returns_none() -> None:
    body = _json(
        _response(
            output=[
                {"type": "web_search_call", "status": "completed"},
                {"type": "file_search_call", "queries": ["x"]},
                {"role": "assistant"},
                {},
            ]
        )
    )

    assert parse_responses_response(body) is None


def test_empty_message_content_list_returns_none() -> None:
    body = _json(_response(output=[{"type": "message", "content": []}]))

    assert parse_responses_response(body) is None


def test_empty_output_text_string_is_recognised_as_output() -> None:
    # One empty part makes content_parts truthy, so content is "" rather than None.
    body = _json(_response(output=[_message("")]))

    parsed = parse_responses_response(body)

    assert parsed is not None
    assert parsed.content == ""
    assert parsed.finish_reason == "completed"


# ── parse_responses_sse ─────────────────────────────────────────────────────


def test_realistic_stream_parses_the_terminal_response_object() -> None:
    body = _sse(
        _event("response.created", {"id": "resp_1", "status": "in_progress", "output": []}),
        json.dumps({"type": "response.output_text.delta", "delta": "he"}),
        json.dumps({"type": "response.output_text.delta", "delta": "llo"}),
        _event(
            "response.completed",
            _response(id="resp_1", output=[_message("he", "llo")]),
        ),
    )

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.id == "resp_1"
    # The deltas are ignored; only the terminal snapshot is trusted.
    assert parsed.content == "hello"
    assert parsed.prompt_tokens == 11
    assert parsed.completion_tokens == 4
    assert parsed.finish_reason == "completed"


@pytest.mark.parametrize(
    ("event_type", "status"),
    [
        ("response.completed", "completed"),
        ("response.incomplete", "incomplete"),
        ("response.failed", "failed"),
    ],
)
def test_every_terminal_event_type_is_recovered(event_type: str, status: str) -> None:
    body = _sse(_event(event_type, _response(status=status)))

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.finish_reason == status


def test_non_terminal_event_types_are_not_parsed() -> None:
    body = _sse(_event("response.in_progress", _response()))

    assert parse_responses_sse(body) is None


def test_last_terminal_event_wins() -> None:
    body = _sse(
        _event("response.completed", _response(id="resp_first", output=[_message("first")])),
        _event("response.completed", _response(id="resp_last", output=[_message("last")])),
    )

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.id == "resp_last"
    assert parsed.content == "last"


def test_terminal_event_before_deltas_is_still_recovered() -> None:
    body = _sse(
        _event("response.completed", _response(output=[_message("early")])),
        json.dumps({"type": "response.output_text.delta", "delta": "late"}),
    )

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "early"


def test_done_sentinel_is_ignored() -> None:
    body = _sse(_event("response.completed", _response()), "[DONE]")

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


def test_done_sentinel_alone_returns_none() -> None:
    assert parse_responses_sse(b"data: [DONE]\n\n") is None


def test_comment_lines_and_event_fields_are_ignored() -> None:
    body = (
        b": keep-alive ping\n"
        b"event: response.completed\n"
        b"id: 42\n"
        b"retry: 1000\n"
        b"data: " + _event("response.completed", _response()).encode() + b"\n\n"
    )

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


def test_multiline_data_fields_are_folded_into_one_payload() -> None:
    pretty = json.dumps(
        {"type": "response.completed", "response": _response()},
        indent=2,
    )
    lines = pretty.split("\n")
    assert len(lines) > 1
    body = "".join(f"data: {line}\n" for line in lines).encode() + b"\n"

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"
    assert parsed.id == "resp_abc"


def test_data_field_without_a_leading_space_is_parsed() -> None:
    body = b"data:" + _event("response.completed", _response()).encode() + b"\n\n"

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


def test_final_event_without_a_trailing_blank_line_is_flushed() -> None:
    body = b"data: " + _event("response.completed", _response()).encode()

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


def test_crlf_line_endings_are_supported() -> None:
    body = b"data: " + _event("response.completed", _response()).encode() + b"\r\n\r\n"

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


def test_bare_carriage_return_separators_are_supported() -> None:
    # bytes.splitlines also splits on a lone \r, unlike the SSEAssembler.
    body = b"data: " + _event("response.completed", _response()).encode() + b"\r\r"

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


@pytest.mark.parametrize("body", [b"", b"\n\n", b": only a comment\n\n", b"event: x\n\n"])
def test_streams_with_no_data_events_return_none(body: bytes) -> None:
    assert parse_responses_sse(body) is None


@pytest.mark.parametrize(
    "response",
    [None, "resp_1", 7, ["resp_1"]],
)
def test_terminal_event_without_a_mapping_response_returns_none(response: Any) -> None:
    body = _sse(_event("response.completed", response))

    assert parse_responses_sse(body) is None


def test_truncated_terminal_event_returns_none() -> None:
    body = b'data: {"type": "response.completed", "response": {"id": "resp_1"'

    assert parse_responses_sse(body) is None


def test_malformed_event_after_a_valid_terminal_keeps_the_terminal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: a truncated trailing frame no longer voids the exchange.

    A terminal event carries the complete response snapshot, so an undecodable
    neighbour cannot make it partial. Recovering it is strictly better than
    losing a whole exchange to a connection dropped one byte into the trailing
    frame — and the skipped frame is logged so the damage is not silent.
    """
    good = _event("response.completed", _response())
    body = _sse(good) + b'data: {"type": "response.out\n\n'

    with caplog.at_level(logging.WARNING, logger="speedlm.gateway.responses"):
        parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"
    assert "skipped 1 unusable" in caplog.text
    assert "recovered" in caplog.text
    # The same stream without the trailing garbage parses identically.
    assert parse_responses_sse(_sse(good)) == parsed


def test_undecodable_events_without_a_terminal_still_return_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _sse(_event("response.output_text.delta", None)) + b'data: {"type": "response.out\n\n'

    with caplog.at_level(logging.WARNING, logger="speedlm.gateway.responses"):
        assert parse_responses_sse(body) is None

    assert "not recovered" in caplog.text


@pytest.mark.parametrize("event", ["[]", "[1,2]", '"text"', "7", "null", "true"])
def test_non_mapping_event_payloads_are_skipped_not_fatal(event: str) -> None:
    """Regression: a stray ``data: [1,2]`` after a good terminal is not fatal."""
    body = _sse(_event("response.completed", _response()), event)

    parsed = parse_responses_sse(body)

    assert parsed is not None
    assert parsed.content == "hello"


@pytest.mark.parametrize("event", ["[1,2]", '"text"', "7"])
def test_non_mapping_event_payloads_alone_still_return_none(event: str) -> None:
    assert parse_responses_sse(_sse(event)) is None


def test_sse_invalid_utf8_returns_none() -> None:
    assert parse_responses_sse(b'data: {"type": "\xff\xfe"}\n\n') is None


def test_sse_accepts_a_bytearray_body() -> None:
    parsed = parse_responses_sse(bytearray(_sse(_event("response.completed", _response()))))

    assert parsed is not None
    assert parsed.content == "hello"


def test_terminal_response_that_fails_payload_parsing_returns_none() -> None:
    body = _sse(_event("response.completed", _response(output=[])))

    assert parse_responses_sse(body) is None


# ── response_status_and_id ──────────────────────────────────────────────────


def test_json_body_reports_id_and_status() -> None:
    body = _json({"id": "resp_bg", "status": "queued", "output": []})

    assert response_status_and_id(body, "application/json") == ("resp_bg", "queued")


def test_json_body_accepts_a_bytearray() -> None:
    body = bytearray(_json({"id": "resp_bg", "status": "in_progress"}))

    assert response_status_and_id(body, "application/json") == ("resp_bg", "in_progress")


def test_sse_body_reads_the_nested_response_object() -> None:
    body = _sse(
        _event("response.created", {"id": "resp_bg", "status": "queued"}),
        _event("response.in_progress", {"id": "resp_bg", "status": "in_progress"}),
    )

    assert response_status_and_id(body, "text/event-stream") == ("resp_bg", "in_progress")


def test_sse_body_falls_back_to_the_top_level_payload() -> None:
    body = _sse(json.dumps({"id": "resp_bg", "status": "queued"}))

    assert response_status_and_id(body, "text/event-stream") == ("resp_bg", "queued")


def test_sse_last_event_without_a_response_object_keeps_the_announced_id() -> None:
    """Regression: a trailing delta must not erase the announced identity.

    The loop kept the LAST event unconditionally instead of the last one
    carrying identity, so a background stream truncated after a delta reported
    ``(None, None)`` — which is exactly the pair capture.py uses to register
    ``_pending_responses``, silently dropping background correlation.
    """
    body = _sse(
        _event("response.created", {"id": "resp_bg", "status": "queued"}),
        json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
    )

    assert response_status_and_id(body, "text/event-stream") == ("resp_bg", "queued")


@pytest.mark.parametrize(
    "content_type",
    [
        "text/event-stream",
        "text/event-stream; charset=utf-8",
        "TEXT/EVENT-STREAM",
        "Text/Event-Stream; charset=UTF-8",
    ],
)
def test_sse_content_type_variations_are_recognised(content_type: str) -> None:
    body = _sse(_event("response.created", {"id": "resp_bg", "status": "queued"}))

    assert response_status_and_id(body, content_type) == ("resp_bg", "queued")


def test_sse_body_under_a_json_content_type_returns_nothing() -> None:
    body = _sse(_event("response.created", {"id": "resp_bg", "status": "queued"}))

    assert response_status_and_id(body, "application/json") == (None, None)


def test_sse_done_only_stream_returns_nothing() -> None:
    assert response_status_and_id(b"data: [DONE]\n\n", "text/event-stream") == (None, None)


def test_sse_stream_with_no_events_returns_nothing() -> None:
    assert response_status_and_id(b"", "text/event-stream") == (None, None)


def test_sse_malformed_event_returns_nothing() -> None:
    body = _sse(_event("response.created", {"id": "resp_bg", "status": "queued"})) + b"data: {\n\n"

    assert response_status_and_id(body, "text/event-stream") == (None, None)


def test_sse_non_mapping_event_is_ignored_and_leaves_the_prior_payload() -> None:
    body = _sse(_event("response.created", {"id": "resp_bg", "status": "queued"}), "[1,2]")

    assert response_status_and_id(body, "text/event-stream") == ("resp_bg", "queued")


@pytest.mark.parametrize("body", [b"", b"   ", b"{", b"[]", b'"resp"', b"7", b"null"])
def test_unparseable_or_non_object_json_bodies_return_nothing(body: bytes) -> None:
    assert response_status_and_id(body, "application/json") == (None, None)


def test_invalid_utf8_json_body_returns_nothing() -> None:
    assert response_status_and_id(b'{"id": "\xff"}', "application/json") == (None, None)


@pytest.mark.parametrize("value", ["", None, 7, [], {}, True])
def test_non_string_or_empty_id_and_status_become_none(value: Any) -> None:
    body = _json({"id": value, "status": value})

    assert response_status_and_id(body, "application/json") == (None, None)


def test_missing_id_and_status_keys_return_nothing() -> None:
    assert response_status_and_id(_json({"object": "response"}), "") == (None, None)


def test_id_and_status_are_reported_independently() -> None:
    assert response_status_and_id(_json({"id": "resp_bg"}), "") == ("resp_bg", None)
    assert response_status_and_id(_json({"status": "queued"}), "") == (None, "queued")


# ── responses_request_messages ──────────────────────────────────────────────


def test_instructions_become_a_system_message() -> None:
    messages = responses_request_messages({"instructions": "Be terse."})

    assert messages == [
        {
            "role": "system",
            "content": "Be terse.",
            "provenance_tag": "client_supplied",
        }
    ]


@pytest.mark.parametrize("instructions", ["", None, 7, ["Be terse."], {"text": "x"}])
def test_unusable_instructions_are_skipped(instructions: Any) -> None:
    messages = responses_request_messages({"instructions": instructions, "input": "hi"})

    assert messages == [
        {"role": "user", "content": "hi", "provenance_tag": "client_supplied"}
    ]


def test_string_input_yields_a_user_message_and_returns_early() -> None:
    messages = responses_request_messages({"instructions": "Be terse.", "input": "hi"})

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": "hi",
        "provenance_tag": "client_supplied",
    }


def test_empty_string_input_still_yields_a_user_message() -> None:
    messages = responses_request_messages({"input": ""})

    assert messages == [{"role": "user", "content": "", "provenance_tag": "client_supplied"}]


@pytest.mark.parametrize("request_input", [None, 7, {"role": "user"}, True])
def test_input_that_is_neither_string_nor_list_yields_instructions_only(
    request_input: Any,
) -> None:
    messages = responses_request_messages(
        {"instructions": "Be terse.", "input": request_input}
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_request_without_input_or_instructions_yields_no_messages() -> None:
    assert responses_request_messages({"model": "gpt-4.1"}) == []


def test_message_items_are_converted_with_and_without_an_explicit_type() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {"type": "message", "role": "user", "content": "typed"},
                {"role": "assistant", "content": "untyped"},
            ]
        }
    )

    assert messages == [
        {
            "type": "message",
            "role": "user",
            "content": "typed",
            "provenance_tag": "client_supplied",
        },
        {
            "role": "assistant",
            "content": "untyped",
            "provenance_tag": "client_supplied",
        },
    ]


def test_message_item_extra_fields_are_preserved() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                    "id": "msg_1",
                    "status": "completed",
                }
            ]
        }
    )

    assert messages[0]["id"] == "msg_1"
    assert messages[0]["status"] == "completed"
    assert messages[0]["content"] == [{"type": "input_text", "text": "hi"}]


def test_message_item_without_a_usable_role_is_skipped() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {"type": "message", "content": "no role"},
                {"type": "message", "role": "", "content": "empty role"},
                {"type": "message", "role": 7, "content": "numeric role"},
            ]
        }
    )

    assert messages == []


def test_function_call_output_becomes_a_tool_message() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temp":21}',
                }
            ]
        }
    )

    assert messages == [
        {
            "role": "tool",
            "content": '{"temp":21}',
            "tool_call_id": "call_1",
            "provenance_tag": "client_supplied",
        }
    ]


def test_function_call_output_without_a_call_id_keeps_a_null_tool_call_id() -> None:
    messages = responses_request_messages(
        {"input": [{"type": "function_call_output", "output": "done"}]}
    )

    assert messages[0]["tool_call_id"] is None
    assert messages[0]["content"] == "done"


def test_function_call_becomes_an_assistant_tool_call() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "fc_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ]
        }
    )

    assert messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
            "provenance_tag": "client_supplied",
        }
    ]


def test_function_call_falls_back_to_id_when_call_id_is_absent_or_empty() -> None:
    messages = responses_request_messages(
        {
            "input": [
                {"type": "function_call", "id": "fc_1", "name": "f", "arguments": "{}"},
                {
                    "type": "function_call",
                    "call_id": "",
                    "id": "fc_2",
                    "name": "f",
                    "arguments": "{}",
                },
            ]
        }
    )

    assert [message["tool_calls"][0]["id"] for message in messages] == ["fc_1", "fc_2"]


def test_function_call_without_any_id_records_none() -> None:
    messages = responses_request_messages(
        {"input": [{"type": "function_call", "name": "f", "arguments": "{}"}]}
    )

    assert messages[0]["tool_calls"][0]["id"] is None


def test_function_call_missing_name_and_arguments_default_to_empty_strings() -> None:
    messages = responses_request_messages(
        {"input": [{"type": "function_call", "call_id": "call_1"}]}
    )

    assert messages[0]["tool_calls"][0]["function"] == {"name": "", "arguments": ""}


def test_non_mapping_input_items_are_skipped() -> None:
    messages = responses_request_messages(
        {"input": ["hello", None, 7, ["nested"], {"role": "user", "content": "kept"}]}
    )

    assert len(messages) == 1
    assert messages[0]["content"] == "kept"


@pytest.mark.parametrize(
    "item",
    [
        {"type": "item_reference", "id": "msg_1"},
        {"type": "reasoning", "summary": []},
        {"type": "computer_call", "call_id": "call_1"},
        {"type": "function_call_output_v2", "output": "x"},
        {"type": 7, "role": "user", "content": "numeric type"},
        {},
    ],
)
def test_unknown_item_types_are_skipped(item: dict[str, Any]) -> None:
    assert responses_request_messages({"input": [item]}) == []


def test_full_tool_round_trip_conversation_is_reconstructed_in_order() -> None:
    messages = responses_request_messages(
        {
            "instructions": "You are helpful.",
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "21C"},
                {"type": "web_search_call", "id": "ws_1"},
            ],
        }
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[3]["tool_call_id"] == messages[2]["tool_calls"][0]["id"]


# ── capture dispatch integration (capture.py:447-448) ───────────────────────


def test_capture_dispatch_routes_json_responses_bodies() -> None:
    parsed = _decode_response_body(
        _json(_response()),
        endpoint="/v1/responses",
        content_type="application/json",
    )

    assert parsed is not None
    assert parsed.id == "resp_abc"
    assert parsed.content == "hello"
    assert parsed.completion_tokens == 4


def test_capture_dispatch_routes_sse_responses_bodies() -> None:
    body = _sse(
        _event("response.created", {"id": "resp_1", "status": "in_progress"}),
        _event(
            "response.completed",
            _response(id="resp_1", output=[_message("streamed")]),
        ),
        "[DONE]",
    )

    parsed = _decode_response_body(
        body,
        endpoint="/v1/responses",
        content_type="text/event-stream; charset=utf-8",
    )

    assert parsed is not None
    assert parsed.id == "resp_1"
    assert parsed.content == "streamed"


def test_capture_dispatch_treats_content_type_case_insensitively() -> None:
    body = _sse(_event("response.completed", _response()))

    parsed = _decode_response_body(
        body,
        endpoint="/v1/responses",
        content_type="TEXT/EVENT-STREAM",
    )

    assert parsed is not None
    assert parsed.content == "hello"


def test_capture_dispatch_returns_none_for_unusable_responses_bodies() -> None:
    assert (
        _decode_response_body(
            b"not json",
            endpoint="/v1/responses",
            content_type="application/json",
        )
        is None
    )
    assert (
        _decode_response_body(
            b"data: {\n\n",
            endpoint="/v1/responses",
            content_type="text/event-stream",
        )
        is None
    )
