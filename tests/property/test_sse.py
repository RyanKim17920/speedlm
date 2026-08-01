from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from speedlm.gateway.sse import (
    AssembledResponse,
    SSEAssembler,
    parse_json_response,
)

CHAT_ENDPOINT = "/v1/chat/completions"


def _wire_event(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _assemble(wire: bytes, chunk_sizes: list[int]) -> tuple[AssembledResponse, bool, bool]:
    assembler = SSEAssembler(CHAT_ENDPOINT)
    offset = 0
    for size in chunk_sizes:
        next_offset = min(len(wire), offset + size)
        assembler.feed(wire[offset:next_offset])
        offset = next_offset
    assembler.feed(wire[offset:])
    response = assembler.finish()
    return response, assembler.valid, assembler.done


@given(st.lists(st.binary(max_size=80), max_size=20))
@settings(max_examples=100, deadline=None)
def test_assembler_never_raises_on_arbitrary_bytes(chunks: list[bytes]) -> None:
    assembler = SSEAssembler(CHAT_ENDPOINT)
    for chunk in chunks:
        assembler.feed(chunk)
    assert isinstance(assembler.finish(), AssembledResponse)


@given(st.binary(max_size=500), st.sampled_from([CHAT_ENDPOINT, "/v1/completions"]))
@settings(max_examples=80, deadline=None)
def test_non_streaming_parser_never_raises_on_arbitrary_bytes(
    body: bytes,
    endpoint: str,
) -> None:
    result = parse_json_response(body, endpoint)
    assert result is None or isinstance(result, AssembledResponse)


@given(st.binary(max_size=500), st.lists(st.integers(min_value=1, max_value=30), max_size=40))
@settings(max_examples=100, deadline=None)
def test_arbitrary_stream_is_chunk_boundary_independent(
    wire: bytes,
    chunk_sizes: list[int],
) -> None:
    assert _assemble(wire, chunk_sizes) == _assemble(wire, [len(wire)])


@given(
    st.lists(st.text(max_size=30), min_size=1, max_size=8),
    st.lists(st.integers(min_value=1, max_value=30), max_size=40),
)
@settings(max_examples=70, deadline=None)
def test_chunk_boundaries_do_not_change_assembly(
    content_parts: list[str],
    chunk_sizes: list[int],
) -> None:
    events = [
        _wire_event(
            {
                "id": "stream-id",
                "model": "stream-model",
                "created": 1_700_000_000,
                "choices": [{"index": 0, "delta": {"content": part}}],
            }
        )
        for part in content_parts
    ]
    events.append(
        _wire_event(
            {
                "choices": [],
                "usage": {"prompt_tokens": 17, "completion_tokens": 9},
            }
        )
    )
    wire = b"".join(events) + b"data: [DONE]\n\n"

    whole = _assemble(wire, [len(wire)])
    chunked = _assemble(wire, chunk_sizes)

    assert chunked == whole
    response, valid, done = chunked
    assert valid
    assert done
    assert response.content == "".join(content_parts)
    assert response.prompt_tokens == 17
    assert response.completion_tokens == 9


@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20),
    st.lists(st.text(max_size=20), min_size=1, max_size=6),
    st.lists(st.integers(min_value=1, max_value=25), max_size=30),
)
@settings(max_examples=60, deadline=None)
def test_tool_call_fragments_are_assembled_in_order(
    name: str,
    argument_parts: list[str],
    chunk_sizes: list[int],
) -> None:
    events: list[bytes] = []
    for index, arguments in enumerate(argument_parts):
        raw_call: dict[str, Any] = {
            "index": 0,
            "function": {"arguments": arguments},
        }
        if index == 0:
            raw_call.update({"id": "call-1", "type": "function"})
            function = raw_call["function"]
            assert isinstance(function, dict)
            function["name"] = name
        events.append(
            _wire_event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "", "tool_calls": [raw_call]},
                        }
                    ]
                }
            )
        )
    wire = b"".join(events) + b"data: [DONE]\n\n"

    response, valid, done = _assemble(wire, chunk_sizes)

    assert valid
    assert done
    assert response.tool_calls == (
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": name, "arguments": "".join(argument_parts)},
        },
    )


@given(st.text(max_size=100))
@settings(max_examples=50, deadline=None)
def test_truncated_json_event_is_marked_invalid(content: str) -> None:
    payload = {
        "choices": [{"index": 0, "delta": {"content": content}}],
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assembler = SSEAssembler(CHAT_ENDPOINT)

    assembler.feed(b"data: " + encoded[:-1])
    response = assembler.finish()

    assert not assembler.valid
    assert response.content is None


@given(
    st.text(max_size=100),
    st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=80, deadline=None)
def test_any_proper_stream_prefix_never_claims_completion(
    content: str,
    raw_cutoff: int,
) -> None:
    wire = _wire_event(
        {"choices": [{"index": 0, "delta": {"content": content}}]}
    ) + b"data: [DONE]\n\n"
    done_marker = b"data: [DONE]\n\n"
    cutoff = raw_cutoff % (len(wire) - len(done_marker) + 1)
    response, _, done = _assemble(wire[:cutoff], [1, 2, 3, 5, 8])

    assert not done
    assert content.startswith(response.content or "")


def test_deeply_nested_json_never_escapes_feed() -> None:
    depth = 10_000
    malformed = b"data: " + b"[" * depth + b"0" + b"]" * depth + b"\n\n"
    assembler = SSEAssembler(CHAT_ENDPOINT)

    assembler.feed(malformed)
    assembler.finish()
