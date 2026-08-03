"""Sampling provenance and capture-loss regressions.

The gate scores the draft head on greedy (temperature 0) argmax agreement, so a
row's temperature is only meaningful together with a statement of where that
number came from. These tests pin the two halves of that: the gateway must not
turn an omitted temperature into a fabricated 0.0, and the recorded set must not
silently drop fields that changed what was generated.

Every assertion here is written to fail if the corresponding production code is
reverted; see the mutation notes in the module docstring of each class.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from speedlm.config import SamplingConfig
from speedlm.gateway.capture import (
    SERVER_DEFAULTS_ORIGIN,
    VLLM_ENGINE_SAMPLING_DEFAULTS,
    CaptureManager,
    sampling_provenance,
)
from speedlm.traces.normalize import (
    SAMPLING_PROVENANCE_KEY,
    SAMPLING_SOURCE_CLIENT,
    SAMPLING_SOURCE_RECORD,
    SAMPLING_SOURCE_SERVER_DEFAULT,
    SAMPLING_SOURCE_UNKNOWN,
    normalize_record,
)
from speedlm.traces.store import TraceRecord, TraceStore, estimate_message_tokens


@pytest.fixture(autouse=True)
def _immediate_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def immediate(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _chat_response(content: str = "hi") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "model",
        "created": 1_700_000_000,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def _capture_chat(tmp_path: Path, request: dict[str, Any]) -> TraceRecord:
    """Run one chat exchange through the real capture path and return the row."""
    async def scenario() -> TraceRecord:
        store = TraceStore(tmp_path / "traces.jsonl")
        capture = CaptureManager(store)
        adapter = capture.match("POST", "/v1/chat/completions")
        assert adapter is not None
        capture.submit_exchange(
            _json_bytes(request),
            _json_bytes(_chat_response()),
            adapter=adapter,
            request_path="/v1/chat/completions",
            method="POST",
            content_type="application/json",
            content_encoding="",
            timestamp=1_700_000_000.0,
        )
        await capture.drain()
        records = list(store.iter_records())
        assert store.stats().total_dropped == 0
        assert len(records) == 1
        return records[0]

    return asyncio.run(scenario())


def _provenance(record: TraceRecord) -> dict[str, Any]:
    for message in reversed(record.messages):
        block = message.get(SAMPLING_PROVENANCE_KEY)
        if isinstance(block, dict):
            return block
    raise AssertionError("record carries no sampling provenance")


_BASE_REQUEST: dict[str, Any] = {
    "model": "model",
    "messages": [{"role": "user", "content": "hello"}],
}


# ── Fabricated greedy ───────────────────────────────────────────────────────


class TestOmittedTemperature:
    """Mutations that must turn these red:

    - capture.py `_build_raw_record`: read the three fields back from
      `request_data` instead of `sampling_params` (the pre-fix behaviour).
    - capture.py: set VLLM_ENGINE_SAMPLING_DEFAULTS["temperature"] = 0.0.
    - capture.py `_build_raw_record`: stop attaching the provenance block.
    """

    def test_absent_temperature_records_the_engine_default_not_zero(
        self, tmp_path: Path
    ) -> None:
        record = _capture_chat(tmp_path, dict(_BASE_REQUEST))
        # vLLM 0.25.1 substitutes 1.0, i.e. full sampling — never greedy.
        assert record.temperature == 1.0
        block = _provenance(record)
        assert block["sources"]["temperature"] == SAMPLING_SOURCE_SERVER_DEFAULT
        assert block["server_defaults_origin"] == SERVER_DEFAULTS_ORIGIN

    def test_explicit_greedy_is_labelled_as_the_client_s_own_choice(
        self, tmp_path: Path
    ) -> None:
        record = _capture_chat(tmp_path, {**_BASE_REQUEST, "temperature": 0.0})
        assert record.temperature == 0.0
        assert _provenance(record)["sources"]["temperature"] == SAMPLING_SOURCE_CLIENT

    def test_omitted_and_explicit_greedy_rows_are_distinguishable(
        self, tmp_path: Path
    ) -> None:
        """The defect: both used to serialize to a byte-identical greedy row."""
        omitted = _capture_chat(tmp_path / "a", dict(_BASE_REQUEST))
        explicit = _capture_chat(
            tmp_path / "b", {**_BASE_REQUEST, "temperature": 0.0}
        )
        assert omitted.to_dict() != explicit.to_dict()
        assert (omitted.temperature, _provenance(omitted)["sources"]["temperature"]) != (
            explicit.temperature,
            _provenance(explicit)["sources"]["temperature"],
        )

    def test_server_defaults_are_flagged_as_model_overridable(self) -> None:
        """vLLM's --generation-config auto lets a model override these five."""
        _, block = sampling_provenance(dict(_BASE_REQUEST))
        assert set(block["server_defaults_model_overridable"]) == set(
            VLLM_ENGINE_SAMPLING_DEFAULTS
        )
        assert "temperature" in block["server_defaults_model_overridable"]

    def test_absent_seed_is_unspecified_not_defaulted(self, tmp_path: Path) -> None:
        """There is no engine default for `seed`; claiming one would be a guess."""
        record = _capture_chat(tmp_path, dict(_BASE_REQUEST))
        block = _provenance(record)
        assert "seed" in block["unspecified"]
        assert block["sources"]["seed"] == SAMPLING_SOURCE_UNKNOWN


# ── Widened allowlist ───────────────────────────────────────────────────────


class TestRecordedRequestFields:
    """Mutation: shrink `_RECORDED_REQUEST_FIELDS` back to
    ("temperature", "top_p", "seed") — every test here goes red.
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("stop", ["\nObservation:"]),
            ("top_k", 40),
            ("min_p", 0.05),
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.25),
            ("repetition_penalty", 1.1),
            ("max_tokens", 128),
            ("response_format", {"type": "json_object"}),
            ("tool_choice", "required"),
        ],
    )
    def test_generation_changing_field_is_recorded(
        self, tmp_path: Path, field: str, value: Any
    ) -> None:
        record = _capture_chat(tmp_path, {**_BASE_REQUEST, field: value})
        block = _provenance(record)
        assert block["params"][field] == value
        assert block["sources"][field] == SAMPLING_SOURCE_CLIENT

    def test_stop_string_truncation_is_not_baked_in_as_a_turn_ending(
        self, tmp_path: Path
    ) -> None:
        """A stop-string cut used to be indistinguishable from an EOS ending."""
        record = _capture_chat(
            tmp_path, {**_BASE_REQUEST, "stop": ["\nUser:"], "max_tokens": 16}
        )
        block = _provenance(record)
        assert block["params"]["stop"] == ["\nUser:"]
        assert block["params"]["max_tokens"] == 16

    def test_unrecorded_request_fields_are_named_on_the_row(
        self, tmp_path: Path
    ) -> None:
        """Mutation: drop the `unrecorded_request_fields` entry from the block."""
        record = _capture_chat(
            tmp_path, {**_BASE_REQUEST, "some_future_decoding_knob": 3}
        )
        assert "some_future_decoding_knob" in _provenance(record)[
            "unrecorded_request_fields"
        ]

    def test_structural_fields_are_not_reported_as_losses(
        self, tmp_path: Path
    ) -> None:
        record = _capture_chat(tmp_path, {**_BASE_REQUEST, "stream": False})
        omissions = _provenance(record)["unrecorded_request_fields"]
        assert "messages" not in omissions
        assert "model" not in omissions
        assert "stream" not in omissions


# ── Tool schemas ────────────────────────────────────────────────────────────


_TOOLS = [{
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object"}},
}]


class TestToolsSurviveEveryShape:
    """Mutation: remove "tools" from the copy loops in
    `_request_response_to_internal` / `_proxy_capture_to_internal`.
    """

    def test_request_response_shape_keeps_tools(self) -> None:
        record = normalize_record(
            {
                "request": {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": _TOOLS,
                },
                "response": {"choices": [{
                    "message": {"role": "assistant", "content": "hi"},
                }]},
            },
            defaults=SamplingConfig(),
        )
        assert [dict(tool) for tool in record.tools] == _TOOLS

    def test_proxy_capture_shape_keeps_tools(self) -> None:
        record = normalize_record(
            {
                "event": {
                    "incoming_payload": {"body": {
                        "model": "m",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": _TOOLS,
                    }},
                    "upstream_result": {"data": {"choices": [{
                        "message": {"role": "assistant", "content": "hi"},
                    }]}},
                },
            },
            defaults=SamplingConfig(),
        )
        assert [dict(tool) for tool in record.tools] == _TOOLS

    def test_malformed_nested_tools_do_not_reject_the_record(self) -> None:
        """The proxy shape is recovered by heuristic walk; a stray non-schema
        `tools` value must not turn an acceptable record into a rejection."""
        record = normalize_record(
            {
                "event": {
                    "incoming_payload": {"body": {
                        "model": "m",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": "not-a-schema-list",
                    }},
                    "upstream_result": {"data": {"choices": [{
                        "message": {"role": "assistant", "content": "hi"},
                    }]}},
                },
            },
            defaults=SamplingConfig(),
        )
        assert record.tools == ()


# ── Assistant prefill ───────────────────────────────────────────────────────


class TestAssistantPrefill:
    """Mutation: always `messages.append(assistant)` instead of merging."""

    def test_continued_prefill_is_one_turn(self, tmp_path: Path) -> None:
        record = _capture_chat(tmp_path, {
            "model": "model",
            "messages": [
                {"role": "user", "content": "count"},
                {"role": "assistant", "content": "one two "},
            ],
            "continue_final_message": True,
            "add_generation_prompt": False,
        })
        roles = [message["role"] for message in record.messages]
        assert roles == ["user", "assistant"]
        assert record.messages[-1]["content"] == "one two hi"
        assert record.messages[-1]["prefill_prefix_chars"] == len("one two ")

    def test_trailing_assistant_without_continuation_stays_two_turns(
        self, tmp_path: Path
    ) -> None:
        """With vLLM's default add_generation_prompt=True the template really
        does render a finished turn plus a new one, so merging would be wrong."""
        record = _capture_chat(tmp_path, {
            "model": "model",
            "messages": [
                {"role": "user", "content": "count"},
                {"role": "assistant", "content": "one two"},
            ],
        })
        roles = [message["role"] for message in record.messages]
        assert roles == ["user", "assistant", "assistant"]

    def test_normalized_request_response_prefill_is_one_turn(self) -> None:
        record = normalize_record(
            {
                "request": {
                    "model": "m",
                    "messages": [
                        {"role": "user", "content": "count"},
                        {"role": "assistant", "content": "one two "},
                    ],
                    "continue_final_message": True,
                },
                "response": {"choices": [{
                    "message": {"role": "assistant", "content": "three"},
                }]},
            },
            defaults=SamplingConfig(),
        )
        assert [m["role"] for m in record.messages] == ["user", "assistant"]
        assert record.messages[-1]["content"] == "one two three"


# ── openai-response history loss ────────────────────────────────────────────


class TestOpenAIResponseShape:
    """Mutation: drop the `history_truncated` flag."""

    def test_bare_response_object_is_flagged_as_a_fragment(self) -> None:
        record = normalize_record(
            {
                "id": "chatcmpl-x",
                "model": "m",
                "choices": [{
                    "message": {"role": "assistant", "content": "hi"},
                }],
                "prompt_text": None,
            },
            defaults=SamplingConfig(),
        )
        assert len(record.messages) == 1
        assert record.messages[0]["history_truncated"] is True

    def test_shapes_with_a_prompt_are_not_flagged(self) -> None:
        record = normalize_record(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "choices": [{
                    "message": {"role": "assistant", "content": "hi"},
                }],
            },
            defaults=SamplingConfig(),
        )
        assert all("history_truncated" not in m for m in record.messages)


# ── Normalize-side provenance and compatibility ─────────────────────────────


class TestNormalizeProvenance:
    """Mutation: delete the `_attach_sampling_provenance` call in
    `normalize_record`, or make it claim `client_explicit`.
    """

    def test_backfilled_temperature_is_labelled_unknown(self) -> None:
        record = normalize_record(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            defaults=SamplingConfig(temperature=0.0),
        )
        assert record.temperature == 0.0  # still a number, but...
        sources = record.messages[-1][SAMPLING_PROVENANCE_KEY]["sources"]
        assert sources["temperature"] == SAMPLING_SOURCE_UNKNOWN

    def test_observed_temperature_is_labelled_as_observed(self) -> None:
        record = normalize_record(
            {
                "model": "m",
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            defaults=SamplingConfig(temperature=0.7),
        )
        sources = record.messages[-1][SAMPLING_PROVENANCE_KEY]["sources"]
        assert sources["temperature"] == SAMPLING_SOURCE_RECORD

    def test_capture_time_label_is_not_overwritten_by_normalization(self) -> None:
        """Mutation: let `_attach_sampling_provenance` overwrite existing keys."""
        record = normalize_record(
            {
                "model": "m",
                "temperature": 1.0,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": "hi",
                        SAMPLING_PROVENANCE_KEY: {
                            "sources": {
                                "temperature": SAMPLING_SOURCE_SERVER_DEFAULT
                            },
                        },
                    },
                ],
            },
            defaults=SamplingConfig(),
        )
        sources = record.messages[-1][SAMPLING_PROVENANCE_KEY]["sources"]
        assert sources["temperature"] == SAMPLING_SOURCE_SERVER_DEFAULT

    def test_provenance_is_not_counted_as_generated_text(self) -> None:
        """Mutation: estimate tokens over the raw messages again."""
        data = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        record = normalize_record(data, defaults=SamplingConfig())
        bare_prompt, bare_completion = estimate_message_tokens(
            list(data["messages"])  # type: ignore[arg-type]
        )
        assert (record.prompt_tokens, record.completion_tokens) == (
            bare_prompt,
            bare_completion,
        )


class TestOnDiskCompatibility:
    """Trace files written before this change carry no provenance at all."""

    def test_legacy_record_without_provenance_reads_back(self) -> None:
        legacy = {
            "id": "tr-legacy",
            "timestamp": 1_700_000_000.0,
            "model": "m",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "tool_calls": [],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "prompt_tokens": 4,
            "completion_tokens": 2,
        }
        record = TraceRecord.from_dict(legacy)
        # No claim of provenance is manufactured on read.
        assert all(SAMPLING_PROVENANCE_KEY not in m for m in record.messages)

    def test_new_record_round_trips_through_from_dict(
        self, tmp_path: Path
    ) -> None:
        """Provenance rides on messages precisely so that `from_dict`'s
        unknown-top-level-key rejection keeps working. Mutation: move the block
        to a top-level key of the raw record and this raises TraceError.
        """
        record = _capture_chat(tmp_path, dict(_BASE_REQUEST))
        restored = TraceRecord.from_dict(record.to_dict())
        assert restored.to_dict() == record.to_dict()
        assert (
            restored.messages[-1][SAMPLING_PROVENANCE_KEY]["sources"]["temperature"]
            == SAMPLING_SOURCE_SERVER_DEFAULT
        )
