"""Tests for speedlm.traces.normalize — normalize_record, normalize_file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from speedlm.config import SamplingConfig
from speedlm.traces.normalize import (
    NormalizeError,
    Rejection,
    normalize_file,
    normalize_record,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _default_data(**overrides: object) -> dict:
    base: dict = {
        "id": "ext-1",
        "timestamp": 1700000000.0,
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    base.update(overrides)
    return base


def _defaults() -> SamplingConfig:
    return SamplingConfig(temperature=0.5, top_p=0.95, seed=17)


# ── Happy path ──────────────────────────────────────────────────────────────


class TestNormalizeRecordHappy:
    def test_basic(self) -> None:
        rec = normalize_record(_default_data(), defaults=_defaults())
        assert rec.id == "ext-1"
        assert rec.model == "gpt-4"
        assert rec.temperature == 0.5
        assert rec.top_p == 0.95
        assert rec.seed == 17
        assert rec.prompt_tokens == 10
        assert rec.completion_tokens == 20


# ── Sampling defaults ──────────────────────────────────────────────────────


class TestSamplingDefaults:
    def test_defaults_applied_when_absent(self) -> None:
        data = _default_data()
        assert "temperature" not in data
        assert "top_p" not in data
        assert "seed" not in data
        rec = normalize_record(data, defaults=_defaults())
        assert rec.temperature == 0.5
        assert rec.top_p == 0.95
        assert rec.seed == 17

    def test_gpt_oss_defaults_applied_when_absent(self) -> None:
        rec = normalize_record(_default_data(), defaults=SamplingConfig())
        assert rec.temperature == 0.0
        assert rec.top_p == 1.0
        assert rec.seed == 0

    def test_record_values_override_defaults(self) -> None:
        data = _default_data(temperature=0.9, top_p=0.4, seed=99)
        rec = normalize_record(data, defaults=_defaults())
        assert rec.temperature == 0.9
        assert rec.top_p == 0.4
        assert rec.seed == 99

    def test_present_but_invalid_temperature_rejected(self) -> None:
        data = _default_data(temperature=-1.0)
        with pytest.raises(NormalizeError, match="temperature"):
            normalize_record(data, defaults=_defaults())

    def test_present_but_invalid_temp_type_rejected(self) -> None:
        data = _default_data(temperature="hot")
        with pytest.raises(NormalizeError):
            normalize_record(data, defaults=_defaults())

    def test_present_but_invalid_temp_bool_rejected(self) -> None:
        data = _default_data(temperature=True)
        with pytest.raises(NormalizeError):
            normalize_record(data, defaults=_defaults())


# ── Missing messages ───────────────────────────────────────────────────────


class TestMissingMessages:
    def test_missing_messages(self) -> None:
        data = _default_data()
        del data["messages"]
        with pytest.raises(NormalizeError, match="messages"):
            normalize_record(data, defaults=_defaults())

    def test_empty_messages(self) -> None:
        data = _default_data(messages=[])
        with pytest.raises(NormalizeError, match="messages"):
            normalize_record(data, defaults=_defaults())


# ── Model handling ─────────────────────────────────────────────────────────


class TestModelHandling:
    def test_missing_model_no_default_uses_unknown(self) -> None:
        data = _default_data()
        del data["model"]
        rec = normalize_record(data, defaults=_defaults(), default_model=None)
        assert rec.model == "unknown"

    def test_default_model_applied(self) -> None:
        data = _default_data()
        del data["model"]
        rec = normalize_record(data, defaults=_defaults(), default_model="fallback")
        assert rec.model == "fallback"


# ── Deterministic ID ───────────────────────────────────────────────────────


class TestDeterministicId:
    def test_id_generated_when_missing(self) -> None:
        data = _default_data()
        del data["id"]
        rec1 = normalize_record(data, defaults=_defaults())
        rec2 = normalize_record(data, defaults=_defaults())
        assert rec1.id == rec2.id
        assert rec1.id.startswith("tr-")

    def test_id_stable_across_calls(self) -> None:
        data = _default_data(id="")  # empty id triggers generation
        ids = {normalize_record(data, defaults=_defaults()).id for _ in range(5)}
        assert len(ids) == 1


# ── Usage token extraction ─────────────────────────────────────────────────


class TestTokenExtraction:
    def test_usage_tokens(self) -> None:
        rec = normalize_record(_default_data(), defaults=_defaults())
        assert rec.prompt_tokens == 10
        assert rec.completion_tokens == 20

    def test_no_usage_is_estimated(self) -> None:
        data = _default_data()
        del data["usage"]
        rec = normalize_record(data, defaults=_defaults())
        assert rec.prompt_tokens > 0
        assert rec.completion_tokens > 0
        assert rec.token_count_source == "estimated"
        assert rec.to_dict()["token_count_source"] == "estimated"

    def test_unparseable_tokens_are_estimated(self) -> None:
        data = _default_data(
            usage={"prompt_tokens": "unknown", "completion_tokens": -1}
        )
        rec = normalize_record(data, defaults=_defaults())
        assert rec.prompt_tokens > 0
        assert rec.completion_tokens > 0
        assert rec.token_count_source == "estimated"

    def test_top_level_tokens(self) -> None:
        data = _default_data()
        del data["usage"]
        data["prompt_tokens"] = 30
        data["completion_tokens"] = 40
        rec = normalize_record(data, defaults=_defaults())
        assert rec.prompt_tokens == 30
        assert rec.completion_tokens == 40


# ── Timestamp handling ─────────────────────────────────────────────────────


class TestTimestamp:
    def test_created_field(self) -> None:
        data = _default_data()
        del data["timestamp"]
        data["created"] = 1700001000
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1700001000.0

    def test_negative_timestamp_rejected(self) -> None:
        data = _default_data(timestamp=-1.0)
        with pytest.raises(NormalizeError):
            normalize_record(data, defaults=_defaults())

    def test_bool_timestamp_rejected(self) -> None:
        data = _default_data(timestamp=True)
        with pytest.raises(NormalizeError):
            normalize_record(data, defaults=_defaults())


# ── normalize_file ─────────────────────────────────────────────────────────


class TestNormalizeFile:
    def test_happy_path(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(_default_data()) + "\n")
        result = normalize_file(f, defaults=_defaults())
        assert result.accepted_count == 1
        assert result.rejected_count == 0
        assert result.shape_counts == {"internal": 1}

    def test_malformed_json_becomes_rejection(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(_default_data()) + "\n{bad json}\n")
        result = normalize_file(f, defaults=_defaults())
        assert result.accepted_count == 1
        assert result.rejected_count == 1
        assert result.rejected[0].line == 2
        assert isinstance(result.rejected[0], Rejection)

    def test_missing_model_accepted(self, tmp_path: Path) -> None:
        data = _default_data()
        del data["model"]
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(data) + "\n")
        result = normalize_file(f, defaults=_defaults(), default_model=None)
        assert result.accepted_count == 1
        assert result.rejected_count == 0
        assert result.accepted[0].model == "unknown"

    def test_default_model_in_file(self, tmp_path: Path) -> None:
        data = _default_data()
        del data["model"]
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(data) + "\n")
        result = normalize_file(f, defaults=_defaults(), default_model="def-model")
        assert result.accepted_count == 1
        assert result.accepted[0].model == "def-model"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NormalizeError):
            normalize_file(tmp_path / "nope.jsonl")

    def test_rejection_preserves_other_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps(_default_data()),
            "{bad}",
            json.dumps(_default_data(id="ext-2")),
        ]
        f.write_text("\n".join(lines) + "\n")
        result = normalize_file(f, defaults=_defaults())
        assert result.accepted_count == 2
        assert result.rejected_count == 1
        assert result.rejected[0].line == 2


# ── External shape auto-detection ──────────────────────────────────────────


class TestExternalShapes:
    def test_openai_chat_completion_response(self) -> None:
        data = {
            "id": "chatcmpl-1",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "created": 1750000000,
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.id == "chatcmpl-1"
        assert rec.timestamp == 1750000000.0
        assert rec.messages == ({"role": "assistant", "content": "hi"},)
        assert rec.prompt_tokens == 5
        assert rec.completion_tokens == 3

    def test_openai_tool_calls_preserved(self) -> None:
        tool_calls = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        }]
        data = {
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            "created": 1750000000,
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.messages[0]["tool_calls"] == tool_calls
        assert rec.tool_calls == tuple(tool_calls)

    def test_openai_response_without_usage_is_estimated(self) -> None:
        data = {
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
            }],
            "created": 1750000000,
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.token_count_source == "estimated"
        assert rec.completion_tokens > 0

    def test_nested_request_response_pair(self) -> None:
        data = {
            "request": {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
            },
            "response": {
                "id": "chatcmpl-pair",
                "model": "gpt-4-0613",
                "choices": [{
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                "created": 1750000000,
            },
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.id == "chatcmpl-pair"
        assert rec.model == "gpt-4-0613"
        assert rec.messages == (
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        )
        assert rec.temperature == 0.2
        assert rec.prompt_tokens == 5
        assert rec.completion_tokens == 3

    def test_flat_request_completion_pair(self) -> None:
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "completion": "hi",
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.messages[-1] == {"role": "assistant", "content": "hi"}
        assert rec.prompt_tokens > 0
        assert rec.completion_tokens > 0
        assert rec.token_count_source == "estimated"

    def test_bare_conversation(self) -> None:
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.messages == tuple(data["messages"])
        assert rec.prompt_tokens > 0
        assert rec.completion_tokens > 0
        assert rec.token_count_source == "estimated"

    def test_combined_messages_and_choices_recovers_conversation(self) -> None:
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.messages == (
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        )

    def test_unrecognized_shape_rejected(self) -> None:
        with pytest.raises(NormalizeError, match="cannot recover a conversation"):
            normalize_record({"model": "gpt-4"}, defaults=_defaults())

    def test_arbitrary_proxy_capture(self) -> None:
        data = {
            "event": {
                "incoming_payload": {
                    "body": {
                        "model": "proxy-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                },
                "upstream_result": {
                    "data": {
                        "id": "proxy-1",
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "hi",
                            },
                        }],
                    },
                },
            },
        }
        rec = normalize_record(data, defaults=_defaults())
        assert rec.id == "proxy-1"
        assert rec.model == "proxy-model"
        assert rec.messages == (
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        )
        assert rec.token_count_source == "estimated"

    def test_file_reports_each_detected_shape(self, tmp_path: Path) -> None:
        internal = _default_data()
        response = {
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "created": 1750000000,
        }
        bare = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        proxy = {
            "event": {
                "input": {
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "output": {"role": "assistant", "content": "hi"},
            },
        }
        path = tmp_path / "shapes.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(item) for item in (response, internal, bare, proxy)
            ) + "\n"
        )
        result = normalize_file(path, defaults=_defaults())
        assert result.accepted_count == 4
        assert result.shape_counts == {
            "openai-response": 1,
            "internal": 1,
            "bare-conversation": 1,
            "proxy-capture": 1,
        }

# ── ISO-8601 timestamp acceptance (Defect 1 regression) ────────────────────


class TestISOTimestamp:
    """ISO-8601 string timestamps must be accepted and normalized to epoch float."""

    def test_iso_with_z(self) -> None:
        data = _default_data(timestamp="2026-07-25T04:00:00Z")
        rec = normalize_record(data, defaults=_defaults())
        # 2026-07-25T04:00:00Z == 1784952000.0
        assert rec.timestamp == 1784952000.0

    def test_iso_with_positive_offset(self) -> None:
        data = _default_data(timestamp="2026-07-25T04:00:00+00:00")
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1784952000.0

    def test_iso_with_nonzero_offset(self) -> None:
        # +05:00 means wall clock is 5h ahead → epoch is 5h less
        data = _default_data(timestamp="2026-07-25T09:00:00+05:00")
        rec = normalize_record(data, defaults=_defaults())
        # 2026-07-25T09:00:00+05:00 == 2026-07-25T04:00:00Z
        assert rec.timestamp == 1784952000.0

    def test_iso_naive_treated_as_utc(self) -> None:
        data = _default_data(timestamp="2026-07-25T04:00:00")
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1784952000.0

    def test_epoch_int(self) -> None:
        data = _default_data(timestamp=1700000000)
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1700000000.0

    def test_epoch_float(self) -> None:
        data = _default_data(timestamp=1700000000.5)
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1700000000.5

    def test_malformed_string_rejected(self) -> None:
        data = _default_data(timestamp="not-a-date")
        with pytest.raises(NormalizeError, match="not a valid ISO-8601"):
            normalize_record(data, defaults=_defaults())

    def test_nan_rejected(self) -> None:

        data = _default_data(timestamp=float("nan"))
        with pytest.raises(NormalizeError, match="non-finite|finite"):
            normalize_record(data, defaults=_defaults())

    def test_inf_rejected(self) -> None:
        data = _default_data(timestamp=float("inf"))
        with pytest.raises(NormalizeError, match="non-finite|finite"):
            normalize_record(data, defaults=_defaults())

    def test_negative_iso_rejected(self) -> None:
        data = _default_data(timestamp="0001-01-01T00:00:00+00:00")
        with pytest.raises(NormalizeError, match=">= 0"):
            normalize_record(data, defaults=_defaults())

    def test_created_field_iso(self) -> None:
        """The 'created' fallback also accepts ISO-8601 strings."""
        data = _default_data()
        del data["timestamp"]
        data["created"] = "2026-07-25T04:00:00Z"
        rec = normalize_record(data, defaults=_defaults())
        assert rec.timestamp == 1784952000.0

    def test_file_level_iso_accepted(self, tmp_path: Path) -> None:
        """End-to-end: normalize_file accepts ISO timestamps."""
        f = tmp_path / "iso.jsonl"
        data = _default_data(timestamp="2026-07-25T04:00:00Z")
        f.write_text(json.dumps(data) + "\n")
        result = normalize_file(f, defaults=_defaults())
        assert result.accepted_count == 1
        assert result.rejected_count == 0
        assert result.accepted[0].timestamp == 1784952000.0

    def test_file_level_malformed_rejected(self, tmp_path: Path) -> None:
        """End-to-end: malformed ISO strings are rejected, not crashed."""
        f = tmp_path / "bad-tz.jsonl"
        data = _default_data(timestamp="yesterday")
        f.write_text(json.dumps(data) + "\n")
        result = normalize_file(f, defaults=_defaults())
        assert result.accepted_count == 0
        assert result.rejected_count == 1
        assert "ISO-8601" in result.rejected[0].reason
