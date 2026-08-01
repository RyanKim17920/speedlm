"""Property tests for external trace normalization."""

from __future__ import annotations

import datetime as dt
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from speedlm.config import SamplingConfig
from speedlm.traces.normalize import (
    NormalizeError,
    normalize_record,
)
from speedlm.traces.store import TraceRecord, TraceStore

DEFAULTS = SamplingConfig(temperature=0.2, top_p=0.9, seed=7)
JSON_SCALAR = st.none() | st.booleans() | st.integers(-10**12, 10**12) | st.floats(
    allow_nan=False,
    allow_infinity=False,
) | st.text(max_size=40)
JSON_VALUE: SearchStrategy[Any] = st.recursive(
    JSON_SCALAR,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=20), children, max_size=4),
    max_leaves=12,
)


def _message(role: str, content: Any) -> dict[str, Any]:
    return {"role": role, "content": content}


@st.composite
def supported_records(draw: DrawFn) -> dict[str, Any]:
    """Build one valid example of each documented external record shape."""
    shape = draw(
        st.sampled_from(
            (
                "internal",
                "bare-conversation",
                "openai-response",
                "request-response",
                "proxy-capture",
            )
        )
    )
    content = draw(st.text(max_size=60))
    reply = draw(st.text(max_size=60))
    timestamp = draw(st.integers(min_value=0, max_value=4_000_000_000))
    common: dict[str, Any] = {
        "id": draw(st.text(min_size=1, max_size=20)),
        "model": draw(st.text(min_size=1, max_size=20)),
        "timestamp": timestamp,
    }
    user = _message("user", content)
    assistant = _message("assistant", reply)
    usage = {
        "prompt_tokens": draw(st.integers(min_value=0, max_value=10_000)),
        "completion_tokens": draw(st.integers(min_value=0, max_value=10_000)),
    }
    if shape == "internal":
        return {**common, "messages": [user, assistant], "usage": usage}
    if shape == "bare-conversation":
        return {**common, "messages": [user, assistant]}
    if shape == "openai-response":
        return {
            **common,
            "choices": [{"index": 0, "message": assistant}],
            "usage": usage,
        }
    if shape == "request-response":
        return {
            "request": {
                "model": common["model"],
                "messages": [user],
                "temperature": 0.2,
            },
            "response": {
                **common,
                "choices": [{"index": 0, "message": assistant}],
                "usage": usage,
            },
        }
    return {
        "capture": {
            "request_body": {
                "model": common["model"],
                "messages": [user],
            },
            "response_body": {
                **common,
                "choices": [{"index": 0, "message": assistant}],
                "usage": usage,
            },
        }
    }


@given(st.dictionaries(st.text(max_size=20), JSON_VALUE, max_size=12))
@settings(max_examples=200, deadline=None)
def test_normalize_never_raises_undeclared_exception(data: dict[str, Any]) -> None:
    try:
        record = normalize_record(data, defaults=DEFAULTS)
    except NormalizeError:
        return
    assert isinstance(record, TraceRecord)
    assert TraceRecord.from_dict(record.to_dict()) == record


@given(st.dictionaries(st.text(max_size=20), JSON_VALUE, max_size=12))
@settings(max_examples=150, deadline=None)
def test_acceptance_or_rejection_is_deterministic(data: dict[str, Any]) -> None:
    outcomes: list[bool] = []
    for _ in range(2):
        try:
            normalize_record(data, defaults=DEFAULTS)
        except NormalizeError:
            outcomes.append(False)
        else:
            outcomes.append(True)
    assert outcomes[0] == outcomes[1]


@given(supported_records())
@settings(max_examples=100, deadline=None)
def test_all_five_shapes_produce_valid_round_trippable_records(
    data: dict[str, Any],
) -> None:
    record = normalize_record(data, defaults=DEFAULTS)
    assert isinstance(record, TraceRecord)

    with tempfile.TemporaryDirectory() as directory:
        store = TraceStore(
            Path(directory) / "traces.jsonl",
            redaction_enabled=False,
        )
        store.append(record)
        assert list(store.iter_records()) == [record]


@st.composite
def iso_datetimes(draw: DrawFn) -> tuple[str, float]:
    timezone = draw(
        st.sampled_from(
            (
                None,
                dt.UTC,
                dt.timezone(dt.timedelta(hours=-7)),
                dt.timezone(dt.timedelta(hours=5, minutes=30)),
            )
        )
    )
    value = draw(
        st.datetimes(
            min_value=dt.datetime(1970, 1, 1),
            max_value=dt.datetime(9998, 12, 31, 23, 59, 59, 999999),
            timezones=st.just(timezone),
        )
    )
    encoded = value.isoformat()
    if value.tzinfo is dt.UTC and draw(st.booleans()):
        encoded = encoded.removesuffix("+00:00") + "Z"
    aware = value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
    return encoded, aware.timestamp()


@given(iso_datetimes())
@settings(max_examples=100, deadline=None)
def test_valid_iso_8601_timestamps_are_accepted(value: tuple[str, float]) -> None:
    encoded, expected = value
    record = normalize_record(
        {"messages": [_message("user", "x")], "timestamp": encoded},
        defaults=DEFAULTS,
    )
    assert math.isclose(record.timestamp, expected, abs_tol=1e-6)


@given(
    st.one_of(
        st.integers(min_value=0, max_value=10**15),
        st.floats(
            min_value=0,
            max_value=10**15,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
@settings(max_examples=100, deadline=None)
def test_finite_non_negative_epoch_timestamps_are_accepted(epoch: int | float) -> None:
    record = normalize_record(
        {"messages": [_message("user", "x")], "timestamp": epoch},
        defaults=DEFAULTS,
    )
    assert record.timestamp == float(epoch)


@given(
    st.sampled_from(
        (
            "",
            "not-a-date",
            "NaN",
            "inf",
            "-1",
            "2024-13-01",
            "2024-00-01",
            "2024-02-30",
            "2024-01-01T25:00:00",
            "2024-01-01T00:61:00",
            "2024-01-01T00:00:61",
            "2024-01-01T00:00:00+99:00",
        )
    )
)
@settings(max_examples=30, deadline=None)
def test_malformed_timestamp_strings_are_rejected(timestamp: str) -> None:
    with pytest.raises(NormalizeError):
        normalize_record(
            {"messages": [_message("user", "x")], "timestamp": timestamp},
            defaults=DEFAULTS,
        )


def test_huge_epoch_is_a_declared_rejection() -> None:
    with pytest.raises(NormalizeError):
        normalize_record(
            {"messages": [_message("user", "x")], "timestamp": 10**309},
            defaults=DEFAULTS,
        )
