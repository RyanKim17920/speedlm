from __future__ import annotations

import itertools
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from speedlm.traces.store import (
    TraceRecord,
    TraceStore,
    estimate_message_tokens,
)

_CASES = itertools.count()


def _case_path(tmp_path: Path) -> Path:
    return tmp_path / f"property-{next(_CASES)}.jsonl"


def _record(
    index: int,
    timestamp: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    token_count_source: str = "measured",
) -> TraceRecord:
    return TraceRecord(
        id=f"record-{index}",
        timestamp=timestamp,
        model="property-model",
        messages=(
            {"role": "user", "content": "ordinary prompt"},
            {"role": "assistant", "content": "ordinary response"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_count_source=token_count_source,
    )


def _tokens(record: TraceRecord) -> int:
    if record.total_tokens is not None:
        return record.total_tokens
    prompt, completion = estimate_message_tokens(list(record.messages))
    return prompt + completion


@given(
    specs=st.lists(
        st.tuples(
            st.floats(
                min_value=0.0,
                max_value=2_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.integers(min_value=0, max_value=500),
            st.integers(min_value=0, max_value=500),
        ),
        max_size=30,
    ),
    max_tokens=st.integers(min_value=1, max_value=4_000),
    max_age_days=st.floats(
        min_value=0.01,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    now=st.floats(
        min_value=0.0,
        max_value=2_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_pruned_store_never_exceeds_age_or_token_bounds(
    specs: list[tuple[float, int, int]],
    max_tokens: int,
    max_age_days: float,
    now: float,
    tmp_path: Path,
) -> None:
    store = TraceStore(
        _case_path(tmp_path),
        max_tokens=max_tokens,
        max_age_days=max_age_days,
        redaction_enabled=False,
    )
    for index, (timestamp, prompt, completion) in enumerate(specs):
        store.append(_record(index, timestamp, prompt, completion))

    store.prune(now=now)
    remaining = list(store.iter_records())

    assert sum(_tokens(record) for record in remaining) <= max_tokens
    assert all(
        now - record.timestamp <= max_age_days * 86_400.0
        for record in remaining
    )


@given(
    token_counts=st.lists(
        st.integers(min_value=0, max_value=500),
        min_size=1,
        max_size=30,
    ),
    max_tokens=st.integers(min_value=1, max_value=5_000),
)
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_token_pruning_drops_an_oldest_prefix_only(
    token_counts: list[int],
    max_tokens: int,
    tmp_path: Path,
) -> None:
    store = TraceStore(
        _case_path(tmp_path),
        max_tokens=max_tokens,
        max_age_days=365.0,
        redaction_enabled=False,
    )
    records = [
        _record(index, float(index + 1), tokens, 0)
        for index, tokens in enumerate(token_counts)
    ]
    for record in records:
        store.append(record)

    store.prune(now=float(len(records) + 1))
    remaining = list(store.iter_records())

    expected = list(records)
    while sum(_tokens(record) for record in expected) > max_tokens:
        expected.pop(0)
    assert [record.id for record in remaining] == [record.id for record in expected]


@given(
    specs=st.lists(
        st.tuples(
            st.floats(
                min_value=0.0,
                max_value=1_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.one_of(st.none(), st.integers(min_value=0, max_value=1_000)),
            st.one_of(st.none(), st.integers(min_value=0, max_value=1_000)),
            st.booleans(),
        ),
        max_size=30,
    )
)
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_stats_match_a_direct_recount(
    specs: list[tuple[float, int | None, int | None, bool]],
    tmp_path: Path,
) -> None:
    store = TraceStore(
        _case_path(tmp_path),
        max_tokens=100_000,
        max_age_days=365.0,
        redaction_enabled=False,
    )
    records: list[TraceRecord] = []
    for index, (timestamp, prompt, completion, prefer_measured) in enumerate(specs):
        source = (
            "measured"
            if prefer_measured and prompt is not None and completion is not None
            else "estimated"
        )
        record = _record(
            index,
            timestamp,
            prompt,
            completion,
            token_count_source=source,
        )
        records.append(record)
        store.append(record)

    stats = store.stats()
    measured = sum(
        _tokens(record)
        for record in records
        if record.token_count_source == "measured" and record.total_tokens is not None
    )
    estimated = sum(
        _tokens(record)
        for record in records
        if record.token_count_source != "measured" or record.total_tokens is None
    )

    assert stats.count == len(records)
    assert stats.tokens == sum(_tokens(record) for record in records)
    assert stats.measured_tokens == measured
    assert stats.estimated_tokens == estimated
    assert stats.unknown_token_records == sum(
        record.total_tokens is None for record in records
    )
    assert stats.oldest == (
        min(record.timestamp for record in records) if records else None
    )
    assert stats.newest == (
        max(record.timestamp for record in records) if records else None
    )
    assert stats.redacted_records == 0
    assert dict(stats.redaction_counts) == {}
