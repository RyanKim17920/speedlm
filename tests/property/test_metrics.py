from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from speedlm.gate.metrics import (
    COUNTER_NAMES,
    AcceptanceStatus,
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)

RESET_COUNTERS = {prom: field for field, prom in COUNTER_NAMES.items()}
GENERATED_TOKENS = COUNTER_NAMES["generated_tokens"]
DECODE_TIME = COUNTER_NAMES["decode_time_seconds"]


def _metric(name: str, value: float) -> str:
    return f"{name} {value:.17g}\n"


@given(st.text(max_size=2_000))
@settings(max_examples=100, deadline=None)
def test_parse_metrics_never_raises_on_arbitrary_text(text: str) -> None:
    assert isinstance(parse_metrics(text), MetricsSnapshot)


@given(
    st.sampled_from(sorted(RESET_COUNTERS)),
    st.floats(min_value=1.0, max_value=1e12, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=70, deadline=None)
def test_counter_resets_never_produce_a_valid_delta(
    prometheus_name: str,
    before_value: float,
    fraction: float,
) -> None:
    before = parse_metrics(_metric(prometheus_name, before_value))
    after = parse_metrics(_metric(prometheus_name, before_value * fraction))

    with pytest.raises(CounterResetError, match=RESET_COUNTERS[prometheus_name]):
        compute_delta(before, after)


@given(
    st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=60, deadline=None)
def test_monotonic_counters_have_non_negative_deltas(
    before_generated: float,
    increment: float,
) -> None:
    before = parse_metrics(_metric(GENERATED_TOKENS, before_generated))
    after = parse_metrics(
        _metric(GENERATED_TOKENS, before_generated + increment)
    )

    delta = compute_delta(before, after)

    assert not delta.reset_detected
    assert math.isclose(delta.drafted_tokens, 0.0)
    assert math.isclose(delta.accepted_tokens, 0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/gate/metrics.py:161 MetricsSnapshot.acceptance_rate and "
        "compute_delta both return 0.0 when no draft counters were scraped, so "
        "AcceptanceStatus.UNAVAILABLE is declared but never returned; the "
        "distinction is carried by has_draft_counters / acceptance_available "
        "instead. NOT FIXED DELIBERATELY: this is a type-design question, not a "
        "security one, and it sits on the promotion gate, which has no "
        "post-promotion rollback. Returning the enum widens the type to "
        "'float | AcceptanceStatus' at src/speedlm/gate/decide.py:382 and 481-482, "
        "where the value flows into GateRepeat.stock_acceptance_rate / "
        "candidate_acceptance_rate (declared float), into the serialized gate "
        "record at decide.py:274-275 and runner.py:826, and into arithmetic at "
        "report.py:1284-1285. A fix must either keep acceptance_rate a float and "
        "add a separate acceptance_status property, or widen the type and update "
        "every consumer plus the on-disk record schema, and must replay "
        "tests/test_gate_decide.py jobs 368670/368689 to prove no promote/reject "
        "decision changes."
    ),
)
def test_absent_draft_counters_are_unavailable_not_zero() -> None:
    snapshot = parse_metrics("")
    delta = compute_delta(MetricsSnapshot(), snapshot)

    assert not snapshot.has_draft_counters
    assert not delta.acceptance_available
    assert snapshot.acceptance_rate is AcceptanceStatus.UNAVAILABLE
    assert delta.acceptance_rate is AcceptanceStatus.UNAVAILABLE


def test_time_sum_counter_reset_is_detected() -> None:
    before = parse_metrics(_metric(DECODE_TIME, 1.0))
    after = parse_metrics(_metric(DECODE_TIME, 0.0))

    with pytest.raises(CounterResetError, match="decode_time_seconds"):
        compute_delta(before, after)
