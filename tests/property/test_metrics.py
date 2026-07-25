from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from speedlm.gate.metrics import (
    AcceptanceStatus,
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)

RESET_COUNTERS = {
    "generated_tokens_total": "generated_tokens",
    "prompt_tokens_total": "prompt_tokens",
    "vllm:speculated_tokens_total": "drafted_tokens",
    "vllm:accepted_tokens_total": "accepted_tokens",
    "vllm:accept_token_count_total": "draft_accept_count",
    "vllm:reject_token_count_total": "draft_reject_count",
}


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
    before = parse_metrics(_metric("generated_tokens_total", before_generated))
    after = parse_metrics(
        _metric("generated_tokens_total", before_generated + increment)
    )

    delta = compute_delta(before, after)

    assert not delta.reset_detected
    assert math.isclose(delta.drafted_tokens, 0.0)
    assert math.isclose(delta.accepted_tokens, 0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/gate/metrics.py:80 and :254 encode absent draft counters as "
        "numeric 0.0 even though AcceptanceStatus.UNAVAILABLE exists; minimal "
        "input is an empty metrics scrape"
    ),
)
def test_absent_draft_counters_are_unavailable_not_zero() -> None:
    snapshot = parse_metrics("")
    delta = compute_delta(MetricsSnapshot(), snapshot)

    assert not snapshot.has_draft_counters
    assert not delta.acceptance_available
    assert snapshot.acceptance_rate is AcceptanceStatus.UNAVAILABLE
    assert delta.acceptance_rate is AcceptanceStatus.UNAVAILABLE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/gate/metrics.py:228 omits time_per_output_token_ns from reset checks"
    ),
)
def test_time_sum_counter_reset_is_detected() -> None:
    before = parse_metrics("time_per_output_token_ns_sum 1\n")
    after = parse_metrics("time_per_output_token_ns_sum 0\n")

    with pytest.raises(CounterResetError, match="time_per_output_token_ns"):
        compute_delta(before, after)
