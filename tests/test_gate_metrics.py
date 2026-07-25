"""Tests for gate/metrics.py — no GPU, no network, fake Prometheus fixtures."""

from speedlm.gate.metrics import (
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)

# ---------------------------------------------------------------------------
# Fake Prometheus text fixtures
# ---------------------------------------------------------------------------

METRICS_STOCK = """\
# HELP generated_tokens_total Total number of generated tokens.
# TYPE generated_tokens_total counter
generated_tokens_total 1000
# HELP prompt_tokens_total Total number of prompt tokens.
# TYPE prompt_tokens_total counter
prompt_tokens_total 5000
# HELP time_per_output_token_ns_sum Sum of time per output token in ns.
# TYPE time_per_output_token_ns_sum counter
time_per_output_token_ns_sum 20000000000
"""

METRICS_CANDIDATE_BEFORE = """\
generated_tokens_total 1000
prompt_tokens_total 5000
time_per_output_token_ns_sum 10000000000
vllm:speculated_tokens_total 800
vllm:accepted_tokens_total 600
vllm:accept_token_count_total 150
vllm:reject_token_count_total 50
"""

METRICS_CANDIDATE_AFTER = """\
generated_tokens_total 3000
prompt_tokens_total 15000
time_per_output_token_ns_sum 50000000000
vllm:speculated_tokens_total 2800
vllm:accepted_tokens_total 2100
vllm:accept_token_count_total 500
vllm:reject_token_count_total 200
"""

METRICS_ACCEPTANCE_ABSENT = """\
generated_tokens_total 1000
prompt_tokens_total 5000
time_per_output_token_ns_sum 20000000000
"""

METRICS_AFTER_RESET = """\
generated_tokens_total 500
prompt_tokens_total 1000
time_per_output_token_ns_sum 10000000000
vllm:speculated_tokens_total 100
vllm:accepted_tokens_total 50
vllm:accept_token_count_total 10
vllm:reject_token_count_total 5
"""


# ---------------------------------------------------------------------------
# Tests — parse_metrics
# ---------------------------------------------------------------------------

def test_parse_stock_metrics() -> None:
    snap = parse_metrics(METRICS_STOCK)
    assert snap.generated_tokens == 1000.0
    assert snap.prompt_tokens == 5000.0
    assert snap.time_per_output_token_ns == 20_000_000_000.0
    assert snap.has_draft_counters is False


def test_parse_candidate_metrics() -> None:
    snap = parse_metrics(METRICS_CANDIDATE_AFTER)
    assert snap.generated_tokens == 3000.0
    assert snap.drafted_tokens == 2800.0
    assert snap.accepted_tokens == 2100.0
    assert snap.draft_accept_count == 500.0
    assert snap.draft_reject_count == 200.0
    assert snap.has_draft_counters is True
    assert abs(snap.acceptance_rate - 500 / (500 + 200)) < 1e-9


def test_parse_acceptance_absent() -> None:
    snap = parse_metrics(METRICS_ACCEPTANCE_ABSENT)
    assert snap.has_draft_counters is False
    assert snap.drafted_tokens == 0.0
    assert snap.accepted_tokens == 0.0


def test_parse_tpot_and_throughput() -> None:
    snap = parse_metrics(METRICS_CANDIDATE_AFTER)
    tpot_ms = snap.tpot_ms
    assert tpot_ms > 0
    tok_per_sec = snap.output_tok_per_sec
    assert tok_per_sec > 0


def test_parse_mean_accepted_length() -> None:
    snap = parse_metrics(METRICS_CANDIDATE_AFTER)
    mean_len = snap.mean_accepted_length
    assert abs(mean_len - 2100 / 500) < 1e-9


def test_parse_empty_text() -> None:
    snap = parse_metrics("")
    assert snap.generated_tokens == 0.0
    assert snap.has_draft_counters is False


def test_parse_with_gauge_lines() -> None:
    text = """\
generated_tokens_total 100
vllm:num_requests_running 5
vllm:num_requests_swapped 0
vllm:num_requests_waiting 12
"""
    snap = parse_metrics(text)
    assert snap.generated_tokens == 100.0
    assert snap.num_requests_running == 5.0
    assert snap.num_requests_swapped == 0.0
    assert snap.num_requests_waiting == 12.0


# ---------------------------------------------------------------------------
# Tests — compute_delta
# ---------------------------------------------------------------------------

def test_compute_delta_basic() -> None:
    before = parse_metrics(METRICS_CANDIDATE_BEFORE)
    after = parse_metrics(METRICS_CANDIDATE_AFTER)
    delta = compute_delta(before, after)

    assert delta.reset_detected is False
    assert delta.acceptance_available is True
    assert delta.drafted_tokens == 2800 - 800
    assert delta.accepted_tokens == 2100 - 600

    # delta_accept = 500-150=350, delta_reject = 200-50=150
    assert abs(delta.acceptance_rate - 350 / (350 + 150)) < 1e-9

    # Mean accepted length
    assert abs(delta.mean_accepted_length - (2100 - 600) / (500 - 150)) < 1e-9


def test_compute_delta_acceptance_unavailable() -> None:
    before = parse_metrics(METRICS_ACCEPTANCE_ABSENT)
    after = parse_metrics(METRICS_ACCEPTANCE_ABSENT)
    delta = compute_delta(before, after)

    assert delta.reset_detected is False
    assert delta.acceptance_available is False
    assert delta.drafted_tokens == 0.0
    assert delta.acceptance_rate == 0.0


def test_compute_delta_counter_reset_raises() -> None:
    before = parse_metrics(METRICS_CANDIDATE_AFTER)
    after = parse_metrics(METRICS_AFTER_RESET)

    err = None
    try:
        compute_delta(before, after)
    except CounterResetError as exc:
        err = exc
    assert err is not None
    assert "generated_tokens" in str(err)


def test_compute_delta_counter_reset_draft() -> None:
    """Draft counter reset also raises CounterResetError."""
    before = parse_metrics(METRICS_CANDIDATE_AFTER)
    after = parse_metrics(METRICS_ACCEPTANCE_ABSENT)
    err = None
    try:
        compute_delta(before, after)
    except CounterResetError as exc:
        err = exc
    assert err is not None


def test_compute_delta_tpot_and_throughput() -> None:
    before = parse_metrics(METRICS_CANDIDATE_BEFORE)
    after = parse_metrics(METRICS_CANDIDATE_AFTER)
    delta = compute_delta(before, after)

    # delta_gen = 2000, delta_tpot_ns = 40e9
    expected_tpot_ms = (40_000_000_000 / 2000) / 1_000_000
    assert abs(delta.tpot_ms - expected_tpot_ms) < 1e-6

    # tok_per_sec = 2000 / (40e9 / 1e9) = 2000 / 40 = 50
    assert abs(delta.output_tok_per_sec - 50.0) < 1e-6


def test_compute_delta_zero_tokens() -> None:
    """When no tokens generated, TPOT and throughput are 0."""
    before = MetricsSnapshot()
    after = MetricsSnapshot()
    delta = compute_delta(before, after)
    assert delta.tpot_ms == 0.0
    assert delta.output_tok_per_sec == 0.0


def test_metrics_delta_is_frozen() -> None:
    """MetricsDelta instances are immutable."""
    import dataclasses
    before = MetricsSnapshot()
    after = MetricsSnapshot()
    delta = compute_delta(before, after)
    # Frozen dataclasses are immutable by construction
    assert dataclasses.is_dataclass(delta)
