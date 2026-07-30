"""Tests for gate/metrics.py — no GPU, no network.

The fixtures under ``tests/data/`` are verbatim excerpts of real vLLM
``/metrics`` responses captured from the vendored runtime.  Parsing those
files, rather than hand-written Prometheus text, is what keeps
:data:`speedlm.gate.metrics.COUNTER_NAMES` honest: a synthetic fixture can
agree with a parser that agrees with nothing vLLM actually emits.
"""

from pathlib import Path

import pytest

from speedlm.gate.metrics import (
    COUNTER_NAMES,
    CounterResetError,
    compute_delta,
    parse_metrics,
)

DATA_DIR = Path(__file__).parent / "data"
REAL_BEFORE = (DATA_DIR / "vllm_metrics_before.prom").read_text()
REAL_AFTER = (DATA_DIR / "vllm_metrics_after.prom").read_text()

# Values read straight out of the captured fixtures.
_AFTER_DRAFTS = 33.0
_AFTER_DRAFT_TOKENS = 99.0
_AFTER_ACCEPTED = 78.0
_AFTER_GENERATED = 112.0
_AFTER_PROMPT = 41.0
_AFTER_DECODE_SECONDS = 1.7592467290814966


# ---------------------------------------------------------------------------
# Synthetic fixtures — same counter names as the real capture, chosen values.
# ---------------------------------------------------------------------------

def _exposition(**counters: float) -> str:
    return "".join(
        f'{COUNTER_NAMES[field]}{{engine="0",model_name="m"}} {value}\n'
        for field, value in counters.items()
    )


METRICS_NO_SPEC = _exposition(
    generated_tokens=1000,
    prompt_tokens=5000,
    decode_time_seconds=20.0,
)

METRICS_SPEC_BEFORE = _exposition(
    generated_tokens=1000,
    prompt_tokens=5000,
    decode_time_seconds=10.0,
    drafted_tokens=800,
    accepted_tokens=600,
    num_drafts=200,
)

METRICS_SPEC_AFTER = _exposition(
    generated_tokens=3000,
    prompt_tokens=15000,
    decode_time_seconds=50.0,
    drafted_tokens=2800,
    accepted_tokens=2100,
    num_drafts=700,
)

METRICS_SPEC_RESET = _exposition(
    generated_tokens=500,
    prompt_tokens=1000,
    decode_time_seconds=10.0,
    drafted_tokens=100,
    accepted_tokens=50,
    num_drafts=25,
)


# ---------------------------------------------------------------------------
# Regression tests — the parser must agree with real vLLM output
# ---------------------------------------------------------------------------

def test_parses_real_vllm_capture_with_labels() -> None:
    """A real labelled /metrics body must yield real numbers, not zeros.

    This is the regression guard for the shipped bug: the parser expected
    counter names (``vllm:accept_token_count_total`` and friends) that vLLM
    never emits, so every scrape parsed to an all-zero snapshot and the gate
    rejected every candidate with ``acceptance_unavailable``.
    """
    snap = parse_metrics(REAL_AFTER)

    assert snap.has_draft_counters is True
    assert snap.drafted_tokens == _AFTER_DRAFT_TOKENS
    assert snap.accepted_tokens == _AFTER_ACCEPTED
    assert snap.num_drafts == _AFTER_DRAFTS
    assert snap.generated_tokens == _AFTER_GENERATED
    assert snap.prompt_tokens == _AFTER_PROMPT
    assert snap.decode_time_seconds == pytest.approx(_AFTER_DECODE_SECONDS)


def test_real_capture_yields_nonzero_acceptance_and_throughput() -> None:
    delta = compute_delta(parse_metrics(REAL_BEFORE), parse_metrics(REAL_AFTER))

    assert delta.acceptance_available is True
    assert delta.acceptance_rate == pytest.approx(
        _AFTER_ACCEPTED / _AFTER_DRAFT_TOKENS
    )
    # vLLM's definition: 1 + accepted / drafts.
    assert delta.mean_accepted_length == pytest.approx(
        1.0 + _AFTER_ACCEPTED / _AFTER_DRAFTS
    )
    assert delta.output_tok_per_sec == pytest.approx(
        _AFTER_GENERATED / _AFTER_DECODE_SECONDS
    )
    assert delta.acceptance_rate > 0.0
    assert delta.output_tok_per_sec > 0.0


def test_every_configured_counter_appears_in_the_real_capture() -> None:
    """Each name the gate scrapes for must exist in real vLLM output."""
    missing = [
        prom_name
        for prom_name in COUNTER_NAMES.values()
        if prom_name not in REAL_AFTER
    ]
    assert missing == []


def test_per_pos_counter_is_not_mistaken_for_the_accepted_counter() -> None:
    text = (
        'vllm:spec_decode_num_accepted_tokens_per_pos_total'
        '{engine="0",position="0"} 900\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 78\n'
    )
    assert parse_metrics(text).accepted_tokens == 78.0


def test_counters_are_summed_across_engines() -> None:
    text = (
        'vllm:generation_tokens_total{engine="0"} 100\n'
        'vllm:generation_tokens_total{engine="1"} 40\n'
    )
    assert parse_metrics(text).generated_tokens == 140.0


# ---------------------------------------------------------------------------
# Tests — parse_metrics
# ---------------------------------------------------------------------------

def test_parse_without_spec_decode_counters() -> None:
    snap = parse_metrics(METRICS_NO_SPEC)
    assert snap.generated_tokens == 1000.0
    assert snap.prompt_tokens == 5000.0
    assert snap.decode_time_seconds == 20.0
    assert snap.has_draft_counters is False
    assert snap.drafted_tokens == 0.0
    assert snap.accepted_tokens == 0.0


def test_parse_spec_decode_counters() -> None:
    snap = parse_metrics(METRICS_SPEC_AFTER)
    assert snap.has_draft_counters is True
    assert snap.drafted_tokens == 2800.0
    assert snap.accepted_tokens == 2100.0
    assert snap.num_drafts == 700.0
    assert snap.acceptance_rate == pytest.approx(2100 / 2800)
    assert snap.mean_accepted_length == pytest.approx(1 + 2100 / 700)


def test_parse_tpot_and_throughput() -> None:
    snap = parse_metrics(METRICS_SPEC_AFTER)
    assert snap.tpot_ms == pytest.approx(50.0 / 3000 * 1000)
    assert snap.output_tok_per_sec == pytest.approx(3000 / 50.0)


def test_parse_empty_text() -> None:
    snap = parse_metrics("")
    assert snap.generated_tokens == 0.0
    assert snap.has_draft_counters is False
    assert snap.acceptance_rate == 0.0
    assert snap.output_tok_per_sec == 0.0


def test_parse_with_gauge_lines() -> None:
    text = (
        'vllm:generation_tokens_total{engine="0"} 100\n'
        'vllm:num_requests_running{engine="0"} 5\n'
        'vllm:num_requests_swapped{engine="0"} 0\n'
        'vllm:num_requests_waiting{engine="0"} 12\n'
    )
    snap = parse_metrics(text)
    assert snap.generated_tokens == 100.0
    assert snap.num_requests_running == 5.0
    assert snap.num_requests_swapped == 0.0
    assert snap.num_requests_waiting == 12.0


# ---------------------------------------------------------------------------
# Tests — compute_delta
# ---------------------------------------------------------------------------

def test_compute_delta_basic() -> None:
    delta = compute_delta(
        parse_metrics(METRICS_SPEC_BEFORE),
        parse_metrics(METRICS_SPEC_AFTER),
    )

    assert delta.reset_detected is False
    assert delta.acceptance_available is True
    assert delta.drafted_tokens == 2000.0
    assert delta.accepted_tokens == 1500.0
    assert delta.acceptance_rate == pytest.approx(1500 / 2000)
    assert delta.mean_accepted_length == pytest.approx(1 + 1500 / 500)


def test_compute_delta_acceptance_unavailable_when_counters_absent() -> None:
    snap = parse_metrics(METRICS_NO_SPEC)
    delta = compute_delta(snap, snap)

    assert delta.reset_detected is False
    assert delta.acceptance_available is False
    assert delta.drafted_tokens == 0.0
    assert delta.acceptance_rate == 0.0


def test_compute_delta_acceptance_unavailable_when_no_drafting_happened() -> None:
    """Present-but-idle counters are not a measured zero-percent acceptance."""
    snap = parse_metrics(METRICS_SPEC_AFTER)
    delta = compute_delta(snap, snap)

    assert delta.acceptance_available is False
    assert delta.acceptance_rate == 0.0


def test_compute_delta_counter_reset_raises() -> None:
    with pytest.raises(CounterResetError, match="generated_tokens"):
        compute_delta(
            parse_metrics(METRICS_SPEC_AFTER),
            parse_metrics(METRICS_SPEC_RESET),
        )


def test_compute_delta_counter_reset_draft() -> None:
    with pytest.raises(CounterResetError):
        compute_delta(
            parse_metrics(METRICS_SPEC_AFTER),
            parse_metrics(METRICS_NO_SPEC),
        )


def test_compute_delta_tpot_and_throughput() -> None:
    delta = compute_delta(
        parse_metrics(METRICS_SPEC_BEFORE),
        parse_metrics(METRICS_SPEC_AFTER),
    )
    # delta_gen = 2000 tokens over delta_decode = 40 s.
    assert delta.tpot_ms == pytest.approx(40.0 / 2000 * 1000)
    assert delta.output_tok_per_sec == pytest.approx(2000 / 40.0)
