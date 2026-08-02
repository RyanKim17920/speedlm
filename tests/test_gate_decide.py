"""Tests for gate/decide.py — no GPU, no network."""

import math

import pytest

from speedlm.config import DivergenceCriterion, PromotionConfig
from speedlm.gate.decide import (
    GATING_ACCEPTANCE_STATISTIC,
    GATING_THROUGHPUT_STATISTIC,
    Decision,
    DispersionBasis,
    DivergenceBasis,
    Reason,
    Verdict,
    decide_promotion,
    first_divergence,
)
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import (
    ReplayResult,
    RequestResult,
    RunResults,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_request_result(
    response_text: str = "ok response",
    *,
    valid: bool = True,
    error: str = "",
    latency_s: float = 0.1,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    output_tokens: tuple[str, ...] = (),
    context_hash: str = "abcd1234",
) -> RequestResult:
    return RequestResult(
        context_hash=context_hash,
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        response_text=response_text,
        valid=valid,
        error=error,
        output_tokens=output_tokens,
    )


def _make_run(
    results: list[RequestResult],
) -> RunResults:
    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count
    total_latency = sum(r.latency_s for r in results)
    total_pt = sum(r.prompt_tokens for r in results)
    total_ct = sum(r.completion_tokens for r in results)
    return RunResults(
        results=tuple(results),
        total_latency_s=total_latency,
        total_prompt_tokens=total_pt,
        total_completion_tokens=total_ct,
        valid_count=valid_count,
        invalid_count=invalid_count,
        invalid_rate=invalid_count / len(results) if results else 0.0,
    )


def _make_replay(
    runs: list[RunResults],
    *,
    suite_hash: str = "test-suite-hash",
) -> ReplayResult:
    return ReplayResult(
        run_results=tuple(runs),
        num_runs=len(runs),
        suite_hash=suite_hash,
    )


def _make_delta(
    *,
    acceptance_rate: float = 0.7,
    output_tok_per_sec: float = 100.0,
    reset_detected: bool = False,
    acceptance_available: bool = True,
) -> MetricsDelta:
    return MetricsDelta(
        reset_detected=reset_detected,
        acceptance_available=acceptance_available,
        drafted_tokens=1000.0,
        accepted_tokens=700.0,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=4.67,
        tpot_ms=10.0,
        output_tok_per_sec=output_tok_per_sec,
    )


def _pcfg(*, acc_pp: float = 1.0, throughput_pct: float = 2.0) -> PromotionConfig:
    return PromotionConfig(
        min_acceptance_delta_pp=acc_pp,
        min_throughput_delta_pct=throughput_pct,
    )


#: Completion tokens per synthetic request.  Large enough that any throughput
#: these helpers are asked for is representable exactly.
_RUN_COMPLETION_TOKENS = 1000


def _valid_run(tps: float = 100.0) -> RunResults:
    """A single run with one valid request at *exactly* ``tps`` tokens/second.

    The gating throughput statistic is the mean of these per-repeat figures, so
    the helper has to reproduce the requested rate exactly.  The previous form
    fixed latency at 0.1 s and rounded completion tokens to a whole number,
    which quantised every request to the nearest 10 tok/s -- fine when the
    gate read throughput from the injected Prometheus delta, silently wrong
    now that it reads it from here.
    """
    if tps <= 0:
        return _make_run([
            _make_request_result("same output", latency_s=1.0, completion_tokens=0),
        ])
    return _make_run([
        _make_request_result(
            "same output",
            latency_s=_RUN_COMPLETION_TOKENS / tps,
            completion_tokens=_RUN_COMPLETION_TOKENS,
        ),
    ])


def _valid_runs_with_tps(tps: float, count: int = 3) -> ReplayResult:
    """ReplayResult with N valid runs at the given throughput."""
    runs = [_valid_run(tps=tps) for _ in range(count)]
    return _make_replay(runs)


def _assert_unmeasured_deltas(decision: Decision) -> None:
    assert decision.acceptance_delta_pp is None
    assert decision.throughput_delta_pct is None
    assert decision.to_dict()["acceptance_delta_pp"] is None
    assert decision.to_dict()["throughput_delta_pct"] is None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_promote_when_both_thresholds_clear() -> None:
    """PROMOTE when acceptance delta >= threshold AND throughput delta >= threshold."""
    stock_metrics = _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0)
    cand_metrics = _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0)

    stock_replay = _valid_runs_with_tps(100.0)
    cand_replay = _valid_runs_with_tps(110.0)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )
    assert dec.verdict == Verdict.PROMOTE
    assert dec.reason == Reason.BOTH_THRESHOLDS_MET
    assert dec.acceptance_delta_pp is not None
    assert dec.throughput_delta_pct is not None
    assert abs(dec.acceptance_delta_pp - 5.0) < 0.01
    assert abs(dec.throughput_delta_pct - 10.0) < 0.01


def test_reject_when_only_acceptance_clears() -> None:
    """REJECT when acceptance clears but throughput does not."""
    stock_metrics = _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0)
    cand_metrics = _make_delta(acceptance_rate=0.65, output_tok_per_sec=101.0)

    stock_replay = _valid_runs_with_tps(100.0)
    cand_replay = _valid_runs_with_tps(101.0)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(acc_pp=1.0, throughput_pct=5.0),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.THROUGHPUT_BELOW_THRESHOLD


def test_reject_when_only_throughput_clears() -> None:
    """REJECT when throughput clears but acceptance does not."""
    stock_metrics = _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0)
    cand_metrics = _make_delta(acceptance_rate=0.602, output_tok_per_sec=110.0)

    stock_replay = _valid_runs_with_tps(100.0)
    cand_replay = _valid_runs_with_tps(110.0)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.ACCEPTANCE_BELOW_THRESHOLD


def test_zero_stock_throughput_is_unmeasured_not_a_threshold_miss() -> None:
    """A zero denominator cannot honestly produce a throughput delta.

    The denominator that matters is the *gating* statistic's -- the replayed
    per-repeat stock throughput -- not the Prometheus window's.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.7, output_tok_per_sec=100.0),
        _valid_runs_with_tps(0.0),
        _valid_runs_with_tps(100.0),
        _pcfg(),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.THROUGHPUT_UNAVAILABLE
    _assert_unmeasured_deltas(dec)


def test_unusable_prometheus_throughput_does_not_veto_a_good_candidate() -> None:
    """The diagnostic statistic is diagnostic: a zero there is not a rejection.

    Acceptance still comes from the Prometheus counters, so a genuinely broken
    scrape is caught by ``ACCEPTANCE_UNAVAILABLE``.  A missing *decode-time*
    series is a reporting gap in a number that no longer gates.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60, output_tok_per_sec=0.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=0.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        PromotionConfig(),
    )

    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET
    assert dec.throughput_delta_pct == pytest.approx(0.0)
    # The unmeasurable diagnostic is reported as unmeasured, not as a zero.
    assert dec.prometheus_throughput_delta_pct is None
    assert dec.throughput_statistic_gap_pp is None
    assert dec.to_dict()["prometheus_throughput_delta_pct"] is None


def test_reject_on_counter_reset() -> None:
    """REJECT when counter reset is detected in stock metrics."""
    stock_metrics = _make_delta(reset_detected=True)
    cand_metrics = _make_delta()

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.COUNTER_RESET
    _assert_unmeasured_deltas(dec)


def test_reject_on_candidate_counter_reset() -> None:
    """REJECT when counter reset is detected in candidate metrics."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta(reset_detected=True)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.COUNTER_RESET
    _assert_unmeasured_deltas(dec)


def test_reject_on_acceptance_unavailable() -> None:
    """REJECT when acceptance counters are unavailable."""
    stock_metrics = _make_delta(acceptance_available=True)
    cand_metrics = _make_delta(acceptance_available=False)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.ACCEPTANCE_UNAVAILABLE
    _assert_unmeasured_deltas(dec)


def test_reject_on_high_invalid_rate() -> None:
    """REJECT when invalid rate exceeds threshold."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()

    # Stock is clean
    stock_replay = _valid_runs_with_tps(100.0)

    # Candidate has high invalid rate (100% invalid)
    invalid_runs = [
        _make_run([_make_request_result("", valid=False, error="error")])
        for _ in range(3)
    ]
    cand_replay = _make_replay(invalid_runs)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.HIGH_INVALID_RATE
    _assert_unmeasured_deltas(dec)


def test_reject_on_too_few_repeats() -> None:
    """REJECT when fewer than 3 repeats."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()

    stock_replay = _make_replay([_valid_run() for _ in range(2)])
    cand_replay = _make_replay([_valid_run() for _ in range(3)])

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.TOO_FEW_REPEATS
    _assert_unmeasured_deltas(dec)


def test_reject_on_output_mismatch() -> None:
    """REJECT when stock and candidate outputs differ."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()

    stock_run = _make_run([_make_request_result("same output")])
    cand_run = _make_run([_make_request_result("different output")])

    stock_replay = _make_replay([stock_run for _ in range(3)])
    cand_replay = _make_replay([cand_run for _ in range(3)])

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.OUTPUT_MISMATCH
    _assert_unmeasured_deltas(dec)


def test_decision_has_per_repeat_data() -> None:
    """Decision includes per-repeat summaries."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()

    runs = [_valid_run() for _ in range(3)]
    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _make_replay(runs),
        _make_replay(runs),
        _pcfg(acc_pp=0.0, throughput_pct=0.0),
    )
    assert len(dec.per_repeat) == 3
    for i, pr in enumerate(dec.per_repeat):
        assert pr.repeat_index == i


def test_decision_to_dict() -> None:
    """Decision.to_dict produces expected structure."""
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()

    runs = [_valid_run() for _ in range(3)]
    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _make_replay(runs),
        _make_replay(runs),
        _pcfg(),
    )
    d = dec.to_dict()
    assert d["verdict"] == dec.verdict.value
    assert d["reason"] == dec.reason.value
    assert d["num_repeats"] == 3
    assert len(d["per_repeat"]) == 3


def test_reject_default_when_neither_clears() -> None:
    """Default REJECT when neither threshold is met."""
    stock_metrics = _make_delta(acceptance_rate=0.7, output_tok_per_sec=100.0)
    cand_metrics = _make_delta(acceptance_rate=0.701, output_tok_per_sec=100.5)

    stock_replay = _valid_runs_with_tps(100.0)
    cand_replay = _valid_runs_with_tps(100.5)

    dec = decide_promotion(
        stock_metrics, cand_metrics,
        stock_replay, cand_replay,
        _pcfg(),
    )
    assert dec.verdict == Verdict.REJECT


def test_decision_is_frozen() -> None:
    """Decision instances are immutable."""
    import dataclasses
    stock_metrics = _make_delta()
    cand_metrics = _make_delta()
    dec = decide_promotion(
        stock_metrics, cand_metrics,
        _make_replay([_valid_run() for _ in range(3)]),
        _make_replay([_valid_run() for _ in range(3)]),
        _pcfg(),
    )
    assert dataclasses.is_dataclass(dec)


# ---------------------------------------------------------------------------
# Provenance: the report must not misrepresent how much was measured
# ---------------------------------------------------------------------------

def test_num_repeats_always_matches_the_reported_per_repeat_rows() -> None:
    """`num_repeats` is a claim about the run and must be backed by rows.

    The shipped bug persisted `num_repeats: 0, per_repeat: []` on the
    acceptance-unavailable path even though three suite passes had run, which
    made a measurement failure indistinguishable from a benchmark that never
    executed.
    """
    for stock_metrics, cand_metrics in (
        (_make_delta(), _make_delta(acceptance_available=False)),
        (_make_delta(), _make_delta(reset_detected=True)),
        (_make_delta(), _make_delta()),
    ):
        dec = decide_promotion(
            stock_metrics, cand_metrics,
            _valid_runs_with_tps(100.0),
            _valid_runs_with_tps(110.0),
            _pcfg(acc_pp=0.0, throughput_pct=0.0),
        )
        assert dec.num_repeats == len(dec.per_repeat), dec.reason
        assert dec.num_repeats == 3, dec.reason


def test_zero_sample_benchmark_can_never_promote() -> None:
    """An empty replay must fail closed, not sail through on metric deltas."""
    empty = _make_replay([])

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.99, output_tok_per_sec=999.0),
        empty,
        empty,
        _pcfg(acc_pp=0.0, throughput_pct=0.0),
    )

    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.TOO_FEW_REPEATS
    assert dec.num_repeats == 0
    assert dec.per_repeat == ()
    _assert_unmeasured_deltas(dec)


def test_a_genuine_comparison_always_reports_both_deltas() -> None:
    """Any threshold verdict must carry the numbers it was based on."""
    for cand_tps, cand_acc in ((110.0, 0.70), (90.0, 0.70), (110.0, 0.50)):
        dec = decide_promotion(
            _make_delta(acceptance_rate=0.60, output_tok_per_sec=100.0),
            _make_delta(acceptance_rate=cand_acc, output_tok_per_sec=cand_tps),
            _valid_runs_with_tps(100.0),
            _valid_runs_with_tps(cand_tps),
            _pcfg(acc_pp=0.0, throughput_pct=0.0),
        )
        assert dec.reason in {
            Reason.BOTH_THRESHOLDS_MET,
            Reason.ACCEPTANCE_BELOW_THRESHOLD,
            Reason.THROUGHPUT_BELOW_THRESHOLD,
        }
        assert dec.acceptance_delta_pp is not None
        assert dec.throughput_delta_pct is not None
        assert dec.num_repeats > 0


# ---------------------------------------------------------------------------
# Shipped defaults: the gate must discriminate improvement from timing noise
# ---------------------------------------------------------------------------

#: Job 368670's measured arms, as recorded in its decision.json.  Acceptance was
#: byte-identical between arms (1155 drafted / 730 accepted on both), and the
#: throughput delta was +0.96% -- inside the run's own 1.43% standard error, and
#: it flips to -0.78% if the first scored repeat is dropped.
_J368670_ACCEPTANCE = 730.0 / 1155.0
_J368670_STOCK_TPS = 82.92028682232012
_J368670_CANDIDATE_TPS = 83.71705649227111


def _decide_with_shipped_defaults(
    *,
    stock_acc: float,
    cand_acc: float,
    stock_tps: float,
    cand_tps: float,
) -> Decision:
    """Run the real gate under the real shipped PromotionConfig defaults."""
    return decide_promotion(
        _make_delta(acceptance_rate=stock_acc, output_tok_per_sec=stock_tps),
        _make_delta(acceptance_rate=cand_acc, output_tok_per_sec=cand_tps),
        _valid_runs_with_tps(stock_tps),
        _valid_runs_with_tps(cand_tps),
        PromotionConfig(),
    )


def test_shipped_defaults_are_the_noise_derived_ones() -> None:
    """Pin the defaults the rest of these tests reason about."""
    pcfg = PromotionConfig()
    assert pcfg.min_acceptance_delta_pp == 1.0
    assert pcfg.min_throughput_delta_pct == -2.0


def test_job_368670_marginal_candidate_is_rejected_under_shipped_defaults() -> None:
    """The regression that started this: a within-noise win must not promote.

    Under the 0.0/0.0 gate this exact measurement returned ``promote`` with
    ``both_thresholds_met``.  Acceptance did not move at all, so there is no
    evidence the candidate draft head is better -- only that the clock wobbled.
    """
    dec = _decide_with_shipped_defaults(
        stock_acc=_J368670_ACCEPTANCE,
        cand_acc=_J368670_ACCEPTANCE,
        stock_tps=_J368670_STOCK_TPS,
        cand_tps=_J368670_CANDIDATE_TPS,
    )

    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.ACCEPTANCE_BELOW_THRESHOLD
    assert dec.acceptance_delta_pp == 0.0
    assert dec.throughput_delta_pct is not None
    assert 0.9 < dec.throughput_delta_pct < 1.0


def test_a_zeroed_gate_would_still_have_promoted_job_368670() -> None:
    """Guard the premise: the reject above comes from the thresholds, not luck."""
    dec = decide_promotion(
        _make_delta(
            acceptance_rate=_J368670_ACCEPTANCE, output_tok_per_sec=_J368670_STOCK_TPS
        ),
        _make_delta(
            acceptance_rate=_J368670_ACCEPTANCE, output_tok_per_sec=_J368670_CANDIDATE_TPS
        ),
        _valid_runs_with_tps(_J368670_STOCK_TPS),
        _valid_runs_with_tps(_J368670_CANDIDATE_TPS),
        _pcfg(acc_pp=0.0, throughput_pct=0.0),
    )

    assert dec.verdict == Verdict.PROMOTE
    assert dec.reason == Reason.BOTH_THRESHOLDS_MET


def test_sub_threshold_acceptance_gains_are_rejected_under_shipped_defaults() -> None:
    """Anything short of a full point of acceptance fails closed.

    0.087 pp is the counter quantum -- a single accepted token out of the ~1155
    drafted in a suite pass -- and 0.99 pp sits just under the bar.  Neither is
    evidence of a better head, so neither ships, however good the clock looks.
    """
    for gain_pp in (0.0, 0.087, 0.5, 0.99):
        dec = _decide_with_shipped_defaults(
            stock_acc=0.60,
            cand_acc=0.60 + gain_pp / 100.0,
            stock_tps=100.0,
            cand_tps=140.0,
        )
        assert dec.verdict == Verdict.REJECT, gain_pp
        assert dec.reason == Reason.ACCEPTANCE_BELOW_THRESHOLD, gain_pp


def test_acceptance_bar_promotes_exactly_at_one_point() -> None:
    """The bar is inclusive: a clean 1.0 pp gain with flat throughput ships."""
    dec = _decide_with_shipped_defaults(
        stock_acc=0.60, cand_acc=0.61, stock_tps=100.0, cand_tps=100.0
    )

    assert dec.verdict == Verdict.PROMOTE
    assert dec.reason == Reason.BOTH_THRESHOLDS_MET
    assert dec.acceptance_delta_pp is not None
    assert abs(dec.acceptance_delta_pp - 1.0) < 1e-9


def test_throughput_guard_tolerates_jitter_but_not_a_regression() -> None:
    """A real acceptance win survives noise-sized slowdowns, not real ones.

    -1.9% is inside the measured jitter band (1.10% standard error at five
    repeats), so it must not veto a candidate that genuinely improved
    acceptance.  -2.1% is past the guard, and -19.2% is job 368648's actual
    un-warmed regression.
    """
    for slowdown_pct, expected in (
        (-1.9, Verdict.PROMOTE),
        (-2.0, Verdict.PROMOTE),
        (-2.1, Verdict.REJECT),
        (-19.2, Verdict.REJECT),
    ):
        dec = _decide_with_shipped_defaults(
            stock_acc=0.60,
            cand_acc=0.65,
            stock_tps=100.0,
            cand_tps=100.0 * (1.0 + slowdown_pct / 100.0),
        )
        assert dec.verdict == expected, slowdown_pct
        if expected is Verdict.REJECT:
            assert dec.reason == Reason.THROUGHPUT_BELOW_THRESHOLD, slowdown_pct


def test_acceptance_is_checked_before_throughput_under_shipped_defaults() -> None:
    """When both bars fail, the reported reason names the promotion criterion."""
    dec = _decide_with_shipped_defaults(
        stock_acc=0.60, cand_acc=0.60, stock_tps=100.0, cand_tps=50.0
    )

    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.ACCEPTANCE_BELOW_THRESHOLD
    # The throughput number is still reported so the operator sees both facts.
    assert dec.throughput_delta_pct is not None
    assert abs(dec.throughput_delta_pct + 50.0) < 1e-9


# ---------------------------------------------------------------------------
# One statistic gates, and the record says which
# ---------------------------------------------------------------------------

#: Job 368689's five scored repeats, verbatim from its ``decision.json``
#: (``log_artifacts/live-idle-thresholds-20260730T213537Z``).
_J368689_STOCK_PER_REPEAT = (
    76.34386880491071,
    76.62329517682556,
    76.52287914081516,
    70.62018912755627,
    73.71546413567889,
)
_J368689_CANDIDATE_PER_REPEAT = (
    74.66176429606165,
    75.091991116685,
    74.66711484793703,
    71.74390864075151,
    74.85118171837253,
)
#: The same run's Prometheus decode-time window: 1315 generated tokens over
#: 15.785 s (stock) and 16.304 s (candidate) of ``request_decode_time_seconds``.
#: A different denominator from the replay figures above -- decode only, no
#: prefill and no client time -- hence a different, larger, delta.
_J368689_STOCK_PROM_TPS = 83.3062096100477
_J368689_CANDIDATE_PROM_TPS = 80.65669725058713
#: Both arms produced byte-identical spec_decode counters: 1925 drafted,
#: 915 accepted, so acceptance did not move at all.
_J368689_ACCEPTANCE = 915.0 / 1925.0


def _replay_with_per_repeat_tps(values: tuple[float, ...]) -> ReplayResult:
    return _make_replay([_valid_run(tps=v) for v in values])


def _decide_368689(
    *,
    stock_acc: float = _J368689_ACCEPTANCE,
    cand_acc: float = _J368689_ACCEPTANCE,
    pcfg: PromotionConfig | None = None,
) -> Decision:
    """Replay job 368689's real measurement through the real gate."""
    return decide_promotion(
        _make_delta(
            acceptance_rate=stock_acc, output_tok_per_sec=_J368689_STOCK_PROM_TPS
        ),
        _make_delta(
            acceptance_rate=cand_acc, output_tok_per_sec=_J368689_CANDIDATE_PROM_TPS
        ),
        _replay_with_per_repeat_tps(_J368689_STOCK_PER_REPEAT),
        _replay_with_per_repeat_tps(_J368689_CANDIDATE_PER_REPEAT),
        pcfg if pcfg is not None else PromotionConfig(),
        warmup_repeats=1,
    )


def test_decision_names_the_statistic_that_gates() -> None:
    """The record must not leave a reader guessing which number is the bar."""
    dec = _decide_368689()

    assert dec.throughput_statistic == GATING_THROUGHPUT_STATISTIC
    assert dec.throughput_statistic == "replay_per_repeat_mean"
    assert dec.to_dict()["throughput_statistic"] == "replay_per_repeat_mean"


def test_reported_averages_are_exactly_the_per_repeat_means() -> None:
    """``*_avg_tok_per_sec`` must be reproducible by hand from ``per_repeat``.

    Job 368689's record was not: its averages came from the Prometheus window
    while its ``per_repeat`` array came from replay timing, so averaging the
    published column did not reproduce the published mean.  That is exactly
    how the two statistics got confused.
    """
    dec = _decide_368689()

    stock_column = [r.stock_tok_per_sec for r in dec.per_repeat]
    cand_column = [r.candidate_tok_per_sec for r in dec.per_repeat]

    assert stock_column == pytest.approx(list(_J368689_STOCK_PER_REPEAT))
    assert cand_column == pytest.approx(list(_J368689_CANDIDATE_PER_REPEAT))
    assert dec.stock_avg_tok_per_sec == pytest.approx(
        sum(stock_column) / len(stock_column)
    )
    assert dec.candidate_avg_tok_per_sec == pytest.approx(
        sum(cand_column) / len(cand_column)
    )
    # ...and the gating delta is the delta of those two published means.
    assert dec.throughput_delta_pct == pytest.approx(
        (dec.candidate_avg_tok_per_sec - dec.stock_avg_tok_per_sec)
        / dec.stock_avg_tok_per_sec
        * 100.0
    )


def test_job_368689_reproduces_both_statistics_and_their_gap() -> None:
    """Pin the real numbers, including the disagreement between them."""
    dec = _decide_368689()

    assert dec.num_repeats == 5
    assert dec.warmup_repeats == 1
    # Gating statistic: the replayed per-repeat means.
    assert dec.stock_avg_tok_per_sec == pytest.approx(74.7651, abs=1e-3)
    assert dec.candidate_avg_tok_per_sec == pytest.approx(74.2032, abs=1e-3)
    assert dec.throughput_delta_pct == pytest.approx(-0.7516, abs=1e-3)
    # Diagnostic statistic: the Prometheus decode-time window.
    assert dec.stock_prometheus_decode_tok_per_sec == pytest.approx(
        _J368689_STOCK_PROM_TPS
    )
    assert dec.candidate_prometheus_decode_tok_per_sec == pytest.approx(
        _J368689_CANDIDATE_PROM_TPS
    )
    assert dec.prometheus_throughput_delta_pct == pytest.approx(-3.1805, abs=1e-3)
    # The gap is published, not left for a reader to rediscover by hand.
    assert dec.throughput_statistic_gap_pp == pytest.approx(2.4289, abs=1e-3)
    assert dec.to_dict()["throughput_statistic_gap_pp"] == pytest.approx(
        2.4289, abs=1e-3
    )


def test_job_368689_rejects_on_acceptance_under_shipped_defaults() -> None:
    """The verdict the live run actually reached, for the reason it reached it."""
    dec = _decide_368689()

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD
    assert dec.acceptance_delta_pp == pytest.approx(0.0)


def test_job_368689_throughput_guard_follows_the_replay_statistic() -> None:
    """The crux: the two statistics fall on opposite sides of the -2.0 bar.

    Job 368689's replay delta is -0.75% (inside the guard) while its Prometheus
    delta is -3.18% (outside it).  With acceptance cleared so the throughput
    guard is the deciding bar, the verdict must follow the statistic the
    threshold was calibrated on -- the replay one -- and promote.
    """
    dec = _decide_368689(
        stock_acc=_J368689_ACCEPTANCE,
        cand_acc=_J368689_ACCEPTANCE + 0.02,
    )

    assert dec.throughput_delta_pct is not None
    assert dec.prometheus_throughput_delta_pct is not None
    # Precondition: the two really do straddle the shipped threshold.
    assert dec.throughput_delta_pct > PromotionConfig().min_throughput_delta_pct
    assert dec.prometheus_throughput_delta_pct < PromotionConfig().min_throughput_delta_pct

    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET


def test_a_real_regression_is_caught_on_either_statistic() -> None:
    """The guard is not weakened by the switch.

    Job 368648's un-warmed candidate arm measured -19.2% on the Prometheus
    window and -17.5% on the replay per-repeat means.  Both are an order of
    magnitude past the -2.0 bar, so the regression this gate exists to catch
    is caught whichever statistic gates.
    """
    stock = (59.6422,) * 3
    candidate = (49.2089,) * 3
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60, output_tok_per_sec=85.1899),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=68.8334),
        _replay_with_per_repeat_tps(stock),
        _replay_with_per_repeat_tps(candidate),
        PromotionConfig(),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.THROUGHPUT_BELOW_THRESHOLD
    assert dec.throughput_delta_pct == pytest.approx(-17.4931, abs=1e-3)
    assert dec.prometheus_throughput_delta_pct == pytest.approx(-19.2000, abs=1e-3)


def test_job_368670_agreed_and_still_rejects_on_acceptance() -> None:
    """The calibration run: there the two statistics agreed to 0.001 pp.

    368670 is where ``min_throughput_delta_pct`` came from -- its pooled sd of
    1.338 tok/s sits on a 76.31 tok/s mean, which is the *replay* mean, not its
    82.92 tok/s Prometheus figure.  That provenance is why the replay statistic
    is the one that gates.
    """
    stock = (74.2624, 76.9898, 77.6637)
    candidate = (77.6678, 76.5205, 76.9296)
    dec = decide_promotion(
        _make_delta(acceptance_rate=_J368670_ACCEPTANCE, output_tok_per_sec=82.92028682232012),
        _make_delta(acceptance_rate=_J368670_ACCEPTANCE, output_tok_per_sec=83.71705649227111),
        _replay_with_per_repeat_tps(stock),
        _replay_with_per_repeat_tps(candidate),
        PromotionConfig(),
    )

    assert dec.stock_avg_tok_per_sec == pytest.approx(76.3053, abs=1e-3)
    assert dec.throughput_delta_pct == pytest.approx(0.9620, abs=1e-3)
    assert dec.prometheus_throughput_delta_pct == pytest.approx(0.9609, abs=1e-3)
    # The two statistics can agree closely; the gap is not a constant offset.
    assert dec.throughput_statistic_gap_pp == pytest.approx(0.0, abs=0.01)

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD


# ---------------------------------------------------------------------------
# Output correctness: divergence position, not string equality
# ---------------------------------------------------------------------------


def _tokens(prefix: int, tail: str, length: int) -> tuple[str, ...]:
    """A generation that agrees for *prefix* tokens then says *tail* forever."""
    return tuple(f"t{i}" for i in range(prefix)) + (tail,) * (length - prefix)


def _correctness_pair(
    *,
    diverge_at: int | None,
    length: int = 1600,
    contexts: int = 1,
) -> tuple[ReplayResult, ReplayResult]:
    """Two single-repeat correctness passes that part at *diverge_at*."""
    stock: list[RequestResult] = []
    candidate: list[RequestResult] = []
    for index in range(contexts):
        shared = _tokens(length, "x", length)
        stock.append(
            _make_request_result(
                "answer",
                output_tokens=shared,
                context_hash=f"ctx-{index}",
            )
        )
        candidate.append(
            _make_request_result(
                "answer",
                output_tokens=(
                    shared if diverge_at is None else _tokens(diverge_at, "OTHER", length)
                ),
                context_hash=f"ctx-{index}",
            )
        )
    return (
        _make_replay([_make_run(stock)]),
        _make_replay([_make_run(candidate)]),
    )


def _decide_with_correctness(
    *,
    diverge_at: int | None,
    min_divergence_token_index: int = 16,
    contexts: int = 1,
    correctness_max_tokens: int | None = None,
) -> Decision:
    stock_correctness, cand_correctness = _correctness_pair(
        diverge_at=diverge_at, contexts=contexts
    )
    return decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.65),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        PromotionConfig(
            min_acceptance_delta_pp=1.0,
            min_throughput_delta_pct=2.0,
            min_divergence_token_index=min_divergence_token_index,
        ),
        stock_correctness=stock_correctness,
        candidate_correctness=cand_correctness,
        correctness_max_tokens=correctness_max_tokens,
    )


def test_first_divergence_reports_a_token_offset_not_a_boolean() -> None:
    stock = _make_request_result("a", output_tokens=("A", "B", "C", "D"))
    candidate = _make_request_result("a", output_tokens=("A", "B", "Z", "D"))

    index, basis, s_len, c_len = first_divergence(stock, candidate)

    assert index == 2
    assert basis is DivergenceBasis.TOKEN
    assert (s_len, c_len) == (4, 4)


def test_first_divergence_falls_back_to_characters_without_token_capture() -> None:
    index, basis, _, _ = first_divergence(
        _make_request_result("hello world"),
        _make_request_result("hello WORLD"),
    )

    assert index == 6
    assert basis is DivergenceBasis.CHARACTER


def test_first_divergence_is_none_only_for_identical_generations() -> None:
    same = ("A", "B", "C")
    assert first_divergence(
        _make_request_result("x", output_tokens=same),
        _make_request_result("x", output_tokens=same),
    )[0] is None
    # A prefix is not a match: stopping early is itself a difference.
    assert first_divergence(
        _make_request_result("x", output_tokens=same),
        _make_request_result("x", output_tokens=same[:2]),
    )[0] == 2


def test_candidate_diverging_at_token_3_is_rejected() -> None:
    """Parting almost immediately is what a broken drafter looks like."""
    dec = _decide_with_correctness(diverge_at=3)

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH
    assert dec.output_early_divergences == 1


def test_candidate_diverging_at_token_900_of_1600_is_not_rejected_for_it() -> None:
    """Float non-determinism deep in a long answer is not a defect."""
    dec = _decide_with_correctness(diverge_at=900)

    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET
    # The divergence is still recorded -- it is simply not disqualifying.
    assert dec.output_total_divergences == 1
    assert dec.output_early_divergences == 0


def test_first_divergence_offsets_are_persisted_for_every_diverging_context() -> None:
    dec = _decide_with_correctness(diverge_at=900, contexts=3)

    record = dec.to_dict()
    assert record["output_total_divergences"] == 3
    assert record["output_early_divergences"] == 0
    assert record["min_divergence_token_index"] == 16
    assert [d["first_divergence_index"] for d in record["output_divergences"]] == [
        900,
        900,
        900,
    ]
    assert {d["context_hash"] for d in record["output_divergences"]} == {
        "ctx-0",
        "ctx-1",
        "ctx-2",
    }
    assert {d["basis"] for d in record["output_divergences"]} == {"token"}
    assert {d["early"] for d in record["output_divergences"]} == {False}


def test_identical_generations_record_no_divergences_at_all() -> None:
    dec = _decide_with_correctness(diverge_at=None, contexts=3)

    assert dec.verdict is Verdict.PROMOTE
    assert dec.output_divergences == ()


def test_divergence_threshold_is_configurable() -> None:
    """The same measurement flips verdict when the bar moves past it."""
    assert _decide_with_correctness(
        diverge_at=20, min_divergence_token_index=16
    ).verdict is Verdict.PROMOTE
    assert _decide_with_correctness(
        diverge_at=20, min_divergence_token_index=64
    ).reason is Reason.OUTPUT_MISMATCH


def test_a_saturated_threshold_is_named_in_the_decision_record() -> None:
    """Job 369373: the rejection was constructed by the config, not measured.

    Threshold 128 against a correctness cap of 128 makes every divergence at
    every observable offset early, so ``OUTPUT_MISMATCH`` here says nothing
    about the drafter -- and a reader of ``decision.json`` must be able to see
    that without re-deriving it from two numbers.
    """
    dec = _decide_with_correctness(
        diverge_at=24, min_divergence_token_index=128, correctness_max_tokens=128
    )

    assert dec.reason is Reason.OUTPUT_MISMATCH
    assert dec.divergence_criterion is DivergenceCriterion.SATURATED
    assert dec.to_dict()["divergence_criterion"] == "saturated"


def test_a_disabled_threshold_is_named_in_the_decision_record() -> None:
    """The other end: no divergence can ever be early, so none can gate."""
    dec = _decide_with_correctness(
        diverge_at=1, min_divergence_token_index=0, correctness_max_tokens=128
    )

    assert dec.output_early_divergences == 0
    assert dec.divergence_criterion is DivergenceCriterion.DISABLED
    assert dec.to_dict()["divergence_criterion"] == "disabled"


def test_the_calibrated_default_relationship_is_recorded_as_such() -> None:
    dec = _decide_with_correctness(
        diverge_at=24, min_divergence_token_index=16, correctness_max_tokens=128
    )

    assert dec.divergence_criterion is DivergenceCriterion.CALIBRATED
    assert dec.to_dict()["divergence_criterion"] == "calibrated"


def test_correctness_pass_is_compared_instead_of_the_throughput_pass() -> None:
    """The batched throughput pass must not be able to veto on its own text."""
    stock_correctness, cand_correctness = _correctness_pair(diverge_at=None)
    # Throughput replays whose *text* differs everywhere -- which is exactly
    # what job 369005 saw at concurrency 8 -- and which used to reject.
    stock_throughput = _make_replay([_make_run([_make_request_result("aaa")])] * 3)
    cand_throughput = _make_replay([_make_run([_make_request_result("bbb")])] * 3)

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.65),
        stock_throughput,
        cand_throughput,
        _pcfg(acc_pp=1.0, throughput_pct=-100.0),
        stock_correctness=stock_correctness,
        candidate_correctness=cand_correctness,
    )

    assert dec.verdict is Verdict.PROMOTE
    assert dec.output_divergences == ()


# ---------------------------------------------------------------------------
# Acceptance is an n-repeat measurement
# ---------------------------------------------------------------------------


def test_per_repeat_acceptance_vector_carries_real_variance() -> None:
    """Each repeat's own metric window reaches its own per-repeat row."""
    stock_windows = [_make_delta(acceptance_rate=r) for r in (0.60, 0.62, 0.58)]
    cand_windows = [_make_delta(acceptance_rate=r) for r in (0.70, 0.66, 0.68)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.68),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert [r.stock_acceptance_rate for r in dec.per_repeat] == [0.60, 0.62, 0.58]
    assert [r.candidate_acceptance_rate for r in dec.per_repeat] == [0.70, 0.66, 0.68]
    # Not one scalar stamped N times: the rows actually differ.
    assert len({r.stock_acceptance_rate for r in dec.per_repeat}) == 3


def test_avg_acceptance_is_the_mean_of_the_column_beside_it() -> None:
    stock_windows = [_make_delta(acceptance_rate=r) for r in (0.60, 0.62, 0.58)]
    cand_windows = [_make_delta(acceptance_rate=r) for r in (0.70, 0.66, 0.68)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.0),
        _make_delta(acceptance_rate=0.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert dec.stock_avg_acceptance == pytest.approx(0.60)
    assert dec.candidate_avg_acceptance == pytest.approx(0.68)
    assert dec.acceptance_delta_pp == pytest.approx(8.0)
    assert dec.acceptance_statistic == GATING_ACCEPTANCE_STATISTIC


def test_acceptance_dispersion_is_published_so_the_bar_can_be_recalibrated() -> None:
    stock_windows = [_make_delta(acceptance_rate=r) for r in (0.60, 0.62, 0.58)]
    cand_windows = [_make_delta(acceptance_rate=r) for r in (0.70, 0.70, 0.70)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert dec.stock_acceptance_stdev == pytest.approx(0.02)
    assert dec.candidate_acceptance_stdev == 0.0
    record = dec.to_dict()
    assert record["stock_acceptance_stdev"] == pytest.approx(0.02)
    assert record["candidate_acceptance_stdev"] == 0.0


def test_without_per_repeat_windows_the_dispersion_is_honestly_zero() -> None:
    """One pooled sample has no dispersion, and must not pretend otherwise."""
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.65),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert {r.stock_acceptance_rate for r in dec.per_repeat} == {0.60}
    assert dec.stock_acceptance_stdev == 0.0
    assert dec.stock_avg_acceptance == pytest.approx(0.60)


def test_a_repeat_window_without_acceptance_falls_back_to_the_pooled_rate() -> None:
    windows = [
        _make_delta(acceptance_rate=0.62),
        _make_delta(acceptance_rate=0.0, acceptance_available=False),
        _make_delta(acceptance_rate=0.58),
    ]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=windows,
    )

    # The idle window reports the arm's pooled rate rather than a measured 0%.
    assert [r.stock_acceptance_rate for r in dec.per_repeat] == [0.62, 0.60, 0.58]


# ---------------------------------------------------------------------------
# Dispersion basis: a zero standard error is not a good measurement
# ---------------------------------------------------------------------------

def _replay_with_tps_series(series: list[float]) -> ReplayResult:
    """A replay whose per-repeat throughput follows *series* exactly."""
    return _make_replay([_valid_run(tps=t) for t in series])


def test_degenerate_acceptance_is_labelled_not_reported_as_a_tight_measurement() -> None:
    """Five bit-identical repeats must not read as an infinitely good measurement.

    This is job 369162 (gpt-oss-20b) in miniature: both arms returned the same
    acceptance rate on every repeat, so ``*_acceptance_stdev`` is 0.0.  A
    consumer computing ``min_acceptance_delta_pp / standard_error`` on that
    would get infinity.  The record has to say the standard error does not
    exist instead.
    """
    stock_windows = [_make_delta(acceptance_rate=0.3558) for _ in range(5)]
    cand_windows = [_make_delta(acceptance_rate=0.3501) for _ in range(5)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.3558),
        _make_delta(acceptance_rate=0.3501),
        _valid_runs_with_tps(100.0, count=5),
        _valid_runs_with_tps(110.0, count=5),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert dec.stock_acceptance_stdev == 0.0
    assert dec.candidate_acceptance_stdev == 0.0
    assert dec.acceptance_dispersion is DispersionBasis.DEGENERATE
    assert dec.acceptance_delta_standard_error_pp is None

    record = dec.to_dict()
    assert record["acceptance_dispersion"] == "degenerate"
    assert record["acceptance_delta_standard_error_pp"] is None


def test_varying_acceptance_is_labelled_measured_and_carries_a_standard_error() -> None:
    """Job 369161's shape: the column actually varies, so the SE means something."""
    stock_windows = [_make_delta(acceptance_rate=r) for r in (0.60, 0.62, 0.58)]
    cand_windows = [_make_delta(acceptance_rate=r) for r in (0.70, 0.66, 0.68)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.68),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert dec.acceptance_dispersion is DispersionBasis.MEASURED
    se = dec.acceptance_delta_standard_error_pp
    assert se is not None
    # sqrt(sd_s^2/n + sd_c^2/n) with sd = 0.02 on both arms, n = 3, in pp.
    assert se == pytest.approx(math.sqrt(2 * 0.02**2 / 3) * 100.0)
    assert dec.to_dict()["acceptance_dispersion"] == "measured"


def test_one_arm_varying_is_still_a_measurement() -> None:
    """A degenerate label needs *both* arms flat; one varying arm gives a real SE."""
    stock_windows = [_make_delta(acceptance_rate=r) for r in (0.60, 0.62, 0.58)]
    cand_windows = [_make_delta(acceptance_rate=0.70) for _ in range(3)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
        stock_repeat_metrics=stock_windows,
        candidate_repeat_metrics=cand_windows,
    )

    assert dec.candidate_acceptance_stdev == 0.0
    assert dec.acceptance_dispersion is DispersionBasis.MEASURED
    assert dec.acceptance_delta_standard_error_pp is not None


def test_throughput_standard_error_is_published_in_percent_of_stock() -> None:
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0, 102.0, 98.0]),
        _replay_with_tps_series([110.0, 108.0, 112.0]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.throughput_dispersion is DispersionBasis.MEASURED
    se = dec.throughput_delta_standard_error_pct
    assert se is not None
    # sd = 2.0 on both arms, n = 3, expressed against the 100 tok/s stock mean.
    assert se == pytest.approx(math.sqrt(2 * 2.0**2 / 3))
    assert dec.to_dict()["throughput_delta_standard_error_pct"] == pytest.approx(se)


def test_a_stubbed_constant_throughput_replay_is_degenerate_not_perfect() -> None:
    """Constant clock readings mean the clock was not read, not that it is noiseless."""
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.throughput_dispersion is DispersionBasis.DEGENERATE
    assert dec.throughput_delta_standard_error_pct is None
    assert dec.to_dict()["throughput_dispersion"] == "degenerate"


# ---------------------------------------------------------------------------
# Warm-up drift: why the repeat count cannot simply be cut
# ---------------------------------------------------------------------------

def test_a_warming_candidate_arm_publishes_a_positive_trend() -> None:
    """The number that says ``sd/sqrt(n)`` is resting on non-exchangeable repeats.

    Modelled on jobs 369161/369162, where the candidate arm (which runs first)
    rose monotonically across all five scored repeats while stock did not.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0, 100.0, 100.0, 100.0, 100.0]),
        _replay_with_tps_series([96.0, 98.0, 100.0, 102.0, 104.0]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.stock_throughput_trend_pct_per_repeat == pytest.approx(0.0)
    # +2 tok/s per repeat on a 100 tok/s mean.
    assert dec.candidate_throughput_trend_pct_per_repeat == pytest.approx(2.0)
    record = dec.to_dict()
    assert record["candidate_throughput_trend_pct_per_repeat"] == pytest.approx(2.0)
    assert record["stock_throughput_trend_pct_per_repeat"] == pytest.approx(0.0)


def test_truncating_a_warming_arm_moves_the_gated_delta_toward_the_reject_bar() -> None:
    """Why ``benchmark_repeats`` was kept at five rather than cut to three.

    The same recorded candidate series, scored over its first three repeats
    instead of all five, reads materially worse -- not because it is noisier
    but because the truncated window is the cold part of a warming arm.  This
    pins the bias as a property of the gate's own arithmetic, so a future
    reduction of ``benchmark_repeats`` cannot be made without this test failing.
    """
    stock_series = [100.0, 100.0, 100.0, 100.0, 100.0]
    candidate_series = [96.0, 98.0, 100.0, 102.0, 104.0]

    full = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series(stock_series),
        _replay_with_tps_series(candidate_series),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )
    truncated = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series(stock_series[:3]),
        _replay_with_tps_series(candidate_series[:3]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert full.throughput_delta_pct == pytest.approx(0.0)
    assert truncated.throughput_delta_pct == pytest.approx(-2.0)
    assert truncated.throughput_delta_pct is not None
    assert full.throughput_delta_pct is not None
    assert truncated.throughput_delta_pct < full.throughput_delta_pct


def test_trend_needs_two_repeats_before_it_means_anything() -> None:
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _make_replay([_valid_run(tps=100.0)]),
        _make_replay([_valid_run(tps=110.0)]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.reason is Reason.TOO_FEW_REPEATS
    assert dec.candidate_throughput_trend_pct_per_repeat is None
    assert dec.throughput_dispersion is DispersionBasis.UNSAMPLED
    assert dec.throughput_delta_standard_error_pct is None
    assert dec.acceptance_dispersion is DispersionBasis.UNSAMPLED


# ---------------------------------------------------------------------------
# Where the warming stops: the measurement ``warmup_repeats`` needs
# ---------------------------------------------------------------------------

def test_a_plateauing_arm_reports_the_repeat_its_warming_stopped_at() -> None:
    """The number ``tuning.warmup_repeats`` has to be argued from.

    A candidate that climbs for two repeats and then settles should report
    ``2``: repeats 2.. are mutually exchangeable, so two extra unscored passes
    would have opened the measurement window warm.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0] * 8),
        _replay_with_tps_series([90.0, 95.0, 100.0, 100.1, 99.9, 100.0, 100.1, 99.9]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.candidate_throughput_flat_from_repeat == 2
    record = dec.to_dict()
    assert record["candidate_throughput_flat_from_repeat"] == 2
    assert record["stock_throughput_flat_from_repeat"] == 0


def test_an_arm_still_warming_at_the_last_repeat_reports_no_flat_index() -> None:
    """The signature jobs 369161/369162 produced, and why they settled nothing.

    Both runs scored five repeats and the candidate trended upward across all
    of them.  ``None`` is the honest answer -- "it had not flattened yet" --
    and it is what says the characterisation run needs more repeats, not that
    warmup should be raised to some guessed value.
    """
    series = [96.0, 98.3, 99.7, 102.2, 104.1]
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0, 100.4, 99.7, 100.2, 99.8]),
        _replay_with_tps_series(series),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.candidate_throughput_flat_from_repeat is None
    assert dec.to_dict()["candidate_throughput_flat_from_repeat"] is None
    # The trend is what says the ``None`` means "still climbing" rather than
    # "too few samples"; the two are published together for exactly that reason.
    trend = dec.candidate_throughput_trend_pct_per_repeat
    assert trend is not None and trend > 0.0


def test_extending_a_warming_run_is_what_finds_the_flat_index() -> None:
    """Why the characterisation run raises ``benchmark_repeats`` rather than warmup.

    The same arm, scored over five repeats and then over ten: five cannot see
    the plateau that ten resolves.  This is the whole payoff of the diagnostic
    mode -- the answer is a property of how long the window is, not of how the
    detector is tuned.
    """
    warming = [90.0, 94.0, 97.0, 99.0, 100.0]
    settled = [100.1, 99.9, 100.0, 100.2, 99.8]
    stock = [100.0, 100.3, 99.8, 100.1, 99.9]

    short = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series(stock),
        _replay_with_tps_series(warming),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )
    long = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series(stock + stock),
        _replay_with_tps_series(warming + settled),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert short.candidate_throughput_flat_from_repeat is None
    assert long.candidate_throughput_flat_from_repeat is not None
    assert long.candidate_throughput_flat_from_repeat > 0


def test_a_perfectly_linear_ramp_never_reads_as_flat() -> None:
    """Zero residual must not be mistaken for zero slope.

    A stubbed arm that ramps exactly has no noise to compare its slope
    against.  Reporting it flat would be the mirror of the bug
    :class:`DispersionBasis` exists to prevent -- a stub reading as a perfect
    measurement.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0] * 5),
        _replay_with_tps_series([96.0, 98.0, 100.0, 102.0, 104.0]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.candidate_throughput_flat_from_repeat is None
    # A constant column is the other exactly-collinear case, and it *is* flat.
    assert dec.stock_throughput_flat_from_repeat == 0


def test_the_flat_index_needs_three_repeats_before_it_means_anything() -> None:
    """Two points fit their own slope exactly, so every pair would read flat."""
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0, 103.0]),
        _replay_with_tps_series([100.0, 106.0]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.num_repeats == 2
    assert dec.candidate_throughput_trend_pct_per_repeat is not None
    assert dec.candidate_throughput_flat_from_repeat is None
    assert dec.stock_throughput_flat_from_repeat is None


def test_the_flat_index_is_the_earliest_qualifying_window_not_the_shortest() -> None:
    """Scanning latest-first would answer ``n - 3`` on almost any column.

    Later suffixes are shorter and easier to call flat by accident, so the
    scan takes the earliest window that clears the bar.  Here every window from
    repeat two on is flat, and two is the answer.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.70),
        _replay_with_tps_series([100.0] * 8),
        _replay_with_tps_series([80.0, 90.0, 100.0, 100.2, 99.8, 100.1, 99.9, 100.0]),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.candidate_throughput_flat_from_repeat == 2
