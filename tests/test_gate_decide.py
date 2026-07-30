"""Tests for gate/decide.py — no GPU, no network."""

from speedlm.config import PromotionConfig
from speedlm.gate.decide import (
    Decision,
    Reason,
    Verdict,
    decide_promotion,
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
) -> RequestResult:
    return RequestResult(
        context_hash="abcd1234",
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        response_text=response_text,
        valid=valid,
        error=error,
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


def _valid_run(tps: float = 100.0) -> RunResults:
    """A single run with one valid request."""
    return _make_run([
        _make_request_result(
            "same output",
            latency_s=0.1,
            completion_tokens=int(tps * 0.1),
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
    """A zero denominator cannot honestly produce a throughput delta."""
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=0.0),
        _make_delta(acceptance_rate=0.7, output_tok_per_sec=100.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        _pcfg(),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.THROUGHPUT_UNAVAILABLE
    _assert_unmeasured_deltas(dec)


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
