"""Tests for gate/decide.py — no GPU, no network."""

import dataclasses
import math

import pytest

from speedlm.config import DivergenceCriterion, PromotionConfig
from speedlm.gate.decide import (
    CUDA_GRAPH_EXECUTION_MODE,
    DIVERGENCE_ALPHA,
    EAGER_EXECUTION_MODE,
    GATING_ACCEPTANCE_CRITERION,
    GATING_ACCEPTANCE_STATISTIC,
    GATING_THROUGHPUT_STATISTIC,
    UNRECORDED_EXECUTION_MODE,
    Decision,
    DecisionError,
    DispersionBasis,
    DivergenceBasis,
    EngineExecution,
    Reason,
    TruncationRegime,
    Verdict,
    classify_truncation,
    decide_promotion,
    divergence_excess_p_value,
    divergence_position_p_value,
    first_divergence,
)
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import (
    ReplayResult,
    RequestResult,
    RunResults,
    _validity_error,
    is_truncated_finish_reason,
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
    finish_reason: str = "",
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
        finish_reason=finish_reason,
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


#: Draft depth these fixtures speak at.  It has to be *a* number because the
#: two acceptance-side statistics are related through it -- ``acceptance_rate ==
#: (mean_accepted_length - 1) / k`` -- and a fixture that pinned
#: ``mean_accepted_length`` to a constant while varying ``acceptance_rate``
#: would describe an engine that cannot exist.  It used to: the constant 4.67
#: sat next to a swept rate, so every arm in this file had an identical accepted
#: length and the gating criterion measured nothing.
FIXTURE_DRAFT_DEPTH = 5


def _make_delta(
    *,
    acceptance_rate: float = 0.7,
    output_tok_per_sec: float = 100.0,
    reset_detected: bool = False,
    acceptance_available: bool = True,
    mean_accepted_length: float | None = None,
    draft_depth: int = FIXTURE_DRAFT_DEPTH,
) -> MetricsDelta:
    """A metrics delta whose two acceptance statistics are mutually consistent.

    *mean_accepted_length* defaults to ``1 + acceptance_rate * draft_depth``,
    which is the identity vLLM's own counters satisfy.  Pass *draft_depth* to
    describe an engine drafting at a different depth -- that is the whole point
    of the criterion -- or *mean_accepted_length* to break the identity
    deliberately.
    """
    if mean_accepted_length is None:
        mean_accepted_length = 1.0 + acceptance_rate * draft_depth
    return MetricsDelta(
        reset_detected=reset_detected,
        acceptance_available=acceptance_available,
        drafted_tokens=1000.0,
        accepted_tokens=700.0,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_length,
        tpot_ms=10.0,
        output_tok_per_sec=output_tok_per_sec,
    )


def _pcfg(
    *,
    acc_pp: float = 1.0,
    throughput_pct: float = 2.0,
    mal: float | None = None,
) -> PromotionConfig:
    """Promotion config whose two acceptance bars describe the same demand.

    *mal* defaults to *acc_pp* converted at :data:`FIXTURE_DRAFT_DEPTH`
    (``delta_mal == k * delta_acc``), so a test that asks for "a 1.0 pp bar" gets
    the bar that means the same thing under the gating criterion.  Pass it
    explicitly to set the two independently.
    """
    if mal is None:
        mal = acc_pp / 100.0 * FIXTURE_DRAFT_DEPTH
    return PromotionConfig(
        min_acceptance_delta_pp=acc_pp,
        min_accepted_length_delta=mal,
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

    # Four contexts rather than one: the criterion is a comparison of rates
    # against a noise floor, so a single disagreeing context is not evidence of
    # anything and no honest criterion can reject on it.  Every context
    # disagreeing from the very first character is.
    stock_run = _make_run(
        [_make_request_result("same output", context_hash=f"ctx-{i}") for i in range(4)]
    )
    cand_run = _make_run(
        [
            _make_request_result("different output", context_hash=f"ctx-{i}")
            for i in range(4)
        ]
    )

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
    assert pcfg.min_accepted_length_delta == 0.05
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


# ---------------------------------------------------------------------------
# The promotion criterion is k-invariant
#
# Every test below is written so that gating on ``acceptance_rate`` -- the
# behaviour these replace -- makes it fail.  The archived measurements are the
# real ones: gpt-oss-20b drafts at k=5 and scores 0.356, Qwen3-8B drafts at k=3
# and scores 0.380, and gpt-oss is nonetheless 28% better per verifier step.
# ---------------------------------------------------------------------------

#: gpt-oss-20b, k=5: acceptance 0.356, i.e. 2.78 tokens per verifier step.
_GPTOSS_DEPTH = 5
_GPTOSS_ACCEPTANCE = 0.356
#: Qwen3-8B, k=3: acceptance 0.380, i.e. 2.14 tokens per verifier step.
_QWEN_DEPTH = 3
_QWEN_ACCEPTANCE = 0.380


def test_the_gating_criterion_is_named_in_the_record() -> None:
    """A reader must never have to infer which acceptance statistic decided."""
    dec = _decide_with_shipped_defaults(
        stock_acc=0.60, cand_acc=0.70, stock_tps=100.0, cand_tps=100.0
    )

    assert dec.acceptance_criterion == GATING_ACCEPTANCE_CRITERION
    assert dec.to_dict()["acceptance_criterion"] == "mean_accepted_length_delta"


def test_a_deeper_draft_chain_that_wins_on_tokens_per_step_is_not_rejected() -> None:
    """The defect, stated as a test: raising k must not veto the speedup.

    gpt-oss's per-position conditional acceptance is still *rising* at position
    5 (0.689 -> 0.715), so k=5 is too shallow.  Going to k=7 drops the
    acceptance *rate* to ~0.31 -- a -4.6 pp delta, four and a half times the old
    1.0 pp bar in the wrong direction -- while raising accepted length from 2.78
    to 3.17 tokens per step, which is the actual speedup.

    Gating on the rate rejects this with ``acceptance_below_threshold``.  That is
    the single best available improvement being thrown away, and it is what this
    test exists to stop.
    """
    stock = _make_delta(
        acceptance_rate=_GPTOSS_ACCEPTANCE,
        draft_depth=_GPTOSS_DEPTH,
        output_tok_per_sec=100.0,
    )
    deeper = _make_delta(acceptance_rate=0.31, draft_depth=7, output_tok_per_sec=100.0)

    # The premise: the rate really did fall, and hard.
    assert deeper.acceptance_rate < stock.acceptance_rate
    assert (deeper.acceptance_rate - stock.acceptance_rate) * 100.0 < -4.0
    # ...while tokens per verifier step really did rise.
    assert deeper.mean_accepted_length > stock.mean_accepted_length

    dec = decide_promotion(
        stock,
        deeper,
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        PromotionConfig(),
    )

    assert dec.verdict == Verdict.PROMOTE
    assert dec.reason == Reason.BOTH_THRESHOLDS_MET
    # The rate delta is still recorded, still negative, and still ignored.
    assert dec.acceptance_delta_pp is not None
    assert dec.acceptance_delta_pp < -4.0
    assert dec.accepted_length_delta == pytest.approx(3.17 - 2.78, abs=1e-9)


def test_the_same_speedup_clears_the_bar_at_every_draft_depth() -> None:
    """One bar, one meaning: 0.10 tokens/step ships at k=3 and at k=5 alike.

    Under the acceptance rate the identical improvement is 3.33 pp at k=3 and
    2.00 pp at k=5 -- the same engine change scored 67% higher for drafting
    shallower.  A single rate threshold therefore cannot mean the same thing at
    two depths, which is what makes it unusable across a k-sweep.
    """
    gain_per_step = 0.10
    verdicts = {}
    for depth, base in ((_QWEN_DEPTH, _QWEN_ACCEPTANCE), (_GPTOSS_DEPTH, _GPTOSS_ACCEPTANCE)):
        stock = _make_delta(
            acceptance_rate=base, draft_depth=depth, output_tok_per_sec=100.0
        )
        candidate = _make_delta(
            acceptance_rate=base,
            draft_depth=depth,
            mean_accepted_length=1.0 + base * depth + gain_per_step,
            output_tok_per_sec=100.0,
        )
        dec = decide_promotion(
            stock,
            candidate,
            _valid_runs_with_tps(100.0),
            _valid_runs_with_tps(100.0),
            PromotionConfig(),
        )
        verdicts[depth] = dec
        assert dec.accepted_length_delta == pytest.approx(gain_per_step)

    assert verdicts[_QWEN_DEPTH].verdict == Verdict.PROMOTE
    assert verdicts[_GPTOSS_DEPTH].verdict == Verdict.PROMOTE


def test_gptoss_at_k5_outranks_qwen_at_k3_on_the_gating_criterion() -> None:
    """The archived cross-model comparison, on the statistic that decides.

    0.356 < 0.380 on the rate; 2.78 > 2.14 on tokens per verifier step.  Both
    numbers are real; only the second is comparable, because the first divides
    by a k that differs between the two rows.
    """
    gptoss = _make_delta(acceptance_rate=_GPTOSS_ACCEPTANCE, draft_depth=_GPTOSS_DEPTH)
    qwen = _make_delta(acceptance_rate=_QWEN_ACCEPTANCE, draft_depth=_QWEN_DEPTH)

    # The rank inversion itself: the two statistics disagree about which head
    # is better, and only one of them is comparable across the two depths.
    assert gptoss.acceptance_rate < qwen.acceptance_rate
    assert gptoss.mean_accepted_length > qwen.mean_accepted_length
    assert gptoss.mean_accepted_length / qwen.mean_accepted_length > 1.25

    # Why: the rate is exactly the accepted length rescaled by that row's own k,
    # so comparing two rows at different k compares two different rescalings.
    for delta, depth in ((gptoss, _GPTOSS_DEPTH), (qwen, _QWEN_DEPTH)):
        assert delta.acceptance_rate == pytest.approx(
            (delta.mean_accepted_length - 1.0) / depth
        )
    # Rescaled onto Qwen's depth, gpt-oss outscores it on the rate too.
    assert (gptoss.mean_accepted_length - 1.0) / _QWEN_DEPTH > qwen.acceptance_rate


def test_the_rate_delta_is_still_recorded_when_it_no_longer_decides() -> None:
    """Backward compatibility: the archived field keeps its name and meaning.

    Every decision in ``/data/ryan.kim/speedlm-runs`` is described by
    ``acceptance_delta_pp`` against ``min_acceptance_delta_pp``.  Both stay in
    the record, computed exactly as before, on every path that reaches the
    deltas -- promote and both threshold rejections alike.
    """
    for cand_acc, expected in ((0.70, Verdict.PROMOTE), (0.60, Verdict.REJECT)):
        dec = _decide_with_shipped_defaults(
            stock_acc=0.60, cand_acc=cand_acc, stock_tps=100.0, cand_tps=100.0
        )
        record = dec.to_dict()
        assert dec.verdict == expected
        assert record["acceptance_delta_pp"] == pytest.approx(
            (cand_acc - 0.60) * 100.0
        )
        assert record["min_acceptance_delta_pp"] == 1.0
        assert record["accepted_length_delta"] == pytest.approx(
            (cand_acc - 0.60) * FIXTURE_DRAFT_DEPTH
        )
        assert record["min_accepted_length_delta"] == 0.05


def test_accepted_length_columns_are_the_mean_of_the_per_repeat_array() -> None:
    """The published averages must be reproducible by hand from the array."""
    dec = _decide_with_shipped_defaults(
        stock_acc=0.60, cand_acc=0.70, stock_tps=100.0, cand_tps=100.0
    )

    assert dec.per_repeat
    stock_column = [r.stock_accepted_length for r in dec.per_repeat]
    cand_column = [r.candidate_accepted_length for r in dec.per_repeat]
    assert dec.stock_avg_accepted_length == pytest.approx(
        sum(stock_column) / len(stock_column)
    )
    assert dec.candidate_avg_accepted_length == pytest.approx(
        sum(cand_column) / len(cand_column)
    )
    assert dec.accepted_length_delta == pytest.approx(
        dec.candidate_avg_accepted_length - dec.stock_avg_accepted_length
    )
    # And the columns reach the record, not just the object.
    row = dec.to_dict()["per_repeat"][0]
    assert row["stock_accepted_length"] == pytest.approx(stock_column[0])
    assert row["candidate_accepted_length"] == pytest.approx(cand_column[0])


def test_a_missing_num_drafts_counter_reports_unavailable_not_a_zero_delta() -> None:
    """The counter asymmetry must not masquerade as a measured verdict.

    ``acceptance_available`` keys off ``spec_decode_num_draft_tokens`` while the
    gating criterion divides by ``spec_decode_num_drafts``.  An endpoint
    exposing only the first hands the gate ``mean_accepted_length == 0.0`` on
    both arms -- a 0.0 delta, which would reject every candidate under
    ``acceptance_below_threshold`` and blame the head for a missing counter.
    """
    blind = _make_delta(acceptance_rate=0.60, mean_accepted_length=0.0)
    assert blind.acceptance_available is True
    assert blind.accepted_length_available is False

    dec = decide_promotion(
        blind,
        _make_delta(acceptance_rate=0.90, mean_accepted_length=0.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(140.0),
        _pcfg(acc_pp=0.0, throughput_pct=0.0, mal=0.0),
    )

    assert dec.verdict == Verdict.REJECT
    assert dec.reason == Reason.ACCEPTANCE_UNAVAILABLE


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


def test_accepted_length_bar_promotes_exactly_at_the_bar() -> None:
    """The bar is inclusive: a gain landing exactly on it ships.

    The three numbers are chosen to be exact in binary (2.0, 2.5, 0.5) so that
    "exactly at the bar" is a real claim rather than a coin flip on the last
    bit.  The gain this test used to make -- 0.60 -> 0.61 acceptance against a
    1.0 pp bar -- was such a coin flip: ``(0.61-0.60)*100`` happens to land
    1e-15 *above* 1.0, while the same gain expressed as accepted length lands
    1.8e-16 *below* 0.05.  Neither ordering says anything about the gate.
    """
    dec = decide_promotion(
        _make_delta(mean_accepted_length=2.0, output_tok_per_sec=100.0),
        _make_delta(mean_accepted_length=2.5, output_tok_per_sec=100.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(100.0),
        _pcfg(acc_pp=0.0, throughput_pct=0.0, mal=0.5),
    )

    assert dec.verdict == Verdict.PROMOTE
    assert dec.reason == Reason.BOTH_THRESHOLDS_MET
    assert dec.accepted_length_delta == pytest.approx(0.5)


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


#: Contexts a correctness fixture uses unless a test says otherwise.
#:
#: Forty, not one.  The criterion these fixtures exercise compares a divergence
#: *rate* against the engine's own divergence rate, so a suite of one context can
#: distinguish nothing at all: 1-of-1 against a floor of 0-of-1 is what you get
#: half the time from a fair coin.  The archived live suites are 103 and 410
#: contexts; forty is enough for a total corruption to be overwhelming and small
#: enough to keep the fixtures cheap.
_CORRECTNESS_CONTEXTS = 40


def _correctness_arms(
    *,
    diverge_at: int | None,
    control_diverge_at: int | None = None,
    contexts: int = _CORRECTNESS_CONTEXTS,
    diverging: int | None = None,
    control_diverging: int | None = None,
    length: int = 1600,
    with_control: bool = True,
) -> tuple[ReplayResult, ReplayResult, ReplayResult | None]:
    """Three single-repeat correctness passes: stock, candidate, and control.

    The control is a *second stock pass*, which is what the runner really
    produces: same engine, same draft, same settings.  Whatever it disagrees
    with the first stock pass about is engine nondeterminism by construction, so
    a fixture that wants to model benign noise gives ``control_diverge_at`` the
    same treatment it gives ``diverge_at``.  A fixture that wants to model a
    broken head leaves the control clean.

    ``diverging`` / ``control_diverging`` bound how many of *contexts* part, so a
    test can express "the engine flips a near-tie on a few contexts" rather than
    only "on all of them or none".
    """
    n_diverging = contexts if diverging is None else diverging
    n_control = contexts if control_diverging is None else control_diverging
    shared = _tokens(length, "x", length)
    stock: list[RequestResult] = []
    candidate: list[RequestResult] = []
    control: list[RequestResult] = []
    for index in range(contexts):
        def _result(tokens: tuple[str, ...], index: int = index) -> RequestResult:
            return _make_request_result(
                "answer", output_tokens=tokens, context_hash=f"ctx-{index}"
            )

        stock.append(_result(shared))
        candidate.append(
            _result(
                shared
                if diverge_at is None or index >= n_diverging
                else _tokens(diverge_at, "OTHER", length)
            )
        )
        control.append(
            _result(
                shared
                if control_diverge_at is None or index >= n_control
                else _tokens(control_diverge_at, "NOISE", length)
            )
        )
    return (
        _make_replay([_make_run(stock)]),
        _make_replay([_make_run(candidate)]),
        _make_replay([_make_run(control)]) if with_control else None,
    )


def _correctness_pair(
    *, diverge_at: int | None, length: int = 1600, contexts: int = 1
) -> tuple[ReplayResult, ReplayResult]:
    """The stock/candidate halves of :func:`_correctness_arms`."""
    stock, candidate, _ = _correctness_arms(
        diverge_at=diverge_at, contexts=contexts, length=length, with_control=False
    )
    return stock, candidate


def _decide_with_correctness(
    *,
    diverge_at: int | None,
    control_diverge_at: int | None = None,
    min_divergence_token_index: int = 16,
    contexts: int = _CORRECTNESS_CONTEXTS,
    diverging: int | None = None,
    control_diverging: int | None = None,
    correctness_max_tokens: int | None = None,
    with_control: bool = True,
) -> Decision:
    stock_correctness, cand_correctness, control = _correctness_arms(
        diverge_at=diverge_at,
        control_diverge_at=control_diverge_at,
        contexts=contexts,
        diverging=diverging,
        control_diverging=control_diverging,
        with_control=with_control,
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
        stock_correctness_control=control,
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
    """Parting almost immediately, on every context, while the engine does not.

    The engine agrees with itself on all forty contexts and the candidate
    disagrees on all forty from the third token: no noise floor explains that,
    which is what a broken drafter looks like.
    """
    dec = _decide_with_correctness(diverge_at=3)

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH
    assert dec.output_early_divergences == _CORRECTNESS_CONTEXTS
    assert dec.control_early_divergences == 0


def test_candidate_diverging_at_token_900_of_1600_is_not_rejected_for_it() -> None:
    """Float non-determinism deep in a long answer is not a defect.

    And it is now *shown* not to be one rather than assumed: the stock arm,
    replayed against itself, parts at exactly the same offset just as often.
    """
    dec = _decide_with_correctness(diverge_at=900, control_diverge_at=900)

    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET
    # The divergence is still recorded -- it is simply not disqualifying.
    assert dec.output_total_divergences == _CORRECTNESS_CONTEXTS
    assert dec.output_early_divergences == 0
    assert dec.control_total_divergences == _CORRECTNESS_CONTEXTS


def test_first_divergence_offsets_are_persisted_for_every_diverging_context() -> None:
    dec = _decide_with_correctness(
        diverge_at=900, control_diverge_at=900, contexts=3
    )

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
    """The same measurement flips verdict when the bar moves past it.

    Both comparisons part on all forty contexts, so the *total* statistic sees
    40 against 40 and can never gate here -- which is what isolates the
    threshold.  The candidate parts at token 20 and the engine parts at token
    900, so moving the bar from 16 to 64 is the only thing that changes: the
    early statistic goes from 0 against 0 to 40 against 0.
    """
    assert _decide_with_correctness(
        diverge_at=20, control_diverge_at=900, min_divergence_token_index=16
    ).verdict is Verdict.PROMOTE
    assert _decide_with_correctness(
        diverge_at=20, control_diverge_at=900, min_divergence_token_index=64
    ).reason is Reason.OUTPUT_MISMATCH


# ---------------------------------------------------------------------------
# The divergence criterion is a comparison against a measured noise floor
# ---------------------------------------------------------------------------


def test_engine_noise_at_the_qwen_rate_does_not_reject() -> None:
    """The false reject this criterion exists to stop, at its measured size.

    Job d993eee replayed 410 held-out contexts and six of them parted, one of
    them at token 4.  The old rule rejected on that single early occurrence
    while the underlying numbers were the best on record (+0.83 pp acceptance,
    +0.0248 tokens per verifier step).  Reproduced at scale here: the candidate
    comparison and the stock-against-stock control diverge at the same rate and
    with the same one-in-six early share, because both are the same engine.
    """
    stock, candidate, control = _correctness_arms(
        diverge_at=None, contexts=410, length=128
    )
    # Six divergences in the candidate comparison, one of them early; six in the
    # control, one of them early.  Same engine, same hazard.
    def _mark(
        replay: ReplayResult, offsets: dict[int, int], tail: str
    ) -> ReplayResult:
        run = replay.run_results[0]
        results = list(run.results)
        for ctx, offset in offsets.items():
            results[ctx] = _make_request_result(
                "answer",
                output_tokens=_tokens(offset, tail, 128),
                context_hash=f"ctx-{ctx}",
            )
        return _make_replay([_make_run(results)])

    candidate = _mark(
        candidate, {0: 4, 1: 44, 2: 55, 3: 78, 4: 110, 5: 119}, "OTHER"
    )
    control = _mark(control, {6: 9, 7: 41, 8: 60, 9: 83, 10: 101, 11: 120}, "NOISE")

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.60),
        _make_delta(acceptance_rate=0.65),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        PromotionConfig(
            min_acceptance_delta_pp=1.0,
            min_throughput_delta_pct=2.0,
            min_divergence_token_index=16,
        ),
        stock_correctness=stock,
        candidate_correctness=candidate,
        stock_correctness_control=control,
        correctness_max_tokens=128,
    )

    # The evidence that used to reject is still recorded in full.
    assert dec.output_total_divergences == 6
    assert dec.output_early_divergences == 1
    # And it is now judged against the floor that explains it.
    assert dec.control_total_divergences == 6
    assert dec.control_early_divergences == 1
    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET


def test_a_saturated_noise_floor_still_does_not_reject() -> None:
    """gpt-oss-20b: 56% of contexts diverged, on a MoE target under eager.

    Every archived gpt-oss gate rejected on ``output_mismatch``.  A criterion
    that only works when the engine is nearly deterministic is not a criterion
    for "any model, any inference configuration"; this one measures the floor
    wherever it happens to sit.
    """
    dec = _decide_with_correctness(
        diverge_at=40,
        diverging=23,
        control_diverge_at=40,
        control_diverging=21,
        contexts=41,
    )

    assert dec.divergence_rate == pytest.approx(23 / 41)
    assert dec.control_divergence_rate == pytest.approx(21 / 41)
    assert dec.verdict is Verdict.PROMOTE


def test_a_corrupted_head_is_still_rejected_against_a_noisy_engine() -> None:
    """Detection must survive the floor being large.

    The same 56%-noise engine as above, but the head is genuinely broken -- a
    mis-wired draft-to-target token map, a vocabulary mismatch, corrupted
    weights all look the same from here: every context, first token.  No noise
    floor short of total reaches that.
    """
    dec = _decide_with_correctness(
        diverge_at=0,
        control_diverge_at=40,
        control_diverging=21,
        contexts=41,
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH
    assert dec.output_early_divergences == 41
    assert dec.control_early_divergences == 0
    assert dec.divergence_early_p_value is not None
    assert dec.divergence_early_p_value < dec.divergence_alpha


def test_a_head_that_merely_diverges_more_often_is_rejected_on_the_total() -> None:
    """The early statistic is not the only one, and does not have to be.

    A head that never parts early but parts on every context while the engine
    parts on a tenth of them is not explained by the engine.
    """
    dec = _decide_with_correctness(
        diverge_at=900, control_diverge_at=900, control_diverging=4
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH
    assert dec.output_early_divergences == 0
    assert dec.divergence_early_p_value == 1.0
    assert dec.divergence_total_p_value is not None
    assert dec.divergence_total_p_value < dec.divergence_alpha


def test_a_modest_excess_over_the_floor_is_not_significant() -> None:
    """The bar has to be a real bar, and it has to sit where it says it does.

    Twelve divergences against a measured floor of five, over forty contexts, is
    a one-sided p of 0.0497 -- an excess this engine produces by itself one run
    in twenty.  Rejecting there would trade one hair trigger for a slower one.
    The criterion runs an order of magnitude tighter than that, and this test
    pins *where*: it fails if the significance level is loosened even to 0.1.
    """
    dec = _decide_with_correctness(
        diverge_at=900, diverging=12, control_diverge_at=900, control_diverging=5
    )

    assert DIVERGENCE_ALPHA == 0.01
    assert dec.divergence_total_p_value == pytest.approx(0.0497, abs=5e-4)
    assert dec.divergence_total_p_value > dec.divergence_alpha
    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET


def test_two_statistics_are_tested_so_the_level_is_split_between_them() -> None:
    """Rejecting on "either statistic is significant" tests twice.

    Without the Bonferroni divisor the criterion runs at twice its stated size.
    Fourteen divergences against a floor of four sits in exactly the gap it
    opens -- p = 0.0072, above the corrected 0.005 and below the uncorrected
    0.01 -- so this fixture promotes only while the correction is applied.
    """
    dec = _decide_with_correctness(
        diverge_at=900, diverging=14, control_diverge_at=900, control_diverging=4
    )

    assert dec.divergence_alpha == pytest.approx(DIVERGENCE_ALPHA / 2)
    assert dec.divergence_total_p_value == pytest.approx(0.00724, abs=5e-5)
    assert dec.verdict is Verdict.PROMOTE


def test_the_measured_noise_floor_is_persisted_for_audit() -> None:
    """``output_mismatch`` was unfalsifiable without the floor beside it."""
    dec = _decide_with_correctness(
        diverge_at=900, control_diverge_at=900, control_diverging=10
    )

    record = dec.to_dict()
    assert record["divergence_control_available"] is True
    assert record["divergence_trials"] == _CORRECTNESS_CONTEXTS
    assert record["control_trials"] == _CORRECTNESS_CONTEXTS
    assert record["control_total_divergences"] == 10
    assert record["control_early_divergences"] == 0
    assert record["divergence_rate"] == pytest.approx(1.0)
    assert record["control_divergence_rate"] == pytest.approx(0.25)
    assert record["divergence_alpha"] == pytest.approx(DIVERGENCE_ALPHA / 2)
    # Every control divergence is recorded individually, exactly as the
    # candidate's are, so the floor can be re-derived rather than trusted.
    assert len(record["control_divergences"]) == 10
    assert {d["first_divergence_index"] for d in record["control_divergences"]} == {900}


def test_a_missing_control_is_recorded_as_missing_not_as_a_clean_engine() -> None:
    """No control means an *assumed* floor, and the record has to say so."""
    dec = _decide_with_correctness(diverge_at=900, with_control=False)

    assert dec.divergence_control_available is False
    assert dec.control_divergence_rate is None
    assert dec.control_total_divergences == 0
    # The assumed floor is zero over an equal number of trials, which is the
    # strictest floor that is still a test -- so this fixture does reject.
    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH


def test_the_exact_test_is_one_sided_and_calibrated() -> None:
    """The p-value is the hypergeometric upper tail, checked by hand."""
    # Nothing diverged anywhere: no evidence, never significance.
    assert divergence_excess_p_value(0, 10, 0, 10) == 1.0
    # Every event in the candidate arm and none in the control: the chance of
    # that split under a common rate is 1 / C(2n, k).
    assert divergence_excess_p_value(5, 5, 0, 5) == pytest.approx(
        1 / math.comb(10, 5)
    )
    assert divergence_excess_p_value(3, 40, 0, 40) == pytest.approx(
        math.comb(40, 3) / math.comb(80, 3)
    )
    # One-sided: an excess in the *control* arm is not evidence against the
    # candidate, so the p-value must be large, not small.
    assert divergence_excess_p_value(0, 40, 8, 40) > 0.99
    # Equal rates sit at the middle of the distribution.
    assert 0.4 < divergence_excess_p_value(8, 40, 8, 40) < 0.8


def test_the_exact_test_rejects_impossible_counts() -> None:
    with pytest.raises(DecisionError, match="cannot exceed trials"):
        divergence_excess_p_value(11, 10, 0, 10)
    with pytest.raises(DecisionError, match="non-negative integer"):
        divergence_excess_p_value(-1, 10, 0, 10)


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


# ---------------------------------------------------------------------------
# Engine execution regime
# ---------------------------------------------------------------------------
def _decide(**kwargs: object) -> Decision:
    runs = [_valid_run() for _ in range(3)]
    return decide_promotion(
        _make_delta(),
        _make_delta(),
        _make_replay(runs),
        _make_replay(runs),
        _pcfg(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_decision_that_was_told_nothing_says_so_rather_than_assuming_eager() -> None:
    """Every archived run was eager, but no record said so.

    Reading those records as eager would infer a fact from a habit.  The
    absence of knowledge is itself what has to be persisted.
    """
    record = _decide().to_dict()

    assert record["engine_execution_mode"] == UNRECORDED_EXECUTION_MODE
    assert record["engine_enforce_eager"] is None
    assert record["engine_max_num_seqs"] is None
    assert UNRECORDED_EXECUTION_MODE not in (
        EAGER_EXECUTION_MODE,
        CUDA_GRAPH_EXECUTION_MODE,
    )


def test_the_two_execution_regimes_are_recorded_distinguishably() -> None:
    """Eager and CUDA-graph are a large throughput difference on one model."""
    eager = _decide(
        engine_execution=EngineExecution(enforce_eager=True)
    ).to_dict()
    graphed = _decide(
        engine_execution=EngineExecution(enforce_eager=False)
    ).to_dict()

    assert eager["engine_execution_mode"] == EAGER_EXECUTION_MODE
    assert graphed["engine_execution_mode"] == CUDA_GRAPH_EXECUTION_MODE
    assert eager["engine_execution_mode"] != graphed["engine_execution_mode"]
    assert eager["engine_enforce_eager"] is True
    assert graphed["engine_enforce_eager"] is False


def test_the_scheduler_flags_that_move_throughput_reach_the_record() -> None:
    record = _decide(
        engine_execution=EngineExecution(
            enforce_eager=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            max_num_seqs=256,
        )
    ).to_dict()

    assert record["engine_enable_chunked_prefill"] is True
    assert record["engine_enable_prefix_caching"] is False
    assert record["engine_max_num_seqs"] == 256


def test_the_engine_regime_is_read_off_the_argv_that_actually_launched_it() -> None:
    """This is the argv ``build_vllm_argv`` produced, not a restatement of it."""
    execution = EngineExecution.from_argv(
        [
            "vllm",
            "serve",
            "openai/gpt-oss-20b",
            "--max-model-len",
            "4096",
            "--enforce-eager",
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--max-num-seqs=128",
        ]
    )

    assert execution.execution_mode == EAGER_EXECUTION_MODE
    assert execution.enforce_eager is True
    assert execution.enable_chunked_prefill is True
    assert execution.enable_prefix_caching is False
    assert execution.max_num_seqs == 128


def test_an_argv_without_enforce_eager_describes_a_graph_capturing_engine() -> None:
    execution = EngineExecution.from_argv(["vllm", "serve", "m"])

    assert execution.enforce_eager is False
    assert execution.execution_mode == CUDA_GRAPH_EXECUTION_MODE


def test_a_flag_the_operator_never_passed_is_unknown_not_false() -> None:
    """vLLM's own defaults for these are version-dependent.

    Recording ``False`` would put a fact in the record that nobody measured.
    """
    execution = EngineExecution.from_argv(["vllm", "serve", "m", "--enforce-eager"])

    assert execution.enable_chunked_prefill is None
    assert execution.enable_prefix_caching is None
    assert execution.max_num_seqs is None


def test_an_engine_execution_rejects_values_it_cannot_mean() -> None:
    with pytest.raises(ValueError, match="enforce_eager"):
        EngineExecution(enforce_eager="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_num_seqs"):
        EngineExecution(enforce_eager=True, max_num_seqs=0)


# ---------------------------------------------------------------------------
# The divergence position statistic
#
# Greedy speculative decoding verifies by exact argmax equality with no
# tolerance band, so a sound engine emits the verifier's own trajectory
# whatever the draft head proposes.  The arms can then only part where
# floating-point noise flips an argmax at a near-tie -- and near-ties are not
# concentrated at any offset.  Every test below is a statement about that
# null, and each one is paired with the perturbation that breaks it.
# ---------------------------------------------------------------------------


def _diverging_arms(
    *,
    contexts: int,
    offsets: dict[int, int],
    window: int,
) -> tuple[ReplayResult, ReplayResult]:
    """Two correctness passes over *contexts* contexts, parting where told.

    ``offsets`` maps a context index to the token offset at which the candidate
    arm's generation departs from the stock arm's.  Every generation is exactly
    *window* tokens long on both sides, so no divergence this builds is a
    length artefact -- the offsets are the only thing under test.
    """
    stock: list[RequestResult] = []
    candidate: list[RequestResult] = []
    for i in range(contexts):
        base = tuple(f"t{j}" for j in range(window))
        stock.append(
            _make_request_result(
                "".join(base), context_hash=f"ctx-{i}", output_tokens=base
            )
        )
        if i in offsets:
            at = offsets[i]
            other = base[:at] + tuple(f"x{j}" for j in range(at, window))
        else:
            other = base
        candidate.append(
            _make_request_result(
                "".join(other), context_hash=f"ctx-{i}", output_tokens=other
            )
        )
    return (
        _make_replay([_make_run(stock)]),
        _make_replay([_make_run(candidate)]),
    )


def test_position_statistic_clears_a_flat_hazard() -> None:
    """The Qwen shape: a handful of late partings, none of them evidence.

    Ten of 410 contexts part somewhere in a 128-token window, three of them
    inside the first sixteen tokens.  A flat per-token hazard calibrated from
    the other seven puts about one context in that first sixteen, and three
    against an expectation of one is not a finding.  This is the observation
    that the old zero-floor Fisher test rejected at p = 9.2e-4.
    """
    p = divergence_position_p_value(
        3, 10, 410, min_divergence_index=16, max_tokens=128
    )
    assert p > DIVERGENCE_ALPHA / 2


def test_position_statistic_clears_a_nondeterministic_engine() -> None:
    """The gpt-oss shape: half the contexts part, and it is still not evidence.

    211 of 410 contexts parting inside 128 tokens is a per-token flip hazard of
    about 0.56%, which predicts roughly 35 of them inside the first sixteen
    tokens.  Five were seen.  The statistic has to calibrate to the engine in
    front of it, or a model whose engine is merely noisier than another one
    fails the gate for being noisy.
    """
    p = divergence_position_p_value(
        5, 211, 410, min_divergence_index=16, max_tokens=128
    )
    assert p > DIVERGENCE_ALPHA / 2


def test_position_statistic_catches_a_head_whose_output_is_its_own() -> None:
    """The failure the gate exists for: verification not actually verifying.

    A head served without verification -- or verified with a tolerance band --
    stops emitting the verifier's trajectory immediately, so the partings pile
    up at the front of the window instead of spreading across it.  400 of 410
    contexts part and 390 of those part inside the first sixteen tokens, where
    the run's own hazard predicts about 39.
    """
    p = divergence_position_p_value(
        390, 400, 410, min_divergence_index=16, max_tokens=128
    )
    assert p < DIVERGENCE_ALPHA / 2


def test_position_statistic_catches_front_loading_without_a_high_rate() -> None:
    """Front-loading is the signal, not volume.

    The same ten partings as the benign case, but six of them inside the first
    sixteen tokens rather than three.  The rate is identical and unremarkable;
    only the shape has changed, and the shape is what the null constrains.
    """
    benign = divergence_position_p_value(
        3, 10, 410, min_divergence_index=16, max_tokens=128
    )
    front_loaded = divergence_position_p_value(
        6, 10, 410, min_divergence_index=16, max_tokens=128
    )
    assert benign > DIVERGENCE_ALPHA / 2
    assert front_loaded < DIVERGENCE_ALPHA / 2


def test_position_statistic_reports_no_evidence_rather_than_significance() -> None:
    """Every branch with nothing to test returns 1.0, never a small number."""
    assert divergence_position_p_value(
        0, 10, 410, min_divergence_index=16, max_tokens=128
    ) == 1.0
    assert divergence_position_p_value(
        0, 0, 410, min_divergence_index=16, max_tokens=128
    ) == 1.0
    assert divergence_position_p_value(
        0, 0, 0, min_divergence_index=16, max_tokens=128
    ) == 1.0
    # A threshold at or past the window makes every divergence early by
    # construction: there is no late window to calibrate against and no
    # contrast to test.  ``DivergenceCriterion`` already names this shape.
    assert divergence_position_p_value(
        10, 10, 410, min_divergence_index=128, max_tokens=128
    ) == 1.0
    # Every context still alive at the threshold parted afterwards, so the
    # calibrated hazard is unbounded and no early count can surprise.
    assert divergence_position_p_value(
        2, 410, 410, min_divergence_index=16, max_tokens=128
    ) == 1.0


def test_a_control_can_exonerate_but_never_condemn() -> None:
    """The asymmetry that keeps an unmeasurable floor from being read as one.

    The gate's control replays stock against stock inside a single engine
    incarnation, while the measurement straddles a restart.  That makes the
    control a lower bound on the true floor: useless as evidence the candidate
    is at fault, sound as evidence that it is not.  So it may only ever raise
    the null rate.
    """
    condemning = divergence_position_p_value(
        390, 400, 410, min_divergence_index=16, max_tokens=128
    )
    assert condemning < DIVERGENCE_ALPHA / 2
    # The same evidence, against an engine that demonstrably front-loads on its
    # own: no longer a finding about the head.
    exonerated = divergence_position_p_value(
        390,
        400,
        410,
        min_divergence_index=16,
        max_tokens=128,
        control_early_rate=390 / 410,
    )
    assert exonerated > DIVERGENCE_ALPHA / 2
    # And a control that saw nothing cannot make a benign run look guilty.
    assert divergence_position_p_value(
        3, 10, 410, min_divergence_index=16, max_tokens=128, control_early_rate=0.0
    ) == divergence_position_p_value(
        3, 10, 410, min_divergence_index=16, max_tokens=128
    )


def test_position_statistic_rejects_impossible_arguments() -> None:
    with pytest.raises(DecisionError, match="early_events"):
        divergence_position_p_value(
            -1, 10, 410, min_divergence_index=16, max_tokens=128
        )
    with pytest.raises(DecisionError, match="cannot exceed the total"):
        divergence_position_p_value(
            11, 10, 410, min_divergence_index=16, max_tokens=128
        )
    with pytest.raises(DecisionError, match="cannot exceed trials"):
        divergence_position_p_value(
            3, 411, 410, min_divergence_index=16, max_tokens=128
        )
    with pytest.raises(DecisionError, match="control_early_rate"):
        divergence_position_p_value(
            3, 10, 410, min_divergence_index=16, max_tokens=128, control_early_rate=1.5
        )


# ---------------------------------------------------------------------------
# The criterion built on it
# ---------------------------------------------------------------------------


def test_a_same_session_control_is_not_a_comparable_null() -> None:
    """The bug this whole change exists for.

    A control whose two passes share a replay session was collected inside one
    engine incarnation; the measured pair, whose two sides carry *different*
    session ids, was not.  Testing the second against the first is not a test,
    and the decision has to say so on its face rather than publish a p-value
    that reads as though it were.
    """
    stock, candidate = _diverging_arms(
        contexts=40, offsets={i: 40 + i for i in range(10)}, window=128
    )
    stock = dataclasses.replace(stock, session_id="stock-session")
    candidate = dataclasses.replace(candidate, session_id="candidate-session")
    control, _ = _diverging_arms(contexts=40, offsets={}, window=128)
    control = dataclasses.replace(control, session_id="stock-session")

    dec = decide_promotion(
        _make_delta(), _make_delta(mean_accepted_length=5.0),
        _make_replay([_valid_run(100.0) for _ in range(3)]),
        _make_replay([_valid_run(110.0) for _ in range(3)]),
        _pcfg(),
        stock_correctness=stock,
        candidate_correctness=candidate,
        stock_correctness_control=control,
        correctness_max_tokens=128,
    )
    assert dec.divergence_control_available is True
    assert dec.divergence_control_comparable is False
    # Ten late partings out of forty, against a control of zero collected the
    # easy way.  The Fisher test says "significant"; it is not a test.
    assert dec.divergence_total_p_value is not None
    assert dec.divergence_total_p_value < DIVERGENCE_ALPHA / 2
    assert dec.reason is not Reason.OUTPUT_MISMATCH


def test_a_cross_session_control_is_comparable_and_gates() -> None:
    """Give the control the boundary the measurement has and the rate channel
    comes back.

    This is the runner change the finding asks for: collect the control's
    second pass in its own replay session, across the same engine restart the
    arms are separated by, and the excess test recovers its meaning.
    """
    stock, candidate = _diverging_arms(
        contexts=40, offsets={i: 40 + i for i in range(10)}, window=128
    )
    stock = dataclasses.replace(stock, session_id="stock-session")
    candidate = dataclasses.replace(candidate, session_id="candidate-session")
    control, _ = _diverging_arms(contexts=40, offsets={}, window=128)
    control = dataclasses.replace(control, session_id="control-session")

    dec = decide_promotion(
        _make_delta(), _make_delta(mean_accepted_length=5.0),
        _make_replay([_valid_run(100.0) for _ in range(3)]),
        _make_replay([_valid_run(110.0) for _ in range(3)]),
        _pcfg(),
        stock_correctness=stock,
        candidate_correctness=candidate,
        stock_correctness_control=control,
        correctness_max_tokens=128,
    )
    assert dec.divergence_control_comparable is True
    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH


def test_front_loaded_divergence_gates_with_no_comparable_control() -> None:
    """The teeth.  A corrupted head is caught without any usable floor.

    Same topology as the non-comparable case above -- same-session control,
    cross-session measurement, so the rate channel is inert -- but the partings
    are at the front of the window instead of spread through it.
    """
    stock, candidate = _diverging_arms(
        contexts=40, offsets={i: i % 4 for i in range(36)}, window=128
    )
    stock = dataclasses.replace(stock, session_id="stock-session")
    candidate = dataclasses.replace(candidate, session_id="candidate-session")
    control, _ = _diverging_arms(contexts=40, offsets={}, window=128)
    control = dataclasses.replace(control, session_id="stock-session")

    dec = decide_promotion(
        _make_delta(), _make_delta(mean_accepted_length=5.0),
        _make_replay([_valid_run(100.0) for _ in range(3)]),
        _make_replay([_valid_run(110.0) for _ in range(3)]),
        _pcfg(),
        stock_correctness=stock,
        candidate_correctness=candidate,
        stock_correctness_control=control,
        correctness_max_tokens=128,
    )
    assert dec.divergence_control_comparable is False
    assert dec.divergence_position_p_value is not None
    assert dec.divergence_position_p_value < DIVERGENCE_ALPHA / 2
    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.OUTPUT_MISMATCH


def test_the_measured_qwen_evidence_no_longer_rejects() -> None:
    """The run that prompted this: 10 partings in 410, three of them early.

    Reproduced at the shape the artifact records -- token basis, every
    generation at the 128-token cap, a same-session control of zero -- because
    the point of the change is that *this* evidence stops being a rejection
    while the evidence in the test above keeps being one.
    """
    offsets = {0: 119, 1: 4, 2: 118, 3: 113, 4: 4, 5: 27, 6: 104, 7: 44, 8: 74, 9: 4}
    stock, candidate = _diverging_arms(contexts=410, offsets=offsets, window=128)
    stock = dataclasses.replace(stock, session_id="stock-session")
    candidate = dataclasses.replace(candidate, session_id="candidate-session")
    control, _ = _diverging_arms(contexts=410, offsets={}, window=128)
    control = dataclasses.replace(control, session_id="stock-session")

    dec = decide_promotion(
        _make_delta(), _make_delta(mean_accepted_length=5.0),
        _make_replay([_valid_run(100.0) for _ in range(3)]),
        _make_replay([_valid_run(110.0) for _ in range(3)]),
        _pcfg(),
        stock_correctness=stock,
        candidate_correctness=candidate,
        stock_correctness_control=control,
        correctness_max_tokens=128,
    )
    assert dec.output_total_divergences == 10
    assert dec.output_early_divergences == 3
    assert dec.control_total_divergences == 0
    # The old criterion's verdict, still recorded, still significant, and now
    # explicitly not a null the gate is entitled to use.
    assert dec.divergence_total_p_value is not None
    assert dec.divergence_total_p_value < DIVERGENCE_ALPHA / 2
    assert dec.divergence_control_comparable is False
    assert dec.reason is not Reason.OUTPUT_MISMATCH


def test_replay_stamps_every_invocation_with_its_own_session() -> None:
    """Two invocations are two sessions; a result rebuilt from slices is none.

    The decider reads "same non-empty id" as "one live engine, no restart in
    between".  Nothing that did not come straight out of ``replay_suite`` may
    make that claim, so the default has to be the empty string rather than
    anything a comparison could mistake for agreement.
    """
    assert ReplayResult(run_results=(), num_runs=0, suite_hash="h").session_id == ""
    first = _make_replay([_valid_run()])
    assert "session_id" in first.to_dict()


# ---------------------------------------------------------------------------
# What the output cap did to the measurement
# ---------------------------------------------------------------------------


def _with_truncation(
    replay: ReplayResult,
    columns: list[tuple[int, int]],
) -> ReplayResult:
    """Stamp one ``(reported, truncated)`` pair onto each repeat of *replay*.

    Layered on top of :func:`_valid_runs_with_tps` rather than written beside
    it, so the throughput and validity the gate reads are exactly what the
    existing fixtures produce and the only thing varying is the finish-reason
    bookkeeping.
    """
    return dataclasses.replace(
        replay,
        run_results=tuple(
            dataclasses.replace(
                run, finish_reason_count=reported, truncated_count=truncated
            )
            for run, (reported, truncated) in zip(
                replay.run_results, columns, strict=True
            )
        ),
    )


def test_an_arm_that_reported_no_finish_reason_classifies_as_untestable() -> None:
    """Zero reported is not zero truncated; it is nothing observed at all.

    Every archived decision reads this, because none of them persisted the
    counts.  Collapsing it into ``BOUNDED`` would turn "we never looked" into
    the affirmative claim that truncation was measured and was low.
    """
    assert classify_truncation(reported=0, truncated=0) is TruncationRegime.UNTESTABLE


def test_an_arm_with_no_natural_stop_whatsoever_classifies_as_saturated() -> None:
    """The bar is exactly zero natural stops, not a tuned fraction of them."""
    assert classify_truncation(reported=7, truncated=7) is TruncationRegime.SATURATED


def test_a_single_natural_stop_is_enough_to_leave_the_saturated_regime() -> None:
    """The boundary is a count, so one observation moves it and is meant to.

    At one natural stop the run has seen where this model stops at least once,
    which is the whole of what ``SATURATED`` says it has not.
    """
    assert classify_truncation(reported=7, truncated=6) is TruncationRegime.MIXED


def test_an_exact_tie_between_truncated_and_natural_lands_on_bounded() -> None:
    """The MIXED split is a *strict* majority, so a tie is not a majority."""
    assert classify_truncation(reported=10, truncated=5) is TruncationRegime.BOUNDED


def test_one_truncation_more_than_natural_stops_flips_bounded_to_mixed() -> None:
    """The other side of the same boundary, one response away from the tie."""
    assert classify_truncation(reported=11, truncated=6) is TruncationRegime.MIXED


def test_reject_when_the_stock_arm_produced_no_natural_stop() -> None:
    """A measurement rejection: the run cannot support a promotion either way.

    Every stock generation was ended by ``benchmark_max_tokens``, so the arm
    contains no observation of the workload's own lengths and its throughput is
    attributable to the cap rather than to the head.  ``invalid_rate`` cannot
    see this -- a response that spent its whole budget generating is a healthy
    response -- which is why it needs its own reason.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), [(5, 5)] * 3),
        _with_truncation(_valid_runs_with_tps(110.0), [(5, 1)] * 3),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert dec.stock_truncation_regime is TruncationRegime.SATURATED
    assert dec.candidate_truncation_regime is not TruncationRegime.SATURATED


def test_reject_when_the_candidate_arm_produced_no_natural_stop() -> None:
    """Either arm saturating is enough; the comparison needs both to be real."""
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), [(5, 1)] * 3),
        _with_truncation(_valid_runs_with_tps(110.0), [(5, 5)] * 3),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert dec.candidate_truncation_regime is TruncationRegime.SATURATED
    assert dec.stock_truncation_regime is not TruncationRegime.SATURATED


def test_heavily_truncated_arms_that_still_stopped_sometimes_are_not_rejected() -> None:
    """The guard must not swallow the regime every archived live run sits in.

    SLURM 8b72d9a captured 92.2% and 85.3% of its two arms' responses at
    ``length`` and those runs are interpretable: the natural stops bound how
    much of the length distribution was clipped.  Rejecting them would make the
    guard a veto on realistic agentic traffic rather than on unmeasured runs.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), [(100, 92)] * 3),
        _with_truncation(_valid_runs_with_tps(110.0), [(100, 99)] * 3),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.PROMOTE
    assert dec.reason is Reason.BOTH_THRESHOLDS_MET
    assert dec.stock_truncation_regime is TruncationRegime.MIXED
    assert dec.candidate_truncation_regime is TruncationRegime.MIXED


def test_truncation_rates_pool_the_per_repeat_columns_rather_than_average_them() -> None:
    """The published rate must be reproducible by hand from ``per_repeat``.

    The repeats deliberately report different numbers of responses, which is
    the only case in which pooling and a mean of per-repeat rates disagree --
    and they disagree by a lot here, ``0.15`` against ``0.3667``.  A repeat in
    which almost everything errored must not weigh as much as a clean one.
    """
    stock_columns = [(10, 1), (2, 2), (8, 0)]
    candidate_columns = [(4, 1), (4, 1), (4, 2)]

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), stock_columns),
        _with_truncation(_valid_runs_with_tps(110.0), candidate_columns),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    # The array the reader reconciles against really is the array handed in.
    assert [
        (r.stock_finish_reasons, r.stock_truncated) for r in dec.per_repeat
    ] == stock_columns
    assert [
        (r.candidate_finish_reasons, r.candidate_truncated) for r in dec.per_repeat
    ] == candidate_columns

    assert dec.stock_finish_reasons_reported == 20
    assert dec.candidate_finish_reasons_reported == 12
    assert dec.stock_truncation_rate == pytest.approx(3 / 20)
    assert dec.candidate_truncation_rate == pytest.approx(4 / 12)
    assert dec.truncation_rate_delta == pytest.approx(4 / 12 - 3 / 20)

    # The discriminating half: a mean of the per-repeat rates is a different
    # number, so this test fails if anybody swaps pooling for averaging.
    assert dec.stock_truncation_rate != pytest.approx((0.1 + 1.0 + 0.0) / 3)


def test_truncation_rates_are_none_rather_than_zero_when_nothing_reported() -> None:
    """An unmeasured record must not be readable as one that measured zero.

    ``0.0`` here would say "no generation hit the cap", which is a claim about
    the workload.  ``None`` says the endpoint never reported a finish reason,
    which is a claim about the harness, and every archived record is the second.
    """
    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _valid_runs_with_tps(100.0),
        _valid_runs_with_tps(110.0),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.stock_finish_reasons_reported == 0
    assert dec.candidate_finish_reasons_reported == 0
    assert dec.stock_truncation_rate is None
    assert dec.candidate_truncation_rate is None
    assert dec.truncation_rate_delta is None
    assert dec.stock_truncation_regime is TruncationRegime.UNTESTABLE
    assert dec.candidate_truncation_regime is TruncationRegime.UNTESTABLE
    # And an unmeasured arm is not a rejection -- see ``UNTESTABLE``.
    assert dec.verdict is Verdict.PROMOTE


def _finish_reason_columns(
    reasons: list[str],
    *,
    repeats: int = 3,
) -> list[tuple[int, int]]:
    """One ``(reported, truncated)`` column per repeat, from real finish reasons.

    Counted with ``replay.is_truncated_finish_reason`` -- the production
    predicate -- rather than with a literal the test picks, so the vocabulary
    the gate actually classifies with is what these tests exercise.  A spelling
    dropped from ``TRUNCATED_FINISH_REASONS`` therefore changes the regime here
    exactly as it would on a live endpoint.
    """
    truncated = sum(1 for reason in reasons if is_truncated_finish_reason(reason))
    return [(len(reasons), truncated)] * repeats


#: Tokens a capped-and-empty response generated before the cap ended it.  The
#: number only has to be non-zero: it is what makes the response "the engine
#: did real work" rather than "the engine answered with nothing".
_EMPTY_CAPPED_COMPLETION_TOKENS = 512


def _empty_capped_result(finish_reason: str, *, latency_s: float) -> RequestResult:
    """One 200-OK response that generated tokens and surfaced none of them.

    Validity is *computed* by ``replay._validity_error``, the production
    classifier, rather than asserted by the fixture.  A fixture that passed
    ``valid=True`` here would make the test below a tautology: the whole
    question is which side of the validity filter this response shape falls on,
    and answering it in the fixture would answer it for the gate too.
    """
    error = _validity_error(
        completion_tokens=_EMPTY_CAPPED_COMPLETION_TOKENS,
        text="",
        reasoning="",
        output_tokens=(),
        finish_reason=finish_reason,
    )
    return _make_request_result(
        "",
        valid=not error,
        error=error,
        latency_s=latency_s,
        completion_tokens=_EMPTY_CAPPED_COMPLETION_TOKENS,
        finish_reason=finish_reason,
    )


def _empty_capped_arm(
    finish_reason: str,
    *,
    tps: float,
    responses: int = 1,
    repeats: int = 3,
) -> ReplayResult:
    """An arm whose every response came back empty at *finish_reason*.

    Throughput is held at *tps* the same way :func:`_valid_run` holds it, so
    this arm differs from a healthy one only in the response shape under test.
    One response per repeat, likewise matching :func:`_valid_run`: the gate
    pairs the two arms request by request, so an arm of a different width would
    fail to zip against the healthy arm it is compared with.
    The finish-reason columns come from :func:`_finish_reason_columns`, i.e.
    from the production truncation predicate over the same reasons the
    responses carry, so the counts and the responses describe one population.
    """
    runs = [
        _make_run(
            [
                _empty_capped_result(
                    finish_reason,
                    latency_s=_EMPTY_CAPPED_COMPLETION_TOKENS / tps,
                )
                for _ in range(responses)
            ]
        )
        for _ in range(repeats)
    ]
    return _with_truncation(
        _make_replay(runs),
        _finish_reason_columns([finish_reason] * responses, repeats=repeats),
    )


def test_an_arm_of_empty_capped_responses_rejects_as_saturated_not_invalid() -> None:
    """The reason is the whole point: this run measured fixed-length decode.

    A Responses-API server whose every generation ran into the cap without
    surfacing anything reports ``incomplete`` on responses that are, for
    throughput purposes, entirely healthy -- 512 tokens each.  The validity
    check tested ``finish_reason == "length"``, so it scored all of them
    invalid, and because only valid responses enter the truncation denominator
    the arm reported *nothing* about where the model stops.  The gate then
    rejected on ``HIGH_INVALID_RATE``, which is a claim about a broken engine,
    for an engine that was working exactly as asked.  ``HIGH_INVALID_RATE`` is
    checked first, so the wrong reason also shadowed the right one entirely:
    ``TRUNCATION_SATURATED`` could not fire for the response shape it exists to
    catch.

    Both arms would promote on their thresholds, so the verdict here comes from
    the truncation guard and nothing else.
    """
    candidate = _empty_capped_arm("incomplete", tps=110.0)

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(
            _valid_runs_with_tps(100.0), _finish_reason_columns(["stop"] * 5)
        ),
        candidate,
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert dec.reason is not Reason.HIGH_INVALID_RATE
    assert dec.candidate_truncation_regime is TruncationRegime.SATURATED
    # And why the reason above is the truncation guard's rather than the
    # invalid-rate guard's, which is checked first: under the production
    # classifier there is nothing invalid here for it to fire on.
    assert candidate.avg_invalid_rate == 0.0


def test_an_arm_of_empty_self_stopped_responses_still_rejects_as_invalid() -> None:
    """The contrast that keeps the widening from becoming a hole.

    Identical to the arm above in every respect except who ended the
    generations: these responses claim they stopped of their own accord and
    have nothing to show for it, which is a broken engine and exactly what
    ``HIGH_INVALID_RATE`` is for.  Accepting the ``incomplete`` shape by
    treating any finish reason as proof the cap was hit would have swallowed
    this arm too -- and it would then have been read as a fully *natural-stop*
    arm, the strongest possible "the cap was not binding" reading of an engine
    returning nothing at all.
    """
    candidate = _empty_capped_arm("stop", tps=110.0)

    assert candidate.avg_invalid_rate == 1.0

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(
            _valid_runs_with_tps(100.0), _finish_reason_columns(["stop"] * 5)
        ),
        candidate,
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.HIGH_INVALID_RATE


def test_an_arm_capped_under_the_responses_api_spelling_is_still_rejected() -> None:
    """``incomplete`` saturates an arm exactly as ``length`` does.

    This project's own gateway emits the Responses API spelling, so an arm in
    which the cap ended every generation can arrive saying ``incomplete``
    throughout.  While the gate counted only ``length`` that arm reported zero
    truncations and five natural stops, classified ``BOUNDED`` -- the strongest
    available claim that the cap was *not* binding -- and promoted a benchmark
    that had measured nothing but fixed-length decode.
    """
    capped = _finish_reason_columns(["incomplete"] * 5)
    assert capped == [(5, 5)] * 3

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), _finish_reason_columns(["stop"] * 5)),
        _with_truncation(_valid_runs_with_tps(110.0), capped),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert dec.candidate_truncation_regime is TruncationRegime.SATURATED


def test_a_run_whose_only_stop_came_from_a_failed_request_is_rejected() -> None:
    """The decide-side half of the invalid-response defect.

    ``replay`` now refuses to count a failed response's finish reason, so the
    population in
    ``test_gate_replay.py::test_an_invalid_response_cannot_supply_the_only_
    natural_stop`` -- ninety-nine capped generations and one broken empty
    ``stop`` -- reaches the gate as ninety-nine reported, ninety-nine
    truncated.  That has to land on a rejection: had the one failure been
    allowed to count, the arm would have read ``MIXED`` and promoted on a
    benchmark that never once saw where this model stops.
    """
    capped = _finish_reason_columns(["length"] * 99)
    assert capped == [(99, 99)] * 3

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        _with_truncation(_valid_runs_with_tps(100.0), capped),
        _with_truncation(_valid_runs_with_tps(110.0), capped),
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert dec.stock_truncation_regime is TruncationRegime.SATURATED
    # And the same counts with one natural stop restored are *not* rejected,
    # which is the whole of what the miscounted failure bought.
    assert classify_truncation(reported=99, truncated=98) is TruncationRegime.MIXED


def test_the_verdict_agrees_with_the_truncation_the_record_persists() -> None:
    """The gate must classify the window it writes down, not a wider pool.

    ``per_repeat`` keeps only ``min(stock_runs, candidate_runs)`` rows, so when
    the arms return unequal run counts a gate classifying the pooled
    ``ReplayResult`` reads evidence the record does not contain.  Here the
    candidate's fourth run -- unpaired, and therefore never persisted -- is the
    only one holding a natural stop: pooled it says ``MIXED`` and promotes,
    while every row the decision actually wrote says ``SATURATED``.  A verdict
    its own evidence contradicts is unauditable after the fact.
    """
    stock = _with_truncation(_valid_runs_with_tps(100.0), [(5, 1)] * 3)
    candidate = _with_truncation(
        _valid_runs_with_tps(110.0, count=4), [(5, 5), (5, 5), (5, 5), (5, 0)]
    )

    # The premise: the two readings genuinely disagree, so this is a test and
    # not a tautology.
    assert classify_truncation(
        reported=candidate.total_finish_reason_count,
        truncated=sum(r.truncated_count for r in candidate.run_results),
    ) is TruncationRegime.MIXED

    dec = decide_promotion(
        _make_delta(acceptance_rate=0.6, output_tok_per_sec=100.0),
        _make_delta(acceptance_rate=0.65, output_tok_per_sec=110.0),
        stock,
        candidate,
        _pcfg(acc_pp=1.0, throughput_pct=2.0),
    )

    # The unpaired run is absent from the record, as it must be.
    assert len(dec.per_repeat) == 3
    assert [
        (r.candidate_finish_reasons, r.candidate_truncated) for r in dec.per_repeat
    ] == [(5, 5)] * 3

    # And the verdict says exactly what those persisted rows imply.
    assert dec.candidate_truncation_regime is TruncationRegime.SATURATED
    assert dec.verdict is Verdict.REJECT
    assert dec.reason is Reason.TRUNCATION_SATURATED
    assert (dec.reason is Reason.TRUNCATION_SATURATED) is (
        TruncationRegime.SATURATED
        in (dec.stock_truncation_regime, dec.candidate_truncation_regime)
    )
