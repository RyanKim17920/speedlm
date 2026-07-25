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
