"""Tests for gate/divergence.py - the output-correctness criterion.

The promotion gate rejects a candidate head whose generations diverge from
the stock. This module is the statistical apparatus: Fisher exact tests,
position-based tests, divergence collection, and the evaluate_divergence
orchestrator. It is imported by gate/decide.py which re-exports its public
names, but decide.py only tests the integration path. Here we test divergence
in isolation so that a change to the statistical machinery is caught before
it reaches the gate.
"""

from __future__ import annotations

import pytest

from speedlm.gate.divergence import (
    DIVERGENCE_PER_STATISTIC_ALPHA,
    ContextDivergence,
    DecisionError,
    DivergenceBasis,
    divergence_excess_p_value,
    divergence_position_p_value,
    evaluate_divergence,
    first_divergence,
)
from speedlm.gate.replay import (
    FINISH_REASON_NATURAL,
    ReplayResult,
    RequestResult,
    RunResults,
)

MIN_DIV_TOKEN = 16


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_request(
    output_tokens: tuple[str, ...] | None = None,
    *,
    context_hash: str = "test-hash",
    response_text: str = "",
    reasoning_text: str = "",
) -> RequestResult:
    return RequestResult(
        context_hash=context_hash,
        latency_s=0.1,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        response_text=response_text,
        valid=True,
        error="",
        output_tokens=output_tokens or (),
        finish_reason=FINISH_REASON_NATURAL,
        reasoning_text=reasoning_text,
    )


def _make_run_results(contexts: int, *, diverge: bool = False) -> RunResults:
    """Build one RunResults with *contexts* identical request results."""
    results: list[RequestResult] = []
    for ctx in range(contexts):
        tokens = tuple(f"tok_{ctx}_{i}" for i in range(32))
        results.append(_make_request(output_tokens=tokens, context_hash=f"ctx-{ctx}"))
    return RunResults(
        results=tuple(results),
        total_latency_s=0.1 * contexts,
        total_prompt_tokens=10 * contexts,
        total_completion_tokens=5 * contexts,
        valid_count=contexts,
        invalid_count=0,
        invalid_rate=0.0,
    )


def _make_replay(
    num_runs: int = 1,
    *,
    contexts_per_run: int = 1,
    diverge: bool = False,
    session_id: str = "default",
) -> ReplayResult:
    """Build stock/candidate ReplayResult with identical runs."""
    run_results: list[RunResults] = []
    for _ in range(num_runs):
        run_results.append(_make_run_results(contexts_per_run, diverge=diverge))
    return ReplayResult(
        run_results=tuple(run_results),
        num_runs=num_runs,
        suite_hash="test-suite",
        session_id=session_id,
    )


def _make_divergence(index: int, *, early: bool, repeat: int = 0) -> ContextDivergence:
    return ContextDivergence(
        context_hash=f"ctx-{index:04d}",
        repeat_index=repeat,
        first_divergence_index=index,
        basis="token",
        stock_length=128,
        candidate_length=128,
        early=early,
    )


# ---------------------------------------------------------------------------
# first_divergence
# ---------------------------------------------------------------------------


def test_first_divergence_identical_returns_none() -> None:
    tokens = tuple(f"t{i}" for i in range(10))
    stock = _make_request(output_tokens=tokens)
    candidate = _make_request(output_tokens=tokens)

    index, basis, s_len, c_len = first_divergence(stock, candidate)

    assert index is None
    assert basis == DivergenceBasis.TOKEN
    assert s_len == c_len == 10


def test_first_divergence_finds_token_offset() -> None:
    stock = _make_request(output_tokens=("a", "b", "c", "d"))
    candidate = _make_request(output_tokens=("a", "X", "c", "d"))

    index, basis, s_len, c_len = first_divergence(stock, candidate)

    assert index == 1
    assert basis == DivergenceBasis.TOKEN
    assert s_len == c_len == 4


def test_first_divergence_diverges_at_position() -> None:
    stock = _make_request(output_tokens=("a", "b", "c"))
    candidate = _make_request(output_tokens=("a", "X", "c"))

    index, basis, _, _ = first_divergence(stock, candidate)

    assert index == 1
    assert basis == DivergenceBasis.TOKEN


def test_first_divergence_shorter_generation_is_divergence() -> None:
    stock = _make_request(output_tokens=("a", "b", "c"))
    candidate = _make_request(output_tokens=("a", "b"))

    index, basis, s_len, c_len = first_divergence(stock, candidate)

    assert index == 2
    assert basis == DivergenceBasis.TOKEN
    assert s_len == 3
    assert c_len == 2


def test_first_divergence_falls_back_to_characters_without_token_capture() -> None:
    """When output_tokens is empty on both sides, we compare characters."""
    stock = _make_request(output_tokens=(), reasoning_text="hello world")
    candidate = _make_request(output_tokens=(), reasoning_text="hello worle")

    index, basis, _, _ = first_divergence(stock, candidate)

    assert index == 10
    assert basis == DivergenceBasis.CHARACTER


def test_first_divergence_reports_a_token_offset_not_a_boolean() -> None:
    stock = _make_request(output_tokens=("a", "b", "c"))
    candidate = _make_request(output_tokens=("x", "b", "c"))

    index, basis, _, _ = first_divergence(stock, candidate)

    assert index == 0
    assert basis == DivergenceBasis.TOKEN


# ---------------------------------------------------------------------------
# divergence_excess_p_value
# ---------------------------------------------------------------------------


def test_divergence_excess_p_value_no_events_returns_one() -> None:
    p = divergence_excess_p_value(0, 100, 0, 100)
    assert p == 1.0


def test_divergence_excess_p_value_no_trials_returns_one() -> None:
    p = divergence_excess_p_value(0, 0, 0, 0)
    assert p == 1.0


def test_divergence_excess_p_value_total_corruption_is_significant() -> None:
    """Every context diverges for candidate, none for control -> small p."""
    p = divergence_excess_p_value(10, 10, 0, 10)
    assert p < DIVERGENCE_PER_STATISTIC_ALPHA


def test_divergence_excess_p_value_equal_rates_not_significant() -> None:
    """Same rate in both arms should not reject."""
    p = divergence_excess_p_value(5, 100, 5, 100)
    assert p >= DIVERGENCE_PER_STATISTIC_ALPHA


def test_divergence_excess_p_value_negative_raises() -> None:
    with pytest.raises(DecisionError, match="non-negative"):
        divergence_excess_p_value(-1, 100, 0, 100)


def test_divergence_excess_p_value_events_exceed_trials_raises() -> None:
    with pytest.raises(DecisionError, match="cannot exceed"):
        divergence_excess_p_value(5, 3, 0, 100)


def test_divergence_excess_p_value_bool_raises() -> None:
    """Passing a bool should raise since it's not a valid int here."""
    with pytest.raises(DecisionError, match="non-negative"):
        divergence_excess_p_value(True, 100, 0, 100)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# divergence_position_p_value
# ---------------------------------------------------------------------------


def test_divergence_position_p_value_no_early_events_returns_one() -> None:
    p = divergence_position_p_value(0, 10, 100, min_divergence_index=16, max_tokens=128)
    assert p == 1.0


def test_divergence_position_p_value_no_trials_returns_one() -> None:
    p = divergence_position_p_value(0, 0, 0, min_divergence_index=16, max_tokens=128)
    assert p == 1.0


def test_divergence_position_p_value_front_loaded_is_significant() -> None:
    """All divergences in the first few tokens -> highly significant.

    80 of 90 divergences are early out of 100 trials. The calibrated hazard
    expects only ~6% to be early, so 80 is far outside the null.
    """
    p = divergence_position_p_value(
        80,
        90,
        100,
        min_divergence_index=16,
        max_tokens=128,
    )
    assert p < DIVERGENCE_PER_STATISTIC_ALPHA


def test_divergence_position_p_value_threshold_at_cap_returns_one() -> None:
    """No late window exists, so no evidence."""
    p = divergence_position_p_value(
        5,
        10,
        100,
        min_divergence_index=128,
        max_tokens=128,
    )
    assert p == 1.0


def test_divergence_position_p_value_early_exceeds_total_raises() -> None:
    with pytest.raises(DecisionError, match="cannot exceed"):
        divergence_position_p_value(10, 5, 100, min_divergence_index=16, max_tokens=128)


def test_divergence_position_p_value_total_exceeds_trials_raises() -> None:
    with pytest.raises(DecisionError, match="cannot exceed"):
        divergence_position_p_value(5, 100, 50, min_divergence_index=16, max_tokens=128)


def test_divergence_position_p_value_invalid_rate_raises() -> None:
    with pytest.raises(DecisionError, match="control_early_rate"):
        divergence_position_p_value(
            5,
            10,
            100,
            min_divergence_index=16,
            max_tokens=128,
            control_early_rate=1.5,
        )


def test_divergence_position_p_value_negative_raises() -> None:
    with pytest.raises(DecisionError, match="non-negative"):
        divergence_position_p_value(-1, 10, 100, min_divergence_index=16, max_tokens=128)


def test_divergence_position_p_value_control_rate_makes_harder() -> None:
    """A control_early_rate raises the null, making p larger."""
    p_without = divergence_position_p_value(
        5,
        10,
        100,
        min_divergence_index=16,
        max_tokens=128,
        control_early_rate=0.0,
    )
    p_with = divergence_position_p_value(
        5,
        10,
        100,
        min_divergence_index=16,
        max_tokens=128,
        control_early_rate=0.1,
    )
    assert p_with >= p_without


# ---------------------------------------------------------------------------
# evaluate_divergence
# ---------------------------------------------------------------------------


def test_evaluate_divergence_no_control_marks_unavailable() -> None:
    replay = _make_replay(num_runs=1, contexts_per_run=5)

    evidence = evaluate_divergence(
        stock_replay=replay,
        candidate_replay=replay,
        stock_correctness=None,
        candidate_correctness=None,
        stock_correctness_control=None,
        min_divergence_index=MIN_DIV_TOKEN,
        correctness_max_tokens=128,
        max_output_mismatches=10,
    )

    assert evidence.control_available is False
    assert evidence.control_comparable is False


def test_evaluate_divergence_identical_replays_no_rejection() -> None:
    replay = _make_replay(num_runs=1, contexts_per_run=5)

    evidence = evaluate_divergence(
        stock_replay=replay,
        candidate_replay=replay,
        stock_correctness=None,
        candidate_correctness=None,
        stock_correctness_control=None,
        min_divergence_index=MIN_DIV_TOKEN,
        correctness_max_tokens=128,
        max_output_mismatches=10,
    )

    assert not evidence.rejects
    assert evidence.trials == 5
    assert evidence.total_early == 0


def test_evaluate_divergence_control_comparable_requires_session_ids() -> None:
    """control_comparable is False when session IDs are missing or equal."""
    replay = _make_replay(num_runs=1, contexts_per_run=2)

    evidence = evaluate_divergence(
        stock_replay=replay,
        candidate_replay=replay,
        stock_correctness=replay,
        candidate_correctness=replay,
        stock_correctness_control=replay,
        min_divergence_index=MIN_DIV_TOKEN,
        correctness_max_tokens=128,
        max_output_mismatches=10,
    )

    # All the same session_id, so control_comparable should be False
    assert not evidence.control_comparable


# ---------------------------------------------------------------------------
# ContextDivergence.to_dict
# ---------------------------------------------------------------------------


def test_context_divergence_to_dict_roundtrips() -> None:
    div = _make_divergence(5, early=True)
    d = div.to_dict()
    assert d["first_divergence_index"] == 5
    assert d["early"] is True
    assert d["basis"] == "token"
    assert d["context_hash"] == "ctx-0005"
