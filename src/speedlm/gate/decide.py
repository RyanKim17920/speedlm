"""Promotion decision logic for candidate speculative draft heads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from speedlm.config import PromotionConfig
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import ReplayResult

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DecisionError(ValueError):
    """Raised when the decision cannot be computed."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """The promotion verdict."""
    PROMOTE = "promote"
    REJECT = "reject"


class Reason(Enum):
    """Enumerated reasons for a verdict."""
    BOTH_THRESHOLDS_MET = "both_thresholds_met"
    ACCEPTANCE_BELOW_THRESHOLD = "acceptance_below_threshold"
    THROUGHPUT_BELOW_THRESHOLD = "throughput_below_threshold"
    COUNTER_RESET = "counter_reset"
    ACCEPTANCE_UNAVAILABLE = "acceptance_unavailable"
    HIGH_INVALID_RATE = "high_invalid_rate"
    TOO_FEW_REPEATS = "too_few_repeats"
    OUTPUT_MISMATCH = "output_mismatch"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class RepeatSummary:
    """Per-repeat summary included in the decision report."""

    repeat_index: int
    stock_tok_per_sec: float
    candidate_tok_per_sec: float
    stock_acceptance_rate: float
    candidate_acceptance_rate: float
    invalid_rate: float
    output_mismatches: int


@dataclass(frozen=True, slots=True)
class Decision:
    """The promotion decision with full reporting."""

    verdict: Verdict
    reason: Reason
    acceptance_delta_pp: float | None
    throughput_delta_pct: float | None
    min_acceptance_delta_pp: float
    min_throughput_delta_pct: float
    num_repeats: int
    per_repeat: tuple[RepeatSummary, ...]
    stock_avg_acceptance: float
    candidate_avg_acceptance: float
    stock_avg_tok_per_sec: float
    candidate_avg_tok_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason.value,
            "acceptance_delta_pp": self.acceptance_delta_pp,
            "throughput_delta_pct": self.throughput_delta_pct,
            "min_acceptance_delta_pp": self.min_acceptance_delta_pp,
            "min_throughput_delta_pct": self.min_throughput_delta_pct,
            "num_repeats": self.num_repeats,
            "per_repeat": [
                {
                    "repeat_index": r.repeat_index,
                    "stock_tok_per_sec": r.stock_tok_per_sec,
                    "candidate_tok_per_sec": r.candidate_tok_per_sec,
                    "stock_acceptance_rate": r.stock_acceptance_rate,
                    "candidate_acceptance_rate": r.candidate_acceptance_rate,
                    "invalid_rate": r.invalid_rate,
                    "output_mismatches": r.output_mismatches,
                }
                for r in self.per_repeat
            ],
            "stock_avg_acceptance": self.stock_avg_acceptance,
            "candidate_avg_acceptance": self.candidate_avg_acceptance,
            "stock_avg_tok_per_sec": self.stock_avg_tok_per_sec,
            "candidate_avg_tok_per_sec": self.candidate_avg_tok_per_sec,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_REPEATS = 3
_INVALID_RATE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_reject(
    reason: Reason,
    acceptance_delta_pp: float | None,
    throughput_delta_pct: float | None,
    pcfg: PromotionConfig,
    num_repeats: int,
    per_repeat: tuple[RepeatSummary, ...],
    stock_acc: float,
    cand_acc: float,
    stock_tps: float,
    cand_tps: float,
) -> Decision:
    return Decision(
        verdict=Verdict.REJECT,
        reason=reason,
        acceptance_delta_pp=acceptance_delta_pp,
        throughput_delta_pct=throughput_delta_pct,
        min_acceptance_delta_pp=pcfg.min_acceptance_delta_pp,
        min_throughput_delta_pct=pcfg.min_throughput_delta_pct,
        num_repeats=num_repeats,
        per_repeat=per_repeat,
        stock_avg_acceptance=stock_acc,
        candidate_avg_acceptance=cand_acc,
        stock_avg_tok_per_sec=stock_tps,
        candidate_avg_tok_per_sec=cand_tps,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_promotion(
    stock_metrics: MetricsDelta,
    candidate_metrics: MetricsDelta,
    stock_replay: ReplayResult,
    candidate_replay: ReplayResult,
    promotion_config: PromotionConfig,
    *,
    max_output_mismatches: int = 0,
) -> Decision:
    """Decide whether to promote the candidate head.

    Args:
        stock_metrics: Prometheus delta for the stock (baseline) run.
        candidate_metrics: Prometheus delta for the candidate run.
        stock_replay: Replay results for stock endpoint.
        candidate_replay: Replay results for candidate endpoint.
        promotion_config: Thresholds from config.
        max_output_mismatches: Allowed output mismatches (default 0).

    Returns:
        A :class:`Decision` with verdict, reason, and per-repeat data.
    """

    s_tps = stock_metrics.output_tok_per_sec
    c_tps = candidate_metrics.output_tok_per_sec
    s_acc = stock_metrics.acceptance_rate
    c_acc = candidate_metrics.acceptance_rate


    # --- Validation: counter reset ---
    if stock_metrics.reset_detected or candidate_metrics.reset_detected:
        return _make_reject(
            reason=Reason.COUNTER_RESET,
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            pcfg=promotion_config,
            num_repeats=0,
            per_repeat=tuple(),
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Validation: acceptance unavailable ---
    if not stock_metrics.acceptance_available or not candidate_metrics.acceptance_available:
        return _make_reject(
            reason=Reason.ACCEPTANCE_UNAVAILABLE,
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            pcfg=promotion_config,
            num_repeats=0,
            per_repeat=tuple(),
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Validation: too few repeats ---
    s_runs = stock_replay.num_runs
    c_runs = candidate_replay.num_runs
    min_runs = min(s_runs, c_runs)

    if s_runs < _MIN_REPEATS or c_runs < _MIN_REPEATS:
        return _make_reject(
            reason=Reason.TOO_FEW_REPEATS,
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            pcfg=promotion_config,
            num_repeats=min_runs,
            per_repeat=tuple(),
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Validation: high invalid rate ---
    if (
        stock_replay.avg_invalid_rate > _INVALID_RATE_THRESHOLD
        or candidate_replay.avg_invalid_rate > _INVALID_RATE_THRESHOLD
    ):
        return _make_reject(
            reason=Reason.HIGH_INVALID_RATE,
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            pcfg=promotion_config,
            num_repeats=min_runs,
            per_repeat=tuple(),
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Build per-repeat summaries + count mismatches ---
    per_repeat_list: list[RepeatSummary] = []
    total_mismatches = 0

    for i in range(min_runs):
        s_run = stock_replay.run_results[i]
        c_run = candidate_replay.run_results[i]

        mismatches = 0
        for s_req, c_req in zip(s_run.results, c_run.results, strict=True):
            if s_req.response_text != c_req.response_text:
                mismatches += 1
        total_mismatches += mismatches

        per_repeat_list.append(
            RepeatSummary(
                repeat_index=i,
                stock_tok_per_sec=s_run.output_tok_per_sec,
                candidate_tok_per_sec=c_run.output_tok_per_sec,
                stock_acceptance_rate=s_acc,
                candidate_acceptance_rate=c_acc,
                invalid_rate=c_run.invalid_rate,
                output_mismatches=mismatches,
            )
        )

    per_repeat_tuple = tuple(per_repeat_list)

    # --- Validation: output mismatch ---
    if total_mismatches > max_output_mismatches:
        return _make_reject(
            reason=Reason.OUTPUT_MISMATCH,
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            pcfg=promotion_config,
            num_repeats=min_runs,
            per_repeat=per_repeat_tuple,
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Compute deltas ---
    acceptance_delta_pp = (c_acc - s_acc) * 100.0

    throughput_delta_pct = (c_tps - s_tps) / s_tps * 100.0 if s_tps > 0 else 0.0

    # --- Threshold: acceptance ---
    if acceptance_delta_pp < promotion_config.min_acceptance_delta_pp:
        return _make_reject(
            reason=Reason.ACCEPTANCE_BELOW_THRESHOLD,
            acceptance_delta_pp=acceptance_delta_pp,
            throughput_delta_pct=throughput_delta_pct,
            pcfg=promotion_config,
            num_repeats=min_runs,
            per_repeat=per_repeat_tuple,
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Threshold: throughput ---
    if throughput_delta_pct < promotion_config.min_throughput_delta_pct:
        return _make_reject(
            reason=Reason.THROUGHPUT_BELOW_THRESHOLD,
            acceptance_delta_pp=acceptance_delta_pp,
            throughput_delta_pct=throughput_delta_pct,
            pcfg=promotion_config,
            num_repeats=min_runs,
            per_repeat=per_repeat_tuple,
            stock_acc=s_acc,
            cand_acc=c_acc,
            stock_tps=s_tps,
            cand_tps=c_tps,
        )

    # --- Both thresholds met: PROMOTE ---
    return Decision(
        verdict=Verdict.PROMOTE,
        reason=Reason.BOTH_THRESHOLDS_MET,
        acceptance_delta_pp=acceptance_delta_pp,
        throughput_delta_pct=throughput_delta_pct,
        min_acceptance_delta_pp=promotion_config.min_acceptance_delta_pp,
        min_throughput_delta_pct=promotion_config.min_throughput_delta_pct,
        num_repeats=min_runs,
        per_repeat=per_repeat_tuple,
        stock_avg_acceptance=s_acc,
        candidate_avg_acceptance=c_acc,
        stock_avg_tok_per_sec=s_tps,
        candidate_avg_tok_per_sec=c_tps,
    )
