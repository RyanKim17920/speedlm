"""Promotion decision logic for candidate speculative draft heads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

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
    THROUGHPUT_UNAVAILABLE = "throughput_unavailable"
    HIGH_INVALID_RATE = "high_invalid_rate"
    TOO_FEW_REPEATS = "too_few_repeats"
    OUTPUT_MISMATCH = "output_mismatch"
    UNCERTAIN = "uncertain"


#: Names the throughput statistic the promotion gate is evaluated against.
#:
#: The gate measures throughput two independent ways and they do not agree:
#:
#: * ``replay_per_repeat_mean`` -- the mean of the per-repeat, client-timed
#:   suite passes reported in ``per_repeat``.  One sample per repeat, so its
#:   run-to-run dispersion is measurable from a single benchmark.
#: * the Prometheus scrape window -- ``vllm:generation_tokens_total`` divided by
#:   ``vllm:request_decode_time_seconds_sum`` across the whole window.  Its
#:   denominator is *decode time only*: it excludes prefill and all client-side
#:   HTTP time, so it reads systematically higher, and it collapses to a single
#:   pooled ratio per arm with no within-run dispersion at all.
#:
#: On job 368689 they disagreed by 2.43 pp (-0.75% replay, -3.18% Prometheus).
#: ``PromotionConfig.min_throughput_delta_pct`` was calibrated from per-repeat
#: replay dispersion -- job 368670's pooled sd of 1.338 tok/s on a 76.31 tok/s
#: mean, which is the *replay* mean for that run, not its 82.92 tok/s Prometheus
#: figure -- so the replay statistic is the one that gates.  Changing this
#: constant without recalibrating that threshold against the new statistic's
#: dispersion would silently move the bar.
GATING_THROUGHPUT_STATISTIC: Final[str] = "replay_per_repeat_mean"

#: What gated in decision records written before the statistic was pinned.
#: Those runs evaluated the threshold against the Prometheus decode window
#: while publishing per-repeat replay figures beside it, so a loader must label
#: them as what they were rather than relabel them as the current statistic.
LEGACY_THROUGHPUT_STATISTIC: Final[str] = "prometheus_decode_window"


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
    #: The throughput delta the guard is evaluated against, computed from
    #: :data:`GATING_THROUGHPUT_STATISTIC`.  Compare against
    #: ``min_throughput_delta_pct``; ``prometheus_throughput_delta_pct`` is the
    #: other measurement of the same quantity and never gates.
    throughput_delta_pct: float | None
    min_acceptance_delta_pp: float
    min_throughput_delta_pct: float
    num_repeats: int
    per_repeat: tuple[RepeatSummary, ...]
    stock_avg_acceptance: float
    candidate_avg_acceptance: float
    #: Mean of the ``per_repeat`` throughput column, by construction: these are
    #: the numbers ``throughput_delta_pct`` is computed from, so a reader can
    #: reproduce the gating delta by hand from the array above them.
    stock_avg_tok_per_sec: float
    candidate_avg_tok_per_sec: float
    #: Unscored suite passes run per arm before the measurement window opened.
    #: Reported so a reader can tell steady state from cold start; these passes
    #: are deliberately absent from ``num_repeats`` and ``per_repeat``.
    warmup_repeats: int = 0
    #: Which statistic gated, spelled out in the record rather than implied.
    throughput_statistic: str = GATING_THROUGHPUT_STATISTIC
    #: Diagnostic only.  vLLM's decode-time throughput over the Prometheus
    #: scrape window: informative because it isolates decode from prefill and
    #: client overhead, but *not* what the threshold is calibrated against.
    stock_prometheus_decode_tok_per_sec: float = 0.0
    candidate_prometheus_decode_tok_per_sec: float = 0.0
    prometheus_throughput_delta_pct: float | None = None

    @property
    def throughput_statistic_gap_pp(self) -> float | None:
        """How far the diagnostic statistic sits from the gating one.

        Published so the disagreement is a reported number rather than
        something a reader has to rediscover by hand.  The two measure
        different denominators and are *expected* to differ; job 368689's
        2.43 pp is the largest gap observed across the archived live runs.
        """
        if self.throughput_delta_pct is None:
            return None
        if self.prometheus_throughput_delta_pct is None:
            return None
        return self.throughput_delta_pct - self.prometheus_throughput_delta_pct

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason.value,
            "acceptance_delta_pp": self.acceptance_delta_pp,
            "throughput_statistic": self.throughput_statistic,
            "throughput_delta_pct": self.throughput_delta_pct,
            "prometheus_throughput_delta_pct": self.prometheus_throughput_delta_pct,
            "throughput_statistic_gap_pp": self.throughput_statistic_gap_pp,
            "min_acceptance_delta_pp": self.min_acceptance_delta_pp,
            "min_throughput_delta_pct": self.min_throughput_delta_pct,
            "num_repeats": self.num_repeats,
            "warmup_repeats": self.warmup_repeats,
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
            "stock_prometheus_decode_tok_per_sec": (
                self.stock_prometheus_decode_tok_per_sec
            ),
            "candidate_prometheus_decode_tok_per_sec": (
                self.candidate_prometheus_decode_tok_per_sec
            ),
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_REPEATS = 3
_INVALID_RATE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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
    warmup_repeats: int = 0,
) -> Decision:
    """Decide whether to promote the candidate head.

    Both thresholds are evaluated against a single, named statistic per
    quantity: acceptance from the Prometheus ``spec_decode`` counter delta
    (which has no timing component), and throughput from
    :data:`GATING_THROUGHPUT_STATISTIC`.  The Prometheus decode-time throughput
    is still measured and reported -- as
    ``prometheus_throughput_delta_pct`` -- but it does not gate, because
    ``min_throughput_delta_pct`` is calibrated from replay dispersion.

    Args:
        stock_metrics: Prometheus delta for the stock (baseline) run.  Supplies
            the gating acceptance rate and the diagnostic decode throughput.
        candidate_metrics: Prometheus delta for the candidate run.
        stock_replay: Replay results for stock endpoint.
        candidate_replay: Replay results for candidate endpoint.
        promotion_config: Thresholds from config.
        max_output_mismatches: Allowed output mismatches (default 0).
        warmup_repeats: Unscored warmup passes each arm ran before the
            measurement window opened, recorded in the decision for audit.

    Returns:
        A :class:`Decision` with verdict, reason, and per-repeat data.
    """

    # Prometheus decode-time throughput.  Reported for diagnosis, never gated
    # on: see :data:`GATING_THROUGHPUT_STATISTIC` for why.
    prom_s_tps = stock_metrics.output_tok_per_sec
    prom_c_tps = candidate_metrics.output_tok_per_sec
    prom_delta_pct = (
        (prom_c_tps - prom_s_tps) / prom_s_tps * 100.0 if prom_s_tps > 0 else None
    )
    s_acc = stock_metrics.acceptance_rate
    c_acc = candidate_metrics.acceptance_rate

    # How many suite passes each arm actually completed.  This is a fact about
    # the benchmark run, so it is established before any validation short
    # circuits: a report claiming zero repeats must mean the replay really did
    # not run, not merely that the decision stopped looking.
    s_runs = stock_replay.num_runs
    c_runs = candidate_replay.num_runs
    min_runs = min(s_runs, c_runs)

    # --- Build per-repeat summaries + count mismatches ---
    # Derived purely from replay data, so these survive a metrics failure and
    # keep ``num_repeats == len(per_repeat)`` true on every path.
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

    # --- The gating throughput statistic ---
    # Both arms replayed the same suite the same number of times, so pairing the
    # repeats and averaging the column keeps this number reconcilable by hand:
    # ``stock_avg_tok_per_sec`` is *exactly* the mean of the ``per_repeat``
    # stock column, and the delta below is exactly the delta of those two means.
    # The Prometheus figures are carried alongside, clearly named, and ignored.
    s_tps = _mean([r.stock_tok_per_sec for r in per_repeat_tuple])
    c_tps = _mean([r.candidate_tok_per_sec for r in per_repeat_tuple])

    def _decide(
        verdict: Verdict,
        reason: Reason,
        *,
        acceptance_delta_pp: float | None = None,
        throughput_delta_pct: float | None = None,
    ) -> Decision:
        return Decision(
            verdict=verdict,
            reason=reason,
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
            warmup_repeats=warmup_repeats,
            throughput_statistic=GATING_THROUGHPUT_STATISTIC,
            stock_prometheus_decode_tok_per_sec=prom_s_tps,
            candidate_prometheus_decode_tok_per_sec=prom_c_tps,
            prometheus_throughput_delta_pct=prom_delta_pct,
        )

    def _reject(reason: Reason, **deltas: float | None) -> Decision:
        return _decide(Verdict.REJECT, reason, **deltas)

    # --- Validation: counter reset ---
    if stock_metrics.reset_detected or candidate_metrics.reset_detected:
        return _reject(Reason.COUNTER_RESET)

    # --- Validation: acceptance unavailable ---
    if not stock_metrics.acceptance_available or not candidate_metrics.acceptance_available:
        return _reject(Reason.ACCEPTANCE_UNAVAILABLE)

    # --- Validation: too few repeats ---
    if s_runs < _MIN_REPEATS or c_runs < _MIN_REPEATS:
        return _reject(Reason.TOO_FEW_REPEATS)

    # --- Validation: high invalid rate ---
    if (
        stock_replay.avg_invalid_rate > _INVALID_RATE_THRESHOLD
        or candidate_replay.avg_invalid_rate > _INVALID_RATE_THRESHOLD
    ):
        return _reject(Reason.HIGH_INVALID_RATE)

    # --- Validation: output mismatch ---
    if total_mismatches > max_output_mismatches:
        return _reject(Reason.OUTPUT_MISMATCH)

    # --- Validation: throughput unavailable ---
    # The denominator that matters is the gating statistic's.  A Prometheus
    # window that reads zero is a diagnostic gap, not grounds to reject.
    if s_tps <= 0:
        return _reject(Reason.THROUGHPUT_UNAVAILABLE)

    # --- Compute deltas ---
    acceptance_delta_pp = (c_acc - s_acc) * 100.0

    throughput_delta_pct = (c_tps - s_tps) / s_tps * 100.0

    # --- Threshold: acceptance ---
    if acceptance_delta_pp < promotion_config.min_acceptance_delta_pp:
        return _reject(
            Reason.ACCEPTANCE_BELOW_THRESHOLD,
            acceptance_delta_pp=acceptance_delta_pp,
            throughput_delta_pct=throughput_delta_pct,
        )

    # --- Threshold: throughput ---
    if throughput_delta_pct < promotion_config.min_throughput_delta_pct:
        return _reject(
            Reason.THROUGHPUT_BELOW_THRESHOLD,
            acceptance_delta_pp=acceptance_delta_pp,
            throughput_delta_pct=throughput_delta_pct,
        )

    # --- Both thresholds met: PROMOTE ---
    return _decide(
        Verdict.PROMOTE,
        Reason.BOTH_THRESHOLDS_MET,
        acceptance_delta_pp=acceptance_delta_pp,
        throughput_delta_pct=throughput_delta_pct,
    )
