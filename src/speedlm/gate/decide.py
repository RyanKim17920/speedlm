"""Promotion decision logic for candidate speculative draft heads."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from speedlm.config import PromotionConfig
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import ReplayResult, RequestResult

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

#: Names the acceptance statistic the promotion gate is evaluated against.
#:
#: Acceptance used to be a single pooled ratio per arm: the runner scraped
#: ``/metrics`` once before the whole replay and once after it, differenced the
#: two, and :func:`decide_promotion` stamped the resulting scalar into every
#: ``RepeatSummary``.  Job 369005 shows what that produces --
#: ``(539777-88070)/(1343865-223074) = 0.40302518489174166`` appears
#: bit-identical in all five per-repeat rows.  The array read as n=5 to any
#: consumer while carrying n=1 of information, and
#: ``min_acceptance_delta_pp`` -- the *primary* promotion criterion -- was being
#: evaluated against one sample per arm with no variance estimate at all.
#:
#: The runner now scrapes between repeats (``repeats + 1`` scrapes per arm
#: rather than 2, so the extra instrumentation costs no extra passes), which
#: makes each repeat a real, independent acceptance sample.  The gate compares
#: the mean of that vector, exactly as it already does for throughput, and
#: publishes the vector's sample standard deviation beside it so the threshold
#: can be recalibrated from evidence instead of assertion.
GATING_ACCEPTANCE_STATISTIC: Final[str] = "per_repeat_mean"

#: What gated in decision records written before acceptance was sampled per
#: repeat: one Prometheus window spanning every repeat of an arm, stamped into
#: each per-repeat row.  A loader must label those records as what they were.
LEGACY_ACCEPTANCE_STATISTIC: Final[str] = "prometheus_pooled_window"


class DivergenceBasis(Enum):
    """What two generations were aligned on when they were compared.

    ``TOKEN`` is the model's own segmentation, recovered from the endpoint's
    per-token logprobs; an index is then a token offset.  ``CHARACTER`` is the
    fallback when logprobs were unavailable on either side, and an index is a
    character offset into the response text -- coarser, and systematically
    *larger* than the token offset it stands in for, so it is recorded rather
    than silently mixed in with token offsets.
    """

    TOKEN = "token"
    CHARACTER = "character"


@dataclass(frozen=True, slots=True)
class ContextDivergence:
    """Where one held-out context's two generations first parted.

    Persisted for every diverging context.  Before this existed the run
    directory kept only ``decision.json`` and ``gate-metrics/*.prom.gz``, so a
    rejection reading ``output_mismatches: 85`` was unfalsifiable after the
    fact: there was no way to tell a drafter that broke at token 3 from float
    noise at token 900.
    """

    context_hash: str
    repeat_index: int
    #: Offset of the first position at which the two generations differ, in
    #: units of ``basis``.  Always >= 0; a context that never diverges is not
    #: recorded at all.
    first_divergence_index: int
    basis: str
    stock_length: int
    candidate_length: int
    #: True when the divergence is early enough to gate against, i.e.
    #: ``first_divergence_index < promotion.min_divergence_token_index``.
    early: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_hash": self.context_hash,
            "repeat_index": self.repeat_index,
            "first_divergence_index": self.first_divergence_index,
            "basis": self.basis,
            "stock_length": self.stock_length,
            "candidate_length": self.candidate_length,
            "early": self.early,
        }


@dataclass(frozen=True, slots=True)
class RepeatSummary:
    """Per-repeat summary included in the decision report."""

    repeat_index: int
    stock_tok_per_sec: float
    candidate_tok_per_sec: float
    #: Acceptance measured over *this repeat's* Prometheus window.  Before the
    #: runner scraped between repeats this column held one pooled scalar
    #: repeated ``num_repeats`` times; see
    #: :data:`GATING_ACCEPTANCE_STATISTIC`.
    stock_acceptance_rate: float
    candidate_acceptance_rate: float
    invalid_rate: float
    #: Early divergences attributed to the correctness-pass repeat with this
    #: index.  The correctness pass is a separate, bounded, single-stream
    #: replay that runs once by default, so with the default configuration only
    #: row 0 is ever non-zero.  ``Decision.output_early_divergences`` is the
    #: total the gate actually compares against ``max_output_mismatches``.
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
    #: Mean of the ``per_repeat`` acceptance column, by construction -- the same
    #: contract ``stock_avg_tok_per_sec`` has held all along.  It was previously
    #: a single pooled ratio wearing the name ``avg``; see
    #: :data:`GATING_ACCEPTANCE_STATISTIC`.
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
    #: Which acceptance statistic gated, spelled out in the record.
    acceptance_statistic: str = GATING_ACCEPTANCE_STATISTIC
    #: Sample standard deviation of the ``per_repeat`` acceptance column, per
    #: arm.  Zero at fewer than two repeats, and zero on any record written
    #: before the runner scraped between repeats (those columns are a single
    #: value repeated).  This is the number ``min_acceptance_delta_pp`` has to
    #: be recalibrated against; publishing it is what makes that possible.
    stock_acceptance_stdev: float = 0.0
    candidate_acceptance_stdev: float = 0.0
    #: The bar a divergence had to clear to be treated as harmless, copied from
    #: ``promotion.min_divergence_token_index`` so an archived decision records
    #: the criterion it was judged under.
    min_divergence_token_index: int = 0
    #: Every context whose two generations parted, with the offset at which
    #: they parted.  ``output_early_divergences`` counts the subset that parted
    #: before ``min_divergence_token_index``; that subset is what gates.
    output_divergences: tuple[ContextDivergence, ...] = ()

    @property
    def output_early_divergences(self) -> int:
        """Divergences early enough to count against ``max_output_mismatches``."""
        return sum(1 for d in self.output_divergences if d.early)

    @property
    def output_total_divergences(self) -> int:
        """Contexts that parted at any offset, early or not."""
        return len(self.output_divergences)

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
            "acceptance_statistic": self.acceptance_statistic,
            "stock_avg_acceptance": self.stock_avg_acceptance,
            "candidate_avg_acceptance": self.candidate_avg_acceptance,
            "stock_acceptance_stdev": self.stock_acceptance_stdev,
            "candidate_acceptance_stdev": self.candidate_acceptance_stdev,
            "min_divergence_token_index": self.min_divergence_token_index,
            "output_early_divergences": self.output_early_divergences,
            "output_total_divergences": self.output_total_divergences,
            "output_divergences": [d.to_dict() for d in self.output_divergences],
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


def _stdev(values: list[float]) -> float:
    """Sample standard deviation, or 0.0 when there is nothing to disperse."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def first_divergence(
    stock: RequestResult,
    candidate: RequestResult,
) -> tuple[int | None, DivergenceBasis, int, int]:
    """Locate where two generations of the same context first differ.

    This replaces whole-string equality, which asks the wrong question.  Two
    greedy generations of 1600 tokens that agree for 1500 of them are, by any
    behavioural standard, the same answer; two that part at token 3 are not.
    Equality collapses both to ``True`` and gates on the collapse.

    Alignment prefers the model's own tokenisation (``output_tokens``, from the
    endpoint's per-token logprobs) and falls back to characters when either
    side did not capture it.  The basis is returned rather than assumed,
    because a character offset is not comparable to a token offset.

    Returns:
        ``(index, basis, stock_length, candidate_length)`` where *index* is the
        first differing position, or ``None`` when one sequence is a prefix of
        the other *and* they are the same length -- i.e. they are identical.
        A shorter-but-otherwise-identical generation diverges at the end of the
        shorter sequence, because stopping early is itself a difference.
    """
    if stock.output_tokens and candidate.output_tokens:
        basis = DivergenceBasis.TOKEN
        left: Sequence[str] = stock.output_tokens
        right: Sequence[str] = candidate.output_tokens
    else:
        basis = DivergenceBasis.CHARACTER
        left = stock.response_text
        right = candidate.response_text

    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index, basis, len(left), len(right)
    if len(left) != len(right):
        return min(len(left), len(right)), basis, len(left), len(right)
    return None, basis, len(left), len(right)


def _repeat_acceptance(
    repeat_metrics: Sequence[MetricsDelta],
    index: int,
    pooled: float,
) -> float:
    """This repeat's own acceptance, or the pooled figure if none was taken.

    Falling back to the pooled scalar is what the gate used to do for *every*
    repeat.  Keeping it as the fallback means a caller that has not been
    updated still produces a self-consistent record -- ``*_avg_acceptance``
    remains the literal mean of the column, and the published standard
    deviation is 0.0, which is the honest reading of one sample repeated.
    """
    if index < len(repeat_metrics):
        delta = repeat_metrics[index]
        if delta.acceptance_available:
            return delta.acceptance_rate
    return pooled


def _collect_divergences(
    stock: ReplayResult,
    candidate: ReplayResult,
    *,
    min_divergence_index: int,
) -> tuple[ContextDivergence, ...]:
    """Locate, classify and record every context whose generations parted."""
    found: list[ContextDivergence] = []
    for repeat_index in range(min(stock.num_runs, candidate.num_runs)):
        s_run = stock.run_results[repeat_index]
        c_run = candidate.run_results[repeat_index]
        for s_req, c_req in zip(s_run.results, c_run.results, strict=True):
            index, basis, s_len, c_len = first_divergence(s_req, c_req)
            if index is None:
                continue
            found.append(
                ContextDivergence(
                    context_hash=s_req.context_hash,
                    repeat_index=repeat_index,
                    first_divergence_index=index,
                    basis=basis.value,
                    stock_length=s_len,
                    candidate_length=c_len,
                    early=index < min_divergence_index,
                )
            )
    return tuple(found)


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
    stock_repeat_metrics: Sequence[MetricsDelta] = (),
    candidate_repeat_metrics: Sequence[MetricsDelta] = (),
    stock_correctness: ReplayResult | None = None,
    candidate_correctness: ReplayResult | None = None,
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
        max_output_mismatches: Allowed *early* divergences -- contexts whose
            two generations parted before
            ``promotion_config.min_divergence_token_index``.  Default 0: the
            gate stays fail-closed, but against a criterion that a correct
            candidate can actually satisfy.
        warmup_repeats: Unscored warmup passes each arm ran before the
            measurement window opened, recorded in the decision for audit.
        stock_repeat_metrics: One :class:`MetricsDelta` per scored repeat,
            from consecutive scrapes taken around each individual suite pass.
            Supplying it is what makes acceptance an n-repeat measurement
            instead of one pooled scalar; when empty, the pooled
            ``stock_metrics`` value stands in for every repeat and the
            published standard deviation is honestly 0.0.
        candidate_repeat_metrics: The same, for the candidate arm.
        stock_correctness: Results of the dedicated output-correctness pass --
            bounded output, concurrency 1, token capture on.  When ``None`` the
            throughput replay is compared instead, which is what the unit tests
            do and what a caller that has not yet been updated will get.
        candidate_correctness: The same, for the candidate arm.

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

    # --- Output correctness ---
    # Compared on its own replay when the caller ran one, so that the question
    # "do these two heads produce the same answer" is asked of a short,
    # single-stream, bounded generation rather than of the batched throughput
    # pass, whose whole purpose is to vary batch composition.
    min_divergence_index = promotion_config.min_divergence_token_index
    divergences = _collect_divergences(
        stock_correctness if stock_correctness is not None else stock_replay,
        candidate_correctness if candidate_correctness is not None else candidate_replay,
        min_divergence_index=min_divergence_index,
    )
    early_by_repeat: dict[int, int] = {}
    for d in divergences:
        if d.early:
            early_by_repeat[d.repeat_index] = early_by_repeat.get(d.repeat_index, 0) + 1
    total_early = sum(early_by_repeat.values())

    # --- Build per-repeat summaries ---
    # Derived purely from replay data plus the per-repeat metric windows, so
    # these survive a metrics failure and keep ``num_repeats == len(per_repeat)``
    # true on every path.
    per_repeat_list: list[RepeatSummary] = []

    for i in range(min_runs):
        s_run = stock_replay.run_results[i]
        c_run = candidate_replay.run_results[i]

        per_repeat_list.append(
            RepeatSummary(
                repeat_index=i,
                stock_tok_per_sec=s_run.output_tok_per_sec,
                candidate_tok_per_sec=c_run.output_tok_per_sec,
                stock_acceptance_rate=_repeat_acceptance(stock_repeat_metrics, i, s_acc),
                candidate_acceptance_rate=_repeat_acceptance(
                    candidate_repeat_metrics, i, c_acc
                ),
                invalid_rate=c_run.invalid_rate,
                output_mismatches=early_by_repeat.get(i, 0),
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

    # --- The gating acceptance statistic ---
    # The same contract as throughput above: the published ``*_avg_acceptance``
    # is *exactly* the mean of the ``per_repeat`` column beside it, and the
    # gated delta is exactly the delta of those two means.  The standard
    # deviations travel with them so a reader can see how much the mean is
    # worth without re-deriving it from the array.
    s_acc_series = [r.stock_acceptance_rate for r in per_repeat_tuple]
    c_acc_series = [r.candidate_acceptance_rate for r in per_repeat_tuple]
    s_acc_mean = _mean(s_acc_series) if s_acc_series else s_acc
    c_acc_mean = _mean(c_acc_series) if c_acc_series else c_acc
    s_acc_sd = _stdev(s_acc_series)
    c_acc_sd = _stdev(c_acc_series)

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
            stock_avg_acceptance=s_acc_mean,
            candidate_avg_acceptance=c_acc_mean,
            stock_avg_tok_per_sec=s_tps,
            candidate_avg_tok_per_sec=c_tps,
            warmup_repeats=warmup_repeats,
            throughput_statistic=GATING_THROUGHPUT_STATISTIC,
            stock_prometheus_decode_tok_per_sec=prom_s_tps,
            candidate_prometheus_decode_tok_per_sec=prom_c_tps,
            prometheus_throughput_delta_pct=prom_delta_pct,
            acceptance_statistic=GATING_ACCEPTANCE_STATISTIC,
            stock_acceptance_stdev=s_acc_sd,
            candidate_acceptance_stdev=c_acc_sd,
            min_divergence_token_index=min_divergence_index,
            output_divergences=divergences,
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
    # Position, not equality: a candidate that parts from stock deep into a
    # long answer has not misbehaved, it has been measured on hardware that is
    # not bitwise reproducible.  One that parts in the first few tokens has.
    if total_early > max_output_mismatches:
        return _reject(Reason.OUTPUT_MISMATCH)

    # --- Validation: throughput unavailable ---
    # The denominator that matters is the gating statistic's.  A Prometheus
    # window that reads zero is a diagnostic gap, not grounds to reject.
    if s_tps <= 0:
        return _reject(Reason.THROUGHPUT_UNAVAILABLE)

    # --- Compute deltas ---
    acceptance_delta_pp = (c_acc_mean - s_acc_mean) * 100.0

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
