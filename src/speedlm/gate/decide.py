"""Promotion decision logic for candidate speculative draft heads.

The output-correctness half of the criterion lives in
:mod:`speedlm.gate.divergence` and is re-exported here, so this module is
still the single import site for the whole gate.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from speedlm.config import (
    DivergenceCriterion,
    PromotionConfig,
    classify_divergence_criterion,
)
from speedlm.gate.divergence import (
    DIVERGENCE_ALPHA,
    DIVERGENCE_PER_STATISTIC_ALPHA,
    DIVERGENCE_STATISTICS,
    ContextDivergence,
    DecisionError,
    DivergenceBasis,
    DivergenceEvidence,
    divergence_excess_p_value,
    divergence_position_p_value,
    evaluate_divergence,
    first_divergence,
)
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import ReplayResult

#: Everything this module offers, including the names it re-exports from
#: :mod:`speedlm.gate.divergence` -- listed so that moving the divergence
#: criterion out did not move a single import site.
__all__ = [
    "CANDIDATE_ARM",
    "CUDA_GRAPH_EXECUTION_MODE",
    "ContextDivergence",
    "DIVERGENCE_ALPHA",
    "DIVERGENCE_PER_STATISTIC_ALPHA",
    "DIVERGENCE_STATISTICS",
    "Decision",
    "DecisionError",
    "DispersionBasis",
    "DivergenceBasis",
    "DivergenceEvidence",
    "EAGER_EXECUTION_MODE",
    "EngineExecution",
    "FLAT_TREND_T_STATISTIC",
    "GATING_ACCEPTANCE_CRITERION",
    "GATING_ACCEPTANCE_STATISTIC",
    "GATING_THROUGHPUT_STATISTIC",
    "LEGACY_ACCEPTANCE_CRITERION",
    "LEGACY_ACCEPTANCE_STATISTIC",
    "LEGACY_THROUGHPUT_STATISTIC",
    "MIN_FLAT_WINDOW",
    "MeasurementBlock",
    "Reason",
    "RepeatSummary",
    "STOCK_ARM",
    "StationarityStatus",
    "ThroughputStationarity",
    "TruncationRegime",
    "UNRECORDED_EXECUTION_MODE",
    "Verdict",
    "classify_truncation",
    "decide_promotion",
    "divergence_excess_p_value",
    "divergence_position_p_value",
    "evaluate_divergence",
    "first_divergence",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """The promotion verdict."""
    PROMOTE = "promote"
    REJECT = "reject"


class StationarityStatus(Enum):
    """What the finite stationarity sample actually established."""

    UNTESTABLE = "untestable"
    STATIONARY = "stationary"
    MATERIAL_SHIFT_UNRESOLVED = "material_shift_unresolved"
    NON_STATIONARY = "non_stationary"


class TruncationRegime(Enum):
    """What the output cap did to the generations a decision was measured over.

    ``finish_reason`` was recorded on every replayed request from the day the
    replay path was written and read by nothing: a run in which the harness
    ended *every* generation reported ``invalid_rate 0.0`` and was gated as
    though it had measured the workload.  It had not -- it had measured
    fixed-length decode.  This is the type that says which.

    The states are ordered by how much of the workload's own stopping behaviour
    survived into the measurement, and the boundaries are counts rather than
    tuned fractions:

    * :attr:`UNTESTABLE` -- no response reported a finish reason this build can
      classify.  Every archived decision reads this, because none of them
      persisted the counts, and so does a run behind a server whose finish-reason
      spellings are all unrecognised (see
      :func:`speedlm.gate.replay.classify_finish_reason`).  It must never be
      collapsed into ``BOUNDED``, which is the claim that truncation was
      *measured* and was low.
    * :attr:`SATURATED` -- finish reasons were reported and **not one**
      generation ended on the model's own terms.  Zero, not "few": at zero the
      run contains no observation whatsoever of where this model stops, so no
      throughput figure from it can be attributed to the workload rather than
      to ``benchmark_max_tokens``.  This is the only state that gates.
    * :attr:`MIXED` -- most generations hit the cap, but some did not.  The
      measurement is dominated by the cap and is still interpretable, because
      the natural stops bound how much of the distribution was clipped.  This
      is the state the archived live runs sit in: SLURM 8b72d9a captured 1889
      of 2049 Qwen3-8B responses and 1747 of 2049 gpt-oss-20b responses at
      ``length``, i.e. 92.2% and 85.3%.
    * :attr:`BOUNDED` -- most generations stopped by themselves; the cap was
      not the binding constraint on length.

    A regime is not a verdict on the draft head.  Truncation is a property of
    the harness configuration, and the same head measured under a larger cap
    would report a different one.
    """

    UNTESTABLE = "untestable"
    BOUNDED = "bounded"
    MIXED = "mixed"
    SATURATED = "saturated"


def classify_truncation(*, reported: int, truncated: int) -> TruncationRegime:
    """Classify one arm's truncation from its pooled finish-reason counts.

    Args:
        reported: Responses whose finish reason the replay could *classify* --
            i.e. truncations plus known natural stops.  Deliberately not "every
            response that carried a non-empty string": a spelling the replay
            does not recognise is excluded from this denominator, exactly as a
            blank or absent reason is, so ``reported - truncated`` is a count of
            observed natural stops rather than an inference from the absence of
            a truncation marker.  See
            :func:`speedlm.gate.replay.classify_finish_reason`.
        truncated: Of those, the ones the output cap ended.
    """
    if reported <= 0:
        return TruncationRegime.UNTESTABLE
    natural_stops = reported - truncated
    if natural_stops <= 0:
        return TruncationRegime.SATURATED
    # A majority boundary, and deliberately a description rather than a
    # threshold: nothing gates on the MIXED/BOUNDED split, so it carries none
    # of the calibration burden that ``SATURATED``'s exact zero avoids.
    if truncated * 2 > reported:
        return TruncationRegime.MIXED
    return TruncationRegime.BOUNDED


#: The two arms every measurement in this module is paired across, spelled once
#: so the per-arm readings below cannot drift apart by a typo.
STOCK_ARM: Final[str] = "stock"
CANDIDATE_ARM: Final[str] = "candidate"


@dataclass(frozen=True, slots=True)
class _TruncationCounts:
    """One arm's pooled finish-reason counts, and the three readings of them.

    The counts are pooled across repeats rather than averaged, because
    ``SATURATED`` is a statement about a *total* count of natural stops and a
    per-repeat mean would round it away.
    """

    #: Responses whose finish reason the replay could classify.
    reported: int
    #: Of those, the ones the output cap ended.
    truncated: int

    @property
    def rate(self) -> float | None:
        """Truncated share of the reported responses, or ``None`` if unmeasured.

        ``None`` rather than ``0.0`` when nothing reported a finish reason, so
        that a record which never measured truncation cannot be read as one
        that measured zero.
        """
        if self.reported <= 0:
            return None
        return self.truncated / self.reported

    @property
    def regime(self) -> TruncationRegime:
        return classify_truncation(reported=self.reported, truncated=self.truncated)


def _truncation_counts(
    per_repeat: Sequence[RepeatSummary], arm: str
) -> _TruncationCounts:
    """Pool one arm's finish-reason counts out of the per-repeat array.

    Both the gate and the ``Decision`` properties read the regime through here,
    from the same array, on purpose.  ``decide_promotion`` used to sum the
    columns itself while the record's properties summed them again: two
    derivations of one fact, and a verdict whose own evidence contradicts it is
    this project's recurring defect.  One reader makes that divergence
    unrepresentable rather than merely unlikely.
    """
    if arm == STOCK_ARM:
        return _TruncationCounts(
            reported=sum(r.stock_finish_reasons for r in per_repeat),
            truncated=sum(r.stock_truncated for r in per_repeat),
        )
    return _TruncationCounts(
        reported=sum(r.candidate_finish_reasons for r in per_repeat),
        truncated=sum(r.candidate_truncated for r in per_repeat),
    )


class Reason(Enum):
    """Enumerated reasons for a verdict."""
    BOTH_THRESHOLDS_MET = "both_thresholds_met"
    ACCEPTANCE_BELOW_THRESHOLD = "acceptance_below_threshold"
    THROUGHPUT_BELOW_THRESHOLD = "throughput_below_threshold"
    COUNTER_RESET = "counter_reset"
    ACCEPTANCE_UNAVAILABLE = "acceptance_unavailable"
    THROUGHPUT_UNAVAILABLE = "throughput_unavailable"
    HIGH_INVALID_RATE = "high_invalid_rate"
    #: An arm produced no generation that ended on the model's own terms -- see
    #: :attr:`TruncationRegime.SATURATED`.  A measurement rejection, in the same
    #: family as ``COUNTER_RESET`` and ``HIGH_INVALID_RATE``: it says the run
    #: cannot support a promotion, not that the head is worse.
    TRUNCATION_SATURATED = "truncation_saturated"
    #: Not one valid response in an arm reported a finish reason, so the
    #: saturation guard above had nothing to read -- see
    #: :attr:`TruncationRegime.UNTESTABLE`.  A missing-instrument rejection in
    #: the same family as ``ACCEPTANCE_UNAVAILABLE``, and distinct from
    #: ``TRUNCATION_SATURATED``: saturation says the cap ended every generation,
    #: this says nobody said what ended any of them.
    TRUNCATION_UNMEASURED = "truncation_unmeasured"
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

#: Names the acceptance-side *quantity* the promotion criterion is applied to.
#:
#: ``GATING_ACCEPTANCE_STATISTIC`` above says how an arm's acceptance figure is
#: pooled across repeats.  This says *which figure* -- a different question, and
#: the answer changed.
#:
#: It used to be the acceptance rate, ``accepted_tokens / drafted_tokens``.
#: Since ``drafted_tokens == num_drafts * k`` for a draft chain of depth ``k``,
#: that is algebraically ``(mean_accepted_length - 1) / k``: the depth sits in
#: the denominator, so the statistic is **not comparable across draft depths**
#: and a deeper chain scores *lower* on it while emitting *more* tokens per
#: step.  gpt-oss-20b at ``k=5`` scores 0.356 against Qwen3-8B's 0.380 at
#: ``k=3`` while emitting 2.82 tokens per verifier step against 2.14.  Worse,
#: gpt-oss's per-position conditional acceptance still rises at the last drafted
#: position (0.689 -> 0.715 at position 5), so raising ``k`` would raise
#: throughput while dropping the rate to ~0.31 -- and the primary promotion
#: criterion would have rejected the largest speedup on the table.
#:
#: ``mean_accepted_length = 1 + accepted_tokens / num_drafts`` is tokens per
#: verifier *step*.  No ``k`` in the denominator, directly proportional to
#: throughput at fixed step cost, and already computed per repeat by
#: :class:`speedlm.gate.metrics.MetricsDelta`.  See
#: :attr:`speedlm.config.PromotionConfig.min_accepted_length_delta` for the
#: threshold and its calibration against the archived runs.
GATING_ACCEPTANCE_CRITERION: Final[str] = "mean_accepted_length_delta"

#: What decided records written before the criterion moved off the rate.  Those
#: runs compared ``acceptance_delta_pp`` against ``min_acceptance_delta_pp``;
#: both fields are still recorded and still mean what they meant, so a loader
#: labels the archived record with this rather than relabelling it as the
#: current criterion.
LEGACY_ACCEPTANCE_CRITERION: Final[str] = "acceptance_rate_delta_pp"

#: Execution mode of an engine launched with ``--enforce-eager``.
EAGER_EXECUTION_MODE: Final[str] = "eager"

#: Execution mode of an engine left to capture CUDA graphs / ``torch.compile``.
CUDA_GRAPH_EXECUTION_MODE: Final[str] = "cuda_graph"

#: What a record says when nothing told the gate which mode it measured under.
#:
#: This is *not* a synonym for eager.  Every archived SpeedLM run happens to
#: have been launched with ``--enforce-eager`` (it is operator passthrough --
#: see ``scripts/make_snapshot_run.sh`` and ``tests/e2e/run_slurm_e2e.sh``),
#: but no record says so, and the gate had no way to know: the runner owns no
#: vLLM process.  Reading those records as eager would be inferring a fact
#: from a habit.  A record that predates this field, or a runner that was
#: given no engine description, says ``unrecorded`` and means it.
UNRECORDED_EXECUTION_MODE: Final[str] = "unrecorded"


@dataclass(frozen=True, slots=True)
class EngineExecution:
    """How the engine both arms replayed against was actually executing.

    None of this changes what the gate *measures*; all of it changes what a
    measured number is worth.  Eager and CUDA-graph execution are a large
    throughput difference on the same weights, and for a MoE target they are
    not even bitwise equivalent -- which is the same property
    :data:`DIVERGENCE_ALPHA` exists to reason about, since the divergence
    floor a run is judged against is a property of its kernels and batch
    shapes.  A ``decision.json`` that records ``56.64`` tok/s without recording
    which of the two produced it cannot be honestly compared to any other run.

    The repo carries a claim that ``--enforce-eager`` is "speed-only, with zero
    acceptance effect".  Nothing in this repository measures that, and this
    class deliberately does not encode it: it records the mode as a fact and
    leaves the question open.  Note the opposite is at least plausible --
    acceptance is counted per verifier step, and the drafted-vs-verified
    comparison happens in kernels that a CUDA-graph capture may select
    differently.
    """

    enforce_eager: bool
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    max_num_seqs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enforce_eager, bool):
            raise ValueError("enforce_eager must be a bool")
        for name in ("enable_chunked_prefill", "enable_prefix_caching"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool or None")
        if self.max_num_seqs is not None and (
            isinstance(self.max_num_seqs, bool)
            or not isinstance(self.max_num_seqs, int)
            or self.max_num_seqs < 1
        ):
            raise ValueError("max_num_seqs must be an integer >= 1 or None")

    @property
    def execution_mode(self) -> str:
        """:data:`EAGER_EXECUTION_MODE` or :data:`CUDA_GRAPH_EXECUTION_MODE`."""

        return (
            EAGER_EXECUTION_MODE if self.enforce_eager else CUDA_GRAPH_EXECUTION_MODE
        )

    @classmethod
    def from_argv(cls, argv: Sequence[str]) -> EngineExecution:
        """Read the execution-relevant flags out of a ``vllm serve`` argv.

        ``argv`` is what
        :func:`speedlm.gateway.process.build_vllm_argv` produced for the engine
        the gate replayed against, so this reads the *served* configuration
        rather than a restatement of it.  Flags absent from the argv are
        recorded as ``None`` -- "the operator did not pass it, so vLLM's
        default applied" -- and never as ``False``, because vLLM's defaults for
        chunked prefill and prefix caching are version-dependent and guessing
        them would put a fact in the record that nobody measured.
        ``enforce_eager`` is the one exception: it is a bare store-true flag,
        so its absence really does mean the engine was left to capture graphs.
        """

        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise ValueError("argv must be a sequence of strings")
        tokens = [str(item) for item in argv]
        enforce_eager = "--enforce-eager" in tokens
        chunked = _argv_toggle(tokens, "enable-chunked-prefill")
        prefix = _argv_toggle(tokens, "enable-prefix-caching")
        max_num_seqs = _argv_int_option(tokens, "--max-num-seqs")
        return cls(
            enforce_eager=enforce_eager,
            enable_chunked_prefill=chunked,
            enable_prefix_caching=prefix,
            max_num_seqs=max_num_seqs,
        )


def _argv_toggle(tokens: Sequence[str], name: str) -> bool | None:
    """Resolve a vLLM ``--flag`` / ``--no-flag`` / ``--flag=true`` triple."""

    result: bool | None = None
    truthy = {"true", "1", "yes"}
    falsy = {"false", "0", "no"}
    for index, token in enumerate(tokens):
        if token == f"--{name}":
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following.lower() in truthy:
                result = True
            elif following.lower() in falsy:
                result = False
            else:
                result = True
        elif token == f"--no-{name}":
            result = False
        elif token.startswith(f"--{name}="):
            value = token.split("=", 1)[1].lower()
            if value in truthy:
                result = True
            elif value in falsy:
                result = False
    return result


def _argv_int_option(tokens: Sequence[str], flag: str) -> int | None:
    """Resolve ``--flag N`` or ``--flag=N`` to an int, or ``None`` if absent."""

    raw: str | None = None
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            raw = tokens[index + 1]
        elif token.startswith(f"{flag}="):
            raw = token.split("=", 1)[1]
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed >= 1 else None



class DispersionBasis(Enum):
    """Whether a published standard deviation is a *measurement* or an artefact.

    A standard deviation of zero and a standard deviation of 0.03 look alike to
    a consumer -- both are "small" -- but they mean opposite things, and the
    difference is not recoverable from the number itself.  Job 369162
    (gpt-oss-20b) published ``candidate_acceptance_stdev = 0.0`` and
    ``stock_acceptance_stdev = 0.0`` across five repeats.  Anything scoring the
    measurement as ``min_acceptance_delta_pp / standard_error`` reads that as
    *infinitely many* standard errors of headroom and concludes the gate is
    superbly resolved.  It is the reverse: five repeats produced five
    bit-identical readings, so there is no variance estimate at all, and the
    threshold's margin is unknown rather than large.

    Naming the three states keeps that distinction in the record:

    * ``MEASURED`` -- two or more repeats, and they disagreed.  The published
      standard error means what it says.
    * ``DEGENERATE`` -- two or more repeats, and every one returned the same
      value in both arms.  The standard error is exactly zero because nothing
      varied, which is evidence of *no measurement*, not of a tight one.  The
      accompanying standard-error field is ``None`` rather than ``0.0``, so a
      consumer dividing by it fails loudly instead of computing infinity.
    * ``UNSAMPLED`` -- fewer than two repeats.  There was never anything to
      disperse.

    This is a reporting distinction only.  ``DEGENERATE`` is not a rejection
    reason: for acceptance it is the *expected* state (see
    :data:`GATING_ACCEPTANCE_STATISTIC`), and turning the expected state into a
    reject would stop the gate promoting anything at all.
    """

    MEASURED = "measured"
    DEGENERATE = "degenerate"
    UNSAMPLED = "unsampled"



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
    #: row 0 is ever non-zero -- and rows at or beyond
    #: ``Decision.correctness_repeats`` report zero because *no correctness
    #: pass ran at that index*, not because one ran and found nothing.  Read
    #: this column against ``Decision.correctness_repeats``, never against
    #: ``num_repeats``: they index different passes that merely share a
    #: counter.  ``Decision.output_early_divergences`` is the total the gate
    #: actually compares against ``max_output_mismatches``.
    output_mismatches: int
    #: Mean accepted length over *this repeat's* Prometheus window, per arm --
    #: ``1 + accepted_tokens / num_drafts``, i.e. tokens emitted per verifier
    #: forward pass.  This is the column the promotion criterion is computed
    #: from; see :data:`GATING_ACCEPTANCE_CRITERION`.
    #:
    #: Defaulted so that an archived ``decision.json`` -- every one of which
    #: predates these two columns -- still rebuilds through
    #: :func:`speedlm.report.parse_decision`.  ``0.0`` is not a reachable
    #: measurement (the floor is 1.0, the bonus token), so "record predates the
    #: field" stays distinguishable from any real reading.
    stock_accepted_length: float = 0.0
    candidate_accepted_length: float = 0.0

    # -- what the output cap did to this repeat ------------------------------
    # Counts, not rates, and one denominator per arm.  Rates are recoverable
    # from counts and counts are not recoverable from rates, and the
    # denominator is the field that distinguishes "nothing was truncated" from
    # "the endpoint never said" -- the distinction the whole
    # :class:`TruncationRegime` classification turns on.  All default to 0, so
    # an archived record rebuilt by :func:`speedlm.report.parse_decision`
    # classifies as ``UNTESTABLE`` rather than being relabelled ``BOUNDED``.

    #: Responses in this repeat whose finish reason was classifiable, per arm --
    #: truncations plus known natural stops.  Unrecognised spellings are in
    #: neither, and so in neither this count nor the one below.
    stock_finish_reasons: int = 0
    candidate_finish_reasons: int = 0
    #: Of those, the ones the output cap ended, per arm.
    stock_truncated: int = 0
    candidate_truncated: int = 0


@dataclass(frozen=True, slots=True)
class MeasurementBlock:
    """One contiguous measurement block, as it was actually run.

    Persisted so a decision record says how its two arms were interleaved
    without anybody having to reconstruct the order from an engine log.  The
    order of ``Decision.block_schedule`` *is* the order the blocks ran in.
    """

    #: ``"stock"`` or ``"candidate"``.
    arm: str
    #: Scored repeats this block contributed.
    repeats: int
    #: Whether opening this block cost an engine restart.  Every scored block
    #: must read ``True``: otherwise exactly one arm can inherit a lifecycle the
    #: cycle established before the benchmark, while its peer is measured after
    #: activation.  See
    #: :data:`speedlm.gate.runner.DEFAULT_ARM_BLOCKS`.
    restarted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "repeats": self.repeats,
            "restarted": self.restarted,
        }


@dataclass(frozen=True, slots=True)
class ThroughputStationarity:
    """Whether the arm-to-arm throughput delta held still during the run.

    Computed by :func:`speedlm.gate.runner._stationarity`, which owns the
    reasoning; this type exists so the answer survives onto disk.  It used to
    be published only on ``GateResult.metrics``, which nothing persists, so a
    run could not be asked afterwards whether its veto was live.
    """

    #: Whether no material shift was observed in a testable sample.  False does
    #: not by itself mean drift was proven: inspect :attr:`status` to distinguish
    #: an unresolved material shift and an untestable sample from a proven one.
    testable: bool
    stationary: bool
    required_for_promotion: bool
    min_repeats: int
    #: Percentage points the paired delta column moved at its best split, and
    #: the studentised size of that move.  ``None`` when untestable.
    delta_shift_pct: float | None
    delta_shift_t_statistic: float | None
    min_shift_t_statistic: float
    materiality_pct: float
    stock_flat_from_repeat: int | None
    candidate_flat_from_repeat: int | None
    stock_trend_pct_per_repeat: float | None
    candidate_trend_pct_per_repeat: float | None

    @property
    def status(self) -> StationarityStatus:
        """Classify absent, immaterial, unresolved, and proven shift evidence."""
        if (
            not self.testable
            or self.delta_shift_pct is None
            or self.delta_shift_t_statistic is None
        ):
            return StationarityStatus.UNTESTABLE
        if abs(self.delta_shift_pct) < self.materiality_pct:
            return StationarityStatus.STATIONARY
        if self.delta_shift_t_statistic >= self.min_shift_t_statistic:
            return StationarityStatus.NON_STATIONARY
        return StationarityStatus.MATERIAL_SHIFT_UNRESOLVED

    @property
    def vetoed(self) -> bool:
        """Whether the stationarity evidence vetoes an otherwise-promotion."""
        return (
            self.required_for_promotion
            and self.status is StationarityStatus.NON_STATIONARY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "testable": self.testable,
            "min_repeats": self.min_repeats,
            "delta_shift_pct": self.delta_shift_pct,
            "delta_shift_t_statistic": self.delta_shift_t_statistic,
            "min_shift_t_statistic": self.min_shift_t_statistic,
            "materiality_pct": self.materiality_pct,
            "stock_flat_from_repeat": self.stock_flat_from_repeat,
            "candidate_flat_from_repeat": self.candidate_flat_from_repeat,
            "stock_trend_pct_per_repeat": self.stock_trend_pct_per_repeat,
            "candidate_trend_pct_per_repeat": self.candidate_trend_pct_per_repeat,
            "status": self.status.value,
            # Derived on serialization so legacy records whose Boolean meant
            # "not proven otherwise" are normalized from their actual evidence.
            "stationary": self.status is StationarityStatus.STATIONARY,
            "vetoed": self.vetoed,
            "required_for_promotion": self.required_for_promotion,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """The promotion decision with full reporting."""

    verdict: Verdict
    reason: Reason
    #: Arm-to-arm delta of the acceptance *rate*, in percentage points.
    #:
    #: **Recorded, not gated** -- see
    #: :attr:`speedlm.config.PromotionConfig.min_acceptance_delta_pp`.  It is
    #: still exactly what it always was (the delta of the two
    #: ``*_avg_acceptance`` means), it is still meaningful at a fixed draft
    #: depth, and it is the number every archived run is described by, so it
    #: keeps its name and is populated on every path that computes deltas.  What
    #: it no longer does is decide: it carries the draft depth in its
    #: denominator and so cannot be compared across depths.
    #: :attr:`accepted_length_delta` is the criterion.
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
    #: before ``min_divergence_token_index``.
    output_divergences: tuple[ContextDivergence, ...] = ()

    # -- the divergence noise floor -----------------------------------------
    # The stock arm replayed against *itself*.  Same engine, same process, same
    # draft, same sampling, same cap, same concurrency -- so every divergence
    # recorded here is engine nondeterminism by construction, and the candidate
    # arm's count is only meaningful relative to it.  See
    # :data:`DIVERGENCE_ALPHA`.
    #
    # Every field below is ``compare=False``.  They are written to
    # ``decision.json`` by :meth:`to_dict` and are fully auditable there, but
    # :func:`speedlm.report.parse_decision` rebuilds a ``Decision`` field by
    # explicitly named field and does not yet know about them, so a record that
    # round-trips through the report layer comes back with them at their
    # defaults.  Excluding them from equality keeps "did this decision survive
    # persistence" answering the question it is asked -- rather than answering
    # "has the report layer been taught the newest evidence fields", which is a
    # different question with a different owner.  Restoring them in
    # ``parse_decision`` is what makes ``speedlm gain`` able to show the floor
    # beside the count; until then the floor lives in the artifact only.

    #: Context comparisons the candidate-versus-stock evidence rests on, i.e.
    #: contexts x correctness repeats.  The denominator of
    #: ``output_total_divergences``.
    divergence_trials: int = field(default=0, compare=False)
    #: The same for the stock-versus-stock control.  Zero means no control was
    #: run; see :attr:`divergence_control_available`.
    control_trials: int = field(default=0, compare=False)
    #: Every context whose two *stock* generations parted.  Persisted in full,
    #: exactly as ``output_divergences`` is: the noise floor is the evidence the
    #: verdict turns on, so a reader must be able to audit it after the fact
    #: rather than take the summary counts on trust.
    control_divergences: tuple[ContextDivergence, ...] = field(
        default=(), compare=False
    )
    #: Whether a stock-versus-stock control pass was actually measured.  When
    #: false the criterion falls back to an *assumed* floor of zero divergences
    #: over an equal number of trials -- the strictest floor that is still a
    #: test rather than a hair trigger -- and this flag is what stops that
    #: assumption being read back as a measurement.
    divergence_control_available: bool = field(default=False, compare=False)
    #: One-sided Fisher exact p-value for "the candidate diverges more often
    #: than the engine does on its own", over all divergences.  ``None`` when
    #: there was nothing to test.
    divergence_total_p_value: float | None = field(default=None, compare=False)
    #: The same, restricted to divergences before ``min_divergence_token_index``.
    divergence_early_p_value: float | None = field(default=None, compare=False)
    #: Whether the control pair spans the same engine boundary as the measured
    #: pair, and is therefore a valid null for it.
    #:
    #: The two Fisher p-values above compare "candidate arm versus stock arm"
    #: against "stock arm versus stock arm".  That is only a test if both pairs
    #: were collected across the same conditions.  In the runner's topology they
    #: are not: the measured pair straddles a full vLLM teardown, weight reload
    #: and cache rebuild, while the control's two passes run back-to-back inside
    #: one live engine.  A same-engine control replays an identical computation
    #: and so reports zero by construction, which turns the Fisher test into a
    #: hair trigger on any nonzero measured count.  When this is false the rate
    #: channel is recorded but does not gate; see
    #: :attr:`divergence_position_p_value` for the channel that does.
    divergence_control_comparable: bool = field(default=False, compare=False)
    #: One-sided p-value for "the divergences are front-loaded", from
    #: :func:`divergence_position_p_value`.  This is the control-free channel,
    #: and it is what rejects a genuinely broken head.  ``None`` when the run
    #: recorded no early divergence to test.
    divergence_position_p_value: float | None = field(default=None, compare=False)
    #: Per-statistic significance level each p-value above was compared against,
    #: i.e. :data:`DIVERGENCE_ALPHA` over :data:`DIVERGENCE_STATISTICS`.
    divergence_alpha: float = field(
        default=DIVERGENCE_PER_STATISTIC_ALPHA, compare=False
    )

    # -- the promotion criterion --------------------------------------------
    # Added beside ``acceptance_delta_pp`` rather than replacing it: the rate
    # delta keeps its name, its units and its meaning, and these carry the
    # decision.  All default so an archived record still rebuilds; see
    # :data:`GATING_ACCEPTANCE_CRITERION`.

    #: Arm-to-arm delta of mean accepted length, in tokens per verifier step.
    #: This is what ``min_accepted_length_delta`` is compared against.
    #: ``None`` on every path that short-circuits above the delta computation,
    #: exactly as ``acceptance_delta_pp`` is.
    accepted_length_delta: float | None = None
    #: The bar :attr:`accepted_length_delta` had to clear.  ``None`` means the
    #: record predates the criterion and was gated on the rate instead -- read
    #: :attr:`acceptance_criterion` rather than inferring from this.
    min_accepted_length_delta: float | None = None
    #: Mean of the ``per_repeat`` accepted-length column, per arm -- the same
    #: by-construction contract ``*_avg_acceptance`` and ``*_avg_tok_per_sec``
    #: hold, so ``accepted_length_delta`` is reproducible by hand from the
    #: array.
    stock_avg_accepted_length: float = 0.0
    candidate_avg_accepted_length: float = 0.0
    #: Sample standard deviation of that column, per arm.  Expected to be 0.0
    #: on a deterministic greedy replay; read it through
    #: :attr:`accepted_length_dispersion`, never as "a very tight measurement".
    stock_accepted_length_stdev: float = 0.0
    candidate_accepted_length_stdev: float = 0.0
    #: Which acceptance-side quantity gated, spelled out rather than implied.
    #: :data:`GATING_ACCEPTANCE_CRITERION` on records written by this gate;
    #: :data:`LEGACY_ACCEPTANCE_CRITERION` is what a loader stamps on an
    #: archived record that predates the field.
    acceptance_criterion: str = GATING_ACCEPTANCE_CRITERION

    # -- measurement context -------------------------------------------------
    # What the numbers above were produced under.  Every field here changes the
    # value of a reported statistic without changing its name, so a decision
    # that omits them is not comparable to another decision: the gate knew all
    # of them (they were on ``GateResult.metrics``) and none of them reached
    # ``decision.json``, which is the only file ``speedlm gain`` reads.  All
    # default to ``None`` -- "this record predates the field" -- so an archived
    # decision stays readable and is never silently relabelled.

    #: Output cap of the throughput/acceptance pass.  It bounds the acceptance
    #: window as well as the wall clock; see
    #: :attr:`speedlm.config.IdleTuningConfig.benchmark_max_tokens`, which
    #: measures the bias as roughly +0.1 pp on the acceptance delta.
    benchmark_max_tokens: int | None = None
    #: Requests kept in flight per arm during that pass.  The gating throughput
    #: statistic divides completion tokens by the *sum* of per-request
    #: latencies, so its absolute value moves with how much the engine batched.
    replay_concurrency: int | None = None
    #: Output cap of the separate, single-stream correctness pass -- the one
    #: that produces ``output_divergences``.
    correctness_max_tokens: int | None = None
    #: Suite passes that correctness pass made, per arm.  Recorded because
    #: ``per_repeat`` is indexed by the *throughput* pass's repeat and carries
    #: an ``output_mismatches`` column anyway: without this field a record
    #: reading ``num_repeats: 5`` and ``output_mismatches: [48, 0, 0, 0, 0]``
    #: is indistinguishable from five correctness passes of which four were
    #: clean, when in fact one pass ran and rows 1..4 were never measured.
    #: Job 369373 produced exactly that record.
    correctness_repeats: int | None = None
    #: Identity of the frozen held-out suite both arms replayed.  Two deltas
    #: measured over different suites are different measurements.
    suite_hash: str | None = None
    #: Contexts in that suite.
    num_contexts: int | None = None
    #: The draft the stock arm ran, i.e. the baseline this delta is *against*.
    #: Recorded because it changes on every promotion: without it a reader
    #: cannot tell improvement over the incumbent from improvement over the
    #: original head.
    stock_draft: str | None = None

    #: How the engine was executing: :data:`EAGER_EXECUTION_MODE`,
    #: :data:`CUDA_GRAPH_EXECUTION_MODE`, or :data:`UNRECORDED_EXECUTION_MODE`
    #: when the gate was told nothing.  Always a string, never ``None``: the
    #: absence of knowledge is itself the fact worth persisting, and a ``None``
    #: here would be indistinguishable from a key a reader forgot to add.
    engine_execution_mode: str = UNRECORDED_EXECUTION_MODE
    #: The flags behind that mode, each ``None`` when unrecorded.  See
    #: :class:`EngineExecution`.
    engine_enforce_eager: bool | None = None
    engine_enable_chunked_prefill: bool | None = None
    engine_enable_prefix_caching: bool | None = None
    engine_max_num_seqs: int | None = None

    # -- how the measurement was taken --------------------------------------
    # Attached by the runner after the numbers are in, because both are
    # properties of the *schedule* rather than of the comparison.  They were
    # published on ``GateResult.metrics``, which nothing persists; a record
    # that cannot say whether it interleaved, or whether the stationarity veto
    # was live, forces the next diagnosis back through the vLLM log.

    #: Blocks each arm's scored repeats were split into.  ``1`` is the
    #: sequential design (all of one arm, then all of the other); ``None`` on
    #: records written before the schedule was persisted.
    arm_blocks: int | None = None
    #: The realized block order, one entry per block, in the order run.
    block_schedule: tuple[MeasurementBlock, ...] = ()
    #: Whether the columns this decision was computed from had stopped moving.
    #: ``None`` means the record predates the test, never "it was stationary".
    throughput_stationarity: ThroughputStationarity | None = None

    # -- dispersion of the gated statistics ---------------------------------
    # Derived from ``per_repeat`` rather than stored, so an archived decision
    # rebuilt by ``speedlm.report.parse_decision`` reports the same basis the
    # gate did -- the per-repeat array is the evidence, and it has always been
    # persisted.  Nothing here gates; see :class:`DispersionBasis`.

    @property
    def acceptance_dispersion(self) -> DispersionBasis:
        """Whether the acceptance repeats actually disagreed."""
        return _dispersion_basis(
            [r.stock_acceptance_rate for r in self.per_repeat],
            [r.candidate_acceptance_rate for r in self.per_repeat],
        )

    @property
    def acceptance_delta_standard_error_pp(self) -> float | None:
        """Standard error of ``acceptance_delta_pp``, or ``None`` if unmeasured.

        ``None`` for both :attr:`DispersionBasis.DEGENERATE` and
        :attr:`DispersionBasis.UNSAMPLED`, because in neither case does a
        standard error exist.  Returning ``0.0`` there would be read as a
        perfect measurement by anything that divides a threshold by it.
        """
        return _delta_standard_error(
            [r.stock_acceptance_rate for r in self.per_repeat],
            [r.candidate_acceptance_rate for r in self.per_repeat],
            scale=100.0,
        )

    @property
    def accepted_length_dispersion(self) -> DispersionBasis:
        """Whether the accepted-length repeats actually disagreed.

        Same reading as :attr:`acceptance_dispersion`, and ``DEGENERATE`` is
        likewise the *expected* state: both columns are derived from the same
        deterministic counter windows, so neither samples anything.
        """
        return _dispersion_basis(
            [r.stock_accepted_length for r in self.per_repeat],
            [r.candidate_accepted_length for r in self.per_repeat],
        )

    @property
    def accepted_length_delta_standard_error(self) -> float | None:
        """Standard error of :attr:`accepted_length_delta`, in tokens/step.

        ``None`` for ``DEGENERATE`` and ``UNSAMPLED`` alike, so anything that
        divides the threshold by it fails loudly rather than computing an
        infinitely well-resolved gate.  See :class:`DispersionBasis`.
        """
        return _delta_standard_error(
            [r.stock_accepted_length for r in self.per_repeat],
            [r.candidate_accepted_length for r in self.per_repeat],
            scale=1.0,
        )

    @property
    def throughput_dispersion(self) -> DispersionBasis:
        """Whether the throughput repeats actually disagreed.

        Unlike acceptance this is expected to be ``MEASURED`` on live hardware;
        ``DEGENERATE`` here means the replay was stubbed or the clock was not
        read, not that the machine is noiseless.
        """
        return _dispersion_basis(
            [r.stock_tok_per_sec for r in self.per_repeat],
            [r.candidate_tok_per_sec for r in self.per_repeat],
        )

    @property
    def throughput_delta_standard_error_pct(self) -> float | None:
        """Standard error of ``throughput_delta_pct``, in percent of stock.

        Read this together with the two ``*_throughput_trend_pct_per_repeat``
        properties.  It is a ``sd/sqrt(n)`` figure, which assumes the repeats
        are exchangeable; when an arm is still warming they are not, and the
        trend is what says so.
        """
        stock = [r.stock_tok_per_sec for r in self.per_repeat]
        mean_stock = _mean(stock)
        if mean_stock <= 0:
            return None
        return _delta_standard_error(
            stock,
            [r.candidate_tok_per_sec for r in self.per_repeat],
            scale=100.0 / mean_stock,
        )

    @property
    def stock_throughput_trend_pct_per_repeat(self) -> float | None:
        """Least-squares slope of the stock throughput column, %/repeat.

        Published because a warming arm breaks the assumption underneath every
        ``sd/sqrt(n)`` figure in this record, and the breakage is invisible in
        the standard deviation alone -- drift inflates it exactly as noise
        does.  On jobs 369161/369162 the *candidate* arm (which runs first, see
        ``IdleTuningConfig.benchmark_candidate_arm_first``) trended
        +0.63%/repeat and +0.80%/repeat, accounting for 83% and 94% of that
        arm's per-repeat variance, while stock trended -0.42%/repeat and
        +0.13%/repeat with no such structure.  A large positive candidate trend
        means the measurement window opened before the arm reached steady
        state, and that the reported delta understates the candidate.
        """
        return _trend_pct_per_repeat([r.stock_tok_per_sec for r in self.per_repeat])

    @property
    def candidate_throughput_trend_pct_per_repeat(self) -> float | None:
        """Least-squares slope of the candidate throughput column, %/repeat."""
        return _trend_pct_per_repeat([r.candidate_tok_per_sec for r in self.per_repeat])

    @property
    def stock_throughput_flat_from_repeat(self) -> int | None:
        """Earliest repeat the stock arm had stopped warming from, if it did."""
        return _flat_from_repeat([r.stock_tok_per_sec for r in self.per_repeat])

    @property
    def candidate_throughput_flat_from_repeat(self) -> int | None:
        """Earliest repeat the candidate arm had stopped warming from, if it did.

        The trend properties say an arm is still warming; this says *for how
        long*, which is the number ``tuning.warmup_repeats`` has to be argued
        from.  ``None`` is the answer jobs 369161/369162 would give: they show
        only that the candidate had not flattened by repeat five, and neither
        run scored enough repeats to find where it does.

        Reading it: a value of ``k`` means repeats ``k..n-1`` are mutually
        exchangeable, so an arm given ``warmup_repeats + k`` unscored passes
        would open its measurement window warm.  ``0`` means the arm was
        already warm at the first scored repeat and the existing warmup was
        sufficient.

        It does not gate, and deliberately so: it is a diagnostic over a single
        run's five-to-N samples, and the gate is the only safeguard with no
        rollback behind it.  Its job is to accumulate across runs so that the
        warmup/repeat trade stops being an argument.  That trade is *not* a
        free efficiency win -- a warmup pass costs exactly what a scored repeat
        costs (see :class:`speedlm.config.IdleTuningConfig.warmup_repeats`), so
        buying an unbiased window at ``k > 0`` strictly adds passes unless
        ``k`` is small enough that ``warmup + repeats`` still falls.
        """
        return _flat_from_repeat([r.candidate_tok_per_sec for r in self.per_repeat])

    # -- what the output cap did ---------------------------------------------
    # Derived from ``per_repeat`` on the same contract the dispersion
    # properties hold: the array is the evidence, and these are readings of it
    # that a reader can reproduce by hand.  Pooled across repeats rather than
    # averaged, because ``SATURATED`` is a statement about a total count of
    # natural stops and a per-repeat mean would round it away.

    @property
    def stock_finish_reasons_reported(self) -> int:
        """Stock responses that carried a finish reason, all repeats pooled."""
        return _truncation_counts(self.per_repeat, STOCK_ARM).reported

    @property
    def candidate_finish_reasons_reported(self) -> int:
        """Candidate responses that carried a finish reason, all repeats pooled."""
        return _truncation_counts(self.per_repeat, CANDIDATE_ARM).reported

    @property
    def stock_truncation_rate(self) -> float | None:
        """Fraction of reported stock generations that hit the output cap.

        ``None`` when nothing reported a finish reason, so that a record which
        never measured truncation cannot be read as one that measured zero.
        """
        return _truncation_counts(self.per_repeat, STOCK_ARM).rate

    @property
    def candidate_truncation_rate(self) -> float | None:
        """Fraction of reported candidate generations that hit the output cap."""
        return _truncation_counts(self.per_repeat, CANDIDATE_ARM).rate

    @property
    def stock_truncation_regime(self) -> TruncationRegime:
        """What the cap did to the stock arm; see :class:`TruncationRegime`."""
        return _truncation_counts(self.per_repeat, STOCK_ARM).regime

    @property
    def candidate_truncation_regime(self) -> TruncationRegime:
        """What the cap did to the candidate arm."""
        return _truncation_counts(self.per_repeat, CANDIDATE_ARM).regime

    @property
    def truncation_rate_delta(self) -> float | None:
        """Candidate truncation rate minus stock's, or ``None`` if unmeasured.

        Recorded and **not** gated, deliberately.  Under exact-argmax
        verification both arms emit the target's own tokens, so the honest
        null is that this is zero and any departure is the two arms having
        generated materially different text -- the same family of fault
        ``OUTPUT_MISMATCH`` catches, but visible on the throughput pass, which
        captures no logprobs and runs no divergence comparison.

        It does not gate because the throughput pass has no control arm.  The
        divergence criterion earned the right to reject by measuring the
        engine's own stock-against-stock noise floor first (see
        :data:`DIVERGENCE_ALPHA`); there is no such floor for this quantity in
        any archived run, and a two-proportion test against a zero null at
        these sample sizes would fire on the batching-dependent nondeterminism
        the control pass exists to absorb.  Publishing the number is what makes
        that floor measurable on the next runs; inventing a threshold for it
        now would be exactly the uncalibrated constant this gate keeps getting
        burned by.
        """
        stock = self.stock_truncation_rate
        candidate = self.candidate_truncation_rate
        if stock is None or candidate is None:
            return None
        return candidate - stock

    @property
    def output_early_divergences(self) -> int:
        """Divergences early enough to count against ``max_output_mismatches``."""
        return sum(1 for d in self.output_divergences if d.early)

    @property
    def output_total_divergences(self) -> int:
        """Contexts that parted at any offset, early or not."""
        return len(self.output_divergences)

    @property
    def control_early_divergences(self) -> int:
        """Noise-floor divergences before ``min_divergence_token_index``."""
        return sum(1 for d in self.control_divergences if d.early)

    @property
    def control_total_divergences(self) -> int:
        """Noise-floor divergences at any offset."""
        return len(self.control_divergences)

    @property
    def divergence_rate(self) -> float | None:
        """Candidate-versus-stock divergences per context comparison."""
        if self.divergence_trials <= 0:
            return None
        return self.output_total_divergences / self.divergence_trials

    @property
    def control_divergence_rate(self) -> float | None:
        """The engine's own divergence rate, measured stock against stock.

        ``None`` when no control ran.  Read it beside :attr:`divergence_rate`:
        two numbers of similar size are the same engine measured twice, which is
        what the losslessness of speculative decoding predicts and what the
        verdict must not punish.
        """
        if not self.divergence_control_available or self.control_trials <= 0:
            return None
        return self.control_total_divergences / self.control_trials

    @property
    def divergence_criterion(self) -> DivergenceCriterion:
        """Whether the recorded position criterion could discriminate.

        Derived from the two fields already on the record -- the threshold the
        run was judged under and the cap the correctness pass ran with -- so it
        is a reading of this decision, not of the config that is live now.  It
        is what stops ``OUTPUT_MISMATCH`` on a saturated threshold from being
        read as evidence about the drafter; see
        :class:`speedlm.config.DivergenceCriterion`.
        """
        return classify_divergence_criterion(
            self.min_divergence_token_index, self.correctness_max_tokens
        )

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
                    "stock_accepted_length": r.stock_accepted_length,
                    "candidate_accepted_length": r.candidate_accepted_length,
                    "stock_finish_reasons": r.stock_finish_reasons,
                    "candidate_finish_reasons": r.candidate_finish_reasons,
                    "stock_truncated": r.stock_truncated,
                    "candidate_truncated": r.candidate_truncated,
                }
                for r in self.per_repeat
            ],
            # Derived from the ``per_repeat`` counts above, emitted so a reader
            # of ``decision.json`` does not have to sum the array to learn
            # whether the cap or the workload chose the generation lengths.
            "stock_truncation_rate": self.stock_truncation_rate,
            "candidate_truncation_rate": self.candidate_truncation_rate,
            "stock_truncation_regime": self.stock_truncation_regime.value,
            "candidate_truncation_regime": self.candidate_truncation_regime.value,
            "truncation_rate_delta": self.truncation_rate_delta,
            # Which acceptance-side quantity decided, and the criterion's own
            # delta/threshold pair.  Emitted next to -- never instead of -- the
            # rate figures below, which every archived run is described by.
            "acceptance_criterion": self.acceptance_criterion,
            "accepted_length_delta": self.accepted_length_delta,
            "min_accepted_length_delta": self.min_accepted_length_delta,
            "stock_avg_accepted_length": self.stock_avg_accepted_length,
            "candidate_avg_accepted_length": self.candidate_avg_accepted_length,
            "stock_accepted_length_stdev": self.stock_accepted_length_stdev,
            "candidate_accepted_length_stdev": self.candidate_accepted_length_stdev,
            "accepted_length_dispersion": self.accepted_length_dispersion.value,
            "accepted_length_delta_standard_error": (
                self.accepted_length_delta_standard_error
            ),
            "acceptance_statistic": self.acceptance_statistic,
            "stock_avg_acceptance": self.stock_avg_acceptance,
            "candidate_avg_acceptance": self.candidate_avg_acceptance,
            "stock_acceptance_stdev": self.stock_acceptance_stdev,
            "candidate_acceptance_stdev": self.candidate_acceptance_stdev,
            # Why the two stdevs above are worth what they are.  A zero stdev
            # is published as ``degenerate`` with a ``null`` standard error,
            # never as a very good measurement; see :class:`DispersionBasis`.
            "acceptance_dispersion": self.acceptance_dispersion.value,
            "acceptance_delta_standard_error_pp": (
                self.acceptance_delta_standard_error_pp
            ),
            "min_divergence_token_index": self.min_divergence_token_index,
            # Whether that threshold had a range to discriminate in, given the
            # correctness cap below.  ``saturated`` means every divergence was
            # early by construction and the verdict is a property of the
            # configuration, not of the drafter.
            "divergence_criterion": self.divergence_criterion.value,
            "output_early_divergences": self.output_early_divergences,
            "output_total_divergences": self.output_total_divergences,
            "output_divergences": [d.to_dict() for d in self.output_divergences],
            # The noise floor the two counts above were judged against, and the
            # test that judged them.  Without these a reader cannot tell a head
            # that broke from an engine that is not bitwise reproducible; see
            # ``DIVERGENCE_ALPHA``.
            "divergence_control_available": self.divergence_control_available,
            "divergence_control_comparable": self.divergence_control_comparable,
            "divergence_position_p_value": self.divergence_position_p_value,
            "divergence_trials": self.divergence_trials,
            "control_trials": self.control_trials,
            "control_early_divergences": self.control_early_divergences,
            "control_total_divergences": self.control_total_divergences,
            "control_divergences": [d.to_dict() for d in self.control_divergences],
            "divergence_rate": self.divergence_rate,
            "control_divergence_rate": self.control_divergence_rate,
            "divergence_total_p_value": self.divergence_total_p_value,
            "divergence_early_p_value": self.divergence_early_p_value,
            "divergence_alpha": self.divergence_alpha,
            "stock_avg_tok_per_sec": self.stock_avg_tok_per_sec,
            "candidate_avg_tok_per_sec": self.candidate_avg_tok_per_sec,
            "throughput_dispersion": self.throughput_dispersion.value,
            "throughput_delta_standard_error_pct": (
                self.throughput_delta_standard_error_pct
            ),
            # Drift, not noise: the number that says whether the standard error
            # above rests on exchangeable repeats.  See
            # ``stock_throughput_trend_pct_per_repeat``.
            "stock_throughput_trend_pct_per_repeat": (
                self.stock_throughput_trend_pct_per_repeat
            ),
            "candidate_throughput_trend_pct_per_repeat": (
                self.candidate_throughput_trend_pct_per_repeat
            ),
            # Where the drift above stops, or ``null`` if it had not stopped by
            # the last scored repeat.  See
            # ``candidate_throughput_flat_from_repeat``.
            "stock_throughput_flat_from_repeat": (
                self.stock_throughput_flat_from_repeat
            ),
            "candidate_throughput_flat_from_repeat": (
                self.candidate_throughput_flat_from_repeat
            ),
            "stock_prometheus_decode_tok_per_sec": (
                self.stock_prometheus_decode_tok_per_sec
            ),
            "candidate_prometheus_decode_tok_per_sec": (
                self.candidate_prometheus_decode_tok_per_sec
            ),
            "benchmark_max_tokens": self.benchmark_max_tokens,
            "replay_concurrency": self.replay_concurrency,
            "correctness_max_tokens": self.correctness_max_tokens,
            "correctness_repeats": self.correctness_repeats,
            "suite_hash": self.suite_hash,
            "num_contexts": self.num_contexts,
            "stock_draft": self.stock_draft,
            "engine_execution_mode": self.engine_execution_mode,
            "engine_enforce_eager": self.engine_enforce_eager,
            "engine_enable_chunked_prefill": self.engine_enable_chunked_prefill,
            "engine_enable_prefix_caching": self.engine_enable_prefix_caching,
            "engine_max_num_seqs": self.engine_max_num_seqs,
            # How the measurement was taken: the interleaving, whether each
            # block paid for its own engine, and whether the delta held still.
            "arm_blocks": self.arm_blocks,
            "block_schedule": [block.to_dict() for block in self.block_schedule],
            "throughput_stationarity": (
                None
                if self.throughput_stationarity is None
                else self.throughput_stationarity.to_dict()
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


def _delta_standard_error_raw(stock: list[float], candidate: list[float]) -> float | None:
    """SE of the arm-to-arm delta of means, in the inputs' own units.

    ``None`` when there are fewer than two paired repeats.  The two arms are
    timed and scraped independently, so their variances add rather than
    cancel; this is the unpaired form, matching how ``*_avg_*`` and the gated
    deltas are computed.
    """
    n = min(len(stock), len(candidate))
    if n < 2:
        return None
    var = _stdev(stock[:n]) ** 2 / n + _stdev(candidate[:n]) ** 2 / n
    return math.sqrt(var)


def _dispersion_basis(stock: list[float], candidate: list[float]) -> DispersionBasis:
    """Classify what a published standard error is worth.  See :class:`DispersionBasis`."""
    se = _delta_standard_error_raw(stock, candidate)
    if se is None:
        return DispersionBasis.UNSAMPLED
    if se == 0.0:
        return DispersionBasis.DEGENERATE
    return DispersionBasis.MEASURED


def _delta_standard_error(
    stock: list[float],
    candidate: list[float],
    *,
    scale: float,
) -> float | None:
    """:func:`_delta_standard_error_raw` in reporting units, ``None`` unless measured."""
    se = _delta_standard_error_raw(stock, candidate)
    if se is None or se == 0.0:
        return None
    return se * scale


def _trend_pct_per_repeat(values: list[float]) -> float | None:
    """OLS slope of a per-repeat column, as a percentage of its own mean.

    ``None`` below two repeats, or when the mean is non-positive and a
    percentage would be meaningless.
    """
    n = len(values)
    if n < 2:
        return None
    mean = _mean(values)
    if mean <= 0:
        return None
    mean_x = (n - 1) / 2.0
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    numerator = sum((i - mean_x) * (v - mean) for i, v in enumerate(values))
    return numerator / denominator / mean * 100.0


#: Repeats a trailing window needs before it may be called flat.
#:
#: Three is the floor at which an OLS slope has a residual degree of freedom
#: (``m - 2``) and therefore a standard error at all.  Two points fit their own
#: slope exactly, leaving zero residual, so on a two-point rule *every* window
#: would read flat and the answer would be "repeat n-2" on every run regardless
#: of the data.
MIN_FLAT_WINDOW: Final = 3

#: ``|slope| / SE(slope)`` below which a trailing window has stopped trending.
#:
#: One standard error -- a deliberately lenient bar, chosen against how large
#: the drift being detected actually is.  The full five-repeat candidate
#: windows on jobs 369161/369162 explain 83% and 94% of their own variance,
#: which for ``n=5`` is ``|t| = sqrt(R^2/(1-R^2) * (n-2))`` = 3.8 and 6.9.  The
#: warming signal this has to *not* miss therefore sits at 4-7 standard errors,
#: so a 1.0 bar separates it by a wide margin while still being reachable by a
#: genuinely settled window.
#:
#: Erring lenient is the right direction here because nothing downstream gates
#: on this: it is published so that raising ``warmup_repeats`` can be argued
#: from a measured index instead of a guess, and a bar so strict that no real
#: column ever clears it would answer "never flat" forever and teach nothing.
FLAT_TREND_T_STATISTIC: Final = 1.0


def _slope_t_statistic(values: list[float]) -> float | None:
    """``|slope| / SE(slope)`` of an OLS fit of *values* against repeat index.

    ``None`` below :data:`MIN_FLAT_WINDOW` points, where the residual variance
    has no degrees of freedom to be estimated from.
    """
    n = len(values)
    if n < MIN_FLAT_WINDOW:
        return None
    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    sxx = sum((i - mean_x) ** 2 for i in range(n))
    sxy = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ssr = sum((v - (intercept + slope * i)) ** 2 for i, v in enumerate(values))
    if ssr <= 0.0:
        # Exactly collinear: no residual to build a standard error from, but
        # the verdict is not ambiguous either way.  A constant column is flat;
        # a perfect ramp is not.  Both are stub signatures rather than live
        # measurements -- see :class:`DispersionBasis` -- and reporting each as
        # what it plainly is beats reporting both as unknown.
        return 0.0 if slope == 0.0 else math.inf
    return abs(slope) / math.sqrt(ssr / (n - 2) / sxx)


def _flat_from_repeat(values: list[float]) -> int | None:
    """Earliest repeat index from which the trailing window stopped trending.

    Answers "how many passes does this arm need before it is warm" by asking,
    for each suffix of the per-repeat column, whether that suffix still has a
    slope its own residual noise cannot explain -- the trend of the trailing
    window falling below its own noise.  The earliest suffix that clears
    :data:`FLAT_TREND_T_STATISTIC` is returned; ``None`` means no suffix of at
    least :data:`MIN_FLAT_WINDOW` repeats did, i.e. the arm was still warming
    when the last scored repeat ended.

    Suffixes are scanned earliest-first on purpose.  The quantity wanted is the
    *first* index that is safe to measure from, and later suffixes are shorter
    and therefore easier to call flat by accident; taking the earliest one that
    passes keeps the answer from drifting toward ``n - MIN_FLAT_WINDOW`` as the
    window shrinks.
    """
    n = len(values)
    for start in range(n - MIN_FLAT_WINDOW + 1):
        statistic = _slope_t_statistic(values[start:])
        if statistic is not None and statistic < FLAT_TREND_T_STATISTIC:
            return start
    return None




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


def _repeat_accepted_length(
    repeat_metrics: Sequence[MetricsDelta],
    index: int,
    pooled: float,
) -> float:
    """This repeat's own mean accepted length, or the pooled figure.

    Mirrors :func:`_repeat_acceptance` exactly, including its fallback, so the
    two per-repeat columns are always populated on the same basis and
    ``*_avg_accepted_length`` stays the literal mean of its column.  It keys off
    :attr:`~speedlm.gate.metrics.MetricsDelta.accepted_length_available` rather
    than ``acceptance_available`` because the two are divided by *different*
    counters and can disagree.
    """
    if index < len(repeat_metrics):
        delta = repeat_metrics[index]
        if delta.accepted_length_available:
            return delta.mean_accepted_length
    return pooled


def _build_per_repeat(
    stock_replay: ReplayResult,
    candidate_replay: ReplayResult,
    *,
    num_repeats: int,
    stock_repeat_metrics: Sequence[MetricsDelta],
    candidate_repeat_metrics: Sequence[MetricsDelta],
    pooled_stock: MetricsDelta,
    pooled_candidate: MetricsDelta,
    early_by_repeat: dict[int, int],
) -> tuple[RepeatSummary, ...]:
    """Summarise the paired repeats both arms actually completed.

    Derived purely from replay data plus the per-repeat metric windows, so
    these survive a metrics failure and keep ``num_repeats == len(per_repeat)``
    true on every path.
    """

    return tuple(
        RepeatSummary(
            repeat_index=i,
            stock_tok_per_sec=stock_replay.run_results[i].output_tok_per_sec,
            candidate_tok_per_sec=candidate_replay.run_results[i].output_tok_per_sec,
            stock_acceptance_rate=_repeat_acceptance(
                stock_repeat_metrics, i, pooled_stock.acceptance_rate
            ),
            candidate_acceptance_rate=_repeat_acceptance(
                candidate_repeat_metrics, i, pooled_candidate.acceptance_rate
            ),
            invalid_rate=candidate_replay.run_results[i].invalid_rate,
            output_mismatches=early_by_repeat.get(i, 0),
            stock_accepted_length=_repeat_accepted_length(
                stock_repeat_metrics, i, pooled_stock.mean_accepted_length
            ),
            candidate_accepted_length=_repeat_accepted_length(
                candidate_repeat_metrics, i, pooled_candidate.mean_accepted_length
            ),
            stock_finish_reasons=stock_replay.run_results[i].finish_reason_count,
            candidate_finish_reasons=candidate_replay.run_results[i].finish_reason_count,
            stock_truncated=stock_replay.run_results[i].truncated_count,
            candidate_truncated=candidate_replay.run_results[i].truncated_count,
        )
        for i in range(num_repeats)
    )


def _column_stats(values: list[float], pooled: float) -> tuple[float, float]:
    """``(mean, sample sd)`` of one per-repeat column.

    The published ``*_avg_*`` figure is *exactly* the mean of the ``per_repeat``
    column beside it, and every gated delta is exactly the delta of two such
    means -- which is what lets a reader reconcile the verdict by hand from the
    array.  The sample standard deviation is returned beside the mean and
    published with it, so a reader can see how much that mean is worth without
    re-deriving it from the array.

    *pooled* stands in only when there is no column at all (a zero-sample
    benchmark), where the whole-window scrape is the only figure there is; the
    standard deviation is then honestly ``0.0``.
    """
    return (_mean(values) if values else pooled), _stdev(values)


def _engine_fields(engine_execution: EngineExecution | None) -> dict[str, Any]:
    """The engine-execution columns of a decision record.

    ``None`` throughout when the gate was told nothing, except the mode itself,
    which says :data:`UNRECORDED_EXECUTION_MODE` out loud: the absence of
    knowledge is the fact worth persisting, and a ``None`` there would be
    indistinguishable from a key a reader forgot to add.
    """
    if engine_execution is None:
        return {
            "engine_execution_mode": UNRECORDED_EXECUTION_MODE,
            "engine_enforce_eager": None,
            "engine_enable_chunked_prefill": None,
            "engine_enable_prefix_caching": None,
            "engine_max_num_seqs": None,
        }
    return {
        "engine_execution_mode": engine_execution.execution_mode,
        "engine_enforce_eager": engine_execution.enforce_eager,
        "engine_enable_chunked_prefill": engine_execution.enable_chunked_prefill,
        "engine_enable_prefix_caching": engine_execution.enable_prefix_caching,
        "engine_max_num_seqs": engine_execution.max_num_seqs,
    }


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
    stock_correctness_control: ReplayResult | None = None,
    benchmark_max_tokens: int | None = None,
    replay_concurrency: int | None = None,
    correctness_max_tokens: int | None = None,
    num_contexts: int | None = None,
    stock_draft: str | None = None,
    engine_execution: EngineExecution | None = None,
) -> Decision:
    """Decide whether to promote the candidate head.

    Both thresholds are evaluated against a single, named statistic per
    quantity: the acceptance side from the Prometheus ``spec_decode`` counter
    delta (which has no timing component), and throughput from
    :data:`GATING_THROUGHPUT_STATISTIC`.

    The acceptance-side criterion is the **mean-accepted-length** delta, in
    tokens per verifier step -- see :data:`GATING_ACCEPTANCE_CRITERION` for why
    the acceptance *rate* cannot serve, and
    :attr:`speedlm.config.PromotionConfig.min_accepted_length_delta` for the
    bar.  ``acceptance_delta_pp`` and ``min_acceptance_delta_pp`` are still
    computed and recorded unchanged; they no longer decide.

    The Prometheus decode-time throughput
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
        max_output_mismatches: Divergences tolerated outright, before the
            criterion is even consulted.  Default 0, which leaves the
            significance test in :data:`DIVERGENCE_ALPHA` fully in charge: a
            divergence count has to *both* exceed this allowance and exceed the
            run's own measured noise floor significantly.  Raising it can only
            make the gate more permissive, never less.
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
        stock_correctness_control: A *second* correctness pass by the stock arm,
            on the same engine and settings as ``stock_correctness``.  Comparing
            the two measures how often this engine disagrees with itself, which
            is the only thing the candidate arm's divergence count can honestly
            be judged against -- see :data:`DIVERGENCE_ALPHA`.  ``None`` falls
            back to an assumed floor of zero over an equal number of trials and
            records ``divergence_control_available: false``.
        benchmark_max_tokens: Output cap the throughput/acceptance pass ran
            under, recorded verbatim in the decision.  These last five are
            *measurement context*: they change what the reported statistics are
            worth without changing their names, and a decision that omits them
            cannot be compared to another run.  Omitting one records ``None``,
            which is what an archived decision written before the field existed
            reads back as.
        replay_concurrency: In-flight requests per arm during that pass.
        correctness_max_tokens: Output cap of the correctness pass.
        num_contexts: Contexts in the frozen suite both arms replayed.  The
            suite's hash is taken from ``stock_replay``.
        stock_draft: The draft the stock arm ran -- the baseline this delta is
            measured against.
        engine_execution: How the engine both arms replayed against was
            executing -- see :class:`EngineExecution`.  ``None`` records
            :data:`UNRECORDED_EXECUTION_MODE`, which is what an archived
            decision written before this field existed reads back as.

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
    s_mal = stock_metrics.mean_accepted_length
    c_mal = candidate_metrics.mean_accepted_length

    # How many suite passes each arm actually completed.  This is a fact about
    # the benchmark run, so it is established before any validation short
    # circuits: a report claiming zero repeats must mean the replay really did
    # not run, not merely that the decision stopped looking.
    s_runs = stock_replay.num_runs
    c_runs = candidate_replay.num_runs
    min_runs = min(s_runs, c_runs)

    # --- Output correctness ---
    # The whole criterion -- the measured pair, the engine's own noise floor,
    # whether that floor is a valid null, and the three channels that can reject
    # on it -- lives in :func:`speedlm.gate.divergence.evaluate_divergence`.
    min_divergence_index = promotion_config.min_divergence_token_index
    evidence = evaluate_divergence(
        stock_replay,
        candidate_replay,
        stock_correctness=stock_correctness,
        candidate_correctness=candidate_correctness,
        stock_correctness_control=stock_correctness_control,
        min_divergence_index=min_divergence_index,
        correctness_max_tokens=correctness_max_tokens,
        max_output_mismatches=max_output_mismatches,
    )

    # --- Build per-repeat summaries ---
    per_repeat_tuple = _build_per_repeat(
        stock_replay,
        candidate_replay,
        num_repeats=min_runs,
        stock_repeat_metrics=stock_repeat_metrics,
        candidate_repeat_metrics=candidate_repeat_metrics,
        pooled_stock=stock_metrics,
        pooled_candidate=candidate_metrics,
        early_by_repeat=evidence.early_by_repeat,
    )

    # --- The three gated statistics, each the mean of its own column ---
    # Both arms replayed the same suite the same number of times, so pairing the
    # repeats and averaging the column keeps every reported number reconcilable
    # by hand: ``stock_avg_tok_per_sec`` is *exactly* the mean of the
    # ``per_repeat`` stock column, and each delta below is exactly the delta of
    # two such means.  The Prometheus figures are carried alongside, clearly
    # named, and ignored -- see :data:`GATING_THROUGHPUT_STATISTIC`.
    #
    # The acceptance *criterion* -- mean accepted length, in tokens per verifier
    # step -- is the one that decides.  Unlike the rate beside it, it carries no
    # draft depth in its denominator, so it is comparable across a k-sweep.  See
    # :data:`GATING_ACCEPTANCE_CRITERION`.
    s_tps, _ = _column_stats([r.stock_tok_per_sec for r in per_repeat_tuple], 0.0)
    c_tps, _ = _column_stats([r.candidate_tok_per_sec for r in per_repeat_tuple], 0.0)
    s_acc_mean, s_acc_sd = _column_stats(
        [r.stock_acceptance_rate for r in per_repeat_tuple], s_acc
    )
    c_acc_mean, c_acc_sd = _column_stats(
        [r.candidate_acceptance_rate for r in per_repeat_tuple], c_acc
    )
    s_mal_mean, s_mal_sd = _column_stats(
        [r.stock_accepted_length for r in per_repeat_tuple], s_mal
    )
    c_mal_mean, c_mal_sd = _column_stats(
        [r.candidate_accepted_length for r in per_repeat_tuple], c_mal
    )

    def _decide(
        verdict: Verdict,
        reason: Reason,
        *,
        acceptance_delta_pp: float | None = None,
        accepted_length_delta: float | None = None,
        throughput_delta_pct: float | None = None,
    ) -> Decision:
        return Decision(
            verdict=verdict,
            reason=reason,
            acceptance_delta_pp=acceptance_delta_pp,
            accepted_length_delta=accepted_length_delta,
            min_accepted_length_delta=promotion_config.min_accepted_length_delta,
            stock_avg_accepted_length=s_mal_mean,
            candidate_avg_accepted_length=c_mal_mean,
            stock_accepted_length_stdev=s_mal_sd,
            candidate_accepted_length_stdev=c_mal_sd,
            acceptance_criterion=GATING_ACCEPTANCE_CRITERION,
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
            output_divergences=evidence.divergences,
            divergence_trials=evidence.trials,
            control_trials=evidence.control_trials,
            control_divergences=evidence.control_divergences,
            divergence_control_available=evidence.control_available,
            divergence_control_comparable=evidence.control_comparable,
            divergence_position_p_value=(
                evidence.position_p if evidence.total_early else None
            ),
            divergence_total_p_value=evidence.total_p,
            divergence_early_p_value=evidence.early_p,
            divergence_alpha=DIVERGENCE_PER_STATISTIC_ALPHA,
            benchmark_max_tokens=benchmark_max_tokens,
            replay_concurrency=replay_concurrency,
            correctness_max_tokens=correctness_max_tokens,
            correctness_repeats=evidence.correctness_repeats,
            suite_hash=stock_replay.suite_hash or None,
            num_contexts=num_contexts,
            stock_draft=stock_draft,
            **_engine_fields(engine_execution),
        )

    def _reject(reason: Reason, **deltas: float | None) -> Decision:
        return _decide(Verdict.REJECT, reason, **deltas)

    # --- Validation: counter reset ---
    if stock_metrics.reset_detected or candidate_metrics.reset_detected:
        return _reject(Reason.COUNTER_RESET)

    # --- Validation: acceptance unavailable ---
    # Both quantities are checked, because they are divided by *different*
    # counters: the rate by ``spec_decode_num_draft_tokens`` and the criterion
    # by ``spec_decode_num_drafts``.  An endpoint exposing only the first would
    # otherwise hand the gate ``0.0 - 0.0 = 0.0`` tokens/step and reject every
    # candidate under a threshold reason, hiding a missing counter behind a
    # verdict about the head.
    if not all(
        (
            stock_metrics.acceptance_available,
            candidate_metrics.acceptance_available,
            stock_metrics.accepted_length_available,
            candidate_metrics.accepted_length_available,
        )
    ):
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

    # --- Validation: the output cap chose every generation length ---
    # ``invalid_rate`` does not see this and cannot be made to: a response that
    # spent the whole budget generating is a *healthy* response, so every one
    # of them is valid and the rate reads 0.0 no matter how completely the
    # harness was the thing that ended them.  That is the shape a realistic
    # agentic workload replayed under a chat-sized ``benchmark_max_tokens``
    # produces, and it passed.
    #
    # Zero natural stops is the bar, so this rejects only runs that observed
    # nothing at all about where the model stops -- it does not disturb the
    # heavily-but-not-wholly truncated regime every archived live run sits in.
    # See :class:`TruncationRegime`.
    # Classified from ``per_repeat_tuple`` -- the paired window -- and NOT from
    # the two ``ReplayResult`` objects, which pool every run each arm made.
    # The two differ whenever the arms returned unequal run counts: the record
    # keeps only ``min_runs`` rows, so a gate reading the unpaired pool could
    # promote on ``MIXED`` while the decision it wrote derived ``SATURATED``
    # from the rows it actually persisted.  A verdict whose own evidence
    # contradicts it is this project's recurring defect; deriving both from one
    # array makes the divergence unrepresentable rather than merely unlikely --
    # which is what ``_truncation_counts``, the one reader both sides go
    # through, enforces.
    regimes = (
        _truncation_counts(per_repeat_tuple, STOCK_ARM).regime,
        _truncation_counts(per_repeat_tuple, CANDIDATE_ARM).regime,
    )
    if TruncationRegime.SATURATED in regimes:
        return _reject(Reason.TRUNCATION_SATURATED)

    # --- Validation: nothing reported a finish reason at all ---
    # ``UNTESTABLE`` is deliberately non-gating *as a description of a record*:
    # every archived ``decision.json`` predates the counts, so its properties
    # must keep reading ``UNTESTABLE`` rather than being relabelled ``BOUNDED``,
    # and :func:`speedlm.report.parse_decision` must keep loading them.  That is
    # a schema gap, and it is not this branch's subject.
    #
    # Here the counts come from a replay that just ran.  ``invalid_rate`` has
    # already been checked above, so this arm returned responses the gate
    # accepted as valid -- and not one of them said what ended it.  That is a
    # measurement gap, not a schema gap: the saturation guard directly above is
    # structurally unable to fire, and without this branch such a run promotes
    # with its strongest measurement check silently inert.  The precedent is
    # ``ACCEPTANCE_UNAVAILABLE`` a few checks up: in this codebase a missing
    # instrument rejects, it does not pass.
    if TruncationRegime.UNTESTABLE in regimes:
        return _reject(Reason.TRUNCATION_UNMEASURED)

    # --- Validation: output mismatch ---
    # Excess over the engine's own noise floor, not any single occurrence.  Both
    # arms run speculative decoding against the same target, so both emit the
    # target's distribution and a disagreement between them is the engine
    # disagreeing with itself -- which the control pass measures directly.  See
    # :data:`DIVERGENCE_ALPHA` for the argument and the false-reject arithmetic
    # the old any-occurrence rule was producing.
    if evidence.rejects:
        return _reject(Reason.OUTPUT_MISMATCH)

    # --- Validation: throughput unavailable ---
    # The denominator that matters is the gating statistic's.  A Prometheus
    # window that reads zero is a diagnostic gap, not grounds to reject.
    if s_tps <= 0:
        return _reject(Reason.THROUGHPUT_UNAVAILABLE)

    # --- Compute deltas ---
    # Recorded, not gated.  Still the delta of the two ``*_avg_acceptance``
    # means, still in percentage points, still what every archived run is
    # described by -- see ``Decision.acceptance_delta_pp``.
    acceptance_delta_pp = (c_acc_mean - s_acc_mean) * 100.0

    # The promotion criterion: tokens per verifier step, k-invariant.
    accepted_length_delta = c_mal_mean - s_mal_mean

    throughput_delta_pct = (c_tps - s_tps) / s_tps * 100.0

    deltas: dict[str, float | None] = {
        "acceptance_delta_pp": acceptance_delta_pp,
        "accepted_length_delta": accepted_length_delta,
        "throughput_delta_pct": throughput_delta_pct,
    }

    # --- Threshold: acceptance ---
    # ``Reason.ACCEPTANCE_BELOW_THRESHOLD`` is deliberately reused rather than
    # split: the reason string is a public contract (``speedlm.report``'s
    # explanations, the e2e harness's ``DELTA_REASONS``, archived records), and
    # what it says -- the acceptance side of the gate missed its bar -- is still
    # exactly true.  Which bar is named by ``acceptance_criterion``.
    if accepted_length_delta < promotion_config.min_accepted_length_delta:
        return _reject(Reason.ACCEPTANCE_BELOW_THRESHOLD, **deltas)

    # --- Threshold: throughput ---
    if throughput_delta_pct < promotion_config.min_throughput_delta_pct:
        return _reject(Reason.THROUGHPUT_BELOW_THRESHOLD, **deltas)

    # --- Both thresholds met: PROMOTE ---
    return _decide(Verdict.PROMOTE, Reason.BOTH_THRESHOLDS_MET, **deltas)
