"""Shared data contract for the benchmarking harness.

Every other module in this package speaks these types and only these types.
They are defined in one file, separately from the code that produces or
consumes them, because the index (:mod:`resultsdb`), the comparator
(:mod:`compare`) and the regression tracker were built against this contract
concurrently.  Changing a field here changes all three.

Design notes that are load-bearing rather than incidental:

* :class:`MetricSeries` carries *the individual observations*, never a
  pre-reduced mean.  A mean cannot be given a standard error after the fact,
  and this project has repeatedly reached conclusions smaller than one SE.  An
  extractor that can only recover a mean must say so by setting
  ``reduced=True``, and the comparator then refuses to publish a resolved
  verdict for it instead of pretending the dispersion is zero.
* :class:`MetricSeries` also carries ``pass_index`` alongside each value.  Two
  arms are only comparable when their observations came from the same position
  in the run -- the ABBA block schedule exists precisely so that they do -- and
  pairing by list position silently mis-pairs when one arm has more repeats.
* Everything is frozen.  An index that can be mutated after it is built is an
  index whose provenance is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Direction(Enum):
    """Which way is better for a metric.

    Required, not inferred.  "Latency went up 4%" and "throughput went up 4%"
    are opposite findings, and a comparator that guesses from the metric's name
    will eventually guess wrong on a metric nobody thought about.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    #: Neither -- a descriptive quantity (token counts, request counts).  These
    #: are diffed and reported but never produce a regression verdict.
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """The observations of one metric, for one arm, in one cell of one run."""

    name: str
    unit: str
    direction: Direction
    #: One entry per observation, in run order.
    values: tuple[float, ...]
    #: Pass index of each observation, parallel to :attr:`values`.  ``None``
    #: when the artifact did not record one; the comparator then falls back to
    #: positional pairing *and says so* in its output.
    pass_indices: tuple[int, ...] | None = None
    #: True when the artifact only preserved a summary statistic, so
    #: :attr:`values` is a single already-reduced number.  A reduced series can
    #: never carry a standard error.
    reduced: bool = False

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"metric {self.name!r} has no observations")
        if self.pass_indices is not None and len(self.pass_indices) != len(self.values):
            raise ValueError(
                f"metric {self.name!r}: {len(self.pass_indices)} pass indices "
                f"for {len(self.values)} values"
            )
        if self.reduced and len(self.values) != 1:
            raise ValueError(
                f"metric {self.name!r} is marked reduced but carries "
                f"{len(self.values)} values; a reduced series is one number"
            )


@dataclass(frozen=True, slots=True)
class CellRecord:
    """One measured configuration within a run (a matrix cell, or the run itself)."""

    name: str
    #: ``arm -> metric name -> series``.  Arm names are the run's own
    #: ("stock"/"candidate", "off"/"on", "direct"/"gateway").  A single-arm cell
    #: uses the arm name ``"single"``.
    arms: dict[str, dict[str, MetricSeries]]
    #: Free-form configuration recovered from the artifact: execution mode,
    #: concurrency, context band, model, speculative depth.  Used to refuse
    #: comparisons between cells that were not configured alike.
    config: dict[str, Any] = field(default_factory=dict)
    #: Non-fatal problems found while extracting.  Present in the index rather
    #: than logged, so a query can filter on them.
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One benchmark run, as recovered from its on-disk artifacts."""

    run_id: str
    run_dir: Path
    flavor: str
    #: Full commit sha the run was built from, when recoverable from the
    #: snapshot provenance.  ``None`` when the run predates provenance
    #: recording -- such runs are indexed but excluded from commit-ordered
    #: regression tracking rather than silently attributed to HEAD.
    commit: str | None
    created_at: str | None
    slurm_job_id: str | None
    #: Name of the workload the run measured, when declared.  ``None`` for
    #: historical runs, which all predate the workload system.
    workload: str | None
    cells: dict[str, CellRecord]
    #: Artifacts that were found but could not be parsed, path -> reason.  An
    #: index that silently drops what it cannot read reports a clean history it
    #: has not got.
    unparsed: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether the run yielded no measurements at all."""
        return not self.cells


class ComparisonVerdict(Enum):
    """What a paired comparison of one metric across two runs established.

    Deliberately mirrors :class:`speedlm.gate.decide.StationarityStatus`'s
    discipline: the interesting distinction is not the sign of the mean but
    whether the sample resolved anything.  A bare mean is not a finding.
    """

    #: Delta is smaller than the materiality threshold, and the sample was
    #: precise enough to have detected a material one.
    PROVEN_FLAT = "proven_flat"
    #: Delta exceeds materiality but is under the significance bar -- the run
    #: saw a shift and could not tell it from noise.  This is the honest
    #: outcome most short runs deserve and the one a bare mean hides.
    MATERIAL_SHIFT_UNRESOLVED = "material_shift_unresolved"
    PROVEN_REGRESSION = "proven_regression"
    PROVEN_IMPROVEMENT = "proven_improvement"
    #: No standard error exists: fewer than two observations per arm, or every
    #: observation returned an identical value so there is no variance estimate.
    #: Not "tight" -- unmeasured.
    UNRESOLVED = "unresolved"
    #: The metric is :attr:`Direction.NEUTRAL`, so it is reported without a
    #: better/worse judgement.
    DESCRIPTIVE = "descriptive"


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """The diff of one metric between a baseline and a candidate run."""

    cell: str
    arm: str
    metric: str
    unit: str
    direction: Direction
    baseline_mean: float
    candidate_mean: float
    delta: float
    delta_pct: float | None
    #: Standard error of :attr:`delta`, in the metric's own units.  ``None``
    #: when the sample cannot support one -- never ``0.0``, so that a consumer
    #: dividing by it fails loudly instead of computing infinite headroom.
    delta_standard_error: float | None
    #: ``speedlm.gate.decide.DispersionBasis`` value: measured / degenerate /
    #: unsampled.
    dispersion_basis: str
    #: ``|delta| / SE``, ``None`` when there is no SE.
    t_statistic: float | None
    verdict: ComparisonVerdict
    baseline_n: int
    candidate_n: int
    #: Set when the two arms had to be paired by list position because at least
    #: one side recorded no pass indices.
    positional_pairing: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunComparison:
    """The full diff of two runs."""

    baseline: str
    candidate: str
    baseline_commit: str | None
    candidate_commit: str | None
    comparisons: tuple[MetricComparison, ...]
    #: Cells or metrics present on one side only.  Reported, because a metric
    #: that vanished between two commits is a finding, not a non-event.
    baseline_only: tuple[str, ...] = ()
    candidate_only: tuple[str, ...] = ()
    #: Cells present on both sides whose recorded configuration differed.  These
    #: are excluded from :attr:`comparisons` rather than compared across a
    #: config change.
    incomparable: tuple[str, ...] = ()

    @property
    def regressions(self) -> tuple[MetricComparison, ...]:
        return tuple(
            c for c in self.comparisons if c.verdict is ComparisonVerdict.PROVEN_REGRESSION
        )

    @property
    def unresolved_shifts(self) -> tuple[MetricComparison, ...]:
        return tuple(
            c
            for c in self.comparisons
            if c.verdict is ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED
        )
