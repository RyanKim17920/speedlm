"""Track one metric across a commit-ordered history of runs.

Two properties make this different from sorting a list of runs by date:

* **Commit order, not wall-clock order.**  Runs on this project are relaunched
  constantly -- a job that failed on an old snapshot is resubmitted after newer
  work has already been measured, and its artifacts are then the *newest* on
  disk while describing the *oldest* code.  Ordering that history by
  ``created_at`` produces a series whose "regressions" are an artefact of the
  queue.  Ordering is therefore taken from ``git rev-list --topo-order`` in the
  repository itself, and adjacency of two commits is checked with
  ``git merge-base --is-ancestor`` so that a step across a branch fork is
  labelled rather than silently reported as a regression.

* **Runs without a commit are excluded, not attributed to HEAD.**
  :attr:`~model.RunRecord.commit` is ``None`` for runs that predate provenance
  recording.  Placing them at HEAD would date every one of them to whatever was
  checked out when the index happened to be built, which is a fabricated
  history.  They are dropped and *named* in :attr:`MetricTrack.excluded`.

Verdicts for consecutive pairs come from :mod:`compare`, so a step in this
history is judged by exactly the same rule as an ad-hoc A/B, including the
"could this sample have resolved a material shift?" test.  The trend summary
reuses :func:`speedlm.gate.decide._slope_t_statistic` and
:func:`speedlm.gate.decide._flat_from_repeat`, which already know that a
two-point window fits its own slope exactly and therefore proves nothing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Protocol

from speedlm.gate.decide import (  # noqa: PLC2701 - see compare.py's module docstring
    FLAT_TREND_T_STATISTIC,
    MIN_FLAT_WINDOW,
    _flat_from_repeat,
    _slope_t_statistic,
)

from .compare import (
    DEFAULT_MATERIALITY_PCT,
    DEFAULT_SIGNIFICANCE_T,
    compare_metric,
    config_differences,
)
from .model import ComparisonVerdict, MetricComparison, MetricSeries, RunRecord

#: The repository whose topology defines "before" and "after".
REPO_ROOT: Final = Path(__file__).resolve().parents[3]


class TrendStatus(Enum):
    """Whether a metric's history is going somewhere, mirroring StationarityStatus.

    The distinction that matters is not the sign of the slope but whether the
    history could support one at all.  A three-run history with a huge slope and
    huger residuals is ``UNRESOLVED``, not ``TRENDING``.
    """

    #: Fewer than :data:`speedlm.gate.decide.MIN_FLAT_WINDOW` points, so an OLS
    #: slope has no residual degrees of freedom and no standard error.
    UNTESTABLE = "untestable"
    #: A slope exists and is at least :data:`FLAT_TREND_T_STATISTIC` standard
    #: errors from zero.
    TRENDING = "trending"
    #: A slope exists but its own residual noise explains it.
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One run's measurement of the tracked metric, placed in commit order."""

    run_id: str
    commit: str
    #: Position in ``git rev-list --topo-order --reverse``; larger is newer.
    topological_index: int
    series: MetricSeries
    mean: float


@dataclass(frozen=True, slots=True)
class MetricTrend:
    """OLS trend of the per-run means across the commit-ordered history."""

    status: TrendStatus
    #: ``|slope| / SE(slope)``; ``None`` when the history is too short.
    slope_t_statistic: float | None
    #: Earliest index from which the trailing window stopped trending, or
    #: ``None`` when no trailing window of at least
    #: :data:`speedlm.gate.decide.MIN_FLAT_WINDOW` points has settled.
    flat_from_index: int | None
    n: int


@dataclass(frozen=True, slots=True)
class MetricTrack:
    """The history of one (cell, arm, metric), plus the step-by-step verdicts."""

    cell: str
    arm: str
    metric: str
    points: tuple[MetricPoint, ...]
    #: One entry per consecutive pair of :attr:`points`, oldest step first.
    steps: tuple[MetricComparison, ...]
    trend: MetricTrend
    #: ``run_id: reason`` for every run that was dropped from the history.
    excluded: tuple[tuple[str, str], ...] = ()
    #: Steps whose two commits are not ancestor-related (a branch fork) or whose
    #: cell configuration changed.  Described, not silently compared.
    caveats: tuple[str, ...] = ()

    @property
    def regressions(self) -> tuple[MetricComparison, ...]:
        return tuple(s for s in self.steps if s.verdict is ComparisonVerdict.PROVEN_REGRESSION)

    @property
    def unresolved_shifts(self) -> tuple[MetricComparison, ...]:
        return tuple(
            s for s in self.steps if s.verdict is ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED
        )


class CommitOrdering(Protocol):
    """Where a commit sits in history, and whether one commit precedes another.

    A protocol rather than a concrete class so that the ordering rule can be
    exercised with a known topology in a test that has no repository, and so
    the git calls are not made once per comparison.
    """

    def position(self, commit: str) -> int | None:
        """Topological index, larger for newer; ``None`` if the commit is unknown."""

    def is_ancestor(self, older: str, newer: str) -> bool:
        """Whether *older* is reachable from *newer*."""


class GitCommitOrdering:
    """:class:`CommitOrdering` backed by the real repository.

    ``git rev-list --topo-order --reverse --all`` is read once and cached: it is
    a parent-before-child linearisation of the whole commit graph, which is
    exactly the "before/after" relation wanted here and is not recoverable from
    commit dates (a rebase or an amended commit rewrites the date freely).
    """

    def __init__(self, repo: Path = REPO_ROOT) -> None:
        self.repo = repo
        self._positions: dict[str, int] | None = None
        self._resolved: dict[str, str | None] = {}

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(self.repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def _load(self) -> dict[str, int]:
        if self._positions is None:
            result = self._git("rev-list", "--topo-order", "--reverse", "--all")
            if result.returncode != 0:
                raise RuntimeError(f"git rev-list failed in {self.repo}: {result.stderr.strip()}")
            self._positions = {
                sha: index for index, sha in enumerate(result.stdout.split()) if sha
            }
        return self._positions

    def _resolve(self, commit: str) -> str | None:
        """Full sha for a possibly abbreviated commit, or ``None`` if unknown."""
        if commit not in self._resolved:
            result = self._git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
            self._resolved[commit] = result.stdout.strip() or None
        return self._resolved[commit]

    def position(self, commit: str) -> int | None:
        positions = self._load()
        if commit in positions:
            return positions[commit]
        full = self._resolve(commit)
        return positions.get(full) if full is not None else None

    def is_ancestor(self, older: str, newer: str) -> bool:
        return self._git("merge-base", "--is-ancestor", older, newer).returncode == 0


def _series_for(
    run: RunRecord, cell: str, arm: str, metric: str
) -> tuple[MetricSeries | None, str | None]:
    cell_record = run.cells.get(cell)
    if cell_record is None:
        return None, f"run does not contain cell {cell!r}"
    arm_metrics = cell_record.arms.get(arm)
    if arm_metrics is None:
        return None, f"cell {cell!r} does not contain arm {arm!r}"
    series = arm_metrics.get(metric)
    if series is None:
        return None, f"cell {cell!r} arm {arm!r} does not record metric {metric!r}"
    return series, None


def track_metric(
    runs: Iterable[RunRecord],
    cell: str,
    arm: str,
    metric: str,
    *,
    ordering: CommitOrdering | None = None,
    materiality_pct: float = DEFAULT_MATERIALITY_PCT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> MetricTrack:
    """Order the runs by git topology and compare each consecutive pair."""
    resolver = ordering if ordering is not None else GitCommitOrdering()
    excluded: list[tuple[str, str]] = []
    points: list[MetricPoint] = []

    for run in runs:
        if run.commit is None:
            # Not placed at HEAD: a run whose provenance was never recorded has
            # no position in history, and inventing one fabricates the history.
            excluded.append((run.run_id, "no recorded commit"))
            continue
        series, problem = _series_for(run, cell, arm, metric)
        if series is None:
            assert problem is not None
            excluded.append((run.run_id, problem))
            continue
        index = resolver.position(run.commit)
        if index is None:
            excluded.append((run.run_id, f"commit {run.commit[:12]} is not in this repository"))
            continue
        points.append(
            MetricPoint(
                run_id=run.run_id,
                commit=run.commit,
                topological_index=index,
                series=series,
                mean=sum(series.values) / len(series.values),
            )
        )

    # Ties (several runs on one commit) keep a deterministic order by run id;
    # they are genuinely simultaneous in history, so any other order would be
    # asserting an ordering the repository does not contain.
    points.sort(key=lambda point: (point.topological_index, point.run_id))

    runs_by_id = {run.run_id: run for run in runs}
    steps: list[MetricComparison] = []
    caveats: list[str] = []
    for older, newer in zip(points, points[1:], strict=False):
        notes: list[str] = []
        if older.commit != newer.commit and not resolver.is_ancestor(older.commit, newer.commit):
            caveat = (
                f"{older.run_id} -> {newer.run_id}: {older.commit[:12]} is not an ancestor of "
                f"{newer.commit[:12]}; these runs are on diverging branches"
            )
            caveats.append(caveat)
            notes.append(caveat)
        differences = config_differences(
            runs_by_id[older.run_id].cells[cell], runs_by_id[newer.run_id].cells[cell]
        )
        if differences:
            caveat = f"{older.run_id} -> {newer.run_id}: configuration changed: " + "; ".join(
                differences
            )
            caveats.append(caveat)
            notes.append(caveat)
        steps.append(
            compare_metric(
                cell=cell,
                arm=arm,
                baseline=older.series,
                candidate=newer.series,
                materiality_pct=materiality_pct,
                significance_t=significance_t,
                extra_notes=notes,
            )
        )

    return MetricTrack(
        cell=cell,
        arm=arm,
        metric=metric,
        points=tuple(points),
        steps=tuple(steps),
        trend=summarize_trend([point.mean for point in points]),
        excluded=tuple(excluded),
        caveats=tuple(caveats),
    )


def summarize_trend(means: Sequence[float]) -> MetricTrend:
    """Classify a history as trending, flat, or too short to say.

    Delegates to the gate's OLS helpers so that "flat" means here what it means
    when the gate decides whether an arm has finished warming up.
    """
    values = list(means)
    t_statistic = _slope_t_statistic(values)
    if t_statistic is None:
        # Fewer than MIN_FLAT_WINDOW points: two points fit their own slope
        # exactly, leaving no residual to build a standard error from, so every
        # short history would read as perfectly resolved.
        return MetricTrend(
            status=TrendStatus.UNTESTABLE,
            slope_t_statistic=None,
            flat_from_index=None,
            n=len(values),
        )
    status = (
        TrendStatus.TRENDING if t_statistic >= FLAT_TREND_T_STATISTIC else TrendStatus.FLAT
    )
    return MetricTrend(
        status=status,
        slope_t_statistic=t_statistic,
        flat_from_index=_flat_from_repeat(values),
        n=len(values),
    )


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Every proven regression and every unresolved shift across a history."""

    tracks: tuple[MetricTrack, ...] = ()
    regressions: tuple[MetricComparison, ...] = ()
    #: Surfaced alongside the regressions on purpose.  A material shift the run
    #: could not resolve is the outcome most short benchmarks deserve, and
    #: burying it under "no regressions found" is how this project has
    #: previously shipped a slowdown it had in fact measured.
    unresolved_shifts: tuple[MetricComparison, ...] = ()
    excluded: tuple[tuple[str, str], ...] = ()
    caveats: tuple[str, ...] = field(default_factory=tuple)


def detect_regressions(
    runs: Iterable[RunRecord],
    *,
    ordering: CommitOrdering | None = None,
    materiality_pct: float = DEFAULT_MATERIALITY_PCT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> RegressionReport:
    """Track every (cell, arm, metric) seen in *runs* and collect the bad news."""
    run_list = list(runs)
    resolver = ordering if ordering is not None else GitCommitOrdering()

    keys: set[tuple[str, str, str]] = set()
    for run in run_list:
        for cell_name, cell in run.cells.items():
            for arm_name, metrics in cell.arms.items():
                for metric_name in metrics:
                    keys.add((cell_name, arm_name, metric_name))

    tracks: list[MetricTrack] = []
    regressions: list[MetricComparison] = []
    unresolved: list[MetricComparison] = []
    excluded: dict[str, str] = {}
    caveats: list[str] = []
    for cell_name, arm_name, metric_name in sorted(keys):
        track = track_metric(
            run_list,
            cell_name,
            arm_name,
            metric_name,
            ordering=resolver,
            materiality_pct=materiality_pct,
            significance_t=significance_t,
        )
        tracks.append(track)
        regressions.extend(track.regressions)
        unresolved.extend(track.unresolved_shifts)
        caveats.extend(track.caveats)
        for run_id, reason in track.excluded:
            if reason == "no recorded commit":
                excluded[run_id] = reason

    return RegressionReport(
        tracks=tuple(tracks),
        regressions=tuple(regressions),
        unresolved_shifts=tuple(unresolved),
        excluded=tuple(sorted(excluded.items())),
        caveats=tuple(dict.fromkeys(caveats)),
    )


def render_track(track: MetricTrack) -> str:
    """A compact history: one line per run, then one line per step verdict."""
    lines = [
        f"{track.cell}/{track.arm}/{track.metric}"
        f"  ({track.points[0].series.unit if track.points else 'no data'},"
        f" {track.points[0].series.direction.value if track.points else 'n/a'})",
        f"  trend: {track.trend.status.value}"
        + (
            f"  |slope|/SE = {track.trend.slope_t_statistic:.2f}"
            f" (bar {FLAT_TREND_T_STATISTIC:g}, min window {MIN_FLAT_WINDOW})"
            if track.trend.slope_t_statistic is not None
            else f"  (fewer than {MIN_FLAT_WINDOW} points)"
        ),
        "",
    ]
    for point in track.points:
        lines.append(
            f"  {point.commit[:12]}  {point.run_id:<40}  "
            f"mean {point.mean:.6g}  n={len(point.series.values)}"
        )
    if track.steps:
        lines.append("")
        for step, point in zip(track.steps, track.points[1:], strict=True):
            error_bar = (
                f"+/- {step.delta_standard_error:.4g}"
                if step.delta_standard_error is not None
                else f"+/- no SE ({step.dispersion_basis})"
            )
            delta_pct = f"{step.delta_pct:+.2f}%" if step.delta_pct is not None else "n/a"
            lines.append(
                f"  -> {point.commit[:12]}  {step.delta:+.4g} {error_bar}  "
                f"{delta_pct}  {step.verdict.value}"
            )
    if track.excluded:
        lines.append("")
        lines.append("  excluded:")
        lines.extend(f"    {run_id}: {reason}" for run_id, reason in track.excluded)
    if track.caveats:
        lines.append("")
        lines.append("  caveats:")
        lines.extend(f"    {caveat}" for caveat in track.caveats)
    return "\n".join(lines)
