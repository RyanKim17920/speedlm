"""Paired comparison of two benchmark runs, with the project's dispersion discipline.

The comparator exists because a bare mean is not a finding.  The repository has
twice reached a conclusion that was smaller than one standard error of the
measurement that produced it, and once published ``stdev = 0.0`` from five
bit-identical readings and read it as infinite headroom.  Both mistakes are
already named and already solved in :mod:`speedlm.gate.decide`; this module
*imports* that machinery rather than growing a second, subtly different copy of
it:

* :func:`speedlm.gate.decide._delta_standard_error_raw` -- the standard error of
  an arm-to-arm delta of means, ``None`` rather than ``0.0`` when the sample
  cannot support one.
* :func:`speedlm.gate.decide._dispersion_basis` -- whether that standard error
  is a measurement (``MEASURED``), an artefact of every repeat returning the
  same number (``DEGENERATE``), or absent because there were fewer than two
  repeats (``UNSAMPLED``).

The private names are imported deliberately.  They are the gate's decision
procedure, and a comparator that reimplemented them would be free to drift from
the thing that actually gates promotion.  If they move, this module should
break loudly rather than quietly disagree.

The verdict ladder mirrors
:class:`speedlm.gate.decide.StationarityStatus`: the interesting distinction is
not the sign of the delta but whether the sample resolved anything.  In
particular a small delta is only :attr:`~model.ComparisonVerdict.PROVEN_FLAT`
when the sample was *precise enough that a material shift would have shown up*.
A small delta measured with a huge error bar is
:attr:`~model.ComparisonVerdict.UNRESOLVED`, not flat -- that distinction is the
entire reason this module exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from speedlm.gate.decide import (  # noqa: PLC2701 - see module docstring
    _delta_standard_error_raw,
    _dispersion_basis,
)

from .model import (
    CellRecord,
    ComparisonVerdict,
    Direction,
    MetricComparison,
    MetricSeries,
    RunComparison,
    RunRecord,
)

#: Below this percentage the two runs are treated as measuring the same thing,
#: *provided* the sample could have resolved a shift of that size.  Two percent
#: is the smallest throughput move this project has ever acted on; anything
#: under it has always been noise or a rounding artefact of the token counters.
DEFAULT_MATERIALITY_PCT: Final = 2.0

#: ``|delta| / SE`` a material delta must clear before it is called proven.
#: Two standard errors, the conventional bar, and the one the promotion gate's
#: throughput criterion already uses.
DEFAULT_SIGNIFICANCE_T: Final = 2.0

#: Configuration keys that make two cells incomparable when they disagree.
#:
#: These are the dimensions the benchmark matrix varies on purpose.  Comparing
#: a concurrency-1 cell against a concurrency-16 cell, or two different
#: speculative depths, produces a delta that is real, large, and about nothing
#: the commit under test did.  Such pairs are reported as ``incomparable``
#: rather than compared.
COMPARABILITY_KEYS: Final[tuple[str, ...]] = (
    "execution_mode",
    "concurrency",
    "context_band",
    "model",
    "speculative_depth",
)

_MISSING: Final = object()


class MispairedObservations(ValueError):
    """Two arms recorded different pass indices and cannot be paired.

    Raised rather than silently intersecting.  ``tests/e2e/
    test_serving_activation_capture_overhead.py::_pair_samples`` learned this
    the hard way and says why: silently pairing fewer observations than were
    measured shrinks the error bar without anyone noticing, which turns an
    unresolved run into a confident one.
    """


@dataclass(frozen=True, slots=True)
class PairedObservations:
    """Two arms' values, aligned so that entry *i* of each came from one pass."""

    baseline: tuple[float, ...]
    candidate: tuple[float, ...]
    #: Set when alignment fell back to list position because at least one side
    #: recorded no pass indices.
    positional: bool
    notes: tuple[str, ...] = ()


def pair_observations(baseline: MetricSeries, candidate: MetricSeries) -> PairedObservations:
    """Align two series by pass index, falling back to list position.

    Two arms are only comparable on observations that came from the same
    position in a run -- the ABBA block schedule exists precisely so that they
    do.  When both sides recorded pass indices and the index *sets* disagree,
    this raises :class:`MispairedObservations`; it never quietly drops the
    unmatched entries.
    """
    if baseline.pass_indices is None or candidate.pass_indices is None:
        notes: list[str] = ["paired by list position: at least one arm recorded no pass indices"]
        n = min(len(baseline.values), len(candidate.values))
        if len(baseline.values) != len(candidate.values):
            notes.append(
                "arms have unequal observation counts "
                f"({len(baseline.values)} vs {len(candidate.values)}); "
                f"compared on the first {n}"
            )
        return PairedObservations(
            baseline=tuple(baseline.values[:n]),
            candidate=tuple(candidate.values[:n]),
            positional=True,
            notes=tuple(notes),
        )

    baseline_by_pass = _by_pass_index(baseline, "baseline")
    candidate_by_pass = _by_pass_index(candidate, "candidate")
    if set(baseline_by_pass) != set(candidate_by_pass):
        baseline_only = sorted(set(baseline_by_pass) - set(candidate_by_pass))
        candidate_only = sorted(set(candidate_by_pass) - set(baseline_by_pass))
        raise MispairedObservations(
            f"metric {baseline.name!r}: the two arms did not cover the same pass set "
            f"(baseline-only={baseline_only} candidate-only={candidate_only}); "
            "pairing the overlap would shrink the error bar without anyone noticing"
        )
    order = sorted(baseline_by_pass)
    return PairedObservations(
        baseline=tuple(baseline_by_pass[p] for p in order),
        candidate=tuple(candidate_by_pass[p] for p in order),
        positional=False,
    )


def _by_pass_index(series: MetricSeries, side: str) -> dict[int, float]:
    assert series.pass_indices is not None
    mapping: dict[int, float] = {}
    for pass_index, value in zip(series.pass_indices, series.values, strict=True):
        if pass_index in mapping:
            raise MispairedObservations(
                f"metric {series.name!r}: {side} arm recorded pass index "
                f"{pass_index} twice; the index cannot identify an observation"
            )
        mapping[pass_index] = value
    return mapping


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def compare_metric(
    *,
    cell: str,
    arm: str,
    baseline: MetricSeries,
    candidate: MetricSeries,
    materiality_pct: float = DEFAULT_MATERIALITY_PCT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
    extra_notes: Iterable[str] = (),
) -> MetricComparison:
    """Compare one metric between two runs and assign a verdict.

    The direction is taken from the series and never inferred from the metric's
    name: "latency went up 4%" and "throughput went up 4%" are opposite
    findings, and a comparator that guesses will eventually guess wrong on a
    metric nobody thought about.
    """
    if baseline.direction is not candidate.direction:
        raise ValueError(
            f"metric {baseline.name!r} is {baseline.direction.value} in the baseline "
            f"but {candidate.direction.value} in the candidate; one of the two "
            "extractors is wrong and the comparison would be meaningless"
        )
    notes = list(extra_notes)
    positional = False
    try:
        paired = pair_observations(baseline, candidate)
    except MispairedObservations as exc:
        # Reported, not silently repaired: the arms disagree about what was
        # measured, so there is no honest error bar for this metric.
        return _unpairable_comparison(
            cell=cell,
            arm=arm,
            baseline=baseline,
            candidate=candidate,
            notes=tuple([*notes, str(exc)]),
        )
    positional = paired.positional
    notes.extend(paired.notes)

    if baseline.reduced or candidate.reduced:
        notes.append(
            "at least one side is a pre-reduced summary statistic; "
            "no dispersion was preserved, so no standard error exists"
        )

    baseline_values = list(paired.baseline)
    candidate_values = list(paired.candidate)
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    delta = candidate_mean - baseline_mean
    delta_pct = delta / abs(baseline_mean) * 100.0 if baseline_mean != 0.0 else None
    if delta_pct is None:
        notes.append("baseline mean is zero; a percentage delta is undefined")

    standard_error = _delta_standard_error_raw(baseline_values, candidate_values)
    basis = _dispersion_basis(baseline_values, candidate_values)
    if standard_error == 0.0:
        # DEGENERATE.  Every repeat returned the same number in both arms, so
        # there is no variance estimate at all.  Publishing 0.0 would let a
        # consumer compute infinite headroom off a measurement that never
        # measured anything; publish None so that division fails loudly.
        standard_error = None
        notes.append(
            "every paired observation was identical, so the standard error is "
            "an artefact rather than a measurement"
        )
    t_statistic = (
        abs(delta) / standard_error if standard_error is not None and standard_error > 0 else None
    )

    verdict, verdict_note = _verdict(
        direction=baseline.direction,
        delta=delta,
        delta_pct=delta_pct,
        baseline_mean=baseline_mean,
        standard_error=standard_error,
        t_statistic=t_statistic,
        materiality_pct=materiality_pct,
        significance_t=significance_t,
    )
    if verdict_note:
        notes.append(verdict_note)

    return MetricComparison(
        cell=cell,
        arm=arm,
        metric=baseline.name,
        unit=baseline.unit,
        direction=baseline.direction,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        delta_pct=delta_pct,
        delta_standard_error=standard_error,
        dispersion_basis=basis.value,
        t_statistic=t_statistic,
        verdict=verdict,
        baseline_n=len(baseline_values),
        candidate_n=len(candidate_values),
        positional_pairing=positional,
        notes=tuple(notes),
    )


def _unpairable_comparison(
    *,
    cell: str,
    arm: str,
    baseline: MetricSeries,
    candidate: MetricSeries,
    notes: tuple[str, ...],
) -> MetricComparison:
    """A comparison that reports the full sample sizes and refuses a verdict.

    ``baseline_n`` / ``candidate_n`` deliberately report what was *measured*,
    not what could be paired, so that a reader can see the mismatch rather than
    a plausible-looking smaller pair count.
    """
    baseline_mean = _mean(baseline.values)
    candidate_mean = _mean(candidate.values)
    delta = candidate_mean - baseline_mean
    return MetricComparison(
        cell=cell,
        arm=arm,
        metric=baseline.name,
        unit=baseline.unit,
        direction=baseline.direction,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        delta_pct=delta / abs(baseline_mean) * 100.0 if baseline_mean != 0.0 else None,
        delta_standard_error=None,
        dispersion_basis=_dispersion_basis([], []).value,
        t_statistic=None,
        verdict=ComparisonVerdict.UNRESOLVED,
        baseline_n=len(baseline.values),
        candidate_n=len(candidate.values),
        positional_pairing=False,
        notes=notes,
    )


def _verdict(
    *,
    direction: Direction,
    delta: float,
    delta_pct: float | None,
    baseline_mean: float,
    standard_error: float | None,
    t_statistic: float | None,
    materiality_pct: float,
    significance_t: float,
) -> tuple[ComparisonVerdict, str | None]:
    """The verdict ladder.  See the module docstring for why it is shaped this way."""
    if direction is Direction.NEUTRAL:
        return ComparisonVerdict.DESCRIPTIVE, None
    if standard_error is None or t_statistic is None:
        return ComparisonVerdict.UNRESOLVED, None
    if delta_pct is None:
        return (
            ComparisonVerdict.UNRESOLVED,
            "materiality is a percentage and the baseline mean is zero",
        )

    if abs(delta_pct) < materiality_pct:
        # A small delta only proves flatness if a material delta would have
        # been detectable.  Otherwise the run simply could not see anything,
        # and calling that "flat" is the error this whole module guards.
        material_delta = materiality_pct / 100.0 * abs(baseline_mean)
        if material_delta >= significance_t * standard_error:
            return ComparisonVerdict.PROVEN_FLAT, None
        return (
            ComparisonVerdict.UNRESOLVED,
            f"delta is under the {materiality_pct:g}% materiality bar, but the sample "
            f"could not have resolved a material shift either "
            f"(SE {standard_error:.4g} needs to be at most "
            f"{material_delta / significance_t:.4g})",
        )

    if t_statistic < significance_t:
        return ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED, None

    worse = delta > 0.0 if direction is Direction.LOWER_IS_BETTER else delta < 0.0
    return (
        ComparisonVerdict.PROVEN_REGRESSION if worse else ComparisonVerdict.PROVEN_IMPROVEMENT
    ), None


def config_differences(baseline: CellRecord, candidate: CellRecord) -> tuple[str, ...]:
    """Comparability-relevant config keys on which two cells disagree."""
    differences: list[str] = []
    for key in COMPARABILITY_KEYS:
        left: Any = baseline.config.get(key, _MISSING)
        right: Any = candidate.config.get(key, _MISSING)
        if left is _MISSING and right is _MISSING:
            continue
        if left != right:
            differences.append(
                f"{key}: {_render_config_value(left)} -> {_render_config_value(right)}"
            )
    return tuple(differences)


def _render_config_value(value: Any) -> str:
    return "<absent>" if value is _MISSING else repr(value)


def compare_runs(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    materiality_pct: float = DEFAULT_MATERIALITY_PCT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> RunComparison:
    """Diff two runs cell by cell, arm by arm, metric by metric."""
    comparisons: list[MetricComparison] = []
    baseline_only: list[str] = []
    candidate_only: list[str] = []
    incomparable: list[str] = []

    for cell_name in sorted(set(baseline.cells) - set(candidate.cells)):
        baseline_only.append(f"cell {cell_name}")
    for cell_name in sorted(set(candidate.cells) - set(baseline.cells)):
        candidate_only.append(f"cell {cell_name}")

    for cell_name in sorted(set(baseline.cells) & set(candidate.cells)):
        baseline_cell = baseline.cells[cell_name]
        candidate_cell = candidate.cells[cell_name]
        differences = config_differences(baseline_cell, candidate_cell)
        if differences:
            incomparable.append(f"cell {cell_name}: {'; '.join(differences)}")
            continue

        for arm_name in sorted(set(baseline_cell.arms) - set(candidate_cell.arms)):
            baseline_only.append(f"cell {cell_name} arm {arm_name}")
        for arm_name in sorted(set(candidate_cell.arms) - set(baseline_cell.arms)):
            candidate_only.append(f"cell {cell_name} arm {arm_name}")

        for arm_name in sorted(set(baseline_cell.arms) & set(candidate_cell.arms)):
            baseline_metrics = baseline_cell.arms[arm_name]
            candidate_metrics = candidate_cell.arms[arm_name]
            for metric in sorted(set(baseline_metrics) - set(candidate_metrics)):
                baseline_only.append(f"cell {cell_name} arm {arm_name} metric {metric}")
            for metric in sorted(set(candidate_metrics) - set(baseline_metrics)):
                candidate_only.append(f"cell {cell_name} arm {arm_name} metric {metric}")
            for metric in sorted(set(baseline_metrics) & set(candidate_metrics)):
                comparisons.append(
                    compare_metric(
                        cell=cell_name,
                        arm=arm_name,
                        baseline=baseline_metrics[metric],
                        candidate=candidate_metrics[metric],
                        materiality_pct=materiality_pct,
                        significance_t=significance_t,
                    )
                )

    return RunComparison(
        baseline=baseline.run_id,
        candidate=candidate.run_id,
        baseline_commit=baseline.commit,
        candidate_commit=candidate.commit,
        comparisons=tuple(comparisons),
        baseline_only=tuple(baseline_only),
        candidate_only=tuple(candidate_only),
        incomparable=tuple(incomparable),
    )


#: Order verdicts are listed in.  Proven bad news first, then the honest
#: "we could not tell" outcomes, then good news, then the descriptive tail.
_VERDICT_RANK: Final[dict[ComparisonVerdict, int]] = {
    ComparisonVerdict.PROVEN_REGRESSION: 0,
    ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED: 1,
    ComparisonVerdict.UNRESOLVED: 2,
    ComparisonVerdict.PROVEN_IMPROVEMENT: 3,
    ComparisonVerdict.PROVEN_FLAT: 4,
    ComparisonVerdict.DESCRIPTIVE: 5,
}

#: Printed in place of the error bar when there is none.  A blank column would
#: read as "small"; this reads as "absent", which is what it means.
NO_STANDARD_ERROR_MARKER: Final = "no SE"


def sort_key(comparison: MetricComparison) -> tuple[int, float, str, str, str]:
    """Most-material-first ordering: worst verdict, then largest relative move."""
    return (
        _VERDICT_RANK[comparison.verdict],
        -abs(comparison.delta_pct if comparison.delta_pct is not None else 0.0),
        comparison.cell,
        comparison.arm,
        comparison.metric,
    )


def render_comparison(comparison: RunComparison) -> str:
    """A human-readable table: delta plus-or-minus its standard error, and a verdict."""
    lines: list[str] = [
        f"baseline  {comparison.baseline} ({_short(comparison.baseline_commit)})",
        f"candidate {comparison.candidate} ({_short(comparison.candidate_commit)})",
        "",
    ]
    rows = [_row(c) for c in sorted(comparison.comparisons, key=sort_key)]
    header = ("cell", "arm", "metric", "delta", "delta %", "t", "verdict")
    if rows:
        widths = [
            max(len(header[i]), max(len(row[i]) for row in rows)) for i in range(len(header))
        ]
        lines.append("  ".join(head.ljust(widths[i]) for i, head in enumerate(header)).rstrip())
        lines.append("  ".join("-" * widths[i] for i in range(len(header))))
        for row in rows:
            lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(row))).rstrip())
    else:
        lines.append("(no comparable metrics)")

    footnotes = [
        c
        for c in sorted(comparison.comparisons, key=sort_key)
        if c.notes or c.positional_pairing
    ]
    if footnotes:
        lines.append("")
        lines.append("notes:")
        for c in footnotes:
            label = f"{c.cell}/{c.arm}/{c.metric}"
            if c.positional_pairing:
                lines.append(f"  {label}: paired by list position, not by pass index")
            for note in c.notes:
                lines.append(f"  {label}: {note}")

    for title, entries in (
        ("baseline only", comparison.baseline_only),
        ("candidate only", comparison.candidate_only),
        ("incomparable (configuration differs)", comparison.incomparable),
    ):
        if entries:
            lines.append("")
            lines.append(f"{title}:")
            lines.extend(f"  {entry}" for entry in entries)

    return "\n".join(lines)


def _row(c: MetricComparison) -> tuple[str, str, str, str, str, str, str]:
    if c.delta_standard_error is None:
        delta = f"{c.delta:+.4g} +/- {NO_STANDARD_ERROR_MARKER} ({c.dispersion_basis}) {c.unit}"
        t_text = "n/a"
    else:
        delta = f"{c.delta:+.4g} +/- {c.delta_standard_error:.4g} {c.unit}"
        t_text = f"{c.t_statistic:.2f}" if c.t_statistic is not None else "n/a"
    delta_pct = f"{c.delta_pct:+.2f}%" if c.delta_pct is not None else "n/a"
    return (c.cell, c.arm, c.metric, delta, delta_pct, t_text, c.verdict.value)


def _short(commit: str | None) -> str:
    if commit is None:
        # Not "unknown": the index records provenance-less runs on purpose, and
        # a reader has to know that this pair cannot be placed in history.
        return "no recorded commit"
    return commit[:12]
