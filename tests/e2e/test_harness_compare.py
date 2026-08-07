"""Tests for :mod:`tests.e2e.harness.compare`.

Deliberately unmarked: these run in CI, on a laptop, with no GPU, no live
server and no ``/data``.  Every fixture is built in-process, and every test
here was demonstrated to fail against a mutation of the code it covers before
being kept -- the recurring defect in this repository is a green check that
measures nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.harness.compare import (
    DEFAULT_MATERIALITY_PCT,
    DEFAULT_SIGNIFICANCE_T,
    MispairedObservations,
    compare_metric,
    compare_runs,
    config_differences,
    pair_observations,
    render_comparison,
)
from tests.e2e.harness.model import (
    CellRecord,
    ComparisonVerdict,
    Direction,
    MetricSeries,
    RunRecord,
)


def make_series(
    name: str,
    values: tuple[float, ...],
    *,
    direction: Direction = Direction.HIGHER_IS_BETTER,
    unit: str = "tok/s",
    passes: tuple[int, ...] | None = None,
) -> MetricSeries:
    return MetricSeries(
        name=name, unit=unit, direction=direction, values=values, pass_indices=passes
    )


def make_cell(
    name: str,
    arms: dict[str, dict[str, MetricSeries]],
    *,
    config: dict[str, object] | None = None,
) -> CellRecord:
    return CellRecord(name=name, arms=arms, config=dict(config or {}))


def make_run(run_id: str, cells: dict[str, CellRecord], *, commit: str | None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        run_dir=Path("/nonexistent") / run_id,
        flavor="config-matrix",
        commit=commit,
        created_at=None,
        slurm_job_id=None,
        workload="synthetic",
        cells=cells,
    )


# A 20% candidate slowdown, the same fault magnitude the live matrix injects.
INJECTED_SLOWDOWN_PCT = 20.0

# Four clean throughput repeats: quiet enough that a 20% move is many standard
# errors wide, which is what makes this a control rather than a coin flip.
CLEAN_THROUGHPUT = (100.0, 101.0, 99.0, 100.0)


def _compare(
    baseline_values: tuple[float, ...],
    candidate_values: tuple[float, ...],
    *,
    direction: Direction = Direction.HIGHER_IS_BETTER,
    baseline_passes: tuple[int, ...] | None = None,
    candidate_passes: tuple[int, ...] | None = None,
    materiality_pct: float = DEFAULT_MATERIALITY_PCT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
):
    return compare_metric(
        cell="concurrency1-short",
        arm="candidate",
        baseline=make_series(
            "tokens_per_second", baseline_values, direction=direction, passes=baseline_passes
        ),
        candidate=make_series(
            "tokens_per_second", candidate_values, direction=direction, passes=candidate_passes
        ),
        materiality_pct=materiality_pct,
        significance_t=significance_t,
    )


# ---------------------------------------------------------------------------
# 1. The control: a known injected regression must be detected.
# ---------------------------------------------------------------------------


def test_injected_twenty_percent_slowdown_is_proven_regression() -> None:
    """A known 20% candidate slowdown must come back PROVEN_REGRESSION.

    This is the control that proves the detector fires at all.  Analogous to
    ``test_synthetic_throughput_regression_is_detected`` in
    ``test_inference_configuration_matrix.py``, but exercising the harness
    comparator's verdict ladder rather than the matrix's budget check.
    """
    slowed = tuple(v * (1.0 - INJECTED_SLOWDOWN_PCT / 100.0) for v in CLEAN_THROUGHPUT)
    result = _compare(CLEAN_THROUGHPUT, slowed)

    assert result.verdict is ComparisonVerdict.PROVEN_REGRESSION
    assert result.delta < 0.0
    assert result.delta_pct == pytest.approx(-INJECTED_SLOWDOWN_PCT, abs=1e-9)
    assert result.dispersion_basis == "measured"
    assert result.delta_standard_error is not None and result.delta_standard_error > 0.0
    # Many standard errors wide: this is not a marginal call.
    assert result.t_statistic is not None and result.t_statistic > DEFAULT_SIGNIFICANCE_T


def test_injected_slowdown_below_materiality_is_not_a_regression() -> None:
    """The same detector must not fire on a move smaller than the materiality bar.

    A detector that returns PROVEN_REGRESSION for everything would pass the
    control above, so the control needs its negative twin.
    """
    barely = tuple(v * 0.999 for v in CLEAN_THROUGHPUT)
    result = _compare(CLEAN_THROUGHPUT, barely)
    assert result.verdict is not ComparisonVerdict.PROVEN_REGRESSION
    # And specifically: this sample was precise enough to have seen a material
    # move, so the honest answer is PROVEN_FLAT rather than a shrug.
    assert result.verdict is ComparisonVerdict.PROVEN_FLAT


# ---------------------------------------------------------------------------
# 2. Zero variance is the absence of a measurement, not a tight one.
# ---------------------------------------------------------------------------


def test_zero_variance_pair_is_unresolved_with_no_standard_error() -> None:
    """Five bit-identical readings must not be reported as a resolved flat result.

    Job 369162 published ``candidate_acceptance_stdev = 0.0`` across five
    repeats and anything dividing by it read infinite headroom.  The standard
    error here has to be ``None`` -- so that a consumer dividing by it fails
    loudly -- and the verdict has to be UNRESOLVED rather than PROVEN_FLAT.
    """
    identical = (100.0, 100.0, 100.0, 100.0, 100.0)
    result = _compare(identical, identical)

    assert result.delta == 0.0
    assert result.delta_standard_error is None, "a degenerate SE must never be published as 0.0"
    assert result.t_statistic is None
    assert result.dispersion_basis == "degenerate"
    assert result.verdict is ComparisonVerdict.UNRESOLVED
    assert result.verdict is not ComparisonVerdict.PROVEN_FLAT


def test_single_observation_pair_is_unresolved_and_unsampled() -> None:
    """One repeat per arm has never had anything to disperse."""
    result = _compare((100.0,), (104.0,))
    assert result.delta_standard_error is None
    assert result.dispersion_basis == "unsampled"
    assert result.verdict is ComparisonVerdict.UNRESOLVED


def test_reduced_series_cannot_be_given_a_standard_error_after_the_fact() -> None:
    """A pre-reduced mean is UNRESOLVED, and says why in its notes."""
    result = compare_metric(
        cell="c",
        arm="a",
        baseline=MetricSeries(
            name="tokens_per_second",
            unit="tok/s",
            direction=Direction.HIGHER_IS_BETTER,
            values=(100.0,),
            reduced=True,
        ),
        candidate=MetricSeries(
            name="tokens_per_second",
            unit="tok/s",
            direction=Direction.HIGHER_IS_BETTER,
            values=(80.0,),
            reduced=True,
        ),
    )
    assert result.verdict is ComparisonVerdict.UNRESOLVED
    assert result.delta_standard_error is None
    assert any("pre-reduced" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 3 and 4. Flat-because-proven vs flat-because-blind.
# ---------------------------------------------------------------------------

# Tight repeats: a 2% move on a mean of 100 is 2.0 units, which is roughly 40
# standard errors here, so a material shift could not have hidden.
TIGHT_BASELINE = (100.00, 100.10, 99.90, 100.05, 99.95)
TIGHT_CANDIDATE = (100.02, 99.98, 100.05, 99.97, 100.00)

# Same means, but a sample so dispersed that a 2% move would be well inside the
# error bar.  The delta is identically small; only the precision differs.
NOISY_BASELINE = (80.0, 120.0, 90.0, 110.0, 100.0)
NOISY_CANDIDATE = (82.0, 118.0, 92.0, 108.0, 100.0)


def test_genuinely_flat_and_well_resolved_pair_is_proven_flat() -> None:
    """A small delta measured precisely enough to have seen a material one."""
    result = _compare(TIGHT_BASELINE, TIGHT_CANDIDATE)

    assert abs(result.delta_pct or 0.0) < DEFAULT_MATERIALITY_PCT
    assert result.delta_standard_error is not None
    material_delta = DEFAULT_MATERIALITY_PCT / 100.0 * abs(result.baseline_mean)
    assert material_delta >= DEFAULT_SIGNIFICANCE_T * result.delta_standard_error
    assert result.verdict is ComparisonVerdict.PROVEN_FLAT


def test_small_delta_on_an_imprecise_sample_is_unresolved_not_proven_flat() -> None:
    """The crux: being unable to see a shift is not the same as there being none.

    The delta here is as small as in the PROVEN_FLAT case above, but the sample
    is so noisy that a full materiality-sized move would have been inside one
    standard error.  Calling that "flat" would publish a conclusion the run did
    not earn.
    """
    result = _compare(NOISY_BASELINE, NOISY_CANDIDATE)

    assert abs(result.delta_pct or 0.0) < DEFAULT_MATERIALITY_PCT
    assert result.delta_standard_error is not None
    material_delta = DEFAULT_MATERIALITY_PCT / 100.0 * abs(result.baseline_mean)
    assert material_delta < DEFAULT_SIGNIFICANCE_T * result.delta_standard_error
    assert result.verdict is ComparisonVerdict.UNRESOLVED
    assert result.verdict is not ComparisonVerdict.PROVEN_FLAT
    assert any("could not have resolved" in note for note in result.notes)


def test_material_delta_on_a_noisy_sample_is_material_shift_unresolved() -> None:
    """A 5% move the sample cannot distinguish from noise is not a regression."""
    shifted = tuple(v - 5.0 for v in NOISY_BASELINE)
    result = _compare(NOISY_BASELINE, shifted)

    assert result.delta_pct is not None
    assert abs(result.delta_pct) >= DEFAULT_MATERIALITY_PCT
    assert result.t_statistic is not None and result.t_statistic < DEFAULT_SIGNIFICANCE_T
    assert result.verdict is ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED
    assert result.verdict is not ComparisonVerdict.PROVEN_REGRESSION


# ---------------------------------------------------------------------------
# 5. Direction decides the sign of the finding.
# ---------------------------------------------------------------------------

DIRECTION_BASELINE = (10.0, 10.5, 9.5, 10.0)
DIRECTION_CANDIDATE = (12.0, 12.5, 11.5, 12.0)


def test_same_numbers_opposite_directions_give_opposite_verdicts() -> None:
    """Going up is a regression for latency and an improvement for throughput.

    Identical inputs, identical delta, identical t: only ``direction`` differs.
    A comparator that inferred direction from the metric name would eventually
    guess wrong on a metric nobody thought about, so it is never inferred.
    """
    lower_better = _compare(
        DIRECTION_BASELINE, DIRECTION_CANDIDATE, direction=Direction.LOWER_IS_BETTER
    )
    higher_better = _compare(
        DIRECTION_BASELINE, DIRECTION_CANDIDATE, direction=Direction.HIGHER_IS_BETTER
    )

    assert lower_better.delta == higher_better.delta > 0.0
    assert lower_better.t_statistic == higher_better.t_statistic
    assert lower_better.verdict is ComparisonVerdict.PROVEN_REGRESSION
    assert higher_better.verdict is ComparisonVerdict.PROVEN_IMPROVEMENT


def test_neutral_direction_is_always_descriptive() -> None:
    """Token and request counts are diffed and reported, never judged."""
    result = _compare(
        DIRECTION_BASELINE, DIRECTION_CANDIDATE, direction=Direction.NEUTRAL
    )
    assert result.verdict is ComparisonVerdict.DESCRIPTIVE
    assert result.delta > 0.0


# ---------------------------------------------------------------------------
# 6. Mis-paired arms must not silently pair fewer observations.
# ---------------------------------------------------------------------------


def test_mispaired_pass_indices_raise_rather_than_pairing_the_overlap() -> None:
    """Pairing the overlap would shrink the error bar without anyone noticing.

    ``_pair_samples`` in ``test_serving_activation_capture_overhead.py`` says
    exactly this; the comparator has to hold the same line.
    """
    baseline = make_series("tokens_per_second", (100.0, 101.0, 99.0, 100.0), passes=(0, 1, 2, 3))
    candidate = make_series("tokens_per_second", (80.0, 81.0, 79.0, 80.0), passes=(0, 1, 2, 7))

    with pytest.raises(MispairedObservations, match="did not cover the same pass set"):
        pair_observations(baseline, candidate)


def test_mispaired_pass_indices_are_reported_with_full_sample_sizes() -> None:
    """The comparison survives, but as UNRESOLVED with no fabricated error bar.

    ``baseline_n`` / ``candidate_n`` report what was *measured* (4 and 4), not
    the 3 an intersection would have yielded: a plausible-looking smaller pair
    count is precisely how this failure hides.
    """
    result = compare_metric(
        cell="c",
        arm="a",
        baseline=make_series(
            "tokens_per_second", (100.0, 101.0, 99.0, 100.0), passes=(0, 1, 2, 3)
        ),
        candidate=make_series("tokens_per_second", (80.0, 81.0, 79.0, 80.0), passes=(0, 1, 2, 7)),
    )

    assert result.verdict is ComparisonVerdict.UNRESOLVED
    assert result.delta_standard_error is None
    assert result.t_statistic is None
    assert (result.baseline_n, result.candidate_n) == (4, 4)
    assert any("did not cover the same pass set" in note for note in result.notes)


def test_duplicate_pass_index_is_rejected() -> None:
    """A repeated pass index cannot identify an observation, so it is an error."""
    baseline = make_series("tokens_per_second", (100.0, 101.0, 99.0), passes=(0, 1, 1))
    candidate = make_series("tokens_per_second", (100.0, 101.0, 99.0), passes=(0, 1, 2))
    with pytest.raises(MispairedObservations, match="twice"):
        pair_observations(baseline, candidate)


def test_pass_indices_are_matched_not_zipped() -> None:
    """Same pass set in a different recorded order still pairs by index."""
    baseline = make_series("tokens_per_second", (10.0, 20.0, 30.0), passes=(0, 1, 2))
    candidate = make_series("tokens_per_second", (30.0, 20.0, 10.0), passes=(2, 1, 0))
    paired = pair_observations(baseline, candidate)
    assert paired.baseline == (10.0, 20.0, 30.0)
    assert paired.candidate == (10.0, 20.0, 30.0)
    assert paired.positional is False


def test_missing_pass_indices_fall_back_to_position_and_say_so() -> None:
    """Positional pairing is allowed, but it is recorded on the result."""
    result = compare_metric(
        cell="c",
        arm="a",
        baseline=make_series("tokens_per_second", (100.0, 101.0), passes=(0, 1)),
        candidate=make_series("tokens_per_second", (100.0, 102.0), passes=None),
    )
    assert result.positional_pairing is True
    assert any("list position" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Run-level assembly: config guards and one-sided cells.
# ---------------------------------------------------------------------------


def _throughput_cell(name: str, values: tuple[float, ...], **config: object) -> CellRecord:
    return make_cell(
        name,
        {"candidate": {"tokens_per_second": make_series("tokens_per_second", values)}},
        config=config,
    )


def test_cells_with_differing_configuration_are_incomparable_not_compared() -> None:
    """A concurrency change explains any delta better than the commit does."""
    baseline = make_run(
        "run-a",
        {"c1": _throughput_cell("c1", CLEAN_THROUGHPUT, concurrency=1, execution_mode="eager")},
        commit="a" * 40,
    )
    candidate = make_run(
        "run-b",
        {
            "c1": _throughput_cell(
                "c1", (60.0, 61.0, 59.0, 60.0), concurrency=16, execution_mode="eager"
            )
        },
        commit="b" * 40,
    )

    result = compare_runs(baseline, candidate)
    assert result.comparisons == ()
    assert len(result.incomparable) == 1
    assert "concurrency" in result.incomparable[0]


def test_identical_configuration_is_compared() -> None:
    """The config guard must not refuse everything -- the negative of the test above."""
    baseline = make_run(
        "run-a", {"c1": _throughput_cell("c1", CLEAN_THROUGHPUT, concurrency=1)}, commit="a" * 40
    )
    candidate = make_run(
        "run-b",
        {"c1": _throughput_cell("c1", tuple(v * 0.8 for v in CLEAN_THROUGHPUT), concurrency=1)},
        commit="b" * 40,
    )
    result = compare_runs(baseline, candidate)
    assert result.incomparable == ()
    assert len(result.comparisons) == 1
    assert result.regressions == result.comparisons


def test_missing_configuration_key_on_one_side_is_a_difference() -> None:
    left = make_cell("c1", {}, config={"speculative_depth": 3})
    right = make_cell("c1", {}, config={})
    assert config_differences(left, right) == ("speculative_depth: 3 -> <absent>",)
    assert config_differences(right, right) == ()


def test_one_sided_cells_and_metrics_are_reported_not_dropped() -> None:
    """A metric that vanished between two commits is a finding, not a non-event."""
    baseline = make_run(
        "run-a",
        {
            "c1": make_cell(
                "c1",
                {
                    "candidate": {
                        "tokens_per_second": make_series("tokens_per_second", CLEAN_THROUGHPUT),
                        "ttft_ms": make_series(
                            "ttft_ms", (10.0, 11.0), direction=Direction.LOWER_IS_BETTER
                        ),
                    }
                },
            ),
            "gone": make_cell("gone", {}),
        },
        commit="a" * 40,
    )
    candidate = make_run(
        "run-b",
        {
            "c1": make_cell(
                "c1",
                {
                    "candidate": {
                        "tokens_per_second": make_series("tokens_per_second", CLEAN_THROUGHPUT)
                    },
                    "stock": {},
                },
            ),
            "new": make_cell("new", {}),
        },
        commit="b" * 40,
    )

    result = compare_runs(baseline, candidate)
    assert "cell gone" in result.baseline_only
    assert "cell c1 arm candidate metric ttft_ms" in result.baseline_only
    assert "cell new" in result.candidate_only
    assert "cell c1 arm stock" in result.candidate_only


def test_direction_disagreement_between_runs_is_an_error() -> None:
    """One of the two extractors is wrong; the comparison would be meaningless."""
    with pytest.raises(ValueError, match="one of the two"):
        compare_metric(
            cell="c",
            arm="a",
            baseline=make_series("x", (1.0, 2.0), direction=Direction.HIGHER_IS_BETTER),
            candidate=make_series("x", (1.0, 2.0), direction=Direction.LOWER_IS_BETTER),
        )


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------


def _rendered() -> str:
    baseline = make_run(
        "run-a",
        {
            "c1": make_cell(
                "c1",
                {
                    "candidate": {
                        "tokens_per_second": make_series("tokens_per_second", CLEAN_THROUGHPUT),
                        "requests": make_series(
                            "requests", (5.0, 5.0, 5.0, 5.0), direction=Direction.NEUTRAL
                        ),
                        "ttft_ms": make_series(
                            "ttft_ms", TIGHT_BASELINE, direction=Direction.LOWER_IS_BETTER
                        ),
                    }
                },
            ),
        },
        commit="a" * 40,
    )
    candidate = make_run(
        "run-b",
        {
            "c1": make_cell(
                "c1",
                {
                    "candidate": {
                        "tokens_per_second": make_series(
                            "tokens_per_second", tuple(v * 0.8 for v in CLEAN_THROUGHPUT)
                        ),
                        "requests": make_series(
                            "requests", (5.0, 5.0, 5.0, 5.0), direction=Direction.NEUTRAL
                        ),
                        "ttft_ms": make_series(
                            "ttft_ms", TIGHT_CANDIDATE, direction=Direction.LOWER_IS_BETTER
                        ),
                    }
                },
            ),
        },
        commit="b" * 40,
    )
    return render_comparison(compare_runs(baseline, candidate))


def test_render_puts_the_proven_regression_first() -> None:
    """Most-material-first: the reader must not have to scan for the bad news."""
    text = _rendered()
    body = [line for line in text.splitlines() if line.startswith("c1")]
    assert body, text
    assert "proven_regression" in body[0]
    assert "tokens_per_second" in body[0]


def test_render_marks_a_missing_standard_error_explicitly() -> None:
    """A blank error-bar column reads as 'small'; this has to read as 'absent'."""
    text = _rendered()
    descriptive = [
        line for line in text.splitlines() if "requests" in line and line.startswith("c1")
    ]
    assert descriptive, text
    assert "no SE" in descriptive[0]
    assert "degenerate" in descriptive[0]


def test_render_always_shows_the_error_bar_next_to_the_delta() -> None:
    text = _rendered()
    regression_line = next(line for line in text.splitlines() if "proven_regression" in line)
    assert "+/-" in regression_line


def test_render_reports_a_provenance_less_run_as_such() -> None:
    """'no recorded commit' rather than a plausible blank."""
    empty = make_run("r", {}, commit=None)
    text = render_comparison(compare_runs(empty, empty))
    assert "no recorded commit" in text
    assert "(no comparable metrics)" in text
