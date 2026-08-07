"""Tests for :mod:`tests.e2e.harness.regression`.

Unmarked and CI-safe: no GPU, no ``/data``, no live server.  The topology tests
drive an injected :class:`~tests.e2e.harness.regression.CommitOrdering` so that
the ordering *rule* is exercised against a known graph rather than against
whatever this repository's history happens to look like; one additional test
checks the real git-backed implementation against commits it resolves itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.harness.model import (
    CellRecord,
    ComparisonVerdict,
    Direction,
    MetricSeries,
    RunRecord,
)
from tests.e2e.harness.regression import (
    REPO_ROOT,
    GitCommitOrdering,
    TrendStatus,
    detect_regressions,
    render_track,
    summarize_trend,
    track_metric,
)

CELL = "concurrency1-short"
ARM = "candidate"
METRIC = "tokens_per_second"


class FakeOrdering:
    """A known commit graph: ``positions`` is the topological linearisation."""

    def __init__(self, positions: dict[str, int], forks: set[tuple[str, str]] = frozenset()):
        self.positions = positions
        # Pairs that are NOT ancestor-related, i.e. sit on diverging branches.
        self.forks = set(forks)

    def position(self, commit: str) -> int | None:
        return self.positions.get(commit)

    def is_ancestor(self, older: str, newer: str) -> bool:
        return (older, newer) not in self.forks


def make_run(
    run_id: str,
    values: tuple[float, ...],
    *,
    commit: str | None,
    created_at: str | None = None,
    config: dict[str, object] | None = None,
    cell: str = CELL,
    arm: str = ARM,
    metric: str = METRIC,
    direction: Direction = Direction.HIGHER_IS_BETTER,
) -> RunRecord:
    series = MetricSeries(
        name=metric, unit="tok/s", direction=direction, values=values, pass_indices=None
    )
    return RunRecord(
        run_id=run_id,
        run_dir=Path("/nonexistent") / run_id,
        flavor="config-matrix",
        commit=commit,
        created_at=created_at,
        slurm_job_id=None,
        workload="synthetic",
        cells={
            cell: CellRecord(
                name=cell, arms={arm: {metric: series}}, config=dict(config or {})
            )
        },
    )


# ---------------------------------------------------------------------------
# 7. Commit-less runs are excluded; ordering is topological, not chronological.
# ---------------------------------------------------------------------------


def test_runs_without_a_commit_are_excluded_and_named() -> None:
    """A provenance-less run has no place in history and must not be given one.

    Attributing it to HEAD would date it to whenever the index happened to be
    built, which fabricates the history the tracker exists to report.
    """
    ordering = FakeOrdering({"aaa": 0, "bbb": 1})
    runs = [
        make_run("older", (100.0, 100.0), commit="aaa"),
        make_run("provenance-less", (10.0, 10.0), commit=None),
        make_run("newer", (101.0, 101.0), commit="bbb"),
    ]

    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)

    assert [point.run_id for point in track.points] == ["older", "newer"]
    assert ("provenance-less", "no recorded commit") in track.excluded
    # And it must not have leaked into the comparisons under another name.
    assert all(point.run_id != "provenance-less" for point in track.points)


def test_history_is_ordered_by_git_topology_not_by_wall_clock_date() -> None:
    """Runs are relaunched out of order, so ``created_at`` is not the history.

    Here the run measuring the OLDEST commit was written LAST.  A date sort
    would report a 20% improvement; the topological sort reports the 20%
    regression that actually happened.
    """
    ordering = FakeOrdering({"old": 0, "mid": 1, "new": 2})
    runs = [
        # created_at deliberately anti-correlated with commit order.
        make_run("run-new-code", (80.0, 81.0, 79.0), commit="new", created_at="2026-08-01T00:00"),
        make_run("run-mid-code", (90.0, 91.0, 89.0), commit="mid", created_at="2026-08-02T00:00"),
        make_run("run-old-code", (100.0, 101.0, 99.0), commit="old", created_at="2026-08-03T00:00"),
    ]

    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)

    assert [point.run_id for point in track.points] == [
        "run-old-code",
        "run-mid-code",
        "run-new-code",
    ]
    assert [point.commit for point in track.points] == ["old", "mid", "new"]
    # Every step is downhill, which is only true in the topological order.
    assert all(step.delta < 0.0 for step in track.steps)
    assert len(track.steps) == 2


def test_a_commit_absent_from_the_repository_is_excluded_not_appended() -> None:
    ordering = FakeOrdering({"aaa": 0})
    runs = [
        make_run("known", (100.0, 100.0), commit="aaa"),
        make_run("stranger", (5.0, 5.0), commit="f" * 40),
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)
    assert [point.run_id for point in track.points] == ["known"]
    assert any("not in this repository" in reason for _, reason in track.excluded)


def test_a_step_across_a_branch_fork_is_flagged() -> None:
    """`git merge-base --is-ancestor` says these two runs are not sequential."""
    ordering = FakeOrdering({"aaa": 0, "bbb": 1}, forks={("aaa", "bbb")})
    runs = [
        make_run("a", (100.0, 101.0, 99.0), commit="aaa"),
        make_run("b", (80.0, 81.0, 79.0), commit="bbb"),
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)
    assert len(track.caveats) == 1
    assert "diverging branches" in track.caveats[0]
    assert any("diverging branches" in note for note in track.steps[0].notes)


def test_git_backed_ordering_agrees_with_the_repository() -> None:
    """The real implementation, checked against commits it resolves itself."""
    revs = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-list", "-n", "3", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revs.returncode != 0 or len(revs.stdout.split()) < 3:
        pytest.skip("no git history available")
    newest, middle, oldest = revs.stdout.split()[:3]

    ordering = GitCommitOrdering(REPO_ROOT)
    positions = [ordering.position(sha) for sha in (oldest, middle, newest)]
    assert all(p is not None for p in positions)
    assert positions == sorted(positions), "rev-list --reverse must place parents first"
    assert ordering.is_ancestor(oldest, newest) is True
    assert ordering.is_ancestor(newest, oldest) is False
    assert ordering.position("0" * 40) is None
    # Abbreviated shas must resolve, since provenance files record short ones.
    assert ordering.position(newest[:9]) == ordering.position(newest)


# ---------------------------------------------------------------------------
# Step verdicts, inherited wholesale from the comparator.
# ---------------------------------------------------------------------------


def test_a_regression_between_two_commits_is_proven_and_attributed() -> None:
    ordering = FakeOrdering({"good": 0, "bad": 1})
    runs = [
        make_run("before", (100.0, 101.0, 99.0, 100.0), commit="good"),
        make_run("after", (80.0, 80.8, 79.2, 80.0), commit="bad"),
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)

    assert len(track.regressions) == 1
    assert track.regressions[0].verdict is ComparisonVerdict.PROVEN_REGRESSION
    assert track.regressions[0].delta_pct == pytest.approx(-20.0, abs=1e-6)


def test_detect_regressions_surfaces_unresolved_shifts_alongside_regressions() -> None:
    """An unresolved material shift must not be buried under 'no regressions'."""
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    noisy = (80.0, 120.0, 90.0, 110.0, 100.0)
    runs = [
        make_run("before", noisy, commit="c0"),
        make_run("after", tuple(v - 5.0 for v in noisy), commit="c1"),
    ]
    report = detect_regressions(runs, ordering=ordering)

    assert report.regressions == ()
    assert len(report.unresolved_shifts) == 1
    assert report.unresolved_shifts[0].verdict is ComparisonVerdict.MATERIAL_SHIFT_UNRESOLVED


def test_detect_regressions_covers_every_cell_arm_and_metric() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})

    def multi(run_id: str, scale: float, commit: str) -> RunRecord:
        base = (100.0, 101.0, 99.0, 100.0)
        return RunRecord(
            run_id=run_id,
            run_dir=Path("/nonexistent") / run_id,
            flavor="f",
            commit=commit,
            created_at=None,
            slurm_job_id=None,
            workload=None,
            cells={
                "cellA": CellRecord(
                    name="cellA",
                    arms={
                        "stock": {
                            "tokens_per_second": MetricSeries(
                                name="tokens_per_second",
                                unit="tok/s",
                                direction=Direction.HIGHER_IS_BETTER,
                                values=base,
                            )
                        },
                        "candidate": {
                            "tokens_per_second": MetricSeries(
                                name="tokens_per_second",
                                unit="tok/s",
                                direction=Direction.HIGHER_IS_BETTER,
                                values=tuple(v * scale for v in base),
                            )
                        },
                    },
                )
            },
        )

    report = detect_regressions([multi("a", 1.0, "c0"), multi("b", 0.8, "c1")], ordering=ordering)
    assert {(t.cell, t.arm, t.metric) for t in report.tracks} == {
        ("cellA", "stock", "tokens_per_second"),
        ("cellA", "candidate", "tokens_per_second"),
    }
    # Only the candidate arm moved; the stock arm is unchanged and so is not a
    # regression.  A tracker that collapsed arms would report both or neither.
    assert {c.arm for c in report.regressions} == {"candidate"}


def test_detect_regressions_reports_provenance_less_runs_once() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    runs = [
        make_run("a", (100.0, 101.0), commit="c0"),
        make_run("orphan", (100.0, 101.0), commit=None),
        make_run("b", (100.0, 101.0), commit="c1"),
    ]
    report = detect_regressions(runs, ordering=ordering)
    assert report.excluded == (("orphan", "no recorded commit"),)


def test_a_configuration_change_between_commits_is_flagged_as_a_caveat() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    runs = [
        make_run("a", (100.0, 101.0, 99.0), commit="c0", config={"concurrency": 1}),
        make_run("b", (60.0, 61.0, 59.0), commit="c1", config={"concurrency": 16}),
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)
    assert len(track.caveats) == 1
    assert "concurrency" in track.caveats[0]


def test_runs_missing_the_tracked_metric_are_excluded_with_a_reason() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    runs = [
        make_run("a", (100.0, 101.0), commit="c0"),
        make_run("b", (100.0, 101.0), commit="c1", metric="something_else"),
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)
    assert [point.run_id for point in track.points] == ["a"]
    assert any(METRIC in reason for _, reason in track.excluded)


# ---------------------------------------------------------------------------
# Trend classification, delegated to the gate's OLS helpers.
# ---------------------------------------------------------------------------


def test_a_two_point_history_is_untestable_not_perfectly_trending() -> None:
    """Two points fit their own slope exactly, leaving no residual to test."""
    trend = summarize_trend([100.0, 80.0])
    assert trend.status is TrendStatus.UNTESTABLE
    assert trend.slope_t_statistic is None


def test_a_steady_decline_is_reported_as_trending() -> None:
    trend = summarize_trend([100.0, 95.0, 90.0, 85.0, 80.0])
    assert trend.status is TrendStatus.TRENDING
    assert trend.slope_t_statistic is not None and trend.slope_t_statistic > 1.0


def test_a_noisy_history_with_no_real_slope_is_reported_as_flat() -> None:
    trend = summarize_trend([100.0, 80.0, 120.0, 90.0, 110.0])
    assert trend.status is TrendStatus.FLAT
    assert trend.slope_t_statistic is not None and trend.slope_t_statistic < 1.0


def test_track_reports_the_trend_of_its_own_history() -> None:
    ordering = FakeOrdering({f"c{i}": i for i in range(5)})
    runs = [
        make_run(f"r{i}", (100.0 - 5 * i, 101.0 - 5 * i, 99.0 - 5 * i), commit=f"c{i}")
        for i in range(5)
    ]
    track = track_metric(runs, CELL, ARM, METRIC, ordering=ordering)
    assert track.trend.n == 5
    assert track.trend.status is TrendStatus.TRENDING


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------


def test_render_track_shows_every_run_the_verdicts_and_the_exclusions() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    runs = [
        make_run("before", (100.0, 101.0, 99.0, 100.0), commit="c0"),
        make_run("after", (80.0, 80.8, 79.2, 80.0), commit="c1"),
        make_run("orphan", (1.0, 2.0), commit=None),
    ]
    text = render_track(track_metric(runs, CELL, ARM, METRIC, ordering=ordering))

    assert "before" in text and "after" in text
    assert "proven_regression" in text
    assert "+/-" in text
    assert "orphan: no recorded commit" in text


def test_render_track_marks_a_missing_standard_error() -> None:
    ordering = FakeOrdering({"c0": 0, "c1": 1})
    runs = [
        make_run("a", (100.0, 100.0, 100.0), commit="c0"),
        make_run("b", (100.0, 100.0, 100.0), commit="c1"),
    ]
    text = render_track(track_metric(runs, CELL, ARM, METRIC, ordering=ordering))
    assert "no SE (degenerate)" in text
