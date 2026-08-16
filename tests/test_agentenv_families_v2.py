"""Anti-vacuity tests for every task family in families_v2.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agentenv import families_v2
from tests.e2e.agentenv.phrasing import Fact
from tests.e2e.agentenv.tasks import Task, TaskUnsolvableError, check_task_is_solvable_and_unsolved

PROJECT_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"

# Three seeds per family.
V2_CASES = [(task, seed) for task in families_v2.TASKS_V2 for seed in range(3)]


@pytest.mark.parametrize(
    ("task", "seed"),
    V2_CASES,
    ids=[f"{task.name}-seed-{seed}" for task, seed in V2_CASES],
)
def test_every_v2_instance_is_unsolved_then_solved_by_its_reference(
    task: Task, seed: int, tmp_path: Path
) -> None:
    """Every seeded v2 task must make both halves of its preflight observable."""
    instance = task.instance(seed)

    check_task_is_solvable_and_unsolved(
        instance,
        tmp_path / instance.id,
        solution=families_v2.CATALOG_V2[task.name](seed),
        python=PROJECT_PYTHON,
    )


def test_v2_solvability_preflight_rejects_no_op(tmp_path: Path) -> None:
    """The preflight must fail when the fix does nothing."""
    instance = families_v2.task_by_name("perf-hotspot").instance(0)

    with pytest.raises(TaskUnsolvableError, match="not solved by its own reference solution"):
        check_task_is_solvable_and_unsolved(
            instance,
            tmp_path / instance.id,
            solution=lambda _sandbox: None,
            python=PROJECT_PYTHON,
        )


# ---------------------------------------------------------------------------
# Per-family negative graders
# ---------------------------------------------------------------------------


def _grade_after(instance, root: Path, cheat) -> Any:
    """Materialize, apply cheat, and grade."""
    sandbox = instance.materialize(root, python=PROJECT_PYTHON)
    cheat(sandbox)
    return instance.grade(sandbox)


def test_flaky_grader_refuses_no_mutation_fix(tmp_path: Path) -> None:
    """The grader must check for copy/sample, not just test passage."""
    instance = families_v2.task_by_name("flaky-test-quarantine").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


def test_dep_grader_refuses_deleted_shim(tmp_path: Path) -> None:
    """The grader checks tests pass, not that the shim is gone."""
    instance = families_v2.task_by_name("dep-version-conflict").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


def test_api_grader_requires_new_field(tmp_path: Path) -> None:
    """The grader must check the client uses the new field name."""
    instance = families_v2.task_by_name("api-contract-drift").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


def test_perf_grader_checks_operation_count(tmp_path: Path) -> None:
    """The grader requires the O(n) test to pass, not just correctness."""
    instance = families_v2.task_by_name("perf-hotspot").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


def test_cfg_grader_checks_precedence(tmp_path: Path) -> None:
    """The grader requires correct precedence, not just file existence."""
    instance = families_v2.task_by_name("config-precedence-bug").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


def test_err_grader_requires_no_bare_except(tmp_path: Path) -> None:
    """The grader must check for absence of bare except."""
    instance = families_v2.task_by_name("error-swallow-audit").instance(0)
    grade = _grade_after(instance, tmp_path / instance.id, lambda _s: None)
    assert grade.solved is False


# ---------------------------------------------------------------------------
# Guard test: every required_fact must be a typed Fact, never a bare str
# ---------------------------------------------------------------------------


def test_all_briefs_use_typed_facts() -> None:
    """All required_facts entries in every v2 Brief must be Fact instances.

    A bare ``str`` defaults silently to FactKind.VALUE and produces
    nonsense sentence frames for paths and commands (bug 1).
    """
    for task in families_v2.TASKS_V2:
        for seed in range(3):
            # Patch render_instruction in families_v2's namespace to capture
            # the Brief's required_facts before it is rendered.
            captured: list[tuple[str | Fact, ...]] = []

            # families_v2 imports render_instruction by name, so we must
            # patch the name in *that* module's namespace, not in phrasing.
            _orig = families_v2.render_instruction  # type: ignore[attr-defined]

            def _capture(brief, *, seed, salt, _c=captured, _orig=_orig):  # noqa: ANN001
                _c.append(brief.required_facts)
                return _orig(brief, seed=seed, salt=salt)

            families_v2.render_instruction = _capture  # type: ignore[attr-defined]
            try:
                task.build(seed)
            finally:
                families_v2.render_instruction = _orig  # type: ignore[attr-defined]

            assert captured, f"{task.name} seed={seed}: render_instruction never called"
            required_facts = captured[0]
            for i, entry in enumerate(required_facts):
                assert isinstance(entry, Fact), (
                    f"{task.name} seed={seed}: required_facts[{i}]={entry!r} "
                    f"is a bare str, not a Fact instance"
                )
