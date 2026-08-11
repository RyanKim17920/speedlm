"""Anti-vacuity tests for every executable task family in the catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.e2e.agentenv import catalog
from tests.e2e.agentenv.tasks import (
    Grade,
    Task,
    TaskInstance,
    TaskUnsolvableError,
    Workspace,
    check_task_is_solvable_and_unsolved,
)
from tests.e2e.agentenv.workspace import WorkspaceSandbox

PROJECT_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"

# Three seeds cover every deterministic variant without making the ordinary
# suite expensive.  This matrix launches about 21 tiny real pytest subprocesses
# and took roughly 30 seconds in the project venv during focused verification.
CATALOG_CASES = [(task, seed) for task in catalog.TASKS for seed in range(3)]


@pytest.mark.parametrize(
    ("task", "seed"),
    CATALOG_CASES,
    ids=[f"{task.name}-seed-{seed}" for task, seed in CATALOG_CASES],
)
def test_every_catalog_instance_is_unsolved_then_solved_by_its_reference(
    task: Task, seed: int, tmp_path: Path
) -> None:
    """Every seeded task must make both halves of its preflight observable.

    The helper grades the untouched real workspace first and the reference edit
    second.  A vacuous baseline or a stale grader makes this matrix go red for
    the exact family and seed instead of allowing a decorative green check.
    """
    instance = task.instance(seed)

    check_task_is_solvable_and_unsolved(
        instance,
        tmp_path / instance.id,
        solution=catalog.CATALOG[task.name](seed),
        python=PROJECT_PYTHON,
    )


def test_solvability_preflight_rejects_a_no_op_solution(tmp_path: Path) -> None:
    """The preflight itself must demonstrably fail when the fix does nothing.

    The read-only trace family is a cheap, unambiguous witness: its untouched
    workspace has no ``answer.txt``, and a no-op cannot satisfy that grader.
    Removing the post-solution check makes this test report a false green.
    """
    instance = catalog.task_by_name("call-chain-trace").instance(0)

    with pytest.raises(TaskUnsolvableError, match="not solved by its own reference solution"):
        check_task_is_solvable_and_unsolved(
            instance,
            tmp_path / instance.id,
            solution=lambda _sandbox: None,
            python=PROJECT_PYTHON,
        )


def test_solvability_preflight_rejects_a_task_that_is_already_solved(tmp_path: Path) -> None:
    """The preflight must also fail on the baseline half, not only on the fix.

    This is the anti-vacuity rule the whole package is built around: a grader
    that is green on an untouched workspace measures nothing, and every model
    would "pass" it.  The reference solution here is a no-op, so only the
    baseline check can raise.  Delete that check and the preflight happily
    certifies a task that grades itself solved before the agent has run --
    exactly the green-but-meaningless shape this suite exists to catch.
    """
    instance = TaskInstance(
        id="always-solved-0000",
        family="always-solved",
        instruction="There is nothing to do; the grader never looks at the workspace.",
        workspace=Workspace(files={"note.txt": "untouched\n"}),
        grader=lambda _sandbox: Grade(True, "this grader ignores the workspace"),
    )

    with pytest.raises(TaskUnsolvableError, match="grades as solved on the untouched workspace"):
        check_task_is_solvable_and_unsolved(
            instance,
            tmp_path / instance.id,
            solution=lambda _sandbox: None,
            python=PROJECT_PYTHON,
        )


# ---------------------------------------------------------------------------
# Per-family negative graders
#
# The preflight matrix above only observes two endpoints: untouched -> not
# solved, reference edit -> solved.  Every cheat that lands between those two
# points is invisible to it, and each family's grader carries exactly one check
# whose job is to refuse such a cheat.  These tests drive the cheat and assert
# the refusal, so deleting that one check turns a specific test red instead of
# leaving the whole suite green.
# ---------------------------------------------------------------------------
_TRIVIALLY_GREEN_TEST = "def test_nothing() -> None:\n    assert True\n"


def _grade_after(
    instance: TaskInstance, root: Path, cheat: Callable[[WorkspaceSandbox], None]
) -> Grade:
    """Materialize the real workspace, apply ``cheat``, and grade what is left."""
    sandbox = instance.materialize(root, python=PROJECT_PYTHON)
    cheat(sandbox)
    return instance.grade(sandbox)


def test_bugfix_grader_refuses_a_neutered_test_suite(tmp_path: Path) -> None:
    """Deleting the failing assertions must not count as fixing the bug.

    Overwriting ``tests/test_pipeline.py`` with a test that cannot fail makes
    pytest exit 0 while the planted defect is still in the pipeline source, so
    ``pytest`` is asserted True here on purpose: it proves the cheat really did
    buy a green suite, and that only ``tests_untouched`` stands between that
    green suite and a solved verdict.  Drop ``tests_untouched`` from the grader
    and this task scores a model that deleted the evidence as having fixed it.
    """
    instance = catalog.task_by_name("bugfix-localize").instance(0)

    def neuter_the_tests(sandbox: WorkspaceSandbox) -> None:
        (sandbox.root / "tests" / "test_pipeline.py").write_text(
            _TRIVIALLY_GREEN_TEST, encoding="utf-8"
        )

    grade = _grade_after(instance, tmp_path / instance.id, neuter_the_tests)

    assert grade.checks["pytest"] is True
    assert grade.checks["tests_untouched"] is False
    assert grade.solved is False


def test_feature_grader_refuses_a_rewritten_specification(tmp_path: Path) -> None:
    """Rewriting the spec tests must not count as implementing the feature.

    The cheat creates an empty ``pipeline/rolling.py`` and replaces the spec
    test with one that cannot fail.  Both of the other checks -- ``pytest`` and
    ``module_created`` -- are therefore True, which isolates ``spec_untouched``
    as the single check standing between "the file exists and the suite is
    green" and a solved verdict.  Without it the grader rewards a model that
    edited the specification to match an empty module.
    """
    instance = catalog.task_by_name("feature-implement").instance(0)

    def rewrite_the_spec(sandbox: WorkspaceSandbox) -> None:
        (sandbox.root / "tests" / "test_rolling.py").write_text(
            _TRIVIALLY_GREEN_TEST, encoding="utf-8"
        )
        (sandbox.root / "pipeline" / "rolling.py").write_text("", encoding="utf-8")

    grade = _grade_after(instance, tmp_path / instance.id, rewrite_the_spec)

    assert grade.checks["pytest"] is True
    assert grade.checks["module_created"] is True
    assert grade.checks["spec_untouched"] is False
    assert grade.solved is False


def test_refactor_grader_refuses_a_workspace_nobody_touched(tmp_path: Path) -> None:
    """Doing nothing must not count as completing the rename.

    A refactor task starts with a green suite -- that is the point of a refactor
    -- so ``pytest`` alone is satisfied by an agent that made no edit at all.
    The cheat is literally the empty edit, and ``old_name_gone`` is the only
    check that can fail on it.  Remove that check and this family grades every
    idle trajectory as solved, which is the vacuous baseline the preflight
    cannot see because the preflight never runs the idle case.
    """
    instance = catalog.task_by_name("refactor-rename").instance(0)

    grade = _grade_after(instance, tmp_path / instance.id, lambda _sandbox: None)

    assert grade.checks["pytest"] is True
    assert grade.checks["old_name_gone"] is False
    assert grade.solved is False


def test_log_grader_refuses_the_right_keys_with_the_wrong_values(tmp_path: Path) -> None:
    """Producing the requested shape must not count as finding the ERROR line.

    ``findings.json`` here is a well-formed JSON object with exactly the three
    keys the instruction names, so ``file_present``, ``json_valid``,
    ``json_object`` and ``no_extra_keys`` are all True -- the document a model
    could write from the instruction alone, without ever reading the log.  Only
    the per-field value comparisons can fail on it.  A grader that checked the
    key set and stopped would score that document as a solved triage.
    """
    instance = catalog.task_by_name("log-triage").instance(0)

    def guess_the_shape(sandbox: WorkspaceSandbox) -> None:
        (sandbox.root / "findings.json").write_text(
            json.dumps(
                {
                    "service": "not-the-service",
                    "request_id": "not-the-request-id",
                    "error_code": "NOT_THE_ERROR_CODE",
                }
            ),
            encoding="utf-8",
        )

    grade = _grade_after(instance, tmp_path / instance.id, guess_the_shape)

    assert grade.checks["no_extra_keys"] is True
    assert grade.checks["file_present"] is True
    assert grade.checks["json_valid"] is True
    assert grade.checks["json_object"] is True
    assert grade.checks["service"] is False
    assert grade.checks["request_id"] is False
    assert grade.checks["error_code"] is False
    assert grade.solved is False
