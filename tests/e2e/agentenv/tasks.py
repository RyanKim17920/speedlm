"""Task, workspace and grading types shared by every environment.

The type that matters here is :class:`Grade`.  Grading is the only thing
standing between "we ran an agent loop" and "we ran an agent loop that did
something", and this project's recurring defect is a check that cannot fail, so
the contract is narrow on purpose:

* a grader receives the :class:`~tests.e2e.agentenv.workspace.WorkspaceSandbox`
  and nothing else.  It cannot see the transcript, so it cannot be satisfied by
  anything the model *said*;
* a grader must record a ``baseline`` -- what the same check returns on the
  freshly materialized workspace, before the agent touches it.  A task whose
  grader already passes at turn zero is not a task, and
  :func:`check_task_is_solvable_and_unsolved` refuses it.

That second rule is the anti-vacuity rule from
``tests/e2e/harness/workloads.py`` applied to environments: a green grader on an
untouched workspace is the data-shaped version of a test that measures nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from tests.e2e.agentenv.workspace import WorkspaceSandbox

__all__ = [
    "AGENT_TOOLS",
    "Grade",
    "SYSTEM_PROMPT",
    "Task",
    "TaskInstance",
    "TaskUnsolvableError",
    "ToolSpec",
    "Workspace",
    "check_task_is_solvable_and_unsolved",
    "materialize",
]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One OpenAI function tool, kept as a typed object rather than raw JSON.

    The wire form is produced by :meth:`schema`.  Holding the pieces separately
    means the loop can assert that every tool the model called was one this task
    declared, which a bare list of dicts makes awkward enough that it gets
    skipped.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def _object(properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


#: The tool surface every environment in this package exposes.
#:
#: It is deliberately the small set a real coding agent actually leans on.  A
#: larger surface would make the tool-schema prefix longer -- which flatters the
#: prompt-length statistics -- without making any trajectory more realistic, and
#: this project has already been burned once by a benchmark that measured its
#: own padding.
AGENT_TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory in the workspace. Returns "
            "workspace-relative paths; directories end with a slash and files "
            "carry their size."
        ),
        parameters=_object(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory. Defaults to the root.",
                }
            },
            [],
        ),
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read a UTF-8 text file. Output is line-numbered so that edit "
            "anchors can be quoted exactly. Use start_line and max_lines to "
            "read part of a large file."
        ),
        parameters=_object(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "start_line": {
                    "type": "integer",
                    "description": "First line to return, 1-based.",
                },
                "max_lines": {"type": "integer", "description": "How many lines to return."},
            },
            ["path"],
        ),
    ),
    ToolSpec(
        name="search",
        description=(
            "Search the workspace for a Python regular expression, one result "
            "per matching line as path:line:text."
        ),
        parameters=_object(
            {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory to search under.",
                },
            },
            ["pattern"],
        ),
    ),
    ToolSpec(
        name="write_file",
        description=(
            "Write a complete UTF-8 file, creating parent directories and "
            "overwriting any existing content. Use replace_in_file for small "
            "edits to an existing file."
        ),
        parameters=_object(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "The complete new file content."},
            },
            ["path", "content"],
        ),
    ),
    ToolSpec(
        name="replace_in_file",
        description=(
            "Replace one exact occurrence of a string in a file. The find text "
            "must appear exactly once, including indentation; quote enough "
            "surrounding lines to make it unique."
        ),
        parameters=_object(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "find": {"type": "string", "description": "Exact text to replace, once."},
                "replace": {"type": "string", "description": "Replacement text."},
            },
            ["path", "find", "replace"],
        ),
    ),
    ToolSpec(
        name="run_tests",
        description=(
            "Run the workspace test suite with pytest and return its output. "
            "Pass a path to run a subset."
        ),
        parameters=_object(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative test file or directory.",
                }
            },
            [],
        ),
    ),
    ToolSpec(
        name="submit",
        description=(
            "Declare the task finished. Call this only after verifying the work; "
            "the result is judged by inspecting the workspace, not by this "
            "message."
        ),
        parameters=_object(
            {
                "summary": {
                    "type": "string",
                    "description": "One or two sentences on what was changed and why.",
                }
            },
            ["summary"],
        ),
    ),
)

#: Naming the grading rule in the system prompt is not a hint, it is the
#: difference between an agent that verifies and one that narrates.  Real coding
#: agents are told this too, so omitting it would make the traffic less
#: realistic rather than more honest.
SYSTEM_PROMPT: Final[str] = """
You are a software engineering agent working inside a sandboxed workspace.

Work only through the provided tools. Explore before you edit: read the files
you are about to change, and quote edit anchors exactly as read_file printed
them. After making a change, run the tests to confirm it. When the work is done
and verified, call submit.

Your result is judged by inspecting the workspace, never by what you say about
it. Describing a change you did not make counts as failure.
""".strip()


@dataclass(frozen=True, slots=True)
class Workspace:
    """The files one task instance starts from.

    A mapping of workspace-relative path to file content.  Held as data rather
    than as a directory template so a task instance is fully described by a JSON
    document, which is what lets the trajectory record reproduce the exact
    starting world months later.
    """

    files: Mapping[str, str]

    def write_to(self, root: Path) -> None:
        for relative, content in self.files.items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"workspace file escapes the root: {relative!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Grade:
    """The verdict on one finished trajectory.

    ``checks`` carries every individual assertion the grader made with its own
    outcome, because "failed" on its own does not distinguish "the agent broke
    the build" from "the agent fixed the bug but deleted the docstring the task
    also asked for", and those are different findings about the same run.
    """

    solved: bool
    detail: str
    checks: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"solved": self.solved, "detail": self.detail, "checks": dict(self.checks)}


#: A grader sees the sandbox and nothing else.  See the module docstring.
Grader = Callable[[WorkspaceSandbox], Grade]


@dataclass(frozen=True, slots=True)
class TaskInstance:
    """One concrete, gradeable piece of work.

    ``instruction`` is the user turn.  It is written the way a real ticket is
    written -- what outcome is wanted, not which files to touch -- because a
    prompt that names the file turns a multi-turn search into a single-turn
    edit, and the search turns are most of what makes agentic traffic long.
    """

    id: str
    family: str
    instruction: str
    workspace: Workspace
    grader: Grader
    #: Free-form notes carried into the trajectory record: which variant seed
    #: produced this instance, which module the bug was planted in, and so on.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, root: Path, *, python: Path | None = None) -> WorkspaceSandbox:
        root.mkdir(parents=True, exist_ok=True)
        self.workspace.write_to(root)
        return WorkspaceSandbox(root, python=python)

    def grade(self, sandbox: WorkspaceSandbox) -> Grade:
        return self.grader(sandbox)


@dataclass(frozen=True, slots=True)
class Task:
    """A family of instances that differ by seed.

    Variation is what stops the corpus from being 600 records of one prompt.
    ``agentic-tool-loop`` has 626 records drawn from 43 session instances and
    16 bug templates, and its own manifest calls that narrowness load-bearing;
    a task here is expected to produce structurally different work per seed --
    a different buggy module, different data, a different call chain -- not the
    same work with the numbers changed.
    """

    name: str
    summary: str
    build: Callable[[int], TaskInstance]

    def instance(self, seed: int) -> TaskInstance:
        return self.build(seed)


class TaskUnsolvableError(AssertionError):
    """A task instance failed its own pre-flight."""


def check_task_is_solvable_and_unsolved(
    instance: TaskInstance,
    root: Path,
    *,
    solution: Callable[[WorkspaceSandbox], None],
    python: Path | None = None,
) -> None:
    """Prove the task is neither already solved nor impossible.

    Both halves are needed and neither is redundant:

    * grading the freshly materialized workspace must return *not solved*.  A
      task that grades green at turn zero measures nothing, and every model
      would "pass" it;
    * applying ``solution`` -- a reference edit written alongside the task --
      must then make grading return *solved*.  Without this, a grader with a
      typo'd path is indistinguishable from a hard task, and the whole family
      would be recorded as "no model can do it".

    Raises :class:`TaskUnsolvableError` naming which half failed.
    """
    sandbox = instance.materialize(root, python=python)
    before = instance.grade(sandbox)
    if before.solved:
        raise TaskUnsolvableError(
            f"task instance {instance.id!r} grades as solved on the untouched "
            f"workspace, so it cannot measure anything: {before.detail}"
        )
    solution(sandbox)
    after = instance.grade(sandbox)
    if not after.solved:
        raise TaskUnsolvableError(
            f"task instance {instance.id!r} is not solved by its own reference "
            f"solution, so the grader or the fixture is wrong: {after.detail}"
        )


def materialize(
    instance: TaskInstance, root: Path, *, python: Path | None = None
) -> WorkspaceSandbox:
    return instance.materialize(root, python=python)
