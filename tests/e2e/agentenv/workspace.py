"""A sandboxed workspace and the tools that really operate on it.

Every tool here performs the action it advertises.  There is no fixture layer
and no canned response: :meth:`WorkspaceSandbox.execute` reaches the filesystem,
and ``run_tests`` starts a real pytest subprocess in the workspace.  A model that
calls ``read_file`` with a path that does not exist receives the same message a
human would, and the recovery turn that follows is real traffic rather than a
scripted one.

Containment is by construction, not by convention:

* every path argument is resolved and then required to lie inside the workspace
  root *after* resolution, so ``../`` and symlinks out are both refused;
* reads, writes and captured subprocess output are byte-capped, so a model that
  asks for a 2 GB file gets a refusal instead of the harness getting an OOM;
* the only subprocess that can be started is the workspace's own pytest, with a
  wall-clock timeout, an empty-ish environment and its own temporary directory.

The caps are deliberately generous enough that a competent agent never meets
them.  They exist so an incompetent one cannot take the allocation down.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MAX_MATCHES",
    "MAX_READ_BYTES",
    "MAX_TOOL_OUTPUT_CHARS",
    "MAX_WRITE_BYTES",
    "ToolError",
    "ToolResult",
    "WorkspaceSandbox",
]

#: Largest file the ``read_file`` tool will return.  Above this the model is
#: told the size and asked to narrow with ``start_line``/``max_lines``, which is
#: what a real coding agent does with a large file anyway.
MAX_READ_BYTES: Final[int] = 256 * 1024

#: Largest body ``write_file`` accepts.  A model that tries to write more than
#: this has lost the plot, and the write would be the thing that fills the disk.
MAX_WRITE_BYTES: Final[int] = 256 * 1024

#: Ceiling on the characters any single tool result puts back into the prompt.
#: This is a *context* budget rather than a safety one: a 40k-character pytest
#: traceback would evict the task description from a 16k window and turn the
#: rest of the trajectory into noise.
MAX_TOOL_OUTPUT_CHARS: Final[int] = 8_000

#: Ceiling on ``search`` hits returned.  A regex like ``.`` must not be able to
#: return the whole workspace.
MAX_MATCHES: Final[int] = 200

#: Files ``search`` and ``list_dir`` never surface.  Caches and VCS metadata are
#: not part of the task and they dominate the listing when present.
_HIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"}
)


class ToolError(Exception):
    """A tool refused or failed.

    Raised for the *ordinary* failures an agent is expected to recover from --
    a missing path, a non-unique edit anchor, a bad regex.  The loop turns it
    into a tool message and hands it back to the model, because that recovery
    turn is the traffic this package exists to produce.  It is deliberately NOT
    used for harness bugs: those propagate and fail the run.
    """


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What one executed tool call produced.

    ``text`` is what goes back to the model.  ``truncated`` and ``elapsed`` are
    for the trajectory record only: a run whose tool results were mostly
    truncated measured a different workload than one whose were not, and that
    has to be recoverable after the fact rather than guessed at.
    """

    text: str
    truncated: bool = False
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] | None = None


def _clip(text: str) -> ToolResult:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return ToolResult(text=text)
    kept = MAX_TOOL_OUTPUT_CHARS - 120
    head = text[: kept // 2]
    tail = text[-(kept // 2) :]
    joined = (
        f"{head}\n\n... [{len(text) - kept} characters omitted from the middle; "
        f"the tool produced {len(text)} characters and the limit is "
        f"{MAX_TOOL_OUTPUT_CHARS}] ...\n\n{tail}"
    )
    return ToolResult(text=joined, truncated=True)


class WorkspaceSandbox:
    """Executes tool calls against one workspace directory.

    One instance per task instance.  It holds no state beyond the root and the
    pytest settings, so a trajectory can be replayed against a freshly
    materialized workspace and produce the same tool results.
    """

    def __init__(
        self,
        root: Path,
        *,
        python: Path | None = None,
        test_timeout_seconds: float = 120.0,
    ) -> None:
        resolved = Path(root).resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace root is not a directory: {resolved}")
        self.root = resolved
        self.python = Path(python) if python is not None else Path(sys.executable)
        self.test_timeout_seconds = float(test_timeout_seconds)

    # -- path containment -------------------------------------------------
    def resolve(self, raw: object, *, argument: str = "path") -> Path:
        """Resolve a model-supplied path, or refuse it.

        Resolution happens first and containment is checked on the *result*, so
        ``a/../../etc/passwd`` and a symlink pointing outside are both caught by
        the same test.  Checking the raw string for ``..`` instead -- the
        obvious implementation -- passes a symlink straight through.
        """
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError(f"{argument} must be a non-empty string")
        candidate = Path(raw)
        if candidate.is_absolute():
            # An absolute path inside the workspace is legal but a model that
            # produced one is usually echoing a path from an error message; keep
            # it working rather than making it a puzzle.
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ToolError(
                f"{argument} {raw!r} resolves to {resolved}, which is outside the "
                f"workspace. Use workspace-relative paths."
            ) from None
        return resolved

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    # -- the tools --------------------------------------------------------
    def list_dir(self, arguments: Mapping[str, Any]) -> ToolResult:
        target = self.resolve(arguments.get("path", "."))
        if not target.exists():
            raise ToolError(f"no such directory: {arguments.get('path')!r}")
        if not target.is_dir():
            raise ToolError(f"{arguments.get('path')!r} is a file, not a directory")
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name in _HIDDEN_NAMES:
                continue
            if child.is_dir():
                entries.append(f"{self.relative(child)}/")
            else:
                entries.append(f"{self.relative(child)}  ({child.stat().st_size} bytes)")
        if not entries:
            return ToolResult(text=f"{self.relative(target) or '.'} is empty")
        return _clip("\n".join(entries))

    def read_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        target = self.resolve(arguments.get("path"))
        if not target.exists():
            raise ToolError(f"no such file: {arguments.get('path')!r}")
        if target.is_dir():
            raise ToolError(f"{arguments.get('path')!r} is a directory; use list_dir")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(
                f"{arguments.get('path')!r} is {size} bytes, above the "
                f"{MAX_READ_BYTES}-byte read limit; pass start_line and max_lines "
                "to read part of it"
            )
        try:
            body = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{arguments.get('path')!r} is not UTF-8 text") from None

        lines = body.splitlines()
        start = _positive_int(arguments.get("start_line"), "start_line", default=1)
        limit = _positive_int(arguments.get("max_lines"), "max_lines", default=len(lines) or 1)
        if start > len(lines) and lines:
            raise ToolError(
                f"start_line {start} is past the end of {arguments.get('path')!r}, "
                f"which has {len(lines)} lines"
            )
        window = lines[start - 1 : start - 1 + limit]
        # Line numbers are supplied because every edit anchor the model will
        # later produce has to be quoted exactly; numbering makes the quoting
        # verifiable to it rather than a guess.
        numbered = "\n".join(f"{start + offset}\t{line}" for offset, line in enumerate(window))
        return _clip(numbered if numbered else "(empty file)")

    def search(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_pattern = arguments.get("pattern")
        if not isinstance(raw_pattern, str) or not raw_pattern:
            raise ToolError("pattern must be a non-empty string")
        try:
            pattern = re.compile(raw_pattern)
        except re.error as error:
            raise ToolError(f"pattern is not a valid regular expression: {error}") from None
        root = self.resolve(arguments.get("path", "."))
        if not root.exists():
            raise ToolError(f"no such path: {arguments.get('path')!r}")

        hits: list[str] = []
        truncated = False
        for file_path in _walk_files(root if root.is_dir() else root.parent):
            if root.is_file() and file_path != root:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{self.relative(file_path)}:{number}:{line.strip()[:200]}")
                    if len(hits) >= MAX_MATCHES:
                        truncated = True
                        break
            if truncated:
                break
        if not hits:
            return ToolResult(text=f"no matches for {raw_pattern!r}")
        body = "\n".join(hits)
        if truncated:
            body += f"\n... stopped at {MAX_MATCHES} matches; narrow the pattern or the path"
        result = _clip(body)
        return ToolResult(
            text=result.text,
            truncated=result.truncated or truncated,
        )

    def write_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        target = self.resolve(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError(
                f"content is {len(encoded)} bytes, above the {MAX_WRITE_BYTES}-byte "
                "write limit"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        verb = "overwrote" if existed else "created"
        return ToolResult(
            text=f"{verb} {self.relative(target)} ({len(encoded)} bytes, "
            f"{len(content.splitlines())} lines)"
        )

    def replace_in_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Exact-string replacement, refused unless the anchor is unique.

        Uniqueness is the whole contract.  An edit tool that replaces the first
        match silently edits the wrong call site in any file that repeats a
        line, and the resulting failure surfaces several turns later as a
        confusing test error -- which is a realistic agent experience but an
        unusable one for grading, because the trajectory then measures the
        tool's ambiguity rather than the model's reasoning.
        """
        target = self.resolve(arguments.get("path"))
        if not target.is_file():
            raise ToolError(f"no such file: {arguments.get('path')!r}")
        find = arguments.get("find")
        replace = arguments.get("replace")
        if not isinstance(find, str) or not find:
            raise ToolError("find must be a non-empty string")
        if not isinstance(replace, str):
            raise ToolError("replace must be a string")
        body = target.read_text(encoding="utf-8")
        occurrences = body.count(find)
        if occurrences == 0:
            raise ToolError(
                f"find text does not occur in {self.relative(target)}; read the file "
                "and quote the exact text including indentation"
            )
        if occurrences > 1:
            raise ToolError(
                f"find text occurs {occurrences} times in {self.relative(target)}; "
                "include surrounding lines so the anchor is unique"
            )
        target.write_text(body.replace(find, replace), encoding="utf-8")
        return ToolResult(
            text=f"replaced 1 occurrence in {self.relative(target)}",
        )

    def run_tests(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Run the workspace's own pytest and return what it printed.

        The exit status is reported in words rather than as a number, because
        the number is the thing a model most often hallucinates agreement with.
        """
        raw_target = arguments.get("path")
        target_args: list[str] = []
        if raw_target is not None:
            target_args = [self.relative(self.resolve(raw_target))]
        completed, elapsed = self._pytest(target_args)
        status = "PASSED" if completed.returncode == 0 else "FAILED"
        body = (
            f"pytest {' '.join(target_args) or '(whole workspace)'} -> {status}\n\n"
            f"{completed.stdout}"
        )
        result = _clip(body)
        return ToolResult(
            text=result.text,
            truncated=result.truncated,
            elapsed_seconds=elapsed,
            metadata={"returncode": completed.returncode},
        )

    # -- the pytest subprocess -------------------------------------------
    def _pytest(self, target_args: Sequence[str]) -> tuple[subprocess.CompletedProcess[str], float]:
        import time

        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "TMPDIR": str(self.root / ".tmp"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(self.root),
            # A task's tests must not reach the network; nothing in the catalog
            # needs it and a hung DNS lookup would burn the wall clock.
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        (self.root / ".tmp").mkdir(exist_ok=True)
        command = [str(self.python), "-m", "pytest", "-q", "-p", "no:cacheprovider", *target_args]
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, sandboxed cwd
                command,
                cwd=self.root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.test_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            elapsed = time.monotonic() - started
            stdout = expired.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            return (
                subprocess.CompletedProcess(
                    command,
                    returncode=124,
                    stdout=(
                        f"{stdout}\n\npytest exceeded the "
                        f"{self.test_timeout_seconds:g}s time limit and was killed"
                    ),
                    stderr="",
                ),
                elapsed,
            )
        elapsed = time.monotonic() - started
        merged = completed.stdout + (
            f"\n--- stderr ---\n{completed.stderr}" if completed.stderr.strip() else ""
        )
        if "No module named pytest" in merged:
            # This is a harness misconfiguration wearing a task failure's
            # clothes, and it is worth its own exception because of how it
            # presents: every pytest-backed family reports "not solved by its
            # own reference solution", which reads as a broken grader or an
            # impossible task. The usual cause is passing a RESOLVED
            # `.venv/bin/python`: that symlink points at the bare uv
            # interpreter, and resolving it leaves the venv (and pytest)
            # behind. Pass the venv path unresolved.
            raise RuntimeError(
                f"the interpreter {self.python} cannot import pytest, so no "
                "workspace test can run and every pytest-graded task would be "
                "recorded as unsolvable. If this path came from resolving a "
                "symlink such as .venv/bin/python, pass it unresolved instead."
            )
        return (
            subprocess.CompletedProcess(command, completed.returncode, merged, ""),
            elapsed,
        )

    def tests_pass(self, target: str | None = None) -> tuple[bool, str]:
        """Grading entry point: did the workspace's tests pass?

        Separate from :meth:`run_tests` so grading never goes through the
        model-facing clipping and formatting.  A grader that read the clipped
        text could be fooled by a task whose output happened to end in
        ``PASSED``.
        """
        target_args = [self.relative(self.resolve(target))] if target is not None else []
        completed, _ = self._pytest(target_args)
        return completed.returncode == 0, completed.stdout

    # -- dispatch ---------------------------------------------------------
    #: Tool name to bound method.  ``submit`` is absent on purpose: it carries
    #: no side effect, so the loop handles it rather than the sandbox.
    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        handler = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "search": self.search,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "run_tests": self.run_tests,
        }.get(name)
        if handler is None:
            raise ToolError(
                f"no tool named {name!r}; available: list_dir, read_file, search, "
                "write_file, replace_in_file, run_tests, submit"
            )
        return handler(arguments)


def _positive_int(value: object, name: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        # Models routinely send "12" rather than 12; accepting the string is not
        # leniency for its own sake, it removes a turn of pure formatting
        # recovery that tells us nothing about drafting.
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        else:
            raise ToolError(f"{name} must be a positive integer")
    if value < 1:
        raise ToolError(f"{name} must be a positive integer, got {value}")
    return int(value)


def _walk_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _HIDDEN_NAMES)
        for filename in sorted(filenames):
            found.append(Path(dirpath) / filename)
    return found
