"""Regression guard: no file that feeds on-screen content may bake in run-specific paths.

Specifically, the following run directory names must not appear in any position
that is consumed directly rather than as a clearly-marked, CLI-overridable default:

  - bigcycle-run1
  - regate-big-run1
  - regate-big-run2
  - demo-video-run2
  - bigcorpus-run1

Acceptable positions (PASS):
  - Python variable names starting with DEFAULT_ that are only used when no CLI
    argument is supplied (i.e., the call site is `args.x or DEFAULT_X`).
  - Code comments and docstrings (checked by AST node type, not regex).

Failing positions (FAIL):
  - Session step `command` fields in any JSON script under demo/ — these run
    verbatim in a real shell and their output appears on screen.
  - A Python constant consumed directly in a function call without an args check
    (e.g., `parse_reproductions(CORROBORATING_DECISIONS)` with no `or args.x`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"

BANNED_RUN_DIRS = [
    "bigcycle-run1",
    "regate-big-run1",
    "regate-big-run2",
    "demo-video-run2",
    "bigcorpus-run1",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains_banned(text: str) -> list[str]:
    return [r for r in BANNED_RUN_DIRS if r in text]


# ---------------------------------------------------------------------------
# Check 1: session script step commands
# ---------------------------------------------------------------------------


def _scan_session_json(path: Path) -> list[str]:
    """Return one violation string per offending step command."""
    with path.open(encoding="utf-8") as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as exc:
            return [f"{path}: invalid JSON — {exc}"]

    steps = doc if isinstance(doc, list) else doc.get("steps", [])

    violations: list[str] = []
    for i, step in enumerate(steps):
        cmd = step.get("command", "") if isinstance(step, dict) else ""
        hits = _contains_banned(cmd)
        for h in hits:
            # Show first 120 chars of the command for context
            snippet = cmd[:120].replace("\n", " ")
            violations.append(
                f"{path.name} step {i}: hardcoded run dir {h!r} in command: {snippet!r}"
            )
    return violations


# ---------------------------------------------------------------------------
# Check 2: Python file — consumed non-default constants
# ---------------------------------------------------------------------------


class _DirectConsumptionVisitor(ast.NodeVisitor):
    """Find calls to parse_reproductions(CORROBORATING_DECISIONS) without CLI guard.

    A guarded call looks like:
        parse_reproductions(args.corroborating or CORROBORATING_DECISIONS)
    or
        corr = args.corroborating if args.corroborating else CORROBORATING_DECISIONS
        parse_reproductions(corr)

    A bare call looks like:
        parse_reproductions(CORROBORATING_DECISIONS)
    which bakes in the hardcoded list regardless of CLI input.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id == "parse_reproductions"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "CORROBORATING_DECISIONS"
        ):
            # Bare CORROBORATING_DECISIONS passed without CLI override
            self.violations.append(
                f"line {node.lineno}: parse_reproductions(CORROBORATING_DECISIONS)"
                " — not CLI-overridable; wrap as"
                " parse_reproductions(args.corroborating or CORROBORATING_DECISIONS)"
            )
        self.generic_visit(node)


def _scan_python_consumed_constants(path: Path) -> list[str]:
    """Return violations in a Python file where run-dir paths feed consumed constants."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error — {exc}"]

    visitor = _DirectConsumptionVisitor()
    visitor.visit(tree)
    return [f"{path.name}: {v}" for v in visitor.violations]


# ---------------------------------------------------------------------------
# Check 3: scan ALL demo Python files for banned run dirs in literal strings
#          that are NOT inside a node whose enclosing assignment name starts
#          with DEFAULT_ and that later appear at a call site guarded by args.
#
# This is deliberately coarser than check 2 (which is exact for the known
# pattern) — it catches future regressions where a new file bakes in a path.
# Only applies to strings that CONTAIN a banned run dir AND that are assigned
# to a name that does NOT start with DEFAULT_.
# ---------------------------------------------------------------------------


class _BannedLiteralVisitor(ast.NodeVisitor):
    """Flag string literals containing a banned run dir in non-DEFAULT_ assignments."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.violations: list[str] = []
        self._current_assign_name: str | None = None

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Track the assignment target name (first target, simple Name only)
        if node.targets and isinstance(node.targets[0], ast.Name):
            prev = self._current_assign_name
            self._current_assign_name = node.targets[0].id
            self.generic_visit(node)
            self._current_assign_name = prev
        else:
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        # Handle type-annotated assignments: name: Type = value
        if isinstance(node.target, ast.Name):
            prev = self._current_assign_name
            self._current_assign_name = node.target.id
            self.generic_visit(node)
            self._current_assign_name = prev
        else:
            self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if not isinstance(node.value, str):
            return
        hits = _contains_banned(node.value)
        if not hits:
            return
        assign_name = self._current_assign_name or ""
        # DEFAULT_* assignments are allowed (they are CLI-fallback defaults)
        if assign_name.startswith("DEFAULT_"):
            return
        # _RUNS is a path-prefix helper, not consumed directly
        if assign_name.startswith("_RUNS") or assign_name == "_RUNS":
            return
        for h in hits:
            self.violations.append(
                f"line {node.lineno}: {h!r} in string literal"
                f" (assigned to {assign_name!r} — not a DEFAULT_* name)"
                if assign_name
                else f"line {node.lineno}: {h!r} in inline string literal"
            )

    # Ignore strings inside docstrings / comments — they are not executed
    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        # A bare string expression is a docstring; skip it
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)


def _scan_python_banned_literals(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error — {exc}"]

    visitor = _BannedLiteralVisitor(path.name)
    visitor.visit(tree)
    return [f"{path.name}: {v}" for v in visitor.violations]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_session_json_no_hardcoded_run_paths() -> None:
    """Step commands in session_fast.json must not contain hardcoded run directory names."""
    script = DEMO_DIR / "session_fast.json"
    assert script.exists(), f"session script not found: {script}"

    violations = _scan_session_json(script)
    msg = "\n  ".join(violations)
    assert not violations, (
        f"Hardcoded run directories found in session step commands:\n  {msg}\n\n"
        "Fix: replace literal paths with environment variables"
        " (e.g. $SPEEDLM_TRAINING_RUN) injected by build_video.py."
    )


def test_extract_series_corroborating_is_cli_overridable() -> None:
    """extract_series.py must not pass CORROBORATING_DECISIONS directly to parse_reproductions.

    The call must be guarded so the CLI --corroborating flag takes precedence,
    e.g.:  parse_reproductions(args.corroborating_entries or CORROBORATING_DECISIONS)
    """
    script = DEMO_DIR / "remotion" / "extract_series.py"
    assert script.exists(), f"extract_series.py not found: {script}"

    violations = _scan_python_consumed_constants(script)
    msg = "\n  ".join(violations)
    assert not violations, (
        f"CORROBORATING_DECISIONS consumed without CLI override:\n  {msg}\n\n"
        "Fix: add --corroborating NAME=PATH (repeatable) CLI option and use"
        " it in parse_reproductions(), falling back to CORROBORATING_DECISIONS"
        " only when the flag is absent."
    )


def test_no_unguarded_banned_literals_in_extract_series() -> None:
    """String literals with banned run dir names in extract_series.py must be in DEFAULT_* vars."""
    script = DEMO_DIR / "remotion" / "extract_series.py"
    assert script.exists(), f"extract_series.py not found: {script}"

    violations = _scan_python_banned_literals(script)
    msg = "\n  ".join(violations)
    assert not violations, (
        f"Banned run-dir literals in non-DEFAULT_ assignments in extract_series.py:\n  {msg}\n\n"
        "Fix: move hardcoded paths into DEFAULT_* constants and ensure CLI flags"
        " override them."
    )
