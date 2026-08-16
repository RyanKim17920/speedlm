"""The task catalog: realistic work, planted deterministically from a seed.

Every family here was chosen because it forces a *shape* of trajectory that a
single request cannot produce:

``bugfix-localize``
    The failing test names a symptom, not a file.  Finding the planted defect
    needs a search, two or three reads across modules, one edit and a re-run --
    the canonical coding-agent loop, and the one where the prefix grows fastest
    because every tool result stays in context.

``feature-implement``
    Tests exist and fail; the implementation does not.  Long generated turns
    (writing a function body) rather than the short tool calls the other
    families produce, which matters because a drafter's acceptance differs
    sharply between the two.

``log-triage``
    A large read-only artifact and a small structured answer.  Produces the
    biggest prompts in the catalog and almost no writing, which is the
    prefill-dominated regime real agents spend much of their time in.

``refactor-rename``
    Multi-site edit under a green-tests invariant.  The agent must not break
    what already works, so the grader checks both the new name and the old
    behaviour -- a task that can be failed by doing too much, not only too
    little.

``schema-migrate``
    Read one format, emit another, satisfy a validator the agent can run.  Its
    edits are file writes rather than string replacements, so the generated
    turns are long and highly structured.

``call-chain-trace``
    Read-only reasoning across six modules, answered by writing a single value
    that exists nowhere in the source and can only be computed by following the
    chain.  Ungameable by grepping, and it is graded on the value.

Determinism
-----------
``instance(seed)`` is a pure function of the seed.  Nothing samples from the
process RNG and nothing reads the clock, so a trajectory recorded today can be
re-graded against a byte-identical workspace later.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import Any, Final

from tests.e2e.agentenv.families_v2 import CATALOG_V2, TASKS_V2
from tests.e2e.agentenv.phrasing import Brief, Fact, FactKind, render_instruction
from tests.e2e.agentenv.tasks import Grade, Task, TaskInstance, Workspace
from tests.e2e.agentenv.workspace import WorkspaceSandbox

__all__ = ["CATALOG", "TASKS", "all_instances", "task_by_name"]


# ---------------------------------------------------------------------------
# Shared fixture: a small but genuine data-pipeline package
# ---------------------------------------------------------------------------
_MODULE_NAMES: Final[tuple[str, ...]] = ("parse", "validate", "aggregate", "report")


def _pipeline_files(*, bug: str | None, seed: int) -> dict[str, str]:
    """A four-module package plus its tests, optionally with one planted defect.

    The defect is always a *plausible* one -- an off-by-one bound, a wrong
    comparison operator, a dropped branch -- rather than a syntax error, because
    a syntax error is found by running anything at all and would collapse the
    trajectory to two turns.
    """
    rng = random.Random(seed)
    threshold = rng.choice([50, 75, 100, 125])
    currency = rng.choice(["USD", "EUR", "GBP"])

    parse = '''"""Turn raw ledger lines into records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    account: str
    amount: int
    currency: str


class ParseError(ValueError):
    """A ledger line could not be read."""


def parse_line(line: str) -> Entry:
    """Parse one ``account,amount,currency`` line."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise ParseError(f"expected 3 fields, got {len(parts)}: {line!r}")
    account, raw_amount, currency = parts
    if not account:
        raise ParseError(f"empty account in {line!r}")
    try:
        amount = int(raw_amount)
    except ValueError:
        raise ParseError(f"amount is not an integer in {line!r}") from None
    return Entry(account=account, amount=amount, currency=currency.upper())


def parse_ledger(text: str) -> list[Entry]:
    entries = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(parse_line(stripped))
    return entries
'''

    # The validate defect flips a boundary from inclusive to exclusive, so the
    # test that fails is the one sitting exactly on the threshold.
    validate_bound = ">" if bug == "validate" else ">="
    validate = f'''"""Business rules applied to parsed entries."""

from __future__ import annotations

from pipeline.parse import Entry

BASE_CURRENCY = "{currency}"
LARGE_ENTRY_THRESHOLD = {threshold}


def is_large(entry: Entry) -> bool:
    """Whether an entry is large enough to need a second approval.

    An entry exactly at the threshold counts as large.
    """
    return entry.amount {validate_bound} LARGE_ENTRY_THRESHOLD


def is_foreign(entry: Entry) -> bool:
    return entry.currency != BASE_CURRENCY


def needs_review(entry: Entry) -> bool:
    return is_large(entry) or is_foreign(entry)
'''

    # The aggregate defect drops negative amounts from the per-account total,
    # which is invisible on any all-positive fixture.
    aggregate_filter = (
        "        if entry.amount < 0:\n            continue\n" if bug == "aggregate" else ""
    )
    aggregate = f'''"""Group entries and total them."""

from __future__ import annotations

from collections.abc import Iterable

from pipeline.parse import Entry


def totals_by_account(entries: Iterable[Entry]) -> dict[str, int]:
    """Sum every entry per account, including refunds (negative amounts)."""
    totals: dict[str, int] = {{}}
    for entry in entries:
{aggregate_filter}        totals[entry.account] = totals.get(entry.account, 0) + entry.amount
    return totals


def account_count(entries: Iterable[Entry]) -> int:
    return len({{entry.account for entry in entries}})
'''

    # The report defect sorts by account name rather than by descending total,
    # so it is only visible when the two orders disagree.
    report_key = (
        "sorted(totals.items())"
        if bug == "report"
        else "sorted(totals.items(), key=lambda item: (-item[1], item[0]))"
    )
    report = f'''"""Render totals for humans."""

from __future__ import annotations


def render_totals(totals: dict[str, int]) -> str:
    """One line per account, largest total first, ties broken by account name."""
    lines = []
    for account, total in {report_key}:
        lines.append(f"{{account}}: {{total}}")
    return "\\n".join(lines)
'''

    tests = f'''from pipeline.aggregate import account_count, totals_by_account
from pipeline.parse import ParseError, parse_ledger, parse_line
from pipeline.report import render_totals
from pipeline.validate import LARGE_ENTRY_THRESHOLD, is_large, needs_review

LEDGER = """
# monthly ledger
alpha, 40, {currency}
beta, {threshold}, {currency}
alpha, -15, {currency}
gamma, 12, XTS
"""


def test_parse_skips_comments_and_blanks():
    entries = parse_ledger(LEDGER)
    assert len(entries) == 4
    assert entries[0].account == "alpha"


def test_parse_line_rejects_short_lines():
    try:
        parse_line("alpha, 12")
    except ParseError:
        return
    raise AssertionError("parse_line accepted a two-field line")


def test_entry_exactly_at_threshold_is_large():
    entries = parse_ledger(LEDGER)
    at_threshold = [e for e in entries if e.amount == LARGE_ENTRY_THRESHOLD]
    assert at_threshold, "fixture no longer contains a threshold entry"
    assert is_large(at_threshold[0])


def test_foreign_currency_needs_review():
    entries = parse_ledger(LEDGER)
    foreign = [e for e in entries if e.currency == "XTS"]
    assert needs_review(foreign[0])


def test_refunds_reduce_the_account_total():
    entries = parse_ledger(LEDGER)
    totals = totals_by_account(entries)
    assert totals["alpha"] == 25


def test_account_count_is_distinct_accounts():
    assert account_count(parse_ledger(LEDGER)) == 3


def test_totals_render_largest_first():
    rendered = render_totals({{"alpha": 25, "beta": {threshold}, "gamma": 12}})
    assert rendered.splitlines()[0].startswith("beta")
'''

    return {
        "pipeline/__init__.py": '"""A small ledger pipeline."""\n',
        "pipeline/parse.py": parse,
        "pipeline/validate.py": validate,
        "pipeline/aggregate.py": aggregate,
        "pipeline/report.py": report,
        "tests/test_pipeline.py": tests,
        "README.md": (
            "# ledger pipeline\n\n"
            "Parse a ledger, apply review rules, total per account, render.\n\n"
            "Run the suite with `pytest`.\n"
        ),
    }


def _tests_pass_grader(target: str | None = None) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass(target)
        return Grade(
            solved=passed,
            detail=output.strip()[-1500:] if not passed else "pytest exited 0",
            checks={"pytest": passed},
        )

    return grade


# ---------------------------------------------------------------------------
# bugfix-localize
# ---------------------------------------------------------------------------
_BUG_SITES: Final[tuple[str, ...]] = ("validate", "aggregate", "report")
_BUGFIX_SALT: Final = 0xC0DE


def _bugfix_brief(seed: int) -> Brief:
    rng = random.Random(seed)
    threshold = rng.choice([50, 75, 100, 125])
    currency = rng.choice(["USD", "EUR", "GBP"])
    return Brief(
        goal="find the source bug and fix the pipeline code so the test suite passes",
        required_facts=(
            Fact(text="pytest", kind=FactKind.COMMAND),
            Fact(text="pipeline/", kind=FactKind.PATH),
            Fact(text=str(threshold), kind=FactKind.NUMBER),
            Fact(text=currency, kind=FactKind.TOKEN),
        ),
        context="the ledger pipeline test suite is currently failing",
        constraints=(
            "do not change anything under tests/ — the tests are correct",
            "change the smallest amount of pipeline code possible",
        ),
        blame_path="",
    )


def _bugfix_instance(seed: int) -> TaskInstance:
    site = _BUG_SITES[seed % len(_BUG_SITES)]
    files = _pipeline_files(bug=site, seed=seed)
    return TaskInstance(
        id=f"bugfix-localize-{seed:04d}",
        family="bugfix-localize",
        instruction=render_instruction(_bugfix_brief(seed), seed=seed, salt=_BUGFIX_SALT),
        workspace=Workspace(files=files),
        grader=_bugfix_grader(files),
        metadata={"planted_bug_module": site, "seed": seed},
    )


def _bugfix_grader(original: Mapping[str, str]) -> Any:
    """Tests must pass *and* the tests themselves must be untouched.

    Without the second half the task has a trivial solution -- delete the
    failing assertions -- and a model that finds it would be scored as having
    fixed the bug.  The check is a byte comparison against the materialized
    fixture rather than a mtime or a diff, because a model that rewrote the file
    to identical bytes has, in fact, not changed it.
    """
    frozen = {path: text for path, text in original.items() if path.startswith("tests/")}

    def grade(sandbox: WorkspaceSandbox) -> Grade:
        untouched = True
        for path, expected in frozen.items():
            current = sandbox.root / path
            if not current.is_file() or current.read_text(encoding="utf-8") != expected:
                untouched = False
                break
        passed, output = sandbox.tests_pass()
        return Grade(
            solved=passed and untouched,
            detail=(
                "pytest exited 0 and tests/ is byte-identical to the fixture"
                if passed and untouched
                else (
                    ("tests/ was modified; " if not untouched else "")
                    + (output.strip()[-1200:] if not passed else "pytest passed")
                )
            ),
            checks={"pytest": passed, "tests_untouched": untouched},
        )

    return grade


def _bugfix_solution(seed: int) -> Any:
    """Reference fix, used only by the solvability pre-flight."""
    site = _BUG_SITES[seed % len(_BUG_SITES)]

    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _pipeline_files(bug=None, seed=seed)
        (sandbox.root / f"pipeline/{site}.py").write_text(
            clean[f"pipeline/{site}.py"], encoding="utf-8"
        )

    return solve


# ---------------------------------------------------------------------------
# feature-implement
# ---------------------------------------------------------------------------
_FEATURE_SALT: Final = 0xCAFE


def _feature_brief(seed: int) -> Brief:
    rng = random.Random(seed ^ 0x5EED)
    window = rng.choice([3, 4, 5])
    return Brief(
        goal="implement rolling_max in pipeline/rolling.py so the test suite passes",
        required_facts=(
            Fact(text="tests/test_rolling.py", kind=FactKind.PATH),
            Fact(text="docs/rolling.md", kind=FactKind.PATH),
            Fact(text="pipeline/rolling.py", kind=FactKind.PATH),
            Fact(text=str(window), kind=FactKind.NUMBER),
        ),
        constraints=("do not change the tests or the docs",),
        creates=("pipeline/rolling.py",),
    )


def _feature_instance(seed: int) -> TaskInstance:
    rng = random.Random(seed ^ 0x5EED)
    window = rng.choice([3, 4, 5])
    files = dict(_pipeline_files(bug=None, seed=seed))
    files["tests/test_rolling.py"] = f"""from pipeline.rolling import rolling_max

WINDOW = {window}


def test_window_shorter_than_series_returns_one_value_per_position():
    assert len(rolling_max([1, 2, 3, 4, 5, 6], WINDOW)) == 6


def test_leading_positions_use_the_partial_window():
    assert rolling_max([5, 1, 1, 1, 1, 1], WINDOW)[0] == 5
    assert rolling_max([5, 1, 1, 1, 1, 1], WINDOW)[1] == 5


def test_value_leaves_the_window_when_it_falls_behind():
    series = [9] + [0] * (WINDOW * 2)
    result = rolling_max(series, WINDOW)
    assert result[WINDOW - 1] == 9
    assert result[WINDOW] == 0


def test_negative_values_are_supported():
    assert rolling_max([-5, -3, -9], 2) == [-5, -3, -3]


def test_window_of_one_is_the_series():
    assert rolling_max([4, 2, 7], 1) == [4, 2, 7]


def test_empty_series_is_empty():
    assert rolling_max([], WINDOW) == []


def test_non_positive_window_is_rejected():
    try:
        rolling_max([1, 2, 3], 0)
    except ValueError:
        return
    raise AssertionError("rolling_max accepted a window of 0")
"""
    files["docs/rolling.md"] = (
        "# rolling_max\n\n"
        "`pipeline.rolling.rolling_max(series, window)` returns a list the same\n"
        "length as `series`. Position *i* holds the maximum of the at most\n"
        "`window` values ending at *i*, so early positions use a shorter window.\n"
        "A window below 1 is a `ValueError`.\n"
    )
    return TaskInstance(
        id=f"feature-implement-{seed:04d}",
        family="feature-implement",
        instruction=render_instruction(_feature_brief(seed), seed=seed, salt=_FEATURE_SALT),
        workspace=Workspace(files=files),
        grader=_feature_grader(files),
        metadata={"window": window, "seed": seed},
    )


def _feature_grader(original: Mapping[str, str]) -> Any:
    frozen = {
        path: text
        for path, text in original.items()
        if path.startswith("tests/") or path.startswith("docs/")
    }

    def grade(sandbox: WorkspaceSandbox) -> Grade:
        untouched = all(
            (sandbox.root / path).is_file()
            and (sandbox.root / path).read_text(encoding="utf-8") == expected
            for path, expected in frozen.items()
        )
        implemented = (sandbox.root / "pipeline" / "rolling.py").is_file()
        passed, output = sandbox.tests_pass()
        return Grade(
            solved=passed and untouched and implemented,
            detail=(
                "pytest exited 0 with tests/ and docs/ untouched"
                if passed and untouched and implemented
                else (
                    ("specification files were modified; " if not untouched else "")
                    + ("pipeline/rolling.py was not created; " if not implemented else "")
                    + (output.strip()[-1200:] if not passed else "")
                )
            ),
            checks={
                "pytest": passed,
                "spec_untouched": untouched,
                "module_created": implemented,
            },
        )

    return grade


def _feature_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        (sandbox.root / "pipeline" / "rolling.py").write_text(
            '"""Rolling window helpers."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def rolling_max(series: list[int], window: int) -> list[int]:\n"
            "    if window < 1:\n"
            '        raise ValueError("window must be at least 1")\n'
            "    return [\n"
            "        max(series[max(0, i - window + 1) : i + 1])\n"
            "        for i in range(len(series))\n"
            "    ]\n",
            encoding="utf-8",
        )

    return solve


# ---------------------------------------------------------------------------
# log-triage
# ---------------------------------------------------------------------------
_LOG_SALT: Final = 0xB0B0


def _log_brief(seed: int) -> Brief:
    rng = random.Random(seed ^ 0xB0B)
    services = ["auth", "ledger", "billing", "search", "notify"]
    culprit = rng.choice(services)
    request_id = f"req-{rng.randrange(16**8):08x}"
    error_code = rng.choice(["E_TIMEOUT", "E_CONFLICT", "E_QUOTA", "E_UPSTREAM"])
    return Brief(
        goal="find the ERROR line and write findings.json with the three fields",
        required_facts=(
            Fact(text="service.log", kind=FactKind.PATH),
            Fact(text="findings.json", kind=FactKind.PATH),
            Fact(text=culprit, kind=FactKind.TOKEN),
            Fact(text=request_id, kind=FactKind.TOKEN),
            Fact(text=error_code, kind=FactKind.TOKEN),
        ),
        context="exactly one line is at ERROR level",
        constraints=(
            'use exactly the keys "service", "request_id" and "error_code"',
            "the error code is the uppercase token that begins the error message",
        ),
        has_test_suite=False,
        creates=("findings.json",),
    )


def _log_instance(seed: int) -> TaskInstance:
    rng = random.Random(seed ^ 0xB0B)
    services = ["auth", "ledger", "billing", "search", "notify"]
    culprit = rng.choice(services)
    request_id = f"req-{rng.randrange(16**8):08x}"
    error_code = rng.choice(["E_TIMEOUT", "E_CONFLICT", "E_QUOTA", "E_UPSTREAM"])
    lines: list[str] = []
    for tick in range(1400):
        service = services[(tick * 7 + seed) % len(services)]
        level = "INFO"
        rid = f"req-{(tick * 2654435761 + seed) % 16**8:08x}"
        message = f"handled in {(tick * 37) % 400}ms"
        if tick == 812:
            service, level, rid = culprit, "ERROR", request_id
            message = f"{error_code}: downstream refused after 3 retries"
        elif tick % 211 == 0:
            level = "WARN"
            message = "retrying after transient failure"
        stamp = f"2026-08-09T04:{tick // 60:02d}:{tick % 60:02d}Z"
        lines.append(f"{stamp} {level} {service} {rid} {message}")
    log = "\n".join(lines) + "\n"

    return TaskInstance(
        id=f"log-triage-{seed:04d}",
        family="log-triage",
        instruction=render_instruction(_log_brief(seed), seed=seed, salt=_LOG_SALT),
        workspace=Workspace(
            files={
                "service.log": log,
                "README.md": (
                    "# incident triage\n\nLogs are `TIMESTAMP LEVEL SERVICE REQUEST_ID MESSAGE`.\n"
                ),
            }
        ),
        grader=_log_grader(
            {"service": culprit, "request_id": request_id, "error_code": error_code}
        ),
        metadata={"seed": seed, "log_lines": len(lines)},
    )


def _log_grader(expected: Mapping[str, str]) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        path = sandbox.root / "findings.json"
        if not path.is_file():
            return Grade(False, "findings.json was not created", {"file_present": False})
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return Grade(False, f"findings.json is not valid JSON: {error}", {"json_valid": False})
        if not isinstance(found, dict):
            return Grade(False, "findings.json is not a JSON object", {"json_object": False})
        checks = {
            "file_present": True,
            "json_valid": True,
            "json_object": True,
            **{key: found.get(key) == value for key, value in expected.items()},
            "no_extra_keys": set(found) == set(expected),
        }
        solved = all(checks.values())
        return Grade(
            solved=solved,
            detail=(
                "all three fields match" if solved else f"got {found!r}, wanted {dict(expected)!r}"
            ),
            checks=checks,
        )

    return grade


def _log_solution(seed: int) -> Any:
    # The reference solution reads the log, exactly as an agent must, so it
    # takes no precomputed answer from the instance metadata.
    def solve(sandbox: WorkspaceSandbox) -> None:
        log = (sandbox.root / "service.log").read_text(encoding="utf-8")
        for line in log.splitlines():
            fields = line.split(" ", 4)
            if len(fields) == 5 and fields[1] == "ERROR":
                _, _, service, request_id, message = fields
                (sandbox.root / "findings.json").write_text(
                    json.dumps(
                        {
                            "service": service,
                            "request_id": request_id,
                            "error_code": message.split(":", 1)[0],
                        }
                    ),
                    encoding="utf-8",
                )
                return
        raise AssertionError("fixture contains no ERROR line")

    return solve


# ---------------------------------------------------------------------------
# refactor-rename
# ---------------------------------------------------------------------------
_REFACTOR_SALT: Final = 0xBEEF


def _refactor_brief(seed: int) -> Brief:
    rng = random.Random(seed)
    threshold = rng.choice([50, 75, 100, 125])
    currency = rng.choice(["USD", "EUR", "GBP"])
    return Brief(
        goal="rename the function and keep the whole test suite green",
        required_facts=(
            Fact(text="needs_review", kind=FactKind.SYMBOL),
            Fact(text="requires_second_approval", kind=FactKind.SYMBOL),
            Fact(text=str(threshold), kind=FactKind.NUMBER),
            Fact(text=currency, kind=FactKind.TOKEN),
        ),
        context="needs_review reads like a noun phrase and callers expect it to return a list",
        constraints=(
            "rename everywhere it is defined or used, including in tests",
            "the old name must not survive anywhere in the workspace",
            "no behaviour may change",
        ),
        creates=("requires_second_approval",),
    )


def _refactor_instance(seed: int) -> TaskInstance:
    files = dict(_pipeline_files(bug=None, seed=seed))
    files["pipeline/cli.py"] = '''"""Command line entry point."""

from __future__ import annotations

import sys

from pipeline.aggregate import totals_by_account
from pipeline.parse import parse_ledger
from pipeline.report import render_totals
from pipeline.validate import needs_review


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: cli.py LEDGER", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        entries = parse_ledger(handle.read())
    flagged = [entry for entry in entries if needs_review(entry)]
    print(render_totals(totals_by_account(entries)))
    print(f"{len(flagged)} entries need review")
    return 0
'''
    files["tests/test_cli.py"] = """import pipeline.cli as cli


def test_main_rejects_wrong_argument_count():
    assert cli.main(["cli.py"]) == 2


def test_main_reads_a_ledger(tmp_path, capsys):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("alpha, 10, USD\\n", encoding="utf-8")
    assert cli.main(["cli.py", str(ledger)]) == 0
    assert "alpha: 10" in capsys.readouterr().out
"""
    return TaskInstance(
        id=f"refactor-rename-{seed:04d}",
        family="refactor-rename",
        instruction=render_instruction(_refactor_brief(seed), seed=seed, salt=_REFACTOR_SALT),
        workspace=Workspace(files=files),
        grader=_refactor_grader(),
        metadata={"seed": seed},
    )


def _refactor_grader() -> Any:
    """Green tests, the new name present, and the old name gone.

    All three are needed.  Tests alone pass if the agent does nothing (the suite
    starts green, which is the point of a refactor task) -- so "no occurrence of
    the old name" is the check that can actually fail, and "the new name is
    called from the CLI" is what stops a model from satisfying that by deleting
    the function.
    """

    def grade(sandbox: WorkspaceSandbox) -> Grade:
        sources = [
            path
            for path in sandbox.root.rglob("*.py")
            if ".tmp" not in path.parts and "__pycache__" not in path.parts
        ]
        bodies = {path: path.read_text(encoding="utf-8") for path in sources}
        old_present = any("needs_review" in text for text in bodies.values())
        new_defined = any("def requires_second_approval" in text for text in bodies.values())
        new_called = sum(1 for text in bodies.values() if "requires_second_approval" in text)
        passed, output = sandbox.tests_pass()
        checks = {
            "pytest": passed,
            "old_name_gone": not old_present,
            "new_name_defined": new_defined,
            "new_name_used_in_more_than_one_file": new_called >= 2,
        }
        solved = all(checks.values())
        return Grade(
            solved=solved,
            detail=(
                "renamed across the workspace with the suite green"
                if solved
                else f"checks={checks}; pytest tail: {output.strip()[-800:]}"
            ),
            checks=checks,
        )

    return grade


def _refactor_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        for path in sandbox.root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if "needs_review" in text:
                path.write_text(
                    text.replace("needs_review", "requires_second_approval"), encoding="utf-8"
                )

    return solve


# ---------------------------------------------------------------------------
# schema-migrate
# ---------------------------------------------------------------------------
_SCHEMA_SALT: Final = 0xF00D


def _schema_brief(seed: int) -> Brief:
    rng = random.Random(seed ^ 0xC0FFEE)
    hosts = [f"{name}-{rng.randrange(10, 99)}" for name in ("edge", "core", "cache")]
    port = 8000 + rng.randrange(0, 900)
    replicas = rng.randrange(1, 6)
    enabled_bool = (0 + seed) % 2 == 0
    return Brief(
        goal="produce settings.json in the new format and run the test suite",
        required_facts=(
            Fact(text="settings.ini", kind=FactKind.PATH),
            Fact(text="settings.json", kind=FactKind.PATH),
            Fact(text=hosts[0], kind=FactKind.TOKEN),
            Fact(text=str(port), kind=FactKind.NUMBER),
            Fact(text=str(replicas), kind=FactKind.NUMBER),
            Fact(text=str(enabled_bool).lower(), kind=FactKind.TOKEN),
        ),
        context="settings.ini is the old deployment config",
        constraints=(
            "ports and replica counts are JSON numbers, not strings",
            "enabled is a JSON boolean, not the string the ini file uses",
        ),
        creates=("settings.json",),
    )


def _schema_instance(seed: int) -> TaskInstance:
    rng = random.Random(seed ^ 0xC0FFEE)
    hosts = [f"{name}-{rng.randrange(10, 99)}" for name in ("edge", "core", "cache")]
    ini_lines = ["# deployment settings", ""]
    expected: dict[str, Any] = {}
    for index, host in enumerate(hosts):
        port = 8000 + rng.randrange(0, 900)
        replicas = rng.randrange(1, 6)
        enabled = "true" if (index + seed) % 2 == 0 else "false"
        ini_lines += [
            f"[{host}]",
            f"port = {port}",
            f"replicas = {replicas}",
            f"enabled = {enabled}",
            "",
        ]
        expected[host] = {
            "port": port,
            "replicas": replicas,
            "enabled": enabled == "true",
        }

    validator = '''"""Validate settings.json against the rules in README.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path("settings.json")
    if not path.is_file():
        print("settings.json is missing")
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"settings.json is not valid JSON: {error}")
        return 1
    if not isinstance(data, dict):
        print("settings.json must be a JSON object keyed by host")
        return 1
    for host, block in data.items():
        if not isinstance(block, dict):
            print(f"{host}: value must be an object")
            return 1
        if set(block) != {"port", "replicas", "enabled"}:
            print(f"{host}: keys must be exactly port, replicas, enabled; got {sorted(block)}")
            return 1
        if not isinstance(block["port"], int) or isinstance(block["port"], bool):
            print(f"{host}: port must be a JSON number, not a string")
            return 1
        if not isinstance(block["replicas"], int) or isinstance(block["replicas"], bool):
            print(f"{host}: replicas must be a JSON number, not a string")
            return 1
        if not isinstance(block["enabled"], bool):
            print(f"{host}: enabled must be a JSON boolean, not a string")
            return 1
    print(f"settings.json is valid: {len(data)} hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    tests = """import subprocess
import sys


def test_settings_json_passes_the_validator():
    completed = subprocess.run(
        [sys.executable, "validate_settings.py"], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
"""
    return TaskInstance(
        id=f"schema-migrate-{seed:04d}",
        family="schema-migrate",
        instruction=render_instruction(_schema_brief(seed), seed=seed, salt=_SCHEMA_SALT),
        workspace=Workspace(
            files={
                "settings.ini": "\n".join(ini_lines),
                "validate_settings.py": validator,
                "tests/test_settings.py": tests,
                "README.md": (
                    "# settings\n\n"
                    "`settings.json` is a JSON object keyed by host name. Each value "
                    "has exactly `port` (number), `replicas` (number) and `enabled` "
                    "(boolean). Run `python validate_settings.py` to check it.\n"
                ),
            }
        ),
        grader=_schema_grader(expected),
        metadata={"seed": seed, "hosts": len(hosts)},
    )


def _schema_grader(expected: Mapping[str, Any]) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        path = sandbox.root / "settings.json"
        if not path.is_file():
            return Grade(False, "settings.json was not created", {"file_present": False})
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return Grade(False, f"settings.json is not valid JSON: {error}", {"json_valid": False})
        matches = found == dict(expected)
        passed, output = sandbox.tests_pass()
        checks = {
            "file_present": True,
            "json_valid": True,
            "content_matches": matches,
            "pytest": passed,
        }
        return Grade(
            solved=matches and passed,
            detail=(
                "settings.json matches the ini source and the validator accepts it"
                if matches and passed
                else f"content_matches={matches}; validator output: {output.strip()[-800:]}"
            ),
            checks=checks,
        )

    return grade


def _schema_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        text = (sandbox.root / "settings.ini").read_text(encoding="utf-8")
        result: dict[str, Any] = {}
        host: str | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                host = line[1:-1]
                result[host] = {}
                continue
            if host is None or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "enabled":
                result[host][key] = value == "true"
            else:
                result[host][key] = int(value)
        (sandbox.root / "settings.json").write_text(json.dumps(result), encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# call-chain-trace
# ---------------------------------------------------------------------------
_TRACE_SALT: Final = 0xA11C


def _trace_brief(seed: int) -> Brief:
    rng = random.Random(seed ^ 0xA11CE)
    base = rng.randrange(3, 40)
    steps = [rng.randrange(2, 9) for _ in range(5)]
    return Brief(
        goal="work out the return value by reading the code and compose the arithmetic",
        required_facts=(
            Fact(text="chain.stage5.apply5", kind=FactKind.SYMBOL),
            Fact(text="answer.txt", kind=FactKind.PATH),
            Fact(text=str(base), kind=FactKind.NUMBER),
            Fact(text=str(steps[0]), kind=FactKind.NUMBER),
        ),
        context="chain.stage5.apply5() returns a single integer computed by a chain of stages",
        constraints=(
            "write that integer and nothing else to answer.txt",
            "there is no test suite and you must not add one",
        ),
        has_test_suite=False,
        creates=("answer.txt",),
    )


def _trace_instance(seed: int) -> TaskInstance:
    rng = random.Random(seed ^ 0xA11CE)
    base = rng.randrange(3, 40)
    steps = [rng.randrange(2, 9) for _ in range(5)]
    # The answer exists nowhere in the source: it is the composition of six
    # constants spread over six modules, so grep cannot find it and the model
    # has to read and compose.
    answer = base
    for step in steps:
        answer = answer * step + 1

    files: dict[str, str] = {
        "chain/__init__.py": '"""A deliberately indirect computation."""\n',
        "chain/stage0.py": f'"""Entry point of the chain."""\n\nSEED_VALUE = {base}\n\n\n'
        "def start() -> int:\n    return SEED_VALUE\n",
    }
    for index, step in enumerate(steps, start=1):
        previous = index - 1
        files[f"chain/stage{index}.py"] = (
            f'"""Stage {index} of the chain."""\n\n'
            f"from chain.stage{previous} import "
            f"{'start' if previous == 0 else f'apply{previous}'}\n\n"
            f"FACTOR = {step}\n\n\n"
            f"def apply{index}() -> int:\n"
            f"    return {'start' if previous == 0 else f'apply{previous}'}() * FACTOR + 1\n"
        )
    files["README.md"] = (
        "# chain\n\n"
        f"`chain.stage{len(steps)}.apply{len(steps)}()` returns a single integer, "
        "computed by a chain of stages that each transform the previous one.\n"
    )

    return TaskInstance(
        id=f"call-chain-trace-{seed:04d}",
        family="call-chain-trace",
        instruction=render_instruction(_trace_brief(seed), seed=seed, salt=_TRACE_SALT),
        workspace=Workspace(files=files),
        grader=_trace_grader(answer),
        metadata={"seed": seed, "answer": answer, "stages": len(steps)},
    )


def _trace_grader(answer: int) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        path = sandbox.root / "answer.txt"
        if not path.is_file():
            return Grade(False, "answer.txt was not created", {"file_present": False})
        raw = path.read_text(encoding="utf-8").strip()
        try:
            value = int(raw)
        except ValueError:
            return Grade(
                False,
                f"answer.txt holds {raw[:120]!r}, which is not an integer",
                {"file_present": True, "integer": False},
            )
        correct = value == answer
        return Grade(
            solved=correct,
            detail=("the value is correct" if correct else f"got {value}, wanted {answer}"),
            checks={"file_present": True, "integer": True, "value_correct": correct},
        )

    return grade


def _trace_solution(seed: int) -> Any:
    instance = _trace_instance(seed)
    answer = int(instance.metadata["answer"])

    def solve(sandbox: WorkspaceSandbox) -> None:
        (sandbox.root / "answer.txt").write_text(f"{answer}\n", encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TASKS: Final[tuple[Task, ...]] = (
    Task(
        name="bugfix-localize",
        summary="A planted defect in one of three modules; failing suite names only the symptom.",
        build=_bugfix_instance,
    ),
    Task(
        name="feature-implement",
        summary="Tests and docs specify a module that does not exist yet.",
        build=_feature_instance,
    ),
    Task(
        name="log-triage",
        summary="Find the single ERROR line in a 1400-line log and report three fields.",
        build=_log_instance,
    ),
    Task(
        name="refactor-rename",
        summary="Rename an API across modules and tests without changing behaviour.",
        build=_refactor_instance,
    ),
    Task(
        name="schema-migrate",
        summary="Translate an ini config to JSON with correct types, checked by a validator.",
        build=_schema_instance,
    ),
    Task(
        name="call-chain-trace",
        summary="Compose six constants across six modules into one value.",
        build=_trace_instance,
    ),
    *TASKS_V2,
)

#: Reference solutions, used only by the solvability pre-flight in
#: ``tests/test_agentenv_catalog.py``.  They live beside the tasks so a task and
#: the proof that it is passable cannot drift apart.
CATALOG: Final[Mapping[str, Any]] = {
    "bugfix-localize": _bugfix_solution,
    "feature-implement": _feature_solution,
    "log-triage": _log_solution,
    "refactor-rename": _refactor_solution,
    "schema-migrate": _schema_solution,
    "call-chain-trace": _trace_solution,
    **CATALOG_V2,
}


def task_by_name(name: str) -> Task:
    for task in TASKS:
        if task.name == name:
            return task
    known = ", ".join(task.name for task in TASKS)
    raise KeyError(f"unknown task {name!r}; available: {known}")


def all_instances(
    *,
    seeds: int,
    families: tuple[str, ...] | None = None,
    seed_start: int = 0,
) -> list[TaskInstance]:
    """``seeds`` instances of every selected family, in a stable order.

    Seeds run over ``range(seed_start, seed_start + seeds)``.  ``seed_start``
    exists so that concurrent traffic shards can each take a DISJOINT seed
    window: ``instance(seed)`` is a pure function of the seed, so two shards
    both left at the default ``seed_start=0`` would plant byte-identical
    workspaces and prompts and merely duplicate each other's corpus.
    """
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    chosen = [task for task in TASKS if families is None or task.name in families]
    if not chosen:
        raise ValueError(f"no task matched families={families!r}")
    return [
        task.instance(seed) for seed in range(seed_start, seed_start + seeds) for task in chosen
    ]
