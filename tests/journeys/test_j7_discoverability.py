from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import assert_clean_cli_result, run_cli

HELP_PATHS = [
    ("--help",),
    ("vllm", "--help"),
    ("vllm", "serve", "--help"),
    ("traces", "--help"),
    ("traces", "import", "--help"),
    ("traces", "stats", "--help"),
    ("status", "--help"),
    ("gain", "--help"),
    ("doctor", "--help"),
]


@pytest.mark.parametrize("args", HELP_PATHS)
def test_every_public_help_path_is_clean(
    args: tuple[str, ...],
    speedlm_home: Path,
) -> None:
    result = run_cli(speedlm_home, *args)
    assert_clean_cli_result(result)
    assert result.stdout.startswith("usage: speedlm")
    assert "not yet implemented" not in result.stdout.lower()


def test_help_matches_the_commands_and_options_users_can_invoke(speedlm_home: Path) -> None:
    top = run_cli(speedlm_home, "--help")
    assert_clean_cli_result(top)
    for command in ("vllm", "traces", "status", "gain", "doctor"):
        assert command in top.stdout
    assert "tune" not in top.stdout
    assert "benchmark" not in top.stdout

    serve = run_cli(speedlm_home, "vllm", "serve", "--help")
    assert_clean_cli_result(serve)
    for item in ("model", "--host", "--port", "--enable-idle-tuning"):
        assert item in serve.stdout

    imported = run_cli(speedlm_home, "traces", "import", "--help")
    assert_clean_cli_result(imported)
    for item in ("path", "--model", "--store", "JSONL", "bootstrapping"):
        assert item in imported.stdout

    status = run_cli(speedlm_home, "status", "--json")
    assert_clean_cli_result(status)
    assert json.loads(status.stdout)["gateway"]["state"] == "stopped"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/cli.py:501 uses parse_known_args and ignores unknown arguments for "
        "non-serve commands, so typos contradict argparse-style help"
    ),
)
@pytest.mark.parametrize(
    "args",
    [
        ("status", "--jsoon"),
        ("gain", "--jsoon"),
        ("doctor", "--jsoon"),
        ("traces", "stats", "--strore", "somewhere"),
    ],
)
def test_help_contract_rejects_unknown_or_misspelled_options(
    args: tuple[str, ...],
    speedlm_home: Path,
) -> None:
    result = run_cli(speedlm_home, *args)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_serve_help_discloses_forwarded_vllm_arguments(speedlm_home: Path) -> None:
    result = run_cli(speedlm_home, "vllm", "serve", "--help")
    assert_clean_cli_result(result)
    lowered = result.stdout.lower()
    assert "vllm_args" in lowered or "additional arguments are forwarded" in lowered
