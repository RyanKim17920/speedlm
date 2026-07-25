from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import assert_clean_cli_result, run_cli


def test_day_one_commands_are_safe_and_understandable_in_order(speedlm_home: Path) -> None:
    assert not speedlm_home.exists()

    results = [
        run_cli(speedlm_home, "--version"),
        run_cli(speedlm_home, "status"),
        run_cli(speedlm_home, "traces", "stats"),
        run_cli(speedlm_home, "gain"),
        run_cli(speedlm_home, "doctor"),
    ]

    assert_clean_cli_result(results[0])
    assert_clean_cli_result(results[1])
    assert_clean_cli_result(results[2])
    assert_clean_cli_result(results[3])
    assert_clean_cli_result(results[4], expected_codes=(0, 1))

    assert re.search(r"speedlm \d+\.\d+\.\d+", results[0].stdout)
    assert "no gateway is running" in results[1].stdout.lower()
    assert "no active draft" in results[1].stdout.lower()
    assert "trace buffer is empty" in results[1].stdout.lower()
    assert "No gate has ever run" in results[3].stdout
    assert "no measured gain to report" in results[3].stdout
    assert "Overall:" in results[4].stdout
    assert "Execution mode:" in results[4].stdout

    for result in results:
        assert "0.0% speedup" not in result.output.lower()
        assert "0.00% speedup" not in result.output.lower()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/cli.py:423-427 renders empty token buckets as measured/estimated "
        "zero measurements instead of saying that no traces exist"
    ),
)
def test_day_one_stats_do_not_present_zero_as_a_measurement(speedlm_home: Path) -> None:
    result = run_cli(speedlm_home, "traces", "stats")
    assert_clean_cli_result(result)
    assert "count    : 0" in result.stdout
    assert "measured : 0" not in result.stdout
    assert "estimated: 0" not in result.stdout
    assert "no traces" in result.stdout.lower()
