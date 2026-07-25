from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import assert_clean_cli_result, run_cli


def decision(
    *,
    verdict: str,
    reason: str,
    acceptance_delta_pp: float,
    throughput_delta_pct: float,
    stock_acceptance: float = 0.62,
    candidate_acceptance: float = 0.68,
    stock_tps: float = 100.0,
    candidate_tps: float = 112.0,
    repeats: int = 2,
) -> dict[str, Any]:
    per_repeat = [
        {
            "repeat_index": index,
            "stock_tok_per_sec": stock_tps,
            "candidate_tok_per_sec": candidate_tps,
            "stock_acceptance_rate": stock_acceptance,
            "candidate_acceptance_rate": candidate_acceptance,
            "invalid_rate": 0.0,
            "output_mismatches": 0,
        }
        for index in range(repeats)
    ]
    return {
        "verdict": verdict,
        "reason": reason,
        "acceptance_delta_pp": acceptance_delta_pp,
        "throughput_delta_pct": throughput_delta_pct,
        "min_acceptance_delta_pp": 1.0,
        "min_throughput_delta_pct": 2.0,
        "num_repeats": repeats,
        "per_repeat": per_repeat,
        "stock_avg_acceptance": stock_acceptance,
        "candidate_avg_acceptance": candidate_acceptance,
        "stock_avg_tok_per_sec": stock_tps,
        "candidate_avg_tok_per_sec": candidate_tps,
    }


def write_decision(home: Path, payload: dict[str, Any], run_name: str) -> None:
    run_dir = home / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "decision.json").write_text(json.dumps(payload), encoding="utf-8")


def test_gain_without_benchmark_plainly_says_no_gate_ran(speedlm_home: Path) -> None:
    result = run_cli(speedlm_home, "gain")
    assert_clean_cli_result(result)
    assert "No gate has ever run" in result.stdout
    assert "no measured gain to report" in result.stdout
    assert "tok/s" not in result.stdout
    assert "0.0%" not in result.stdout


def test_gain_shows_measured_numbers_verdict_and_reason(speedlm_home: Path) -> None:
    write_decision(
        speedlm_home,
        decision(
            verdict="promote",
            reason="both_thresholds_met",
            acceptance_delta_pp=6.0,
            throughput_delta_pct=12.0,
        ),
        "promote-run",
    )

    result = run_cli(speedlm_home, "gain")
    assert_clean_cli_result(result)
    assert "verdict           : promote" in result.stdout
    assert "both_thresholds_met" in result.stdout
    assert "acceptance delta  : +6.00 pp" in result.stdout
    assert "throughput delta  : +12.00%" in result.stdout
    assert "promoted on measured gains" in result.stdout


def test_rejected_gain_explains_why(speedlm_home: Path) -> None:
    write_decision(
        speedlm_home,
        decision(
            verdict="reject",
            reason="acceptance_below_threshold",
            acceptance_delta_pp=0.2,
            throughput_delta_pct=5.0,
            candidate_acceptance=0.622,
            candidate_tps=105.0,
        ),
        "reject-run",
    )

    result = run_cli(speedlm_home, "gain")
    assert_clean_cli_result(result)
    assert "verdict           : reject" in result.stdout
    assert "reason            : acceptance_below_threshold" in result.stdout
    assert "acceptance delta  : +0.20 pp" in result.stdout
    assert "threshold >= 1.00 pp" in result.stdout
    assert "rejected because" in result.stdout


def test_unmeasured_reject_never_formats_zero_as_measured(speedlm_home: Path) -> None:
    payload = decision(
        verdict="reject",
        reason="counter_reset",
        acceptance_delta_pp=0.0,
        throughput_delta_pct=0.0,
        stock_acceptance=0.0,
        candidate_acceptance=0.0,
        stock_tps=0.0,
        candidate_tps=0.0,
        repeats=0,
    )
    write_decision(speedlm_home, payload, "unmeasured-run")

    result = run_cli(speedlm_home, "gain")
    assert_clean_cli_result(result)
    assert "verdict           : reject" in result.stdout
    assert "counter_reset" in result.stdout
    assert "acceptance        : not measured" in result.stdout
    assert "throughput        : not measured" in result.stdout
    assert "speedup           : not measured" in result.stdout
    assert "tok/s" not in result.stdout
    assert "0.00%" not in result.stdout
