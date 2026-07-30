from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from speedlm.gate.decide import Reason, Verdict
from speedlm.report import (
    GainStatus,
    GatewayState,
    build_gain_report,
    build_status_report,
    find_latest_decision,
    load_decision,
)
from speedlm.storage import resolve_layout

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "speedlm-home"
    monkeypatch.setenv("SPEEDLM_HOME", str(root))
    return root


def _decision_dict(
    *,
    verdict: str = "promote",
    reason: str = "both_thresholds_met",
    acceptance_delta_pp: float | None = 6.0,
    throughput_delta_pct: float | None = 12.0,
    num_repeats: int = 3,
    per_repeat: list[dict[str, Any]] | None = None,
    stock_acc: float = 0.62,
    cand_acc: float = 0.68,
    stock_tps: float = 100.0,
    cand_tps: float = 112.0,
) -> dict[str, Any]:
    if per_repeat is None:
        per_repeat = [
            {
                "repeat_index": i,
                "stock_tok_per_sec": stock_tps,
                "candidate_tok_per_sec": cand_tps,
                "stock_acceptance_rate": stock_acc,
                "candidate_acceptance_rate": cand_acc,
                "invalid_rate": 0.0,
                "output_mismatches": 0,
            }
            for i in range(num_repeats)
        ]
    return {
        "verdict": verdict,
        "reason": reason,
        "acceptance_delta_pp": acceptance_delta_pp,
        "throughput_delta_pct": throughput_delta_pct,
        "min_acceptance_delta_pp": 1.0,
        "min_throughput_delta_pct": 2.0,
        "num_repeats": num_repeats,
        "per_repeat": per_repeat,
        "stock_avg_acceptance": stock_acc,
        "candidate_avg_acceptance": cand_acc,
        "stock_avg_tok_per_sec": stock_tps,
        "candidate_avg_tok_per_sec": cand_tps,
    }


def _write_decision(home_dir: Path, payload: dict[str, Any], *, run: str = "run-1") -> Path:
    run_dir = home_dir / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_traces(home_dir: Path, records: list[dict[str, Any]]) -> Path:
    traces_dir = home_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / "traces.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _trace_record(rid: str, timestamp: float, prompt: int, completion: int) -> dict[str, Any]:
    return {
        "id": rid,
        "timestamp": timestamp,
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_calls": [],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


# ---------------------------------------------------------------------------
# status — fresh install
# ---------------------------------------------------------------------------


def test_status_fresh_install_renders_cleanly(home: Path) -> None:
    report = build_status_report(now=1_000.0)
    text = report.render_text()

    assert not home.exists(), "status must never create SPEEDLM_HOME"
    assert "SpeedLM status" in text
    assert "no active draft" in text
    assert "0 record(s), 0 token(s)" in text
    assert "no tuner state" in text
    assert report.gateway.state is GatewayState.STOPPED
    assert report.home_exists is False
    assert report.traces.count == 0


def test_status_json_shape_fresh(home: Path) -> None:
    payload = json.loads(build_status_report(now=1_000.0).to_json())
    assert set(payload) == {
        "home",
        "home_exists",
        "generated_at",
        "gateway",
        "active_draft",
        "traces",
        "tuner",
        "scheduler",
        "models",
        "profile",
    }
    assert payload["gateway"]["state"] == "stopped"
    assert payload["active_draft"]["present"] is False
    assert payload["traces"]["count"] == 0
    assert payload["tuner"]["present"] is False
    assert payload["scheduler"]["present"] is False
    assert payload["models"]["configured"] is False
    assert payload["models"]["verifier"]
    assert isinstance(payload["profile"], dict)
    assert payload["profile"]["status"] == "profiled"
    assert payload["models"]["profile"] == payload["profile"]


# ---------------------------------------------------------------------------
# status — populated install
# ---------------------------------------------------------------------------


def test_status_with_traces_present(home: Path) -> None:
    _write_traces(
        home,
        [
            _trace_record("a", 500.0, 10, 20),
            _trace_record("b", 900.0, 5, 5),
        ],
    )
    report = build_status_report(now=1_000.0)
    text = report.render_text()

    assert report.traces.count == 2
    assert report.traces.tokens == 40
    assert report.traces.oldest == 500.0
    assert report.traces.newest == 900.0
    assert "2 record(s), 40 token(s)" in text
    assert "oldest" in text


def test_status_with_active_draft_and_tuner_state(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "active.json").write_text(
        json.dumps(
            {"artifact_id": "abc123", "history": ["old1", "old2"], "updated_at": 900.0}
        ),
        encoding="utf-8",
    )
    runs = home / "runs"
    runs.mkdir(parents=True)
    (runs / "state.json").write_text(
        json.dumps(
            {"state": "TRAINING", "sequence": 4, "updated_at": 950.0, "reason": "idle window"}
        ),
        encoding="utf-8",
    )

    report = build_status_report(now=1_000.0)
    text = report.render_text()

    assert report.active_draft.present is True
    assert report.active_draft.artifact_id == "abc123"
    assert report.active_draft.history == ("old1", "old2")
    assert "abc123" in text
    assert "2 prior artifact(s)" in text
    assert report.tuner.present is True
    assert report.tuner.state == "TRAINING"
    assert "TRAINING" in text
    assert "idle window" in text


def test_status_reports_durable_scheduler_lifecycle_and_last_cycle(
    home: Path,
) -> None:
    runs = home / "runs"
    runs.mkdir(parents=True)
    (runs / "scheduler.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "lifecycle": "stopped",
                "created_at": 800.0,
                "updated_at": 990.0,
                "lifecycle_changed_at": 990.0,
                "last_attempt_at": 900.0,
                "last_result_at": 950.0,
                "last_error_at": None,
                "last_watermark": {
                    "count": 40,
                    "tokens": 400,
                    "oldest": 100.0,
                    "newest": 899.0,
                    "unknown_token_records": 0,
                },
                "last_result": {
                    "outcome": "promoted",
                    "artifact_id": "artifact-1",
                    "error": None,
                    "decision_path": "runs/run-1/decision.json",
                },
                "last_error": None,
            }
        ),
        encoding="utf-8",
    )

    report = build_status_report(now=1_000.0)
    payload = report.to_dict()["scheduler"]
    text = report.render_text()

    assert report.scheduler.present is True
    assert report.scheduler.enabled is True
    assert report.scheduler.lifecycle == "stopped"
    assert report.scheduler.last_watermark == {
        "count": 40,
        "tokens": 400,
        "oldest": 100.0,
        "newest": 899.0,
        "unknown_token_records": 0,
    }
    assert isinstance(payload, dict)
    assert payload["last_result"]["outcome"] == "promoted"
    assert "scheduler    : stopped (enabled)" in text
    assert "last cycle : promoted" in text


def test_status_gateway_running_and_stale(home: Path) -> None:
    home.mkdir(parents=True)
    gateway = home / "gateway.json"

    gateway.write_text(
        json.dumps({"pid": os.getpid(), "host": "127.0.0.1", "port": 8100}),
        encoding="utf-8",
    )
    running = build_status_report(now=1_000.0)
    assert running.gateway.state is GatewayState.RUNNING
    assert "127.0.0.1:8100" in running.gateway.detail

    # PID 2**31 - 1 is above the default pid_max and cannot be live.
    gateway.write_text(json.dumps({"pid": 2**31 - 1}), encoding="utf-8")
    stale = build_status_report(now=1_000.0)
    assert stale.gateway.state is GatewayState.STALE
    assert "stale" in stale.gateway.detail


def test_status_tolerates_corrupt_files(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "gateway.json").write_text("{not json", encoding="utf-8")
    (home / "active.json").write_text("[]", encoding="utf-8")
    (home / "runs").mkdir()
    (home / "runs" / "state.json").write_text(json.dumps({"sequence": 1}), encoding="utf-8")

    report = build_status_report(now=1_000.0)
    text = report.render_text()

    assert report.gateway.state is GatewayState.UNREADABLE
    assert report.active_draft.present is False
    assert report.tuner.present is False
    assert "unreadable" in text or "no usable" in text


def test_status_model_pair_from_config(home: Path) -> None:
    from speedlm.doctor import PRIMARY_DRAFT, PRIMARY_VERIFIER

    home.mkdir(parents=True)
    (home / "config.json").write_text(
        json.dumps({"model": PRIMARY_VERIFIER}), encoding="utf-8"
    )
    report = build_status_report(now=1_000.0)
    assert report.models.configured is True
    assert report.models.verifier == PRIMARY_VERIFIER
    assert report.models.draft == PRIMARY_DRAFT


def test_status_model_pair_unsupported_verifier(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "config.json").write_text(json.dumps({"model": "some/other"}), encoding="utf-8")
    report = build_status_report(now=1_000.0)
    assert report.models.configured is True
    assert report.models.draft == "unknown"
    assert "no profile matched" in report.models.detail
    payload = json.loads(report.to_json())
    profile = payload["profile"]
    assert isinstance(profile, dict)
    assert profile["status"] == "unprofiled"
    assert profile["name"] == "unprofiled"
    assert profile["verifier_model"] == "some/other"
    assert profile["trainable"] is False
    assert profile["tuning_available"] is False
    assert "no profile matched" in profile["detail"]
    assert "tuning is unavailable" in profile["detail"]


def test_status_resolves_profile_from_served_hf_snapshot_path(home: Path) -> None:
    home.mkdir(parents=True)
    served_model = (
        "/root/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-9B/snapshots/0123456789abcdef"
    )
    (home / "gateway.json").write_text(
        json.dumps({"pid": os.getpid(), "model": served_model}),
        encoding="utf-8",
    )

    payload = json.loads(build_status_report(now=1_000.0).to_json())

    assert payload["profile"]["status"] == "profiled"
    assert payload["profile"]["name"] == "qwen3.5-9b-mtp"
    assert payload["profile"]["verifier_model"] == "Qwen/Qwen3.5-9B"
    assert payload["profile"]["speculative_method"] == "mtp"
    assert payload["profile"]["chat_template_kind"] == "chatml"


# ---------------------------------------------------------------------------
# gain — nothing measured
# ---------------------------------------------------------------------------


def test_gain_no_benchmark_ever_run(home: Path) -> None:
    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.NO_GATE_RUN
    assert report.decision is None
    assert "No gate has ever run" in text
    assert "%" not in text
    assert "tok/s" not in text


def test_gain_no_benchmark_json_shape(home: Path) -> None:
    payload = json.loads(build_gain_report(now=1_000.0).to_json())
    assert payload["status"] == "no_gate_run"
    assert payload["verdict"] is None
    assert payload["reason"] is None
    assert payload["measurement"] is None
    assert payload["per_repeat"] == []
    assert payload["deltas_measured"] is False


def test_gain_unreadable_decision(home: Path) -> None:
    run_dir = home / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "decision.json").write_text("{oops", encoding="utf-8")

    report = build_gain_report(now=1_000.0)
    assert report.status is GainStatus.UNREADABLE
    assert report.decision is None
    assert "could not be read" in report.render_text()


# ---------------------------------------------------------------------------
# gain — real decisions
# ---------------------------------------------------------------------------


def test_gain_promote_decision(home: Path) -> None:
    _write_decision(home, _decision_dict())
    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.MEASURED
    assert report.deltas_measured is True
    assert report.decision is not None
    assert report.decision.verdict is Verdict.PROMOTE
    assert "verdict           : promote" in text
    assert "both_thresholds_met" in text
    assert "+6.00 pp" in text
    assert "+12.00%" in text
    assert "per-repeat:" in text
    assert "[0] stock 100.00 tok/s" in text
    assert "promoted on measured gains" in text


def test_gain_reject_decision_includes_reason(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="acceptance_below_threshold",
            acceptance_delta_pp=0.2,
            throughput_delta_pct=5.0,
        ),
    )
    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.MEASURED
    assert report.decision is not None
    assert report.decision.verdict is Verdict.REJECT
    assert report.decision.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD
    assert "verdict           : reject" in text
    assert "acceptance_below_threshold" in text
    assert "+0.20 pp" in text
    assert "rejected because" in text


def test_gain_acceptance_unavailable_prints_no_number(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="acceptance_unavailable",
            acceptance_delta_pp=0.0,
            throughput_delta_pct=0.0,
            num_repeats=0,
            per_repeat=[],
            stock_acc=0.0,
            cand_acc=0.0,
            stock_tps=0.0,
            cand_tps=0.0,
        ),
    )
    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.NOT_MEASURED
    assert report.acceptance_available is False
    assert report.deltas_measured is False
    assert "acceptance        : not measured" in text
    assert "speedup           : not measured" in text
    assert "UNAVAILABLE" in text
    # No fabricated zero speedup anywhere.
    assert "0.00%" not in text
    assert "+0.00" not in text
    assert "tok/s" not in text


def test_gain_counter_reset_prints_no_number(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="counter_reset",
            acceptance_delta_pp=0.0,
            throughput_delta_pct=0.0,
            num_repeats=0,
            per_repeat=[],
        ),
    )
    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.NOT_MEASURED
    assert report.counter_reset is True
    assert report.deltas_measured is False
    assert "counter_reset" in text
    assert "counter reset invalidated the benchmark" in text
    assert "speedup           : not measured" in text
    assert "tok/s" not in text
    assert "+0.00" not in text


def test_gain_too_few_repeats_is_not_measured(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="too_few_repeats",
            acceptance_delta_pp=0.0,
            throughput_delta_pct=0.0,
            num_repeats=1,
            per_repeat=[],
        ),
    )
    report = build_gain_report(now=1_000.0)
    assert report.status is GainStatus.NOT_MEASURED
    assert report.deltas_measured is False
    assert "not measured" in report.render_text()


def test_gain_json_shape_measured(home: Path) -> None:
    _write_decision(home, _decision_dict())
    payload = json.loads(build_gain_report(now=1_000.0).to_json())

    assert payload["status"] == "measured"
    assert payload["verdict"] == "promote"
    assert payload["reason"] == "both_thresholds_met"
    assert payload["deltas_measured"] is True
    assert payload["acceptance_available"] is True
    assert payload["counter_reset"] is False
    assert payload["measurement"]["throughput_delta_pct"] == 12.0
    assert payload["thresholds"]["min_acceptance_delta_pp"] == 1.0
    assert len(payload["per_repeat"]) == 3


def test_gain_json_omits_measurement_when_unavailable(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="acceptance_unavailable",
            acceptance_delta_pp=0.0,
            throughput_delta_pct=0.0,
            num_repeats=0,
            per_repeat=[],
        ),
    )
    payload = json.loads(build_gain_report(now=1_000.0).to_json())

    assert payload["status"] == "not_measured"
    assert payload["measurement"] is None
    assert payload["per_repeat"] == []
    assert payload["acceptance_available"] is False


def test_gain_accepts_new_unmeasured_decision_with_null_deltas(home: Path) -> None:
    path = _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="counter_reset",
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            num_repeats=0,
            per_repeat=[],
        ),
    )

    decision = load_decision(path)
    assert decision.acceptance_delta_pp is None
    assert decision.throughput_delta_pct is None
    assert decision.to_dict()["acceptance_delta_pp"] is None
    assert decision.to_dict()["throughput_delta_pct"] is None

    report = build_gain_report(now=1_000.0)
    assert report.status is GainStatus.NOT_MEASURED
    assert report.to_dict()["measurement"] is None
    assert "not measured" in report.render_text()


def test_gain_reports_zero_stock_throughput_as_unmeasured(home: Path) -> None:
    _write_decision(
        home,
        _decision_dict(
            verdict="reject",
            reason="throughput_unavailable",
            acceptance_delta_pp=None,
            throughput_delta_pct=None,
            num_repeats=3,
        ),
    )

    report = build_gain_report(now=1_000.0)

    assert report.status is GainStatus.NOT_MEASURED
    assert report.to_dict()["measurement"] is None
    assert "no measurable throughput" in report.render_text()


# ---------------------------------------------------------------------------
# decision discovery / parsing
# ---------------------------------------------------------------------------


def test_find_latest_decision_uses_explicit_run_not_newest_mtime(home: Path) -> None:
    current = _write_decision(home, _decision_dict(), run="run-a")
    foreign = _write_decision(
        home, _decision_dict(verdict="reject", reason="uncertain"), run="run-b"
    )
    os.utime(current, (1_000, 1_000))
    os.utime(foreign, (2_000, 2_000))
    (home / "runs" / "state.json").write_text(
        json.dumps(
            {
                "state": "READY",
                "sequence": 1,
                "updated_at": 2_001.0,
                "reason": "candidate serving",
                "run_id": "run-a",
            }
        ),
        encoding="utf-8",
    )

    layout = resolve_layout()
    assert find_latest_decision(layout) == current
    assert build_gain_report(now=1_000.0).source_path == current


def test_foreign_decision_is_ignored_when_known_run_has_none(home: Path) -> None:
    _write_decision(home, _decision_dict(), run="foreign-fixture")
    (home / "runs" / "state.json").write_text(
        json.dumps(
            {
                "state": "READY",
                "sequence": 1,
                "updated_at": 2_001.0,
                "reason": "cycle failed before gate",
                "run_id": "current-run",
            }
        ),
        encoding="utf-8",
    )

    assert find_latest_decision(resolve_layout()) is None
    report = build_gain_report(now=2_001.0)
    assert report.status is GainStatus.NO_GATE_RUN
    assert "No gate has ever run" in report.render_text()


def test_find_latest_decision_absent(home: Path) -> None:
    assert find_latest_decision(resolve_layout()) is None


def test_load_decision_roundtrip(home: Path) -> None:
    path = _write_decision(home, _decision_dict())
    decision = load_decision(path)
    assert decision.to_dict() == _decision_dict()


# ---------------------------------------------------------------------------
# gain — provenance validation (Defect 2 regression)
# ---------------------------------------------------------------------------


def test_gain_inconsistent_per_repeat_is_unreadable(home: Path) -> None:
    """A decision with num_repeats != len(per_repeat) is untrusted."""
    payload = _decision_dict(num_repeats=3, per_repeat=[
        {
            "repeat_index": 0,
            "stock_tok_per_sec": 100.0,
            "candidate_tok_per_sec": 112.0,
            "stock_acceptance_rate": 0.62,
            "candidate_acceptance_rate": 0.68,
            "invalid_rate": 0.0,
            "output_mismatches": 0,
        },
    ])
    _write_decision(home, payload)
    report = build_gain_report(now=1_000.0)
    assert report.status is GainStatus.UNREADABLE
    assert report.decision is None
    assert "inconsistent provenance" in report.render_text()


def test_gain_decision_shows_mtime(home: Path) -> None:
    """The decision file mtime must appear in the rendered text."""
    _write_decision(home, _decision_dict())
    report = build_gain_report(now=1_000.0)
    text = report.render_text()
    assert "source mtime" in text
    assert report.source_mtime is not None


def test_gain_decision_to_dict_includes_mtime(home: Path) -> None:
    """The JSON output must contain source_mtime."""
    _write_decision(home, _decision_dict())
    payload = json.loads(build_gain_report(now=1_000.0).to_json())
    assert "source_mtime" in payload
    assert payload["source_mtime"] is not None
