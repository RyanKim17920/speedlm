from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from speedlm.gate.decide import DispersionBasis, Reason, Verdict
from speedlm.report import (
    GainStatus,
    GatewayState,
    build_gain_report,
    build_status_report,
    find_latest_decision,
    load_decision,
    parse_decision,
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
    warmup_repeats: int = 1,
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
        "warmup_repeats": warmup_repeats,
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

    loaded = decision.to_dict()
    original = _decision_dict()
    assert {k: loaded[k] for k in original} == original


def test_loading_a_pre_pinned_decision_labels_the_statistic_it_gated_on(
    home: Path,
) -> None:
    """A record with no ``throughput_statistic`` gated on the Prometheus window.

    Its ``throughput_delta_pct`` and ``*_avg_tok_per_sec`` came from that
    window, so they are mirrored into the Prometheus fields and labelled as
    such rather than being reattributed to the statistic that gates today.
    """
    path = _write_decision(home, _decision_dict())
    decision = load_decision(path)

    assert decision.throughput_statistic == "prometheus_decode_window"
    assert decision.prometheus_throughput_delta_pct == 12.0
    assert decision.stock_prometheus_decode_tok_per_sec == 100.0
    assert decision.candidate_prometheus_decode_tok_per_sec == 112.0


def test_decision_written_by_todays_gate_round_trips_its_statistic(
    home: Path,
) -> None:
    """A record carrying both statistics keeps them distinct across a reload."""
    payload = _decision_dict()
    payload["throughput_statistic"] = "replay_per_repeat_mean"
    payload["prometheus_throughput_delta_pct"] = -3.1805
    payload["stock_prometheus_decode_tok_per_sec"] = 83.3062
    payload["candidate_prometheus_decode_tok_per_sec"] = 80.6567
    path = _write_decision(home, payload)

    decision = load_decision(path)

    assert decision.throughput_statistic == "replay_per_repeat_mean"
    assert decision.throughput_delta_pct == 12.0
    assert decision.prometheus_throughput_delta_pct == -3.1805
    assert decision.throughput_statistic_gap_pp == pytest.approx(15.1805, abs=1e-4)
    assert decision.to_dict()["throughput_statistic"] == "replay_per_repeat_mean"


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


def test_decision_without_warmup_field_reads_back_as_zero(home: Path) -> None:
    payload = _decision_dict()
    del payload["warmup_repeats"]
    path = _write_decision(home, payload)

    decision = load_decision(path)

    assert decision.warmup_repeats == 0


def test_measurement_context_survives_a_decision_round_trip(home: Path) -> None:
    """The knobs a cross-run comparison needs must reload as themselves."""
    payload = _decision_dict()
    payload["benchmark_max_tokens"] = 512
    payload["replay_concurrency"] = 8
    payload["correctness_max_tokens"] = 128
    payload["suite_hash"] = "abc123"
    payload["num_contexts"] = 103
    payload["stock_draft"] = "/runs/artifacts/incumbent"
    path = _write_decision(home, payload)

    decision = load_decision(path)

    assert decision.benchmark_max_tokens == 512
    assert decision.replay_concurrency == 8
    assert decision.correctness_max_tokens == 128
    assert decision.suite_hash == "abc123"
    assert decision.num_contexts == 103
    assert decision.stock_draft == "/runs/artifacts/incumbent"
    reloaded = decision.to_dict()
    assert {key: reloaded[key] for key in payload} == payload


def test_a_decision_predating_the_measurement_context_stays_readable(
    home: Path,
) -> None:
    """Absent is ``None``, never zero.

    Zero would claim an unbatched, uncapped run, which is a measurement the
    archive never made.  The schema is additive precisely so archived records
    keep loading; this pins that they do, and that they say so honestly.
    """
    path = _write_decision(home, _decision_dict())

    decision = load_decision(path)

    assert decision.benchmark_max_tokens is None
    assert decision.replay_concurrency is None
    assert decision.correctness_max_tokens is None
    assert decision.suite_hash is None
    assert decision.num_contexts is None
    assert decision.stock_draft is None
    assert build_gain_report(now=2_001.0).status is GainStatus.MEASURED


# ---------------------------------------------------------------------------
# status — serving-unrestored is an incident, not a status
# ---------------------------------------------------------------------------


def _write_scheduler(home_dir: Path, **overrides: Any) -> Path:
    runs = home_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "enabled": True,
        "lifecycle": "running",
        "last_result": {"outcome": "rolled_back", "error": None},
        "last_error": None,
    }
    record.update(overrides)
    path = runs / "scheduler.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_status_shouts_when_the_scheduler_reports_serving_unrestored(
    home: Path,
) -> None:
    _write_scheduler(home, serving_unrestored=True)

    report = build_status_report(now=1_000.0)
    payload = report.to_dict()["scheduler"]
    text = report.render_text()

    assert report.scheduler.serving_unrestored is True
    assert isinstance(payload, dict)
    assert payload["serving_unrestored"] is True
    assert "SERVING    : NOT RESTORED" in text
    assert "the active pointer does not name" in text


def test_status_stays_quiet_when_serving_was_restored(home: Path) -> None:
    _write_scheduler(home, serving_unrestored=False)

    report = build_status_report(now=1_000.0)
    text = report.render_text()

    assert report.scheduler.serving_unrestored is False
    assert "NOT RESTORED" not in text


def test_a_scheduler_record_predating_the_field_reads_back_as_unknown(
    home: Path,
) -> None:
    """Absent is ``None``, not ``False``.

    A scheduler that never reported the condition is not evidence that the
    condition is absent, and flattening the two would let an old record
    reassure a reader about something it never checked.
    """
    _write_scheduler(home)

    report = build_status_report(now=1_000.0)
    payload = report.to_dict()["scheduler"]

    assert report.scheduler.serving_unrestored is None
    assert isinstance(payload, dict)
    assert payload["serving_unrestored"] is None
    assert "NOT RESTORED" not in report.render_text()


def test_a_non_boolean_serving_unrestored_is_not_believed(home: Path) -> None:
    _write_scheduler(home, serving_unrestored="yes")

    report = build_status_report(now=1_000.0)

    assert report.scheduler.serving_unrestored is None


# ---------------------------------------------------------------------------
# gain — the quoted gain may not be the gain being delivered
# ---------------------------------------------------------------------------


def _write_unrestored_marker(home_dir: Path, payload: object) -> Path:
    runs = home_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "serving-unrestored.json"
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_gain_banners_a_measured_gain_that_is_not_being_delivered(
    home: Path,
) -> None:
    """A perfectly good measurement still gets the banner.

    The measurement is not what is wrong -- what is wrong is that the engine is
    not serving the draft it was measured for -- so the status stays
    ``MEASURED`` and the incident is reported beside it.
    """
    _write_decision(home, _decision_dict())
    _write_unrestored_marker(
        home,
        {
            "schema_version": 1,
            "detected_at": 950.0,
            "expected_active_draft": "artifact-7",
            "error": "runtime restore failed: engine did not come back",
        },
    )

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.status is GainStatus.MEASURED
    assert report.serving_unrestored is not None
    assert report.serving_unrestored.expected_active_draft == "artifact-7"
    assert "SERVING NOT RESTORED" in text
    assert "is NOT being delivered" in text
    assert "artifact-7" in text
    assert "engine did not come back" in text
    # The banner precedes every number it qualifies.
    assert text.index("SERVING NOT RESTORED") < text.index("acceptance delta")


def test_gain_banner_json_shape(home: Path) -> None:
    _write_decision(home, _decision_dict())
    _write_unrestored_marker(
        home,
        {"schema_version": 1, "detected_at": 950.0, "expected_active_draft": "a-7"},
    )

    payload = json.loads(build_gain_report(now=1_000.0).to_json())

    assert payload["status"] == "measured"
    assert payload["serving_unrestored"]["expected_active_draft"] == "a-7"
    assert payload["serving_unrestored"]["detected_at"] == 950.0
    assert payload["serving_unrestored"]["error"] is None


def test_gain_banner_survives_an_unreadable_marker(home: Path) -> None:
    """Presence is the signal; a corrupt payload must not silence it."""
    _write_decision(home, _decision_dict())
    _write_unrestored_marker(home, "{truncated")

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.serving_unrestored is not None
    assert report.serving_unrestored.expected_active_draft is None
    assert "SERVING NOT RESTORED" in text
    assert "the active pointer names unknown" in text


def test_gain_banners_even_when_nothing_was_ever_measured(home: Path) -> None:
    _write_unrestored_marker(
        home, {"schema_version": 1, "expected_active_draft": "artifact-7"}
    )

    report = build_gain_report(now=1_000.0)

    assert report.status is GainStatus.NO_GATE_RUN
    assert "SERVING NOT RESTORED" in report.render_text()


def test_gain_has_no_banner_when_serving_is_sound(home: Path) -> None:
    _write_decision(home, _decision_dict())

    report = build_gain_report(now=1_000.0)
    payload = json.loads(report.to_json())

    assert report.serving_unrestored is None
    assert payload["serving_unrestored"] is None
    assert "SERVING NOT RESTORED" not in report.render_text()


# ---------------------------------------------------------------------------
# gain — dispersion must not read as a tight measurement
# ---------------------------------------------------------------------------


def _varying_repeats(count: int) -> list[dict[str, Any]]:
    return [
        {
            "repeat_index": i,
            "stock_tok_per_sec": 100.0 + i,
            "candidate_tok_per_sec": 112.0 - i,
            "stock_acceptance_rate": 0.62 + i / 1000.0,
            "candidate_acceptance_rate": 0.68 - i / 1000.0,
            "invalid_rate": 0.0,
            "output_mismatches": 0,
        }
        for i in range(count)
    ]


def test_gain_degenerate_dispersion_never_prints_a_bare_delta(home: Path) -> None:
    """Five bit-identical repeats are no measurement, and must not read as one.

    This is job 369162's shape: greedy replay over a frozen suite returned the
    same counters every repeat, so the acceptance stdev was exactly ``0.0``.
    Printing ``-0.57 pp`` alone would be read as a superbly resolved number.
    """
    _write_decision(home, _decision_dict(num_repeats=5))

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.decision is not None
    assert (
        "acceptance delta  : +6.00 pp "
        "(no variance observed across 5 identical repeats; "
        "threshold >= 1.00 pp)" in text
    )
    # No standard error is offered where none exists.
    assert "+6.00 pp +/-" not in text


def test_gain_measured_dispersion_prints_the_standard_error(home: Path) -> None:
    _write_decision(
        home, _decision_dict(num_repeats=4, per_repeat=_varying_repeats(4))
    )

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.decision is not None
    assert "acceptance delta  : +6.00 pp +/- " in text
    assert "(n=4; threshold >= 1.00 pp)" in text
    assert "throughput delta  : +12.00% +/- " in text
    assert "(n=4; threshold >= 2.00%)" in text
    assert "no variance observed" not in text


def test_gain_unsampled_dispersion_says_there_was_nothing_to_disperse(
    home: Path,
) -> None:
    _write_decision(
        home, _decision_dict(num_repeats=1, per_repeat=_varying_repeats(1))
    )

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert "acceptance delta  : +6.00 pp (no dispersion: 1 repeat(s);" in text
    assert "throughput delta  : +12.00% (no dispersion: 1 repeat(s);" in text


def test_gain_json_publishes_a_null_standard_error_when_degenerate(
    home: Path,
) -> None:
    """``null``, never ``0.0``: a consumer dividing by it must fail loudly."""
    _write_decision(home, _decision_dict(num_repeats=5))

    payload = json.loads(build_gain_report(now=1_000.0).to_json())
    measurement = payload["measurement"]

    assert measurement["acceptance_dispersion"] == "degenerate"
    assert measurement["acceptance_delta_standard_error_pp"] is None
    assert measurement["throughput_dispersion"] == "degenerate"
    assert measurement["throughput_delta_standard_error_pct"] is None


def test_gain_json_publishes_the_standard_error_when_measured(home: Path) -> None:
    _write_decision(
        home, _decision_dict(num_repeats=4, per_repeat=_varying_repeats(4))
    )

    measurement = json.loads(build_gain_report(now=1_000.0).to_json())["measurement"]

    assert measurement["acceptance_dispersion"] == "measured"
    assert measurement["acceptance_delta_standard_error_pp"] > 0.0
    assert measurement["throughput_dispersion"] == "measured"
    assert measurement["throughput_delta_standard_error_pct"] > 0.0


# ---------------------------------------------------------------------------
# gain — measurement context
# ---------------------------------------------------------------------------


def test_gain_renders_the_context_two_runs_must_share_to_be_comparable(
    home: Path,
) -> None:
    payload = _decision_dict()
    payload.update(
        {
            "suite_hash": "e945cac6e677",
            "num_contexts": 103,
            "stock_draft": "RedHatAI/Qwen3-8B-speculator.eagle3",
            "benchmark_max_tokens": 512,
            "replay_concurrency": 8,
            "correctness_max_tokens": 128,
        }
    )
    _write_decision(home, payload)

    report = build_gain_report(now=1_000.0)
    text = report.render_text()
    context = json.loads(report.to_json())["measurement_context"]

    assert "suite             : e945cac6e677 (103 contexts)" in text
    assert "baseline draft    : RedHatAI/Qwen3-8B-speculator.eagle3" in text
    assert "replay            : concurrency 8, max 512 tokens" in text
    # Bounds only the separate correctness replay, so it is machine-readable
    # rather than spent on a line the operator does not need.
    assert "correctness" not in text
    assert context["correctness_max_tokens"] == 128


def test_gain_context_is_reported_for_an_aborted_decision_too(home: Path) -> None:
    """It describes the attempt, not the result."""
    payload = _decision_dict(
        verdict="reject",
        reason="high_invalid_rate",
        acceptance_delta_pp=None,
        throughput_delta_pct=None,
    )
    payload["suite_hash"] = "e945cac6e677"
    _write_decision(home, payload)

    report = build_gain_report(now=1_000.0)

    assert report.deltas_measured is False
    assert "suite             : e945cac6e677" in report.render_text()


def test_gain_omits_context_lines_a_legacy_decision_never_recorded(
    home: Path,
) -> None:
    _write_decision(home, _decision_dict())

    report = build_gain_report(now=1_000.0)
    text = report.render_text()
    context = json.loads(report.to_json())["measurement_context"]

    assert "suite             :" not in text
    assert "baseline draft    :" not in text
    assert "replay            :" not in text
    assert set(context.values()) == {None}


# ---------------------------------------------------------------------------
# gain — archived artifacts
# ---------------------------------------------------------------------------

#: Real gate decisions from completed GPU runs.  They predate every field added
#: above, which is exactly why they are the test: the dispersion figures are
#: *derived* from ``per_repeat``, so an archived record must report the same
#: basis the gate would have, without carrying any of the new keys.
_ARCHIVE_ROOT = Path("/data/ryan.kim/speedlm-runs")
_ARCHIVED_DECISIONS = sorted(
    _ARCHIVE_ROOT.glob(
        "*/results/live-idle-tuning/speedlm_home/runs/*/decision.json"
    )
)


@pytest.mark.skipif(
    not _ARCHIVED_DECISIONS, reason="no archived gate decisions on this host"
)
@pytest.mark.parametrize(
    "path", _ARCHIVED_DECISIONS, ids=lambda p: p.parent.name[:8]
)
def test_archived_decisions_round_trip_without_the_new_keys(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    assert "acceptance_dispersion" not in record, (
        "this fixture is only meaningful while it predates the field"
    )

    decision = load_decision(path)

    # Derived, so they exist even though the file never stored them.
    assert decision.acceptance_dispersion in set(DispersionBasis)
    assert decision.throughput_dispersion in set(DispersionBasis)
    # And the invariant that motivated the enum: no zero standard errors.
    for basis, error in (
        (decision.acceptance_dispersion, decision.acceptance_delta_standard_error_pp),
        (
            decision.throughput_dispersion,
            decision.throughput_delta_standard_error_pct,
        ),
    ):
        if basis is DispersionBasis.MEASURED:
            assert error is None or error > 0.0
        else:
            assert error is None

    # Re-serialising and re-parsing must be a fixed point.
    assert parse_decision(decision.to_dict(), source=path).to_dict() == decision.to_dict()


@pytest.mark.skipif(
    not _ARCHIVED_DECISIONS, reason="no archived gate decisions on this host"
)
@pytest.mark.parametrize(
    "path", _ARCHIVED_DECISIONS, ids=lambda p: p.parent.name[:8]
)
def test_archived_homes_render_a_gain_report(path: Path) -> None:
    archived_home = path.parent.parent.parent
    report = build_gain_report(home=archived_home, now=1_000.0)
    text = report.render_text()

    assert report.status in {GainStatus.MEASURED, GainStatus.NOT_MEASURED}
    assert "SERVING NOT RESTORED" not in text
    if report.deltas_measured:
        assert "acceptance delta  :" in text
        # Whatever the basis, the delta is qualified -- never bare.
        for label in ("acceptance delta  :", "throughput delta  :"):
            line = next(ln for ln in text.splitlines() if ln.startswith(label))
            assert "no variance observed" in line or "+/-" in line
