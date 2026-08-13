from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from speedlm.gate.decide import (
    DIVERGENCE_ALPHA,
    DIVERGENCE_STATISTICS,
    DispersionBasis,
    Reason,
    TruncationRegime,
    Verdict,
)
from speedlm.report import (
    GainReport,
    GainStatus,
    GatewayState,
    ReportError,
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
    draft_depth: int = 5,
    min_accepted_length_delta: float = 0.05,
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
                # Derived through *draft_depth* rather than pinned, because the
                # two acceptance columns are related by
                # ``acceptance_rate == (mean_accepted_length - 1) / k``.
                "stock_accepted_length": 1.0 + stock_acc * draft_depth,
                "candidate_accepted_length": 1.0 + cand_acc * draft_depth,
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
        # Accepted-length criterion fields (added in GAP 1).
        # Derived from acceptance rates via the same relation:
        # mean_accepted_length = 1 + acceptance_rate * draft_depth.
        "accepted_length_delta": (cand_acc - stock_acc) * draft_depth,
        "min_accepted_length_delta": min_accepted_length_delta,
        "stock_avg_accepted_length": 1.0 + stock_acc * draft_depth,
        "candidate_avg_accepted_length": 1.0 + cand_acc * draft_depth,
        "acceptance_criterion": "mean_accepted_length_delta",
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


def _vetoed_stationarity() -> dict[str, Any]:
    """A stationarity block whose evidence vetoes an otherwise-promotion."""
    return {
        "testable": True,
        "stationary": False,
        "required_for_promotion": True,
        "min_repeats": 4,
        "delta_shift_pct": -14.2,
        "delta_shift_t_statistic": 9.4,
        "min_shift_t_statistic": 4.0,
        "materiality_pct": 2.0,
        "stock_flat_from_repeat": None,
        "candidate_flat_from_repeat": None,
        "stock_trend_pct_per_repeat": -0.27,
        "candidate_trend_pct_per_repeat": 3.69,
        "status": "non_stationary",
        "vetoed": True,
    }


def test_gain_reports_a_vetoed_promotion_as_a_rejection(home: Path) -> None:
    """``speedlm gain`` must not announce a promotion that was rolled back.

    This is the shape bigcycle-run1 wrote: ``verdict: promote``,
    ``reason: both_thresholds_met``, and a non-stationary throughput veto that
    sent the cycle to ``ROLLING_BACK``.  The veto is applied after
    ``decide_promotion`` returns, so the headline verdict on the record was the
    threshold comparison rather than the outcome.
    """
    payload = _decision_dict()
    payload["throughput_stationarity"] = _vetoed_stationarity()
    _write_decision(home, payload)

    report = build_gain_report(now=1_000.0)
    rendered = json.loads(report.to_json())

    assert report.decision is not None
    assert report.decision.verdict is Verdict.PROMOTE
    assert report.decision.final_verdict is Verdict.REJECT
    assert rendered["verdict"] == "reject"
    assert rendered["threshold_verdict"] == "promote"
    assert rendered["vetoed"] is True
    text = report.render_text()
    assert "verdict           : reject" in text
    assert "throughput_not_stationary" in text
    # The one sentence a skimming operator reads must not claim a promotion.
    assert "was promoted" not in text
    assert "was NOT promoted" in text


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
    assert {
        **{k: loaded[k] for k in original if k != "per_repeat"},
        "per_repeat": [
            {k2: row[k2] for k2 in original["per_repeat"][0]}
            for row in loaded["per_repeat"]
        ],
    } == original


def test_archived_per_repeat_row_lacking_truncation_fields_reads_as_untestable(
    home: Path,
) -> None:
    """An archived per_repeat row whose four new truncation columns are absent
    must parse to zeros, and the Decision must classify both arms as
    ``TruncationRegime.UNTESTABLE`` -- not as ``BOUNDED``.

    The distinction is critical: ``UNTESTABLE`` says "no finish reasons were
    recorded, so we cannot tell"; ``BOUNDED`` says "finish reasons were
    recorded and most stopped naturally".  An archived record that never
    persisted the counts must never be silently relabelled as "measured, and
    truncation was low".
    """
    payload = _decision_dict()
    assert "stock_finish_reasons" not in payload["per_repeat"][0]
    path = _write_decision(home, payload)

    decision = load_decision(path)

    for row in decision.per_repeat:
        assert row.stock_finish_reasons == 0
        assert row.candidate_finish_reasons == 0
        assert row.stock_truncated == 0
        assert row.candidate_truncated == 0

    assert (
        decision.stock_truncation_regime is TruncationRegime.UNTESTABLE
    ), "stock arm must be UNTESTABLE, not BOUNDED"
    assert (
        decision.candidate_truncation_regime is TruncationRegime.UNTESTABLE
    ), "candidate arm must be UNTESTABLE, not BOUNDED"
    assert decision.stock_truncation_rate is None
    assert decision.candidate_truncation_rate is None
    assert decision.truncation_rate_delta is None


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
    assert {
        **{key: reloaded[key] for key in payload if key != "per_repeat"},
        "per_repeat": [
            {k2: row[k2] for k2 in payload["per_repeat"][0]}
            for row in reloaded["per_repeat"]
        ],
    } == payload


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
            # Accepted-length fields varying with the acceptance rate.
            "stock_accepted_length": 1.0 + (0.62 + i / 1000.0) * 5,
            "candidate_accepted_length": 1.0 + (0.68 - i / 1000.0) * 5,
        }
        for i in range(count)
    ]


def test_gain_names_the_bar_that_actually_gated(home: Path) -> None:
    """`speedlm gain` must quote the criterion's threshold, not a dead one.

    The acceptance-rate bar used to be printed as ``threshold >= 1.00 pp``
    beside the rate delta.  It no longer gates -- the rate divides by the draft
    depth -- so printing it there would tell a reader the gate applied a bar it
    did not.  The gating bar is printed against the statistic it applies to.
    """
    payload = _decision_dict()
    payload["accepted_length_delta"] = 0.3
    payload["min_accepted_length_delta"] = 0.05
    payload["stock_avg_accepted_length"] = 4.1
    payload["candidate_avg_accepted_length"] = 4.4
    payload["acceptance_criterion"] = "mean_accepted_length_delta"
    _write_decision(home, payload)

    text = build_gain_report(now=1_000.0).render_text()

    assert "accepted len stock: 4.100 tok/step" in text
    assert "accepted len cand : 4.400 tok/step" in text
    assert "accepted len delta: +0.300 tok/step" in text
    assert "threshold >= 0.050 tok/step" in text
    # And the rate delta is still shown, explicitly as a non-gating figure.
    assert "acceptance delta  : +6.00 pp" in text
    assert "recorded, not gated" in text
    assert "threshold >= 1.00 pp" not in text


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
    # The acceptance delta shows degenerate dispersion qualifier.
    assert (
        "acceptance delta  : +6.00 pp "
        "(no variance observed across 5 identical repeats; "
        "recorded, not gated)" in text
    )
    # No standard error is offered for degenerate dispersion.
    assert "+6.00 pp +/-" not in text
    # The accepted-length block also shows degenerate dispersion.
    assert "accepted len delta:" in text
    assert "tok/step (no variance observed across 5 identical repeats;" in text


def test_gain_measured_dispersion_prints_the_standard_error(home: Path) -> None:
    _write_decision(
        home, _decision_dict(num_repeats=4, per_repeat=_varying_repeats(4))
    )

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.decision is not None
    assert "acceptance delta  : +6.00 pp +/- " in text
    assert "(n=4; recorded, not gated)" in text
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

#: Real gate decisions from completed GPU runs.  The older ones predate every
#: field added above, which is exactly why they are the test: the dispersion
#: figures are *derived* from ``per_repeat``, so an archived record must report
#: the same basis the gate would have, without carrying any of the new keys.
#: Newer runs land in the same tree and do carry them; both are replayed.
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
    # The archive now holds records from both sides of the field's
    # introduction, so this branches on the record instead of asserting the
    # tree stays legacy -- which it does not.  Job 369373 wrote a modern
    # decision.json into the same glob and turned this into a red suite about
    # the fixture rather than about the code.  Both sides are worth replaying:
    # a legacy record proves the dispersion figures are *derived*, a modern one
    # proves the stored basis survives the round trip unchanged.
    legacy = "acceptance_dispersion" not in record

    decision = load_decision(path)

    # Derived, so they exist even though a legacy file never stored them.
    assert decision.acceptance_dispersion in set(DispersionBasis)
    assert decision.throughput_dispersion in set(DispersionBasis)
    if not legacy:
        assert decision.acceptance_dispersion.value == record["acceptance_dispersion"]
        assert decision.throughput_dispersion.value == record["throughput_dispersion"]
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


# ---------------------------------------------------------------------------
# gain — to_dict() JSON output includes accepted-length fields (GAP 1)
# ---------------------------------------------------------------------------


def test_gain_json_thresholds_includes_min_accepted_length_delta(
    home: Path,
) -> None:
    """to_dict() thresholds must expose min_accepted_length_delta."""
    payload = _decision_dict()
    payload["accepted_length_delta"] = 0.3
    payload["min_accepted_length_delta"] = 0.05
    payload["stock_avg_accepted_length"] = 4.1
    payload["candidate_avg_accepted_length"] = 4.4
    payload["acceptance_criterion"] = "mean_accepted_length_delta"
    _write_decision(home, payload)

    result = build_gain_report(now=1_000.0).to_dict()

    assert "min_accepted_length_delta" in result["thresholds"]
    assert result["thresholds"]["min_accepted_length_delta"] == 0.05


def test_gain_json_measurement_includes_accepted_length_block(
    home: Path,
) -> None:
    """to_dict() measurement block must carry accepted-length stats when deltas
    are measured."""
    payload = _decision_dict()
    payload["accepted_length_delta"] = 0.3
    payload["min_accepted_length_delta"] = 0.05
    payload["stock_avg_accepted_length"] = 4.1
    payload["candidate_avg_accepted_length"] = 4.4
    payload["acceptance_criterion"] = "mean_accepted_length_delta"
    _write_decision(home, payload)

    result = build_gain_report(now=1_000.0).to_dict()
    measurement = result["measurement"]

    assert measurement is not None
    assert measurement["acceptance_criterion"] == "mean_accepted_length_delta"
    assert measurement["accepted_length_delta"] == 0.3
    assert measurement["stock_accepted_length"] == 4.1
    assert measurement["candidate_accepted_length"] == 4.4
    assert "accepted_length_dispersion" in measurement
    assert "accepted_length_delta_standard_error" in measurement


def test_gain_json_per_repeat_rows_include_accepted_length(
    home: Path,
) -> None:
    """to_dict() per_repeat rows must carry stock/candidate accepted_length."""
    payload = _decision_dict(num_repeats=2)
    payload["accepted_length_delta"] = 0.3
    payload["min_accepted_length_delta"] = 0.05
    payload["stock_avg_accepted_length"] = 4.1
    payload["candidate_avg_accepted_length"] = 4.4
    _write_decision(home, payload)

    result = build_gain_report(now=1_000.0).to_dict()

    for row in result["per_repeat"]:
        assert "stock_accepted_length" in row
        assert "candidate_accepted_length" in row
        # _decision_dict derives these from acceptance_rate:
        # 1.0 + rate * draft_depth == 1.0 + 0.62 * 5 == 4.1
        assert row["stock_accepted_length"] == pytest.approx(4.1)
        assert row["candidate_accepted_length"] == pytest.approx(4.4)


# ---------------------------------------------------------------------------
# gain -- deltas_measured recognises EITHER acceptance statistic
#
# The gate's promotion criterion moved from the acceptance rate to the mean
# accepted length.  Requiring *both* deltas made every decision.json written
# before that move -- which is every archived run on disk -- render as "not
# measured", silently destroying the readout of the whole archive.  These pin
# the either-not-both predicate.
# ---------------------------------------------------------------------------


def _legacy_decision_dict() -> dict[str, Any]:
    """A decision exactly as the gate wrote it before the criterion moved.

    Every accepted-length key is absent, at the top level and per repeat --
    which is what the records under ``/data/ryan.kim/speedlm-runs`` actually
    look like.
    """
    payload = _decision_dict()
    for key in (
        "accepted_length_delta",
        "min_accepted_length_delta",
        "stock_avg_accepted_length",
        "candidate_avg_accepted_length",
        "acceptance_criterion",
    ):
        del payload[key]
    for row in payload["per_repeat"]:
        del row["stock_accepted_length"]
        del row["candidate_accepted_length"]
    return payload


def test_legacy_decision_without_accepted_length_renders_acceptance_delta(
    home: Path,
) -> None:
    """REGRESSION GUARD.  A record carrying ``acceptance_delta_pp`` and no
    ``accepted_length_delta`` must print its acceptance delta, not "not
    measured".  This is the exact shape of every archived run on disk."""
    payload = _legacy_decision_dict()
    assert "accepted_length_delta" not in payload
    _write_decision(home, payload)

    report = build_gain_report(now=1_000.0)
    text = report.render_text()

    assert report.deltas_measured is True
    assert "acceptance delta  : +6.00 pp" in text
    assert "throughput delta  : +12.00%" in text
    assert "not measured" not in text
    # It carries no accepted-length statistic, so it must claim none.
    assert "accepted len" not in text


def test_legacy_decision_json_reports_deltas_measured(home: Path) -> None:
    """The JSON readout of a legacy record must agree with the text one."""
    _write_decision(home, _legacy_decision_dict())

    result = build_gain_report(now=1_000.0).to_dict()

    assert result["deltas_measured"] is True
    assert result["measurement"] is not None
    assert result["measurement"]["acceptance_delta_pp"] == pytest.approx(6.0)
    # Absent, never a measured zero.
    assert result["measurement"]["accepted_length_delta"] is None
    assert result["thresholds"]["min_accepted_length_delta"] is None


def test_deltas_measured_true_on_accepted_length_alone() -> None:
    """The mirror case: only the new statistic present."""
    decision = replace(
        parse_decision(_decision_dict(), source=Path("d.json")),
        acceptance_delta_pp=None,
    )
    report = GainReport(
        status=GainStatus.MEASURED, detail="Verdict: ok.", decision=decision
    )

    assert report.deltas_measured is True
    text = report.render_text()
    assert "accepted len delta: +0.300 tok/step" in text
    # No rate delta was recorded, so none may be printed.
    assert "acceptance delta" not in text
    assert "acceptance stock" not in text


def test_deltas_measured_false_when_neither_acceptance_statistic_present() -> None:
    """Either-not-both is not "anything goes": with no acceptance statistic at
    all there is nothing to report."""
    decision = replace(
        parse_decision(_decision_dict(), source=Path("d.json")),
        acceptance_delta_pp=None,
        accepted_length_delta=None,
    )
    report = GainReport(
        status=GainStatus.MEASURED, detail="Verdict: ok.", decision=decision
    )

    assert report.deltas_measured is False
    assert "acceptance        : not measured" in report.render_text()


def test_deltas_measured_false_without_throughput_delta() -> None:
    """Throughput has no second statistic to fall back on, so it stays
    mandatory even when an acceptance statistic is present."""
    decision = replace(
        parse_decision(_decision_dict(), source=Path("d.json")),
        throughput_delta_pct=None,
    )
    report = GainReport(
        status=GainStatus.MEASURED, detail="Verdict: ok.", decision=decision
    )

    assert report.deltas_measured is False


# ---------------------------------------------------------------------------
# gain -- the divergence noise floor round-trips and is surfaced
#
# The gate rejects on divergence only when the candidate-versus-stock count
# significantly exceeds a stock-versus-stock floor measured in the same run.
# ``parse_decision`` rebuilds a Decision field by explicit field, so those
# fields have to be named there or the floor never reaches ``speedlm gain``.
# ---------------------------------------------------------------------------


def _divergence_entry(
    *, index: int, early: bool, context: str = "ctx-a", repeat: int = 0
) -> dict[str, Any]:
    return {
        "context_hash": context,
        "repeat_index": repeat,
        "first_divergence_index": index,
        "basis": "token_mismatch",
        "stock_length": 128,
        "candidate_length": 128,
        "early": early,
    }


def _decision_with_control() -> dict[str, Any]:
    """A divergence rejection carrying its measured noise floor."""
    payload = _decision_dict(
        verdict="reject",
        reason="output_mismatch",
        acceptance_delta_pp=None,
        throughput_delta_pct=None,
    )
    payload["min_divergence_token_index"] = 16
    payload["output_divergences"] = [
        _divergence_entry(index=9, early=True, context="ctx-1"),
        _divergence_entry(index=80, early=False, context="ctx-2"),
    ]
    payload["control_divergences"] = [
        _divergence_entry(index=4, early=True, context="ctx-3"),
        _divergence_entry(index=61, early=False, context="ctx-4"),
        _divergence_entry(index=99, early=False, context="ctx-5"),
    ]
    payload["divergence_trials"] = 410
    payload["control_trials"] = 410
    payload["divergence_control_available"] = True
    payload["divergence_total_p_value"] = 0.42
    payload["divergence_early_p_value"] = 0.61
    # Deliberately NOT the default DIVERGENCE_ALPHA/DIVERGENCE_STATISTICS, so a
    # hardcoded or defaulted alpha cannot pass for a round-tripped one.
    payload["divergence_alpha"] = 0.0025
    return payload


def test_parse_decision_round_trips_the_divergence_noise_floor() -> None:
    decision = parse_decision(_decision_with_control(), source=Path("d.json"))

    assert decision.divergence_trials == 410
    assert decision.control_trials == 410
    assert decision.divergence_control_available is True
    assert len(decision.control_divergences) == 3
    assert decision.control_total_divergences == 3
    assert decision.control_early_divergences == 1
    assert decision.control_divergence_rate == pytest.approx(3 / 410)
    assert decision.divergence_rate == pytest.approx(2 / 410)
    assert decision.divergence_total_p_value == pytest.approx(0.42)
    assert decision.divergence_early_p_value == pytest.approx(0.61)
    assert decision.divergence_alpha == pytest.approx(0.0025)
    # The control's own evidence survives, not just its summary counts.
    assert decision.control_divergences[0].context_hash == "ctx-3"
    assert decision.control_divergences[0].first_divergence_index == 4


def test_parse_decision_round_trips_the_schedule_and_the_stationarity_verdict() -> None:
    """How the measurement was taken has to survive onto disk and back.

    ``parse_decision`` rebuilds a Decision field by explicit field, so a field
    the gate persists but the parser does not name is a field ``speedlm gain``
    and every archived-record reader silently loses.
    """
    payload = _decision_dict()
    payload["arm_blocks"] = 2
    payload["block_schedule"] = [
        {"arm": "candidate", "repeats": 3, "restarted": False},
        {"arm": "stock", "repeats": 3, "restarted": True},
        {"arm": "stock", "repeats": 2, "restarted": True},
        {"arm": "candidate", "repeats": 2, "restarted": True},
    ]
    payload["throughput_stationarity"] = {
        "testable": True,
        "min_repeats": 4,
        "delta_shift_pct": 2.5,
        "delta_shift_t_statistic": 9.75,
        "min_shift_t_statistic": 4.0,
        "materiality_pct": 2.0,
        "stock_flat_from_repeat": 0,
        "candidate_flat_from_repeat": None,
        "stock_trend_pct_per_repeat": 1.25,
        "candidate_trend_pct_per_repeat": None,
        "stationary": False,
        "required_for_promotion": True,
    }

    decision = parse_decision(payload, source=Path("d.json"))

    assert decision.arm_blocks == 2
    # The realized order, not a re-derivation of it: ABBA, and the one block
    # that reused a running engine was the first.
    assert [b.arm for b in decision.block_schedule] == [
        "candidate",
        "stock",
        "stock",
        "candidate",
    ]
    assert [b.restarted for b in decision.block_schedule] == [False, True, True, True]
    assert [b.repeats for b in decision.block_schedule] == [3, 3, 2, 2]
    stationarity = decision.throughput_stationarity
    assert stationarity is not None
    assert stationarity.stationary is False
    assert stationarity.testable is True
    assert stationarity.delta_shift_pct == pytest.approx(2.5)
    assert stationarity.delta_shift_t_statistic == pytest.approx(9.75)
    assert stationarity.materiality_pct == pytest.approx(2.0)
    assert stationarity.candidate_flat_from_repeat is None
    # Re-serialising is a fixed point, which is what makes the record readable
    # by the next parse rather than only by this one.
    assert parse_decision(decision.to_dict(), source=Path("d.json")).to_dict() == (
        decision.to_dict()
    )


def test_a_record_without_a_schedule_never_reads_as_a_measured_one() -> None:
    """Absence is "not recorded", not "one block, stationary"."""
    decision = parse_decision(_decision_dict(), source=Path("d.json"))

    assert decision.arm_blocks is None
    assert decision.block_schedule == ()
    assert decision.throughput_stationarity is None


def test_absent_divergence_control_never_reads_as_a_measured_zero() -> None:
    """A record predating the control must come back saying "no control ran",
    not "the engine diverged zero times"."""
    decision = parse_decision(_decision_dict(), source=Path("d.json"))

    assert decision.divergence_control_available is False
    assert decision.control_divergences == ()
    assert decision.control_trials == 0
    assert decision.divergence_trials == 0
    # Rates are None -- never 0.0, which would claim a perfectly deterministic
    # engine was measured.
    assert decision.control_divergence_rate is None
    assert decision.divergence_rate is None
    assert decision.divergence_total_p_value is None
    assert decision.divergence_early_p_value is None
    # No alpha was recorded, so the criterion's own per-statistic level is
    # what the record reads back as -- not zero, which would be no test at all.
    assert decision.divergence_alpha == pytest.approx(
        DIVERGENCE_ALPHA / DIVERGENCE_STATISTICS
    )


def test_gain_text_shows_the_noise_floor_beside_the_candidate_rate(
    home: Path,
) -> None:
    """An operator reading a rejection must see the engine's own divergence
    rate; without it the candidate's count means nothing."""
    _write_decision(home, _decision_with_control())

    text = build_gain_report(now=1_000.0).render_text()

    assert "divergence rate   : candidate 2/410 (0.49% of context comparisons)" in text
    assert (
        "engine noise floor: stock vs stock 3/410 (0.73%), 1 before token 16" in text
    )
    assert "divergence test   : total p 0.4200, early p 0.6100" in text
    assert "alpha 0.0025" in text


def test_gain_text_says_so_when_the_floor_was_only_assumed(home: Path) -> None:
    payload = _decision_with_control()
    payload["divergence_control_available"] = False
    payload["control_divergences"] = []
    payload["control_trials"] = 0
    _write_decision(home, payload)

    text = build_gain_report(now=1_000.0).render_text()

    assert "divergence rate   : candidate 2/410" in text
    assert "engine noise floor: no stock-vs-stock control ran" in text
    assert "assumed a zero-divergence floor" in text
    # "not measured" is this report's phrase for a statistic with no numbers at
    # all; the divergence counts here ARE measured.
    assert "engine noise floor: not measured" not in text


def test_legacy_divergence_record_claims_no_floor_at_all(home: Path) -> None:
    """A record from before the criterion existed was judged against no floor,
    so the report must not describe one -- assumed or otherwise."""
    payload = _legacy_decision_dict()
    payload["verdict"] = "reject"
    payload["reason"] = "output_mismatch"
    payload["acceptance_delta_pp"] = None
    payload["throughput_delta_pct"] = None
    payload["min_divergence_token_index"] = 16
    payload["output_divergences"] = [_divergence_entry(index=9, early=True)]
    _write_decision(home, payload)

    text = build_gain_report(now=1_000.0).render_text()

    # The evidence it does carry is still shown.
    assert "divergences       : 1 contexts parted" in text
    assert "engine noise floor" not in text
    assert "divergence rate" not in text
    assert "divergence test" not in text


def test_gain_json_exposes_the_divergence_floor_on_an_unmeasured_reject(
    home: Path,
) -> None:
    """``output_mismatch`` has no speed deltas at all, so the divergence block
    must live outside the ``deltas_measured`` branch -- it is the entire basis
    of that verdict."""
    _write_decision(home, _decision_with_control())

    result = build_gain_report(now=1_000.0).to_dict()
    divergence = result["divergence"]

    assert result["deltas_measured"] is False
    assert result["measurement"] is None
    assert divergence["candidate_total"] == 2
    assert divergence["candidate_early"] == 1
    assert divergence["candidate_trials"] == 410
    assert divergence["candidate_rate"] == pytest.approx(2 / 410)
    assert divergence["control_available"] is True
    assert divergence["control_total"] == 3
    assert divergence["control_early"] == 1
    assert divergence["control_trials"] == 410
    assert divergence["control_rate"] == pytest.approx(3 / 410)
    assert divergence["total_p_value"] == pytest.approx(0.42)
    assert divergence["early_p_value"] == pytest.approx(0.61)
    assert divergence["alpha"] == pytest.approx(0.0025)
    assert len(divergence["control_divergences"]) == 3


def test_gain_json_control_fields_are_null_when_no_control_ran(home: Path) -> None:
    _write_decision(home, _legacy_decision_dict())

    divergence = build_gain_report(now=1_000.0).to_dict()["divergence"]

    assert divergence["control_available"] is False
    for key in ("control_total", "control_early", "control_trials", "control_rate"):
        assert divergence[key] is None, key
    assert divergence["candidate_trials"] is None
    assert divergence["candidate_rate"] is None


def test_control_divergences_must_be_a_list() -> None:
    """The control array gets the same validation as the candidate one; it is
    the evidence the verdict turns on."""
    payload = _decision_with_control()
    payload["control_divergences"] = {"not": "a list"}

    with pytest.raises(ReportError, match="control_divergences"):
        parse_decision(payload, source=Path("d.json"))


# ---------------------------------------------------------------------------
# Every verdict the gate can write must survive its own reader
# ---------------------------------------------------------------------------


def _saturated_decision() -> Any:
    """A saturated ``Decision`` produced by the gate, not assembled by hand.

    Hand-building one would let this test pass against a ``Decision`` the gate
    can never actually emit -- and the defect was precisely that the gate's own
    output could not be read back.  So the record under test comes out of
    ``decide_promotion``, on an arm whose every generation was ended by the
    output cap.
    """
    from speedlm.config import PromotionConfig
    from speedlm.gate.decide import decide_promotion
    from speedlm.gate.metrics import MetricsDelta
    from speedlm.gate.replay import ReplayResult, RunResults

    def _delta(acceptance_rate: float, tok_per_sec: float) -> MetricsDelta:
        return MetricsDelta(
            reset_detected=False,
            acceptance_available=True,
            drafted_tokens=1000.0,
            accepted_tokens=700.0,
            acceptance_rate=acceptance_rate,
            mean_accepted_length=1.0 + acceptance_rate * 5,
            tpot_ms=10.0,
            output_tok_per_sec=tok_per_sec,
        )

    def _replay_arm(tok_per_sec: float, truncated: int) -> ReplayResult:
        runs = tuple(
            RunResults(
                results=(),
                total_latency_s=1000.0 / tok_per_sec,
                total_prompt_tokens=100,
                total_completion_tokens=1000,
                valid_count=5,
                invalid_count=0,
                invalid_rate=0.0,
                truncated_count=truncated,
                natural_stop_count=5 - truncated,
            )
            for _ in range(3)
        )
        return ReplayResult(run_results=runs, num_runs=len(runs), suite_hash="suite-h")

    return decide_promotion(
        _delta(0.60, 100.0),
        _delta(0.65, 110.0),
        _replay_arm(100.0, truncated=5),
        _replay_arm(110.0, truncated=1),
        PromotionConfig(
            min_acceptance_delta_pp=1.0,
            min_accepted_length_delta=0.05,
            min_throughput_delta_pct=2.0,
        ),
    )


def test_a_saturated_decision_round_trips_through_the_report_layer() -> None:
    """The gate must not be able to write a verdict its own reader rejects.

    ``TRUNCATION_SATURATED`` returns above the delta computation, exactly as
    ``OUTPUT_MISMATCH`` does, so the record it writes carries null deltas.
    While the reason was missing from ``UNMEASURED_REASONS`` the loader demanded
    numbers the gate had never computed and raised ``'acceptance_delta_pp' must
    be numeric`` on every saturated run -- so the one rejection reason that
    exists to say "this benchmark measured nothing" produced an artifact
    ``speedlm gain`` could not open.
    """
    decision = _saturated_decision()

    assert decision.verdict is Verdict.REJECT
    assert decision.reason is Reason.TRUNCATION_SATURATED
    assert decision.stock_truncation_regime is TruncationRegime.SATURATED
    # The shape that made the loader fail: the deltas really are absent.
    assert decision.to_dict()["acceptance_delta_pp"] is None

    parsed = parse_decision(decision.to_dict(), source=Path("d.json"))

    assert parsed.reason is Reason.TRUNCATION_SATURATED
    assert parsed.verdict is Verdict.REJECT
    assert parsed.to_dict() == decision.to_dict()


def test_a_saturated_decision_renders_a_gain_report(home: Path) -> None:
    """The other half of the same defect: three renderers index the mapping.

    A reason absent from ``_REASON_EXPLANATIONS`` is a ``KeyError`` waiting in
    the text renderer, the JSON renderer and the summary line, so loading the
    record back is not enough -- it has to be reportable.
    """
    _write_decision(home, _saturated_decision().to_dict())

    report = build_gain_report()

    assert report.status is GainStatus.NOT_MEASURED
    assert not report.deltas_measured
    # All three renderers index ``_REASON_EXPLANATIONS`` directly.
    assert "output cap" in report.to_dict()["reason_detail"]
    assert "output cap" in report.to_json()
    assert "output cap" in report.render_text()


def test_every_rejection_reason_carries_an_explanation() -> None:
    """Enumerated so the next reason added cannot repeat this defect.

    ``_REASON_EXPLANATIONS`` is indexed, never ``.get``-ed, in three renderers.
    Asserting over the enum rather than over a list the test maintains means a
    new ``Reason`` member fails here at once, instead of at whatever moment a
    live gate first writes it.
    """
    from speedlm.report import _REASON_EXPLANATIONS

    missing = [reason.name for reason in Reason if reason not in _REASON_EXPLANATIONS]

    assert missing == []
    assert all(_REASON_EXPLANATIONS[reason].strip() for reason in Reason)
