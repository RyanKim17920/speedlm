"""Tests for speedlm.workload — measured corpus shape versus declared config."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from speedlm.config import IdleTuningConfig, SpeedLMConfig, WorkloadConfig
from speedlm.traces.store import TraceStats
from speedlm.tuner.orchestrator import CycleOutcome, CycleResult
from speedlm.tuner.service import TunerService
from speedlm.workload import (
    COMPLETION_CAP_HEADROOM_FRACTION,
    MATERIAL_FRACTION,
    WorkloadAssessment,
    WorkloadShape,
    evaluate_workload,
    format_workload_findings,
)

VERIFIER = "openai/gpt-oss-20b"


def _shape(**overrides: object) -> WorkloadShape:
    """A plain single-turn chat corpus: the shape every default assumes."""
    base = WorkloadShape(
        records=100,
        multi_turn_records=0,
        tool_schema_records=0,
        tool_call_records=0,
        system_prompt_records=0,
        assistant_history_records=0,
        client_supplied_assistant_records=0,
        truncated_completion_records=0,
        training_dropped_records=0,
        prompt_tokens_p50=117,
        prompt_tokens_p95=250,
        prompt_tokens_max=800,
        completion_tokens_p50=64,
        completion_tokens_p95=200,
        completion_tokens_max=400,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _evaluate(
    shape: WorkloadShape,
    *,
    workload: WorkloadConfig | None = None,
    sequence_length: int = 16_384,
    benchmark_max_tokens: int = 512,
) -> dict[str, str]:
    """Findings keyed by code, with the declaration defaulted to 'declared'.

    ``domain`` is set by default so the always-on undeclared-domain finding
    does not have to be filtered out of every assertion; the tests that care
    about it pass an empty domain explicitly.
    """
    findings = evaluate_workload(
        shape,
        workload=workload if workload is not None else WorkloadConfig(domain="chat"),
        sequence_length=sequence_length,
        benchmark_max_tokens=benchmark_max_tokens,
    )
    return {finding.code: finding.render() for finding in findings}


# ---------------------------------------------------------------------------
# The corpus the defaults were built for
# ---------------------------------------------------------------------------


def test_the_calibration_corpus_produces_no_findings() -> None:
    """Single-turn, tool-free chat under stock bounds must stay silent.

    If this fires, every finding below is noise and operators will learn to
    ignore the report.
    """
    assert _evaluate(_shape()) == {}


def test_an_empty_corpus_produces_no_findings() -> None:
    """Nothing captured is not a mismatch; it is nothing to say."""
    assert (
        evaluate_workload(
            _shape(records=0, prompt_tokens_p50=None, prompt_tokens_p95=None),
            workload=WorkloadConfig(),
            sequence_length=16_384,
            benchmark_max_tokens=512,
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Material fraction
# ---------------------------------------------------------------------------


def test_a_single_stray_request_is_not_a_workload() -> None:
    """One tool-carrying probe out of 100 must not rewrite the domain."""
    assert "tool_schemas_present" not in _evaluate(_shape(tool_schema_records=1))


def test_tool_schemas_at_the_material_fraction_are_reported() -> None:
    material = int(100 * MATERIAL_FRACTION)
    findings = _evaluate(_shape(tool_schema_records=material, tool_call_records=3))
    assert "tool_schemas_present" in findings
    report = findings["tool_schemas_present"]
    assert "tool schemas" in report
    assert "sequence_length" in report and "benchmark_max_tokens" in report


def test_material_fraction_is_relative_to_the_corpus() -> None:
    """Two records out of ten is material; two out of a thousand is not."""
    assert "tool_schemas_present" in _evaluate(
        _shape(records=10, tool_schema_records=2)
    )
    assert "tool_schemas_present" not in _evaluate(
        _shape(records=1000, tool_schema_records=2)
    )


# ---------------------------------------------------------------------------
# Silent degradation modes
# ---------------------------------------------------------------------------


def test_client_supplied_assistant_turns_are_reported_as_a_training_loss() -> None:
    findings = _evaluate(
        _shape(
            multi_turn_records=40,
            client_supplied_assistant_records=40,
            training_dropped_records=40,
        )
    )
    report = findings["client_supplied_assistant_turns_dropped"]
    assert "40 of 100" in report
    assert "drops" in report
    assert "trust_untagged_assistant_messages" in report


def test_p95_prompt_above_sequence_length_is_reported() -> None:
    findings = _evaluate(
        _shape(prompt_tokens_p95=20_000, prompt_tokens_max=32_000),
        sequence_length=16_384,
    )
    report = findings["prompt_tokens_exceed_sequence_length"]
    assert "20000" in report
    assert "tuning.sequence_length=16384" in report


def test_p95_prompt_at_the_bound_is_not_reported() -> None:
    """The bound is inclusive: fitting exactly is fitting."""
    assert "prompt_tokens_exceed_sequence_length" not in _evaluate(
        _shape(prompt_tokens_p95=16_384), sequence_length=16_384
    )


def test_completions_near_the_benchmark_cap_are_reported() -> None:
    """Near the cap, not merely over it: the distribution is already clipped."""
    near = math.ceil(512 * COMPLETION_CAP_HEADROOM_FRACTION)
    findings = _evaluate(
        _shape(completion_tokens_p50=near, truncated_completion_records=9),
        benchmark_max_tokens=512,
    )
    report = findings["completions_near_benchmark_cap"]
    assert "tuning.benchmark_max_tokens=512" in report
    assert "truncated" in report


def test_short_completions_leave_the_benchmark_cap_alone() -> None:
    assert "completions_near_benchmark_cap" not in _evaluate(
        _shape(completion_tokens_p50=64), benchmark_max_tokens=512
    )


def test_a_raised_cap_clears_the_completion_finding() -> None:
    """The remedy the finding names must actually silence it."""
    shape = _shape(completion_tokens_p50=500)
    assert "completions_near_benchmark_cap" in _evaluate(
        shape, benchmark_max_tokens=512
    )
    assert "completions_near_benchmark_cap" not in _evaluate(
        shape, benchmark_max_tokens=4096
    )


# ---------------------------------------------------------------------------
# Declared expectations
# ---------------------------------------------------------------------------


def test_declared_tool_free_traffic_that_carries_tools_is_reported() -> None:
    findings = _evaluate(
        _shape(tool_schema_records=30),
        workload=WorkloadConfig(domain="chat", expect_tools=False),
    )
    report = findings["declared_shape_present"]
    assert "workload.expect_tools=false" in report
    assert "30 of 100" in report


def test_declared_tools_that_never_arrive_are_reported() -> None:
    """The capture path may be stripping what the operator thinks it sees."""
    findings = _evaluate(
        _shape(tool_schema_records=0),
        workload=WorkloadConfig(domain="agentic", expect_tools=True),
    )
    assert "declared_shape_absent" in findings
    assert "workload.expect_tools=true" in findings["declared_shape_absent"]


def test_declared_tool_workload_resolves_the_defaults_mismatch() -> None:
    findings = _evaluate(
        _shape(tool_schema_records=95, tool_call_records=60),
        workload=WorkloadConfig(domain="agentic-coding", expect_tools=True),
    )
    assert "tool_schemas_present" not in findings


def test_declared_client_assistant_history_resolves_the_expected_loss_warning() -> None:
    findings = _evaluate(
        _shape(
            multi_turn_records=90,
            client_supplied_assistant_records=70,
            training_dropped_records=70,
        ),
        workload=WorkloadConfig(
            domain="agentic-coding",
            expect_client_supplied_assistant_turns=True,
        ),
    )
    assert "client_supplied_assistant_turns_dropped" not in findings


def test_an_undeclared_expectation_makes_no_claim() -> None:
    """None must never be read as False."""
    findings = _evaluate(
        _shape(tool_schema_records=30, system_prompt_records=100),
        workload=WorkloadConfig(domain="chat"),
    )
    assert "declared_shape_present" not in findings
    assert "declared_shape_absent" not in findings


def test_a_correct_declaration_is_silent() -> None:
    assert (
        _evaluate(
            _shape(system_prompt_records=100),
            workload=WorkloadConfig(domain="chat", expect_system_prompts=True),
        )
        == {}
    )


@pytest.mark.parametrize(
    ("field_name", "count_name"),
    [
        ("expect_tools", "tool_schema_records"),
        ("expect_multi_turn", "multi_turn_records"),
        ("expect_system_prompts", "system_prompt_records"),
        (
            "expect_client_supplied_assistant_turns",
            "client_supplied_assistant_records",
        ),
    ],
)
def test_every_declaration_is_actually_checked(
    field_name: str, count_name: str
) -> None:
    """A declaration nothing checks is the decorative knob this must not be."""
    findings = _evaluate(
        _shape(**{count_name: 50}),
        workload=WorkloadConfig(domain="chat", **{field_name: False}),
    )
    assert "declared_shape_present" in findings
    assert f"workload.{field_name}=false" in findings["declared_shape_present"]


def test_an_undeclared_domain_is_itself_a_finding() -> None:
    findings = _evaluate(_shape(), workload=WorkloadConfig())
    assert "workload_domain_undeclared" in findings
    assert "workload.domain" in findings["workload_domain_undeclared"]


def test_a_declared_domain_removes_the_undeclared_finding() -> None:
    assert "workload_domain_undeclared" not in _evaluate(
        _shape(), workload=WorkloadConfig(domain="retrieval")
    )


# ---------------------------------------------------------------------------
# Finding shape
# ---------------------------------------------------------------------------


def test_every_finding_names_measurement_contradiction_and_remedy() -> None:
    """A finding that omits any of the three cannot be acted on."""
    findings = evaluate_workload(
        _shape(
            tool_schema_records=50,
            client_supplied_assistant_records=50,
            training_dropped_records=50,
            prompt_tokens_p95=99_999,
            completion_tokens_p50=511,
        ),
        workload=WorkloadConfig(expect_tools=False, expect_multi_turn=True),
        sequence_length=16_384,
        benchmark_max_tokens=512,
    )
    assert len(findings) >= 5
    for finding in findings:
        assert finding.code
        assert finding.severity in {"warning", "info"}
        assert finding.measured.strip()
        assert finding.contradicts.strip()
        assert finding.remedy.strip()
        # The serialized form is what reaches the cycle record, so it must
        # carry all four parts, not merely the object in memory.
        payload = finding.to_dict()
        assert payload == {
            "code": finding.code,
            "severity": finding.severity,
            "measured": finding.measured,
            "contradicts": finding.contradicts,
            "remedy": finding.remedy,
        }
        assert all(value.strip() for value in payload.values())


def test_findings_render_as_one_named_block() -> None:
    findings = evaluate_workload(
        _shape(tool_schema_records=50),
        workload=WorkloadConfig(domain="chat"),
        sequence_length=16_384,
        benchmark_max_tokens=512,
    )
    rendered = format_workload_findings(findings)
    assert "1 finding(s)" in rendered
    assert "tool_schemas_present" in rendered
    assert "no mismatch" in format_workload_findings(())


# ---------------------------------------------------------------------------
# Shape projection
# ---------------------------------------------------------------------------


def test_shape_carries_every_measured_counter_from_stats() -> None:
    stats = TraceStats(
        count=7,
        tokens=70,
        oldest=1.0,
        newest=2.0,
        multi_turn_records=1,
        tool_schema_records=2,
        tool_call_records=3,
        system_prompt_records=4,
        assistant_history_records=5,
        client_supplied_assistant_records=6,
        truncated_completion_records=7,
        training_dropped_records=7,
        prompt_tokens_p50=11,
        prompt_tokens_p95=22,
        prompt_tokens_max=33,
        completion_tokens_p50=44,
        completion_tokens_p95=55,
        completion_tokens_max=66,
    )
    shape = WorkloadShape.from_stats(stats)
    assert shape.records == 7
    assert shape.multi_turn_records == 1
    assert shape.tool_schema_records == 2
    assert shape.tool_call_records == 3
    assert shape.system_prompt_records == 4
    assert shape.assistant_history_records == 5
    assert shape.client_supplied_assistant_records == 6
    assert shape.truncated_completion_records == 7
    assert shape.prompt_tokens_p95 == 22
    assert shape.completion_tokens_p95 == 55
    assert shape.to_dict()["prompt_tokens_max"] == 33


def test_assessment_records_the_domain_and_the_findings() -> None:
    stats = TraceStats(
        count=100,
        tokens=1000,
        oldest=1.0,
        newest=2.0,
        tool_schema_records=100,
    )
    assessment = WorkloadAssessment.measure(
        stats,
        workload=WorkloadConfig(domain="agentic", expect_tools=False),
        sequence_length=16_384,
        benchmark_max_tokens=512,
    )
    payload = assessment.to_dict()
    assert payload["domain"] == "agentic"
    assert payload["shape"]["tool_schema_records"] == 100
    codes = {finding["code"] for finding in payload["findings"]}
    assert {"tool_schemas_present", "declared_shape_present"} <= codes
    # The record must be JSON, because it is written to scheduler.json.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Wiring: every cycle records the domain it trained on
# ---------------------------------------------------------------------------


class _IdleActivity:
    in_flight = 0
    last_activity = 0.0


@dataclass
class _AcceptingRunner:
    calls: int = 0

    def run_once(self) -> CycleResult:
        self.calls += 1
        return CycleResult(CycleOutcome.REJECTED)

    def recover(self) -> tuple[str, ...]:
        return ()


class _ShapedTraces:
    """A trace source whose corpus shape the test dictates."""

    def __init__(self, stats: TraceStats) -> None:
        self._stats = stats

    def stats(self) -> TraceStats:
        return self._stats

    def prune(self) -> int:
        return 0


def _service(tmp_path: Path, stats: TraceStats, workload: WorkloadConfig) -> TunerService:
    config = SpeedLMConfig(
        model=VERIFIER,
        idle_threshold_seconds=0.01,
        workload=workload,
        tuning=IdleTuningConfig(idle_confirmations=1, benchmark_max_tokens=512),
    )
    return TunerService(
        config,
        activity=_IdleActivity(),
        traces=_ShapedTraces(stats),
        orchestrator_factory=lambda _activity: _AcceptingRunner(),
        enabled=True,
        min_trace_records=2,
        min_corpus_records=2,
        clock=lambda: 1_000_000.0,
        status_path=tmp_path / "scheduler.json",
    )


def _agentic_stats() -> TraceStats:
    return TraceStats(
        count=100,
        tokens=100_000,
        oldest=1.0,
        newest=2.0,
        multi_turn_records=90,
        tool_schema_records=95,
        tool_call_records=60,
        system_prompt_records=100,
        assistant_history_records=80,
        client_supplied_assistant_records=70,
        truncated_completion_records=40,
        training_dropped_records=75,
        prompt_tokens_p50=4_000,
        prompt_tokens_p95=40_000,
        prompt_tokens_max=90_000,
        completion_tokens_p50=500,
        completion_tokens_p95=512,
        completion_tokens_max=512,
    )


def test_a_cycle_records_the_domain_it_trained_on(tmp_path: Path) -> None:
    """scheduler.json must say what the corpus was, not just how big it was."""
    service = _service(
        tmp_path, _agentic_stats(), WorkloadConfig(domain="agentic-tool-dispatch")
    )
    service._poll_once()

    payload = json.loads((tmp_path / "scheduler.json").read_text(encoding="utf-8"))
    recorded = payload["workload"]
    assert recorded["domain"] == "agentic-tool-dispatch"
    assert recorded["shape"]["tool_schema_records"] == 95
    assert recorded["shape"]["prompt_tokens_p95"] == 40_000
    codes = {finding["code"] for finding in recorded["findings"]}
    assert "tool_schemas_present" in codes
    assert "client_supplied_assistant_turns_dropped" in codes
    assert "prompt_tokens_exceed_sequence_length" in codes
    assert "completions_near_benchmark_cap" in codes


def test_the_watermark_dedupe_key_is_not_widened(tmp_path: Path) -> None:
    """Shape rides alongside the watermark, never inside its equality.

    ``_TraceWatermark`` is the scheduler's "have I already tried this buffer"
    key. Folding shape into it would make a re-measured corpus look like new
    work and re-arm cycles that were already attempted.
    """
    service = _service(tmp_path, _agentic_stats(), WorkloadConfig(domain="agentic"))
    service._poll_once()

    payload = json.loads((tmp_path / "scheduler.json").read_text(encoding="utf-8"))
    assert set(payload["last_watermark"]) == {
        "count",
        "tokens",
        "oldest",
        "newest",
        "unknown_token_records",
    }


def test_findings_are_reported_but_do_not_block_the_cycle(tmp_path: Path) -> None:
    """A mismatched corpus is still tuned on; the gate remains the judge."""
    service = _service(tmp_path, _agentic_stats(), WorkloadConfig(domain="agentic"))
    runner = service.orchestrator
    assert isinstance(runner, _AcceptingRunner)

    service._poll_once()

    assert runner.calls == 1
    payload = json.loads((tmp_path / "scheduler.json").read_text(encoding="utf-8"))
    assert payload["workload"]["findings"]
    assert payload["last_result"]["outcome"] == "rejected"


def test_mismatches_are_logged_by_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Recorded is not enough: the operator must see it in the log too."""
    service = _service(tmp_path, _agentic_stats(), WorkloadConfig(domain="agentic"))
    with caplog.at_level("WARNING", logger="speedlm.tuner.service"):
        service._poll_once()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "workload mismatch" in messages
    assert "tool_schemas_present" in messages


def test_a_clean_corpus_records_a_shape_with_no_findings(tmp_path: Path) -> None:
    """The record is unconditional: silence must still be written down."""
    stats = TraceStats(
        count=100,
        tokens=10_000,
        oldest=1.0,
        newest=2.0,
        prompt_tokens_p50=117,
        prompt_tokens_p95=250,
        prompt_tokens_max=800,
        completion_tokens_p50=64,
        completion_tokens_p95=200,
        completion_tokens_max=400,
    )
    service = _service(tmp_path, stats, WorkloadConfig(domain="generic-chat"))
    service._poll_once()

    recorded = json.loads((tmp_path / "scheduler.json").read_text(encoding="utf-8"))[
        "workload"
    ]
    assert recorded["domain"] == "generic-chat"
    assert recorded["findings"] == []
    assert recorded["shape"]["records"] == 100
