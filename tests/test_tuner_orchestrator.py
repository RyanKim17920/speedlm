from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.config import PromotionConfig
from speedlm.gate.decide import Decision, Reason, Verdict, decide_promotion
from speedlm.gate.metrics import MetricsDelta
from speedlm.gate.replay import ReplayResult, RequestResult, RunResults
from speedlm.gateway.control import DraftSwapCorrupted
from speedlm.gateway.control import RuntimeController as RealRuntimeController
from speedlm.report import GainStatus, build_gain_report, load_decision
from speedlm.training.base import BackendInfo, SpeculatorBackend
from speedlm.training.masking import FinalAssistantMaskError
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.eagle3 import (
    Eagle3Adapter,
    Eagle3Config,
    PreparedData,
    ScratchQuotaExceeded,
    TraceSnapshot,
    TrainingResult,
)
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    BENCHMARK_MAX_SECONDS,
    BENCHMARK_MIN_SECONDS,
    BENCHMARK_SAFETY_FACTOR,
    BENCHMARK_SECONDS_PER_TOKEN,
    DECISION_FILE_NAME,
    METRICS_DIR_NAME,
    SERVING_UNRESTORED_FILE_NAME,
    BenchmarkGate,
    CycleOutcome,
    DecisionPersistError,
    GateFailure,
    GateResult,
    OrchestratorTimeouts,
    TunerOrchestrator,
    derive_benchmark_timeout,
    write_decision,
    write_metrics_bodies,
)
from speedlm.tuner.state import TunerState, TunerStateMachine


@dataclass
class FakeActivity:
    in_flight: int = 0
    last_activity: float = 0.0


@dataclass
class FakeLeaser:
    def lease_snapshot(
        self,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> TraceSnapshot:
        destination.mkdir()
        (destination / "traces.jsonl").write_text("{}\n", encoding="utf-8")
        return TraceSnapshot(destination, "trace-hash")


@dataclass
class FakeRenderer:
    mask_error: bool = False
    bytes_to_write: int = 2

    def render_rows(
        self,
        snapshot: TraceSnapshot,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        if self.mask_error:
            raise FinalAssistantMaskError("no final assistant tokens")
        destination.mkdir()
        rows = destination / "rows.jsonl"
        rows.write_bytes(b"x" * self.bytes_to_write)
        return rows


@dataclass
class FakeExtractor:
    def extract_hidden_states(
        self,
        rows_path: Path,
        destination: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        destination.mkdir()
        output = destination / "hidden.pt"
        output.write_bytes(b"h")
        return output


@dataclass
class FakeTrainer:
    returncode: int = 0
    stderr: str = ""
    seen_from_pretrained: str | None = None

    def train(
        self,
        hidden_states_path: Path,
        destination: Path,
        *,
        from_pretrained: str,
        training_params: Mapping[str, object],
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> TrainingResult:
        self.seen_from_pretrained = from_pretrained
        destination.mkdir()
        checkpoint = destination / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"trained")
        return TrainingResult(checkpoint, self.returncode, self.stderr)


@dataclass
class FakeMaterializer:
    def materialize(
        self,
        checkpoint_best: Path,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        destination.mkdir()
        (destination / "config.json").write_text('{"model_type":"eagle3"}', encoding="utf-8")
        (destination / "weights.bin").write_bytes(
            (checkpoint_best / "weights.bin").read_bytes()
        )
        return destination


@dataclass
class FakeValidator:
    validated: list[Path] = field(default_factory=list)

    def validate(
        self,
        draft_directory: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        self.validated.append(draft_directory)


@dataclass
class FakeRuntime:
    activity: FakeActivity
    preempt_on_sleep: bool = False
    #: Make :meth:`restore` raise until it has been called this many times.
    #: ``None`` means "never fail"; a large number means "never recovers".
    fail_restores: int = 0
    calls: list[str] = field(default_factory=list)

    def quiesce(
        self, *, timeout_seconds: float, should_abort: Callable[[], bool]
    ) -> None:
        self.calls.append("quiesce")

    def sleep(
        self, *, timeout_seconds: float, should_abort: Callable[[], bool]
    ) -> None:
        self.calls.append("sleep")
        if self.preempt_on_sleep:
            self.activity.last_activity += 100.0

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        self.calls.append("start_candidate")

    def restore(self, active_draft: Path | str, *, timeout_seconds: float) -> None:
        self.calls.append(f"restore:{Path(active_draft).name}")
        if self.fail_restores > 0:
            self.fail_restores -= 1
            raise RuntimeError("simulated launch failure for the rollback")

    def wake(self, *, timeout_seconds: float) -> None:
        self.calls.append("wake")


@dataclass
class FakeGate:
    passed: bool
    decision: Decision | None = None
    metrics_bodies: dict[str, str] = field(default_factory=dict)
    failure: GateFailure | None = None
    #: Deadlines the orchestrator handed to :meth:`benchmark`, in order.
    seen_timeouts: list[float] = field(default_factory=list)

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        self.seen_timeouts.append(timeout_seconds)
        if self.failure is not None:
            return GateResult(
                False,
                f"benchmark {self.failure.value}",
                metrics={self.failure.value: True},
                metrics_bodies=dict(self.metrics_bodies),
                failure=self.failure,
            )
        return GateResult(
            self.passed,
            "thresholds met" if self.passed else "regression",
            metrics_bodies=dict(self.metrics_bodies),
            decision=self.decision,
        )


@dataclass
class FakeBackend:
    """Small structural implementation of the generalized backend protocol."""

    mask_error: bool = False
    training_error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def describe(self) -> BackendInfo:
        return BackendInfo(
            verifier_model="fake/verifier",
            draft_model="fake/draft",
            from_pretrained="fake/base-draft",
            training_params={"steps": 2},
        )

    def prepare(
        self,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> PreparedData:
        self.calls.append("prepare")
        if self.mask_error:
            raise FinalAssistantMaskError("no final assistant tokens")
        snapshot_path = work_dir / "trace-snapshot"
        snapshot_path.mkdir()
        rows_path = work_dir / "rows.jsonl"
        rows_path.write_text("{}\n", encoding="utf-8")
        return PreparedData(
            snapshot=TraceSnapshot(snapshot_path, "trace-hash"),
            rows_path=rows_path,
        )

    def extract(
        self,
        prepared: PreparedData,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        self.calls.append("extract")
        hidden_states_path = work_dir / "hidden.pt"
        hidden_states_path.write_bytes(b"hidden")
        return hidden_states_path

    def train(
        self,
        extracted: Path,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> TrainingResult:
        self.calls.append("train")
        if self.training_error is not None:
            raise self.training_error
        checkpoint = work_dir / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"trained")
        return TrainingResult(checkpoint, 0)

    def materialize(
        self,
        trained: TrainingResult,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        self.calls.append("materialize")
        draft_directory = work_dir / "draft-model"
        draft_directory.mkdir()
        (draft_directory / "config.json").write_text(
            '{"model_type":"fake"}',
            encoding="utf-8",
        )
        (draft_directory / "weights.bin").write_bytes(
            (trained.checkpoint_best / "weights.bin").read_bytes()
        )
        return draft_directory

    def validate(
        self,
        artifact: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> None:
        self.calls.append("validate")


# ---------------------------------------------------------------------------
# Real gate decisions (built through decide_promotion, not hand-rolled dicts)
# ---------------------------------------------------------------------------


def _run(
    *, completion_tokens: int, latency_s: float, response_text: str = "ok"
) -> RunResults:
    results = tuple(
        RequestResult(
            context_hash="abcd1234",
            latency_s=latency_s,
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            total_tokens=10 + completion_tokens,
            response_text=response_text,
            valid=True,
        )
        for _ in range(1)
    )
    return RunResults(
        results=results,
        total_latency_s=latency_s,
        total_prompt_tokens=10,
        total_completion_tokens=completion_tokens,
        valid_count=len(results),
        invalid_count=0,
        invalid_rate=0.0,
    )


def _replay(*, completion_tokens: int, latency_s: float, repeats: int = 3) -> ReplayResult:
    runs = [
        _run(completion_tokens=completion_tokens, latency_s=latency_s)
        for _ in range(repeats)
    ]
    return ReplayResult(
        run_results=tuple(runs), num_runs=len(runs), suite_hash="suite-hash"
    )


#: Draft depth these fixtures speak at.  ``mean_accepted_length`` below is
#: derived through it rather than pinned to a constant: the two acceptance
#: statistics are related by ``acceptance_rate == (mean_accepted_length - 1)/k``
#: and the gate's promotion criterion is now the accepted-length delta, so a
#: fixture holding the length fixed while sweeping the rate would describe an
#: impossible engine and hand the gate a zero delta on every arm.
_FIXTURE_DRAFT_DEPTH = 5


def _delta(*, acceptance_rate: float, output_tok_per_sec: float) -> MetricsDelta:
    return MetricsDelta(
        reset_detected=False,
        acceptance_available=True,
        drafted_tokens=1000.0,
        accepted_tokens=1000.0 * acceptance_rate,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=1.0 + acceptance_rate * _FIXTURE_DRAFT_DEPTH,
        tpot_ms=10.0,
        output_tok_per_sec=output_tok_per_sec,
    )


def _real_decision(*, promote: bool) -> Decision:
    """Produce a genuine :class:`Decision` from the real gate logic."""
    # The reject arm mirrors the real failure this gate exists to catch: a
    # candidate whose acceptance did not move (0.0 pp, below the 1.0 pp bar)
    # while throughput drifted up by an amount well inside timing noise.
    candidate_acceptance = 0.78 if promote else 0.60
    candidate_tps = 130.0 if promote else 101.0
    decision = decide_promotion(
        _delta(acceptance_rate=0.60, output_tok_per_sec=100.0),
        _delta(acceptance_rate=candidate_acceptance, output_tok_per_sec=candidate_tps),
        _replay(completion_tokens=100, latency_s=1.0),
        _replay(completion_tokens=130, latency_s=1.0),
        PromotionConfig(),
    )
    expected = Verdict.PROMOTE if promote else Verdict.REJECT
    assert decision.verdict is expected, decision
    return decision


def _real_throughput_regression_decision() -> Decision:
    """A genuine reject that trips the *throughput* guard, not the acceptance bar.

    Acceptance clears its bar comfortably (+18 pp) while throughput drops by
    the -19.2% actually observed on job 368648's un-warmed candidate arm, so
    only the regression guard can be responsible for the reject.
    """
    decision = decide_promotion(
        _delta(acceptance_rate=0.60, output_tok_per_sec=100.0),
        _delta(acceptance_rate=0.78, output_tok_per_sec=80.8),
        _replay(completion_tokens=100, latency_s=1.0),
        _replay(completion_tokens=81, latency_s=1.0),
        PromotionConfig(),
    )
    assert decision.verdict is Verdict.REJECT, decision
    return decision


def _eagle3_adapter(
    *,
    trainer: FakeTrainer | None = None,
    renderer: FakeRenderer | None = None,
    quota: int = 1024 * 1024,
) -> Eagle3Adapter:
    return Eagle3Adapter(
        Eagle3Config(
            verifier_model="openai/gpt-oss-20b",
            draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
            from_pretrained="RedHatAI/gpt-oss-20b-speculator.eagle3",
            training_params={"steps": 2},
            scratch_quota_bytes=quota,
        ),
        leaser=FakeLeaser(),
        renderer=renderer or FakeRenderer(),
        extractor=FakeExtractor(),
        trainer=trainer or FakeTrainer(),
        materializer=FakeMaterializer(),
        validator=FakeValidator(),
    )


def _orchestrator(
    tmp_path: Path,
    *,
    activity: FakeActivity | None = None,
    runtime: FakeRuntime | None = None,
    gate_passed: bool = True,
    backend: SpeculatorBackend | None = None,
    decision: Decision | None = None,
    metrics_bodies: dict[str, str] | None = None,
    work_root: Path | None = None,
    gate: BenchmarkGate | None = None,
    timeouts: OrchestratorTimeouts | None = None,
) -> tuple[TunerOrchestrator, TunerStateMachine, ArtifactRegistry, FakeRuntime]:
    activity = activity or FakeActivity()
    runtime = runtime or FakeRuntime(activity)
    state = TunerStateMachine(tmp_path / "state")
    artifacts = ArtifactRegistry(tmp_path / "registry")
    orchestrator = TunerOrchestrator(
        state=state,
        idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
        backend=backend or FakeBackend(),
        artifacts=artifacts,
        runtime=runtime,
        gate=gate or FakeGate(gate_passed, decision, metrics_bodies or {}),
        work_root=work_root or (tmp_path / "work"),
        run_id_factory=lambda: "run-1",
        **({"timeouts": timeouts} if timeouts is not None else {}),
    )
    return orchestrator, state, artifacts, runtime


def test_happy_path_reaches_promoting_then_ready(tmp_path: Path) -> None:
    backend = FakeBackend()
    orchestrator, state, artifacts, runtime = _orchestrator(
        tmp_path,
        backend=backend,
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    active = artifacts.active()
    assert active is not None
    assert active.manifest.base_draft == "fake/base-draft"
    assert backend.calls == [
        "prepare",
        "extract",
        "train",
        "materialize",
        "validate",
    ]
    events = state.events_path.read_text(encoding="utf-8")
    assert '"to": "PROMOTING"' in events
    assert runtime.calls == ["quiesce", "sleep", "start_candidate", "wake"]


def test_gate_rejection_rolls_back_without_changing_active(tmp_path: Path) -> None:
    orchestrator, state, artifacts, runtime = _orchestrator(
        tmp_path,
        gate_passed=False,
    )
    baseline_source = tmp_path / "baseline"
    baseline_source.mkdir()
    (baseline_source / "weights.bin").write_bytes(b"baseline")
    baseline = artifacts.publish(
        baseline_source,
        ArtifactSpec(
            verifier_model="openai/gpt-oss-20b",
            draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
            base_draft="RedHatAI/gpt-oss-20b-speculator.eagle3",
            trace_hash="old-traces",
            training_params={"steps": 1},
        ),
    )
    artifacts.promote(baseline.artifact_id, gate_passed=True)
    active_before = artifacts.active_path.read_bytes()

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.REJECTED
    assert state.state is TunerState.READY
    assert artifacts.active_path.read_bytes() == active_before
    assert artifacts.active().artifact_id == baseline.artifact_id  # type: ignore[union-attr]
    assert '"to": "ROLLING_BACK"' in state.events_path.read_text(encoding="utf-8")
    assert any(call.startswith("restore:") for call in runtime.calls)


def test_preemption_aborts_and_restores_serving(tmp_path: Path) -> None:
    activity = FakeActivity()
    runtime = FakeRuntime(activity, preempt_on_sleep=True)
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert state.state is TunerState.READY
    assert "start_candidate" not in runtime.calls
    assert runtime.calls[-1] == "wake"


def test_a_preemption_whose_restore_fails_is_not_reported_as_merely_preempted(
    tmp_path: Path,
) -> None:
    """The worst failure this system has, made observable.

    A preemption that reaches the rollback and cannot respawn leaves the engine
    loaded with the abandoned candidate while the durable pointer names stock.
    Before this was fixed the only trace of that was a substring inside
    ``result.error``: the outcome stayed ``PREEMPTED``, the state machine walked
    to ``READY``, and ``TunerService`` -- which re-attempts recovery only for
    ``FAILED`` -- never looked again.  So the cycle result must carry the fact
    as its own typed field, and it must outlive the cycle in a durable marker,
    because the state machine cannot carry it: ``READY`` is where the cycle
    legitimately ends.
    """
    activity = FakeActivity()
    runtime = FakeRuntime(activity, preempt_on_sleep=True, fail_restores=1)
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert result.error is not None
    assert "runtime restore failed" in result.error
    # The fact itself, typed rather than spelled inside a message.
    assert result.serving_restored is False
    # And still outstanding after the cycle ended at READY, which is exactly
    # the window in which the wrong draft answers live traffic.
    assert state.state is TunerState.READY
    assert orchestrator.serving_unrestored is True

    marker = state.state_path.parent / SERVING_UNRESTORED_FILE_NAME
    assert marker.exists()
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert "runtime restore failed" in record["error"]

    # The state journal says so too: "active draft restored" was a lie here.
    events = state.events_path.read_text(encoding="utf-8")
    assert "SERVING NOT RESTORED" in events


def test_recover_re_attempts_a_failed_restore_from_ready(tmp_path: Path) -> None:
    """``recover`` used to no-op at READY, so nothing ever retried the restore."""
    activity = FakeActivity()
    runtime = FakeRuntime(activity, preempt_on_sleep=True, fail_restores=1)
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
    )
    assert orchestrator.run_once().serving_restored is False
    assert state.state is TunerState.READY

    errors = orchestrator.recover()

    assert errors == ()
    assert orchestrator.serving_unrestored is False
    assert not (state.state_path.parent / SERVING_UNRESTORED_FILE_NAME).exists()
    # The retry is a real restore attempt, not just a cleared flag.
    assert len([call for call in runtime.calls if call.startswith("restore:")]) == 2


def test_a_restore_that_keeps_failing_keeps_reporting(tmp_path: Path) -> None:
    activity = FakeActivity()
    runtime = FakeRuntime(activity, preempt_on_sleep=True, fail_restores=99)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
    )
    orchestrator.run_once()

    errors = orchestrator.recover()

    assert len(errors) == 1
    assert "runtime restore failed" in errors[0]
    assert orchestrator.serving_unrestored is True


def test_training_stderr_is_surfaced_and_runtime_restored(tmp_path: Path) -> None:
    backend = FakeBackend(training_error=RuntimeError("CUDA out of memory"))
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        backend=backend,
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.FAILED
    assert result.error is not None
    assert "CUDA out of memory" in result.error
    assert state.state is TunerState.READY


def test_final_assistant_mask_error_is_distinct(tmp_path: Path) -> None:
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        backend=FakeBackend(mask_error=True),
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.FINAL_ASSISTANT_MASK_ERROR
    assert "no final assistant tokens" in (result.error or "")
    assert state.state is TunerState.READY


# ---------------------------------------------------------------------------
# Decision persistence — the write half of `speedlm gain`
# ---------------------------------------------------------------------------


def test_decision_is_persisted_on_promote(tmp_path: Path) -> None:
    decision = _real_decision(promote=True)
    orchestrator, _, _, _ = _orchestrator(tmp_path, decision=decision)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    path = tmp_path / "work" / "run-1" / DECISION_FILE_NAME
    assert result.decision_path == path
    assert json.loads(path.read_text(encoding="utf-8")) == decision.to_dict()


def test_decision_is_persisted_on_reject(tmp_path: Path) -> None:
    """A rejection is a real measured result and must stay reportable."""
    decision = _real_decision(promote=False)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path, gate_passed=False, decision=decision
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.REJECTED
    path = tmp_path / "work" / "run-1" / DECISION_FILE_NAME
    assert result.decision_path == path
    assert load_decision(path) == decision


def test_no_decision_file_when_the_gate_produced_none(tmp_path: Path) -> None:
    orchestrator, _, _, _ = _orchestrator(tmp_path)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert result.decision_path is None
    assert not (tmp_path / "work" / "run-1" / DECISION_FILE_NAME).exists()


def test_inconsistent_decision_is_refused_rather_than_written(tmp_path: Path) -> None:
    """`speedlm gain` distrusts these, so they must never reach disk."""
    good = _real_decision(promote=True)
    bad = Decision(
        verdict=good.verdict,
        reason=good.reason,
        acceptance_delta_pp=good.acceptance_delta_pp,
        throughput_delta_pct=good.throughput_delta_pct,
        min_acceptance_delta_pp=good.min_acceptance_delta_pp,
        min_throughput_delta_pct=good.min_throughput_delta_pct,
        num_repeats=good.num_repeats + 1,
        per_repeat=good.per_repeat,
        stock_avg_acceptance=good.stock_avg_acceptance,
        candidate_avg_acceptance=good.candidate_avg_acceptance,
        stock_avg_tok_per_sec=good.stock_avg_tok_per_sec,
        candidate_avg_tok_per_sec=good.candidate_avg_tok_per_sec,
    )
    run_dir = tmp_path / "run"

    with pytest.raises(DecisionPersistError, match="inconsistent"):
        write_decision(run_dir, bad)

    assert not (run_dir / DECISION_FILE_NAME).exists()


@pytest.mark.parametrize("promote", [True, False])
def test_decision_round_trips_into_a_measured_gain_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, promote: bool
) -> None:
    """End-to-end contract: gate Decision -> decision.json -> `speedlm gain`.

    This is the test that proves the two halves connect: the orchestrator writes
    under the runs directory, and ``report.py`` finds, trusts, and renders it.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("SPEEDLM_HOME", str(home))
    decision = _real_decision(promote=promote)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate_passed=promote,
        decision=decision,
        work_root=home / "runs",
    )

    result = orchestrator.run_once()
    assert result.outcome is (
        CycleOutcome.PROMOTED if promote else CycleOutcome.REJECTED
    )

    report = build_gain_report()
    assert report.status is GainStatus.MEASURED
    assert report.deltas_measured is True
    assert report.source_path == home / "runs" / "run-1" / DECISION_FILE_NAME
    assert report.decision == decision

    text = report.render_text()
    assert f"verdict           : {decision.verdict.value}" in text
    assert "throughput stock  : 100.00 tok/s" in text
    assert f"repeats           : {decision.num_repeats}" in text
    assert "not measured" not in text

    payload = json.loads(report.to_json())
    measurement = payload["measurement"]
    assert measurement is not None
    assert measurement["stock_tok_per_sec"] == decision.stock_avg_tok_per_sec
    assert measurement["throughput_delta_pct"] == decision.throughput_delta_pct
    assert len(payload["per_repeat"]) == decision.num_repeats


def test_rejected_decision_keeps_its_reason_through_the_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SPEEDLM_HOME", str(home))
    decision = _real_throughput_regression_decision()
    assert decision.reason is Reason.THROUGHPUT_BELOW_THRESHOLD
    orchestrator, _, _, _ = _orchestrator(
        tmp_path, gate_passed=False, decision=decision, work_root=home / "runs"
    )

    orchestrator.run_once()

    report = build_gain_report()
    assert report.decision is not None
    assert report.decision.reason is Reason.THROUGHPUT_BELOW_THRESHOLD
    assert "rejected because" in report.detail


def test_scratch_quota_exceeded(tmp_path: Path) -> None:
    adapter = _eagle3_adapter(renderer=FakeRenderer(bytes_to_write=32), quota=20)
    work = tmp_path / "scratch"

    try:
        adapter.prepare(work, should_abort=lambda: False)
    except ScratchQuotaExceeded as exc:
        assert exc.used_bytes > exc.quota_bytes
    else:
        raise AssertionError("expected scratch quota failure")


def test_gate_metrics_bodies_are_persisted_beside_the_decision(tmp_path: Path) -> None:
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate_passed=False,
        decision=_real_decision(promote=False),
        metrics_bodies={
            "stock-before": "vllm:generation_tokens_total 1\n",
            "candidate-after": "vllm:generation_tokens_total 9\n",
        },
    )

    result = orchestrator.run_once()

    assert result.decision_path is not None
    metrics_dir = result.decision_path.parent / METRICS_DIR_NAME
    assert sorted(path.name for path in metrics_dir.iterdir()) == [
        "candidate-after.prom.gz",
        "stock-before.prom.gz",
    ]
    body = gzip.decompress((metrics_dir / "stock-before.prom.gz").read_bytes())
    assert body.decode("utf-8") == "vllm:generation_tokens_total 1\n"


def test_gate_metrics_bodies_are_persisted_without_a_decision(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate_passed=False,
        decision=None,
        metrics_bodies={"stock-before": "vllm:generation_tokens_total 1\n"},
        work_root=work_root,
    )

    result = orchestrator.run_once()

    assert result.decision_path is None
    assert (work_root / "run-1" / METRICS_DIR_NAME / "stock-before.prom.gz").is_file()


def test_unsafe_metrics_label_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DecisionPersistError, match="unsafe label"):
        write_metrics_bodies(tmp_path, {"../escape": "body"})


# ---------------------------------------------------------------------------
# Infrastructure failures are not scientific results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (GateFailure.TIMED_OUT, CycleOutcome.BENCHMARK_TIMED_OUT),
        (GateFailure.ABORTED, CycleOutcome.BENCHMARK_ABORTED),
    ],
)
def test_gate_failure_gets_its_own_terminal_outcome(
    tmp_path: Path,
    failure: GateFailure,
    expected: CycleOutcome,
) -> None:
    """A benchmark that never measured must not read as a rejection."""
    gate = FakeGate(False, failure=failure)
    orchestrator, state, artifacts, runtime = _orchestrator(tmp_path, gate=gate)

    result = orchestrator.run_once()

    assert result.outcome is expected
    assert result.outcome is not CycleOutcome.REJECTED
    # Rollback and serving restoration are unchanged: only the label differs.
    assert state.state is TunerState.READY
    assert artifacts.active() is None
    assert runtime.calls == [
        "quiesce",
        "sleep",
        "start_candidate",
        "restore:base-draft",
        "wake",
    ]


def test_gate_failure_leaves_no_decision_to_report(tmp_path: Path) -> None:
    gate = FakeGate(False, failure=GateFailure.TIMED_OUT)
    orchestrator, _, _, _ = _orchestrator(tmp_path, gate=gate)

    result = orchestrator.run_once()

    assert result.decision_path is None
    assert result.gate is not None
    assert result.gate.decision is None
    assert result.gate.failure is GateFailure.TIMED_OUT


def test_measured_rejection_keeps_the_rejected_outcome(tmp_path: Path) -> None:
    """The distinction must not swallow genuine rejections."""
    orchestrator, _, _, _ = _orchestrator(tmp_path, gate_passed=False)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.REJECTED
    assert result.gate is not None
    assert result.gate.failure is None


def test_every_gate_failure_maps_to_a_distinct_outcome() -> None:
    """A new GateFailure must not silently fall back to ``rejected``."""
    from speedlm.tuner.orchestrator import _GATE_FAILURE_OUTCOMES

    assert set(_GATE_FAILURE_OUTCOMES) == set(GateFailure)
    assert len(set(_GATE_FAILURE_OUTCOMES.values())) == len(GateFailure)
    assert CycleOutcome.REJECTED not in set(_GATE_FAILURE_OUTCOMES.values())


def test_gate_result_failure_cannot_claim_a_measurement() -> None:
    with pytest.raises(ValueError, match="cannot have passed"):
        GateResult(True, "ok", failure=GateFailure.TIMED_OUT)


def test_gate_result_failure_cannot_carry_a_decision() -> None:
    decision = _real_decision(promote=True)
    with pytest.raises(ValueError, match="cannot carry a decision"):
        GateResult(False, "nope", decision=decision, failure=GateFailure.ABORTED)


def test_gate_result_failure_must_be_typed() -> None:
    with pytest.raises(TypeError, match="GateFailure"):
        GateResult(False, "nope", failure="timed_out")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Derived benchmark deadline
# ---------------------------------------------------------------------------


def test_derived_timeout_scales_with_suite_repeats_and_arms() -> None:
    small = derive_benchmark_timeout(num_contexts=10, repeats=5, concurrency=1)
    big = derive_benchmark_timeout(num_contexts=100, repeats=5, concurrency=1)
    more_repeats = derive_benchmark_timeout(
        num_contexts=10, repeats=50, concurrency=1
    )

    assert big > small
    assert more_repeats > small


def test_derived_timeout_shrinks_with_concurrency() -> None:
    serial = derive_benchmark_timeout(num_contexts=100, repeats=5, concurrency=1)
    parallel = derive_benchmark_timeout(num_contexts=100, repeats=5, concurrency=8)

    assert parallel < serial


def test_derived_timeout_covers_the_measured_production_shape() -> None:
    """The shape the gate actually runs, sized against what it actually cost.

    103 held-out contexts, five scored repeats plus one warmup, two arms, at
    concurrency 8 under a 512-token throughput cap and a 1x128-token
    correctness pass.  Job 369040 measured the whole benchmark at ~1740s for
    gpt-oss and ~1140s for Qwen, and the fixed 1800s deadline this derivation
    replaced expired inside the stock arm's warmup pass.
    """
    budget = derive_benchmark_timeout(
        num_contexts=103,
        repeats=5,
        warmup_repeats=1,
        concurrency=8,
        correctness_repeats=1,
        benchmark_max_tokens=512,
        correctness_max_tokens=128,
    )

    assert budget == pytest.approx(2_709.45, abs=1.0)
    assert budget > 1_800.0
    # Headroom over both measured profiles, without the 3.5x slack the flat
    # per-generation constant produced (9612s for a benchmark that used 2713s).
    assert budget > 1_740.0
    assert budget < 2 * 1_740.0


def test_derived_timeout_scales_with_the_throughput_cap() -> None:
    """The deadline must move when the cap it is sized against moves."""
    shape: dict[str, int] = {
        "num_contexts": 103,
        "repeats": 5,
        "warmup_repeats": 1,
        "concurrency": 8,
        "correctness_repeats": 1,
    }
    tight = derive_benchmark_timeout(**shape, benchmark_max_tokens=256)  # type: ignore[arg-type]
    loose = derive_benchmark_timeout(**shape, benchmark_max_tokens=1_024)  # type: ignore[arg-type]

    assert loose > tight
    # The generation term is linear in the cap and nothing else in the formula
    # depends on it, so the difference is exactly the extra tokens' cost.
    extra_tokens = 2 * (1 + 5) * 103 * (1_024 - 256)
    assert loose - tight == pytest.approx(
        extra_tokens * BENCHMARK_SECONDS_PER_TOKEN / 8 * BENCHMARK_SAFETY_FACTOR
    )


def test_the_correctness_pass_is_not_charged_at_the_throughput_rate() -> None:
    """Job 369040's real bug: a 128-token pass billed like a 512-token one.

    The formula charged the correctness pass ``2 x 1 x 103 x 20.0 = 4120s``
    against a measured 396s -- 10.4x over, and the majority of a 9612s
    deadline.  Costing tokens makes the correctness term exactly its own cap's
    share, so it can never again dominate a budget it does not spend.
    """
    shape: dict[str, int] = {
        "num_contexts": 103,
        "repeats": 5,
        "warmup_repeats": 1,
        "concurrency": 8,
        "benchmark_max_tokens": 512,
        "correctness_max_tokens": 128,
    }
    without = derive_benchmark_timeout(**shape, correctness_repeats=0)  # type: ignore[arg-type]
    with_pass = derive_benchmark_timeout(**shape, correctness_repeats=1)  # type: ignore[arg-type]

    correctness_share = with_pass - without
    # 2 arms x 1 repeat x 103 contexts x 128 tokens x 0.016 s x 1.25 safety.
    assert correctness_share == pytest.approx(527.36, abs=1.0)
    # The old formula charged it 4120s before the safety factor; the point is
    # that it is now a small fraction of the deadline, not the bulk of it.
    assert correctness_share < 0.25 * with_pass


def test_derived_timeout_is_clamped_at_both_ends() -> None:
    tiny = derive_benchmark_timeout(
        num_contexts=1,
        repeats=3,
        concurrency=64,
        fixed_overhead_seconds=1.0,
    )
    huge = derive_benchmark_timeout(
        num_contexts=100_000, repeats=100, concurrency=1
    )

    assert tiny == BENCHMARK_MIN_SECONDS
    assert huge == BENCHMARK_MAX_SECONDS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_contexts": 0},
        {"repeats": 0},
        {"arms": 0},
        {"concurrency": 0},
        {"warmup_repeats": -1},
        {"seconds_per_token": 0.0},
        {"benchmark_max_tokens": 0},
        {"correctness_max_tokens": 0},
        {"safety_factor": -1.0},
        {"fixed_overhead_seconds": 0},
    ],
)
def test_derived_timeout_rejects_nonsense(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"num_contexts": 10, "repeats": 5}
    with pytest.raises(ValueError):
        derive_benchmark_timeout(**{**base, **kwargs})  # type: ignore[arg-type]


@dataclass
class SizingGate(FakeGate):
    """Gate that can size its own benchmark, as the real runner does."""

    estimate: float | None = None

    def estimated_benchmark_seconds(self) -> float | None:
        if isinstance(self.estimate, str):
            raise RuntimeError("estimator blew up")
        return self.estimate


def test_orchestrator_uses_the_gate_estimate_when_it_is_smaller(
    tmp_path: Path,
) -> None:
    gate = SizingGate(True, estimate=900.0)
    orchestrator, _, _, _ = _orchestrator(tmp_path, gate=gate)

    orchestrator.run_once()

    assert gate.seen_timeouts == [900.0]


def test_orchestrator_never_lets_an_estimate_exceed_the_ceiling(
    tmp_path: Path,
) -> None:
    """The ceiling is what bounds a genuine hang and must stay authoritative."""
    gate = SizingGate(True, estimate=BENCHMARK_MAX_SECONDS * 10)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate=gate,
        timeouts=OrchestratorTimeouts(benchmark=1_200.0),
    )

    orchestrator.run_once()

    assert gate.seen_timeouts == [1_200.0]


@pytest.mark.parametrize("estimate", [None, 0.0, -5.0, "boom"])
def test_orchestrator_falls_back_to_the_ceiling(
    tmp_path: Path,
    estimate: object,
) -> None:
    gate = SizingGate(True, estimate=estimate)  # type: ignore[arg-type]
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate=gate,
        timeouts=OrchestratorTimeouts(benchmark=1_234.0),
    )

    orchestrator.run_once()

    assert gate.seen_timeouts == [1_234.0]


def test_gate_without_an_estimator_still_gets_the_configured_ceiling(
    tmp_path: Path,
) -> None:
    gate = FakeGate(True)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        gate=gate,
        timeouts=OrchestratorTimeouts(benchmark=777.0),
    )

    orchestrator.run_once()

    assert gate.seen_timeouts == [777.0]


def test_default_benchmark_ceiling_is_no_longer_the_fixed_1800s() -> None:
    assert OrchestratorTimeouts().benchmark == BENCHMARK_MAX_SECONDS
    assert OrchestratorTimeouts().benchmark > 1_800.0


# ---------------------------------------------------------------------------
# Draft hot-swap sequencing
#
# The cycle sleeps vLLM at level 1 before training to release device memory,
# and a level-1 sleep offloads the drafter's weights.  Weights that are not
# resident cannot be overwritten, so ``start_candidate`` used to arrive at a
# sleeping engine every single time and the swap was unreachable in production
# no matter how it was implemented.  These tests pin the wake that fixes that,
# and pin equally hard that it changes nothing when the feature is off.
# ---------------------------------------------------------------------------


@dataclass
class HotSwapRuntime(FakeRuntime):
    """A runtime that advertises the optional hot-swap capability."""

    #: Mirrors ``config.tuning.draft_hot_swap_enabled`` reaching the controller.
    supports_draft_hot_swap: bool = True
    #: Raised by :meth:`wake_for_swap`, to exercise the fallback.
    wake_error: Exception | None = None

    def wake_for_swap(self, *, timeout_seconds: float) -> None:
        self.calls.append(f"wake_for_swap:{timeout_seconds:g}")
        if self.wake_error is not None:
            raise self.wake_error


def test_flag_off_sequencing_is_byte_identical_to_today(tmp_path: Path) -> None:
    activity = FakeActivity()
    runtime = HotSwapRuntime(activity, supports_draft_hot_swap=False)
    orchestrator, _, _, _ = _orchestrator(tmp_path, activity=activity, runtime=runtime)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    # Exactly the order asserted by test_happy_path_reaches_promoting_then_ready.
    assert runtime.calls == ["quiesce", "sleep", "start_candidate", "wake"]


def test_a_runtime_without_the_capability_is_never_asked_to_wake(
    tmp_path: Path,
) -> None:
    # The plain FakeRuntime has no ``wake_for_swap`` at all: the orchestrator
    # must treat that as an ordinary runtime, not as a failure.
    orchestrator, _, _, runtime = _orchestrator(tmp_path)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert runtime.calls == ["quiesce", "sleep", "start_candidate", "wake"]


def test_flag_on_wakes_for_the_swap_immediately_before_start_candidate(
    tmp_path: Path,
) -> None:
    activity = FakeActivity()
    runtime = HotSwapRuntime(activity)
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
        timeouts=OrchestratorTimeouts(wake=31.0),
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    assert runtime.calls == [
        "quiesce",
        "sleep",
        "wake_for_swap:31",
        "start_candidate",
        "wake",
    ]


def test_a_failed_wake_falls_back_to_the_full_restart_and_still_completes(
    tmp_path: Path,
) -> None:
    activity = FakeActivity()
    runtime = HotSwapRuntime(activity, wake_error=RuntimeError("wake_up refused"))
    orchestrator, state, artifacts, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
    )

    result = orchestrator.run_once()

    # start_candidate still runs, and it restarts because the engine is still
    # asleep -- which is exactly the sequencing used before the wake existed.
    assert result.outcome is CycleOutcome.PROMOTED
    assert result.error is None
    assert state.state is TunerState.READY
    assert artifacts.active() is not None
    assert runtime.calls == [
        "quiesce",
        "sleep",
        "wake_for_swap:30",
        "start_candidate",
        "wake",
    ]


@dataclass
class OrderedRuntime(HotSwapRuntime):
    """Records its effects into a timeline shared with the backend."""

    timeline: list[str] = field(default_factory=list)

    def quiesce(
        self, *, timeout_seconds: float, should_abort: Callable[[], bool]
    ) -> None:
        self.timeline.append("quiesce")
        super().quiesce(timeout_seconds=timeout_seconds, should_abort=should_abort)

    def sleep(
        self, *, timeout_seconds: float, should_abort: Callable[[], bool]
    ) -> None:
        # Standing in for the controller's ``_await_gpu_memory``: sleep only
        # returns once the device memory the trainer needs is actually free.
        self.timeline.append("sleep")
        self.timeline.append("gpu-memory-released")
        super().sleep(timeout_seconds=timeout_seconds, should_abort=should_abort)

    def wake_for_swap(self, *, timeout_seconds: float) -> None:
        self.timeline.append("wake_for_swap")
        super().wake_for_swap(timeout_seconds=timeout_seconds)

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        self.timeline.append("start_candidate")
        super().start_candidate(
            draft_directory,
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )


@dataclass
class OrderedBackend(FakeBackend):
    """A backend that appends every GPU-consuming step to a shared timeline."""

    timeline: list[str] = field(default_factory=list)

    def prepare(
        self, work_dir: Path, *, should_abort: Callable[[], bool]
    ) -> PreparedData:
        self.timeline.append("prepare")
        return super().prepare(work_dir, should_abort=should_abort)

    def extract(
        self,
        prepared: PreparedData,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        self.timeline.append("extract")
        return super().extract(prepared, work_dir, should_abort=should_abort)

    def train(
        self,
        extracted: Path,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> TrainingResult:
        self.timeline.append("train")
        return super().train(extracted, work_dir, should_abort=should_abort)

    def validate(
        self, draft_directory: Path, *, should_abort: Callable[[], bool]
    ) -> None:
        self.timeline.append("validate")
        super().validate(draft_directory, should_abort=should_abort)


@pytest.mark.parametrize("hot_swap", [False, True])
def test_the_memory_release_still_precedes_every_training_step(
    tmp_path: Path, hot_swap: bool
) -> None:
    """The wake must never move in front of the work the sleep exists for."""
    timeline: list[str] = []
    activity = FakeActivity()
    runtime = OrderedRuntime(
        activity,
        supports_draft_hot_swap=hot_swap,
        timeline=timeline,
    )
    backend = OrderedBackend(timeline=timeline)
    orchestrator, _, _, _ = _orchestrator(
        tmp_path,
        activity=activity,
        runtime=runtime,
        backend=backend,
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    expected = [
        "quiesce",
        "sleep",
        "gpu-memory-released",
        "prepare",
        "extract",
        "train",
        "validate",
        *(["wake_for_swap"] if hot_swap else []),
        "start_candidate",
    ]
    assert timeline == expected
    # The invariant stated positively: nothing re-acquires device memory until
    # the last training step has finished.
    assert timeline.index("gpu-memory-released") < timeline.index("prepare")
    if hot_swap:
        assert timeline.index("validate") < timeline.index("wake_for_swap")


# --- the real controller, to prove the path is actually reachable ----------


@dataclass
class RecordingAdmission:
    calls: list[str] = field(default_factory=list)

    def stop_admitting(self) -> None:
        self.calls.append("stop")

    def start_admitting(self) -> None:
        self.calls.append("start")


@dataclass
class RecordingControlHTTP:
    posts: list[str] = field(default_factory=list)

    def post(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        query: Mapping[str, str] | None = None,
    ) -> None:
        self.posts.append(endpoint)

    def wait_ready(self, *, timeout_seconds: float) -> None:
        return None

    def canary(self, *, timeout_seconds: float) -> None:
        return None

    def wait_sleeping(
        self,
        sleeping: bool,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        return None


@dataclass
class RecordingProcess:
    restarts: list[str] = field(default_factory=list)

    def restart(self, draft: Path | str, *, timeout_seconds: float) -> None:
        self.restarts.append(str(draft))


@dataclass
class RecordingDraftSwap:
    swaps: list[str] = field(default_factory=list)
    error: Exception | None = None

    def hot_swap_draft(self, weights_path: str, *, timeout_seconds: float) -> None:
        self.swaps.append(weights_path)
        if self.error is not None:
            raise self.error


def _real_runtime(
    *,
    swap: RecordingDraftSwap | None,
) -> tuple[RealRuntimeController, RecordingProcess, RecordingControlHTTP]:
    http = RecordingControlHTTP()
    process = RecordingProcess()
    controller = RealRuntimeController(
        activity=FakeActivity(),
        admission=RecordingAdmission(),
        http=http,
        process=process,
        active_draft="base-draft",
        clock=lambda: 0.0,
        sleeper=lambda seconds: None,
        draft_swap_http=swap,
    )
    return controller, process, http


@pytest.mark.parametrize("enabled", [False, True])
def test_the_real_controller_reaches_the_swap_only_when_enabled(
    tmp_path: Path, enabled: bool
) -> None:
    """End-to-end: an orchestrator cycle driving the production controller.

    This is the test the blocker was invisible to before -- with the flag on,
    the drafter is swapped and the child is never replaced; with it off, the
    cycle restarts exactly as it always did.
    """
    swap = RecordingDraftSwap() if enabled else None
    controller, process, http = _real_runtime(swap=swap)
    orchestrator, state, artifacts, _ = _orchestrator(tmp_path, runtime=controller)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    active = artifacts.active()
    assert active is not None
    if enabled:
        assert swap is not None
        assert swap.swaps == [str(active.path)]
        # The whole point: no child process was replaced for the candidate.
        assert process.restarts == []
        # Sleep for training, then one mid-cycle wake, then the end-of-cycle
        # wake finds the engine already awake and only reopens admission.
        assert http.posts == ["/sleep", "/wake_up"]
    else:
        assert process.restarts == [str(active.path)]
        # The restart is what ends the sleep, so the end-of-cycle wake has no
        # engine left to wake -- unchanged from before this feature existed.
        assert http.posts == ["/sleep"]


def test_a_swap_failure_under_the_real_controller_restarts_the_child(
    tmp_path: Path,
) -> None:
    swap = RecordingDraftSwap(error=DraftSwapCorrupted("half-applied"))
    controller, process, _ = _real_runtime(swap=swap)
    orchestrator, state, artifacts, _ = _orchestrator(tmp_path, runtime=controller)

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    active = artifacts.active()
    assert active is not None
    assert swap.swaps == [str(active.path)]
    # A swap that may have half-applied must still leave a serving engine.
    assert process.restarts == [str(active.path)]
