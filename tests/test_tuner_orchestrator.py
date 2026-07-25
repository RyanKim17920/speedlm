from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.eagle3 import (
    Eagle3Adapter,
    Eagle3Config,
    FinalAssistantMaskError,
    ScratchQuotaExceeded,
    TraceSnapshot,
    TrainingResult,
)
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    GateResult,
    TunerOrchestrator,
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

    def wake(self, *, timeout_seconds: float) -> None:
        self.calls.append("wake")


@dataclass
class FakeGate:
    passed: bool

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        return GateResult(self.passed, "thresholds met" if self.passed else "regression")


def _adapter(
    *,
    trainer: FakeTrainer | None = None,
    renderer: FakeRenderer | None = None,
    quota: int = 1024 * 1024,
) -> Eagle3Adapter:
    return Eagle3Adapter(
        Eagle3Config(training_params={"steps": 2}, scratch_quota_bytes=quota),
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
    adapter: Eagle3Adapter | None = None,
) -> tuple[TunerOrchestrator, TunerStateMachine, ArtifactRegistry, FakeRuntime]:
    activity = activity or FakeActivity()
    runtime = runtime or FakeRuntime(activity)
    state = TunerStateMachine(tmp_path / "state")
    artifacts = ArtifactRegistry(tmp_path / "registry")
    orchestrator = TunerOrchestrator(
        state=state,
        idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
        eagle3=adapter or _adapter(),
        artifacts=artifacts,
        runtime=runtime,
        gate=FakeGate(gate_passed),
        work_root=tmp_path / "work",
        run_id_factory=lambda: "run-1",
    )
    return orchestrator, state, artifacts, runtime


def test_happy_path_reaches_promoting_then_ready(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    orchestrator, state, artifacts, runtime = _orchestrator(
        tmp_path,
        adapter=_adapter(trainer=trainer),
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    assert artifacts.active() is not None
    assert trainer.seen_from_pretrained == "RedHatAI/gpt-oss-20b-speculator.eagle3"
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


def test_training_stderr_is_surfaced_and_runtime_restored(tmp_path: Path) -> None:
    trainer = FakeTrainer(returncode=9, stderr="CUDA out of memory")
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        adapter=_adapter(trainer=trainer),
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.FAILED
    assert result.error is not None
    assert "CUDA out of memory" in result.error
    assert state.state is TunerState.READY


def test_final_assistant_mask_error_is_distinct(tmp_path: Path) -> None:
    orchestrator, state, _, _ = _orchestrator(
        tmp_path,
        adapter=_adapter(renderer=FakeRenderer(mask_error=True)),
    )

    result = orchestrator.run_once()

    assert result.outcome is CycleOutcome.FINAL_ASSISTANT_MASK_ERROR
    assert "no final assistant tokens" in (result.error or "")
    assert state.state is TunerState.READY


def test_scratch_quota_exceeded(tmp_path: Path) -> None:
    adapter = _adapter(renderer=FakeRenderer(bytes_to_write=32), quota=20)
    work = tmp_path / "scratch"

    try:
        adapter.prepare(work, should_abort=lambda: False)
    except ScratchQuotaExceeded as exc:
        assert exc.used_bytes > exc.quota_bytes
    else:
        raise AssertionError("expected scratch quota failure")
