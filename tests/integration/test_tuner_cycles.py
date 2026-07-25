from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from speedlm.gateway.activity import ActivityTracker
from speedlm.training.base import BackendInfo
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    GateResult,
    TunerOrchestrator,
)
from speedlm.tuner.state import TunerState, TunerStateMachine


@dataclass(frozen=True, slots=True)
class _Snapshot:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    snapshot: _Snapshot


@dataclass
class _Backend:
    payload: bytes

    def describe(self) -> BackendInfo:
        return BackendInfo(
            verifier_model="verifier",
            draft_model="draft",
            from_pretrained="stock-draft",
            training_params={"steps": 1},
        )

    def prepare(
        self,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> _Prepared:
        del work_dir
        assert not should_abort()
        return _Prepared(_Snapshot(self.payload.hex()))

    def extract(
        self,
        prepared: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> object:
        del prepared, work_dir
        assert not should_abort()
        return object()

    def train(
        self,
        extracted: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> object:
        del extracted, work_dir
        assert not should_abort()
        return object()

    def materialize(
        self,
        trained: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del trained
        assert not should_abort()
        draft = work_dir / "draft"
        draft.mkdir()
        (draft / "weights.bin").write_bytes(self.payload)
        return draft

    def validate(
        self,
        artifact: Any,
        *,
        should_abort: Callable[[], bool],
    ) -> None:
        assert Path(artifact).is_dir()
        assert not should_abort()


@dataclass
class _Gate:
    passed: bool

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        assert candidate_draft.is_dir()
        assert timeout_seconds > 0 and not should_abort()
        return GateResult(
            passed=self.passed,
            reason="thresholds met" if self.passed else "candidate regressed",
        )


@dataclass
class _Runtime:
    activity: ActivityTracker
    clock_value: list[float]
    traffic_on_sleep: bool = False
    serving: Path | str | None = None
    calls: list[str] = field(default_factory=list)

    def quiesce(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.calls.append("quiesce")

    def sleep(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.calls.append("sleep")
        if self.traffic_on_sleep:
            self.clock_value[0] += 1.0
            self.activity.begin()
            self.activity.end()

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.serving = draft_directory
        self.calls.append("start_candidate")

    def restore(self, active_draft: Path | str, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.serving = active_draft
        self.calls.append("restore")

    def wake(self, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.calls.append("wake")


def _idle_pair() -> tuple[list[float], ActivityTracker, IdleDetector]:
    now = [0.0]
    activity = ActivityTracker(clock=lambda: now[0])
    now[0] = 10.0
    idle = IdleDetector(activity, threshold_seconds=5.0, clock=lambda: now[0])
    return now, activity, idle


def test_full_tuner_promotes_then_rejection_preserves_active_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    runs = tmp_path / "runs"
    state = TunerStateMachine(runs / "state")
    artifacts = ArtifactRegistry(runs)
    now, activity, idle = _idle_pair()
    runtime = _Runtime(activity, now)

    promoted = TunerOrchestrator(
        state=state,
        idle=idle,
        backend=_Backend(b"promoted-candidate"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(True),
        work_root=runs,
        run_id_factory=lambda: "promote-cycle",
    ).run_once()

    assert promoted.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    assert promoted.artifact_id is not None
    pointer_after_promotion = json.loads(
        artifacts.active_path.read_text(encoding="utf-8")
    )
    assert pointer_after_promotion["artifact_id"] == promoted.artifact_id
    active_bytes = artifacts.active_path.read_bytes()

    rejected = TunerOrchestrator(
        state=state,
        idle=idle,
        backend=_Backend(b"rejected-candidate"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(False),
        work_root=runs,
        run_id_factory=lambda: "reject-cycle",
    ).run_once()

    assert rejected.outcome is CycleOutcome.REJECTED
    assert rejected.artifact_id != promoted.artifact_id
    assert state.state is TunerState.READY
    assert artifacts.active_path.read_bytes() == active_bytes
    assert artifacts.active_pointer() is not None
    assert artifacts.active_pointer().artifact_id == promoted.artifact_id  # type: ignore[union-attr]


def test_mid_cycle_traffic_preempts_and_restores_previous_active_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    runs = tmp_path / "runs"
    artifacts = ArtifactRegistry(runs)
    baseline_source = tmp_path / "baseline"
    baseline_source.mkdir()
    (baseline_source / "weights.bin").write_bytes(b"known-good")
    baseline = artifacts.publish(
        baseline_source,
        ArtifactSpec(
            verifier_model="verifier",
            draft_model="draft",
            base_draft="stock-draft",
            trace_hash="baseline-traces",
            training_params={"steps": 1},
        ),
    )
    artifacts.promote(baseline.artifact_id, gate_passed=True)
    active_before = artifacts.active_path.read_bytes()
    now, activity, idle = _idle_pair()
    runtime = _Runtime(
        activity,
        now,
        traffic_on_sleep=True,
        serving=baseline.path,
    )

    result = TunerOrchestrator(
        state=TunerStateMachine(runs / "state"),
        idle=idle,
        backend=_Backend(b"must-not-train"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(True),
        work_root=runs,
        run_id_factory=lambda: "preempted-cycle",
    ).run_once()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert artifacts.active_path.read_bytes() == active_before
    assert runtime.serving == baseline.path
    assert runtime.calls == ["quiesce", "sleep", "restore", "wake"]
