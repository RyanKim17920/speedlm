"""Fail-closed orchestration of one idle EAGLE-3 tuning cycle."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.eagle3 import Eagle3Adapter, FinalAssistantMaskError
from speedlm.tuner.idle import IdleDetector, TuningPreempted
from speedlm.tuner.state import TunerState, TunerStateMachine

AbortCheck = Callable[[], bool]
DraftReference = Path | str


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome returned by the injected benchmark/promotion gate."""

    passed: bool
    reason: str
    metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("gate result passed must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("gate result reason must be a non-empty string")


class BenchmarkGate(Protocol):
    """Benchmark a running candidate against stock and make a gate decision."""

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> GateResult: ...


class RuntimeController(Protocol):
    """Gateway and child-vLLM effects required by the tuner.

    Implementations must make all methods idempotent. ``restore`` must restart
    the child vLLM with the supplied active draft (or the configured base draft).
    """

    def quiesce(
        self, *, timeout_seconds: float, should_abort: AbortCheck
    ) -> None: ...

    def sleep(
        self, *, timeout_seconds: float, should_abort: AbortCheck
    ) -> None: ...

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None: ...

    def restore(
        self,
        active_draft: DraftReference,
        *,
        timeout_seconds: float,
    ) -> None: ...

    def wake(self, *, timeout_seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class OrchestratorTimeouts:
    """Timeouts for runtime and gate effects."""

    quiesce: float = 30.0
    sleep: float = 120.0
    candidate_start: float = 600.0
    benchmark: float = 1_800.0
    restore: float = 600.0
    wake: float = 30.0

    def __post_init__(self) -> None:
        for name, value in (
            ("quiesce", self.quiesce),
            ("sleep", self.sleep),
            ("candidate_start", self.candidate_start),
            ("benchmark", self.benchmark),
            ("restore", self.restore),
            ("wake", self.wake),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} timeout must be a positive number")


_DEFAULT_TIMEOUTS = OrchestratorTimeouts()


class CycleOutcome(StrEnum):
    """Terminal result of one scheduler poll."""

    NOT_IDLE = "not_idle"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    PREEMPTED = "preempted"
    FINAL_ASSISTANT_MASK_ERROR = "final_assistant_mask_error"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Observable cycle result for logs and scheduling policy."""

    outcome: CycleOutcome
    artifact_id: str | None = None
    gate: GateResult | None = None
    error: str | None = None


class TunerOrchestrator:
    """Drive one complete cycle and restore serving on every terminal branch."""

    def __init__(
        self,
        *,
        state: TunerStateMachine,
        idle: IdleDetector,
        eagle3: Eagle3Adapter,
        artifacts: ArtifactRegistry,
        runtime: RuntimeController,
        gate: BenchmarkGate,
        work_root: Path,
        timeouts: OrchestratorTimeouts = _DEFAULT_TIMEOUTS,
        run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._state = state
        self._idle = idle
        self._eagle3 = eagle3
        self._artifacts = artifacts
        self._runtime = runtime
        self._gate = gate
        self._work_root = work_root
        self._timeouts = timeouts
        self._run_id_factory = run_id_factory

    def run_once(self) -> CycleResult:
        """Poll idle state and, when eligible, execute one complete tuning cycle."""
        if self._state.state is not TunerState.READY:
            recovery_errors = self.recover()
            if recovery_errors:
                return CycleResult(
                    outcome=CycleOutcome.FAILED,
                    error="; ".join(recovery_errors),
                )
        if not self._idle.should_tune:
            return CycleResult(outcome=CycleOutcome.NOT_IDLE)

        guard = self._idle.arm()
        work_dir = self._work_root / self._run_id_factory()
        work_dir.mkdir(parents=True, exist_ok=False)
        artifact_id: str | None = None
        pointer_changed = False

        try:
            self._state.transition(TunerState.QUIESCING, reason="idle threshold reached")
            guard.check()
            self._runtime.quiesce(
                timeout_seconds=self._timeouts.quiesce,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.SLEEPING, reason="gateway quiesced")
            self._runtime.sleep(
                timeout_seconds=self._timeouts.sleep,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.EXTRACTING, reason="vLLM sleeping")
            prepared = self._eagle3.prepare(
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            hidden_states = self._eagle3.extract(
                prepared,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.TRAINING, reason="hidden states extracted")
            training = self._eagle3.train(
                hidden_states,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            draft_directory = self._eagle3.materialize_and_validate(
                training,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            artifact = self._artifacts.publish(
                draft_directory,
                ArtifactSpec(
                    verifier_model=self._eagle3.config.verifier_model,
                    draft_model=self._eagle3.config.draft_model,
                    base_draft=self._eagle3.config.from_pretrained,
                    trace_hash=prepared.snapshot.content_hash,
                    training_params=self._eagle3.config.training_params,
                ),
            )
            artifact_id = artifact.artifact_id
            guard.check()

            self._state.transition(
                TunerState.CANDIDATE_STARTING,
                reason=f"candidate {artifact_id} materialized",
            )
            self._runtime.start_candidate(
                artifact.path,
                timeout_seconds=self._timeouts.candidate_start,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.BENCHMARKING, reason="candidate running")
            gate_result = self._gate.benchmark(
                artifact.path,
                timeout_seconds=self._timeouts.benchmark,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            if not gate_result.passed:
                self._state.transition(
                    TunerState.ROLLING_BACK,
                    reason=f"gate rejected candidate: {gate_result.reason}",
                )
                cleanup_errors = self._finish_rollback(pointer_changed=False)
                return CycleResult(
                    outcome=(
                        CycleOutcome.REJECTED
                        if not cleanup_errors
                        else CycleOutcome.FAILED
                    ),
                    artifact_id=artifact_id,
                    gate=gate_result,
                    error="; ".join(cleanup_errors) if cleanup_errors else None,
                )

            self._state.transition(
                TunerState.PROMOTING,
                reason=f"gate passed candidate: {gate_result.reason}",
            )
            self._artifacts.promote(artifact_id, gate_passed=True)
            pointer_changed = True
            self._state.transition(TunerState.WAKING, reason="active pointer promoted")
            self._runtime.wake(timeout_seconds=self._timeouts.wake)
            self._state.transition(TunerState.READY, reason="candidate serving")
            return CycleResult(
                outcome=CycleOutcome.PROMOTED,
                artifact_id=artifact_id,
                gate=gate_result,
            )
        except FinalAssistantMaskError as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.FINAL_ASSISTANT_MASK_ERROR,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
            )
        except TuningPreempted as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.PREEMPTED,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
            )
        except Exception as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.FAILED,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
            )

    def recover(self) -> tuple[str, ...]:
        """Restore the durable active draft after an interrupted process."""
        if self._state.state is TunerState.READY:
            return ()
        errors: list[str] = []
        if self._state.state not in {TunerState.ROLLING_BACK, TunerState.WAKING}:
            try:
                self._state.transition(
                    TunerState.ROLLING_BACK,
                    reason="process restart recovery",
                )
            except Exception as exc:
                errors.append(f"state recovery failed: {exc}")
        if self._state.state is TunerState.ROLLING_BACK:
            try:
                self._runtime.restore(
                    self._active_draft(),
                    timeout_seconds=self._timeouts.restore,
                )
            except Exception as exc:
                errors.append(f"runtime restore failed: {exc}")
            try:
                self._state.transition(TunerState.WAKING, reason="runtime restored")
            except Exception as exc:
                errors.append(f"state recovery failed: {exc}")
        if self._state.state is TunerState.WAKING:
            try:
                self._runtime.wake(timeout_seconds=self._timeouts.wake)
            except Exception as exc:
                errors.append(f"runtime wake failed: {exc}")
            try:
                self._state.transition(TunerState.READY, reason="restart recovery complete")
            except Exception as exc:
                errors.append(f"state recovery failed: {exc}")
        return tuple(errors)

    def _rollback_after_failure(self, pointer_changed: bool) -> tuple[str, ...]:
        if self._state.state is TunerState.READY:
            return ()
        errors: list[str] = []
        needs_rollback_transition = self._state.state not in {
            TunerState.ROLLING_BACK,
            TunerState.WAKING,
        } or (self._state.state is TunerState.WAKING and pointer_changed)
        if needs_rollback_transition:
            try:
                self._state.transition(
                    TunerState.ROLLING_BACK,
                    reason="cycle failed or was preempted",
                )
            except Exception as exc:
                errors.append(f"state rollback failed: {exc}")
        errors.extend(self._finish_rollback(pointer_changed))
        return tuple(errors)

    def _finish_rollback(self, pointer_changed: bool) -> tuple[str, ...]:
        errors: list[str] = []
        if pointer_changed:
            try:
                self._artifacts.rollback()
            except Exception as exc:
                errors.append(f"active pointer rollback failed: {exc}")
        if self._state.state is TunerState.ROLLING_BACK:
            try:
                self._runtime.restore(
                    self._active_draft(),
                    timeout_seconds=self._timeouts.restore,
                )
            except Exception as exc:
                errors.append(f"runtime restore failed: {exc}")
            try:
                self._state.transition(TunerState.WAKING, reason="active draft restored")
            except Exception as exc:
                errors.append(f"state rollback failed: {exc}")
        if self._state.state is TunerState.WAKING:
            try:
                self._runtime.wake(timeout_seconds=self._timeouts.wake)
            except Exception as exc:
                errors.append(f"runtime wake failed: {exc}")
            try:
                self._state.transition(TunerState.READY, reason="serving restored")
            except Exception as exc:
                errors.append(f"state rollback failed: {exc}")
        return tuple(errors)

    def _active_draft(self) -> DraftReference:
        active = self._artifacts.active()
        if active is not None:
            return active.path
        return self._eagle3.config.from_pretrained


def _combine_error(error: Exception, cleanup_errors: tuple[str, ...]) -> str:
    message = str(error)
    if cleanup_errors:
        return f"{message}; cleanup: {'; '.join(cleanup_errors)}"
    return message
