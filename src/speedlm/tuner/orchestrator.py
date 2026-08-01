"""Fail-closed orchestration of one idle speculative tuning cycle."""

from __future__ import annotations

import gzip
import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from speedlm.config import ValLossPreFilterConfig
from speedlm.gate.decide import Decision
from speedlm.storage import atomic_write_bytes, atomic_write_json
from speedlm.training.base import SpeculatorBackend
from speedlm.training.masking import FinalAssistantMaskError
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.idle import IdleDetector, TuningPreempted
from speedlm.tuner.state import TunerState, TunerStateMachine

logger = logging.getLogger(__name__)

AbortCheck = Callable[[], bool]
DraftReference = Path | str

#: Name of the persisted gate decision inside a run directory.  Must match
#: :data:`speedlm.report.DECISION_FILE_NAME`, which is what ``speedlm gain``
#: looks for underneath the runs directory.
DECISION_FILE_NAME: Final = "decision.json"

#: Directory, inside a run directory, holding the verbatim Prometheus bodies
#: the gate scraped.  It sits next to ``decision.json`` so a decision and the
#: evidence behind it are always found together.
METRICS_DIR_NAME: Final = "gate-metrics"

#: Marker recorded, next to ``state.json``, while serving is known to be on a
#: draft the durable active pointer does not name.
#:
#: This condition cannot be carried by :class:`~speedlm.tuner.state.TunerState`,
#: and that is the whole reason the marker exists.  ``READY`` is where a
#: preempted cycle legitimately ends, so the state machine has nowhere to put
#: "the cycle is over *and* serving is wrong"; before this file existed the only
#: record was a substring inside ``CycleResult.error``, which
#: :class:`~speedlm.tuner.service.TunerService` did not read and no restart
#: survived.  A durable file survives both, and it is what
#: :meth:`TunerOrchestrator.recover` consults to know it must try again even
#: from ``READY``.
SERVING_UNRESTORED_FILE_NAME: Final = "serving-unrestored.json"

_SAFE_LABEL: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DecisionPersistError(RuntimeError):
    """Raised when a gate decision cannot be persisted for later reporting."""


class GateFailure(StrEnum):
    """Why a gate returned without measuring anything.

    A gate that ran to completion leaves this ``None``, whatever its verdict:
    a rejection is a measurement.  These two values mean the opposite -- the
    benchmark never produced a :class:`~speedlm.gate.decide.Decision` -- and
    exist so that fact survives into ``scheduler.json`` instead of being
    flattened into the same ``rejected`` outcome a real comparison produces.
    """

    #: Serving activity preempted the benchmark.
    ABORTED = "aborted"
    #: The whole-run benchmark deadline expired.
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome returned by the injected benchmark/promotion gate.

    ``decision`` carries the gate's full :class:`~speedlm.gate.decide.Decision`
    when the benchmark produced one.  It is what gets persisted as
    ``decision.json`` so ``speedlm gain`` can report the measurement; a gate
    that never got far enough to build one leaves it ``None``.

    ``metrics_bodies`` maps a scrape label to the verbatim Prometheus body the
    gate read.  It is the evidence behind every derived rate, persisted next to
    the decision so provenance questions are answerable from artifacts.

    ``failure`` is set only when the gate gave up without measuring; it is the
    typed form of the ``{"aborted": True}`` / ``{"timed_out": True}`` markers
    the runner also writes into ``metrics``, and it is what lets the
    orchestrator emit an infrastructure outcome rather than a scientific one.
    """

    passed: bool
    reason: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    metrics_bodies: Mapping[str, str] = field(default_factory=dict)
    decision: Decision | None = None
    failure: GateFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("gate result passed must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("gate result reason must be a non-empty string")
        if self.decision is not None and not isinstance(self.decision, Decision):
            raise TypeError("gate result decision must be a Decision or None")
        if self.failure is not None:
            if not isinstance(self.failure, GateFailure):
                raise TypeError("gate result failure must be a GateFailure or None")
            # A failure means "never measured".  Allowing it alongside a pass
            # or a decision would recreate exactly the ambiguity it removes.
            if self.passed:
                raise ValueError("a failed gate result cannot have passed")
            if self.decision is not None:
                raise ValueError("a failed gate result cannot carry a decision")


def write_decision(run_dir: Path, decision: Decision) -> Path:
    """Atomically persist *decision* as ``<run_dir>/decision.json``.

    The payload is exactly :meth:`Decision.to_dict`, which is the shape
    :func:`speedlm.report.parse_decision` reads back.

    ``speedlm gain`` distrusts any decision whose ``num_repeats`` disagrees with
    the number of per-repeat rows, so an internally inconsistent decision is
    rejected here rather than written out as an unreportable file.

    Raises:
        DecisionPersistError: If the decision is inconsistent or cannot be
            written.
    """
    if decision.per_repeat and decision.num_repeats != len(decision.per_repeat):
        raise DecisionPersistError(
            "refusing to persist an inconsistent decision: "
            f"num_repeats={decision.num_repeats} but "
            f"len(per_repeat)={len(decision.per_repeat)}"
        )
    path = run_dir / DECISION_FILE_NAME
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, decision.to_dict())
    except OSError as exc:
        raise DecisionPersistError(f"cannot write gate decision {path}: {exc}") from exc
    return path


def write_metrics_bodies(
    run_dir: Path,
    bodies: Mapping[str, str],
) -> tuple[Path, ...]:
    """Persist each raw ``/metrics`` body under ``<run_dir>/gate-metrics``.

    Bodies are gzipped: a vLLM exposition is several hundred kilobytes of
    repetitive text, and keeping it compressed is what makes retaining it on
    every cycle affordable.

    Raises:
        DecisionPersistError: If a label is unusable as a file name or a body
            cannot be written.  Evidence that silently fails to land is the
            same blindness this write exists to remove.
    """
    if not bodies:
        return ()
    directory = run_dir / METRICS_DIR_NAME
    written: list[Path] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DecisionPersistError(
            f"cannot create gate metrics directory {directory}: {exc}"
        ) from exc
    for label in sorted(bodies):
        if not _SAFE_LABEL.match(label):
            raise DecisionPersistError(
                f"refusing to persist gate metrics under unsafe label {label!r}"
            )
        path = directory / f"{label}.prom.gz"
        try:
            atomic_write_bytes(
                path,
                gzip.compress(bodies[label].encode("utf-8"), mtime=0),
            )
        except OSError as exc:
            raise DecisionPersistError(
                f"cannot write gate metrics {path}: {exc}"
            ) from exc
        written.append(path)
    return tuple(written)


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

    Two *optional* members extend this contract without being part of it, so
    that an implementation which predates draft hot-swap -- or a test double --
    stays a valid ``RuntimeController``. They are read defensively, in the same
    spirit as ``estimated_benchmark_seconds`` on :class:`BenchmarkGate`:

    ``supports_draft_hot_swap``
        A read-only ``bool`` saying whether an in-place draft swap is wired up
        at all. Absent or false means the orchestrator sequences exactly as it
        always did.
    ``wake_for_swap(*, timeout_seconds)``
        Wake the sleeping engine *without* reopening gateway admission, so
        that the following :meth:`start_candidate` can swap the drafter's
        weights in place instead of restarting the child. Raising is a normal
        outcome; the orchestrator falls back to the restart path.
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


#: Wall-clock one generated token is assumed to cost on one stream.
#:
#: Replaces a flat ``BENCHMARK_SECONDS_PER_GENERATION = 20.0``, which was wrong
#: in two ways that a per-token rate fixes structurally rather than by
#: re-tuning a constant.
#:
#: First, a per-*generation* constant cannot span models, because a generation
#: is not a fixed amount of work.  Single-stream-equivalent cost measured 28.7
#: s/gen for gpt-oss and 13.2 s/gen for Qwen on the same suite -- 20.0 is 1.44x
#: too small for one and 1.5x too large for the other -- and essentially all of
#: that spread is output length, not per-token speed.
#:
#: Second, it charged the bounded correctness pass at the same rate as the
#: throughput pass.  On job 369040 the correctness pass took 396s while the
#: formula charged it ``2 x 1 x 103 x 20.0 = 4120s``, 10.4x over, which made it
#: the majority of a 9612s deadline for a benchmark that used 2713s.  A
#: per-token rate charges each pass its own cap, so the two cannot diverge.
#:
#: 0.016 is measured: 0.01347 s/token for gpt-oss and 0.00820 for Qwen, so this
#: carries ~19% margin over the slower of the two rather than being sized off
#: the exact observation that produced it.
BENCHMARK_SECONDS_PER_TOKEN: Final = 0.016

#: Fixed, per-benchmark cost that is not generation: two engine activations
#: (~90s each on job 368959), four Prometheus scrapes, suite load and the
#: leakage check.  600s is roughly three times the observed total, which is
#: cheap insurance on a term that does not scale with the suite.
BENCHMARK_FIXED_OVERHEAD_SECONDS: Final = 600.0

#: Multiplier applied to the derived generation budget.
#:
#: The budget divides the serial cost by the full replay concurrency, which
#: assumes linear speedup from batching.  Real speedup is sublinear -- and
#: sublinear specifically for speculative decoding, whose per-step cost grows
#: with batch size -- so this factor is the slack that absorbs the gap.  It is
#: not a safety margin against a hang; :data:`BENCHMARK_MAX_SECONDS` is.
BENCHMARK_SAFETY_FACTOR: Final = 1.25

#: Floor for a derived deadline, so a tiny suite still tolerates one slow
#: engine start rather than timing out on overhead alone.
BENCHMARK_MIN_SECONDS: Final = 300.0

#: Hard ceiling on any benchmark deadline, derived or configured.
#:
#: A genuine hang must still terminate the cycle, and the tuner only runs while
#: serving is idle -- ``should_abort`` cuts a benchmark short the moment real
#: traffic arrives -- so the ceiling only has to bound the pathological case.
BENCHMARK_MAX_SECONDS: Final = 14_400.0


def derive_benchmark_timeout(
    *,
    num_contexts: int,
    repeats: int,
    warmup_repeats: int = 1,
    arms: int = 2,
    concurrency: int = 1,
    correctness_repeats: int = 0,
    benchmark_max_tokens: int = 512,
    correctness_max_tokens: int = 128,
    seconds_per_token: float = BENCHMARK_SECONDS_PER_TOKEN,
    fixed_overhead_seconds: float = BENCHMARK_FIXED_OVERHEAD_SECONDS,
    safety_factor: float = BENCHMARK_SAFETY_FACTOR,
) -> float:
    """Size a benchmark deadline from the work the benchmark will do.

    The work is ``arms x (warmup_repeats + repeats) x num_contexts``
    generations of at most ``benchmark_max_tokens`` each, spread over
    ``concurrency`` in-flight requests, plus
    ``arms x correctness_repeats x num_contexts`` single-stream generations of
    at most ``correctness_max_tokens`` each for the output-correctness pass,
    plus a fixed overhead that does not scale with the suite.  The result is
    clamped into ``[BENCHMARK_MIN_SECONDS, BENCHMARK_MAX_SECONDS]``.

    Costing tokens rather than generations is what lets the two passes be
    charged differently: the correctness pass is bounded an order of magnitude
    tighter than the throughput pass, and a per-generation rate had no way to
    express that -- see :data:`BENCHMARK_SECONDS_PER_TOKEN`.

    This exists because a fixed 1800s deadline is not a statement about
    anything: on job 368959 it was smaller than one arm's warmup pass, so the
    gate died before its first measurement and the failure was indistinguishable
    from a rejection.  A deadline that moves with the suite cannot silently
    become too small when the suite grows.

    Raises:
        ValueError: If any count is not a positive integer or any cost factor
            is not a positive number.
    """
    for name, count in (
        ("num_contexts", num_contexts),
        ("repeats", repeats),
        ("arms", arms),
        ("concurrency", concurrency),
        ("benchmark_max_tokens", benchmark_max_tokens),
        ("correctness_max_tokens", correctness_max_tokens),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    for name, count in (
        ("warmup_repeats", warmup_repeats),
        ("correctness_repeats", correctness_repeats),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be an integer >= 0")
    for name, factor in (
        ("seconds_per_token", seconds_per_token),
        ("fixed_overhead_seconds", fixed_overhead_seconds),
        ("safety_factor", safety_factor),
    ):
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(f"{name} must be a positive number")

    generations = arms * (warmup_repeats + repeats) * num_contexts
    generation_seconds = (
        generations * benchmark_max_tokens * float(seconds_per_token) / concurrency
    )
    # The correctness pass gives up all batching, so it does not get to divide
    # by ``concurrency`` -- but it *is* bounded an order of magnitude tighter
    # than the throughput pass, and charging it that tighter cap is the whole
    # point of costing tokens.  Job 369040 measured this pass at 396s while the
    # per-generation formula charged it 4120s.
    correctness_seconds = (
        arms
        * correctness_repeats
        * num_contexts
        * correctness_max_tokens
        * float(seconds_per_token)
    )
    budget = (
        (generation_seconds + correctness_seconds) * float(safety_factor)
        + float(fixed_overhead_seconds)
    )
    return min(max(budget, BENCHMARK_MIN_SECONDS), BENCHMARK_MAX_SECONDS)


@dataclass(frozen=True, slots=True)
class OrchestratorTimeouts:
    """Timeouts for runtime and gate effects."""

    quiesce: float = 30.0
    sleep: float = 120.0
    candidate_start: float = 600.0
    #: Hard upper bound on one benchmark, not the expected duration.
    #:
    #: The deadline actually handed to the gate is
    #: ``min(gate.estimated_benchmark_seconds(), benchmark)`` -- see
    #: :meth:`TunerOrchestrator._benchmark_timeout` -- so this field is what
    #: stops a genuine hang, while :func:`derive_benchmark_timeout` is what
    #: keeps a healthy-but-large suite from being cut off mid-measurement.
    benchmark: float = BENCHMARK_MAX_SECONDS
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
    #: The gate measured both arms and declined to promote.  A scientific
    #: result: ``decision_path`` points at the measurement behind it.
    REJECTED = "rejected"
    VAL_LOSS_NOT_IMPROVED = "val_loss_not_improved"
    PREEMPTED = "preempted"
    #: The benchmark's whole-run deadline expired.  Infrastructure failure,
    #: not a result: no arms were compared and there is no decision to read.
    BENCHMARK_TIMED_OUT = "benchmark_timed_out"
    #: Serving activity preempted the benchmark before it could measure.
    BENCHMARK_ABORTED = "benchmark_aborted"
    FINAL_ASSISTANT_MASK_ERROR = "final_assistant_mask_error"
    FAILED = "failed"


#: Terminal outcome for each way the gate can fail without measuring.
_GATE_FAILURE_OUTCOMES: Final[Mapping[GateFailure, CycleOutcome]] = {
    GateFailure.ABORTED: CycleOutcome.BENCHMARK_ABORTED,
    GateFailure.TIMED_OUT: CycleOutcome.BENCHMARK_TIMED_OUT,
}


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Observable cycle result for logs and scheduling policy."""

    outcome: CycleOutcome
    artifact_id: str | None = None
    gate: GateResult | None = None
    error: str | None = None
    #: Where the gate decision was persisted, when the gate produced one.
    decision_path: Path | None = None
    val_loss: float | None = None
    #: Whether serving was left on the draft the durable active pointer names.
    #:
    #: Deliberately a field of its own rather than another ``CycleOutcome``.
    #: The outcome answers "why did this cycle end", which is a fact about the
    #: cycle; this answers "what is the engine serving now", which is a fact
    #: about the runtime and outlives the cycle entirely.  Folding the second
    #: into the first is what produced the defect: a preemption whose rollback
    #: could not respawn reported ``PREEMPTED`` -- true, and the reason the
    #: cycle stopped -- while an unvalidated draft answered live traffic under
    #: a pointer naming the incumbent.  Keeping them orthogonal also means
    #: every outcome can carry the flag, including the ``PROMOTED`` and
    #: ``NOT_IDLE`` paths where a *previous* cycle's failure is still
    #: outstanding.
    serving_restored: bool = True


class TunerOrchestrator:
    """Drive one complete cycle and restore serving on every terminal branch."""

    def __init__(
        self,
        *,
        state: TunerStateMachine,
        idle: IdleDetector,
        backend: SpeculatorBackend,
        artifacts: ArtifactRegistry,
        runtime: RuntimeController,
        gate: BenchmarkGate,
        work_root: Path,
        timeouts: OrchestratorTimeouts = _DEFAULT_TIMEOUTS,
        run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        val_loss_prefilter: ValLossPreFilterConfig | None = None,
    ) -> None:
        self._state = state
        self._idle = idle
        self._backend = backend
        self._artifacts = artifacts
        self._runtime = runtime
        self._gate = gate
        self._work_root = work_root
        self._timeouts = timeouts
        self._run_id_factory = run_id_factory
        self._val_loss_prefilter = val_loss_prefilter
        # Beside ``state.json``, not under ``work_root``: the condition belongs
        # to the runtime, not to the run that happened to expose it, and it has
        # to be findable without knowing which run directory was last written.
        self._unrestored_path = state.state_path.parent / SERVING_UNRESTORED_FILE_NAME

    @property
    def serving_unrestored(self) -> bool:
        """Whether a recorded restore failure is still outstanding.

        Read from the durable marker rather than from memory, so it survives a
        process restart -- the supervisor relaunches the child vLLM, but from
        *its* view of the active draft, which is exactly the bookkeeping this
        condition says cannot be trusted.
        """
        return self._unrestored_path.exists()

    def run_once(self) -> CycleResult:
        """Poll idle state and, when eligible, execute one complete tuning cycle."""
        if self._state.state is not TunerState.READY or self.serving_unrestored:
            recovery_errors = self.recover()
            if recovery_errors:
                return CycleResult(
                    outcome=CycleOutcome.FAILED,
                    error="; ".join(recovery_errors),
                    serving_restored=not self.serving_unrestored,
                )
        if not self._idle.should_tune:
            return CycleResult(
                outcome=CycleOutcome.NOT_IDLE,
                serving_restored=not self.serving_unrestored,
            )

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
            prepared = self._backend.prepare(
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            extracted = self._backend.extract(
                prepared,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.TRAINING, reason="training signals extracted")
            trained = self._backend.train(
                extracted,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            draft_directory = self._backend.materialize(
                trained,
                work_dir,
                should_abort=lambda: guard.is_preempted,
            )
            self._backend.validate(
                draft_directory,
                should_abort=lambda: guard.is_preempted,
            )
            backend_info = self._backend.describe()
            # Read defensively: ``train`` is a protocol method, and a backend
            # that predates this field returns a result without one.  A missing
            # val_loss means "unavailable", which the pre-filter below treats
            # as fail-open -- it must never turn into a failed cycle.
            val_loss = getattr(trained, "val_loss", None)
            artifact = self._artifacts.publish(
                draft_directory,
                ArtifactSpec(
                    verifier_model=backend_info.verifier_model,
                    draft_model=backend_info.draft_model,
                    base_draft=backend_info.from_pretrained,
                    trace_hash=prepared.snapshot.content_hash,
                    training_params=backend_info.training_params,
                    val_loss=val_loss,
                ),
            )
            artifact_id = artifact.artifact_id

            # Pre-filter: skip the expensive benchmark if validation loss
            # did not improve.  This is a COST FILTER, not a promotion
            # criterion — the acceptance gate remains the sole authority.
            incumbent_val_loss = None
            incumbent = self._artifacts.active()
            if incumbent is not None:
                incumbent_val_loss = incumbent.manifest.val_loss

            if (
                self._val_loss_prefilter is not None
                and self._val_loss_prefilter.enabled
                and val_loss is not None
                and incumbent_val_loss is not None
            ):
                improvement = incumbent_val_loss - val_loss
                if improvement < self._val_loss_prefilter.min_improvement:
                    # Validation loss did not improve enough — skip benchmark.
                    self._state.transition(
                        TunerState.ROLLING_BACK,
                        reason=(
                            f"val_loss not improved: candidate={val_loss:.4f}, "
                            f"incumbent={incumbent_val_loss:.4f}, "
                            f"improvement={improvement:.4f}, "
                            f"threshold={self._val_loss_prefilter.min_improvement:.4f}"
                        ),
                    )
                    cleanup_errors = self._finish_rollback(pointer_changed=False)
                    return CycleResult(
                        outcome=(
                            CycleOutcome.VAL_LOSS_NOT_IMPROVED
                            if not cleanup_errors
                            else CycleOutcome.FAILED
                        ),
                        artifact_id=artifact_id,
                        error="; ".join(cleanup_errors) if cleanup_errors else None,
                        val_loss=val_loss,
                        serving_restored=not self.serving_unrestored,
                    )
            guard.check()

            self._state.transition(
                TunerState.CANDIDATE_STARTING,
                reason=f"candidate {artifact_id} materialized",
            )
            self._prepare_draft_hot_swap()
            self._runtime.start_candidate(
                artifact.path,
                timeout_seconds=self._timeouts.candidate_start,
                should_abort=lambda: guard.is_preempted,
            )
            guard.check()

            self._state.transition(TunerState.BENCHMARKING, reason="candidate running")
            gate_result = self._gate.benchmark(
                artifact.path,
                timeout_seconds=self._benchmark_timeout(),
                should_abort=lambda: guard.is_preempted,
            )
            # Persist before the preemption check: a completed benchmark is a
            # real measurement even if the cycle is abandoned immediately after.
            decision_path = self._persist_decision(work_dir, gate_result)
            guard.check()

            if not gate_result.passed:
                # A gate that never measured is not a rejection.  Both paths
                # roll back identically; only the reported outcome differs, so
                # that a reader of scheduler.json can tell an infrastructure
                # failure from a scientific one without opening the metrics.
                failed_outcome = (
                    CycleOutcome.REJECTED
                    if gate_result.failure is None
                    else _GATE_FAILURE_OUTCOMES[gate_result.failure]
                )
                verb = (
                    "rejected candidate"
                    if gate_result.failure is None
                    else f"could not measure candidate ({gate_result.failure.value})"
                )
                self._state.transition(
                    TunerState.ROLLING_BACK,
                    reason=f"gate {verb}: {gate_result.reason}",
                )
                cleanup_errors = self._finish_rollback(pointer_changed=False)
                return CycleResult(
                    outcome=(
                        failed_outcome
                        if not cleanup_errors
                        else CycleOutcome.FAILED
                    ),
                    artifact_id=artifact_id,
                    gate=gate_result,
                    error="; ".join(cleanup_errors) if cleanup_errors else None,
                    decision_path=decision_path,
                    val_loss=val_loss,
                    serving_restored=not self.serving_unrestored,
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
                decision_path=decision_path,
                val_loss=val_loss,
                serving_restored=not self.serving_unrestored,
            )
        except FinalAssistantMaskError as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.FINAL_ASSISTANT_MASK_ERROR,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
                serving_restored=not self.serving_unrestored,
            )
        except TuningPreempted as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.PREEMPTED,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
                serving_restored=not self.serving_unrestored,
            )
        except Exception as exc:
            cleanup_errors = self._rollback_after_failure(pointer_changed)
            return CycleResult(
                outcome=CycleOutcome.FAILED,
                artifact_id=artifact_id,
                error=_combine_error(exc, cleanup_errors),
                serving_restored=not self.serving_unrestored,
            )

    def _prepare_draft_hot_swap(self) -> None:
        """Wake the engine here, and only here, so a hot-swap is reachable.

        The cycle sleeps vLLM at level 1 *before* training, and that sleep is
        load-bearing: it is what releases the device memory the training engine
        then sizes itself from (``RuntimeController._await_gpu_memory``). But a
        level-1 sleep offloads the drafter's weights, and weights that are not
        resident cannot be overwritten -- so on arrival at
        ``start_candidate`` the engine was always asleep and the swap was
        unreachable no matter how it was implemented.

        The fix is a wake placed at exactly the point a full restart would
        otherwise have happened. That is what keeps the memory guarantee
        intact: every training step has already finished by the time this runs,
        and re-mapping the sleeping engine's own allocations costs no more
        device memory than launching a replacement child would have. Nothing
        about the sleep-before-training ordering moves.

        Waking restores the weights resident at sleep time -- the stock active
        draft -- so the candidate is applied *on top of* a woken engine rather
        than replacing a configured one.

        Every failure mode lands on the pre-existing full-restart path:
        the runtime cannot swap (no endpoint, method absent), it can but the
        wake failed, or the swap itself failed. In all three the engine stays
        asleep or gets restarted by ``start_candidate``, which is what it did
        before this method existed. It therefore never raises.
        """
        if not getattr(self._runtime, "supports_draft_hot_swap", False):
            return
        wake_for_swap = getattr(self._runtime, "wake_for_swap", None)
        if not callable(wake_for_swap):
            return
        try:
            wake_for_swap(timeout_seconds=self._timeouts.wake)
        except Exception:
            # Not a cycle failure: start_candidate restarts the child, which is
            # exactly the sequencing this cycle would have used anyway.
            logger.warning(
                "could not wake vLLM for an in-place draft hot-swap; the "
                "candidate will be started by a full restart instead",
                exc_info=True,
            )

    def _benchmark_timeout(self) -> float:
        """Deadline for this cycle's benchmark, derived where the gate can.

        Read defensively, in the same spirit as ``val_loss`` above:
        ``estimated_benchmark_seconds`` is not part of the
        :class:`BenchmarkGate` protocol, so a gate that predates it -- or one
        that cannot size itself yet because no suite exists -- simply falls
        back to the configured hard ceiling.  Sizing the deadline must never be
        able to fail a cycle.

        The estimate is clamped by ``timeouts.benchmark`` in both directions of
        intent: it may shrink the ceiling but never raise it, so the hard bound
        on a hang stays exactly where it was configured.
        """
        ceiling = self._timeouts.benchmark
        estimator = getattr(self._gate, "estimated_benchmark_seconds", None)
        if not callable(estimator):
            return ceiling
        try:
            estimate = estimator()
        except Exception:
            return ceiling
        if (
            isinstance(estimate, bool)
            or not isinstance(estimate, (int, float))
            or estimate <= 0
        ):
            return ceiling
        return min(float(estimate), ceiling)

    def _persist_decision(self, run_dir: Path, gate_result: GateResult) -> Path | None:
        """Record the gate's decision so ``speedlm gain`` can report it.

        Both promotions and rejections are written: a rejection is a real
        measured result and must stay reportable.  A failure to persist is
        fail-closed — the exception propagates, the cycle rolls back and
        restores serving — because an unrecorded benchmark is exactly the
        blindness this write exists to remove.

        The raw metrics bodies are written first and unconditionally: a
        benchmark that aborted before producing a decision still scraped
        counters, and that partial evidence is worth keeping.
        """
        write_metrics_bodies(run_dir, gate_result.metrics_bodies)
        if gate_result.decision is None:
            return None
        return write_decision(run_dir, gate_result.decision)

    def recover(self) -> tuple[str, ...]:
        """Restore the durable active draft after an interrupted process.

        Also the re-attempt path for a cycle that *finished* with serving on the
        wrong draft.  That case reaches here at ``READY`` -- the cycle really did
        end, and ``READY`` is the honest state for it -- so the early return can
        no longer be conditioned on the state alone: it is conditioned on the
        state *and* the durable marker.  Without the second half nothing ever
        retried the restore, which is what let an abandoned candidate keep
        answering live traffic indefinitely.
        """
        if self._state.state is TunerState.READY:
            if not self.serving_unrestored:
                return ()
            logger.error(
                "re-attempting a restore that previously failed; serving is "
                "still on a draft the active pointer does not name"
            )
            error = self._restore_serving()
            return () if error is None else (error,)
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
            restore_error = self._restore_serving()
            if restore_error is not None:
                errors.append(restore_error)
            try:
                self._state.transition(
                    TunerState.WAKING,
                    reason=(
                        "runtime restored"
                        if restore_error is None
                        else f"SERVING NOT RESTORED: {restore_error}"
                    ),
                )
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
        """Undo the cycle's effects and hand serving back, loudly if it cannot.

        The wake at the end runs *even when the restore failed*, which is a
        deliberate choice and the one worth arguing.  Waking is what reopens the
        gateway's admission gate (``RuntimeController.wake`` ->
        ``AdmissionGate._start_admitting``), so skipping it would keep admission
        closed -- and this system never refuses traffic, it only queues it:
        ``speedlm.gateway.proxy`` waits on admission in an unbounded loop.  A
        closed gate is therefore an unbounded outage for every client, not a
        polite 503.

        What it would be buying is bounded by comparison.  The draft head is a
        *proposer*: vLLM's verifier accepts or rejects every drafted token, so a
        wrong or unvalidated draft changes speed, not answers.  (This is the
        same losslessness ``PromotionConfig.min_divergence_token_index``
        documents from the other side.)  So the choice is between a total
        outage of unknown duration and a throughput regression of bounded size,
        and the regression wins -- but only because it is made loud here and
        retried promptly by
        :meth:`speedlm.tuner.service.TunerService._recover_unrestored_serving`,
        which re-attempts ahead of the retry cooldown rather than behind it.

        If the draft head were ever able to change *outputs* -- a non-lossless
        speculative mode, a swap that also moved the verifier -- this trade
        reverses and admission should stay closed.  It is the losslessness that
        makes serving-on-the-wrong-draft survivable, not the cost of downtime.
        """
        errors: list[str] = []
        if pointer_changed:
            try:
                self._artifacts.rollback()
            except Exception as exc:
                errors.append(f"active pointer rollback failed: {exc}")
        restore_error: str | None = None
        if self._state.state is TunerState.ROLLING_BACK:
            restore_error = self._restore_serving()
            if restore_error is not None:
                errors.append(restore_error)
            try:
                self._state.transition(
                    TunerState.WAKING,
                    reason=(
                        "active draft restored"
                        if restore_error is None
                        else f"SERVING NOT RESTORED: {restore_error}"
                    ),
                )
            except Exception as exc:
                errors.append(f"state rollback failed: {exc}")
        if self._state.state is TunerState.WAKING:
            try:
                self._runtime.wake(timeout_seconds=self._timeouts.wake)
            except Exception as exc:
                errors.append(f"runtime wake failed: {exc}")
            try:
                self._state.transition(
                    TunerState.READY,
                    reason=(
                        "serving restored"
                        if not self.serving_unrestored
                        else "SERVING NOT RESTORED: admission reopened onto a "
                        "draft the active pointer does not name"
                    ),
                )
            except Exception as exc:
                errors.append(f"state rollback failed: {exc}")
        return tuple(errors)

    def _restore_serving(self) -> str | None:
        """Put the engine back on the durable active draft, recording failure.

        Returns the error message when the restore did not take, and ``None``
        when it did.  Success clears the durable marker, so the condition can
        only ever be outstanding while it is still true -- there is no separate
        "all clear" path that could forget to run.
        """
        try:
            self._runtime.restore(
                self._active_draft(),
                timeout_seconds=self._timeouts.restore,
            )
        except Exception as exc:
            message = f"runtime restore failed: {exc}"
            self._mark_serving_unrestored(message)
            return message
        self._clear_serving_unrestored()
        return None

    def _mark_serving_unrestored(self, error: str) -> None:
        """Record, durably and at ERROR level, that serving is on a wrong draft.

        Marking must never itself raise: it runs on a cleanup path that is
        already handling one failure, and a marker that cannot be written is
        strictly less bad than a cleanup that aborts partway through.  The log
        line is emitted first for exactly that reason -- it is the half that
        cannot fail on a full disk.
        """
        try:
            expected = str(self._active_draft())
        except Exception:
            expected = "unknown"
        logger.error(
            "SERVING NOT RESTORED: could not return the engine to the active "
            "draft %s (%s). Live traffic is being answered by whichever draft "
            "the abandoned cycle left loaded, which the durable active pointer "
            "does not name. Speculative decoding is lossless, so answers are "
            "unaffected, but throughput is unvalidated until this clears.",
            expected,
            error,
        )
        try:
            atomic_write_json(
                self._unrestored_path,
                {
                    "schema_version": 1,
                    "detected_at": time.time(),
                    "expected_active_draft": expected,
                    "error": error,
                },
            )
        except OSError as exc:
            logger.warning(
                "could not persist the serving-unrestored marker %s: %s",
                self._unrestored_path,
                exc,
            )

    def _clear_serving_unrestored(self) -> None:
        try:
            self._unrestored_path.unlink(missing_ok=True)
        except OSError as exc:
            # Leaving it behind only costs a redundant restore attempt on the
            # next poll, which is the safe direction to fail in.
            logger.warning(
                "could not clear the serving-unrestored marker %s: %s",
                self._unrestored_path,
                exc,
            )

    def _active_draft(self) -> DraftReference:
        active = self._artifacts.active()
        if active is not None:
            return active.path
        return self._backend.describe().from_pretrained


def _combine_error(error: Exception, cleanup_errors: tuple[str, ...]) -> str:
    message = str(error)
    if cleanup_errors:
        return f"{message}; cleanup: {'; '.join(cleanup_errors)}"
    return message
