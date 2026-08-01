"""Opt-in background service for idle speculative tuning."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Protocol

from speedlm.config import SpeedLMConfig
from speedlm.profiles import ModelProfile, resolve_profile
from speedlm.storage import atomic_write_json, ensure_layout, resolve_layout
from speedlm.traces.store import TraceStats
from speedlm.training.base import SpeculatorBackend
from speedlm.tuner.artifacts import ArtifactRegistry
from speedlm.tuner.idle import ActivitySource, IdleDetector
from speedlm.tuner.orchestrator import (
    BenchmarkGate,
    CycleOutcome,
    CycleResult,
    OrchestratorTimeouts,
    RuntimeController,
    TunerOrchestrator,
)
from speedlm.tuner.state import TunerStateMachine

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_MIN_TRACE_RECORDS = 32
DEFAULT_MIN_CORPUS_RECORDS = 256
SCHEDULER_STATUS_FILE = "scheduler.json"

#: Consecutive idle polls required before arming, when config carries no
#: preference.  One preserves the historical single-sample behaviour; the
#: production default lives on
#: :attr:`speedlm.config.IdleTuningConfig.idle_confirmations`.
DEFAULT_IDLE_CONFIRMATIONS = 1

#: Quiet period after an unproductive cycle, when config carries no preference.
#: Zero preserves the historical "retry as soon as the watermark moves"
#: behaviour; see :attr:`speedlm.config.IdleTuningConfig.retry_cooldown_seconds`.
DEFAULT_RETRY_COOLDOWN_SECONDS = 0.0

#: Outcomes that arm the retry cooldown.
#:
#: These are the cycles that spent engine restarts without producing a
#: measurement: the gateway was not really idle, or the cycle could not
#: complete.  Repeating them immediately re-pays the restart to reach the same
#: conclusion, which is what job 369040 did 9.2s after a preempted cycle.
#:
#: PROMOTED, REJECTED and VAL_LOSS_NOT_IMPROVED are deliberately absent: those
#: cycles produced a real result, and the existing watermark dedupe already
#: stops them from repeating on unchanged traces.
COOLDOWN_OUTCOMES: frozenset[CycleOutcome] = frozenset(
    {
        CycleOutcome.PREEMPTED,
        CycleOutcome.BENCHMARK_ABORTED,
        CycleOutcome.BENCHMARK_TIMED_OUT,
        CycleOutcome.FAILED,
        CycleOutcome.FINAL_ASSISTANT_MASK_ERROR,
    }
)


class TunerServiceConfigurationError(ValueError):
    """Raised when service dependencies disagree with the selected profile."""


class TunerServiceStopError(RuntimeError):
    """Raised when a tuner worker does not stop within the requested timeout."""


class TunerServiceStartupError(RuntimeError):
    """Raised when startup serving recovery fails or does not complete."""


class TraceStatsSource(Protocol):
    """Trace-buffer surface used by the scheduling policy and by retention."""

    def stats(self) -> TraceStats: ...

    def prune(self) -> int: ...


class CycleRunner(Protocol):
    """Orchestrator surface needed by :class:`TunerService`."""

    def run_once(self) -> CycleResult: ...

    def recover(self) -> tuple[str, ...]: ...


OrchestratorFactory = Callable[[ActivitySource], CycleRunner]


@dataclass(frozen=True, slots=True)
class _TraceWatermark:
    count: int
    tokens: int
    oldest: float | None
    newest: float | None
    unknown_token_records: int

    @classmethod
    def from_stats(cls, stats: TraceStats) -> _TraceWatermark:
        return cls(
            count=stats.count,
            tokens=stats.tokens,
            oldest=stats.oldest,
            newest=stats.newest,
            unknown_token_records=stats.unknown_token_records,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "tokens": self.tokens,
            "oldest": self.oldest,
            "newest": self.newest,
            "unknown_token_records": self.unknown_token_records,
        }


class _PreemptibleActivitySource:
    """Add service shutdown to the gateway's request-driven abort signal."""

    def __init__(self, source: ActivitySource, stop_requested: Event) -> None:
        self._source = source
        self._stop_requested = stop_requested

    @property
    def in_flight(self) -> int:
        if self._stop_requested.is_set():
            return max(1, self._source.in_flight)
        return self._source.in_flight

    @property
    def last_activity(self) -> float:
        if self._stop_requested.is_set():
            return float("inf")
        return self._source.last_activity


def build_tuner_orchestrator(
    config: SpeedLMConfig,
    *,
    activity: ActivitySource,
    backend: SpeculatorBackend,
    gate: BenchmarkGate,
    runtime: RuntimeController,
    home: Path | None = None,
    profiles: Mapping[str, ModelProfile] | None = None,
    state: TunerStateMachine | None = None,
    artifacts: ArtifactRegistry | None = None,
    work_root: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    timeouts: OrchestratorTimeouts | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> TunerOrchestrator:
    """Assemble a ready orchestrator from config and concrete effect objects.

    The profile is resolved before any durable tuner objects are created, and
    the injected backend must describe the same verifier and draft. This keeps
    an explicit profile selection from silently training the wrong model pair.
    """

    profile = resolve_profile(
        {"model": config.model, "profile": config.profile},
        profiles=profiles,
        home=home,
    )
    _validate_backend_profile(profile, backend)

    layout = ensure_layout(home)
    state_machine = state or TunerStateMachine(layout.runs_dir)
    artifact_registry = artifacts or ArtifactRegistry(layout.runs_dir)

    kwargs: dict[str, object] = {}
    if timeouts is not None:
        kwargs["timeouts"] = timeouts
    if run_id_factory is not None:
        kwargs["run_id_factory"] = run_id_factory
    kwargs["val_loss_prefilter"] = config.tuning.val_loss_prefilter

    return TunerOrchestrator(
        state=state_machine,
        idle=IdleDetector(
            activity,
            threshold_seconds=config.idle_threshold_seconds,
            clock=clock,
        ),
        backend=backend,
        artifacts=artifact_registry,
        runtime=runtime,
        gate=gate,
        work_root=work_root or layout.runs_dir,
        **kwargs,  # type: ignore[arg-type]
    )


def create_tuner_service(
    config: SpeedLMConfig,
    *,
    activity: ActivitySource,
    traces: TraceStatsSource,
    backend: SpeculatorBackend,
    gate: BenchmarkGate,
    runtime: RuntimeController,
    enabled: bool | None = None,
    min_trace_records: int = DEFAULT_MIN_TRACE_RECORDS,
    min_corpus_records: int = DEFAULT_MIN_CORPUS_RECORDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    home: Path | None = None,
    profiles: Mapping[str, ModelProfile] | None = None,
    state: TunerStateMachine | None = None,
    artifacts: ArtifactRegistry | None = None,
    work_root: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    timeouts: OrchestratorTimeouts | None = None,
    run_id_factory: Callable[[], str] | None = None,
    status_path: Path | None = None,
) -> TunerService:
    """Create the production service while retaining every effect seam.

    ``enabled`` defaults to ``config.tuning_enabled`` when that config seam is
    available, and otherwise to ``False``. An explicit argument permits CLI
    wiring before or independently of persisted configuration.
    """

    def orchestrator_factory(cycle_activity: ActivitySource) -> TunerOrchestrator:
        return build_tuner_orchestrator(
            config,
            activity=cycle_activity,
            backend=backend,
            gate=gate,
            runtime=runtime,
            home=home,
            profiles=profiles,
            state=state,
            artifacts=artifacts,
            work_root=work_root,
            clock=clock,
            timeouts=timeouts,
            run_id_factory=run_id_factory,
        )

    return TunerService(
        config,
        activity=activity,
        traces=traces,
        orchestrator_factory=orchestrator_factory,
        enabled=enabled,
        min_trace_records=min_trace_records,
        min_corpus_records=min_corpus_records,
        poll_interval_seconds=poll_interval_seconds,
        clock=clock,
        wall_clock=wall_clock,
        status_path=(
            resolve_layout(home).runs_dir / SCHEDULER_STATUS_FILE
            if status_path is None
            else status_path
        ),
    )


class TunerService:
    """Run at most one tuning cycle for each eligible trace-buffer watermark."""

    def __init__(
        self,
        config: SpeedLMConfig,
        *,
        activity: ActivitySource,
        traces: TraceStatsSource,
        orchestrator_factory: OrchestratorFactory,
        enabled: bool | None = None,
        min_trace_records: int = DEFAULT_MIN_TRACE_RECORDS,
        min_corpus_records: int = DEFAULT_MIN_CORPUS_RECORDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        idle_confirmations: int | None = None,
        retry_cooldown_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        status_path: Path | None = None,
    ) -> None:
        if enabled is None:
            enabled = getattr(config, "tuning_enabled", False)
        if not isinstance(enabled, bool):
            raise TunerServiceConfigurationError("tuning enabled flag must be boolean")
        if (
            isinstance(min_trace_records, bool)
            or not isinstance(min_trace_records, int)
            or min_trace_records < 1
        ):
            raise TunerServiceConfigurationError("min_trace_records must be a positive integer")
        if (
            isinstance(min_corpus_records, bool)
            or not isinstance(min_corpus_records, int)
            or min_corpus_records < 1
        ):
            raise TunerServiceConfigurationError("min_corpus_records must be a positive integer")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or poll_interval_seconds <= 0
        ):
            raise TunerServiceConfigurationError("poll_interval_seconds must be a positive number")

        tuning = getattr(config, "tuning", None)
        if idle_confirmations is None:
            idle_confirmations = getattr(
                tuning, "idle_confirmations", DEFAULT_IDLE_CONFIRMATIONS
            )
        if retry_cooldown_seconds is None:
            retry_cooldown_seconds = getattr(
                tuning, "retry_cooldown_seconds", DEFAULT_RETRY_COOLDOWN_SECONDS
            )
        if (
            isinstance(idle_confirmations, bool)
            or not isinstance(idle_confirmations, int)
            or idle_confirmations < 1
        ):
            raise TunerServiceConfigurationError(
                "idle_confirmations must be a positive integer"
            )
        if (
            isinstance(retry_cooldown_seconds, bool)
            or not isinstance(retry_cooldown_seconds, (int, float))
            or retry_cooldown_seconds < 0
        ):
            raise TunerServiceConfigurationError(
                "retry_cooldown_seconds must be a non-negative number"
            )

        self._enabled = enabled
        self._traces = traces
        self._min_trace_records = min_trace_records
        self._min_corpus_records = min_corpus_records
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._idle_confirmations = int(idle_confirmations)
        self._retry_cooldown_seconds = float(retry_cooldown_seconds)
        #: Consecutive polls so far that observed an idle gateway.
        self._idle_streak = 0
        #: Monotonic instant before which no new cycle may be attempted.
        self._cooldown_until: float | None = None
        self._clock = clock
        self._stop_requested = Event()
        self._startup_complete = Event()
        self._activation_requested = Event()
        self._activity = _PreemptibleActivitySource(activity, self._stop_requested)
        self._idle = IdleDetector(
            self._activity,
            threshold_seconds=config.idle_threshold_seconds,
            clock=clock,
        )
        self._orchestrator = orchestrator_factory(self._activity)
        self._lock = Lock()
        self._status_write_lock = Lock()
        self._thread: Thread | None = None
        self._last_attempted: _TraceWatermark | None = None
        self._last_status_watermark: _TraceWatermark | None = None
        self._last_result: CycleResult | None = None
        self._last_status_result: CycleResult | None = None
        self._last_error: str | None = None
        self._startup_error: str | None = None
        self._status_path = status_path
        self._wall_clock = wall_clock
        created_at = self._wall_clock()
        self._created_at = created_at
        self._lifecycle = "stopped"
        self._lifecycle_changed_at = created_at
        self._last_attempt_at: float | None = None
        self._last_result_at: float | None = None
        self._last_error_at: float | None = None
        self._persist_scheduler_status()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self) -> CycleResult | None:
        with self._lock:
            return self._last_result

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def orchestrator(self) -> CycleRunner:
        """Return the assembled runner for diagnostics and tests."""

        return self._orchestrator

    def start(self, *, paused: bool = False) -> None:
        """Start the watcher once; disabled services remain inert."""

        if not isinstance(paused, bool):
            raise ValueError("paused must be boolean")
        if not self._enabled:
            logger.info("idle tuner is disabled")
            self._set_lifecycle("stopped")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested.clear()
            self._startup_complete.clear()
            self._startup_error = None
            if paused:
                self._activation_requested.clear()
            else:
                self._activation_requested.set()
            thread = Thread(
                target=self._run,
                name="speedlm-idle-tuner",
                daemon=False,
            )
            self._thread = thread
        self._set_lifecycle("startup")
        try:
            thread.start()
        except BaseException as exc:
            with self._lock:
                if self._thread is thread:
                    self._thread = None
                self._startup_error = str(exc)
            self._record_error(exc)
            self._set_lifecycle("stopped")
            self._startup_complete.set()
            raise

    def activate(self) -> None:
        """Allow polling after an externally held admission gate is released."""
        self._activation_requested.set()

    def wait_started(self, *, timeout_seconds: float | None = None) -> None:
        """Wait until startup recovery has completed successfully."""
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number or None")
        if not self._startup_complete.wait(timeout_seconds):
            raise TunerServiceStartupError(
                f"tuner startup recovery did not finish within {timeout_seconds} seconds"
            )
        with self._lock:
            error = self._startup_error
            lifecycle = self._lifecycle
        if error is not None:
            raise TunerServiceStartupError(
                f"tuner startup recovery failed: {error}"
            )
        if lifecycle != "running":
            raise TunerServiceStartupError(
                f"tuner stopped during startup recovery (lifecycle={lifecycle})"
            )

    def stop(self, *, timeout_seconds: float | None = None) -> None:
        """Preempt any active cycle, restore serving, and join the watcher."""

        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number or None")

        self._stop_requested.set()
        self._activation_requested.set()
        self._set_lifecycle("stopping")
        with self._lock:
            thread = self._thread
        if thread is None or thread is current_thread():
            self._set_lifecycle("stopped")
            return

        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TunerServiceStopError(
                f"tuner service did not stop within {timeout_seconds} seconds"
            )
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        try:
            try:
                errors = self._orchestrator.recover()
                if errors:
                    raise TunerServiceStartupError("; ".join(errors))
            except Exception as exc:
                logger.exception("idle tuner could not recover serving during startup")
                with self._lock:
                    self._startup_error = str(exc)
                self._record_error(exc)
                return
            self._set_lifecycle("running")
            self._startup_complete.set()
            while (
                not self._stop_requested.is_set()
                and not self._activation_requested.wait(self._poll_interval_seconds)
            ):
                pass
            while not self._stop_requested.is_set():
                self._poll_once()
                self._stop_requested.wait(self._poll_interval_seconds)
        finally:
            self._recover_serving("service shutdown")
            self._set_lifecycle("stopped")
            self._startup_complete.set()

    def _poll_once(self) -> None:
        if self._in_cooldown():
            self._idle_streak = 0
            return
        if not self._idle.should_tune or self._stop_requested.is_set():
            # One busy sample resets the window: the confirmations must be
            # *consecutive* or they measure nothing a single sample did not.
            self._idle_streak = 0
            return
        self._idle_streak += 1
        if self._idle_streak < self._idle_confirmations:
            return

        try:
            watermark = _TraceWatermark.from_stats(self._traces.stats())
        except Exception as exc:
            logger.exception("idle tuner could not inspect the trace buffer")
            self._record_error(exc)
            return

        if (
            watermark.count < self._min_trace_records
            or watermark.count < self._min_corpus_records
            or watermark == self._last_attempted
            or self._stop_requested.is_set()
        ):
            return

        self._idle_streak = 0
        self._record_attempt(watermark)
        try:
            result = self._orchestrator.run_once()
        except Exception as exc:
            self._last_attempted = watermark
            self._arm_cooldown("cycle exception")
            logger.exception("idle tuning cycle raised an exception")
            self._record_error(exc)
            self._recover_serving("cycle exception")
            return

        if result.outcome is not CycleOutcome.NOT_IDLE:
            self._last_attempted = watermark
        if result.outcome in COOLDOWN_OUTCOMES:
            self._arm_cooldown(result.outcome.value)
        self._prune_traces()
        self._record_result(result)
        if result.outcome in {
            CycleOutcome.FAILED,
            CycleOutcome.FINAL_ASSISTANT_MASK_ERROR,
        }:
            logger.error("idle tuning cycle failed: %s", result.error or "unknown error")
            if result.outcome is CycleOutcome.FAILED:
                self._recover_serving("failed cycle")
        elif result.outcome is CycleOutcome.BENCHMARK_TIMED_OUT:
            # An infrastructure failure, not a rejection: the cycle already
            # rolled back and restored serving, but a deadline that cannot fit
            # the benchmark will recur every cycle until it is noticed.
            logger.error(
                "idle tuning benchmark exceeded its deadline without measuring: %s",
                result.gate.reason if result.gate is not None else "unknown reason",
            )
        elif result.outcome is CycleOutcome.BENCHMARK_ABORTED:
            logger.info("idle tuning benchmark preempted by serving activity")
        elif result.outcome is CycleOutcome.PREEMPTED:
            logger.info("idle tuning cycle preempted by serving activity")
        elif result.outcome is not CycleOutcome.NOT_IDLE:
            logger.info("idle tuning cycle completed: %s", result.outcome.value)

    def _in_cooldown(self) -> bool:
        """Whether a previous unproductive cycle still bars a new attempt.

        The cooldown expires by wall progress alone, so an expired one is
        cleared here rather than being re-tested on every later poll.
        """
        until = self._cooldown_until
        if until is None:
            return False
        if self._clock() < until:
            return True
        self._cooldown_until = None
        logger.info("idle tuning retry cooldown expired")
        return False

    def _arm_cooldown(self, reason: str) -> None:
        """Bar the next attempt for the configured quiet period.

        Deliberately *not* gated on the trace watermark.  The watermark dedupe
        cannot see this case at all: the request that preempts a cycle is
        itself traced, so it advances the watermark and the next poll looks
        like a brand-new opportunity even though nothing about the situation
        has changed.
        """
        if self._retry_cooldown_seconds <= 0:
            return
        self._cooldown_until = self._clock() + self._retry_cooldown_seconds
        self._idle_streak = 0
        logger.info(
            "idle tuning cycle ended as %s; suppressing retries for %gs",
            reason,
            self._retry_cooldown_seconds,
        )

    def _prune_traces(self) -> None:
        """Apply the buffer's retention policy between cycles.

        ``TraceStore.prune`` has existed since the buffer was written and was
        never called outside tests, so the trace corpus grew without bound and
        every cycle paid for all of it.  The retention policy itself is not
        new -- it is ``buffer.max_age_days`` and ``buffer.max_tokens``, already
        carried by the store -- only the call is.

        This runs on the tuner thread strictly *after* ``run_once`` returns, so
        there is by construction no cycle in flight whose records could be
        pulled out from under it.  The held-out suite is equally safe: it is
        persisted as its own frozen contexts under the run directory and the
        training snapshot is a copy, so neither depends on the source buffer
        surviving.

        Retention is best effort.  A pruning failure must not fail a cycle
        that has already produced a real measurement, so it is logged and
        swallowed; ``prune`` itself is already atomic and leaves the file
        untouched when it cannot rewrite it.
        """
        if self._stop_requested.is_set():
            return
        # Do not evict records while the corpus is still below the accumulation
        # threshold; pruning here would prevent accumulation from ever completing.
        try:
            record_count = self._traces.stats().count
        except Exception:
            record_count = 0
        if record_count < self._min_corpus_records:
            return
        try:
            dropped = self._traces.prune()
        except Exception:
            logger.warning("trace retention pass failed", exc_info=True)
            return
        # Logged unconditionally.  A pass that logged only when it dropped
        # something was invisible exactly when it mattered -- "retention never
        # ran" and "retention ran and had nothing to drop" produced identical
        # artifacts, so the feature could not be confirmed from a run at all.
        logger.info("trace retention pass dropped %d record(s)", dropped)

    def _record_result(self, result: CycleResult) -> None:
        with self._lock:
            self._last_result = result
            self._last_status_result = result
            self._last_error = result.error
            now = self._wall_clock()
            self._last_result_at = now
            self._last_error_at = now if result.error is not None else None
        self._persist_scheduler_status()

    def _record_error(self, error: BaseException) -> None:
        with self._lock:
            self._last_result = None
            self._last_error = str(error)
            self._last_error_at = self._wall_clock()
        self._persist_scheduler_status()

    def _record_attempt(self, watermark: _TraceWatermark) -> None:
        with self._lock:
            self._last_status_watermark = watermark
            self._last_attempt_at = self._wall_clock()
        self._persist_scheduler_status()

    def _set_lifecycle(self, lifecycle: str) -> None:
        if self._status_path is None:
            with self._lock:
                self._lifecycle = lifecycle
                self._lifecycle_changed_at = self._wall_clock()
            return
        with self._status_write_lock:
            with self._lock:
                self._lifecycle = lifecycle
                self._lifecycle_changed_at = self._wall_clock()
                payload = self._scheduler_payload_locked(self._wall_clock())
            self._write_scheduler_status(payload)

    def _persist_scheduler_status(self) -> None:
        if self._status_path is None:
            return
        with self._status_write_lock:
            with self._lock:
                payload = self._scheduler_payload_locked(self._wall_clock())
            self._write_scheduler_status(payload)

    def _scheduler_payload_locked(self, now: float) -> dict[str, object]:
        result = self._last_status_result
        # Published so an operator reading scheduler.json can tell "idle and
        # waiting for traces" from "suppressed after an unproductive cycle".
        # Without it the two look identical from outside the process.
        cooldown_until = self._cooldown_until
        cooldown_remaining = (
            max(0.0, cooldown_until - self._clock())
            if cooldown_until is not None
            else None
        )
        return {
            "schema_version": 1,
            "enabled": self._enabled,
            "lifecycle": self._lifecycle,
            "created_at": self._created_at,
            "updated_at": now,
            "lifecycle_changed_at": self._lifecycle_changed_at,
            "last_attempt_at": self._last_attempt_at,
            "last_result_at": self._last_result_at,
            "last_error_at": self._last_error_at,
            "cooldown_remaining_seconds": cooldown_remaining,
            "last_watermark": (
                self._last_status_watermark.to_dict()
                if self._last_status_watermark is not None
                else None
            ),
            "last_result": (
                {
                    "outcome": result.outcome.value,
                    "artifact_id": result.artifact_id,
                    "error": result.error,
                    "decision_path": (
                        str(result.decision_path)
                        if result.decision_path is not None
                        else None
                    ),
                    "val_loss": result.val_loss,
                }
                if result is not None
                else None
            ),
            "last_error": self._last_error,
        }

    def _write_scheduler_status(self, payload: Mapping[str, object]) -> None:
        path = self._status_path
        if path is None:
            return
        try:
            atomic_write_json(path, payload)
        except OSError as exc:
            logger.warning("could not persist idle tuner scheduler status: %s", exc)

    def _recover_serving(self, context: str) -> None:
        try:
            errors = self._orchestrator.recover()
        except Exception:
            logger.exception("idle tuner could not recover serving during %s", context)
            return
        if errors:
            logger.error(
                "idle tuner serving recovery had errors during %s: %s",
                context,
                "; ".join(errors),
            )


def _validate_backend_profile(
    profile: ModelProfile,
    backend: SpeculatorBackend,
) -> None:
    if not profile.trainable:
        raise TunerServiceConfigurationError(
            f"profile {profile.name!r} uses non-trainable method {profile.speculative_method!r}"
        )
    info = backend.describe()
    if info.verifier_model != profile.verifier_model:
        raise TunerServiceConfigurationError(
            f"backend verifier {info.verifier_model!r} does not match profile "
            f"{profile.name!r} verifier {profile.verifier_model!r}"
        )
    if profile.draft_model is not None and info.draft_model != profile.draft_model:
        raise TunerServiceConfigurationError(
            f"backend draft {info.draft_model!r} does not match profile "
            f"{profile.name!r} draft {profile.draft_model!r}"
        )
