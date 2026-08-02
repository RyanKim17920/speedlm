"""Assemble a whole tuning cycle out of production parts and a simulated engine.

Only the three protocols the GPU sits behind are simulated --
:class:`~speedlm.training.base.SpeculatorBackend`,
:class:`~speedlm.tuner.orchestrator.RuntimeController` and (optionally)
:class:`~speedlm.tuner.orchestrator.BenchmarkGate`.  The state machine, the
artifact registry, the gate runner, the metrics parser and the promotion
decision are all the real thing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulation.engine import DraftProfile, SimulatedEngine
from speedlm.config import (
    IdleTuningConfig,
    PromotionConfig,
    SamplingConfig,
    SpeedLMConfig,
    ValLossPreFilterConfig,
)
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gateway.activity import ActivityTracker
from speedlm.traces.store import TraceRecord, TraceStats
from speedlm.training.base import BackendInfo
from speedlm.tuner.artifacts import ArtifactRegistry
from speedlm.tuner.composition import warm_start_reference
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleResult,
    GateFailure,
    GateResult,
    OrchestratorTimeouts,
    TunerOrchestrator,
)
from speedlm.tuner.state import TunerStateMachine

#: The cycle stages a test can hook.  These are the boundaries the orchestrator
#: itself sequences, named so a preemption test reads as the story it tells.
STAGES = (
    "quiesce",
    "sleep",
    "extract",
    "train",
    "candidate_start",
    "benchmark",
    "restore",
    "wake",
)


class _Unset:
    """Sentinel distinguishing "not supplied" from an explicit ``None``."""


UNSET = _Unset()


class MutableClock:
    """A clock the test drives, so nothing waits on real time.

    ``advance_per_call`` exists for the one place a *passive* clock cannot
    express the scenario: the gate's deadline is only ever consulted at stage
    boundaries, so a benchmark can only be made to time out by having the
    clock move while the gate is running.
    """

    def __init__(self, start: float = 0.0, *, advance_per_call: float = 0.0) -> None:
        self._now = float(start)
        self._advance_per_call = float(advance_per_call)
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        value = self._now
        self._now += self._advance_per_call
        return value

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


@dataclass
class StageHooks:
    """Callbacks fired as the cycle crosses each stage boundary.

    A hook may raise, which is how "the engine died during extraction" is
    expressed without teaching the backend about every failure mode.
    """

    hooks: dict[str, Callable[[], None]] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)

    def fire(self, stage: str) -> None:
        self.seen.append(stage)
        hook = self.hooks.get(stage)
        if hook is not None:
            hook()

    def at(self, stage: str, action: Callable[[], None]) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        self.hooks[stage] = action


def traffic_arrival(activity: ActivityTracker, clock: MutableClock) -> Callable[[], None]:
    """A hook that simulates one real request landing on the gateway.

    Advancing the clock first is what makes the arrival *newer* than the
    watermark :meth:`speedlm.tuner.idle.IdleDetector.arm` captured, which is
    exactly how :class:`~speedlm.tuner.idle.PreemptionGuard` notices it.
    """

    def arrive() -> None:
        clock.advance(1.0)
        activity.begin()
        activity.end()

    return arrive


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Snapshot:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    snapshot: _Snapshot


@dataclass(frozen=True, slots=True)
class _Trained:
    val_loss: float | None


@dataclass
class SimulatedBackend:
    """A :class:`~speedlm.training.base.SpeculatorBackend` that can fail.

    Each cycle produces a distinct draft directory, so the content-addressed
    registry mints a distinct artifact ID -- which is what makes "the active
    pointer moved" and "the active pointer did not move" different observable
    facts across a multi-cycle run.
    """

    payload: bytes
    hooks: StageHooks = field(default_factory=StageHooks)
    val_loss: float | None = 0.50
    trace_hash: str = "sim-trace-hash"
    verifier_model: str = "sim/verifier-8b"
    draft_model: str = "sim/draft-eagle3"
    base_draft: str = "sim/stock-draft"
    #: Resolves the checkpoint each cycle warm-starts from, exactly as
    #: ``create_production_tuner`` wires it.  ``None`` is the historical
    #: behaviour: every cycle trains from ``base_draft``.
    warm_start_resolver: Callable[[], str] | None = None
    #: One entry per :meth:`train`, in cycle order -- the assertion surface for
    #: "did learning actually accumulate across cycles".
    trained_from: list[str] = field(default_factory=list)
    fail_in: str | None = None
    failure: Exception | None = None
    prepared_calls: int = 0
    #: ``(stage, should_abort())`` at every point the backend consulted the
    #: abort check.  Proves the orchestrator threads a live check all the way
    #: down rather than passing a constant ``False``.
    aborts_seen: list[tuple[str, bool]] = field(default_factory=list)

    def describe(self) -> BackendInfo:
        # The *resolved* base once a cycle has trained, so the manifest's
        # ``base_draft`` names the head this artifact was actually built on.
        return BackendInfo(
            verifier_model=self.verifier_model,
            draft_model=self.draft_model,
            from_pretrained=(
                self.trained_from[-1] if self.trained_from else self.base_draft
            ),
            training_params={"steps": 4, "lr": 1e-4},
        )

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_in == stage:
            raise self.failure or RuntimeError(f"simulated backend failure in {stage}")

    def prepare(self, work_dir: Path, *, should_abort: Callable[[], bool]) -> _Prepared:
        del work_dir
        self.prepared_calls += 1
        self.hooks.fire("extract")
        self._maybe_fail("extract")
        self.aborts_seen.append(("prepare", should_abort()))
        return _Prepared(_Snapshot(self.trace_hash))

    def extract(
        self,
        prepared: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> object:
        del prepared, work_dir
        self.aborts_seen.append(("extract", should_abort()))
        return object()

    def train(
        self,
        extracted: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> _Trained:
        del extracted, work_dir
        self.hooks.fire("train")
        self._maybe_fail("train")
        self.trained_from.append(
            self.base_draft
            if self.warm_start_resolver is None
            else self.warm_start_resolver()
        )
        self.aborts_seen.append(("train", should_abort()))
        return _Trained(val_loss=self.val_loss)

    def materialize(
        self,
        trained: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del trained
        self._maybe_fail("materialize")
        self.aborts_seen.append(("materialize", should_abort()))
        draft = work_dir / "draft"
        draft.mkdir()
        (draft / "model.safetensors").write_bytes(self.payload)
        (draft / "config.json").write_text(
            '{"architectures": ["Eagle3Speculator"]}', encoding="utf-8"
        )
        return draft

    def validate(self, artifact: Any, *, should_abort: Callable[[], bool]) -> None:
        self._maybe_fail("validate")
        assert Path(artifact).is_dir()
        self.aborts_seen.append(("validate", should_abort()))


# ---------------------------------------------------------------------------
# Runtime controller
# ---------------------------------------------------------------------------

@dataclass
class SimulatedRuntime:
    """A :class:`~speedlm.tuner.orchestrator.RuntimeController` over the engine.

    ``serving`` is the assertion surface every preemption test lands on: after
    a cycle ends -- promoted, rejected, preempted or failed -- serving must be
    pointed at the correct draft, and this records what it was actually
    pointed at.
    """

    engine: SimulatedEngine
    hooks: StageHooks = field(default_factory=StageHooks)
    base_draft: str = "sim/stock-draft"
    serving: str | Path | None = None
    calls: list[str] = field(default_factory=list)
    fail_in: str | None = None
    #: ``(effect, should_abort())`` at every effect boundary.
    aborts_seen: list[tuple[str, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.serving is None:
            self.serving = self.base_draft

    def _effect(self, name: str) -> None:
        self.calls.append(name)
        self.hooks.fire(name)
        if self.fail_in == name:
            raise RuntimeError(f"simulated runtime failure in {name}")

    def quiesce(self, *, timeout_seconds: float, should_abort: Callable[[], bool]) -> None:
        assert timeout_seconds > 0
        self._effect("quiesce")
        self.aborts_seen.append(("quiesce", should_abort()))

    def sleep(self, *, timeout_seconds: float, should_abort: Callable[[], bool]) -> None:
        assert timeout_seconds > 0
        self._effect("sleep")
        self.engine.activate_sleep()
        self.aborts_seen.append(("sleep", should_abort()))

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0
        self._effect("candidate_start")
        self.engine.activate(draft_directory)
        self.serving = draft_directory
        self.aborts_seen.append(("candidate_start", should_abort()))

    def restore(self, active_draft: Path | str, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self._effect("restore")
        self.engine.activate(active_draft)
        self.serving = active_draft

    def wake(self, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self._effect("wake")
        self.engine.wake()


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

@dataclass
class ScriptedGate:
    """A gate that returns preset verdicts, one per cycle.

    Used where the *cycle* is under test rather than the measurement: multi-
    cycle pointer bookkeeping, preemption boundaries, rollback.  Scenarios
    about the measurement itself use the real
    :class:`~speedlm.gate.runner.BenchmarkGateRunner` -- see
    :func:`real_gate`.
    """

    verdicts: list[GateResult]
    hooks: StageHooks = field(default_factory=StageHooks)
    calls: int = 0
    seen_drafts: list[Path] = field(default_factory=list)

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        assert timeout_seconds > 0
        assert candidate_draft.is_dir()
        self.seen_drafts.append(candidate_draft)
        self.calls += 1
        self.hooks.fire("benchmark")
        if should_abort():
            return GateResult(
                passed=False,
                reason="benchmark aborted during replay",
                metrics={"aborted": True},
                failure=GateFailure.ABORTED,
            )
        if not self.verdicts:
            raise AssertionError("ScriptedGate ran out of scripted verdicts")
        return self.verdicts.pop(0)


def passing_gate_result(reason: str = "both_thresholds_met") -> GateResult:
    return GateResult(passed=True, reason=reason)


def failing_gate_result(reason: str = "acceptance_below_threshold") -> GateResult:
    return GateResult(passed=False, reason=reason)


@dataclass
class EngineEndpoint:
    """A :class:`~speedlm.gate.runner.DraftEndpoint` over the simulated engine."""

    engine: SimulatedEngine
    hooks: StageHooks = field(default_factory=StageHooks)
    activations: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return self.engine.url

    def activate(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0
        self.activations.append(str(draft))
        self.engine.activate(draft)
        if self.engine.faults.never_ready:
            raise RuntimeError(f"engine never became ready for {draft}")


@dataclass
class EngineMetrics:
    """A :class:`~speedlm.gate.runner.MetricsSource` over the simulated engine."""

    engine: SimulatedEngine

    def scrape(self, *, timeout_seconds: float, should_abort: Callable[[], bool]) -> str:
        assert timeout_seconds > 0
        del should_abort
        return self.engine.scrape()


@dataclass
class RecordSource:
    """A :class:`~speedlm.gate.runner.TraceSource` over frozen records."""

    records: tuple[TraceRecord, ...]

    def iter_records(self) -> Iterator[TraceRecord]:
        yield from self.records


def simulation_config(
    *,
    model: str = "sim/verifier-8b",
    promotion: PromotionConfig | None = None,
    idle_threshold_seconds: float = 300.0,
    tuning: IdleTuningConfig | None = None,
) -> SpeedLMConfig:
    """A production :class:`~speedlm.config.SpeedLMConfig` for the simulation."""
    kwargs: dict[str, Any] = {}
    if tuning is not None:
        kwargs["tuning"] = tuning
    return SpeedLMConfig(
        model=model,
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, seed=0),
        promotion=promotion or PromotionConfig(),
        idle_threshold_seconds=idle_threshold_seconds,
        **kwargs,
    )


@dataclass
class SimulatedTraceBuffer:
    """A :class:`~speedlm.tuner.service.TraceStatsSource` the test advances.

    The scheduler dedupes on the *watermark* -- the buffer's whole
    :class:`~speedlm.traces.store.TraceStats` tuple -- so "a request arrived
    and was traced" has to be expressible as a change to it.  That is the whole
    subject of the retry-cooldown tests: the request that preempts a cycle is
    itself traced, so it advances this watermark and the next poll looks like a
    brand-new opportunity even though nothing has changed.
    """

    count: int = 512
    tokens: int = 65536
    prunes: int = 0

    def record_request(self, *, tokens: int = 128) -> None:
        """Model one more served request landing in the trace buffer."""
        self.count += 1
        self.tokens += tokens

    def stats(self) -> TraceStats:
        return TraceStats(
            count=self.count,
            tokens=self.tokens,
            oldest=0.0,
            newest=float(self.count),
        )

    def prune(self) -> int:
        self.prunes += 1
        return 0


def real_gate(
    engine: SimulatedEngine,
    records: Sequence[TraceRecord],
    suite_dir: Path,
    *,
    config: SpeedLMConfig | None = None,
    stock_draft: str = "sim/stock-draft",
    repeats: int = 3,
    warmup_repeats: int = 1,
    held_out_fraction: float = 1.0,
    clock: Callable[[], float] | None = None,
    endpoint: Any | None = None,
    candidate_arm_first: bool = False,
) -> BenchmarkGateRunner:
    """The *production* gate runner, wired to the simulated engine over HTTP.

    ``held_out_fraction=1.0`` puts every supplied record in the suite, and the
    leakage check is satisfied with an empty training-hash set: the simulation
    supplies benchmark records and training records separately, so there is
    genuinely no overlap to find.  A leakage *positive* is asserted elsewhere by
    handing the runner a hash set that does overlap.

    ``endpoint`` overrides the default unconditional-activation endpoint.  A
    test about *arm order* has to supply
    :class:`simulation.production.SimulatedDraftEndpoint` instead, because the
    saving the order exists to produce -- reusing the engine
    ``CANDIDATE_STARTING`` already built -- is only expressible by an endpoint
    that can decline to restart.
    """
    return BenchmarkGateRunner(
        config=config or simulation_config(),
        trace_source=RecordSource(tuple(records)),
        suite_dir=suite_dir,
        stock_draft=stock_draft,
        endpoint=endpoint if endpoint is not None else EngineEndpoint(engine),
        metrics_source=EngineMetrics(engine),
        repeats=repeats,
        warmup_repeats=warmup_repeats,
        held_out_fraction=held_out_fraction,
        training_context_hashes=frozenset(),
        candidate_arm_first=candidate_arm_first,
        clock=clock or (lambda: 0.0),
    )


# ---------------------------------------------------------------------------
# Cycle assembly
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    """One assembled, runnable tuning cycle plus everything to assert on."""

    root: Path
    clock: MutableClock
    activity: ActivityTracker
    state: TunerStateMachine
    artifacts: ArtifactRegistry
    runtime: SimulatedRuntime
    backend: SimulatedBackend
    gate: Any
    hooks: StageHooks
    engine: SimulatedEngine
    timeouts: OrchestratorTimeouts
    val_loss_prefilter: ValLossPreFilterConfig | None
    cycles: int = 0

    def orchestrator(self, *, run_id: str | None = None) -> TunerOrchestrator:
        self.cycles += 1
        label = run_id or f"cycle-{self.cycles:02d}"
        return TunerOrchestrator(
            state=self.state,
            idle=IdleDetector(
                self.activity,
                threshold_seconds=5.0,
                clock=self.clock,
            ),
            backend=self.backend,
            artifacts=self.artifacts,
            runtime=self.runtime,
            gate=self.gate,
            work_root=self.root / "runs",
            timeouts=self.timeouts,
            run_id_factory=lambda: label,
            val_loss_prefilter=self.val_loss_prefilter,
        )

    def run_cycle(
        self,
        *,
        run_id: str | None = None,
        payload: bytes | None = None,
        val_loss: float | None | _Unset = UNSET,
    ) -> CycleResult:
        """Run one cycle, optionally re-rolling what the backend will produce."""
        if payload is not None:
            self.backend.payload = payload
        if not isinstance(val_loss, _Unset):
            self.backend.val_loss = val_loss
        self.go_idle()
        return self.orchestrator(run_id=run_id).run_once()

    def go_idle(self) -> None:
        """Move the clock past the idle threshold so a cycle is eligible."""
        self.clock.advance(60.0)

    @property
    def active_artifact_id(self) -> str | None:
        pointer = self.artifacts.active_pointer()
        return None if pointer is None else pointer.artifact_id


def build_simulation(
    root: Path,
    *,
    engine: SimulatedEngine,
    gate: Any,
    payload: bytes = b"candidate-weights-0",
    val_loss: float | None = 0.50,
    hooks: StageHooks | None = None,
    timeouts: OrchestratorTimeouts | None = None,
    val_loss_prefilter: ValLossPreFilterConfig | None = None,
    base_draft: str = "sim/stock-draft",
    compounding_warm_start: bool = False,
    warm_start_max_chain_depth: int | None = None,
) -> Simulation:
    """Wire production orchestration around the simulated engine.

    ``compounding_warm_start`` defaults to *off* here, unlike production: the
    existing simulations assert cycle sequencing and pointer movement, and none
    of them should change behaviour because a warm start started following the
    registry.  The compounding simulation opts in explicitly.
    """
    shared_hooks = hooks or StageHooks()
    clock = MutableClock(start=100.0)
    activity = ActivityTracker(clock=clock)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactRegistry(root / "registry", clock=lambda: clock.now)
    return Simulation(
        root=root,
        clock=clock,
        activity=activity,
        state=TunerStateMachine(root / "state", clock=lambda: clock.now),
        artifacts=artifacts,
        runtime=SimulatedRuntime(
            engine=engine,
            hooks=shared_hooks,
            base_draft=base_draft,
        ),
        backend=SimulatedBackend(
            payload=payload,
            hooks=shared_hooks,
            val_loss=val_loss,
            base_draft=base_draft,
            warm_start_resolver=(
                (
                    lambda: warm_start_reference(
                        artifacts,
                        base_draft,
                        max_chain_depth=warm_start_max_chain_depth,
                    )
                )
                if compounding_warm_start
                else None
            ),
        ),
        gate=gate,
        hooks=shared_hooks,
        engine=engine,
        timeouts=timeouts
        or OrchestratorTimeouts(
            quiesce=5.0,
            sleep=5.0,
            candidate_start=5.0,
            benchmark=60.0,
            restore=5.0,
            wake=5.0,
        ),
        val_loss_prefilter=val_loss_prefilter,
    )


def profile_pair(
    *,
    stock_acceptance: float,
    candidate_acceptance: float,
    stock_seconds: float = 0.006,
    candidate_seconds: float = 0.003,
    candidate_divergence_at: int | None = None,
    candidate_invalid_every: int | None = None,
) -> tuple[DraftProfile, DraftProfile]:
    """A stock/candidate behaviour pair for the gate to measure and compare."""
    return (
        DraftProfile(
            name="stock",
            acceptance_rate=stock_acceptance,
            seconds_per_request=stock_seconds,
        ),
        DraftProfile(
            name="candidate",
            acceptance_rate=candidate_acceptance,
            seconds_per_request=candidate_seconds,
            divergence_at_token=candidate_divergence_at,
            invalid_every=candidate_invalid_every,
        ),
    )


def register_pair(
    engine: SimulatedEngine,
    stock_reference: str,
    candidate_reference: str | Path,
    profiles: Mapping[str, DraftProfile] | tuple[DraftProfile, DraftProfile],
) -> None:
    """Bind a stock/candidate profile pair to the references the gate will use."""
    if isinstance(profiles, tuple):
        stock, candidate = profiles
    else:
        stock, candidate = profiles["stock"], profiles["candidate"]
    engine.register(stock_reference, stock)
    engine.register(str(candidate_reference), candidate)
