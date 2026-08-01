"""The cycle again, but with the *production* runtime controller in the loop.

:mod:`simulation.harness` stops at the
:class:`~speedlm.tuner.orchestrator.RuntimeController` protocol and supplies its
own implementation, which is the right boundary for questions about the
*orchestrator*: what gets published, which pointer moves, what happens when
traffic arrives.

It is the wrong boundary for questions about the controller itself.  The restore
fast path, the hot-swap mutation flag, ``matches_running_draft`` and the
"do not restart what is already running" decisions the benchmark now depends on
all live inside :class:`speedlm.gateway.control.RuntimeController`, below that
seam.  So this module moves the seam down: the controller, its HTTP transport
(:class:`speedlm.gateway.vllm_http.VLLMControlClient`), its admission gate
(:class:`speedlm.gateway.control.AdmissionGate`) and its draft-swap client are
all the shipped classes, talking real HTTP to the simulated engine.  Only two
things are simulated here, and both are the process boundary itself:

* :class:`SimulatedProcessControl` -- "fork/exec a new ``vllm serve``" becomes
  :meth:`~simulation.engine.SimulatedEngine.activate`, which is already modelled
  as a restart (counters reset, sleep cleared, a journal entry appended).
* :class:`SimulatedDraftEndpoint` -- a line-for-line mirror of the private
  ``speedlm.tuner.composition._DraftEndpoint``, which cannot be reused directly
  because it is annotated against ``ThreadsafeProcessControl`` and would drag a
  real asyncio supervisor in with it.

Fidelity boundaries specific to this module
-------------------------------------------

* **No GPU memory precondition.**  ``gpu_memory`` is left unset, so
  ``RuntimeController._await_gpu_memory`` is never entered.  Nothing here can
  observe device memory, so nothing here should pretend to test the wait for it.
* **A restart is free.**  The production docstring puts a restart at ~100s and a
  wake at ~0.04s; here both are microseconds.  Restart *counts* and their
  ordering are therefore assertable, elapsed seconds are not.
* **The child cannot die on its own.**  A restart either succeeds or raises
  because the test asked it to; there is no crashed-child-that-still-holds-the-
  port failure mode.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulation.engine import SimulatedEngine
from simulation.harness import (
    MutableClock,
    SimulatedBackend,
    StageHooks,
)
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.control import (
    AbortCheck,
    AdmissionGate,
    ControlAborted,
    DraftReference,
    RuntimeController,
)
from speedlm.gateway.vllm_http import VLLMControlClient, VLLMDraftSwapClient
from speedlm.tuner.artifacts import ArtifactRegistry
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleResult,
    OrchestratorTimeouts,
    TunerOrchestrator,
)
from speedlm.tuner.state import TunerStateMachine

#: Deadlines generous enough that nothing in a GPU-free run can reach them by
#: accident, so a test that *wants* a timeout has to arrange one explicitly.
DEFAULT_TIMEOUTS = OrchestratorTimeouts(
    quiesce=10.0,
    sleep=10.0,
    candidate_start=30.0,
    benchmark=60.0,
    restore=30.0,
    wake=30.0,
)


class ProcessControlFailure(RuntimeError):
    """Raised when the simulated child refuses to come up."""


@dataclass
class SimulatedProcessControl:
    """A :class:`~speedlm.gateway.control.ChildProcessControl` over the engine.

    ``fail_next`` models a launch that does not take -- a port still held by the
    corpse of the previous child, a driver that has not released the device.
    It decrements, so a controller that retries can be made to succeed on the
    attempt after the one that failed.
    """

    engine: SimulatedEngine
    fail_next: int = 0
    restarts: list[str] = field(default_factory=list)

    def restart(self, draft: DraftReference, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ProcessControlFailure(f"simulated launch failure for {draft}")
        self.restarts.append(str(draft))
        self.engine.activate(draft)


@dataclass
class SimulatedDraftEndpoint:
    """A :class:`~speedlm.gate.runner.DraftEndpoint` that shares the controller.

    Mirrors ``speedlm.tuner.composition._DraftEndpoint``: consult the
    controller before restarting, and tell it afterwards.  Without both halves
    the gate and the controller keep independent ideas of what the child is
    running, and every decision built on ``matches_running_draft`` -- including
    restore's fast path -- reasons about the wrong process.
    """

    engine: SimulatedEngine
    process: SimulatedProcessControl
    http: VLLMControlClient
    runtime: RuntimeController
    #: Every draft the gate asked for, whether or not it cost a restart.
    requested: list[str] = field(default_factory=list)
    #: The subset that actually cost one.
    restarted: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return self.engine.url

    def activate(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        if should_abort():
            raise ControlAborted("draft activation aborted")
        self.requested.append(str(draft))
        if self.runtime.matches_running_draft(draft):
            self.http.wait_ready(
                timeout_seconds=timeout_seconds,
                should_abort=should_abort,
            )
            return
        self.process.restart(draft, timeout_seconds=timeout_seconds)
        self.restarted.append(str(draft))
        self.http.wait_ready(
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )
        self.runtime.note_external_restart(draft)


@dataclass
class ProductionSimulation:
    """A cycle assembled around the shipped :class:`RuntimeController`."""

    root: Path
    clock: MutableClock
    activity: ActivityTracker
    admission: AdmissionGate
    state: TunerStateMachine
    artifacts: ArtifactRegistry
    runtime: RuntimeController
    process: SimulatedProcessControl
    http: VLLMControlClient
    backend: SimulatedBackend
    gate: Any
    hooks: StageHooks
    engine: SimulatedEngine
    timeouts: OrchestratorTimeouts
    endpoint: SimulatedDraftEndpoint | None = None
    cycles: int = 0

    def orchestrator(self, *, run_id: str | None = None) -> TunerOrchestrator:
        self.cycles += 1
        label = run_id or f"cycle-{self.cycles:02d}"
        return TunerOrchestrator(
            state=self.state,
            idle=IdleDetector(self.activity, threshold_seconds=5.0, clock=self.clock),
            backend=self.backend,
            artifacts=self.artifacts,
            runtime=self.runtime,
            gate=self.gate,
            work_root=self.root / "runs",
            timeouts=self.timeouts,
            run_id_factory=lambda: label,
        )

    def run_cycle(
        self,
        *,
        run_id: str | None = None,
        payload: bytes | None = None,
    ) -> CycleResult:
        if payload is not None:
            self.backend.payload = payload
        self.go_idle()
        return self.orchestrator(run_id=run_id).run_once()

    def go_idle(self) -> None:
        self.clock.advance(60.0)

    @property
    def active_artifact_id(self) -> str | None:
        pointer = self.artifacts.active_pointer()
        return None if pointer is None else pointer.artifact_id

    @property
    def restarts(self) -> int:
        """Engine launches this simulation paid for.

        Counted on the process control rather than
        :attr:`~simulation.engine.EngineJournal.restarts`, which accumulates
        across every simulation that shared one engine fixture -- a test
        comparing two orders against the same engine would otherwise read the
        first order's launches into the second's total.
        """
        return len(self.process.restarts)

    @property
    def serving(self) -> str | None:
        """What the *engine* is loaded with, not what the controller believes."""
        return self.engine.loaded_reference

    def expected_active_draft(self) -> str:
        """The draft the durable state says must be serving right now."""
        active = self.artifacts.active()
        if active is not None:
            return str(active.path)
        return self.backend.describe().from_pretrained


@contextmanager
def production_simulation(
    root: Path,
    *,
    engine: SimulatedEngine,
    gate: Any = None,
    gate_factory: Callable[[ProductionSimulation], Any] | None = None,
    hooks: StageHooks | None = None,
    payload: bytes = b"candidate-weights-0",
    val_loss: float | None = 0.50,
    base_draft: str = "sim/stock-draft",
    timeouts: OrchestratorTimeouts | None = None,
    draft_hot_swap: bool = False,
    restore_fast_path_timeout_seconds: float = 120.0,
    controller_clock: Callable[[], float] = time.monotonic,
) -> Iterator[ProductionSimulation]:
    """Assemble a production-runtime cycle and close its HTTP client after.

    ``gate_factory`` exists because a gate that shares the controller (the real
    :class:`~speedlm.gate.runner.BenchmarkGateRunner` wired through
    :class:`SimulatedDraftEndpoint`) cannot be built before the controller is.
    """
    clock = MutableClock(start=100.0)
    activity = ActivityTracker(clock=clock)
    shared_hooks = hooks or StageHooks()
    (root / "runs").mkdir(parents=True, exist_ok=True)
    http = VLLMControlClient(
        engine.url,
        poll_interval_seconds=0.01,
        attempt_timeout_seconds=2.0,
        clock=controller_clock,
    )
    try:
        process = SimulatedProcessControl(engine=engine)
        admission = AdmissionGate(activity)
        runtime = RuntimeController(
            activity=activity,
            admission=admission,
            http=http,
            process=process,
            active_draft=base_draft,
            clock=controller_clock,
            poll_interval_seconds=0.01,
            recovery_timeout_seconds=30.0,
            draft_swap_http=VLLMDraftSwapClient(http) if draft_hot_swap else None,
            restore_fast_path_timeout_seconds=restore_fast_path_timeout_seconds,
        )
        simulation = ProductionSimulation(
            root=root,
            clock=clock,
            activity=activity,
            admission=admission,
            state=TunerStateMachine(root / "state", clock=lambda: clock.now),
            artifacts=ArtifactRegistry(root / "registry", clock=lambda: clock.now),
            runtime=runtime,
            process=process,
            http=http,
            backend=SimulatedBackend(
                payload=payload,
                hooks=shared_hooks,
                val_loss=val_loss,
                base_draft=base_draft,
            ),
            gate=gate,
            hooks=shared_hooks,
            engine=engine,
            timeouts=timeouts or DEFAULT_TIMEOUTS,
        )
        if gate_factory is not None:
            simulation.gate = gate_factory(simulation)
        yield simulation
    finally:
        http.close()


def shared_endpoint(
    simulation: ProductionSimulation,
) -> SimulatedDraftEndpoint:
    """The gate endpoint that shares *simulation*'s controller, recorded on it."""
    endpoint = SimulatedDraftEndpoint(
        engine=simulation.engine,
        process=simulation.process,
        http=simulation.http,
        runtime=simulation.runtime,
    )
    simulation.endpoint = endpoint
    return endpoint
