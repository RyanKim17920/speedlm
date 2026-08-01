"""When the scheduler is allowed to start a cycle, and when it must not.

Two guards, both added after job 369040 spent its GPU restarting engines to
reach conclusions it had already reached:

* ``idle_confirmations`` -- a single idle poll no longer arms a cycle.  One
  sample of a request-driven watermark is a coin flip; three consecutive ones
  are a window.
* ``retry_cooldown_seconds`` -- an unproductive outcome (preempted, failed,
  aborted, timed out) bars the next attempt for a quiet period.  A *measured*
  outcome does not, because the watermark dedupe already handles those.

The existing unit tests for both drive a stub orchestrator that returns a
chosen :class:`~speedlm.tuner.orchestrator.CycleOutcome`.  These drive the real
:class:`~speedlm.tuner.orchestrator.TunerOrchestrator` against the simulated
engine, so the outcomes are *produced* -- a cycle is preempted because a
request really arrived at extraction, and it is rejected because the gate
really declined.  That matters here specifically, because the bug the cooldown
fixes is about which outcome arms it, and a stub cannot get that wrong.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    SimulatedTraceBuffer,
    Simulation,
    StageHooks,
    build_simulation,
    failing_gate_result,
    simulation_config,
    traffic_arrival,
)
from speedlm.config import IdleTuningConfig
from speedlm.tuner.idle import ActivitySource, IdleDetector
from speedlm.tuner.orchestrator import CycleOutcome, TunerOrchestrator
from speedlm.tuner.service import TunerService

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)
IDLE_THRESHOLD = 5.0
IDLE_CONFIRMATIONS = 3
COOLDOWN_SECONDS = 600.0


@dataclass
class Scheduler:
    """A real :class:`TunerService` over a real cycle, polled one step at a time.

    ``_poll_once`` is called directly rather than starting the watcher thread:
    the whole subject is *which poll* is allowed to arm a cycle, and a thread
    polling on wall clock cannot express "the third consecutive one".
    """

    simulation: Simulation
    service: TunerService
    traces: SimulatedTraceBuffer
    hooks: StageHooks
    outcomes: list[CycleOutcome]

    def poll(self, times: int = 1) -> None:
        for _ in range(times):
            self.service._poll_once()  # noqa: SLF001

    def go_idle(self) -> None:
        """Move past the idle threshold so the next poll can see an idle gateway."""
        self.simulation.clock.advance(IDLE_THRESHOLD * 2)

    def wait(self, seconds: float) -> None:
        self.simulation.clock.advance(seconds)

    @property
    def cycles_run(self) -> int:
        return len(self.outcomes)


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        yield running


@pytest.fixture()
def scheduler(tmp_path: Path, engine: SimulatedEngine) -> Scheduler:
    hooks = StageHooks()
    simulation = build_simulation(
        tmp_path,
        engine=engine,
        # Enough verdicts that no test runs the gate dry; each is a measured
        # rejection, which is deliberately *not* a cooldown outcome.
        gate=ScriptedGate([failing_gate_result() for _ in range(8)], hooks=hooks),
        hooks=hooks,
    )
    traces = SimulatedTraceBuffer()
    outcomes: list[CycleOutcome] = []
    runs = itertools.count(1)

    class _RecordingRunner:
        """The real orchestrator, with each cycle's outcome kept for assertions.

        A fresh orchestrator per cycle, because the run directory is minted
        from ``run_id_factory`` with ``exist_ok=False`` -- one orchestrator
        reused across polls would collide with itself on the second cycle and
        turn a scheduling question into a filesystem one.
        """

        def run_once(self) -> object:
            orchestrator = TunerOrchestrator(
                state=simulation.state,
                idle=IdleDetector(
                    simulation.activity,
                    threshold_seconds=IDLE_THRESHOLD,
                    clock=simulation.clock,
                ),
                backend=simulation.backend,
                artifacts=simulation.artifacts,
                runtime=simulation.runtime,
                gate=simulation.gate,
                work_root=tmp_path / "runs",
                timeouts=simulation.timeouts,
                run_id_factory=lambda: f"cycle-{next(runs):02d}",
            )
            result = orchestrator.run_once()
            outcomes.append(result.outcome)
            return result

        def recover(self) -> tuple[str, ...]:
            return ()

    def factory(activity: ActivitySource) -> _RecordingRunner:
        del activity
        return _RecordingRunner()

    service = TunerService(
        simulation_config(
            idle_threshold_seconds=IDLE_THRESHOLD,
            tuning=IdleTuningConfig(
                idle_confirmations=IDLE_CONFIRMATIONS,
                retry_cooldown_seconds=COOLDOWN_SECONDS,
            ),
        ),
        activity=simulation.activity,
        traces=traces,
        orchestrator_factory=factory,  # type: ignore[arg-type]
        enabled=True,
        min_trace_records=2,
        min_corpus_records=2,
        poll_interval_seconds=0.01,
        clock=simulation.clock,
    )
    return Scheduler(
        simulation=simulation,
        service=service,
        traces=traces,
        hooks=hooks,
        outcomes=outcomes,
    )


class TestIdleConfirmation:
    def test_one_idle_poll_is_not_enough_to_arm_a_cycle(
        self, scheduler: Scheduler
    ) -> None:
        scheduler.go_idle()

        scheduler.poll()
        assert scheduler.cycles_run == 0
        scheduler.poll()
        assert scheduler.cycles_run == 0
        scheduler.poll()
        assert scheduler.outcomes == [CycleOutcome.REJECTED]

    def test_one_busy_sample_resets_the_streak(self, scheduler: Scheduler) -> None:
        scheduler.go_idle()
        scheduler.poll(2)

        # A request is in flight for exactly one poll.  The confirmations have
        # to be *consecutive* or they measure nothing a single sample did not.
        scheduler.simulation.activity.begin()
        scheduler.poll()
        scheduler.simulation.activity.end()
        assert scheduler.cycles_run == 0

        scheduler.go_idle()
        scheduler.poll(2)
        assert scheduler.cycles_run == 0, "the busy sample did not reset the streak"
        scheduler.poll()
        assert scheduler.outcomes == [CycleOutcome.REJECTED]


class TestRetryCooldown:
    def test_a_preempted_cycle_suppresses_the_retry_the_watermark_would_allow(
        self, scheduler: Scheduler
    ) -> None:
        # Exactly job 369040.  The request that preempts the cycle is itself
        # traced, so it *advances* the watermark -- the dedupe sees a brand-new
        # opportunity and re-pays the engine restarts to reach the same
        # conclusion.  Only a cooldown can see this case.
        scheduler.hooks.at(
            "extract",
            traffic_arrival(scheduler.simulation.activity, scheduler.simulation.clock),
        )
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS)
        assert scheduler.outcomes == [CycleOutcome.PREEMPTED]

        # The preempting request lands in the trace buffer, defeating the dedupe.
        scheduler.traces.record_request()
        scheduler.hooks.hooks.pop("extract")
        scheduler.wait(9.2)
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS * 2)

        assert scheduler.outcomes == [CycleOutcome.PREEMPTED], (
            "a fresh watermark must not defeat the cooldown"
        )

    def test_the_cooldown_expires_and_the_next_cycle_runs(
        self, scheduler: Scheduler
    ) -> None:
        scheduler.hooks.at(
            "extract",
            traffic_arrival(scheduler.simulation.activity, scheduler.simulation.clock),
        )
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS)
        assert scheduler.outcomes == [CycleOutcome.PREEMPTED]
        scheduler.hooks.hooks.pop("extract")
        scheduler.traces.record_request()

        # Still inside the quiet period.
        scheduler.wait(COOLDOWN_SECONDS - 60.0)
        scheduler.poll(IDLE_CONFIRMATIONS)
        assert scheduler.cycles_run == 1

        # And past it.  The streak restarts from zero, because a poll spent in
        # cooldown is not an idle confirmation.
        scheduler.wait(120.0)
        scheduler.poll(IDLE_CONFIRMATIONS - 1)
        assert scheduler.cycles_run == 1
        scheduler.poll()
        assert scheduler.outcomes == [CycleOutcome.PREEMPTED, CycleOutcome.REJECTED]

    def test_a_measured_rejection_does_not_arm_the_cooldown(
        self, scheduler: Scheduler
    ) -> None:
        # A rejection spent its restarts and got a real answer for them.  The
        # watermark dedupe already stops it repeating on unchanged traces, so
        # suppressing it as well would just delay real work.
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS)
        assert scheduler.outcomes == [CycleOutcome.REJECTED]

        scheduler.traces.record_request()
        scheduler.wait(1.0)
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS)

        assert scheduler.outcomes == [CycleOutcome.REJECTED, CycleOutcome.REJECTED]

    def test_the_cycle_that_armed_the_cooldown_still_restored_serving(
        self, scheduler: Scheduler
    ) -> None:
        # The cooldown is a scheduling decision, not a recovery one.  Whatever
        # it suppresses next, the preempted cycle must already have put the
        # engine back on the active draft and woken it.
        scheduler.hooks.at(
            "extract",
            traffic_arrival(scheduler.simulation.activity, scheduler.simulation.clock),
        )
        scheduler.go_idle()
        scheduler.poll(IDLE_CONFIRMATIONS)

        assert scheduler.outcomes == [CycleOutcome.PREEMPTED]
        assert scheduler.simulation.runtime.serving == "sim/stock-draft"
        assert scheduler.simulation.engine.sleeping is False
