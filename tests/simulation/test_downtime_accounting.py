"""What every terminal path costs, and what none of them may leave behind.

The GPU runs measure downtime as "time the gateway was not serving normally".
The simulation can measure the same thing, but only up to a point, and the point
matters: **the simulation models arithmetic, not physics.**  A simulated engine
launch is microseconds; a real one is ~100-105s.  So nothing here asserts
seconds.

What it does assert is the two things that survive the abstraction:

1. **A cost model over observed operations.**  Downtime is dominated by two
   countable events -- engine launches and wakes -- and the simulation observes
   both exactly.  Costing them with the production docstring's own figures gives
   a number that is comparable *between paths*, never against a stopwatch.
2. **The terminal invariant, over every terminal path.**  However a cycle ends
   -- promoted, rejected, preempted, timed out, aborted, backend-failed -- the
   engine must be awake and loaded with the draft the durable pointer names.
   An engine left asleep is an outage; an engine left on the wrong draft is
   worse, because it answers.

The one path that cannot satisfy the invariant -- restore itself failing -- is
covered too, asserting that it is *reported* rather than silently accepted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    StageHooks,
    failing_gate_result,
    passing_gate_result,
    traffic_arrival,
)
from simulation.production import ProductionSimulation, production_simulation
from speedlm.tuner.orchestrator import CycleOutcome, GateFailure, GateResult
from speedlm.tuner.state import TunerState

STOCK_REFERENCE = "sim/stock-draft"
STOCK = DraftProfile(name="stock", seconds_per_request=0.0)

#: Seconds attributed to one engine launch, from the production docstring on
#: :meth:`speedlm.gateway.control.RuntimeController.restore`: ~62.5s to
#: launch-ready, ~100-105s including teardown.
RESTART_SECONDS = 100.0
#: Seconds attributed to one wake, measured WAKING->READY at 0.04s in the same
#: docstring and rounded up generously.  The ratio is the point: a wake is
#: three orders of magnitude cheaper than a launch.
WAKE_SECONDS = 0.1


def not_serving_seconds(simulation: ProductionSimulation) -> float:
    """Model the cycle's not-serving-normally time from what was observed.

    Deliberately a *model*.  Every operation it prices happens with the
    gateway's admission gate closed, so the sum is the shape of the real
    downtime; the constants are the production measurements, not this
    simulation's own clock, which would report microseconds.
    """
    wakes = simulation.engine.journal.events.count("wake_up")
    return simulation.restarts * RESTART_SECONDS + wakes * WAKE_SECONDS


@dataclass(frozen=True, slots=True)
class TerminalPath:
    """One way a cycle can end, and how to make it end that way."""

    name: str
    expected: CycleOutcome
    gate: Callable[[StageHooks], ScriptedGate]
    arm: Callable[[ProductionSimulation], None] = lambda _s: None


def _scripted(*results: GateResult) -> Callable[[StageHooks], ScriptedGate]:
    return lambda hooks: ScriptedGate(list(results), hooks=hooks)


def _preempt(simulation: ProductionSimulation) -> None:
    simulation.hooks.at(
        "extract", traffic_arrival(simulation.activity, simulation.clock)
    )


def _preempt_at_benchmark(simulation: ProductionSimulation) -> None:
    simulation.hooks.at(
        "benchmark", traffic_arrival(simulation.activity, simulation.clock)
    )


def _fail_training(simulation: ProductionSimulation) -> None:
    simulation.backend.fail_in = "train"


TERMINAL_PATHS = (
    TerminalPath("promote", CycleOutcome.PROMOTED, _scripted(passing_gate_result())),
    TerminalPath("reject", CycleOutcome.REJECTED, _scripted(failing_gate_result())),
    TerminalPath(
        "preempt-before-mutation",
        CycleOutcome.PREEMPTED,
        _scripted(passing_gate_result()),
        _preempt,
    ),
    TerminalPath(
        "preempt-after-candidate-start",
        CycleOutcome.PREEMPTED,
        _scripted(passing_gate_result()),
        _preempt_at_benchmark,
    ),
    TerminalPath(
        "gate-timeout",
        CycleOutcome.BENCHMARK_TIMED_OUT,
        _scripted(
            GateResult(
                passed=False,
                reason="benchmark timed out",
                metrics={"timed_out": True},
                failure=GateFailure.TIMED_OUT,
            )
        ),
    ),
    TerminalPath(
        "gate-abort",
        CycleOutcome.BENCHMARK_ABORTED,
        _scripted(
            GateResult(
                passed=False,
                reason="benchmark aborted",
                metrics={"aborted": True},
                failure=GateFailure.ABORTED,
            )
        ),
    ),
    TerminalPath(
        "backend-failure",
        CycleOutcome.FAILED,
        _scripted(passing_gate_result()),
        _fail_training,
    ),
)


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        running.register(STOCK_REFERENCE, STOCK)
        yield running


def state_sequence(simulation: ProductionSimulation) -> list[str]:
    """Every state the durable journal recorded, in order."""
    raw = simulation.state.events_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["to"] for line in raw if line.strip()]


@pytest.mark.parametrize("path", TERMINAL_PATHS, ids=lambda p: p.name)
def test_no_terminal_path_leaves_the_engine_asleep_or_on_the_wrong_draft(
    tmp_path: Path, engine: SimulatedEngine, path: TerminalPath
) -> None:
    hooks = StageHooks()
    with production_simulation(
        tmp_path, engine=engine, gate=path.gate(hooks), hooks=hooks
    ) as simulation:
        path.arm(simulation)

        result = simulation.run_cycle(run_id=path.name)

        assert result.outcome is path.expected, result.error
        # 1. The state machine came all the way home.
        assert simulation.state.state is TunerState.READY
        assert state_sequence(simulation)[-1] == "READY"
        # 2. The engine is awake.  Asleep is an outage: level-1 sleep offloads
        #    the weights, so the child answers nothing.
        assert engine.sleeping is False
        # 3. And it holds the draft the durable pointer names.  Serving the
        #    wrong draft is worse than downtime, because it looks like success.
        expected = simulation.expected_active_draft()
        assert (simulation.serving or STOCK_REFERENCE) == expected
        assert str(simulation.runtime.running_draft) == expected

        # 4. And the downtime window it opened is closed and finite.  A cycle
        #    that slept must have ended that sleep, whichever branch it took to
        #    get here.  Both endings count: an explicit ``/wake_up``, and a
        #    restart, which replaces the process and so cannot inherit a sleep.
        events = simulation.engine.journal.events
        lifecycle = [
            e for e in events if e in {"sleep", "wake_up"} or e.startswith("activate:")
        ]
        if lifecycle:
            assert lifecycle[-1] != "sleep", lifecycle
        assert not_serving_seconds(simulation) > 0
        assert simulation.restarts <= 3, (
            "no terminal path should need more than the candidate launch, the "
            "stock arm launch and one rollback launch"
        )


def test_the_cheapest_path_is_the_one_that_mutated_nothing(
    tmp_path: Path, engine: SimulatedEngine
) -> None:
    # The ordering property the downtime work exists to produce: a cycle
    # abandoned before it touched the engine must cost strictly less than one
    # that ran a benchmark, and it must cost no engine launches at all.
    compared = {
        "preempt-before-mutation",
        "preempt-after-candidate-start",
        "promote",
        "reject",
        "backend-failure",
    }
    costs: dict[str, float] = {}
    for path in TERMINAL_PATHS:
        if path.name not in compared:
            continue
        hooks = StageHooks()
        with production_simulation(
            tmp_path / path.name, engine=engine, gate=path.gate(hooks), hooks=hooks
        ) as simulation:
            path.arm(simulation)
            simulation.run_cycle(run_id=path.name)
            costs[path.name] = not_serving_seconds(simulation)

    assert costs["preempt-before-mutation"] < costs["promote"]
    assert costs["preempt-before-mutation"] < costs["reject"]
    assert costs["preempt-before-mutation"] < costs["preempt-after-candidate-start"]
    # It is cheap because it is a wake, not because the model says so.
    assert costs["preempt-before-mutation"] < RESTART_SECONDS
    # A failure during training also mutated nothing the engine can see.
    assert costs["backend-failure"] < RESTART_SECONDS


def test_a_restore_that_cannot_respawn_is_reported_not_hidden(
    tmp_path: Path, engine: SimulatedEngine
) -> None:
    # The one terminal path where the invariant genuinely cannot hold.  The
    # engine is loaded with the abandoned candidate, the rollback's restart
    # cannot run, and no bookkeeping can make that safe -- so the only correct
    # behaviour is to say so, loudly, in the cycle result.
    hooks = StageHooks()
    with production_simulation(
        tmp_path,
        engine=engine,
        gate=ScriptedGate([passing_gate_result()], hooks=hooks),
        hooks=hooks,
    ) as simulation:
        _preempt_at_benchmark(simulation)
        original_restart = simulation.process.restart

        def refuse_rollback(draft: object, *, timeout_seconds: float) -> None:
            if str(draft) == STOCK_REFERENCE:
                raise RuntimeError("simulated launch failure for the rollback")
            original_restart(draft, timeout_seconds=timeout_seconds)  # type: ignore[arg-type]

        simulation.process.restart = refuse_rollback  # type: ignore[method-assign]

        result = simulation.run_cycle(run_id="restore-failure")

        # The failure is reported, and it is the only thing that reports it.
        assert result.error is not None
        assert "runtime restore failed" in result.error
        # Honest about the consequence: the abandoned candidate is still loaded
        # and the durable pointer still names stock, so serving and the pointer
        # disagree.
        assert simulation.serving != simulation.expected_active_draft()
        assert simulation.artifacts.active_pointer() is None

        # Pinned deliberately, because it is the weakest link on this path and
        # a future change should have to look at it.  The outcome stays
        # PREEMPTED -- cleanup errors are folded into ``error`` rather than
        # promoting the outcome to FAILED -- and the state machine still walks
        # all the way home to READY.  ``TunerService._poll_once`` calls
        # ``_recover_serving`` only for FAILED, so on this path nothing
        # re-attempts the restore; the wrong draft keeps answering until a
        # later cycle happens to replace it.  The retry cooldown does still
        # arm, because PREEMPTED is one of ``COOLDOWN_OUTCOMES``.
        assert result.outcome is CycleOutcome.PREEMPTED
        assert simulation.state.state is TunerState.READY
