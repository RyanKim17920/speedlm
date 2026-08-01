"""``restore`` may wake instead of respawning -- but only when it can prove it.

A restart is ~100s of pure downtime, and job 369040 paid it for a cycle that
was preempted 0.1s into extraction having mutated nothing.  So
:meth:`speedlm.gateway.control.RuntimeController.restore` now tries a wake
first.  The trade is deliberately asymmetric: skipping a restart that *was*
needed leaves the wrong draft answering live traffic, which is the worst thing
this system can do, and is strictly worse than 100s of downtime.

Every test here therefore comes in the same shape -- one case where the fast
path must be taken, and one case per uncertainty where it must *not* be, each
asserting on what the **engine** is loaded with rather than on what the
controller believes.  The two disagreeing is the bug.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    StageHooks,
    passing_gate_result,
    traffic_arrival,
)
from simulation.production import (
    ProductionSimulation,
    production_simulation,
)
from speedlm.gateway.control import DraftSwapCorrupted, ServiceRecoveryError
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

STOCK_REFERENCE = "sim/stock-draft"
STOCK = DraftProfile(name="stock", seconds_per_request=0.0)
#: Small enough that a fast path which cannot confirm serving gives up inside a
#: test, large enough that a healthy one never trips it.
SHORT_FAST_PATH = 0.15


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        running.register(STOCK_REFERENCE, STOCK)
        yield running


@contextmanager
def preempted_at_extraction(
    root: Path,
    engine: SimulatedEngine,
    **kwargs: object,
) -> Iterator[ProductionSimulation]:
    """A cycle that traffic abandons during extraction, mutating nothing."""
    hooks = StageHooks()
    with production_simulation(
        root,
        engine=engine,
        gate=ScriptedGate([passing_gate_result()], hooks=hooks),
        hooks=hooks,
        **kwargs,  # type: ignore[arg-type]
    ) as simulation:
        hooks.at("extract", traffic_arrival(simulation.activity, simulation.clock))
        yield simulation


class TestTheFastPathIsTakenWhenItIsSafe:
    def test_a_cycle_that_mutated_nothing_is_restored_by_a_wake(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        with preempted_at_extraction(tmp_path, engine) as simulation:
            result = simulation.run_cycle(run_id="preempted")

            assert result.outcome is CycleOutcome.PREEMPTED
            assert simulation.state.state is TunerState.READY
            # The whole point: no engine was launched.  Not by the cycle, which
            # never reached CANDIDATE_STARTING, and not by the rollback, which
            # is what used to spend 104.5s here.
            assert simulation.restarts == 0
            assert simulation.process.restarts == []
            # And serving is genuinely back: awake, on the active draft.
            assert engine.sleeping is False
            assert str(simulation.runtime.running_draft) == STOCK_REFERENCE
            assert simulation.runtime.matches_running_draft(STOCK_REFERENCE)

    def test_the_fast_path_really_slept_and_really_woke_the_engine(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # A "fast path" that skipped the wake as well as the restart would pass
        # the test above while leaving an offloaded engine answering nothing.
        with preempted_at_extraction(tmp_path, engine) as simulation:
            simulation.run_cycle(run_id="preempted")

            events = engine.journal.events
            assert "sleep" in events, "the cycle must have actually slept vLLM"
            assert events.index("sleep") < events.index("wake_up")
            assert engine.sleeping is False


class TestTheFastPathIsRefusedOnEveryUncertainty:
    def test_a_different_draft_running_forces_the_restart(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # Preempt *after* CANDIDATE_STARTING, so the engine really is loaded
        # with the candidate when the rollback asks for the active draft.
        hooks = StageHooks()
        with production_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([passing_gate_result()], hooks=hooks),
            hooks=hooks,
        ) as simulation:
            hooks.at(
                "benchmark", traffic_arrival(simulation.activity, simulation.clock)
            )
            result = simulation.run_cycle(run_id="preempted")

            assert result.outcome is CycleOutcome.PREEMPTED
            assert result.artifact_id is not None
            # One launch for the candidate, one to put stock back.  The second
            # is the one that must not be skipped.
            assert simulation.process.restarts[0].endswith(result.artifact_id)
            assert simulation.process.restarts[-1] == STOCK_REFERENCE
            # The engine's own answer, not the controller's bookkeeping: an
            # abandoned candidate must not be left answering live traffic.
            assert simulation.serving == STOCK_REFERENCE
            assert engine.sleeping is False
            assert simulation.artifacts.active_pointer() is None
            assert simulation.state.state is TunerState.READY

    def test_a_failed_canary_forces_the_restart(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # One completion succeeds -- enough for ``wait_ready`` -- and the next
        # fails, which is exactly the canary.  An engine that is *up* but can
        # no longer emit a token is the case readiness alone cannot see.
        engine.set_completion_budget(1)
        with preempted_at_extraction(
            tmp_path, engine, restore_fast_path_timeout_seconds=SHORT_FAST_PATH
        ) as simulation:
            result = simulation.run_cycle(run_id="preempted")

            assert result.outcome is CycleOutcome.PREEMPTED
            assert simulation.process.restarts == [STOCK_REFERENCE]
            assert simulation.serving == STOCK_REFERENCE
            assert engine.sleeping is False
            assert simulation.state.state is TunerState.READY

    def test_an_engine_that_cannot_become_ready_forces_the_restart(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # Wedged until something replaces the process.  The fast path must give
        # up inside its *own* budget and leave enough of the restore deadline
        # for the restart -- which does fix it -- to still run.
        #
        # Driven directly, because a whole cycle would mask the question: the
        # wake that follows a rollback fails against a wedged engine too, and
        # its own recovery restart would repair the deployment no matter what
        # restore did.  Only ``restore`` is under test here.
        with production_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([]),
            restore_fast_path_timeout_seconds=SHORT_FAST_PATH,
        ) as simulation:
            engine.activate(STOCK_REFERENCE)
            simulation.runtime.note_external_restart(STOCK_REFERENCE)
            before = simulation.restarts
            engine.wedge()

            # Four times the fast path's budget.  If the fast path were allowed
            # to spend the whole restore deadline instead of its own, there
            # would be nothing left and this would raise ControlTimeout.
            simulation.runtime.restore(
                STOCK_REFERENCE, timeout_seconds=SHORT_FAST_PATH * 4
            )

            assert simulation.restarts == before + 1
            assert simulation.process.restarts[-1] == STOCK_REFERENCE
            assert simulation.serving == STOCK_REFERENCE
            assert engine.sleeping is False
            # The restart cleared the wedge, so the engine really can serve now.
            simulation.http.canary(timeout_seconds=5.0)

    def test_a_stale_sleep_belief_forces_the_restart(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # Driven directly rather than through a cycle: the controller's sleep
        # bookkeeping is only ever wrong when something outside it slept the
        # engine, which by construction no cycle stage can arrange.
        with production_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([]),
            restore_fast_path_timeout_seconds=SHORT_FAST_PATH,
        ) as simulation:
            engine.activate(STOCK_REFERENCE)
            simulation.runtime.note_external_restart(STOCK_REFERENCE)
            before = simulation.restarts
            # vLLM slept behind the controller's back.  Bookkeeping now says
            # awake; the engine says otherwise, and only the engine is right.
            engine.activate_sleep()

            simulation.runtime.restore(STOCK_REFERENCE, timeout_seconds=10.0)

            assert simulation.restarts == before + 1
            assert engine.sleeping is False
            assert simulation.serving == STOCK_REFERENCE

    def test_a_possibly_mutated_engine_forces_the_restart(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # The only route by which ``_engine_may_be_mutated`` survives to a
        # restore is a hot-swap that may have half-applied *followed by* a
        # restart that could not run -- every successful restart clears the
        # flag, because a fresh process cannot carry a previous one's mutation.
        candidate = tmp_path / "candidate-draft"
        candidate.mkdir()
        with production_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([]),
            draft_hot_swap=True,
        ) as simulation:
            engine.activate(STOCK_REFERENCE)
            simulation.runtime.note_external_restart(STOCK_REFERENCE)
            # The simulated ``/collective_rpc`` never confirms a swap, so the
            # shipped VLLMDraftSwapClient classifies it as corrupted -- the
            # pessimistic reading, which is the only safe one.
            simulation.process.fail_next = 2
            with pytest.raises(ServiceRecoveryError):
                simulation.runtime.start_candidate(
                    candidate, timeout_seconds=10.0, should_abort=lambda: False
                )
            assert simulation.process.fail_next == 0
            before = simulation.restarts

            simulation.runtime.restore(STOCK_REFERENCE, timeout_seconds=10.0)

            # Bookkeeping still names the stock draft and the engine is awake
            # and healthy, so every cheap signal says "no restart needed".  The
            # mutation flag is the only thing standing between that and a
            # possibly half-swapped drafter serving live traffic.
            assert simulation.restarts == before + 1
            assert simulation.serving == STOCK_REFERENCE
            assert simulation.runtime.matches_running_draft(STOCK_REFERENCE)

    def test_the_swap_that_sets_the_flag_is_classified_as_corrupted(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # Pins the premise of the test above: an unconfirmed swap RPC is a
        # *corrupted* outcome, not an unavailable one.  If it were classified
        # as unavailable the engine would be declared untouched and the flag
        # would never be set at all.
        with production_simulation(
            tmp_path, engine=engine, gate=ScriptedGate([]), draft_hot_swap=True
        ) as simulation:
            from speedlm.gateway.vllm_http import VLLMDraftSwapClient  # noqa: PLC0415

            swap = VLLMDraftSwapClient(simulation.http)
            with pytest.raises(DraftSwapCorrupted):
                swap.hot_swap_draft(str(tmp_path), timeout_seconds=5.0)


def test_a_faulted_engine_still_ends_on_the_promoted_draft(
    tmp_path: Path, engine: SimulatedEngine
) -> None:
    # The composite the fast path is most dangerous in: an incumbent exists, so
    # "restore" means a promoted artifact directory rather than the base draft.
    hooks = StageHooks()
    with production_simulation(
        tmp_path,
        engine=engine,
        gate=ScriptedGate([passing_gate_result(), passing_gate_result()], hooks=hooks),
        hooks=hooks,
        restore_fast_path_timeout_seconds=SHORT_FAST_PATH,
    ) as simulation:
        promoted = simulation.run_cycle(run_id="promote", payload=b"weights-promoted")
        assert promoted.outcome is CycleOutcome.PROMOTED
        assert promoted.artifact_id is not None
        assert simulation.serving is not None
        assert simulation.serving.endswith(promoted.artifact_id)

        # Now preempt a later cycle with a canary that will not answer.
        engine.set_completion_budget(1)
        hooks.at("extract", traffic_arrival(simulation.activity, simulation.clock))
        result = simulation.run_cycle(run_id="preempted", payload=b"weights-abandoned")

        assert result.outcome is CycleOutcome.PREEMPTED
        assert simulation.active_artifact_id == promoted.artifact_id
        # Restored to the incumbent by a real restart, not by a fast path that
        # took the controller's word for it.
        assert simulation.process.restarts[-1].endswith(promoted.artifact_id)
        assert simulation.serving == simulation.expected_active_draft()
        assert engine.sleeping is False
