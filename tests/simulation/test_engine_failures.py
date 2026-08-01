"""The engine misbehaves at each stage; serving must survive every time.

The scenarios here are infrastructure, not science: an engine that will not
sleep, will not wake, will not come back, or dies partway through.  In every
one the contract is the same and it is not "the cycle succeeds" -- it is *the
cycle reports a failure and puts serving back where it found it*.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    Simulation,
    StageHooks,
    build_simulation,
    failing_gate_result,
    passing_gate_result,
)
from speedlm.training.masking import FinalAssistantMaskError
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)

#: Backend stages that can fail, and the cycle stage each one belongs to.
BACKEND_STAGES = ("extract", "train", "materialize", "validate")

#: Runtime effects that can fail.  ``restore`` and ``wake`` are excluded here
#: because failing them is failing the recovery itself -- covered separately.
RUNTIME_EFFECTS = ("quiesce", "sleep", "candidate_start")


@dataclass(frozen=True, slots=True)
class FailureFactory:
    root: Path
    engine: SimulatedEngine

    def __call__(self, *, cycles: int = 2) -> Simulation:
        hooks = StageHooks()
        return build_simulation(
            self.root,
            engine=self.engine,
            gate=ScriptedGate([passing_gate_result() for _ in range(cycles)], hooks=hooks),
            hooks=hooks,
        )


@pytest.fixture()
def failing(tmp_path: Path) -> Iterator[FailureFactory]:
    with running_engine(default_profile=STOCK) as engine:
        yield FailureFactory(root=tmp_path, engine=engine)


class TestBackendFailures:
    @pytest.mark.parametrize("stage", BACKEND_STAGES)
    def test_a_backend_failure_at_any_stage_restores_serving(
        self, failing: FailureFactory, stage: str
    ) -> None:
        simulation = failing()
        simulation.backend.fail_in = stage

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FAILED
        assert result.error is not None and stage in result.error
        assert simulation.state.state is TunerState.READY
        assert simulation.runtime.serving == "sim/stock-draft"
        assert simulation.artifacts.active_pointer() is None
        assert simulation.runtime.calls[-2:] == ["restore", "wake"]
        assert simulation.engine.sleeping is False

    def test_a_masking_failure_reports_its_own_outcome(
        self, failing: FailureFactory
    ) -> None:
        # This one is singled out because the operator's next action differs:
        # a mask error means the captured traces are unusable for this template,
        # not that the machine broke.  Flattening it into FAILED would hide that.
        simulation = failing()
        simulation.backend.fail_in = "train"
        simulation.backend.failure = FinalAssistantMaskError("no final assistant turn")

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FINAL_ASSISTANT_MASK_ERROR
        assert result.outcome is not CycleOutcome.FAILED
        assert simulation.state.state is TunerState.READY
        assert simulation.runtime.serving == "sim/stock-draft"

    def test_a_failed_cycle_does_not_disturb_an_existing_promotion(
        self, failing: FailureFactory
    ) -> None:
        simulation = failing(cycles=3)
        promoted = simulation.run_cycle(run_id="good", payload=b"weights-good")
        assert promoted.outcome is CycleOutcome.PROMOTED
        pointer_bytes = simulation.artifacts.active_path.read_bytes()

        simulation.backend.fail_in = "train"
        failed = simulation.run_cycle(run_id="bad", payload=b"weights-bad")

        assert failed.outcome is CycleOutcome.FAILED
        assert simulation.artifacts.active_path.read_bytes() == pointer_bytes
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id


class TestRuntimeFailures:
    @pytest.mark.parametrize("effect", RUNTIME_EFFECTS)
    def test_a_runtime_failure_at_any_effect_still_ends_ready(
        self, failing: FailureFactory, effect: str
    ) -> None:
        simulation = failing()
        simulation.runtime.fail_in = effect

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FAILED
        assert simulation.state.state is TunerState.READY
        # Rollback still ran, and still ended by waking the engine.  A cycle
        # that fails while the engine is asleep and leaves it asleep is an
        # outage caused by the tuner.
        assert simulation.runtime.calls[-1] == "wake"
        assert simulation.engine.sleeping is False

    def test_a_restore_that_fails_is_reported_rather_than_swallowed(
        self, failing: FailureFactory
    ) -> None:
        simulation = failing()
        simulation.backend.fail_in = "train"
        simulation.runtime.fail_in = "restore"

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FAILED
        # Both the original failure and the cleanup failure are in the error:
        # an operator needs to know the tuner could not put serving back.
        assert result.error is not None
        assert "train" in result.error
        assert "runtime restore failed" in result.error
        # And the machine still lands on READY, because leaving it stuck in
        # ROLLING_BACK would block every future cycle.
        assert simulation.state.state is TunerState.READY

    def test_a_wake_that_fails_leaves_the_journal_honest(
        self, failing: FailureFactory
    ) -> None:
        simulation = failing()
        simulation.runtime.fail_in = "wake"

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FAILED
        assert result.error is not None and "runtime wake failed" in result.error


class TestEngineFaultsDuringACycle:
    def test_a_crashed_engine_does_not_stop_the_cycle_from_ending_ready(
        self, failing: FailureFactory
    ) -> None:
        # The engine dies during training.  The tuner has no traffic to serve
        # while it is asleep, so the cycle only discovers this at the gate --
        # and the contract is still that it reports failure and restores.
        simulation = failing()
        simulation.hooks.at(
            "train",
            lambda: setattr(simulation.engine.faults, "crash_after_requests", 0),
        )
        simulation.gate = ScriptedGate([failing_gate_result("engine unavailable")])

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.REJECTED
        assert simulation.state.state is TunerState.READY
        assert simulation.runtime.serving == "sim/stock-draft"

    def test_an_engine_that_refuses_to_wake_is_surfaced(
        self, failing: FailureFactory
    ) -> None:
        simulation = failing()
        simulation.engine.faults.refuse_wake = True

        result = simulation.run_cycle()

        # The runtime's wake raises out of the engine, so even a promoted cycle
        # reports failure rather than claiming the candidate is serving.
        assert result.outcome is CycleOutcome.FAILED
        assert result.error is not None and "refused to wake" in result.error

    def test_an_engine_that_refuses_to_sleep_aborts_before_any_training(
        self, failing: FailureFactory
    ) -> None:
        simulation = failing()
        simulation.engine.faults.refuse_sleep = True

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.FAILED
        # Training never started: nothing was extracted, so no candidate exists
        # and nothing could have been published.
        assert simulation.backend.prepared_calls == 0
        assert result.artifact_id is None
        assert simulation.artifacts.active_pointer() is None
        assert simulation.state.state is TunerState.READY

    def test_stalling_the_engine_does_not_stall_the_test_suite(
        self, failing: FailureFactory
    ) -> None:
        # Guardrail on the simulator itself: ``stall_seconds`` must be an
        # injected, bounded delay, not an unbounded hang, or a failure in this
        # package would hang CI rather than fail it.
        simulation = failing()
        simulation.engine.faults.stall_seconds = 0.01
        result = simulation.run_cycle()
        assert result.outcome is CycleOutcome.PROMOTED
