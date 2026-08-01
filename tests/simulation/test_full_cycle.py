"""Whole cycles, end to end, against production orchestration.

The GPU e2e can run one cycle in about fourteen minutes and cannot choose its
outcome.  These run several in under a second and choose every one, which is
what makes "the active pointer is correct after a promote *and* after the
reject that follows it" a thing that can be asserted at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    Simulation,
    build_simulation,
    failing_gate_result,
    passing_gate_result,
)
from speedlm.tuner.artifacts import ArtifactError
from speedlm.tuner.orchestrator import CycleOutcome, GateFailure, GateResult
from speedlm.tuner.state import TunerState

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)


@dataclass(frozen=True, slots=True)
class SimulationFactory:
    """Builds a simulation driven by a scripted list of gate verdicts."""

    root: Path
    engine: SimulatedEngine

    def __call__(self, verdicts: Sequence[GateResult]) -> Simulation:
        return build_simulation(
            self.root,
            engine=self.engine,
            gate=ScriptedGate(list(verdicts)),
        )


@pytest.fixture()
def sim(tmp_path: Path) -> Iterator[SimulationFactory]:
    """A simulation factory whose engine is torn down with the test."""
    with running_engine(default_profile=STOCK) as engine:
        yield SimulationFactory(root=tmp_path, engine=engine)


class TestPromoteThenReject:
    def test_promote_moves_the_pointer_and_the_reject_after_it_does_not(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result(), failing_gate_result()])

        promoted = simulation.run_cycle(run_id="promote", payload=b"weights-promoted")
        assert promoted.outcome is CycleOutcome.PROMOTED
        assert promoted.artifact_id is not None
        assert simulation.active_artifact_id == promoted.artifact_id
        assert simulation.state.state is TunerState.READY
        # Serving must be pointed at the promoted artifact's directory, not at
        # the base draft: a promotion that leaves the old head serving has
        # promoted nothing.
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id

        pointer_bytes = simulation.artifacts.active_path.read_bytes()

        rejected = simulation.run_cycle(run_id="reject", payload=b"weights-rejected")
        assert rejected.outcome is CycleOutcome.REJECTED
        assert rejected.artifact_id is not None
        assert rejected.artifact_id != promoted.artifact_id
        # Byte-for-byte: a rejection must not rewrite the pointer even with the
        # same value, because a rewrite is a chance to write the wrong value.
        assert simulation.artifacts.active_path.read_bytes() == pointer_bytes
        assert simulation.active_artifact_id == promoted.artifact_id
        assert simulation.state.state is TunerState.READY
        # And serving is restored to the *promoted* artifact, not the base.
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id

    def test_the_rejected_candidate_is_still_published_and_readable(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([failing_gate_result("acceptance_below_threshold")])
        result = simulation.run_cycle(payload=b"weights-rejected")

        assert result.outcome is CycleOutcome.REJECTED
        assert result.artifact_id is not None
        # A rejected candidate is evidence, not garbage: it stays in the
        # registry, content-verified, so a later run can compare against it.
        artifact = simulation.artifacts.get(result.artifact_id)
        assert artifact.manifest.val_loss == pytest.approx(0.50)
        assert artifact.manifest.base_draft == "sim/stock-draft"
        assert artifact.manifest.trace_hash == "sim-trace-hash"
        assert (artifact.path / "model.safetensors").read_bytes() == b"weights-rejected"

    def test_rejecting_without_an_incumbent_leaves_no_active_pointer(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([failing_gate_result()])
        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.REJECTED
        assert simulation.artifacts.active_pointer() is None
        # With nothing promoted, serving falls back to the backend's base draft.
        assert simulation.runtime.serving == "sim/stock-draft"


class TestMultipleCycles:
    def test_three_cycles_with_mixed_outcomes_leak_no_state(self, sim: SimulationFactory) -> None:
        simulation = sim(
            [
                passing_gate_result(),
                failing_gate_result(),
                passing_gate_result(),
                failing_gate_result(),
            ]
        )

        first = simulation.run_cycle(run_id="c1", payload=b"weights-1")
        second = simulation.run_cycle(run_id="c2", payload=b"weights-2")
        third = simulation.run_cycle(run_id="c3", payload=b"weights-3")
        fourth = simulation.run_cycle(run_id="c4", payload=b"weights-4")

        assert [r.outcome for r in (first, second, third, fourth)] == [
            CycleOutcome.PROMOTED,
            CycleOutcome.REJECTED,
            CycleOutcome.PROMOTED,
            CycleOutcome.REJECTED,
        ]
        ids = [r.artifact_id for r in (first, second, third, fourth)]
        assert len(set(ids)) == 4, "each cycle must mint a distinct artifact"

        # The pointer ends on the last *promoted* candidate, and its history is
        # exactly the promotions before it -- rejections never enter history,
        # which is what stops a rollback restoring a head that never gated.
        pointer = simulation.artifacts.active_pointer()
        assert pointer is not None
        assert pointer.artifact_id == third.artifact_id
        assert pointer.history == (first.artifact_id,)

        assert simulation.state.state is TunerState.READY
        # Every cycle ran the same effect sequence; none of them left the
        # runtime mid-cycle.
        assert simulation.runtime.calls.count("quiesce") == 4
        assert simulation.runtime.calls.count("wake") == 4

    def test_each_cycle_gets_its_own_run_directory(self, sim: SimulationFactory) -> None:
        simulation = sim([passing_gate_result(), failing_gate_result()])
        simulation.run_cycle(run_id="run-a", payload=b"weights-a")
        simulation.run_cycle(run_id="run-b", payload=b"weights-b")

        runs = sorted(p.name for p in (simulation.root / "runs").iterdir() if p.is_dir())
        assert runs == ["run-a", "run-b"]

    def test_a_reused_run_directory_fails_the_cycle_rather_than_overwriting(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result(), passing_gate_result()])
        first = simulation.run_cycle(run_id="collide", payload=b"weights-a")
        assert first.outcome is CycleOutcome.PROMOTED

        # ``work_dir.mkdir(exist_ok=False)`` is deliberate: silently reusing a
        # run directory would mix two cycles' evidence into one.
        #
        # It sits *before* ``run_once``'s try block, so a collision propagates
        # instead of becoming a ``FAILED`` CycleResult.  That is safe rather
        # than tidy -- nothing has been quiesced, slept or transitioned yet, so
        # there is nothing to roll back -- and with the default
        # ``uuid4().hex`` run-id factory a collision cannot occur in practice.
        # Pinned here so a future change that moves the mkdir *after* a state
        # transition, and therefore starts leaving a cycle stranded mid-flight,
        # is a test failure rather than a silent regression.
        with pytest.raises(FileExistsError):
            simulation.run_cycle(run_id="collide", payload=b"weights-b")

        assert simulation.state.state is TunerState.READY
        assert simulation.active_artifact_id == first.artifact_id
        assert Path(str(simulation.runtime.serving)).name == first.artifact_id

    def test_an_identical_candidate_republishes_to_the_same_artifact(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result(), passing_gate_result()])
        first = simulation.run_cycle(run_id="c1", payload=b"identical")
        second = simulation.run_cycle(run_id="c2", payload=b"identical")

        # Content addressing means retraining to the same weights is a no-op on
        # the registry, and promoting the already-active artifact must not push
        # it onto its own history.
        assert first.artifact_id == second.artifact_id
        pointer = simulation.artifacts.active_pointer()
        assert pointer is not None
        assert pointer.history == ()


class TestGateFailuresAreNotRejections:
    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            (GateFailure.TIMED_OUT, CycleOutcome.BENCHMARK_TIMED_OUT),
            (GateFailure.ABORTED, CycleOutcome.BENCHMARK_ABORTED),
        ],
    )
    def test_a_gate_that_never_measured_reports_infrastructure_not_science(
        self,
        sim: SimulationFactory,
        failure: GateFailure,
        expected: CycleOutcome,
    ) -> None:
        simulation = sim(
            [
                GateResult(
                    passed=False,
                    reason=f"benchmark {failure.value}",
                    metrics={failure.value: True},
                    failure=failure,
                )
            ]
        )
        result = simulation.run_cycle()

        assert result.outcome is expected
        assert result.outcome is not CycleOutcome.REJECTED
        # No decision was produced, so none is persisted: a reader of the run
        # directory can tell "not measured" from "measured and declined".
        assert result.decision_path is None
        assert not (simulation.root / "runs" / "cycle-01" / "decision.json").exists()
        assert simulation.state.state is TunerState.READY
        assert simulation.runtime.serving == "sim/stock-draft"


class TestPersistedEvidence:
    def test_raw_metrics_bodies_are_persisted_beside_the_run(self, sim: SimulationFactory) -> None:
        body = "# HELP x\nvllm:generation_tokens_total{engine=\"0\"} 5.0\n"
        simulation = sim(
            [
                GateResult(
                    passed=False,
                    reason="acceptance_below_threshold",
                    metrics_bodies={"stock-before": body, "candidate-after": body},
                )
            ]
        )
        result = simulation.run_cycle(run_id="evidence")

        assert result.outcome is CycleOutcome.REJECTED
        metrics_dir = simulation.root / "runs" / "evidence" / "gate-metrics"
        assert sorted(p.name for p in metrics_dir.iterdir()) == [
            "candidate-after.prom.gz",
            "stock-before.prom.gz",
        ]

    def test_an_unsafe_metrics_label_fails_the_cycle_closed(self, sim: SimulationFactory) -> None:
        simulation = sim(
            [
                GateResult(
                    passed=True,
                    reason="both_thresholds_met",
                    metrics_bodies={"../escape": "x 1\n"},
                )
            ]
        )
        result = simulation.run_cycle()

        # Fail-closed: evidence that cannot be filed safely aborts the cycle
        # rather than being dropped, and serving is restored regardless.
        assert result.outcome is CycleOutcome.FAILED
        assert simulation.state.state is TunerState.READY
        assert simulation.artifacts.active_pointer() is None

    def test_the_state_journal_records_every_transition_of_every_cycle(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result(), failing_gate_result()])
        simulation.run_cycle(run_id="c1", payload=b"weights-1")
        simulation.run_cycle(run_id="c2", payload=b"weights-2")

        events = [
            json.loads(line)
            for line in simulation.state.events_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        transitions = [(e["from"], e["to"]) for e in events]
        assert transitions[0] == (None, "READY")
        # The promoting cycle and the rejecting cycle take structurally
        # different paths through the machine, and both end at READY.
        assert ("BENCHMARKING", "PROMOTING") in transitions
        assert ("BENCHMARKING", "ROLLING_BACK") in transitions
        assert transitions[-1] == ("WAKING", "READY")
        assert all(e["recovery"] is False for e in events)
        # Sequence numbers are dense and strictly increasing across cycles.
        assert [e["sequence"] for e in events] == list(range(len(events)))


class TestRecovery:
    def test_a_cycle_interrupted_mid_flight_recovers_before_the_next_one_starts(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result()])
        # Simulate a process that died in TRAINING: the durable journal says
        # TRAINING, nothing else does.
        simulation.state.transition(TunerState.QUIESCING, reason="crashed run")
        simulation.state.transition(TunerState.SLEEPING, reason="crashed run")
        simulation.state.transition(TunerState.EXTRACTING, reason="crashed run")
        simulation.state.transition(TunerState.TRAINING, reason="crashed run")

        result = simulation.run_cycle()

        assert result.outcome is CycleOutcome.PROMOTED
        assert simulation.state.state is TunerState.READY
        # Recovery restored serving from the durable pointer *before* the new
        # cycle touched anything.
        assert simulation.runtime.calls[:2] == ["restore", "wake"]

    def test_recovery_restores_the_promoted_draft_not_the_base(
        self, sim: SimulationFactory
    ) -> None:
        simulation = sim([passing_gate_result(), passing_gate_result()])
        promoted = simulation.run_cycle(run_id="c1", payload=b"weights-1")
        assert promoted.artifact_id is not None

        simulation.state.transition(TunerState.QUIESCING, reason="crashed run")
        simulation.state.transition(TunerState.SLEEPING, reason="crashed run")
        simulation.runtime.serving = "wrong-draft"
        errors = simulation.orchestrator().recover()

        assert errors == ()
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id


def test_artifact_registry_refuses_activation_without_a_gate_pass(
    tmp_path: Path,
) -> None:
    # The registry is the last line of defence: even holding a published
    # artifact ID, nothing can activate it by asserting success.
    with running_engine(default_profile=STOCK) as engine:
        simulation = build_simulation(
            tmp_path, engine=engine, gate=ScriptedGate([failing_gate_result()])
        )
        result = simulation.run_cycle()
        assert result.artifact_id is not None
        with pytest.raises(ArtifactError, match="without a gate pass"):
            simulation.artifacts.promote(result.artifact_id, gate_passed=False)
