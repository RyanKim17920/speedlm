"""A real request arrives at every stage boundary, one boundary per test.

The invariant under test is the whole reason idle tuning is allowed to touch a
serving engine at all: *whenever* traffic arrives, the cycle abandons itself and
serving comes back on the correct draft.  A GPU e2e can demonstrate this once,
at whichever boundary the timing happened to land on.  Here every boundary is
reached deliberately.
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
    passing_gate_result,
    traffic_arrival,
)
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)

#: Every boundary at which a request can land during a cycle, in the order the
#: orchestrator crosses them.
BOUNDARIES = ("quiesce", "sleep", "extract", "train", "candidate_start", "benchmark")


@dataclass(frozen=True, slots=True)
class PreemptionFactory:
    root: Path
    engine: SimulatedEngine

    def __call__(self, boundary: str, *, cycles: int = 1) -> Simulation:
        hooks = StageHooks()
        simulation = build_simulation(
            self.root,
            engine=self.engine,
            gate=ScriptedGate([passing_gate_result() for _ in range(cycles + 1)], hooks=hooks),
            hooks=hooks,
        )
        hooks.at(boundary, traffic_arrival(simulation.activity, simulation.clock))
        return simulation


@pytest.fixture()
def preemptible(tmp_path: Path) -> Iterator[PreemptionFactory]:
    with running_engine(default_profile=STOCK) as engine:
        yield PreemptionFactory(root=tmp_path, engine=engine)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_traffic_at_every_boundary_abandons_the_cycle_and_restores_the_base_draft(
    preemptible: PreemptionFactory,
    boundary: str,
) -> None:
    simulation = preemptible(boundary)
    result = simulation.run_cycle()

    assert result.outcome is CycleOutcome.PREEMPTED, (
        f"traffic during {boundary} must preempt, got {result.outcome} ({result.error})"
    )
    assert simulation.state.state is TunerState.READY
    # Nothing was ever promoted, so serving must land back on the base draft --
    # not on a candidate that was mid-flight when the request arrived.
    assert simulation.runtime.serving == "sim/stock-draft"
    assert simulation.artifacts.active_pointer() is None
    # Rollback always ends with a restore followed by a wake, whatever stage it
    # started from.  An engine left asleep is an outage.
    assert simulation.runtime.calls[-2:] == ["restore", "wake"]
    assert simulation.engine.sleeping is False


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_traffic_at_every_boundary_restores_the_promoted_draft_when_one_exists(
    preemptible: PreemptionFactory,
    boundary: str,
) -> None:
    simulation = preemptible(boundary, cycles=2)
    # Land a promotion first, with the preemption hook disarmed for that cycle.
    armed = simulation.hooks.hooks.pop(boundary)
    promoted = simulation.run_cycle(run_id="promote", payload=b"weights-promoted")
    assert promoted.outcome is CycleOutcome.PROMOTED
    assert promoted.artifact_id is not None
    pointer_bytes = simulation.artifacts.active_path.read_bytes()

    simulation.hooks.hooks[boundary] = armed
    result = simulation.run_cycle(run_id="preempted", payload=b"weights-abandoned")

    assert result.outcome is CycleOutcome.PREEMPTED
    assert simulation.state.state is TunerState.READY
    # The incumbent survives untouched, byte for byte...
    assert simulation.artifacts.active_path.read_bytes() == pointer_bytes
    assert simulation.active_artifact_id == promoted.artifact_id
    # ...and serving is put back on it, not on the abandoned candidate.
    assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id


def test_preemption_before_the_candidate_starts_never_swaps_the_engine(
    preemptible: PreemptionFactory,
) -> None:
    # Boundaries before ``candidate_start`` must leave the engine on whatever it
    # was already serving: a candidate that was never benchmarked must never
    # have been loaded.
    for boundary in ("quiesce", "sleep", "extract", "train"):
        simulation = preemptible(boundary)
        simulation.run_cycle(run_id=f"pre-{boundary}")
        assert "candidate_start" not in simulation.runtime.calls, boundary


def test_preemption_after_the_candidate_starts_still_publishes_the_artifact(
    preemptible: PreemptionFactory,
) -> None:
    # Training work is not thrown away because a request arrived after it: the
    # candidate is published and remains available for a later cycle to gate.
    simulation = preemptible("benchmark")
    result = simulation.run_cycle()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert result.artifact_id is not None
    artifact = simulation.artifacts.get(result.artifact_id)
    assert artifact.manifest.val_loss == pytest.approx(0.50)
    # Published, but emphatically not active.
    assert simulation.artifacts.active_pointer() is None


def test_the_abort_check_is_live_all_the_way_down(
    preemptible: PreemptionFactory,
) -> None:
    # The orchestrator hands every backend and runtime call a ``should_abort``
    # closure.  If any of them received a constant ``False`` the guard would be
    # decorative, so assert the check actually flips underneath them.
    simulation = preemptible("extract")
    simulation.run_cycle()

    seen = dict(simulation.backend.aborts_seen)
    assert seen["prepare"] is True, "abort check did not observe the arrived request"


def test_an_in_flight_request_alone_preempts_without_any_new_arrival(
    preemptible: PreemptionFactory,
) -> None:
    # ``PreemptionGuard`` trips on ``in_flight > 0`` as well as on a newer
    # watermark: a long-running request that started before the cycle armed
    # must still stop it.
    simulation = preemptible("quiesce")
    simulation.hooks.hooks["quiesce"] = simulation.activity.begin
    result = simulation.run_cycle()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert simulation.state.state is TunerState.READY
    assert simulation.runtime.serving == "sim/stock-draft"


def test_a_cycle_can_promote_normally_after_a_preempted_one(
    preemptible: PreemptionFactory,
) -> None:
    # Preemption must be a clean abandonment, not a poisoned state machine.
    simulation = preemptible("train", cycles=2)
    first = simulation.run_cycle(run_id="preempted", payload=b"weights-abandoned")
    assert first.outcome is CycleOutcome.PREEMPTED

    simulation.hooks.hooks.pop("train")
    second = simulation.run_cycle(run_id="promoted", payload=b"weights-good")

    assert second.outcome is CycleOutcome.PROMOTED
    assert simulation.active_artifact_id == second.artifact_id
    assert simulation.state.state is TunerState.READY


def test_a_gateway_that_is_not_idle_is_never_armed_at_all(
    preemptible: PreemptionFactory,
) -> None:
    simulation = preemptible("quiesce")
    simulation.activity.begin()

    result = simulation.orchestrator().run_once()

    assert result.outcome is CycleOutcome.NOT_IDLE
    # Not merely "no promotion": nothing was touched.  A busy gateway must not
    # even be quiesced.
    assert simulation.runtime.calls == []
    assert simulation.state.state is TunerState.READY
