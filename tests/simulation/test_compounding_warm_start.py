"""Whether learning accumulates across cycles, against production orchestration.

The one property the GPU e2e cannot show: it runs a single cycle, and a warm
start frozen at composition time is indistinguishable from a correct one until
a *second* cycle asks what the first one promoted.  These run several cycles
in under a second and choose every verdict.
"""

from __future__ import annotations

from collections.abc import Iterator
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
from speedlm.tuner.composition import promotion_chain_depth
from speedlm.tuner.orchestrator import CycleOutcome

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)
BASE_DRAFT = "sim/stock-draft"


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        yield running


def _simulation(
    root: Path,
    engine: SimulatedEngine,
    *,
    verdicts: list[object],
    compounding: bool = True,
    max_chain_depth: int | None = None,
) -> Simulation:
    return build_simulation(
        root,
        engine=engine,
        gate=ScriptedGate(list(verdicts)),  # type: ignore[arg-type]
        base_draft=BASE_DRAFT,
        compounding_warm_start=compounding,
        warm_start_max_chain_depth=max_chain_depth,
    )


class TestCompoundingAcrossCycles:
    def test_a_promotion_becomes_the_next_cycles_warm_start(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        simulation = _simulation(
            tmp_path,
            engine,
            verdicts=[passing_gate_result(), passing_gate_result()],
        )

        first = simulation.run_cycle(run_id="one", payload=b"weights-one")
        assert first.outcome is CycleOutcome.PROMOTED
        assert first.artifact_id is not None
        first_path = simulation.artifacts.get(first.artifact_id).path

        second = simulation.run_cycle(run_id="two", payload=b"weights-two")
        assert second.outcome is CycleOutcome.PROMOTED
        assert second.artifact_id is not None

        assert simulation.backend.trained_from == [BASE_DRAFT, str(first_path)]
        # Provenance, not just behaviour: the manifest has to say which head
        # each cycle built on, or the chain is unreconstructable afterwards.
        assert (
            simulation.artifacts.get(second.artifact_id).manifest.base_draft
            == str(first_path)
        )
        assert (
            promotion_chain_depth(simulation.artifacts.get(second.artifact_id).path) == 2
        )

    def test_a_rejection_leaves_the_warm_start_on_the_incumbent(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        """A rejected candidate is not a new base, and stock is not one either.

        Both wrong answers are reachable: following the candidate would build on
        a head the gate refused, and falling back to stock would silently
        restart the chain on every rejection -- which, given how often the
        archived runs reject, is most cycles.
        """
        simulation = _simulation(
            tmp_path,
            engine,
            verdicts=[
                passing_gate_result(),
                failing_gate_result(),
                failing_gate_result(),
            ],
        )

        promoted = simulation.run_cycle(run_id="one", payload=b"weights-one")
        assert promoted.outcome is CycleOutcome.PROMOTED
        assert promoted.artifact_id is not None
        incumbent = simulation.artifacts.get(promoted.artifact_id).path

        for index, payload in enumerate((b"weights-two", b"weights-three")):
            rejected = simulation.run_cycle(run_id=f"reject-{index}", payload=payload)
            assert rejected.outcome is CycleOutcome.REJECTED
            assert rejected.artifact_id != promoted.artifact_id

        assert simulation.backend.trained_from == [
            BASE_DRAFT,
            str(incumbent),
            str(incumbent),
        ]
        assert simulation.active_artifact_id == promoted.artifact_id

    def test_an_empty_registry_falls_back_to_the_stock_drafter(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        """First cycle ever, and any state where nothing has been promoted."""
        simulation = _simulation(
            tmp_path,
            engine,
            verdicts=[failing_gate_result(), failing_gate_result()],
        )

        for index in range(2):
            result = simulation.run_cycle(run_id=f"cycle-{index}")
            assert result.outcome is CycleOutcome.REJECTED

        assert simulation.active_artifact_id is None
        assert simulation.backend.trained_from == [BASE_DRAFT, BASE_DRAFT]

    def test_switching_compounding_off_restores_the_from_stock_behaviour(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        simulation = _simulation(
            tmp_path,
            engine,
            verdicts=[passing_gate_result(), passing_gate_result()],
            compounding=False,
        )

        for index in range(2):
            result = simulation.run_cycle(
                run_id=f"cycle-{index}",
                payload=f"weights-{index}".encode(),
            )
            assert result.outcome is CycleOutcome.PROMOTED

        assert simulation.backend.trained_from == [BASE_DRAFT, BASE_DRAFT]

    def test_the_chain_bound_re_baselines_without_touching_the_gate(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        """Bounding compounding must move only where training *starts*.

        The gate still measures the candidate against the current incumbent and
        still promotes on its own verdict; a re-baselined cycle is an ordinary
        cycle whose base happens to be stock.
        """
        simulation = _simulation(
            tmp_path,
            engine,
            verdicts=[passing_gate_result(), passing_gate_result()],
            max_chain_depth=1,
        )

        first = simulation.run_cycle(run_id="one", payload=b"weights-one")
        second = simulation.run_cycle(run_id="two", payload=b"weights-two")

        assert first.outcome is second.outcome is CycleOutcome.PROMOTED
        assert simulation.backend.trained_from == [BASE_DRAFT, BASE_DRAFT]
        assert second.artifact_id is not None
        assert simulation.active_artifact_id == second.artifact_id
        assert (
            promotion_chain_depth(simulation.artifacts.get(second.artifact_id).path) == 1
        )
