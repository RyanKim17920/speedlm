"""The candidate arm runs first, so a rejecting cycle needs no rollback restart.

``CANDIDATE_STARTING`` already spends a full engine launch on the candidate.
Benchmarking stock first threw that engine away, measured stock, launched the
candidate again, and then -- on a rejection -- launched stock a third time to
roll back.  Running the candidate arm first reuses the engine the cycle just
built and leaves the benchmark ending on stock, which is what a rejection wants
serving anyway.

The saving is only expressible against an endpoint that can decline to restart
(``simulation.production.SimulatedDraftEndpoint``, mirroring production's
``_DraftEndpoint``) and a controller that knows what is running.  So these tests
run the *production* runtime controller and the *production*
:class:`~speedlm.gate.runner.BenchmarkGateRunner` against the simulated engine,
and count launches from the engine's own journal.

The counts are launches, not seconds.  A simulated restart is free; the
production docstring puts a real one at ~100-105s.  Asserting the count is the
honest form of "downtime went down".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from simulation.corpus import simulation_traces
from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import real_gate, simulation_config
from simulation.production import (
    ProductionSimulation,
    production_simulation,
    shared_endpoint,
)
from speedlm.config import PromotionConfig
from speedlm.gate.decide import Verdict
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

#: See ``test_end_to_end.ACCEPTANCE_ONLY``: the throughput guard is
#: client-measured wall clock and carries host jitter, so acceptance -- exact
#: counter arithmetic -- does the gating here.  Arm *order* is the subject.
ACCEPTANCE_ONLY = PromotionConfig(
    min_acceptance_delta_pp=1.0,
    min_throughput_delta_pct=-95.0,
)

STOCK_REFERENCE = "sim/stock-draft"
SUITE_SIZE = 3
STOCK = DraftProfile(name="stock", acceptance_rate=0.40, seconds_per_request=0.0)
BETTER = DraftProfile(name="candidate", acceptance_rate=0.55, seconds_per_request=0.0)
WORSE = DraftProfile(name="candidate", acceptance_rate=0.25, seconds_per_request=0.0)


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        running.register(STOCK_REFERENCE, STOCK)
        yield running


def measured_cycle(
    root: Path,
    engine: SimulatedEngine,
    candidate: DraftProfile,
    *,
    candidate_arm_first: bool,
) -> AbstractContextManager[ProductionSimulation]:
    """A cycle whose gate really measures the engine, sharing the controller."""
    engine.set_fallback(candidate)
    return production_simulation(
        root,
        engine=engine,
        gate_factory=lambda simulation: real_gate(
            engine,
            simulation_traces(SUITE_SIZE),
            root / "suite",
            stock_draft=STOCK_REFERENCE,
            config=simulation_config(promotion=ACCEPTANCE_ONLY),
            endpoint=shared_endpoint(simulation),
            candidate_arm_first=candidate_arm_first,
            # Arm order is about *which engine gets reused*, not about warm
            # kernels, and every warmup pass is a whole extra suite replay.
            warmup_repeats=0,
        ),
    )


class TestARejectingCycle:
    def test_candidate_first_pays_three_launches_where_stock_first_pays_four(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        with measured_cycle(
            tmp_path / "first", engine, WORSE, candidate_arm_first=True
        ) as simulation:
            result = simulation.run_cycle(run_id="reject", payload=b"weights-worse")
            assert result.outcome is CycleOutcome.REJECTED
            candidate_first = simulation.restarts
            # The candidate arm deliberately rebuilds the engine that
            # CANDIDATE_STARTING built so both scored arms begin from the same
            # lifecycle state.
            assert simulation.endpoint is not None
            assert simulation.endpoint.requested[0].endswith(str(result.artifact_id))
            assert simulation.endpoint.restarted[0].endswith(str(result.artifact_id))
            # The benchmark ended on stock, so rollback still takes the
            # wake-in-place fast path.
            assert simulation.serving == STOCK_REFERENCE
            assert simulation.state.state is TunerState.READY

        with measured_cycle(
            tmp_path / "second", engine, WORSE, candidate_arm_first=False
        ) as simulation:
            result = simulation.run_cycle(run_id="reject", payload=b"weights-worse")
            assert result.outcome is CycleOutcome.REJECTED
            stock_first = simulation.restarts
            assert simulation.serving == STOCK_REFERENCE

        assert candidate_first == 3
        assert stock_first == 4
        assert candidate_first < stock_first

    def test_the_old_order_still_works_and_still_ends_on_the_right_draft(
        self, tmp_path: Path, engine: SimulatedEngine
    ) -> None:
        # ``benchmark_candidate_arm_first: false`` is a supported configuration,
        # not dead code: it is the escape hatch if the new order ever biases a
        # measurement.  It must still reach the same verdict.
        with measured_cycle(
            tmp_path, engine, WORSE, candidate_arm_first=False
        ) as simulation:
            result = simulation.run_cycle(run_id="reject", payload=b"weights-worse")

            assert result.outcome is CycleOutcome.REJECTED
            assert result.gate is not None
            assert result.gate.decision is not None
            assert result.gate.decision.verdict is Verdict.REJECT
            assert simulation.endpoint is not None
            # Stock, then candidate -- and the stock arm cost a launch, because
            # the engine was holding the candidate when the benchmark began.
            assert simulation.endpoint.requested[0] == STOCK_REFERENCE
            assert simulation.endpoint.restarted[0] == STOCK_REFERENCE
            assert simulation.serving == STOCK_REFERENCE
            assert simulation.artifacts.active_pointer() is None


class TestAPromotingCycle:
    @pytest.mark.parametrize("candidate_first", [True, False])
    def test_the_promoted_candidate_is_what_actually_serves(
        self, tmp_path: Path, engine: SimulatedEngine, candidate_first: bool
    ) -> None:
        # This is the trap in candidate-first.  The orchestrator promotes by
        # flipping the durable pointer and *waking* -- it never restarts -- so
        # a benchmark that ended on stock would leave stock answering traffic
        # under a pointer that claims the candidate.  Serving and the pointer
        # must agree.
        with measured_cycle(
            tmp_path, engine, BETTER, candidate_arm_first=candidate_first
        ) as simulation:
            result = simulation.run_cycle(run_id="promote", payload=b"weights-better")

            assert result.outcome is CycleOutcome.PROMOTED
            assert result.artifact_id is not None
            assert simulation.active_artifact_id == result.artifact_id
            assert simulation.serving is not None
            assert simulation.serving.endswith(result.artifact_id)
            assert simulation.serving == simulation.expected_active_draft()
            assert str(simulation.runtime.running_draft) == simulation.serving
            assert engine.sleeping is False
            assert simulation.state.state is TunerState.READY
