"""The whole lifecycle with nothing scripted between the orchestrator and HTTP.

Elsewhere in this package the gate is scripted, so the *cycle* is under test.
Here the gate is the shipped :class:`~speedlm.gate.runner.BenchmarkGateRunner`
and the verdict is whatever it measures off the simulated engine.  The chain
from "the tuner decided to tune" to "the active pointer moved" runs unbroken:
suite freezing, HTTP replay, Prometheus scraping, promotion arithmetic,
artifact publication, pointer promotion, and the persisted ``decision.json``
that ``speedlm gain`` later reads.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.corpus import simulation_traces
from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    MutableClock,
    Simulation,
    build_simulation,
    real_gate,
    simulation_config,
)
from speedlm.config import PromotionConfig
from speedlm.gate.decide import Reason, Verdict
from speedlm.report import parse_decision
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

#: See ``test_gate_reasons.ACCEPTANCE_ONLY``: the gating throughput statistic is
#: client-measured wall clock and therefore carries the host's scheduling
#: jitter, which on a loaded machine can swamp the millisecond-scale latency
#: difference between the two simulated arms.  These tests are about the chain
#: from decision to active pointer, not about the throughput guard, so the
#: guard is widened and acceptance -- exact counter arithmetic -- does the
#: gating.  ``test_gate_reasons`` exercises the throughput guard on its own.
ACCEPTANCE_ONLY = PromotionConfig(
    min_acceptance_delta_pp=1.0,
    min_throughput_delta_pct=-95.0,
)

STOCK_REFERENCE = "sim/stock-draft"
SUITE_SIZE = 5

STOCK = DraftProfile(
    name="stock",
    acceptance_rate=0.40,
    seconds_per_request=0.005,
)


@dataclass(frozen=True, slots=True)
class MeasuredFactory:
    """Builds a simulation whose gate really measures the engine.

    The candidate's reference is the artifact directory, which does not exist
    until the registry mints it mid-cycle -- so the candidate behaviour is
    supplied as the engine's *fallback* profile, which any unregistered
    reference resolves to.
    """

    root: Path
    engine: SimulatedEngine

    def __call__(
        self,
        candidate: DraftProfile,
        *,
        clock: MutableClock | None = None,
    ) -> Simulation:
        self.engine.set_fallback(candidate)
        return build_simulation(
            self.root,
            engine=self.engine,
            gate=real_gate(
                self.engine,
                simulation_traces(SUITE_SIZE),
                self.root / "suite",
                stock_draft=STOCK_REFERENCE,
                config=simulation_config(promotion=ACCEPTANCE_ONLY),
                clock=clock,
            ),
        )


@pytest.fixture()
def measured(tmp_path: Path) -> Iterator[MeasuredFactory]:
    with running_engine(default_profile=STOCK) as engine:
        engine.register(STOCK_REFERENCE, STOCK)
        yield MeasuredFactory(root=tmp_path, engine=engine)


class TestMeasuredPromotion:
    def test_a_genuinely_better_head_is_measured_promoted_and_served(
        self, measured: MeasuredFactory, tmp_path: Path
    ) -> None:
        simulation = measured(
            DraftProfile(
                name="candidate", acceptance_rate=0.55, seconds_per_request=0.002
            )
        )
        result = simulation.run_cycle(run_id="measured", payload=b"weights-better")

        assert result.outcome is CycleOutcome.PROMOTED
        assert result.artifact_id is not None
        assert simulation.active_artifact_id == result.artifact_id
        assert simulation.state.state is TunerState.READY

        # The verdict came from real arithmetic on real counters.
        gate_result = result.gate
        assert gate_result is not None
        decision = gate_result.decision
        assert decision is not None
        assert decision.verdict is Verdict.PROMOTE
        assert decision.reason is Reason.BOTH_THRESHOLDS_MET
        assert decision.acceptance_delta_pp == pytest.approx(15.0, abs=1e-4)

        # ...and it is on disk in the shape the reporting layer reads back.
        assert result.decision_path is not None
        assert result.decision_path.name == "decision.json"
        round_tripped = parse_decision(
            json.loads(result.decision_path.read_text(encoding="utf-8")),
            source=result.decision_path,
        )
        assert round_tripped.verdict is Verdict.PROMOTE
        assert round_tripped.reason is Reason.BOTH_THRESHOLDS_MET
        assert round_tripped.acceptance_delta_pp == pytest.approx(
            decision.acceptance_delta_pp
        )
        assert round_tripped.num_repeats == decision.num_repeats

    def test_the_raw_scrapes_behind_the_decision_are_stored_beside_it(
        self, measured: MeasuredFactory, tmp_path: Path
    ) -> None:
        simulation = measured(
            DraftProfile(
                name="candidate", acceptance_rate=0.55, seconds_per_request=0.002
            )
        )
        result = simulation.run_cycle(run_id="evidence", payload=b"weights-better")
        assert result.outcome is CycleOutcome.PROMOTED

        metrics_dir = tmp_path / "runs" / "evidence" / "gate-metrics"
        names = sorted(p.name for p in metrics_dir.iterdir())
        assert names == sorted(
            f"{arm}-{label}.prom.gz"
            for arm in ("stock", "candidate")
            for label in ("before", "after-repeat-0", "after-repeat-1", "after")
        )
        # The stored body must be the verbatim exposition, so a reader can
        # re-derive the reported rate from the absolute counters.
        body = gzip.decompress(
            (metrics_dir / "candidate-after.prom.gz").read_bytes()
        ).decode("utf-8")
        assert "vllm:spec_decode_num_accepted_tokens_total" in body


class TestMeasuredRejection:
    def test_a_worse_head_is_measured_rejected_and_never_serves(
        self, measured: MeasuredFactory, tmp_path: Path
    ) -> None:
        simulation = measured(
            DraftProfile(
                name="candidate", acceptance_rate=0.30, seconds_per_request=0.005
            )
        )
        result = simulation.run_cycle(run_id="measured", payload=b"weights-worse")

        assert result.outcome is CycleOutcome.REJECTED
        assert simulation.artifacts.active_pointer() is None
        assert simulation.runtime.serving == STOCK_REFERENCE
        assert simulation.state.state is TunerState.READY

        gate_result = result.gate
        assert gate_result is not None
        assert gate_result.decision is not None
        assert gate_result.decision.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD
        # A rejection is a measurement, so it is persisted like one.
        assert result.decision_path is not None
        assert result.decision_path.exists()

    def test_a_promotion_then_a_measured_rejection_keeps_the_incumbent(
        self, measured: MeasuredFactory, tmp_path: Path
    ) -> None:
        # Two cycles against one engine, each measured independently, with the
        # engine's behaviour changing between them -- the sequence a real
        # deployment produces and the GPU e2e cannot arrange.
        simulation = measured(
            DraftProfile(
                name="candidate-good", acceptance_rate=0.55, seconds_per_request=0.002
            )
        )
        promoted = simulation.run_cycle(run_id="c1", payload=b"weights-good")
        assert promoted.outcome is CycleOutcome.PROMOTED
        pointer_bytes = simulation.artifacts.active_path.read_bytes()

        # The next candidate trains out worse.
        simulation.engine.set_fallback(
            DraftProfile(
                name="candidate-bad",
                acceptance_rate=0.25,
                seconds_per_request=0.005,
            )
        )
        rejected = simulation.run_cycle(run_id="c2", payload=b"weights-bad")

        assert rejected.outcome is CycleOutcome.REJECTED
        assert simulation.artifacts.active_path.read_bytes() == pointer_bytes
        assert simulation.active_artifact_id == promoted.artifact_id
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id
        # Both runs left their own evidence; neither overwrote the other's.
        assert (tmp_path / "runs" / "c1" / "decision.json").exists()
        assert (tmp_path / "runs" / "c2" / "decision.json").exists()


def test_a_benchmark_timeout_end_to_end_is_not_reported_as_a_rejection(
    measured: MeasuredFactory,
    tmp_path: Path,
) -> None:
    simulation = measured(
        DraftProfile(name="candidate", acceptance_rate=0.55, seconds_per_request=0.002),
        # The runner only reads its clock at stage boundaries, so a clock that
        # ticks on every read expires the deadline without anything being slow.
        clock=MutableClock(advance_per_call=0.05),
    )
    engine = simulation.engine
    # The orchestrator clamps the gate's own estimate by this ceiling, so a
    # small ceiling is what makes the deadline reachable.
    object.__setattr__(simulation.timeouts, "benchmark", 1.0)

    result = simulation.run_cycle(run_id="timeout")

    assert result.outcome is CycleOutcome.BENCHMARK_TIMED_OUT
    assert result.outcome is not CycleOutcome.REJECTED
    # No decision was reached, so none was written: a report built from the
    # runs directory cannot mistake this for a measured loss.
    assert result.decision_path is None
    assert not (tmp_path / "runs" / "timeout" / "decision.json").exists()
    assert result.gate is not None
    assert result.gate.decision is None
    # Serving is still restored, and the engine is awake.
    assert simulation.state.state is TunerState.READY
    assert simulation.runtime.serving == STOCK_REFERENCE
    assert engine.sleeping is False
