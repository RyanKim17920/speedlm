"""The cheap pre-filter, and the line it must not cross.

The pre-filter is a *cost* filter: it may decline to spend a benchmark, and it
may never promote anything.  The scenarios that matter are therefore the ones
where it declines (the benchmark must not run) and the ones where it declines
to decline (the benchmark must run, and the gate must remain the only
authority).
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
    build_simulation,
    failing_gate_result,
    passing_gate_result,
)
from speedlm.config import ValLossPreFilterConfig
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)


@dataclass(frozen=True, slots=True)
class PrefilterFactory:
    root: Path
    engine: SimulatedEngine

    def __call__(
        self,
        *,
        prefilter: ValLossPreFilterConfig | None,
        val_loss: float | None = 0.50,
        cycles: int = 3,
    ) -> Simulation:
        return build_simulation(
            self.root,
            engine=self.engine,
            gate=ScriptedGate([passing_gate_result() for _ in range(cycles)]),
            val_loss=val_loss,
            val_loss_prefilter=prefilter,
        )


@pytest.fixture()
def prefiltered(tmp_path: Path) -> Iterator[PrefilterFactory]:
    with running_engine(default_profile=STOCK) as engine:
        yield PrefilterFactory(root=tmp_path, engine=engine)


class TestSkipping:
    def test_a_candidate_that_did_not_improve_never_reaches_the_benchmark(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        incumbent = simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        assert incumbent.outcome is CycleOutcome.PROMOTED
        benchmarks_after_first = simulation.gate.calls
        effects_after_first = len(simulation.runtime.calls)

        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.499)

        assert result.outcome is CycleOutcome.VAL_LOSS_NOT_IMPROVED
        assert result.val_loss == pytest.approx(0.499)
        # The whole point of the filter: the expensive part did not run.
        assert simulation.gate.calls == benchmarks_after_first
        # And it did not run because it was never started -- the skipped cycle
        # quiesced and slept, then went straight to rollback without ever
        # loading a candidate into the engine.
        assert simulation.runtime.calls[effects_after_first:] == [
            "quiesce",
            "sleep",
            "restore",
            "wake",
        ]
        assert simulation.state.state is TunerState.READY
        assert simulation.active_artifact_id == incumbent.artifact_id

    def test_a_skipped_candidate_is_still_published(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.60)

        assert result.outcome is CycleOutcome.VAL_LOSS_NOT_IMPROVED
        assert result.artifact_id is not None
        artifact = simulation.artifacts.get(result.artifact_id)
        # The manifest records the loss that got it skipped, so the decision is
        # auditable without re-running anything.
        assert artifact.manifest.val_loss == pytest.approx(0.60)

    def test_serving_is_restored_to_the_incumbent_after_a_skip(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        incumbent = simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.50)

        assert Path(str(simulation.runtime.serving)).name == incumbent.artifact_id
        assert simulation.engine.sleeping is False


class TestNotSkipping:
    def test_an_improvement_at_exactly_the_threshold_proceeds(
        self, prefiltered: PrefilterFactory
    ) -> None:
        # ``improvement < min_improvement`` skips, so equality must pass: an
        # off-by-one here silently discards every marginal candidate.
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.49)

        assert result.outcome is CycleOutcome.PROMOTED
        assert simulation.gate.calls == 2

    def test_a_disabled_prefilter_never_skips(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=False, min_improvement=0.01)
        )
        simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.99)

        assert result.outcome is CycleOutcome.PROMOTED
        assert simulation.gate.calls == 2

    def test_a_missing_val_loss_fails_open_to_the_benchmark(
        self, prefiltered: PrefilterFactory
    ) -> None:
        # A backend that cannot report validation loss must not have every
        # candidate silently skipped; the gate is the authority, so the
        # unavailable-metric path has to fall through to it.
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=None)

        assert result.outcome is CycleOutcome.PROMOTED
        assert result.val_loss is None
        assert simulation.gate.calls == 2

    def test_the_first_cycle_has_no_incumbent_to_compare_against(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(
            prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01)
        )
        result = simulation.run_cycle(payload=b"weights-1", val_loss=99.0)

        assert result.outcome is CycleOutcome.PROMOTED
        assert simulation.gate.calls == 1

    def test_no_prefilter_configured_behaves_as_disabled(
        self, prefiltered: PrefilterFactory
    ) -> None:
        simulation = prefiltered(prefilter=None)
        simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=5.0)

        assert result.outcome is CycleOutcome.PROMOTED


def test_the_prefilter_cannot_promote_what_the_gate_rejects(tmp_path: Path) -> None:
    # The invariant the pre-filter must never violate: passing it is permission
    # to spend a benchmark, not permission to ship.
    with running_engine(default_profile=STOCK) as engine:
        simulation = build_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([passing_gate_result(), failing_gate_result()]),
            val_loss_prefilter=ValLossPreFilterConfig(enabled=True, min_improvement=0.01),
        )
        incumbent = simulation.run_cycle(run_id="c1", payload=b"weights-1", val_loss=0.50)
        # A dramatic validation-loss improvement, and still rejected.
        result = simulation.run_cycle(run_id="c2", payload=b"weights-2", val_loss=0.01)

        assert result.outcome is CycleOutcome.REJECTED
        assert simulation.active_artifact_id == incumbent.artifact_id
        assert simulation.state.state is TunerState.READY
