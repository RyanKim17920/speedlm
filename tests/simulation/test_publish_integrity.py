"""Publishing hashes twice, not three times -- and still catches a bad copy.

``ArtifactRegistry.publish`` used to hash the source, hash the copy, and then
hash the copy *again* via ``get``.  For a 2 GB EAGLE-3 draft that is three full
SHA-256 passes, all of them while the gateway's admission gate is closed, so the
third was paid for in serving downtime.  It was dropped on the argument that it
never added a guarantee: ``copied_hash`` already proved that exact tree hashes
to the artifact ID, ``os.rename`` is atomic, and nothing in between can alter
content.

That argument is only worth as much as the check it left behind, so these tests
make the copy actually land wrong.  ``shutil.copytree`` is patched *inside the
registry's own module*, which is the one seam that reproduces a bad copy without
pretending: the bytes on the way in are fine, the bytes that arrive are not.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from draft_weights import write_draft_weights

from simulation.engine import DraftProfile, SimulatedEngine, running_engine
from simulation.harness import (
    ScriptedGate,
    build_simulation,
    passing_gate_result,
)
from speedlm.training.masking import MaskPolicy
from speedlm.tuner.artifacts import ArtifactError, ArtifactRegistry, ArtifactSpec
from speedlm.tuner.composition import ProductionTuningError, _PublishedWeightGuard
from speedlm.tuner.eagle3 import (
    DraftWeightsError,
    Eagle3Adapter,
    Eagle3Config,
    weight_fingerprint,
)
from speedlm.tuner.orchestrator import CycleOutcome
from speedlm.tuner.state import TunerState

STOCK = DraftProfile(name="stock", seconds_per_request=0.0)
CORRUPTION = b"bytes-that-never-left-the-source"

SPEC = ArtifactSpec(
    verifier_model="sim/verifier-8b",
    draft_model="sim/draft-eagle3",
    base_draft="sim/stock-draft",
    trace_hash="sim-trace-hash",
    training_params={"steps": 4},
)


@pytest.fixture()
def engine() -> Iterator[SimulatedEngine]:
    with running_engine(default_profile=STOCK) as running:
        yield running


def corrupt_the_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every publish copy land with one file's bytes altered.

    The source is untouched, so the artifact ID computed before the copy is
    correct and the tree that arrives does not match it -- which is exactly the
    shape of a truncated write, a bit flip, or a filesystem that lied about a
    completed copy.
    """
    real_copytree = shutil.copytree

    def landing_wrong(source: Any, destination: Any, **kwargs: Any) -> Any:
        result = real_copytree(source, destination, **kwargs)
        for entry in sorted(Path(destination).rglob("*")):
            if entry.is_file():
                entry.chmod(0o644)
                entry.write_bytes(CORRUPTION)
                break
        return result

    monkeypatch.setattr("speedlm.tuner.artifacts.shutil.copytree", landing_wrong)


def a_draft(root: Path, payload: bytes = b"weights") -> Path:
    draft = root / "draft"
    draft.mkdir()
    (draft / "model.safetensors").write_bytes(payload)
    (draft / "config.json").write_text('{"architectures": ["Eagle3Speculator"]}')
    return draft


class TestTheRegistryAlone:
    def test_a_copy_that_lands_wrong_is_rejected_not_published(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = ArtifactRegistry(tmp_path / "registry")
        draft = a_draft(tmp_path)
        corrupt_the_copy(monkeypatch)

        with pytest.raises(ArtifactError, match="changed while publishing"):
            registry.publish(draft, SPEC)

        # Nothing published, and no half-copied temp tree left behind for a
        # later reader to mistake for an artifact.
        assert list((tmp_path / "registry" / "artifacts").iterdir()) == []
        assert registry.active_pointer() is None

    def test_an_honest_copy_still_publishes_and_verifies(self, tmp_path: Path) -> None:
        registry = ArtifactRegistry(tmp_path / "registry")
        draft = a_draft(tmp_path, b"weights-good")

        artifact = registry.publish(draft, SPEC)

        # ``get`` is the pass that was *kept*: it re-hashes on every later read,
        # which is where a bit-rot or tamper check belongs.
        reread = registry.get(artifact.artifact_id)
        assert reread.artifact_id == artifact.artifact_id
        assert (reread.path / "model.safetensors").read_bytes() == b"weights-good"

    def test_corruption_after_publication_is_still_caught_on_read(
        self, tmp_path: Path
    ) -> None:
        # Dropping the third hash did not drop the *later* verification, so an
        # artifact that rots on disk after publication still fails closed.
        registry = ArtifactRegistry(tmp_path / "registry")
        artifact = registry.publish(a_draft(tmp_path, b"weights-good"), SPEC)
        victim = artifact.path / "model.safetensors"
        victim.chmod(0o644)
        victim.write_bytes(CORRUPTION)

        with pytest.raises(ArtifactError, match="content hash mismatch"):
            registry.get(artifact.artifact_id)


class TestInsideACycle:
    def test_a_bad_copy_fails_the_cycle_and_restores_serving(
        self,
        tmp_path: Path,
        engine: SimulatedEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        simulation = build_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([passing_gate_result()]),
        )
        corrupt_the_copy(monkeypatch)

        result = simulation.run_cycle(run_id="corrupt")

        # Fail-closed: a candidate whose bytes cannot be trusted is never
        # benchmarked, never published and never activated...
        assert result.outcome is CycleOutcome.FAILED
        assert result.error is not None
        assert "changed while publishing" in result.error
        assert result.artifact_id is None
        assert list((tmp_path / "registry" / "artifacts").iterdir()) == []
        assert simulation.artifacts.active_pointer() is None
        # ...and the engine is put back exactly as any other failed cycle.
        assert simulation.state.state is TunerState.READY
        assert simulation.runtime.serving == "sim/stock-draft"
        assert engine.sleeping is False

    def test_a_bad_copy_never_dislodges_a_healthy_incumbent(
        self,
        tmp_path: Path,
        engine: SimulatedEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        simulation = build_simulation(
            tmp_path,
            engine=engine,
            gate=ScriptedGate([passing_gate_result(), passing_gate_result()]),
        )
        promoted = simulation.run_cycle(run_id="good", payload=b"weights-good")
        assert promoted.outcome is CycleOutcome.PROMOTED
        assert promoted.artifact_id is not None
        pointer_bytes = simulation.artifacts.active_path.read_bytes()

        corrupt_the_copy(monkeypatch)
        result = simulation.run_cycle(run_id="corrupt", payload=b"weights-corrupt")

        assert result.outcome is CycleOutcome.FAILED
        # Byte for byte: a publish that failed must not touch the pointer.
        assert simulation.artifacts.active_path.read_bytes() == pointer_bytes
        assert simulation.active_artifact_id == promoted.artifact_id
        # And the incumbent is still readable, i.e. its own bytes survived the
        # neighbouring failure.
        assert simulation.artifacts.get(promoted.artifact_id).path.is_dir()
        assert Path(str(simulation.runtime.serving)).name == promoted.artifact_id


class TestTheTrainedWeightsReachThePublication:
    """``hash_directory`` proves the copy is faithful, not that it is *ours*.

    The registry can only say "this tree is the tree I was handed".  Whether
    the tree it was handed holds the weights the cycle trained is a separate
    claim, and until the adapter fingerprinted them nothing in the pipeline
    could make it -- the checkpoint_best -> materialize -> publish path was
    believed correct because it reads correct, never because it was checked.
    """

    def _adapter(self, stock: Path) -> Eagle3Adapter:
        adapter = object.__new__(Eagle3Adapter)
        adapter.config = Eagle3Config(
            verifier_model="sim/verifier-8b",
            draft_model=str(stock),
            from_pretrained=str(stock),
            mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
        )
        return adapter

    def test_a_tree_that_is_not_the_trained_draft_is_refused(
        self, tmp_path: Path
    ) -> None:
        stock = tmp_path / "stock"
        write_draft_weights(stock, seed=1)
        trained = tmp_path / "trained"
        write_draft_weights(trained, seed=2)
        someone_elses = tmp_path / "someone-elses"
        write_draft_weights(someone_elses, seed=3)

        adapter = self._adapter(stock)
        adapter._record_draft_weights(trained)
        guard = _PublishedWeightGuard()
        guard.bind(adapter)
        registry = ArtifactRegistry(tmp_path / "registry", before_publish=guard)

        with pytest.raises(DraftWeightsError, match="do not match"):
            registry.publish(someone_elses, SPEC)

        assert list((tmp_path / "registry" / "artifacts").iterdir()) == []
        assert registry.active_pointer() is None

    def test_the_trained_draft_publishes_and_carries_its_fingerprint(
        self, tmp_path: Path
    ) -> None:
        stock = tmp_path / "stock"
        write_draft_weights(stock, seed=1)
        trained = tmp_path / "trained"
        write_draft_weights(trained, seed=2)

        adapter = self._adapter(stock)
        adapter._record_draft_weights(trained)
        guard = _PublishedWeightGuard()
        guard.bind(adapter)
        registry = ArtifactRegistry(tmp_path / "registry", before_publish=guard)

        artifact = registry.publish(
            trained,
            ArtifactSpec(
                verifier_model="sim/verifier-8b",
                draft_model="sim/draft-eagle3",
                base_draft=str(stock),
                trace_hash="sim-trace-hash",
                training_params=adapter.describe().training_params,
            ),
        )

        recorded = artifact.manifest.training_params["draft_weight_fingerprint"]
        assert recorded == weight_fingerprint(artifact.path)
        assert (
            artifact.manifest.training_params["draft_weight_baseline_fingerprint"]
            == weight_fingerprint(stock)
        )

    def test_an_unbound_guard_is_an_error_not_a_skipped_check(
        self, tmp_path: Path
    ) -> None:
        trained = tmp_path / "trained"
        write_draft_weights(trained, seed=2)
        registry = ArtifactRegistry(
            tmp_path / "registry", before_publish=_PublishedWeightGuard()
        )

        with pytest.raises(ProductionTuningError, match="never bound"):
            registry.publish(trained, SPEC)
