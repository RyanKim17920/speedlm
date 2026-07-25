from __future__ import annotations

import json
from pathlib import Path

import pytest

from speedlm.tuner.artifacts import ArtifactError, ArtifactRegistry, ArtifactSpec


def _source(path: Path, content: str) -> Path:
    path.mkdir()
    (path / "config.json").write_text(content, encoding="utf-8")
    weights = path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(content.encode())
    return path


def _spec(trace_hash: str = "trace-1") -> ArtifactSpec:
    return ArtifactSpec(
        verifier_model="openai/gpt-oss-20b",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        base_draft="RedHatAI/gpt-oss-20b-speculator.eagle3",
        trace_hash=trace_hash,
        training_params={"steps": 8},
    )


def test_publish_is_content_addressed_and_promote_is_gate_guarded(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "registry", clock=lambda: 1.0)
    artifact = registry.publish(_source(tmp_path / "candidate", "one"), _spec())

    assert artifact.path.name == artifact.artifact_id
    assert artifact.manifest.trace_hash == "trace-1"
    assert registry.publish(tmp_path / "candidate", _spec()).artifact_id == artifact.artifact_id
    with pytest.raises(ArtifactError, match="gate pass"):
        registry.promote(artifact.artifact_id, gate_passed=False)
    assert not registry.active_path.exists()

    registry.promote(artifact.artifact_id, gate_passed=True)

    assert registry.active() is not None
    assert registry.active().artifact_id == artifact.artifact_id  # type: ignore[union-attr]


def test_rollback_restores_previous_active(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "registry", clock=lambda: 1.0)
    first = registry.publish(_source(tmp_path / "first", "one"), _spec("trace-1"))
    second = registry.publish(_source(tmp_path / "second", "two"), _spec("trace-2"))
    registry.promote(first.artifact_id, gate_passed=True)
    registry.promote(second.artifact_id, gate_passed=True)

    pointer = registry.rollback()

    assert pointer is not None
    assert pointer.artifact_id == first.artifact_id
    assert registry.active().artifact_id == first.artifact_id  # type: ignore[union-attr]


def test_atomic_publish_crash_never_exposes_partial_artifact(tmp_path: Path) -> None:
    def crash(_: Path) -> None:
        raise RuntimeError("simulated power loss before rename")

    registry = ArtifactRegistry(tmp_path / "registry", before_publish=crash)

    with pytest.raises(RuntimeError, match="power loss"):
        registry.publish(_source(tmp_path / "candidate", "data"), _spec())

    artifact_entries = list((tmp_path / "registry" / "artifacts").iterdir())
    assert artifact_entries == []
    assert not registry.active_path.exists()


def test_active_pointer_swap_is_valid_json(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "registry", clock=lambda: 7.0)
    artifact = registry.publish(_source(tmp_path / "candidate", "data"), _spec())
    registry.promote(artifact.artifact_id, gate_passed=True)

    pointer = json.loads(registry.active_path.read_text(encoding="utf-8"))

    assert pointer == {
        "artifact_id": artifact.artifact_id,
        "history": [],
        "updated_at": 7.0,
    }
