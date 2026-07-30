"""CPU-only tests for production idle-tuner composition."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import speedlm.tuner.composition as composition
from speedlm.config import IdleTuningConfig, SpeedLMConfig
from speedlm.profiles import ModelProfile
from speedlm.storage import ensure_layout
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.composition import (
    ProductionTuningError,
    build_tuning_launch_plan,
    create_production_tuner,
)


def _profile() -> ModelProfile:
    return ModelProfile(
        name="private-eagle-profile",
        verifier_model="acme/verifier-that-is-not-a-builtin",
        draft_model="acme/warm-start-that-is-not-a-builtin",
        speculative_method="eagle3",
        num_speculative_tokens=7,
        target_layer_ids=(1, 8, 19),
        chat_template_kind="auto",
        max_seq_len=12_345,
    )


def _config(tmp_path: Path, profile: ModelProfile) -> SpeedLMConfig:
    repo = tmp_path / "speculators"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "prepare_data.py",
        "launch_vllm.py",
        "data_generation_offline.py",
        "train.py",
    ):
        (scripts / name).write_text("# test fixture\n", encoding="utf-8")
    python = tmp_path / "training-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    validator = tmp_path / "validate.py"
    validator.write_text("# test fixture\n", encoding="utf-8")
    return SpeedLMConfig(
        model=profile.verifier_model,
        tuning_enabled=True,
        tuning=IdleTuningConfig(
            speculators_repo=str(repo),
            training_python=str(python),
            prepared_validator_script=str(validator),
            sequence_length=20_000,
            learning_rate=4e-6,
            epochs=2,
            concurrency=3,
            training_port=9_123,
            scratch_quota_bytes=100_000,
        ),
    )


def _speculative_config(argv: list[str]) -> dict[str, object]:
    option_index = argv.index("--speculative-config")
    value = json.loads(argv[option_index + 1])
    assert isinstance(value, dict)
    return value


def test_launch_plan_uses_only_profile_derived_speculative_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=("--dtype", "bfloat16", "--enable-sleep-mode"),
        child_port=8_765,
        home=tmp_path / "home",
    )
    argv = plan.argv_factory("acme/candidate-draft")

    assert plan.profile is profile
    assert plan.active_draft == profile.draft_model
    assert argv[:3] == ["vllm", "serve", profile.verifier_model]
    assert argv.count("--enable-sleep-mode") == 1
    assert "--dtype" in argv
    assert _speculative_config(argv) == {
        "method": profile.speculative_method,
        "model": "acme/candidate-draft",
        "num_speculative_tokens": profile.num_speculative_tokens,
    }
    rendered = "\0".join(argv)
    assert "openai/gpt-oss-20b" not in rendered
    assert "RedHatAI/gpt-oss-20b-speculator.eagle3" not in rendered


def test_launch_plan_selects_the_promoted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    home = tmp_path / "home"
    registry = ArtifactRegistry(ensure_layout(home).runs_dir, clock=lambda: 1.0)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.safetensors").write_bytes(b"trained")
    artifact = registry.publish(
        candidate,
        ArtifactSpec(
            verifier_model=profile.verifier_model,
            draft_model=profile.draft_model or "missing",
            base_draft=profile.draft_model or "missing",
            trace_hash="trace-hash",
            training_params={"epochs": 2},
        ),
    )
    registry.promote(artifact.artifact_id, gate_passed=True)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=home,
    )

    assert plan.active_draft == artifact.path
    assert _speculative_config(plan.argv_factory(plan.active_draft))["model"] == str(
        artifact.path
    )


@pytest.mark.parametrize(
    "passthrough",
    (
        ("--speculative-config", "{}"),
        ("--speculative-config={}",),
    ),
)
def test_launch_plan_rejects_user_owned_speculative_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passthrough: tuple[str, ...],
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    with pytest.raises(ProductionTuningError, match="owned by idle tuning"):
        build_tuning_launch_plan(
            config,
            passthrough=passthrough,
            child_port=8_765,
            home=tmp_path / "home",
        )


def test_missing_training_dependencies_fail_before_layout_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = SpeedLMConfig(
        model=profile.verifier_model,
        tuning_enabled=True,
        tuning=IdleTuningConfig(
            speculators_repo=str(tmp_path / "missing-repo"),
            training_python=str(tmp_path / "missing-python"),
            prepared_validator_script=str(tmp_path / "missing-validator.py"),
        ),
    )
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    def unexpected_layout(_home: Path) -> None:
        raise AssertionError("layout creation must follow dependency validation")

    monkeypatch.setattr(composition, "ensure_layout", unexpected_layout)

    with pytest.raises(ProductionTuningError, match="dependencies are missing") as error:
        build_tuning_launch_plan(
            config,
            passthrough=(),
            child_port=8_765,
            home=tmp_path / "home",
        )

    assert "prepare_data.py" in str(error.value)
    assert "missing-python" in str(error.value)
    assert "missing-validator.py" in str(error.value)
    assert not (tmp_path / "home").exists()


def test_create_production_tuner_assembles_profile_bound_collaborators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    captured: dict[str, Any] = {}
    state = object()
    artifacts = object()
    split = SimpleNamespace(
        suite_dir=tmp_path / "held-out",
        training_context_hashes=frozenset({"train-hash"}),
    )
    backend = object()
    runtime = object()
    gate = object()
    service = object()

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: state)
    monkeypatch.setattr(composition, "ArtifactRegistry", lambda _path: artifacts)

    def build_split(_traces: object, **kwargs: object) -> object:
        captured["split"] = kwargs
        return split

    def build_pipeline(**kwargs: object) -> object:
        captured["pipeline"] = kwargs
        return SimpleNamespace()

    def build_backend(_pipeline: object, **kwargs: object) -> object:
        captured["backend"] = kwargs
        return backend

    def build_runtime(**kwargs: object) -> object:
        captured["runtime"] = kwargs
        return runtime

    def build_gate(**kwargs: object) -> object:
        captured["gate"] = kwargs
        return gate

    def build_service(_config: SpeedLMConfig, **kwargs: object) -> object:
        captured["service"] = kwargs
        return service

    monkeypatch.setattr(composition, "HeldOutTraceSnapshotLeaser", build_split)
    monkeypatch.setattr(composition, "SpeculatorsPipelineConfig", build_pipeline)
    monkeypatch.setattr(
        composition,
        "Eagle3Backend",
        SimpleNamespace(from_speculators=build_backend),
    )
    monkeypatch.setattr(composition, "RuntimeController", build_runtime)
    monkeypatch.setattr(composition, "BenchmarkGateRunner", build_gate)
    monkeypatch.setattr(composition, "create_tuner_service", build_service)

    activity = object()
    admission = object()
    traces = object()
    capture = object()
    process = object()
    http = object()
    loop = asyncio.new_event_loop()
    try:
        result = create_production_tuner(
            config,
            profile=profile,
            active_draft="acme/active-draft",
            activity=activity,  # type: ignore[arg-type]
            admission=admission,  # type: ignore[arg-type]
            traces=traces,  # type: ignore[arg-type]
            capture=capture,  # type: ignore[arg-type]
            process=process,  # type: ignore[arg-type]
            http=http,  # type: ignore[arg-type]
            child_url="http://127.0.0.1:8765",
            loop=loop,
            home=tmp_path / "home",
        )
    finally:
        loop.close()

    assert result is service
    assert captured["split"] == {
        "held_out_fraction": config.tuning.held_out_fraction,
        "scratch_quota_bytes": config.tuning.scratch_quota_bytes,
    }
    pipeline = captured["pipeline"]
    assert pipeline["verifier_model"] == profile.verifier_model
    assert pipeline["warm_start_model"] == profile.draft_model
    assert pipeline["target_layer_ids"] == profile.target_layer_ids
    assert pipeline["sequence_length"] == profile.max_seq_len
    assert pipeline["learning_rate"] == config.tuning.learning_rate
    assert captured["backend"] == {"trace_leaser": split}
    assert captured["runtime"]["active_draft"] == "acme/active-draft"
    assert captured["gate"]["stock_draft"] == "acme/active-draft"
    assert captured["gate"]["suite_dir"]() == split.suite_dir
    assert captured["gate"]["training_context_hashes"]() == frozenset({"train-hash"})
    assert captured["service"]["backend"] is backend
    assert captured["service"]["gate"] is gate
    assert captured["service"]["runtime"] is runtime
    assert captured["service"]["state"] is state
    assert captured["service"]["artifacts"] is artifacts
    assert captured["service"]["enabled"] is True
