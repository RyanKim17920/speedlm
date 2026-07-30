"""CPU-only tests for production idle-tuner composition."""

from __future__ import annotations

import asyncio
import builtins
import json
from dataclasses import replace
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
    resolve_verifier_revision,
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
            verifier_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
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


def test_launch_plan_rejects_profile_for_a_different_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = replace(_config(tmp_path, profile), model="acme/different-verifier")
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    with pytest.raises(ProductionTuningError, match="does not match served model"):
        build_tuning_launch_plan(
            config,
            passthrough=(),
            child_port=8_765,
            home=tmp_path / "home",
        )


def test_launch_plan_accepts_cache_snapshot_path_for_profile_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    snapshot_model = (
        tmp_path
        / "hub"
        / "models--acme--verifier-that-is-not-a-builtin"
        / "snapshots"
        / "revision"
    )
    config = replace(_config(tmp_path, profile), model=str(snapshot_model))
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )

    assert plan.argv_factory(plan.active_draft)[2] == str(snapshot_model)


def test_launch_plan_rejects_active_artifact_for_a_different_model_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    home = tmp_path / "home"
    registry = ArtifactRegistry(ensure_layout(home).runs_dir, clock=lambda: 1.0)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.safetensors").write_bytes(b"stale")
    artifact = registry.publish(
        candidate,
        ArtifactSpec(
            verifier_model="acme/old-verifier",
            draft_model="acme/old-draft",
            base_draft="acme/old-draft",
            trace_hash="old-traces",
            training_params={},
        ),
    )
    registry.promote(artifact.artifact_id, gate_passed=True)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    with pytest.raises(ProductionTuningError, match="belongs to verifier/draft"):
        build_tuning_launch_plan(
            config,
            passthrough=(),
            child_port=8_765,
            home=home,
        )


def test_launch_plan_rejects_conflicting_served_model_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = replace(_config(tmp_path, profile), model_alias="expected-alias")
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    with pytest.raises(ProductionTuningError, match="must match configured model alias"):
        build_tuning_launch_plan(
            config,
            passthrough=("--served-model-name", "different-alias"),
            child_port=8_765,
            home=tmp_path / "home",
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
        return SimpleNamespace(gpu_memory_utilization=0.80)

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
        "training_window_records": config.tuning.training_window_records,
    }
    pipeline = captured["pipeline"]
    assert pipeline["verifier_model"] == profile.verifier_model
    # An unpinned verifier is an unreproducible cycle.
    assert pipeline["verifier_revision"] == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert pipeline["warm_start_model"] == profile.draft_model
    assert pipeline["target_layer_ids"] == profile.target_layer_ids
    assert pipeline["sequence_length"] == profile.max_seq_len
    assert pipeline["learning_rate"] == config.tuning.learning_rate
    assert captured["backend"] == {"trace_leaser": split}
    assert captured["runtime"]["active_draft"] == "acme/active-draft"
    # The post-sleep memory precondition must demand exactly what the
    # hidden-state engine will be launched with, or it is decorative.
    assert captured["runtime"]["gpu_memory"].required_fraction == 0.80
    assert captured["gate"]["stock_draft"] == "acme/active-draft"
    assert captured["gate"]["suite_dir"]() == split.suite_dir
    assert captured["gate"]["training_context_hashes"]() == frozenset({"train-hash"})
    assert captured["service"]["backend"] is backend
    assert captured["service"]["gate"] is gate
    assert captured["service"]["runtime"] is runtime
    assert captured["service"]["state"] is state
    assert captured["service"]["artifacts"] is artifacts
    assert captured["service"]["enabled"] is True


def test_configured_verifier_revision_is_pinned_verbatim() -> None:
    assert (
        resolve_verifier_revision(
            "acme/verifier",
            "6cee5e81ee83917806bbde320786a8fb61efebee",
            resolver=lambda _model: pytest.fail("must not query the Hub"),
        )
        == "6cee5e81ee83917806bbde320786a8fb61efebee"
    )


def test_an_unconfigured_verifier_revision_is_resolved_once() -> None:
    seen: list[str] = []

    def resolver(model: str) -> str:
        seen.append(model)
        return "6cee5e81ee83917806bbde320786a8fb61efebee"

    revision = resolve_verifier_revision("acme/verifier", None, resolver=resolver)

    assert revision == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert seen == ["acme/verifier"]


def test_an_unresolvable_verifier_revision_degrades_to_unpinned() -> None:
    """Pinning is provenance, not a precondition; it must not block startup."""

    def resolver(_model: str) -> str:
        raise ConnectionError("hub unreachable")

    assert resolve_verifier_revision("acme/verifier", None, resolver=resolver) is None


def test_an_empty_verifier_revision_degrades_to_unpinned() -> None:
    assert (
        resolve_verifier_revision("acme/verifier", None, resolver=lambda _model: "")
        is None
    )


def test_resolution_prefers_the_local_hub_cache_over_the_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = (
        tmp_path / "hub" / "models--openai--gpt-oss-20b" / "refs"
    )
    refs.mkdir(parents=True)
    (refs / "main").write_text(
        "6cee5e81ee83917806bbde320786a8fb61efebee\n", encoding="utf-8"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(
        composition,
        "_hub_revision",
        lambda _model: pytest.fail("must not query the Hub when the cache answers"),
    )

    assert (
        resolve_verifier_revision("openai/gpt-oss-20b", None)
        == "6cee5e81ee83917806bbde320786a8fb61efebee"
    )


def test_composition_starts_when_huggingface_hub_is_unimportable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heavy stack is installed out-of-band and may simply be absent.

    Reproduces job 368708: ``No module named 'huggingface_hub'`` on the
    startup path turned an undeclared dependency into a refusal to serve.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(
        composition.Path, "home", classmethod(lambda _cls: tmp_path / "nowhere")
    )
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)

    assert resolve_verifier_revision("openai/gpt-oss-20b", None) is None


def test_a_local_verifier_path_is_its_own_pin(tmp_path: Path) -> None:
    local = tmp_path / "verifier"
    local.mkdir()

    assert (
        resolve_verifier_revision(
            str(local),
            None,
            resolver=lambda _model: pytest.fail("must not query the Hub"),
        )
        is None
    )
