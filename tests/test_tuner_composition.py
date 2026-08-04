"""CPU-only tests for production idle-tuner composition."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import speedlm.tuner.composition as composition
from speedlm.config import (
    REFERENCE_LEARNING_RATE,
    IdleTuningConfig,
    SpeedLMConfig,
)
from speedlm.gate.decide import (
    CUDA_GRAPH_EXECUTION_MODE,
    EAGER_EXECUTION_MODE,
    EngineExecution,
)
from speedlm.gateway.control import ControlAborted
from speedlm.gateway.vllm_http import VLLMDraftSwapClient
from speedlm.profiles import AuxLayerError, ContextWindowError, ModelProfile
from speedlm.storage import ensure_layout
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.composition import (
    DRAFT_SWAP_WORKER_EXTENSION_CLS,
    WORKER_EXTENSION_OPTION,
    ProductionTuningError,
    build_tuning_launch_plan,
    create_production_tuner,
    resolve_verifier_revision,
)


@dataclass(frozen=True)
class _FakeBackendConfig:
    """Just enough of ``Eagle3Config`` for the manifest-provenance merge.

    ``create_production_tuner`` folds the resolved context window into the
    backend's ``training_params`` -- that mapping *is* the artifact manifest's
    provenance -- so a double that has no ``config`` is not modelling the
    collaborator it stands in for.
    """

    training_params: dict[str, object] = field(default_factory=dict)


class _FakeBackend:
    def __init__(self) -> None:
        self.config = _FakeBackendConfig()


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
            # The Speculators reference rate.  At the old 4e-6 this fixture
            # exercised a value that made training a measured no-op, and it
            # could not have caught the second copy of the 1e-5 bound in the
            # Eagle-3 backend that rejected the reference rate outright.
            learning_rate=REFERENCE_LEARNING_RATE,
            epochs=2,
            extraction_concurrency=3,
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


def test_the_launch_plan_reads_the_engine_regime_off_the_argv_it_will_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded regime must be the served one, not a restatement of config.

    ``build_tuning_launch_plan`` keeps editing ``base_passthrough`` after the
    operator's flags are read -- it appends ``--served-model-name`` and, under
    the ``align`` policy, ``--max-model-len`` -- so only the finished argv is
    the truth about what vLLM was told.  This asserts the plan's regime against
    that finished argv rather than against the passthrough it started from.
    """
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(
            "--enforce-eager",
            "--max-num-seqs",
            "32",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
        ),
        child_port=8_765,
        home=tmp_path / "home",
    )

    assert plan.engine_execution.execution_mode == EAGER_EXECUTION_MODE
    assert plan.engine_execution.enforce_eager is True
    assert plan.engine_execution.max_num_seqs == 32
    assert plan.engine_execution.enable_prefix_caching is False
    assert plan.engine_execution.enable_chunked_prefill is True
    # The load-bearing claim: this is what the supervisor execs, byte for byte.
    # ``cli._run_tuned_vllm_gateway`` hands ``plan.argv_factory`` straight to
    # ``VLLMSupervisor``, which calls it with the draft and passes the result
    # to ``VLLMProcess``.
    assert plan.engine_execution == EngineExecution.from_argv(
        plan.argv_factory(plan.active_draft)
    )


def test_a_launch_plan_told_no_execution_flags_does_not_invent_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent toggles stay ``None``; vLLM's own defaults are version-dependent."""
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )

    # ``--enforce-eager`` is a bare store-true flag, so its absence really does
    # mean graphs were captured; the other three genuinely were not observed.
    assert plan.engine_execution.execution_mode == CUDA_GRAPH_EXECUTION_MODE
    assert plan.engine_execution.enable_prefix_caching is None
    assert plan.engine_execution.enable_chunked_prefill is None
    assert plan.engine_execution.max_num_seqs is None


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
    # A registry that answers ``active()``: the gate is now handed a
    # resolver that consults it, not a value captured at composition time.
    active_artifact: list[Any] = [None]
    artifacts = SimpleNamespace(active=lambda: active_artifact[0])
    split = SimpleNamespace(
        suite_dir=tmp_path / "held-out",
        training_context_hashes=frozenset({"train-hash"}),
    )
    backend = _FakeBackend()
    runtime = object()
    gate = object()
    service = object()

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: state)
    monkeypatch.setattr(
        composition, "ArtifactRegistry", lambda _path, **_kwargs: artifacts
    )

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
    # Training depth comes from the serving depth, not from Speculators'
    # default of 3.  Threading this is what stops a profile serving a chain
    # deeper than the one its head was fitted on.
    assert pipeline["num_speculative_steps"] == profile.num_speculative_tokens
    assert pipeline["learning_rate"] == config.tuning.learning_rate
    assert captured["backend"]["trace_leaser"] is split
    # The training-side twin of ``stock_draft`` below, and resolved for the same
    # reason: frozen at composition time it names the profile's stock speculator
    # forever, so every cycle re-runs the same one-shot fine-tune of the
    # original head and discards the previous cycle's promotion.
    warm_start_resolver = captured["backend"]["warm_start_resolver"]
    assert callable(warm_start_resolver)
    assert warm_start_resolver() == profile.draft_model
    assert set(captured["backend"]) == {"trace_leaser", "warm_start_resolver"}
    assert captured["runtime"]["active_draft"] == "acme/active-draft"
    # The post-sleep memory precondition must demand exactly what the
    # hidden-state engine will be launched with, or it is decorative.
    assert captured["runtime"]["gpu_memory"].required_fraction == 0.80
    # A knob that never leaves config is not a knob: this was pinned at the
    # controller's own default with no way to reach it from a config file.
    assert (
        captured["runtime"]["restore_fast_path_timeout_seconds"]
        == config.tuning.restore_fast_path_timeout_seconds
    )
    # The stock arm's baseline is resolved when the benchmark runs, not frozen
    # here: with nothing promoted it is the startup draft, and after a
    # promotion it has to follow the registry -- otherwise every cycle after
    # the first reports gain over the original head instead of over what is
    # actually serving, and the gate is the only safeguard there is.
    stock_draft = captured["gate"]["stock_draft"]
    assert callable(stock_draft)
    assert stock_draft() == "acme/active-draft"
    active_artifact[0] = SimpleNamespace(path=Path("/runs/artifacts/promoted"))
    assert stock_draft() == Path("/runs/artifacts/promoted")
    # The gate replay was serialized until this reached it; a default that
    # never leaves composition is the same bug in a different place.
    assert captured["gate"]["repeats"] == config.tuning.benchmark_repeats
    # Same bug, same place: warmup was the runner's own constructor default and
    # composition never passed it, so the one knob ``benchmark_repeats``' own
    # analysis names as the honest direction to move was unreachable from a
    # config file.
    assert captured["gate"]["warmup_repeats"] == config.tuning.warmup_repeats
    assert (
        captured["gate"]["replay_concurrency"]
        == config.tuning.benchmark_concurrency
    )
    # ``extraction_concurrency`` drives the training-side extraction engine and
    # must NOT be what the gate replays at.  While it was named ``concurrency``
    # job 369006 set it to 4, recorded that in its config, and the gate replayed
    # at 8; the fixture keeps the two values distinct so a rewiring shows up
    # here rather than in a post-hoc analysis.
    assert config.tuning.extraction_concurrency != config.tuning.benchmark_concurrency
    assert (
        captured["gate"]["replay_concurrency"]
        != config.tuning.extraction_concurrency
    )
    assert captured["pipeline"]["concurrency"] == config.tuning.extraction_concurrency
    # ...and the extraction engine must be allowed to *schedule* what the
    # extraction client sends it.  Unwired, ``max_num_seqs`` stayed at the
    # backend dataclass default of 1 and every prefill serialized behind the
    # previous one no matter what ``concurrency`` was set to.
    assert captured["pipeline"]["max_num_seqs"] == config.tuning.extraction_concurrency
    assert (
        captured["gate"]["candidate_arm_first"]
        == config.tuning.benchmark_candidate_arm_first
    )
    assert (
        captured["gate"]["benchmark_max_tokens"]
        == config.tuning.benchmark_max_tokens
    )
    assert captured["gate"]["suite_dir"]() == split.suite_dir
    assert captured["gate"]["training_context_hashes"]() == frozenset({"train-hash"})
    assert captured["service"]["backend"] is backend
    assert captured["service"]["gate"] is gate
    assert captured["service"]["runtime"] is runtime
    assert captured["service"]["state"] is state
    assert captured["service"]["artifacts"] is artifacts
    assert captured["service"]["enabled"] is True
    # Nothing was told, so nothing is claimed.  ``BenchmarkGateRunner`` turns
    # ``None`` into ``engine_execution_mode: unrecorded`` rather than into an
    # assumed eager or graphed run.
    assert captured["gate"]["engine_execution"] is None


def _gate_kwargs_from_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble a production tuner against stubs and return the gate's kwargs."""
    profile = _profile()
    config = _config(tmp_path, profile)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: object())
    monkeypatch.setattr(
        composition,
        "ArtifactRegistry",
        lambda _path, **_kwargs: SimpleNamespace(active=lambda: None),
    )
    monkeypatch.setattr(
        composition,
        "HeldOutTraceSnapshotLeaser",
        lambda _traces, **_kwargs: SimpleNamespace(
            suite_dir=tmp_path / "held-out",
            training_context_hashes=frozenset(),
        ),
    )
    monkeypatch.setattr(
        composition,
        "SpeculatorsPipelineConfig",
        lambda **_kwargs: SimpleNamespace(gpu_memory_utilization=0.80),
    )
    monkeypatch.setattr(
        composition,
        "Eagle3Backend",
        SimpleNamespace(from_speculators=lambda _pipeline, **_kwargs: _FakeBackend()),
    )
    monkeypatch.setattr(composition, "RuntimeController", lambda **_kwargs: object())

    def build_gate(**kwargs: object) -> object:
        captured["gate"] = kwargs
        return object()

    monkeypatch.setattr(composition, "BenchmarkGateRunner", build_gate)
    monkeypatch.setattr(
        composition,
        "create_tuner_service",
        lambda _config, **_kwargs: object(),
    )

    loop = asyncio.new_event_loop()
    try:
        create_production_tuner(
            config,
            profile=profile,
            active_draft="acme/active-draft",
            activity=object(),  # type: ignore[arg-type]
            admission=object(),  # type: ignore[arg-type]
            traces=object(),  # type: ignore[arg-type]
            capture=object(),  # type: ignore[arg-type]
            process=object(),  # type: ignore[arg-type]
            http=object(),  # type: ignore[arg-type]
            child_url="http://127.0.0.1:8765",
            loop=loop,
            home=tmp_path / "home",
            **extra,
        )
    finally:
        loop.close()

    gate_kwargs = captured["gate"]
    assert isinstance(gate_kwargs, dict)
    return gate_kwargs


def test_the_gate_is_handed_the_engine_regime_it_will_measure_under(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EngineExecution`` reached the runner from nothing before this.

    ``BenchmarkGateRunner`` has accepted ``engine_execution`` and
    ``decide.EngineExecution.from_argv`` has existed, but no production caller
    supplied either -- so every live ``decision.json`` recorded
    ``engine_execution_mode: unrecorded`` while the engine was in fact running
    ``--enforce-eager`` (job d993eee's vLLM banner says ``enforce_eager=True``).
    """
    execution = EngineExecution(
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=64,
    )

    gate_kwargs = _gate_kwargs_from_assembly(
        tmp_path, monkeypatch, engine_execution=execution
    )

    assert gate_kwargs["engine_execution"] is execution


def test_the_launch_plans_regime_survives_the_trip_to_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: operator flags -> launch argv -> plan -> gate constructor.

    Asserting the two halves separately would leave the join untested, and the
    join is where the wiring was missing.
    """
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)
    plan = build_tuning_launch_plan(
        config,
        passthrough=("--enforce-eager", "--max-num-seqs", "64"),
        child_port=8_765,
        home=tmp_path / "home",
    )

    # A directory of its own: the plan above already populated ``tmp_path``.
    gate_kwargs = _gate_kwargs_from_assembly(
        tmp_path / "assembly", monkeypatch, engine_execution=plan.engine_execution
    )

    recorded = gate_kwargs["engine_execution"]
    assert isinstance(recorded, EngineExecution)
    assert recorded.execution_mode == EAGER_EXECUTION_MODE
    assert recorded.max_num_seqs == 64
    assert recorded == EngineExecution.from_argv(plan.argv_factory(plan.active_draft))


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


def test_min_corpus_records_reaches_the_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: create_production_tuner should pass min_corpus_records."""
    profile = _profile()
    base_config = _config(tmp_path, profile)
    config = replace(
        base_config,
        tuning=replace(base_config.tuning, min_corpus_records=512, training_window_records=512),
    )
    captured: dict[str, Any] = {}
    state = object()
    artifacts = object()
    split = SimpleNamespace(
        suite_dir=tmp_path / "held-out",
        training_context_hashes=frozenset({"train-hash"}),
    )
    backend = _FakeBackend()
    runtime = object()
    gate = object()
    service = object()

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: state)
    monkeypatch.setattr(
        composition, "ArtifactRegistry", lambda _path, **_kwargs: artifacts
    )

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
        create_production_tuner(
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

    assert captured["service"]["min_corpus_records"] == 512


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


# --- draft hot-swap wiring ------------------------------------------------


def _hot_swap_config(tmp_path: Path, profile: ModelProfile) -> SpeedLMConfig:
    config = _config(tmp_path, profile)
    return replace(
        config,
        tuning=replace(config.tuning, draft_hot_swap_enabled=True),
    )


def test_the_hot_swap_flag_defaults_off_and_registers_no_worker_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    assert config.tuning.draft_hot_swap_enabled is False

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )

    assert WORKER_EXTENSION_OPTION not in plan.argv_factory("acme/candidate")


def test_the_hot_swap_flag_registers_the_combined_extension_by_dotted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _hot_swap_config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )
    argv = plan.argv_factory("acme/candidate")

    index = argv.index(WORKER_EXTENSION_OPTION)
    assert argv[index + 1] == DRAFT_SWAP_WORKER_EXTENSION_CLS
    # vLLM resolves the class with rsplit(".", 1); a colon form silently fails.
    assert ":" not in DRAFT_SWAP_WORKER_EXTENSION_CLS
    assert argv.count(WORKER_EXTENSION_OPTION) == 1


def test_the_combined_extension_dotted_path_actually_imports() -> None:
    module_name, _, class_name = DRAFT_SWAP_WORKER_EXTENSION_CLS.rpartition(".")
    module = importlib.import_module(module_name)

    assert getattr(module, class_name).__name__ == class_name


@pytest.mark.parametrize(
    "passthrough",
    [
        (WORKER_EXTENSION_OPTION, "speedlm.activation_capture.hook.X"),
        (f"{WORKER_EXTENSION_OPTION}=speedlm.activation_capture.hook.X",),
    ],
)
def test_two_worker_extension_classes_are_refused_before_any_engine_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passthrough: tuple[str, ...],
) -> None:
    profile = _profile()
    config = _hot_swap_config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    # vLLM accepts exactly one; discovering that at engine start would cost a
    # full launch per candidate and surface as an attribute-collision assert.
    with pytest.raises(ProductionTuningError, match="exactly one"):
        build_tuning_launch_plan(
            config,
            passthrough=passthrough,
            child_port=8_765,
            home=tmp_path / "home",
        )


def test_a_passthrough_worker_extension_is_allowed_while_hot_swap_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_args, **_kwargs: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(WORKER_EXTENSION_OPTION, "speedlm.activation_capture.hook.X"),
        child_port=8_765,
        home=tmp_path / "home",
    )
    argv = plan.argv_factory("acme/candidate")

    assert argv.count(WORKER_EXTENSION_OPTION) == 1
    assert DRAFT_SWAP_WORKER_EXTENSION_CLS not in argv


@pytest.mark.parametrize("enabled", [False, True])
def test_the_swap_client_is_wired_only_when_the_flag_is_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    if enabled:
        config = replace(
            config,
            tuning=replace(config.tuning, draft_hot_swap_enabled=True),
        )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: object())
    monkeypatch.setattr(
        composition, "ArtifactRegistry", lambda _path, **_kwargs: object()
    )
    monkeypatch.setattr(
        composition,
        "HeldOutTraceSnapshotLeaser",
        lambda _traces, **_kwargs: SimpleNamespace(
            suite_dir=tmp_path / "held-out",
            training_context_hashes=frozenset(),
        ),
    )
    monkeypatch.setattr(
        composition,
        "SpeculatorsPipelineConfig",
        lambda **_kwargs: SimpleNamespace(gpu_memory_utilization=0.80),
    )
    monkeypatch.setattr(
        composition,
        "Eagle3Backend",
        SimpleNamespace(from_speculators=lambda _pipeline, **_kwargs: _FakeBackend()),
    )

    def build_runtime(**kwargs: object) -> object:
        captured["runtime"] = kwargs
        return object()

    monkeypatch.setattr(composition, "RuntimeController", build_runtime)
    monkeypatch.setattr(composition, "BenchmarkGateRunner", lambda **_kwargs: object())
    monkeypatch.setattr(
        composition,
        "create_tuner_service",
        lambda _config, **_kwargs: object(),
    )

    http = object()
    loop = asyncio.new_event_loop()
    try:
        create_production_tuner(
            config,
            profile=profile,
            active_draft="acme/active-draft",
            activity=object(),  # type: ignore[arg-type]
            admission=object(),  # type: ignore[arg-type]
            traces=object(),  # type: ignore[arg-type]
            capture=object(),  # type: ignore[arg-type]
            process=object(),  # type: ignore[arg-type]
            http=http,  # type: ignore[arg-type]
            child_url="http://127.0.0.1:8765",
            loop=loop,
            home=tmp_path / "home",
        )
    finally:
        loop.close()

    swap = captured["runtime"]["draft_swap_http"]
    if not enabled:
        # Wiring a client without the worker extension registered would only
        # manufacture failures against an engine that has no RPC handler.
        assert swap is None
    else:
        assert isinstance(swap, VLLMDraftSwapClient)
        assert swap.control is http


@dataclass
class _RecordingProcess:
    restarts: list[object] = field(default_factory=list)

    def restart(self, draft: object, *, timeout_seconds: float) -> None:
        del timeout_seconds
        self.restarts.append(draft)


@dataclass
class _RecordingHTTP:
    ready_calls: int = 0

    def wait_ready(
        self,
        *,
        timeout_seconds: float,
        should_abort: object = None,
    ) -> None:
        del timeout_seconds, should_abort
        self.ready_calls += 1


@dataclass
class _FakeRuntimeView:
    """The two-method slice of ``RuntimeController`` the endpoint consults."""

    running: object
    notified: list[object] = field(default_factory=list)

    def matches_running_draft(self, draft: object) -> bool:
        return str(self.running) == str(draft)

    def note_external_restart(self, draft: object) -> None:
        self.notified.append(draft)
        self.running = draft


def _endpoint(runtime: object | None) -> tuple[object, _RecordingProcess, _RecordingHTTP]:
    process = _RecordingProcess()
    http = _RecordingHTTP()
    endpoint = composition._DraftEndpoint(
        url="http://127.0.0.1:8765",
        process=process,  # type: ignore[arg-type]
        http=http,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
    )
    return endpoint, process, http


def test_activating_the_draft_that_already_runs_skips_the_engine_restart() -> None:
    """The candidate arm must reuse the engine CANDIDATE_STARTING just built."""
    runtime = _FakeRuntimeView(running=Path("/artifacts/candidate"))
    endpoint, process, http = _endpoint(runtime)

    endpoint.activate(  # type: ignore[attr-defined]
        Path("/artifacts/candidate"),
        timeout_seconds=30.0,
        should_abort=lambda: False,
    )

    assert process.restarts == []
    # Readiness is still confirmed: the arm never measures a mute engine.
    assert http.ready_calls == 1
    assert runtime.notified == []


def test_activating_a_different_draft_restarts_and_tells_the_controller() -> None:
    runtime = _FakeRuntimeView(running=Path("/artifacts/candidate"))
    endpoint, process, http = _endpoint(runtime)

    endpoint.activate(  # type: ignore[attr-defined]
        "stock",
        timeout_seconds=30.0,
        should_abort=lambda: False,
    )

    assert process.restarts == ["stock"]
    assert http.ready_calls == 1
    # Without this the controller would still believe the candidate is running
    # and would skip the rollback restart it genuinely needs.
    assert runtime.notified == ["stock"]


def test_an_endpoint_without_a_controller_always_restarts() -> None:
    endpoint, process, _ = _endpoint(None)

    endpoint.activate(  # type: ignore[attr-defined]
        "stock",
        timeout_seconds=30.0,
        should_abort=lambda: False,
    )

    assert process.restarts == ["stock"]


def test_an_aborted_activation_never_touches_the_engine() -> None:
    runtime = _FakeRuntimeView(running="stock")
    endpoint, process, http = _endpoint(runtime)

    with pytest.raises(ControlAborted):
        endpoint.activate(  # type: ignore[attr-defined]
            "stock",
            timeout_seconds=30.0,
            should_abort=lambda: True,
        )

    assert process.restarts == []
    assert http.ready_calls == 0


# ---------------------------------------------------------------------------
# Warm-start resolution
# ---------------------------------------------------------------------------


def _publish(
    artifacts: ArtifactRegistry,
    source: Path,
    payload: bytes,
    base_draft: str,
) -> Any:
    source.mkdir(parents=True)
    (source / "weights.bin").write_bytes(payload)
    return artifacts.publish(
        source,
        ArtifactSpec(
            verifier_model="acme/verifier",
            draft_model="acme/stock",
            base_draft=base_draft,
            trace_hash=payload.hex(),
            training_params={},
        ),
    )


def test_warm_start_falls_back_to_stock_while_nothing_has_been_promoted(
    tmp_path: Path,
) -> None:
    """The first cycle of a fresh install, and any state with no incumbent."""
    artifacts = ArtifactRegistry(tmp_path / "runs")
    assert composition.warm_start_reference(artifacts, "acme/stock") == "acme/stock"


def test_warm_start_follows_the_registry_once_something_is_promoted(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactRegistry(tmp_path / "runs")
    first = _publish(artifacts, tmp_path / "one", b"one", "acme/stock")
    assert composition.warm_start_reference(artifacts, "acme/stock") == "acme/stock"
    artifacts.promote(first.artifact_id, gate_passed=True)
    assert composition.warm_start_reference(artifacts, "acme/stock") == str(first.path)


def test_the_chain_depth_is_read_from_ancestry_not_from_promotion_count(
    tmp_path: Path,
) -> None:
    """``base_draft``, not the pointer's ``history``.

    ``history`` counts promotions, including ones made with compounding off,
    so it says nothing about what a head was actually built on.
    """
    artifacts = ArtifactRegistry(tmp_path / "runs")
    first = _publish(artifacts, tmp_path / "one", b"one", "acme/stock")
    second = _publish(artifacts, tmp_path / "two", b"two", str(first.path))
    # Promoted, but trained from stock rather than from its predecessor: a
    # promotion that is not a link in the chain.
    third = _publish(artifacts, tmp_path / "three", b"three", "acme/stock")
    for artifact in (first, second, third):
        artifacts.promote(artifact.artifact_id, gate_passed=True)

    assert composition.promotion_chain_depth("acme/stock") == 0
    assert composition.promotion_chain_depth(first.path) == 1
    assert composition.promotion_chain_depth(second.path) == 2
    assert composition.promotion_chain_depth(third.path) == 1


def test_a_chain_at_its_bound_re_baselines_to_the_stock_drafter(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactRegistry(tmp_path / "runs")
    first = _publish(artifacts, tmp_path / "one", b"one", "acme/stock")
    artifacts.promote(first.artifact_id, gate_passed=True)

    # One deep: still inside a bound of two, already at a bound of one.
    assert (
        composition.warm_start_reference(artifacts, "acme/stock", max_chain_depth=2)
        == str(first.path)
    )
    assert (
        composition.warm_start_reference(artifacts, "acme/stock", max_chain_depth=1)
        == "acme/stock"
    )


def test_an_unreadable_ancestor_ends_the_walk_rather_than_spinning(
    tmp_path: Path,
) -> None:
    """A broken link biases the depth *down*, toward compounding.

    Failing the other way would silently re-baseline a healthy chain because a
    manifest could not be read, which is a worse outcome than under-counting:
    the bound is a precaution, the chain is the product.
    """
    artifacts = ArtifactRegistry(tmp_path / "runs")
    first = _publish(artifacts, tmp_path / "one", b"one", str(tmp_path / "vanished"))
    assert composition.promotion_chain_depth(first.path) == 1

    # A manifest naming itself is the degenerate cycle; it must terminate.
    self_referential = tmp_path / "loop"
    self_referential.mkdir()
    (self_referential / "manifest.json").write_text(
        json.dumps({"base_draft": str(self_referential)}),
        encoding="utf-8",
    )
    assert composition.promotion_chain_depth(self_referential) == 1


# ---------------------------------------------------------------------------
# Draft-chain depth threading
# ---------------------------------------------------------------------------


def test_declared_draft_depth_reads_a_cached_drafter_config(tmp_path: Path) -> None:
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps(
            {
                "speculators_model_type": "eagle3",
                "speculators_config": {
                    "algorithm": "eagle3",
                    "proposal_methods": [{"speculative_tokens": 3}],
                },
            }
        ),
        encoding="utf-8",
    )

    assert composition.declared_draft_depth(str(drafter)) == 3


@pytest.mark.parametrize(
    "contents",
    [None, "not json", json.dumps([1, 2]), json.dumps({})],
)
def test_declared_draft_depth_is_best_effort(
    tmp_path: Path, contents: str | None
) -> None:
    """Provenance, not a precondition: an unreadable drafter must not block."""
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    if contents is not None:
        (drafter / "config.json").write_text(contents, encoding="utf-8")

    assert composition.declared_draft_depth(str(drafter)) is None
    assert composition.declared_draft_depth("acme/not-in-the-cache") is None


# ---------------------------------------------------------------------------
# Drafter aux-layer arity, read from fc.weight
# ---------------------------------------------------------------------------

#: The two stock EAGLE-3 heads, and the ``fc.weight`` shape each one actually
#: publishes.  Measured off the local HF cache rather than asserted from the
#: config: gpt-oss-20b fuses 3 x 2880 into 2880 and Qwen3-8B fuses 3 x 4096
#: into 4096, so both imply an aux-layer arity of 3 -- which is the count of
#: target layer ids both profiles resolve today.  This is why wiring the real
#: value in changes no production value.
STOCK_FC_WEIGHT_SHAPES: dict[str, tuple[int, int]] = {
    "RedHatAI/gpt-oss-20b-speculator.eagle3": (2880, 8640),
    "RedHatAI/Qwen3-8B-speculator.eagle3": (4096, 12288),
}


def _write_bf16_safetensors(
    directory: Path, tensors: dict[str, tuple[int, ...]]
) -> Path:
    """Write a minimal, structurally valid bf16 safetensors shard."""
    directory.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        size = 2
        for extent in shape:
            size *= extent
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path = directory / "model.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))
    return path


def test_drafter_aux_count_is_read_from_the_fc_weight_shape(tmp_path: Path) -> None:
    """The arity is a property of the tensor, not of anything a config claims."""
    drafter = tmp_path / "drafter"
    _write_bf16_safetensors(drafter, {"fc.weight": (8, 32)})

    assert (
        composition.resolve_drafter_aux_count(
            "acme/drafter", snapshot_resolver=lambda _model: drafter
        )
        == 4
    )


def test_an_uncached_drafter_degrades_to_an_unchecked_arity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Hub id with no local snapshot must not stop the service starting."""
    with caplog.at_level("INFO", logger=composition.logger.name):
        assert (
            composition.resolve_drafter_aux_count(
                "acme/never-downloaded",
                snapshot_resolver=lambda _model: None,
                counter=lambda _path: pytest.fail("must not read absent weights"),
            )
            is None
        )

    assert "no local snapshot" in caplog.text
    assert "acme/never-downloaded" in caplog.text


def test_a_drafter_without_a_usable_fc_weight_degrades_to_an_unchecked_arity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``DraftWeightsError`` is provenance lost, not a startup failure."""
    drafter = tmp_path / "drafter"
    # Structurally valid safetensors, but no fusion projection at all.
    _write_bf16_safetensors(drafter, {"norm.weight": (8,)})

    with caplog.at_level("INFO", logger=composition.logger.name):
        assert (
            composition.resolve_drafter_aux_count(
                "acme/headless", snapshot_resolver=lambda _model: drafter
            )
            is None
        )

    assert "fc.weight" in caplog.text


@pytest.mark.parametrize("model", sorted(STOCK_FC_WEIGHT_SHAPES))
def test_both_stock_heads_imply_three_aux_layers(model: str) -> None:
    """The no-change claim, checked against the real cached snapshots.

    Skipped rather than failed when the head is not on this host: the point of
    the resolver is that an absent snapshot is survivable, so a test asserting
    against one would be asserting about the machine, not the code.
    """
    snapshot = composition.cached_snapshot_dir(model)
    if snapshot is None:
        pytest.skip(f"{model} is not in this host's HF cache")

    from speedlm.tuner.eagle3 import _collect_draft_tensors

    assert _collect_draft_tensors(snapshot)["fc.weight"].shape == (
        STOCK_FC_WEIGHT_SHAPES[model]
    )
    assert composition.resolve_drafter_aux_count(model) == 3


def _stub_composition_collaborators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, captured: dict[str, Any]
) -> None:
    """Replace everything ``create_production_tuner`` builds after the profile."""
    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: object())
    monkeypatch.setattr(
        composition,
        "ArtifactRegistry",
        lambda _path, **_kwargs: SimpleNamespace(active=lambda: None),
    )
    monkeypatch.setattr(
        composition,
        "HeldOutTraceSnapshotLeaser",
        lambda *_args, **_kwargs: SimpleNamespace(
            suite_dir=tmp_path / "held-out",
            training_context_hashes=frozenset(),
        ),
    )

    def build_pipeline(**kwargs: object) -> object:
        captured["pipeline"] = kwargs
        return SimpleNamespace(gpu_memory_utilization=0.80)

    monkeypatch.setattr(composition, "SpeculatorsPipelineConfig", build_pipeline)
    monkeypatch.setattr(
        composition,
        "Eagle3Backend",
        SimpleNamespace(from_speculators=lambda *_a, **_k: _FakeBackend()),
    )
    monkeypatch.setattr(composition, "RuntimeController", lambda **_k: object())
    monkeypatch.setattr(composition, "BenchmarkGateRunner", lambda **_k: object())
    monkeypatch.setattr(
        composition, "create_tuner_service", lambda *_a, **_k: object()
    )


def _compose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: ModelProfile,
    captured: dict[str, Any],
) -> None:
    config = _config(tmp_path, profile)
    _stub_composition_collaborators(monkeypatch, tmp_path, captured)
    loop = asyncio.new_event_loop()
    try:
        create_production_tuner(
            config,
            profile=profile,
            active_draft="acme/active-draft",
            activity=object(),  # type: ignore[arg-type]
            admission=object(),  # type: ignore[arg-type]
            traces=object(),  # type: ignore[arg-type]
            capture=object(),  # type: ignore[arg-type]
            process=object(),  # type: ignore[arg-type]
            http=object(),  # type: ignore[arg-type]
            child_url="http://127.0.0.1:8765",
            loop=loop,
            home=tmp_path / "home",
        )
    finally:
        loop.close()


def test_a_drafter_arity_contradicting_the_configured_layer_ids_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the wiring.

    The profile pins three target layer ids; the head it would warm-start from
    fuses *four* aux hidden states.  Those cannot both be right, and the tensor
    is the one that cannot lie -- so composition must refuse rather than launch
    a cycle that dies as a shape error hundreds of GPU-seconds into extraction.

    Until the real count reached ``resolve_target_layer_ids`` this was
    unreachable: the only production call site passed ``drafter_aux_count=None``
    and the arity branch never executed.
    """
    drafter = tmp_path / "stock-drafter"
    # hidden 8, fused 32 -> four aux hidden states, one more than the pin.
    _write_bf16_safetensors(drafter, {"fc.weight": (8, 32)})
    monkeypatch.setattr(composition, "cached_snapshot_dir", lambda _model: drafter)
    profile = replace(_profile(), target_layer_ids=(2, 12, 21), num_hidden_layers=24)

    with pytest.raises(AuxLayerError) as error:
        _compose(tmp_path, monkeypatch, profile, {})

    assert "3 entries" in str(error.value)
    assert "4 aux layers" in str(error.value)


def test_an_agreeing_drafter_arity_leaves_the_configured_layer_ids_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-change case: three ids, a head that fuses three, pin preserved."""
    drafter = tmp_path / "stock-drafter"
    _write_bf16_safetensors(drafter, {"fc.weight": (8, 24)})
    monkeypatch.setattr(composition, "cached_snapshot_dir", lambda _model: drafter)
    profile = replace(_profile(), target_layer_ids=(2, 12, 21), num_hidden_layers=24)
    captured: dict[str, Any] = {}

    _compose(tmp_path, monkeypatch, profile, captured)

    assert captured["pipeline"]["target_layer_ids"] == (2, 12, 21)


def test_an_unpinned_profile_derives_its_layer_ids_from_the_drafter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no pin, the *head's* arity sets how many layers are extracted.

    Previously this fell through to the documented default of 3 whatever the
    head expected, which is only correct by coincidence.
    """
    drafter = tmp_path / "stock-drafter"
    _write_bf16_safetensors(drafter, {"fc.weight": (8, 32)})
    monkeypatch.setattr(composition, "cached_snapshot_dir", lambda _model: drafter)
    profile = replace(_profile(), target_layer_ids=None, num_hidden_layers=24)
    captured: dict[str, Any] = {}

    _compose(tmp_path, monkeypatch, profile, captured)

    assert len(captured["pipeline"]["target_layer_ids"]) == 4


def test_an_unreadable_drafter_preserves_the_historical_layer_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best effort means best effort: no snapshot, no check, service starts."""
    monkeypatch.setattr(composition, "cached_snapshot_dir", lambda _model: None)
    profile = replace(_profile(), target_layer_ids=None, num_hidden_layers=24)
    captured: dict[str, Any] = {}

    _compose(tmp_path, monkeypatch, profile, captured)

    assert len(captured["pipeline"]["target_layer_ids"]) == 3


# ---------------------------------------------------------------------------
# Context window: the position axis of train/serve alignment
# ---------------------------------------------------------------------------


def _compose_tuner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: SpeedLMConfig,
    profile: ModelProfile,
    **kwargs: Any,
) -> tuple[dict[str, Any], _FakeBackend]:
    """Run ``create_production_tuner`` against inert collaborators."""
    captured: dict[str, Any] = {}
    backend = _FakeBackend()

    monkeypatch.setattr(
        composition,
        "ensure_layout",
        lambda _home: SimpleNamespace(runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(composition, "TunerStateMachine", lambda _path: object())
    monkeypatch.setattr(
        composition,
        "ArtifactRegistry",
        lambda _path, **_kwargs: SimpleNamespace(active=lambda: None),
    )
    monkeypatch.setattr(
        composition,
        "HeldOutTraceSnapshotLeaser",
        lambda _traces, **_kwargs: SimpleNamespace(
            suite_dir=tmp_path / "held-out",
            training_context_hashes=frozenset(),
        ),
    )

    def build_pipeline(**pipeline_kwargs: object) -> object:
        captured["pipeline"] = pipeline_kwargs
        return SimpleNamespace(gpu_memory_utilization=0.80)

    monkeypatch.setattr(composition, "SpeculatorsPipelineConfig", build_pipeline)
    monkeypatch.setattr(
        composition,
        "Eagle3Backend",
        SimpleNamespace(from_speculators=lambda _pipeline, **_kwargs: backend),
    )
    monkeypatch.setattr(
        composition, "RuntimeController", lambda **_kwargs: object()
    )
    monkeypatch.setattr(composition, "BenchmarkGateRunner", lambda **_kwargs: object())
    monkeypatch.setattr(
        composition, "create_tuner_service", lambda _config, **_kwargs: object()
    )

    loop = asyncio.new_event_loop()
    try:
        create_production_tuner(
            config,
            profile=profile,
            active_draft="acme/active-draft",
            activity=object(),  # type: ignore[arg-type]
            admission=object(),  # type: ignore[arg-type]
            traces=object(),  # type: ignore[arg-type]
            capture=object(),  # type: ignore[arg-type]
            process=object(),  # type: ignore[arg-type]
            http=object(),  # type: ignore[arg-type]
            child_url="http://127.0.0.1:8765",
            loop=loop,
            home=tmp_path / "home",
            **kwargs,
        )
    finally:
        loop.close()
    return captured, backend


def test_the_training_window_is_the_lower_of_the_two_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch production actually takes: config below the profile.

    The sibling assertion above pins ``sequence_length == profile.max_seq_len``
    against a fixture whose ``tuning.sequence_length`` is *higher*, so ``min``
    can only ever pick the profile side and the production branch -- 16384
    against a 131072 profile -- is never exercised.
    """
    profile = _profile()
    base = _config(tmp_path, profile)
    config = replace(base, tuning=replace(base.tuning, sequence_length=4_096))
    assert config.tuning.sequence_length < profile.max_seq_len

    captured, _backend = _compose_tuner(tmp_path, monkeypatch, config, profile)

    assert captured["pipeline"]["sequence_length"] == 4_096


def test_the_context_window_lands_in_the_published_training_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate delta is only attributable if the artifact records the axis."""
    profile = _profile()
    config = _config(tmp_path, profile)

    _captured, backend = _compose_tuner(
        tmp_path,
        monkeypatch,
        config,
        profile,
        passthrough=("--max-model-len", "131072"),
    )

    assert backend.config.training_params == {
        "training_sequence_length": profile.max_seq_len,
        "serving_context_window": 131_072,
        "serving_context_window_source": "passthrough",
        "context_window_policy": "record",
        "context_window_ratio": 131_072 / profile.max_seq_len,
        "context_window_aligned": False,
    }


def test_a_launch_plan_alignment_is_preferred_over_re_deriving_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest must name the window the engine was really launched with."""
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_a, **_k: profile)
    plan = build_tuning_launch_plan(
        config,
        passthrough=("--max-model-len", "65536"),
        child_port=8_765,
        home=tmp_path / "plan-home",
    )

    _captured, backend = _compose_tuner(
        tmp_path,
        monkeypatch,
        config,
        profile,
        context_window=plan.context_window,
    )

    assert backend.config.training_params["serving_context_window"] == 65_536


def test_the_default_policy_does_not_cap_what_clients_may_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitting --max-model-len is a product change; it must never be silent."""
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.delenv(composition.CONTEXT_WINDOW_POLICY_ENV, raising=False)
    monkeypatch.setattr(composition, "resolve_profile", lambda *_a, **_k: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )

    assert "--max-model-len" not in plan.argv_factory("acme/candidate")
    assert plan.context_window is not None
    assert plan.context_window.aligned is False


def test_the_align_policy_makes_the_two_windows_agree_by_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setenv(composition.CONTEXT_WINDOW_POLICY_ENV, "align")
    monkeypatch.setattr(composition, "resolve_profile", lambda *_a, **_k: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=(),
        child_port=8_765,
        home=tmp_path / "home",
    )
    argv = plan.argv_factory("acme/candidate")

    assert argv[argv.index("--max-model-len") + 1] == str(profile.max_seq_len)
    assert plan.context_window is not None
    assert plan.context_window.aligned is True
    assert plan.context_window.serving_tokens == profile.max_seq_len


def test_the_align_policy_never_overrides_an_explicit_operator_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setenv(composition.CONTEXT_WINDOW_POLICY_ENV, "align")
    monkeypatch.setattr(composition, "resolve_profile", lambda *_a, **_k: profile)

    plan = build_tuning_launch_plan(
        config,
        passthrough=("--max-model-len", "2048"),
        child_port=8_765,
        home=tmp_path / "home",
    )
    argv = plan.argv_factory("acme/candidate")

    assert argv.count("--max-model-len") == 1
    assert argv[argv.index("--max-model-len") + 1] == "2048"


def test_the_strict_policy_fails_before_the_layout_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    config = _config(tmp_path, profile)
    monkeypatch.setenv(composition.CONTEXT_WINDOW_POLICY_ENV, "strict")
    monkeypatch.setattr(composition, "resolve_profile", lambda *_a, **_k: profile)

    with pytest.raises(ContextWindowError):
        build_tuning_launch_plan(
            config,
            passthrough=("--max-model-len", "131072"),
            child_port=8_765,
            home=tmp_path / "home",
        )

    assert not (tmp_path / "home").exists()


def test_an_unknown_policy_environment_value_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(composition.CONTEXT_WINDOW_POLICY_ENV, "strct")

    with pytest.raises(ContextWindowError):
        composition.context_window_policy()


def test_an_absent_policy_environment_value_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(composition.CONTEXT_WINDOW_POLICY_ENV, raising=False)

    assert composition.context_window_policy() == "record"
