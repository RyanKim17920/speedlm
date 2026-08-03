from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import speedlm.profiles as profiles_module
from speedlm.config import SpeedLMConfig
from speedlm.profiles import (
    BUILTIN_PROFILES,
    DEFAULT_SPECULATIVE_TOKENS,
    GPT_OSS_EAGLE3_PROFILE,
    LLAMA_31_8B_EAGLE3_PROFILE,
    MAX_SPECULATIVE_TOKENS,
    QWEN_3_8B_EAGLE3_PROFILE,
    QWEN_35_9B_MTP_PROFILE,
    AuxLayerError,
    ContextWindowError,
    ModelProfile,
    ParserRegistry,
    ProfileError,
    ServingContextWindow,
    SpeculativeDepthError,
    default_aux_layers,
    discover_vllm_parser_registry,
    drafter_declared_speculative_tokens,
    load_profiles,
    resolve_model_parsers,
    resolve_profile,
    resolve_serving_context_window,
    resolve_speculative_tokens,
    resolve_target_layer_ids,
    spread_aux_layers,
    validate_training_context_window,
    validate_training_depth,
)


def _write_profile(home: Path, filename: str, data: dict[str, object]) -> Path:
    profiles_dir = home / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_default_aux_layers_uses_vllm_rule() -> None:
    assert default_aux_layers(24) == (2, 12, 21)
    assert default_aux_layers(32) == (2, 16, 29)
    assert default_aux_layers(48) == (2, 24, 45)


# -- spread_aux_layers tests --


def test_spread_aux_layers_reproduces_vllm_rule_k3() -> None:
    """k=3, n>=7 must match vLLM's (2, n//2, n-3) exactly."""
    assert spread_aux_layers(3, 24) == (2, 12, 21)
    assert spread_aux_layers(3, 32) == (2, 16, 29)
    assert spread_aux_layers(3, 48) == (2, 24, 45)
    assert spread_aux_layers(3, 7) == (2, 3, 4)


def test_spread_aux_layers_small_models_raise() -> None:
    """n < 7 with k=3 is provably invalid."""
    for n in (2, 4, 5, 6):
        with pytest.raises(ValueError):
            spread_aux_layers(3, n)


def test_spread_aux_layers_k4_produces_four_valid_indices() -> None:
    """k=4 should produce four strictly increasing indices."""
    result = spread_aux_layers(4, 24)
    assert len(result) == 4
    assert result == tuple(sorted(set(result)))
    assert all(0 <= idx < 24 for idx in result)


def test_spread_aux_layers_k1_returns_single() -> None:
    assert spread_aux_layers(1, 24) == (2,)


def test_spread_aux_layers_k2_returns_anchors() -> None:
    assert spread_aux_layers(2, 24) == (2, 21)


# -- resolve_target_layer_ids tests --


def test_resolve_explicit_profile_override_wins() -> None:
    explicit = (1, 8, 19)
    resolved = resolve_target_layer_ids(
        explicit=explicit,
        num_hidden_layers=24,
        drafter_aux_count=3,
    )
    assert resolved is explicit


def test_resolve_drafter_aux_count_drives_k() -> None:
    """When drafter_aux_count is known, use it for spread_aux_layers."""
    resolved = resolve_target_layer_ids(
        explicit=None,
        num_hidden_layers=24,
        drafter_aux_count=4,
    )
    assert len(resolved) == 4
    assert resolved == spread_aux_layers(4, 24)


def test_resolve_fallback_to_k3() -> None:
    """Without drafter_aux_count, default to k=3."""
    resolved = resolve_target_layer_ids(
        explicit=None,
        num_hidden_layers=32,
        drafter_aux_count=None,
    )
    assert resolved == (2, 16, 29)


def test_resolve_raises_without_num_hidden_layers() -> None:
    with pytest.raises(AuxLayerError, match="num_hidden_layers"):
        resolve_target_layer_ids(
            explicit=None,
            num_hidden_layers=None,
            drafter_aux_count=None,
        )


def test_resolve_validates_explicit_arity_mismatch() -> None:
    with pytest.raises(AuxLayerError, match="4 entries") as exc_info:
        resolve_target_layer_ids(
            explicit=(2, 12, 16, 20),
            num_hidden_layers=24,
            drafter_aux_count=3,
        )
    assert "drafter expects 3 aux layers" in str(exc_info.value)


def test_resolve_validates_explicit_out_of_range() -> None:
    with pytest.raises(AuxLayerError, match="out of range"):
        resolve_target_layer_ids(
            explicit=(2, 12, 99),
            num_hidden_layers=24,
            drafter_aux_count=3,
        )


def test_builtin_profiles_load_and_validate(tmp_path: Path) -> None:
    profiles = load_profiles(tmp_path)

    assert profiles == dict(BUILTIN_PROFILES)
    assert GPT_OSS_EAGLE3_PROFILE.target_layer_ids == (2, 12, 21)
    assert GPT_OSS_EAGLE3_PROFILE.chat_template_kind == "harmony"
    assert GPT_OSS_EAGLE3_PROFILE.tool_call_parser == "openai"
    assert GPT_OSS_EAGLE3_PROFILE.reasoning_parser == "openai_gptoss"
    assert LLAMA_31_8B_EAGLE3_PROFILE.speculative_method == "eagle3"
    assert LLAMA_31_8B_EAGLE3_PROFILE.target_layer_ids == (2, 16, 29)
    assert QWEN_35_9B_MTP_PROFILE.verifier_model == "Qwen/Qwen3.5-9B"
    assert QWEN_35_9B_MTP_PROFILE.draft_model is None
    assert QWEN_35_9B_MTP_PROFILE.speculative_method == "mtp"
    assert QWEN_35_9B_MTP_PROFILE.chat_template_kind == "chatml"
    assert QWEN_35_9B_MTP_PROFILE.tool_call_parser == "hermes"
    assert QWEN_35_9B_MTP_PROFILE.reasoning_parser is None
    assert QWEN_35_9B_MTP_PROFILE.speculative_config() == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    assert QWEN_3_8B_EAGLE3_PROFILE.verifier_model == "Qwen/Qwen3-8B"
    assert QWEN_3_8B_EAGLE3_PROFILE.target_layer_ids is None
    assert QWEN_3_8B_EAGLE3_PROFILE.num_hidden_layers == 36


def test_user_profile_loads_and_overrides_builtin(tmp_path: Path) -> None:
    replacement = GPT_OSS_EAGLE3_PROFILE.to_dict()
    replacement["num_speculative_tokens"] = 7
    _write_profile(tmp_path, "replacement.json", replacement)

    profiles = load_profiles(tmp_path)

    assert profiles[GPT_OSS_EAGLE3_PROFILE.name].num_speculative_tokens == 7
    assert len(profiles) == len(BUILTIN_PROFILES)


def test_profile_parser_fields_round_trip() -> None:
    data = GPT_OSS_EAGLE3_PROFILE.to_dict()

    profile = ModelProfile.from_dict(data)

    assert profile.tool_call_parser == "openai"
    assert profile.reasoning_parser == "openai_gptoss"
    assert profile.to_dict() == data


def test_parser_discovery_reads_lazy_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_module = SimpleNamespace(
        ToolParserManager=SimpleNamespace(
            tool_parsers={},
            lazy_parsers={
                "openai": ("vllm.tool_parsers.gptoss_tool_parser", "GptOssToolParser"),
                "hermes": ("vllm.tool_parsers.hermes_tool_parser", "Hermes2ProToolParser"),
            },
        )
    )
    reasoning_module = SimpleNamespace(
        ReasoningParserManager=SimpleNamespace(
            reasoning_parsers={},
            lazy_parsers={
                "openai_gptoss": (
                    "vllm.reasoning.gptoss_reasoning_parser",
                    "GptOssReasoningParser",
                ),
            },
        )
    )
    modules = {
        "vllm.tool_parsers": tool_module,
        "vllm.reasoning": reasoning_module,
    }
    monkeypatch.setattr(
        profiles_module.importlib,
        "import_module",
        modules.__getitem__,
    )

    registry = discover_vllm_parser_registry()

    assert registry.tool_parsers == ("hermes", "openai")
    assert registry.reasoning_parsers == ("openai_gptoss",)
    assert "gptoss_tool_parser" in registry.tool_metadata["openai"]
    assert registry.errors == ()


def test_parser_discovery_reads_separate_vllm_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "vllm-env"
    executable = environment / "bin" / "vllm"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    package = environment / "lib" / "python3.12" / "site-packages" / "vllm"
    tool_package = package / "tool_parsers"
    reasoning_package = package / "reasoning"
    tool_package.mkdir(parents=True)
    reasoning_package.mkdir(parents=True)
    (tool_package / "__init__.py").write_text(
        "_TOOL_PARSERS_TO_REGISTER = {\n"
        "    'format_x': ('format_x_parser', 'FormatXParser'),\n"
        "}\n",
        encoding="utf-8",
    )
    (reasoning_package / "__init__.py").write_text(
        "_REASONING_PARSERS_TO_REGISTER = {\n"
        "    'reason_x': ('reason_x_parser', 'ReasonXParser'),\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(profiles_module.shutil, "which", lambda _name: str(executable))

    registry = profiles_module._external_vllm_parser_registry()

    assert registry is not None
    assert registry.tool_parsers == ("format_x",)
    assert registry.reasoning_parsers == ("reason_x",)
    assert "FormatXParser" in registry.tool_metadata["format_x"]


def test_parser_discovery_merges_partial_local_and_external_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def local_registry(
        module_name: str,
        _manager_name: str,
        _eager_attribute: str,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        if module_name == "vllm.tool_parsers":
            return ("local_tool",), {"local_tool": "local_tool LocalToolParser"}
        raise ImportError("reasoning package unavailable")

    monkeypatch.setattr(profiles_module, "_manager_registry", local_registry)
    monkeypatch.setattr(
        profiles_module,
        "_external_vllm_parser_registry",
        lambda: ParserRegistry(
            tool_parsers=("external_tool",),
            reasoning_parsers=("external_reasoning",),
        ),
    )

    registry = discover_vllm_parser_registry()

    assert registry.tool_parsers == ("external_tool", "local_tool")
    assert registry.reasoning_parsers == ("external_reasoning",)
    assert registry.errors and "reasoning parsers" in registry.errors[0]


@pytest.mark.parametrize(
    ("model_type", "tool_parser", "reasoning_parser"),
    [
        ("gpt_oss", "openai", "openai_gptoss"),
        ("qwen2", None, None),
        ("qwen3", "qwen3_xml", "qwen3"),
        ("llama", None, None),
    ],
)
def test_model_type_resolves_against_discovered_parser_keys(
    model_type: str,
    tool_parser: str,
    reasoning_parser: str | None,
) -> None:
    registry = ParserRegistry(
        tool_parsers=(
            "hermes",
            "llama3_json",
            "llama4_json",
            "openai",
            "qwen3_coder",
            "qwen3_xml",
        ),
        reasoning_parsers=("openai_gptoss", "qwen3"),
        tool_metadata={
            "openai": "openai vllm.tool_parsers.gptoss_tool_parser.GptOssToolParser",
        },
    )

    resolution = resolve_model_parsers(
        "acme/not-a-builtin-profile",
        model_type=model_type,
        registry=registry,
    )

    assert resolution.tool_call_parser == tool_parser
    assert resolution.reasoning_parser == reasoning_parser
    expected_tool_source = "auto-detected" if tool_parser is not None else "none"
    assert resolution.tool_call_parser_source == expected_tool_source
    expected_reasoning_source = (
        "auto-detected" if reasoning_parser is not None else "none"
    )
    assert resolution.reasoning_parser_source == expected_reasoning_source


def test_non_builtin_local_model_uses_config_model_type(tmp_path: Path) -> None:
    model = tmp_path / "custom-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    registry = ParserRegistry(
        tool_parsers=("hermes", "qwen3_coder", "qwen3_xml"),
        reasoning_parsers=("qwen3",),
    )

    resolution = resolve_model_parsers(str(model), registry=registry)

    assert resolution.model_type == "qwen3_5"
    assert resolution.tool_call_parser == "qwen3_xml"
    assert resolution.tool_call_parser_source == "auto-detected"
    assert resolution.reasoning_parser == "qwen3"


def test_unknown_model_type_safely_resolves_no_parsers(tmp_path: Path) -> None:
    model = tmp_path / "custom-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "acme_transformer"}),
        encoding="utf-8",
    )

    resolution = resolve_model_parsers(
        str(model),
        registry=ParserRegistry(
            tool_parsers=("hermes", "openai"),
            reasoning_parsers=("openai_gptoss",),
        ),
    )

    assert resolution.tool_call_parser is None
    assert resolution.reasoning_parser is None
    assert resolution.tool_call_parser_source == "none"
    assert resolution.reasoning_parser_source == "none"


def test_profile_pins_override_auto_detection() -> None:
    resolution = resolve_model_parsers(
        GPT_OSS_EAGLE3_PROFILE.verifier_model,
        model_type="gpt_oss",
        profile=GPT_OSS_EAGLE3_PROFILE,
        registry=ParserRegistry(
            tool_parsers=("generic_gptoss",),
            reasoning_parsers=("generic_gptoss",),
        ),
    )

    assert resolution.tool_call_parser == "openai"
    assert resolution.reasoning_parser == "openai_gptoss"
    assert resolution.tool_call_parser_source == "profile-pinned"
    assert resolution.reasoning_parser_source == "profile-pinned"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"speculative_method": "magic"}, "speculative_method must be one of"),
        ({"num_speculative_tokens": 0}, "must be a positive integer"),
        ({"target_layer_ids": [2, 2]}, "unique non-negative integers"),
        ({"unexpected": True}, "unknown keys"),
    ],
)
def test_malformed_profile_rejected_with_clear_reason(
    tmp_path: Path,
    mutation: dict[str, object],
    reason: str,
) -> None:
    data = GPT_OSS_EAGLE3_PROFILE.to_dict()
    data.update(mutation)
    path = _write_profile(tmp_path, "malformed.json", data)

    with pytest.raises(ProfileError, match=reason) as exc_info:
        load_profiles(tmp_path)

    assert str(path) in str(exc_info.value)


def test_non_object_profile_is_rejected(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    path = profiles_dir / "array.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ProfileError, match="profile must be a JSON object"):
        load_profiles(tmp_path)


def test_resolver_explicit_profile_wins_over_name_match() -> None:
    config = SpeedLMConfig(
        model=QWEN_35_9B_MTP_PROFILE.verifier_model,
        profile=GPT_OSS_EAGLE3_PROFILE.name,
    )

    resolved = resolve_profile(
        config,
        served_model=LLAMA_31_8B_EAGLE3_PROFILE.verifier_model,
        profiles=BUILTIN_PROFILES,
    )

    assert resolved is GPT_OSS_EAGLE3_PROFILE


@pytest.mark.parametrize(
    "name_or_verifier",
    [
        QWEN_35_9B_MTP_PROFILE.name,
        QWEN_35_9B_MTP_PROFILE.verifier_model,
    ],
)
def test_resolver_exact_name_match(name_or_verifier: str) -> None:
    assert (
        resolve_profile(served_model=name_or_verifier, profiles=BUILTIN_PROFILES)
        is QWEN_35_9B_MTP_PROFILE
    )


def test_resolver_matches_huggingface_cache_snapshot_path() -> None:
    served_model = (
        "~/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-9B/snapshots/0123456789abcdef"
    )

    assert (
        resolve_profile(served_model=served_model, profiles=BUILTIN_PROFILES)
        is QWEN_35_9B_MTP_PROFILE
    )


@pytest.mark.parametrize(
    "served_model",
    [
        GPT_OSS_EAGLE3_PROFILE.name,
        GPT_OSS_EAGLE3_PROFILE.verifier_model,
        (
            "/opt/huggingface/hub/models--openai--gpt-oss-20b/"
            "snapshots/abcdef0123456789"
        ),
    ],
)
def test_gpt_oss_eagle3_resolution_regression(served_model: str) -> None:
    resolved = resolve_profile(served_model=served_model, profiles=BUILTIN_PROFILES)

    assert resolved is GPT_OSS_EAGLE3_PROFILE
    assert resolved.speculative_method == "eagle3"
    assert resolved.draft_model == "RedHatAI/gpt-oss-20b-speculator.eagle3"


def test_resolver_uses_config_model_when_served_model_is_absent() -> None:
    config = SpeedLMConfig(model=LLAMA_31_8B_EAGLE3_PROFILE.verifier_model)

    assert (
        resolve_profile(config, profiles=BUILTIN_PROFILES)
        is LLAMA_31_8B_EAGLE3_PROFILE
    )


def test_unknown_verifier_never_gets_a_guessed_draft() -> None:
    with pytest.raises(ProfileError, match="no model profile matches") as exc_info:
        resolve_profile(
            SpeedLMConfig(model="acme/unknown-verifier"),
            profiles=BUILTIN_PROFILES,
        )

    message = str(exc_info.value)
    assert "acme/unknown-verifier" in message
    assert "available profiles" in message.lower()
    assert GPT_OSS_EAGLE3_PROFILE.name in message


def test_unknown_explicit_profile_lists_available_profiles() -> None:
    with pytest.raises(ProfileError, match="unknown explicit profile") as exc_info:
        resolve_profile(
            {"model": "anything", "profile": "missing"},
            profiles=BUILTIN_PROFILES,
        )

    assert QWEN_35_9B_MTP_PROFILE.name in str(exc_info.value)


def test_ngram_profile_is_serve_benchmark_only() -> None:
    profile = ModelProfile(
        name="generic-ngram",
        verifier_model="acme/verifier",
        draft_model=None,
        speculative_method="ngram",
        num_speculative_tokens=5,
        target_layer_ids=None,
        chat_template_kind="auto",
        max_seq_len=4096,
    )

    assert profile.trainable is False
    assert profile.serve_benchmark_only is True


def test_ngram_json_cannot_claim_to_be_trainable() -> None:
    data = {
        "name": "generic-ngram",
        "verifier_model": "acme/verifier",
        "draft_model": None,
        "speculative_method": "ngram",
        "num_speculative_tokens": 5,
        "target_layer_ids": None,
        "chat_template_kind": "auto",
        "max_seq_len": 4096,
        "trainable": True,
    }

    with pytest.raises(ProfileError, match="trainable must be False"):
        ModelProfile.from_dict(data)


def test_qwen3_8b_eagle3_profile_fields() -> None:
    """Qwen3-8B EAGLE-3 profile reflects the real model/drafter configs."""
    assert QWEN_3_8B_EAGLE3_PROFILE.verifier_model == "Qwen/Qwen3-8B"
    assert QWEN_3_8B_EAGLE3_PROFILE.draft_model == "RedHatAI/Qwen3-8B-speculator.eagle3"
    assert QWEN_3_8B_EAGLE3_PROFILE.speculative_method == "eagle3"
    assert QWEN_3_8B_EAGLE3_PROFILE.num_speculative_tokens == 3
    assert QWEN_3_8B_EAGLE3_PROFILE.num_hidden_layers == 36
    assert QWEN_3_8B_EAGLE3_PROFILE.max_seq_len == 40_960
    assert QWEN_3_8B_EAGLE3_PROFILE.chat_template_kind == "chatml"
    assert QWEN_3_8B_EAGLE3_PROFILE.target_layer_ids is None
    assert QWEN_3_8B_EAGLE3_PROFILE.trainable is True


def test_qwen3_8b_aux_layers_derived_not_hardcoded() -> None:
    """Qwen3-8B aux layers come from derivation, not literal pinning.

    The profile has target_layer_ids=None, so resolve_target_layer_ids must
    derive them via default_aux_layers(36) -> (2, 18, 33).  We also verify
    that gpt-oss and llama resolutions are unaffected by the new profile.
    """
    # Qwen3-8B: 36 layers -> default_aux_layers(36) = (2, 18, 33)
    assert default_aux_layers(36) == (2, 18, 33)

    resolved = resolve_target_layer_ids(
        explicit=QWEN_3_8B_EAGLE3_PROFILE.target_layer_ids,
        num_hidden_layers=QWEN_3_8B_EAGLE3_PROFILE.num_hidden_layers,
        drafter_aux_count=None,
    )
    assert resolved == (2, 18, 33)

    # gpt-oss (24 layers) and Llama (32 layers) are unchanged
    gpt_oss_resolved = resolve_target_layer_ids(
        explicit=GPT_OSS_EAGLE3_PROFILE.target_layer_ids,
        num_hidden_layers=GPT_OSS_EAGLE3_PROFILE.num_hidden_layers,
        drafter_aux_count=None,
    )
    assert gpt_oss_resolved == (2, 12, 21)

    llama_resolved = resolve_target_layer_ids(
        explicit=LLAMA_31_8B_EAGLE3_PROFILE.target_layer_ids,
        num_hidden_layers=LLAMA_31_8B_EAGLE3_PROFILE.num_hidden_layers,
        drafter_aux_count=None,
    )
    assert llama_resolved == (2, 16, 29)


def test_qwen3_8b_profile_in_builtin_registry() -> None:
    assert "qwen3-8b-eagle3" in BUILTIN_PROFILES
    assert BUILTIN_PROFILES["qwen3-8b-eagle3"] is QWEN_3_8B_EAGLE3_PROFILE


def test_qwen3_8b_profile_resolution_regression() -> None:
    assert (
        resolve_profile(
            served_model="Qwen/Qwen3-8B", profiles=BUILTIN_PROFILES
        ) is QWEN_3_8B_EAGLE3_PROFILE
    )
    assert (
        resolve_profile(
            served_model="qwen3-8b-eagle3", profiles=BUILTIN_PROFILES
        ) is QWEN_3_8B_EAGLE3_PROFILE
    )


# -- activation-capture E2E aux-layer derivation --------------------------
#
# The serving-activation-capture E2E used to default its target layers to a
# hardcoded [4, 12, 20], which matched neither drafter this deployment runs
# and aborted job 369214.  Its replacement derives them from the drafter and
# verifier configs through ``resolve_target_layer_ids``, so it is covered
# here alongside the resolution it delegates to rather than only behind the
# GPU gate.
from e2e.test_serving_activation_capture import (  # noqa: E402
    _drafter_aux_declaration,
    _resolve_model_dir,
    _target_layer_ids,
    _verifier_num_hidden_layers,
)

#: The published RedHatAI speculators as they actually sit in the HF cache:
#: Qwen3-8B omits ``eagle_aux_hidden_state_layer_ids`` entirely, gpt-oss-20b
#: carries it as null.  Neither states the layers, so both must come from the
#: derivation path.
_QWEN_DRAFTER_CONFIG: dict[str, object] = {
    "architectures": ["Eagle3Speculator"],
    "speculators_model_type": "eagle3",
    "speculators_config": {
        "algorithm": "eagle3",
        "verifier": {"name_or_path": "Qwen/Qwen3-8B"},
    },
}
_GPT_OSS_DRAFTER_CONFIG: dict[str, object] = {
    "architectures": ["Eagle3DraftModel"],
    "speculators_model_type": "eagle3",
    "eagle_aux_hidden_state_layer_ids": None,
    "speculators_config": {
        "algorithm": "eagle3",
        "verifier": {"name_or_path": "openai/gpt-oss-20b"},
    },
}


def _write_cached_model(
    hf_home: Path,
    repository: str,
    config: dict[str, object],
) -> Path:
    slug = "models--" + repository.replace("/", "--")
    snapshot = hf_home / "hub" / slug / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def _fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hf_home = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("SPEEDLM_E2E_TARGET_LAYER_IDS", raising=False)
    monkeypatch.delenv("SPEEDLM_E2E_DRAFTER_DIR", raising=False)
    _write_cached_model(hf_home, "Qwen/Qwen3-8B", {"num_hidden_layers": 36})
    _write_cached_model(
        hf_home, "RedHatAI/Qwen3-8B-speculator.eagle3", _QWEN_DRAFTER_CONFIG
    )
    _write_cached_model(hf_home, "openai/gpt-oss-20b", {"num_hidden_layers": 24})
    _write_cached_model(
        hf_home, "RedHatAI/gpt-oss-20b-speculator.eagle3", _GPT_OSS_DRAFTER_CONFIG
    )
    return hf_home


def test_e2e_target_layers_derived_for_both_cached_drafters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither published drafter states its layers; both must be derived."""
    _fake_cache(tmp_path, monkeypatch)

    assert _target_layer_ids(
        "Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3"
    ) == [2, 18, 33]
    assert _target_layer_ids(
        "openai/gpt-oss-20b", "RedHatAI/gpt-oss-20b-speculator.eagle3"
    ) == [2, 12, 21]


def test_e2e_target_layers_track_verifier_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different verifier depth moves the layers -- nothing is hardcoded."""
    hf_home = _fake_cache(tmp_path, monkeypatch)
    _write_cached_model(hf_home, "acme/deep-48", {"num_hidden_layers": 48})

    assert _target_layer_ids(
        "acme/deep-48", "RedHatAI/Qwen3-8B-speculator.eagle3"
    ) == list(spread_aux_layers(3, 48))


def test_e2e_target_layers_honour_drafter_declared_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drafter that pins its own layer ids wins over the heuristic."""
    hf_home = _fake_cache(tmp_path, monkeypatch)
    _write_cached_model(
        hf_home,
        "acme/pinned.eagle3",
        {"eagle_aux_hidden_state_layer_ids": [33, 2, 18]},
    )

    assert _target_layer_ids("Qwen/Qwen3-8B", "acme/pinned.eagle3") == [2, 18, 33]


def test_e2e_target_layers_use_drafter_declared_arity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared aux count drives k, not the k == 3 fallback."""
    hf_home = _fake_cache(tmp_path, monkeypatch)
    _write_cached_model(hf_home, "acme/four.eagle3", {"num_aux_hidden_states": 4})

    assert _target_layer_ids("Qwen/Qwen3-8B", "acme/four.eagle3") == list(
        spread_aux_layers(4, 36)
    )


def test_e2e_target_layers_explicit_env_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("SPEEDLM_E2E_TARGET_LAYER_IDS", "[11, 3, 7]")

    assert _target_layer_ids(
        "Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3"
    ) == [3, 7, 11]


def test_e2e_target_layers_reject_malformed_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("SPEEDLM_E2E_TARGET_LAYER_IDS", "not json")
    with pytest.raises(AssertionError, match="must be JSON"):
        _target_layer_ids("Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3")

    monkeypatch.setenv("SPEEDLM_E2E_TARGET_LAYER_IDS", "[]")
    with pytest.raises(AssertionError, match="non-empty JSON array"):
        _target_layer_ids("Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3")


def test_e2e_uncached_model_reports_instead_of_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline runs must name the missing snapshot, not hang on a fetch."""
    _fake_cache(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="cannot resolve"):
        _resolve_model_dir("acme/never-pulled")


def test_e2e_aux_declaration_readers_handle_null_and_absent() -> None:
    assert _drafter_aux_declaration(_QWEN_DRAFTER_CONFIG) == (None, None)
    assert _drafter_aux_declaration(_GPT_OSS_DRAFTER_CONFIG) == (None, None)
    assert _drafter_aux_declaration({"eagle_aux_hidden_state_layer_ids": [2, 5]}) == (
        (2, 5),
        2,
    )
    assert _drafter_aux_declaration(
        {"eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 4, 9]}}
    ) == ((1, 4, 9), 3)
    assert _drafter_aux_declaration({"num_aux_hidden_states": 5}) == (None, 5)


def test_e2e_verifier_depth_reader_handles_text_config() -> None:
    assert _verifier_num_hidden_layers({"num_hidden_layers": 36}) == 36
    assert _verifier_num_hidden_layers({"text_config": {"num_hidden_layers": 28}}) == 28
    assert _verifier_num_hidden_layers({"num_hidden_layers": 0}) is None
    assert _verifier_num_hidden_layers({}) is None


# ---------------------------------------------------------------------------
# Draft-chain depth: what is trained must be what is served
# ---------------------------------------------------------------------------


def test_the_profile_pin_is_the_serving_truth_and_wins() -> None:
    """gpt-oss serves 5; the trainer's default of 3 must not override it."""
    assert (
        resolve_speculative_tokens(explicit=5, drafter_declared=3) == 5
    )


def test_a_drafter_declaration_is_used_when_the_profile_pins_nothing() -> None:
    assert resolve_speculative_tokens(drafter_declared=4) == 4


def test_nothing_named_falls_back_to_the_trainer_default() -> None:
    assert resolve_speculative_tokens() == DEFAULT_SPECULATIVE_TOKENS == 3


@pytest.mark.parametrize("depth", [0, -1, MAX_SPECULATIVE_TOKENS + 1])
def test_an_out_of_range_depth_is_rejected(depth: int) -> None:
    with pytest.raises(SpeculativeDepthError):
        resolve_speculative_tokens(explicit=depth)


def test_the_ceiling_is_not_a_hardcoded_five() -> None:
    """gpt-oss's conditional acceptance still rises at position 5.

    Whatever replaced the hardcoded 3 must leave going deeper reachable, so a
    depth above the current serving value has to resolve cleanly.
    """
    assert resolve_speculative_tokens(explicit=8) == 8
    assert MAX_SPECULATIVE_TOKENS > 5


def test_the_stock_drafters_declaration_is_read_not_guessed() -> None:
    """Both cached RedHatAI EAGLE-3 drafters declare 3 here."""
    config = {
        "speculators_config": {
            "algorithm": "eagle3",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 3},
            ],
        }
    }

    assert drafter_declared_speculative_tokens(config) == 3


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"speculators_config": {}},
        {"speculators_config": {"proposal_methods": []}},
        {"speculators_config": {"proposal_methods": [{"speculative_tokens": 0}]}},
        {
            "speculators_config": {
                "proposal_methods": [
                    {"speculative_tokens": 3},
                    {"speculative_tokens": 5},
                ]
            }
        },
    ],
)
def test_an_unusable_declaration_is_no_evidence(config: dict[str, object]) -> None:
    assert drafter_declared_speculative_tokens(config) is None


def test_a_training_serving_mismatch_fails_before_the_gpu_cycle() -> None:
    with pytest.raises(SpeculativeDepthError, match="extrapolation"):
        validate_training_depth(
            profile_name="gpt-oss-20b-eagle3",
            serving_tokens=5,
            training_steps=3,
        )


def test_matching_depths_pass() -> None:
    validate_training_depth(
        profile_name="qwen3-8b-eagle3",
        serving_tokens=3,
        training_steps=3,
    )


def test_every_builtin_profile_can_be_trained_at_the_depth_it_serves() -> None:
    for profile in BUILTIN_PROFILES.values():
        depth = resolve_speculative_tokens(explicit=profile.num_speculative_tokens)
        validate_training_depth(
            profile_name=profile.name,
            serving_tokens=profile.num_speculative_tokens,
            training_steps=depth,
        )


def test_a_profile_cannot_request_an_unbounded_chain() -> None:
    with pytest.raises(SpeculativeDepthError):
        ModelProfile(
            name="too-deep",
            verifier_model="acme/verifier",
            draft_model="acme/draft",
            speculative_method="eagle3",
            num_speculative_tokens=MAX_SPECULATIVE_TOKENS + 1,
            target_layer_ids=None,
            chat_template_kind="auto",
            max_seq_len=4096,
        )


# ---------------------------------------------------------------------------
# Context window: the position axis, the twin of validate_training_depth
# ---------------------------------------------------------------------------


def _cache_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository: str = "openai/gpt-oss-20b",
    config: dict[str, object] | None = None,
) -> None:
    hf_home = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    if config is not None:
        _write_cached_model(hf_home, repository, config)


def test_an_explicit_max_model_len_outranks_the_verifier_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vLLM lets the flag win, so the resolver must too."""
    _cache_verifier(
        tmp_path, monkeypatch, config={"max_position_embeddings": 131_072}
    )

    resolved = resolve_serving_context_window(
        "openai/gpt-oss-20b", ("--dtype", "bfloat16", "--max-model-len", "8192")
    )

    assert resolved == ServingContextWindow(tokens=8_192, source="passthrough")


def test_the_equals_form_of_max_model_len_is_read_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache_verifier(
        tmp_path, monkeypatch, config={"max_position_embeddings": 131_072}
    )

    resolved = resolve_serving_context_window(
        "openai/gpt-oss-20b", ("--max-model-len=4096",)
    )

    assert resolved == ServingContextWindow(tokens=4_096, source="passthrough")


def test_the_serving_window_falls_back_to_max_position_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What vLLM itself falls back to when no flag is given."""
    _cache_verifier(
        tmp_path, monkeypatch, config={"max_position_embeddings": 131_072}
    )

    resolved = resolve_serving_context_window("openai/gpt-oss-20b", ())

    assert resolved == ServingContextWindow(
        tokens=131_072, source="verifier_config"
    )


def test_an_uncached_verifier_reports_unresolved_rather_than_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache_verifier(tmp_path, monkeypatch)

    resolved = resolve_serving_context_window("openai/gpt-oss-20b", ())

    assert resolved == ServingContextWindow(tokens=None, source="unresolved")


def test_a_config_without_a_usable_position_count_is_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache_verifier(
        tmp_path, monkeypatch, config={"max_position_embeddings": "lots"}
    )

    resolved = resolve_serving_context_window("openai/gpt-oss-20b", ())

    assert resolved == ServingContextWindow(tokens=None, source="unresolved")


@pytest.mark.parametrize("value", ["twelve", "0", "-1"])
def test_an_unusable_max_model_len_is_rejected_before_a_gpu_cycle(
    value: str,
) -> None:
    with pytest.raises(ContextWindowError):
        resolve_serving_context_window("openai/gpt-oss-20b", ("--max-model-len", value))


def test_a_valueless_max_model_len_is_rejected() -> None:
    with pytest.raises(ContextWindowError):
        resolve_serving_context_window(
            "openai/gpt-oss-20b", ("--max-model-len", "--dtype", "bfloat16")
        )


def test_the_stock_mismatch_is_recorded_rather_than_raised() -> None:
    """Stock defaults disagree 8x; a policy that raised would reject them all."""
    alignment = validate_training_context_window(
        profile_name="gpt-oss-20b-eagle3",
        training_tokens=16_384,
        serving=ServingContextWindow(tokens=131_072, source="verifier_config"),
    )

    assert alignment.aligned is False
    assert alignment.ratio == 8.0
    assert alignment.as_manifest_fields() == {
        "training_sequence_length": 16_384,
        "serving_context_window": 131_072,
        "serving_context_window_source": "verifier_config",
        "context_window_policy": "record",
        "context_window_ratio": 8.0,
        "context_window_aligned": False,
    }


def test_an_unresolved_serving_window_is_recorded_as_null_not_omitted() -> None:
    alignment = validate_training_context_window(
        profile_name="gpt-oss-20b-eagle3",
        training_tokens=16_384,
        serving=ServingContextWindow(tokens=None, source="unresolved"),
    )

    assert alignment.aligned is False
    assert alignment.ratio is None
    assert alignment.as_manifest_fields()["serving_context_window"] is None


def test_equal_windows_are_reported_as_aligned() -> None:
    alignment = validate_training_context_window(
        profile_name="gpt-oss-20b-eagle3",
        training_tokens=4_096,
        serving=ServingContextWindow(tokens=4_096, source="passthrough"),
        policy="strict",
    )

    assert alignment.aligned is True
    assert alignment.ratio == 1.0


def test_the_strict_policy_names_the_extrapolated_positions() -> None:
    """The same argument validate_training_depth makes, on the other axis."""
    with pytest.raises(ContextWindowError) as excinfo:
        validate_training_context_window(
            profile_name="gpt-oss-20b-eagle3",
            training_tokens=16_384,
            serving=ServingContextWindow(tokens=131_072, source="verifier_config"),
            policy="strict",
        )

    message = str(excinfo.value)
    assert "16385..131072" in message
    assert "extrapolation" in message


def test_the_strict_policy_refuses_an_unresolved_serving_window() -> None:
    with pytest.raises(ContextWindowError):
        validate_training_context_window(
            profile_name="gpt-oss-20b-eagle3",
            training_tokens=16_384,
            serving=ServingContextWindow(tokens=None, source="unresolved"),
            policy="strict",
        )


def test_an_unknown_policy_is_rejected() -> None:
    with pytest.raises(ContextWindowError):
        validate_training_context_window(
            profile_name="gpt-oss-20b-eagle3",
            training_tokens=16_384,
            serving=ServingContextWindow(tokens=16_384, source="passthrough"),
            policy="warn",  # type: ignore[arg-type]
        )


def test_context_window_errors_are_profile_errors() -> None:
    """Callers already catching ProfileError must not miss this one."""
    assert issubclass(ContextWindowError, ProfileError)
