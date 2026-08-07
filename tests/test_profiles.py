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


# ---------------------------------------------------------------------------
# Dynamic-K schedules (num_speculative_tokens_per_batch_size)
# ---------------------------------------------------------------------------
def _scheduled(schedule: object, *, tokens: int = 5) -> ModelProfile:
    """A gpt-oss-shaped profile carrying *schedule*."""

    return ModelProfile(
        name="scheduled",
        verifier_model="openai/gpt-oss-20b",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        speculative_method="eagle3",
        num_speculative_tokens=tokens,
        target_layer_ids=(2, 12, 21),
        chat_template_kind="harmony",
        max_seq_len=4096,
        speculative_schedule=schedule,  # type: ignore[arg-type]
    )


def test_no_builtin_profile_emits_a_batch_size_schedule() -> None:
    """The feature is additive: today's engines must stay on fixed K.

    Emitting ``num_speculative_tokens_per_batch_size`` is precisely what makes
    ``Scheduler.dynamic_sd_lookup`` stop being ``None``, so a profile that
    emits it by accident silently changes how every batch size is drafted.
    """
    for profile in BUILTIN_PROFILES.values():
        assert profile.speculative_schedule is None
        assert "num_speculative_tokens_per_batch_size" not in profile.speculative_config()
    assert GPT_OSS_EAGLE3_PROFILE.speculative_config() == {
        "method": "eagle3",
        "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
        "num_speculative_tokens": 5,
    }


def test_a_schedule_reaches_the_engine_under_the_key_vllm_reads() -> None:
    """The key name is the whole contract with ``config/speculative.py:172``."""
    profile = _scheduled(((1, 8, 5), (9, 64, 2)))

    config = profile.speculative_config()

    assert config["num_speculative_tokens_per_batch_size"] == [[1, 8, 5], [9, 64, 2]]
    # The scalar stays, because vLLM still clamps the schedule against it and
    # falls back to it whenever the batch is empty.
    assert config["num_speculative_tokens"] == 5


def test_a_schedule_survives_a_json_round_trip() -> None:
    profile = _scheduled(((1, 8, 5), (9, 64, 0)))

    restored = ModelProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert restored.speculative_schedule == ((1, 8, 5), (9, 64, 0))
    assert restored == profile


def test_a_json_profile_may_declare_a_schedule(tmp_path: Path) -> None:
    """``from_dict`` rejects unknown keys, so the allowlist must know it."""
    data = GPT_OSS_EAGLE3_PROFILE.to_dict()
    data["speculative_schedule"] = [[1, 16, 5], [17, 256, 3]]

    restored = ModelProfile.from_dict(data, source=str(tmp_path))

    assert restored.speculative_schedule == ((1, 16, 5), (17, 256, 3))


@pytest.mark.parametrize(
    ("schedule", "match"),
    [
        (((2, 8, 5),), "must start at batch size 1"),
        (((1, 8, 5), (5, 64, 3)), "overlap or are out of order"),
        (((1, 8, 5), (64, 32, 3)), "ends before it starts"),
        (((0, 8, 5),), "range_start must be at least 1"),
        (((1, 8, 5), (9, 64, -1)), "must not be negative"),
        ((), "must not be empty"),
        (((1, 8),), "range_start, range_end, num_speculative_tokens"),
        (((1, 8, True),), "range_start, range_end, num_speculative_tokens"),
        ("1,8,5", "array of bands"),
    ],
)
def test_a_malformed_schedule_is_rejected_at_profile_load(
    schedule: object, match: str
) -> None:
    """Every rule here mirrors one vLLM would raise hours later, at engine boot."""
    with pytest.raises(ProfileError, match=match):
        _scheduled(schedule)


def test_a_schedule_cannot_ask_for_more_depth_than_the_profile_declares() -> None:
    """vLLM clamps with ``min(num_speculative_tokens, K)`` and says nothing.

    A profile whose deepest band reads 8 while the engine serves 5 is a profile
    that lies about what it serves, which is the failure mode this module
    exists to remove.
    """
    with pytest.raises(ProfileError, match="silently clamp"):
        _scheduled(((1, 8, 8),), tokens=5)


def test_a_schedule_must_actually_reach_the_declared_depth() -> None:
    """Otherwise ``num_speculative_tokens`` stops describing the serving depth.

    It is the number the drafter is trained against, so it has to be the depth
    the schedule reaches -- see ``validate_training_depth``.
    """
    with pytest.raises(ProfileError, match="must be the depth the schedule"):
        _scheduled(((1, 8, 3), (9, 64, 2)), tokens=5)


def test_the_deepest_band_is_what_the_profile_reports_as_served() -> None:
    assert GPT_OSS_EAGLE3_PROFILE.max_served_speculative_tokens == 5
    assert _scheduled(((1, 8, 5), (9, 64, 1))).max_served_speculative_tokens == 5


def test_the_depth_guard_measures_the_deepest_band_not_the_shallowest() -> None:
    """Run d993eee-gptoss-idle is the measurement behind this choice.

    A drafter declaring ``speculative_tokens: 3`` served at 5 was at parity
    with stock at draft position 0 (0.6850 vs 0.6871) and lost 5.3 pp at
    position 4.  Only the deep positions are extrapolation, so the deepest
    band is what training depth must match.
    """
    schedule = ((1, 8, 5), (9, 64, 3))

    # Training to the shallow band is not enough, even though most batch
    # sizes in that schedule only ever draft 3.
    with pytest.raises(SpeculativeDepthError, match="positions 4..5"):
        validate_training_depth(
            profile_name="scheduled",
            serving_tokens=5,
            training_steps=3,
            serving_schedule=schedule,
        )

    validate_training_depth(
        profile_name="scheduled",
        serving_tokens=5,
        training_steps=5,
        serving_schedule=schedule,
    )


def test_the_depth_guard_names_the_schedule_when_one_is_in_play() -> None:
    with pytest.raises(SpeculativeDepthError, match="deepest band of 2"):
        validate_training_depth(
            profile_name="scheduled",
            serving_tokens=5,
            training_steps=2,
            serving_schedule=((1, 8, 5), (9, 64, 3)),
        )


def test_every_builtin_profile_trains_to_the_depth_its_schedule_reaches() -> None:
    for profile in BUILTIN_PROFILES.values():
        validate_training_depth(
            profile_name=profile.name,
            serving_tokens=profile.num_speculative_tokens,
            training_steps=profile.max_served_speculative_tokens,
            serving_schedule=profile.speculative_schedule,
        )


# ---------------------------------------------------------------------------
# A brand-new model, declared only in user JSON.
#
# Everything above proves the *error* path (an unknown verifier raises) and the
# *override* path (a user file replacing a built-in by name).  Neither proves
# the path the product's generality claim actually rests on: an operator points
# SpeedLM at a model SpeedLM has never heard of, declares it in a JSON file, and
# the whole resolution chain works with no code change.  These tests are that
# path, from the file on disk to the argv the engine would be launched with.
# ---------------------------------------------------------------------------

#: A vendor and model that appear in no built-in profile.  Deliberately not a
#: real Hub repository: nothing in this section may reach the network, so the
#: name must be one that could never accidentally resolve to a cached snapshot.
NEW_VERIFIER = "vertexis/Atlas-7B-Instruct"
NEW_DRAFTER = "vertexis/Atlas-7B-Instruct-speculator.eagle3"
NEW_PROFILE_NAME = "atlas-7b-instruct-eagle3"

#: The JSON an operator would write.  Every optional field is populated, because
#: an optional field left unset proves nothing about whether it is reachable.
NEW_PROFILE_JSON: dict[str, object] = {
    "name": NEW_PROFILE_NAME,
    "verifier_model": NEW_VERIFIER,
    "draft_model": NEW_DRAFTER,
    "speculative_method": "eagle3",
    "num_speculative_tokens": 4,
    "target_layer_ids": [3, 11, 19, 25],
    "chat_template_kind": "chatml",
    "max_seq_len": 8192,
    "num_hidden_layers": 28,
    "max_scratch_bytes": 123_456_789,
    "tool_call_parser": "atlas_xml",
    "reasoning_parser": "atlas_think",
    "speculative_schedule": [[1, 4, 2], [5, 64, 4]],
}

#: What each declared key must become on the resolved :class:`ModelProfile`.
#: Keyed by dataclass field name so the structural test below can assert this
#: covers every field -- a new field with no expectation here is a field nobody
#: has proved reachable.
NEW_PROFILE_EXPECTED: dict[str, object] = {
    "name": NEW_PROFILE_NAME,
    "verifier_model": NEW_VERIFIER,
    "draft_model": NEW_DRAFTER,
    "speculative_method": "eagle3",
    "num_speculative_tokens": 4,
    "target_layer_ids": (3, 11, 19, 25),
    "chat_template_kind": "chatml",
    "max_seq_len": 8192,
    "num_hidden_layers": 28,
    "max_scratch_bytes": 123_456_789,
    "tool_call_parser": "atlas_xml",
    "reasoning_parser": "atlas_think",
    "speculative_schedule": ((1, 4, 2), (5, 64, 4)),
    "trainable": True,
}


def _new_model_profile(home: Path) -> ModelProfile:
    """Write the new-model JSON under *home* and return the loaded profile."""

    _write_profile(home, "atlas.json", dict(NEW_PROFILE_JSON))
    registry = load_profiles(home)
    return registry[NEW_PROFILE_NAME]


def test_a_model_with_no_builtin_profile_is_declarable_purely_in_json(
    tmp_path: Path,
) -> None:
    """The success path: a new model added by a file, not by a code change."""

    assert NEW_PROFILE_NAME not in BUILTIN_PROFILES
    assert all(
        builtin.verifier_model != NEW_VERIFIER for builtin in BUILTIN_PROFILES.values()
    )

    _write_profile(tmp_path, "atlas.json", dict(NEW_PROFILE_JSON))
    registry = load_profiles(tmp_path)

    # Added, not substituted: every built-in survives alongside it.
    assert len(registry) == len(BUILTIN_PROFILES) + 1
    for name, builtin in BUILTIN_PROFILES.items():
        assert registry[name] == builtin
    assert NEW_PROFILE_NAME in registry


def test_every_declared_field_arrives_on_the_resolved_profile(
    tmp_path: Path,
) -> None:
    """Assume nothing is reachable until the declared value has been seen.

    The loop is over the dataclass's own fields rather than a hand-written list
    of assertions, so a field added to :class:`ModelProfile` and never proved
    loadable fails here instead of shipping unreachable.
    """

    profile = _new_model_profile(tmp_path)

    assert set(NEW_PROFILE_EXPECTED) == profiles_module.PROFILE_FIELD_NAMES
    for field_name in sorted(profiles_module.PROFILE_FIELD_NAMES):
        assert getattr(profile, field_name) == NEW_PROFILE_EXPECTED[field_name], (
            f"{field_name} did not survive the JSON round trip"
        )

    # And the reverse direction: nothing is dropped on the way back out.
    assert profile.to_dict() == {**NEW_PROFILE_JSON, "trainable": True}


def test_the_new_model_resolves_by_profile_name(tmp_path: Path) -> None:
    _new_model_profile(tmp_path)

    resolved = resolve_profile(
        {"model": "some/other-model", "profile": NEW_PROFILE_NAME},
        home=tmp_path,
    )
    assert resolved.name == NEW_PROFILE_NAME
    assert resolved.verifier_model == NEW_VERIFIER


def test_the_new_model_resolves_by_config_model_verifier_string(
    tmp_path: Path,
) -> None:
    """Resolution by ``config.model``, not only by an explicit profile name.

    This is the shape a deployment actually has: the operator names the model
    they are serving and never names a profile at all.
    """

    _new_model_profile(tmp_path)

    resolved = resolve_profile({"model": NEW_VERIFIER, "profile": None}, home=tmp_path)

    assert resolved.name == NEW_PROFILE_NAME
    assert resolved.draft_model == NEW_DRAFTER


def test_the_new_model_resolves_by_served_model_and_by_cache_path(
    tmp_path: Path,
) -> None:
    _new_model_profile(tmp_path)

    by_served = resolve_profile(served_model=NEW_VERIFIER, home=tmp_path)
    assert by_served.name == NEW_PROFILE_NAME

    snapshot = (
        "/data/hf-cache/hub/models--vertexis--Atlas-7B-Instruct/snapshots/abc123"
    )
    by_snapshot = resolve_profile(served_model=snapshot, home=tmp_path)
    assert by_snapshot.name == NEW_PROFILE_NAME


def test_the_new_models_declared_depth_and_schedule_reach_the_engine_fragment(
    tmp_path: Path,
) -> None:
    """The vLLM speculative-config fragment carries the declared values."""

    profile = _new_model_profile(tmp_path)

    assert profile.speculative_config() == {
        "method": "eagle3",
        "num_speculative_tokens": 4,
        "model": NEW_DRAFTER,
        "num_speculative_tokens_per_batch_size": [[1, 4, 2], [5, 64, 4]],
    }
    assert profile.max_served_speculative_tokens == 4


def test_the_new_models_declared_depth_drives_the_training_depth(
    tmp_path: Path,
) -> None:
    profile = _new_model_profile(tmp_path)

    training_depth = resolve_speculative_tokens(
        explicit=profile.num_speculative_tokens,
        drafter_declared=None,
    )
    assert training_depth == 4

    validate_training_depth(
        profile_name=profile.name,
        serving_tokens=profile.num_speculative_tokens,
        training_steps=training_depth,
        serving_schedule=profile.speculative_schedule,
    )
    with pytest.raises(SpeculativeDepthError, match="3-deep"):
        validate_training_depth(
            profile_name=profile.name,
            serving_tokens=profile.num_speculative_tokens,
            training_steps=3,
            serving_schedule=profile.speculative_schedule,
        )


def test_the_new_models_declared_aux_layers_drive_the_resolver(
    tmp_path: Path,
) -> None:
    """``target_layer_ids`` and ``num_hidden_layers`` reach the aux resolver.

    This is the call ``build_tuning_launch_plan`` makes: the profile's pin goes
    *through* the resolver so it is cross-checked against the drafter's arity
    rather than trusted.
    """

    profile = _new_model_profile(tmp_path)

    resolved = resolve_target_layer_ids(
        explicit=profile.target_layer_ids,
        num_hidden_layers=profile.num_hidden_layers,
        drafter_aux_count=4,
    )
    assert resolved == (3, 11, 19, 25)

    # The declared layer count is what bounds them.
    with pytest.raises(AuxLayerError, match="28 hidden layers"):
        resolve_target_layer_ids(
            explicit=(3, 11, 19, 99),
            num_hidden_layers=profile.num_hidden_layers,
            drafter_aux_count=4,
        )

    # And with no pin, the declared depth is what the spread is computed over.
    derived = resolve_target_layer_ids(
        explicit=None,
        num_hidden_layers=profile.num_hidden_layers,
        drafter_aux_count=4,
    )
    assert derived == spread_aux_layers(4, 28)
    assert max(derived) < profile.num_hidden_layers


def test_the_new_models_max_seq_len_caps_the_training_window(
    tmp_path: Path,
) -> None:
    """``max_seq_len`` reaches the window a cycle would actually train on.

    ``tuning.sequence_length`` defaults to 16384; the declared 8192 is lower, so
    a profile that failed to reach ``training_sequence_length`` would show up as
    16384 here.
    """

    from speedlm.tuner.composition import (
        resolve_context_window_alignment,
        training_sequence_length,
    )

    profile = _new_model_profile(tmp_path)
    config = SpeedLMConfig(model=NEW_VERIFIER)
    assert config.tuning.sequence_length == 16_384

    assert training_sequence_length(config, profile) == 8192

    aligned = resolve_context_window_alignment(
        config,
        profile,
        ["--max-model-len", "8192"],
        policy="record",
    )
    assert aligned.training_tokens == 8192
    assert aligned.serving_tokens == 8192
    assert aligned.source == "passthrough"
    assert aligned.aligned is True

    skewed = resolve_context_window_alignment(
        config,
        profile,
        ["--max-model-len", "65536"],
        policy="record",
    )
    assert skewed.aligned is False
    assert skewed.ratio == 8.0
    assert skewed.as_manifest_fields()["training_sequence_length"] == 8192


def test_the_new_models_declared_parsers_win_over_auto_detection(
    tmp_path: Path,
) -> None:
    """The declared parser names arrive as profile pins, not as guesses."""

    profile = _new_model_profile(tmp_path)

    resolution = resolve_model_parsers(
        NEW_VERIFIER,
        (),
        model_type="atlas",
        profile=profile,
        registry=ParserRegistry(
            tool_parsers=("hermes",),
            reasoning_parsers=("deepseek_r1",),
        ),
    )
    assert resolution.tool_call_parser == "atlas_xml"
    assert resolution.tool_call_parser_source == "profile-pinned"
    assert resolution.reasoning_parser == "atlas_think"
    assert resolution.reasoning_parser_source == "profile-pinned"


def test_the_new_model_reaches_the_vllm_argv_the_gateway_would_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As far downstream as this goes without a GPU: the serving argv itself.

    ``speedlm vllm serve`` builds its child argv from
    ``_profiled_vllm_passthrough`` plus ``build_vllm_argv``.  Both are pure, so
    the declared parsers and the declared speculative config can be shown to
    land in the exact argument list a real launch would use.
    """

    from speedlm.cli import _profiled_vllm_passthrough
    from speedlm.gateway.process import build_vllm_argv

    profile = _new_model_profile(tmp_path)

    # No network and no cache lookup: this model exists nowhere but the JSON.
    monkeypatch.setattr(
        profiles_module,
        "read_model_type",
        lambda model, allow_remote=False: None,
    )
    monkeypatch.setattr(
        profiles_module,
        "discover_vllm_parser_registry",
        ParserRegistry,
    )

    passthrough = _profiled_vllm_passthrough(
        NEW_VERIFIER,
        [],
        profile=profile,
        home=tmp_path,
    )
    argv = build_vllm_argv(
        NEW_VERIFIER,
        [
            *passthrough,
            "--speculative-config",
            json.dumps(profile.speculative_config(), sort_keys=True),
        ],
        host="127.0.0.1",
        port=8000,
    )

    assert argv[:3] == ["vllm", "serve", NEW_VERIFIER]
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "atlas_xml"
    assert argv[argv.index("--reasoning-parser") + 1] == "atlas_think"
    speculative = json.loads(argv[argv.index("--speculative-config") + 1])
    assert speculative["model"] == NEW_DRAFTER
    assert speculative["num_speculative_tokens"] == 4
    assert speculative["num_speculative_tokens_per_batch_size"] == [
        [1, 4, 2],
        [5, 64, 4],
    ]


def test_a_new_model_profile_is_still_validated(tmp_path: Path) -> None:
    """Generality is not permissiveness: the new-model path still refuses junk."""

    broken = dict(NEW_PROFILE_JSON)
    broken["target_layer_ids"] = [3, 11, 11]
    _write_profile(tmp_path, "atlas.json", broken)
    with pytest.raises(ProfileError, match="target_layer_ids"):
        load_profiles(tmp_path)


# ---------------------------------------------------------------------------
# Structural checks on the from_dict key sets.
# ---------------------------------------------------------------------------


def test_the_profile_allow_list_is_derived_from_the_dataclass() -> None:
    """A hand-maintained allow-list is a copy of a structure, and copies drift.

    Modelled on ``tests/test_config.py``'s structural tests, which derive the
    config section keys from ``dataclasses.fields`` for the same reason: the
    promotion allow-list once omitted a documented, validated field and made it
    unsettable from JSON.
    """

    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ModelProfile)}

    assert field_names == profiles_module.PROFILE_FIELD_NAMES
    assert field_names >= profiles_module.REQUIRED_PROFILE_KEYS
    # to_dict is the other half of the same contract.
    assert set(GPT_OSS_EAGLE3_PROFILE.to_dict()) == field_names


def test_every_dataclass_field_is_actually_accepted_by_from_dict() -> None:
    """Derivation is only worth having if the derived set is honoured."""

    import dataclasses

    complete = {**NEW_PROFILE_JSON, "trainable": True}
    assert set(complete) == {f.name for f in dataclasses.fields(ModelProfile)}

    profile = ModelProfile.from_dict(complete)

    assert profile.name == NEW_PROFILE_NAME
    assert profile.trainable is True


def test_an_unknown_profile_key_is_still_rejected() -> None:
    data = {**NEW_PROFILE_JSON, "_structural_test_fake_key_xyz": 1}

    with pytest.raises(ProfileError, match="unknown keys: _structural_test_fake_key_xyz"):
        ModelProfile.from_dict(data)


def test_a_missing_required_key_is_still_rejected() -> None:
    data = dict(NEW_PROFILE_JSON)
    del data["draft_model"]

    with pytest.raises(ProfileError, match="missing required keys: draft_model"):
        ModelProfile.from_dict(data)


def test_a_contradicting_trainable_is_still_rejected() -> None:
    data = {**NEW_PROFILE_JSON, "trainable": False}

    with pytest.raises(ProfileError, match="trainable must be True"):
        ModelProfile.from_dict(data)


def test_every_built_in_chat_template_is_declarable_in_a_profile() -> None:
    """``chat_template_kind`` must be able to name every shipped template.

    ``CHAT_TEMPLATE_KINDS`` is a hand-written frozenset while the templates are
    classes carrying their own ``name``.  A template added to
    ``speedlm.training.templates`` without a matching kind would be a renderer
    no profile could ever select, so the link is asserted rather than assumed.
    ``auto`` has no class by design -- it defers to the tokenizer -- so this is
    a subset check, not an equality.
    """

    from speedlm.training import templates as templates_module
    from speedlm.training.templates.base import ChatTemplate

    exported = [
        getattr(templates_module, attribute) for attribute in templates_module.__all__
    ]
    template_names = {
        candidate.name
        for candidate in exported
        if isinstance(candidate, type)
        and candidate is not ChatTemplate
        and isinstance(getattr(candidate, "name", None), str)
    }

    assert template_names, "no chat templates were discovered"
    undeclarable = template_names - profiles_module.CHAT_TEMPLATE_KINDS
    assert not undeclarable, (
        f"chat templates no profile can select: {sorted(undeclarable)}"
    )


def test_default_aux_layers_has_no_production_caller() -> None:
    """The hardcoded ``k == 3`` in ``default_aux_layers`` is a dead fallback.

    ``resolve_target_layer_ids`` is the live path and reads the drafter's own
    aux arity (``build_tuning_launch_plan`` passes ``drafter_aux_count``).
    ``default_aux_layers`` is documented as deprecated and is called by nothing
    under ``src/``; this pins that, so re-introducing a caller -- and with it a
    silently 3-way spread for a drafter that fuses a different number of aux
    states -- fails here rather than at drafter load time.
    """

    import re

    source_root = Path(profiles_module.__file__).parent
    call = re.compile(r"(?<!def )\bdefault_aux_layers\s*\(")
    callers = [
        str(path.relative_to(source_root))
        for path in sorted(source_root.rglob("*.py"))
        if call.search(path.read_text(encoding="utf-8"))
    ]

    assert callers == [], f"default_aux_layers has production callers: {callers}"
