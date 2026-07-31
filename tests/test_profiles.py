from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import speedlm.profiles as profiles_module
from speedlm.config import SpeedLMConfig
from speedlm.profiles import (
    BUILTIN_PROFILES,
    GPT_OSS_EAGLE3_PROFILE,
    LLAMA_31_8B_EAGLE3_PROFILE,
    QWEN_35_9B_MTP_PROFILE,
    AuxLayerError,
    ModelProfile,
    ParserRegistry,
    ProfileError,
    default_aux_layers,
    discover_vllm_parser_registry,
    load_profiles,
    resolve_model_parsers,
    resolve_profile,
    resolve_target_layer_ids,
    spread_aux_layers,
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
