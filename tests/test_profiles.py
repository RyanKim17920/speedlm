from __future__ import annotations

import json
from pathlib import Path

import pytest

from speedlm.config import SpeedLMConfig
from speedlm.profiles import (
    BUILTIN_PROFILES,
    GPT_OSS_EAGLE3_PROFILE,
    LLAMA_31_8B_EAGLE3_PROFILE,
    QWEN_35_9B_MTP_PROFILE,
    ModelProfile,
    ProfileError,
    load_profiles,
    resolve_profile,
)


def _write_profile(home: Path, filename: str, data: dict[str, object]) -> Path:
    profiles_dir = home / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_builtin_profiles_load_and_validate(tmp_path: Path) -> None:
    profiles = load_profiles(tmp_path)

    assert profiles == dict(BUILTIN_PROFILES)
    assert GPT_OSS_EAGLE3_PROFILE.target_layer_ids == (2, 12, 21)
    assert GPT_OSS_EAGLE3_PROFILE.chat_template_kind == "harmony"
    assert GPT_OSS_EAGLE3_PROFILE.tool_call_parser == "openai"
    assert GPT_OSS_EAGLE3_PROFILE.reasoning_parser == "openai_gptoss"
    assert LLAMA_31_8B_EAGLE3_PROFILE.speculative_method == "eagle3"
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
