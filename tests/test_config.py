from __future__ import annotations

from pathlib import Path

import pytest

from speedlm.config import (
    ConfigError,
    IdleTuningConfig,
    PromotionConfig,
    RedactionConfig,
    SamplingConfig,
    SpeedLMConfig,
    TargetConfig,
    TraceBufferConfig,
    WrapperConfig,
    load_config,
    save_config,
    speedlm_home,
    startup_stall_seconds,
    startup_timeout_seconds,
)

# ---------------------------------------------------------------------------
# speedlm_home
# ---------------------------------------------------------------------------


def test_home_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEEDLM_HOME", raising=False)
    # We can't easily override Path.home, so just check the suffix
    home = speedlm_home()
    assert home.name == ".speedlm"


def test_home_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    home = speedlm_home()
    assert home == tmp_path.resolve()


def test_startup_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEEDLM_STARTUP_TIMEOUT_SECONDS", raising=False)
    assert startup_timeout_seconds() == 900.0
    assert SpeedLMConfig(model="model").startup_timeout_seconds == 900.0


def test_startup_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_STARTUP_TIMEOUT_SECONDS", "123.5")
    assert startup_timeout_seconds() == 123.5
    assert SpeedLMConfig(model="model").startup_timeout_seconds == 123.5


def test_startup_timeout_config_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_STARTUP_TIMEOUT_SECONDS", "123.5")
    config = SpeedLMConfig.from_dict(
        {"model": "model", "startup_timeout_seconds": 456}
    )
    assert config.startup_timeout_seconds == 456


@pytest.mark.parametrize("value", [0, -1, True, "slow"])
def test_invalid_startup_timeout(value: object) -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig(model="model", startup_timeout_seconds=value)  # type: ignore[arg-type]


def test_startup_stall_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEEDLM_STARTUP_STALL_SECONDS", raising=False)
    assert startup_stall_seconds() == 600.0
    assert SpeedLMConfig(model="model").startup_stall_seconds == 600.0


def test_startup_stall_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_STARTUP_STALL_SECONDS", "720.5")
    assert startup_stall_seconds() == 720.5
    assert SpeedLMConfig(model="model").startup_stall_seconds == 720.5


def test_startup_stall_config_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_STARTUP_STALL_SECONDS", "720.5")
    config = SpeedLMConfig.from_dict(
        {"model": "model", "startup_stall_seconds": 840}
    )
    assert config.startup_stall_seconds == 840


@pytest.mark.parametrize("value", [0, -1, True, "slow"])
def test_invalid_startup_stall(value: object) -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig(model="model", startup_stall_seconds=value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SamplingConfig
# ---------------------------------------------------------------------------


def test_sampling_defaults() -> None:
    sc = SamplingConfig()
    assert sc.temperature == 0.0
    assert sc.top_p == 1.0
    assert sc.seed == 0


def test_sampling_negative_temperature() -> None:
    with pytest.raises(ConfigError):
        SamplingConfig(temperature=-0.1)


def test_sampling_bad_top_p_zero() -> None:
    with pytest.raises(ConfigError):
        SamplingConfig(top_p=0.0)


def test_sampling_bad_top_p_over_one() -> None:
    with pytest.raises(ConfigError):
        SamplingConfig(top_p=1.5)


def test_sampling_negative_seed() -> None:
    with pytest.raises(ConfigError):
        SamplingConfig(seed=-1)


def test_sampling_bool_seed_rejected() -> None:
    with pytest.raises(ConfigError):
        SamplingConfig(seed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TargetConfig / WrapperConfig
# ---------------------------------------------------------------------------


def test_target_defaults() -> None:
    tc = TargetConfig()
    assert tc.host == "127.0.0.1"
    assert tc.port == 8000


def test_wrapper_defaults() -> None:
    wc = WrapperConfig()
    assert wc.host == "127.0.0.1"
    assert wc.port == 8100


def test_bad_port_zero() -> None:
    with pytest.raises(ConfigError):
        TargetConfig(port=0)


def test_bad_port_too_high() -> None:
    with pytest.raises(ConfigError):
        TargetConfig(port=65536)


def test_bool_port_rejected() -> None:
    with pytest.raises(ConfigError):
        TargetConfig(port=True)  # type: ignore[arg-type]


def test_empty_host_rejected() -> None:
    with pytest.raises(ConfigError):
        TargetConfig(host="")


# ---------------------------------------------------------------------------
# TraceBufferConfig
# ---------------------------------------------------------------------------


def test_buffer_defaults() -> None:
    bc = TraceBufferConfig()
    assert bc.max_tokens == 8_000_000
    assert bc.max_age_days == 14.0


def test_buffer_zero_tokens() -> None:
    with pytest.raises(ConfigError):
        TraceBufferConfig(max_tokens=0)


# ---------------------------------------------------------------------------
# RedactionConfig
# ---------------------------------------------------------------------------


def test_redaction_enabled_by_default() -> None:
    assert RedactionConfig().enabled is True
    assert SpeedLMConfig(model="model").redaction.enabled is True


def test_redaction_can_be_disabled_and_round_trips() -> None:
    cfg = SpeedLMConfig(model="model", redaction=RedactionConfig(enabled=False))
    assert SpeedLMConfig.from_dict(cfg.to_dict()).redaction.enabled is False


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_redaction_enabled_requires_bool(value: object) -> None:
    with pytest.raises(ConfigError, match="redaction.enabled must be a bool"):
        RedactionConfig(enabled=value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PromotionConfig
# ---------------------------------------------------------------------------


def test_promotion_defaults() -> None:
    pc = PromotionConfig()
    assert pc.min_acceptance_delta_pp == 1.0
    assert pc.min_throughput_delta_pct == 2.0


def test_idle_tuning_config_round_trip_without_machine_specific_paths() -> None:
    tuning = IdleTuningConfig(
        min_trace_records=64,
        speculators_repo="/opt/speculators",
        training_python="/opt/training/bin/python",
        held_out_fraction=0.25,
    )
    config = SpeedLMConfig(model="org/model", tuning=tuning)

    restored = SpeedLMConfig.from_dict(config.to_dict())

    assert restored.tuning == tuning
    assert restored.tuning.speculators_repo == "/opt/speculators"


@pytest.mark.parametrize(
    "tuning",
    [
        {"min_trace_records": 1},
        {"held_out_fraction": 0},
        {"benchmark_repeats": 2},
        {"speculators_repo": ""},
        {"learning_rate": 1e-4},
    ],
)
def test_invalid_idle_tuning_config_is_rejected(tuning: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({"model": "org/model", "tuning": tuning})


# ---------------------------------------------------------------------------
# SpeedLMConfig
# ---------------------------------------------------------------------------


def test_alias_resolution() -> None:
    cfg = SpeedLMConfig(model="meta-llama/Llama-2-7b")
    assert cfg.alias == "meta-llama/Llama-2-7b"
    cfg2 = SpeedLMConfig(model="meta-llama/Llama-2-7b", model_alias="llama-2")
    assert cfg2.alias == "llama-2"


def test_profile_reference() -> None:
    cfg = SpeedLMConfig(model="hf/model", profile="custom-eagle3")
    assert cfg.profile == "custom-eagle3"
    assert SpeedLMConfig.from_dict(cfg.to_dict()).profile == "custom-eagle3"


@pytest.mark.parametrize("value", ["", 1, False])
def test_invalid_profile_reference(value: object) -> None:
    with pytest.raises(ConfigError, match="profile must be a non-empty string or null"):
        SpeedLMConfig(model="hf/model", profile=value)  # type: ignore[arg-type]


def test_round_trip() -> None:
    cfg = SpeedLMConfig(
        model="bigscience/bloom",
        model_alias="bloom",
        idle_threshold_seconds=600.0,
    )
    d = cfg.to_dict()
    cfg2 = SpeedLMConfig.from_dict(d)
    assert cfg2.to_dict() == d


def test_missing_model() -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({})


def test_unknown_top_level_key() -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({"model": "x", "weird": 1})


def test_unknown_nested_key() -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({"model": "x", "target": {"host": "a", "port": 80, "bogus": 1}})


def test_nested_must_be_mapping() -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({"model": "x", "target": "not-a-mapping"})


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.json")


def test_load_config_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text('[1, 2, 3]', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg = SpeedLMConfig(
        model="hf/model",
        model_alias="m",
        idle_threshold_seconds=120.0,
    )
    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()
