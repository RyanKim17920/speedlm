from __future__ import annotations

from pathlib import Path

import pytest

from speedlm.config import (
    MAX_LEARNING_RATE,
    REFERENCE_LEARNING_RATE,
    ConfigError,
    DivergenceCriterion,
    IdleTuningConfig,
    PromotionConfig,
    RedactionConfig,
    SamplingConfig,
    SpeedLMConfig,
    TargetConfig,
    TraceBufferConfig,
    WrapperConfig,
    classify_divergence_criterion,
    load_config,
    save_config,
    speedlm_home,
    startup_stall_seconds,
    startup_timeout_seconds,
)
from speedlm.gateway.control import DEFAULT_RESTORE_FAST_PATH_TIMEOUT_SECONDS

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
    # Negative by design: throughput is a regression guard, not the promotion
    # criterion.  Requiring a *positive* throughput delta is unachievable at
    # the gate's sample size (one-sided 95% MDE ~3.0% at three repeats against
    # a measured 1.43% standard error), so it would be cleared by noise rather
    # than by merit.  See PromotionConfig's docstring for the derivation.
    assert pc.min_throughput_delta_pct == -2.0


def test_promotion_rejects_a_zeroed_gate_being_mistaken_for_a_default() -> None:
    """A 0.0/0.0 gate is loadable but is emphatically not what ships."""
    zeroed = PromotionConfig(min_acceptance_delta_pp=0.0, min_throughput_delta_pct=0.0)
    default = PromotionConfig()
    assert zeroed != default
    assert default.min_acceptance_delta_pp > 0.0


def test_promotion_accepts_a_negative_throughput_regression_floor() -> None:
    pc = PromotionConfig(min_throughput_delta_pct=-5.0)
    assert pc.min_throughput_delta_pct == -5.0


def test_promotion_rejects_a_non_finite_throughput_floor() -> None:
    with pytest.raises(ConfigError):
        PromotionConfig(min_throughput_delta_pct=float("nan"))
    with pytest.raises(ConfigError):
        PromotionConfig(min_throughput_delta_pct=float("-inf"))


def test_promotion_still_rejects_a_negative_acceptance_bar() -> None:
    """Acceptance must improve; a negative bar would license a regression."""
    with pytest.raises(ConfigError):
        PromotionConfig(min_acceptance_delta_pp=-0.5)


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
        {"benchmark_concurrency": 0},
        {"benchmark_concurrency": "eight"},
        {"restore_fast_path_timeout_seconds": 0},
        {"restore_fast_path_timeout_seconds": "fast"},
        {"speculators_repo": ""},
        {"learning_rate": 0},
        {"learning_rate": -1e-4},
        {"learning_rate": 1.1e-3},
        {"learning_rate": 1.0},
        {"learning_rate": True},
        {"learning_rate": "1e-4"},
    ],
)
def test_invalid_idle_tuning_config_is_rejected(tuning: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        SpeedLMConfig.from_dict({"model": "org/model", "tuning": tuning})


def test_reference_learning_rate_is_accepted_and_is_the_default() -> None:
    """1e-4 -- the Speculators reference ``--lr`` -- must load.

    The old ceiling of 1e-5 rejected it, and at 1e-5 the AdamW update over a
    cycle's ~202 steps was below half a bf16 ULP: every RMSNorm tensor in the
    published artifacts moved exactly 0.000000.  Training that cannot move a
    weight is the regression this test exists to prevent, so both halves are
    pinned -- the reference value loads, and it is what you get by default.
    """
    assert REFERENCE_LEARNING_RATE == 1e-4
    assert SpeedLMConfig(model="org/model").tuning.learning_rate == 1e-4

    configured = SpeedLMConfig.from_dict(
        {"model": "org/model", "tuning": {"learning_rate": REFERENCE_LEARNING_RATE}}
    )
    assert configured.tuning.learning_rate == 1e-4
    assert (
        SpeedLMConfig.from_dict(configured.to_dict()).tuning.learning_rate == 1e-4
    )


def test_learning_rate_ceiling_leaves_headroom_above_the_reference() -> None:
    """The bound is a fat-finger guard, not a bound sitting on the default.

    A ceiling equal to the value everyone configures is indistinguishable from
    no headroom: the first sweep above the reference would hit it.  Pin that
    the boundary itself loads, that a decade of room exists above 1e-4, and
    that the classic dropped-exponent typos are still refused.
    """
    assert MAX_LEARNING_RATE == 1e-3
    assert MAX_LEARNING_RATE >= REFERENCE_LEARNING_RATE * 10

    at_bound = SpeedLMConfig.from_dict(
        {"model": "org/model", "tuning": {"learning_rate": MAX_LEARNING_RATE}}
    )
    assert at_bound.tuning.learning_rate == MAX_LEARNING_RATE

    # A 5x sweep around the reference stays inside the guard.
    for rate in (2e-4, 5e-4):
        assert (
            SpeedLMConfig.from_dict(
                {"model": "org/model", "tuning": {"learning_rate": rate}}
            ).tuning.learning_rate
            == rate
        )

    # ...while a dropped or shifted exponent does not.
    for typo in (1e-1, 1.0):
        with pytest.raises(ConfigError, match="learning_rate"):
            SpeedLMConfig.from_dict(
                {"model": "org/model", "tuning": {"learning_rate": typo}}
            )


def test_benchmark_concurrency_round_trips_and_defaults_above_one() -> None:
    """The gate replay was serial until this knob existed; 1 must be opt-in."""
    default = SpeedLMConfig(model="org/model")
    assert default.tuning.benchmark_concurrency > 1

    configured = SpeedLMConfig.from_dict(
        {"model": "org/model", "tuning": {"benchmark_concurrency": 3}}
    )
    assert configured.tuning.benchmark_concurrency == 3
    assert (
        SpeedLMConfig.from_dict(configured.to_dict()).tuning.benchmark_concurrency
        == 3
    )


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


# ---------------------------------------------------------------------------
# IdleTuningConfig
# ---------------------------------------------------------------------------


def test_idle_tuning_config_default_min_corpus_records() -> None:
    cfg = IdleTuningConfig()
    assert cfg.min_corpus_records == 256
    assert cfg.min_trace_records == 32
    assert cfg.training_window_records == 256


def test_idle_tuning_config_window_must_be_at_least_corpus() -> None:
    with pytest.raises(ConfigError, match="training_window_records"):
        IdleTuningConfig(training_window_records=128, min_corpus_records=256)


def test_idle_tuning_config_corpus_too_small() -> None:
    with pytest.raises(ConfigError, match="min_corpus_records"):
        IdleTuningConfig(min_corpus_records=1)


def test_divergence_threshold_round_trips_through_config() -> None:
    config = SpeedLMConfig.from_dict(
        {
            "model": "m",
            "promotion": {"min_divergence_token_index": 32},
            "tuning": {"correctness_max_tokens": 64},
        }
    )

    assert config.promotion.min_divergence_token_index == 32
    assert config.tuning.correctness_max_tokens == 64
    assert config.to_dict()["promotion"]["min_divergence_token_index"] == 32
    assert config.to_dict()["tuning"]["correctness_max_tokens"] == 64


def test_divergence_threshold_defaults_are_the_documented_ones() -> None:
    config = SpeedLMConfig(model="m")

    assert config.promotion.min_divergence_token_index == 16
    assert config.tuning.correctness_max_tokens == 128


@pytest.mark.parametrize("value", [-1, "x", 1.5])
def test_invalid_divergence_threshold_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="min_divergence_token_index"):
        PromotionConfig(min_divergence_token_index=value)


@pytest.mark.parametrize("value", [0, -1, "x"])
def test_invalid_correctness_max_tokens_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="correctness_max_tokens"):
        IdleTuningConfig(correctness_max_tokens=value)


# --- the promotion/tuning threshold relationship ---------------------------
# Neither field is wrong on its own; it is the pair that decides whether the
# position criterion has a range to discriminate in.  Both degenerate ends are
# legitimate to configure, so they warn at load rather than raising -- job
# 369373 set them equal deliberately -- but they must not be silent.


@pytest.mark.parametrize(
    ("threshold", "cap", "expected"),
    [
        (16, 128, DivergenceCriterion.CALIBRATED),
        (127, 128, DivergenceCriterion.CALIBRATED),
        (128, 128, DivergenceCriterion.SATURATED),
        (256, 128, DivergenceCriterion.SATURATED),
        (0, 128, DivergenceCriterion.DISABLED),
        # An archived decision predating the cap field: only the disabled end
        # is decidable without a range.
        (16, None, DivergenceCriterion.CALIBRATED),
        (0, None, DivergenceCriterion.DISABLED),
    ],
)
def test_the_divergence_criterion_is_classified_at_both_ends(
    threshold: int,
    cap: int | None,
    expected: DivergenceCriterion,
) -> None:
    assert classify_divergence_criterion(threshold, cap) is expected


def test_a_saturated_divergence_threshold_warns_at_config_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Job 369373: threshold 128 against a cap of 128 rejects everything.

    Every divergence at every observable offset classifies early, so with
    ``max_output_mismatches = 0`` the gate demands a bitwise identical
    generation -- which non-reproducible hardware will not produce.
    """
    with caplog.at_level("WARNING", logger="speedlm.config"):
        SpeedLMConfig.from_dict(
            {
                "model": "m",
                "promotion": {"min_divergence_token_index": 128},
                "tuning": {"correctness_max_tokens": 128},
            }
        )

    assert "every divergence" in caplog.text
    assert "min_divergence_token_index" in caplog.text


def test_a_disabled_divergence_threshold_warns_at_config_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other end matters too: the gate is the only safeguard."""
    with caplog.at_level("WARNING", logger="speedlm.config"):
        SpeedLMConfig.from_dict(
            {"model": "m", "promotion": {"min_divergence_token_index": 0}}
        )

    assert "disabled" in caplog.text


def test_a_calibrated_divergence_threshold_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="speedlm.config"):
        SpeedLMConfig(model="m")

    assert caplog.text == ""


def test_a_degenerate_threshold_relationship_is_not_fatal() -> None:
    """The 369373 experiment must stay runnable; the warning is the safeguard."""
    config = SpeedLMConfig.from_dict(
        {
            "model": "m",
            "promotion": {"min_divergence_token_index": 128},
            "tuning": {"correctness_max_tokens": 128},
        }
    )

    assert config.promotion.min_divergence_token_index == 128


def test_benchmark_max_tokens_round_trips_and_defaults_to_the_served_cap() -> None:
    """512 is the cap the live harness puts on production traffic."""
    default = SpeedLMConfig(model="m")
    assert default.tuning.benchmark_max_tokens == 512

    configured = SpeedLMConfig.from_dict(
        {"model": "org/model", "tuning": {"benchmark_max_tokens": 256}}
    )
    assert configured.tuning.benchmark_max_tokens == 256
    assert configured.to_dict()["tuning"]["benchmark_max_tokens"] == 256
    assert (
        SpeedLMConfig.from_dict(configured.to_dict()).tuning.benchmark_max_tokens == 256
    )


@pytest.mark.parametrize("value", [0, -1, "x", 1.5])
def test_invalid_benchmark_max_tokens_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="benchmark_max_tokens"):
        IdleTuningConfig(benchmark_max_tokens=value)


# ---------------------------------------------------------------------------
# The two concurrency knobs
# ---------------------------------------------------------------------------


def test_the_legacy_concurrency_key_is_rejected_with_both_alternatives() -> None:
    """A config that lies about what ran must not load.

    ``tuning.concurrency`` never reached the gate -- it is the Speculators
    extraction degree -- but job 369006 set it to 4 and the archived config was
    read as a statement about gate replay, which actually ran at 8.  Loading it
    silently is what made that analysis nearly go wrong, so the key now fails
    validation and the error names both knobs it could have meant.
    """
    with pytest.raises(ConfigError) as excinfo:
        SpeedLMConfig.from_dict(
            {"model": "org/model", "tuning": {"concurrency": 4}}
        )

    message = str(excinfo.value)
    assert "tuning.concurrency" in message
    assert "extraction_concurrency" in message
    assert "benchmark_concurrency" in message


def test_the_two_concurrency_knobs_are_independent() -> None:
    config = SpeedLMConfig.from_dict(
        {
            "model": "org/model",
            "tuning": {"extraction_concurrency": 4, "benchmark_concurrency": 8},
        }
    )

    assert config.tuning.extraction_concurrency == 4
    assert config.tuning.benchmark_concurrency == 8
    dumped = config.to_dict()["tuning"]
    assert dumped["extraction_concurrency"] == 4
    assert dumped["benchmark_concurrency"] == 8
    # Round-tripping a dump must not resurrect the ambiguous key.
    assert "concurrency" not in dumped
    assert SpeedLMConfig.from_dict(config.to_dict()).tuning.extraction_concurrency == 4


@pytest.mark.parametrize("value", [0, -1, "x", True])
def test_invalid_extraction_concurrency_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="extraction_concurrency"):
        IdleTuningConfig(extraction_concurrency=value)


# ---------------------------------------------------------------------------
# Downtime guards: idle confirmation, retry cooldown, benchmark arm order
# ---------------------------------------------------------------------------


def test_scheduling_guards_default_to_the_documented_values() -> None:
    tuning = SpeedLMConfig(model="m").tuning
    assert tuning.idle_confirmations == 3
    assert tuning.retry_cooldown_seconds == 600.0
    assert tuning.benchmark_candidate_arm_first is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("idle_confirmations", 5),
        ("retry_cooldown_seconds", 30.0),
        ("benchmark_candidate_arm_first", False),
    ],
)
def test_scheduling_guards_round_trip(key: str, value: object) -> None:
    configured = SpeedLMConfig.from_dict({"model": "org/model", "tuning": {key: value}})
    assert getattr(configured.tuning, key) == value
    assert configured.to_dict()["tuning"][key] == value
    assert getattr(SpeedLMConfig.from_dict(configured.to_dict()).tuning, key) == value


@pytest.mark.parametrize("value", [0, -1, "x", 1.5, True])
def test_invalid_idle_confirmations_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="idle_confirmations"):
        IdleTuningConfig(idle_confirmations=value)


@pytest.mark.parametrize("value", [-1.0, "x", None])
def test_invalid_retry_cooldown_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="retry_cooldown_seconds"):
        IdleTuningConfig(retry_cooldown_seconds=value)


def test_a_zero_retry_cooldown_is_allowed_to_restore_the_old_behaviour() -> None:
    assert IdleTuningConfig(retry_cooldown_seconds=0.0).retry_cooldown_seconds == 0.0


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_invalid_benchmark_arm_order_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="benchmark_candidate_arm_first"):
        IdleTuningConfig(benchmark_candidate_arm_first=value)


def test_restore_fast_path_timeout_is_configurable_and_defaults_unchanged() -> None:
    """The knob existed on the controller but nothing could reach it.

    Composition built ``RuntimeController`` without it, so the fast path was
    pinned at the controller's own 120s default with no config key at all.  The
    default must stay 120s -- wiring it up is not allowed to move behaviour --
    and an operator on a slower host must now be able to raise it.
    """
    assert IdleTuningConfig().restore_fast_path_timeout_seconds == 120.0
    assert (
        IdleTuningConfig().restore_fast_path_timeout_seconds
        == DEFAULT_RESTORE_FAST_PATH_TIMEOUT_SECONDS
    )

    config = SpeedLMConfig.from_dict(
        {
            "model": "org/model",
            "tuning": {"restore_fast_path_timeout_seconds": 300.0},
        }
    )

    assert config.tuning.restore_fast_path_timeout_seconds == 300.0
    assert config.to_dict()["tuning"]["restore_fast_path_timeout_seconds"] == 300.0
    assert (
        SpeedLMConfig.from_dict(config.to_dict()).tuning
        == config.tuning
    )


def test_warmup_repeats_round_trips_through_config() -> None:
    """The knob that was previously reachable only as a runner constructor default."""
    config = SpeedLMConfig.from_dict(
        {"model": "m", "tuning": {"warmup_repeats": 4, "benchmark_repeats": 12}}
    )

    assert config.tuning.warmup_repeats == 4
    assert config.tuning.benchmark_repeats == 12
    assert config.to_dict()["tuning"]["warmup_repeats"] == 4
    assert config.to_dict()["tuning"]["benchmark_repeats"] == 12


def test_warmup_repeats_default_matches_what_production_already_ran() -> None:
    """Surfacing the knob must not move the measurement it exposes."""
    assert SpeedLMConfig(model="m").tuning.warmup_repeats == 1


def test_warmup_repeats_may_be_disabled_but_not_negative() -> None:
    assert IdleTuningConfig(warmup_repeats=0).warmup_repeats == 0
    with pytest.raises(ConfigError, match="tuning.warmup_repeats"):
        IdleTuningConfig(warmup_repeats=-1)


@pytest.mark.parametrize("value", ["x", 1.5, True])
def test_invalid_warmup_repeats_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="tuning.warmup_repeats"):
        IdleTuningConfig(warmup_repeats=value)


def test_benchmark_repeats_is_unbounded_above_for_characterisation_runs() -> None:
    """The diagnostic mode is a config value, not a separate code path.

    Finding where the warming curve flattens needs scored repeats past five,
    and nothing may cap that: a run configured for twenty is the measurement,
    and the production default of five is untouched by its existence.
    """
    assert IdleTuningConfig(benchmark_repeats=20).benchmark_repeats == 20
    assert SpeedLMConfig(model="m").tuning.benchmark_repeats == 5


# ---------------------------------------------------------------------------
# Compounding warm start
# ---------------------------------------------------------------------------


def test_compounding_warm_start_is_on_by_default_and_unbounded() -> None:
    """The default is the argued one, so a change to it is a visible diff.

    Every archived realistic-data run trained a professionally built speculator
    from a standing start over ~409 records and lost to it -- so re-running that
    from-stock fine-tune every cycle is the configuration with the measured
    negative record, and compounding is the only one in which the product's
    premise is testable at all.  The bound is null because no compounding cycle
    has ever run and any N would be invented; see the field docstrings.
    """
    tuning = SpeedLMConfig(model="m").tuning
    assert tuning.compounding_warm_start is True
    assert tuning.warm_start_max_chain_depth is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("compounding_warm_start", False),
        ("warm_start_max_chain_depth", 5),
    ],
)
def test_warm_start_knobs_round_trip(key: str, value: object) -> None:
    configured = SpeedLMConfig.from_dict({"model": "org/model", "tuning": {key: value}})
    assert getattr(configured.tuning, key) == value
    assert configured.to_dict()["tuning"][key] == value
    assert getattr(SpeedLMConfig.from_dict(configured.to_dict()).tuning, key) == value


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_invalid_compounding_warm_start_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="compounding_warm_start"):
        IdleTuningConfig(compounding_warm_start=value)


@pytest.mark.parametrize("value", [0, -1, "x", 1.5, True])
def test_invalid_warm_start_chain_bound_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError, match="warm_start_max_chain_depth"):
        IdleTuningConfig(warm_start_max_chain_depth=value)
