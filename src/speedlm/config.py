from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------
from speedlm.storage import atomic_write_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HOME_NAME = ".speedlm"
HOME_ENV_VAR = "SPEEDLM_HOME"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
STARTUP_TIMEOUT_ENV_VAR = "SPEEDLM_STARTUP_TIMEOUT_SECONDS"
DEFAULT_STARTUP_STALL_SECONDS = 600.0
STARTUP_STALL_ENV_VAR = "SPEEDLM_STARTUP_STALL_SECONDS"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _validate_host(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}.host must be a non-empty string, got {value!r}")
    return value


def _validate_port(value: Any, name: str) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(f"{name}.port must be an int, got {type(value).__name__!r}")
    if not (1 <= value <= 65535):
        raise ConfigError(f"{name}.port must be in 1..65535, got {value}")
    return value


def _validate_float_gte(value: Any, name: str, minimum: float) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {type(value).__name__!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def _validate_float(value: Any, name: str) -> float:
    """Validate a finite numeric value with no bound on either side."""
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {type(value).__name__!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {value}")
    return float(value)


def _validate_int_gte(value: Any, name: str, minimum: int) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an int, got {type(value).__name__!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def startup_timeout_seconds() -> float:
    """Return the configured vLLM startup hard ceiling.

    ``SPEEDLM_STARTUP_TIMEOUT_SECONDS`` overrides the default for process
    launches that do not load a model config.
    """
    raw = os.environ.get(STARTUP_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_STARTUP_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{STARTUP_TIMEOUT_ENV_VAR} must be numeric, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{STARTUP_TIMEOUT_ENV_VAR} must be > 0, got {raw!r}")
    return value


def startup_stall_seconds() -> float:
    """Return the configured vLLM startup liveness window.

    ``SPEEDLM_STARTUP_STALL_SECONDS`` overrides the default for process
    launches that do not load a model config.
    """
    raw = os.environ.get(STARTUP_STALL_ENV_VAR)
    if raw is None:
        return DEFAULT_STARTUP_STALL_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{STARTUP_STALL_ENV_VAR} must be numeric, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{STARTUP_STALL_ENV_VAR} must be > 0, got {raw!r}")
    return value


# ---------------------------------------------------------------------------
# speedlm_home
# ---------------------------------------------------------------------------


def speedlm_home() -> Path:
    """Return the SpeedLM home directory.

    Uses ``SPEEDLM_HOME`` environment variable if set (expanded and resolved
    to an absolute path), otherwise falls back to ``~/.speedlm``.
    Does **not** create any directories.
    """
    env = os.environ.get(HOME_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / DEFAULT_HOME_NAME


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if _is_bool(self.temperature) or not isinstance(self.temperature, (int, float)):
            raise ConfigError(
                f"temperature must be numeric, got {type(self.temperature).__name__!r}"
            )
        if self.temperature < 0:
            raise ConfigError(f"temperature must be >= 0, got {self.temperature}")
        if _is_bool(self.top_p) or not isinstance(self.top_p, (int, float)):
            raise ConfigError(f"top_p must be numeric, got {type(self.top_p).__name__!r}")
        if not (0 < self.top_p <= 1):
            raise ConfigError(f"top_p must be in (0, 1], got {self.top_p}")
        if _is_bool(self.seed) or not isinstance(self.seed, int):
            raise ConfigError(f"seed must be an int, got {type(self.seed).__name__!r}")
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")


@dataclass(frozen=True, slots=True)
class TargetConfig:
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        _validate_host(self.host, "target")
        _validate_port(self.port, "target")


@dataclass(frozen=True, slots=True)
class WrapperConfig:
    host: str = "127.0.0.1"
    port: int = 8100

    def __post_init__(self) -> None:
        _validate_host(self.host, "wrapper")
        _validate_port(self.port, "wrapper")


@dataclass(frozen=True, slots=True)
class TraceBufferConfig:
    max_tokens: int = 8_000_000
    max_age_days: float = 14.0

    def __post_init__(self) -> None:
        if _is_bool(self.max_tokens) or not isinstance(self.max_tokens, int):
            raise ConfigError(f"max_tokens must be an int, got {type(self.max_tokens).__name__!r}")
        if self.max_tokens <= 0:
            raise ConfigError(f"max_tokens must be > 0, got {self.max_tokens}")
        if _is_bool(self.max_age_days) or not isinstance(self.max_age_days, (int, float)):
            raise ConfigError(
                f"max_age_days must be numeric, got {type(self.max_age_days).__name__!r}"
            )
        if self.max_age_days <= 0:
            raise ConfigError(f"max_age_days must be > 0, got {self.max_age_days}")


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    """Privacy controls for newly persisted traces."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError(
                f"redaction.enabled must be a bool, got {type(self.enabled).__name__!r}"
            )


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Bars a candidate draft head must clear before it is promoted.

    The two knobs play *different* roles, and the defaults are derived from the
    measured dispersion of the live gate, not chosen for symmetry.

    ``min_acceptance_delta_pp`` is the **promotion criterion**.  Improving draft
    acceptance is the entire point of idle tuning, so a candidate that does not
    measurably raise acceptance is not worth shipping however good its clock
    looks.  Acceptance is read from vLLM's ``spec_decode`` counters over a
    deterministic, greedy replay of a fixed held-out suite, so the same suite
    replayed against two arms has *no* timing component and no measured noise:
    on job 368670 the two arms produced byte-identical draft/accept counters
    (1155 drafted, 730 accepted, delta 0.0 pp).  The noise floor is therefore
    the counter quantum itself -- one accepted token out of ~1155 drafted, or
    0.087 pp.  ``1.0`` pp is ~12 accepted tokens, a ~2% relative lift in
    acceptance: comfortably resolvable, and small enough that a genuinely
    better head clears it.

    ``min_throughput_delta_pct`` is a **regression guard**, deliberately
    negative.  Throughput *is* timing, and it is noisy: across the three scored
    repeats of job 368670 the within-arm standard deviation was 1.80 tok/s
    (stock) and 0.58 tok/s (candidate), pooled 1.34 tok/s on a ~76.7 tok/s
    mean, giving a standard error on the arm-to-arm delta of 1.43% at three
    repeats.  The one-sided 95% minimum detectable effect is ~3.0%, so any
    *positive* threshold below that cannot be earned on merit -- it is cleared
    by whichever way the timing noise happened to fall.  Job 368670 is the
    worked example: it measured +0.96% (t=0.67, p~0.28) and would have promoted
    under the old ``2.0`` bar roughly one run in seven by chance alone; drop its
    first scored repeat and the same data reads -0.78%.  Requiring throughput to
    *prove* an improvement is therefore not achievable at this sample size, so
    the gate instead requires only that throughput not visibly regress.  ``-2.0``
    sits ~1.8 standard errors below zero at five repeats (SE 1.10%), so ordinary
    jitter does not trip it, while the real regression this gate has already
    caught -- the un-warmed candidate arm of job 368648, at -19.2% -- is more
    than sixteen standard errors past it.

    Both values remain fully configurable via ``promotion`` in ``config.json``;
    these are defaults, not policy.  Note that setting them to ``0.0``/``0.0``
    reduces the gate to "not measurably worse", which promotes on noise
    indefinitely -- see DEMO.md on why lowering the gate is the failure mode
    this system exists to prevent.
    """

    min_acceptance_delta_pp: float = 1.0
    #: Negative by design: a floor on regression, not a required speedup.
    min_throughput_delta_pct: float = -2.0

    def __post_init__(self) -> None:
        _validate_float_gte(self.min_acceptance_delta_pp, "min_acceptance_delta_pp", 0)
        # No lower bound of zero here: a negative value is the intended
        # regression-guard form ("reject anything more than N% slower").
        _validate_float(self.min_throughput_delta_pct, "min_throughput_delta_pct")


@dataclass(frozen=True, slots=True)
class IdleTuningConfig:
    """Production composition settings for the opt-in idle tuner."""

    min_trace_records: int = 32
    poll_interval_seconds: float = 1.0
    held_out_fraction: float = 0.2
    #: Scored suite passes per arm.  Five, not three, because the gate's
    #: throughput regression guard is only as trustworthy as its standard
    #: error: job 368670's pooled within-arm dispersion of 1.34 tok/s on a
    #: ~76.7 tok/s mean puts the arm-to-arm standard error at 1.43% over three
    #: repeats but 1.10% over five, which moves the -2.0% guard from 1.4 to 1.8
    #: standard errors clear of zero.  The cost is four extra suite passes per
    #: tuning cycle -- on job 368670 a pass took ~8s, so ~32s added to a ~275s
    #: benchmark phase inside a ~1200s cycle (~3%).  Engine restarts, not
    #: repeats, dominate that phase.
    benchmark_repeats: int = 5
    #: How many of the newest trace records one cycle may train on.
    #:
    #: Trace selection is a sliding window, not a full rescan: without a bound
    #: every cycle re-extracted and re-trained on the entire corpus, which is
    #: why cycle 1 and cycle 2 of the archived runs produced byte-identical
    #: trace snapshots while the watermark advanced by a single record.  The
    #: window is what makes a cycle cost O(recent traffic) instead of
    #: O(everything ever captured).
    #:
    #: 256 is chosen against ``min_trace_records`` (32): eight arming
    #: thresholds of history is enough that the per-cycle training
    #: distribution is a stable sample of recent traffic rather than a
    #: high-variance snapshot, while still bounding hidden-state extraction --
    #: the stage the window actually pays for -- to a fixed ceiling.  Set to
    #: null to restore the unbounded full-corpus scan.
    training_window_records: int | None = 256
    speculators_repo: str | None = None
    training_python: str | None = None
    vllm_python: str | None = None
    prepared_validator_script: str | None = None
    sequence_length: int = 16_384
    learning_rate: float = 1e-5
    epochs: int = 1
    concurrency: int = 8
    training_port: int = 8_131
    scratch_quota_bytes: int = 5 * 1024 * 1024 * 1024
    shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _validate_int_gte(self.min_trace_records, "tuning.min_trace_records", 2)
        _validate_float_gte(
            self.poll_interval_seconds,
            "tuning.poll_interval_seconds",
            0.001,
        )
        if (
            isinstance(self.held_out_fraction, bool)
            or not isinstance(self.held_out_fraction, (int, float))
            or not 0 < self.held_out_fraction < 1
        ):
            raise ConfigError("tuning.held_out_fraction must be in (0, 1)")
        _validate_int_gte(self.benchmark_repeats, "tuning.benchmark_repeats", 3)
        if self.training_window_records is not None:
            _validate_int_gte(
                self.training_window_records,
                "tuning.training_window_records",
                self.min_trace_records,
            )
        for name, value in (
            ("speculators_repo", self.speculators_repo),
            ("training_python", self.training_python),
            ("vllm_python", self.vllm_python),
            ("prepared_validator_script", self.prepared_validator_script),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ConfigError(f"tuning.{name} must be a non-empty string or null")
        _validate_int_gte(self.sequence_length, "tuning.sequence_length", 1)
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not 0 < self.learning_rate <= 1e-5
        ):
            raise ConfigError("tuning.learning_rate must be in (0, 1e-5]")
        _validate_int_gte(self.epochs, "tuning.epochs", 1)
        _validate_int_gte(self.concurrency, "tuning.concurrency", 1)
        _validate_port(self.training_port, "tuning")
        _validate_int_gte(self.scratch_quota_bytes, "tuning.scratch_quota_bytes", 1)
        _validate_float_gte(
            self.shutdown_timeout_seconds,
            "tuning.shutdown_timeout_seconds",
            0.001,
        )


@dataclass(frozen=True, slots=True)
class SpeedLMConfig:
    model: str
    model_alias: str = ""
    profile: str | None = None
    target: TargetConfig = field(default_factory=TargetConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    buffer: TraceBufferConfig = field(default_factory=TraceBufferConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    tuning: IdleTuningConfig = field(default_factory=IdleTuningConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    idle_threshold_seconds: float = 300.0
    startup_timeout_seconds: float = field(default_factory=startup_timeout_seconds)
    startup_stall_seconds: float = field(default_factory=startup_stall_seconds)
    tuning_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ConfigError("model must be a non-empty string")
        if not isinstance(self.model_alias, str):
            raise ConfigError(
                f"model_alias must be a string, got {type(self.model_alias).__name__!r}"
            )
        if self.profile is not None and (
            not isinstance(self.profile, str) or not self.profile
        ):
            raise ConfigError(
                "profile must be a non-empty string or null, got "
                f"{self.profile!r}"
            )
        if _is_bool(self.idle_threshold_seconds) or not isinstance(
            self.idle_threshold_seconds, (int, float)
        ):
            raise ConfigError(
                "idle_threshold_seconds must be numeric, got "
                f"{type(self.idle_threshold_seconds).__name__!r}"
            )
        if self.idle_threshold_seconds <= 0:
            raise ConfigError(
                f"idle_threshold_seconds must be > 0, got {self.idle_threshold_seconds}"
            )
        if _is_bool(self.startup_timeout_seconds) or not isinstance(
            self.startup_timeout_seconds, (int, float)
        ):
            raise ConfigError(
                "startup_timeout_seconds must be numeric, got "
                f"{type(self.startup_timeout_seconds).__name__!r}"
            )
        if (
            not math.isfinite(self.startup_timeout_seconds)
            or self.startup_timeout_seconds <= 0
        ):
            raise ConfigError(
                f"startup_timeout_seconds must be > 0, got {self.startup_timeout_seconds}"
            )
        if _is_bool(self.startup_stall_seconds) or not isinstance(
            self.startup_stall_seconds, (int, float)
        ):
            raise ConfigError(
                "startup_stall_seconds must be numeric, got "
                f"{type(self.startup_stall_seconds).__name__!r}"
            )
        if (
            not math.isfinite(self.startup_stall_seconds)
            or self.startup_stall_seconds <= 0
        ):
            raise ConfigError(
                f"startup_stall_seconds must be > 0, got {self.startup_stall_seconds}"
            )
        if not isinstance(self.tuning_enabled, bool):
            raise ConfigError(
                f"tuning_enabled must be a bool, got {type(self.tuning_enabled).__name__!r}"
            )

    @property
    def alias(self) -> str:
        return self.model_alias if self.model_alias else self.model

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "model_alias": self.model_alias,
            "profile": self.profile,
            "idle_threshold_seconds": self.idle_threshold_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "startup_stall_seconds": self.startup_stall_seconds,
            "tuning_enabled": self.tuning_enabled,
        }
        result["target"] = {
            "host": self.target.host,
            "port": self.target.port,
        }
        result["wrapper"] = {
            "host": self.wrapper.host,
            "port": self.wrapper.port,
        }
        result["buffer"] = {
            "max_tokens": self.buffer.max_tokens,
            "max_age_days": self.buffer.max_age_days,
        }
        result["redaction"] = {
            "enabled": self.redaction.enabled,
        }
        result["promotion"] = {
            "min_acceptance_delta_pp": self.promotion.min_acceptance_delta_pp,
            "min_throughput_delta_pct": self.promotion.min_throughput_delta_pct,
        }
        result["tuning"] = {
            "min_trace_records": self.tuning.min_trace_records,
            "poll_interval_seconds": self.tuning.poll_interval_seconds,
            "held_out_fraction": self.tuning.held_out_fraction,
            "benchmark_repeats": self.tuning.benchmark_repeats,
            "training_window_records": self.tuning.training_window_records,
            "speculators_repo": self.tuning.speculators_repo,
            "training_python": self.tuning.training_python,
            "vllm_python": self.tuning.vllm_python,
            "prepared_validator_script": self.tuning.prepared_validator_script,
            "sequence_length": self.tuning.sequence_length,
            "learning_rate": self.tuning.learning_rate,
            "epochs": self.tuning.epochs,
            "concurrency": self.tuning.concurrency,
            "training_port": self.tuning.training_port,
            "scratch_quota_bytes": self.tuning.scratch_quota_bytes,
            "shutdown_timeout_seconds": self.tuning.shutdown_timeout_seconds,
        }
        result["sampling"] = {
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "seed": self.sampling.seed,
        }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpeedLMConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("config data must be a mapping")
        if "model" not in data:
            raise ConfigError("'model' is required")

        known_keys = {
            "model",
            "model_alias",
            "profile",
            "target",
            "wrapper",
            "buffer",
            "redaction",
            "promotion",
            "tuning",
            "sampling",
            "idle_threshold_seconds",
            "startup_timeout_seconds",
            "startup_stall_seconds",
            "tuning_enabled",
        }
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise ConfigError(f"unknown top-level keys: {', '.join(sorted(unknown))}")

        def _nested(mapping: Mapping[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
            if section in mapping:
                val = mapping[section]
                if not isinstance(val, Mapping):
                    raise ConfigError(f"{section} must be a mapping, got {type(val).__name__!r}")
                sub_unknown = set(val.keys()) - allowed
                if sub_unknown:
                    raise ConfigError(
                        f"unknown keys in {section}: {', '.join(sorted(sub_unknown))}"
                    )
                return cast(dict[str, Any], val)
            return {}

        target_data = _nested(data, "target", {"host", "port"})
        wrapper_data = _nested(data, "wrapper", {"host", "port"})
        buffer_data = _nested(data, "buffer", {"max_tokens", "max_age_days"})
        redaction_data = _nested(data, "redaction", {"enabled"})
        promotion_data = _nested(
            data, "promotion", {"min_acceptance_delta_pp", "min_throughput_delta_pct"}
        )
        tuning_data = _nested(
            data,
            "tuning",
            {
                "min_trace_records",
                "poll_interval_seconds",
                "held_out_fraction",
                "benchmark_repeats",
                "training_window_records",
                "speculators_repo",
                "training_python",
                "vllm_python",
                "prepared_validator_script",
                "sequence_length",
                "learning_rate",
                "epochs",
                "concurrency",
                "training_port",
                "scratch_quota_bytes",
                "shutdown_timeout_seconds",
            },
        )
        sampling_data = _nested(data, "sampling", {"temperature", "top_p", "seed"})

        return cls(
            model=data["model"],
            model_alias=data.get("model_alias", ""),
            profile=data.get("profile"),
            target=TargetConfig(**target_data),
            wrapper=WrapperConfig(**wrapper_data),
            buffer=TraceBufferConfig(**buffer_data),
            redaction=RedactionConfig(**redaction_data),
            promotion=PromotionConfig(**promotion_data),
            tuning=IdleTuningConfig(**tuning_data),
            sampling=SamplingConfig(**sampling_data),
            idle_threshold_seconds=data.get("idle_threshold_seconds", 300.0),
            startup_timeout_seconds=data.get(
                "startup_timeout_seconds", startup_timeout_seconds()
            ),
            startup_stall_seconds=data.get(
                "startup_stall_seconds", startup_stall_seconds()
            ),
            tuning_enabled=data.get("tuning_enabled", False),
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_config(path: Path) -> SpeedLMConfig:
    """Load a ``SpeedLMConfig`` from a JSON file.

    Raises ``ConfigError`` for missing files, invalid JSON, or non-object
    top-level values.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {path}") from exc

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(obj, dict):
        raise ConfigError(f"top-level JSON value must be an object, got {type(obj).__name__}")

    return SpeedLMConfig.from_dict(obj)


def save_config(config: SpeedLMConfig, path: Path) -> None:
    """Write ``config`` to *path* as JSON using an atomic rename."""
    atomic_write_json(path, config.to_dict())
