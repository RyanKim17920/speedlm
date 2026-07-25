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
    min_acceptance_delta_pp: float = 1.0
    min_throughput_delta_pct: float = 2.0

    def __post_init__(self) -> None:
        _validate_float_gte(self.min_acceptance_delta_pp, "min_acceptance_delta_pp", 0)
        _validate_float_gte(self.min_throughput_delta_pct, "min_throughput_delta_pct", 0)


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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    idle_threshold_seconds: float = 300.0
    startup_timeout_seconds: float = field(default_factory=startup_timeout_seconds)
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
            "sampling",
            "idle_threshold_seconds",
            "startup_timeout_seconds",
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
            sampling=SamplingConfig(**sampling_data),
            idle_threshold_seconds=data.get("idle_threshold_seconds", 300.0),
            startup_timeout_seconds=data.get(
                "startup_timeout_seconds", startup_timeout_seconds()
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
