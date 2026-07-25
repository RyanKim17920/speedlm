"""Speculative-decoding model profiles.

Profiles make the relationship between a verifier, its draft mechanism, and
the serving/training parameters explicit.  Built-ins are defaults only:
operators can replace them by name with JSON files in
``<SPEEDLM_HOME>/profiles``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

from speedlm.config import speedlm_home

SpeculativeMethod = Literal["eagle3", "mtp", "medusa", "ngram", "draft_model"]
ChatTemplateKind = Literal["harmony", "chatml", "auto"]

SPECULATIVE_METHODS: Final = frozenset(
    {"eagle3", "mtp", "medusa", "ngram", "draft_model"}
)
CHAT_TEMPLATE_KINDS: Final = frozenset({"harmony", "chatml", "auto"})
NON_TRAINABLE_METHODS: Final = frozenset({"ngram"})


class ProfileError(ValueError):
    """Raised when a model profile cannot be loaded or resolved safely."""


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Complete, immutable contract for one speculative-decoding setup."""

    name: str
    verifier_model: str
    draft_model: str | None
    speculative_method: SpeculativeMethod
    num_speculative_tokens: int
    target_layer_ids: tuple[int, ...] | None
    chat_template_kind: ChatTemplateKind
    max_seq_len: int
    trainable: bool = field(init=False)

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "name")
        _non_empty_string(self.verifier_model, "verifier_model")
        if self.draft_model is not None:
            _non_empty_string(self.draft_model, "draft_model")
        if self.speculative_method not in SPECULATIVE_METHODS:
            allowed = ", ".join(sorted(SPECULATIVE_METHODS))
            raise ProfileError(
                f"speculative_method must be one of: {allowed}; "
                f"got {self.speculative_method!r}"
            )
        _positive_int(self.num_speculative_tokens, "num_speculative_tokens")
        _positive_int(self.max_seq_len, "max_seq_len")
        if self.chat_template_kind not in CHAT_TEMPLATE_KINDS:
            allowed = ", ".join(sorted(CHAT_TEMPLATE_KINDS))
            raise ProfileError(
                f"chat_template_kind must be one of: {allowed}; "
                f"got {self.chat_template_kind!r}"
            )
        if self.target_layer_ids is not None and (
                not isinstance(self.target_layer_ids, tuple)
                or not self.target_layer_ids
                or any(
                    isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                    for layer in self.target_layer_ids
                )
                or len(set(self.target_layer_ids)) != len(self.target_layer_ids)
        ):
            raise ProfileError(
                "target_layer_ids must be null or unique non-negative integers"
            )
        if self.speculative_method == "eagle3" and not self.target_layer_ids:
            raise ProfileError("target_layer_ids are required for eagle3 profiles")
        if (
            self.speculative_method in {"eagle3", "medusa", "draft_model"}
            and self.draft_model is None
        ):
            raise ProfileError(
                f"draft_model is required for {self.speculative_method} profiles"
            )

        object.__setattr__(
            self,
            "trainable",
            self.speculative_method not in NON_TRAINABLE_METHODS,
        )

    @property
    def serve_benchmark_only(self) -> bool:
        """Whether this profile may be served/benchmarked but never tuned."""

        return not self.trainable

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation of this profile."""

        return {
            "name": self.name,
            "verifier_model": self.verifier_model,
            "draft_model": self.draft_model,
            "speculative_method": self.speculative_method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "target_layer_ids": (
                list(self.target_layer_ids) if self.target_layer_ids is not None else None
            ),
            "chat_template_kind": self.chat_template_kind,
            "max_seq_len": self.max_seq_len,
            "trainable": self.trainable,
        }

    def speculative_config(self) -> dict[str, object]:
        """Build the vLLM speculative-config fragment for this profile."""

        result: dict[str, object] = {
            "method": self.speculative_method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.draft_model is not None and self.speculative_method in {
            "eagle3",
            "medusa",
            "draft_model",
        }:
            result["model"] = self.draft_model
        return result

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source: str = "profile",
    ) -> ModelProfile:
        """Validate and construct a profile from a JSON-like mapping."""

        if not isinstance(data, Mapping):
            raise ProfileError(f"{source}: profile must be a JSON object")

        required = {
            "name",
            "verifier_model",
            "draft_model",
            "speculative_method",
            "num_speculative_tokens",
            "chat_template_kind",
            "max_seq_len",
        }
        allowed = required | {"target_layer_ids", "trainable"}
        missing = required - set(data)
        if missing:
            raise ProfileError(f"{source}: missing required keys: {', '.join(sorted(missing))}")
        unknown = set(data) - allowed
        if unknown:
            raise ProfileError(f"{source}: unknown keys: {', '.join(sorted(unknown))}")

        draft_model_value = data["draft_model"]
        if draft_model_value is not None:
            draft_model_value = _non_empty_string(draft_model_value, "draft_model")

        method_value = data["speculative_method"]
        if not isinstance(method_value, str) or method_value not in SPECULATIVE_METHODS:
            allowed_methods = ", ".join(sorted(SPECULATIVE_METHODS))
            raise ProfileError(
                f"{source}: speculative_method must be one of: {allowed_methods}; "
                f"got {method_value!r}"
            )

        template_value = data["chat_template_kind"]
        if not isinstance(template_value, str) or template_value not in CHAT_TEMPLATE_KINDS:
            allowed_templates = ", ".join(sorted(CHAT_TEMPLATE_KINDS))
            raise ProfileError(
                f"{source}: chat_template_kind must be one of: {allowed_templates}; "
                f"got {template_value!r}"
            )

        layer_value = data.get("target_layer_ids")
        target_layer_ids: tuple[int, ...] | None
        if layer_value is None:
            target_layer_ids = None
        elif isinstance(layer_value, list):
            target_layer_ids = tuple(layer_value)
        else:
            raise ProfileError(f"{source}: target_layer_ids must be an array or null")

        try:
            profile = cls(
                name=_non_empty_string(data["name"], "name"),
                verifier_model=_non_empty_string(
                    data["verifier_model"], "verifier_model"
                ),
                draft_model=cast(str | None, draft_model_value),
                speculative_method=cast(SpeculativeMethod, method_value),
                num_speculative_tokens=_positive_int(
                    data["num_speculative_tokens"], "num_speculative_tokens"
                ),
                target_layer_ids=target_layer_ids,
                chat_template_kind=cast(ChatTemplateKind, template_value),
                max_seq_len=_positive_int(data["max_seq_len"], "max_seq_len"),
            )
        except ProfileError as exc:
            raise ProfileError(f"{source}: {exc}") from exc

        declared_trainable = data.get("trainable")
        if declared_trainable is not None:
            if not isinstance(declared_trainable, bool):
                raise ProfileError(f"{source}: trainable must be a boolean")
            if declared_trainable is not profile.trainable:
                raise ProfileError(
                    f"{source}: trainable must be {profile.trainable} for "
                    f"method {profile.speculative_method!r}"
                )
        return profile


GPT_OSS_EAGLE3_PROFILE: Final = ModelProfile(
    name="gpt-oss-20b-eagle3",
    verifier_model="openai/gpt-oss-20b",
    draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
    speculative_method="eagle3",
    num_speculative_tokens=5,
    target_layer_ids=(2, 12, 21),
    chat_template_kind="harmony",
    max_seq_len=131_072,
)

LLAMA_31_8B_EAGLE3_PROFILE: Final = ModelProfile(
    name="llama-3.1-8b-instruct-eagle3",
    verifier_model="meta-llama/Llama-3.1-8B-Instruct",
    draft_model="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
    speculative_method="eagle3",
    num_speculative_tokens=5,
    target_layer_ids=(2, 12, 21),
    chat_template_kind="auto",
    max_seq_len=131_072,
)

QWEN_35_9B_MTP_PROFILE: Final = ModelProfile(
    name="qwen3.5-9b-mtp",
    verifier_model="Qwen/Qwen3.5-9B",
    draft_model=None,
    speculative_method="mtp",
    num_speculative_tokens=3,
    target_layer_ids=None,
    chat_template_kind="chatml",
    max_seq_len=262_144,
)

BUILTIN_PROFILES: Final[Mapping[str, ModelProfile]] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            GPT_OSS_EAGLE3_PROFILE,
            LLAMA_31_8B_EAGLE3_PROFILE,
            QWEN_35_9B_MTP_PROFILE,
        )
    }
)


def load_profiles(home: Path | None = None) -> dict[str, ModelProfile]:
    """Load built-ins plus validated user JSON profiles.

    User profiles override a built-in only when their ``name`` is identical.
    Any malformed file aborts the entire load; no partial registry is returned.
    """

    profiles = dict(BUILTIN_PROFILES)
    profiles_dir = (home if home is not None else speedlm_home()) / "profiles"
    if not profiles_dir.exists():
        return profiles
    if not profiles_dir.is_dir():
        raise ProfileError(f"profile path is not a directory: {profiles_dir}")

    user_profile_names: set[str] = set()
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{path}: invalid JSON: {exc.msg}") from exc
        except OSError as exc:
            raise ProfileError(f"{path}: cannot read profile: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProfileError(f"{path}: profile must be a JSON object")
        profile = ModelProfile.from_dict(raw, source=str(path))
        if profile.name in user_profile_names:
            raise ProfileError(
                f"{path}: duplicate user profile name {profile.name!r}"
            )
        user_profile_names.add(profile.name)
        profiles[profile.name] = profile
    return profiles


class ProfileConfig(Protocol):
    """Minimal config shape accepted by :func:`resolve_profile`."""

    model: str
    profile: str | None


def _config_value(
    config: ProfileConfig | Mapping[str, object],
    key: str,
) -> object | None:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _available_profiles(profiles: Mapping[str, ModelProfile]) -> str:
    return ", ".join(
        f"{name} ({profile.verifier_model})"
        for name, profile in sorted(profiles.items())
    )


def _match_profile(
    name_or_verifier: str,
    profiles: Mapping[str, ModelProfile],
) -> ModelProfile | None:
    by_name = profiles.get(name_or_verifier)
    if by_name is not None:
        return by_name
    verifier_matches = [
        profile
        for profile in profiles.values()
        if profile.verifier_model == name_or_verifier
    ]
    if len(verifier_matches) > 1:
        names = ", ".join(sorted(profile.name for profile in verifier_matches))
        raise ProfileError(
            f"multiple profiles match verifier {name_or_verifier!r}: {names}; "
            "set config.profile explicitly"
        )
    return verifier_matches[0] if verifier_matches else None


def resolve_profile(
    config: ProfileConfig | Mapping[str, object] | None = None,
    served_model: str | None = None,
    *,
    profiles: Mapping[str, ModelProfile] | None = None,
    home: Path | None = None,
) -> ModelProfile:
    """Resolve a profile without ever guessing an unknown verifier's draft.

    An explicit ``config.profile`` wins.  Otherwise the served model, followed
    by ``config.model``, must exactly match a profile name or verifier model.
    """

    registry = profiles if profiles is not None else load_profiles(home)
    if not registry:
        raise ProfileError("no model profiles are available")

    explicit_profile: object | None = None
    config_model: object | None = None
    if config is not None:
        explicit_profile = _config_value(config, "profile")
        config_model = _config_value(config, "model")

    if explicit_profile is not None:
        if not isinstance(explicit_profile, str) or not explicit_profile:
            raise ProfileError("config.profile must be a non-empty string or null")
        match = registry.get(explicit_profile)
        if match is None:
            raise ProfileError(
                f"unknown explicit profile {explicit_profile!r}; "
                f"available profiles: {_available_profiles(registry)}"
            )
        return match

    candidates: list[str] = []
    if served_model is not None:
        if not isinstance(served_model, str) or not served_model:
            raise ProfileError("served_model must be a non-empty string or null")
        candidates.append(served_model)
    if config_model is not None:
        if not isinstance(config_model, str) or not config_model:
            raise ProfileError("config.model must be a non-empty string")
        if config_model not in candidates:
            candidates.append(config_model)

    for candidate in candidates:
        match = _match_profile(candidate, registry)
        if match is not None:
            return match

    attempted = ", ".join(repr(candidate) for candidate in candidates) or "none"
    raise ProfileError(
        f"no model profile matches {attempted}; set config.profile explicitly. "
        f"Available profiles: {_available_profiles(registry)}"
    )
