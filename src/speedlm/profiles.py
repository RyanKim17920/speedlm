"""Speculative-decoding model profiles.

Profiles make the relationship between a verifier, its draft mechanism, and
the serving/training parameters explicit.  Built-ins are defaults only:
operators can replace them by name with JSON files in
``<SPEEDLM_HOME>/profiles``.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

from speedlm.config import speedlm_home

SpeculativeMethod = Literal["eagle3", "mtp", "medusa", "ngram", "draft_model"]
ChatTemplateKind = Literal["harmony", "chatml", "auto"]
ParserSource = Literal[
    "auto-detected",
    "profile-pinned",
    "user-supplied",
    "none",
]

SPECULATIVE_METHODS: Final = frozenset(
    {"eagle3", "mtp", "medusa", "ngram", "draft_model"}
)
CHAT_TEMPLATE_KINDS: Final = frozenset({"harmony", "chatml", "auto"})
NON_TRAINABLE_METHODS: Final = frozenset({"ngram"})


class ProfileError(ValueError):
    """Raised when a model profile cannot be loaded or resolved safely."""


@dataclass(frozen=True, slots=True)
class ParserRegistry:
    """Names and lazy-registration metadata discovered from the installed vLLM."""

    tool_parsers: tuple[str, ...] = ()
    reasoning_parsers: tuple[str, ...] = ()
    tool_metadata: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    reasoning_metadata: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParserResolution:
    """Effective parser values plus the winning level of the override chain."""

    model_type: str | None
    registry: ParserRegistry
    tool_call_parser: str | None
    reasoning_parser: str | None
    tool_call_parser_source: ParserSource
    reasoning_parser_source: ParserSource


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
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
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
        if self.tool_call_parser is not None:
            _non_empty_string(self.tool_call_parser, "tool_call_parser")
        if self.reasoning_parser is not None:
            _non_empty_string(self.reasoning_parser, "reasoning_parser")
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
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
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
        allowed = required | {
            "target_layer_ids",
            "tool_call_parser",
            "reasoning_parser",
            "trainable",
        }
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

        tool_call_parser_value = data.get("tool_call_parser")
        if tool_call_parser_value is not None:
            tool_call_parser_value = _non_empty_string(
                tool_call_parser_value, "tool_call_parser"
            )

        reasoning_parser_value = data.get("reasoning_parser")
        if reasoning_parser_value is not None:
            reasoning_parser_value = _non_empty_string(
                reasoning_parser_value, "reasoning_parser"
            )

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
                tool_call_parser=cast(str | None, tool_call_parser_value),
                reasoning_parser=cast(str | None, reasoning_parser_value),
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
    tool_call_parser="openai",
    reasoning_parser="openai_gptoss",
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
    tool_call_parser="hermes",
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

    verifier_reference = canonical_verifier_reference(name_or_verifier)

    verifier_matches = [
        profile
        for profile in profiles.values()
        if profile.verifier_model == verifier_reference
    ]
    if len(verifier_matches) > 1:
        names = ", ".join(sorted(profile.name for profile in verifier_matches))
        raise ProfileError(
            f"multiple profiles match verifier {verifier_reference!r}: {names}; "
            "set config.profile explicitly"
        )
    return verifier_matches[0] if verifier_matches else None


def canonical_verifier_reference(model: str) -> str:
    """Return the repository ID embedded in a Hugging Face cache path."""

    for part in Path(model).parts:
        if not part.startswith("models--"):
            continue
        repository_parts = part.removeprefix("models--").split("--")
        if all(repository_parts):
            return "/".join(repository_parts)
    return model


def resolve_profile(
    config: ProfileConfig | Mapping[str, object] | None = None,
    served_model: str | None = None,
    *,
    profiles: Mapping[str, ModelProfile] | None = None,
    home: Path | None = None,
) -> ModelProfile:
    """Resolve a profile without ever guessing an unknown verifier's draft.

    An explicit ``config.profile`` wins.  Otherwise the served model, followed
    by ``config.model``, must match a profile name or verifier model. Resolved
    Hugging Face cache snapshot paths are matched by their embedded repository
    ID (for example ``models--Qwen--Qwen3.5-9B``).
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


def _lazy_target(value: object) -> str:
    if (
        isinstance(value, tuple)
        and len(value) >= 2
        and isinstance(value[0], str)
        and isinstance(value[1], str)
    ):
        return f"{value[0]}.{value[1]}"
    return ""


def _manager_registry(
    module_name: str,
    manager_name: str,
    eager_attribute: str,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Read eager and lazy parser names after importing the registering package."""

    module = importlib.import_module(module_name)
    manager = getattr(module, manager_name)
    lazy = getattr(manager, "lazy_parsers", {})
    eager = getattr(manager, eager_attribute, {})

    lazy_mapping = lazy if isinstance(lazy, Mapping) else {}
    eager_mapping = eager if isinstance(eager, Mapping) else {}
    names = tuple(
        sorted(
            key
            for key in set(lazy_mapping) | set(eager_mapping)
            if isinstance(key, str) and key
        )
    )
    metadata_map = {
        name: " ".join(
            part
            for part in (
                name,
                _lazy_target(lazy_mapping.get(name)),
            )
            if part
        )
        for name in names
    }
    return names, MappingProxyType(metadata_map)


def discover_vllm_parser_registry() -> ParserRegistry:
    """Discover the installed vLLM parser set without loading parser classes.

    Importing each public parser package runs vLLM's lazy-registration hook.
    The lazy maps are therefore authoritative even while the eager maps remain
    empty, which is the normal state before a parser is first used.
    """

    errors: list[str] = []
    tool_parsers: tuple[str, ...] = ()
    reasoning_parsers: tuple[str, ...] = ()
    tool_metadata: Mapping[str, str] = {}
    reasoning_metadata: Mapping[str, str] = {}

    try:
        tool_parsers, tool_metadata = _manager_registry(
            "vllm.tool_parsers",
            "ToolParserManager",
            "tool_parsers",
        )
    except Exception as exc:
        errors.append(f"tool parsers: {type(exc).__name__}: {exc}")

    try:
        reasoning_parsers, reasoning_metadata = _manager_registry(
            "vllm.reasoning",
            "ReasoningParserManager",
            "reasoning_parsers",
        )
    except Exception as exc:
        errors.append(f"reasoning parsers: {type(exc).__name__}: {exc}")

    if not tool_parsers or not reasoning_parsers:
        external = _external_vllm_parser_registry()
        if external is not None:
            tool_parsers = tuple(sorted(set(tool_parsers) | set(external.tool_parsers)))
            reasoning_parsers = tuple(
                sorted(set(reasoning_parsers) | set(external.reasoning_parsers))
            )
            tool_metadata = MappingProxyType(
                {**external.tool_metadata, **tool_metadata}
            )
            reasoning_metadata = MappingProxyType(
                {**external.reasoning_metadata, **reasoning_metadata}
            )
            errors.extend(external.errors)

    return ParserRegistry(
        tool_parsers=tool_parsers,
        reasoning_parsers=reasoning_parsers,
        tool_metadata=tool_metadata,
        reasoning_metadata=reasoning_metadata,
        errors=tuple(errors),
    )


def _external_vllm_parser_registry() -> ParserRegistry | None:
    """Read lazy parser declarations from the vLLM executable environment.

    SpeedLM intentionally does not depend on vLLM in its own lightweight
    virtualenv. Parsing vLLM's literal lazy-registration tables avoids loading
    torch/CUDA in the gateway process and works for any parser names shipped by
    the installed vLLM version.
    """
    executable = shutil.which("vllm")
    if executable is None:
        return None
    environment = Path(executable).resolve().parent.parent
    site_packages = sorted(environment.glob("lib/python*/site-packages"))
    for site in site_packages:
        package = site / "vllm"
        tool = _literal_parser_registry(
            package / "tool_parsers" / "__init__.py",
            "_TOOL_PARSERS_TO_REGISTER",
            module_prefix="vllm.tool_parsers",
        )
        reasoning = _literal_parser_registry(
            package / "reasoning" / "__init__.py",
            "_REASONING_PARSERS_TO_REGISTER",
            module_prefix="vllm.reasoning",
        )
        if tool is None and reasoning is None:
            continue
        tool_names, tool_metadata = tool or ((), {})
        reasoning_names, reasoning_metadata = reasoning or ((), {})
        return ParserRegistry(
            tool_parsers=tool_names,
            reasoning_parsers=reasoning_names,
            tool_metadata=tool_metadata,
            reasoning_metadata=reasoning_metadata,
        )
    return None


def _literal_parser_registry(
    path: Path,
    variable_name: str,
    *,
    module_prefix: str,
) -> tuple[tuple[str, ...], Mapping[str, str]] | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return None
    raw: object | None = None
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in targets
        ):
            continue
        if statement.value is None:
            return None
        try:
            raw = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            return None
        break
    if not isinstance(raw, dict):
        return None
    entries: dict[str, str] = {}
    for name, target in raw.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(target, tuple)
            or len(target) < 2
            or not isinstance(target[0], str)
            or not isinstance(target[1], str)
        ):
            continue
        entries[name] = f"{name} {module_prefix}.{target[0]}.{target[1]}"
    return tuple(sorted(entries)), MappingProxyType(entries)


def read_model_type(
    model: str,
    *,
    allow_remote: bool = False,
) -> str | None:
    """Read ``model_type`` from config.json, optionally using Transformers."""

    candidate = Path(model).expanduser()
    config_path = candidate / "config.json" if candidate.is_dir() else candidate
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, Mapping):
            return None
        model_type = raw.get("model_type")
        return model_type if isinstance(model_type, str) and model_type else None

    cached_config = _cached_model_config(model)
    if cached_config is not None:
        return read_model_type(str(cached_config))

    if not allow_remote:
        return None
    try:
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(
            model,
            trust_remote_code=False,
        )
        model_type = getattr(config, "model_type", None)
        return model_type if isinstance(model_type, str) and model_type else None
    except Exception:
        return None


def _cached_model_config(model: str) -> Path | None:
    if "/" not in model or model.startswith(("/", ".")):
        return None
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache is not None:
        hub = Path(hub_cache).expanduser()
    else:
        hf_home = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ).expanduser()
        hub = hf_home / "hub"
    repository = hub / f"models--{model.replace('/', '--')}"
    reference = repository / "refs" / "main"
    candidates: list[Path] = []
    try:
        revision = reference.read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision:
        candidates.append(repository / "snapshots" / revision / "config.json")
    snapshots = repository / "snapshots"
    if snapshots.is_dir():
        candidates.extend(
            path / "config.json"
            for path in sorted(snapshots.iterdir(), reverse=True)
            if path.is_dir()
        )
    return next((path for path in candidates if path.is_file()), None)


_PARSER_NOISE_TERMS: Final = frozenset(
    {
        "adapter",
        "class",
        "engine",
        "json",
        "model",
        "module",
        "parser",
        "py",
        "reasoning",
        "tool",
        "vllm",
        "xml",
    }
)


def _name_terms(value: str) -> set[str]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    parts = [
        part.lower()
        for part in re.split(r"[^A-Za-z0-9]+", separated)
        if part
    ]
    terms = set(parts)
    compact = "".join(parts)
    if compact:
        terms.add(compact)
    for part in parts:
        family = re.sub(r"\d.*$", "", part)
        if family:
            terms.add(family)
    return terms - _PARSER_NOISE_TERMS


def _expanded_model_terms(model_type: str) -> set[str]:
    return _name_terms(model_type)


def _best_parser_match(
    model_type: str,
    parsers: Sequence[str],
    metadata: Mapping[str, str],
    *,
    tool_dialect: str | None = None,
) -> str | None:
    model_terms = _name_terms(model_type)
    expanded_terms = _expanded_model_terms(model_type)
    dialect_terms = _name_terms(tool_dialect) if tool_dialect is not None else set()
    model_version_terms = {
        term for term in model_terms if any(character.isdigit() for character in term)
    }
    candidates: list[tuple[tuple[int, int, int, int, str], str]] = []

    for parser in parsers:
        key_terms = _name_terms(parser)
        descriptor_terms = _name_terms(metadata.get(parser, parser))
        parser_version_terms = {
            term
            for term in key_terms | descriptor_terms
            if any(character.isdigit() for character in term)
        }
        if parser_version_terms and not (
            model_version_terms & parser_version_terms
        ):
            continue
        if not model_terms & descriptor_terms and not expanded_terms & key_terms:
            continue
        if dialect_terms and not dialect_terms & key_terms:
            continue

        compact_key = re.sub(r"[^a-z0-9]", "", parser.lower())
        exact_dialect = int(compact_key in expanded_terms)
        direct_matches = len(model_terms & descriptor_terms)
        synonym_matches = len(expanded_terms & key_terms)
        # Prefer a parser whose whole key names the output dialect (for example
        # ``hermes``) over specialized variants that happen to share a family
        # prefix. Shorter keys are the safer generic fallback.
        rank = (
            -exact_dialect,
            -direct_matches,
            -synonym_matches,
            len(parser),
            parser,
        )
        candidates.append((rank, parser))

    if not candidates:
        return None
    return min(candidates)[1]


def _option_value(
    arguments: Sequence[str],
    option: str,
) -> tuple[bool, str | None]:
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            value = argument.partition("=")[2]
            return True, value or None
        if argument == option:
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                return True, arguments[index + 1]
            return True, None
    return False, None


def resolve_model_parsers(
    model: str,
    passthrough: Sequence[str] = (),
    *,
    model_type: str | None = None,
    profile: ModelProfile | None = None,
    registry: ParserRegistry | None = None,
    home: Path | None = None,
    allow_remote_config: bool = False,
) -> ParserResolution:
    """Resolve parser flags through user, profile, auto-detected, then none."""

    discovered = registry if registry is not None else discover_vllm_parser_registry()
    effective_model_type = (
        model_type
        if model_type is not None
        else read_model_type(model, allow_remote=allow_remote_config)
    )

    matched_profile = profile
    if matched_profile is None:
        try:
            matched_profile = resolve_profile(served_model=model, home=home)
        except ProfileError:
            matched_profile = None

    auto_tool = (
        _best_parser_match(
            effective_model_type,
            discovered.tool_parsers,
            discovered.tool_metadata,
        )
        if effective_model_type is not None
        else None
    )
    effective_tool = (
        matched_profile.tool_call_parser
        if matched_profile is not None and matched_profile.tool_call_parser is not None
        else auto_tool
    )
    auto_reasoning = (
        _best_parser_match(
            effective_model_type,
            discovered.reasoning_parsers,
            discovered.reasoning_metadata,
            tool_dialect=effective_tool,
        )
        if effective_model_type is not None
        else None
    )

    if matched_profile is not None and matched_profile.tool_call_parser is not None:
        tool_parser = matched_profile.tool_call_parser
        tool_source: ParserSource = "profile-pinned"
    elif auto_tool is not None:
        tool_parser = auto_tool
        tool_source = "auto-detected"
    else:
        tool_parser = None
        tool_source = "none"

    if matched_profile is not None and matched_profile.reasoning_parser is not None:
        reasoning_parser = matched_profile.reasoning_parser
        reasoning_source: ParserSource = "profile-pinned"
    elif auto_reasoning is not None:
        reasoning_parser = auto_reasoning
        reasoning_source = "auto-detected"
    else:
        reasoning_parser = None
        reasoning_source = "none"

    user_tool_supplied, user_tool = _option_value(passthrough, "--tool-call-parser")
    if user_tool_supplied:
        tool_parser = user_tool
        tool_source = "user-supplied"

    user_reasoning_supplied, user_reasoning = _option_value(
        passthrough,
        "--reasoning-parser",
    )
    if user_reasoning_supplied:
        reasoning_parser = user_reasoning
        reasoning_source = "user-supplied"

    return ParserResolution(
        model_type=effective_model_type,
        registry=discovered,
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        tool_call_parser_source=tool_source,
        reasoning_parser_source=reasoning_source,
    )
