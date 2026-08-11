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
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

from speedlm.storage import speedlm_home

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

#: Draft-chain depth assumed when nothing else names one.
#:
#: This is the Speculators trainer's own ``--ttt-steps`` default
#: (``scripts/train.py:660``) and the depth both stock RedHatAI EAGLE-3
#: drafters declare in ``speculators_config.proposal_methods[0]``.  It is a
#: fallback, never a contract: the serving depth is
#: :attr:`ModelProfile.num_speculative_tokens`.
DEFAULT_SPECULATIVE_TOKENS: Final = 3

#: Deepest draft chain SpeedLM will train or serve.
#:
#: Not an architectural bound.  The Speculators EAGLE-3 head is a *single*
#: transformer layer rolled out autoregressively, so ``ttt_steps`` is a loop
#: bound (``src/speculators/models/eagle3/core.py:199``) rather than a shape,
#: and the upstream suite drives one checkpoint at 1, 3 and 5 steps
#: (``tests/integration/models/test_model_forward.py:264``).  Nothing in the
#: drafter forbids a deeper chain -- gpt-oss-20b's measured *conditional*
#: acceptance is flat-to-rising through position 5 (0.689, 0.665, 0.677,
#: 0.690, 0.715), which is an argument for going deeper, not shallower.
#:
#: The ceiling exists only so a typo in a user profile cannot request a chain
#: whose per-step training and verification cost grows without bound.  Raise
#: it deliberately when a measurement asks for more.
MAX_SPECULATIVE_TOKENS: Final = 16


class ProfileError(ValueError):
    """Raised when a model profile cannot be loaded or resolved safely."""


class SpeculativeDepthError(ProfileError):
    """Raised when training depth and serving depth cannot be reconciled."""


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


#: One ``(range_start, range_end, num_speculative_tokens)`` band, inclusive.
SpeculativeScheduleEntry = tuple[int, int, int]

#: A batch-size-indexed draft-depth schedule, ordered and non-overlapping.
SpeculativeSchedule = tuple[SpeculativeScheduleEntry, ...]


def _validate_speculative_schedule(
    value: Any,
    *,
    num_speculative_tokens: int,
) -> SpeculativeSchedule:
    """Validate a dynamic-K schedule against the vLLM fork's own contract.

    The fork this project serves on replaced ``disable_by_batch_size`` with
    ``SpeculativeConfig.num_speculative_tokens_per_batch_size``
    (``vllm/config/speculative.py:172``), a list of inclusive
    ``(range_start, range_end, K)`` bands that
    ``vllm/v1/spec_decode/dynamic/utils.py:7`` expands into a dense per-batch
    lookup consumed at ``vllm/v1/core/sched/scheduler.py:1085``.  Every rule
    below mirrors a ``ValueError`` that validator would raise -- the point of
    duplicating them is that a typo in a profile fails at profile load rather
    than after a multi-hour engine boot.

    Two rules are *stricter* than vLLM's:

    ``K <= num_speculative_tokens``
        vLLM silently clamps with ``min(vllm_num_speculative_tokens, K)``
        (``utils.py:113``), so a schedule asking for more than the declared
        depth is served quietly shallower than it reads.  A profile that lies
        about the depth it serves is exactly the failure this module exists to
        prevent, so it is rejected instead.

    ``max(K) == num_speculative_tokens``
        This is what keeps :func:`validate_training_depth` protecting
        something real once ``K`` is a schedule rather than a scalar.  See that
        function's docstring for the argument.
    """
    if not isinstance(value, (tuple, list)) or isinstance(value, (str, bytes)):
        raise ProfileError("speculative_schedule must be an array of bands or null")
    if not value:
        raise ProfileError("speculative_schedule must not be empty")

    bands: list[SpeculativeScheduleEntry] = []
    for entry in value:
        if (
            not isinstance(entry, (tuple, list))
            or isinstance(entry, (str, bytes))
            or len(entry) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in entry)
        ):
            raise ProfileError(
                "each speculative_schedule band must be "
                "[range_start, range_end, num_speculative_tokens] integers; "
                f"got {entry!r}"
            )
        start, end, depth = (int(entry[0]), int(entry[1]), int(entry[2]))
        if start < 1:
            raise ProfileError(
                f"speculative_schedule range_start must be at least 1; got {start}"
            )
        if end < start:
            raise ProfileError(
                f"speculative_schedule band ({start}, {end}) ends before it starts"
            )
        if depth < 0:
            raise ProfileError(
                f"speculative_schedule depth must not be negative; got {depth}"
            )
        if depth > num_speculative_tokens:
            raise ProfileError(
                f"speculative_schedule band ({start}, {end}) drafts {depth} tokens "
                f"but the profile declares num_speculative_tokens="
                f"{num_speculative_tokens}; vLLM would silently clamp it"
            )
        bands.append((start, end, depth))

    if bands[0][0] != 1:
        raise ProfileError(
            "the first speculative_schedule band must start at batch size 1 so "
            f"every runtime batch size has a defined depth; got {bands[0][0]}"
        )
    for previous, current in zip(bands, bands[1:], strict=False):
        if current[0] <= previous[1]:
            raise ProfileError(
                f"speculative_schedule bands ({previous[0]}, {previous[1]}) and "
                f"({current[0]}, {current[1]}) overlap or are out of order"
            )
    deepest = max(depth for _, _, depth in bands)
    if deepest != num_speculative_tokens:
        raise ProfileError(
            f"speculative_schedule serves at most {deepest} draft tokens but the "
            f"profile declares num_speculative_tokens={num_speculative_tokens}; "
            "the declared depth is what the drafter is trained to, so it must be "
            "the depth the schedule actually reaches"
        )
    return tuple(bands)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Complete, immutable contract for one speculative-decoding setup."""

    name: str
    verifier_model: str
    draft_model: str | None
    speculative_method: SpeculativeMethod
    num_speculative_tokens: int
    target_layer_ids: tuple[int, ...] | None
    max_seq_len: int
    #: Descriptive only, and defaulted because it gates nothing.
    #:
    #: No code in ``src/`` branches on this value -- it is validated on load,
    #: copied into ``to_dict`` and printed by ``speedlm status``/``doctor``,
    #: and that is all.  Chat rendering is done by the *server*: replay posts
    #: OpenAI-shaped messages to ``/v1/chat/completions`` and vLLM applies the
    #: model's own HF template (no ``--chat-template`` argv is ever passed),
    #: while training rows are rendered downstream by Speculators'
    #: ``prepare_data.py --model <verifier>``, again from the verifier's own
    #: tokenizer.  The local template stack under ``training/templates/`` is
    #: reachable only from tests.
    #:
    #: It was required until now, against a closed vocabulary of three values.
    #: That combination could only ever *reject* a legitimate model: a family
    #: that is neither Harmony nor ChatML had to claim to be one of them to be
    #: declarable at all, and the claim changed no behaviour once accepted.  A
    #: required gate that admits nothing true and blocks something true is
    #: worse than no gate, so it now defaults to ``"auto"`` -- which is what
    #: every value already meant operationally.
    chat_template_kind: ChatTemplateKind = "auto"
    num_hidden_layers: int | None = None
    max_scratch_bytes: int | None = None
    """Hidden-state scratch scales with hidden_size x num_aux_layers x tokens."""
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    speculative_schedule: SpeculativeSchedule | None = None
    """Optional batch-size-indexed draft depth; ``None`` keeps ``K`` fixed."""
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
        if self.num_speculative_tokens > MAX_SPECULATIVE_TOKENS:
            raise SpeculativeDepthError(
                f"num_speculative_tokens must be at most "
                f"{MAX_SPECULATIVE_TOKENS}; got {self.num_speculative_tokens}"
            )
        _positive_int(self.max_seq_len, "max_seq_len")
        if self.num_hidden_layers is not None:
            _positive_int(self.num_hidden_layers, "num_hidden_layers")
        if self.max_scratch_bytes is not None:
            _positive_int(self.max_scratch_bytes, "max_scratch_bytes")
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
        if (
            self.speculative_method in {"eagle3", "medusa", "draft_model"}
            and self.draft_model is None
        ):
            raise ProfileError(
                f"draft_model is required for {self.speculative_method} profiles"
            )
        if self.speculative_schedule is not None:
            object.__setattr__(
                self,
                "speculative_schedule",
                _validate_speculative_schedule(
                    self.speculative_schedule,
                    num_speculative_tokens=self.num_speculative_tokens,
                ),
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

    @property
    def max_served_speculative_tokens(self) -> int:
        """Deepest draft position this profile can ever ask the drafter for.

        With a fixed ``K`` that is ``num_speculative_tokens``.  With a
        schedule it is the deepest band, which
        :func:`_validate_speculative_schedule` pins to the same number -- so
        this is a single, honest answer either way, and the one
        :func:`validate_training_depth` must be handed.
        """

        if self.speculative_schedule is None:
            return self.num_speculative_tokens
        return max(depth for _, _, depth in self.speculative_schedule)

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
            "num_hidden_layers": self.num_hidden_layers,
            "max_scratch_bytes": self.max_scratch_bytes,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
            "speculative_schedule": (
                [list(band) for band in self.speculative_schedule]
                if self.speculative_schedule is not None
                else None
            ),
            "trainable": self.trainable,
        }

    def speculative_config(self) -> dict[str, object]:
        """Build the vLLM speculative-config fragment for this profile.

        ``num_speculative_tokens_per_batch_size`` is emitted *only* when the
        profile carries a schedule.  Leaving it out is what makes
        ``Scheduler.dynamic_sd_lookup`` stay ``None``
        (``vllm/v1/core/sched/scheduler.py:236``) and the engine fall back to
        the fixed ``num_speculative_tokens`` -- i.e. exactly today's
        behaviour for every profile that does not opt in.
        """

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
        if self.speculative_schedule is not None:
            result["num_speculative_tokens_per_batch_size"] = [
                list(band) for band in self.speculative_schedule
            ]
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

        required = REQUIRED_PROFILE_KEYS
        allowed = PROFILE_FIELD_NAMES
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

        # Absent is legitimate and means ``"auto"``; a value that IS supplied is
        # still held to the vocabulary, so an existing profile carrying a typo
        # is not silently downgraded to the default.
        template_value = data.get("chat_template_kind", "auto")
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

        num_hidden_layers_value = data.get("num_hidden_layers")
        if num_hidden_layers_value is not None:
            num_hidden_layers_value = _positive_int(
                num_hidden_layers_value, "num_hidden_layers"
            )

        max_scratch_bytes_value = data.get("max_scratch_bytes")
        if max_scratch_bytes_value is not None:
            max_scratch_bytes_value = _positive_int(
                data["max_scratch_bytes"], "max_scratch_bytes"
            )

        schedule_value = data.get("speculative_schedule")
        speculative_schedule: SpeculativeSchedule | None
        if schedule_value is None:
            speculative_schedule = None
        else:
            # Fully validated by ``__post_init__``; this only fixes the shape
            # so a JSON array of arrays becomes the tuple the dataclass holds.
            speculative_schedule = cast(SpeculativeSchedule, schedule_value)

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
                num_hidden_layers=cast(int | None, num_hidden_layers_value),
                max_scratch_bytes=cast(int | None, max_scratch_bytes_value),
                tool_call_parser=cast(str | None, tool_call_parser_value),
                reasoning_parser=cast(str | None, reasoning_parser_value),
                speculative_schedule=speculative_schedule,
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


#: Every key :meth:`ModelProfile.from_dict` will accept, derived from the
#: dataclass rather than hand-listed.
#:
#: A hand-maintained allow-list is a copy of a structure, and copies drift.
#: This project has already shipped that exact bug once on the config side:
#: ``PromotionConfig.min_accepted_length_delta`` existed, was validated in
#: ``__post_init__`` and was documented as settable, but was missing from the
#: promotion allow-list, so every attempt to set it from JSON raised
#: ``unknown keys in promotion``.  A field added to :class:`ModelProfile` is by
#: definition part of the profile contract, so deriving the allow-list makes
#: that class of defect unrepresentable instead of merely tested for.
#:
#: ``trainable`` is included because it is a real dataclass field: it is
#: ``init=False`` and derived from ``speculative_method``, and ``to_dict``
#: emits it, so a round-trip of a profile through JSON must be accepted back.
#: ``from_dict`` still rejects a declared value that contradicts the derived
#: one -- accepting the key is not the same as honouring it.
PROFILE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    profile_field.name for profile_field in fields(ModelProfile)
)

#: The keys a profile JSON object must carry.
#:
#: Deliberately *not* derived.  Which fields are mandatory is a semantic
#: choice, not a structural fact: ``draft_model`` is required even though it
#: may be null, because a profile that simply omits it is far more likely to be
#: an author who forgot their drafter than one who meant "no drafter", and the
#: optional fields all have defaults that are safe to leave unstated.  A
#: structural test pins this set as a subset of the dataclass fields so a typo
#: or a renamed field cannot make a key required that no longer exists.
REQUIRED_PROFILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "verifier_model",
        "draft_model",
        "speculative_method",
        "num_speculative_tokens",
        "max_seq_len",
    }
)


def spread_aux_layers(k: int, n: int) -> tuple[int, ...]:
    """Spread *k* aux-layer indices across *n* hidden layers.

    This generalises vLLM's three-position heuristic
    ``(2, n//2, n-3)`` to an arbitrary count *k*, while reproducing it
    exactly when ``k == 3`` and ``n >= 7``.

    The rule places the first index at 2, the last at ``n - 3``, and
    distributes the remaining ``k - 2`` indices evenly between them.
    The result is validated: indices must be unique, strictly increasing,
    and all within ``[0, n)``.

    Raises ``ValueError`` when the requested count cannot be satisfied —
    for example when the model is too shallow.  The error message names
    *k*, *n*, and why the resolution failed so the operator can adjust
    the model or the aux-layer count.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"aux-layer count must be a positive integer, got {k!r}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"num_hidden_layers must be a positive integer, got {n!r}")

    first = 2
    last = n - 3

    if k == 1:
        indices = [first]
    elif k == 2:
        indices = [first, last]
    elif k == 3:
        if last < first + 2:
            raise ValueError(
                f"cannot spread {k} aux layers across {n} hidden layers: "
                f"the available range ({first}..{last}) is too narrow"
            )
        # Reproduce vLLM's formula exactly: (2, n//2, n-3).
        indices = [first, n // 2, last]
    else:
        if last - first < k - 1:
            raise ValueError(
                f"cannot spread {k} aux layers across {n} hidden layers: "
                f"the available range ({first}..{last}) is too narrow "
                f"(need at least {k - 1} slots for {k} indices)"
            )
        # Place k - 2 intermediate indices evenly between `first` and `last`.
        # We divide the span (last - first) into (k - 1) equal segments,
        # so position j is: first + j * (last - first) / (k - 1), rounded down.
        span = last - first
        seg_count = k - 1
        indices = [first]
        for j in range(1, seg_count):
            idx = first + (j * span) // seg_count
            indices.append(idx)
        indices.append(last)

    # Validate: unique, strictly increasing, within [0, n)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"cannot spread {k} aux layers across {n} hidden layers: "
            f"generated indices {tuple(indices)} contain duplicates"
        )
    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            raise ValueError(
                f"cannot spread {k} aux layers across {n} hidden layers: "
                f"indices are not strictly increasing at position {i} "
                f"({indices[i - 1]} >= {indices[i]})"
            )
    if any(idx < 0 or idx >= n for idx in indices):
        raise ValueError(
            f"cannot spread {k} aux layers across {n} hidden layers: "
            f"indices {tuple(indices)} are out of range [0, {n})"
        )

    return tuple(indices)


def default_aux_layers(num_hidden_layers: int) -> tuple[int, ...]:
    """Return vLLM's default three aux-layer IDs for *num_hidden_layers*.

    DEPRECATED: use :func:`spread_aux_layers` with the drafter's actual
    aux-layer count instead.  This function remains for backward
    compatibility with code that still hardcodes ``k == 3``.

    For example, gpt-oss-20b (24 layers) gives ``(2, 12, 21)``
    and Llama-3.1-8B (32 layers) gives ``(2, 16, 29)``.

    Raises ``ValueError`` when the model is too small (n < 7).
    """
    return spread_aux_layers(3, num_hidden_layers)


class AuxLayerError(ValueError):
    """Raised when aux-layer resolution produces an invalid result."""


def resolve_target_layer_ids(
    *,
    explicit: tuple[int, ...] | None = None,
    num_hidden_layers: int | None = None,
    drafter_aux_count: int | None = None,
) -> tuple[int, ...]:
    """Resolve EAGLE-3 target aux-layer IDs with a documented precedence.

    Resolution order:

    1. **Explicit profile override** — ``explicit``, if set, is the
       authoritative pin.  It is validated against ``num_hidden_layers``
       when that is known.

    2. **Drafter-driven derivation** — when ``drafter_aux_count`` is known,
       the drafter's declared count of aux layers is used as *k* for
       :func:`spread_aux_layers`.  This matches the drafter's
       ``fc_input_size`` expectation (see ``llama_eagle3.py``).

    3. **Heuristic fallback** — as a last resort, default to ``k == 3``.
       This is the vLLM default when the drafter config carries no
       ``num_aux_hidden_states`` or ``eagle_aux_hidden_state_layer_ids``.

    Raises :class:`AuxLayerError` when resolution cannot produce a valid
    set of indices.  The error message names the model, its layer count,
    and the required count so the operator can take corrective action.
    """
    if explicit is not None:
        if num_hidden_layers is not None:
            _validate_layer_ids(
                explicit, num_hidden_layers, expected_count=drafter_aux_count
            )
        return explicit

    if num_hidden_layers is None:
        raise AuxLayerError(
            "cannot resolve aux layers: num_hidden_layers is not known"
        )

    k = drafter_aux_count if drafter_aux_count is not None else 3

    return spread_aux_layers(k, num_hidden_layers)


def _validate_layer_ids(
    indices: tuple[int, ...],
    num_hidden_layers: int,
    *,
    expected_count: int | None = None,
) -> None:
    """Validate that *indices* is a valid aux-layer set."""
    if len(indices) == 0:
        raise AuxLayerError("target_layer_ids must not be empty")
    if len(set(indices)) != len(indices):
        raise AuxLayerError(
            f"target_layer_ids contains duplicates: {indices}"
        )
    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            raise AuxLayerError(
                f"target_layer_ids is not strictly increasing: {indices}"
            )
    if any(idx < 0 or idx >= num_hidden_layers for idx in indices):
        raise AuxLayerError(
            f"target_layer_ids {indices} is out of range for a model "
            f"with {num_hidden_layers} hidden layers"
        )
    if expected_count is not None and len(indices) != expected_count:
        raise AuxLayerError(
            f"target_layer_ids has {len(indices)} entries but the drafter "
            f"expects {expected_count} aux layers"
        )


def resolve_speculative_tokens(
    *,
    explicit: int | None = None,
    drafter_declared: int | None = None,
) -> int:
    """Resolve the EAGLE-3 draft-chain depth with a documented precedence.

    This is the single answer to "how deep is the chain", used for *both* the
    trainer's ``--ttt-steps`` and vLLM's ``num_speculative_tokens``.  Before it
    existed the two were independent constants -- the profile served
    ``num_speculative_tokens`` while
    :class:`speedlm.tuner.eagle3.Eagle3Config` hard-required 3 TTT steps and
    the backend never passed ``--ttt-steps`` at all, so gpt-oss-20b trained a
    3-deep head and served it 5-deep and positions 4-5 were pure
    extrapolation.

    Resolution order, deliberately the same shape as
    :func:`resolve_target_layer_ids`:

    1. **Explicit profile override** -- ``explicit``, if set, wins.  The
       profile's ``num_speculative_tokens`` is what vLLM is actually told to
       serve, so it is the truth training has to match, not the other way
       round.

    2. **Drafter-declared arity** -- otherwise the checkpoint's own
       ``speculators_config.proposal_methods[*].speculative_tokens``, read via
       :func:`drafter_declared_speculative_tokens`.  A checkpoint that names
       the chain it was fitted for is better evidence than any default.

    3. **Fallback** -- :data:`DEFAULT_SPECULATIVE_TOKENS`.

    A declared value that disagrees with an explicit profile pin is *not* an
    error, and that asymmetry against ``expected_count`` in
    :func:`_validate_layer_ids` is deliberate.  Aux-layer count is a shape:
    ``fc.weight`` is ``hidden x (aux_count * hidden)`` and the wrong count
    cannot even load.  Chain depth is a rollout length; the stock drafters
    declare 3 because 3 is what they were trained at, and deepening that is
    exactly what a tuning cycle is for.  Treating the declaration as a veto
    would forbid the fix.

    Raises:
        SpeculativeDepthError: if the resolved depth is not an integer in
            ``1..MAX_SPECULATIVE_TOKENS``.
    """
    for name, value in (("explicit", explicit), ("drafter_declared", drafter_declared)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpeculativeDepthError(f"{name} speculative tokens must be an integer")
        if value < 1 or value > MAX_SPECULATIVE_TOKENS:
            raise SpeculativeDepthError(
                f"{name} speculative tokens must be in "
                f"1..{MAX_SPECULATIVE_TOKENS}; got {value}"
            )
    if explicit is not None:
        return explicit
    if drafter_declared is not None:
        return drafter_declared
    return DEFAULT_SPECULATIVE_TOKENS


def drafter_declared_speculative_tokens(config: Mapping[str, Any]) -> int | None:
    """Return the chain depth a Speculators draft *config* declares, if any.

    Reads ``speculators_config.proposal_methods[*].speculative_tokens``, which
    is what ``Eagle3DraftModel.from_training_args`` writes from ``ttt_steps``
    (``src/speculators/models/eagle3/core.py:317``) and what
    ``Eagle3SpeculatorConfig`` exposes to vLLM.  Both stock RedHatAI EAGLE-3
    drafters in the local cache declare 3.

    Returns ``None`` when the config carries no usable declaration, and when
    the declared depths disagree with each other -- an ambiguous checkpoint is
    no evidence at all, and guessing which proposal method "counts" would be
    the same class of silent assumption this module exists to remove.
    """
    speculators = config.get("speculators_config")
    if not isinstance(speculators, Mapping):
        return None
    methods = speculators.get("proposal_methods")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
        return None
    declared: set[int] = set()
    for method in methods:
        if not isinstance(method, Mapping):
            continue
        value = method.get("speculative_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            continue
        declared.add(value)
    if len(declared) != 1:
        return None
    return declared.pop()


def validate_training_depth(
    *,
    profile_name: str,
    serving_tokens: int,
    training_steps: int,
    serving_schedule: SpeculativeSchedule | None = None,
) -> None:
    """Fail before a GPU cycle when the two depths disagree.

    Composition derives ``training_steps`` from ``serving_tokens``, so this
    can only fire after someone reintroduces an independent source for one of
    them.  That is precisely the regression it exists to catch: the original
    defect was silent for a whole tuning cycle and only showed up as a
    -0.567 pp gate delta on gpt-oss against -0.189 pp on the 3/3 Qwen3
    profile.

    When ``serving_schedule`` is given, ``K`` is no longer a scalar and the
    invariant has to be restated.  The depth that matters is the **deepest**
    band the schedule can reach, not the shallowest and not any average,
    because acceptance degrades monotonically with position and only the deep
    positions are extrapolation.  Run ``d993eee-gptoss-idle`` is the direct
    measurement: a candidate drafter whose own ``speculators_config`` declared
    ``speculative_tokens: 3`` was served at 5, and its per-position acceptance
    against the stock drafter was at parity at position 0 (0.6850 vs 0.6871)
    and collapsed only in the tail -- -2.4 pp at position 1, -4.4 pp at
    position 2, -5.3 pp at position 3, -5.3 pp at position 4.  A schedule
    whose shallow bands are trained but whose deepest band is not would
    reproduce that regression on exactly the batch sizes that use the deepest
    band, so the deepest band is what must be trained.
    """
    served = serving_tokens
    if serving_schedule is not None:
        served = max(depth for _, _, depth in serving_schedule)
    if training_steps != served:
        detail = (
            f" (deepest band of {len(serving_schedule)} in the batch-size "
            f"schedule)"
            if serving_schedule is not None
            else ""
        )
        raise SpeculativeDepthError(
            f"profile {profile_name!r} would train a {training_steps}-deep "
            f"draft chain and serve a {served}-deep one{detail}; positions "
            f"{training_steps + 1}..{served} would be extrapolation. "
            "Training depth must equal num_speculative_tokens."
        )


#: What to do when the training window and the serving context window disagree.
#:
#: ``record`` is the default and changes no behaviour: it resolves both numbers,
#: logs the ratio and hands the caller a record for the artifact manifest.  It
#: is the default because on *stock* configuration the two disagree by
#: construction -- ``tuning.sequence_length`` defaults to 16384 while every
#: built-in profile serves 40960..262144 -- so a policy that raised here would
#: reject every supported deployment on first start and teach operators to
#: disable the check.
#:
#: ``align`` makes them agree by construction, by emitting
#: ``--max-model-len <training window>`` on the tuned engine.  That is a real
#: product change and not a safe default: it *caps what clients may send*.  A
#: deployment whose users legitimately submit 100k-token prompts would start
#: rejecting them, which is a far worse failure than a drafter that extrapolates.
#:
#: ``strict`` refuses any disagreement.  Useful for a benchmark rig that intends
#: to hold the two axes equal and wants that intent enforced; useless as a
#: default, for the reason above.
ContextWindowPolicy = Literal["record", "align", "strict"]

CONTEXT_WINDOW_POLICIES: Final = frozenset({"record", "align", "strict"})

#: Where a resolved serving context window came from.
ContextWindowSource = Literal["passthrough", "verifier_config", "unresolved"]

MAX_MODEL_LEN_OPTION: Final = "--max-model-len"


class ContextWindowError(ProfileError):
    """Raised when serving and training context windows cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class ServingContextWindow:
    """The positional range the serving engine will actually accept.

    ``tokens`` is ``None`` only when neither source could answer -- the
    operator passed no ``--max-model-len`` *and* the verifier's ``config.json``
    is not in the local cache (or carries no usable
    ``max_position_embeddings``).  Unresolved is reported as itself rather than
    guessed at: an invented serving window would put a fabricated number in the
    artifact manifest, and the manifest is the only place a gate delta can
    later be attributed from.
    """

    tokens: int | None
    source: ContextWindowSource


@dataclass(frozen=True, slots=True)
class ContextWindowAlignment:
    """Resolved training/serving position axes and their disagreement.

    ``ratio`` is serving over training, i.e. how many times further out the
    engine may place a token than the deepest position the cycle could have
    fitted.  ``None`` when the serving window is unresolved.
    """

    training_tokens: int
    serving_tokens: int | None
    source: ContextWindowSource
    policy: ContextWindowPolicy
    aligned: bool

    @property
    def ratio(self) -> float | None:
        if self.serving_tokens is None:
            return None
        return self.serving_tokens / self.training_tokens

    def as_manifest_fields(self) -> dict[str, object]:
        """The provenance a published artifact must carry.

        Every key is always present, ``None`` included, for the same reason
        ``verifier_revision`` is: an absent key is ambiguous between "this
        build does not record the position axis" and "this cycle could not
        resolve it", and only one of those is attributable.
        """
        return {
            "training_sequence_length": self.training_tokens,
            "serving_context_window": self.serving_tokens,
            "serving_context_window_source": self.source,
            "context_window_policy": self.policy,
            "context_window_ratio": self.ratio,
            "context_window_aligned": self.aligned,
        }


def resolve_serving_context_window(
    verifier_model: str,
    passthrough: Sequence[str] = (),
) -> ServingContextWindow:
    """Resolve what the serving engine will accept, from runtime facts only.

    Two sources, in the order vLLM itself resolves them:

    1. the operator's ``--max-model-len`` passthrough, which wins outright;
    2. otherwise ``max_position_embeddings`` from the verifier's cached
       ``config.json`` -- the value vLLM falls back to, read here with
       :func:`_cached_model_config`, the same stdlib-only lookup the rest of
       this module uses.

    No torch, no network, no GPU: this is safe to call before a cycle commits
    to anything.  Model *names* are deliberately not consulted -- a profile's
    ``max_seq_len`` is a declaration, and the whole point of this function is
    to read what will actually happen.
    """
    supplied, raw = _option_value(passthrough, MAX_MODEL_LEN_OPTION)
    if supplied:
        if raw is None:
            raise ContextWindowError(
                f"{MAX_MODEL_LEN_OPTION} was passed through without a value"
            )
        try:
            tokens = int(raw)
        except ValueError as exc:
            raise ContextWindowError(
                f"{MAX_MODEL_LEN_OPTION} must be an integer, got {raw!r}"
            ) from exc
        if tokens < 1:
            raise ContextWindowError(
                f"{MAX_MODEL_LEN_OPTION} must be positive, got {tokens}"
            )
        return ServingContextWindow(tokens=tokens, source="passthrough")

    config_path = _cached_model_config(verifier_model)
    if config_path is None:
        return ServingContextWindow(tokens=None, source="unresolved")
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ServingContextWindow(tokens=None, source="unresolved")
    if not isinstance(raw_config, Mapping):
        return ServingContextWindow(tokens=None, source="unresolved")
    declared = raw_config.get("max_position_embeddings")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        return ServingContextWindow(tokens=None, source="unresolved")
    return ServingContextWindow(tokens=declared, source="verifier_config")


def validate_training_context_window(
    *,
    profile_name: str,
    training_tokens: int,
    serving: ServingContextWindow,
    policy: ContextWindowPolicy = "record",
) -> ContextWindowAlignment:
    """Reconcile the position axis the way :func:`validate_training_depth`
    reconciles the depth axis.

    ``validate_training_depth`` refuses to train a chain shallower than the one
    it will serve because "positions N+1..M would be extrapolation".  The
    sequence axis carries the identical argument and had no guard at all:
    ``tuning.sequence_length`` caps the training window at 16384 by default
    while nothing on the serving path emits ``--max-model-len``, so the engine
    falls back to the verifier's ``max_position_embeddings`` -- 131072 for
    gpt-oss-20b and Llama-3.1-8B, 262144 for Qwen3.5-9B, 40960 for Qwen3-8B.

    It is a *weaker* argument than the depth one, and the policy default
    reflects that.  ``sequence_length`` is a cap, not a target: it becomes
    ``--seq-length`` for ``prepare_data.py`` and ``--total-seq-len`` for
    ``train.py``, so the positions the head actually sees are bounded by the
    captured conversations, which are far shorter than 16384.  Raising the cap
    would therefore not buy positional coverage the corpus does not contain --
    which is why the default policy records the disagreement instead of trying
    to close it.  See :data:`ContextWindowPolicy`.

    Pure: every value it reads is already resolved by the caller, so it can run
    before any filesystem mutation or engine start.
    """
    if policy not in CONTEXT_WINDOW_POLICIES:
        raise ContextWindowError(
            f"context window policy must be one of "
            f"{', '.join(sorted(CONTEXT_WINDOW_POLICIES))}, got {policy!r}"
        )
    if isinstance(training_tokens, bool) or not isinstance(training_tokens, int):
        raise ContextWindowError("training_tokens must be an integer")
    if training_tokens < 1:
        raise ContextWindowError(
            f"training_tokens must be positive, got {training_tokens}"
        )
    serving_tokens = serving.tokens
    aligned = serving_tokens is not None and serving_tokens == training_tokens
    if policy == "strict" and not aligned:
        if serving_tokens is None:
            raise ContextWindowError(
                f"profile {profile_name!r} requested the 'strict' context window "
                f"policy, but the serving context window could not be resolved "
                f"({serving.source}); pass {MAX_MODEL_LEN_OPTION} explicitly or "
                "use the 'record' policy."
            )
        raise ContextWindowError(
            f"profile {profile_name!r} would train on positions "
            f"1..{training_tokens} and serve positions 1..{serving_tokens}; "
            f"positions {training_tokens + 1}..{serving_tokens} would be "
            f"extrapolation. Set tuning.sequence_length and "
            f"{MAX_MODEL_LEN_OPTION} to the same value, or use the 'align' "
            "policy to have SpeedLM emit it."
        )
    return ContextWindowAlignment(
        training_tokens=training_tokens,
        serving_tokens=serving_tokens,
        source=serving.source,
        policy=policy,
        aligned=aligned,
    )


def cached_snapshot_dir(model: str) -> Path | None:
    """Return a local directory holding *model*'s weights, or ``None``.

    Accepts either an on-disk path (a promoted artifact, returned as-is) or a
    Hub repo id, whose snapshot is located by reading the cache layout
    directly -- the same stdlib-only approach as
    :func:`speedlm.tuner.composition.cached_hub_revision`, so this stays
    importable on hosts without ``huggingface_hub``.
    """
    candidate = Path(model)
    if candidate.is_dir():
        return candidate
    config = _cached_model_config(model)
    if config is None:
        return None
    return config.parent


GPT_OSS_EAGLE3_PROFILE: Final = ModelProfile(
    name="gpt-oss-20b-eagle3",
    verifier_model="openai/gpt-oss-20b",
    draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
    speculative_method="eagle3",
    num_speculative_tokens=5,
    target_layer_ids=(2, 12, 21),
    chat_template_kind="harmony",
    max_seq_len=131_072,
    num_hidden_layers=24,
    tool_call_parser="openai",
    reasoning_parser="openai_gptoss",
)

LLAMA_31_8B_EAGLE3_PROFILE: Final = ModelProfile(
    name="llama-3.1-8b-instruct-eagle3",
    verifier_model="meta-llama/Llama-3.1-8B-Instruct",
    draft_model="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
    speculative_method="eagle3",
    num_speculative_tokens=5,
    target_layer_ids=(2, 16, 29),
    chat_template_kind="auto",
    max_seq_len=131_072,
    num_hidden_layers=32,
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

QWEN_3_8B_EAGLE3_PROFILE: Final = ModelProfile(
    name="qwen3-8b-eagle3",
    verifier_model="Qwen/Qwen3-8B",
    draft_model="RedHatAI/Qwen3-8B-speculator.eagle3",
    speculative_method="eagle3",
    num_speculative_tokens=3,
    target_layer_ids=None,
    chat_template_kind="chatml",
    max_seq_len=40_960,
    num_hidden_layers=36,
    #: The model-owned template wraps a JSON object with ``name`` and
    #: ``arguments`` fields.  Hermes consumes that body; qwen3_xml instead
    #: requires function-assignment and parameter-assignment elements.
    tool_call_parser="hermes",
)

BUILTIN_PROFILES: Final[Mapping[str, ModelProfile]] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            GPT_OSS_EAGLE3_PROFILE,
            LLAMA_31_8B_EAGLE3_PROFILE,
            QWEN_35_9B_MTP_PROFILE,
            QWEN_3_8B_EAGLE3_PROFILE,
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
    resolved_dir = profiles_dir.resolve()
    for path in sorted(profiles_dir.glob("*.json")):
        #: The profiles directory is user-writable, so a *.json entry may be a
        #: symlink aimed anywhere on the filesystem. Loading one would let an
        #: attacker with write access to the directory (but not to the profile
        #: contents) redirect the verifier/draft models a run trains against.
        if path.is_symlink() and not path.resolve().is_relative_to(resolved_dir):
            raise ProfileError(
                f"{path}: refusing to load profile symlink pointing outside"
                f" {profiles_dir}"
            )
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


#: A term that is nothing but a version marker: an optional leading ``v``
#: followed by digits. ``v1``, ``v2``, ``v`` and ``3`` all qualify; ``qwen3``
#: and ``a13b`` do not, because :func:`_name_terms` already contributed their
#: digit-stripped family stem (``qwen``, ``a``) to the term set separately.
_BARE_VERSION_TERM: Final = re.compile(r"v\d*|\d+")


def _carries_family_signal(shared_terms: set[str]) -> bool:
    """Is there anything in the overlap besides a version number?

    :func:`_name_terms` deliberately over-generates. It emits each part, the
    compact join, and a digit-stripped family stem, so that a model typed
    ``qwen3`` still matches a parser keyed ``qwen``. The price of that
    generosity is that a version fragment like ``v1`` becomes a first-class
    term on both sides -- and the match gate in :func:`_best_parser_match`
    demands only ONE shared term, on either the descriptor or the key side.

    Two unrelated families that both happen to be on their first revision
    therefore "match" on the version number alone. Measured against the real
    installed vLLM registry (43 tool parsers, 27 reasoning parsers),
    ``hunyuan_v1_dense`` selected ``poolside_v1`` on the shared set
    ``{"v", "v1"}`` and nothing else, and reported it as ``auto-detected`` for
    BOTH the tool-call and the reasoning parser. Hunyuan and Poolside are
    unrelated models with unrelated output dialects. The only thing the two
    names have in common is the digit 1.

    So: a bare version token may never be the sole basis for a match. If the
    version number is all two names share, they share nothing, and we abstain.
    Abstaining is the honest outcome here -- :func:`resolve_model_parsers`
    degrades the source to ``"none"`` and ``cli.py`` then emits no
    ``--tool-call-parser`` / ``--reasoning-parser`` flag at all. Not parsing
    tool calls is a visible, correctable gap. Parsing them with another
    family's grammar is a silent one, and silent wrongness is the failure mode
    this codebase exists to avoid.
    """
    return any(not _BARE_VERSION_TERM.fullmatch(term) for term in shared_terms)


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
        direct_shared = model_terms & descriptor_terms
        synonym_shared = expanded_terms & key_terms
        if not direct_shared and not synonym_shared:
            continue
        if dialect_terms and not dialect_terms & key_terms:
            continue
        # Confidence floor. The gates above are satisfied by a single shared
        # term, and a shared version number is a coincidence, not evidence of
        # a shared dialect. See :func:`_carries_family_signal` for the
        # hunyuan/poolside case this rejects.
        if not _carries_family_signal(direct_shared | synonym_shared):
            continue

        compact_key = re.sub(r"[^a-z0-9]", "", parser.lower())
        exact_dialect = int(compact_key in expanded_terms)
        direct_matches = len(direct_shared)
        synonym_matches = len(synonym_shared)
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
