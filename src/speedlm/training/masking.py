"""Explicit supervision policies and EAGLE-style training-window audits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from speedlm.training.templates.base import AssistantSpan


class MaskPolicy(StrEnum):
    """Which assistant spans receive loss."""

    FINAL_SPAN = "final_span"
    FINAL_TURN_ALL_CHANNELS = "final_turn_all_channels"
    ALL_ASSISTANT_TURNS = "all_assistant_turns"


class FinalAssistantMaskError(ValueError):
    """A named row has no supervised tokens under its selected policy."""

    def __init__(
        self,
        row_id: str,
        policy: MaskPolicy | None = None,
        detail: str = "",
    ) -> None:
        if policy is None:
            # Backward-compatible construction for the existing tuner effect
            # protocol. New row preparation always supplies row identity/policy.
            self.row_id = "<unknown>"
            self.policy = MaskPolicy.FINAL_SPAN
            super().__init__(row_id)
            return
        self.row_id = row_id
        self.policy = policy
        suffix = f": {detail}" if detail else ""
        super().__init__(
            f"training row {row_id!r} has an all-zero {policy.value} loss mask{suffix}"
        )


def select_spans(
    spans: Sequence[AssistantSpan],
    policy: MaskPolicy,
) -> tuple[AssistantSpan, ...]:
    """Select assistant character spans according to an explicit policy."""
    if not spans:
        return ()
    if policy is MaskPolicy.ALL_ASSISTANT_TURNS:
        return tuple(spans)
    if policy is MaskPolicy.FINAL_TURN_ALL_CHANNELS:
        final_turn = max(span.turn for span in spans)
        return tuple(span for span in spans if span.turn == final_turn)

    final_spans = [span for span in spans if span.channel == "final"]
    if final_spans:
        return (final_spans[-1],)
    return (spans[-1],)


def loss_mask_from_offsets(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[AssistantSpan],
    *,
    policy: MaskPolicy,
    row_id: str,
) -> tuple[bool, ...]:
    """Project selected character spans onto tokenizer offset mappings."""
    selected = select_spans(spans, policy)
    mask = tuple(
        token_end > token_start
        and any(token_start < span.end and token_end > span.start for span in selected)
        for token_start, token_end in offsets
    )
    if not any(mask):
        raise FinalAssistantMaskError(
            row_id,
            policy,
            "the rendered assistant selection produced no tokenizer overlap",
        )
    return mask


def require_nonzero_loss_mask(
    loss_mask: Sequence[object],
    *,
    row_id: str,
    policy: MaskPolicy,
) -> None:
    """Fail with row identity instead of crashing on an empty mask."""
    if not any(bool(value) for value in loss_mask):
        raise FinalAssistantMaskError(row_id, policy)


@dataclass(frozen=True, slots=True)
class TrainingWindowSummary:
    """Post-shift supervision retained by a configured sequence window."""

    row_count: int
    total_supervised_tokens: int
    retained_supervised_tokens: int
    truncated_supervised_tokens: int
    rows_with_truncated_supervision: int
    rows_with_all_supervision_truncated: int
    rows_without_retained_supervision: int
    maximum_sequence_length: int
    all_supervision_truncated_indices: tuple[int, ...]
    no_retained_supervision_indices: tuple[int, ...]
    all_supervision_truncated_row_ids: tuple[str, ...]
    no_retained_supervision_row_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "all_supervision_truncated_indices",
            "no_retained_supervision_indices",
            "all_supervision_truncated_row_ids",
            "no_retained_supervision_row_ids",
        ):
            result[key] = list(result[key])
        return result


def _value(row: object, field: str) -> object:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _integer(value: object, field: str, row_id: str) -> int:
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"prepared row {row_id!r} has a non-integer {field}: {value!r}"
        ) from error
    if result < 0:
        raise ValueError(f"prepared row {row_id!r} has a negative {field}: {result}")
    return result


def _sequence(row: object, field: str, row_id: str) -> Sequence[Any]:
    value = _value(row, field)
    if (
        value is None
        or isinstance(value, (str, bytes))
        or not hasattr(value, "__len__")
        or not hasattr(value, "__getitem__")
    ):
        raise ValueError(f"prepared row {row_id!r} has no sequence-valued {field}")
    return value  # type: ignore[return-value]


def summarize_training_window(
    rows: Iterable[object],
    total_seq_len: int,
) -> TrainingWindowSummary:
    """Audit loss after EAGLE-3's position-zero shift and window slice.

    The retained original mask region is ``[1 : total_seq_len + 1]``.
    """
    if (
        isinstance(total_seq_len, bool)
        or not isinstance(total_seq_len, int)
        or total_seq_len < 1
    ):
        raise ValueError("total_seq_len must be a positive integer")

    row_count = 0
    total_supervised = 0
    retained_supervised = 0
    truncated_supervised = 0
    rows_truncated = 0
    all_truncated_indices: list[int] = []
    no_retained_indices: list[int] = []
    all_truncated_ids: list[str] = []
    no_retained_ids: list[str] = []
    maximum_sequence_length = 0

    for row_index, row in enumerate(rows):
        raw_id = _value(row, "id")
        row_id = raw_id if isinstance(raw_id, str) and raw_id else str(row_index)
        input_ids = _sequence(row, "input_ids", row_id)
        loss_mask = _sequence(row, "loss_mask", row_id)
        sequence_length = _integer(_value(row, "seq_len"), "seq_len", row_id)
        if sequence_length > len(input_ids) or sequence_length > len(loss_mask):
            raise ValueError(
                f"prepared row {row_id!r} seq_len={sequence_length} exceeds "
                f"input_ids={len(input_ids)} or loss_mask={len(loss_mask)}"
            )

        row_count += 1
        maximum_sequence_length = max(maximum_sequence_length, sequence_length)
        supervised = sum(1 for value in loss_mask[1:sequence_length] if bool(value))
        retained_end = min(sequence_length, total_seq_len + 1)
        retained = sum(1 for value in loss_mask[1:retained_end] if bool(value))
        truncated = supervised - retained
        total_supervised += supervised
        retained_supervised += retained
        truncated_supervised += truncated
        if truncated:
            rows_truncated += 1
        if supervised and not retained:
            all_truncated_indices.append(row_index)
            all_truncated_ids.append(row_id)
        if not retained:
            no_retained_indices.append(row_index)
            no_retained_ids.append(row_id)

    return TrainingWindowSummary(
        row_count=row_count,
        total_supervised_tokens=total_supervised,
        retained_supervised_tokens=retained_supervised,
        truncated_supervised_tokens=truncated_supervised,
        rows_with_truncated_supervision=rows_truncated,
        rows_with_all_supervision_truncated=len(all_truncated_indices),
        rows_without_retained_supervision=len(no_retained_indices),
        maximum_sequence_length=maximum_sequence_length,
        all_supervision_truncated_indices=tuple(all_truncated_indices),
        no_retained_supervision_indices=tuple(no_retained_indices),
        all_supervision_truncated_row_ids=tuple(all_truncated_ids),
        no_retained_supervision_row_ids=tuple(no_retained_ids),
    )


def require_trainable_window(
    rows: Iterable[object],
    total_seq_len: int,
) -> TrainingWindowSummary:
    """Return the audit or reject a window that erases any target row."""
    summary = summarize_training_window(rows, total_seq_len)
    if summary.row_count == 0:
        raise ValueError("prepared dataset has no rows")
    if summary.rows_without_retained_supervision:
        row_ids = ",".join(summary.no_retained_supervision_row_ids[:20])
        suffix = "..." if len(summary.no_retained_supervision_row_ids) > 20 else ""
        raise ValueError(
            f"total_seq_len={total_seq_len} leaves no post-shift supervised target "
            f"tokens inside the effective training window for "
            f"{summary.rows_without_retained_supervision}/{summary.row_count} rows "
            f"(row ids: {row_ids}{suffix}); retained="
            f"{summary.retained_supervised_tokens}, truncated="
            f"{summary.truncated_supervised_tokens}"
        )
    return summary
