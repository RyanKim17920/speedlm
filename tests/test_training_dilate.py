"""Tests for training/dilate_prepared_loss_mask.py.

left_dilate_loss_mask extends each nonzero span one position to the left,
creating a guard row at the boundary between masked and unmasked regions.
dilate_prepared_dataset applies this across a persisted dataset.

Both are pure data transforms, so they are easy to test against hand-computed
expected outputs.
"""

from __future__ import annotations

import pytest

from speedlm.training.dilate_prepared_loss_mask import (
    _dilate_row,
    _plain_mask,
    left_dilate_loss_mask,
)

# ---------------------------------------------------------------------------
# left_dilate_loss_mask
# ---------------------------------------------------------------------------


def test_left_dilate_preserves_zeros_when_no_spans() -> None:
    mask = [0, 0, 0, 0]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [False, False, False, False]


def test_left_dilate_extends_span_start_left() -> None:
    mask = [0, 1, 1, 0]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [True, True, True, False]


def test_left_dilate_does_not_wrap_around() -> None:
    mask = [1, 1, 0, 0]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [True, True, False, False]


def test_left_dilate_extends_multiple_spans_independently() -> None:
    mask = [0, 1, 1, 0, 0, 1, 1, 1]
    dilated = left_dilate_loss_mask(mask)
    # Span at [1,2]: dilates position 0 left. Span at [5,6,7]: dilates position 4 left.
    # Each span start extends one position left into the gap before it.
    assert dilated == [True, True, True, False, True, True, True, True]


def test_left_dilate_does_not_chain_dilations() -> None:
    """Decisions are made from the original mask, so a newly enabled position
    never triggers another dilation. A span at position 2 does not dilate
    position 1, and then dilate position 0."""
    mask = [0, 0, 1, 0]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [False, True, True, False]


def test_left_dilate_single_element_span() -> None:
    mask = [0, 0, 1, 0, 0]
    dilated = left_dilate_loss_mask(mask)
    # Span starts at index 2, so position 1 gets dilated left.
    assert dilated == [False, True, True, False, False]


def test_left_dilate_single_element_span_at_start() -> None:
    mask = [1, 0, 0]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [True, False, False]


def test_left_dilate_full_mask_unchanged() -> None:
    mask = [1, 1, 1, 1]
    dilated = left_dilate_loss_mask(mask)
    assert dilated == [True, True, True, True]


def test_left_dilate_empty_mask() -> None:
    dilated = left_dilate_loss_mask([])
    assert dilated == []


def test_left_dilate_single_element() -> None:
    dilated = left_dilate_loss_mask([1])
    assert dilated == [True]


# ---------------------------------------------------------------------------
# _plain_mask
# ---------------------------------------------------------------------------


def test_plain_mask_list() -> None:
    assert _plain_mask([1, 0, 1]) == [1, 0, 1]


def test_plain_mask_tuple() -> None:
    assert _plain_mask((True, False, True)) == [True, False, True]


def test_plain_mask_raises_on_string() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        _plain_mask("not a mask")


def test_plain_mask_raises_on_bytes() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        _plain_mask(b"not a mask")


# ---------------------------------------------------------------------------
# _dilate_row
# ---------------------------------------------------------------------------


def test_dilate_row_on_missing_loss_mask() -> None:
    with pytest.raises(ValueError, match="no loss_mask"):
        _dilate_row({"input_ids": [1, 2, 3]})


def test_dilate_row_applies_dilation() -> None:
    row = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 0]}
    result = _dilate_row(row)
    assert result["loss_mask"] == [True, True, True, False]
