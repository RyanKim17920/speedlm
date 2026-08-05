from __future__ import annotations

from speedlm.training.dilate_prepared_loss_mask import left_dilate_loss_mask


def test_left_dilation_adds_exactly_span_start_predecessors() -> None:
    masks = [
        [],
        [False, False, False],
        [True],
        [True, True, False],
        [False, True, True, False, True, False, False],
        [False, True, False, True, False, True],
    ]

    for mask in masks:
        original = [bool(value) for value in mask]
        dilated = left_dilate_loss_mask(mask)
        expected_additions = {
            index - 1
            for index, value in enumerate(original)
            if index > 0 and value and not original[index - 1]
        }
        actual_additions = {
            index
            for index, (before, after) in enumerate(zip(original, dilated, strict=True))
            if after and not before
        }

        assert actual_additions == expected_additions
        assert len(dilated) == len(original)
        assert all(after for before, after in zip(original, dilated, strict=True) if before)

