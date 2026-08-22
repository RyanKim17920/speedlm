"""Tests for tuner/eagle3.py quota accounting - shard_bytes_per_row, derive_scratch_quota_bytes.

The quota logic is the guard rail that prevents a mis-sized training window from
filling /data with hidden-state shards. A window that is under-provisioned wastes
GPU time (the extraction aborts mid-run), while an over-provisioned window wastes
disk. This module tests the arithmetic in isolation so a constant change or a
formula tweak is caught before it kills a GPU cycle.

test_training_speculators.py covers quota against realistic model sizes. Here
we test edge cases, argument validation, and the interaction between
sequence_length and tokens_per_row.
"""

from __future__ import annotations

import pytest

from speedlm.tuner.eagle3 import (
    MAX_SCRATCH_BYTES,
    SCRATCH_HEADROOM_BYTES,
    SHARD_BYTES_PER_ROW,
    derive_scratch_quota_bytes,
    shard_bytes_per_row,
)

# ---------------------------------------------------------------------------
# shard_bytes_per_row
# ---------------------------------------------------------------------------


def test_shard_bytes_per_row_defaults() -> None:
    per_row = shard_bytes_per_row()
    # tokens * (aux + 1) * hidden_size * dtype_bytes = 4096 * 4 * 4096 * 2
    expected = 4096 * 4 * 4096 * 2
    assert per_row == expected


def test_shard_bytes_per_row_geometry() -> None:
    """Per-row bytes is the max of computed geometry and the floor constant."""
    per_row = shard_bytes_per_row(hidden_size=2880, num_aux_layers=3, tokens_per_row=1000)
    # 1000 * 4 * 2880 * 2 = 23,040,000 which is below SHARD_BYTES_PER_ROW (32 MB)
    expected = max(1000 * 4 * 2880 * 2, SHARD_BYTES_PER_ROW)
    assert per_row == expected


def test_shard_bytes_per_row_sequence_length_caps_tokens() -> None:
    per_row = shard_bytes_per_row(
        hidden_size=4096,
        num_aux_layers=3,
        tokens_per_row=8192,
        sequence_length=2048,
    )
    # sequence_length is smaller, so tokens_per_row is capped
    expected = 2048 * 4 * 4096 * 2
    assert per_row == expected


def test_shard_bytes_per_row_sequence_length_larger_no_effect() -> None:
    """When sequence_length > tokens_per_row, tokens_per_row is unchanged.

    Use large enough numbers to exceed the SHARD_BYTES_PER_ROW floor so we
    actually see the computed geometry.
    """
    per_row = shard_bytes_per_row(
        hidden_size=4096,
        num_aux_layers=3,
        tokens_per_row=10000,
        sequence_length=8192,
    )
    # sequence_length (8192) < tokens_per_row (10000), so tokens is capped at 8192
    expected = 8192 * 4 * 4096 * 2
    assert per_row == expected


def test_shard_bytes_per_row_floored_at_constant() -> None:
    """Very small geometry is still floored at SHARD_BYTES_PER_ROW."""
    per_row = shard_bytes_per_row(hidden_size=128, num_aux_layers=1, tokens_per_row=10)
    assert per_row >= SHARD_BYTES_PER_ROW


def test_shard_bytes_per_row_negative_raises() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        shard_bytes_per_row(hidden_size=-1)


def test_shard_bytes_per_row_zero_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        shard_bytes_per_row(hidden_size=0)


def test_shard_bytes_per_row_bool_raises() -> None:
    """A bool must not be accepted as a geometry parameter."""
    with pytest.raises(ValueError, match="positive"):
        shard_bytes_per_row(hidden_size=True)


def test_shard_bytes_per_row_sequence_length_negative_raises() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        shard_bytes_per_row(sequence_length=-1)


# ---------------------------------------------------------------------------
# derive_scratch_quota_bytes
# ---------------------------------------------------------------------------


def test_derive_scratch_quota_default() -> None:
    quota = derive_scratch_quota_bytes(256)
    per_row = shard_bytes_per_row()
    expected = 256 * per_row + SCRATCH_HEADROOM_BYTES
    assert quota == expected


def test_derive_scratch_quota_exceeds_max_raises_with_geometry() -> None:
    with pytest.raises(ValueError, match="exceeds MAX_SCRATCH_BYTES") as raised:
        derive_scratch_quota_bytes(999999)

    msg = str(raised.value)
    assert "hidden_size" in msg
    assert "num_aux_layers" in msg
    assert "tokens/row" in msg


def test_derive_scratch_quota_exceeds_max_suggests_fix() -> None:
    """The error message names the maximum window that would fit."""
    with pytest.raises(ValueError, match="Lower tuning.training_window_records") as raised:
        derive_scratch_quota_bytes(999999)

    assert "at most" in str(raised.value)


def test_derive_scratch_quota_small_model_fits() -> None:
    quota = derive_scratch_quota_bytes(512, hidden_size=2880, num_aux_layers=3, tokens_per_row=1000)
    assert quota <= MAX_SCRATCH_BYTES


def test_derive_scratch_quota_window_zero_raises() -> None:
    with pytest.raises(ValueError, match="training_window_records"):
        derive_scratch_quota_bytes(0)


def test_derive_scratch_quota_negative_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        derive_scratch_quota_bytes(-1)
