from __future__ import annotations

from speedlm.training.backends.eagle3 import Eagle3Backend, Eagle3Config
from speedlm.training.base import SpeculatorBackend
from speedlm.training.masking import (
    MaskPolicy,
    require_trainable_window,
    summarize_training_window,
)


def test_shifted_label_and_truncation_accounting() -> None:
    rows = [
        {
            "id": "long",
            "input_ids": [1] * 10,
            "loss_mask": [False] * 7 + [True] * 3,
            "seq_len": 10,
        },
        {
            "id": "short",
            "input_ids": [1] * 5,
            "loss_mask": [True, False, False, True, True],
            "seq_len": 5,
        },
    ]

    summary = summarize_training_window(rows, total_seq_len=8)

    assert summary.total_supervised_tokens == 5
    assert summary.retained_supervised_tokens == 4
    assert summary.truncated_supervised_tokens == 1
    assert summary.rows_with_truncated_supervision == 1
    assert summary.rows_with_all_supervision_truncated == 0
    assert require_trainable_window(rows, 8) == summary


def test_shift_drops_position_zero_and_reports_row_id() -> None:
    rows = [
        {
            "id": "position-zero-only",
            "input_ids": [1, 2],
            "loss_mask": [True, False],
            "seq_len": 2,
        }
    ]

    try:
        require_trainable_window(rows, 8)
    except ValueError as error:
        assert "position-zero-only" in str(error)
        assert "post-shift" in str(error)
    else:
        raise AssertionError("expected shifted all-zero supervision to fail")


def test_eagle3_backend_satisfies_protocol_and_declares_distillation_contract() -> None:
    backend = object.__new__(Eagle3Backend)
    config = Eagle3Config(mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS)

    assert isinstance(backend, SpeculatorBackend)
    assert config.from_pretrained == "RedHatAI/gpt-oss-20b-speculator.eagle3"
    assert config.effective_training_params["distillation_loss"] == "soft_kl"
    assert config.effective_training_params["draft_vocabulary"] == "reduced_d2t_t2d"
    assert config.effective_training_params["num_speculative_steps"] == 3
    assert config.effective_training_params["ttt_loss_reduction"] == "sum"
