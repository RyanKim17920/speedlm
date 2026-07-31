"""Generalized speculative-training primitives."""

from speedlm.training.base import BackendInfo, SpeculatorBackend, TrainingBackendError
from speedlm.training.masking import (
    FinalAssistantMaskError,
    MaskPolicy,
    TrainingWindowSummary,
    require_trainable_window,
    summarize_training_window,
)
from speedlm.training.rows import (
    PreparedTrainingRow,
    TrainingRow,
    training_row_from_trace,
)

__all__ = [
    "BackendInfo",
    "FinalAssistantMaskError",
    "MaskPolicy",
    "PreparedTrainingRow",
    "SpeculatorBackend",
    "TrainingBackendError",
    "TrainingRow",
    "TrainingWindowSummary",
    "require_trainable_window",
    "summarize_training_window",
    "training_row_from_trace",
]
