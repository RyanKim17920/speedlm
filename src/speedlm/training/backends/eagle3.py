"""EAGLE-3 backend powered by an injected Speculators process boundary.

The implementation lives in the tuner compatibility module while existing
orchestrator imports are supported.  This training namespace is the canonical
pluggable-backend entry point.
"""

from speedlm.training.masking import FinalAssistantMaskError
from speedlm.tuner.eagle3 import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_VERIFIER_MODEL,
    MAX_SCRATCH_BYTES,
    DraftMaterializer,
    DraftValidator,
    Eagle3Adapter,
    Eagle3Config,
    Eagle3Error,
    Eagle3Timeouts,
    HiddenStateExtractor,
    PreparedData,
    ScratchQuotaExceeded,
    SpeculatorsTrainer,
    StageTimeoutError,
    TraceSnapshot,
    TraceSnapshotLeaser,
    TrainingError,
    TrainingResult,
    TrainingRowRenderer,
    scratch_usage,
)


class Eagle3Backend(Eagle3Adapter):
    """Canonical name for the five-stage generalized EAGLE-3 backend."""


__all__ = [
    "DEFAULT_DRAFT_MODEL",
    "DEFAULT_VERIFIER_MODEL",
    "MAX_SCRATCH_BYTES",
    "DraftMaterializer",
    "DraftValidator",
    "Eagle3Adapter",
    "Eagle3Backend",
    "Eagle3Config",
    "Eagle3Error",
    "Eagle3Timeouts",
    "FinalAssistantMaskError",
    "HiddenStateExtractor",
    "PreparedData",
    "ScratchQuotaExceeded",
    "SpeculatorsTrainer",
    "StageTimeoutError",
    "TraceSnapshot",
    "TraceSnapshotLeaser",
    "TrainingError",
    "TrainingResult",
    "TrainingRowRenderer",
    "scratch_usage",
]
