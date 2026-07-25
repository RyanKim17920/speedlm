"""Idle-time EAGLE-3 auto-tuning for SpeedLM."""

from speedlm.training.base import BackendInfo, SpeculatorBackend
from speedlm.training.masking import FinalAssistantMaskError
from speedlm.tuner.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactManifest,
    ArtifactRegistry,
    ArtifactSpec,
)
from speedlm.tuner.eagle3 import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_VERIFIER_MODEL,
    Eagle3Adapter,
    Eagle3Config,
    ScratchQuotaExceeded,
    TrainingError,
)
from speedlm.tuner.idle import ActivitySource, IdleDetector, TuningPreempted
from speedlm.tuner.orchestrator import (
    BenchmarkGate,
    CycleOutcome,
    CycleResult,
    GateResult,
    RuntimeController,
    TunerOrchestrator,
)
from speedlm.tuner.state import (
    IllegalTransitionError,
    StateSnapshot,
    TunerState,
    TunerStateMachine,
)

__all__ = [
    "DEFAULT_DRAFT_MODEL",
    "DEFAULT_VERIFIER_MODEL",
    "ActivitySource",
    "Artifact",
    "ArtifactError",
    "ArtifactManifest",
    "ArtifactRegistry",
    "ArtifactSpec",
    "BackendInfo",
    "BenchmarkGate",
    "CycleOutcome",
    "CycleResult",
    "Eagle3Adapter",
    "Eagle3Config",
    "FinalAssistantMaskError",
    "GateResult",
    "IdleDetector",
    "IllegalTransitionError",
    "RuntimeController",
    "ScratchQuotaExceeded",
    "SpeculatorBackend",
    "StateSnapshot",
    "TrainingError",
    "TunerOrchestrator",
    "TunerState",
    "TunerStateMachine",
    "TuningPreempted",
]
