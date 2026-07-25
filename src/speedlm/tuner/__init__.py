"""Idle-time EAGLE-3 auto-tuning for SpeedLM."""

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
    FinalAssistantMaskError,
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
    "StateSnapshot",
    "TrainingError",
    "TunerOrchestrator",
    "TunerState",
    "TunerStateMachine",
    "TuningPreempted",
]
