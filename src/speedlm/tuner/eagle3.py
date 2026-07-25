"""Injectable GPT-OSS EAGLE-3 training pipeline.

This module owns validation, quotas, abort propagation, and the
``--from-pretrained`` contract. GPU/process mechanics remain behind protocols so
the login-node test suite never imports CUDA, vLLM, or Speculators.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

MAX_SCRATCH_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_VERIFIER_MODEL = "openai/gpt-oss-20b"
DEFAULT_DRAFT_MODEL = "RedHatAI/gpt-oss-20b-speculator.eagle3"

AbortCheck = Callable[[], bool]


class Eagle3Error(RuntimeError):
    """Base class for EAGLE-3 adapter failures."""


class FinalAssistantMaskError(Eagle3Error):
    """Training rows contain no trainable final-assistant tokens."""


class ScratchQuotaExceeded(Eagle3Error):
    """The per-cycle scratch directory exceeded its hard byte limit."""

    def __init__(self, used_bytes: int, quota_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.quota_bytes = quota_bytes
        super().__init__(
            f"scratch quota exceeded: used {used_bytes} bytes, limit {quota_bytes} bytes"
        )


class StageTimeoutError(Eagle3Error):
    """An external stage exceeded its configured wall-clock timeout."""


class TrainingError(Eagle3Error):
    """Speculators training failed, retaining stderr for diagnosis."""

    def __init__(self, message: str, *, stderr: str) -> None:
        self.stderr = stderr
        detail = f"{message}; stderr: {stderr}" if stderr else message
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    """A leased, immutable trace snapshot."""

    path: Path
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Trace lease and rendered Speculators training rows."""

    snapshot: TraceSnapshot
    rows_path: Path


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result returned by the Speculators process boundary."""

    checkpoint_best: Path
    returncode: int
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class Eagle3Timeouts:
    """Per-effect wall clock limits in seconds."""

    lease: float = 60.0
    render: float = 300.0
    extract: float = 3_600.0
    train: float = 14_400.0
    materialize: float = 600.0
    validate: float = 300.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lease", self.lease),
            ("render", self.render),
            ("extract", self.extract),
            ("train", self.train),
            ("materialize", self.materialize),
            ("validate", self.validate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} timeout must be a positive number")


@dataclass(frozen=True, slots=True)
class Eagle3Config:
    """Models and training controls for one EAGLE-3 adapter."""

    verifier_model: str = DEFAULT_VERIFIER_MODEL
    draft_model: str = DEFAULT_DRAFT_MODEL
    from_pretrained: str = DEFAULT_DRAFT_MODEL
    training_params: Mapping[str, object] = field(default_factory=dict)
    timeouts: Eagle3Timeouts = field(default_factory=Eagle3Timeouts)
    scratch_quota_bytes: int = MAX_SCRATCH_BYTES

    def __post_init__(self) -> None:
        for name, value in (
            ("verifier_model", self.verifier_model),
            ("draft_model", self.draft_model),
            ("from_pretrained", self.from_pretrained),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.scratch_quota_bytes, bool)
            or not isinstance(self.scratch_quota_bytes, int)
            or self.scratch_quota_bytes <= 0
            or self.scratch_quota_bytes > MAX_SCRATCH_BYTES
        ):
            raise ValueError("scratch_quota_bytes must be in 1..5 GiB")


class TraceSnapshotLeaser(Protocol):
    """Lease a stable trace snapshot into *destination*."""

    def lease_snapshot(
        self,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> TraceSnapshot: ...


class TrainingRowRenderer(Protocol):
    """Render leased traces into final-assistant-masked training rows."""

    def render_rows(
        self,
        snapshot: TraceSnapshot,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> Path: ...


class HiddenStateExtractor(Protocol):
    """Extract verifier hidden states on the GPU node."""

    def extract_hidden_states(
        self,
        rows_path: Path,
        destination: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> Path: ...


class SpeculatorsTrainer(Protocol):
    """Run multi-step Speculators training from an existing draft model."""

    def train(
        self,
        hidden_states_path: Path,
        destination: Path,
        *,
        from_pretrained: str,
        training_params: Mapping[str, object],
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> TrainingResult: ...


class DraftMaterializer(Protocol):
    """Convert ``checkpoint_best`` into a standalone draft-model directory."""

    def materialize(
        self,
        checkpoint_best: Path,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> Path: ...


class DraftValidator(Protocol):
    """Validate the standalone draft against the verifier model."""

    def validate(
        self,
        draft_directory: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None: ...


class Eagle3Adapter:
    """Coordinate injected EAGLE-3 effects with hard safety contracts."""

    def __init__(
        self,
        config: Eagle3Config,
        *,
        leaser: TraceSnapshotLeaser,
        renderer: TrainingRowRenderer,
        extractor: HiddenStateExtractor,
        trainer: SpeculatorsTrainer,
        materializer: DraftMaterializer,
        validator: DraftValidator,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._leaser = leaser
        self._renderer = renderer
        self._extractor = extractor
        self._trainer = trainer
        self._materializer = materializer
        self._validator = validator
        self._clock = clock

    def prepare(self, work_dir: Path, *, should_abort: AbortCheck) -> PreparedData:
        """Lease traces and render training rows without touching a GPU."""
        work_dir.mkdir(parents=True, exist_ok=True)
        self._check(work_dir, should_abort)
        started = self._clock()
        snapshot = self._leaser.lease_snapshot(
            work_dir / "trace-snapshot",
            timeout_seconds=self.config.timeouts.lease,
            should_abort=should_abort,
        )
        self._finish_stage(
            "trace lease", started, self.config.timeouts.lease, work_dir, should_abort
        )
        started = self._clock()
        try:
            rows_path = self._renderer.render_rows(
                snapshot,
                work_dir / "training-rows",
                timeout_seconds=self.config.timeouts.render,
                should_abort=should_abort,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "FinalAssistantMaskError":
                raise FinalAssistantMaskError(str(exc)) from exc
            raise
        self._finish_stage(
            "training row render",
            started,
            self.config.timeouts.render,
            work_dir,
            should_abort,
        )
        return PreparedData(snapshot=snapshot, rows_path=rows_path)

    def extract(
        self,
        prepared: PreparedData,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Path:
        """Extract verifier hidden states."""
        self._check(work_dir, should_abort)
        started = self._clock()
        hidden_states = self._extractor.extract_hidden_states(
            prepared.rows_path,
            work_dir / "hidden-states",
            verifier_model=self.config.verifier_model,
            timeout_seconds=self.config.timeouts.extract,
            should_abort=should_abort,
        )
        self._finish_stage(
            "hidden-state extraction",
            started,
            self.config.timeouts.extract,
            work_dir,
            should_abort,
        )
        return hidden_states

    def train(
        self,
        hidden_states: Path,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> TrainingResult:
        """Run Speculators training, always with ``--from-pretrained``."""
        self._check(work_dir, should_abort)
        if not self.config.from_pretrained:
            raise Eagle3Error("refusing to train EAGLE-3 from scratch")
        started = self._clock()
        result = self._trainer.train(
            hidden_states,
            work_dir / "speculators-training",
            from_pretrained=self.config.from_pretrained,
            training_params=self.config.training_params,
            timeout_seconds=self.config.timeouts.train,
            should_abort=should_abort,
        )
        self._finish_stage(
            "Speculators training",
            started,
            self.config.timeouts.train,
            work_dir,
            should_abort,
        )
        if result.returncode != 0:
            raise TrainingError(
                f"Speculators exited with status {result.returncode}",
                stderr=result.stderr,
            )
        if not result.checkpoint_best.exists():
            raise TrainingError(
                f"checkpoint_best is missing: {result.checkpoint_best}",
                stderr=result.stderr,
            )
        return result

    def materialize_and_validate(
        self,
        result: TrainingResult,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Path:
        """Build and validate the separate standalone draft-model directory."""
        self._check(work_dir, should_abort)
        started = self._clock()
        draft_directory = self._materializer.materialize(
            result.checkpoint_best,
            work_dir / "draft-model",
            timeout_seconds=self.config.timeouts.materialize,
            should_abort=should_abort,
        )
        self._finish_stage(
            "draft materialization",
            started,
            self.config.timeouts.materialize,
            work_dir,
            should_abort,
        )
        if not draft_directory.is_dir():
            raise Eagle3Error(
                f"materializer did not return a draft directory: {draft_directory}"
            )
        started = self._clock()
        self._validator.validate(
            draft_directory,
            verifier_model=self.config.verifier_model,
            timeout_seconds=self.config.timeouts.validate,
            should_abort=should_abort,
        )
        self._finish_stage(
            "draft validation",
            started,
            self.config.timeouts.validate,
            work_dir,
            should_abort,
        )
        return draft_directory

    def _finish_stage(
        self,
        stage: str,
        started: float,
        timeout: float,
        work_dir: Path,
        should_abort: AbortCheck,
    ) -> None:
        elapsed = self._clock() - started
        if elapsed > timeout:
            raise StageTimeoutError(
                f"{stage} exceeded {timeout:.3f}s timeout (elapsed {elapsed:.3f}s)"
            )
        self._check(work_dir, should_abort)

    def _check(self, work_dir: Path, should_abort: AbortCheck) -> None:
        if should_abort():
            from speedlm.tuner.idle import TuningPreempted

            raise TuningPreempted("incoming request preempted EAGLE-3 stage")
        used = scratch_usage(work_dir)
        if used > self.config.scratch_quota_bytes:
            raise ScratchQuotaExceeded(used, self.config.scratch_quota_bytes)


def scratch_usage(path: Path) -> int:
    """Return bytes occupied by regular files beneath *path* without following links."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            total += entry.lstat().st_size
        elif entry.is_file():
            total += entry.stat().st_size
    return total
