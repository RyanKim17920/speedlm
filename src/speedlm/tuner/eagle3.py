"""Injectable GPT-OSS EAGLE-3 training pipeline.

This module owns validation, quotas, abort propagation, and the
``--from-pretrained`` contract. GPU/process mechanics remain behind protocols so
the login-node test suite never imports CUDA, vLLM, or Speculators.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from speedlm.training.base import BackendInfo
from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy

# Hidden-state scratch scales with hidden_size x num_aux_layers x tokens.
# Larger models need proportionally more space.  This constant is a hard
# safety ceiling; the per-instance scratch_quota_bytes in Eagle3Config
# should typically be smaller.
MAX_SCRATCH_BYTES = 20 * 1024 * 1024 * 1024

AbortCheck = Callable[[], bool]


class Eagle3Error(RuntimeError):
    """Base class for EAGLE-3 adapter failures."""


class AuxLayerCountMismatch(Eagle3Error):
    """The aux-layer count does not match the drafter's expectation.

    The drafter's ``fc_input_size`` is derived from its configured
    ``num_aux_hidden_states`` (or the length of
    ``eagle_aux_hidden_state_layer_ids`` in ``eagle_config``).
    A mismatch means a shape error in the forward pass.
    """

    def __init__(
        self, expected: int, actual: int, drafter_model: str | None = None
    ) -> None:
        self.expected = expected
        self.actual = actual
        source = f" (drafter: {drafter_model})" if drafter_model else ""
        super().__init__(
            f"aux-layer count mismatch: drafter expects {expected} aux layers{source}, "
            f"but {actual} were provided"
        )


class ScratchQuotaExceeded(Eagle3Error):
    """The per-cycle scratch directory exceeded its hard byte limit."""

    def __init__(self, used_bytes: int, quota_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.quota_bytes = quota_bytes
        super().__init__(
            f"scratch quota exceeded: used {used_bytes} bytes, limit {quota_bytes} bytes "
            f"(raise tuning.scratch_quota_bytes to increase)"
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
    val_loss: float | None = None


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

    verifier_model: str
    draft_model: str
    from_pretrained: str
    verifier_revision: str | None = None
    draft_revision: str | None = None
    target_layer_ids: tuple[int, ...] | None = None
    sequence_length: int = 16_384
    num_speculative_steps: int = 3
    mask_policy: MaskPolicy = MaskPolicy.FINAL_TURN_ALL_CHANNELS
    training_params: Mapping[str, object] = field(default_factory=dict)
    timeouts: Eagle3Timeouts = field(default_factory=Eagle3Timeouts)
    scratch_quota_bytes: int = MAX_SCRATCH_BYTES

    def __post_init__(self) -> None:
        for model_name, model_value in (
            ("verifier_model", self.verifier_model),
            ("draft_model", self.draft_model),
            ("from_pretrained", self.from_pretrained),
        ):
            if not isinstance(model_value, str) or not model_value:
                raise ValueError(f"{model_name} must be a non-empty string")
        for revision_name, revision_value in (
            ("verifier_revision", self.verifier_revision),
            ("draft_revision", self.draft_revision),
        ):
            if revision_value is not None and (
                not isinstance(revision_value, str) or not revision_value
            ):
                raise ValueError(f"{revision_name} must be a non-empty string or null")
        if self.target_layer_ids is not None and (
            not isinstance(self.target_layer_ids, tuple)
            or not self.target_layer_ids
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in self.target_layer_ids
            )
            or len(set(self.target_layer_ids)) != len(self.target_layer_ids)
        ):
            raise ValueError("target_layer_ids must be unique non-negative integers")
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 1
        ):
            raise ValueError("sequence_length must be a positive integer")
        if self.num_speculative_steps != 3:
            raise ValueError("EAGLE-3 requires exactly 3 speculative/TTT steps")
        if not isinstance(self.mask_policy, MaskPolicy):
            raise ValueError("mask_policy must be an explicit MaskPolicy")
        if (
            isinstance(self.scratch_quota_bytes, bool)
            or not isinstance(self.scratch_quota_bytes, int)
            or self.scratch_quota_bytes <= 0
            or self.scratch_quota_bytes > MAX_SCRATCH_BYTES
        ):
            raise ValueError(
                "scratch_quota_bytes must be in 1..20 GiB "
                "(field: tuning.scratch_quota_bytes)"
            )

    @property
    def effective_training_params(self) -> Mapping[str, object]:
        """Parameters including the verified EAGLE-3 distillation contract."""
        result = dict(self.training_params)
        result.update(
            {
                "sequence_length": self.sequence_length,
                "num_speculative_steps": self.num_speculative_steps,
                "distillation_loss": "soft_kl",
                "draft_vocabulary": "reduced_d2t_t2d",
                "ttt_loss_reduction": "sum",
                "mask_policy": self.mask_policy.value,
            }
        )
        if self.verifier_revision is not None:
            result["verifier_revision"] = self.verifier_revision
        if self.draft_revision is not None:
            result["draft_revision"] = self.draft_revision
        if self.target_layer_ids is not None:
            result["target_layer_ids"] = self.target_layer_ids
        return result


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

    def describe(self) -> BackendInfo:
        """Return backend-neutral metadata for orchestration and provenance.

        The pinned verifier revision travels with the training parameters so
        it lands in the published artifact manifest.  Without it the manifest
        names the verifier but not *which* verifier, and a cycle trained
        against a silently updated upstream model is indistinguishable from
        one that was not.

        The field is always present, null included.  Resolution is best
        effort, so an absent key would be ambiguous between "this build does
        not record revisions" and "this cycle could not be pinned"; an
        explicit null says the cycle ran unpinned and says it in the artifact.
        """
        params = dict(self.config.training_params)
        params["verifier_revision"] = self.config.verifier_revision
        return BackendInfo(
            verifier_model=self.config.verifier_model,
            draft_model=self.config.draft_model,
            from_pretrained=self.config.from_pretrained,
            training_params=params,
        )

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
            rows_path = _call_with_supported_keywords(
                self._renderer.render_rows,
                snapshot,
                work_dir / "training-rows",
                timeout_seconds=self.config.timeouts.render,
                should_abort=should_abort,
                mask_policy=self.config.mask_policy,
                sequence_length=self.config.sequence_length,
            )
        except FinalAssistantMaskError:
            raise
        except Exception as exc:
            if exc.__class__.__name__ == "FinalAssistantMaskError":
                raise FinalAssistantMaskError(
                    "<unknown>", self.config.mask_policy, str(exc)
                ) from exc
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
        hidden_states = _call_with_supported_keywords(
            self._extractor.extract_hidden_states,
            prepared.rows_path,
            work_dir / "hidden-states",
            verifier_model=self.config.verifier_model,
            verifier_revision=self.config.verifier_revision,
            target_layer_ids=self.config.target_layer_ids,
            sequence_length=self.config.sequence_length,
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
            training_params=self.config.effective_training_params,
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
        """Compatibility helper that runs the distinct materialize/validate stages."""
        draft_directory = self.materialize(
            result,
            work_dir,
            should_abort=should_abort,
        )
        self.validate(draft_directory, should_abort=should_abort)
        return draft_directory

    def materialize(
        self,
        result: TrainingResult,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Path:
        """Build the separate standalone draft-model directory."""
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
        return draft_directory

    def validate(
        self,
        draft_directory: Path,
        *,
        should_abort: AbortCheck,
    ) -> None:
        """Validate a materialized draft against the configured verifier."""
        work_dir = draft_directory.parent
        self._check(work_dir, should_abort)
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


def _call_with_supported_keywords[T](
    function: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    """Pass generalized parameters while preserving legacy injected effects."""
    parameters = inspect.signature(function).parameters.values()
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    supported_names = {parameter.name for parameter in parameters}
    supported = (
        kwargs
        if accepts_arbitrary
        else {name: value for name, value in kwargs.items() if name in supported_names}
    )
    return function(*args, **supported)
