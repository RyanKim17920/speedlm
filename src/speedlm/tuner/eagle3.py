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

from speedlm.traces.redact import Redactor
from speedlm.training.base import BackendInfo
from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy

# Hidden-state scratch scales with hidden_size x num_aux_layers x tokens.
# Larger models need proportionally more space.  This constant is a hard
# safety ceiling; the per-instance scratch_quota_bytes in Eagle3Config
# should typically be smaller.
MAX_SCRATCH_BYTES = 20 * 1024 * 1024 * 1024

#: Byte budget charged to one leased training row's hidden-state shard.
#:
#: Extraction writes exactly one ``hs_<index>.safetensors`` per leased row --
#: ``data_generation_offline.py --max-samples`` is the leased row count -- and a
#: shard is ``tokens_in_row x (num_aux_layers + 1) x hidden_size x 2`` bytes.
#: For gpt-oss-20b (``hidden_size`` 2880, three aux layers plus the appended
#: target layer) that is ``4 x 2880 x 2 = 23,040`` bytes per token, so 32 MiB
#: buys a ~1,456-token row.  The number is not a guess: job 369325's failure
#: inventory sampled 64 shards with a mean of 15.8 MB and a **maximum of
#: 31,460,688 B**, and 32 MiB is that maximum rounded up to a power of two.
#:
#: It is a budget, not a bound.  ``sequence_length`` permits rows several times
#: longer, which is what :data:`SCRATCH_HEADROOM_BYTES` and the
#: :data:`MAX_SCRATCH_BYTES` ceiling above it are for.
SHARD_BYTES_PER_ROW = 32 * 1024 * 1024

#: Scratch occupied by everything that is not a hidden-state shard.
#:
#: At the moment job 369325 aborted, its inventory held 2,096,091 B of trace
#: snapshot, 1,951,186 B of rendered conversations and 1,252,347 B of training
#: rows -- 5.3 MB in total.  Those three are negligible; the term this constant
#: actually covers is ``speculators-training``, the trainer's checkpoint and
#: optimizer state, which had not been created yet when that run died and so
#: was never measured.  1 GiB is therefore rounded up hard rather than fitted,
#: which is the honest treatment of a term with no observation behind it.
SCRATCH_HEADROOM_BYTES = 1024 * 1024 * 1024


def derive_scratch_quota_bytes(training_window_records: int) -> int:
    """Return the scratch quota a *training_window_records*-row cycle needs.

    The quota is derived from the one thing that actually sizes scratch --
    the number of hidden-state shards, which equals the number of leased
    training rows, which is bounded above by ``tuning.training_window_records``
    -- rather than picked as a round number::

        quota = training_window_records * SHARD_BYTES_PER_ROW
                + SCRATCH_HEADROOM_BYTES

    Job 369325 is the worked example of getting this wrong.  It leased 409 rows
    against a 5 GiB (5,368,709,120 B) quota and aborted at 5,384,048,233 B, a
    0.29 % overshoot that read like a rounding accident.  It was not: at the
    observed 15.8 MB mean shard those 409 rows needed ``409 x 15.8 MB =
    6.47 GB``, so the run was ~21 % under-provisioned on the mean and would
    have needed ``512 x 15.8 MB = 8.09 GB`` had the window filled.  The quota
    was not marginally too small, it could never have completed.

    Args:
        training_window_records: the configured lease ceiling in records.

    Returns:
        The derived quota in bytes.

    Raises:
        ValueError: if *training_window_records* is not a positive integer, or
            if the derived quota exceeds :data:`MAX_SCRATCH_BYTES` -- a window
            that cannot be provisioned within the hard ceiling is a
            configuration error, not something to silently clamp.
    """
    if isinstance(training_window_records, bool) or not isinstance(
        training_window_records, int
    ):
        raise ValueError("training_window_records must be a positive integer")
    if training_window_records <= 0:
        raise ValueError(
            f"training_window_records must be a positive integer, "
            f"got {training_window_records}"
        )
    derived = training_window_records * SHARD_BYTES_PER_ROW + SCRATCH_HEADROOM_BYTES
    if derived > MAX_SCRATCH_BYTES:
        raise ValueError(
            f"a {training_window_records}-record window needs {derived} bytes of "
            f"scratch, which exceeds MAX_SCRATCH_BYTES ({MAX_SCRATCH_BYTES}); "
            f"lower tuning.training_window_records or raise the ceiling"
        )
    return derived


AbortCheck = Callable[[], bool]

#: Names the checkpoint one cycle warm-starts from, resolved when it trains.
#:
#: The same shape as :data:`speedlm.gate.runner.StockDraft`, and for the same
#: reason: what counts as "the head we are improving on" is a *durable pointer
#: that moves*, so anything naming it has to ask at the moment it needs the
#: answer rather than capture a value at composition time.
WarmStartResolver = Callable[[], str]


class Eagle3Error(RuntimeError):
    """Base class for EAGLE-3 adapter failures."""

    #: Cleanup failures that occurred *while this error was propagating*.
    #:
    #: Job 369325 lost its root cause because cleanup raised on the way out and
    #: the secondary ``OSError`` replaced the ``ScratchQuotaExceeded`` it was
    #: cleaning up after.  Cleanup problems are now recorded here (and as
    #: exception notes) instead of being allowed to become the reported error.
    cleanup_errors: tuple[str, ...] = ()


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
        #: ``stderr`` stays verbatim for callers that persist it to the
        #: owner-only training-log sidecar. The exception *message* travels
        #: much further -- tracebacks, CLI output, structured logs -- so the
        #: copy interpolated there is redacted first. A subprocess that echoes
        #: an API key or a token in a diagnostic line must not turn a training
        #: failure into a credential leak.
        self.stderr = stderr
        if stderr:
            redacted, _ = Redactor().redact_text(stderr)
            detail = f"{message}; stderr: {redacted}"
        else:
            detail = message
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

    #: Held unresolved on purpose; see :meth:`_warm_start`.  ``None`` keeps the
    #: historical behaviour exactly: every cycle trains from
    #: ``config.from_pretrained``.
    _warm_start_resolver: WarmStartResolver | None = None

    #: What the most recent :meth:`train` actually trained from.
    #:
    #: :meth:`describe` reports this rather than the configured value, because
    #: the configured value stopped being the answer the moment a resolver could
    #: return something else -- and the artifact manifest's ``base_draft`` is
    #: the *only* record of which head a cycle built on.  Recording a
    #: configured-but-unused value there is the same class of defect as a
    #: manifest asserting a verifier revision the cycle could not satisfy (see
    #: :meth:`~speedlm.training.backends.eagle3.Eagle3Backend.describe`).
    #:
    #: Both are class-level defaults so that an adapter assembled without
    #: ``__init__`` -- which tests do to exercise ``describe`` in isolation --
    #: still answers, exactly as ``Eagle3Backend._state`` does.
    _resolved_warm_start: str | None = None

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
        warm_start_resolver: WarmStartResolver | None = None,
    ) -> None:
        self.config = config
        self._leaser = leaser
        self._renderer = renderer
        self._extractor = extractor
        self._trainer = trainer
        self._materializer = materializer
        self._validator = validator
        self._clock = clock
        self._warm_start_resolver = warm_start_resolver

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

        ``from_pretrained`` is the *resolved* warm start once a cycle has
        trained, so the chain is reconstructable: each artifact's manifest
        ``base_draft`` names the artifact directory it was trained from, and
        following that field back terminates at the profile's stock drafter.

        Before any training it is the configured stock drafter, which is also
        what :meth:`speedlm.tuner.orchestrator.TunerOrchestrator._active_draft`
        needs from it: that caller reads this field only when the registry has
        no active artifact, and a resolved value can only ever *be* an artifact
        directory when the registry had one.  So the fallback branch is
        unreachable with a stale directory by construction.
        """
        params = dict(self.config.training_params)
        params["verifier_revision"] = self.config.verifier_revision
        return BackendInfo(
            verifier_model=self.config.verifier_model,
            draft_model=self.config.draft_model,
            from_pretrained=self._resolved_warm_start or self.config.from_pretrained,
            training_params=params,
        )

    def _warm_start(self) -> str:
        """The checkpoint this cycle trains from, resolved when it trains.

        Fails closed.  A resolver that returns nothing is a broken pointer, and
        quietly substituting the stock drafter would silently restart the chain
        -- which is indistinguishable, in the artifacts, from a chain that was
        never compounding at all.
        """
        if self._warm_start_resolver is None:
            return self.config.from_pretrained
        resolved = self._warm_start_resolver()
        if not isinstance(resolved, str) or not resolved:
            raise Eagle3Error(
                "warm-start resolver named no checkpoint; refusing to guess a "
                "base for EAGLE-3 training"
            )
        return resolved

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
        """Extract verifier hidden states.

        Unconditionally, every cycle, over the whole leased window.  That is
        the dominant non-benchmark cost of a cycle -- 205.8s on job 369161 and
        303.8s on job 369162 -- and it is *not* incremental: with
        ``training_window_records=512`` over 513 buffered records, 511 of the
        512 rows were re-extracted from scratch.  A content-hash cache over
        those rows was assessed and deliberately not built, for four reasons in
        increasing order of severity.

        The hit rate would be ~0 at the only granularity that is addressable.
        :meth:`~speedlm.training.split.HeldOutTraceSnapshotLeaser._select_window`
        recomputes its offset from ``count_records()`` on every lease, so one
        new trace shifts the whole window by one record and the snapshot digest
        changes completely.  A snapshot-keyed cache therefore misses every time
        despite ~511/512 rows being identical; only a *per-row* cache would hit.

        Per-row is not reachable from here.  Extraction is a subprocess over a
        whole rows file that emits packed ``hs_*.safetensors`` shards, with no
        per-row addressing and shard boundaries that depend on the row set.
        Serving a partially-hit window would mean splicing third-party shards
        by hand -- which is exactly the code whose bugs produce a head trained
        against the wrong hidden states.

        Two of the four required invalidation keys are not observable.
        ``verifier_revision`` silently degrades to unpinned when the local HF
        cache cannot satisfy the pin (see
        :mod:`speedlm.training.backends.eagle3`), so the configured revision is
        not the identity of the weights that ran; and there is no ``dtype``
        knob anywhere in this pipeline -- precision is inherited from the
        verifier snapshot's own config and never read back -- so a dtype change
        cannot be detected at all.  A key that cannot see two of the things it
        must invalidate on is not a cache, it is a silent correctness hazard,
        and a stale entry here trains the head on another model's activations.

        And the saving is smaller than it looks.  Extraction stands up its own
        ``--enforce-eager`` vLLM server, loads the verifier and tears it down;
        engine lifecycle, not per-row forward passes, is most of that 200-300s,
        and no row cache touches it.  The addressable win in this stage is
        removing an engine start, not avoiding re-extraction.
        """
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
        from_pretrained = self._warm_start()
        # Recorded before the run, not after it: ``describe`` must be able to
        # say what a *failed* cycle attempted to build on, and a value written
        # only on success cannot.
        self._resolved_warm_start = from_pretrained
        started = self._clock()
        result = self._trainer.train(
            hidden_states,
            work_dir / "speculators-training",
            from_pretrained=from_pretrained,
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
    """Return bytes occupied by regular files beneath *path* without following links.

    Entries that vanish between enumeration and ``stat`` are skipped rather
    than raised.  This walk runs as the abort check of every EAGLE-3 stage, so
    it re-walks the scratch tree roughly ten times a second *while a
    subprocess is writing into it* -- and hidden-state extraction in
    particular churns that tree hard: the server writes each shard as
    ``cmpl-<request id>-<n>-<hash>.safetensors`` and the client immediately
    renames it to ``hs_<index>.safetensors``, so hundreds of paths appear and
    disappear under the walk.  A path enumerated a moment before the rename is
    simply gone when it is stat'd.

    Letting that race escape turned an ordinary interleaving into a failed
    cycle whose ``FileNotFoundError`` named a transient shard, which read like
    a missing output rather than the measurement artefact it was.  A vanished
    file occupies no bytes; skipping it is exact, not a weakened check -- a
    file that is still there is still counted, and the quota still trips.
    """
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_symlink():
                total += entry.lstat().st_size
            elif entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
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
