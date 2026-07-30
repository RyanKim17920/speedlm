"""Concrete subprocess-driven EAGLE-3 backend for Speculators."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from speedlm.training.backends.speculators_runner import (
    ProcessResult,
    ProcessRunner,
    RunningProcess,
    SubprocessRunner,
)
from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy
from speedlm.tuner.eagle3 import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_VERIFIER_MODEL,
    MAX_SCRATCH_BYTES,
    AbortCheck,
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
from speedlm.tuner.idle import TuningPreempted

DEFAULT_SPECULATORS_REPO = Path("/admin/home/ryan.kim/speedlm/.preflight/speculators")
DEFAULT_SPECULATORS_PYTHON = Path(
    "/admin/home/ryan.kim/speedlm/.preflight/venvs/speculators/bin/python"
)
_ZERO_MASK = re.compile(
    r"(?:all[- ]zero|no trainable|nonzero loss|loss[- ]mask tokens)",
    re.IGNORECASE,
)
_ROW_ID = re.compile(
    r"(?:row(?:_id)?|index)\s*(?:=|:|is)?\s*['\"]?([A-Za-z0-9_.:/-]+)",
    re.IGNORECASE,
)
_RESOLVE_MODEL = (
    "from huggingface_hub import snapshot_download;"
    "import sys;"
    "print(snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2]))"
)
_AUDIT_MASKS = """
from datasets import load_from_disk
import sys
for index, row in enumerate(load_from_disk(sys.argv[1])):
    if not any(bool(value) for value in row["loss_mask"]):
        print(f"SPEEDLM_ZERO_MASK_ROW={row.get('id', index)}")
        raise SystemExit(3)
""".strip()
_VALIDATE_DRAFT = """
import json
import sys
from pathlib import Path
from safetensors import safe_open
root = Path(sys.argv[1])
verifier = sys.argv[2]
layers = [int(value) for value in sys.argv[3:]]
config_path = root / "config.json"
if not config_path.is_file():
    raise SystemExit(f"missing draft config: {config_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("speculators_model_type") != "eagle3":
    raise SystemExit("materialized draft is not an EAGLE-3 Speculators model")
speculators = config.get("speculators_config", {})
if speculators.get("algorithm") != "eagle3":
    raise SystemExit("materialized draft has a non-eagle3 algorithm")
actual_verifier = speculators.get("verifier", {}).get("name_or_path")
if actual_verifier != verifier:
    raise SystemExit(f"draft verifier mismatch: {actual_verifier!r} != {verifier!r}")
if config.get("eagle_aux_hidden_state_layer_ids") != layers:
    raise SystemExit("draft target layer ids do not match the training contract")
weights = sorted(root.glob("*.safetensors"))
if not weights:
    raise SystemExit("materialized draft has no safetensors weights")
keys = set()
for path in weights:
    with safe_open(str(path), framework="pt") as handle:
        keys.update(handle.keys())
missing = {"d2t", "t2d"} - keys
if missing:
    raise SystemExit(f"materialized draft is missing vocab mappings: {sorted(missing)}")
""".strip()


@dataclass(frozen=True, slots=True)
class SpeculatorsPipelineConfig:
    """Configurable reproduction of the verified Speculators pipeline."""

    prepared_validator_script: Path
    row_count: int | None = None
    speculators_repo: Path = DEFAULT_SPECULATORS_REPO
    training_python: Path = DEFAULT_SPECULATORS_PYTHON
    vllm_python: Path | None = None
    verifier_model: str = DEFAULT_VERIFIER_MODEL
    verifier_revision: str | None = None
    warm_start_model: str = DEFAULT_DRAFT_MODEL
    warm_start_revision: str | None = None
    target_layer_ids: tuple[int, ...] = (2, 12, 21)
    sequence_length: int = 16_384
    learning_rate: float = 1e-5
    epochs: int = 1
    seed: int = 0
    port: int = 8_131
    concurrency: int = 8
    mask_policy: MaskPolicy = MaskPolicy.ALL_ASSISTANT_TURNS
    max_num_seqs: int = 1
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.80
    scratch_quota_bytes: int = MAX_SCRATCH_BYTES
    model_resolve_timeout_seconds: float = 600.0
    server_shutdown_timeout_seconds: float = 10.0
    health_poll_interval_seconds: float = 0.25
    health_request_timeout_seconds: float = 1.0
    timeouts: Eagle3Timeouts = field(default_factory=Eagle3Timeouts)

    def __post_init__(self) -> None:
        for name, path in (
            ("prepared_validator_script", self.prepared_validator_script),
            ("speculators_repo", self.speculators_repo),
            ("training_python", self.training_python),
        ):
            if not isinstance(path, Path):
                raise ValueError(f"{name} must be a Path")
        if self.vllm_python is not None and not isinstance(self.vllm_python, Path):
            raise ValueError("vllm_python must be a Path or null")
        for name, model_value in (
            ("verifier_model", self.verifier_model),
            ("warm_start_model", self.warm_start_model),
        ):
            if not isinstance(model_value, str) or not model_value:
                raise ValueError(f"{name} must be non-empty")
        for name, revision_value in (
            ("verifier_revision", self.verifier_revision),
            ("warm_start_revision", self.warm_start_revision),
        ):
            if revision_value is not None and (
                not isinstance(revision_value, str) or not revision_value
            ):
                raise ValueError(f"{name} must be non-empty or null")
        for name, integer_value in (
            ("sequence_length", self.sequence_length),
            ("epochs", self.epochs),
            ("port", self.port),
            ("concurrency", self.concurrency),
            ("max_num_seqs", self.max_num_seqs),
        ):
            _positive_int(name, integer_value)
        if self.row_count is not None:
            _positive_int("row_count", self.row_count)
        if self.port > 65_535:
            raise ValueError("port must be at most 65535")
        if (
            not self.target_layer_ids
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in self.target_layer_ids
            )
            or len(set(self.target_layer_ids)) != len(self.target_layer_ids)
        ):
            raise ValueError("target_layer_ids must be unique non-negative integers")
        if not isinstance(self.mask_policy, MaskPolicy):
            raise ValueError("mask_policy must be a MaskPolicy")
        if self.mask_policy is not MaskPolicy.ALL_ASSISTANT_TURNS:
            raise ValueError(
                "the Speculators all_assistant pipeline requires "
                "MaskPolicy.ALL_ASSISTANT_TURNS"
            )
        if isinstance(self.learning_rate, bool) or not 0 < self.learning_rate <= 1e-5:
            raise ValueError("learning_rate must be positive and no greater than safe 1e-5")
        if (
            isinstance(self.gpu_memory_utilization, bool)
            or not isinstance(self.gpu_memory_utilization, (int, float))
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if (
            isinstance(self.scratch_quota_bytes, bool)
            or not isinstance(self.scratch_quota_bytes, int)
            or not 0 < self.scratch_quota_bytes <= MAX_SCRATCH_BYTES
        ):
            raise ValueError("scratch_quota_bytes must be in 1..5 GiB")
        for name, timeout_value in (
            ("model_resolve_timeout_seconds", self.model_resolve_timeout_seconds),
            ("server_shutdown_timeout_seconds", self.server_shutdown_timeout_seconds),
            ("health_poll_interval_seconds", self.health_poll_interval_seconds),
            ("health_request_timeout_seconds", self.health_request_timeout_seconds),
        ):
            if (
                isinstance(timeout_value, bool)
                or not isinstance(timeout_value, (int, float))
                or timeout_value <= 0
            ):
                raise ValueError(f"{name} must be positive")

    @property
    def effective_vllm_python(self) -> Path:
        return self.vllm_python or self.training_python


@dataclass(slots=True)
class _State:
    prepared: Path | None = None
    row_count: int | None = None
    verifier: str | None = None
    warm_start: str | None = None


class _Resolver:
    def __init__(
        self,
        config: SpeculatorsPipelineConfig,
        runner: ProcessRunner,
        state: _State,
    ) -> None:
        self.config = config
        self.runner = runner
        self.state = state

    def verifier(self, guard: AbortCheck, scratch: Path) -> str:
        if self.state.verifier is None:
            self.state.verifier = self._resolve(
                self.config.verifier_model,
                self.config.verifier_revision,
                "verifier model resolution",
                guard,
                scratch,
            )
        return self.state.verifier

    def warm_start(self, model: str, guard: AbortCheck, scratch: Path) -> str:
        if self.state.warm_start is None:
            revision = (
                self.config.warm_start_revision
                if model == self.config.warm_start_model
                else None
            )
            self.state.warm_start = self._resolve(
                model, revision, "warm-start model resolution", guard, scratch
            )
        return self.state.warm_start

    def _resolve(
        self,
        model: str,
        revision: str | None,
        stage: str,
        guard: AbortCheck,
        scratch: Path,
    ) -> str:
        if revision is None or Path(model).exists():
            return model
        result = self.runner.run(
            [
                str(self.config.training_python),
                "-c",
                _RESOLVE_MODEL,
                model,
                revision,
            ],
            cwd=self.config.speculators_repo,
            env=_environment(self.config),
            timeout_seconds=self.config.model_resolve_timeout_seconds,
            should_abort=_guard(
                scratch, self.config.scratch_quota_bytes, guard, ()
            ),
        )
        _success(stage, result)
        paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not paths:
            raise TrainingError(f"{stage} returned no snapshot path", stderr=result.stderr)
        return paths[-1]


class FilesystemTraceSnapshotLeaser:
    """Create an immutable, content-hashed trace lease."""

    def __init__(self, source: Path, *, scratch_quota_bytes: int) -> None:
        self.source = source
        self.scratch_quota_bytes = scratch_quota_bytes

    def lease_snapshot(
        self,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> TraceSnapshot:
        if not self.source.is_file():
            raise Eagle3Error(f"trace source is not a regular file: {self.source}")
        started = time.monotonic()
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / self.source.name
        digest = hashlib.sha256()
        guard = _guard(
            destination.parent,
            self.scratch_quota_bytes,
            should_abort,
            (destination,),
        )
        try:
            with self.source.open("rb") as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    _deadline(started, timeout_seconds, "trace snapshot lease")
                    _abort(guard, "trace snapshot lease")
                    output.write(chunk)
                    digest.update(chunk)
                    _abort(guard, "trace snapshot lease")
            _deadline(started, timeout_seconds, "trace snapshot lease")
            _abort(guard, "trace snapshot lease")
            target.chmod(0o444)
            return TraceSnapshot(target, digest.hexdigest())
        except BaseException:
            _remove(destination)
            raise


class SpeculatorsTrainingRowRenderer:
    """Run prepare_data.py and check_prepared_dataset.py."""

    def __init__(
        self,
        config: SpeculatorsPipelineConfig,
        runner: ProcessRunner,
        resolver: _Resolver,
        state: _State,
    ) -> None:
        self.config = config
        self.runner = runner
        self.resolver = resolver
        self.state = state

    def render_rows(
        self,
        snapshot: TraceSnapshot,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
        mask_policy: MaskPolicy = MaskPolicy.ALL_ASSISTANT_TURNS,
        sequence_length: int | None = None,
    ) -> Path:
        if mask_policy is not self.config.mask_policy:
            raise Eagle3Error("renderer mask policy does not match configured policy")
        seq_len = sequence_length or self.config.sequence_length
        scratch = destination.parent
        guard = _guard(
            scratch, self.config.scratch_quota_bytes, should_abort, (destination,)
        )
        try:
            verifier = self.resolver.verifier(guard, scratch)
            observed_row_count = _jsonl_record_count(snapshot.path)
            if observed_row_count < 1:
                raise Eagle3Error("training trace snapshot contains no records")
            row_count = self.config.row_count or observed_row_count
            prepare = self.runner.run(
                [
                    str(self.config.training_python),
                    str(self.config.speculators_repo / "scripts" / "prepare_data.py"),
                    "--model",
                    verifier,
                    "--data",
                    str(snapshot.path),
                    "--output",
                    str(destination),
                    "--seq-length",
                    str(seq_len),
                    "--seed",
                    str(self.config.seed),
                    "--num-preprocessing-workers",
                    "0",
                    "--overwrite",
                ],
                cwd=self.config.speculators_repo,
                env=_environment(self.config),
                timeout_seconds=timeout_seconds,
                should_abort=guard,
            )
            _success("Speculators prepare", prepare)
            checked = self.runner.run(
                [
                    str(self.config.training_python),
                    str(self.config.prepared_validator_script),
                    str(destination),
                    str(row_count),
                    "--require-nonzero-loss-mask",
                    "--max-seq-len",
                    str(seq_len),
                ],
                cwd=self.config.speculators_repo,
                env=_environment(self.config),
                timeout_seconds=timeout_seconds,
                should_abort=guard,
            )
            if checked.returncode and _ZERO_MASK.search(checked.stderr):
                row_id = self._zero_row(destination, timeout_seconds, guard, checked)
                raise FinalAssistantMaskError(row_id, mask_policy)
            _success("prepared dataset validation", checked)
            self.state.prepared = destination
            self.state.row_count = row_count
            return destination
        except BaseException:
            _remove(destination)
            raise

    def _zero_row(
        self,
        destination: Path,
        timeout_seconds: float,
        guard: AbortCheck,
        checked: ProcessResult,
    ) -> str:
        match = _ROW_ID.search(f"{checked.stdout}\n{checked.stderr}")
        if match is not None and match.group(1).lower() not in {"a", "with"}:
            return match.group(1)
        audit = self.runner.run(
            [str(self.config.training_python), "-c", _AUDIT_MASKS, str(destination)],
            cwd=self.config.speculators_repo,
            env=_environment(self.config),
            timeout_seconds=timeout_seconds,
            should_abort=guard,
        )
        match = re.search(
            r"SPEEDLM_ZERO_MASK_ROW=([^\r\n]+)",
            f"{audit.stdout}\n{audit.stderr}",
        )
        if match is None:
            raise TrainingError(
                "could not identify the all-zero loss-mask row",
                stderr=checked.stderr,
            )
        return match.group(1).strip()


class SpeculatorsHiddenStateExtractor:
    """Manage vLLM and offline hidden-state generation."""

    def __init__(
        self,
        config: SpeculatorsPipelineConfig,
        runner: ProcessRunner,
        resolver: _Resolver,
        *,
        health_check: Callable[[str, float], bool],
        sleeper: Callable[[float], None],
        clock: Callable[[], float],
        state: _State | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.resolver = resolver
        self.health_check = health_check
        self.sleeper = sleeper
        self.clock = clock
        self.state = state

    def extract_hidden_states(
        self,
        rows_path: Path,
        destination: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: AbortCheck,
        verifier_revision: str | None = None,
        target_layer_ids: Sequence[int] | None = None,
        sequence_length: int | None = None,
    ) -> Path:
        del verifier_model, verifier_revision, sequence_length
        layers = tuple(target_layer_ids or self.config.target_layer_ids)
        row_count = (
            self.state.row_count
            if self.state is not None and self.state.row_count is not None
            else self.config.row_count
        )
        if row_count is None:
            raise Eagle3Error("training row count was not recorded before extraction")
        scratch = destination.parent
        guard = _guard(
            scratch, self.config.scratch_quota_bytes, should_abort, (destination,)
        )
        started = self.clock()
        server: RunningProcess | None = None
        try:
            verifier = self.resolver.verifier(guard, scratch)
            server = self.runner.start(
                [
                    str(self.config.effective_vllm_python),
                    str(self.config.speculators_repo / "scripts" / "launch_vllm.py"),
                    verifier,
                    "--hidden-states-path",
                    str(destination),
                    "--target-layer-ids",
                    *(str(layer) for layer in layers),
                    "--",
                    "--port",
                    str(self.config.port),
                    "--max-num-seqs",
                    str(self.config.max_num_seqs),
                    *(["--enforce-eager"] if self.config.enforce_eager else []),
                    "--gpu-memory-utilization",
                    f"{self.config.gpu_memory_utilization:.2f}",
                ],
                cwd=self.config.speculators_repo,
                env=_environment(self.config),
                timeout_seconds=timeout_seconds,
            )
            health_url = f"http://127.0.0.1:{self.config.port}/health"
            while not self.health_check(
                health_url, self.config.health_request_timeout_seconds
            ):
                returncode = self.runner.check_running(server, should_abort=guard)
                if returncode is not None:
                    stopped = self.runner.terminate(
                        server,
                        grace_seconds=self.config.server_shutdown_timeout_seconds,
                    )
                    raise TrainingError(
                        f"vLLM hidden-state server exited with status {returncode}",
                        stderr=stopped.stderr,
                    )
                _deadline(started, timeout_seconds, "vLLM health check")
                self.sleeper(self.config.health_poll_interval_seconds)
            self.runner.check_running(server, should_abort=guard)
            remaining = timeout_seconds - (self.clock() - started)
            if remaining <= 0:
                raise StageTimeoutError("hidden-state extraction exhausted its timeout")
            generated = self.runner.run(
                [
                    str(self.config.training_python),
                    str(
                        self.config.speculators_repo
                        / "scripts"
                        / "data_generation_offline.py"
                    ),
                    "--endpoint",
                    f"http://127.0.0.1:{self.config.port}/v1",
                    "--preprocessed-data",
                    str(rows_path),
                    "--output",
                    str(destination),
                    "--max-samples",
                    str(row_count),
                    "--concurrency",
                    str(self.config.concurrency),
                    "--validate-outputs",
                    "--fail-on-error",
                ],
                cwd=self.config.speculators_repo,
                env=_environment(self.config),
                timeout_seconds=remaining,
                should_abort=guard,
            )
            _success("offline hidden-state generation", generated)
            return destination
        except BaseException:
            _remove(destination)
            raise
        finally:
            if server is not None:
                self.runner.terminate(
                    server,
                    grace_seconds=self.config.server_shutdown_timeout_seconds,
                )


class SpeculatorsTrainingProcess:
    """Run train.py with the mandatory warm start."""

    def __init__(
        self,
        config: SpeculatorsPipelineConfig,
        runner: ProcessRunner,
        resolver: _Resolver,
        state: _State,
    ) -> None:
        self.config = config
        self.runner = runner
        self.resolver = resolver
        self.state = state

    def train(
        self,
        hidden_states_path: Path,
        destination: Path,
        *,
        from_pretrained: str,
        training_params: Mapping[str, object],
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> TrainingResult:
        if not from_pretrained:
            raise Eagle3Error("refusing to train EAGLE-3 from scratch")
        if self.state.prepared is None:
            raise Eagle3Error("prepared dataset was not recorded before training")
        raw_layers = training_params.get("target_layer_ids")
        layer_values = (
            raw_layers
            if isinstance(raw_layers, Sequence)
            and not isinstance(raw_layers, (str, bytes))
            else self.config.target_layer_ids
        )
        layers = tuple(_integer("target_layer_ids", value) for value in layer_values)
        seq_len = _integer(
            "sequence_length",
            training_params.get("sequence_length", self.config.sequence_length),
        )
        scratch = destination.parent
        guard = _guard(
            scratch, self.config.scratch_quota_bytes, should_abort, (destination,)
        )
        try:
            verifier = self.resolver.verifier(guard, scratch)
            resolved_warm_start = self.resolver.warm_start(
                from_pretrained, guard, scratch
            )
            warm_start = _pin_warm_start(
                resolved_warm_start,
                verifier,
                scratch / "warm-start-pinned",
                guard,
                timeout_seconds,
            )
            result = self.runner.run(
                [
                    str(self.config.training_python),
                    str(self.config.speculators_repo / "scripts" / "train.py"),
                    "--verifier-name-or-path",
                    verifier,
                    "--from-pretrained",
                    warm_start,
                    "--data-path",
                    str(self.state.prepared),
                    "--hidden-states-path",
                    str(hidden_states_path),
                    "--on-missing",
                    "raise",
                    "--save-path",
                    str(destination),
                    "--speculator-type",
                    "eagle3",
                    "--target-layer-ids",
                    *(str(layer) for layer in layers),
                    "--seed",
                    str(self.config.seed),
                    "--epochs",
                    str(self.config.epochs),
                    "--lr",
                    str(self.config.learning_rate),
                    "--total-seq-len",
                    str(seq_len),
                    "--no-resume-from-checkpoint",
                    "--save-best",
                ],
                cwd=self.config.speculators_repo,
                env=_training_environment(self.config),
                timeout_seconds=timeout_seconds,
                should_abort=guard,
            )
            _success("Speculators training", result)
            checkpoint = destination / "checkpoint_best"
            if not checkpoint.is_dir():
                raise TrainingError(
                    f"checkpoint_best is missing: {checkpoint}",
                    stderr=result.stderr,
                )
            return TrainingResult(checkpoint, result.returncode, result.stderr)
        except BaseException:
            _remove(destination)
            _remove(scratch / "warm-start-pinned")
            raise


class SpeculatorsDraftMaterializer:
    """Copy inference files from checkpoint_best into an immutable directory."""

    _TRANSIENT = {
        "optimizer_state_dict.pt",
        "scheduler_state_dict.pt",
        "val_metrics.json",
    }

    def __init__(self, *, scratch_quota_bytes: int) -> None:
        self.scratch_quota_bytes = scratch_quota_bytes

    def materialize(
        self,
        checkpoint_best: Path,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> Path:
        source = checkpoint_best.resolve(strict=True)
        if not source.is_dir():
            raise Eagle3Error(f"checkpoint_best is not a directory: {checkpoint_best}")
        if destination.exists():
            raise Eagle3Error(f"refusing to overwrite draft directory: {destination}")
        destination.mkdir(parents=True)
        started = time.monotonic()
        guard = _guard(
            destination.parent,
            self.scratch_quota_bytes,
            should_abort,
            (destination,),
        )
        try:
            for path in sorted(source.rglob("*")):
                _deadline(started, timeout_seconds, "draft materialization")
                _abort(guard, "draft materialization")
                if path.name in self._TRANSIENT or path.name.startswith("."):
                    continue
                target = destination / path.relative_to(source)
                if path.is_dir():
                    target.mkdir(exist_ok=True)
                elif path.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _copy(path, target, guard, started, timeout_seconds)
            for path in sorted(destination.rglob("*"), reverse=True):
                path.chmod(0o555 if path.is_dir() else 0o444)
            destination.chmod(0o555)
            _abort(guard, "draft materialization")
            _cleanup_transients(destination.parent)
            return destination
        except BaseException:
            _writable(destination)
            _remove(destination)
            raise


class SpeculatorsDraftValidator:
    """Validate standalone EAGLE-3 config and safetensors in a subprocess."""

    def __init__(
        self,
        config: SpeculatorsPipelineConfig,
        runner: ProcessRunner,
        resolver: _Resolver,
    ) -> None:
        self.config = config
        self.runner = runner
        self.resolver = resolver

    def validate(
        self,
        draft_directory: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        del verifier_model
        scratch = draft_directory.parent
        guard = _guard(scratch, self.config.scratch_quota_bytes, should_abort, ())
        verifier = self.resolver.verifier(guard, scratch)
        result = self.runner.run(
            [
                str(self.config.training_python),
                "-c",
                _VALIDATE_DRAFT,
                str(draft_directory),
                verifier,
                *(str(layer) for layer in self.config.target_layer_ids),
            ],
            cwd=self.config.speculators_repo,
            env=_environment(self.config),
            timeout_seconds=timeout_seconds,
            should_abort=guard,
        )
        _success("standalone EAGLE-3 draft validation", result)


class Eagle3Backend(Eagle3Adapter):
    """Canonical adapter with a factory for all concrete effects."""

    @classmethod
    def from_speculators(
        cls,
        pipeline: SpeculatorsPipelineConfig,
        *,
        trace_source: Path | None = None,
        trace_leaser: TraceSnapshotLeaser | None = None,
        runner: ProcessRunner | None = None,
        health_check: Callable[[str, float], bool] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> Eagle3Backend:
        if (trace_source is None) == (trace_leaser is None):
            raise ValueError("provide exactly one of trace_source or trace_leaser")
        process_runner = runner or SubprocessRunner(clock=clock)
        state = _State()
        resolver = _Resolver(pipeline, process_runner, state)
        config = Eagle3Config(
            verifier_model=pipeline.verifier_model,
            verifier_revision=pipeline.verifier_revision,
            draft_model=pipeline.warm_start_model,
            draft_revision=pipeline.warm_start_revision,
            from_pretrained=pipeline.warm_start_model,
            target_layer_ids=pipeline.target_layer_ids,
            sequence_length=pipeline.sequence_length,
            mask_policy=pipeline.mask_policy,
            training_params={
                "learning_rate": pipeline.learning_rate,
                "epochs": pipeline.epochs,
                "seed": pipeline.seed,
            },
            timeouts=pipeline.timeouts,
            scratch_quota_bytes=pipeline.scratch_quota_bytes,
        )
        return cls(
            config,
            leaser=(
                trace_leaser
                if trace_leaser is not None
                else FilesystemTraceSnapshotLeaser(
                    trace_source,  # type: ignore[arg-type]
                    scratch_quota_bytes=pipeline.scratch_quota_bytes,
                )
            ),
            renderer=SpeculatorsTrainingRowRenderer(
                pipeline, process_runner, resolver, state
            ),
            extractor=SpeculatorsHiddenStateExtractor(
                pipeline,
                process_runner,
                resolver,
                health_check=health_check or _health,
                sleeper=sleeper,
                clock=clock,
                state=state,
            ),
            trainer=SpeculatorsTrainingProcess(
                pipeline, process_runner, resolver, state
            ),
            materializer=SpeculatorsDraftMaterializer(
                scratch_quota_bytes=pipeline.scratch_quota_bytes
            ),
            validator=SpeculatorsDraftValidator(pipeline, process_runner, resolver),
            clock=clock,
        )

    def _check(self, work_dir: Path, should_abort: AbortCheck) -> None:
        try:
            super()._check(work_dir, should_abort)
        except ScratchQuotaExceeded:
            _cleanup_transients(work_dir)
            raise


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _jsonl_record_count(path: Path) -> int:
    try:
        with path.open("rb") as source:
            return sum(1 for line in source if line.strip())
    except OSError as error:
        raise Eagle3Error(f"cannot count training records in {path}: {error}") from error


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise Eagle3Error(f"{name} must be integer-valued")
    try:
        return int(value)
    except ValueError as error:
        raise Eagle3Error(f"{name} must be integer-valued") from error


def _environment(config: SpeculatorsPipelineConfig) -> dict[str, str]:
    env = dict(os.environ)
    source = str(config.speculators_repo / "src")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not previous else f"{source}{os.pathsep}{previous}"
    return env


def _training_environment(config: SpeculatorsPipelineConfig) -> dict[str, str]:
    env = _environment(config)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _pin_warm_start(
    resolved_warm_start: str,
    verifier: str,
    destination: Path,
    guard: AbortCheck,
    timeout_seconds: float,
) -> str:
    """Patch verifier provenance without copying immutable model weights."""
    source = Path(resolved_warm_start)
    if not source.is_dir():
        return resolved_warm_start
    config_path = source / "config.json"
    if not config_path.is_file():
        raise TrainingError(
            f"warm-start checkpoint has no config.json: {source}",
            stderr="",
        )
    if destination.exists():
        raise Eagle3Error(f"refusing existing warm-start pin: {destination}")
    started = time.monotonic()
    destination.mkdir()
    try:
        for entry in sorted(source.iterdir()):
            _deadline(started, timeout_seconds, "warm-start pinning")
            _abort(guard, "warm-start pinning")
            target = destination / entry.name
            if entry.name == "config.json":
                value = json.loads(entry.read_text(encoding="utf-8"))
                value["speculators_config"]["verifier"]["name_or_path"] = verifier
                target.write_text(
                    json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                target.chmod(0o444)
            else:
                target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        _abort(guard, "warm-start pinning")
        return str(destination)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        _remove(destination)
        raise TrainingError(
            f"warm-start config is not a Speculators draft: {config_path}",
            stderr=str(error),
        ) from error
    except BaseException:
        _remove(destination)
        raise


def _success(stage: str, result: ProcessResult) -> None:
    if result.returncode:
        raise TrainingError(
            f"{stage} exited with status {result.returncode}",
            stderr=result.stderr,
        )


def _guard(
    scratch: Path,
    quota: int,
    should_abort: AbortCheck,
    cleanup: Sequence[Path],
) -> AbortCheck:
    def check() -> bool:
        used = scratch_usage(scratch)
        if used > quota:
            for path in cleanup:
                _remove(path)
            _cleanup_transients(scratch)
            raise ScratchQuotaExceeded(used, quota)
        return should_abort()

    return check


def _abort(guard: AbortCheck, stage: str) -> None:
    if guard():
        raise TuningPreempted(f"incoming request preempted {stage}")


def _deadline(started: float, timeout: float, stage: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > timeout:
        raise StageTimeoutError(
            f"{stage} exceeded {timeout:.3f}s timeout (elapsed {elapsed:.3f}s)"
        )


def _copy(
    source: Path,
    destination: Path,
    guard: AbortCheck,
    started: float,
    timeout: float,
) -> None:
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        while chunk := input_file.read(1024 * 1024):
            _deadline(started, timeout, "draft materialization")
            _abort(guard, "draft materialization")
            output_file.write(chunk)
            _abort(guard, "draft materialization")


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        _writable(path)
        shutil.rmtree(path)


def _writable(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob("*"):
        child.chmod(0o755 if child.is_dir() else 0o644)
    path.chmod(0o755)


def _cleanup_transients(work_dir: Path) -> None:
    for name in (
        "trace-snapshot",
        "training-rows",
        "hidden-states",
        "speculators-training",
        "warm-start-pinned",
    ):
        _remove(work_dir / name)


def _health(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


__all__ = [
    "DEFAULT_DRAFT_MODEL",
    "DEFAULT_SPECULATORS_PYTHON",
    "DEFAULT_SPECULATORS_REPO",
    "DEFAULT_VERIFIER_MODEL",
    "MAX_SCRATCH_BYTES",
    "DraftMaterializer",
    "DraftValidator",
    "Eagle3Adapter",
    "Eagle3Backend",
    "Eagle3Config",
    "Eagle3Error",
    "Eagle3Timeouts",
    "FilesystemTraceSnapshotLeaser",
    "FinalAssistantMaskError",
    "HiddenStateExtractor",
    "PreparedData",
    "ProcessResult",
    "ProcessRunner",
    "ScratchQuotaExceeded",
    "SpeculatorsDraftMaterializer",
    "SpeculatorsDraftValidator",
    "SpeculatorsHiddenStateExtractor",
    "SpeculatorsPipelineConfig",
    "SpeculatorsTrainer",
    "SpeculatorsTrainingProcess",
    "SpeculatorsTrainingRowRenderer",
    "StageTimeoutError",
    "SubprocessRunner",
    "TraceSnapshot",
    "TraceSnapshotLeaser",
    "TrainingError",
    "TrainingResult",
    "TrainingRowRenderer",
    "scratch_usage",
]
