"""Concrete subprocess-driven EAGLE-3 backend for Speculators."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speedlm.training.backends.speculators_runner import (
    ProcessResult,
    ProcessRunner,
    RunningProcess,
    SubprocessRunner,
)
from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy
from speedlm.tuner.eagle3 import (
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

logger = logging.getLogger(__name__)

#: Per-stream cap on the persisted training log.  Two mebibytes is far more
#: than a Speculators run emits normally, and small enough that a pathological
#: run cannot fill the scratch quota with its own diagnostics.
MAX_TRAINING_LOG_BYTES = 2 * 1024 * 1024

DEFAULT_SPECULATORS_REPO = Path(
    os.environ.get("SPEEDLM_SPECULATORS_REPO", "speculators")
)
DEFAULT_SPECULATORS_PYTHON = Path(
    os.environ.get("SPEEDLM_TRAINING_PYTHON", sys.executable)
)
_SPECULATORS_DATA_NAME = "speculators-conversations.jsonl"
_SPECULATORS_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}
_ZERO_MASK = re.compile(
    # A row the validator rejects for carrying no usable sequence supervises
    # nothing, exactly like an all-zero mask, so route it to the named error
    # with row identity instead of losing it in a generic raise.
    r"(?:all[- ]zero|no trainable|nonzero loss|loss[- ]mask tokens"
    r"|no input_ids|empty input_ids)",
    re.IGNORECASE,
)
_ROW_ID = re.compile(
    r"(?:row(?:_id)?|index)\s*(?:=|:|is)?\s*['\"]?([A-Za-z0-9_.:/-]+)",
    re.IGNORECASE,
)
#: Resolve a repo to an on-disk snapshot without letting the pin tighten what
#: counts as resolvable.  ``snapshot_download`` only runs
#: ``_raise_if_incomplete_snapshot`` when an explicit revision is given, so a
#: pinned revision demands a byte-complete snapshot where the unpinned call
#: was happy with whatever the cache held.  Deployments here pull the cache
#: minimally on purpose -- shards, tokenizer and configs, not licences and
#: READMEs -- so the pin alone turned a working offline cache into
#: IncompleteSnapshotError (job 368710).
#:
#: ``allow_patterns`` narrows completeness to the files training actually
#: reads.  It is a real narrowing but not a guarantee: huggingface_hub matches
#: patterns with ``fnmatch``, whose ``*`` crosses ``/``, so ``*.json`` also
#: claims ``original/config.json`` in an auxiliary directory a minimal cache
#: never pulled (job 368719).
#:
#: The guarantee is the fallback, and it is *not* another download.  The
#: unpinned path never called ``snapshot_download`` at all -- it handed the
#: bare repo id downstream and let the loader resolve it -- so an unpinned
#: call cannot be reproduced by passing ``revision=None``, which still runs
#: the completeness check against ``main``.  When the cache cannot satisfy the
#: pin, resolution reports that and the caller falls back to exactly what an
#: unpinned cycle did.
_UNRESOLVED_SNAPSHOT = "SPEEDLM_UNRESOLVED"
_RESOLVE_MODEL = f"""
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError
repo = sys.argv[1]
revision = sys.argv[2]
patterns = sys.argv[3:] or None
try:
    path = snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns)
except IncompleteSnapshotError as error:
    print(f"SPEEDLM_INCOMPLETE_SNAPSHOT={{error}}", file=sys.stderr)
    path = "{_UNRESOLVED_SNAPSHOT}"
print(path)
""".strip()
_AUDIT_MASKS = """
from datasets import load_from_disk
import sys
for index, row in enumerate(load_from_disk(sys.argv[1])):
    if not any(bool(value) for value in row["loss_mask"]):
        print(f"SPEEDLM_ZERO_MASK_ROW={row.get('id', index)}")
        raise SystemExit(3)
""".strip()
#: Validate a materialized draft against the training contract.
#:
#: Both comparisons here are identity questions, not string questions, and
#: writing them as ``!=`` failed a genuinely correct cross-model run (job
#: 368962, Qwen3-8B).
#:
#: *Verifier.*  A Speculators config records its verifier as the repo id it was
#: published under ("Qwen/Qwen3-8B"), while the caller passes the resolved
#: snapshot directory the cache handed back.  Those name the same model, so the
#: comparison is on canonical form: a snapshot path carries the repo id in its
#: ``models--<org>--<name>`` cache segment, which maps back by ``--`` -> ``/``.
#: A genuinely different model still fails, because the two canonical repo ids
#: differ.
#:
#: *Layer ids.*  ``eagle_aux_hidden_state_layer_ids`` is ``null`` in both
#: drafters this deployment warm-starts from -- absent entirely in
#: RedHatAI/Qwen3-8B-speculator.eagle3, present-but-null in
#: RedHatAI/gpt-oss-20b-speculator.eagle3 -- and neither carries the key under
#: ``speculators_config``.  Null means the drafter did not pin layers, which
#: contradicts nothing, so it passes; a *present* list that disagrees with the
#: contract is a real conflict and still fails.  The comparison is on lists of
#: ints so the config's JSON ``[2, 18, 33]`` matches the contract's tuple.  The
#: fallback lookup under ``speculators_config`` costs nothing and only ever
#: turns a silent pass into a loud failure, so it guards configs that nest the
#: key where these two do not.
_VALIDATE_DRAFT = """
import json
import sys
from pathlib import Path
from safetensors import safe_open


def canonical_model(value):
    # Repo id for a HF snapshot path, else the identifier as given.
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    for part in reversed(Path(text).parts):
        if part.startswith("models--"):
            text = part[len("models--"):].replace("--", "/")
            break
    return text.lower()


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
if canonical_model(actual_verifier) != canonical_model(verifier):
    raise SystemExit(f"draft verifier mismatch: {actual_verifier!r} != {verifier!r}")
actual_layers = config.get("eagle_aux_hidden_state_layer_ids")
if actual_layers is None:
    actual_layers = speculators.get("eagle_aux_hidden_state_layer_ids")
if actual_layers is not None and layers:
    if [int(value) for value in actual_layers] != layers:
        raise SystemExit(
            "draft target layer ids do not match the training contract: "
            f"{actual_layers!r} != {layers!r}"
        )
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


class EmptySpeculatorsDatasetError(Eagle3Error):
    """No captured record survived conversion into the Speculators contract."""

    def __init__(self, source: Path) -> None:
        self.source = source
        super().__init__(
            f"no captured trace in {source} converted to a Speculators conversation"
        )


@dataclass(frozen=True, slots=True)
class SpeculatorsPipelineConfig:
    """Configurable reproduction of the verified Speculators pipeline."""

    prepared_validator_script: Path
    verifier_model: str
    warm_start_model: str
    row_count: int | None = None
    speculators_repo: Path = DEFAULT_SPECULATORS_REPO
    training_python: Path = DEFAULT_SPECULATORS_PYTHON
    vllm_python: Path | None = None
    verifier_revision: str | None = None
    warm_start_revision: str | None = None
    target_layer_ids: tuple[int, ...] = ()
    #: Sequence length for training. A value of 16384 collapsed a 512-record
    #: corpus into 1 batch; 4096 yielded 44 steps, making this a key lever
    #: for sampler throughput and gradient frequency.
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
    #: Files a pinned snapshot must contain to count as resolved.  Weights,
    #: the shard index, configs, the tokenizer and the chat template are what
    #: extraction and training open; licences, model cards and alternative
    #: runtime formats are not, and requiring them only breaks minimal caches.
    model_resolve_allow_patterns: tuple[str, ...] = (
        "*.json",
        "*.safetensors",
        "*.jinja",
    )
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
        if self.target_layer_ids and (
            any(
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
            raise ValueError(
                "scratch_quota_bytes must be in 1..20 GiB "
                "(field: tuning.scratch_quota_bytes)"
            )
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
                *self.config.model_resolve_allow_patterns,
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
        resolved = paths[-1]
        if resolved == _UNRESOLVED_SNAPSHOT:
            # The pin could not be satisfied from the cache.  Degrade to the
            # pre-pin behaviour -- hand the bare repo id downstream and let the
            # loader resolve it, which is what a cycle did before revisions
            # were pinned at all.  A pin is provenance; it must not be able to
            # stop a cycle the unpinned path would have run.
            logger.warning(
                "%s could not satisfy revision %s from the cache; continuing "
                "unpinned with %r",
                stage,
                revision,
                model,
            )
            return model
        return resolved


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
        started = time.monotonic()
        scratch = destination.parent
        data = scratch / _SPECULATORS_DATA_NAME
        guard = _guard(
            scratch, self.config.scratch_quota_bytes, should_abort, (destination, data)
        )
        try:
            verifier = self.resolver.verifier(guard, scratch)
            _remove(data)
            observed_row_count = _render_speculators_dataset(
                snapshot,
                data,
                guard=guard,
                started=started,
                timeout=timeout_seconds,
            )
            row_count = self.config.row_count or observed_row_count
            prepare = self.runner.run(
                [
                    str(self.config.training_python),
                    str(self.config.speculators_repo / "scripts" / "prepare_data.py"),
                    "--model",
                    verifier,
                    "--data",
                    str(data),
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
            _remove(data)
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
        del verifier_model, verifier_revision
        layers = tuple(target_layer_ids or self.config.target_layer_ids)
        seq_len = sequence_length or self.config.sequence_length
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
                    # launch_vllm.py appends the verifier's final layer to this
                    # list; training slices it back off as the regression target
                    # (hidden_states[:, :-1] are the aux layers), so the server
                    # must emit len(layers) + 1 layers, not len(layers).
                    "--",
                    "--port",
                    str(self.config.port),
                    "--max-model-len",
                    str(seq_len),
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
            _check_hidden_state_layers(destination, len(layers), generated)
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
            # Persisted before the status check so a *failed* training run
            # leaves the same evidence a successful one does.
            persist_training_output(scratch, result)
            _success("Speculators training", result)
            checkpoint = destination / "checkpoint_best"
            if not checkpoint.is_dir():
                raise TrainingError(
                    f"checkpoint_best is missing: {checkpoint}",
                    stderr=result.stderr,
                )
            val_loss = _parse_val_loss(checkpoint)
            return TrainingResult(checkpoint, result.returncode, result.stderr, val_loss=val_loss)
        except BaseException:
            _remove(destination)
            _remove(scratch / "warm-start-pinned")
            raise


class SpeculatorsDraftMaterializer:
    """Copy inference files from checkpoint_best into an immutable directory."""

    _TRANSIENT = {
        "optimizer_state_dict.pt",
        "scheduler_state_dict.pt",
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


def _speculators_content(value: object) -> str:
    """Flatten a captured content value into the plain text the loader reads."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "".join(
            part["text"]
            for part in value
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _speculators_turn(turn: object) -> dict[str, Any] | None:
    """Convert one captured turn, or return None when the loader would drop it."""
    if not isinstance(turn, Mapping):
        return None
    raw_role = turn.get("from", turn.get("role"))
    role = _SPECULATORS_ROLES.get(raw_role) if isinstance(raw_role, str) else None
    if role is None:
        return None
    raw_content = turn.get("value")
    if raw_content is None:
        raw_content = turn.get("content")
    converted: dict[str, Any] = {"role": role, "content": _speculators_content(raw_content)}
    calls = turn.get("tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)) and calls:
        converted["tool_calls"] = list(calls)
    call_id = turn.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        converted["tool_call_id"] = call_id
    # The loader reads either key and re-emits both; carry whichever was captured.
    reasoning = turn.get("thinking") or turn.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        converted["thinking"] = reasoning
        converted["reasoning_content"] = reasoning
    return converted


def _speculators_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert one captured trace, or return None when it trains nothing."""
    turns_value = record.get("conversations")
    if not isinstance(turns_value, (list, tuple)):
        turns_value = record.get("messages")
    if not isinstance(turns_value, (list, tuple)):
        return None
    turns = [
        converted
        for converted in (_speculators_turn(turn) for turn in turns_value)
        if converted is not None
    ]
    if not any(turn["role"] == "assistant" for turn in turns):
        return None
    converted_record: dict[str, Any] = {"conversations": turns}
    row_id = record.get("id")
    if isinstance(row_id, str) and row_id:
        converted_record["id"] = row_id
    tools = record.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        converted_record["tools"] = list(tools)
    return converted_record


def _render_speculators_dataset(
    snapshot: TraceSnapshot,
    destination: Path,
    *,
    guard: AbortCheck,
    started: float,
    timeout: float,
) -> int:
    """Rewrite a leased trace snapshot into the Speculators loader contract.

    SpeedLM captures a top-level ``messages`` key while the Speculators loader
    reads a top-level ``conversations`` key, so handing the snapshot over verbatim
    builds an empty dataset without reporting an error. Records that carry no
    trainable assistant turn are dropped, and an entirely empty result fails loudly
    rather than reaching training as a silently empty dataset.
    """
    written = 0
    try:
        with (
            snapshot.path.open("r", encoding="utf-8") as source,
            destination.open("x", encoding="utf-8") as output,
        ):
            for number, line in enumerate(source, start=1):
                _deadline(started, timeout, "Speculators row rendering")
                _abort(guard, "Speculators row rendering")
                if not line.strip():
                    continue
                location = f"{snapshot.path} line {number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Eagle3Error(f"{location} is not valid JSON") from error
                if not isinstance(record, Mapping):
                    raise Eagle3Error(f"{location} is not a JSON object")
                converted = _speculators_record(record)
                if converted is None:
                    continue
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1
    except OSError as error:
        raise Eagle3Error(
            f"cannot render Speculators rows from {snapshot.path}: {error}"
        ) from error
    if not written:
        raise EmptySpeculatorsDatasetError(snapshot.path)
    return written


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


def persist_training_output(
    run_dir: Path,
    result: ProcessResult,
    *,
    max_bytes: int = MAX_TRAINING_LOG_BYTES,
) -> Path | None:
    """Write the training subprocess's streams under ``<run_dir>/training-logs``.

    Training is the single longest stage of a cycle and its output was only
    ever surfaced by attaching stderr to an exception -- so a *successful*
    cycle discarded it entirely and there was no way to see how the ~180-210s
    divided between engine/model startup and actual gradient steps.  Every
    other stage leaves evidence in the run directory; this one now does too.

    Each stream is capped at *max_bytes* with the head and tail kept and the
    middle elided, because the two things worth reading in a training log are
    the startup banner and whatever happened at the end, and because a long
    run must not be able to fill the disk through its own logs.

    Persisting is best effort and returns ``None`` on failure: losing
    diagnostics must never be what fails a cycle that otherwise succeeded.
    """
    directory = run_dir / "training-logs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
            (directory / f"{name}.log").write_text(
                _bounded(stream or "", max_bytes), encoding="utf-8"
            )
    except OSError as exc:
        logger.warning("could not persist training output to %s: %s", directory, exc)
        return None
    return directory


def _bounded(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace")
    half = max_bytes // 2
    elided = len(raw) - 2 * half
    head = raw[:half].decode("utf-8", errors="replace")
    tail = raw[-half:].decode("utf-8", errors="replace")
    return f"{head}\n...[{elided} bytes elided]...\n{tail}"


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


def _check_hidden_state_layers(
    destination: Path,
    layer_count: int,
    generated: ProcessResult,
) -> None:
    """Fail at EXTRACTING when the emitted layer count breaks the contract.

    launch_vllm.py appends the verifier's final layer to --target-layer-ids,
    so each hs_*.safetensors must carry len(target_layer_ids) + 1 layers.
    Training slices the final layer back off as the regression target; a
    mismatch otherwise surfaces ~100s later as an opaque Dynamo broadcast
    error against a flattened hidden size.
    """
    expected = layer_count + 1
    shards = sorted(destination.glob("hs_*.safetensors"))
    if not shards:
        raise TrainingError(
            f"hidden-state extraction emitted no shards: {destination}",
            stderr=generated.stderr,
        )
    for shard in shards:
        shape = _safetensors_shape(shard, "hidden_states", generated)
        if len(shape) != 3 or shape[1] != expected:
            raise TrainingError(
                "hidden-state layer count breaks the EAGLE-3 contract: "
                f"{shard.name} has shape {shape}, expected "
                f"[sequence_length, {expected}, hidden_size] for "
                f"{layer_count} aux layers plus the verifier's final layer",
                stderr=generated.stderr,
            )


def _safetensors_shape(
    path: Path,
    tensor: str,
    generated: ProcessResult,
) -> list[int]:
    """Read one tensor's shape from a safetensors header, without torch."""
    try:
        with path.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(
            f"cannot read hidden-state header: {path}: {exc!r}",
            stderr=generated.stderr,
        ) from exc
    entry = header.get(tensor) if isinstance(header, dict) else None
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if not isinstance(shape, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in shape
    ):
        raise TrainingError(
            f"hidden-state shard has no usable {tensor} shape: {path}",
            stderr=generated.stderr,
        )
    return shape


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


def _parse_val_loss(checkpoint_best: Path) -> float | None:
    """Read ``loss_epoch`` from the checkpoint's ``val_metrics.json``.

    The Speculators trainer writes this file after each epoch; with
    ``--save-best`` the ``checkpoint_best`` symlink points at the best
    epoch, so this is the best validation loss.

    Returns ``None`` if the file is missing or malformed — the caller
    must treat that as "unavailable" and NOT fail the cycle.
    """
    path = checkpoint_best / "val_metrics.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    loss = data.get("loss_epoch")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool):
        return None
    return float(loss)


def _cleanup_transients(work_dir: Path) -> None:
    for name in (
        "trace-snapshot",
        _SPECULATORS_DATA_NAME,
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
    "DEFAULT_SPECULATORS_PYTHON",
    "DEFAULT_SPECULATORS_REPO",
    "MAX_SCRATCH_BYTES",
    "DraftMaterializer",
    "DraftValidator",
    "Eagle3Adapter",
    "Eagle3Backend",
    "Eagle3Config",
    "Eagle3Error",
    "Eagle3Timeouts",
    "EmptySpeculatorsDatasetError",
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
