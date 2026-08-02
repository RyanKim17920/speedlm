"""Concrete subprocess-driven EAGLE-3 backend for Speculators."""

from __future__ import annotations

import contextlib
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from speedlm.training.backends.speculators_runner import (
    ProcessResult,
    ProcessRunner,
    RunningProcess,
    SubprocessRunner,
    process_output,
)
from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy
from speedlm.tuner.eagle3 import (
    MAX_SCRATCH_BYTES,
    SCRATCH_HEADROOM_BYTES,
    SHARD_BYTES_PER_ROW,
    AbortCheck,
    BackendInfo,
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
    derive_scratch_quota_bytes,
    scratch_usage,
)
from speedlm.tuner.idle import TuningPreempted

logger = logging.getLogger(__name__)

#: Passes :func:`_remove_tree` makes at a directory whose contents reappear
#: under it.  With the writers already stopped, one retry is enough for a
#: process inside its SIGTERM grace period; three bounds the pathological case
#: without turning a genuinely undeletable tree into a hang.
_REMOVE_TREE_ATTEMPTS: Final = 3

#: Per-stream cap on the persisted training log.  Two mebibytes is far more
#: than a Speculators run emits normally, and small enough that a pathological
#: run cannot fill the scratch quota with its own diagnostics.
MAX_TRAINING_LOG_BYTES = 2 * 1024 * 1024

#: Directory beneath a cycle's scratch that holds per-stage diagnostics.
#:
#: Deliberately a *sibling* of every stage's output directory and never a
#: child of one.  The failure paths below destroy the stage's output, and a
#: log written inside that output is destroyed with it -- which is exactly how
#: the one artifact that would have explained a missing hidden-state shard was
#: deleted by the error path that was cleaning up after it.  It is also absent
#: from :data:`_TRANSIENT_NAMES`, so the scratch-quota sweep leaves it alone.
STAGE_LOG_DIR_NAME: Final = "stage-logs"

#: Output entries named individually in a failed stage's inventory.
#:
#: The inventory answers "what did this stage actually produce" -- a count, a
#: total size, and enough names to see an off-by-one or a naming mismatch --
#: without retaining gigabytes of unusable shards.  Sixty-four names is enough
#: to recognise a pattern at both ends of a sorted listing and small enough
#: that the record stays readable.
MAX_INVENTORY_ENTRIES: Final = 64

#: Bytes moved between two checkpoints while copying one materialized draft file.
#:
#: The unit here is not the read size, it is how often the copy stops to check
#: the scratch quota -- and that check is expensive: :func:`scratch_usage`
#: ``rglob``s and ``stat``s the entire scratch tree on every call.  At the
#: previous 1 MiB, with a check on *both* sides of every write, a ~2 GB draft
#: paid roughly four thousand full-tree walks, all of them on the cycle's
#: critical path with serving stopped (the materialize/validate/publish tail
#: measured ~153s).
#:
#: 16 MiB keeps every property the small chunk had -- bounded memory, and a
#: deadline/abort/quota checkpoint *inside* the copy so a large file is still
#: interruptible and still cannot silently blow the quota -- while cutting the
#: number of walks by 32x.  The trailing checkpoint after the loop is what
#: keeps the last write covered, so nothing is merely deferred to the caller.
DRAFT_COPY_CHUNK_BYTES: Final = 16 * 1024 * 1024

DEFAULT_SPECULATORS_REPO = Path(
    os.environ.get("SPEEDLM_SPECULATORS_REPO", "speculators")
)
DEFAULT_SPECULATORS_PYTHON = Path(
    os.environ.get("SPEEDLM_TRAINING_PYTHON", sys.executable)
)
_SPECULATORS_DATA_NAME = "speculators-conversations.jsonl"

#: Stage outputs a scratch-quota trip removes, and that a failure inventories.
_TRANSIENT_NAMES: Final = (
    "trace-snapshot",
    _SPECULATORS_DATA_NAME,
    "training-rows",
    "hidden-states",
    "speculators-training",
    "warm-start-pinned",
)
#: Stage names used for stage-logs subdirectories.  Named constants because the
#: same string identifies a stage on its success path and its failure path.
_EXTRACTION_STAGE: Final = "hidden-state-extraction"
_ROW_RENDER_STAGE: Final = "training-row-rendering"
_TRAINING_STAGE: Final = "training"
_QUOTA_STAGE: Final = "scratch-quota"
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
#: reads, but on its own it does not narrow far enough, and the claim that it
#: did was the bug.  huggingface_hub filters with ``fnmatch``, whose ``*``
#: crosses ``/``, so ``*.json`` also claims ``original/config.json`` and
#: ``*.safetensors`` claims the 13.7 GB ``original/model.safetensors`` --
#: auxiliary paths a deliberately minimal cache never pulled.  There is no
#: character class that repairs this (``[!/]*.json`` still matches, because the
#: trailing ``*`` crosses the separator all the same), so the anchoring is done
#: with ``ignore_patterns`` instead: ``*/*`` matches exactly the paths that
#: contain a separator, which leaves the top level and nothing else.  Verified
#: against the cached tree listing for ``openai/gpt-oss-20b``, where the
#: unanchored patterns expected three files the cache does not hold and the
#: anchored ones expect ten, all present (job 369373).
#:
#: The guarantee is still the fallback, and it is *not* another download.  The
#: unpinned path never called ``snapshot_download`` at all -- it handed the
#: bare repo id downstream and let the loader resolve it -- so an unpinned
#: call cannot be reproduced by passing ``revision=None``, which still runs
#: the completeness check against ``main``.  When the cache cannot satisfy the
#: pin, resolution reports that and the caller falls back to exactly what an
#: unpinned cycle did.
_UNRESOLVED_SNAPSHOT = "SPEEDLM_UNRESOLVED"
#: argv markers separating the two pattern lists.  Positional lists cannot be
#: split by count -- both are operator-configurable and either may be empty.
_ALLOW_FLAG = "--allow"
_IGNORE_FLAG = "--ignore"
_RESOLVE_MODEL = f"""
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError
repo = sys.argv[1]
revision = sys.argv[2]
rest = sys.argv[3:]
split = rest.index("{_IGNORE_FLAG}")
allow = rest[1:split] or None
ignore = rest[split + 1:] or None
try:
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )
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
    #: What the patterns above must *not* be allowed to reach.  They are matched
    #: with ``fnmatch``, whose ``*`` crosses ``/``, so without this they claim
    #: nested auxiliary paths -- ``original/config.json``,
    #: ``original/model.safetensors`` (13.7 GB on ``openai/gpt-oss-20b``) --
    #: that a minimal cache never downloaded, and the pin is then recorded as
    #: unsatisfied on a cache that holds everything training reads.  ``*/*``
    #: matches exactly the paths containing a separator; see ``_RESOLVE_MODEL``.
    model_resolve_ignore_patterns: tuple[str, ...] = ("*/*",)
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
        for name, patterns in (
            ("model_resolve_allow_patterns", self.model_resolve_allow_patterns),
            ("model_resolve_ignore_patterns", self.model_resolve_ignore_patterns),
        ):
            # The two lists travel to the resolver subprocess as one argv split
            # on the marker tokens, so a pattern equal to a marker would silently
            # move files from one list to the other.
            if any(
                not isinstance(pattern, str)
                or not pattern
                or pattern in (_ALLOW_FLAG, _IGNORE_FLAG)
                for pattern in patterns
            ):
                raise ValueError(
                    f"{name} must be non-empty strings other than "
                    f"{_ALLOW_FLAG!r} and {_IGNORE_FLAG!r}"
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
    #: Whether the configured verifier revision was actually satisfied.
    #:
    #: ``None`` until the verifier is resolved, or when no revision was pinned
    #: at all.  ``False`` records that the pin could not be met and the cycle
    #: continued unpinned -- the one state the published manifest previously
    #: could not express, because it copied the *requested* revision whether or
    #: not resolution had honoured it.
    verifier_pinned: bool | None = None


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
            resolved, pinned = self._resolve(
                self.config.verifier_model,
                self.config.verifier_revision,
                "verifier model resolution",
                guard,
                scratch,
            )
            self.state.verifier = resolved
            self.state.verifier_pinned = pinned
            if pinned is False:
                _record_unpinned_verifier(
                    scratch, self.config.verifier_model, self.config.verifier_revision
                )
        return self.state.verifier

    def warm_start(self, model: str, guard: AbortCheck, scratch: Path) -> str:
        if self.state.warm_start is None:
            revision = (
                self.config.warm_start_revision
                if model == self.config.warm_start_model
                else None
            )
            self.state.warm_start, _ = self._resolve(
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
    ) -> tuple[str, bool | None]:
        """Resolve *model*, reporting whether its configured pin was honoured.

        The second element is ``None`` when no pin applied -- either none was
        configured, or *model* is already a concrete on-disk path, which names
        exact weights and leaves nothing for a revision to pin down.
        """
        if revision is None or Path(model).exists():
            return model, None
        result = self.runner.run(
            [
                str(self.config.training_python),
                "-c",
                _RESOLVE_MODEL,
                model,
                revision,
                _ALLOW_FLAG,
                *self.config.model_resolve_allow_patterns,
                _IGNORE_FLAG,
                *self.config.model_resolve_ignore_patterns,
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
            return model, False
        return resolved, True


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
        except BaseException as error:
            preserve_failure_evidence(
                destination.parent, "trace-snapshot-lease", error, outputs=(destination,)
            )
            _discard((destination,), primary=error)
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
            persist_stage_output(scratch, _ROW_RENDER_STAGE, prepare)
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
        except BaseException as error:
            preserve_failure_evidence(
                scratch, _ROW_RENDER_STAGE, error, outputs=(destination, data)
            )
            _discard((destination, data), primary=error)
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
        server: RunningProcess | None = None

        def stop_server() -> ProcessResult | None:
            """Terminate the hidden-state server once, persisting its output.

            Idempotent by design: the quota guard, the health-check failure
            path, the failure handler and the ``finally`` below all call it, and
            only the first does the work.  The vLLM launcher's own output lives
            in the returned ``ProcessResult`` and nowhere else; dropping it left
            the extraction server's side of every failure unexaminable.

            Returns:
                The terminated server's output, or ``None`` if it was already
                stopped.
            """
            nonlocal server
            running, server = server, None
            if running is None:
                return None
            stopped = self.runner.terminate(
                running,
                grace_seconds=self.config.server_shutdown_timeout_seconds,
            )
            persist_stage_output(scratch, "hidden-state-server", stopped)
            return stopped

        guard = _guard(
            scratch,
            self.config.scratch_quota_bytes,
            should_abort,
            (destination,),
            stop=stop_server,
        )
        started = self.clock()
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
                    stopped = stop_server()
                    raise TrainingError(
                        f"vLLM hidden-state server exited with status {returncode}",
                        stderr=stopped.stderr if stopped is not None else "",
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
            # Persisted before the status check so a *failed* generation leaves
            # the same evidence a successful one does.
            persist_stage_output(scratch, _EXTRACTION_STAGE, generated)
            _success("offline hidden-state generation", generated)
            _check_hidden_state_layers(destination, len(layers), generated)
            return destination
        except BaseException as error:
            # Order matters twice over.  The inventory of what extraction
            # produced has to be taken while the shards are still there, so it
            # comes first -- and the server has to be gone before the shards
            # are deleted, because it writes ``cmpl-*.safetensors`` and their
            # ``.lock`` siblings straight into ``destination``.  Deleting under
            # a live server is what turned job 369325's quota trip into an
            # ``Errno 39`` that replaced its own root cause.
            preserve_failure_evidence(
                scratch, _EXTRACTION_STAGE, error, outputs=(destination,)
            )
            _stop_quietly(stop_server, primary=error)
            _discard((destination,), primary=error)
            raise
        finally:
            stop_server()


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
        except BaseException as error:
            preserve_failure_evidence(
                scratch,
                _TRAINING_STAGE,
                error,
                outputs=(destination, scratch / "warm-start-pinned"),
            )
            _discard((destination, scratch / "warm-start-pinned"), primary=error)
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
        except BaseException as error:
            preserve_failure_evidence(
                destination.parent,
                "draft-materialization",
                error,
                outputs=(destination,),
            )
            _writable(destination)
            _discard((destination,), primary=error)
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

    #: Resolution state shared with the stage components, when built by the
    #: factory.  ``None`` for an adapter assembled directly in a test.
    _state: _State | None = None

    def describe(self) -> BackendInfo:
        """Report the verifier revision that ran, not the one that was asked for.

        The base implementation copies the *configured* pin into the training
        parameters that become the artifact manifest.  Resolution, though, is
        allowed to fall back: when the cache cannot satisfy the pin the cycle
        continues against the bare repo id, because a pin is provenance and
        must not be able to stop a cycle the unpinned path would have run.

        That fallback is kept.  What is not kept is the manifest asserting a
        revision the cycle did not verify -- a recorded pin that does not
        describe the weights that ran is worse than no pin, because nothing
        downstream can tell the two apart.  So the record is made honest
        instead of the failure made fatal: ``verifier_revision`` goes null
        exactly when the cycle could not be pinned, which the base docstring
        already defines as "ran unpinned", and the request that could not be
        met is preserved beside it so the drop is visible rather than merely
        absent.  ``describe`` is read after training, so the resolution this
        consults is the one the cycle actually used.

        ``verifier_revision_satisfied`` is written on *both* outcomes.  Written
        only on failure it is unfalsifiable: a manifest without it could mean
        the pin held, or that the cycle predates the field, or that the state
        was never consulted, and a provenance record that cannot say "yes"
        cannot be compared against one that says "no".  It stays absent only
        when no pin applied at all -- nothing was asked, so nothing was
        satisfied or missed.
        """
        info = super().describe()
        if self._state is None or self._state.verifier_pinned is None:
            return info
        params = dict(info.training_params)
        params["verifier_revision_satisfied"] = self._state.verifier_pinned
        if self._state.verifier_pinned:
            return BackendInfo(
                verifier_model=info.verifier_model,
                draft_model=info.draft_model,
                from_pretrained=info.from_pretrained,
                training_params=params,
            )
        params["verifier_revision"] = None
        params["verifier_revision_requested"] = self.config.verifier_revision
        return BackendInfo(
            verifier_model=info.verifier_model,
            draft_model=info.draft_model,
            from_pretrained=info.from_pretrained,
            training_params=params,
        )

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
        backend = cls(
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
        backend._state = state
        return backend

    def _check(self, work_dir: Path, should_abort: AbortCheck) -> None:
        try:
            super()._check(work_dir, should_abort)
        except ScratchQuotaExceeded as error:
            preserve_failure_evidence(
                work_dir,
                _QUOTA_STAGE,
                error,
                outputs=tuple(work_dir / name for name in _TRANSIENT_NAMES),
            )
            #: ``primary=error`` so a stray lock or temp file left behind by a
            #: stage's subprocess cannot substitute an ``OSError`` for the
            #: quota breach that is the actual reason this cycle is ending.
            _cleanup_transients(work_dir, primary=error)
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
        preserve_failure_evidence(
            destination.parent, "warm-start-pinning", error, outputs=(destination,)
        )
        _discard((destination,), primary=error)
        raise TrainingError(
            f"warm-start config is not a Speculators draft: {config_path}",
            stderr=str(error),
        ) from error
    except BaseException as error:
        preserve_failure_evidence(
            destination.parent, "warm-start-pinning", error, outputs=(destination,)
        )
        _discard((destination,), primary=error)
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
    return _persist_streams(run_dir / "training-logs", result, max_bytes=max_bytes)


def persist_stage_output(
    run_dir: Path,
    stage: str,
    result: ProcessResult,
    *,
    max_bytes: int = MAX_TRAINING_LOG_BYTES,
) -> Path | None:
    """Write one stage's subprocess streams under ``<run_dir>/stage-logs/<stage>``.

    Training was the only stage that kept its subprocess output.  Every other
    stage surfaced it solely by attaching stderr to an exception, so a stage
    that failed for a reason *other* than a non-zero exit -- an abort, a
    timeout, a quota trip, or an error raised in this process while the child
    ran -- left no trace of what the child had been doing, and the failure
    path then deleted the child's output directory as well.

    Persisting is best effort and returns ``None`` on failure: losing
    diagnostics must never be what fails a cycle.
    """
    return _persist_streams(
        run_dir / STAGE_LOG_DIR_NAME / _slug(stage), result, max_bytes=max_bytes
    )


def preserve_failure_evidence(
    run_dir: Path,
    stage: str,
    error: BaseException,
    *,
    outputs: Sequence[Path] = (),
) -> Path | None:
    """Record why a stage failed and what it had produced, before that is deleted.

    A stage's cleanup exists so a half-written output cannot be mistaken for a
    complete one by a later stage, and that requirement is real -- partial
    hidden states or a partial draft must not survive in consumable form.  But
    deleting the output also deleted the only evidence of the failure, so the
    root cause of a failed cycle was unrecoverable by construction.

    The two are separated here rather than traded off.  What survives is the
    child's streams (if the exception carries them) and an *inventory* of the
    output: how many entries it held, how many bytes, and up to
    :data:`MAX_INVENTORY_ENTRIES` names with sizes.  That answers the
    questions a missing shard raises -- how many shards existed, whether the
    count was off by one, whether a name did not match -- while the bytes
    themselves, which are what a downstream stage could consume and what would
    fill the scratch quota, are still destroyed by the caller.

    Writing evidence must never mask the failure it documents, so every error
    here is swallowed and reported as ``None``.
    """
    directory = run_dir / STAGE_LOG_DIR_NAME / _slug(stage)
    captured = process_output(error)
    if captured is not None:
        _persist_streams(directory, captured)
    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "recorded_at": time.time(),
        "error_type": type(error).__name__,
        "error": str(error),
        "outputs": [_inventory(path) for path in dict.fromkeys(outputs)],
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "failure.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("could not preserve failure evidence in %s: %s", directory, exc)
        return None
    return directory


def _record_unpinned_verifier(
    run_dir: Path,
    model: str,
    revision: str | None,
) -> Path | None:
    """Leave a durable record that a cycle ran on an unpinned verifier.

    The warning that used to be the only trace of this lived in the gateway
    log, which is not part of the cycle's artifacts and is not retained with
    them.  A cycle that then failed -- as this one did -- left nothing to say
    the pin had been dropped.  Best effort, like every other diagnostic write.
    """
    directory = run_dir / STAGE_LOG_DIR_NAME / "provenance"
    record = {
        "schema_version": 1,
        "recorded_at": time.time(),
        "verifier_model": model,
        "verifier_revision_requested": revision,
        "verifier_revision_satisfied": False,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verifier.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("could not record verifier provenance in %s: %s", directory, exc)
        return None
    return directory


def _inventory(path: Path) -> dict[str, Any]:
    """Summarise what a stage produced at *path* without retaining it.

    Tolerates entries vanishing under the walk for the same reason
    :func:`~speedlm.tuner.eagle3.scratch_usage` does: the stage's subprocess
    may still be terminating while this runs.
    """
    entry: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not entry["exists"]:
        return entry
    if path.is_file():
        with contextlib.suppress(OSError):
            entry["bytes"] = path.stat().st_size
        return entry
    names: list[dict[str, Any]] = []
    entries = 0
    total = 0
    for child in sorted(path.rglob("*")):
        try:
            if not child.is_file():
                continue
            size = child.stat().st_size
        except OSError:
            continue
        entries += 1
        total += size
        if len(names) < MAX_INVENTORY_ENTRIES:
            names.append({"name": str(child.relative_to(path)), "bytes": size})
    entry["entries"] = entries
    entry["bytes"] = total
    entry["sample"] = names
    entry["truncated"] = entries > len(names)
    return entry


def _persist_streams(
    directory: Path,
    result: ProcessResult,
    *,
    max_bytes: int = MAX_TRAINING_LOG_BYTES,
) -> Path | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "command.txt").write_text(
            " ".join(result.argv) + "\n", encoding="utf-8"
        )
        for name, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
            (directory / f"{name}.log").write_text(
                _bounded(stream or "", max_bytes), encoding="utf-8"
            )
    except OSError as exc:
        logger.warning("could not persist subprocess output to %s: %s", directory, exc)
        return None
    return directory


def _slug(stage: str) -> str:
    """Reduce a stage name to a safe single path component."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", stage.lower()).strip("-")
    return cleaned or "stage"


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
    stop: Callable[[], object] | None = None,
) -> AbortCheck:
    """Return an abort check that also enforces the scratch quota.

    Args:
        scratch: the cycle's scratch directory, re-walked on every check.
        quota: the byte ceiling; see
            :func:`speedlm.tuner.eagle3.derive_scratch_quota_bytes`.
        should_abort: the preemption check this one wraps.
        cleanup: stage outputs a quota trip removes on top of the transients.
        stop: shuts down the subprocesses that are writing into *scratch*.
            Optional only because most stages have no live writer at the moment
            their guard fires; the stage that does -- hidden-state extraction --
            must pass one.  See the rationale inside.
    """

    def check() -> bool:
        used = scratch_usage(scratch)
        if used > quota:
            # This sweep is the broadest deleter in the pipeline -- it removes
            # every stage's output, not just the running stage's -- so the
            # inventory is taken first.  Without it a quota trip erased the
            # very sizes that would show which stage overran the quota.
            error = ScratchQuotaExceeded(used, quota)
            preserve_failure_evidence(
                scratch,
                _QUOTA_STAGE,
                error,
                outputs=(*cleanup, *(scratch / name for name in _TRANSIENT_NAMES)),
            )
            #: Stop the writers BEFORE deleting the tree they are writing into.
            #:
            #: This check runs from inside ``SubprocessRunner.run``'s poll loop,
            #: so on job 369325 both the vLLM hidden-state server and the
            #: Speculators data-generation client were still alive and still
            #: creating ``cmpl-*.safetensors`` and sibling ``*.safetensors.lock``
            #: files in ``hidden-states`` while ``rmtree`` walked it.  Deleting
            #: first does not merely risk ``Errno 39`` on our side; it also
            #: provokes the client's own ``os.remove(lock_path)`` at
            #: ``speculators/data_generation/vllm_client.py:144`` into a
            #: ``FileNotFoundError``, in vendored code we cannot patch.  Both
            #: races disappear if the tree is quiet before it is removed, which
            #: is why the ordering matters more than the ``rmtree`` retry.
            if stop is not None:
                _stop_quietly(stop, primary=error)
            _discard(cleanup, primary=error)
            _cleanup_transients(scratch, primary=error)
            raise error
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
        while chunk := input_file.read(DRAFT_COPY_CHUNK_BYTES):
            _deadline(started, timeout, "draft materialization")
            _abort(guard, "draft materialization")
            output_file.write(chunk)
    _deadline(started, timeout, "draft materialization")
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
        _remove_tree(path)


def _remove_tree(path: Path) -> None:
    """Delete a directory tree, re-trying entries that reappear under the walk.

    Job 369325: a scratch-quota trip removed ``hidden-states`` while the
    Speculators data-generation client and the vLLM hidden-state server were
    both still running, and ``shutil.rmtree`` raised
    ``OSError: [Errno 39] Directory not empty`` on a single leftover 0-byte
    ``*.safetensors.lock``.  ``rmtree`` enumerates a directory and then
    ``rmdir``s it; a writer that creates one file in that gap is enough.

    The real fix is ordering -- :func:`_guard` and the extraction stage now
    stop their writers *before* deleting the tree, so nothing should be
    creating files here at all.  This is the belt to that braces, for the
    handful of paths the ordering fix cannot reach (a process in its SIGTERM
    grace period, an NFS client flushing).

    It is not a weakening: the final attempt is unguarded, so a tree that
    genuinely cannot be removed still raises the same ``OSError`` it always
    did.

    Raises:
        OSError: if the tree still exists after :data:`_REMOVE_TREE_ATTEMPTS`.
    """
    for attempt in range(_REMOVE_TREE_ATTEMPTS):
        if not path.exists():
            return
        _writable(path)
        try:
            shutil.rmtree(path)
        except OSError:
            if attempt == _REMOVE_TREE_ATTEMPTS - 1:
                raise
        else:
            return


def _discard(paths: Iterable[Path], *, primary: BaseException) -> None:
    """Remove *paths* without ever letting cleanup replace *primary*.

    ``preserve_failure_evidence`` already refuses to let *writing* evidence
    mask the failure it documents.  Deletion had no such treatment, so job
    369325 reported ``[Errno 39] Directory not empty`` -- its own cleanup's
    secondary symptom -- while the ``ScratchQuotaExceeded`` that caused the
    abort survived only in ``stage-logs/scratch-quota/failure.json``, invisible
    to the cycle result, the gateway log and the SLURM output.

    A cleanup failure is still evidence, so it is recorded on the primary
    exception as a note and in :attr:`Eagle3Error.cleanup_errors` rather than
    discarded.  What it may not do is become the reported error.
    """
    for path in paths:
        try:
            _remove(path)
        except OSError as error:
            _note(primary, f"cleanup could not remove {path}: {error}")


def _stop_quietly(stop: Callable[[], object], *, primary: BaseException) -> None:
    """Run *stop* on a failure path, recording rather than raising its errors.

    Same contract as :func:`_discard`: shutting the stage's processes down is
    housekeeping on the way out of a failure, and housekeeping must not become
    the failure.
    """
    try:
        stop()
    except Exception as error:  # noqa: BLE001 - see docstring
        _note(primary, f"could not stop the stage's processes: {error}")


def _note(primary: BaseException, message: str) -> None:
    """Attach *message* to *primary* without changing its type."""
    primary.add_note(message)
    if isinstance(primary, Eagle3Error):
        primary.cleanup_errors = (*primary.cleanup_errors, message)


def _writable(path: Path) -> None:
    """Re-open a tree for deletion, tolerating entries that vanish under the walk.

    Same race as :func:`~speedlm.tuner.eagle3.scratch_usage`: this runs on the
    failure path while the stage's subprocess may still be renaming files out
    from under it, and an entry that disappeared needs no chmod.  Suppressing
    the error hides nothing -- if a permission really cannot be relaxed, the
    ``shutil.rmtree`` this prepares for fails loudly on the same path.
    """
    if not path.exists():
        return
    for child in path.rglob("*"):
        with contextlib.suppress(OSError):
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


def _cleanup_transients(
    work_dir: Path, *, primary: BaseException | None = None
) -> None:
    """Remove every transient stage output beneath *work_dir*.

    Args:
        work_dir: the cycle's scratch directory.
        primary: the failure this cleanup is running on behalf of.  When given,
            removal errors are recorded on it via :func:`_discard` instead of
            replacing it.  ``None`` keeps the old raise-on-failure behaviour for
            the success paths, where there is no primary error to protect.
    """
    paths = [work_dir / name for name in _TRANSIENT_NAMES]
    if primary is None:
        for path in paths:
            _remove(path)
        return
    _discard(paths, primary=primary)


def _health(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


__all__ = [
    "DEFAULT_SPECULATORS_PYTHON",
    "MAX_INVENTORY_ENTRIES",
    "STAGE_LOG_DIR_NAME",
    "persist_stage_output",
    "preserve_failure_evidence",
    "DEFAULT_SPECULATORS_REPO",
    "DRAFT_COPY_CHUNK_BYTES",
    "MAX_SCRATCH_BYTES",
    "SHARD_BYTES_PER_ROW",
    "SCRATCH_HEADROOM_BYTES",
    "derive_scratch_quota_bytes",
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
