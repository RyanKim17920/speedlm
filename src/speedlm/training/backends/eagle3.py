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
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from speedlm.config import MAX_LEARNING_RATE, REFERENCE_LEARNING_RATE
from speedlm.profiles import DEFAULT_SPECULATIVE_TOKENS, MAX_SPECULATIVE_TOKENS
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
    REQUIRED_DRAFT_TENSORS,
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
    WarmStartResolver,
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

#: No decay: every TTT step's loss carries the same weight in the summed
#: objective.  Matches the Speculators trainer's own ``--ttt-step-loss-decay``
#: default (``scripts/train.py:667``), so naming the flag changes nothing.
#: See :attr:`SpeculatorsPipelineConfig.ttt_step_loss_decay` for what moving it
#: would mean and what evidence would justify it.
DEFAULT_TTT_STEP_LOSS_DECAY: Final = 1.0

DEFAULT_SPECULATORS_REPO = Path(
    os.environ.get("SPEEDLM_SPECULATORS_REPO", "speculators")
)
DEFAULT_SPECULATORS_PYTHON = Path(
    os.environ.get("SPEEDLM_TRAINING_PYTHON", sys.executable)
)
LOSS_MASK_DILATION_SCRIPT: Final = (
    Path(__file__).resolve().parents[1] / "dilate_prepared_loss_mask.py"
)
_SPECULATORS_DATA_NAME = "speculators-conversations.jsonl"


class TruncatedRowPolicy(StrEnum):
    """What a rendered corpus does with a row whose completion was cut off."""

    #: Emit the row unchanged.  Restores the historical behaviour.
    KEEP = "keep"
    #: Omit the row from the rendered dataset.
    DROP = "drop"


#: ``finish_reason`` values that mean "this completion was cut off", i.e. the
#: assistant turn on disk is a mid-sentence fragment and not a finished turn.
#:
#: Why this matters for a drafter: production capture caps ``max_tokens`` at
#: 512 (``config.benchmark_max_tokens``) and ``gate/replay.py`` measured 76 of
#: 103 held-out contexts burning the whole budget inside the thinking block and
#: returning null content with ``finish_reason="length"``.  The tokenizer sees
#: such a row exactly as it sees a natural stop, and the chat template appends
#: its end-of-turn marker after the fragment -- teaching the draft that a
#: mid-sentence fragment is a complete turn, so at serving the draft proposes a
#: continuation precisely where the verifier emits stop.
#:
#: ``"length"`` is the OpenAI chat value for exhausting ``max_tokens``
#: (:data:`speedlm.gate.replay.FINISH_REASON_LENGTH`).  ``"incomplete"`` is the
#: Responses API's equivalent: that path files the response *status* into this
#: same field (``gateway/responses.py``), where a truncated response reads
#: ``"incomplete"`` and a finished one reads ``"completed"``.
#:
#: Deliberately a positive list and not "anything that is not ``stop``".  Three
#: real sources produce an *unknown* value -- ``TraceRecord.finish_reason``
#: defaults to ``None`` so corpora archived before the field existed read back
#: ``None``, streaming chunks carry ``null`` until the terminal chunk, and
#: ``gate/replay.py`` coerces a missing value to ``""`` -- and ``"tool_calls"``
#: is a perfectly complete stop.  An inverted test would discard all four.
TRUNCATED_FINISH_REASONS: Final = frozenset({"length", "incomplete"})

#: Key holding a Speculators draft's auxiliary layer ids in its ``config.json``.
_AUX_LAYER_IDS_KEY: Final = "eagle_aux_hidden_state_layer_ids"

#: Largest ``safetensors`` header this module will parse, in bytes.
#:
#: The header is a JSON object of tensor names and shapes; a draft's runs to a
#: few kilobytes.  The bound exists so a corrupt or hostile length prefix
#: cannot make the guard read a gigabyte into memory.
_SAFETENSORS_HEADER_LIMIT: Final = 16 * 1024 * 1024

#: One wrapped ``val/...`` record from the trainer's stdout, up to its epoch.
#:
#: The trainer logs through Rich, which hard-wraps every record to 80 columns
#: with a 20-space continuation indent, so one logical record spans six to ten
#: physical lines and a per-line regex sees only whichever steps happened to
#: land on the line it is looking at.  Matching the whole record with DOTALL is
#: what makes the parse independent of where the wrap fell.
#:
#: ``\bepoch=`` cannot match inside ``loss_0_epoch=`` -- the preceding ``_`` is
#: a word character, so there is no boundary there -- which is what lets the
#: non-greedy span stop at the record's own trailing ``epoch=<n>`` field.
_VAL_EPOCH_RECORD: Final = re.compile(r"val/loss_0_epoch=.*?\bepoch=(\d+)", re.DOTALL)
_VAL_EPOCH_METRIC: Final = re.compile(
    r"val/([A-Za-z0-9_]+)_epoch=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
#: Rich's right-aligned source tag, e.g. ``trainer.py:290``, stripped before parsing.
_LOG_SOURCE_TAG: Final = re.compile(r"[ \t]+[A-Za-z0-9_]+\.py:\d+[ \t]*$", re.MULTILINE)

#: The metric that governs acceptance at the first speculative position.
#:
#: The trainer emits two accuracy series per TTT step.  ``full_acc_k`` is the
#: unconditional chain accuracy -- the fraction of *all* supervised positions
#: whose draft is correct through step ``k``.  ``cond_acc_k`` is conditional on
#: surviving step ``k-1``.  They share a numerator and differ only in
#: denominator, so ``cond_acc_k == full_acc_k / full_acc_{k-1}``; verified
#: against a real epoch where the series read 0.648 / 0.372 / 0.168 and 0.648 /
#: 0.574 / 0.452, since 0.372/0.648 = 0.574 and 0.168/0.372 = 0.452.  At step 0
#: there is nothing to condition on, so the two are equal and either would do;
#: ``full_acc_0_epoch`` is used because it stays comparable across steps.
ACCEPTANCE_METRIC: Final = "full_acc_0_epoch"

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
_MASK_DILATION_STAGE: Final = "loss-mask-left-dilation"
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
#:
#: *Tensors.*  This asked only for ``d2t``/``t2d`` -- two vocabulary index maps
#: that say nothing about whether a draft head is present, let alone a trained
#: one.  It now requires all of
#: :data:`speedlm.tuner.eagle3.REQUIRED_DRAFT_TENSORS`, ``fc.weight`` included.
#: That is a *packaging* check; whether the weights actually moved is a
#: different question, answered by the fingerprint comparison in
#: :meth:`speedlm.tuner.eagle3.Eagle3Adapter._record_draft_weights`.
_VALIDATE_DRAFT_BODY = """
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
missing = set(REQUIRED_TENSORS) - keys
if missing:
    raise SystemExit(f"materialized draft is missing required tensors: {sorted(missing)}")
""".strip()

#: The validator script with its required-tensor set spliced in.
#:
#: Injected as a literal assignment rather than formatted into the body, so
#: the body's own braces (f-strings, set literals) stay untouched, and so the
#: subprocess and :func:`speedlm.tuner.eagle3.draft_tensor_keys` can never
#: disagree about what a trained head must contain.
_VALIDATE_DRAFT: Final = (
    f"REQUIRED_TENSORS = {sorted(REQUIRED_DRAFT_TENSORS)!r}\n{_VALIDATE_DRAFT_BODY}"
)


class EmptySpeculatorsDatasetError(Eagle3Error):
    """No captured record survived conversion into the Speculators contract.

    Raised when nothing was written and no *one* filter dominates the loss --
    including when the dominant bucket is ``dropped_untrainable``, which has no
    dedicated remedy to print because there is no knob that makes a row without
    an assistant turn trainable.  ``counts`` is optional only because one older
    call site predates the reconciliation; when it is supplied the message
    carries the full accounting, so "nothing converted" is never the *whole*
    answer where a per-bucket breakdown exists.
    """

    def __init__(
        self, source: Path, counts: RenderedRowCounts | None = None
    ) -> None:
        self.source = source
        self.counts = counts
        message = (
            f"no captured trace in {source} converted to a Speculators conversation"
        )
        if counts is not None:
            message += f". {counts.drop_accounting()}"
        super().__init__(message)


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
    #: Depth of the draft chain to train, passed through as ``--ttt-steps``.
    #:
    #: Previously absent, so the trainer ran at its own default of 3
    #: (``scripts/train.py:660``) no matter what the profile served.  It is set
    #: from the profile's ``num_speculative_tokens`` in
    #: :func:`speedlm.tuner.composition.create_production_tuner`, which is the
    #: serving truth, and checked against it by
    #: :func:`speedlm.profiles.validate_training_depth`.
    num_speculative_steps: int = DEFAULT_SPECULATIVE_TOKENS
    #: Per-TTT-step loss weight decay, passed through as
    #: ``--ttt-step-loss-decay``.
    #:
    #: The trainer weights step ``s``'s loss by ``gamma ** s``
    #: (``exp_loss_decay`` in ``speculators/models/metrics.py``) and *sums* the
    #: weighted per-step losses into one objective -- which is what
    #: ``ttt_loss_reduction: "sum"`` in the manifest records.  So ``gamma``
    #: chooses how the gradient is split between the near positions, which
    #: dominate expected accepted length, and the deep ones, which do not.
    #:
    #: ``1.0`` is no decay: every step counts equally.  It is the trainer's own
    #: default (``scripts/train.py:667``) and was previously inherited rather
    #: than passed, so nothing recorded it and nothing could vary it.  The
    #: default here is deliberately that same ``1.0`` -- naming the flag must
    #: not silently change what any cycle trains.
    #:
    #: The reason to move it: with no decay and a summed objective, the deep
    #: steps dominate simply because they are harder.  Measured per-step
    #: training losses on the gpt-oss run under
    #: ``/data/ryan.kim/speedlm-runs/d993eee-gptoss-idle`` were 0.755, 2.000,
    #: 2.959, 3.622, 4.144 for steps 0..4, so most of the gradient went to the
    #: positions that contribute least.  That arm measured -7.38% throughput
    #: with position-0 acceptance at parity (0.6850 vs 0.6871) and the loss
    #: confined to the tail (-2.4/-4.4/-5.3/-5.3 pp at positions 1..4), while
    #: the depth-3 Qwen arm was flat (-0.29%).  A ``gamma < 1`` is the lever
    #: that would shift that gradient back toward position 0.
    #:
    #: What would justify changing it: a GPU sweep of ``gamma`` at fixed depth
    #: showing an *end-to-end throughput* gain against the stock baseline on
    #: the same traffic -- not a lower training loss, and not a higher
    #: position-0 acceptance alone, since trading tail acceptance for head
    #: acceptance can leave tokens-per-verifier-step unchanged.  Until such a
    #: sweep exists this stays at 1.0; it is a value to measure, not to guess.
    ttt_step_loss_decay: float = DEFAULT_TTT_STEP_LOSS_DECAY
    #: Sequence length for training. A value of 16384 collapsed a 512-record
    #: corpus into 1 batch; 4096 yielded 44 steps, making this a key lever
    #: for sampler throughput and gradient frequency.
    sequence_length: int = 16_384
    #: Learning rate handed to the trainer as ``--lr``.
    #:
    #: Both the default and the bound below are *imported* from
    #: :mod:`speedlm.config` rather than restated here.  Restating them is what
    #: produced the bug this comment exists for: ``tuning.learning_rate`` was
    #: raised to the Speculators reference ``1e-4`` in :mod:`speedlm.config`
    #: while this file kept an independent copy of the old ``1e-5`` bound, so a
    #: real cycle stopped failing at training and started failing at
    #: composition instead.  One definition, two readers.
    learning_rate: float = REFERENCE_LEARNING_RATE
    epochs: int = 1
    seed: int = 0
    port: int = 8_131
    concurrency: int = 8
    mask_policy: MaskPolicy = MaskPolicy.ALL_ASSISTANT_TURNS
    #: What to do with a captured row whose completion hit the token cap.
    #:
    #: ``DROP`` by default because a truncated row is not merely uninformative,
    #: it is actively wrong supervision: see :data:`TRUNCATED_FINISH_REASONS`.
    #: The cost of that default is real and is why the floor below exists --
    #: the measured truncation rate on seed traffic under a 512-token cap is
    #: high, so on some corpora this filter removes most rows.  When it does,
    #: the cycle fails by name rather than training on the remainder.
    truncated_row_policy: TruncatedRowPolicy = TruncatedRowPolicy.DROP
    #: Rendered rows the truncation filter must leave behind, or the cycle fails.
    #:
    #: The accumulation gate that admits a cycle (``tuning.min_corpus_records``,
    #: default 256, against a ``tuning.training_window_records`` window of 256)
    #: counts *raw trace records*, before rendering and therefore before this
    #: filter runs.  So nothing upstream can notice that a 256-record window
    #: became sixteen trainable rows.  This is that check, and it is deliberately
    #: a hard failure: a silently tiny corpus produces a checkpoint that looks
    #: like every other checkpoint.  32 matches ``tuning.min_trace_records``,
    #: the codebase's existing floor for "enough to mean anything".
    #:
    #: Only consulted when the filter actually removed rows; a corpus that is
    #: small for its own reasons is not this check's business.
    min_rendered_rows: int = 32
    #: Accept assistant turns that carry no ``provenance_tag`` as this
    #: verifier's own output.
    #:
    #: Off by default and deliberately: the gateway tags every message it
    #: captures, so an untagged assistant turn means the corpus predates
    #: tagging, and "predates tagging" is not evidence of authorship.  Mirrors
    #: ``trust_untagged_assistant_messages`` in
    #: :func:`speedlm.training.rows.training_row_from_trace`, which made the
    #: same call for the same reason.  It exists so an operator with a trusted
    #: offline corpus has a named opt-in rather than a corpus the authorship
    #: filter empties with no way back.
    trust_untagged_assistant_messages: bool = False
    #: Extend every prepared loss-mask span one position to the left.
    #:
    #: Speculators shifts ``loss_mask`` when it constructs labels, so the mask
    #: position immediately before assistant content supervises its first token.
    #: Off by default because this changes the training objective; operators
    #: must opt in explicitly and the value is recorded in the draft manifest.
    dilate_loss_mask_span_starts: bool = False
    #: Fail the cycle when the final epoch does not improve on the first.
    #:
    #: Measured on :data:`ACCEPTANCE_METRIC` and not on the loss, because the
    #: two disagree: the summed TTT loss is dominated by the later steps
    #: (loss_0=0.643 against loss_2=3.571 in a total of 6.676 -> 6.425), so it
    #: falls across epochs while step-0 accuracy -- the direct proxy for what
    #: acceptance does at serving -- sits flat at 0.706/0.705/0.705 on gpt-oss
    #: and 0.648/0.647/0.648 on Qwen.  A cycle gated on the loss promotes that.
    require_accuracy_improvement: bool = True
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
        _positive_int("num_speculative_steps", self.num_speculative_steps)
        if self.num_speculative_steps > MAX_SPECULATIVE_TOKENS:
            raise ValueError(
                f"num_speculative_steps must be at most {MAX_SPECULATIVE_TOKENS}"
            )
        # Bounded above by 1.0 because a gamma above 1 would weight the *deep*
        # steps even more heavily, which is the direction the gpt-oss
        # regression already came from; and above 0 because 0 would train
        # nothing but step 0 while still paying for the deeper rollout.
        if (
            isinstance(self.ttt_step_loss_decay, bool)
            or not isinstance(self.ttt_step_loss_decay, (int, float))
            or not 0 < self.ttt_step_loss_decay <= 1
        ):
            raise ValueError("ttt_step_loss_decay must be in (0, 1]")
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
        _positive_int("min_rendered_rows", self.min_rendered_rows)
        if not isinstance(self.truncated_row_policy, TruncatedRowPolicy):
            raise ValueError("truncated_row_policy must be a TruncatedRowPolicy")
        if not isinstance(self.require_accuracy_improvement, bool):
            raise ValueError("require_accuracy_improvement must be a bool")
        if not isinstance(self.trust_untagged_assistant_messages, bool):
            raise ValueError("trust_untagged_assistant_messages must be a bool")
        if not isinstance(self.dilate_loss_mask_span_starts, bool):
            raise ValueError("dilate_loss_mask_span_starts must be a bool")
        if not isinstance(self.mask_policy, MaskPolicy):
            raise ValueError("mask_policy must be a MaskPolicy")
        if self.mask_policy is not MaskPolicy.ALL_ASSISTANT_TURNS:
            raise ValueError(
                "the Speculators all_assistant pipeline requires "
                "MaskPolicy.ALL_ASSISTANT_TURNS"
            )
        # A fat-finger guard, not a safety boundary: nothing here has ever
        # established where "unsafe" begins, and the previous bound of ``1e-5``
        # called itself safe while making training a measured no-op.  See
        # :data:`speedlm.config.MAX_LEARNING_RATE` for what the bound is and is
        # not claiming.
        if (
            isinstance(self.learning_rate, bool)
            or not 0 < self.learning_rate <= MAX_LEARNING_RATE
        ):
            raise ValueError(
                "learning_rate must be in (0, "
                f"{MAX_LEARNING_RATE:g}] (fat-finger guard, not a safety limit)"
            )
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
    #: Which model string :attr:`warm_start` was resolved *from*.
    #:
    #: The state object lives for the whole process, not for one cycle, so a
    #: memo keyed on nothing returns the first cycle's resolution forever.  That
    #: was harmless only while the warm start was a process-wide constant; once
    #: each cycle can warm-start from the current incumbent, an unkeyed memo
    #: silently trains every later cycle from the first cycle's base -- exactly
    #: the defect the per-cycle resolver exists to remove, reintroduced one
    #: layer down.
    warm_start_source: str | None = None
    #: Whether the configured verifier revision was actually satisfied.
    #:
    #: ``None`` until the verifier is resolved, or when no revision was pinned
    #: at all.  ``False`` records that the pin could not be met and the cycle
    #: continued unpinned -- the one state the published manifest previously
    #: could not express, because it copied the *requested* revision whether or
    #: not resolution had honoured it.
    verifier_pinned: bool | None = None
    #: Kept/dropped row counts from the truncation filter, for the manifest.
    rendered_rows: Mapping[str, object] | None = None
    #: What the warm-start checkpoint declared about its aux layers, and the
    #: verdict of the alignment check against the ids extraction actually ran.
    warm_start_aux: Mapping[str, object] | None = None
    #: Per-epoch validation accuracy series parsed from the trainer's stdout.
    val_accuracy: Mapping[str, object] | None = None


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
        """Resolve *model*, re-resolving whenever the requested base changes.

        Two guards keep a promoted artifact -- a local directory of
        materialized weights -- off the Hub path, and they are independent:
        ``warm_start_revision`` is applied only to the *configured* stock repo
        id, so a directory is never pinned to the stock drafter's commit; and
        :meth:`_resolve` short-circuits any ``model`` that exists on disk, so no
        directory is ever handed to snapshot resolution in the first place.
        """
        if self.state.warm_start is None or self.state.warm_start_source != model:
            revision = (
                self.config.warm_start_revision
                if model == self.config.warm_start_model
                else None
            )
            self.state.warm_start, _ = self._resolve(
                model, revision, "warm-start model resolution", guard, scratch
            )
            self.state.warm_start_source = model
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
            counts = _render_speculators_dataset(
                snapshot,
                data,
                guard=guard,
                started=started,
                timeout=timeout_seconds,
                policy=self.config.truncated_row_policy,
                minimum_rows=self.config.min_rendered_rows,
                trust_untagged_assistant_messages=(
                    self.config.trust_untagged_assistant_messages
                ),
            )
            self.state.rendered_rows = counts.to_dict()
            row_count = self.config.row_count or counts.written
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
            if self.config.dilate_loss_mask_span_starts:
                dilation = self.runner.run(
                    [
                        str(self.config.training_python),
                        str(LOSS_MASK_DILATION_SCRIPT),
                        str(destination),
                    ],
                    cwd=self.config.speculators_repo,
                    env=_environment(self.config),
                    timeout_seconds=timeout_seconds,
                    should_abort=guard,
                )
                persist_stage_output(scratch, _MASK_DILATION_STAGE, dilation)
                _success("prepared loss-mask dilation", dilation)
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
        # The depth actually trained.  Taken from the training params -- which
        # ``Eagle3Config.effective_training_params`` fills from the profile's
        # serving ``num_speculative_tokens`` -- and passed explicitly, because
        # omitting the flag silently accepted Speculators' own default of 3.
        ttt_steps = _integer(
            "num_speculative_steps",
            training_params.get(
                "num_speculative_steps", self.config.num_speculative_steps
            ),
        )
        if ttt_steps < 1 or ttt_steps > MAX_SPECULATIVE_TOKENS:
            raise Eagle3Error(
                f"num_speculative_steps must be in 1..{MAX_SPECULATIVE_TOKENS}, "
                f"got {ttt_steps}"
            )
        # Same shape as the depth above, and for the same reason: the flag was
        # never passed, so the trainer used its own default and the manifest
        # could not say which weighting produced a gate result.
        decay = _fraction(
            "ttt_step_loss_decay",
            training_params.get(
                "ttt_step_loss_decay", self.config.ttt_step_loss_decay
            ),
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
            # Before launching, not after: a misaligned fc trains to completion
            # and publishes a checkpoint that looks exactly like a good one, so
            # the only useful place to catch it is ahead of the GPU hours.
            self.state.warm_start_aux = _check_warm_start_alignment(
                warm_start, verifier, layers
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
                    "--ttt-steps",
                    str(ttt_steps),
                    "--ttt-step-loss-decay",
                    repr(decay),
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
            # Recorded before the gate raises, so a cycle rejected for a flat
            # acceptance proxy still leaves the series that explains why.
            epochs = parse_val_accuracy_epochs(result.stdout)
            self.state.val_accuracy = summarize_val_accuracy(epochs)
            try:
                self.state.val_accuracy = _check_accuracy_improved(
                    epochs,
                    result.stderr,
                    required=self.config.require_accuracy_improvement,
                )
            except AccuracyRegressionError:
                self.state.val_accuracy = {
                    **summarize_val_accuracy(epochs),
                    "verdict": "not_improved",
                }
                raise
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
        state = self._state
        if state is None:
            return info
        params = dict(info.training_params)
        # Recorded whenever the stage that produces them ran, so the manifest
        # can distinguish "the check passed" from "the check never ran" --
        # a record written only on one outcome is unfalsifiable.
        if state.rendered_rows is not None:
            params["rendered_rows"] = dict(state.rendered_rows)
        if state.warm_start_aux is not None:
            params["warm_start_aux"] = dict(state.warm_start_aux)
        if state.val_accuracy is not None:
            params["val_accuracy"] = dict(state.val_accuracy)
        if state.verifier_pinned is not None:
            params["verifier_revision_satisfied"] = state.verifier_pinned
            if not state.verifier_pinned:
                params["verifier_revision"] = None
                params["verifier_revision_requested"] = self.config.verifier_revision
        if params == dict(info.training_params):
            return info
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
        warm_start_resolver: WarmStartResolver | None = None,
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
            num_speculative_steps=pipeline.num_speculative_steps,
            mask_policy=pipeline.mask_policy,
            training_params={
                "learning_rate": pipeline.learning_rate,
                "epochs": pipeline.epochs,
                "seed": pipeline.seed,
                # Lands beside the ``ttt_loss_reduction: "sum"`` that
                # ``Eagle3Config.effective_training_params`` already writes --
                # the two describe one objective, and "sum" alone does not say
                # how the summed terms were weighted.  Recorded here so a gate
                # result can be attributed to the weighting that produced it.
                "ttt_step_loss_decay": pipeline.ttt_step_loss_decay,
                "dilate_loss_mask_span_starts": (
                    pipeline.dilate_loss_mask_span_starts
                ),
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
            warm_start_resolver=warm_start_resolver,
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
    # Carried, not merely consulted -- the same argument as ``finish_reason``
    # below.  ``_normalize_conversation`` in the Speculators loader rebuilds
    # each turn from a fixed key set (role/content/tool_calls/tool_call_id/
    # thinking/reasoning_content), so this key is inert there; it exists so the
    # rendered artifact records *why* a row was admitted rather than only this
    # process's logs.
    provenance = turn.get("provenance_tag")
    if isinstance(provenance, str) and provenance:
        converted["provenance_tag"] = provenance
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
    # Carried, not merely consulted.  The filter below decides on it, but the
    # value also rides into the rendered dataset so the decision is auditable
    # from the artifact rather than only from this process's logs -- and so a
    # ``keep`` policy still leaves downstream tooling (``prepare_data.py``
    # ignores unknown keys; they become an unused dataset column) able to see
    # which rows were fragments.
    finish_reason = record.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        converted_record["finish_reason"] = finish_reason
    return converted_record


def _assistant_authorship(
    turn: Mapping[str, Any],
    *,
    trust_untagged: bool,
) -> bool:
    """Whether *this* deployment's verifier demonstrably produced *turn*.

    Only the exact tag ``"generated"`` establishes provider authorship; the
    gateway writes it on the response it assembled and ``"client_supplied"`` on
    everything that arrived in the request (``gateway/capture.py``).  An absent
    tag is *not* evidence of authorship -- it is a corpus that predates the tag
    -- so it fails closed unless an operator opts in.

    A prefill continuation is folded into one assistant turn carrying the
    generated tag, with ``prefill_prefix_chars`` recording how much of the
    merged content the *client* wrote (``traces/normalize.py``).  That prefix
    renders inside the assistant span the loss mask covers, so a non-zero (or
    unknown) prefix is not fully owned either.
    """
    tag = turn.get("provenance_tag")
    if tag is None:
        if not trust_untagged:
            return False
    elif tag != "generated":
        return False
    if "prefill_prefix_chars" not in turn:
        return True
    prefix = turn.get("prefill_prefix_chars")
    return isinstance(prefix, int) and not isinstance(prefix, bool) and prefix == 0


def _unowned_assistant_turns(
    record: Mapping[str, Any],
    *,
    trust_untagged: bool,
) -> int:
    """Count assistant turns in *record* that the verifier did not produce.

    Reads the *captured* messages rather than the converted ones so the answer
    does not depend on the conversion keeping any particular key.
    """
    turns = record.get("conversations")
    if not isinstance(turns, (list, tuple)):
        turns = record.get("messages")
    if not isinstance(turns, (list, tuple)):
        return 0
    unowned = 0
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        raw_role = turn.get("from", turn.get("role"))
        if not isinstance(raw_role, str):
            continue
        if _SPECULATORS_ROLES.get(raw_role) != "assistant":
            continue
        if not _assistant_authorship(turn, trust_untagged=trust_untagged):
            unowned += 1
    return unowned


def _is_truncated(record: Mapping[str, Any]) -> bool:
    """Whether *record*'s completion was cut off rather than finished."""
    value = record.get("finish_reason")
    return isinstance(value, str) and value.strip().lower() in TRUNCATED_FINISH_REASONS


#: Share of read rows the client-authorship filter may drop before the rendering
#: pass reports a diagnosis, even when enough rows survive to clear
#: ``min_rendered_rows``.
#:
#: The floors below only fire when this filter left too *few* rows outright.  A
#: large multi-turn or agentic corpus can lose most of its rows here and still
#: clear the floor, and what survives is exactly the single-turn slice the
#: gateway tagged end to end -- a biased remnant that trains, promotes and
#: reports like any other corpus.  One in five is well above the noise floor for
#: a corpus that is genuinely mostly this verifier's own single-turn traffic and
#: well below the "every multi-turn row is gone" case this exists to name.
CLIENT_SUPPLIED_DROP_ALERT_FRACTION: Final = 0.2


#: Every bucket a read row can end up in other than ``written``, with the
#: clause each one contributes to a failure message.
#:
#: Keyed by field name and consulted by :meth:`RenderedRowCounts.drop_accounting`
#: so that the two corpus-filter failures cannot report a partial account.  Job
#: 375414 is what a partial account costs: 409 rows read, 0 written, 36 dropped
#: by the truncation filter and 373 by the authorship filter, and the message
#: named only the 36 -- so the remedy it printed (raise the token cap) would
#: have changed nothing.  A drop bucket added to :class:`RenderedRowCounts`
#: without an entry here leaves ``read`` un-reconciled, which
#: ``test_every_drop_bucket_is_named_in_the_failure`` fails on.
_DROP_BUCKETS: Final = {
    "dropped_untrainable": "carried no trainable assistant turn",
    "dropped_truncated": "were truncated",
    "dropped_client_supplied": (
        "carried an assistant turn this verifier did not produce"
    ),
}


@dataclass(frozen=True, slots=True)
class RenderedRowCounts:
    """What one rendering pass read, kept and discarded."""

    read: int
    written: int
    dropped_untrainable: int
    dropped_truncated: int
    truncated_seen: int
    policy: TruncatedRowPolicy
    #: Rows dropped because an assistant turn was not this verifier's output.
    dropped_client_supplied: int = 0
    #: Assistant turns that were not this verifier's output, across all rows.
    client_supplied_turns_seen: int = 0

    @property
    def client_supplied_drop_fraction(self) -> float:
        """Share of read rows the client-authorship filter removed."""
        if not self.read:
            return 0.0
        return self.dropped_client_supplied / self.read

    def dominant_drop_bucket(self) -> str | None:
        """The one bucket that took strictly more rows than any other, if any.

        Considers *every* bucket in :data:`_DROP_BUCKETS`, which is the whole
        point.  The predecessor compared ``dropped_client_supplied`` against
        ``dropped_truncated`` and nothing else, so ``dropped_untrainable`` --
        a first-class reconciled bucket that ``drop_accounting`` has always
        printed -- could not win no matter how large it got.  On 90 untrainable,
        6 truncated, 4 client-supplied and 0 written, the caller raised
        :class:`TruncationFilteredCorpusError` and told the operator to raise the
        token cap: a remedy that would return 6 of the 100 missing rows, aimed at
        the smallest of the three causes.

        ``None`` when no bucket is strictly greatest.  A tie is not a dominant
        cause, and the predecessor resolved one by always answering
        "truncation", which is a coin flip wearing a diagnosis.  The caller
        reports the full accounting instead of naming a winner that does not
        exist -- see :class:`UnattributedCorpusShortfallError`.
        """
        counted = {name: int(getattr(self, name)) for name in _DROP_BUCKETS}
        largest = max(counted.values())
        if largest <= 0:
            return None
        leaders = [name for name, value in counted.items() if value == largest]
        if len(leaders) != 1:
            return None
        return leaders[0]

    def drop_accounting(self) -> str:
        """Account for every read row, whichever filter is being blamed.

        A failure message that names one filter's count invites the reader to
        treat that count as the whole loss, and the two filters run in the same
        pass over the same corpus.  Reconciling ``read`` against ``written``
        plus every bucket is what makes the dominant cause visible instead of
        inferred.
        """
        clauses = ", ".join(
            f"{getattr(self, name)} {clause}" for name, clause in _DROP_BUCKETS.items()
        )
        return (
            f"Of {self.read} rows read, {self.written} were written and the "
            f"rest were dropped: {clauses}"
        )

    def client_supplied_diagnosis(self) -> str:
        """Name the authorship filter, its counts and the knob that relaxes it.

        Shared by the hard failure and the survived-but-biased warning so an
        operator reads the same diagnosis either way.  Naming the counts is the
        point: "not enough rows" sends someone hunting for a capture problem,
        while "9 of 10 rows carried an assistant turn this verifier did not
        produce" points straight at the corpus and at the one knob that changes
        the outcome.
        """
        return (
            f"{self.dropped_client_supplied} of {self.read} rows "
            f"({self.client_supplied_drop_fraction:.1%}) carried an assistant "
            f"turn this verifier did not produce "
            f"({self.client_supplied_turns_seen} such turns in total) and were "
            f"dropped, leaving {self.written}. Multi-turn and agentic traffic "
            "replays earlier assistant turns in every request, so a corpus "
            "whose messages carry no provenance_tag loses every multi-turn row "
            "here. Set trust_untagged_assistant_messages=True if every "
            "assistant turn in the corpus is known to be this verifier's own "
            "output."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "read": self.read,
            "written": self.written,
            "dropped_untrainable": self.dropped_untrainable,
            "dropped_truncated": self.dropped_truncated,
            "truncated_seen": self.truncated_seen,
            "truncated_row_policy": self.policy.value,
            "dropped_client_supplied": self.dropped_client_supplied,
            "client_supplied_turns_seen": self.client_supplied_turns_seen,
            "client_supplied_drop_fraction": self.client_supplied_drop_fraction,
            # Carried into ``state.rendered_rows`` and from there into the cycle
            # report, so a corpus this filter reshaped is diagnosable from the
            # artifact and not only from this process's logs.
            "client_supplied_drop_diagnosis": (
                self.client_supplied_diagnosis()
                if self.dropped_client_supplied
                else None
            ),
        }


class TruncationFilteredCorpusError(Eagle3Error):
    """The truncation filter left too few rows to train on.

    Raised only when truncation is the *dominant* cause of the shortfall.  It
    used to be raised whenever the truncation filter had dropped anything at
    all, because it was checked first: on job 375414 that reported 36 truncated
    rows as the reason a 409-row corpus rendered nothing, when 373 rows had
    gone to the authorship filter, and the remedy it printed (raise the token
    cap) would have left the corpus exactly as empty.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        remedy = (
            "Training on the remainder would be a silently tiny corpus. Either "
            "raise the serving max_tokens cap so completions finish, lower "
            "min_rendered_rows, or set truncated_row_policy to 'keep' to accept "
            "training on truncated fragments."
        )
        if counts.dropped_client_supplied:
            # Truncation dominates here, but it is not the whole loss, and the
            # remedy above does nothing for the rows the other filter took.
            remedy += (
                f" The authorship filter also dropped "
                f"{counts.dropped_client_supplied} rows: "
                f"{counts.client_supplied_diagnosis()}"
            )
        super().__init__(
            f"dropping truncated rows left {counts.written} trainable rows from "
            f"{source}, below the floor of {minimum}: "
            f"{counts.dropped_truncated} of {counts.read} rows were truncated "
            f"(finish_reason in {sorted(TRUNCATED_FINISH_REASONS)}). "
            f"{counts.drop_accounting()}. {remedy}"
        )


class ClientSupervisedCorpusError(Eagle3Error):
    """The client-authorship filter left too few rows to train on.

    Raised when the authorship filter is the dominant cause of the shortfall,
    including when the truncation filter also removed rows -- that overlap used
    to be resolved in favour of truncation by check order alone.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        super().__init__(
            f"dropping rows with client-supplied assistant turns left "
            f"{counts.written} trainable rows from {source}, below the floor of "
            f"{minimum}: {counts.client_supplied_diagnosis()} "
            f"{counts.drop_accounting()}. "
            "Supervising those turns would teach the draft head to predict "
            "another model's outputs, so dropping the row is the only "
            "representable answer."
        )


class UnattributedCorpusShortfallError(Eagle3Error):
    """The corpus rendered too few rows and no single filter is to blame.

    The honest answer when :meth:`RenderedRowCounts.dominant_drop_bucket` finds
    no winner, or finds one that has no remedy of its own to offer.  Both
    sibling failures lead with a specific fix -- raise the token cap, set
    ``trust_untagged_assistant_messages`` -- and printing either of those for a
    shortfall it did not cause is worse than printing neither: an operator who
    follows it changes a setting, re-runs, and lands in exactly the same place
    having ruled nothing out.  Job 375414 was that failure with the buckets
    reversed.

    So this one names no cause and prescribes no knob.  It reconciles ``read``
    against ``written`` and every bucket and stops there, which is a complete
    statement of what is known.  It earns a class of its own rather than reusing
    a sibling precisely because the remedy paragraph is the part that must be
    absent.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        dominant = counts.dominant_drop_bucket()
        if dominant is None:
            cause = (
                "no single filter dominates the loss, so there is no one remedy "
                "to recommend"
            )
        else:
            cause = (
                f"the largest cause is that {counts.__getattribute__(dominant)} "
                f"rows {_DROP_BUCKETS[dominant]}, which no setting on this "
                "pipeline can recover"
            )
        super().__init__(
            f"rendering {source} produced {counts.written} trainable rows, below "
            f"the floor of {minimum}, and {cause}. {counts.drop_accounting()}. "
            "Fixing this needs a corpus whose rows carry trainable assistant "
            "turns, not a change to the filters that dropped them."
        )


def _render_speculators_dataset(
    snapshot: TraceSnapshot,
    destination: Path,
    *,
    guard: AbortCheck,
    started: float,
    timeout: float,
    policy: TruncatedRowPolicy = TruncatedRowPolicy.KEEP,
    minimum_rows: int = 1,
    trust_untagged_assistant_messages: bool = False,
) -> RenderedRowCounts:
    """Rewrite a leased trace snapshot into the Speculators loader contract.

    SpeedLM captures a top-level ``messages`` key while the Speculators loader
    reads a top-level ``conversations`` key, so handing the snapshot over verbatim
    builds an empty dataset without reporting an error. Records that carry no
    trainable assistant turn are dropped, and an entirely empty result fails loudly
    rather than reaching training as a silently empty dataset.

    Rows whose completion was truncated are dropped under
    :attr:`TruncatedRowPolicy.DROP`; see :data:`TRUNCATED_FINISH_REASONS` for
    why a truncated row is worse than a missing one.  *minimum_rows* is applied
    only when that filter actually removed something, so it bounds the damage
    this filter can do without imposing a floor on corpora that are small for
    unrelated reasons.

    Rows carrying an assistant turn this verifier did not produce are dropped
    outright.  The Speculators loader builds its loss mask by matching assistant
    spans in the *rendered* text and has no per-turn input, so there is no way
    to render a client-supplied assistant turn as context while withholding loss
    from it -- and the turn cannot simply be deleted either, because it is part
    of the prompt the verifier actually saw.  Dropping the row is therefore the
    only representable answer, and it is fail-closed.  *minimum_rows* bounds
    this filter's damage the same way it bounds the truncation filter's.
    """
    read = 0
    written = 0
    dropped_untrainable = 0
    dropped_truncated = 0
    truncated_seen = 0
    dropped_client_supplied = 0
    client_supplied_turns_seen = 0
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
                read += 1
                converted = _speculators_record(record)
                if converted is None:
                    dropped_untrainable += 1
                    continue
                unowned = _unowned_assistant_turns(
                    record, trust_untagged=trust_untagged_assistant_messages
                )
                if unowned:
                    client_supplied_turns_seen += unowned
                    dropped_client_supplied += 1
                    continue
                if _is_truncated(record):
                    truncated_seen += 1
                    if policy is TruncatedRowPolicy.DROP:
                        dropped_truncated += 1
                        continue
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1
    except OSError as error:
        raise Eagle3Error(
            f"cannot render Speculators rows from {snapshot.path}: {error}"
        ) from error
    counts = RenderedRowCounts(
        read=read,
        written=written,
        dropped_untrainable=dropped_untrainable,
        dropped_truncated=dropped_truncated,
        truncated_seen=truncated_seen,
        policy=policy,
        dropped_client_supplied=dropped_client_supplied,
        client_supplied_turns_seen=client_supplied_turns_seen,
    )
    logger.info(
        "rendered %d of %d captured rows (%d untrainable, %d truncated dropped, "
        "%d truncated seen, %d client-supplied dropped, %d client-supplied turns, "
        "policy=%s)",
        written,
        read,
        dropped_untrainable,
        dropped_truncated,
        truncated_seen,
        dropped_client_supplied,
        client_supplied_turns_seen,
        policy.value,
    )
    # The floors below fire only when this filter left too FEW rows.  A corpus
    # it merely reshaped clears them and then trains, promotes and reports like
    # any other corpus while every multi-turn row is gone.  Warn on the drop
    # FRACTION instead, carrying the same diagnosis the hard failure carries, so
    # the survivable case and the fatal case read alike.
    if counts.client_supplied_drop_fraction >= CLIENT_SUPPLIED_DROP_ALERT_FRACTION:
        logger.warning(
            "client-authorship filter dropped a material fraction of the "
            "corpus: %s",
            counts.client_supplied_diagnosis(),
        )
    # Checked before the empty-dataset error so a corpus a filter emptied is
    # reported as that filter's doing, not as the generic "nothing converted"
    # that predates the filters and names the wrong cause.
    #
    # Which filter is blamed is decided by which one dropped more rows, not by
    # which is checked first.  Fixed order made truncation the reported cause
    # of every shortfall it contributed anything to: job 375414 rendered 0 of
    # 409 rows with 36 truncated and 373 client-supplied, and was told to raise
    # the token cap.
    #
    # The two-way comparison that replaced that fixed order was the same bug one
    # bucket narrower.  It weighed ``dropped_client_supplied`` against
    # ``dropped_truncated`` and ignored ``dropped_untrainable`` entirely, so a
    # corpus losing 90 rows to untrainability, 6 to truncation and 4 to
    # authorship still reported truncation and still prescribed the token cap --
    # a remedy for 6 of the 100 missing rows.  And on an exact tie it always
    # answered truncation, where by construction neither cause dominates.
    #
    # So: attribute across ALL buckets, and only when one genuinely dominates.
    # A bucket with a dedicated remedy gets its own failure; anything else --
    # untrainability, which no knob can undo, or a tie, which has no single
    # cause -- gets the full accounting and no invented prescription.
    if written < minimum_rows and (
        dropped_untrainable or dropped_truncated or dropped_client_supplied
    ):
        dominant = counts.dominant_drop_bucket()
        if dominant == "dropped_client_supplied":
            raise ClientSupervisedCorpusError(snapshot.path, counts, minimum_rows)
        if dominant == "dropped_truncated":
            raise TruncationFilteredCorpusError(snapshot.path, counts, minimum_rows)
        # Nothing was written at all, so the long-standing "nothing converted"
        # failure is still the truest headline -- it now carries the accounting
        # behind it rather than leaving the breakdown unsaid.
        if not written:
            raise EmptySpeculatorsDatasetError(snapshot.path, counts)
        raise UnattributedCorpusShortfallError(snapshot.path, counts, minimum_rows)
    if not written:
        raise EmptySpeculatorsDatasetError(snapshot.path)
    return counts


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise Eagle3Error(f"{name} must be integer-valued")
    try:
        return int(value)
    except ValueError as error:
        raise Eagle3Error(f"{name} must be integer-valued") from error


def _fraction(name: str, value: object) -> float:
    """Parse a training-parameter value that must land in ``(0, 1]``.

    Deliberately does *not* clamp.  A clamp would turn a configuration mistake
    into a silently different training run, which is the exact failure the
    explicit ``--ttt-steps`` pass-through was added to prevent.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise Eagle3Error(f"{name} must be a number in (0, 1]")
    try:
        parsed = float(value)
    except ValueError as error:
        raise Eagle3Error(f"{name} must be a number in (0, 1]") from error
    if not 0 < parsed <= 1:
        raise Eagle3Error(f"{name} must be a number in (0, 1], got {parsed}")
    return parsed


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


class WarmStartLayerMismatchError(Eagle3Error):
    """The warm-start checkpoint would train on different aux layers."""


def _safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """Read tensor shapes from a ``safetensors`` file's header alone.

    The format is an 8-byte little-endian header length followed by that many
    bytes of JSON, so the shapes are readable without materializing any
    weights -- which matters because the file this is pointed at is measured in
    gigabytes and sits on the cycle's critical path.

    Returns an empty mapping when the file cannot be parsed as safetensors;
    callers treat that as "unavailable", never as "mismatched".
    """
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return {}
            length = int.from_bytes(prefix, "little")
            if not 0 < length <= _SAFETENSORS_HEADER_LIMIT:
                return {}
            raw = handle.read(length)
            if len(raw) != length:
                return {}
            header = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(header, Mapping):
        return {}
    shapes: dict[str, list[int]] = {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, Mapping):
            continue
        shape = entry.get("shape")
        if isinstance(shape, list) and all(isinstance(dim, int) for dim in shape):
            shapes[name] = list(shape)
    return shapes


def _checkpoint_aux_count(directory: Path, config: Mapping[str, Any]) -> int | None:
    """How many aux hidden states the checkpoint's ``fc`` was built to consume.

    EAGLE-3's ``fc`` projects the concatenated auxiliary hidden states down to
    one hidden size, so its input dimension *is* the aux count times the hidden
    size.  That number is baked into the weights and cannot be renegotiated by
    a flag, which makes it the one piece of the contract that is checkable
    without trusting any config field.

    ``None`` when it cannot be determined -- a sharded checkpoint, an
    unreadable header, an unexpected ``fc`` shape.
    """
    layer_config = config.get("transformer_layer_config")
    if not isinstance(layer_config, Mapping):
        return None
    hidden_size = layer_config.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        return None
    shape = _safetensors_shapes(directory / "model.safetensors").get("fc.weight")
    if shape is None or len(shape) != 2:
        return None
    inputs = shape[1]
    if inputs <= 0 or inputs % hidden_size:
        return None
    return inputs // hidden_size


def _verifier_hidden_layers(verifier: str) -> int | None:
    """``num_hidden_layers`` from a resolved verifier snapshot, if readable.

    Only a local directory is consulted.  A bare repo id would need the network
    to answer, and a guard that reaches the Hub is a guard that can fail for
    reasons unrelated to what it is guarding.
    """
    directory = Path(verifier)
    if not directory.is_dir():
        return None
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, Mapping):
        return None
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and "num_hidden_layers" in text_config:
        config = text_config
    layers = config.get("num_hidden_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        return None
    return layers


def _speculators_default_aux_ids(num_hidden_layers: int) -> tuple[int, ...]:
    """The ids ``speculators.models.utils.resolve_target_layer_ids`` substitutes.

    Reproduced rather than imported because the vendored tree is read-only and
    is not on this process's import path; the formula is pinned by the guard's
    own tests, so a vendored change that moved it would surface as a failure
    here rather than as a silent divergence at training time.
    """
    return (2, num_hidden_layers // 2, num_hidden_layers - 3)


def _check_warm_start_alignment(
    warm_start: str,
    verifier: str,
    layers: tuple[int, ...],
) -> dict[str, object]:
    """Prove the trainer will consume the aux layers extraction produced.

    ``scripts/train.py`` calls ``model_class.from_pretrained(path, t2d=, d2t=)``
    on the ``--from-pretrained`` path and forwards nothing else, so
    ``--target-layer-ids`` -- the flag this backend passes, and the flag that
    decided which hidden states the extraction server emitted -- has no effect
    on what the trainer consumes.  ``Eagle3DraftModel.from_pretrained`` instead
    re-resolves the ids from the checkpoint's own config, and when that config
    declares none ``resolve_target_layer_ids`` substitutes ``[2, n//2, n-3]``
    with only a ``warnings.warn``.  A live run logged exactly that: "Setting
    target layers to [2, 12, 21]".

    Nothing raises when the two diverge.  The fc input is simply misaligned
    against the extracted states and the run completes.  Today it survives on
    the arithmetic coincidence that both sides compute ``[2, n//2, n-3]``, and
    ``tuning.compounding_warm_start`` defaults to true, so this is the default
    path, not an edge case.

    The other flags ``train.py`` drops on this path -- ``--num-layers``,
    ``--draft-arch``, ``--norm-before-fc``, ``--embed-requires-grad`` -- are
    *not* load-bearing for us: this backend never passes any of them, so there
    is no requested value for the checkpoint to silently override.  Their
    checkpoint values are recorded below rather than asserted, because they do
    describe the architecture that will train (``norm_before_fc`` is true on
    the gpt-oss drafter and false on the Qwen one) and a manifest that cannot
    say which one ran cannot explain a cross-model result.

    Returns the record to publish.  Raises
    :class:`WarmStartLayerMismatchError` when divergence is proven.
    """
    record: dict[str, object] = {"extracted_layer_ids": list(layers)}
    directory = Path(warm_start)
    config_path = directory / "config.json"
    if not directory.is_dir() or not config_path.is_file():
        # A bare repo id: nothing on disk to read, and reaching the Hub here
        # would let a network fault fail a cycle that is otherwise fine.
        record["verdict"] = "unverified_no_local_checkpoint"
        logger.warning(
            "warm start %s is not a local checkpoint; cannot prove the trainer "
            "will use the extracted aux layer ids %s",
            warm_start,
            list(layers),
        )
        return record
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WarmStartLayerMismatchError(
            f"warm-start config is unreadable: {config_path}: {error}"
        ) from error
    if not isinstance(config, Mapping):
        raise WarmStartLayerMismatchError(
            f"warm-start config is not a JSON object: {config_path}"
        )

    for name in ("norm_before_fc", "embed_requires_grad", "norm_before_residual"):
        if name in config:
            record[name] = config[name]
    layer_config = config.get("transformer_layer_config")
    if isinstance(layer_config, Mapping) and "num_hidden_layers" in layer_config:
        record["num_layers"] = layer_config["num_hidden_layers"]

    declared = config.get(_AUX_LAYER_IDS_KEY)
    aux_count = _checkpoint_aux_count(directory, config)
    if aux_count is not None:
        record["checkpoint_aux_count"] = aux_count

    if isinstance(declared, (list, tuple)) and all(
        not isinstance(value, bool) and isinstance(value, int) for value in declared
    ):
        checkpoint_ids = tuple(int(value) for value in declared)
        record["checkpoint_layer_ids"] = list(checkpoint_ids)
        record["verdict"] = "checkpoint_declared"
        if checkpoint_ids != layers:
            raise WarmStartLayerMismatchError(
                "warm-start aux layer ids do not match the ids this cycle "
                f"extracted with: checkpoint {list(checkpoint_ids)} != extracted "
                f"{list(layers)} ({config_path}). scripts/train.py drops "
                "--target-layer-ids on the --from-pretrained path, so training "
                "would silently use the checkpoint's "
                f"{list(checkpoint_ids)} against hidden states computed for "
                f"{list(layers)}, misaligning the fc input with no error raised."
            )
    elif declared is not None:
        raise WarmStartLayerMismatchError(
            f"warm-start config has a malformed {_AUX_LAYER_IDS_KEY}: "
            f"{declared!r} ({config_path})"
        )
    else:
        record["checkpoint_layer_ids"] = None
        num_hidden_layers = _verifier_hidden_layers(verifier)
        if num_hidden_layers is None:
            record["verdict"] = "unverified_no_verifier_config"
            logger.warning(
                "warm start %s declares no %s and the verifier's layer count is "
                "not readable from %s; cannot prove the trainer will use the "
                "extracted aux layer ids %s",
                warm_start,
                _AUX_LAYER_IDS_KEY,
                verifier,
                list(layers),
            )
        else:
            derived = _speculators_default_aux_ids(num_hidden_layers)
            record["derived_layer_ids"] = list(derived)
            record["verifier_num_hidden_layers"] = num_hidden_layers
            record["verdict"] = "derived_from_verifier"
            if derived != layers:
                raise WarmStartLayerMismatchError(
                    "warm-start aux layer ids do not match the ids this cycle "
                    f"extracted with: checkpoint {list(derived)} != extracted "
                    f"{list(layers)} ({config_path}). The checkpoint declares no "
                    f"{_AUX_LAYER_IDS_KEY}, so speculators' "
                    "resolve_target_layer_ids substitutes [2, n//2, n-3] from the "
                    f"verifier's {num_hidden_layers} layers, and scripts/train.py "
                    "drops --target-layer-ids on the --from-pretrained path -- so "
                    f"training would silently use the checkpoint's {list(derived)} "
                    f"against hidden states computed for {list(layers)}, "
                    "misaligning the fc input with no error raised."
                )

    if aux_count is not None and aux_count != len(layers):
        raise WarmStartLayerMismatchError(
            f"warm-start checkpoint's fc consumes {aux_count} auxiliary hidden "
            f"states but this cycle extracted {len(layers)} ({list(layers)}); "
            f"scripts/train.py drops --target-layer-ids on the --from-pretrained "
            f"path, so training would silently use the checkpoint's arity "
            f"({config_path})."
        )
    return record


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


class AccuracyRegressionError(TrainingError):
    """Training's acceptance proxy did not improve across epochs."""


def parse_val_accuracy_epochs(stdout: str) -> list[dict[str, float]]:
    """Per-epoch validation metrics, in the order the trainer logged them.

    Each entry maps a bare metric name (``full_acc_0``, ``cond_acc_1``,
    ``loss_2``, ``loss``) to its value; the trainer's ``_epoch`` suffix is
    stripped, and the epoch index the record reported is stored under
    ``"epoch"``.

    Robust to Rich's 80-column wrapping by matching whole records with DOTALL
    rather than scanning line by line -- see :data:`_VAL_EPOCH_RECORD`.  Records
    spanning the elision marker :func:`_persist_streams` inserts into an
    oversized log are skipped rather than parsed across the hole.
    """
    text = _LOG_SOURCE_TAG.sub("", stdout)
    epochs: list[dict[str, float]] = []
    for match in _VAL_EPOCH_RECORD.finditer(text):
        span = match.group(0)
        if "elided" in span:
            continue
        metrics: dict[str, float] = {}
        for name, value in _VAL_EPOCH_METRIC.findall(span):
            try:
                metrics[name] = float(value)
            except ValueError:  # pragma: no cover - regex admits only floats
                continue
        if not metrics:
            continue
        metrics["epoch"] = float(match.group(1))
        epochs.append(metrics)
    return epochs


def summarize_val_accuracy(epochs: Sequence[Mapping[str, float]]) -> dict[str, object]:
    """Condense a parsed epoch series into the record the manifest carries."""
    key = ACCEPTANCE_METRIC.removesuffix("_epoch")
    summary: dict[str, object] = {
        "metric": ACCEPTANCE_METRIC,
        "epochs": [dict(sorted(entry.items())) for entry in epochs],
    }
    series = [entry[key] for entry in epochs if key in entry]
    if series:
        summary["first"] = series[0]
        summary["final"] = series[-1]
        summary["delta"] = series[-1] - series[0]
        summary["series"] = series
    return summary


def _check_accuracy_improved(
    epochs: Sequence[Mapping[str, float]],
    stderr: str,
    *,
    required: bool,
) -> dict[str, object]:
    """Gate the cycle on the acceptance proxy rather than on the loss.

    The trainer's summed TTT loss falls across epochs while
    :data:`ACCEPTANCE_METRIC` sits flat, so a cycle gated on ``val_loss``
    promotes a checkpoint whose step-0 acceptance did not move.  This is that
    gate.  Silent when fewer than two epochs were logged -- there is nothing to
    compare a single epoch against, and inventing a verdict there would be the
    kind of check that cannot fail.
    """
    summary = summarize_val_accuracy(epochs)
    series = summary.get("series")
    if not isinstance(series, list) or len(series) < 2:
        summary["verdict"] = "not_evaluated"
        return summary
    improved = series[-1] > series[0]
    summary["verdict"] = "improved" if improved else "not_improved"
    if improved or not required:
        return summary
    raise AccuracyRegressionError(
        f"{ACCEPTANCE_METRIC} did not improve across training: "
        f"{series[0]} -> {series[-1]} over {len(series)} epochs (series "
        f"{series}). This is the drafter's top-1 at TTT step 0 and the direct "
        "proxy for step-0 acceptance; the summed TTT loss can fall while it "
        "stays flat, because the sum is dominated by the later steps. Refusing "
        "to promote a checkpoint that did not move it. Set "
        "require_accuracy_improvement to false to train anyway.",
        stderr=stderr,
    )


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
    "WarmStartResolver",
    "scratch_usage",
]
