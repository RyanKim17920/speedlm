"""Injectable GPT-OSS EAGLE-3 training pipeline.

This module owns validation, quotas, abort propagation, and the
``--from-pretrained`` contract. GPU/process mechanics remain behind protocols so
the login-node test suite never imports CUDA, vLLM, or Speculators.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import struct
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

from speedlm.profiles import (
    DEFAULT_SPECULATIVE_TOKENS,
    MAX_SPECULATIVE_TOKENS,
    cached_snapshot_dir,
)
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


class DraftWeightsError(Eagle3Error):
    """A materialized or published draft does not carry usable trained weights."""


class StockIdenticalDraftError(DraftWeightsError):
    """The candidate's weights are byte-identical to the head it trained from.

    Training was a no-op: every tensor came out of the cycle exactly as it
    went in.  Benchmarking such a candidate cannot do anything except spend a
    gate cycle proving that a head is as good as itself, so the cycle fails
    here instead.
    """

    def __init__(self, baseline: str, fingerprint: str) -> None:
        self.baseline = baseline
        self.fingerprint = fingerprint
        super().__init__(
            f"trained draft is byte-identical to its baseline {baseline!r} "
            f"(weight fingerprint {fingerprint}); training was a no-op"
        )


class NoOpTrainingDeltaError(DraftWeightsError):
    """The candidate differs from its baseline only by bf16 re-rounding.

    :class:`StockIdenticalDraftError` catches the case where *no byte* moved.
    This catches the case that actually happened: every byte-identity check
    passed, the fingerprints differed, and the head was still statistically
    the head it started from.  See :data:`MIN_TRAINED_RELATIVE_DELTA` for the
    measurements and the derivation.

    The per-tensor deltas travel on the exception *and* in its message,
    because the alternative -- re-deriving them -- means re-running a whole
    GPU cycle to find out why one failed.
    """

    def __init__(
        self,
        baseline: str,
        deltas: Mapping[str, float],
        *,
        threshold: float,
        norm_epsilon: float,
    ) -> None:
        self.baseline = baseline
        #: Per-tensor relative Frobenius delta, plain floats for the manifest.
        self.deltas = dict(deltas)
        self.max_delta = max(self.deltas.values(), default=0.0)
        self.threshold = threshold
        self.norm_epsilon = norm_epsilon
        listing = ", ".join(
            f"{name}={value:.6f}"
            for name, value in sorted(
                self.deltas.items(), key=lambda item: (-item[1], item[0])
            )
        )
        super().__init__(
            f"trained draft is statistically identical to its baseline {baseline!r}: "
            f"largest relative Frobenius delta {self.max_delta:.6f} is below the "
            f"trained-head floor {threshold:.6f}, and every norm-type tensor moved "
            f"at most {norm_epsilon:.6f}; training moved weights by at most one bf16 "
            f"ULP, which is dither, not learning. Per-tensor deltas: {listing}"
        )


#: Tensor keys every trained EAGLE-3 head must publish.
#:
#: Read off the two stock drafters in the local HF cache rather than assumed:
#: ``RedHatAI/gpt-oss-20b-speculator.eagle3`` publishes 17 tensors and
#: ``RedHatAI/Qwen3-8B-speculator.eagle3`` publishes 16.  This is their
#: intersection.  gpt-oss additionally carries ``input_norm.weight`` because
#: its config sets ``norm_before_fc``; Qwen3 does not, so requiring that key
#: would reject a valid Qwen3 head, and it is deliberately excluded.
#:
#: ``fc.weight`` is what makes this a check on *training* rather than on
#: packaging.  It is the fusion projection from the concatenated aux hidden
#: states into the draft layer -- ``hidden x (aux_count * hidden)``, so
#: ``(2880, 8640)`` for gpt-oss and ``(4096, 12288)`` for Qwen3 -- and it is
#: the tensor EAGLE-3 training actually fits.  The previous validator required
#: only ``d2t``/``t2d``, two vocabulary index maps that a directory holding no
#: draft head at all would still satisfy.
REQUIRED_DRAFT_TENSORS: Final[frozenset[str]] = frozenset(
    {
        "d2t",
        "t2d",
        "fc.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
    }
)

#: Domain separator for :func:`weight_fingerprint`, versioned so a future
#: change to what the digest covers can never be mistaken for a weight change.
_FINGERPRINT_DOMAIN: Final = b"speedlm-eagle3-weights-v1"

#: Ceiling on a safetensors JSON header, which is read whole into memory.
#:
#: The format is an 8-byte little-endian header length followed by that many
#: bytes of JSON.  A corrupt or hostile file can claim an arbitrary length, so
#: the claim is bounded before it is honoured.  Real headers here are a few
#: kilobytes: the largest of the two stock drafters declares 17 tensors.
_MAX_SAFETENSORS_HEADER_BYTES: Final = 100 * 1024 * 1024

#: Read granularity when digesting tensor payloads.
_FINGERPRINT_CHUNK_BYTES: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _TensorLocation:
    """Where one named tensor's bytes live, and what shape they claim to be."""

    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def _safetensors_tensors(path: Path) -> dict[str, _TensorLocation]:
    """Parse *path*'s safetensors header without importing ``safetensors``.

    The tuner process is deliberately free of the GPU stack -- it must stay
    importable on the login node -- so the header is decoded from its
    documented on-disk layout instead: ``u64`` header length, that many bytes
    of JSON, then the tensor buffer.  ``data_offsets`` are relative to the
    start of that buffer.
    """
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise DraftWeightsError(f"truncated safetensors file: {path}")
            header_bytes = int.from_bytes(prefix, "little")
            if header_bytes <= 0 or header_bytes > _MAX_SAFETENSORS_HEADER_BYTES:
                raise DraftWeightsError(
                    f"safetensors header length {header_bytes} is not plausible: {path}"
                )
            raw = stream.read(header_bytes)
            if len(raw) != header_bytes:
                raise DraftWeightsError(f"truncated safetensors header: {path}")
    except OSError as exc:
        raise DraftWeightsError(f"cannot read safetensors file: {path}") from exc
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DraftWeightsError(f"unreadable safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise DraftWeightsError(f"safetensors header is not an object: {path}")
    base = 8 + header_bytes
    tensors: dict[str, _TensorLocation] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise DraftWeightsError(f"safetensors entry {name!r} is not an object: {path}")
        offsets = entry.get("data_offsets")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in shape)
        ):
            raise DraftWeightsError(
                f"safetensors entry {name!r} has an unusable descriptor: {path}"
            )
        tensors[name] = _TensorLocation(
            path=path,
            dtype=dtype,
            shape=tuple(shape),
            start=base + offsets[0],
            end=base + offsets[1],
        )
    return tensors


def _collect_draft_tensors(directory: Path) -> dict[str, _TensorLocation]:
    """Index every tensor across *directory*'s safetensors shards, by name."""
    if not directory.is_dir():
        raise DraftWeightsError(f"draft directory does not exist: {directory}")
    collected: dict[str, _TensorLocation] = {}
    for path in sorted(directory.glob("*.safetensors")):
        for name, location in _safetensors_tensors(path).items():
            previous = collected.get(name)
            if previous is not None:
                raise DraftWeightsError(
                    f"tensor {name!r} appears in both {previous.path.name} and "
                    f"{path.name}; the draft's shard layout is ambiguous"
                )
            collected[name] = location
    return collected


def draft_tensor_keys(directory: Path) -> frozenset[str]:
    """Return every tensor name published by *directory*'s safetensors."""
    return frozenset(_collect_draft_tensors(directory))


def weight_fingerprint(directory: Path) -> str:
    """Return a SHA-256 fingerprint of *directory*'s draft weights.

    The digest covers each tensor's name, dtype, shape and raw bytes, walked
    in sorted name order across all shards.  It deliberately does **not**
    cover file names, shard boundaries, config or tokenizer files: two
    directories with the same fingerprint hold the same weights however they
    were packaged, which is what makes the comparison against the warm-start
    head in :meth:`Eagle3Adapter.materialize` meaningful.

    This is a *different* question from
    :func:`speedlm.tuner.artifacts.hash_directory`, which hashes every byte of
    every file and answers "is this the same publication".  A tuning cycle
    can change the config and leave the weights untouched; only this digest
    can tell that apart.
    """
    tensors = _collect_draft_tensors(directory)
    if not tensors:
        raise DraftWeightsError(f"draft directory has no safetensors weights: {directory}")
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_DOMAIN)
    for name in sorted(tensors):
        location = tensors[name]
        digest.update(b"tensor\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(location.dtype.encode("utf-8"))
        digest.update(b"\0")
        digest.update(",".join(str(value) for value in location.shape).encode("utf-8"))
        digest.update(b"\0")
        remaining = location.end - location.start
        try:
            with location.path.open("rb") as stream:
                stream.seek(location.start)
                while remaining > 0:
                    chunk = stream.read(min(remaining, _FINGERPRINT_CHUNK_BYTES))
                    if not chunk:
                        raise DraftWeightsError(
                            f"tensor {name!r} is truncated in {location.path}"
                        )
                    digest.update(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            raise DraftWeightsError(f"cannot read tensor {name!r}: {location.path}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


def drafter_aux_count(directory: Path) -> int:
    """Return how many aux hidden states *directory*'s drafter expects.

    EAGLE-3 fuses the verifier's aux hidden states with ``fc.weight``, whose
    shape is ``(hidden, aux_count * hidden)``.  The count is therefore
    ``shape[1] // shape[0]`` and is a *property of the checkpoint*, not of
    whatever the caller believes it configured.  Verified against both stock
    drafters in the local HF cache: gpt-oss-20b's ``fc.weight`` is
    ``(2880, 8640)`` and Qwen3-8B's is ``(4096, 12288)``; both give 3.

    Only the safetensors header is read -- a few kilobytes -- so this answers
    on the login node, before a GPU exists, which is the whole point: the
    alternative is discovering the arity mismatch as a shape error several
    hundred GPU-seconds into a forward pass.

    Raises:
        DraftWeightsError: if ``fc.weight`` is absent, is not 2-D, or has a
            second dimension that is not a whole multiple of the first.  None
            of those is a *count*; they are a malformed or non-EAGLE-3 head,
            and reporting a floor-divided guess for them would launder a
            structural problem into a plausible-looking integer.
    """
    tensors = _collect_draft_tensors(directory)
    location = tensors.get("fc.weight")
    if location is None:
        raise DraftWeightsError(
            f"draft directory publishes no fc.weight, so its aux-layer count is "
            f"undefined: {directory}"
        )
    if len(location.shape) != 2:
        raise DraftWeightsError(
            f"fc.weight must be 2-D to imply an aux-layer count, got shape "
            f"{list(location.shape)}: {directory}"
        )
    hidden, fused = location.shape
    if hidden <= 0 or fused <= 0 or fused % hidden:
        raise DraftWeightsError(
            f"fc.weight shape {list(location.shape)} is not "
            f"(hidden, aux_count * hidden); this is a malformed EAGLE-3 head, "
            f"not an aux-layer count: {directory}"
        )
    return fused // hidden


#: Relative spacing of the bf16 grid, i.e. one ULP as a fraction of the value.
#:
#: bf16 keeps 8 significand bits (7 stored).  For ``x`` in the binade
#: ``[2^e, 2^(e+1))`` the ULP is ``2^(e-7)``, so the *relative* spacing lies in
#: ``(2^-8, 2^-7] = (0.00391, 0.00781]``.  Re-rounding every element of a
#: tensor by at most one ULP therefore cannot produce a relative Frobenius
#: delta above ``2^-7``, whatever fraction of the elements moves.
BF16_RELATIVE_ULP: Final = 2.0**-7

#: Smallest relative Frobenius delta that counts as "this head trained".
#:
#: Not a taste call.  Three real candidates were measured against the stock
#: warm-start snapshot they trained from, per tensor, as
#: ``||cand - base||_F / ||base||_F`` in float64 after widening bf16::
#:
#:     run                 max delta   where
#:     gptoss-pinned-2     0.005818    layers.0.self_attn.v_proj.weight
#:     full-gptoss-head    0.005810    layers.0.self_attn.v_proj.weight
#:     qwen-pinned-2       0.006090    layers.0.self_attn.k_proj.weight
#:
#: with the same signature in all three: only ~8 % of elements changed at all
#: -- the ones sitting on a rounding boundary -- and every norm-type tensor was
#: either exactly zero or ~1e-5.  For gptoss-pinned-2, ``fc.weight`` moved
#: 1,994,546 of 24,883,200 elements for a delta of 0.003153, while
#: ``embed_tokens.weight``, ``lm_head.weight``, ``norm.weight`` and both
#: layer norms moved zero elements out of hundreds of millions.  Every one of
#: those numbers sits inside the ``2^-7`` dither envelope above: the cycles
#: had re-rounded the warm-start weights and changed nothing else, yet the
#: byte-identity guard passed them because the bytes *did* differ.
#:
#: The floor is ``2 x 2^-7 = 0.015625``: 2.57x the largest observed dither.
#: Be clear about what is *not* behind that factor of two -- no genuinely
#: trained candidate was available to calibrate the upper side, so nothing
#: here measures how far a head that actually learned something moves.  The
#: constant is derived from the dither envelope plus an explicit margin, and
#: it is a module-level name rather than an inline literal precisely so a run
#: that finds real training landing under it can raise it without a patch.
MIN_TRAINED_RELATIVE_DELTA: float = 2.0 * BF16_RELATIVE_ULP

#: Relative delta below which a norm-type tensor counts as "did not move".
#:
#: The RMSNorm gains are the second signal, and they are the sharper one: a
#: re-rounding pass leaves them alone (all three measured runs had every norm
#: at 0.000000 except ``input_norm.weight`` at 0.000028/0.000030, from 37 of
#: 8,640 elements, and qwen's ``layers.0.input_layernorm.weight`` at 0.000010
#: from a *single* element of 4,096), whereas a head that actually fits data
#: moves its gains materially.
#:
#: Note what that qwen number rules out: a strict "norms are EXACTLY zero"
#: test would already have failed to fire on qwen-pinned-2, because one
#: element moved.  Hence an epsilon and not an equality.  1e-4 is ~3.3x the
#: largest observed non-zero norm delta (3.0e-5) and still far below anything
#: a trained gain vector would show.
NORM_MOVED_EPSILON: float = 1.0e-4

#: Float dtypes the magnitude analysis understands, and their item sizes.
#:
#: Integer and boolean dtypes are deliberately absent rather than defaulted.
#: The integer tensors an EAGLE-3 head publishes are ``d2t``/``t2d``, the
#: reduced-vocabulary index maps -- a Frobenius norm over token *indices* is
#: not a magnitude, it is arithmetic on labels, and its "relative delta" would
#: be a meaningless number that still voted in the max.  They are excluded
#: from the report and reported as skips instead.
_DELTA_FLOAT_DTYPES: Final[dict[str, int]] = {"BF16": 2, "F16": 2, "F32": 4}

#: Elements decoded per chunk when differencing a tensor pair.
#:
#: The whole pass is ~25 s over a 1.2 GB gpt-oss head and runs once per tuning
#: cycle, against cycles measured in thousands of seconds.  A million elements
#: is ~8 MB of CPython floats per side, which keeps the peak bounded while
#: leaving the per-chunk work large enough that the C-speed primitives
#: (:func:`struct.unpack`, :func:`map`, :func:`math.sumprod`) dominate.
_DELTA_CHUNK_ELEMENTS: Final = 1 << 20


def _bf16_to_floats(raw: bytes) -> tuple[float, ...]:
    """Decode little-endian bf16 *raw* into floats, without numpy or torch.

    A bf16 bit pattern is exactly the top 16 bits of the float32 with the same
    value -- same sign, same 8-bit exponent, mantissa truncated from 23 bits to
    7 -- so widening is a byte move, not arithmetic: drop each bf16 pair into
    the *high* half of a little-endian float32 and zero the low half.  Slice
    assignment on a ``bytearray`` does that at C speed, and
    :func:`struct.unpack` then reads the result in one call.

    This is why the tuner can do magnitude analysis at all while staying
    importable on a login node with neither torch nor numpy installed.
    """
    if len(raw) % 2:
        raise DraftWeightsError("bf16 payload is not a whole number of elements")
    widened = bytearray(2 * len(raw))
    widened[2::4] = raw[0::2]
    widened[3::4] = raw[1::2]
    return struct.unpack(f"<{len(raw) // 2}f", bytes(widened))


def _decode_floats(dtype: str, raw: bytes) -> tuple[float, ...]:
    """Decode *raw* according to *dtype*, which must be a float dtype."""
    if dtype == "BF16":
        return _bf16_to_floats(raw)
    itemsize = _DELTA_FLOAT_DTYPES.get(dtype)
    if itemsize is None:
        raise DraftWeightsError(f"dtype {dtype!r} is not part of the magnitude analysis")
    code = "f" if dtype == "F32" else "e"
    return struct.unpack(f"<{len(raw) // itemsize}{code}", raw)


def _tensor_chunks(location: _TensorLocation) -> Iterator[tuple[float, ...]]:
    """Yield *location*'s values in fixed-size chunks, widened to float."""
    itemsize = _DELTA_FLOAT_DTYPES[location.dtype]
    count = math.prod(location.shape)
    span = location.end - location.start
    if count * itemsize != span:
        raise DraftWeightsError(
            f"tensor at {location.path} declares shape {list(location.shape)} of "
            f"{location.dtype} ({count * itemsize} bytes) but occupies {span}"
        )
    try:
        with location.path.open("rb") as stream:
            stream.seek(location.start)
            remaining = count
            while remaining > 0:
                wanted = min(remaining, _DELTA_CHUNK_ELEMENTS) * itemsize
                buffer = bytearray()
                while len(buffer) < wanted:
                    piece = stream.read(wanted - len(buffer))
                    if not piece:
                        raise DraftWeightsError(f"tensor is truncated in {location.path}")
                    buffer.extend(piece)
                yield _decode_floats(location.dtype, bytes(buffer))
                remaining -= wanted // itemsize
    except OSError as exc:
        raise DraftWeightsError(f"cannot read tensor bytes: {location.path}") from exc


def _relative_delta(candidate: _TensorLocation, baseline: _TensorLocation) -> float:
    """Return ``||cand - base||_F / ||base||_F`` for one tensor pair.

    Accumulated in CPython floats (C doubles) across chunks, so widening bf16
    to float64 costs nothing and the sum never rounds in the input precision.
    ``sumprod`` is the whole reason this is affordable in pure Python: the
    per-element work is one C subtraction and one fused multiply-add.
    """
    delta_square = 0.0
    baseline_square = 0.0
    for candidate_values, baseline_values in zip(
        _tensor_chunks(candidate), _tensor_chunks(baseline), strict=True
    ):
        difference = list(map(operator.sub, candidate_values, baseline_values))
        delta_square += math.sumprod(difference, difference)
        baseline_square += math.sumprod(baseline_values, baseline_values)
    if baseline_square == 0.0:
        # An all-zero baseline has no scale to be relative to.  Saying "0"
        # when the candidate is also all zeros is exact; saying "infinity"
        # otherwise refuses to divide by nothing and still votes "moved".
        return 0.0 if delta_square == 0.0 else math.inf
    return math.sqrt(delta_square) / math.sqrt(baseline_square)


@dataclass(frozen=True, slots=True)
class WeightDeltaReport:
    """Per-tensor magnitude comparison of a candidate head against a baseline."""

    #: Tensor name -> relative Frobenius delta, over float tensors that both
    #: directories publish with the same dtype and shape.
    deltas: dict[str, float]
    #: Tensor name -> why it was left out.  Skips are *recorded*, never
    #: silent: a tensor that vanished, changed dtype or changed shape between
    #: the two heads is the single most interesting thing a report can hold,
    #: and dropping it on the floor is how a comparison ends up quietly
    #: covering three tensors out of seventeen.
    skipped: dict[str, str]

    @property
    def max_delta(self) -> float:
        """Largest relative delta observed, or ``0.0`` if nothing compared."""
        return max(self.deltas.values(), default=0.0)


def weight_delta_report(candidate: Path, baseline: Path) -> WeightDeltaReport:
    """Compare *candidate*'s weights against *baseline*'s, tensor by tensor.

    Both members of :class:`WeightDeltaReport` are plain ``dict``s of ``str``
    to JSON-native values, so the whole thing lands in the artifact manifest
    unchanged -- a passing cycle is then as diagnosable as a failing one,
    which matters because the interesting question after this guard ships is
    "how far *does* a real cycle move", and only recorded passes can answer it.
    """
    candidate_tensors = _collect_draft_tensors(candidate)
    baseline_tensors = _collect_draft_tensors(baseline)
    deltas: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in sorted(set(candidate_tensors) | set(baseline_tensors)):
        left = candidate_tensors.get(name)
        right = baseline_tensors.get(name)
        if left is None:
            skipped[name] = "absent from the candidate"
            continue
        if right is None:
            skipped[name] = "absent from the baseline"
            continue
        if left.dtype != right.dtype:
            skipped[name] = f"dtype {left.dtype} vs baseline {right.dtype}"
            continue
        if left.shape != right.shape:
            skipped[name] = (
                f"shape {list(left.shape)} vs baseline {list(right.shape)}"
            )
            continue
        if left.dtype not in _DELTA_FLOAT_DTYPES:
            skipped[name] = f"{left.dtype} is not a float dtype"
            continue
        deltas[name] = _relative_delta(left, right)
    return WeightDeltaReport(deltas=deltas, skipped=skipped)


def is_noop_training_delta(report: WeightDeltaReport) -> bool:
    """Is *report* the signature of a cycle that only re-rounded its input?

    Two signals, and deliberately their conjunction:

    (a) every norm-type tensor is at or below :data:`NORM_MOVED_EPSILON` --
        the gains did not move; and
    (b) no tensor at all reaches :data:`MIN_TRAINED_RELATIVE_DELTA` -- nothing
        moved further than one bf16 ULP could carry it.

    Either alone is a worse test.  (b) alone would reject a genuine but small
    update; (a) alone would reject a real head that happens to leave its
    RMSNorm gains where they were, which is an ordinary outcome and not a
    no-op.  Requiring both means the guard fires only on the shape all three
    measured failures actually had.
    """
    if not report.deltas:
        return False
    norms_still = all(
        value <= NORM_MOVED_EPSILON
        for name, value in report.deltas.items()
        if "norm" in name
    )
    nothing_moved = all(
        value < MIN_TRAINED_RELATIVE_DELTA for value in report.deltas.values()
    )
    return norms_still and nothing_moved


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
    #: Depth of the draft chain this cycle trains, i.e. Speculators'
    #: ``--ttt-steps``.  It must equal the profile's serving
    #: ``num_speculative_tokens``; :func:`speedlm.profiles.validate_training_depth`
    #: is what enforces that, at composition time, before a GPU cycle starts.
    num_speculative_steps: int = DEFAULT_SPECULATIVE_TOKENS
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
        # Formerly ``!= 3``, which was an assumption rather than a constraint.
        # The Speculators EAGLE-3 head is one transformer layer rolled out
        # autoregressively -- ``ttt_steps`` is the loop bound at
        # ``src/speculators/models/eagle3/core.py:199``, its upstream suite
        # drives 1, 3 and 5 steps against a single checkpoint, and the trainer
        # merely *defaults* ``--ttt-steps`` to 3.  Pinning 3 here is what let
        # gpt-oss-20b train a 3-deep head and serve it 5-deep.
        if (
            isinstance(self.num_speculative_steps, bool)
            or not isinstance(self.num_speculative_steps, int)
            or self.num_speculative_steps < 1
            or self.num_speculative_steps > MAX_SPECULATIVE_TOKENS
        ):
            raise ValueError(
                f"num_speculative_steps must be an integer in "
                f"1..{MAX_SPECULATIVE_TOKENS}"
            )
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

    #: Weight fingerprint of the draft this cycle materialized, or ``None``
    #: before :meth:`materialize` has run.  Class-level for the same reason as
    #: the two attributes above: ``describe`` must answer on an adapter that a
    #: test assembled without ``__init__``.
    _draft_weight_fingerprint: str | None = None

    #: What :meth:`materialize` compared that fingerprint against, and the
    #: baseline's own fingerprint.  Both are ``None`` when no baseline could be
    #: resolved to local weights.  They are recorded either way: an artifact
    #: whose manifest cannot say *whether* the no-op check ran is
    #: indistinguishable from one where it ran and passed, which is the exact
    #: unfalsifiability ``verifier_revision_satisfied`` exists to avoid.
    _draft_weight_baseline: str | None = None
    _draft_weight_baseline_fingerprint: str | None = None

    #: Per-tensor relative Frobenius deltas against that baseline, and their
    #: maximum.  ``None`` when no baseline directory could be resolved, for
    #: the same reason as the fingerprints above: "the magnitude check did not
    #: run" and "it ran and passed" must not look alike in the manifest.
    #:
    #: A *passing* cycle records these too.  The threshold in
    #: :data:`MIN_TRAINED_RELATIVE_DELTA` is calibrated only from below (see
    #: its derivation), and the only way that ever improves is if every cycle
    #: leaves its measured deltas in the artifact.
    _draft_weight_relative_deltas: dict[str, float] | None = None
    _draft_weight_max_relative_delta: float | None = None

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
        # The depth the cycle trained at, recorded beside the weights it
        # produced.  Without it nothing downstream can compare training depth
        # against the profile's serving ``num_speculative_tokens``, which is
        # how a 3-deep head came to be served 5-deep unnoticed.
        params["num_speculative_steps"] = self.config.num_speculative_steps
        # Proof, not inference, that the trained weights are the published
        # ones: the fingerprint is computed at the materialized draft and
        # re-checked against the artifact tree by
        # :meth:`assert_published_weights` before the publish commits.
        params["draft_weight_fingerprint"] = self._draft_weight_fingerprint
        params["draft_weight_baseline"] = self._draft_weight_baseline
        params["draft_weight_baseline_fingerprint"] = (
            self._draft_weight_baseline_fingerprint
        )
        # How far the weights actually moved, not merely that they moved.
        # Byte-identity answered the second question; three cycles then
        # shipped heads that differed from their baseline by one bf16 ULP.
        params["draft_weight_relative_deltas"] = self._draft_weight_relative_deltas
        params["draft_weight_max_relative_delta"] = self._draft_weight_max_relative_delta
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
        self._record_draft_weights(draft_directory)
        return draft_directory

    def _record_draft_weights(self, draft_directory: Path) -> None:
        """Prove the materialized draft is a trained head, and fingerprint it.

        Three separate claims, none of which the pipeline could previously
        make:

        1. **It is a draft head at all.**  Every key in
           :data:`REQUIRED_DRAFT_TENSORS` must be present -- ``fc.weight``
           above all.  The subprocess validator asked only for ``d2t``/``t2d``,
           which a directory containing no head would still satisfy.

        2. **Training changed something.**  The candidate is compared against
           the head it warm-started from (the stock drafter on the first
           cycle, the incumbent artifact afterwards).  Byte-identical means
           the cycle was a no-op and the gate would be asked to distinguish a
           head from itself, so it raises :class:`StockIdenticalDraftError`.

           Byte-identity turned out to be necessary but nowhere near
           sufficient.  Three real cycles produced heads whose every tensor
           differed from the baseline in the bytes and by *at most one bf16
           ULP* in value -- see :data:`MIN_TRAINED_RELATIVE_DELTA` -- so the
           identity check passed and a statistically identical head went to
           the gate anyway.  :func:`is_noop_training_delta` is the magnitude
           test that catches those, and it runs in addition to, never instead
           of, the identity test.

        3. **These weights are the published ones.**  The fingerprint travels
           into the artifact manifest via :meth:`describe` and is re-derived
           from the artifact tree by :meth:`assert_published_weights`.

        Claim 1 now also checks *arity*: ``fc.weight``'s shape states how many
        aux hidden states the head consumes, and a cycle configured to extract
        a different number of ``target_layer_ids`` is a shape error waiting to
        happen in the forward pass.  Reading it from the header here costs a
        few kilobytes and turns that into an :class:`AuxLayerCountMismatch`
        naming both numbers.

        A baseline that cannot be resolved to local weights -- a Hub id absent
        from the cache -- leaves claim 2 unmade rather than assumed, and the
        manifest records nulls so the gap is visible in the artifact.
        """
        missing = REQUIRED_DRAFT_TENSORS - draft_tensor_keys(draft_directory)
        if missing:
            raise DraftWeightsError(
                f"materialized draft is missing required tensors: {sorted(missing)} "
                f"(directory: {draft_directory})"
            )
        configured_layers = self.config.target_layer_ids
        if configured_layers is not None:
            expected = drafter_aux_count(draft_directory)
            if expected != len(configured_layers):
                # ``expected`` is the drafter's own claim, read off fc.weight;
                # ``actual`` is what this cycle supplied.  The message reads
                # "drafter expects {expected} ... but {actual} were provided",
                # so they must not be swapped.
                raise AuxLayerCountMismatch(
                    expected=expected,
                    actual=len(configured_layers),
                    drafter_model=self.config.draft_model,
                )
        fingerprint = weight_fingerprint(draft_directory)
        baseline, baseline_fingerprint, baseline_directory = self._baseline_weights()
        if baseline_fingerprint is not None and baseline_fingerprint == fingerprint:
            raise StockIdenticalDraftError(str(baseline), fingerprint)
        deltas: dict[str, float] | None = None
        max_delta: float | None = None
        if baseline_directory is not None:
            report = weight_delta_report(draft_directory, baseline_directory)
            deltas = report.deltas
            max_delta = report.max_delta
            if is_noop_training_delta(report):
                raise NoOpTrainingDeltaError(
                    str(baseline),
                    report.deltas,
                    threshold=MIN_TRAINED_RELATIVE_DELTA,
                    norm_epsilon=NORM_MOVED_EPSILON,
                )
        self._draft_weight_fingerprint = fingerprint
        self._draft_weight_baseline = baseline
        self._draft_weight_baseline_fingerprint = baseline_fingerprint
        self._draft_weight_relative_deltas = deltas
        self._draft_weight_max_relative_delta = max_delta

    def _baseline_weights(self) -> tuple[str | None, str | None, Path | None]:
        """Return the head this cycle built on, and its weight fingerprint.

        Preference order is the resolved warm start -- what training was
        actually handed -- then the configured stock drafter.  The second is
        not merely a fallback for the first cycle: when compounding is off the
        two are the same string, and when a resolver returned something the
        cache cannot back, falling through to stock still catches the
        no-op case that matters most.

        The resolved *directory* is returned alongside, because the magnitude
        comparison needs the baseline's tensors and not merely a digest of
        them: re-resolving it at the call site would let the two checks
        disagree about which head "the baseline" is.
        """
        candidates = [
            reference
            for reference in (self._resolved_warm_start, self.config.draft_model)
            if reference
        ]
        for reference in candidates:
            directory = cached_snapshot_dir(reference)
            if directory is None:
                continue
            try:
                return reference, weight_fingerprint(directory), directory
            except DraftWeightsError:
                # A baseline we cannot read is not evidence of anything; try
                # the next one and, failing that, record the absence.
                continue
        return (candidates[0] if candidates else None), None, None

    def assert_published_weights(self, published: Path) -> None:
        """Fail the publish unless *published* holds the weights we trained.

        Wired as the registry's ``before_publish`` hook, so it runs against
        the staged artifact tree while the rename is still revocable.

        This closes a gap rather than fixing a known bug: the
        ``checkpoint_best -> materialize -> publish -> --speculative-config``
        path traces correct, but nothing in it could *prove* the published
        directory holds the trained tensors, so the hypothesis had only ever
        been ruled out by reading code.  It now costs one pass over the
        safetensors payload, which is deliberate: the redundant pass
        :func:`speedlm.tuner.artifacts.hash_directory` dropped was a third
        whole-tree digest that added no guarantee, whereas this is the only
        step that ties the artifact back to the trained weights.
        """
        expected = self._draft_weight_fingerprint
        if expected is None:
            raise DraftWeightsError(
                "no materialized draft fingerprint is on record; refusing to "
                f"publish {published} as a trained draft"
            )
        actual = weight_fingerprint(published)
        if actual != expected:
            raise DraftWeightsError(
                f"published draft weights do not match the trained draft: "
                f"expected fingerprint {expected}, got {actual} (path: {published})"
            )

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
