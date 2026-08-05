"""Write minimal-but-real safetensors files for draft-model tests.

The production fingerprint parses the safetensors container directly (see
``speedlm.tuner.eagle3._safetensors_tensors``), so a zero-byte placeholder is
no longer a usable stand-in for a draft's weights.  These helpers emit the
real on-disk layout -- ``u64`` header length, JSON header, tensor buffer --
with tiny payloads, so tests stay fast while exercising the actual parser.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from speedlm.tuner.eagle3 import REQUIRED_DRAFT_TENSORS

#: Bytes per element of the ``F32`` dtype these helpers declare.
_F32_ITEMSIZE = 4

#: Aux hidden states the stand-in ``fc.weight`` claims to fuse.
#:
#: ``fc.weight`` is no longer just another opaque payload: production reads its
#: *shape* to derive the drafter's aux-layer count (``shape[1] // shape[0]``,
#: see ``speedlm.tuner.eagle3.drafter_aux_count``) and rejects a head whose
#: count disagrees with the configured ``target_layer_ids``.  Both stock
#: drafters imply 3 -- gpt-oss ``(2880, 8640)``, Qwen3 ``(4096, 12288)`` -- and
#: both pinned profiles configure three target layers, so a fixture that wants
#: to stand in for a real head has to declare 3 as well.  It is emitted as
#: ``(1, 3)`` rather than the real dimensions because nothing in these tests
#: reads the payload's magnitudes, only its structure.
_FIXTURE_AUX_COUNT = 3

#: The reduced vocabulary the fixture draft claims, matching the
#: ``draft_vocab_size`` in :func:`speculators_config_payload`.
FIXTURE_DRAFT_VOCAB = 8

#: The verifier vocabulary the fixture draft is validated against, written by
#: :func:`write_verifier_config`.  Twice the draft vocabulary, so the fixture
#: exercises a genuine *reduction* -- an equal-sized one would make the
#: identity map legal and the all-zeros check vacuous.
FIXTURE_VERIFIER_VOCAB = 16

#: Draft id ``i`` maps to target id ``i * _FIXTURE_VOCAB_STRIDE``.
_FIXTURE_VOCAB_STRIDE = 2


def fixture_d2t(
    draft_vocab: int = FIXTURE_DRAFT_VOCAB, stride: int = _FIXTURE_VOCAB_STRIDE
) -> list[int]:
    """A well-formed ``d2t``, stored as the on-disk *delta* to the target id."""
    return [index * stride - index for index in range(draft_vocab)]


def fixture_t2d(
    draft_vocab: int = FIXTURE_DRAFT_VOCAB,
    verifier_vocab: int = FIXTURE_VERIFIER_VOCAB,
    stride: int = _FIXTURE_VOCAB_STRIDE,
) -> list[int]:
    """The ``t2d`` mask matching :func:`fixture_d2t`."""
    mask = [0] * verifier_vocab
    for index in range(draft_vocab):
        mask[index * stride] = 1
    return mask


def d2t_bytes(values: Sequence[int]) -> bytes:
    """Pack *values* as little-endian ``I64``, the dtype ``d2t`` ships as."""
    return b"".join(int(value).to_bytes(8, "little", signed=True) for value in values)


def t2d_bytes(values: Sequence[int]) -> bytes:
    """Pack *values* as safetensors ``BOOL`` -- one byte per element."""
    return bytes(1 if value else 0 for value in values)


#: How the two index maps are declared, since they are not ``F32``.
#:
#: ``d2t`` is ``I64`` and ``t2d`` is ``BOOL`` in every real checkpoint, and
#: production now decodes both for real (``speedlm.tuner.eagle3``
#: ``assert_draft_vocab_mapping``).  A fixture that declared them ``F32``
#: would be rejected as "not an integer index map", so the writer states the
#: true dtypes and emits payloads that are actually a valid mapping.
_VOCAB_MAP_DECLARATIONS: dict[str, tuple[str, int]] = {"d2t": ("I64", 8), "t2d": ("BOOL", 1)}


def safetensors_bytes(payloads: Mapping[str, bytes]) -> bytes:
    """Serialize *payloads* as a safetensors file body.

    Entries are declared ``F32`` with a one-dimensional shape derived from
    their byte length, which is all the fingerprint reads -- except the two
    vocabulary index maps, which carry their real dtypes (see
    :data:`_VOCAB_MAP_DECLARATIONS`).
    """
    header: dict[str, object] = {}
    offset = 0
    for name in sorted(payloads):
        blob = payloads[name]
        declaration = _VOCAB_MAP_DECLARATIONS.get(name)
        if declaration is not None:
            dtype, itemsize = declaration
        else:
            dtype, itemsize = "F32", _F32_ITEMSIZE
        if len(blob) % itemsize:
            raise ValueError(
                f"payload for {name!r} is not a whole number of {dtype} values"
            )
        count = len(blob) // itemsize
        header[name] = {
            "dtype": dtype,
            # Only ``fc.weight`` is shaped, and only because its shape is read.
            "shape": (
                [count // _FIXTURE_AUX_COUNT, _FIXTURE_AUX_COUNT]
                if name == "fc.weight"
                else [count]
            ),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
    raw = json.dumps(header).encode("utf-8")
    body = b"".join(payloads[name] for name in sorted(payloads))
    return len(raw).to_bytes(8, "little") + raw + body


def write_verifier_config(
    directory: Path, *, vocab_size: int = FIXTURE_VERIFIER_VOCAB
) -> Path:
    """Write a minimal verifier snapshot declaring *vocab_size*.

    Production reads the verifier's vocabulary from its own ``config.json``
    via :func:`speedlm.profiles.cached_snapshot_dir`, which returns an on-disk
    path unchanged.  Passing this directory as ``verifier_model`` therefore
    gives tests a real runtime fact to validate against instead of a constant.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps({"vocab_size": vocab_size}), encoding="utf-8")
    return path


def draft_payloads(
    *,
    seed: int = 0,
    names: Iterable[str] = REQUIRED_DRAFT_TENSORS,
) -> dict[str, bytes]:
    """Deterministic payloads for *names*, distinct for each *seed*.

    One ``F32`` value each, except ``fc.weight``, which carries
    :data:`_FIXTURE_AUX_COUNT` so its declared shape is ``(1, 3)``.

    ``d2t``/``t2d`` are the exception to the exception: they are *index maps*,
    not weights, and production validates their contents.  They therefore
    carry the canonical valid mapping and do **not** vary with *seed* -- a
    seed-dithered index map would be an out-of-range or self-inconsistent one,
    which is precisely the state these fixtures must not accidentally be in.
    Fingerprint tests that need two distinguishable drafts still get them:
    every other tensor still varies.
    """
    payloads: dict[str, bytes] = {}
    for index, name in enumerate(sorted(names)):
        if name == "d2t":
            payloads[name] = d2t_bytes(fixture_d2t())
            continue
        if name == "t2d":
            payloads[name] = t2d_bytes(fixture_t2d())
            continue
        payloads[name] = bytes(
            ((seed + index + position) % 256)
            for position in range(
                _F32_ITEMSIZE * (_FIXTURE_AUX_COUNT if name == "fc.weight" else 1)
            )
        )
    return payloads


#: Bytes per element of every dtype these helpers can emit.
DTYPE_ITEMSIZE: dict[str, int] = {"BOOL": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8}


def typed_safetensors_bytes(
    tensors: Mapping[str, tuple[str, tuple[int, ...], bytes]],
) -> bytes:
    """Serialize *name -> (dtype, shape, payload)* as a safetensors file body.

    :func:`safetensors_bytes` above declares everything ``F32`` with a 1-D
    shape, which is all the *fingerprint* reads.  The magnitude analysis reads
    dtype and shape for real -- it has to widen bf16 by hand and divide
    ``fc.weight``'s dimensions -- so fixtures for it must state both, and the
    payload length is cross-checked against them here rather than trusted.
    """
    header: dict[str, object] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, blob = tensors[name]
        itemsize = DTYPE_ITEMSIZE[dtype]
        expected = itemsize
        for extent in shape:
            expected *= extent
        if len(blob) != expected:
            raise ValueError(
                f"payload for {name!r} is {len(blob)} bytes, but {dtype}{list(shape)} "
                f"needs {expected}"
            )
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
    raw = json.dumps(header).encode("utf-8")
    body = b"".join(tensors[name][2] for name in sorted(tensors))
    return len(raw).to_bytes(8, "little") + raw + body


def bf16_bytes(patterns: Sequence[int]) -> bytes:
    """Pack raw little-endian bf16 *patterns* (16-bit integers) into bytes.

    Fixtures work in bit patterns rather than Python floats on purpose: "one
    ULP" is exactly "the next bit pattern", so incrementing the integer is the
    only encoding-independent way to build the dither the guard must catch.
    """
    return b"".join(int(pattern).to_bytes(2, "little") for pattern in patterns)


def write_typed_draft(
    directory: Path,
    tensors: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    *,
    filename: str = "model.safetensors",
) -> Path:
    """Write one typed safetensors shard into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(typed_safetensors_bytes(tensors))
    return path


def write_draft_weights(
    directory: Path,
    *,
    seed: int = 0,
    names: Iterable[str] = REQUIRED_DRAFT_TENSORS,
    filename: str = "model.safetensors",
) -> Path:
    """Write one safetensors shard holding *names* into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(safetensors_bytes(draft_payloads(seed=seed, names=names)))
    return path


def speculators_config_payload(
    depth: int = 3,
    *,
    methods: int = 1,
    verifier: str = "acme/verifier",
) -> dict[str, object]:
    """A stock-shaped Speculators EAGLE-3 draft config declaring *depth*.

    Shaped after the real ``config.json`` in
    ``RedHatAI/gpt-oss-20b-speculator.eagle3``: production reads
    ``speculators_config.proposal_methods[*].speculative_tokens`` out of it
    (``speedlm.profiles.drafter_declared_speculative_tokens``) and rewrites
    that field at materialization
    (``speedlm.tuner.eagle3.declare_speculative_tokens``), so a fixture draft
    that carries no such block is not a stand-in for a materialized head.
    """
    return {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 8,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": depth,
                    "verifier_accept_k": 1,
                }
                for _ in range(methods)
            ],
            "verifier": {"architectures": [], "name_or_path": verifier},
        },
    }


def write_draft_config(directory: Path, payload: object | None = None) -> Path:
    """Write *payload* (default: a 3-deep stock-shaped config) as config.json."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if payload is None:
        payload = speculators_config_payload()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
