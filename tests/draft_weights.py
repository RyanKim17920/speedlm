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


def safetensors_bytes(payloads: Mapping[str, bytes]) -> bytes:
    """Serialize *payloads* as a safetensors file body.

    Each entry is declared ``F32`` with a one-dimensional shape derived from
    its byte length, which is all the fingerprint reads.
    """
    header: dict[str, object] = {}
    offset = 0
    for name in sorted(payloads):
        blob = payloads[name]
        if len(blob) % _F32_ITEMSIZE:
            raise ValueError(f"payload for {name!r} is not a whole number of F32 values")
        count = len(blob) // _F32_ITEMSIZE
        header[name] = {
            "dtype": "F32",
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


def draft_payloads(
    *,
    seed: int = 0,
    names: Iterable[str] = REQUIRED_DRAFT_TENSORS,
) -> dict[str, bytes]:
    """Deterministic payloads for *names*, distinct for each *seed*.

    One ``F32`` value each, except ``fc.weight``, which carries
    :data:`_FIXTURE_AUX_COUNT` so its declared shape is ``(1, 3)``.
    """
    return {
        name: bytes(
            ((seed + index + position) % 256)
            for position in range(
                _F32_ITEMSIZE * (_FIXTURE_AUX_COUNT if name == "fc.weight" else 1)
            )
        )
        for index, name in enumerate(sorted(names))
    }


#: Bytes per element of every dtype these helpers can emit.
DTYPE_ITEMSIZE: dict[str, int] = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}


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
