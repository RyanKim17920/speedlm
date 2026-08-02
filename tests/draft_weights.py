"""Write minimal-but-real safetensors files for draft-model tests.

The production fingerprint parses the safetensors container directly (see
``speedlm.tuner.eagle3._safetensors_tensors``), so a zero-byte placeholder is
no longer a usable stand-in for a draft's weights.  These helpers emit the
real on-disk layout -- ``u64`` header length, JSON header, tensor buffer --
with tiny payloads, so tests stay fast while exercising the actual parser.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from speedlm.tuner.eagle3 import REQUIRED_DRAFT_TENSORS

#: Bytes per element of the ``F32`` dtype these helpers declare.
_F32_ITEMSIZE = 4


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
        header[name] = {
            "dtype": "F32",
            "shape": [len(blob) // _F32_ITEMSIZE],
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
    """Deterministic payloads for *names*, distinct for each *seed*."""
    return {
        name: bytes(((seed + index + position) % 256) for position in range(_F32_ITEMSIZE))
        for index, name in enumerate(sorted(names))
    }


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
