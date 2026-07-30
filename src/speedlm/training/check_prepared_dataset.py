"""Validate a Speculators prepared dataset from its training environment."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("expected_rows", type=int)
    parser.add_argument("--require-nonzero-loss-mask", action="store_true")
    parser.add_argument("--max-seq-len", type=int, required=True)
    return parser


def _column(value: Any) -> list[Any] | None:
    """Coerce one persisted column value into a plain sequence.

    ``Dataset.save_to_disk`` persists the formatting applied while preparing the
    rows, so ``load_from_disk`` hands a torch-formatted dataset back as tensors
    rather than lists. Normalize anything sequence-shaped instead of rejecting a
    well-formed row on the container type its producer happened to choose.
    """
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        listed = tolist()
        return listed if isinstance(listed, list) else None
    if isinstance(value, Sequence):
        return list(value)
    return None


def main() -> int:
    args = _parser().parse_args()
    load_from_disk = importlib.import_module("datasets").load_from_disk
    dataset: Any = load_from_disk(str(args.dataset))
    if len(dataset) != args.expected_rows:
        raise SystemExit(
            f"prepared dataset row count is {len(dataset)}, "
            f"expected {args.expected_rows}"
        )
    for index, row in enumerate(dataset):
        input_ids = _column(row.get("input_ids"))
        loss_mask = _column(row.get("loss_mask"))
        if input_ids is None:
            raise SystemExit(f"row {index} has no input_ids sequence")
        if len(input_ids) > args.max_seq_len:
            raise SystemExit(
                f"row {index} length {len(input_ids)} exceeds {args.max_seq_len}"
            )
        if args.require_nonzero_loss_mask and (
            loss_mask is None or not any(bool(value) for value in loss_mask)
        ):
            raise SystemExit(f"row {index} has an all-zero loss mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
