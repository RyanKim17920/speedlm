"""Validate a Speculators prepared dataset from its training environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("expected_rows", type=int)
    parser.add_argument("--require-nonzero-loss-mask", action="store_true")
    parser.add_argument("--max-seq-len", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    from datasets import load_from_disk

    dataset: Any = load_from_disk(str(args.dataset))
    if len(dataset) != args.expected_rows:
        raise SystemExit(
            f"prepared dataset row count is {len(dataset)}, "
            f"expected {args.expected_rows}"
        )
    for index, row in enumerate(dataset):
        input_ids = row.get("input_ids")
        loss_mask = row.get("loss_mask")
        if not isinstance(input_ids, (list, tuple)):
            raise SystemExit(f"row {index} has no input_ids sequence")
        if len(input_ids) > args.max_seq_len:
            raise SystemExit(
                f"row {index} length {len(input_ids)} exceeds {args.max_seq_len}"
            )
        if args.require_nonzero_loss_mask and (
            not isinstance(loss_mask, (list, tuple))
            or not any(bool(value) for value in loss_mask)
        ):
            raise SystemExit(f"row {index} has an all-zero loss mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
