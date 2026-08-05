"""Left-dilate assistant span starts in a prepared Speculators dataset."""

from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def left_dilate_loss_mask(mask: Iterable[object]) -> list[bool]:
    """Return *mask* with each nonzero span extended one position left.

    Decisions are made from the original mask, so a newly enabled position can
    never trigger another dilation. A span already beginning at position zero
    is left alone rather than wrapping around to the final position.
    """
    original = [bool(value) for value in mask]
    dilated = original.copy()
    for index in range(1, len(original)):
        if original[index] and not original[index - 1]:
            dilated[index - 1] = True
    return dilated


def _plain_mask(value: object) -> list[object]:
    """Normalize a persisted list or tensor without importing torch."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("loss_mask must be a sequence")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        listed = tolist()
        if isinstance(listed, list):
            return listed
        raise TypeError("loss_mask tolist() result must be a list")
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError("loss_mask must be iterable") from error


def _dilate_row(row: Mapping[str, object]) -> dict[str, list[bool]]:
    if "loss_mask" not in row:
        raise ValueError("prepared dataset row has no loss_mask")
    return {"loss_mask": left_dilate_loss_mask(_plain_mask(row["loss_mask"]))}


def _span_starts_with_predecessors(mask: Iterable[object]) -> int:
    values = [bool(value) for value in mask]
    return sum(values[index] and not values[index - 1] for index in range(1, len(values)))


def dilate_prepared_dataset(dataset_path: Path) -> int:
    """Dilate every row in *dataset_path* and return the exact additions."""
    load_from_disk = importlib.import_module("datasets").load_from_disk
    dataset: Any = load_from_disk(str(dataset_path))
    added_positions = sum(
        _span_starts_with_predecessors(_plain_mask(row["loss_mask"]))
        for row in dataset
    )
    dilated: Any = dataset.map(_dilate_row, load_from_cache_file=False)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{dataset_path.name}.dilated-", dir=dataset_path.parent)
    )
    backup = Path(
        tempfile.mkdtemp(prefix=f".{dataset_path.name}.undilated-", dir=dataset_path.parent)
    )
    backup.rmdir()
    try:
        dilated.save_to_disk(str(temporary))
        dataset_path.replace(backup)
        try:
            temporary.replace(dataset_path)
        except BaseException:
            backup.replace(dataset_path)
            raise
        shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return added_positions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"prepared dataset is not a directory: {args.dataset}")
    added_positions = dilate_prepared_dataset(args.dataset)
    print(f"SPEEDLM_DILATED_MASK_POSITIONS={added_positions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
