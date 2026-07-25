from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageError(RuntimeError):
    """Raised for storage I/O errors."""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    traces_dir: Path
    profiles_dir: Path
    runs_dir: Path


def resolve_layout(home: Path | None = None) -> Layout:
    """Resolve directory paths without creating them.

    When *home* is ``None``, ``config.speedlm_home()`` is used.
    """
    if home is None:
        from speedlm.config import speedlm_home
        home = speedlm_home()
    return Layout(
        root=home,
        traces_dir=home / "traces",
        profiles_dir=home / "profiles",
        runs_dir=home / "runs",
    )


def ensure_layout(home: Path | None = None) -> Layout:
    """Like :func:`resolve_layout` but creates all four directories."""
    layout = resolve_layout(home)
    for dir_path in (layout.root, layout.traces_dir, layout.profiles_dir, layout.runs_dir):
        dir_path.mkdir(parents=True, exist_ok=True)
    return layout


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def new_run_dir(
    layout: Layout,
    prefix: str = "run",
    *,
    now: datetime | None = None,
) -> Path:
    """Create a uniquely-named run directory inside ``layout.runs_dir``.

    The directory name follows the pattern ``{prefix}-{YYYYmmdd-HHMMSS}``.
    If that name already exists a numeric suffix (``-1``, ``-2``, ...) is
    appended until a free name is found.  The directory is created and
    returned.
    """
    if now is None:
        now = datetime.now()
    ts = now.strftime("%Y%m%d-%H%M%S")
    base = f"{prefix}-{ts}"
    candidate = layout.runs_dir / base
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    counter = 1
    while True:
        candidate = layout.runs_dir / f"{base}-{counter}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write *text* to *path* via same-dir tmp + rename."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    except BaseException:
        # Best-effort cleanup of the temp file
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: object) -> None:
    """Serialize *obj* to pretty JSON and write atomically."""
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, obj: object) -> None:
    """Append a single JSON line to *path*, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield JSON objects from *path*.

    Blank lines are silently skipped.  Malformed lines or lines that are
    not JSON objects raise ``StorageError`` with the 1-based line number.
    A missing file raises ``StorageError``.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except FileNotFoundError:
        raise StorageError(f"JSONL file not found: {path}") from None
    except OSError as exc:
        raise StorageError(f"cannot open JSONL file: {path}") from exc

    with os.fdopen(fd, "r", encoding="utf-8") as f:
        line_no = 0
        for raw_line in f:
            line_no += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StorageError(
                    f"malformed JSON on line {line_no} in {path}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise StorageError(
                    f"line {line_no} in {path} is not a JSON object"
                    f" (got {type(obj).__name__})"
                )
            yield obj