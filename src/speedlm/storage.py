from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    _fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_FILE_LOCK_TIMEOUT_SECONDS = 0.25
_FILE_LOCK_POLL_SECONDS = 0.01
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

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


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _file_lock_path(path: Path) -> Path:
    """Return the stable sidecar locked by both append and atomic rewrite."""
    return path.with_name(f".{path.name}.lock")


@contextmanager
def _exclusive_file_lock(
    path: Path,
    *,
    timeout: float = _FILE_LOCK_TIMEOUT_SECONDS,
) -> Iterator[bool]:
    """Yield whether the bounded exclusive lock for *path* was acquired.

    POSIX uses ``fcntl.flock`` on a stable sidecar so an atomic replacement of
    *path* cannot invalidate the lock. Platforms without ``fcntl`` fall back to
    a process-local thread lock; that preserves thread safety but cannot
    coordinate writers in separate processes.
    """
    timeout = max(0.0, timeout)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("dropping write after lock setup failed for %s: %s", path, exc)
        yield False
        return

    if _fcntl is None:
        lock = _thread_lock_for(path)
        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            logger.warning("dropping write after file lock deadline for %s", path)
            yield False
            return
        try:
            yield True
        finally:
            lock.release()
        return

    lock_path = _file_lock_path(path)
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        logger.warning("dropping write after lock open failed for %s: %s", path, exc)
        yield False
        return

    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _fcntl.flock(lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("dropping write after file lock deadline for %s", path)
                    yield False
                    return
                time.sleep(min(_FILE_LOCK_POLL_SECONDS, remaining))
            except OSError as exc:
                logger.warning("dropping write after file lock failed for %s: %s", path, exc)
                yield False
                return

        yield True
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(lock_fd)


def _append_jsonl(path: Path, obj: object) -> bool:
    """Append one JSON line, returning false when the bounded write is dropped."""
    line = json.dumps(obj) + "\n"
    with _exclusive_file_lock(path) as acquired:
        if not acquired:
            return False
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            logger.warning("dropping JSONL write for %s: %s", path, exc)
            return False
    return True


def append_jsonl(path: Path, obj: object) -> None:
    """Append a JSON line, logging and dropping it if the bounded lock fails."""
    _append_jsonl(path, obj)


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
