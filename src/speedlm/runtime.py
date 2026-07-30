"""Durable runtime record for a live SpeedLM gateway.

``speedlm status`` answers "is a gateway running?" by reading
``<SPEEDLM_HOME>/gateway.json`` and probing the recorded pid with signal 0 — it
never touches the network.  This module is the *writer* half of that contract:
it creates the record when ``speedlm vllm serve`` starts and removes it again on
a clean shutdown, including on SIGINT/SIGTERM.

Crash safety
------------
The record is written atomically (same-dir temp file + ``os.replace``), so a
reader never observes a partial object.  If the writing process is killed
outright (``SIGKILL``, power loss) the record survives; that is deliberate.
:func:`speedlm.report.read_gateway_status` probes the recorded pid and reports
such a leftover as ``stale`` rather than ``running``, so no extra bookkeeping is
needed here to make a crash detectable.

Ownership
---------
Removal is always conditional on the record still naming *this* process.  A
successor gateway that reused the same home must never have its record deleted
by a slow-exiting predecessor.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final

from speedlm.storage import atomic_write_json, resolve_layout

#: Name of the runtime record inside ``SPEEDLM_HOME``.  Must match
#: :data:`speedlm.report.GATEWAY_FILE_NAME`.
GATEWAY_FILE_NAME: Final = "gateway.json"

#: Signals whose default disposition terminates the process and which the
#: guard therefore cleans up after.
_GUARDED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)

#: What :func:`signal.getsignal` can hand back: a Python callable, one of the
#: ``SIG_DFL``/``SIG_IGN`` constants, or ``None`` for a non-Python handler.
_SignalHandler = Callable[[int, FrameType | None], Any] | int | None


class RuntimeRecordError(RuntimeError):
    """Raised when the gateway runtime record cannot be written."""


def gateway_record_path(home: Path | None = None) -> Path:
    """Return the runtime record path for *home* (default ``SPEEDLM_HOME``)."""
    return resolve_layout(home).root / GATEWAY_FILE_NAME


@dataclass(frozen=True, slots=True)
class GatewayRecord:
    """The on-disk description of a running gateway.

    The field names are the contract read by :mod:`speedlm.report`; ``pid`` is
    the only field a reader strictly requires, but every field is always
    written so ``status`` can name the endpoint it found.
    """

    pid: int
    host: str
    port: int
    model: str
    child_pid: int | None = None
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise RuntimeRecordError("gateway record pid must be a positive integer")
        if not isinstance(self.host, str) or not self.host:
            raise RuntimeRecordError("gateway record host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise RuntimeRecordError("gateway record port must be an integer")
        if not isinstance(self.model, str) or not self.model:
            raise RuntimeRecordError("gateway record model must be a non-empty string")
        if self.child_pid is not None and (
            isinstance(self.child_pid, bool) or not isinstance(self.child_pid, int)
        ):
            raise RuntimeRecordError("gateway record child_pid must be an integer or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "child_pid": self.child_pid,
            "host": self.host,
            "port": self.port,
            "model": self.model,
            "started_at": self.started_at,
        }


def read_record_pid(path: Path) -> int | None:
    """Return the pid named by the record at *path*, or ``None``.

    A missing, unreadable, malformed or pid-less record all yield ``None``:
    this is only ever used to decide whether *we* still own the file, and an
    uninterpretable file is by definition not ours to delete.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    pid = value.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    return pid


class GatewayRuntimeRecord:
    """Owns ``<home>/gateway.json`` for the lifetime of one serve.

    Both :meth:`write` and :meth:`remove` are idempotent, so the guard handler
    and the ``finally`` unwind can each run without stepping on the other.
    """

    def __init__(self, path: Path, record: GatewayRecord) -> None:
        self._path = path
        self._record = record

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record(self) -> GatewayRecord:
        return self._record

    def write(self) -> None:
        """Atomically publish the record, creating the home directory if needed."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._path, self._record.to_dict())
        except OSError as exc:
            raise RuntimeRecordError(
                f"cannot write gateway runtime record {self._path}: {exc}"
            ) from exc

    def update_child_pid(self, child_pid: int | None) -> None:
        """Atomically refresh the replaceable child pid while retaining ownership."""
        self._record = GatewayRecord(
            pid=self._record.pid,
            host=self._record.host,
            port=self._record.port,
            model=self._record.model,
            child_pid=child_pid,
            started_at=self._record.started_at,
        )
        self.write()

    def remove(self) -> bool:
        """Delete the record iff it still names this process.

        Returns:
            True when a record owned by this process was removed.
        """
        if read_record_pid(self._path) != self._record.pid:
            return False
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True


def _install_cleanup_guard(runtime: GatewayRuntimeRecord) -> Callable[[], None]:
    """Remove the record if a terminating signal arrives; return an uninstaller.

    The guard covers the window before the serve loop installs its own
    graceful-shutdown handlers (and the window after it restores them).  It
    deliberately re-raises the signal through the previously installed handler
    so the process keeps its original exit semantics.

    Signal handlers can only be installed from the main thread; off the main
    thread the guard degrades to a no-op and the ``finally`` unwind (or the
    stale-record detection in ``status``) remains the safety net.
    """
    previous: dict[int, _SignalHandler] = {}

    def _restore(signum: int, handler_to_restore: _SignalHandler) -> None:
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(
                signal.Signals(signum),
                handler_to_restore if handler_to_restore is not None else signal.SIG_DFL,
            )

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        runtime.remove()
        _restore(signum, previous.get(signum, signal.SIG_DFL))
        signal.raise_signal(signal.Signals(signum))

    try:
        for guarded in _GUARDED_SIGNALS:
            previous[int(guarded)] = signal.getsignal(guarded)
            signal.signal(guarded, handler)
    except ValueError:
        # Not the main thread — restore whatever was already swapped in.
        for installed, original in previous.items():
            _restore(installed, original)
        previous.clear()

    def uninstall() -> None:
        for installed, original in previous.items():
            # Only restore if our handler is still the installed one; the serve
            # loop legitimately swaps in its own handler for its lifetime.
            if signal.getsignal(signal.Signals(installed)) is not handler:
                continue
            _restore(installed, original)

    return uninstall


@contextlib.contextmanager
def gateway_runtime_record(
    *,
    host: str,
    port: int,
    model: str,
    child_pid: int | None = None,
    home: Path | None = None,
    path: Path | None = None,
    pid: int | None = None,
    now: float | None = None,
    guard_signals: bool = True,
) -> Iterator[GatewayRuntimeRecord]:
    """Publish a gateway runtime record for the duration of the block.

    The record is written on entry and removed on exit — normally, on an
    exception, and on SIGINT/SIGTERM when *guard_signals* is set.

    Args:
        host: Host the gateway listens on (as advertised to clients).
        port: Port the gateway listens on.
        model: Model name the child vLLM was launched with.
        child_pid: Pid of the child vLLM process, when it is already running.
        home: ``SPEEDLM_HOME`` override; ignored when *path* is given.
        path: Explicit record path, primarily for tests.
        pid: Owning pid; defaults to this process.
        now: Timestamp override for ``started_at``.
        guard_signals: Install SIGINT/SIGTERM cleanup handlers.
    """
    record = GatewayRecord(
        pid=os.getpid() if pid is None else pid,
        host=host,
        port=port,
        model=model,
        child_pid=child_pid,
        started_at=time.time() if now is None else now,
    )
    runtime = GatewayRuntimeRecord(
        gateway_record_path(home) if path is None else path,
        record,
    )
    runtime.write()
    uninstall = _install_cleanup_guard(runtime) if guard_signals else None
    try:
        yield runtime
    finally:
        if uninstall is not None:
            uninstall()
        runtime.remove()
