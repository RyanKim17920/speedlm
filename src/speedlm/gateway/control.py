"""Safe, synchronous runtime control for tuner-driven vLLM replacement."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import Final, Protocol

from speedlm.gateway.activity import ActivityTracker

AbortCheck = Callable[[], bool]
CaptureBarrier = Callable[[float, AbortCheck], None]
DraftReference = Path | str

# Keep these paths here rather than spreading vLLM's control API through the
# gateway. They deliberately do not use the public proxy's /v1 namespace.
VLLM_SLEEP_ENDPOINT: Final = "/sleep"
VLLM_WAKE_ENDPOINT: Final = "/wake_up"
VLLM_IS_SLEEPING_ENDPOINT: Final = "/is_sleeping"
VLLM_SLEEP_LEVEL: Final = "1"
VLLM_SLEEP_MODE: Final = "wait"

DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.05
DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final = 600.0


class AdmissionControl(Protocol):
    """Open and close the gateway's admission gate.

    Implementations must be idempotent. Closing the gate affects new requests
    only; requests already counted by :class:`ActivityTracker` keep draining.
    """

    def stop_admitting(self) -> None: ...

    def start_admitting(self) -> None: ...


class VLLMControlHTTP(Protocol):
    """The small part of vLLM's HTTP control surface used by the tuner."""

    def post(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        query: Mapping[str, str] | None = None,
    ) -> None:
        """POST a control request, raising unless vLLM accepts it."""

    def wait_ready(self, *, timeout_seconds: float) -> None:
        """Return only after the child is ready to serve requests."""

    def wait_sleeping(
        self,
        sleeping: bool,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        """Return only when vLLM reports the requested authoritative state."""


class ChildProcessControl(Protocol):
    """Replace the managed child without exposing process-library details."""

    def restart(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
    ) -> None:
        """Restart the child configured with *draft*."""


class ControlAborted(RuntimeError):
    """Raised when an idle-cycle preemption asks runtime control to stop."""


class ControlTimeout(TimeoutError):
    """Raised when a runtime-control operation exhausts its time budget."""


class ServiceRecoveryError(RuntimeError):
    """Raised when both an operation and its serving recovery fail."""


class AdmissionGate:
    """Thread-safe admission control shared by the gateway and tuner.

    A request attempted while the gate is closed updates the shared activity
    watermark. That is what preempts an idle cycle even though the request
    cannot yet be forwarded to the sleeping child.
    """

    def __init__(self, activity: ActivityTracker) -> None:
        self._activity = activity
        self._lock = Lock()
        self._closed = False

    @property
    def is_admitting(self) -> bool:
        with self._lock:
            return not self._closed and self._activity.is_admitting

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def try_begin(self) -> bool:
        with self._lock:
            return not self._closed and self._activity.try_begin()

    async def wait_to_begin(self, *, poll_interval_seconds: float = 0.01) -> bool:
        """Wait for normal reopening, or return false after terminal shutdown."""
        _validate_timeout(poll_interval_seconds, name="admission poll interval")
        while True:
            if self.try_begin():
                return True
            if self.is_closed:
                return False
            await asyncio.sleep(poll_interval_seconds)

    def stop_admitting(self) -> None:
        with self._lock:
            self._activity.stop_admitting()

    def start_admitting(self) -> None:
        with self._lock:
            if not self._closed:
                self._activity.start_admitting()

    def close(self) -> None:
        """Permanently reject queued/new requests during gateway shutdown."""
        with self._lock:
            self._closed = True
            self._activity.stop_admitting()


class _Deadline:
    def __init__(
        self,
        timeout_seconds: float,
        *,
        clock: Callable[[], float],
        operation: str,
    ) -> None:
        _validate_timeout(timeout_seconds)
        self._clock = clock
        self._expires_at = clock() + timeout_seconds
        self._operation = operation

    def remaining(self) -> float:
        remaining = self._expires_at - self._clock()
        if remaining <= 0:
            raise ControlTimeout(f"{self._operation} timed out")
        return remaining

    def check(self) -> None:
        self.remaining()


class RuntimeController:
    """Concrete implementation of the tuner's runtime-control protocol.

    Normal operation deadlines are strict. Recovery after a failed sleep,
    candidate start, or wake gets a separate budget because restoring serving
    is more important than returning at the original deadline.
    """

    def __init__(
        self,
        *,
        activity: ActivityTracker,
        admission: AdmissionControl,
        http: VLLMControlHTTP,
        process: ChildProcessControl,
        active_draft: DraftReference,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        recovery_timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        capture_barrier: CaptureBarrier | None = None,
    ) -> None:
        if not str(active_draft):
            raise ValueError("active draft must be non-empty")
        _validate_timeout(poll_interval_seconds, name="poll interval")
        _validate_timeout(recovery_timeout_seconds, name="recovery timeout")
        self._activity = activity
        self._admission = admission
        self._http = http
        self._process = process
        self._active_draft = active_draft
        self._running_draft: DraftReference = active_draft
        self._clock = clock
        self._sleeper = sleeper
        self._poll_interval_seconds = poll_interval_seconds
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._capture_barrier = capture_barrier
        self._admissions_stopped = False
        self._sleeping = False

    def quiesce(
        self,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        """Close admission and wait until every tracked request has drained."""
        deadline = self._deadline(timeout_seconds, "quiesce")
        try:
            self._check_abort(should_abort, "quiesce")
            deadline.check()
            self._stop_admitting()
            while self._activity.in_flight:
                self._check_abort(should_abort, "quiesce")
                remaining = deadline.remaining()
                self._sleeper(min(self._poll_interval_seconds, remaining))
            if self._capture_barrier is not None:
                self._capture_barrier(deadline.remaining(), should_abort)
            self._check_abort(should_abort, "quiesce")
            deadline.check()
        except Exception:
            self._best_effort_start_admitting()
            raise

    def sleep(
        self,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        """Put vLLM into level-1 sleep, restoring the active draft on failure."""
        deadline = self._deadline(timeout_seconds, "sleep")
        sleep_requested = False
        try:
            self._check_abort(should_abort, "sleep")
            deadline.check()
            if not self._sleeping:
                sleep_requested = True
                self._http.post(
                    VLLM_SLEEP_ENDPOINT,
                    timeout_seconds=deadline.remaining(),
                    query={
                        "level": VLLM_SLEEP_LEVEL,
                        "mode": VLLM_SLEEP_MODE,
                    },
                )
                self._http.wait_sleeping(
                    True,
                    timeout_seconds=deadline.remaining(),
                    should_abort=should_abort,
                )
                deadline.check()
                self._sleeping = True
            self._check_abort(should_abort, "sleep")
            deadline.check()
        except Exception as error:
            if sleep_requested or self._sleeping:
                self._recover_or_raise("sleep", error)
            else:
                self._best_effort_start_admitting()
            raise

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        """Replace the sleeping child with a candidate EAGLE-3 draft."""
        deadline = self._deadline(timeout_seconds, "candidate start")
        try:
            self._check_abort(should_abort, "candidate start")
            deadline.check()
            if self._sleeping or self._running_draft != draft_directory:
                self._restart_and_wait(draft_directory, deadline)
                self._running_draft = draft_directory
                self._sleeping = False
            else:
                self._http.wait_ready(timeout_seconds=deadline.remaining())
                deadline.check()
            self._check_abort(should_abort, "candidate start")
            deadline.check()
        except Exception as error:
            self._recover_or_raise("candidate start", error)
            raise

    def restore(
        self,
        active_draft: DraftReference,
        *,
        timeout_seconds: float,
    ) -> None:
        """Reliably restart on the supplied durable active draft."""
        if not str(active_draft):
            raise ValueError("active draft must be non-empty")
        self._active_draft = active_draft
        deadline = self._deadline(timeout_seconds, "restore")
        try:
            self._restart_and_wait(active_draft, deadline)
        except Exception as error:
            # A failed restart may have stopped the child. Retry the same known
            # good draft with a fresh safety budget before reporting failure.
            self._recover_or_raise("restore", error)
            raise
        self._running_draft = active_draft
        self._sleeping = False

    def wake(self, *, timeout_seconds: float) -> None:
        """Wake vLLM, confirm readiness, and reopen gateway admission."""
        deadline = self._deadline(timeout_seconds, "wake")
        try:
            if self._sleeping:
                self._http.post(
                    VLLM_WAKE_ENDPOINT,
                    timeout_seconds=deadline.remaining(),
                )
                self._http.wait_sleeping(
                    False,
                    timeout_seconds=deadline.remaining(),
                    should_abort=lambda: False,
                )
                deadline.check()
                self._sleeping = False
            self._http.wait_ready(timeout_seconds=deadline.remaining())
            deadline.check()
        except Exception as error:
            self._recover_or_raise("wake", error)
            raise
        self._start_admitting()

    def _deadline(self, timeout_seconds: float, operation: str) -> _Deadline:
        return _Deadline(timeout_seconds, clock=self._clock, operation=operation)

    @staticmethod
    def _check_abort(should_abort: AbortCheck, operation: str) -> None:
        if should_abort():
            raise ControlAborted(f"{operation} aborted")

    def _restart_and_wait(self, draft: DraftReference, deadline: _Deadline) -> None:
        self._process.restart(draft, timeout_seconds=deadline.remaining())
        deadline.check()
        self._http.wait_ready(timeout_seconds=deadline.remaining())
        deadline.check()

    def _recover_or_raise(self, operation: str, error: Exception) -> None:
        try:
            recovery = self._deadline(
                self._recovery_timeout_seconds,
                f"{operation} service recovery",
            )
            self._restart_and_wait(self._active_draft, recovery)
            self._running_draft = self._active_draft
            self._sleeping = False
            self._start_admitting()
        except Exception as recovery_error:
            self._best_effort_start_admitting()
            raise ServiceRecoveryError(
                f"{operation} failed ({error}); restoring service also failed "
                f"({recovery_error})"
            ) from error

    def _stop_admitting(self) -> None:
        if not self._admissions_stopped:
            self._admission.stop_admitting()
            self._admissions_stopped = True

    def _start_admitting(self) -> None:
        if self._admissions_stopped:
            self._admission.start_admitting()
            self._admissions_stopped = False

    def _best_effort_start_admitting(self) -> None:
        with contextlib.suppress(Exception):
            self._start_admitting()


def _validate_timeout(value: float, *, name: str = "timeout") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
