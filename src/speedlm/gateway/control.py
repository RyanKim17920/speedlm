"""Safe, synchronous runtime control for tuner-driven vLLM replacement."""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

# Device-memory observation. ``nvidia-smi`` is the only reading available to
# this process: the gateway venv deliberately does not depend on torch (see the
# pinned-stack note in pyproject.toml), and even where torch is importable,
# ``torch.cuda.mem_get_info`` would first have to create a CUDA context inside
# the gateway process. That context permanently occupies a few hundred MiB of
# the very device memory the next engine is about to size its KV cache from, so
# the in-process query would degrade what it exists to measure. ``nvidia-smi``
# reads the driver's device-global accounting without touching the device, and
# is already how :mod:`speedlm.doctor` inspects GPUs.
NVIDIA_SMI_BINARY: Final = "nvidia-smi"
NVIDIA_SMI_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_GPU_MEMORY_TIMEOUT_SECONDS: Final = 120.0
DEFAULT_GPU_MEMORY_POLL_INTERVAL_SECONDS: Final = 1.0

_BYTES_PER_MIB: Final = 1024 * 1024
_BYTES_PER_GIB: Final = 1024 * 1024 * 1024


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


class GPUMemoryProbeError(RuntimeError):
    """Raised when free device memory cannot be observed at all."""


class GPUMemoryNotReleased(RuntimeError):
    """Raised when a sleeping engine has not given back enough device memory.

    Fail-closed by design: the alternative is letting the next engine start and
    die on an opaque CUDA out-of-memory error deep inside a child process.
    """


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """Free and total memory for one visible CUDA device, in bytes."""

    index: int
    free_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError("device total memory must be positive")
        if not 0 <= self.free_bytes <= self.total_bytes:
            raise ValueError("device free memory must be within 0..total")


class GPUMemoryProbe(Protocol):
    """Read current device memory for every CUDA device the child may use."""

    def read(self) -> tuple[DeviceMemory, ...]:
        """Return one entry per visible device, or raise ``GPUMemoryProbeError``."""


@dataclass(frozen=True, slots=True)
class NvidiaSmiMemoryProbe:
    """Observe device-global memory by shelling out to ``nvidia-smi``.

    vLLM runs as a *separate child process*, so only a device-global reading
    says anything about whether the sleeping engine actually gave its memory
    back. ``CUDA_VISIBLE_DEVICES`` is honoured explicitly because ``nvidia-smi``
    itself ignores it and would otherwise report devices this deployment never
    uses.
    """

    timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS
    binary: str = NVIDIA_SMI_BINARY
    #: Overridable only so tests need not mutate the real process environment.
    environ: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _validate_timeout(self.timeout_seconds, name="nvidia-smi timeout")

    def read(self) -> tuple[DeviceMemory, ...]:
        command = [
            self.binary,
            "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise GPUMemoryProbeError(f"{self.binary} is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise GPUMemoryProbeError(
                f"{self.binary} timed out after {self.timeout_seconds:g}s"
            ) from exc
        except OSError as exc:
            raise GPUMemoryProbeError(f"could not execute {self.binary}: {exc}") from exc
        if completed.returncode != 0:
            detail = _one_line(completed.stderr) or f"exit status {completed.returncode}"
            raise GPUMemoryProbeError(f"{self.binary} failed: {detail}")
        return _select_visible(_parse_nvidia_smi(completed.stdout), environ=self.environ)


def _parse_nvidia_smi(output: str) -> tuple[tuple[str, DeviceMemory], ...]:
    """Parse ``index, uuid, total, free`` MiB rows into ``(uuid, memory)`` pairs."""
    devices: list[tuple[str, DeviceMemory]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise GPUMemoryProbeError(f"unparsable nvidia-smi row: {line.strip()!r}")
        index_text, uuid, total_text, free_text = fields
        try:
            device = DeviceMemory(
                index=int(index_text),
                free_bytes=int(free_text) * _BYTES_PER_MIB,
                total_bytes=int(total_text) * _BYTES_PER_MIB,
            )
        except ValueError as exc:
            raise GPUMemoryProbeError(f"unparsable nvidia-smi row: {line.strip()!r}") from exc
        devices.append((uuid, device))
    if not devices:
        raise GPUMemoryProbeError("nvidia-smi reported no GPUs")
    return tuple(devices)


def _select_visible(
    devices: tuple[tuple[str, DeviceMemory], ...],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[DeviceMemory, ...]:
    """Restrict *devices* to the ones ``CUDA_VISIBLE_DEVICES`` exposes."""
    source = os.environ if environ is None else environ
    visible = source.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return tuple(device for _, device in devices)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if not entries:
        raise GPUMemoryProbeError("CUDA_VISIBLE_DEVICES exposes no devices")
    by_index = {str(device.index): device for _, device in devices}
    by_uuid = {uuid: device for uuid, device in devices}
    selected: list[DeviceMemory] = []
    for entry in entries:
        device = by_index.get(entry) or by_uuid.get(entry)
        if device is None:
            raise GPUMemoryProbeError(
                f"CUDA_VISIBLE_DEVICES entry {entry!r} matches no device reported by nvidia-smi"
            )
        selected.append(device)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class GPUMemoryPrecondition:
    """How much device memory the *next* engine needs before it may be launched.

    The requirement is derived from configuration rather than hardcoded: vLLM
    reads ``--gpu-memory-utilization`` as a fraction of each device's *total*
    memory, so ``required_fraction`` is exactly the value the next engine will
    be started with, and the byte requirement follows from whatever card the
    deployment happens to run on.
    """

    probe: GPUMemoryProbe
    required_fraction: float
    timeout_seconds: float = DEFAULT_GPU_MEMORY_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_GPU_MEMORY_POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_fraction, bool)
            or not isinstance(self.required_fraction, (int, float))
            or not 0 < self.required_fraction <= 1
        ):
            raise ValueError("required_fraction must be in (0, 1]")
        _validate_timeout(self.timeout_seconds, name="gpu memory timeout")
        _validate_timeout(self.poll_interval_seconds, name="gpu memory poll interval")

    def required_bytes(self, device: DeviceMemory) -> int:
        return int(device.total_bytes * self.required_fraction)

    def shortfall(self) -> str | None:
        """Describe the worst under-provisioned device, or ``None`` if all fit."""
        worst: tuple[int, DeviceMemory] | None = None
        for device in self.probe.read():
            required = self.required_bytes(device)
            deficit = required - device.free_bytes
            if deficit > 0 and (worst is None or deficit > worst[0]):
                worst = (deficit, device)
        if worst is None:
            return None
        _, device = worst
        return (
            f"device {device.index} has {_format_gib(device.free_bytes)} free of "
            f"{_format_gib(device.total_bytes)} but the next engine needs "
            f"{_format_gib(self.required_bytes(device))} "
            f"(gpu_memory_utilization={self.required_fraction:g})"
        )


def _one_line(value: str) -> str:
    return " ".join(value.strip().split())


def _format_gib(value: int) -> str:
    return f"{value / _BYTES_PER_GIB:.2f} GiB"


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
        self._held = False

    @property
    def is_admitting(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and not self._held
                and self._activity.is_admitting
            )

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def try_begin(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and not self._held
                and self._activity.try_begin()
            )

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
            if not self._closed and not self._held:
                self._activity.start_admitting()

    def hold(self) -> None:
        """Keep admission closed across tuner recovery start/stop calls."""
        with self._lock:
            if not self._closed:
                self._held = True
                self._activity.stop_admitting()

    def release(self) -> None:
        """Release a startup hold after durable serving recovery completes."""
        with self._lock:
            if not self._closed:
                self._held = False
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
        gpu_memory: GPUMemoryPrecondition | None = None,
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
        self._gpu_memory = gpu_memory
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
        """Put vLLM into level-1 sleep, restoring the active draft on failure.

        Sleep is only reported complete once the device memory the next engine
        needs is actually free — ``/is_sleeping`` alone is not enough. See
        :meth:`_await_gpu_memory`.
        """
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
            self._await_gpu_memory(should_abort)
            self._check_abort(should_abort, "sleep")
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

    def _await_gpu_memory(self, should_abort: AbortCheck) -> None:
        """Block until the sleeping engine has actually given its memory back.

        ``/is_sleeping`` only reports vLLM's own bookkeeping. At sleep level 1
        the weights are offloaded to host memory, but the CUDA allocator pool
        the engine held is released asynchronously and is not reflected in that
        flag. The next engine sizes its KV cache from *free* device memory, so
        launching on the strength of ``/is_sleeping`` alone turns a slow release
        into an opaque CUDA out-of-memory failure inside a child process.

        Waiting is bounded by its own budget rather than the sleep deadline:
        the release is a separate physical event from vLLM accepting the sleep,
        and it is the caller's last chance to fail legibly.

        Raises:
            GPUMemoryNotReleased: If the requirement is still unmet at timeout.
            GPUMemoryProbeError: If free memory cannot be observed at all.
        """
        precondition = self._gpu_memory
        if precondition is None:
            return
        deadline = self._deadline(precondition.timeout_seconds, "gpu memory release")
        shortfall = precondition.shortfall()
        while shortfall is not None:
            self._check_abort(should_abort, "gpu memory release")
            try:
                remaining = deadline.remaining()
            except ControlTimeout:
                raise GPUMemoryNotReleased(
                    f"GPU memory was not released within "
                    f"{precondition.timeout_seconds:g}s of vLLM reporting sleep: "
                    f"{shortfall}"
                ) from None
            self._sleeper(min(precondition.poll_interval_seconds, remaining))
            shortfall = precondition.shortfall()

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
