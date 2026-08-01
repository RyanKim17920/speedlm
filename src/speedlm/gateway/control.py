"""Safe, synchronous runtime control for tuner-driven vLLM replacement."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final, Protocol

from speedlm.gateway.activity import ActivityTracker

logger = logging.getLogger(__name__)

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

#: Budget for :meth:`RuntimeController.restore`'s wake-in-place fast path.
#:
#: The fast path is a wake (measured WAKING->READY at 0.04s), one readiness
#: wait and one bounded canary completion.  It is bounded separately from the
#: restore deadline it runs inside so that a fast path which hangs cannot eat
#: the budget the full restart behind it still needs: on expiry the restart
#: runs with whatever the caller's own deadline still holds.  120s is roughly
#: twice a cold engine's launch->ready time, i.e. deliberately generous for an
#: operation that normally completes in well under a second.
DEFAULT_RESTORE_FAST_PATH_TIMEOUT_SECONDS: Final = 120.0

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

    def canary(self, *, timeout_seconds: float) -> None:
        """Run one bounded completion, raising unless the child answers it."""

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


class DraftSwapHTTP(Protocol):
    """Hot-swap draft weights in a live vLLM engine without restarting.

    The implementation pauses generation, sends a collective-RPC to the
    drafter worker to swap weights in place via the layerwise reload helpers,
    and resumes generation.  Requires ``VLLM_SERVER_DEV_MODE=1`` so that the
    ``/pause``, ``/resume`` and ``/collective_rpc`` endpoints are available.

    Failure is reported through two deliberately distinct exception types so
    that the caller can tell a swap that provably never touched the engine
    from one that may have left it half-mutated.  Anything a caller cannot
    prove was rejected before dispatch must be reported as
    :class:`DraftSwapCorrupted`.
    """

    def hot_swap_draft(
        self,
        weights_path: str,
        *,
        timeout_seconds: float,
    ) -> None:
        """Swap the drafter's weights from *weights_path* in place.

        Args:
            weights_path: The candidate draft *directory*.  Resolving it to
                individual shards is the worker extension's job, so this
                transport passes the directory through verbatim.
            timeout_seconds: Budget for the pause and the swap RPC.  Resuming
                gets its own independent budget: a swap that runs out of time
                must still never leave the engine paused.

        Raises:
            DraftSwapUnavailable: The swap provably did not run.
            DraftSwapCorrupted: The swap may have run, and the engine must be
                treated as serving-broken until it has been restarted.
        """


class DraftSwapUnavailable(RuntimeError):
    """Raised when a hot-swap was rejected before any weight was touched.

    This is an ordinary negative result: the engine is exactly as it was, so
    the caller may quietly fall back to a full restart.
    """


class DraftSwapCorrupted(RuntimeError):
    """Raised when a hot-swap may have left the engine unable to serve.

    Covers a failure inside the weight application, an indeterminate RPC
    outcome, a failed resume, and a post-swap verification that did not pass.
    The engine must be restarted before it admits traffic again.
    """


# vLLM dev-mode endpoints, all gated behind ``VLLM_SERVER_DEV_MODE=1`` and
# mounted under the same origin as the health/completions endpoints.
# ``/pause`` and ``/resume`` come from vLLM's RLHF dev router; ``mode=wait``
# drains in-flight requests without discarding the resident weights, which
# ``/sleep`` at level 1 would.
VLLM_PAUSE_ENDPOINT: Final = "/pause"
VLLM_RESUME_ENDPOINT: Final = "/resume"
VLLM_COLLECTIVE_RPC_ENDPOINT: Final = "/collective_rpc"
VLLM_PAUSE_MODE: Final = "wait"
#: Worker-side RPC implemented by ``speedlm.gateway.draft_swap``.
DRAFT_SWAP_RPC_METHOD: Final = "hot_swap_draft"
#: Resuming is a safety obligation, so it is budgeted independently of the
#: swap deadline it may be cleaning up after.
DRAFT_SWAP_RESUME_TIMEOUT_SECONDS: Final = 30.0


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
        draft_swap_http: DraftSwapHTTP | None = None,
        restore_fast_path_timeout_seconds: float = (
            DEFAULT_RESTORE_FAST_PATH_TIMEOUT_SECONDS
        ),
    ) -> None:
        if not str(active_draft):
            raise ValueError("active draft must be non-empty")
        _validate_timeout(poll_interval_seconds, name="poll interval")
        _validate_timeout(recovery_timeout_seconds, name="recovery timeout")
        _validate_timeout(
            restore_fast_path_timeout_seconds,
            name="restore fast path timeout",
        )
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
        self._restore_fast_path_timeout_seconds = restore_fast_path_timeout_seconds
        self._admissions_stopped = False
        self._sleeping = False
        self._draft_swap_http: DraftSwapHTTP | None = draft_swap_http
        # Whether anything may have altered the *contents* of the running
        # engine since it was last launched. ``_running_draft`` records which
        # draft the child was configured with; this records whether that record
        # can still be believed. Only a full restart clears it, because only a
        # restart replaces the process the mutation happened inside.
        self._engine_may_be_mutated = False

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
        """Replace the sleeping child with a candidate EAGLE-3 draft.

        If a ``draft_swap_http`` endpoint is configured and the engine is not
        sleeping, attempts an in-place hot-swap of the draft weights before
        falling back to a full restart.  Any hot-swap failure leaves
        ``_running_draft`` unchanged, so the restart below runs unconditionally;
        a failure that may have broken the engine also closes admission first.
        """
        deadline = self._deadline(timeout_seconds, "candidate start")
        try:
            self._check_abort(should_abort, "candidate start")
            deadline.check()
            if self._sleeping or self._running_draft != draft_directory:
                # Try hot-swap when engine is awake and draft actually changed.
                if (
                    not self._sleeping
                    and self._running_draft != draft_directory
                    and self._draft_swap_http is not None
                ):
                    self._try_hot_swap(draft_directory, deadline)
                if self._running_draft != draft_directory:
                    self._restart_and_wait(draft_directory, deadline)
                    self._running_draft = draft_directory
                    self._sleeping = False
                else:
                    self._http.wait_ready(timeout_seconds=deadline.remaining())
                    deadline.check()
            else:
                self._http.wait_ready(timeout_seconds=deadline.remaining())
                deadline.check()
            self._check_abort(should_abort, "candidate start")
            deadline.check()
        except Exception as error:
            self._recover_or_raise("candidate start", error)
            raise

    @property
    def supports_draft_hot_swap(self) -> bool:
        """Whether an in-place draft swap is wired up at all.

        True exactly when ``tuning.draft_hot_swap_enabled`` is set, because
        that flag is the only thing that supplies ``draft_swap_http``. Exposed
        so a caller can decide whether the mid-cycle wake below is worth doing
        without having to know about the config object.
        """
        return self._draft_swap_http is not None

    def wake_for_swap(self, *, timeout_seconds: float) -> None:
        """Wake vLLM mid-cycle, deliberately leaving admission closed.

        :meth:`start_candidate` can only hot-swap an engine whose drafter
        weights are resident, and level-1 sleep offloads exactly those weights.
        So a cycle that slept to free device memory for training has to wake
        before the swap is even expressible -- and it must wake *without*
        reopening the gateway, which is the one thing :meth:`wake` also does.
        Reopening here would let real traffic both preempt the idle cycle and
        interleave with the benchmark that follows, so the two wakes cannot be
        the same call.

        The wake restores the weights that were resident at sleep time, i.e.
        the stock active draft; the candidate is applied on top of it.

        Failure is deliberately *not* recovered: leaving the controller's
        sleeping bookkeeping untouched is what makes :meth:`start_candidate`
        fall back to a full restart, which restores serving anyway. Recovering
        here would restart the child twice for one failure.

        Raises:
            DraftSwapUnavailable: No swap endpoint is configured, so there is
                nothing this wake could enable.
        """
        if self._draft_swap_http is None:
            raise DraftSwapUnavailable(
                "no draft hot-swap endpoint is configured for this controller"
            )
        deadline = self._deadline(timeout_seconds, "wake for swap")
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

    @property
    def running_draft(self) -> DraftReference:
        """The draft the managed child is currently configured with."""
        return self._running_draft

    def matches_running_draft(self, draft: DraftReference) -> bool:
        """Whether the live, awake engine already serves *draft*.

        This is the shared predicate behind every "do not restart what is
        already running" decision, so that the controller and the benchmark's
        endpoint cannot disagree about what the child is doing. It is false
        whenever the engine is asleep -- its weights are offloaded, so it
        serves nothing -- or may have been mutated in place by a failed
        hot-swap.
        """
        return self._is_configured_with(draft) and not self._sleeping

    def _is_configured_with(self, draft: DraftReference) -> bool:
        """Whether the child process was launched with *draft* and untouched.

        Says nothing about whether the engine is awake. The comparison is on
        the string form: the same directory reaches this controller as a
        :class:`~pathlib.Path` from the artifact registry and as a ``str`` from
        a profile's warm-start draft, and those denote the same engine.
        """
        if self._engine_may_be_mutated:
            return False
        return str(self._running_draft) == str(draft)

    def note_external_restart(self, draft: DraftReference) -> None:
        """Record that another component restarted the managed child.

        The benchmark gate drives ``ThreadsafeProcessControl`` directly when it
        activates an arm, so without this the controller's ``_running_draft``
        would silently describe an engine that no longer exists -- and every
        decision built on it, including :meth:`restore`'s fast path below,
        would be reasoning about the wrong process. Callers must call this only
        *after* the replacement child has proved ready.
        """
        if not str(draft):
            raise ValueError("draft must be non-empty")
        self._running_draft = draft
        self._sleeping = False
        self._engine_may_be_mutated = False

    def restore(
        self,
        active_draft: DraftReference,
        *,
        timeout_seconds: float,
    ) -> None:
        """Return the child to the supplied durable active draft.

        A restart is a genuine fork/exec of vLLM -- measured at ~62.5s to
        launch-ready and ~100-105s including teardown -- and it is pure
        downtime, because ``restore`` only ever runs while the gateway's gate is
        closed. On job 369040 an idle cycle that was preempted 0.1s into
        extraction, having mutated nothing but one ``mkdir``, still paid 104.5s
        for a rollback that had nothing to roll back.

        So the restart is now conditional, on exactly the evidence that makes it
        unnecessary: :meth:`matches_running_draft` says the child is already
        configured with *active_draft*, is not asleep, and has not been touched
        in place. That is a claim about bookkeeping, so it is not trusted on its
        own -- the fast path additionally makes the engine *prove* it can serve,
        with a readiness wait and one bounded canary completion, before it is
        accepted. Anything less than a clean pass falls through to the restart.

        This is deliberately asymmetric. A skipped restart that should have
        happened would leave the wrong draft serving live traffic, which is far
        worse than 100s of downtime, so every uncertainty resolves towards the
        restart: unknown draft, sleeping engine, possible in-place mutation,
        readiness failure, canary failure, or the fast path exceeding its own
        (separately budgeted) deadline.
        """
        if not str(active_draft):
            raise ValueError("active draft must be non-empty")
        self._active_draft = active_draft
        deadline = self._deadline(timeout_seconds, "restore")
        if self._restore_in_place(active_draft, deadline):
            return
        try:
            self._restart_and_wait(active_draft, deadline)
        except Exception as error:
            # A failed restart may have stopped the child. Retry the same known
            # good draft with a fresh safety budget before reporting failure.
            self._recover_or_raise("restore", error)
            raise
        self._running_draft = active_draft
        self._sleeping = False

    def _restore_in_place(
        self,
        active_draft: DraftReference,
        deadline: _Deadline,
    ) -> bool:
        """Try to satisfy :meth:`restore` by waking, not respawning.

        Returns ``True`` only when the running engine has proved it already
        serves *active_draft*. Every failure returns ``False`` so the caller
        restarts, and none of them raise: a fast path that cannot be confirmed
        is an ordinary negative result, not a cycle failure.
        """
        if not self._is_configured_with(active_draft):
            return False
        try:
            budget = min(
                deadline.remaining(),
                self._restore_fast_path_timeout_seconds,
            )
            fast = self._deadline(budget, "restore fast path")
            if self._sleeping:
                self._wake_now(fast)
            self._http.wait_ready(timeout_seconds=fast.remaining())
            fast.check()
            self._http.canary(timeout_seconds=fast.remaining())
            fast.check()
        except Exception:
            logger.warning(
                "restore could not confirm the running engine still serves %s; "
                "falling back to a full restart",
                active_draft,
                exc_info=True,
            )
            return False
        logger.info(
            "restore reused the running engine for %s without a restart",
            active_draft,
        )
        return True

    def _wake_now(self, deadline: _Deadline) -> None:
        """Wake the sleeping child without touching admission."""
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
        # A fresh process cannot carry a previous one's in-place mutation.
        self._engine_may_be_mutated = False

    def _try_hot_swap(
        self, draft_directory: DraftReference, deadline: _Deadline
    ) -> None:
        """Attempt an in-place draft weight hot-swap.

        Success is only recorded once the mutated engine has proved it can
        still serve: ``wait_ready`` plus one bounded canary completion. An RPC
        that reports success but produces a broken drafter is therefore a
        failure here, not a silent regression the gate has to discover later.

        Both failure modes leave ``_running_draft`` unchanged, which is what
        makes :meth:`start_candidate` restart on the very next statement. They
        differ in blast radius, so they differ in how loudly they report:

        * A swap that provably never ran leaves the engine exactly as it was.
          The restart is ordinary rollback, so a warning is enough.
        * Anything else may have half-applied weights or failed verification.
          The engine is serving-broken until the restart lands, so admission is
          closed first and the failure is logged at error level.
        """
        swap_http = self._draft_swap_http
        if swap_http is None or self._sleeping:
            return
        try:
            deadline.check()
            swap_http.hot_swap_draft(
                str(draft_directory),
                timeout_seconds=deadline.remaining(),
            )
            deadline.check()
        except DraftSwapUnavailable as exc:
            logger.warning(
                "draft hot-swap did not run for %s (%s: %s); the engine is "
                "untouched and a full restart will follow",
                draft_directory,
                type(exc).__name__,
                exc,
            )
            return
        except Exception as exc:
            self._fail_hot_swap(draft_directory, exc, "weight application")
            return
        try:
            self._http.wait_ready(timeout_seconds=deadline.remaining())
            deadline.check()
            self._http.canary(timeout_seconds=deadline.remaining())
            deadline.check()
        except Exception as exc:
            self._fail_hot_swap(draft_directory, exc, "post-swap verification")
            return
        self._running_draft = draft_directory
        self._sleeping = False
        logger.info("hot-swap succeeded for draft %s", draft_directory)

    def _fail_hot_swap(
        self,
        draft_directory: DraftReference,
        error: Exception,
        phase: str,
    ) -> None:
        """Close the gate after a swap that may have broken the live engine."""
        # ``_running_draft`` still names the pre-swap draft, but the engine may
        # no longer agree with it, so no cheap path may believe it again until
        # a restart has replaced the process.
        self._engine_may_be_mutated = True
        self._stop_admitting()
        logger.error(
            "draft hot-swap failed during %s for %s (%s: %s); the engine may be "
            "unable to serve, so admission is closed and a full restart is forced",
            phase,
            draft_directory,
            type(error).__name__,
            error,
        )

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
