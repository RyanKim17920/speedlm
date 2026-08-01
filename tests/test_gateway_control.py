from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier, Thread

import pytest

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.control import (
    VLLM_SLEEP_ENDPOINT,
    VLLM_SLEEP_LEVEL,
    VLLM_SLEEP_MODE,
    VLLM_WAKE_ENDPOINT,
    AdmissionGate,
    ControlAborted,
    ControlTimeout,
    DeviceMemory,
    DraftSwapCorrupted,
    DraftSwapUnavailable,
    GPUMemoryNotReleased,
    GPUMemoryPrecondition,
    GPUMemoryProbeError,
    NvidiaSmiMemoryProbe,
    RuntimeController,
)
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import RuntimeController as RuntimeControllerProtocol


@dataclass
class FakeClock:
    now: float = 0.0
    sleep_hook: Callable[[], None] | None = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.sleep_hook is not None:
            self.sleep_hook()


@dataclass
class FakeAdmission:
    admitting: bool = True
    calls: list[str] = field(default_factory=list)

    def stop_admitting(self) -> None:
        self.calls.append("stop")
        self.admitting = False

    def start_admitting(self) -> None:
        self.calls.append("start")
        self.admitting = True


@dataclass(frozen=True)
class HTTPCall:
    endpoint: str
    timeout_seconds: float
    query: Mapping[str, str] | None


@dataclass
class FakeHTTP:
    clock: FakeClock
    post_calls: list[HTTPCall] = field(default_factory=list)
    ready_timeouts: list[float] = field(default_factory=list)
    canary_timeouts: list[float] = field(default_factory=list)
    fail_post: set[str] = field(default_factory=set)
    fail_ready_count: int = 0
    fail_canary_count: int = 0
    advance_post: float = 0.0
    advance_ready: float = 0.0
    sleeping_waits: list[tuple[bool, float]] = field(default_factory=list)
    fail_sleeping_wait: bool = False

    def post(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        query: Mapping[str, str] | None = None,
    ) -> None:
        self.post_calls.append(HTTPCall(endpoint, timeout_seconds, query))
        self.clock.now += self.advance_post
        self.advance_post = 0.0
        if endpoint in self.fail_post:
            raise RuntimeError(f"{endpoint} failed")

    def wait_ready(self, *, timeout_seconds: float) -> None:
        self.ready_timeouts.append(timeout_seconds)
        self.clock.now += self.advance_ready
        self.advance_ready = 0.0
        if self.fail_ready_count:
            self.fail_ready_count -= 1
            raise RuntimeError("readiness failed")

    def canary(self, *, timeout_seconds: float) -> None:
        self.canary_timeouts.append(timeout_seconds)
        if self.fail_canary_count:
            self.fail_canary_count -= 1
            raise RuntimeError("canary failed")

    def wait_sleeping(
        self,
        sleeping: bool,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        self.sleeping_waits.append((sleeping, timeout_seconds))
        if should_abort():
            raise ControlAborted("sleep-state wait aborted")
        if self.fail_sleeping_wait:
            raise RuntimeError("sleep-state wait failed")


@dataclass(frozen=True)
class ProcessCall:
    draft: Path | str
    timeout_seconds: float


@dataclass
class FakeProcess:
    clock: FakeClock
    running_draft: Path | str
    calls: list[ProcessCall] = field(default_factory=list)
    fail_drafts: set[Path | str] = field(default_factory=set)
    advance_restart: float = 0.0
    running: bool = True

    def restart(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
    ) -> None:
        self.calls.append(ProcessCall(draft, timeout_seconds))
        self.running = False
        self.clock.now += self.advance_restart
        self.advance_restart = 0.0
        if draft in self.fail_drafts:
            raise RuntimeError(f"restart failed for {draft}")
        self.running_draft = draft
        self.running = True


@dataclass
class FakeDraftSwap:
    """A ``DraftSwapHTTP`` whose outcome each test chooses explicitly."""

    calls: list[tuple[str, float]] = field(default_factory=list)
    error: Exception | None = None

    def hot_swap_draft(self, weights_path: str, *, timeout_seconds: float) -> None:
        self.calls.append((weights_path, timeout_seconds))
        if self.error is not None:
            raise self.error


@dataclass
class Rig:
    controller: RuntimeController
    activity: ActivityTracker
    admission: FakeAdmission
    http: FakeHTTP
    process: FakeProcess
    clock: FakeClock
    draft_swap: FakeDraftSwap | None = None


def make_rig(
    *,
    active_draft: Path | str = "base-draft",
    gpu_memory: GPUMemoryPrecondition | None = None,
    draft_swap: FakeDraftSwap | None = None,
) -> Rig:
    clock = FakeClock()
    activity = ActivityTracker(clock=clock)
    admission = FakeAdmission()
    http = FakeHTTP(clock)
    process = FakeProcess(clock, running_draft=active_draft)
    controller = RuntimeController(
        activity=activity,
        admission=admission,
        http=http,
        process=process,
        active_draft=active_draft,
        clock=clock,
        sleeper=clock.sleep,
        poll_interval_seconds=0.1,
        recovery_timeout_seconds=5.0,
        gpu_memory=gpu_memory,
        draft_swap_http=draft_swap,
    )
    return Rig(controller, activity, admission, http, process, clock, draft_swap)


def quiesce(rig: Rig) -> None:
    rig.controller.quiesce(timeout_seconds=1.0, should_abort=lambda: False)


def test_controller_implements_tuner_protocol() -> None:
    controller: RuntimeControllerProtocol = make_rig().controller
    assert controller is not None


def test_admission_close_and_begin_are_one_atomic_decision() -> None:
    for _ in range(100):
        activity = ActivityTracker()
        admission = AdmissionGate(activity)
        barrier = Barrier(3)
        admitted: list[bool] = []

        def close_gate(
            current_barrier: Barrier = barrier,
            current_admission: AdmissionGate = admission,
        ) -> None:
            current_barrier.wait()
            current_admission.stop_admitting()

        def begin_request(
            current_barrier: Barrier = barrier,
            current_admission: AdmissionGate = admission,
            results: list[bool] = admitted,
        ) -> None:
            current_barrier.wait()
            results.append(current_admission.try_begin())

        close_thread = Thread(target=close_gate)
        begin_thread = Thread(target=begin_request)
        close_thread.start()
        begin_thread.start()
        barrier.wait()
        close_thread.join()
        begin_thread.join()

        assert admitted in ([False], [True])
        assert not admission.is_admitting
        assert activity.in_flight == int(admitted[0])
        if admitted[0]:
            activity.end()
        assert activity.in_flight == 0


def test_rejected_request_updates_watermark_and_preempts_idle_guard() -> None:
    clock = FakeClock()
    activity = ActivityTracker(clock=clock)
    admission = AdmissionGate(activity)
    detector = IdleDetector(
        activity,
        threshold_seconds=1.0,
        clock=clock,
    )
    clock.now = 2.0
    guard = detector.arm()
    admission.stop_admitting()
    clock.now = 3.0

    assert not admission.try_begin()

    assert activity.in_flight == 0
    assert activity.last_activity == 3.0
    assert guard.is_preempted


def test_startup_hold_cannot_be_reopened_by_runtime_recovery() -> None:
    activity = ActivityTracker()
    admission = AdmissionGate(activity)

    admission.hold()
    admission.start_admitting()

    assert not admission.is_admitting
    assert not admission.try_begin()

    admission.release()

    assert admission.is_admitting
    assert admission.try_begin()
    activity.end()


def test_quiesce_stops_admission_and_waits_for_in_flight_to_reach_zero() -> None:
    rig = make_rig()
    rig.activity.begin()
    sleeps = 0

    def finish_request_after_two_polls() -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            rig.activity.end()

    rig.clock.sleep_hook = finish_request_after_two_polls

    quiesce(rig)

    assert sleeps == 2
    assert rig.activity.in_flight == 0
    assert not rig.admission.admitting
    assert rig.admission.calls == ["stop"]


def test_quiesce_aborts_and_reopens_admission() -> None:
    rig = make_rig()
    rig.activity.begin()
    checks = 0

    def should_abort() -> bool:
        nonlocal checks
        checks += 1
        return checks == 3

    with pytest.raises(ControlAborted, match="quiesce aborted"):
        rig.controller.quiesce(timeout_seconds=1.0, should_abort=should_abort)

    assert rig.admission.admitting
    assert rig.admission.calls == ["stop", "start"]
    assert rig.process.running


def test_sleep_uses_level_one_endpoint() -> None:
    rig = make_rig()
    quiesce(rig)

    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    assert rig.http.post_calls == [
        HTTPCall(
            VLLM_SLEEP_ENDPOINT,
            2.0,
            {"level": VLLM_SLEEP_LEVEL, "mode": VLLM_SLEEP_MODE},
        )
    ]
    assert rig.http.sleeping_waits == [(True, 2.0)]
    assert not rig.admission.admitting


def test_sleep_failure_restores_active_service_and_admission() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.http.fail_post.add(VLLM_SLEEP_ENDPOINT)

    with pytest.raises(RuntimeError, match="/sleep failed"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    assert [call.draft for call in rig.process.calls] == ["base-draft"]
    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.http.ready_timeouts
    assert rig.admission.admitting


def test_sleep_state_confirmation_failure_restores_service() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.http.fail_sleeping_wait = True

    with pytest.raises(RuntimeError, match="sleep-state wait failed"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    assert rig.http.sleeping_waits == [(True, 2.0)]
    assert [call.draft for call in rig.process.calls] == ["base-draft"]
    assert rig.admission.admitting


def test_abort_after_sleep_restores_service() -> None:
    rig = make_rig()
    quiesce(rig)
    checks = iter((False, True))

    with pytest.raises(ControlAborted, match="aborted"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: next(checks))

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.admission.admitting


def test_candidate_start_failure_rolls_back_to_active_draft() -> None:
    rig = make_rig()
    candidate = Path("/artifacts/candidate")
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)
    rig.process.fail_drafts.add(candidate)

    with pytest.raises(RuntimeError, match="restart failed"):
        rig.controller.start_candidate(
            candidate,
            timeout_seconds=3.0,
            should_abort=lambda: False,
        )

    assert [call.draft for call in rig.process.calls] == [candidate, "base-draft"]
    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.admission.admitting


def test_candidate_readiness_failure_rolls_back() -> None:
    rig = make_rig()
    candidate = Path("/artifacts/candidate")
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)
    rig.http.fail_ready_count = 1

    with pytest.raises(RuntimeError, match="readiness failed"):
        rig.controller.start_candidate(
            candidate,
            timeout_seconds=3.0,
            should_abort=lambda: False,
        )

    assert [call.draft for call in rig.process.calls] == [candidate, "base-draft"]
    assert rig.process.running
    assert rig.process.running_draft == "base-draft"


def test_restore_restarts_supplied_draft_and_confirms_readiness() -> None:
    rig = make_rig()
    active = Path("/artifacts/active")

    rig.controller.restore(active, timeout_seconds=4.0)

    assert [call.draft for call in rig.process.calls] == [active]
    assert rig.http.ready_timeouts == [4.0]
    assert rig.process.running
    assert rig.process.running_draft == active


def test_restore_retries_after_process_failure_so_child_is_not_left_stopped() -> None:
    rig = make_rig()
    original_restart = rig.process.restart
    failed = False

    def fail_once(draft: Path | str, *, timeout_seconds: float) -> None:
        nonlocal failed
        if not failed:
            failed = True
            rig.process.calls.append(ProcessCall(draft, timeout_seconds))
            rig.process.running = False
            raise RuntimeError("transient restart failure")
        original_restart(draft, timeout_seconds=timeout_seconds)

    rig.process.restart = fail_once  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="transient restart failure"):
        rig.controller.restore("base-draft", timeout_seconds=2.0)

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert [call.draft for call in rig.process.calls] == ["base-draft", "base-draft"]


def test_wake_calls_endpoint_confirms_readiness_and_reopens_admission() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    rig.controller.wake(timeout_seconds=3.0)

    assert rig.http.post_calls[-1] == HTTPCall(VLLM_WAKE_ENDPOINT, 3.0, None)
    assert rig.http.sleeping_waits[-1] == (False, 3.0)
    assert rig.http.ready_timeouts == [3.0]
    assert rig.admission.admitting
    assert rig.process.running


def test_wake_state_confirmation_failure_recovers_before_reopening() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)
    rig.http.fail_sleeping_wait = True

    with pytest.raises(RuntimeError, match="sleep-state wait failed"):
        rig.controller.wake(timeout_seconds=3.0)

    assert rig.http.sleeping_waits[-1] == (False, 3.0)
    assert [call.draft for call in rig.process.calls] == ["base-draft"]
    assert rig.http.ready_timeouts
    assert rig.admission.admitting


def test_quiesce_honors_timeout_and_reopens_admission() -> None:
    rig = make_rig()
    rig.activity.begin()

    with pytest.raises(ControlTimeout, match="quiesce timed out"):
        rig.controller.quiesce(timeout_seconds=0.2, should_abort=lambda: False)

    assert rig.clock.now == pytest.approx(0.2)
    assert rig.admission.admitting
    assert rig.process.running


def test_sleep_honors_timeout_and_restores_service() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.http.advance_post = 0.2

    with pytest.raises(ControlTimeout, match="sleep timed out"):
        rig.controller.sleep(timeout_seconds=0.1, should_abort=lambda: False)

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.admission.admitting


def test_candidate_start_honors_timeout_and_rolls_back() -> None:
    rig = make_rig()
    candidate = Path("/artifacts/candidate")
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=1.0, should_abort=lambda: False)
    rig.process.advance_restart = 0.2

    with pytest.raises(ControlTimeout, match="candidate start timed out"):
        rig.controller.start_candidate(
            candidate,
            timeout_seconds=0.1,
            should_abort=lambda: False,
        )

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.admission.admitting


def test_restore_honors_timeout_but_recovers_service() -> None:
    rig = make_rig()
    rig.process.advance_restart = 0.2

    with pytest.raises(ControlTimeout, match="restore timed out"):
        rig.controller.restore("base-draft", timeout_seconds=0.1)

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"


def test_wake_honors_timeout_but_recovers_service() -> None:
    rig = make_rig()
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=1.0, should_abort=lambda: False)
    rig.http.advance_post = 0.2

    with pytest.raises(ControlTimeout, match="wake timed out"):
        rig.controller.wake(timeout_seconds=0.1)

    assert rig.process.running
    assert rig.process.running_draft == "base-draft"
    assert rig.admission.admitting


@pytest.mark.parametrize("method", ["quiesce", "sleep", "candidate", "restore", "wake"])
def test_every_method_rejects_non_positive_timeout_without_stopping_child(
    method: str,
) -> None:
    rig = make_rig()

    with pytest.raises(ValueError, match="timeout must be a positive number"):
        if method == "quiesce":
            rig.controller.quiesce(timeout_seconds=0, should_abort=lambda: False)
        elif method == "sleep":
            rig.controller.sleep(timeout_seconds=0, should_abort=lambda: False)
        elif method == "candidate":
            rig.controller.start_candidate(
                Path("/candidate"),
                timeout_seconds=0,
                should_abort=lambda: False,
            )
        elif method == "restore":
            rig.controller.restore("base-draft", timeout_seconds=0)
        else:
            rig.controller.wake(timeout_seconds=0)

    assert rig.process.running
    assert rig.process.calls == []


GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


@dataclass
class FakeMemoryProbe:
    """Replay a scripted sequence of device-memory readings."""

    readings: list[tuple[DeviceMemory, ...]]
    error: Exception | None = None
    reads: int = 0

    def read(self) -> tuple[DeviceMemory, ...]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return self.readings[min(self.reads - 1, len(self.readings) - 1)]


def device(free_gib: float, *, index: int = 0, total_gib: float = 80.0) -> DeviceMemory:
    return DeviceMemory(
        index=index,
        free_bytes=int(free_gib * GIB),
        total_bytes=int(total_gib * GIB),
    )


def precondition(
    probe: FakeMemoryProbe,
    *,
    required_fraction: float = 0.80,
    timeout_seconds: float = 3.0,
) -> GPUMemoryPrecondition:
    return GPUMemoryPrecondition(
        probe=probe,
        required_fraction=required_fraction,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=0.5,
    )


def test_sleep_waits_until_the_engine_actually_releases_device_memory() -> None:
    # Level-1 sleep offloads weights but frees the CUDA pool asynchronously,
    # so /is_sleeping can be true while the device is still occupied.
    probe = FakeMemoryProbe(
        readings=[
            (device(1.13),),
            (device(40.0),),
            (device(79.0),),
        ]
    )
    rig = make_rig(gpu_memory=precondition(probe))
    quiesce(rig)

    rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    assert probe.reads == 3
    assert rig.clock.now == pytest.approx(1.0)
    assert rig.process.calls == []
    assert rig.process.running


def test_sleep_fails_closed_when_device_memory_is_never_released() -> None:
    # Removing the precondition makes this sleep succeed and the cycle proceed
    # straight into the child-side CUDA OOM this check exists to prevent.
    probe = FakeMemoryProbe(readings=[(device(1.13),)])
    rig = make_rig(gpu_memory=precondition(probe))
    quiesce(rig)

    with pytest.raises(GPUMemoryNotReleased) as excinfo:
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    message = str(excinfo.value)
    assert "GPU memory was not released within 3s" in message
    assert "device 0 has 1.13 GiB free of 80.00 GiB" in message
    assert "needs 64.00 GiB (gpu_memory_utilization=0.8)" in message
    # A cycle that cannot get memory rolls back like any other cycle failure.
    assert rig.process.calls[-1].draft == "base-draft"
    assert rig.process.running
    assert rig.admission.admitting


def test_sleep_reports_the_worst_device_of_several() -> None:
    probe = FakeMemoryProbe(
        readings=[(device(79.0, index=0), device(2.0, index=1))]
    )
    rig = make_rig(gpu_memory=precondition(probe))
    quiesce(rig)

    with pytest.raises(GPUMemoryNotReleased, match="device 1 has 2.00 GiB free"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)


def test_memory_requirement_is_derived_from_each_device_total() -> None:
    check = precondition(FakeMemoryProbe(readings=[]), required_fraction=0.5)

    assert check.required_bytes(device(0.0, total_gib=80.0)) == 40 * GIB
    assert check.required_bytes(device(0.0, total_gib=24.0)) == 12 * GIB


def test_sleep_fails_closed_when_device_memory_cannot_be_observed() -> None:
    probe = FakeMemoryProbe(readings=[], error=GPUMemoryProbeError("nvidia-smi failed"))
    rig = make_rig(gpu_memory=precondition(probe))
    quiesce(rig)

    with pytest.raises(GPUMemoryProbeError, match="nvidia-smi failed"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: False)

    assert rig.process.running
    assert rig.admission.admitting


def test_memory_wait_aborts_on_preemption_and_restores_service() -> None:
    probe = FakeMemoryProbe(readings=[(device(1.13),)])
    rig = make_rig(gpu_memory=precondition(probe))
    quiesce(rig)

    with pytest.raises(ControlAborted, match="gpu memory release aborted"):
        rig.controller.sleep(timeout_seconds=2.0, should_abort=lambda: probe.reads >= 2)

    assert rig.process.running
    assert rig.admission.admitting


@pytest.mark.parametrize(
    ("fraction", "timeout", "interval"),
    [(0.0, 1.0, 1.0), (1.5, 1.0, 1.0), (0.8, 0.0, 1.0), (0.8, 1.0, 0.0)],
)
def test_precondition_rejects_nonsensical_configuration(
    fraction: float, timeout: float, interval: float
) -> None:
    with pytest.raises(ValueError):
        GPUMemoryPrecondition(
            probe=FakeMemoryProbe(readings=[]),
            required_fraction=fraction,
            timeout_seconds=timeout,
            poll_interval_seconds=interval,
        )


@dataclass(frozen=True)
class FakeCompleted:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def stub_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
    result: FakeCompleted | Exception,
) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> FakeCompleted:
        commands.append(command)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("speedlm.gateway.control.subprocess.run", fake_run)
    return commands


NVIDIA_SMI_TWO_GPUS = (
    "0, GPU-aaaa, 81559, 1156\n"
    "1, GPU-bbbb, 81559, 81000\n"
)


def test_nvidia_smi_probe_parses_mib_rows_into_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = stub_nvidia_smi(monkeypatch, FakeCompleted(0, NVIDIA_SMI_TWO_GPUS))

    devices = NvidiaSmiMemoryProbe(environ={}).read()

    assert commands[0][0] == "nvidia-smi"
    assert devices == (
        DeviceMemory(index=0, free_bytes=1156 * MIB, total_bytes=81559 * MIB),
        DeviceMemory(index=1, free_bytes=81000 * MIB, total_bytes=81559 * MIB),
    )


@pytest.mark.parametrize("visible", ["1", "GPU-bbbb", " 1 "])
def test_nvidia_smi_probe_honours_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch, visible: str
) -> None:
    stub_nvidia_smi(monkeypatch, FakeCompleted(0, NVIDIA_SMI_TWO_GPUS))

    devices = NvidiaSmiMemoryProbe(environ={"CUDA_VISIBLE_DEVICES": visible}).read()

    assert [item.index for item in devices] == [1]


@pytest.mark.parametrize(
    ("environ", "match"),
    [
        ({"CUDA_VISIBLE_DEVICES": "7"}, "matches no device"),
        ({"CUDA_VISIBLE_DEVICES": ""}, "exposes no devices"),
    ],
)
def test_nvidia_smi_probe_refuses_unresolvable_visible_devices(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], match: str
) -> None:
    stub_nvidia_smi(monkeypatch, FakeCompleted(0, NVIDIA_SMI_TWO_GPUS))

    with pytest.raises(GPUMemoryProbeError, match=match):
        NvidiaSmiMemoryProbe(environ=environ).read()


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (FileNotFoundError("nvidia-smi"), "not installed"),
        (OSError("permission denied"), "could not execute"),
        (FakeCompleted(9, "", "driver not loaded"), "driver not loaded"),
        (FakeCompleted(0, ""), "reported no GPUs"),
        (FakeCompleted(0, "0, GPU-aaaa, lots, 1156\n"), "unparsable"),
        (FakeCompleted(0, "0, GPU-aaaa, 81559\n"), "unparsable"),
    ],
)
def test_nvidia_smi_probe_reports_unusable_readings_as_probe_errors(
    monkeypatch: pytest.MonkeyPatch, result: FakeCompleted | Exception, match: str
) -> None:
    stub_nvidia_smi(monkeypatch, result)

    with pytest.raises(GPUMemoryProbeError, match=match):
        NvidiaSmiMemoryProbe(environ={}).read()


# --- draft hot-swap -------------------------------------------------------


CANDIDATE_DRAFT = Path("/candidate/draft")


def start_candidate(rig: Rig, draft: Path = CANDIDATE_DRAFT) -> None:
    rig.controller.start_candidate(
        draft,
        timeout_seconds=10.0,
        should_abort=lambda: False,
    )


def test_a_verified_hot_swap_replaces_the_restart() -> None:
    rig = make_rig(draft_swap=FakeDraftSwap())

    start_candidate(rig)

    assert rig.draft_swap is not None
    assert [call[0] for call in rig.draft_swap.calls] == [str(CANDIDATE_DRAFT)]
    # The whole point of the swap is that the process is never replaced.
    assert rig.process.calls == []
    # Success is only recorded after the mutated engine answers a canary.
    assert len(rig.http.canary_timeouts) == 1


def test_a_hot_swap_that_never_ran_falls_back_to_a_quiet_restart() -> None:
    rig = make_rig(
        draft_swap=FakeDraftSwap(error=DraftSwapUnavailable("route is absent")),
    )

    start_candidate(rig)

    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
    # Nothing was mutated, so the gate must not be slammed shut on the way out.
    assert rig.admission.calls == []
    assert rig.http.canary_timeouts == []


def test_a_mid_mutation_failure_closes_admission_and_forces_the_restart() -> None:
    rig = make_rig(
        draft_swap=FakeDraftSwap(error=DraftSwapCorrupted("half-applied")),
    )

    start_candidate(rig)

    assert rig.admission.calls[0] == "stop"
    assert not rig.admission.admitting
    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
    # A possibly-broken engine must never be verified into looking healthy.
    assert rig.http.canary_timeouts == []


def test_a_failed_canary_restarts_rather_than_recording_success() -> None:
    rig = make_rig(draft_swap=FakeDraftSwap())
    rig.http.fail_canary_count = 1

    start_candidate(rig)

    assert len(rig.http.canary_timeouts) == 1
    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
    assert rig.admission.calls[0] == "stop"


def test_a_failed_readiness_probe_restarts_rather_than_recording_success() -> None:
    rig = make_rig(draft_swap=FakeDraftSwap())
    rig.http.fail_ready_count = 1

    start_candidate(rig)

    # The canary is never reached: readiness gates it.
    assert rig.http.canary_timeouts == []
    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
    assert rig.admission.calls[0] == "stop"


def test_no_swap_client_leaves_the_restart_path_exactly_as_it_was() -> None:
    rig = make_rig()

    start_candidate(rig)

    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
    assert rig.http.canary_timeouts == []


def test_a_sleeping_engine_is_never_hot_swapped() -> None:
    rig = make_rig(draft_swap=FakeDraftSwap())
    quiesce(rig)
    rig.controller.sleep(timeout_seconds=10.0, should_abort=lambda: False)

    start_candidate(rig)

    assert rig.draft_swap is not None
    # Swapping weights into an engine whose weights are offloaded is nonsense.
    assert rig.draft_swap.calls == []
    assert [call.draft for call in rig.process.calls] == [CANDIDATE_DRAFT]
