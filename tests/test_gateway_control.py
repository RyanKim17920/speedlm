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
    fail_post: set[str] = field(default_factory=set)
    fail_ready_count: int = 0
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
class Rig:
    controller: RuntimeController
    activity: ActivityTracker
    admission: FakeAdmission
    http: FakeHTTP
    process: FakeProcess
    clock: FakeClock


def make_rig(*, active_draft: Path | str = "base-draft") -> Rig:
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
    )
    return Rig(controller, activity, admission, http, process, clock)


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
