from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import pytest

from speedlm.config import SpeedLMConfig
from speedlm.gateway.activity import ActivityTracker
from speedlm.traces.store import TraceStats
from speedlm.training.base import BackendInfo
from speedlm.tuner.eagle3 import TraceSnapshot
from speedlm.tuner.idle import ActivitySource, TuningPreempted
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    CycleResult,
    GateResult,
)
from speedlm.tuner.service import (
    TunerService,
    TunerServiceConfigurationError,
    build_tuner_orchestrator,
    create_tuner_service,
)

VERIFIER = "openai/gpt-oss-20b"
DRAFT = "RedHatAI/gpt-oss-20b-speculator.eagle3"


@dataclass(frozen=True, slots=True)
class FakePrepared:
    snapshot: TraceSnapshot


@dataclass
class FakeBackend:
    verifier_model: str = VERIFIER
    block_prepare: bool = False
    entered_prepare: Event = field(default_factory=Event)
    calls: list[str] = field(default_factory=list)

    def describe(self) -> BackendInfo:
        return BackendInfo(
            verifier_model=self.verifier_model,
            draft_model=DRAFT,
            from_pretrained=DRAFT,
            training_params={"steps": 1},
        )

    def prepare(
        self,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> FakePrepared:
        self.calls.append("prepare")
        self.entered_prepare.set()
        while self.block_prepare:
            if should_abort():
                raise TuningPreempted("fake training was preempted")
            time.sleep(0.001)
        return FakePrepared(TraceSnapshot(work_dir / "snapshot", "trace-hash"))

    def extract(
        self,
        prepared: object,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del prepared, should_abort
        self.calls.append("extract")
        return work_dir / "hidden"

    def train(
        self,
        extracted: object,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del extracted, should_abort
        self.calls.append("train")
        return work_dir / "checkpoint"

    def materialize(
        self,
        trained: object,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del trained, should_abort
        self.calls.append("materialize")
        candidate = work_dir / "candidate"
        candidate.mkdir()
        (candidate / "weights.bin").write_bytes(b"candidate")
        return candidate

    def validate(
        self,
        artifact: object,
        *,
        should_abort: Callable[[], bool],
    ) -> None:
        del artifact, should_abort
        self.calls.append("validate")


@dataclass
class FakeGate:
    calls: int = 0

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        del candidate_draft, timeout_seconds
        self.calls += 1
        if should_abort():
            raise TuningPreempted("fake benchmark was preempted")
        return GateResult(False, "test rejection")


@dataclass
class FakeRuntime:
    calls: list[str] = field(default_factory=list)

    def quiesce(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        del timeout_seconds
        self.calls.append("quiesce")
        if should_abort():
            raise TuningPreempted("fake quiesce was preempted")

    def sleep(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        del timeout_seconds
        self.calls.append("sleep")
        if should_abort():
            raise TuningPreempted("fake sleep was preempted")

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        del draft_directory, timeout_seconds
        self.calls.append("start_candidate")
        if should_abort():
            raise TuningPreempted("fake candidate start was preempted")

    def restore(
        self,
        active_draft: Path | str,
        *,
        timeout_seconds: float,
    ) -> None:
        del active_draft, timeout_seconds
        self.calls.append("restore")

    def wake(self, *, timeout_seconds: float) -> None:
        del timeout_seconds
        self.calls.append("wake")


class FakeTraces:
    def __init__(self, count: int) -> None:
        self._count = count
        self._lock = Lock()

    def set_count(self, count: int) -> None:
        with self._lock:
            self._count = count

    def stats(self) -> TraceStats:
        with self._lock:
            count = self._count
        return TraceStats(
            count=count,
            tokens=count * 10,
            oldest=1.0 if count else None,
            newest=float(count) if count else None,
        )


@dataclass
class ExplodingRunner:
    calls: int = 0
    recoveries: int = 0

    def run_once(self) -> CycleResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("training exploded")
        return CycleResult(CycleOutcome.REJECTED)

    def recover(self) -> tuple[str, ...]:
        self.recoveries += 1
        return ()


def _config() -> SpeedLMConfig:
    return SpeedLMConfig(
        model=VERIFIER,
        idle_threshold_seconds=0.01,
    )


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


def _service(
    tmp_path: Path,
    *,
    traces: FakeTraces,
    backend: FakeBackend | None = None,
    activity: ActivityTracker | None = None,
    enabled: bool | None = True,
    min_trace_records: int = 2,
) -> tuple[TunerService, FakeBackend, FakeGate, FakeRuntime, ActivityTracker]:
    actual_backend = backend or FakeBackend()
    actual_activity = activity or ActivityTracker()
    gate = FakeGate()
    runtime = FakeRuntime()
    run_number = 0

    def run_id() -> str:
        nonlocal run_number
        run_number += 1
        return f"run-{run_number}"

    service = create_tuner_service(
        _config(),
        activity=actual_activity,
        traces=traces,
        backend=actual_backend,
        gate=gate,
        runtime=runtime,
        enabled=enabled,
        min_trace_records=min_trace_records,
        poll_interval_seconds=0.005,
        home=tmp_path,
        run_id_factory=run_id,
    )
    return service, actual_backend, gate, runtime, actual_activity


def test_idle_threshold_with_enough_traces_runs_exactly_one_cycle(
    tmp_path: Path,
) -> None:
    service, backend, gate, _, _ = _service(tmp_path, traces=FakeTraces(2))
    service.start()
    try:
        _wait_until(lambda: gate.calls == 1)
        time.sleep(0.04)
        assert gate.calls == 1
        assert backend.calls == [
            "prepare",
            "extract",
            "train",
            "materialize",
            "validate",
        ]
    finally:
        service.stop(timeout_seconds=1.0)


def test_not_enough_traces_does_not_start_a_cycle(tmp_path: Path) -> None:
    service, backend, gate, _, _ = _service(tmp_path, traces=FakeTraces(1))
    service.start()
    try:
        time.sleep(0.05)
        assert gate.calls == 0
        assert backend.calls == []
        assert service.is_running
    finally:
        service.stop(timeout_seconds=1.0)


def test_traffic_arriving_mid_cycle_preempts_and_restores_serving(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(block_prepare=True)
    service, _, gate, runtime, activity = _service(
        tmp_path,
        traces=FakeTraces(2),
        backend=backend,
    )
    service.start()
    try:
        assert backend.entered_prepare.wait(1.0)
        activity.begin()
        _wait_until(
            lambda: (
                service.last_result is not None
                and service.last_result.outcome is CycleOutcome.PREEMPTED
            )
        )
        assert gate.calls == 0
        assert runtime.calls[-2:] == ["restore", "wake"]
        assert service.is_running
        activity.end()
    finally:
        backend.block_prepare = False
        service.stop(timeout_seconds=1.0)


def test_cycle_exception_is_logged_and_service_keeps_watching(
    caplog: pytest.LogCaptureFixture,
) -> None:
    traces = FakeTraces(2)
    activity = ActivityTracker()
    runner = ExplodingRunner()
    service = TunerService(
        _config(),
        activity=activity,
        traces=traces,
        orchestrator_factory=lambda cycle_activity: _runner_for(
            cycle_activity,
            runner,
        ),
        enabled=True,
        min_trace_records=2,
        poll_interval_seconds=0.005,
    )

    with caplog.at_level(logging.ERROR, logger="speedlm.tuner.service"):
        service.start()
        try:
            _wait_until(lambda: runner.calls == 1)
            assert service.is_running
            traces.set_count(3)
            _wait_until(lambda: runner.calls == 2)
            assert service.is_running
        finally:
            service.stop(timeout_seconds=1.0)

    assert "idle tuning cycle raised an exception" in caplog.text
    assert runner.recoveries >= 1


def test_stop_mid_cycle_preempts_and_finishes_recovery(tmp_path: Path) -> None:
    backend = FakeBackend(block_prepare=True)
    service, _, _, runtime, _ = _service(
        tmp_path,
        traces=FakeTraces(2),
        backend=backend,
    )
    service.start()
    assert backend.entered_prepare.wait(1.0)

    service.stop(timeout_seconds=1.0)

    assert not service.is_running
    assert service.last_result is not None
    assert service.last_result.outcome is CycleOutcome.PREEMPTED
    assert runtime.calls[-2:] == ["restore", "wake"]
    state = (tmp_path / "runs" / "state.json").read_text(encoding="utf-8")
    assert '"state": "READY"' in state


def test_disabled_by_default_never_runs_a_cycle(tmp_path: Path) -> None:
    service, backend, gate, _, _ = _service(
        tmp_path,
        traces=FakeTraces(2),
        enabled=None,
    )

    service.start()
    time.sleep(0.03)
    service.stop(timeout_seconds=1.0)

    assert not service.enabled
    assert not service.is_running
    assert gate.calls == 0
    assert backend.calls == []


def test_two_idle_periods_do_not_reuse_the_same_trace_watermark(
    tmp_path: Path,
) -> None:
    traces = FakeTraces(2)
    service, _, gate, _, activity = _service(tmp_path, traces=traces)
    service.start()
    try:
        _wait_until(lambda: gate.calls == 1)
        activity.begin()
        activity.end()
        time.sleep(0.05)
        assert gate.calls == 1
    finally:
        service.stop(timeout_seconds=1.0)


def test_factory_resolves_profile_before_creating_durable_state(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TunerServiceConfigurationError,
        match="does not match profile",
    ):
        build_tuner_orchestrator(
            _config(),
            activity=ActivityTracker(),
            backend=FakeBackend(verifier_model="wrong/verifier"),
            gate=FakeGate(),
            runtime=FakeRuntime(),
            home=tmp_path,
        )

    assert not (tmp_path / "runs" / "state.json").exists()


def _runner_for(
    activity: ActivitySource,
    runner: ExplodingRunner,
) -> ExplodingRunner:
    del activity
    return runner
