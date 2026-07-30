from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from speedlm.gateway.supervisor import (
    ThreadsafeProcessControl,
    VLLMSupervisor,
)


@dataclass
class FakeVLLMProcess:
    argv: list[str]
    health_url: str
    pid: int
    shutdown_returncode: int = 0
    returncode: int | None = None
    started: bool = False
    shutdown_calls: int = 0
    ready_timeouts: list[float | None] = field(default_factory=list)
    forwarded_signals: list[int] = field(default_factory=list)
    _exited: asyncio.Event | None = None

    async def start(self) -> None:
        self.started = True
        self._exited = asyncio.Event()

    async def wait_ready(self, *, timeout: float | None = None) -> None:
        assert self.started
        self.ready_timeouts.append(timeout)

    async def wait(self) -> int:
        assert self._exited is not None
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    async def shutdown(self) -> int:
        self.shutdown_calls += 1
        if self.returncode is None:
            self.returncode = self.shutdown_returncode
        assert self._exited is not None
        self._exited.set()
        return self.returncode

    def forward_signal(self, signum: int) -> None:
        self.forwarded_signals.append(signum)

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        assert self._exited is not None
        self._exited.set()


@dataclass
class FakeProcessFactory:
    shutdown_returncodes: list[int] = field(default_factory=list)
    children: list[FakeVLLMProcess] = field(default_factory=list)

    def __call__(
        self,
        argv: list[str],
        *,
        health_url: str,
        **kwargs: Any,
    ) -> FakeVLLMProcess:
        assert kwargs == {}
        index = len(self.children)
        shutdown_returncode = (
            self.shutdown_returncodes[index]
            if index < len(self.shutdown_returncodes)
            else 0
        )
        child = FakeVLLMProcess(
            argv,
            health_url,
            pid=4100 + index,
            shutdown_returncode=shutdown_returncode,
        )
        self.children.append(child)
        return child


def _supervisor(
    factory: FakeProcessFactory,
    *,
    pid_changes: list[int | None] | None = None,
) -> VLLMSupervisor:
    return VLLMSupervisor(
        argv_factory=lambda draft: ["vllm", "serve", "model", "--draft", str(draft)],
        health_url="http://127.0.0.1:9000/health",
        process_factory=factory,
        on_pid_changed=(
            pid_changes.append
            if pid_changes is not None
            else None
        ),
    )


def test_initial_start_and_readiness_use_current_child() -> None:
    async def scenario() -> None:
        factory = FakeProcessFactory()
        pid_changes: list[int | None] = []
        supervisor = _supervisor(factory, pid_changes=pid_changes)

        await supervisor.start("draft-a")
        await supervisor.wait_ready(timeout=12.5)
        supervisor.forward_signal(15)

        assert len(factory.children) == 1
        child = factory.children[0]
        assert child.started
        assert child.argv == [
            "vllm",
            "serve",
            "model",
            "--draft",
            "draft-a",
        ]
        assert child.health_url == "http://127.0.0.1:9000/health"
        assert child.ready_timeouts == [12.5]
        assert child.forwarded_signals == [15]
        assert supervisor.child is child
        assert supervisor.pid == child.pid
        assert pid_changes == [child.pid]

        assert await supervisor.shutdown() == 0
        assert pid_changes == [child.pid, None]

    asyncio.run(scenario())


def test_intentional_restart_does_not_complete_fatal_wait() -> None:
    async def scenario() -> None:
        factory = FakeProcessFactory()
        pid_changes: list[int | None] = []
        supervisor = _supervisor(factory, pid_changes=pid_changes)
        await supervisor.start("baseline")
        fatal_wait = asyncio.create_task(supervisor.wait())
        await asyncio.sleep(0)

        await supervisor.restart("candidate", timeout_seconds=1.0)
        await asyncio.sleep(0)

        assert not fatal_wait.done()
        assert len(factory.children) == 2
        baseline, candidate = factory.children
        assert baseline.shutdown_calls == 1
        assert candidate.ready_timeouts == [None]
        assert supervisor.child is candidate
        assert pid_changes == [
            baseline.pid,
            None,
            candidate.pid,
        ]

        assert await supervisor.shutdown() == 0
        with pytest.raises(asyncio.CancelledError):
            await fatal_wait

    asyncio.run(scenario())


def test_unexpected_exit_from_current_child_completes_fatal_wait() -> None:
    async def scenario() -> None:
        factory = FakeProcessFactory()
        supervisor = _supervisor(factory)
        await supervisor.start(Path("/draft/current"))
        child = factory.children[0]
        fatal_wait = asyncio.create_task(supervisor.wait())

        child.exit(23)

        assert await asyncio.wait_for(fatal_wait, timeout=1.0) == 23
        assert supervisor.child is child
        assert await supervisor.shutdown() == 23

    asyncio.run(scenario())


def test_shutdown_stops_child_clears_pid_and_cancels_fatal_wait() -> None:
    async def scenario() -> None:
        factory = FakeProcessFactory(shutdown_returncodes=[-15])
        pid_changes: list[int | None] = []
        supervisor = _supervisor(factory, pid_changes=pid_changes)
        await supervisor.start("draft")
        fatal_wait = asyncio.create_task(supervisor.wait())
        await asyncio.sleep(0)

        assert await supervisor.shutdown() == -15

        child = factory.children[0]
        assert child.shutdown_calls == 1
        assert supervisor.child is None
        assert supervisor.pid is None
        assert pid_changes == [child.pid, None]
        with pytest.raises(asyncio.CancelledError):
            await fatal_wait
        assert await supervisor.shutdown() == 0

    asyncio.run(scenario())


def test_threadsafe_process_control_restarts_on_owner_loop() -> None:
    async def scenario() -> None:
        factory = FakeProcessFactory()
        supervisor = _supervisor(factory)
        await supervisor.start("baseline")
        control = ThreadsafeProcessControl(
            supervisor,
            asyncio.get_running_loop(),
        )
        finished = Event()
        failures: list[BaseException] = []

        def restart_from_tuner_thread() -> None:
            try:
                control.restart("candidate", timeout_seconds=1.0)
            except BaseException as exc:
                failures.append(exc)
            finally:
                finished.set()

        thread = Thread(
            target=restart_from_tuner_thread,
            name="test-tuner-thread",
        )
        thread.start()
        while not finished.is_set():
            await asyncio.sleep(0.001)
        thread.join()

        assert failures == []
        assert len(factory.children) == 2
        assert factory.children[0].shutdown_calls == 1
        assert factory.children[1].argv[-1] == "candidate"
        assert factory.children[1].ready_timeouts == [None]
        await supervisor.shutdown()

    asyncio.run(scenario())
