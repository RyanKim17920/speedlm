"""Replaceable vLLM child supervision for idle tuning."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path

from speedlm.gateway.control import DraftReference
from speedlm.gateway.process import VLLMProcess

ArgvFactory = Callable[[DraftReference], list[str]]
ProcessFactory = Callable[..., VLLMProcess]
PidCallback = Callable[[int | None], None]


class VLLMSupervisor:
    """Own one replaceable vLLM child while exposing one stable lifecycle.

    Planned draft replacements do not look like child crashes to the gateway.
    An unplanned exit from whichever generation is current completes
    :meth:`wait` and shuts the gateway down.
    """

    def __init__(
        self,
        *,
        argv_factory: ArgvFactory,
        health_url: str,
        process_factory: ProcessFactory = VLLMProcess,
        on_pid_changed: PidCallback | None = None,
    ) -> None:
        self._argv_factory = argv_factory
        self._health_url = health_url
        self._process_factory = process_factory
        self._on_pid_changed = on_pid_changed
        self._child: VLLMProcess | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._unexpected_exit: asyncio.Future[int] | None = None
        self._lock = asyncio.Lock()
        self._replacing = False
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self._child.pid if self._child is not None else None

    @property
    def child(self) -> VLLMProcess | None:
        return self._child

    async def start(self, draft: DraftReference) -> None:
        async with self._lock:
            if self._child is not None:
                raise RuntimeError("vLLM supervisor has already been started")
            if self._closed:
                raise RuntimeError("vLLM supervisor is closed")
            self._unexpected_exit = asyncio.get_running_loop().create_future()
            await self._start_locked(draft)

    async def wait_ready(self, *, timeout: float | None = None) -> None:
        child = self._require_child()
        await child.wait_ready(timeout=timeout)

    async def restart(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
    ) -> None:
        async with asyncio.timeout(timeout_seconds):
            async with self._lock:
                if self._closed:
                    raise RuntimeError("vLLM supervisor is closed")
                self._replacing = True
                try:
                    await self._stop_locked()
                    await self._start_locked(draft)
                    await self._require_child().wait_ready()
                finally:
                    self._replacing = False

    async def wait(self) -> int:
        future = self._unexpected_exit
        if future is None:
            raise RuntimeError("vLLM supervisor has not been started")
        return await asyncio.shield(future)

    def forward_signal(self, signum: int) -> None:
        child = self._child
        if child is not None:
            child.forward_signal(signum)

    async def shutdown(self) -> int:
        async with self._lock:
            if self._closed:
                child = self._child
                return child.returncode if child is not None and child.returncode is not None else 0
            self._closed = True
            self._replacing = True
            try:
                return await self._stop_locked()
            finally:
                self._replacing = False
                future = self._unexpected_exit
                if future is not None and not future.done():
                    future.cancel()

    async def _start_locked(self, draft: DraftReference) -> None:
        child = self._process_factory(
            self._argv_factory(draft),
            health_url=self._health_url,
        )
        await child.start()
        self._child = child
        if self._on_pid_changed is not None:
            self._on_pid_changed(child.pid)
        self._watcher = asyncio.create_task(
            self._watch(child),
            name="speedlm-vllm-child-watch",
        )

    async def _stop_locked(self) -> int:
        watcher = self._watcher
        self._watcher = None
        if watcher is not None:
            watcher.cancel()
        child = self._child
        self._child = None
        if child is None:
            return 0
        try:
            return await child.shutdown()
        finally:
            if watcher is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            if self._on_pid_changed is not None:
                self._on_pid_changed(None)

    async def _watch(self, child: VLLMProcess) -> None:
        try:
            returncode = await child.wait()
        except asyncio.CancelledError:
            raise
        if self._child is not child or self._replacing or self._closed:
            return
        future = self._unexpected_exit
        if future is not None and not future.done():
            future.set_result(returncode)

    def _require_child(self) -> VLLMProcess:
        if self._child is None:
            raise RuntimeError("vLLM supervisor has no active child")
        return self._child


class ThreadsafeProcessControl:
    """Synchronous tuner-thread facade over an asyncio-owned supervisor."""

    def __init__(
        self,
        supervisor: VLLMSupervisor,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._supervisor = supervisor
        self._loop = loop

    def restart(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._supervisor.restart(draft, timeout_seconds=timeout_seconds),
            self._loop,
        )
        future.result(timeout=timeout_seconds + 1.0)
