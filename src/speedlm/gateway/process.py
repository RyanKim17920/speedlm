from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import socket
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import FrameType
from typing import BinaryIO, cast

import httpx

from speedlm.config import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS as CONFIG_DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from speedlm.config import startup_timeout_seconds

DEFAULT_STARTUP_TIMEOUT_SECONDS = CONFIG_DEFAULT_STARTUP_TIMEOUT_SECONDS
DEFAULT_STARTUP_STALL_SECONDS = 180.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0
LOG_TAIL_LINES = 20
LOG_TAIL_MAX_BYTES = 64 * 1024
LOOPBACK_HOST = "127.0.0.1"


class ProcessError(RuntimeError):
    """Raised when the managed vLLM process cannot be started or made ready."""


def build_vllm_argv(
    model: str,
    passthrough: Sequence[str],
    *,
    host: str,
    port: int,
    executable: str = "vllm",
) -> list[str]:
    """Build vLLM argv while forcing the managed server onto loopback."""
    if not model:
        raise ValueError("model must be non-empty")
    sanitized = _remove_option(passthrough, "--host")
    sanitized = _remove_option(sanitized, "--port")
    return [
        executable,
        "serve",
        model,
        *sanitized,
        "--host",
        host,
        "--port",
        str(port),
    ]


def reserve_loopback_port() -> int:
    """Ask the kernel for an unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return int(sock.getsockname()[1])


class VLLMProcess:
    """Supervise one ``vllm serve`` child process."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        health_url: str,
        startup_timeout: float | None = None,
    ) -> None:
        if not argv:
            raise ValueError("argv must not be empty")
        self.argv = tuple(argv)
        self.health_url = health_url
        self.startup_timeout = (
            startup_timeout_seconds() if startup_timeout is None else startup_timeout
        )
        _validate_duration(self.startup_timeout, "startup timeout")
        self._process: asyncio.subprocess.Process | None = None
        self._log_file: BinaryIO | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._saved_log_tail: str | None = None

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def log_path(self) -> Path | None:
        if self._log_file is None:
            return None
        return Path(str(self._log_file.name))

    async def start(self) -> None:
        if self._process is not None:
            raise ProcessError("vLLM process has already been started")
        log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lives with the child
            mode="w+b",
            prefix="speedlm-vllm-",
            suffix=".log",
        )
        self._log_file = cast(BinaryIO, log_file)
        self._saved_log_tail = None
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            log_file.close()
            self._log_file = None
            raise ProcessError(f"cannot launch {self.argv[0]!r}: {exc}") from exc
        if self._process.stdout is not None:
            self._log_task = asyncio.create_task(self._capture_output(self._process.stdout))

    async def wait_ready(
        self,
        *,
        timeout: float | None = None,
        stall_timeout: float = DEFAULT_STARTUP_STALL_SECONDS,
        poll_interval: float = 0.25,
    ) -> None:
        timeout = self.startup_timeout if timeout is None else timeout
        _validate_duration(timeout, "startup timeout")
        _validate_duration(stall_timeout, "startup stall timeout")
        _validate_duration(poll_interval, "poll interval")
        process = self._require_process()
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        hard_deadline = started_at + timeout
        last_progress_at = started_at
        log_size = self._log_size()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(1.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            while True:
                if process.returncode is not None:
                    await self._fail_readiness(
                        f"vLLM exited before readiness with code {process.returncode}"
                    )

                now = loop.time()
                current_log_size = self._log_size()
                if current_log_size > log_size:
                    log_size = current_log_size
                    last_progress_at = now

                hard_remaining = hard_deadline - now
                if hard_remaining <= 0:
                    await self._fail_readiness(
                        f"vLLM hit the absolute startup hard ceiling of {timeout:g} "
                        f"seconds before becoming ready at {self.health_url}"
                    )
                stall_remaining = stall_timeout - (now - last_progress_at)
                if stall_remaining <= 0:
                    await self._fail_readiness(
                        f"vLLM startup stalled: child remained alive but its log did not "
                        f"grow for {stall_timeout:g} seconds"
                    )

                attempt_timeout = min(hard_remaining, stall_remaining)
                try:
                    async with asyncio.timeout(attempt_timeout):
                        response = await client.get(self.health_url)
                    if 200 <= response.status_code < 300:
                        return
                except (httpx.HTTPError, TimeoutError):
                    pass

                now = loop.time()
                await asyncio.sleep(
                    min(
                        poll_interval,
                        max(0.0, hard_deadline - now),
                        max(0.0, stall_timeout - (now - last_progress_at)),
                    )
                )

    async def wait(self) -> int:
        return await self._require_process().wait()

    def forward_signal(self, signum: int) -> None:
        process = self._require_process()
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signum)

    async def shutdown(
        self,
        *,
        timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> int:
        process = self._require_process()
        if process.returncode is not None:
            returncode = await process.wait()
        else:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                returncode = await process.wait()
        await self._finish_log_capture()
        self._saved_log_tail = self._log_tail()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        return returncode

    async def _capture_output(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(64 * 1024):
            log_file = self._log_file
            if log_file is not None:
                log_file.write(chunk)
                log_file.flush()
            _write_stderr(chunk)

    async def _finish_log_capture(self) -> None:
        task = self._log_task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        finally:
            self._log_task = None

    def _log_size(self) -> int:
        log_file = self._log_file
        if log_file is None:
            return 0
        log_file.flush()
        return os.fstat(log_file.fileno()).st_size

    def _log_tail(self) -> str:
        log_file = self._log_file
        if log_file is None:
            return self._saved_log_tail or "(no child log was captured)"
        log_file.flush()
        size = os.fstat(log_file.fileno()).st_size
        log_file.seek(max(0, size - LOG_TAIL_MAX_BYTES))
        data = log_file.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        if size > LOG_TAIL_MAX_BYTES and lines:
            lines = lines[1:]
        return "\n".join(lines[-LOG_TAIL_LINES:]) or "(child log was empty)"

    async def _fail_readiness(self, reason: str) -> None:
        await self.shutdown()
        raise ProcessError(
            f"{reason}\nLast {LOG_TAIL_LINES} lines of vLLM log:\n{self._log_tail()}"
        )

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise ProcessError("vLLM process has not been started")
        return self._process


@contextlib.contextmanager
def forwarded_signals(
    child: VLLMProcess,
    on_signal: Callable[[int], None],
) -> Iterator[None]:
    """Install temporary SIGINT/SIGTERM handlers that also notify the child."""
    previous = {}

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        child.forward_signal(signum)
        on_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)


def _remove_option(args: Sequence[str], option: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        value = args[index]
        if value == option:
            index += 2
            continue
        if value.startswith(f"{option}="):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def _write_stderr(data: bytes) -> None:
    buffer = getattr(sys.stderr, "buffer", None)
    if buffer is not None:
        buffer.write(data)
        buffer.flush()
        return
    sys.stderr.write(data.decode("utf-8", errors="replace"))
    sys.stderr.flush()


def _validate_duration(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
