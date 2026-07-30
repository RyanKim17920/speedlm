from __future__ import annotations

import asyncio
import contextlib
import math
import os
import queue
import signal
import socket
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import BinaryIO, cast

import httpx

from speedlm.config import (
    DEFAULT_STARTUP_STALL_SECONDS as CONFIG_DEFAULT_STARTUP_STALL_SECONDS,
)
from speedlm.config import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS as CONFIG_DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from speedlm.config import startup_stall_seconds, startup_timeout_seconds

DEFAULT_STARTUP_TIMEOUT_SECONDS = CONFIG_DEFAULT_STARTUP_TIMEOUT_SECONDS
DEFAULT_STARTUP_STALL_SECONDS = CONFIG_DEFAULT_STARTUP_STALL_SECONDS
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0
LOG_TAIL_LINES = 20
LOG_TAIL_MAX_BYTES = 64 * 1024
LOOPBACK_HOST = "127.0.0.1"
STDERR_MIRROR_QUEUE_CHUNKS = 64


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
        startup_stall_timeout: float | None = None,
        env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if not argv:
            raise ValueError("argv must not be empty")
        self.argv = tuple(argv)
        self.health_url = health_url
        self.startup_timeout = (
            startup_timeout_seconds() if startup_timeout is None else startup_timeout
        )
        self.startup_stall_timeout = (
            startup_stall_seconds()
            if startup_stall_timeout is None
            else startup_stall_timeout
        )
        self._env_overrides = _validate_env_overrides(env_overrides)
        _validate_duration(self.startup_timeout, "startup timeout")
        _validate_duration(self.startup_stall_timeout, "startup stall timeout")
        self._process: asyncio.subprocess.Process | None = None
        self._log_file: BinaryIO | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._saved_log_tail: str | None = None
        self._stderr_mirror: _StderrMirror | None = None

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
        self._stderr_mirror = _StderrMirror()
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.update(self._env_overrides)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
            )
        except OSError as exc:
            if self._stderr_mirror is not None:
                self._stderr_mirror.close()
                self._stderr_mirror = None
            log_file.close()
            self._log_file = None
            raise ProcessError(f"cannot launch {self.argv[0]!r}: {exc}") from exc
        if self._process.stdout is not None:
            self._log_task = asyncio.create_task(self._capture_output(self._process.stdout))

    async def wait_ready(
        self,
        *,
        timeout: float | None = None,
        stall_timeout: float | None = None,
        poll_interval: float = 0.25,
    ) -> None:
        timeout = self.startup_timeout if timeout is None else timeout
        stall_timeout = (
            self.startup_stall_timeout if stall_timeout is None else stall_timeout
        )
        _validate_duration(timeout, "startup timeout")
        _validate_duration(stall_timeout, "startup stall timeout")
        _validate_duration(poll_interval, "poll interval")
        process = self._require_process()
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        hard_deadline = started_at + timeout
        last_progress_at = started_at
        last_log_progress_at = started_at
        log_size = self._log_size()
        cpu_ticks = _process_tree_cpu_ticks(process.pid)
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
                    last_log_progress_at = now

                current_cpu_ticks = _process_tree_cpu_ticks(process.pid)
                cpu_advanced = any(
                    ticks > cpu_ticks.get(pid, ticks)
                    for pid, ticks in current_cpu_ticks.items()
                )
                cpu_ticks.update(current_cpu_ticks)
                if cpu_advanced:
                    last_progress_at = now

                hard_remaining = hard_deadline - now
                if hard_remaining <= 0:
                    await self._fail_readiness(
                        f"vLLM hit the absolute startup hard ceiling of {timeout:g} "
                        f"seconds before becoming ready at {self.health_url}"
                    )
                stall_remaining = stall_timeout - (now - last_progress_at)
                if stall_remaining <= 0:
                    stalled_for = now - last_progress_at
                    log_silence = now - last_log_progress_at
                    await self._fail_readiness(
                        f"vLLM startup stalled after {stalled_for:.1f} seconds without "
                        f"log or CPU progress (log silence: {log_silence:.1f} seconds; "
                        f"process alive: {'yes' if process.returncode is None else 'no'}; "
                        "CPU time advanced: no)"
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
                returncode = await _wait_for_returncode(process, timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                returncode = await _wait_for_returncode(process)
        await self._finish_log_capture()
        self._saved_log_tail = self._log_tail()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._stderr_mirror is not None:
            self._stderr_mirror.close()
            self._stderr_mirror = None
        return returncode

    async def _capture_output(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(64 * 1024):
            log_file = self._log_file
            if log_file is not None:
                log_file.write(chunk)
                log_file.flush()
            mirror = self._stderr_mirror
            if mirror is not None:
                mirror.submit(chunk)

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
        on_signal(signum)
        if child.pid is not None:
            child.forward_signal(signum)

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


def _validate_env_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    result = dict(overrides or {})
    for key, value in result.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
        ):
            raise ValueError(
                "vLLM environment overrides must contain valid string names and values"
            )
    return result


class _StderrMirror:
    """Best-effort child-log mirroring that can never block the gateway loop."""

    def __init__(self, *, queue_chunks: int = STDERR_MIRROR_QUEUE_CHUNKS) -> None:
        if isinstance(queue_chunks, bool) or not isinstance(queue_chunks, int) or queue_chunks <= 0:
            raise ValueError("stderr mirror queue size must be a positive integer")
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_chunks)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="speedlm-vllm-stderr-mirror",
            daemon=True,
        )
        self._thread.start()

    def submit(self, data: bytes) -> None:
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # The durable tempfile remains complete. Only the optional live
            # mirror is dropped when a parent harness stops draining stderr.
            return

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        # A blocked writer may leave the queue full. Once that write returns,
        # the stop event makes the worker exit without requiring a sentinel.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    def _run(self) -> None:
        while not self._stop.is_set():
            data = self._queue.get()
            if data is None:
                return
            try:
                _write_stderr(data)
            except Exception:
                # Mirroring is optional and must never make the supervisor
                # noisy or unhealthy when its inherited stderr disappears.
                return


def _process_tree_cpu_ticks(root_pid: int) -> dict[int, int]:
    """Read user and system CPU ticks for a process and its live descendants."""
    cpu_ticks: dict[int, int] = {}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in cpu_ticks:
            continue
        ticks = _process_cpu_ticks(pid)
        if ticks is None:
            continue
        cpu_ticks[pid] = ticks
        pending.extend(_process_child_pids(pid))
    return cpu_ticks


def _process_cpu_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat[closing_paren + 2 :].split()
    try:
        # fields starts at proc(5)'s field 3 (state); indexes 11/12 are
        # fields 14/15 (utime/stime).
        return int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def _process_child_pids(pid: int) -> set[int]:
    children: set[int] = set()
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tasks = list(task_dir.iterdir())
    except OSError:
        return children
    for task in tasks:
        try:
            raw_children = (task / "children").read_text(encoding="utf-8")
            children.update(int(child_pid) for child_pid in raw_children.split())
        except (OSError, ValueError):
            continue
    return children


async def _wait_for_returncode(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None = None,
) -> int:
    async def poll() -> int:
        while process.returncode is None:
            await asyncio.sleep(0.01)
        return process.returncode

    if timeout is None:
        return await poll()
    async with asyncio.timeout(timeout):
        return await poll()


def _validate_duration(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
