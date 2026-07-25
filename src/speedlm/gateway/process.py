from __future__ import annotations

import asyncio
import contextlib
import signal
import socket
from collections.abc import Callable, Iterator, Sequence
from types import FrameType

import httpx

DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0
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
    ) -> None:
        if not argv:
            raise ValueError("argv must not be empty")
        self.argv = tuple(argv)
        self.health_url = health_url
        self._process: asyncio.subprocess.Process | None = None

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def start(self) -> None:
        if self._process is not None:
            raise ProcessError("vLLM process has already been started")
        try:
            self._process = await asyncio.create_subprocess_exec(*self.argv)
        except OSError as exc:
            raise ProcessError(f"cannot launch {self.argv[0]!r}: {exc}") from exc

    async def wait_ready(
        self,
        *,
        timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        poll_interval: float = 0.25,
    ) -> None:
        if timeout <= 0:
            raise ValueError("startup timeout must be positive")
        process = self._require_process()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(1.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            while True:
                if process.returncode is not None:
                    raise ProcessError(
                        f"vLLM exited before readiness with code {process.returncode}"
                    )
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ProcessError(
                        f"vLLM did not become ready at {self.health_url} "
                        f"within {timeout:g} seconds"
                    )
                try:
                    async with asyncio.timeout(remaining):
                        response = await client.get(self.health_url)
                    if 200 <= response.status_code < 300:
                        return
                except (httpx.HTTPError, TimeoutError):
                    pass
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ProcessError(
                        f"vLLM did not become ready at {self.health_url} "
                        f"within {timeout:g} seconds"
                    )
                await asyncio.sleep(min(poll_interval, remaining))

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
            return process.returncode
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            return await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            return await process.wait()

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
