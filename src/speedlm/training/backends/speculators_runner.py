"""Abort-aware, output-capturing subprocess boundary for Speculators.

The training backend depends on the small :class:`ProcessRunner` protocol so
login-node tests can inject a fake.  :class:`SubprocessRunner` is the production
implementation and deliberately uses process groups: vLLM and training launchers
may create children which must be stopped together when serving preempts tuning.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, Protocol

from speedlm.tuner.eagle3 import StageTimeoutError, TrainingError
from speedlm.tuner.idle import TuningPreempted

AbortCheck = Callable[[], bool]

#: Attribute under which a terminated child's captured streams travel on the
#: exception that killed it.
#:
#: A dedicated attribute rather than a new exception type because the
#: exceptions in question -- :class:`TuningPreempted`, :class:`StageTimeoutError`,
#: :class:`~speedlm.tuner.eagle3.ScratchQuotaExceeded`, and whatever the abort
#: check itself raises -- are raised by four different modules and are matched
#: by type elsewhere; wrapping them would change control flow, and adding a
#: ``stderr`` field to each would not cover the ones raised by the standard
#: library.
_OUTPUT_ATTRIBUTE: Final = "_speedlm_process_output"


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Captured result of a completed child process."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class RunningProcess(Protocol):
    """Opaque process handle needed by the long-running vLLM stage."""

    argv: tuple[str, ...]
    timeout_seconds: float


class ProcessRunner(Protocol):
    """Injectable child-process operations used by the concrete backend."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> ProcessResult: ...

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
    ) -> RunningProcess: ...

    def check_running(
        self,
        process: RunningProcess,
        *,
        should_abort: AbortCheck,
    ) -> int | None: ...

    def terminate(
        self,
        process: RunningProcess,
        *,
        grace_seconds: float,
    ) -> ProcessResult: ...


@dataclass(slots=True)
class _SubprocessHandle:
    argv: tuple[str, ...]
    timeout_seconds: float
    process: subprocess.Popen[str]
    stdout_file: IO[str]
    stderr_file: IO[str]
    started_at: float
    result: ProcessResult | None = None


class SubprocessRunner:
    """Production runner with polling aborts, hard timeouts, and captured output."""

    def __init__(
        self,
        *,
        poll_interval_seconds: float = 0.1,
        terminate_grace_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if terminate_grace_seconds <= 0:
            raise ValueError("terminate_grace_seconds must be positive")
        self.poll_interval_seconds = poll_interval_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self._clock = clock

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> ProcessResult:
        """Run a finite command while checking abort and timeout during execution."""
        handle = self._handle(
            self.start(
                argv,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
            )
        )
        while True:
            try:
                returncode = self.check_running(handle, should_abort=should_abort)
            except BaseException as error:
                # terminate() reaps the child and reads its captured streams out
                # of unnamed temporary files, which are then closed and gone.
                # Discarding that result threw away the *only* record of what
                # the child was doing when an abort, a timeout, or a scratch
                # quota trip killed it.  Carry it on the exception so the
                # stage's failure path can persist it before it deletes the
                # stage's output.
                attach_process_output(
                    error,
                    self.terminate(handle, grace_seconds=self.terminate_grace_seconds),
                )
                raise
            if returncode is not None:
                return self._collect(handle, returncode)
            time.sleep(self.poll_interval_seconds)

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
    ) -> RunningProcess:
        """Start a process in a new session with output held in temporary files."""
        command = _validated_argv(argv)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        # The files intentionally outlive this method and are closed by _collect.
        stdout_file = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+t", encoding="utf-8"
        )
        stderr_file = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+t", encoding="utf-8"
        )
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                cwd=cwd,
                env=None if env is None else dict(env),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            stdout_file.close()
            stderr_file.close()
            raise TrainingError(
                f"could not start subprocess: {' '.join(command)}",
                stderr=str(error),
            ) from error
        except BaseException:
            stdout_file.close()
            stderr_file.close()
            raise
        return _SubprocessHandle(
            argv=command,
            timeout_seconds=timeout_seconds,
            process=process,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            started_at=self._clock(),
        )

    def check_running(
        self,
        process: RunningProcess,
        *,
        should_abort: AbortCheck,
    ) -> int | None:
        """Poll a process, raising typed preemption/timeout errors when required."""
        handle = self._handle(process)
        if handle.result is not None:
            return handle.result.returncode
        if should_abort():
            raise TuningPreempted("incoming request preempted Speculators subprocess")
        elapsed = self._clock() - handle.started_at
        if elapsed > handle.timeout_seconds:
            raise StageTimeoutError(
                f"subprocess exceeded {handle.timeout_seconds:.3f}s timeout "
                f"(elapsed {elapsed:.3f}s): {' '.join(handle.argv)}"
            )
        return handle.process.poll()

    def terminate(
        self,
        process: RunningProcess,
        *,
        grace_seconds: float,
    ) -> ProcessResult:
        """Terminate the whole child process group and always reap the leader."""
        handle = self._handle(process)
        if handle.result is not None:
            return handle.result
        returncode = handle.process.poll()
        if returncode is None:
            self._signal_group(handle.process, signal.SIGTERM)
            try:
                returncode = handle.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self._signal_group(handle.process, signal.SIGKILL)
                returncode = handle.process.wait()
        return self._collect(handle, returncode)

    @staticmethod
    def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return

    @staticmethod
    def _handle(process: RunningProcess) -> _SubprocessHandle:
        if not isinstance(process, _SubprocessHandle):
            raise TypeError("process handle was not created by this runner")
        return process

    @staticmethod
    def _collect(handle: _SubprocessHandle, returncode: int) -> ProcessResult:
        if handle.result is not None:
            return handle.result
        handle.stdout_file.flush()
        handle.stderr_file.flush()
        handle.stdout_file.seek(0)
        handle.stderr_file.seek(0)
        result = ProcessResult(
            argv=handle.argv,
            returncode=returncode,
            stdout=handle.stdout_file.read(),
            stderr=handle.stderr_file.read(),
        )
        handle.stdout_file.close()
        handle.stderr_file.close()
        handle.result = result
        return result


def attach_process_output(error: BaseException, result: ProcessResult) -> None:
    """Record a terminated child's streams on the exception that killed it.

    Best effort: an exception that refuses attribute assignment must not
    replace the failure it was carrying with an ``AttributeError``.
    """
    with contextlib.suppress(AttributeError, TypeError):
        setattr(error, _OUTPUT_ATTRIBUTE, result)


def process_output(error: BaseException) -> ProcessResult | None:
    """Return the streams :func:`attach_process_output` recorded, if any."""
    value = getattr(error, _OUTPUT_ATTRIBUTE, None)
    return value if isinstance(value, ProcessResult) else None


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    command = tuple(argv)
    if not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("argv must contain non-empty strings")
    return command


__all__ = [
    "ProcessResult",
    "attach_process_output",
    "process_output",
    "ProcessRunner",
    "RunningProcess",
    "SubprocessRunner",
]
