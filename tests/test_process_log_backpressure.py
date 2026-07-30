from __future__ import annotations

import asyncio
import tempfile
import threading
from collections.abc import Callable, Sequence

import pytest

from speedlm.gateway import process as process_module
from speedlm.gateway.process import VLLMProcess


class _ChunkStream:
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = iter(chunks)
        self.read_count = 0

    async def read(self, size: int) -> bytes:
        assert size > 0
        await asyncio.sleep(0)
        self.read_count += 1
        return next(self._chunks, b"")


async def _wait_until(predicate: Callable[[], bool], *, timeout: float) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def test_blocked_stderr_writer_does_not_stall_stdout_drain_or_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = (b"first\n", b"second\n", b"third\n")
    writer_started = threading.Event()
    release_writer = threading.Event()
    writer_timed_out = threading.Event()

    def blocked_write(data: bytes) -> None:
        assert data
        writer_started.set()
        if not release_writer.wait(timeout=1.0):
            writer_timed_out.set()

    monkeypatch.setattr(process_module, "_write_stderr", blocked_write)

    async def scenario() -> None:
        process = VLLMProcess(["vllm"], health_url="http://127.0.0.1/health")
        log_file = tempfile.TemporaryFile(  # noqa: SIM115 - closed after async cleanup
            mode="w+b"
        )
        process._log_file = log_file
        mirror = process_module._StderrMirror(queue_chunks=1)
        process._stderr_mirror = mirror
        stream = _ChunkStream(chunks)
        heartbeat = asyncio.Event()

        async def beat() -> None:
            await asyncio.sleep(0)
            heartbeat.set()

        capture_task = asyncio.create_task(process._capture_output(stream))  # type: ignore[arg-type]
        heartbeat_task = asyncio.create_task(beat())
        try:
            await _wait_until(writer_started.is_set, timeout=0.5)
            await asyncio.wait_for(heartbeat.wait(), timeout=0.2)
            await asyncio.wait_for(capture_task, timeout=0.2)

            assert stream.read_count == len(chunks) + 1
            assert not release_writer.is_set()
            log_file.seek(0)
            assert log_file.read() == b"".join(chunks)
        finally:
            release_writer.set()
            mirror.close()
            await heartbeat_task
            await _wait_until(
                lambda: not mirror._thread.is_alive(),
                timeout=0.5,
            )
            log_file.close()

        assert not mirror._thread.is_alive()
        assert not writer_timed_out.is_set()

    asyncio.run(scenario())


def test_close_returns_when_blocked_writer_and_queue_are_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_started = threading.Event()
    release_writer = threading.Event()

    def blocked_write(data: bytes) -> None:
        assert data == b"first"
        writer_started.set()
        release_writer.wait(timeout=1.0)

    monkeypatch.setattr(process_module, "_write_stderr", blocked_write)
    mirror = process_module._StderrMirror(queue_chunks=1)
    try:
        mirror.submit(b"first")
        assert writer_started.wait(timeout=0.5)
        mirror.submit(b"queued")

        mirror.close()
        assert mirror._stop.is_set()
    finally:
        release_writer.set()
        mirror.close()
        mirror._thread.join(timeout=0.5)

    assert not mirror._thread.is_alive()


@pytest.mark.parametrize("queue_chunks", [0, -1, True, 1.5])
def test_stderr_mirror_rejects_invalid_queue_sizes(queue_chunks: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        process_module._StderrMirror(queue_chunks=queue_chunks)  # type: ignore[arg-type]
