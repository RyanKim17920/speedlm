from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from speedlm.gateway import worker as worker_module
from speedlm.gateway.worker import await_worker


def test_await_worker_returns_worker_result() -> None:
    async def scenario() -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert await await_worker(executor.submit(lambda: 7)) == 7

    asyncio.run(scenario())


def test_await_worker_propagates_worker_exception() -> None:
    def explode() -> None:
        raise OSError("write failed")

    async def scenario() -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(explode)
        with pytest.raises(OSError, match="write failed"):
            await await_worker(future)

    asyncio.run(scenario())


def test_await_worker_does_not_pay_a_fixed_poll_tick_per_call() -> None:
    """Fast workers must resolve on completion, not on the backstop interval."""
    calls = 400

    async def scenario() -> float:
        with ThreadPoolExecutor(max_workers=1) as executor:
            start = time.perf_counter()
            for index in range(calls):
                assert await await_worker(executor.submit(lambda i=index: i)) == index
            return time.perf_counter() - start

    elapsed = asyncio.run(scenario())
    #: The previous flat 1 ms poll made this at least ``calls`` milliseconds.
    assert elapsed < calls * worker_module._POLL_TIMEOUT_SECONDS * 0.5


def test_await_worker_survives_a_lost_event_loop_wakeup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timer backstop must still complete the await if the wakeup is lost."""

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop,
            "call_soon_threadsafe",
            lambda *args, **kwargs: None,
        )
        future: Future[str] = Future()

        def resolve_later() -> None:
            time.sleep(0.02)
            future.set_result("done")

        thread = threading.Thread(target=resolve_later)
        thread.start()
        try:
            assert await await_worker(future) == "done"
        finally:
            thread.join()

    asyncio.run(scenario())


def test_await_worker_drains_the_worker_before_reraising_cancellation() -> None:
    started = threading.Event()
    finished = threading.Event()

    def slow() -> str:
        started.set()
        time.sleep(0.05)
        finished.set()
        return "late"

    async def scenario() -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow)
            task = asyncio.create_task(await_worker(future))
            await asyncio.to_thread(started.wait, 5.0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            assert future.done()

    asyncio.run(scenario())
