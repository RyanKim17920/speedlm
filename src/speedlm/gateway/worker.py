from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import Future

#: Backstop poll interval. A completion callback normally wakes the waiter
#: immediately; this only bounds how long a *lost* wakeup can stall the caller,
#: and matches the interval this module used to poll at unconditionally.
_POLL_TIMEOUT_SECONDS = 0.001


def _arm_wakeup(future: Future[object]) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[None] = loop.create_future()

    def _resolve() -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _on_done(_completed: Future[object]) -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(_resolve)

    future.add_done_callback(_on_done)
    return waiter


async def _wait_until_done(
    future: Future[object],
    waiter: asyncio.Future[None],
) -> None:
    while not future.done():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(waiter),
                _POLL_TIMEOUT_SECONDS,
            )


async def await_worker[T](future: Future[T]) -> T:
    """Await a worker future without depending on selector thread wakeups.

    A few managed Python runtimes lose the wakeup used by
    ``run_in_executor``/``asyncio.to_thread`` after filesystem operations.
    The completion callback is therefore paired with a short timer backstop:
    the common case resolves as soon as the worker finishes, and a dropped
    wakeup degrades to the same bounded polling this module used before,
    preserving strict ordering and bounded memory.
    """
    waiter = _arm_wakeup(future)  # type: ignore[arg-type]
    try:
        await _wait_until_done(future, waiter)  # type: ignore[arg-type]
    except asyncio.CancelledError:
        await _wait_until_done(future, waiter)  # type: ignore[arg-type]
        future.result()
        raise
    return future.result()
