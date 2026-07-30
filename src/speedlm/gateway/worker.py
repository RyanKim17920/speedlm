from __future__ import annotations

import asyncio
from concurrent.futures import Future


async def await_worker[T](future: Future[T]) -> T:
    """Await a worker future without depending on selector thread wakeups.

    A few managed Python runtimes lose the wakeup used by
    ``run_in_executor``/``asyncio.to_thread`` after filesystem operations.
    Short timer polling keeps the event loop runnable while preserving strict
    ordering and bounded memory.
    """
    try:
        while not future.done():
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        while not future.done():
            await asyncio.sleep(0.001)
        future.result()
        raise
    return future.result()
