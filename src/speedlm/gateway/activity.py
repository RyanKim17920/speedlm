from __future__ import annotations

import time
from collections.abc import Callable


class ActivityTracker:
    """Track active proxy requests and time since the gateway became idle."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._in_flight = 0
        self._last_activity = clock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def last_activity(self) -> float:
        return self._last_activity

    def begin(self) -> None:
        self._in_flight += 1
        self._last_activity = self._clock()

    def end(self) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("activity request count would become negative")
        self._in_flight -= 1
        self._last_activity = self._clock()

    def idle_seconds(self) -> float:
        """Return zero while busy, otherwise seconds since the last activity."""
        if self._in_flight:
            return 0.0
        return max(0.0, self._clock() - self._last_activity)
