from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock


class ActivityTracker:
    """Track active proxy requests and time since the gateway became idle."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._in_flight = 0
        self._last_activity = clock()
        self._admitting = True
        self._lock = Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def begin(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._last_activity = self._clock()

    def end(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("activity request count would become negative")
            self._in_flight -= 1
            self._last_activity = self._clock()

    def touch(self) -> None:
        """Record an attempted request without changing the in-flight count."""
        with self._lock:
            self._last_activity = self._clock()

    def try_begin(self) -> bool:
        """Atomically check admission and acquire one in-flight request."""
        with self._lock:
            self._last_activity = self._clock()
            if not self._admitting:
                return False
            self._in_flight += 1
            return True

    def stop_admitting(self) -> None:
        with self._lock:
            self._admitting = False

    def start_admitting(self) -> None:
        with self._lock:
            self._admitting = True

    @property
    def is_admitting(self) -> bool:
        with self._lock:
            return self._admitting

    def idle_seconds(self) -> float:
        """Return zero while busy, otherwise seconds since the last activity."""
        with self._lock:
            if self._in_flight:
                return 0.0
            return max(0.0, self._clock() - self._last_activity)
