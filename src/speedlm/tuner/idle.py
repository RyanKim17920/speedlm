"""Idle detection and request-driven tuning preemption."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ActivitySource(Protocol):
    """Gateway activity needed by the tuner.

    Both timestamps and the detector clock must use the same monotonic clock.
    """

    @property
    def in_flight(self) -> int:
        """Number of requests currently handled by the gateway."""

    @property
    def last_activity(self) -> float:
        """Monotonic timestamp of the most recent request begin/end."""


class TuningPreempted(RuntimeError):
    """Raised when request activity invalidates an idle tuning lease."""


@dataclass(frozen=True, slots=True)
class PreemptionGuard:
    """Detect activity that occurs after a tuning cycle is armed."""

    source: ActivitySource
    armed_at_activity: float

    @property
    def is_preempted(self) -> bool:
        return (
            self.source.in_flight > 0
            or self.source.last_activity > self.armed_at_activity
        )

    def check(self) -> None:
        if self.is_preempted:
            raise TuningPreempted("incoming request preempted idle tuning")


@dataclass(frozen=True, slots=True)
class IdleDetector:
    """Evaluate the gateway's current idle window."""

    source: ActivitySource
    threshold_seconds: float
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold_seconds, bool)
            or not isinstance(self.threshold_seconds, (int, float))
            or self.threshold_seconds <= 0
        ):
            raise ValueError("threshold_seconds must be a positive number")

    @property
    def idle_seconds(self) -> float:
        if self.source.in_flight > 0:
            return 0.0
        return max(0.0, self.clock() - self.source.last_activity)

    @property
    def should_tune(self) -> bool:
        return self.source.in_flight == 0 and self.idle_seconds >= self.threshold_seconds

    def arm(self) -> PreemptionGuard:
        """Capture the activity watermark after verifying the gateway is idle."""
        if not self.should_tune:
            raise TuningPreempted("gateway is not idle enough to start tuning")
        return PreemptionGuard(
            source=self.source,
            armed_at_activity=self.source.last_activity,
        )
