from __future__ import annotations

from dataclasses import dataclass

import pytest

from speedlm.tuner.idle import IdleDetector, TuningPreempted


@dataclass
class FakeActivity:
    in_flight: int
    last_activity: float


def test_fires_only_when_empty_and_past_threshold() -> None:
    activity = FakeActivity(in_flight=0, last_activity=10.0)
    detector = IdleDetector(activity, threshold_seconds=30.0, clock=lambda: 40.0)

    assert detector.idle_seconds == 30.0
    assert detector.should_tune

    activity.in_flight = 1
    assert detector.idle_seconds == 0.0
    assert not detector.should_tune


def test_guard_detects_completed_request_since_arming() -> None:
    activity = FakeActivity(in_flight=0, last_activity=10.0)
    detector = IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 20.0)
    guard = detector.arm()

    activity.last_activity = 21.0

    assert guard.is_preempted
    with pytest.raises(TuningPreempted):
        guard.check()


def test_guard_detects_in_flight_request() -> None:
    activity = FakeActivity(in_flight=0, last_activity=0.0)
    guard = IdleDetector(
        activity,
        threshold_seconds=1.0,
        clock=lambda: 2.0,
    ).arm()

    activity.in_flight = 1

    with pytest.raises(TuningPreempted):
        guard.check()
