from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from speedlm.tuner.state import (
    IllegalTransitionError,
    TunerState,
    TunerStateMachine,
)


def test_illegal_transition_raises(tmp_path: Path) -> None:
    machine = TunerStateMachine(tmp_path, clock=lambda: 10.0)

    with pytest.raises(IllegalTransitionError, match="READY -> TRAINING"):
        machine.transition(TunerState.TRAINING)

    assert machine.state is TunerState.READY


def test_state_is_atomic_and_events_are_auditable(tmp_path: Path) -> None:
    ticks = iter((1.0, 2.0, 3.0))
    machine = TunerStateMachine(tmp_path, clock=lambda: next(ticks))
    machine.transition(TunerState.QUIESCING, reason="idle")
    machine.transition(TunerState.SLEEPING, reason="drained")

    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert stored["state"] == "SLEEPING"
    assert stored["sequence"] == 2
    assert [event["to"] for event in events] == ["READY", "QUIESCING", "SLEEPING"]


def test_state_transitions_are_visible_in_operator_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="speedlm.tuner.state")
    machine = TunerStateMachine(tmp_path, clock=lambda: 10.0)

    machine.transition(TunerState.QUIESCING, reason="idle threshold reached")

    assert "idle tuner state initialized: READY" in caplog.messages
    assert (
        "idle tuner state READY -> QUIESCING: idle threshold reached"
        in caplog.messages
    )


def test_crash_restart_resumes_to_safe_ready_state(tmp_path: Path) -> None:
    machine = TunerStateMachine(tmp_path, clock=lambda: 1.0)
    machine.transition(TunerState.QUIESCING)
    machine.transition(TunerState.SLEEPING)
    machine.transition(TunerState.EXTRACTING)

    restarted = TunerStateMachine(tmp_path, clock=lambda: 2.0)

    assert restarted.state is TunerState.EXTRACTING
    assert restarted.resume() is TunerState.READY
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["to"] for event in events[-3:]] == [
        "ROLLING_BACK",
        "WAKING",
        "READY",
    ]
    assert all(event["recovery"] for event in events[-3:])
