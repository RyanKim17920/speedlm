"""Crash-safe state tracking for the idle tuner."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from speedlm.storage import append_jsonl, atomic_write_json


class StateError(RuntimeError):
    """Base class for tuner state errors."""


class IllegalTransitionError(StateError):
    """Raised when a transition is not present in the transition table."""


class TunerState(StrEnum):
    """Durable stages in one tuning cycle."""

    READY = "READY"
    QUIESCING = "QUIESCING"
    SLEEPING = "SLEEPING"
    EXTRACTING = "EXTRACTING"
    TRAINING = "TRAINING"
    CANDIDATE_STARTING = "CANDIDATE_STARTING"
    BENCHMARKING = "BENCHMARKING"
    PROMOTING = "PROMOTING"
    ROLLING_BACK = "ROLLING_BACK"
    WAKING = "WAKING"


_ACTIVE_STATES = frozenset(TunerState) - {
    TunerState.READY,
    TunerState.ROLLING_BACK,
    TunerState.WAKING,
}

VALID_TRANSITIONS: Mapping[TunerState, frozenset[TunerState]] = {
    TunerState.READY: frozenset({TunerState.QUIESCING}),
    TunerState.QUIESCING: frozenset({TunerState.SLEEPING, TunerState.ROLLING_BACK}),
    TunerState.SLEEPING: frozenset({TunerState.EXTRACTING, TunerState.ROLLING_BACK}),
    TunerState.EXTRACTING: frozenset({TunerState.TRAINING, TunerState.ROLLING_BACK}),
    TunerState.TRAINING: frozenset(
        {TunerState.CANDIDATE_STARTING, TunerState.ROLLING_BACK}
    ),
    TunerState.CANDIDATE_STARTING: frozenset(
        {TunerState.BENCHMARKING, TunerState.ROLLING_BACK}
    ),
    TunerState.BENCHMARKING: frozenset(
        {TunerState.PROMOTING, TunerState.ROLLING_BACK}
    ),
    TunerState.PROMOTING: frozenset({TunerState.WAKING, TunerState.ROLLING_BACK}),
    TunerState.ROLLING_BACK: frozenset({TunerState.WAKING}),
    TunerState.WAKING: frozenset({TunerState.READY, TunerState.ROLLING_BACK}),
}


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """The current durable tuner state."""

    state: TunerState
    sequence: int
    updated_at: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "sequence": self.sequence,
            "updated_at": self.updated_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateSnapshot:
        try:
            state = TunerState(value["state"])
            sequence = value["sequence"]
            updated_at = value["updated_at"]
            reason = value["reason"]
        except (KeyError, TypeError, ValueError) as exc:
            raise StateError("invalid tuner state file") from exc
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise StateError("state sequence must be a non-negative integer")
        if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
            raise StateError("state updated_at must be numeric")
        if not isinstance(reason, str):
            raise StateError("state reason must be a string")
        return cls(
            state=state,
            sequence=sequence,
            updated_at=float(updated_at),
            reason=reason,
        )


class TunerStateMachine:
    """Persist transitions atomically and append an fsynced audit event."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._root = root
        self._state_path = root / "state.json"
        self._events_path = root / "events.jsonl"
        self._clock = clock
        root.mkdir(parents=True, exist_ok=True)
        self._snapshot = self._load_or_initialize()

    @property
    def snapshot(self) -> StateSnapshot:
        return self._snapshot

    @property
    def state(self) -> TunerState:
        return self._snapshot.state

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def events_path(self) -> Path:
        return self._events_path

    def transition(self, target: TunerState, *, reason: str = "") -> StateSnapshot:
        """Move to *target* if legal, persisting state before its audit event."""
        current = self._snapshot
        if target not in VALID_TRANSITIONS[current.state]:
            raise IllegalTransitionError(
                f"illegal tuner transition {current.state.value} -> {target.value}"
            )
        next_snapshot = StateSnapshot(
            state=target,
            sequence=current.sequence + 1,
            updated_at=self._clock(),
            reason=reason,
        )
        atomic_write_json(self._state_path, next_snapshot.to_dict())
        append_jsonl(
            self._events_path,
            {
                "sequence": next_snapshot.sequence,
                "timestamp": next_snapshot.updated_at,
                "from": current.state.value,
                "to": target.value,
                "reason": reason,
                "recovery": False,
            },
        )
        self._snapshot = next_snapshot
        return next_snapshot

    def resume(self) -> TunerState:
        """Recover an interrupted cycle to ``READY`` using safe transition paths.

        Effects are deliberately not replayed here. The process supervisor must
        independently ensure that the child vLLM starts from the durable active
        artifact. This method only repairs the tuner journal.
        """
        state = self.state
        if state is TunerState.READY:
            return state
        if state in _ACTIVE_STATES:
            self._recovery_transition(TunerState.ROLLING_BACK)
            state = self.state
        if state is TunerState.ROLLING_BACK:
            self._recovery_transition(TunerState.WAKING)
            state = self.state
        if state is TunerState.WAKING:
            self._recovery_transition(TunerState.READY)
        return self.state

    def _load_or_initialize(self) -> StateSnapshot:
        if not self._state_path.exists():
            initial = StateSnapshot(
                state=TunerState.READY,
                sequence=0,
                updated_at=self._clock(),
                reason="initialized",
            )
            atomic_write_json(self._state_path, initial.to_dict())
            append_jsonl(
                self._events_path,
                {
                    "sequence": 0,
                    "timestamp": initial.updated_at,
                    "from": None,
                    "to": TunerState.READY.value,
                    "reason": initial.reason,
                    "recovery": False,
                },
            )
            return initial
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(f"cannot load tuner state from {self._state_path}") from exc
        if not isinstance(raw, dict):
            raise StateError("tuner state file must contain a JSON object")
        return StateSnapshot.from_dict(raw)

    def _recovery_transition(self, target: TunerState) -> None:
        current = self._snapshot
        if target not in VALID_TRANSITIONS[current.state]:
            raise IllegalTransitionError(
                f"no recovery path for {current.state.value} -> {target.value}"
            )
        next_snapshot = StateSnapshot(
            state=target,
            sequence=current.sequence + 1,
            updated_at=self._clock(),
            reason=f"restart recovery from {current.state.value}",
        )
        atomic_write_json(self._state_path, next_snapshot.to_dict())
        append_jsonl(
            self._events_path,
            {
                "sequence": next_snapshot.sequence,
                "timestamp": next_snapshot.updated_at,
                "from": current.state.value,
                "to": target.value,
                "reason": next_snapshot.reason,
                "recovery": True,
            },
        )
        self._snapshot = next_snapshot
