"""Backend contracts for pluggable speculative training methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

AbortCheck = Callable[[], bool]


class TrainingBackendError(RuntimeError):
    """Base class for failures at a speculative-training backend boundary."""


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Backend-neutral model and training metadata used by orchestration."""

    verifier_model: str
    draft_model: str
    from_pretrained: str
    training_params: Mapping[str, object]


@runtime_checkable
class SpeculatorBackend(Protocol):
    """Contract implemented by EAGLE-3, MTP, and future backends.

    The deliberately opaque stage values let each method own its native
    intermediate representation without coupling the orchestrator to a
    particular training library.
    """

    def describe(self) -> BackendInfo:
        """Return backend-neutral model and training metadata."""
        ...

    def prepare(self, work_dir: Path, *, should_abort: AbortCheck) -> Any:
        """Lease traces and create template/mask-specific training rows."""
        ...

    def extract(
        self,
        prepared: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        """Extract verifier-side training signals."""
        ...

    def train(
        self,
        extracted: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        """Fit the speculative backend."""
        ...

    def materialize(
        self,
        trained: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        """Turn a training checkpoint into a deployable artifact."""
        ...

    def validate(self, artifact: Any, *, should_abort: AbortCheck) -> None:
        """Validate a materialized artifact against its verifier."""
        ...
