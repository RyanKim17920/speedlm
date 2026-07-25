"""Native MTP backend contract.

SpeedLM's pinned runtime does not expose a validated native-MTP materialization
path yet.  The class is intentionally a fail-closed protocol-complete stub so
callers can select the backend without pretending that an artifact is valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from speedlm.training.base import AbortCheck, TrainingBackendError


class MTPBackendUnavailable(TrainingBackendError):
    """Native MTP is selected but has not been validated for this runtime."""


class MTPBackend:
    """Protocol-complete documented stub for a future native MTP backend."""

    @staticmethod
    def _unavailable() -> NoReturn:
        raise MTPBackendUnavailable(
            "native MTP training/materialization is not validated for the pinned "
            "SpeedLM runtime"
        )

    def prepare(self, work_dir: Path, *, should_abort: AbortCheck) -> Any:
        del work_dir, should_abort
        self._unavailable()

    def extract(
        self,
        prepared: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        del prepared, work_dir, should_abort
        self._unavailable()

    def train(
        self,
        extracted: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        del extracted, work_dir, should_abort
        self._unavailable()

    def materialize(
        self,
        trained: Any,
        work_dir: Path,
        *,
        should_abort: AbortCheck,
    ) -> Any:
        del trained, work_dir, should_abort
        self._unavailable()

    def validate(self, artifact: Any, *, should_abort: AbortCheck) -> None:
        del artifact, should_abort
        self._unavailable()
