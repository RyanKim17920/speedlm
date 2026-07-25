"""Shared dependency-light contracts for cross-feature collaboration."""

from typing import Protocol


class SamplingOptions(Protocol):
    """Sampling fields consumed by trace and benchmark workflows."""

    @property
    def temperature(self) -> float: ...

    @property
    def top_p(self) -> float: ...

    @property
    def seed(self) -> int: ...
