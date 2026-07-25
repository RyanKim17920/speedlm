"""Parse vLLM Prometheus /metrics exposition and compute deltas."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CounterResetError(RuntimeError):
    """Raised when a Prometheus counter decreased between scrapes."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AcceptanceStatus(Enum):
    """Status of acceptance-rate data."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """A single point-in-time snapshot of vLLM counters.

    All fields are non-negative floats parsed from Prometheus text exposition.
    Missing counters default to 0.0 (they were never emitted).
    """

    # Token-level counters
    generated_tokens: float = 0.0
    prompt_tokens: float = 0.0

    # Latency / time
    time_per_output_token_ns: float = 0.0  # total TPOT in nanoseconds
    num_requests_running: float = 0.0
    num_requests_swapped: float = 0.0
    num_requests_waiting: float = 0.0

    # Speculative decoding / draft counters
    _drafted_tokens: float = field(default=0.0, repr=False)
    _accepted_tokens: float = field(default=0.0, repr=False)
    _draft_accept_count: float = field(default=0.0, repr=False)
    _draft_reject_count: float = field(default=0.0, repr=False)

    # Track which draft counters were actually present in the scrape
    _has_draft_counters: bool = field(default=False, repr=False)

    @property
    def drafted_tokens(self) -> float:
        return self._drafted_tokens

    @property
    def accepted_tokens(self) -> float:
        return self._accepted_tokens

    @property
    def draft_accept_count(self) -> float:
        return self._draft_accept_count

    @property
    def draft_reject_count(self) -> float:
        return self._draft_reject_count

    @property
    def has_draft_counters(self) -> bool:
        return self._has_draft_counters

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft attempts that were accepted (0..1)."""
        total = self._draft_accept_count + self._draft_reject_count
        if total == 0:
            return 0.0
        return self._draft_accept_count / total

    @property
    def mean_accepted_length(self) -> float:
        """Average accepted-token length per draft accept."""
        if self._draft_accept_count == 0:
            return 0.0
        return self._accepted_tokens / self._draft_accept_count

    @property
    def tpot_ms(self) -> float:
        """Time-per-output-token in milliseconds."""
        if self.generated_tokens == 0:
            return 0.0
        return (self.time_per_output_token_ns / self.generated_tokens) / 1_000_000

    @property
    def output_tok_per_sec(self) -> float:
        """Output throughput in tokens per second."""
        if self.time_per_output_token_ns == 0:
            return 0.0
        return self.generated_tokens / (self.time_per_output_token_ns / 1e9)


# Well-known counter names (vLLM Prometheus)
COUNTER_NAMES: Final[dict[str, str]] = {
    "generated_tokens": "generated_tokens_total",
    "prompt_tokens": "prompt_tokens_total",
    "time_per_output_token_ns": "time_per_output_token_ns_sum",
    "drafted_tokens": "vllm:speculated_tokens_total",
    "accepted_tokens": "vllm:accepted_tokens_total",
    "draft_accept_count": "vllm:accept_token_count_total",
    "draft_reject_count": "vllm:reject_token_count_total",
}

# Gauge names (point-in-time, not deltas)
GAUGE_NAMES: Final[dict[str, str]] = {
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_swapped": "vllm:num_requests_swapped",
    "num_requests_waiting": "vllm:num_requests_waiting",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\S+?)(?:\{[^}]*\})?\s+([\d.eE+\-]+)$"
)


def parse_metrics(text: str) -> MetricsSnapshot:
    """Parse vLLM Prometheus text exposition format into a snapshot.

    Args:
        text: Raw /metrics response body.

    Returns:
        A frozen :class:`MetricsSnapshot` with all parsed values.
    """
    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    has_draft = False

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            continue
        name, value_str = m.group(1), m.group(2)
        try:
            value = float(value_str)
        except ValueError:
            continue

        for field_name, prom_name in COUNTER_NAMES.items():
            if name == prom_name:
                counters[field_name] = value
                if field_name in (
                    "drafted_tokens", "accepted_tokens",
                    "draft_accept_count", "draft_reject_count",
                ):
                    has_draft = True
                break

        for field_name, prom_name in GAUGE_NAMES.items():
            if name == prom_name:
                gauges[field_name] = value
                break

    return MetricsSnapshot(
        generated_tokens=counters.get("generated_tokens", 0.0),
        prompt_tokens=counters.get("prompt_tokens", 0.0),
        time_per_output_token_ns=counters.get("time_per_output_token_ns", 0.0),
        num_requests_running=gauges.get("num_requests_running", 0.0),
        num_requests_swapped=gauges.get("num_requests_swapped", 0.0),
        num_requests_waiting=gauges.get("num_requests_waiting", 0.0),
        _drafted_tokens=counters.get("drafted_tokens", 0.0),
        _accepted_tokens=counters.get("accepted_tokens", 0.0),
        _draft_accept_count=counters.get("draft_accept_count", 0.0),
        _draft_reject_count=counters.get("draft_reject_count", 0.0),
        _has_draft_counters=has_draft,
    )


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MetricsDelta:
    """Delta between two metrics snapshots.

    Attributes:
        reset_detected: True if any counter went backwards (invalid delta).
        acceptance_available: True if draft counters were present.
        drafted_tokens: Change in drafted tokens.
        accepted_tokens: Change in accepted tokens.
        acceptance_rate: Acceptance rate over the delta window.
        mean_accepted_length: Mean accepted length over the delta window.
        tpot_ms: Time-per-output-token in milliseconds.
        output_tok_per_sec: Output throughput in tokens/s.
    """

    reset_detected: bool
    acceptance_available: bool
    drafted_tokens: float
    accepted_tokens: float
    acceptance_rate: float
    mean_accepted_length: float
    tpot_ms: float
    output_tok_per_sec: float


def compute_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    """Compute per-counter deltas between two snapshots.

    Raises:
        CounterResetError: If any counter decreased (vLLM restart detected).

    Returns:
        A frozen :class:`MetricsDelta` with derived metrics.
    """
    # Check for counter resets on ALL monotonic counters
    counter_fields = [
        "generated_tokens", "prompt_tokens",
        "drafted_tokens", "accepted_tokens",
        "draft_accept_count", "draft_reject_count",
    ]
    for attr in counter_fields:
        after_val = getattr(after, attr)
        before_val = getattr(before, attr)
        if after_val < before_val:
            raise CounterResetError(
                f"Counter '{attr}' reset: {before_val} -> {after_val}"
            )

    delta_draft_accept = after._draft_accept_count - before._draft_accept_count
    delta_draft_reject = after._draft_reject_count - before._draft_reject_count
    total_attempts = delta_draft_accept + delta_draft_reject

    if after.has_draft_counters:
        acceptance_rate = (
            delta_draft_accept / total_attempts if total_attempts > 0 else 0.0
        )
        mean_accepted_len = (
            (after.accepted_tokens - before.accepted_tokens) / delta_draft_accept
            if delta_draft_accept > 0 else 0.0
        )
    else:
        acceptance_rate = 0.0
        mean_accepted_len = 0.0

    delta_gen = after.generated_tokens - before.generated_tokens
    delta_tpot_ns = after.time_per_output_token_ns - before.time_per_output_token_ns

    tpot_ms = (
        (delta_tpot_ns / delta_gen) / 1_000_000 if delta_gen > 0 else 0.0
    )
    elapsed_s = delta_tpot_ns / 1e9 if delta_tpot_ns > 0 else 0.0
    tok_per_sec = delta_gen / elapsed_s if elapsed_s > 0 else 0.0

    return MetricsDelta(
        reset_detected=False,
        acceptance_available=after.has_draft_counters,
        drafted_tokens=after.drafted_tokens - before.drafted_tokens,
        accepted_tokens=after.accepted_tokens - before.accepted_tokens,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_len,
        tpot_ms=tpot_ms,
        output_tok_per_sec=tok_per_sec,
    )
