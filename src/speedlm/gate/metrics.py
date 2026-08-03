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

    # Latency / time.  vLLM exposes decode wall-time as the ``_sum`` series of
    # the ``vllm:request_decode_time_seconds`` histogram.
    decode_time_seconds: float = 0.0
    num_requests_running: float = 0.0
    num_requests_swapped: float = 0.0
    num_requests_waiting: float = 0.0

    # Speculative decoding / draft counters
    _drafted_tokens: float = field(default=0.0, repr=False)
    _accepted_tokens: float = field(default=0.0, repr=False)
    _num_drafts: float = field(default=0.0, repr=False)

    # Track which draft counters were actually present in the scrape
    _has_draft_counters: bool = field(default=False, repr=False)

    @property
    def drafted_tokens(self) -> float:
        return self._drafted_tokens

    @property
    def accepted_tokens(self) -> float:
        return self._accepted_tokens

    @property
    def num_drafts(self) -> float:
        return self._num_drafts

    @property
    def has_draft_counters(self) -> bool:
        return self._has_draft_counters

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that the verifier accepted (0..1).

        Matches vLLM's own dashboard definition:
        ``spec_decode_num_accepted_tokens / spec_decode_num_draft_tokens``.
        """
        if self._drafted_tokens == 0:
            return 0.0
        return self._accepted_tokens / self._drafted_tokens

    @property
    def mean_accepted_length(self) -> float:
        """Mean tokens emitted per decode step, including the bonus token.

        Matches vLLM's definition:
        ``1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts``.
        """
        if self._num_drafts == 0:
            return 0.0
        return 1.0 + self._accepted_tokens / self._num_drafts

    @property
    def tpot_ms(self) -> float:
        """Time-per-output-token in milliseconds."""
        if self.generated_tokens == 0:
            return 0.0
        return (self.decode_time_seconds / self.generated_tokens) * 1000.0

    @property
    def output_tok_per_sec(self) -> float:
        """Output throughput in tokens per second."""
        if self.decode_time_seconds == 0:
            return 0.0
        return self.generated_tokens / self.decode_time_seconds


# Well-known counter names, verified against the vLLM Prometheus exposition
# emitted by the vendored runtime (see vllm/v1/metrics/loggers.py and
# vllm/v1/spec_decode/metrics.py).  These names are the contract: if vLLM
# renames them the gate reports acceptance as unavailable rather than zero.
COUNTER_NAMES: Final[dict[str, str]] = {
    "generated_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "decode_time_seconds": "vllm:request_decode_time_seconds_sum",
    "drafted_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "num_drafts": "vllm:spec_decode_num_drafts_total",
}

# Counters whose presence proves the endpoint is reporting speculative decoding.
DRAFT_COUNTER_FIELDS: Final[frozenset[str]] = frozenset(
    {"drafted_tokens", "accepted_tokens", "num_drafts"}
)

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
    counter_by_prom = {prom: field for field, prom in COUNTER_NAMES.items()}
    gauge_by_prom = {prom: field for field, prom in GAUGE_NAMES.items()}
    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    has_draft = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
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

        counter_field = counter_by_prom.get(name)
        if counter_field is not None:
            # vLLM labels every series by engine (and model); one process can
            # expose several engines, so aggregate rather than last-write-wins.
            counters[counter_field] = counters.get(counter_field, 0.0) + value
            if counter_field in DRAFT_COUNTER_FIELDS:
                has_draft = True
            continue

        gauge_field = gauge_by_prom.get(name)
        if gauge_field is not None:
            gauges[gauge_field] = gauges.get(gauge_field, 0.0) + value

    return MetricsSnapshot(
        generated_tokens=counters.get("generated_tokens", 0.0),
        prompt_tokens=counters.get("prompt_tokens", 0.0),
        decode_time_seconds=counters.get("decode_time_seconds", 0.0),
        num_requests_running=gauges.get("num_requests_running", 0.0),
        num_requests_swapped=gauges.get("num_requests_swapped", 0.0),
        num_requests_waiting=gauges.get("num_requests_waiting", 0.0),
        _drafted_tokens=counters.get("drafted_tokens", 0.0),
        _accepted_tokens=counters.get("accepted_tokens", 0.0),
        _num_drafts=counters.get("num_drafts", 0.0),
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

    @property
    def accepted_length_available(self) -> bool:
        """True when :attr:`mean_accepted_length` is a measurement, not a filler.

        :attr:`acceptance_available` keys off ``spec_decode_num_draft_tokens``
        alone, but :attr:`mean_accepted_length` is divided by
        ``spec_decode_num_drafts``, a *different* counter.  An endpoint that
        exposes the first and not the second -- or that drafted tokens without
        completing a draft inside the window -- yields
        ``acceptance_available=True`` beside ``mean_accepted_length=0.0``, and
        0.0 is not a mean accepted length: the floor is 1.0 (the bonus token)
        whenever a draft happened at all.

        The gate's promotion criterion is now the mean-accepted-length delta
        (see :data:`speedlm.gate.decide.GATING_ACCEPTANCE_CRITERION`), so that
        asymmetry would otherwise turn a missing counter into a measured
        ``0.0 - 0.0 = 0.0`` delta and reject every candidate for a reason the
        record does not name.  Asking this instead makes the gap report as
        ``acceptance_unavailable``, which is what it is.
        """
        return self.acceptance_available and self.mean_accepted_length > 0.0


def compute_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    """Compute per-counter deltas between two snapshots.

    Raises:
        CounterResetError: If any counter decreased (vLLM restart detected).

    Returns:
        A frozen :class:`MetricsDelta` with derived metrics.
    """
    # Check for counter resets on ALL monotonic counters
    counter_fields = [
        "generated_tokens", "prompt_tokens", "decode_time_seconds",
        "drafted_tokens", "accepted_tokens", "num_drafts",
    ]
    for attr in counter_fields:
        after_val = getattr(after, attr)
        before_val = getattr(before, attr)
        if after_val < before_val:
            raise CounterResetError(
                f"Counter '{attr}' reset: {before_val} -> {after_val}"
            )

    delta_drafted = after.drafted_tokens - before.drafted_tokens
    delta_accepted = after.accepted_tokens - before.accepted_tokens
    delta_drafts = after.num_drafts - before.num_drafts

    # Acceptance is only a real measurement when the endpoint exposed the
    # speculative counters AND drafting actually happened in the window.  A
    # present-but-idle counter must not be reported as a measured 0% rate.
    acceptance_available = after.has_draft_counters and delta_drafted > 0

    if acceptance_available:
        acceptance_rate = delta_accepted / delta_drafted
        mean_accepted_len = (
            1.0 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0
        )
    else:
        acceptance_rate = 0.0
        mean_accepted_len = 0.0

    delta_gen = after.generated_tokens - before.generated_tokens
    delta_decode_s = after.decode_time_seconds - before.decode_time_seconds

    tpot_ms = (delta_decode_s / delta_gen) * 1000.0 if delta_gen > 0 else 0.0
    tok_per_sec = delta_gen / delta_decode_s if delta_decode_s > 0 else 0.0

    return MetricsDelta(
        reset_detected=False,
        acceptance_available=acceptance_available,
        drafted_tokens=delta_drafted,
        accepted_tokens=delta_accepted,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_len,
        tpot_ms=tpot_ms,
        output_tok_per_sec=tok_per_sec,
    )
