from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from speedlm.storage import (
    StorageError,
    _append_jsonl,
    _exclusive_file_lock,
    atomic_write_json,
    atomic_write_text,
    count_jsonl,
    read_jsonl,
)
from speedlm.traces.redact import RedactionReport, Redactor

logger = logging.getLogger(__name__)

_REDACTION_PLACEHOLDER_RE = re.compile(r"<REDACTED:([a-z0-9_]+)>")
_PROVENANCE_TAGS = {"generated", "client_supplied"}
DROP_REASONS = (
    "lock_timeout",
    "capture_error",
    "body_overflow",
    "redaction_failure",
    "stream_observer_error",
    "shutdown_pending",
)

# ── Exceptions ──────────────────────────────────────────────────────────────

class TraceError(ValueError):
    """Raised when a trace record fails validation or I/O errors occur."""


class TraceBufferOptions(Protocol):
    """Buffer limits required to configure a trace store."""

    @property
    def max_tokens(self) -> int: ...

    @property
    def max_age_days(self) -> float: ...


class RedactionOptions(Protocol):
    """Privacy setting required to configure a trace store."""

    @property
    def enabled(self) -> bool: ...


class TraceRedactor(Protocol):
    """Structure-preserving redactor used by the persistence boundary."""

    def redact(self, value: Any) -> tuple[Any, RedactionReport]: ...


# ── TraceRecord ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Immutable record of a proxied chat/completion request.

    Each message may carry a ``provenance_tag``. Only the exact value
    ``"generated"`` establishes provider authorship; an absent tag is
    deliberately treated as client-supplied/unknown by consumers.
    """

    id: str
    timestamp: float
    model: str
    messages: tuple[Mapping[str, Any], ...]
    tool_calls: tuple[Mapping[str, Any], ...]
    temperature: float
    top_p: float
    seed: int
    prompt_tokens: int | None
    completion_tokens: int | None
    tools: tuple[Mapping[str, Any], ...] = ()
    token_count_source: str = "measured"
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    exchange_id: str | None = None

    def __post_init__(self) -> None:
        _validate_record(
            self.id,
            self.timestamp,
            self.model,
            self.messages,
            self.tool_calls,
            self.temperature,
            self.top_p,
            self.seed,
            self.prompt_tokens,
            self.completion_tokens,
            self.tools,
            self.token_count_source,
            self.finish_reason,
            self.stop_reason,
            self.exchange_id,
        )

    @property
    def total_tokens(self) -> int | None:
        """Sum of token counts, or ``None`` when either count is unknown."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict. Messages/tool_calls become lists."""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "messages": [dict(m) for m in self.messages],
            "tool_calls": [dict(tc) for tc in self.tool_calls],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "token_count_source": self.token_count_source,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
        }
        if self.tools:
            result["tools"] = [dict(tool) for tool in self.tools]
        if self.exchange_id is not None:
            result["exchange_id"] = self.exchange_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceRecord:
        """Deserialize a dict to a TraceRecord.

        Fail closed: missing keys or unknown keys raise TraceError.
        """
        required_keys = {
            "id", "timestamp", "model", "messages", "tool_calls",
            "temperature", "top_p", "seed", "prompt_tokens", "completion_tokens",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise TraceError(f"missing required key: {sorted(missing)[0]}")
        optional_keys = {
            "exchange_id",
            "finish_reason",
            "stop_reason",
            "token_count_source",
            "tools",
        }
        unknown = set(data.keys()) - required_keys - optional_keys
        if unknown:
            raise TraceError(f"unknown key: {sorted(unknown)[0]}")

        msgs = data["messages"]
        tcs = data["tool_calls"]
        raw_tools = data.get("tools", ())
        if not isinstance(raw_tools, (tuple, list)):
            raise TraceError("tools must be a sequence")
        for index, tool in enumerate(raw_tools):
            if not isinstance(tool, Mapping):
                raise TraceError(f"tools[{index}] must be a mapping")
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            model=data["model"],
            messages=tuple(dict(m) for m in msgs),
            tool_calls=tuple(dict(t) for t in tcs),
            temperature=data["temperature"],
            top_p=data["top_p"],
            seed=data["seed"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            tools=tuple(dict(tool) for tool in raw_tools),
            token_count_source=data.get(
                "token_count_source",
                "estimated"
                if (
                    data["prompt_tokens"] is None
                    or data["completion_tokens"] is None
                    or (
                        data["prompt_tokens"] == 0
                        and data["completion_tokens"] == 0
                    )
                )
                else "measured",
            ),
            finish_reason=data.get("finish_reason"),
            stop_reason=data.get("stop_reason"),
            exchange_id=data.get("exchange_id"),
        )


def _validate_record(
    rid: str,
    timestamp: float,
    model: str,
    messages: tuple[Mapping[str, Any], ...],
    tool_calls: tuple[Mapping[str, Any], ...],
    temperature: float,
    top_p: float,
    seed: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    tools: tuple[Mapping[str, Any], ...],
    token_count_source: str,
    finish_reason: str | None,
    stop_reason: int | str | None,
    exchange_id: str | None,
) -> None:
    if not isinstance(rid, str) or not rid:
        raise TraceError("id must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise TraceError("model must be a non-empty string")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise TraceError("timestamp must be a real number (not bool)")
    if timestamp < 0:
        raise TraceError("timestamp must be >= 0")
    if not isinstance(messages, (tuple, list)) or not messages:
        raise TraceError("messages must be a non-empty sequence")
    for i, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise TraceError(f"messages[{i}] must be a mapping")
        if "role" not in msg:
            raise TraceError(f"messages[{i}] missing 'role'")
        role = msg["role"]
        if not isinstance(role, str) or not role:
            raise TraceError(f"messages[{i}]['role'] must be a non-empty string")
        if "content" not in msg:
            raise TraceError(f"messages[{i}] missing 'content'")
        provenance_tag = msg.get("provenance_tag")
        if provenance_tag is not None and provenance_tag not in _PROVENANCE_TAGS:
            raise TraceError(
                f"messages[{i}]['provenance_tag'] must be "
                "'generated' or 'client_supplied'"
            )
    if not isinstance(tool_calls, (tuple, list)):
        raise TraceError("tool_calls must be a sequence")
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, Mapping):
            raise TraceError(f"tool_calls[{i}] must be a mapping")
    if not isinstance(tools, (tuple, list)):
        raise TraceError("tools must be a sequence")
    for i, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise TraceError(f"tools[{i}] must be a mapping")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TraceError("temperature must be a number (not bool)")
    if temperature < 0:
        raise TraceError("temperature must be >= 0")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TraceError("top_p must be a number (not bool)")
    if not (0 < top_p <= 1):
        raise TraceError("top_p must satisfy 0 < top_p <= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TraceError("seed must be an int (not bool)")
    if seed < 0:
        raise TraceError("seed must be >= 0")
    if prompt_tokens is not None:
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise TraceError("prompt_tokens must be an int or None (not bool)")
        if prompt_tokens < 0:
            raise TraceError("prompt_tokens must be >= 0")
    if completion_tokens is not None:
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
            raise TraceError("completion_tokens must be an int or None (not bool)")
        if completion_tokens < 0:
            raise TraceError("completion_tokens must be >= 0")
    if token_count_source not in {"measured", "estimated"}:
        raise TraceError("token_count_source must be 'measured' or 'estimated'")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TraceError("finish_reason must be a string or None")
    if stop_reason is not None and not (
        isinstance(stop_reason, str)
        or not isinstance(stop_reason, bool)
        and isinstance(stop_reason, int)
    ):
        raise TraceError("stop_reason must be a string, int, or None")
    if exchange_id is not None and (
        not isinstance(exchange_id, str) or not exchange_id
    ):
        raise TraceError("exchange_id must be a non-empty string or None")


def estimate_message_tokens(
    messages: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[int, int]:
    """Cheap chars/4 estimate split into prompt and assistant tokens."""
    prompt_chars = 0
    completion_chars = 0
    for message in messages:
        chars = len(json.dumps(dict(message), ensure_ascii=False, default=str))
        if message.get("role") == "assistant":
            completion_chars += chars
        else:
            prompt_chars += chars
    prompt_tokens = (prompt_chars + 3) // 4
    completion_tokens = (completion_chars + 3) // 4
    return prompt_tokens, completion_tokens


# ── TraceStats ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceStats:
    """Aggregated statistics over a trace store."""

    count: int
    tokens: int
    oldest: float | None
    newest: float | None
    unknown_token_records: int = 0
    measured_tokens: int = 0
    estimated_tokens: int = 0
    redacted_records: int = 0
    redaction_counts: Mapping[str, int] = field(default_factory=dict)
    total_dropped: int = 0
    drops_by_reason: Mapping[str, int] = field(
        default_factory=lambda: dict.fromkeys(DROP_REASONS, 0)
    )
    truncated_at_line: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "redaction_counts",
            MappingProxyType(dict(self.redaction_counts)),
        )
        object.__setattr__(
            self,
            "drops_by_reason",
            MappingProxyType(dict(self.drops_by_reason)),
        )

    @property
    def dropped(self) -> int:
        """Compatibility-friendly shorthand for the total dropped count."""
        return self.total_dropped

    @property
    def drop_counts(self) -> Mapping[str, int]:
        """Compatibility-friendly shorthand for the per-reason counters."""
        return self.drops_by_reason


@dataclass(frozen=True, slots=True)
class RedactionSummary:
    """Aggregate reports for one batch of newly written traces."""

    records: int
    counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def summarize_redactions(
    reports: Iterable[RedactionReport | None],
) -> RedactionSummary:
    """Aggregate successful append reports without retaining sensitive values."""
    counts: Counter[str] = Counter()
    records = 0
    for report in reports:
        if report is None:
            continue
        counts.update(report.counts)
        if report.total:
            records += 1
    return RedactionSummary(records=records, counts=dict(sorted(counts.items())))


def format_redaction_summary(summary: RedactionSummary) -> str:
    """Render a secret-free summary suitable for ``traces import`` output."""
    categories = ", ".join(
        f"{category}: {count}" for category, count in summary.counts.items()
    )
    if not categories:
        categories = "none"
    return (
        f"redacted: {summary.records} record(s), "
        f"{summary.total} replacement(s) [{categories}]"
    )


# ── TraceStore ──────────────────────────────────────────────────────────────

class TraceStore:
    """File-backed trace buffer with age and token-bounded pruning."""

    def __init__(
        self,
        path: Path,
        *,
        max_tokens: int = 8_000_000,
        max_age_days: float = 14.0,
        redaction_enabled: bool = True,
        redactor: TraceRedactor | None = None,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise TraceError("max_tokens must be a positive integer")
        if (isinstance(max_age_days, bool)
                or not isinstance(max_age_days, (int, float))
                or max_age_days <= 0):
            raise TraceError("max_age_days must be a positive number")
        if not isinstance(redaction_enabled, bool):
            raise TraceError("redaction_enabled must be a bool")
        self._path: Path = path
        self._max_tokens: int = max_tokens
        self._max_age_days: float = float(max_age_days)
        # One redactor per long-lived store, safe for concurrent calls because
        # Redactor scopes all mutable traversal state to each redact() call.
        self._redactor: TraceRedactor | None = (
            redactor or Redactor() if redaction_enabled else None
        )
        #: Running token total for the buffer, so ``append`` can decide whether
        #: pruning is due without re-reading the file on every request. ``None``
        #: means "not yet known"; it is seeded from disk on the first append and
        #: recomputed after each prune. Another process appending concurrently
        #: only makes this an under-estimate, and that process runs the same
        #: check, so the ceiling still holds.
        self._buffered_tokens: int | None = None

    @classmethod
    def from_config(
        cls,
        path: Path,
        buffer: TraceBufferOptions,
        *,
        redaction: RedactionOptions | None = None,
    ) -> TraceStore:
        """Create a store from trace-buffer and optional redaction config."""
        return cls(
            path,
            max_tokens=buffer.max_tokens,
            max_age_days=buffer.max_age_days,
            redaction_enabled=redaction.enabled if redaction is not None else True,
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def stats_path(self) -> Path:
        """Return the drop-counter sidecar adjacent to the trace JSONL file."""
        return self._path.with_suffix(".stats.json")

    def record_drop(self, reason: str) -> bool:
        """Persist one dropped trace without ever propagating an I/O failure."""
        if reason not in DROP_REASONS:
            logger.warning("cannot count trace drop with unknown reason: %s", reason)
            return False
        try:
            with _exclusive_file_lock(self.stats_path) as acquired:
                if not acquired:
                    return False
                counts = self._read_drop_counts_unlocked()
                counts[reason] += 1
                atomic_write_json(
                    self.stats_path,
                    {
                        "total_dropped": sum(counts.values()),
                        "drops_by_reason": counts,
                    },
                )
        except Exception as exc:
            logger.warning(
                "failed to persist trace drop counter for %s: %s",
                reason,
                exc,
            )
            return False
        return True

    def _read_drop_counts_unlocked(self) -> dict[str, int]:
        counts = dict.fromkeys(DROP_REASONS, 0)
        if not self.stats_path.exists():
            return counts
        raw = json.loads(self.stats_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("trace stats sidecar is not a JSON object")
        raw_counts = raw.get("drops_by_reason", {})
        if not isinstance(raw_counts, Mapping):
            raise ValueError("trace stats sidecar has invalid drops_by_reason")
        for reason in DROP_REASONS:
            value = raw_counts.get(reason, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"trace stats sidecar has invalid count for {reason}")
            counts[reason] = value
        return counts

    def _read_drop_counts(self) -> dict[str, int]:
        if not self.stats_path.exists():
            return dict.fromkeys(DROP_REASONS, 0)
        try:
            with _exclusive_file_lock(self.stats_path) as acquired:
                if not acquired:
                    return dict.fromkeys(DROP_REASONS, 0)
                return self._read_drop_counts_unlocked()
        except Exception as exc:
            logger.warning("failed to read trace drop counters: %s", exc)
            return dict.fromkeys(DROP_REASONS, 0)

    def append(self, record: TraceRecord) -> RedactionReport | None:
        """Redact and append one record, dropping it if redaction fails.

        Returning ``None`` means privacy processing failed and nothing was
        written. Storing the original record would turn a redactor fault into a
        plaintext secret leak, so this persistence boundary fails closed.
        """
        trace_record = record.to_dict()
        if self._redactor is not None:
            try:
                redacted, report = self._redactor.redact(trace_record)
            except Exception as exc:
                logger.warning(
                    "dropping trace after redaction failure (%s)",
                    type(exc).__name__,
                )
                self.record_drop("redaction_failure")
                return None
            if not isinstance(redacted, Mapping):
                logger.warning("dropping trace after redaction returned a non-mapping")
                self.record_drop("redaction_failure")
                return None
            trace_record = dict(redacted)
        else:
            report = RedactionReport({})

        if not _append_jsonl(self._path, trace_record):
            self.record_drop("lock_timeout")
            return None
        self._enforce_budget_after_append(record)
        return report

    def _enforce_budget_after_append(self, record: TraceRecord) -> None:
        """Prune when the newly appended record pushed the buffer over budget.

        Pruning rewrites the whole file, so it must not run on every append:
        this is the live capture path. The running total makes the common case
        a single addition and defers the full read to the crossings.
        """
        if self._buffered_tokens is None:
            self._buffered_tokens = self._measure_buffered_tokens()
        self._buffered_tokens += _accounting_tokens(record)
        if self._buffered_tokens <= self._max_tokens:
            return
        try:
            self._prune(now=None, enforce_age=False)
        except Exception as exc:  # pragma: no cover - defensive on the hot path
            logger.warning("trace prune after append failed: %s", exc)
            return
        self._buffered_tokens = self._measure_buffered_tokens()

    def _measure_buffered_tokens(self) -> int:
        """Sum the on-disk token cost, treating unreadable tails as zero."""
        total = 0
        try:
            for rec in self.iter_records():
                total += _accounting_tokens(rec)
        except (StorageError, TraceError) as exc:
            logger.warning("trace token accounting truncated: %s", exc)
        return total

    def count_records(self) -> int:
        """Return how many records the buffer holds, without deserializing any.

        This is the cursor a consumer needs to address the *tail* of the
        buffer.  It is deliberately parse-free: the point of an incremental
        consumer is that the records it does not want cost it nothing.
        """
        return count_jsonl(self._path)

    def iter_records(self, *, start: int = 0) -> Iterator[TraceRecord]:
        """Yield records from the JSONL file, beginning at offset *start*.

        ``start`` is a record offset, not a byte offset, and counts blank
        lines out the same way :meth:`count_records` does, so
        ``iter_records(start=count_records() - n)`` yields exactly the newest
        ``n`` records.  Records before *start* are never deserialized.

        A missing or empty file yields nothing (not an error).
        """
        if not self._path.exists():
            return
        for raw in read_jsonl(self._path, skip=start):
            yield TraceRecord.from_dict(raw)

    def stats(self) -> TraceStats:
        """Compute aggregate statistics over all records."""
        count = 0
        tokens = 0
        unknown_token_records = 0
        measured_tokens = 0
        estimated_tokens = 0
        redacted_records = 0
        redaction_counts: Counter[str] = Counter()
        oldest: float | None = None
        newest: float | None = None
        truncated_at_line: int | None = None
        try:
            for rec in self.iter_records():
                count += 1
                record_tokens = _accounting_tokens(rec)
                tokens += record_tokens
                if rec.token_count_source == "measured" and rec.total_tokens is not None:
                    measured_tokens += record_tokens
                else:
                    estimated_tokens += record_tokens
                    if rec.total_tokens is None:
                        unknown_token_records += 1
                record_redactions = Counter(
                    _REDACTION_PLACEHOLDER_RE.findall(
                        json.dumps(rec.to_dict(), ensure_ascii=False)
                    )
                )
                if record_redactions:
                    redacted_records += 1
                    redaction_counts.update(record_redactions)
                if oldest is None or rec.timestamp < oldest:
                    oldest = rec.timestamp
                if newest is None or rec.timestamp > newest:
                    newest = rec.timestamp
        except StorageError as exc:
            truncated_at_line = exc.line_number
            if truncated_at_line is None:
                raise
        drops_by_reason = self._read_drop_counts()
        return TraceStats(
            count=count,
            tokens=tokens,
            oldest=oldest,
            newest=newest,
            unknown_token_records=unknown_token_records,
            measured_tokens=measured_tokens,
            estimated_tokens=estimated_tokens,
            redacted_records=redacted_records,
            redaction_counts=dict(sorted(redaction_counts.items())),
            total_dropped=sum(drops_by_reason.values()),
            drops_by_reason=drops_by_reason,
            truncated_at_line=truncated_at_line,
        )

    def prune(self, *, now: float | None = None) -> int:
        """Prune by age then token budget; return number of records dropped.

        1. Drop records where ``now - record.timestamp > max_age_days * 86400``.
        2. If surviving ``total_tokens > max_tokens``, drop oldest-first (stable).
        3. Rewrite the file atomically only if at least one record was dropped.
        4. A missing file returns 0.
        5. Records exactly at the age boundary are KEPT (strict > comparison).
        """
        return self._prune(now=now, enforce_age=True)

    def _prune(self, *, now: float | None, enforce_age: bool) -> int:
        """Prune the buffer, optionally skipping the age filter.

        ``append`` enforces only the token ceiling: age is measured against a
        caller-supplied clock, and silently applying wall-clock retention as a
        side effect of a capture would delete records the caller never asked
        about. Unbounded disk growth is the append-path risk; retention stays
        with the callers that schedule it.
        """
        if now is None:
            now = time.time()

        if not self._path.exists():
            return 0

        with _exclusive_file_lock(self._path) as acquired:
            if not acquired:
                return 0

            if not self._path.exists():
                return 0

            records: list[TraceRecord] = []
            try:
                for rec in self.iter_records():
                    records.append(rec)
            except TraceError:
                return 0

            if not records:
                return 0

            # Sort ascending by timestamp (stable)
            records.sort(key=lambda r: r.timestamp)

            initial_count = len(records)
            age_seconds = self._max_age_days * 86400.0

            # Step 1: age filter (strict > — boundary records kept)
            if enforce_age:
                records = [r for r in records if (now - r.timestamp) <= age_seconds]

            # Step 2: token budget — drop oldest-first
            total = sum(_accounting_tokens(record) for record in records)
            while total > self._max_tokens and records:
                removed = records.pop(0)
                total -= _accounting_tokens(removed)

            dropped = initial_count - len(records)
            if dropped == 0:
                return 0

            # Step 3: atomic rewrite in ascending-timestamp order
            lines = [json.dumps(r.to_dict()) for r in records]
            text = "\n".join(lines)
            if text:
                text += "\n"
            try:
                atomic_write_text(self._path, text)
            except OSError as exc:
                logger.warning("leaving trace file unchanged after prune write failed: %s", exc)
                return 0
            return dropped


def _accounting_tokens(record: TraceRecord) -> int:
    """Return the token cost the prune budget charges *record*.

    A reported total of zero is never trusted as a measurement: an upstream
    that answers ``usage: {"prompt_tokens": 0, "completion_tokens": 0}``
    alongside megabytes of message text would otherwise sit in the buffer at
    zero cost and defeat the token ceiling entirely. Zero and missing counts
    both fall back to the estimate.
    """
    total = record.total_tokens
    if total is not None and total > 0:
        return total
    prompt_tokens, completion_tokens = estimate_message_tokens(list(record.messages))
    return prompt_tokens + completion_tokens
