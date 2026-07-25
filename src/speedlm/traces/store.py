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
    atomic_write_text,
    read_jsonl,
)
from speedlm.traces.redact import RedactionReport, Redactor

logger = logging.getLogger(__name__)

_REDACTION_PLACEHOLDER_RE = re.compile(r"<REDACTED:([a-z0-9_]+)>")

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
    """Immutable record of a proxied chat/completion request."""

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
    token_count_source: str = "measured"

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
            self.token_count_source,
        )

    @property
    def total_tokens(self) -> int | None:
        """Sum of token counts, or ``None`` when either count is unknown."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict. Messages/tool_calls become lists."""
        return {
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
        }

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
        unknown = set(data.keys()) - required_keys - {"token_count_source"}
        if unknown:
            raise TraceError(f"unknown key: {sorted(unknown)[0]}")

        msgs = data["messages"]
        tcs = data["tool_calls"]
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
            token_count_source=data.get(
                "token_count_source",
                "estimated"
                if data["prompt_tokens"] is None or data["completion_tokens"] is None
                else "measured",
            ),
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
    token_count_source: str,
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
    if not isinstance(tool_calls, (tuple, list)):
        raise TraceError("tool_calls must be a sequence")
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, Mapping):
            raise TraceError(f"tool_calls[{i}] must be a mapping")
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "redaction_counts",
            MappingProxyType(dict(self.redaction_counts)),
        )


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
                return None
            if not isinstance(redacted, Mapping):
                logger.warning("dropping trace after redaction returned a non-mapping")
                return None
            trace_record = dict(redacted)
        else:
            report = RedactionReport({})

        if not _append_jsonl(self._path, trace_record):
            return None
        return report

    def iter_records(self) -> Iterator[TraceRecord]:
        """Yield records from the JSONL file.

        A missing or empty file yields nothing (not an error).
        """
        if not self._path.exists():
            return
        try:
            for raw in read_jsonl(self._path):
                yield TraceRecord.from_dict(raw)
        except (StorageError, TraceError):
            return

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
        )

    def prune(self, *, now: float | None = None) -> int:
        """Prune by age then token budget; return number of records dropped.

        1. Drop records where ``now - record.timestamp > max_age_days * 86400``.
        2. If surviving ``total_tokens > max_tokens``, drop oldest-first (stable).
        3. Rewrite the file atomically only if at least one record was dropped.
        4. A missing file returns 0.
        5. Records exactly at the age boundary are KEPT (strict > comparison).
        """
        if now is None:
            now = time.time()

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
    total = record.total_tokens
    if total is not None:
        return total
    prompt_tokens, completion_tokens = estimate_message_tokens(list(record.messages))
    return prompt_tokens + completion_tokens
