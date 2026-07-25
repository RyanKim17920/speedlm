from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.config import TraceBufferConfig
from speedlm.storage import StorageError, append_jsonl, atomic_write_text, read_jsonl

# ── Exceptions ──────────────────────────────────────────────────────────────

class TraceError(ValueError):
    """Raised when a trace record fails validation or I/O errors occur."""


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
    prompt_tokens: int
    completion_tokens: int

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
        )

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
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
        unknown = set(data.keys()) - required_keys
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
    prompt_tokens: int,
    completion_tokens: int,
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
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise TraceError("prompt_tokens must be an int (not bool)")
    if prompt_tokens < 0:
        raise TraceError("prompt_tokens must be >= 0")
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
        raise TraceError("completion_tokens must be an int (not bool)")
    if completion_tokens < 0:
        raise TraceError("completion_tokens must be >= 0")


# ── TraceStats ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceStats:
    """Aggregated statistics over a trace store."""

    count: int
    tokens: int
    oldest: float | None
    newest: float | None


# ── TraceStore ──────────────────────────────────────────────────────────────

class TraceStore:
    """File-backed trace buffer with age and token-bounded pruning."""

    def __init__(
        self,
        path: Path,
        *,
        max_tokens: int = 8_000_000,
        max_age_days: float = 14.0,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise TraceError("max_tokens must be a positive integer")
        if (isinstance(max_age_days, bool)
                or not isinstance(max_age_days, (int, float))
                or max_age_days <= 0):
            raise TraceError("max_age_days must be a positive number")
        self._path: Path = path
        self._max_tokens: int = max_tokens
        self._max_age_days: float = float(max_age_days)

    @classmethod
    def from_config(cls, path: Path, buffer: TraceBufferConfig) -> TraceStore:
        """Create a store from a TraceBufferConfig."""
        return cls(path, max_tokens=buffer.max_tokens, max_age_days=buffer.max_age_days)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TraceRecord) -> None:
        """Append a record to the JSONL file, creating parent dirs if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(self._path, record.to_dict())

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
        oldest: float | None = None
        newest: float | None = None
        for rec in self.iter_records():
            count += 1
            tokens += rec.total_tokens
            if oldest is None or rec.timestamp < oldest:
                oldest = rec.timestamp
            if newest is None or rec.timestamp > newest:
                newest = rec.timestamp
        return TraceStats(count=count, tokens=tokens, oldest=oldest, newest=newest)

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
        total = sum(r.total_tokens for r in records)
        while total > self._max_tokens and records:
            removed = records.pop(0)
            total -= removed.total_tokens

        dropped = initial_count - len(records)
        if dropped == 0:
            return 0

        # Step 3: atomic rewrite in ascending-timestamp order
        lines = [json.dumps(r.to_dict()) for r in records]
        text = "\n".join(lines)
        if text:
            text += "\n"
        atomic_write_text(self._path, text)
        return dropped