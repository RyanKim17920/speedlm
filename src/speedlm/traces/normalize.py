from __future__ import annotations

import datetime
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.config import SamplingConfig
from speedlm.traces.store import TraceRecord

# ── Exceptions ──────────────────────────────────────────────────────────────

class NormalizeError(ValueError):
    """Raised when external input cannot be normalized (fail closed)."""


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Rejection:
    """A single rejected line from an external JSONL file."""

    line: int  # 1-based line number
    reason: str
    raw: str  # truncated to 500 chars


@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Outcome of normalizing an external JSONL file."""

    accepted: tuple[TraceRecord, ...]
    rejected: tuple[Rejection, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _generate_id(data: Mapping[str, Any]) -> str:
    """Deterministic ID: ``"tr-" + sha256(canonical_json)[:16]``."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"tr-{digest}"


def _parse_timestamp_to_epoch(value: Any, *, index: int) -> float:
    """Convert a timestamp value to epoch-seconds float.

    Accepts:
    - int or float (epoch seconds)
    - ISO-8601 string (e.g., "2026-07-25T04:00:00Z", "+00:00" offset, naive)

    Raises NormalizeError on malformed input.
    """
    if isinstance(value, bool):
        raise NormalizeError(
            f"record[{index}]: "
            f"'timestamp' must be a number or ISO-8601 string (got bool)"
        )
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be a finite number"
            )
        if value < 0:
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be >= 0"
            )
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise NormalizeError(
                f"record[{index}]: "
                f"'timestamp' is not a valid ISO-8601 string: {exc}"
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        epoch = dt.timestamp()
        if not math.isfinite(epoch):
            raise NormalizeError(
                f"record[{index}]: 'timestamp' converted to a non-finite epoch"
            )
        if epoch < 0:
            raise NormalizeError(
                f"record[{index}]: 'timestamp' must be >= 0"
            )
        return epoch
    raise NormalizeError(
        f"record[{index}]: 'timestamp' must be a number or "
        f"ISO-8601 string (got {type(value).__name__})"
    )


# ── Public API ──────────────────────────────────────────────────────────────

def normalize_record(
    data: Mapping[str, Any],
    *,
    defaults: SamplingConfig,
    default_model: str | None = None,
    index: int = 0,
) -> TraceRecord:
    """Validate and normalize a single external OpenAI-format record.

    Raises NormalizeError on any violation.

    Note: unknown top-level keys in external records are silently ignored
    because external OpenAI logs carry noise.
    """

    # --- messages (required, non-empty list of {role, content}) ---
    messages = data.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise NormalizeError(
            f"record[{index}]: 'messages' must be a non-empty list"
        )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise NormalizeError(
                f"record[{index}]: messages[{i}] must be an object"
            )
        if "role" not in msg or not isinstance(msg["role"], str) or not msg["role"]:
            raise NormalizeError(
                f"record[{index}]: "
                f"messages[{i}] must have a non-empty string 'role'"
            )
        if "content" not in msg:
            raise NormalizeError(
                f"record[{index}]: messages[{i}] must have a 'content' key"
            )
    messages_tuple: tuple[dict[str, Any], ...] = tuple(dict(m) for m in messages)

    # --- model ---
    model = data.get("model")
    if isinstance(model, str) and model:
        pass
    elif default_model is not None:
        model = default_model
    else:
        raise NormalizeError(
            f"record[{index}]: 'model' is required (no default_model provided)"
        )

    # --- id ---
    rid = data.get("id")
    if isinstance(rid, str) and rid:
        pass
    else:
        rid = _generate_id(data)

    # --- timestamp ---
    ts = data.get("timestamp")
    created = data.get("created")
    if ts is not None:
        ts = _parse_timestamp_to_epoch(ts, index=index)
    elif created is not None:
        ts = _parse_timestamp_to_epoch(created, index=index)
    else:
        ts = time.time()

    # --- sampling parameters ---
    temperature = data.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise NormalizeError(
                f"record[{index}]: 'temperature' must be a number"
            )
        if temperature < 0:
            raise NormalizeError(
                f"record[{index}]: 'temperature' must be >= 0"
            )
    else:
        temperature = defaults.temperature

    top_p = data.get("top_p")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise NormalizeError(
                f"record[{index}]: 'top_p' must be a number"
            )
        if not (0 < top_p <= 1):
            raise NormalizeError(
                f"record[{index}]: 'top_p' must satisfy 0 < top_p <= 1"
            )
    else:
        top_p = defaults.top_p

    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise NormalizeError(
                f"record[{index}]: 'seed' must be an int"
            )
        if seed < 0:
            raise NormalizeError(
                f"record[{index}]: 'seed' must be >= 0"
            )
    else:
        seed = defaults.seed

    # --- tool_calls ---
    tool_calls_raw = data.get("tool_calls")
    if tool_calls_raw is not None:
        if not isinstance(tool_calls_raw, list):
            raise NormalizeError(
                f"record[{index}]: 'tool_calls' must be a list"
            )
        for i, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                raise NormalizeError(
                    f"record[{index}]: tool_calls[{i}] must be an object"
                )
        tool_calls: tuple[dict[str, Any], ...] = tuple(
            dict(tc) for tc in tool_calls_raw
        )
    else:
        tool_calls = ()

    # --- token counts ---
    prompt_tokens, completion_tokens = _extract_tokens(data, index)

    return TraceRecord(
        id=rid,
        timestamp=float(ts),
        model=str(model),
        messages=messages_tuple,
        tool_calls=tool_calls,
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
    )


def _extract_tokens(data: Mapping[str, Any], index: int) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an external record."""
    usage = data.get("usage")
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if pt is not None:
            if isinstance(pt, bool) or not isinstance(pt, int):
                raise NormalizeError(
                    f"record[{index}]: usage.prompt_tokens must be an int"
                )
            if pt < 0:
                raise NormalizeError(
                    f"record[{index}]: usage.prompt_tokens must be >= 0"
                )
        else:
            pt = 0
        if ct is not None:
            if isinstance(ct, bool) or not isinstance(ct, int):
                raise NormalizeError(
                    f"record[{index}]: usage.completion_tokens must be an int"
                )
            if ct < 0:
                raise NormalizeError(
                    f"record[{index}]: usage.completion_tokens must be >= 0"
                )
        else:
            ct = 0
        return pt, ct

    # No usage dict: try top-level keys
    pt = data.get("prompt_tokens")
    ct = data.get("completion_tokens")
    if pt is not None:
        if isinstance(pt, bool) or not isinstance(pt, int):
            raise NormalizeError(
                f"record[{index}]: 'prompt_tokens' must be an int"
            )
        if pt < 0:
            raise NormalizeError(
                f"record[{index}]: 'prompt_tokens' must be >= 0"
            )
    else:
        pt = 0
    if ct is not None:
        if isinstance(ct, bool) or not isinstance(ct, int):
            raise NormalizeError(
                f"record[{index}]: 'completion_tokens' must be an int"
            )
        if ct < 0:
            raise NormalizeError(
                f"record[{index}]: 'completion_tokens' must be >= 0"
            )
    else:
        ct = 0
    return pt, ct


def normalize_file(
    path: Path,
    *,
    defaults: SamplingConfig | None = None,
    default_model: str | None = None,
) -> NormalizeResult:
    """Normalize an external OpenAI-format JSONL file.

    Reads the file line-by-line directly (not via ``storage.read_jsonl``)
    so that malformed JSON becomes a ``Rejection`` rather than an exception.
    A missing file raises ``NormalizeError``.

    Unknown top-level keys in external records are silently ignored because
    external OpenAI logs carry noise.
    """
    if defaults is None:
        defaults = SamplingConfig()

    if not path.exists():
        raise NormalizeError(f"file not found: {path}")

    accepted: list[TraceRecord] = []
    rejected: list[Rejection] = []

    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue  # skip blank lines
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason=f"malformed JSON: {e}",
                        raw=stripped[:500],
                    )
                )
                continue
            if not isinstance(obj, dict):
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason="line is not a JSON object",
                        raw=stripped[:500],
                    )
                )
                continue
            try:
                record = normalize_record(
                    obj,
                    defaults=defaults,
                    default_model=default_model,
                    index=line_no,
                )
                accepted.append(record)
            except NormalizeError as e:
                rejected.append(
                    Rejection(
                        line=line_no,
                        reason=str(e),
                        raw=stripped[:500],
                    )
                )

    return NormalizeResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )
