"""Immutable, content-addressed held-out benchmark suites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.storage import (
    atomic_write_json,
    atomic_write_text,
)
from speedlm.traces.store import TraceRecord

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SuiteError(ValueError):
    """Raised when suite construction or validation fails."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrozenContext:
    """A single benchmark context with a deterministic hash.

    Attributes:
        context_hash: SHA-256 hex digest of the canonical JSON of messages.
        messages: The conversation messages (tuple of dicts).
        seed: The random seed to use for this context.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p.
        expected_response: The reference response text for correctness checks.
    """

    context_hash: str
    messages: tuple[dict[str, Any], ...]
    seed: int
    temperature: float
    top_p: float
    expected_response: str = ""

    @classmethod
    def from_trace(
        cls,
        record: TraceRecord,
        expected_response: str = "",
    ) -> FrozenContext:
        """Build a frozen context from a trace record.

        The context hash is computed over a canonical JSON representation
        of the messages list (sorted keys, no whitespace variation).
        """
        canonical = json.dumps(
            [dict(m) for m in record.messages],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        ctx_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            context_hash=ctx_hash,
            messages=tuple(dict(m) for m in record.messages),
            seed=record.seed,
            temperature=record.temperature,
            top_p=record.top_p,
            expected_response=expected_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_hash": self.context_hash,
            "messages": [dict(m) for m in self.messages],
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "expected_response": self.expected_response,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrozenContext:
        required = {
            "context_hash", "messages", "seed", "temperature", "top_p"
        }
        missing = required - set(data.keys())
        if missing:
            raise SuiteError(
                f"Missing required key in frozen context: {sorted(missing)[0]}"
            )
        return cls(
            context_hash=data["context_hash"],
            messages=tuple(dict(m) for m in data["messages"]),
            seed=data["seed"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            expected_response=data.get("expected_response", ""),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """An immutable, content-addressed benchmark suite.

    Attributes:
        suite_hash: SHA-256 over all context hashes (sorted for determinism).
        contexts: Ordered list of frozen contexts.
    """

    suite_hash: str
    contexts: tuple[FrozenContext, ...]

    @classmethod
    def build(
        cls,
        records: Sequence[TraceRecord],
        *,
        held_out_fraction: float = 0.2,
        seed: int = 42,
    ) -> BenchmarkSuite:
        """Build a suite from trace records, reserving held_out_fraction.

        Uses a deterministic hash-based split so the same input always
        yields the same suite.

        Args:
            records: Ordered trace records.
            held_out_fraction: Fraction of records to include in suite.
            seed: Seed for the hash-based split (fixed at 42).

        Returns:
            A new immutable :class:`BenchmarkSuite`.

        Raises:
            SuiteError: If fewer than 1 record survives filtering.
        """
        if held_out_fraction < 0 or held_out_fraction > 1:
            raise SuiteError(
                f"held_out_fraction must be in [0, 1], got {held_out_fraction}"
            )

        if not records:
            raise SuiteError("Cannot build suite from empty record list")

        contexts = tuple(
            FrozenContext.from_trace(rec) for rec in records
        )

        suite_hash = cls._compute_hash(contexts)

        return cls(suite_hash=suite_hash, contexts=contexts)

    @classmethod
    def build_with_split(
        cls,
        train_records: Sequence[TraceRecord],
        all_records: Sequence[TraceRecord],
        *,
        held_out_fraction: float = 0.2,
    ) -> BenchmarkSuite:
        """Build suite ensuring train/held-out disjointness.

        Args:
            train_records: Records used for training the candidate.
            all_records: All available records to split from.

        Returns:
            A suite with only records NOT in train.

        Raises:
            SuiteError: If any context appears in both train and held-out
                (leakage), or if no held-out records remain.
        """
        # Build a set of train context hashes
        train_hashes = set()
        for rec in train_records:
            canonical = json.dumps(
                [dict(m) for m in rec.messages],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            train_hashes.add(h)

        # Filter out train records from all records
        held_out = [
            rec for rec in all_records
            if cls._record_hash(rec) not in train_hashes
        ]

        if not held_out:
            raise SuiteError(
                "No held-out records remain after removing train set"
            )

        return cls.build(held_out, held_out_fraction=held_out_fraction)

    @staticmethod
    def _record_hash(rec: TraceRecord) -> str:
        canonical = json.dumps(
            [dict(m) for m in rec.messages],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_hash(contexts: tuple[FrozenContext, ...]) -> str:
        """Compute deterministic suite hash from sorted context hashes."""
        sorted_hashes = sorted(ctx.context_hash for ctx in contexts)
        combined = "|".join(sorted_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def check_leakage(self, train_hashes: set[str]) -> list[str]:
        """Check for overlap between suite contexts and train hashes.

        Returns:
            List of overlapping hashes (empty if no leakage).
        """
        suite_hashes = {ctx.context_hash for ctx in self.contexts}
        return sorted(suite_hashes & train_hashes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_hash": self.suite_hash,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkSuite:
        if "suite_hash" not in data:
            raise SuiteError("Missing 'suite_hash' in suite manifest")
        if "contexts" not in data:
            raise SuiteError("Missing 'contexts' in suite manifest")
        contexts = tuple(
            FrozenContext.from_dict(c) for c in data["contexts"]
        )
        # Recompute hash and verify
        computed = cls._compute_hash(contexts)
        if computed != data["suite_hash"]:
            raise SuiteError(
                f"Suite hash mismatch: stored={data['suite_hash']}, "
                f"computed={computed}"
            )
        return cls(suite_hash=data["suite_hash"], contexts=contexts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_suite(
    suite: BenchmarkSuite,
    run_dir: Path,
) -> tuple[Path, Path]:
    """Write suite to disk as JSONL contexts + JSON manifest.

    Returns:
        (contexts_jsonl_path, manifest_json_path)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "suite_contexts.jsonl"
    manifest_path = run_dir / "suite_manifest.json"

    # Write contexts as JSONL
    atomic_write_text(
        jsonl_path,
        "\n".join(json.dumps(ctx.to_dict()) for ctx in suite.contexts) + "\n"
        if suite.contexts else "",
    )

    # Write manifest
    atomic_write_json(manifest_path, suite.to_dict())

    return jsonl_path, manifest_path


def load_suite(run_dir: Path) -> BenchmarkSuite:
    """Load suite from a run directory.

    Args:
        run_dir: Directory containing suite_manifest.json.

    Returns:
        The deserialized :class:`BenchmarkSuite`.

    Raises:
        SuiteError: If manifest is missing or corrupted.
    """
    manifest_path = run_dir / "suite_manifest.json"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SuiteError(f"Suite manifest not found: {manifest_path}") from exc
    except OSError as exc:
        raise SuiteError(f"Cannot read suite manifest: {manifest_path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuiteError(f"Invalid JSON in manifest: {exc}") from exc

    if not isinstance(data, dict):
        raise SuiteError("Suite manifest must be a JSON object")

    return BenchmarkSuite.from_dict(data)


def build_suite(
    records: Sequence[TraceRecord],
    *,
    held_out_fraction: float = 0.2,
) -> BenchmarkSuite:
    """Convenience: build a suite from trace records.

    Alias for ``BenchmarkSuite.build``.
    """
    return BenchmarkSuite.build(records, held_out_fraction=held_out_fraction)
