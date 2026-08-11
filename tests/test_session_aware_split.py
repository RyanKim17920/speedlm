"""Tests for session-atomic held-out reservation.

Multi-turn agentic records are nested: turn N's ``messages`` list contains
turn N-1's messages verbatim.  A per-record split therefore hands the draft
head a training row whose prefix *is* the continuation a held-out gate
context scores it on predicting.  These tests pin the grouping that stops it,
and pin that single-turn corpora are untouched by it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from speedlm.gate.suite import (
    BenchmarkSuite,
    FrozenContext,
    load_suite,
    session_key,
)
from speedlm.traces.store import TraceRecord
from speedlm.training import split as split_module
from speedlm.training.split import HeldOutTraceSnapshotLeaser
from speedlm.tuner.eagle3 import Eagle3Error

# ── corpus builders ─────────────────────────────────────────────────────────


def _session_records(session: str, turns: int) -> list[TraceRecord]:
    """Build one agent session as ``turns`` nested records.

    Turn 1 is ``[system, user, assistant]``; every later turn re-sends the
    whole prior transcript and appends the next exchange, exactly as a client
    replaying an agent loop does.  Only the final assistant message of each
    record carries ``provenance_tag="generated"`` -- the re-sent history
    arrives untagged, which is why session identity must ignore that key.
    """
    records: list[TraceRecord] = []
    history: list[dict[str, Any]] = [
        {"role": "system", "content": f"you are agent {session}"},
    ]
    for turn in range(1, turns + 1):
        history.append({"role": "user", "content": f"{session} step {turn}"})
        answer = {"role": "assistant", "content": f"{session} reply {turn}"}
        messages = [dict(m) for m in history]
        messages.append({**answer, "provenance_tag": "generated"})
        records.append(
            TraceRecord(
                id=f"{session}-turn-{turn}",
                timestamp=float(turn),
                model="model",
                messages=tuple(messages),
                tool_calls=(),
                temperature=0.0,
                top_p=1.0,
                seed=0,
                prompt_tokens=10,
                completion_tokens=5,
            )
        )
        history.append(dict(answer))
    return records


def _nested_corpus(sessions: int = 4, turns: int = 3) -> tuple[TraceRecord, ...]:
    records: list[TraceRecord] = []
    for index in range(sessions):
        records.extend(_session_records(f"s{index}", turns))
    return tuple(records)


def _single_turn_record(identifier: str, prompt: str) -> TraceRecord:
    return TraceRecord(
        id=identifier,
        timestamp=1.0,
        model="model",
        messages=({"role": "user", "content": prompt},),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=2,
        completion_tokens=1,
    )


def _legacy_build(
    records: Any,
    *,
    held_out_fraction: float = 0.2,
    seed: int = 42,
) -> BenchmarkSuite:
    """The per-record reservation this change replaced.

    Kept in the tests, not the source, so the single-turn equivalence claim is
    checked against the real prior algorithm rather than against itself.
    """
    record_hashes = [BenchmarkSuite._record_hash(rec) for rec in records]
    unique_hashes = set(record_hashes)
    held_out_count = math.ceil(len(unique_hashes) * held_out_fraction)
    ranked = sorted(
        unique_hashes,
        key=lambda context_hash: (
            hashlib.sha256(f"{seed}:{context_hash}".encode("ascii")).digest(),
            context_hash,
        ),
    )
    held_out_hashes = set(ranked[:held_out_count])
    contexts = tuple(
        FrozenContext.from_trace(rec)
        for rec, context_hash in zip(records, record_hashes, strict=True)
        if context_hash in held_out_hashes
    )
    return BenchmarkSuite(
        suite_hash=BenchmarkSuite._compute_hash(contexts),
        contexts=contexts,
    )


@dataclass(frozen=True)
class _TraceSource:
    path: Path
    records: tuple[TraceRecord, ...]

    def iter_records(self) -> Iterator[TraceRecord]:
        yield from self.records


def _leaser(
    tmp_path: Path,
    records: tuple[TraceRecord, ...],
    *,
    held_out_fraction: float = 0.25,
) -> HeldOutTraceSnapshotLeaser:
    return HeldOutTraceSnapshotLeaser(
        _TraceSource(tmp_path / "traces.jsonl", records),  # type: ignore[arg-type]
        held_out_fraction=held_out_fraction,
        scratch_quota_bytes=4_000_000,
    )


def _lease(leaser: HeldOutTraceSnapshotLeaser, destination: Path) -> None:
    leaser.lease_snapshot(
        destination,
        timeout_seconds=30,
        should_abort=lambda: False,
    )


def _snapshot_records(destination: Path) -> tuple[TraceRecord, ...]:
    rows = (destination / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(TraceRecord.from_dict(json.loads(row)) for row in rows)


def _shape(messages: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(m.get("role")), str(m.get("content"))) for m in messages
    )


# ── grouping ────────────────────────────────────────────────────────────────


def test_every_turn_of_a_session_shares_one_session_key() -> None:
    """Nesting means later turns must hash to the opening exchange's key."""
    records = _session_records("alpha", 4)
    other = _session_records("beta", 2)

    keys = {session_key(record) for record in records}
    assert len(keys) == 1
    assert keys.isdisjoint({session_key(record) for record in other})


def test_records_below_three_messages_become_their_own_session() -> None:
    """No opening exchange to hash: fall back to the record's own context."""
    lonely = _single_turn_record("solo-a", "alpha")
    other = _single_turn_record("solo-b", "bravo")

    assert session_key(lonely) == BenchmarkSuite._record_hash(lonely)
    assert session_key(lonely) != session_key(other)


def test_sessions_are_never_split_across_the_boundary() -> None:
    """Every session lands wholly inside or wholly outside the suite."""
    records = _nested_corpus(sessions=4, turns=3)

    suite = BenchmarkSuite.build(records, held_out_fraction=0.25)

    held_out = {context.context_hash for context in suite.contexts}
    by_session: dict[str, set[str]] = {}
    for record in records:
        by_session.setdefault(session_key(record), set()).add(
            BenchmarkSuite._record_hash(record)
        )
    assert held_out  # the reservation is not vacuous
    reserved_sessions = 0
    for contexts in by_session.values():
        assert contexts <= held_out or contexts.isdisjoint(held_out)
        reserved_sessions += contexts <= held_out
    assert reserved_sessions >= 1
    assert reserved_sessions < len(by_session)


def test_nested_corpus_no_longer_leaks_continuations(tmp_path: Path) -> None:
    """The end-to-end leak: no training row may extend a gate context.

    Under the per-record split this corpus leaks -- some turn of a reserved
    session survives into the snapshot and its message list has the reserved
    context as a strict prefix.
    """
    records = _nested_corpus(sessions=4, turns=3)
    leaser = _leaser(tmp_path, records)

    _lease(leaser, tmp_path / "cycle" / "snapshot")

    training = _snapshot_records(tmp_path / "cycle" / "snapshot")
    suite = load_suite(leaser.suite_dir)
    assert training
    assert suite.contexts
    held_out_shapes = [_shape(ctx.messages) for ctx in suite.contexts]
    for record in training:
        trained = _shape(record.messages)
        for reserved in held_out_shapes:
            assert trained[: len(reserved)] != reserved, (
                f"training record {record.id!r} continues a held-out context"
            )
    # And the grouping itself held, not merely the prefix accident.
    held_sessions = {
        session_key(record)
        for record in records
        if BenchmarkSuite._record_hash(record)
        in {ctx.context_hash for ctx in suite.contexts}
    }
    assert {session_key(record) for record in training}.isdisjoint(held_sessions)


def test_the_old_row_level_split_would_have_leaked_this_corpus() -> None:
    """Guards the test above from being vacuous.

    If the nested corpus did not leak under the previous algorithm, then
    ``test_nested_corpus_no_longer_leaks_continuations`` proves nothing.
    """
    records = _nested_corpus(sessions=4, turns=3)

    legacy = _legacy_build(records, held_out_fraction=0.25)

    legacy_held = {ctx.context_hash for ctx in legacy.contexts}
    legacy_training = [
        record
        for record in records
        if BenchmarkSuite._record_hash(record) not in legacy_held
    ]
    leaked = [
        record
        for record in legacy_training
        for reserved in (_shape(ctx.messages) for ctx in legacy.contexts)
        if _shape(record.messages)[: len(reserved)] == reserved
    ]
    assert leaked, "corpus does not exercise the nesting leak"


# ── single-turn corpora are untouched ───────────────────────────────────────


def test_single_turn_corpus_selection_is_byte_identical_to_the_old_split() -> None:
    """UltraChat-shaped traffic must not notice this change at all."""
    records = [_single_turn_record(f"r{i}", f"content-{i}") for i in range(20)]

    for fraction in (0.1, 0.2, 0.35, 0.5):
        for seed in (1, 42, 7):
            new = BenchmarkSuite.build(
                records, held_out_fraction=fraction, seed=seed
            )
            old = _legacy_build(records, held_out_fraction=fraction, seed=seed)
            assert new.suite_hash == old.suite_hash
            assert new.contexts == old.contexts


def test_single_turn_lease_reserves_exactly_the_requested_fraction(
    tmp_path: Path,
) -> None:
    """Session atomicity must not silently inflate a chat-only suite."""
    records = tuple(_single_turn_record(f"r{i}", f"content-{i}") for i in range(10))
    leaser = _leaser(tmp_path, records, held_out_fraction=0.3)

    _lease(leaser, tmp_path / "cycle" / "snapshot")

    suite = load_suite(leaser.suite_dir)
    assert len(suite.contexts) == 3
    assert len(_snapshot_records(tmp_path / "cycle" / "snapshot")) == 7


# ── the guard ───────────────────────────────────────────────────────────────


def test_session_guard_fires_when_reservation_stops_being_session_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the per-record reservation; the leaser must refuse the lease."""
    records = _nested_corpus(sessions=4, turns=3)
    monkeypatch.setattr(
        split_module.BenchmarkSuite,
        "build",
        staticmethod(_legacy_build),
    )
    leaser = _leaser(tmp_path, records)

    with pytest.raises(Eagle3Error) as excinfo:
        _lease(leaser, tmp_path / "cycle" / "snapshot")

    message = str(excinfo.value)
    offending = sorted(
        {session_key(record) for record in records}
        & {
            session_key(record)
            for record in records
            if BenchmarkSuite._record_hash(record)
            in {
                ctx.context_hash
                for ctx in _legacy_build(records, held_out_fraction=0.25).contexts
            }
        }
    )
    assert any(key in message for key in offending)
    assert "straddles" in message
    assert "nest" in message
    assert not (tmp_path / "cycle" / "snapshot").exists()


def test_row_level_leakage_guard_is_still_enforced(tmp_path: Path) -> None:
    """The new session guard is additive; the row guard must remain."""
    records = _nested_corpus(sessions=4, turns=3)
    leaser = _leaser(tmp_path, records)

    _lease(leaser, tmp_path / "cycle" / "snapshot")

    suite = load_suite(leaser.suite_dir)
    training_hashes = {
        BenchmarkSuite._record_hash(record)
        for record in _snapshot_records(tmp_path / "cycle" / "snapshot")
    }
    assert training_hashes.isdisjoint({ctx.context_hash for ctx in suite.contexts})
    assert training_hashes == leaser.training_context_hashes


# ── determinism ─────────────────────────────────────────────────────────────


def test_session_reservation_is_deterministic_under_the_seed() -> None:
    """Same records, same seed, same suite -- in any record order."""
    records = _nested_corpus(sessions=5, turns=3)
    shuffled = tuple(records[7:] + records[:7])

    first = BenchmarkSuite.build(records, held_out_fraction=0.2, seed=42)
    again = BenchmarkSuite.build(records, held_out_fraction=0.2, seed=42)
    reordered = BenchmarkSuite.build(shuffled, held_out_fraction=0.2, seed=42)
    other_seed = BenchmarkSuite.build(records, held_out_fraction=0.2, seed=7)

    assert first.suite_hash == again.suite_hash
    assert first.suite_hash == reordered.suite_hash
    assert first.suite_hash != other_seed.suite_hash
    # A different seed must still reserve whole sessions.
    by_session: dict[str, set[str]] = {}
    for record in records:
        by_session.setdefault(session_key(record), set()).add(
            BenchmarkSuite._record_hash(record)
        )
    held_out = {ctx.context_hash for ctx in other_seed.contexts}
    for contexts in by_session.values():
        assert contexts <= held_out or contexts.isdisjoint(held_out)
