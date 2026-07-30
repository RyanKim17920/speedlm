"""Contract tests for leakage-safe training snapshot leases."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from speedlm.gate.suite import BenchmarkSuite, load_suite
from speedlm.traces.store import TraceRecord
from speedlm.training.split import HeldOutTraceSnapshotLeaser
from speedlm.tuner.eagle3 import Eagle3Error
from speedlm.tuner.idle import TuningPreempted


@dataclass(frozen=True)
class _TraceSource:
    path: Path
    records: tuple[TraceRecord, ...]

    def iter_records(self) -> Iterator[TraceRecord]:
        yield from self.records


def _record(identifier: str, prompt: str) -> TraceRecord:
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


def _leaser(
    tmp_path: Path,
    records: tuple[TraceRecord, ...],
    *,
    held_out_fraction: float = 0.4,
) -> HeldOutTraceSnapshotLeaser:
    source = _TraceSource(tmp_path / "traces.jsonl", records)
    return HeldOutTraceSnapshotLeaser(
        source,  # type: ignore[arg-type]
        held_out_fraction=held_out_fraction,
        scratch_quota_bytes=1_000_000,
    )


def _lease(
    leaser: HeldOutTraceSnapshotLeaser,
    destination: Path,
) -> None:
    leaser.lease_snapshot(
        destination,
        timeout_seconds=30,
        should_abort=lambda: False,
    )


def _snapshot_records(snapshot_dir: Path) -> tuple[TraceRecord, ...]:
    rows = (snapshot_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(TraceRecord.from_dict(json.loads(row)) for row in rows)


def test_split_is_deterministic_and_persists_the_same_suite(tmp_path: Path) -> None:
    records = tuple(_record(f"trace-{index}", f"prompt-{index}") for index in range(8))
    first = _leaser(tmp_path, records)
    second = _leaser(tmp_path, records)

    with pytest.raises(Eagle3Error, match="suite has not been frozen"):
        _ = first.suite_dir

    first_snapshot = first.lease_snapshot(
        tmp_path / "cycle-a" / "snapshot",
        timeout_seconds=30,
        should_abort=lambda: False,
    )
    second_snapshot = second.lease_snapshot(
        tmp_path / "cycle-b" / "snapshot",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    first_suite = load_suite(first.suite_dir)
    second_suite = load_suite(second.suite_dir)
    assert first_snapshot.content_hash == second_snapshot.content_hash
    assert first_snapshot.path.read_bytes() == second_snapshot.path.read_bytes()
    assert first_suite == second_suite
    assert first.suite_dir == tmp_path / "cycle-a" / "held-out"
    assert (first.suite_dir / "suite_contexts.jsonl").is_file()
    assert (first.suite_dir / "suite_manifest.json").is_file()


def test_duplicate_contexts_never_straddle_train_and_held_out(
    tmp_path: Path,
) -> None:
    records = (
        _record("duplicate-a", "same"),
        _record("unique-b", "bravo"),
        _record("duplicate-c", "same"),
        _record("unique-d", "delta"),
        _record("unique-e", "echo"),
    )
    leaser = _leaser(tmp_path, records)

    _lease(leaser, tmp_path / "cycle" / "snapshot")

    training = _snapshot_records(tmp_path / "cycle" / "snapshot")
    suite = load_suite(leaser.suite_dir)
    training_hashes = {
        BenchmarkSuite._record_hash(record)
        for record in training
    }
    held_out_hashes = {context.context_hash for context in suite.contexts}
    all_hashes = {BenchmarkSuite._record_hash(record) for record in records}
    duplicate_hash = BenchmarkSuite._record_hash(records[0])

    assert training_hashes == leaser.training_context_hashes
    assert training_hashes.isdisjoint(held_out_hashes)
    assert training_hashes | held_out_hashes == all_hashes
    assert duplicate_hash in training_hashes or duplicate_hash in held_out_hashes
    assert sum(
        BenchmarkSuite._record_hash(record) == duplicate_hash
        for record in training
    ) in {0, 2}
    assert sum(
        context.context_hash == duplicate_hash
        for context in suite.contexts
    ) in {0, 2}


def test_preemption_during_copy_removes_partial_snapshot_and_suite(
    tmp_path: Path,
) -> None:
    records = tuple(_record(f"trace-{index}", f"prompt-{index}") for index in range(5))
    leaser = _leaser(tmp_path, records, held_out_fraction=0.2)
    checkpoints = iter((False, False, True))

    with pytest.raises(TuningPreempted, match="preempted trace split"):
        leaser.lease_snapshot(
            tmp_path / "cycle" / "snapshot",
            timeout_seconds=30,
            should_abort=lambda: next(checkpoints),
        )

    assert not (tmp_path / "cycle" / "snapshot").exists()
    assert not (tmp_path / "cycle" / "held-out").exists()
    assert leaser.training_context_hashes == frozenset()


def test_timeout_during_copy_removes_partial_snapshot_and_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(_record(f"trace-{index}", f"prompt-{index}") for index in range(5))
    leaser = _leaser(tmp_path, records, held_out_fraction=0.2)
    monotonic_values = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(
        "speedlm.training.split.time.monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(TimeoutError, match="trace split timed out"):
        leaser.lease_snapshot(
            tmp_path / "cycle" / "snapshot",
            timeout_seconds=1,
            should_abort=lambda: False,
        )

    assert not (tmp_path / "cycle" / "snapshot").exists()
    assert not (tmp_path / "cycle" / "held-out").exists()
    assert leaser.training_context_hashes == frozenset()


def test_split_requires_two_unique_contexts(tmp_path: Path) -> None:
    records = (
        _record("duplicate-a", "same"),
        _record("duplicate-b", "same"),
    )
    leaser = _leaser(tmp_path, records)

    with pytest.raises(Eagle3Error, match="collect more unique contexts"):
        _lease(leaser, tmp_path / "cycle" / "snapshot")

    assert not (tmp_path / "cycle" / "snapshot").exists()
    assert not (tmp_path / "cycle" / "held-out").exists()
