"""Tests for speedlm.traces.store — TraceRecord, TraceStore, TraceStats."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from speedlm.config import TraceBufferConfig
from speedlm.storage import _exclusive_file_lock
from speedlm.traces.store import TraceError, TraceRecord, TraceStats, TraceStore

# ── helpers ─────────────────────────────────────────────────────────────────

def _rec(
    rid: str = "r1",
    ts: float | None = None,
    model: str = "m1",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    tool_calls: tuple[dict, ...] | None = None,
) -> TraceRecord:
    return TraceRecord(
        id=rid,
        timestamp=ts if ts is not None else time.time() - 60,
        model=model,
        messages=(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ),
        tool_calls=tool_calls if tool_calls is not None else (),
        temperature=0.7,
        top_p=0.9,
        seed=42,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ── TraceRecord to_dict / from_dict ────────────────────────────────────────


class TestRecordRoundTrip:
    def test_to_dict_from_dict_identity(self) -> None:
        rec = replace(_rec(), finish_reason="stop", stop_reason=128009)
        d = rec.to_dict()
        rec2 = TraceRecord.from_dict(d)
        assert rec == rec2

    def test_messages_become_lists_in_dict(self) -> None:
        rec = _rec()
        d = rec.to_dict()
        assert isinstance(d["messages"], list)
        assert isinstance(d["tool_calls"], list)

    def test_tuples_in_record(self) -> None:
        rec = _rec()
        d = rec.to_dict()
        rec2 = TraceRecord.from_dict(d)
        assert isinstance(rec2.messages, tuple)
        assert isinstance(rec2.tool_calls, tuple)

    def test_total_tokens(self) -> None:
        rec = _rec(prompt_tokens=200, completion_tokens=300)
        assert rec.total_tokens == 500

    def test_legacy_zero_pair_is_inferred_as_estimated(self, tmp_path: Path) -> None:
        payload = _rec(prompt_tokens=0, completion_tokens=0).to_dict()
        del payload["token_count_source"]
        del payload["finish_reason"]
        del payload["stop_reason"]

        rec = TraceRecord.from_dict(payload)
        assert rec.token_count_source == "estimated"
        assert rec.finish_reason is None
        assert rec.stop_reason is None

        store = TraceStore(tmp_path / "t.jsonl")
        store.append(rec)
        stats = store.stats()
        assert stats.measured_tokens == 0
        assert stats.estimated_tokens == 0

    def test_message_provenance_round_trip_and_legacy_default(self) -> None:
        tagged = _rec().to_dict()
        tagged["messages"][0]["provenance_tag"] = "client_supplied"
        tagged["messages"][1]["provenance_tag"] = "generated"

        record = TraceRecord.from_dict(tagged)

        assert record.to_dict()["messages"] == tagged["messages"]

        legacy = _rec().to_dict()
        assert all(
            message.get("provenance_tag") != "generated"
            for message in TraceRecord.from_dict(legacy).messages
        )


# ── TraceRecord validation rejections ──────────────────────────────────────


class TestRecordValidation:
    def test_missing_key(self) -> None:
        d = _rec().to_dict()
        del d["model"]
        with pytest.raises(TraceError, match="missing"):
            TraceRecord.from_dict(d)

    def test_unknown_key(self) -> None:
        d = _rec().to_dict()
        d["extra"] = True
        with pytest.raises(TraceError, match="unknown"):
            TraceRecord.from_dict(d)

    def test_bad_role(self) -> None:
        with pytest.raises(TraceError):
            TraceRecord(
                id="x", timestamp=1.0, model="m",
                messages=(("bad", "role"),),  # tuple, not mapping
                tool_calls=(),
                temperature=0.0, top_p=1.0,
                seed=0, prompt_tokens=0, completion_tokens=0,
            )

    def test_top_p_zero(self) -> None:
        with pytest.raises(TraceError):
            TraceRecord(
                id="x", timestamp=1.0, model="m",
                messages=({"role": "user", "content": "c"},),
                tool_calls=(),
                temperature=0.0, top_p=0.0,
                seed=0, prompt_tokens=0, completion_tokens=0,
            )

    def test_negative_temperature(self) -> None:
        with pytest.raises(TraceError):
            TraceRecord(
                id="x", timestamp=1.0, model="m",
                messages=({"role": "user", "content": "c"},),
                tool_calls=(),
                temperature=-1.0, top_p=1.0,
                seed=0, prompt_tokens=0, completion_tokens=0,
            )

    def test_bool_as_int_seed(self) -> None:
        with pytest.raises(TraceError):
            TraceRecord(
                id="x", timestamp=1.0, model="m",
                messages=({"role": "user", "content": "c"},),
                tool_calls=(),
                temperature=0.0, top_p=1.0,
                seed=True,  # type: ignore[arg-type]
                prompt_tokens=0, completion_tokens=0,
            )

    def test_bool_as_float_temperature(self) -> None:
        with pytest.raises(TraceError):
            TraceRecord(
                id="x", timestamp=1.0, model="m",
                messages=({"role": "user", "content": "c"},),
                tool_calls=(),
                temperature=True,  # type: ignore[arg-type]
                top_p=1.0,
                seed=0, prompt_tokens=0, completion_tokens=0,
            )

    def test_invalid_message_provenance_tag(self) -> None:
        with pytest.raises(TraceError, match="provenance_tag"):
            TraceRecord(
                id="x",
                timestamp=1.0,
                model="m",
                messages=(
                    {
                        "role": "assistant",
                        "content": "c",
                        "provenance_tag": "untrusted",
                    },
                ),
                tool_calls=(),
                temperature=0.0,
                top_p=1.0,
                seed=0,
                prompt_tokens=0,
                completion_tokens=0,
            )


# ── TraceStore append + iter ───────────────────────────────────────────────


class TestStoreAppendIter:
    def test_append_and_iter_round_trip(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "t.jsonl")
        rec = _rec()
        store.append(rec)
        assert list(store.iter_records()) == [rec]


# ── TraceStats ─────────────────────────────────────────────────────────────


class TestStoreStats:
    def test_empty_store_none(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "t.jsonl")
        s = store.stats()
        assert s == TraceStats(count=0, tokens=0, oldest=None, newest=None)

    def test_stats(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "t.jsonl")
        t0 = time.time() - 20
        t1 = time.time() - 10
        store.append(_rec(ts=t0, prompt_tokens=100, completion_tokens=0))
        store.append(_rec(ts=t1, prompt_tokens=200, completion_tokens=50))
        s = store.stats()
        assert s.count == 2
        assert s.tokens == 350
        assert s.oldest == t0
        assert s.newest == t1


# ── Prune by age ───────────────────────────────────────────────────────────


class TestPruneAge:
    def test_prune_by_age(self, tmp_path: Path) -> None:
        now = time.time()
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=10_000_000,
            max_age_days=1.0,
        )
        store.append(_rec(ts=now - 86400 * 3, prompt_tokens=10))  # 3 days ago
        store.append(_rec(ts=now - 3600, prompt_tokens=20))       # 1 hour ago

        dropped = store.prune(now=now)
        assert dropped == 1
        remaining = list(store.iter_records())
        assert len(remaining) == 1
        assert remaining[0].timestamp == now - 3600

    def test_boundary_kept(self, tmp_path: Path) -> None:
        now = time.time()
        boundary = now - 14.0 * 86400
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=10_000_000,
            max_age_days=14.0,
        )
        store.append(_rec(ts=boundary, prompt_tokens=10))
        dropped = store.prune(now=now)
        assert dropped == 0
        assert len(list(store.iter_records())) == 1


# ── Prune by tokens ────────────────────────────────────────────────────────


class TestPruneTokens:
    def test_prune_by_tokens_oldest_first(self, tmp_path: Path) -> None:
        now = time.time()
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=180,
            max_age_days=365.0,
        )
        store.append(_rec(ts=now - 30, prompt_tokens=100, completion_tokens=0))
        store.append(_rec(ts=now - 20, prompt_tokens=100, completion_tokens=0))
        store.append(_rec(ts=now - 10, prompt_tokens=100, completion_tokens=0))

        dropped = store.prune(now=now)
        # 300 > 180, drop oldest (100) -> 200 > 180, drop next (100) -> 100 <= 180
        assert dropped == 2
        remaining = list(store.iter_records())
        assert len(remaining) == 1
        assert remaining[0].timestamp == now - 10


# ── Prune by both ──────────────────────────────────────────────────────────


class TestPruneBoth:
    def test_age_then_tokens(self, tmp_path: Path) -> None:
        now = time.time()
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=150,
            max_age_days=1.0,
        )
        store.append(_rec(ts=now - 86400 * 3, prompt_tokens=200, completion_tokens=0))
        store.append(_rec(ts=now - 3600, prompt_tokens=100, completion_tokens=0))
        store.append(_rec(ts=now - 1800, prompt_tokens=100, completion_tokens=0))

        dropped = store.prune(now=now)
        # age drops 1 (3d ago), left: 100+100=200 > 150 -> drop oldest young -> 100 <= 150
        assert dropped == 2
        assert len(list(store.iter_records())) == 1


# ── Prune no-op ────────────────────────────────────────────────────────────


class TestPruneNoOp:
    def test_nothing_dropped_file_unchanged(self, tmp_path: Path) -> None:
        now = time.time()
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=10_000_000,
            max_age_days=365.0,
        )
        store.append(_rec(ts=now - 10, prompt_tokens=10))
        before = store._path.read_text()
        dropped = store.prune(now=now)
        after = store._path.read_text()
        assert dropped == 0
        assert before == after

    def test_missing_file_returns_0(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "missing.jsonl")
        assert store.prune() == 0


# ── Concurrent append / prune locking ──────────────────────────────────────


class TestStoreConcurrency:
    def test_prune_concurrent_with_appends_loses_no_new_record(
        self,
        tmp_path: Path,
    ) -> None:
        now = time.time()
        append_count = 12
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=append_count * 10,
            max_age_days=1.0,
            redaction_enabled=False,
        )
        store.append(_rec(rid="expired", ts=now - 2 * 86400, prompt_tokens=10,
                          completion_tokens=0))
        barrier = threading.Barrier(append_count + 1)

        def append_new(idx: int) -> object:
            barrier.wait()
            return store.append(
                _rec(
                    rid=f"new-{idx}",
                    ts=now + idx / 1_000,
                    prompt_tokens=10,
                    completion_tokens=0,
                )
            )

        def prune() -> int:
            barrier.wait()
            return store.prune(now=now)

        with ThreadPoolExecutor(max_workers=append_count + 1) as pool:
            append_futures = [pool.submit(append_new, idx) for idx in range(append_count)]
            prune_future = pool.submit(prune)
            append_results = [future.result() for future in append_futures]
            dropped = prune_future.result()

        records = list(store.iter_records())
        assert all(result is not None for result in append_results)
        assert dropped == 1
        assert {record.id for record in records} == {
            f"new-{idx}" for idx in range(append_count)
        }
        assert store.stats().tokens <= append_count * 10

    def test_two_concurrent_prunes_cannot_reintroduce_a_record(
        self,
        tmp_path: Path,
    ) -> None:
        now = time.time()
        path = tmp_path / "t.jsonl"
        loose_store = TraceStore(
            path,
            max_tokens=250,
            max_age_days=365.0,
            redaction_enabled=False,
        )
        strict_store = TraceStore(
            path,
            max_tokens=150,
            max_age_days=365.0,
            redaction_enabled=False,
        )
        for idx in range(3):
            loose_store.append(
                _rec(
                    rid=f"r{idx}",
                    ts=now - (30 - idx),
                    prompt_tokens=100,
                    completion_tokens=0,
                )
            )

        barrier = threading.Barrier(2)

        def prune(store: TraceStore) -> int:
            barrier.wait()
            return store.prune(now=now)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prune, store) for store in (loose_store, strict_store)]
            dropped = [future.result() for future in futures]

        assert sum(dropped) == 2
        assert [record.id for record in strict_store.iter_records()] == ["r2"]

    def test_append_drops_capture_when_lock_deadline_expires(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "t.jsonl"
        store = TraceStore(path, redaction_enabled=False)

        with _exclusive_file_lock(path) as acquired:
            assert acquired
            started = time.monotonic()
            result = store.append(_rec())
            elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 1.0
        assert not path.exists()


# ── Missing file behaviour ─────────────────────────────────────────────────


class TestMissingFile:
    def test_missing_iter_empty(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "missing.jsonl")
        assert list(store.iter_records()) == []

    def test_missing_stats_zeros(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "missing.jsonl")
        s = store.stats()
        assert s.count == 0
        assert s.tokens == 0
        assert s.oldest is None
        assert s.newest is None


# ── from_config ────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_from_config(self, tmp_path: Path) -> None:
        cfg = TraceBufferConfig(max_tokens=5_000_000, max_age_days=7.0)
        store = TraceStore.from_config(tmp_path / "t.jsonl", cfg)
        assert store.path == tmp_path / "t.jsonl"
