"""Tests for speedlm.traces.store — TraceRecord, TraceStore, TraceStats."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from speedlm.config import TraceBufferConfig
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
        rec = _rec()
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