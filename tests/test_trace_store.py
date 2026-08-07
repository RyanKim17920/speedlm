"""Tests for speedlm.traces.store — TraceRecord, TraceStore, TraceStats."""
from __future__ import annotations

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from speedlm.config import TraceBufferConfig
from speedlm.storage import StorageError, _exclusive_file_lock
from speedlm.traces.redact import RedactionReport
from speedlm.traces.store import (
    TRUNCATED_FINISH_REASONS,
    TraceError,
    TraceRecord,
    TraceStats,
    TraceStore,
)

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


def _record_drop_in_process(path: str, reason: str) -> None:
    TraceStore(Path(path)).record_drop(reason)


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

    def test_request_tool_schemas_round_trip(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        rec = replace(_rec(), tools=(tool,))

        payload = rec.to_dict()
        restored = TraceRecord.from_dict(payload)

        assert payload["tools"] == [tool]
        assert restored == rec

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
        # A legacy 0/0 pair is not a measurement, so prune accounting charges
        # the record its estimate rather than letting it sit in the buffer free.
        assert stats.estimated_tokens > 0

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

    def test_malformed_jsonl_reports_truncation_line(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(_rec())
        with store.path.open("a", encoding="utf-8") as stream:
            stream.write("NOT JSON\n")

        stats = store.stats()

        assert stats.count == 1
        assert stats.truncated_at_line == 2
        with pytest.raises(StorageError, match="line 2"):
            list(store.iter_records())

    def test_drop_counters_survive_across_processes(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "traces.jsonl")
        process = multiprocessing.get_context("spawn").Process(
            target=_record_drop_in_process,
            args=(str(store.path), "capture_error"),
        )

        process.start()
        process.join(timeout=10)

        assert process.exitcode == 0
        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["capture_error"] == 1
        assert store.stats_path.name == "traces.stats.json"

    def test_redaction_failure_is_counted(self, tmp_path: Path) -> None:
        class FailingRedactor:
            def redact(self, value: Any) -> tuple[Any, RedactionReport]:
                del value
                raise RuntimeError("redactor unavailable")

        store = TraceStore(tmp_path / "t.jsonl", redactor=FailingRedactor())

        assert store.append(_rec()) is None
        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["redaction_failure"] == 1


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
        # Seeded through a permissive store: append now enforces the token
        # ceiling itself, so a store configured at 180 would never let the
        # over-budget file that prune() is under test for exist.
        seed = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=10_000,
            max_age_days=365.0,
        )
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=180,
            max_age_days=365.0,
        )
        seed.append(_rec(ts=now - 30, prompt_tokens=100, completion_tokens=0))
        seed.append(_rec(ts=now - 20, prompt_tokens=100, completion_tokens=0))
        seed.append(_rec(ts=now - 10, prompt_tokens=100, completion_tokens=0))

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
        # Seeded through a permissive store; see TestPruneTokens.
        seed = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=10_000,
            max_age_days=365.0,
        )
        store = TraceStore(
            tmp_path / "t.jsonl",
            max_tokens=150,
            max_age_days=1.0,
        )
        seed.append(_rec(ts=now - 86400 * 3, prompt_tokens=200, completion_tokens=0))
        seed.append(_rec(ts=now - 3600, prompt_tokens=100, completion_tokens=0))
        seed.append(_rec(ts=now - 1800, prompt_tokens=100, completion_tokens=0))

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
        # Either the explicit prune or an append that crossed the token ceiling
        # may be the caller that evicts "expired"; which one wins the lock is a
        # race. The invariant under test is the end state asserted below.
        assert dropped in (0, 1)
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
        seed_store = TraceStore(
            path,
            max_tokens=10_000,
            max_age_days=365.0,
            redaction_enabled=False,
        )
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
            seed_store.append(
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
        stats = store.stats()
        assert stats.total_dropped == 1
        assert stats.drops_by_reason["lock_timeout"] == 1


# ── Corpus shape ───────────────────────────────────────────────────────────


def _shaped(
    rid: str,
    messages: tuple[dict[str, Any], ...],
    *,
    tools: tuple[dict[str, Any], ...] = (),
    tool_calls: tuple[dict[str, Any], ...] = (),
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    finish_reason: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        id=rid,
        timestamp=1.0,
        model="m1",
        messages=messages,
        tool_calls=tool_calls,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools=tools,
        finish_reason=finish_reason,
    )


_ANSWER = {"role": "assistant", "content": "hello", "provenance_tag": "generated"}


class TestCorpusShape:
    """The shape measurements added so a domain cannot stay implicit."""

    def test_single_turn_chat_corpus_reports_no_shape(self, tmp_path: Path) -> None:
        """The corpus every default was calibrated on must read as plain."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        for index in range(3):
            store.append(
                _shaped(f"r{index}", ({"role": "user", "content": "hi"}, _ANSWER))
            )
        stats = store.stats()
        assert stats.count == 3
        assert stats.multi_turn_records == 0
        assert stats.tool_schema_records == 0
        assert stats.tool_call_records == 0
        assert stats.system_prompt_records == 0
        assert stats.assistant_history_records == 0
        assert stats.client_supplied_assistant_records == 0
        assert stats.truncated_completion_records == 0
        assert stats.training_dropped_records == 0

    def test_multi_turn_counts_non_assistant_turns(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(
            _shaped("single", ({"role": "user", "content": "hi"}, _ANSWER))
        )
        store.append(
            _shaped(
                "multi",
                (
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                    _ANSWER,
                ),
            )
        )
        stats = store.stats()
        assert stats.multi_turn_records == 1
        assert stats.system_prompt_records == 1

    def test_tool_schemas_and_tool_calls_are_counted_apart(
        self, tmp_path: Path
    ) -> None:
        """Offering tools and using them are different facts about traffic."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(
            _shaped(
                "offered",
                ({"role": "user", "content": "hi"}, _ANSWER),
                tools=({"type": "function", "function": {"name": "search"}},),
            )
        )
        store.append(
            _shaped(
                "called",
                (
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "provenance_tag": "generated",
                        "tool_calls": [{"id": "c1", "function": {"name": "search"}}],
                    },
                ),
            )
        )
        stats = store.stats()
        assert stats.tool_schema_records == 1
        assert stats.tool_call_records == 1

    def test_assistant_history_excludes_the_captured_answer(
        self, tmp_path: Path
    ) -> None:
        """The trailing generated answer is the answer, not history."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(
            _shaped("answer-only", ({"role": "user", "content": "hi"}, _ANSWER))
        )
        store.append(
            _shaped(
                "history",
                (
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "earlier",
                        "provenance_tag": "generated",
                    },
                    {"role": "user", "content": "again"},
                    _ANSWER,
                ),
            )
        )
        stats = store.stats()
        assert stats.assistant_history_records == 1

    def test_untagged_assistant_turn_counts_as_client_supplied(
        self, tmp_path: Path
    ) -> None:
        """Absent provenance is not evidence of authorship; it is a drop."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(
            _shaped(
                "untagged",
                (
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "replayed"},
                    {"role": "user", "content": "again"},
                    _ANSWER,
                ),
            )
        )
        stats = store.stats()
        assert stats.client_supplied_assistant_records == 1
        assert stats.training_dropped_records == 1

    def test_training_dropped_is_the_union_of_the_three_filters(
        self, tmp_path: Path
    ) -> None:
        """Client-supplied, truncated and answerless rows, counted once each."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(_shaped("keep", ({"role": "user", "content": "hi"}, _ANSWER)))
        store.append(
            _shaped(
                "client",
                (
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "theirs"},
                    _ANSWER,
                ),
            )
        )
        store.append(
            _shaped(
                "truncated",
                ({"role": "user", "content": "hi"}, _ANSWER),
                finish_reason="length",
            )
        )
        store.append(_shaped("no-answer", ({"role": "user", "content": "hi"},)))
        store.append(
            _shaped(
                "both",
                (
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "theirs"},
                    _ANSWER,
                ),
                finish_reason="length",
            )
        )
        stats = store.stats()
        assert stats.count == 5
        assert stats.client_supplied_assistant_records == 2
        assert stats.truncated_completion_records == 2
        # "both" is dropped once, not twice.
        assert stats.training_dropped_records == 4

    def test_prompt_percentiles_use_the_prune_token_accounting(
        self, tmp_path: Path
    ) -> None:
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        for index, prompt in enumerate([10, 20, 30, 40, 1000]):
            store.append(
                _shaped(
                    f"r{index}",
                    ({"role": "user", "content": "hi"}, _ANSWER),
                    prompt_tokens=prompt,
                    completion_tokens=5,
                )
            )
        stats = store.stats()
        # Nearest rank over [10, 20, 30, 40, 1000].
        assert stats.prompt_tokens_p50 == 30
        assert stats.prompt_tokens_p95 == 1000
        assert stats.prompt_tokens_max == 1000
        assert stats.completion_tokens_p50 == 5
        assert stats.completion_tokens_max == 5
        assert stats.tokens == 10 + 20 + 30 + 40 + 1000 + 25

    def test_untrusted_zero_usage_falls_back_to_the_estimate(
        self, tmp_path: Path
    ) -> None:
        """A zeroed usage block must not report a zero-token prompt."""
        store = TraceStore(tmp_path / "t.jsonl", redaction_enabled=False)
        store.append(
            _shaped(
                "zeroed",
                ({"role": "user", "content": "x" * 400}, _ANSWER),
                prompt_tokens=0,
                completion_tokens=0,
            )
        )
        stats = store.stats()
        assert stats.prompt_tokens_p50 is not None
        assert stats.prompt_tokens_p50 > 50
        assert stats.prompt_tokens_p50 + stats.completion_tokens_p50 == stats.tokens

    def test_empty_store_reports_no_distribution(self, tmp_path: Path) -> None:
        stats = TraceStore(tmp_path / "t.jsonl").stats()
        assert stats.prompt_tokens_p50 is None
        assert stats.prompt_tokens_p95 is None
        assert stats.prompt_tokens_max is None
        assert stats.completion_tokens_p95 is None

    def test_truncated_finish_reasons_match_the_training_filter(self) -> None:
        """The drop population must be the one the renderer actually drops.

        ``store.TRUNCATED_FINISH_REASONS`` is a deliberate copy of the training
        backend's set, so that the store does not import a training module.  A
        copy that drifts would make ``training_dropped_records`` a number about
        nothing, so the copy is pinned here.
        """
        eagle3 = pytest.importorskip(
            "speedlm.training.backends.eagle3",
            reason="training backend not importable in this environment",
        )
        assert TRUNCATED_FINISH_REASONS == eagle3.TRUNCATED_FINISH_REASONS


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
