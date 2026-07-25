"""Tests for gate/suite.py — no GPU, no network."""

from pathlib import Path

from speedlm.gate.suite import (
    BenchmarkSuite,
    FrozenContext,
    SuiteError,
    build_suite,
    load_suite,
    persist_suite,
)
from speedlm.traces.store import TraceRecord


def _make_record(rid: str, content: str) -> TraceRecord:
    """Build a minimal TraceRecord for testing."""
    return TraceRecord(
        id=rid,
        timestamp=1000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": content},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_suite_build_deterministic_hash() -> None:
    """Same input records produce the same suite hash."""
    records = [
        _make_record("r1", "hello"),
        _make_record("r2", "world"),
    ]
    suite1 = BenchmarkSuite.build(records)
    suite2 = BenchmarkSuite.build(records)
    assert suite1.suite_hash == suite2.suite_hash
    assert len(suite1.contexts) == len(records)


def test_suite_context_hash_deterministic() -> None:
    """Same message content produces the same context hash."""
    rec = _make_record("r1", "same content")
    ctx1 = FrozenContext.from_trace(rec)
    ctx2 = FrozenContext.from_trace(rec)
    assert ctx1.context_hash == ctx2.context_hash


def test_suite_different_content_different_hash() -> None:
    """Different message content produces different hashes."""
    rec1 = _make_record("r1", "hello")
    rec2 = _make_record("r2", "goodbye")
    ctx1 = FrozenContext.from_trace(rec1)
    ctx2 = FrozenContext.from_trace(rec2)
    assert ctx1.context_hash != ctx2.context_hash


def test_check_leakage_no_overlap() -> None:
    """check_leakage returns empty when no overlap."""
    records = [_make_record("r1", "hello")]
    suite = BenchmarkSuite.build(records)
    overlaps = suite.check_leakage(set())
    assert len(overlaps) == 0


def test_check_leakage_detects_overlap() -> None:
    """check_leakage detects overlapping hashes."""
    records = [_make_record("r1", "hello")]
    suite = BenchmarkSuite.build(records)
    train_hashes = {suite.contexts[0].context_hash}
    overlaps = suite.check_leakage(train_hashes)
    assert len(overlaps) == 1


def test_suite_serialization_roundtrip() -> None:
    """suite.to_dict() -> from_dict() returns equivalent suite."""
    records = [
        _make_record("r1", "hello"),
        _make_record("r2", "world"),
    ]
    suite = BenchmarkSuite.build(records)
    d = suite.to_dict()
    suite2 = BenchmarkSuite.from_dict(d)
    assert suite2.suite_hash == suite.suite_hash
    assert len(suite2.contexts) == len(suite.contexts)


def test_suite_from_dict_hash_mismatch_raises() -> None:
    """from_dict raises if the hash doesn't match."""
    records = [_make_record("r1", "hello")]
    suite = BenchmarkSuite.build(records)
    d = suite.to_dict()
    d["suite_hash"] = "deadbeef"
    err = None
    try:
        BenchmarkSuite.from_dict(d)
    except SuiteError as exc:
        err = exc
    assert err is not None
    assert "hash mismatch" in str(err).lower()


def test_persist_and_load_suite(tmp_path: Path) -> None:
    """persist_suite + load_suite round-trips correctly."""
    records = [
        _make_record("r1", "hello"),
        _make_record("r2", "world"),
    ]
    suite = BenchmarkSuite.build(records)
    run_dir = tmp_path / "run"
    persist_suite(suite, run_dir)

    loaded = load_suite(run_dir)
    assert loaded.suite_hash == suite.suite_hash
    assert len(loaded.contexts) == len(suite.contexts)
    for orig, loaded_ctx in zip(suite.contexts, loaded.contexts, strict=True):
        assert orig.context_hash == loaded_ctx.context_hash


def test_load_suite_missing_manifest(tmp_path: Path) -> None:
    """load_suite raises when manifest is missing."""
    run_dir = tmp_path / "missing"
    run_dir.mkdir()
    err = None
    try:
        load_suite(run_dir)
    except SuiteError as exc:
        err = exc
    assert err is not None


def test_build_suite_convenience() -> None:
    """build_suite is an alias for BenchmarkSuite.build."""
    records = [_make_record("r1", "hello")]
    suite = build_suite(records)
    assert isinstance(suite, BenchmarkSuite)
    assert len(suite.contexts) == 1


def test_frozen_context_serialization() -> None:
    """FrozenContext.to_dict/from_dict round-trips."""
    rec = _make_record("r1", "hello")
    ctx = FrozenContext.from_trace(rec, expected_response="hi there")
    d = ctx.to_dict()
    ctx2 = FrozenContext.from_dict(d)
    assert ctx.context_hash == ctx2.context_hash
    assert ctx.expected_response == ctx2.expected_response


def test_build_with_split_no_leakage() -> None:
    """build_with_split ensures train/held-out disjointness."""
    train = [_make_record("t1", "train-content")]
    all_recs = [
        _make_record("t1", "train-content"),
        _make_record("h1", "held-out-1"),
        _make_record("h2", "held-out-2"),
    ]
    suite = BenchmarkSuite.build_with_split(train, all_recs)
    # Should have only 2 contexts (the non-train ones)
    assert len(suite.contexts) == 2


def test_build_with_split_all_in_train_raises() -> None:
    """build_with_split raises when all records are in train."""
    records = [_make_record("r1", "content")]
    err = None
    try:
        BenchmarkSuite.build_with_split(records, records)
    except SuiteError as exc:
        err = exc
    assert err is not None


def test_suite_empty_records_raises() -> None:
    """build raises when given empty records."""
    err = None
    try:
        BenchmarkSuite.build([])
    except SuiteError as exc:
        err = exc
    assert err is not None


def test_suite_invalid_fraction_raises() -> None:
    """build raises when held_out_fraction is out of range."""
    records = [_make_record("r1", "hello")]
    for bad in [-0.1, 1.5]:
        err = None
        try:
            BenchmarkSuite.build(records, held_out_fraction=bad)
        except SuiteError as exc:
            err = exc
        assert err is not None
