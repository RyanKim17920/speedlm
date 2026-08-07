"""Tests for gate/suite.py — no GPU, no network."""

from pathlib import Path

import pytest

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
    assert len(suite1.contexts) == 1


def test_suite_build_honors_held_out_fraction() -> None:
    """build includes only the requested deterministic held-out subset."""
    records = [_make_record(f"r{i}", f"content-{i}") for i in range(10)]

    suite = BenchmarkSuite.build(records, held_out_fraction=0.3)
    reversed_suite = BenchmarkSuite.build(
        list(reversed(records)),
        held_out_fraction=0.3,
    )

    assert len(suite.contexts) == 3
    assert suite.suite_hash == reversed_suite.suite_hash


def test_suite_build_honors_split_seed() -> None:
    """Changing the split seed changes the deterministic held-out subset."""
    records = [_make_record(f"r{i}", f"content-{i}") for i in range(10)]

    seed_one = BenchmarkSuite.build(records, held_out_fraction=0.3, seed=1)
    seed_two = BenchmarkSuite.build(records, held_out_fraction=0.3, seed=2)

    assert seed_one.suite_hash != seed_two.suite_hash


def test_suite_build_keeps_duplicate_contexts_on_one_side() -> None:
    """Content-identical records cannot straddle the deterministic split."""
    records = [
        _make_record("duplicate-a", "same"),
        _make_record("duplicate-b", "same"),
        _make_record("other", "other"),
    ]

    suite = BenchmarkSuite.build(records, held_out_fraction=0.5)
    selected_hashes = [context.context_hash for context in suite.contexts]

    duplicate_hash = FrozenContext.from_trace(records[0]).context_hash
    assert selected_hashes.count(duplicate_hash) in {0, 2}


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

    with pytest.raises(SuiteError, match=r"Suite manifest not found: .*suite_manifest\.json"):
        load_suite(run_dir)


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


def test_frozen_context_replays_input_not_captured_generated_output() -> None:
    record = TraceRecord(
        id="captured",
        timestamp=1000.0,
        model="test-model",
        messages=(
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "hi there",
                "provenance_tag": "generated",
            },
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=5,
    )

    context = FrozenContext.from_trace(record)

    assert context.messages == (
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    )
    assert context.expected_response == "hi there"


def test_build_with_split_no_leakage() -> None:
    """build_with_split ensures train/held-out disjointness."""
    train = [_make_record("t1", "train-content")]
    all_recs = [
        _make_record("t1", "train-content"),
        _make_record("h1", "held-out-1"),
        _make_record("h2", "held-out-2"),
    ]
    suite = BenchmarkSuite.build_with_split(
        train,
        all_recs,
        held_out_fraction=1.0,
    )
    # Should have only 2 contexts (the non-train ones)
    assert len(suite.contexts) == 2


def test_build_with_split_all_in_train_raises() -> None:
    """build_with_split raises when all records are in train."""
    records = [_make_record("r1", "content")]

    # The message must name the *split* failure. Without the matcher this test
    # also passed on "Cannot build suite from empty record list" and on the
    # held-out-fraction guard, neither of which is what it claims to check.
    with pytest.raises(
        SuiteError, match=r"No held-out records remain after removing train set"
    ):
        BenchmarkSuite.build_with_split(records, records)


def test_suite_empty_records_raises() -> None:
    """build raises when given empty records."""
    with pytest.raises(SuiteError, match=r"Cannot build suite from empty record list"):
        BenchmarkSuite.build([])


def test_suite_invalid_fraction_raises() -> None:
    """build raises when held_out_fraction is out of range."""
    records = [_make_record("r1", "hello")]
    for bad in [-0.1, 1.5]:
        # The rejected value has to appear in the message, so the guard that
        # fired is provably the fraction guard and not some later failure.
        with pytest.raises(
            SuiteError,
            match=rf"held_out_fraction must be in \[0, 1\], got {bad}",
        ):
            BenchmarkSuite.build(records, held_out_fraction=bad)


def test_suite_zero_fraction_raises_when_nothing_is_reserved() -> None:
    records = [_make_record("r1", "hello")]

    try:
        BenchmarkSuite.build(records, held_out_fraction=0.0)
    except SuiteError as exc:
        assert "No held-out records selected" in str(exc)
    else:
        raise AssertionError("zero held-out fraction must not produce a suite")


# ── Tool schemas ────────────────────────────────────────────────────────────
#
# The gate must be able to benchmark agentic traffic, not just chat. A captured
# request that offered tool schemas is only reproduced if those schemas survive
# freezing; otherwise the replayed prompt has no tool block and the gate scores
# a request production never served.

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_CLOCK_TOOL = {
    "type": "function",
    "function": {"name": "get_time", "parameters": {"type": "object"}},
}


def _make_tool_record(
    rid: str,
    content: str,
    tools: tuple[dict[str, object], ...],
) -> TraceRecord:
    return TraceRecord(
        id=rid,
        timestamp=1000.0,
        model="test-model",
        messages=({"role": "user", "content": content},),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=5,
        tools=tools,
    )


def test_frozen_context_carries_the_captured_tool_schemas() -> None:
    """The tool schemas offered in the request survive freezing."""
    record = _make_tool_record("agentic", "weather in Paris?", (_WEATHER_TOOL,))

    ctx = FrozenContext.from_trace(record)

    assert [dict(tool) for tool in ctx.tools] == [_WEATHER_TOOL]


def test_chat_traffic_freezes_with_no_tools() -> None:
    """A record with no tools freezes to an empty tool tuple."""
    ctx = FrozenContext.from_trace(_make_record("r1", "hello"))

    assert ctx.tools == ()


def test_tool_schemas_change_the_context_hash() -> None:
    """Same messages + different tools are different benchmark contexts."""
    same_messages = "weather in Paris?"
    no_tools = FrozenContext.from_trace(_make_tool_record("a", same_messages, ()))
    weather = FrozenContext.from_trace(
        _make_tool_record("b", same_messages, (_WEATHER_TOOL,))
    )
    clock = FrozenContext.from_trace(
        _make_tool_record("c", same_messages, (_CLOCK_TOOL,))
    )

    assert len({no_tools.context_hash, weather.context_hash, clock.context_hash}) == 3


def test_chat_context_hash_is_unchanged_by_tool_support() -> None:
    """Tool-free traffic keeps the pre-tools digest byte for byte.

    The digest is the SHA-256 of the canonical JSON of the input messages, and
    must stay exactly that when no tools are present -- persisted manifests and
    stored train-set hashes are compared against it.
    """
    import hashlib
    import json

    record = _make_record("r1", "hello")
    ctx = FrozenContext.from_trace(record)

    canonical = json.dumps(
        [{"role": "user", "content": "hello"}],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert ctx.context_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_suite_split_hash_agrees_with_the_frozen_context_hash() -> None:
    """The split's record hash and the frozen context's hash must match.

    ``build`` selects on ``_record_hash`` while ``check_leakage`` compares the
    frozen contexts' own hashes. If tools entered one and not the other, a
    tool-carrying context would be unreachable by any leakage check.
    """
    records = [
        _make_tool_record(f"r{i}", f"content-{i}", (_WEATHER_TOOL,))
        for i in range(4)
    ]

    suite = BenchmarkSuite.build(records, held_out_fraction=1.0)
    record_hashes = {BenchmarkSuite._record_hash(rec) for rec in records}

    assert {ctx.context_hash for ctx in suite.contexts} == record_hashes
    assert suite.check_leakage(record_hashes) == sorted(record_hashes)


def test_tool_schemas_round_trip_through_serialization() -> None:
    """to_dict/from_dict preserves the tool schemas."""
    ctx = FrozenContext.from_trace(
        _make_tool_record("agentic", "weather?", (_WEATHER_TOOL, _CLOCK_TOOL))
    )

    restored = FrozenContext.from_dict(ctx.to_dict())

    assert [dict(t) for t in restored.tools] == [_WEATHER_TOOL, _CLOCK_TOOL]
    assert restored.context_hash == ctx.context_hash


def test_chat_context_serialization_omits_the_tools_key() -> None:
    """A tool-free context serialises to exactly the pre-tools payload."""
    ctx = FrozenContext.from_trace(_make_record("r1", "hello"))

    assert "tools" not in ctx.to_dict()


def test_persisted_suite_reloads_with_its_tool_schemas(tmp_path: Path) -> None:
    """Tool schemas survive the persist/load round trip through disk."""
    records = [_make_tool_record("agentic", "weather?", (_WEATHER_TOOL,))]
    suite = BenchmarkSuite.build(records, held_out_fraction=1.0)

    persist_suite(suite, tmp_path / "run")
    loaded = load_suite(tmp_path / "run")

    assert [dict(t) for t in loaded.contexts[0].tools] == [_WEATHER_TOOL]
