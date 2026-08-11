"""Tests for the unseen-session suite builder (scripts/build_unseen_session_suite.py).

The experiment this supports stands or falls on two claims: that records are
grouped into the sessions they really came from, and that no context drawn from
a "fully unseen" session can share a session with a row the candidate trained
on.  Both are asserted here against inputs that make the wrong answer visible --
in particular a corpus where every session shares one system prompt and one
first user message, which is the shape of the real agentic capture and the shape
that makes the obvious session keys collapse to a task-family id.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from speedlm.gate.suite import BenchmarkSuite, FrozenContext
from speedlm.traces.store import TraceRecord

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_unseen_session_suite.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_unseen_session_suite", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


SYSTEM = {"role": "system", "content": "you are an agent", "provenance_tag": "client_supplied"}
#: Every session in the real capture opens on this exact pair: one system prompt
#: and one of six per-family user prompts, byte-identical across all 30 seeds.
USER = {"role": "user", "content": "trace the call chain", "provenance_tag": "client_supplied"}


def _assistant(text: str, *, generated: bool) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "provenance_tag": "generated" if generated else "client_supplied",
    }


def _record(record_id: str, messages: list[dict[str, Any]]) -> TraceRecord:
    return TraceRecord(
        id=record_id,
        timestamp=1.0,
        model="m",
        messages=tuple(dict(m) for m in messages),
        tool_calls=(),
        temperature=0.7,
        top_p=0.95,
        seed=0,
        prompt_tokens=1,
        completion_tokens=1,
    )


def _session(prefix: str, opening: str, turns: int) -> list[TraceRecord]:
    """A nested multi-turn session: record N's messages contain record N-1's.

    This is the shape that defeats a row-level split, so it is the shape the
    tests must use.
    """
    records = []
    messages = [SYSTEM, USER, _assistant(opening, generated=True)]
    records.append(_record(f"{prefix}-0", messages))
    for turn in range(1, turns):
        messages = [
            *[
                m if m["role"] != "assistant" else _assistant(m["content"], generated=False)
                for m in messages
            ],
            {"role": "user", "content": f"{prefix} follow-up {turn}"},
            _assistant(f"{prefix} reply {turn}", generated=True),
        ]
        records.append(_record(f"{prefix}-{turn}", messages))
    return records


# ---------------------------------------------------------------------------
# session_key
# ---------------------------------------------------------------------------

def test_session_key_groups_every_turn_of_one_session() -> None:
    records = _session("alpha", "alpha opening", turns=5)
    keys = {builder.session_key(record) for record in records}
    assert len(keys) == 1, "a nested session must not split across its own turns"


def test_session_key_survives_the_provenance_tag_flip() -> None:
    """Turn 2 carries turn 1's assistant message re-tagged as client-supplied.

    Including ``provenance_tag`` in the key would split every session at its own
    second turn -- and a split session leaks: half of it could be "unseen" while
    the other half was trained on.
    """
    records = _session("alpha", "alpha opening", turns=2)
    assert records[1].messages[2]["provenance_tag"] == "client_supplied"
    assert records[0].messages[2]["provenance_tag"] == "generated"
    assert builder.session_key(records[0]) == builder.session_key(records[1])


def test_session_key_separates_sessions_sharing_a_system_and_user_prompt() -> None:
    """The family trap.

    Every session in the capture opens on the same system prompt and the same
    per-family user prompt; only the first assistant turn differs.  A key built
    on the first one or two messages yields 6 groups for 181 sessions and would
    hold out whole task families instead of sessions.
    """
    left = _session("alpha", "alpha opening", turns=3)
    right = _session("beta", "beta opening", turns=3)
    assert left[0].messages[:2] == right[0].messages[:2]
    assert builder.session_key(left[0]) != builder.session_key(right[0])


def test_session_key_refuses_a_record_shorter_than_the_opening_exchange() -> None:
    record = _record("short", [SYSTEM, USER])
    with pytest.raises(builder.UnseenSuiteError):
        builder.session_key(record)


def test_group_sessions_returns_corpus_indices_per_session() -> None:
    records = [*_session("alpha", "a", turns=2), *_session("beta", "b", turns=3)]
    grouped = builder.group_sessions(records)
    assert sorted(len(v) for v in grouped.values()) == [2, 3]
    assert set().union(*grouped.values()) == set(range(5))


# ---------------------------------------------------------------------------
# window and unseen-session selection
# ---------------------------------------------------------------------------

def test_select_window_takes_the_newest_records() -> None:
    assert builder.select_window(1065, 512) == 553
    assert builder.select_window(100, 512) == 0
    assert builder.select_window(1065, None) == 0


def test_a_session_with_one_record_in_the_window_is_not_unseen() -> None:
    """One leased row taints the whole session.

    Its later turns contain that row verbatim as a prefix, so scoring on them is
    scoring on text the candidate was fitted on.
    """
    alpha = _session("alpha", "a", turns=3)
    beta = _session("beta", "b", turns=3)
    records = [*alpha[:2], *beta, alpha[2]]
    # window covers only the last record, which is alpha's third turn
    unseen = builder.unseen_sessions(records, window_start=len(records) - 1)
    assert builder.session_key(beta[0]) in unseen
    assert builder.session_key(alpha[0]) not in unseen


def test_sessions_entirely_before_the_window_are_unseen() -> None:
    alpha = _session("alpha", "a", turns=2)
    beta = _session("beta", "b", turns=2)
    records = [*alpha, *beta]
    unseen = builder.unseen_sessions(records, window_start=2)
    assert unseen == {builder.session_key(alpha[0])}


# ---------------------------------------------------------------------------
# split recovery
# ---------------------------------------------------------------------------

def test_reproduce_split_leaves_out_every_held_out_context() -> None:
    records = [*_session("alpha", "a", turns=4), *_session("beta", "b", turns=4)]
    window = records[2:]
    held_out = frozenset({BenchmarkSuite._record_hash(window[0])})
    split = builder.reproduce_split(
        records, training_window_records=len(window), held_out_hashes=held_out
    )
    assert split.window_start == 2
    assert len(split.training_records) == len(window) - 1
    assert not (split.training_hashes & held_out)


def test_reproduce_split_refuses_a_window_that_cannot_contain_the_suite() -> None:
    """The window bounds are an assumption; this is what makes it checkable.

    If a held-out context is not reachable inside the assumed window, the window
    is wrong, and every training hash derived from it would describe a split
    that never happened.
    """
    records = [*_session("alpha", "a", turns=4), *_session("beta", "b", turns=4)]
    stale = frozenset({BenchmarkSuite._record_hash(records[0])})
    with pytest.raises(builder.UnseenSuiteError):
        builder.reproduce_split(
            records, training_window_records=3, held_out_hashes=stale
        )


# ---------------------------------------------------------------------------
# candidate pool and selection
# ---------------------------------------------------------------------------

def _candidates_for(records, *, excluded=frozenset()):
    grouped = builder.group_sessions(records)
    sessions = {i: k for k, idx in grouped.items() for i in idx}
    return builder.build_candidates(
        records,
        sessions=sessions,
        eligible_sessions=set(grouped),
        excluded_hashes=excluded,
    )


def test_build_candidates_drops_contexts_whose_hash_was_trained_on() -> None:
    """Being in an unseen session does not make a context unseen.

    A session's opening record replays as system + user, which is identical
    across every session of the family and is therefore almost certainly in the
    training set already.
    """
    records = _session("alpha", "a", turns=3)
    opening_hash = FrozenContext.from_trace(records[0]).context_hash
    candidates, dropped = _candidates_for(records, excluded=frozenset({opening_hash}))
    assert opening_hash not in {c.context_hash for c in candidates}
    assert dropped["context_hash_in_training_or_original_suite"] == 1


def test_build_candidates_deduplicates_identical_contexts() -> None:
    """Two sessions whose openings differ but whose replayed context matches.

    ``BenchmarkSuite.build`` emits one context per *record*, so a duplicate
    would be replayed twice and counted twice in every arm's mean.
    """
    alpha = _session("alpha", "a", turns=1)
    beta = _session("beta", "b", turns=1)
    records = [*alpha, *beta]
    # Both replay as [system, user] once the generated assistant is stripped.
    assert (
        FrozenContext.from_trace(alpha[0]).context_hash
        == FrozenContext.from_trace(beta[0]).context_hash
    )
    candidates, dropped = _candidates_for(records)
    assert len(candidates) == 1
    assert dropped["duplicate_context_hash"] == 1


def test_selection_spreads_across_sessions_before_taking_a_second_turn() -> None:
    """Round-robin, not "the deepest N".

    The deepest contexts cluster in a handful of long sessions; a suite that
    took them greedily would measure three sessions and call it a hundred
    contexts.
    """
    records = [
        *_session("alpha", "a", turns=6),
        *_session("beta", "b", turns=6),
        *_session("gamma", "c", turns=2),
    ]
    candidates, _ = _candidates_for(records)
    selected = builder.select_contexts(candidates, target=3)
    assert len({c.session for c in selected}) == 3


def test_selection_prefers_deeper_turns_within_a_session() -> None:
    records = _session("alpha", "a", turns=5)
    candidates, _ = _candidates_for(records)
    selected = builder.select_contexts(candidates, target=1)
    assert selected[0].turn_depth == max(c.turn_depth for c in candidates)


def test_selection_refuses_to_pad_a_short_pool() -> None:
    """Fail rather than return 40 contexts when 100 were asked for.

    A silently short suite would be compared against a 100-context decision as
    though the two measured the same thing.
    """
    records = _session("alpha", "a", turns=3)
    candidates, _ = _candidates_for(records)
    with pytest.raises(builder.UnseenSuiteError):
        builder.select_contexts(candidates, target=len(candidates) + 1)


def test_selection_is_deterministic() -> None:
    records = [*_session("alpha", "a", turns=5), *_session("beta", "b", turns=5)]
    candidates, _ = _candidates_for(records)
    first = builder.select_contexts(candidates, target=4)
    second = builder.select_contexts(list(reversed(candidates)), target=4)
    assert [c.context_hash for c in first] == [c.context_hash for c in second]
