"""``trust_self_play_assistant_turns`` must earn its trust, and must be able to refuse.

The flag next to it, ``trust_untagged_assistant_messages``, is an unverified
promise: the operator asserts the corpus is the verifier's own output and nothing
checks it.  An earlier attempt to unblock agentic training that way was reverted
on safety grounds, so the replacement is only worth having if it genuinely fails
on traffic that is not self-play.

Every test here is therefore paired: one that the flag lets legitimate traffic
through, and one that it stops illegitimate traffic.  A file containing only the
first half would be this project's recurring defect -- a guard nobody has watched
go red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from speedlm.training.provenance import iter_jsonl, self_play_attestation

REAL_CAPTURE = Path(
    "/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run2/speedlm_home/"
    "traces/traces.jsonl"
)
REAL_SNAPSHOT = Path(
    "/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run2/speedlm_home/runs/"
    "35ef9ccd70a04415b75db216cf543a8f/trace-snapshot/traces.jsonl"
)


def _assistant(content: str, tag: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "provenance_tag": tag}


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "provenance_tag": "client_supplied"}


def _self_play_rows() -> list[dict[str, Any]]:
    """Two rows as the gateway would capture them from one two-turn session.

    Row 1 is the first request: user turn in, one generated assistant turn out.
    Row 2 is the second request, whose prefix therefore REPLAYS row 1's assistant
    turn -- tagged ``client_supplied`` because it arrived in the request, even
    though this same server wrote it seconds earlier.  That gap between how the
    turn is tagged and who actually authored it is the entire problem this flag
    solves.
    """
    first = _assistant("I will read the file.", "generated")
    return [
        {"id": "row-0", "messages": [_user("fix the bug"), first]},
        {
            "id": "row-1",
            "messages": [
                _user("fix the bug"),
                {**first, "provenance_tag": "client_supplied"},
                {"role": "tool", "content": "ok", "tool_call_id": "c1"},
                _assistant("Done.", "generated"),
            ],
        },
    ]


def _three_row_self_play_rows() -> list[dict[str, Any]]:
    """Three cumulative requests from one tool-using self-play session."""
    first = _assistant("I will inspect the repository.", "generated")
    second = _assistant("I found the failing module.", "generated")
    return [
        {"id": "row-0", "messages": [_user("fix the bug"), first]},
        {
            "id": "row-1",
            "messages": [
                _user("fix the bug"),
                {**first, "provenance_tag": "client_supplied"},
                {"role": "tool", "content": "files", "tool_call_id": "c1"},
                second,
            ],
        },
        {
            "id": "row-2",
            "messages": [
                _user("fix the bug"),
                {**first, "provenance_tag": "client_supplied"},
                {"role": "tool", "content": "files", "tool_call_id": "c1"},
                {**second, "provenance_tag": "client_supplied"},
                {"role": "tool", "content": "source", "tool_call_id": "c2"},
                _assistant("The fix is complete.", "generated"),
            ],
        },
    ]


def test_genuine_self_play_attests() -> None:
    result = self_play_attestation(_self_play_rows())

    assert result.attested
    assert result.prefix_assistant_turns == 1
    assert result.matched_prefix_turns == 1
    assert result.unmatched == ()


def test_held_out_middle_row_requires_full_capture_reference() -> None:
    """A middle held-out row must not orphan its generated assistant turn.

    Attesting only the survivors is the obvious implementation, but it makes a
    genuine cumulative trajectory impossible to admit when the split removes
    the row that first generated one of a later request's prefix turns.
    """
    full_corpus = _three_row_self_play_rows()
    training_rows = [full_corpus[0], full_corpus[2]]

    survivors_only = self_play_attestation(training_rows)
    against_full_capture = self_play_attestation(
        training_rows, reference_rows=full_corpus
    )

    assert not survivors_only.attested
    assert survivors_only.prefix_assistant_turns == 2
    assert survivors_only.matched_prefix_turns == 1
    assert len(survivors_only.unmatched) == 1
    assert survivors_only.unmatched[0].startswith("row-2[3]")
    assert "never generated" in survivors_only.unmatched[0]

    assert against_full_capture.attested
    assert against_full_capture.prefix_assistant_turns == 2
    assert against_full_capture.matched_prefix_turns == 2
    assert against_full_capture.unmatched == ()


def test_one_foreign_turn_refuses_and_names_the_row() -> None:
    """The paired negative. Change one character and the batch must be refused.

    A single foreign assistant turn is enough, because it is enough to teach the
    draft head another model's distribution for that span.
    """
    rows = _self_play_rows()
    rows[1]["messages"][1]["content"] = "I will read the file!"

    result = self_play_attestation(rows)

    assert not result.attested
    assert len(result.unmatched) == 1
    assert "row-1" in result.unmatched[0]


def test_a_row_cannot_attest_itself() -> None:
    """A prefix turn may only match a turn generated in an EARLIER row.

    Matching against the whole batch regardless of order is the obvious
    implementation and it is vacuous: a single row whose prefix repeats its own
    generated turn would attest, so replayed foreign traffic would pass simply by
    arriving one row at a time.
    """
    turn = _assistant("hello", "generated")
    rows = [
        {
            "id": "only",
            "messages": [
                _user("hi"),
                {**turn, "provenance_tag": "client_supplied"},
                turn,
            ],
        }
    ]

    result = self_play_attestation(rows)

    assert not result.attested


def test_a_row_cannot_attest_itself_with_its_own_reference() -> None:
    """An explicit reference must retain the strict earlier-than boundary.

    Treating the reference as an unordered evidence set, or changing ``<`` to
    ``<=``, lets this circular one-row corpus manufacture its own provenance.
    """
    turn = _assistant("This turn exists only in this row.", "generated")
    rows = [
        {
            "id": "self-reference",
            "messages": [
                {**turn, "provenance_tag": "client_supplied"},
                turn,
            ],
        }
    ]

    result = self_play_attestation(rows, reference_rows=rows)

    assert not result.attested
    assert result.prefix_assistant_turns == 1
    assert result.matched_prefix_turns == 0
    assert len(result.unmatched) == 1
    assert result.unmatched[0].startswith("self-reference[0]")
    assert "generated later" in result.unmatched[0]


def test_empty_corpus_does_not_attest() -> None:
    """No rows means no evidence, so there is nothing to attest.

    Returning true here would make every caller's guard structurally incapable of
    firing on a run whose capture layer silently wrote nothing -- the failure this
    whole mechanism exists to prevent, arriving through the back door.
    """
    result = self_play_attestation([])

    assert not result.attested
    assert "no rows were examined" in result.detail


def test_reordering_breaks_attestation() -> None:
    """Order is load-bearing, so a shuffled corpus must not attest.

    The trace store appends one line per completed response, so file order is
    request order. If a caller ever sorted or grouped the rows before attesting,
    this test is what tells them they broke the guarantee.
    """
    result = self_play_attestation(list(reversed(_self_play_rows())))

    assert not result.attested


def test_reference_still_requires_generation_in_an_earlier_row() -> None:
    """A reference is an ordered timeline, not an anywhere-match corpus.

    Indexing the complete capture without retaining positions would accept the
    first prefix below merely because an identical turn appears later, while a
    single generic diagnostic would hide that ordering bug as missing evidence.
    """
    target = {
        "id": "target",
        "messages": [
            _assistant("This exact turn is generated later.", "client_supplied"),
            _assistant("This turn is never generated.", "client_supplied"),
            _assistant("Current response.", "generated"),
        ],
    }
    reference = [
        target,
        {
            "id": "later",
            "messages": [
                _assistant("This exact turn is generated later.", "generated")
            ],
        },
    ]

    result = self_play_attestation([target], reference_rows=reference)

    assert not result.attested
    assert result.prefix_assistant_turns == 2
    assert result.matched_prefix_turns == 0
    assert len(result.unmatched) == 2
    assert result.unmatched[0].startswith("target[0]")
    assert "generated later" in result.unmatched[0]
    assert result.unmatched[1].startswith("target[1]")
    assert "never generated" in result.unmatched[1]


def test_foreign_reference_turns_do_not_launder_training_traffic() -> None:
    """Client-supplied turns in the reference are not authorship evidence.

    Indexing every assistant turn in the full capture, instead of only those
    tagged ``generated``, would let a foreign model's replay launder an identical
    later prefix into all-assistant training supervision.
    """
    foreign = _assistant("A foreign model wrote this.", "client_supplied")
    reference = [
        {
            "id": "foreign-origin",
            "model": "foreign/model",
            "messages": [foreign, _assistant("Local response one.", "generated")],
        },
        {
            "id": "training-row",
            "messages": [
                {**foreign},
                _assistant("Local response two.", "generated"),
            ],
        },
    ]

    result = self_play_attestation([reference[1]], reference_rows=reference)

    assert not result.attested
    assert result.prefix_assistant_turns == 1
    assert result.matched_prefix_turns == 0
    assert len(result.unmatched) == 1
    assert result.unmatched[0].startswith("training-row[0]")
    assert "never generated" in result.unmatched[0]


def test_reference_rows_none_preserves_legacy_behavior() -> None:
    """Omitted and explicit ``None`` references retain the original contract.

    Treating ``None`` as an empty evidence set would leave generated-only rows
    looking healthy while silently breaking the ordinary two-row self-play case.
    """
    cases = [
        (
            [{"id": "generated-only", "messages": [_assistant("Hi.", "generated")]}],
            True,
            0,
        ),
        (
            [
                {
                    "id": "single-foreign",
                    "messages": [_assistant("Not ours.", "client_supplied")],
                }
            ],
            False,
            0,
        ),
        (_self_play_rows(), True, 1),
    ]

    for rows, expected_attested, expected_matches in cases:
        omitted = self_play_attestation(rows)
        explicit_none = self_play_attestation(rows, reference_rows=None)

        assert omitted == explicit_none
        assert explicit_none.attested is expected_attested
        assert explicit_none.matched_prefix_turns == expected_matches


def test_non_assistant_turns_are_not_examined() -> None:
    """User and tool turns are client-supplied as a matter of fact, not provenance.

    They must never count toward the prefix-assistant tally, or a long tool-using
    trajectory would look like a corpus full of foreign turns.
    """
    result = self_play_attestation(_self_play_rows())

    # Two user turns and one tool turn exist across the two rows; none of them
    # may appear in the assistant accounting.
    assert result.prefix_assistant_turns == 1
    assert result.generated_turns == 2


def test_relabelling_only_touches_assistant_turns(tmp_path: Path) -> None:
    """The renderer's relabel step must leave user/tool/system roles alone.

    Rewriting a user turn's provenance would corrupt the loss mask the Speculators
    loader derives from role boundaries, and the corruption would be invisible
    until acceptance came back wrong.
    """
    from speedlm.training.backends.eagle3 import _relabel_self_play_turns

    record = {
        "id": "r",
        "messages": [
            _user("hi"),
            _assistant("prior", "client_supplied"),
            {"role": "tool", "content": "x", "provenance_tag": "client_supplied"},
            _assistant("now", "generated"),
        ],
    }

    updated = _relabel_self_play_turns(record)
    tags = [(m["role"], m["provenance_tag"]) for m in updated["messages"]]

    assert tags == [
        ("user", "client_supplied"),
        ("assistant", "generated"),
        ("tool", "client_supplied"),
        ("assistant", "generated"),
    ]
    # The input must not be mutated in place: the caller still holds it.
    assert record["messages"][1]["provenance_tag"] == "client_supplied"


def test_config_rejects_a_non_boolean_flag() -> None:
    from speedlm.config import ConfigError, IdleTuningConfig

    with pytest.raises(ConfigError, match="trust_self_play_assistant_turns"):
        IdleTuningConfig(trust_self_play_assistant_turns="yes")  # type: ignore[arg-type]


def test_flag_defaults_to_off() -> None:
    """Default-off is the whole safety story: existing runs must not change.

    If this ever flips, every corpus in the project silently starts relabelling
    its replayed turns as trainable.
    """
    from speedlm.config import IdleTuningConfig
    from speedlm.training.backends.eagle3 import SpeculatorsPipelineConfig

    assert IdleTuningConfig().trust_self_play_assistant_turns is False
    assert (
        SpeculatorsPipelineConfig.__dataclass_fields__[
            "trust_self_play_assistant_turns"
        ].default
        is False
    )


def test_snapshot_rows_preserve_capture_order(tmp_path: Path) -> None:
    """``_snapshot_rows`` must not reorder, or attestation loses its meaning."""
    from speedlm.training.backends.eagle3 import _snapshot_rows

    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": f"row-{i}", "messages": []}) for i in range(5)),
        encoding="utf-8",
    )

    class _Snapshot:
        def __init__(self, p: Path) -> None:
            self.path = p

    rows = _snapshot_rows(_Snapshot(path))  # type: ignore[arg-type]

    assert [row["id"] for row in rows] == [f"row-{i}" for i in range(5)]


def test_full_capture_rows_raises_when_capture_is_missing(tmp_path: Path) -> None:
    """A missing capture must not fall back to attesting the snapshot itself.

    The snapshot is deliberately valid, so the obvious fallback would return it
    successfully and make the exception assertion go red.
    """
    from speedlm.training.backends.eagle3 import _full_capture_rows
    from speedlm.tuner.eagle3 import Eagle3Error, TraceSnapshot

    home = tmp_path / "speedlm-home"
    snapshot_path = home / "runs" / "run-id" / "trace-snapshot" / "traces.jsonl"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text('{"id": "survivor", "messages": []}\n', encoding="utf-8")
    snapshot = TraceSnapshot(snapshot_path, "snapshot-hash")
    expected_capture = home / "traces" / "traces.jsonl"

    with pytest.raises(Eagle3Error, match="does not exist") as raised:
        _full_capture_rows(snapshot)

    assert str(expected_capture) in str(raised.value)
    assert "refuses rather than guessing" in str(raised.value)


def test_full_capture_rows_raises_for_shallow_snapshot_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed snapshot layout must not guess a capture or use the subset.

    An absolute ``tmp_path`` always has enough lexical parents, so this uses a
    real relative snapshot whose valid contents would make a fallback go green.
    """
    from speedlm.training.backends.eagle3 import _full_capture_rows
    from speedlm.tuner.eagle3 import Eagle3Error, TraceSnapshot

    monkeypatch.chdir(tmp_path)
    snapshot_path = Path("trace-snapshot/traces.jsonl")
    snapshot_path.parent.mkdir()
    snapshot_path.write_text('{"id": "survivor", "messages": []}\n', encoding="utf-8")
    snapshot = TraceSnapshot(snapshot_path, "snapshot-hash")

    with pytest.raises(Eagle3Error, match="cannot locate the full trace capture"):
        _full_capture_rows(snapshot)


def _leased_snapshot(tmp_path: Path, capture: list[dict[str, Any]], snapshot_ids: set[str]):
    """A snapshot laid out the way the leaser really leaves it on disk.

    ``_full_capture_rows`` navigates from the snapshot back up to
    ``<home>/traces/traces.jsonl``, so a test that writes a bare JSONL file
    exercises the attestation but not the wiring that finds its evidence.
    """
    from speedlm.tuner.eagle3 import TraceSnapshot

    home = tmp_path / "speedlm-home"
    capture_path = home / "traces" / "traces.jsonl"
    capture_path.parent.mkdir(parents=True)
    capture_path.write_text(
        "".join(json.dumps(row) + "\n" for row in capture), encoding="utf-8"
    )
    snapshot_path = home / "runs" / "run-id" / "trace-snapshot" / "traces.jsonl"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        "".join(
            json.dumps(row) + "\n" for row in capture if row["id"] in snapshot_ids
        ),
        encoding="utf-8",
    )
    return TraceSnapshot(snapshot_path, "snapshot-hash")


def _render(snapshot, destination: Path, *, trust: bool):
    import time

    from speedlm.training.backends.eagle3 import _render_speculators_dataset

    return _render_speculators_dataset(
        snapshot,
        destination,
        guard=lambda: False,
        started=time.monotonic(),
        timeout=60.0,
        minimum_rows=1,
        trust_self_play_assistant_turns=trust,
    )


def test_render_refuses_a_foreign_corpus_when_the_flag_is_set(tmp_path: Path) -> None:
    """The flag must abort the cycle, not merely compute a false attestation.

    The attestation itself is covered above; this pins the WIRING. Deleting the
    renderer's ``if not attestation.attested`` check leaves every other test in
    this file green while the flag launders another model's tokens into
    supervision -- which is precisely the failure it exists to prevent.
    """
    from speedlm.tuner.eagle3 import Eagle3Error

    capture = _self_play_rows()
    # One turn this server never generated, arriving in the request prefix.
    capture[1]["messages"][1] = _assistant("Another model wrote this.", "client_supplied")
    snapshot = _leased_snapshot(tmp_path, capture, {"row-0", "row-1"})

    with pytest.raises(Eagle3Error, match="not self-play traffic") as raised:
        _render(snapshot, tmp_path / "out.jsonl", trust=True)

    # The row is named, so the operator can tell a bug from a false premise.
    assert "row-1" in str(raised.value)


def test_render_admits_self_play_turns_the_flag_is_off_for(tmp_path: Path) -> None:
    """The paired half: genuine self-play must actually change what is rendered.

    Flag off, the replayed prefix turn makes the renderer drop the row -- the
    failure that killed three agentic runs. Flag on, the same corpus renders it.
    A test that only asserted the refusal above would leave the relabel step
    free to do nothing.
    """
    capture = _self_play_rows()
    snapshot = _leased_snapshot(tmp_path, capture, {"row-0", "row-1"})

    untrusted = _render(snapshot, tmp_path / "off.jsonl", trust=False)
    trusted = _render(snapshot, tmp_path / "on.jsonl", trust=True)

    assert untrusted.dropped_client_supplied == 1
    assert untrusted.written == 1
    assert trusted.dropped_client_supplied == 0
    assert trusted.written == 2


def test_render_attests_the_held_out_snapshot_against_the_full_capture(
    tmp_path: Path,
) -> None:
    """Held-out rows are removed from the MIDDLE, and the renderer must survive it.

    This is the job-376291 defect at the wiring level: attesting the leased
    subset against itself orphans row-2's prefix and refuses genuine self-play.
    Passing ``reference_rows`` is what makes this pass, so dropping that argument
    turns this red.
    """
    capture = _three_row_self_play_rows()
    # The leaser holds out row-1, orphaning the turn row-2 replays from it.
    snapshot = _leased_snapshot(tmp_path, capture, {"row-0", "row-2"})

    counts = _render(snapshot, tmp_path / "out.jsonl", trust=True)

    assert counts.written == 2
    assert counts.dropped_client_supplied == 0


@pytest.mark.skipif(
    not (REAL_CAPTURE.is_file() and REAL_SNAPSHOT.is_file()),
    reason=(
        "GPU job 376291 full capture and held-out snapshot are not mounted at "
        "the recorded /data paths"
    ),
)
def test_real_held_out_snapshot_matches_job_376291() -> None:
    """The hardware corpus pins both the historical refusal and repaired pass.

    A synthetic-only regression can miss capture-shape details; ignoring the full
    reference here restores the measured 378 orphaned turns and makes this red.
    """
    with REAL_CAPTURE.open(encoding="utf-8") as lines:
        full_rows = iter_jsonl(lines)
    with REAL_SNAPSHOT.open(encoding="utf-8") as lines:
        snapshot_rows = iter_jsonl(lines)

    full = self_play_attestation(full_rows)
    snapshot_alone = self_play_attestation(snapshot_rows)
    snapshot_with_reference = self_play_attestation(
        snapshot_rows, reference_rows=full_rows
    )

    assert full.attested
    assert full.rows_examined == 539
    assert full.prefix_assistant_turns == full.matched_prefix_turns == 2460
    assert len(full.unmatched) == 0

    assert not snapshot_alone.attested
    assert snapshot_alone.rows_examined == 434
    assert snapshot_alone.prefix_assistant_turns == 1974
    assert snapshot_alone.matched_prefix_turns == 1596
    assert len(snapshot_alone.unmatched) == 378

    assert snapshot_with_reference.attested
    assert snapshot_with_reference.rows_examined == 434
    assert (
        snapshot_with_reference.prefix_assistant_turns
        == snapshot_with_reference.matched_prefix_turns
        == 1974
    )
    assert len(snapshot_with_reference.unmatched) == 0
