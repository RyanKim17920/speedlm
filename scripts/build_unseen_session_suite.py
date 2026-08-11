#!/usr/bin/env python
"""Build a benchmark suite from agent sessions that contributed ZERO training rows.

Why this exists
---------------
The agentic self-play gate on run5 reported +0.65 accepted length and +32% tok/s
against a held-out suite that was split *row-level*.  Agentic traffic records are
nested: record N of a session carries record N-1's messages as a prefix, so a
row-level split routinely puts turn 7 of a session in the training set and turn
9 of the *same* session in the benchmark suite.  A draft head that memorised the
session still scores well.  The measured gain may therefore be recall rather
than generalisation.

This script builds the suite that can tell the two apart: contexts drawn only
from sessions **none of whose records** entered the training window, so no row
the candidate trained on shares a session with any row it is scored on.

Session identity
----------------
A session is keyed by the SHA-256 of its *opening exchange* -- the first three
messages (system, first user, first assistant), compared on role + content +
tool_calls.  This is not an arbitrary choice; it was measured against two
independent ground truths on this corpus:

* transitive prefix-containment over the records themselves, and
* the traffic driver's own ``traffic/trajectories/*.json`` files.

On run5 it yields 181 groups against 181 ground-truth components with **0
trajectories split and 0 trajectories merged**, and the same on run4.  The key
works because the first assistant turn is sampled at temperature 0.7 and is
effectively a fingerprint, while the prefix property guarantees every later
record of the session carries those three messages verbatim.

Keys that do NOT work, and why they are rejected here:

* the system prompt alone -- exactly 1 distinct value across the corpus;
* the first user message (with or without the system prompt) -- only 6 distinct
  values, one per *task family*, because the driver hands all 30 seeds of a
  family a byte-identical prompt.  Splitting on it holds out whole families,
  which measures domain shift, not held-out sessions;
* the first *four* messages -- 291 groups and 110 trajectories split, because
  the 3-message opening records have no fourth message and fall out of their
  own session.

Do not widen the key past message index 2.

What is emitted
---------------
``<out>/suite_contexts.jsonl`` + ``<out>/suite_manifest.json``
    Written by :func:`speedlm.gate.suite.persist_suite` from a
    :class:`~speedlm.gate.suite.BenchmarkSuite` built by the module's own
    :meth:`BenchmarkSuite.build`.  The suite hash is therefore rolled up by the
    same code the gate verifies it with, and a hand-edited context file would be
    rejected by ``load_suite``'s hash check.  Nothing here hand-authors JSON that
    the gate will later trust.

``<out>/training_context_hashes.json``
    The **real** context hashes of the 412 leased training records, reproduced
    by re-running the production split.  The gate's leakage proof is mandatory
    but is trivially satisfied by an empty set (``training_context_hashes=
    frozenset()`` is what the simulation harness passes); handing it the true
    training hashes is what makes ``_check_suite_leakage`` an assertion that
    could actually fire.

``<out>/selection.json``
    Per-context provenance: which session it came from, its turn depth, its task
    family, and the record id.  Plus the distributions and every verification
    number printed to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from speedlm.gate.suite import (  # noqa: E402
    BenchmarkSuite,
    FrozenContext,
    load_suite,
    persist_suite,
)
from speedlm.traces.store import TraceRecord  # noqa: E402

#: Messages of the opening exchange that key a session.  See the module
#: docstring: 2 under-groups (family id), 4 splits sessions.  3 is measured.
SESSION_KEY_MESSAGES = 3


class UnseenSuiteError(RuntimeError):
    """Raised when the requested unseen suite cannot be built honestly."""


# ---------------------------------------------------------------------------
# Session grouping
# ---------------------------------------------------------------------------

def session_key(record: TraceRecord, *, salt: str = "") -> str:
    """Identify the agent session *record* belongs to.

    Only role, content and tool_calls participate: ``provenance_tag`` is
    rewritten as a message moves from "the assistant just generated this" to
    "the client sent this back", so including it would split a session at its
    own second turn.

    ``salt`` distinguishes runs when several captures are pooled.  Measured on
    run4+run5 together, 6 single-record sessions collide across runs (the model
    emitted a long think block and no tool call, byte-identically); the salt
    removes even that.  It is empty by default so a single-run key is stable.
    """
    opening = list(record.messages[:SESSION_KEY_MESSAGES])
    if len(opening) < SESSION_KEY_MESSAGES:
        # A record shorter than the opening exchange cannot be keyed against
        # deeper records of the same session, and silently keying it on a
        # shorter prefix would merge it with every other short record.  This
        # corpus has none (every record is system + user + >=1 assistant), so
        # this is a guard, not a code path with a fallback.
        raise UnseenSuiteError(
            f"record {record.id!r} has only {len(opening)} message(s); "
            f"a session key needs {SESSION_KEY_MESSAGES}"
        )
    payload = [
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
        for message in opening
    ]
    canonical = json.dumps(
        {"salt": salt, "opening": payload} if salt else payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def group_sessions(
    records: Sequence[TraceRecord],
    *,
    salt: str = "",
) -> dict[str, list[int]]:
    """Map each session key to the indices of its records, in corpus order."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[session_key(record, salt=salt)].append(index)
    return dict(grouped)


def select_window(count: int, window: int | None) -> int:
    """Start offset of the training window, as ``_select_window`` computes it.

    Mirrors :meth:`speedlm.training.split.HeldOutTraceSnapshotLeaser._select_window`
    (src/speedlm/training/split.py:192): the newest ``window`` records, or the
    whole corpus when no window is configured.
    """
    if window is None:
        return 0
    return max(0, count - window)


def unseen_sessions(
    records: Sequence[TraceRecord],
    *,
    window_start: int,
    salt: str = "",
) -> set[str]:
    """Sessions that contributed no record at or after *window_start*.

    "Unseen" is a property of the whole session, not of a record: one record
    inside the window taints every other record of that session, because they
    share a message prefix the candidate trained on.
    """
    grouped = group_sessions(records, salt=salt)
    return {
        key
        for key, indices in grouped.items()
        if all(index < window_start for index in indices)
    }


# ---------------------------------------------------------------------------
# Reproducing the production split
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReproducedSplit:
    """The lease this cycle actually took, recovered from the run artifacts."""

    window_start: int
    window_records: tuple[TraceRecord, ...]
    training_records: tuple[TraceRecord, ...]
    training_hashes: frozenset[str]
    held_out_hashes: frozenset[str]


def reproduce_split(
    records: Sequence[TraceRecord],
    *,
    training_window_records: int | None,
    held_out_hashes: frozenset[str],
) -> ReproducedSplit:
    """Recover the leased training rows from the window and the persisted suite.

    The leaser's rule is one line
    (:meth:`HeldOutTraceSnapshotLeaser.lease_snapshot`, src/speedlm/training/
    split.py): the training set is every record of the window whose context hash
    is not reserved for the benchmark.  So the training set is fully determined
    by (i) the window bounds and (ii) the suite the cycle actually persisted --
    both of which are on disk.

    This deliberately does **not** re-run ``BenchmarkSuite.build`` to re-derive
    the reservation.  That function's selection policy is the thing under active
    change in this working tree (it now reserves whole sessions rather than a
    ranked prefix of rows), and re-deriving the split with today's policy would
    silently produce a *different* training set from the one the run5 candidate
    was actually fitted on -- which would make the leakage proof describe a
    split that never happened.  The persisted ``held-out/suite_manifest.json``
    is the run's own record of what it reserved; the caller cross-checks the
    resulting row count against the lease's own log line.

    Only the *hashing* comes from the gate module, via
    :meth:`BenchmarkSuite._record_hash`, so the hashes emitted here are directly
    comparable to the ones ``check_leakage`` will compare against.
    """
    start = select_window(len(records), training_window_records)
    window = tuple(records[start:])
    if len(window) < 2:
        raise UnseenSuiteError("training window holds fewer than two records")
    unreachable = held_out_hashes - {
        BenchmarkSuite._record_hash(record) for record in window
    }
    if unreachable:
        raise UnseenSuiteError(
            f"{len(unreachable)} held-out context(s) are not in the "
            f"records[{start}:] window; the window bounds assumed here are not "
            "the ones the cycle used"
        )
    training = tuple(
        record
        for record in window
        if BenchmarkSuite._record_hash(record) not in held_out_hashes
    )
    if not training:
        raise UnseenSuiteError("recovered split left no training records")
    return ReproducedSplit(
        window_start=start,
        window_records=window,
        training_records=training,
        training_hashes=frozenset(
            BenchmarkSuite._record_hash(record) for record in training
        ),
        held_out_hashes=held_out_hashes,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One replayable context drawn from an unseen session."""

    context_hash: str
    record: TraceRecord
    session: str
    turn_depth: int
    family: str


def _family(record: TraceRecord) -> str:
    """Task family, keyed on the first user message.

    Measured: exactly 6 distinct first-user contents across the corpus, one per
    family, because every seed of a family gets a byte-identical prompt.  That
    makes it useless as a session key and exactly right as a family label.
    """
    for message in record.messages:
        if message.get("role") == "user":
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return "no-user"


def build_candidates(
    records: Sequence[TraceRecord],
    *,
    sessions: Mapping[int, str],
    eligible_sessions: set[str],
    excluded_hashes: frozenset[str],
) -> tuple[list[Candidate], Counter[str]]:
    """Every distinct replayable context from *eligible_sessions*.

    Contexts are deduplicated by ``context_hash`` -- a suite is a set of
    distinct prompts, and ``BenchmarkSuite.build`` would otherwise emit the same
    prompt twice and double-count it in every arm's mean.

    Returns the candidates and a tally of why records were dropped, so the
    caller can print a pool that accounts for every eligible record instead of
    an unexplained count.
    """
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for index, record in enumerate(records):
        session = sessions[index]
        if session not in eligible_sessions:
            dropped["session_touched_training_window"] += 1
            continue
        messages, _ = FrozenContext._input_messages(record)
        context = FrozenContext.from_trace(record)
        if context.context_hash in excluded_hashes:
            # Overwhelmingly the bare openings: a session's first record has
            # context = system + user, which is family-constant, so it is
            # byte-identical to the opening of 30 other sessions and is very
            # likely already in the training set.  Being in an unseen session
            # does not make a context unseen.
            dropped["context_hash_in_training_or_original_suite"] += 1
            continue
        if context.context_hash in seen:
            dropped["duplicate_context_hash"] += 1
            continue
        seen.add(context.context_hash)
        candidates.append(
            Candidate(
                context_hash=context.context_hash,
                record=record,
                session=session,
                turn_depth=len(messages),
                family=_family(record),
            )
        )
    return candidates, dropped


def select_contexts(
    candidates: Sequence[Candidate],
    *,
    target: int,
) -> list[Candidate]:
    """Pick *target* contexts, maximising distinct sessions, then depth.

    Round-robin over sessions rather than "the deepest N contexts": the deepest
    contexts cluster in the handful of long sessions, and a suite drawn from
    five sessions measures five sessions.  Within a session the deepest context
    is taken first, because the original suite's contexts were mid-session and a
    replacement made of bare openings would not be comparable to it.

    Deterministic: sessions are ordered by (descending depth of their deepest
    context, session key) and contexts within a session by (descending depth,
    context hash), so the same corpus always yields the same suite.
    """
    if target < 1:
        raise UnseenSuiteError("target must be >= 1")
    by_session: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_session[candidate.session].append(candidate)
    for bucket in by_session.values():
        bucket.sort(key=lambda c: (-c.turn_depth, c.context_hash))
    order = sorted(
        by_session,
        key=lambda s: (-by_session[s][0].turn_depth, s),
    )
    selected: list[Candidate] = []
    round_index = 0
    while len(selected) < target:
        progressed = False
        for session in order:
            bucket = by_session[session]
            if round_index >= len(bucket):
                continue
            progressed = True
            selected.append(bucket[round_index])
            if len(selected) == target:
                break
        if not progressed:
            break
        round_index += 1
    if len(selected) < target:
        raise UnseenSuiteError(
            f"only {len(selected)} distinct unseen context(s) available, "
            f"need {target}"
        )
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_records(path: Path) -> tuple[TraceRecord, ...]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(TraceRecord.from_dict(json.loads(line)))
    return tuple(records)


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument(
        "--original-suite-dir",
        type=Path,
        required=True,
        help="the cycle's own held-out/ directory, used as a leakage control",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--training-window-records", type=int, default=512)
    parser.add_argument("--session-salt", default="")
    parser.add_argument(
        "--expect-training-records",
        type=int,
        default=None,
        help="fail unless the reproduced split leases exactly this many rows",
    )
    args = parser.parse_args(argv)

    records = read_records(args.traces)
    print(f"corpus records: {len(records)}")

    original = load_suite(args.original_suite_dir)
    original_hashes = frozenset(c.context_hash for c in original.contexts)
    print(
        f"original suite: {len(original.contexts)} context(s), "
        f"{len(original_hashes)} distinct hash(es), "
        f"hash {original.suite_hash}"
    )

    split = reproduce_split(
        records,
        training_window_records=args.training_window_records,
        held_out_hashes=original_hashes,
    )
    print(
        f"recovered lease: window[{split.window_start}:{len(records)}] "
        f"= {len(split.window_records)} record(s) -> "
        f"{len(split.training_records)} training row(s), "
        f"{len(split.window_records) - len(split.training_records)} "
        f"held-out record(s)"
    )
    if (
        args.expect_training_records is not None
        and len(split.training_records) != args.expect_training_records
    ):
        raise UnseenSuiteError(
            f"expected {args.expect_training_records} leased training "
            f"record(s), reproduced {len(split.training_records)}"
        )

    grouped = group_sessions(records, salt=args.session_salt)
    sessions = {
        index: key for key, indices in grouped.items() for index in indices
    }
    eligible = unseen_sessions(
        records, window_start=split.window_start, salt=args.session_salt
    )
    training_sessions = {
        sessions[index]
        for index in range(split.window_start, len(records))
    }
    print(
        f"sessions: {len(grouped)} total, {len(training_sessions)} touched the "
        f"training window, {len(eligible)} fully unseen"
    )

    candidates, dropped = build_candidates(
        records,
        sessions=sessions,
        eligible_sessions=eligible,
        excluded_hashes=split.training_hashes | original_hashes,
    )
    print(
        f"candidate pool: {len(candidates)} distinct unseen context(s) from "
        f"{len({c.session for c in candidates})} session(s); "
        f"dropped {dict(dropped)}"
    )

    selected = select_contexts(candidates, target=args.target)
    suite = BenchmarkSuite.build(
        tuple(c.record for c in selected),
        held_out_fraction=1.0,
    )
    if len(suite.contexts) != len(selected):
        raise UnseenSuiteError(
            f"suite holds {len(suite.contexts)} context(s) for "
            f"{len(selected)} selected record(s); selection is not distinct"
        )

    # --- verification, printed and stored ---------------------------------
    suite_hashes = {c.context_hash for c in suite.contexts}
    training_overlap = sorted(suite_hashes & split.training_hashes)
    original_overlap = sorted(suite_hashes & original_hashes)
    chosen_sessions = {c.session for c in selected}
    session_overlap = sorted(chosen_sessions & training_sessions)

    print("\n=== verification ===")
    print(f"(a) chosen contexts intersecting training hashes: {len(training_overlap)}")
    print(f"(b) chosen sessions that contributed training rows: {len(session_overlap)}")
    print(f"(c) chosen contexts present in the ORIGINAL suite: {len(original_overlap)}")
    depths = [c.turn_depth for c in selected]
    families = Counter(c.family for c in selected)
    per_session = Counter(c.session for c in selected)
    print(f"(d) contexts: {len(selected)} over {len(chosen_sessions)} session(s)")
    print(f"    turn depth (input messages): {_distribution(depths)}")
    print(f"    min/median/max depth: {min(depths)}/"
          f"{sorted(depths)[len(depths) // 2]}/{max(depths)}")
    print(f"    contexts per session: {_distribution(list(per_session.values()))}")
    print(f"    task families: {dict(families)}")
    original_depths = [len(c.messages) for c in original.contexts]
    print(f"    ORIGINAL suite depth for comparison: {_distribution(original_depths)}")

    if training_overlap or session_overlap or original_overlap:
        raise UnseenSuiteError(
            "refusing to emit a suite that overlaps the training set: "
            f"{len(training_overlap)} context hash(es), "
            f"{len(session_overlap)} session(s), "
            f"{len(original_overlap)} original-suite context(s)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path, manifest_path = persist_suite(suite, args.out)
    hashes_path = args.out / "training_context_hashes.json"
    hashes_path.write_text(
        json.dumps(
            {
                "traces": str(args.traces),
                "window_start": split.window_start,
                "training_window_records": args.training_window_records,
                "original_suite_dir": str(args.original_suite_dir),
                "original_suite_hash": original.suite_hash,
                "training_record_count": len(split.training_records),
                "training_context_hashes": sorted(split.training_hashes),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    selection: dict[str, Any] = {
        "suite_hash": suite.suite_hash,
        "target": args.target,
        "session_key_messages": SESSION_KEY_MESSAGES,
        "session_salt": args.session_salt,
        "corpus_records": len(records),
        "sessions_total": len(grouped),
        "sessions_touching_training_window": len(training_sessions),
        "sessions_fully_unseen": len(eligible),
        "candidate_pool": len(candidates),
        "candidate_pool_sessions": len({c.session for c in candidates}),
        "candidate_drops": dict(dropped),
        "verification": {
            "contexts_intersecting_training_hashes": training_overlap,
            "sessions_contributing_training_rows": session_overlap,
            "contexts_in_original_suite": original_overlap,
        },
        "distributions": {
            "turn_depth": _distribution(depths),
            "contexts_per_session": _distribution(list(per_session.values())),
            "task_family": dict(families),
            "original_suite_turn_depth": _distribution(original_depths),
        },
        "contexts": [
            {
                "context_hash": c.context_hash,
                "session": c.session,
                "turn_depth": c.turn_depth,
                "task_family": c.family,
                "record_id": c.record.id,
                "record_timestamp": c.record.timestamp,
            }
            for c in selected
        ],
    }
    (args.out / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    # Round-trip through the loader the gate itself uses: if the manifest and
    # the contexts ever disagree, this is where it surfaces, not on the GPU.
    reloaded = load_suite(args.out)
    if reloaded.suite_hash != suite.suite_hash:
        raise UnseenSuiteError("persisted suite did not round-trip")
    print(
        f"\nwrote {jsonl_path}\n"
        f"wrote {manifest_path}\n"
        f"wrote {hashes_path} ({len(split.training_hashes)} hash(es))\n"
        f"wrote {args.out / 'selection.json'}\n"
        f"suite hash: {suite.suite_hash} (reloaded OK)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
