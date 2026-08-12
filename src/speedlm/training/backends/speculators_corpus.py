"""Turning a leased trace snapshot into the corpus Speculators will train on.

Split out of :mod:`speedlm.training.backends.eagle3`, which had grown to hold
two unrelated jobs: driving the Speculators subprocesses through a training
cycle, and deciding which captured rows are allowed to become training rows at
all.  Only the second one lives here.

It is a separable concern because it touches none of the pipeline's machinery.
There is no subprocess, no scratch quota, no vLLM server and no draft
checkpoint below -- just a JSONL file in, a JSONL file out, and the three
filters that stand between them: a row must convert to the loader's
``conversations`` contract, its assistant turns must be this verifier's own
output, and its completion must not have been cut off.  Most of the length is
the reasoning behind those filters and behind which failure gets reported when
they leave too little to train on, because attributing a shortfall to the wrong
filter prints a remedy that cannot work (job 375414).

:mod:`speedlm.training.backends.eagle3` re-exports every name below, so the
split moved no import site.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from speedlm.training.backends.stage_limits import check_abort, check_deadline
from speedlm.training.provenance import self_play_attestation
from speedlm.tuner.eagle3 import AbortCheck, Eagle3Error, TraceSnapshot

logger = logging.getLogger(__name__)

__all__ = [
    "CLIENT_SUPPLIED_DROP_ALERT_FRACTION",
    "TRUNCATED_FINISH_REASONS",
    "ClientSupervisedCorpusError",
    "EmptySpeculatorsDatasetError",
    "RenderedRowCounts",
    "TruncatedRowPolicy",
    "TruncationFilteredCorpusError",
    "UnattributedCorpusShortfallError",
]


class TruncatedRowPolicy(StrEnum):
    """What a rendered corpus does with a row whose completion was cut off."""

    #: Emit the row unchanged.  Restores the historical behaviour.
    KEEP = "keep"
    #: Omit the row from the rendered dataset.
    DROP = "drop"


#: ``finish_reason`` values that mean "this completion was cut off", i.e. the
#: assistant turn on disk is a mid-sentence fragment and not a finished turn.
#:
#: Why this matters for a drafter: production capture caps ``max_tokens`` at
#: 512 (``config.benchmark_max_tokens``) and ``gate/replay.py`` measured 76 of
#: 103 held-out contexts burning the whole budget inside the thinking block and
#: returning null content with ``finish_reason="length"``.  The tokenizer sees
#: such a row exactly as it sees a natural stop, and the chat template appends
#: its end-of-turn marker after the fragment -- teaching the draft that a
#: mid-sentence fragment is a complete turn, so at serving the draft proposes a
#: continuation precisely where the verifier emits stop.
#:
#: ``"length"`` is the OpenAI chat value for exhausting ``max_tokens``
#: (:data:`speedlm.gate.replay.FINISH_REASON_LENGTH`).  ``"incomplete"`` is the
#: Responses API's equivalent: that path files the response *status* into this
#: same field (``gateway/responses.py``), where a truncated response reads
#: ``"incomplete"`` and a finished one reads ``"completed"``.
#:
#: Deliberately a positive list and not "anything that is not ``stop``".  Three
#: real sources produce an *unknown* value -- ``TraceRecord.finish_reason``
#: defaults to ``None`` so corpora archived before the field existed read back
#: ``None``, streaming chunks carry ``null`` until the terminal chunk, and
#: ``gate/replay.py`` coerces a missing value to ``""`` -- and ``"tool_calls"``
#: is a perfectly complete stop.  An inverted test would discard all four.
TRUNCATED_FINISH_REASONS: Final = frozenset({"length", "incomplete"})

_SPECULATORS_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def _speculators_content(value: object) -> str:
    """Flatten a captured content value into the plain text the loader reads."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "".join(
            part["text"]
            for part in value
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _speculators_turn(turn: object) -> dict[str, Any] | None:
    """Convert one captured turn, or return None when the loader would drop it."""
    if not isinstance(turn, Mapping):
        return None
    raw_role = turn.get("from", turn.get("role"))
    role = _SPECULATORS_ROLES.get(raw_role) if isinstance(raw_role, str) else None
    if role is None:
        return None
    raw_content = turn.get("value")
    if raw_content is None:
        raw_content = turn.get("content")
    converted: dict[str, Any] = {"role": role, "content": _speculators_content(raw_content)}
    calls = turn.get("tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)) and calls:
        converted["tool_calls"] = list(calls)
    call_id = turn.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        converted["tool_call_id"] = call_id
    # The loader reads either key and re-emits both; carry whichever was captured.
    reasoning = turn.get("thinking") or turn.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        converted["thinking"] = reasoning
        converted["reasoning_content"] = reasoning
    # Carried, not merely consulted -- the same argument as ``finish_reason``
    # below.  ``_normalize_conversation`` in the Speculators loader rebuilds
    # each turn from a fixed key set (role/content/tool_calls/tool_call_id/
    # thinking/reasoning_content), so this key is inert there; it exists so the
    # rendered artifact records *why* a row was admitted rather than only this
    # process's logs.
    provenance = turn.get("provenance_tag")
    if isinstance(provenance, str) and provenance:
        converted["provenance_tag"] = provenance
    return converted


def _speculators_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert one captured trace, or return None when it trains nothing."""
    turns_value = record.get("conversations")
    if not isinstance(turns_value, (list, tuple)):
        turns_value = record.get("messages")
    if not isinstance(turns_value, (list, tuple)):
        return None
    turns = [
        converted
        for converted in (_speculators_turn(turn) for turn in turns_value)
        if converted is not None
    ]
    if not any(turn["role"] == "assistant" for turn in turns):
        return None
    converted_record: dict[str, Any] = {"conversations": turns}
    row_id = record.get("id")
    if isinstance(row_id, str) and row_id:
        converted_record["id"] = row_id
    tools = record.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        converted_record["tools"] = list(tools)
    # Carried, not merely consulted.  The filter below decides on it, but the
    # value also rides into the rendered dataset so the decision is auditable
    # from the artifact rather than only from this process's logs -- and so a
    # ``keep`` policy still leaves downstream tooling (``prepare_data.py``
    # ignores unknown keys; they become an unused dataset column) able to see
    # which rows were fragments.
    finish_reason = record.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        converted_record["finish_reason"] = finish_reason
    return converted_record


def _assistant_authorship(
    turn: Mapping[str, Any],
    *,
    trust_untagged: bool,
) -> bool:
    """Whether *this* deployment's verifier demonstrably produced *turn*.

    Only the exact tag ``"generated"`` establishes provider authorship; the
    gateway writes it on the response it assembled and ``"client_supplied"`` on
    everything that arrived in the request (``gateway/capture.py``).  An absent
    tag is *not* evidence of authorship -- it is a corpus that predates the tag
    -- so it fails closed unless an operator opts in.

    A prefill continuation is folded into one assistant turn carrying the
    generated tag, with ``prefill_prefix_chars`` recording how much of the
    merged content the *client* wrote (``traces/normalize.py``).  That prefix
    renders inside the assistant span the loss mask covers, so a non-zero (or
    unknown) prefix is not fully owned either.
    """
    tag = turn.get("provenance_tag")
    if tag is None:
        if not trust_untagged:
            return False
    elif tag != "generated":
        return False
    if "prefill_prefix_chars" not in turn:
        return True
    prefix = turn.get("prefill_prefix_chars")
    return isinstance(prefix, int) and not isinstance(prefix, bool) and prefix == 0


def _unowned_assistant_turns(
    record: Mapping[str, Any],
    *,
    trust_untagged: bool,
) -> int:
    """Count assistant turns in *record* that the verifier did not produce.

    Reads the *captured* messages rather than the converted ones so the answer
    does not depend on the conversion keeping any particular key.
    """
    turns = record.get("conversations")
    if not isinstance(turns, (list, tuple)):
        turns = record.get("messages")
    if not isinstance(turns, (list, tuple)):
        return 0
    unowned = 0
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        raw_role = turn.get("from", turn.get("role"))
        if not isinstance(raw_role, str):
            continue
        if _SPECULATORS_ROLES.get(raw_role) != "assistant":
            continue
        if not _assistant_authorship(turn, trust_untagged=trust_untagged):
            unowned += 1
    return unowned


def _is_truncated(record: Mapping[str, Any]) -> bool:
    """Whether *record*'s completion was cut off rather than finished."""
    value = record.get("finish_reason")
    return isinstance(value, str) and value.strip().lower() in TRUNCATED_FINISH_REASONS


#: Share of read rows the client-authorship filter may drop before the rendering
#: pass reports a diagnosis, even when enough rows survive to clear
#: ``min_rendered_rows``.
#:
#: The floors below only fire when this filter left too *few* rows outright.  A
#: large multi-turn or agentic corpus can lose most of its rows here and still
#: clear the floor, and what survives is exactly the single-turn slice the
#: gateway tagged end to end -- a biased remnant that trains, promotes and
#: reports like any other corpus.  One in five is well above the noise floor for
#: a corpus that is genuinely mostly this verifier's own single-turn traffic and
#: well below the "every multi-turn row is gone" case this exists to name.
CLIENT_SUPPLIED_DROP_ALERT_FRACTION: Final = 0.2


#: Every bucket a read row can end up in other than ``written``, with the
#: clause each one contributes to a failure message.
#:
#: Keyed by field name and consulted by :meth:`RenderedRowCounts.drop_accounting`
#: so that the two corpus-filter failures cannot report a partial account.  Job
#: 375414 is what a partial account costs: 409 rows read, 0 written, 36 dropped
#: by the truncation filter and 373 by the authorship filter, and the message
#: named only the 36 -- so the remedy it printed (raise the token cap) would
#: have changed nothing.  A drop bucket added to :class:`RenderedRowCounts`
#: without an entry here leaves ``read`` un-reconciled, which
#: ``test_every_drop_bucket_is_named_in_the_failure`` fails on.
_DROP_BUCKETS: Final = {
    "dropped_untrainable": "carried no trainable assistant turn",
    "dropped_truncated": "were truncated",
    "dropped_client_supplied": (
        "carried an assistant turn this verifier did not produce"
    ),
}


@dataclass(frozen=True, slots=True)
class RenderedRowCounts:
    """What one rendering pass read, kept and discarded."""

    read: int
    written: int
    dropped_untrainable: int
    dropped_truncated: int
    truncated_seen: int
    policy: TruncatedRowPolicy
    #: Rows dropped because an assistant turn was not this verifier's output.
    dropped_client_supplied: int = 0
    #: Assistant turns that were not this verifier's output, across all rows.
    client_supplied_turns_seen: int = 0

    @property
    def client_supplied_drop_fraction(self) -> float:
        """Share of read rows the client-authorship filter removed."""
        if not self.read:
            return 0.0
        return self.dropped_client_supplied / self.read

    def dominant_drop_bucket(self) -> str | None:
        """The one bucket that took strictly more rows than any other, if any.

        Considers *every* bucket in :data:`_DROP_BUCKETS`, which is the whole
        point.  The predecessor compared ``dropped_client_supplied`` against
        ``dropped_truncated`` and nothing else, so ``dropped_untrainable`` --
        a first-class reconciled bucket that ``drop_accounting`` has always
        printed -- could not win no matter how large it got.  On 90 untrainable,
        6 truncated, 4 client-supplied and 0 written, the caller raised
        :class:`TruncationFilteredCorpusError` and told the operator to raise the
        token cap: a remedy that would return 6 of the 100 missing rows, aimed at
        the smallest of the three causes.

        ``None`` when no bucket is strictly greatest.  A tie is not a dominant
        cause, and the predecessor resolved one by always answering
        "truncation", which is a coin flip wearing a diagnosis.  The caller
        reports the full accounting instead of naming a winner that does not
        exist -- see :class:`UnattributedCorpusShortfallError`.
        """
        counted = {name: int(getattr(self, name)) for name in _DROP_BUCKETS}
        largest = max(counted.values())
        if largest <= 0:
            return None
        leaders = [name for name, value in counted.items() if value == largest]
        if len(leaders) != 1:
            return None
        return leaders[0]

    def drop_accounting(self) -> str:
        """Account for every read row, whichever filter is being blamed.

        A failure message that names one filter's count invites the reader to
        treat that count as the whole loss, and the two filters run in the same
        pass over the same corpus.  Reconciling ``read`` against ``written``
        plus every bucket is what makes the dominant cause visible instead of
        inferred.
        """
        clauses = ", ".join(
            f"{getattr(self, name)} {clause}" for name, clause in _DROP_BUCKETS.items()
        )
        return (
            f"Of {self.read} rows read, {self.written} were written and the "
            f"rest were dropped: {clauses}"
        )

    def client_supplied_diagnosis(self) -> str:
        """Name the authorship filter, its counts and the knob that relaxes it.

        Shared by the hard failure and the survived-but-biased warning so an
        operator reads the same diagnosis either way.  Naming the counts is the
        point: "not enough rows" sends someone hunting for a capture problem,
        while "9 of 10 rows carried an assistant turn this verifier did not
        produce" points straight at the corpus and at the one knob that changes
        the outcome.
        """
        return (
            f"{self.dropped_client_supplied} of {self.read} rows "
            f"({self.client_supplied_drop_fraction:.1%}) carried an assistant "
            f"turn this verifier did not produce "
            f"({self.client_supplied_turns_seen} such turns in total) and were "
            f"dropped, leaving {self.written}. Multi-turn and agentic traffic "
            "replays earlier assistant turns in every request, so a corpus "
            "whose messages carry no provenance_tag loses every multi-turn row "
            "here. Set trust_self_play_assistant_turns=True for self-play "
            "traffic -- it attests that every such turn reproduces one this "
            "server generated earlier, and fails the cycle if it does not. "
            "Set trust_untagged_assistant_messages=True only if every "
            "assistant turn in the corpus is known to be this verifier's own "
            "output."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "read": self.read,
            "written": self.written,
            "dropped_untrainable": self.dropped_untrainable,
            "dropped_truncated": self.dropped_truncated,
            "truncated_seen": self.truncated_seen,
            "truncated_row_policy": self.policy.value,
            "dropped_client_supplied": self.dropped_client_supplied,
            "client_supplied_turns_seen": self.client_supplied_turns_seen,
            "client_supplied_drop_fraction": self.client_supplied_drop_fraction,
            # Carried into ``state.rendered_rows`` and from there into the cycle
            # report, so a corpus this filter reshaped is diagnosable from the
            # artifact and not only from this process's logs.
            "client_supplied_drop_diagnosis": (
                self.client_supplied_diagnosis()
                if self.dropped_client_supplied
                else None
            ),
        }


class TruncationFilteredCorpusError(Eagle3Error):
    """The truncation filter left too few rows to train on.

    Raised only when truncation is the *dominant* cause of the shortfall.  It
    used to be raised whenever the truncation filter had dropped anything at
    all, because it was checked first: on job 375414 that reported 36 truncated
    rows as the reason a 409-row corpus rendered nothing, when 373 rows had
    gone to the authorship filter, and the remedy it printed (raise the token
    cap) would have left the corpus exactly as empty.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        remedy = (
            "Training on the remainder would be a silently tiny corpus. Either "
            "raise the serving max_tokens cap so completions finish, lower "
            "min_rendered_rows, or set truncated_row_policy to 'keep' to accept "
            "training on truncated fragments."
        )
        if counts.dropped_client_supplied:
            # Truncation dominates here, but it is not the whole loss, and the
            # remedy above does nothing for the rows the other filter took.
            remedy += (
                f" The authorship filter also dropped "
                f"{counts.dropped_client_supplied} rows: "
                f"{counts.client_supplied_diagnosis()}"
            )
        super().__init__(
            f"dropping truncated rows left {counts.written} trainable rows from "
            f"{source}, below the floor of {minimum}: "
            f"{counts.dropped_truncated} of {counts.read} rows were truncated "
            f"(finish_reason in {sorted(TRUNCATED_FINISH_REASONS)}). "
            f"{counts.drop_accounting()}. {remedy}"
        )


class ClientSupervisedCorpusError(Eagle3Error):
    """The client-authorship filter left too few rows to train on.

    Raised when the authorship filter is the dominant cause of the shortfall,
    including when the truncation filter also removed rows -- that overlap used
    to be resolved in favour of truncation by check order alone.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        super().__init__(
            f"dropping rows with client-supplied assistant turns left "
            f"{counts.written} trainable rows from {source}, below the floor of "
            f"{minimum}: {counts.client_supplied_diagnosis()} "
            f"{counts.drop_accounting()}. "
            "Supervising those turns would teach the draft head to predict "
            "another model's outputs, so dropping the row is the only "
            "representable answer."
        )


class UnattributedCorpusShortfallError(Eagle3Error):
    """The corpus rendered too few rows and no single filter is to blame.

    The honest answer when :meth:`RenderedRowCounts.dominant_drop_bucket` finds
    no winner, or finds one that has no remedy of its own to offer.  Both
    sibling failures lead with a specific fix -- raise the token cap, set
    ``trust_untagged_assistant_messages`` -- and printing either of those for a
    shortfall it did not cause is worse than printing neither: an operator who
    follows it changes a setting, re-runs, and lands in exactly the same place
    having ruled nothing out.  Job 375414 was that failure with the buckets
    reversed.

    So this one names no cause and prescribes no knob.  It reconciles ``read``
    against ``written`` and every bucket and stops there, which is a complete
    statement of what is known.  It earns a class of its own rather than reusing
    a sibling precisely because the remedy paragraph is the part that must be
    absent.  A tie between remediable filters is not told that filters *cannot*
    help, though: either setting can recover rows even though the accounting
    supplies no principled reason to choose one.
    """

    def __init__(self, source: Path, counts: RenderedRowCounts, minimum: int) -> None:
        self.counts = counts
        self.minimum = minimum
        dominant = counts.dominant_drop_bucket()
        if dominant is None:
            cause = (
                "no single filter dominates the loss, so there is no one remedy "
                "to recommend"
            )
            resolution = (
                "Individual filter settings can recover rows in some tied cases, "
                "but this accounting gives no principled reason to choose one; "
                "inspect every bucket before changing the corpus or its filters."
            )
        else:
            cause = (
                f"the largest cause is that {counts.__getattribute__(dominant)} "
                f"rows {_DROP_BUCKETS[dominant]}, which no setting on this "
                "pipeline can recover"
            )
            resolution = (
                "Fixing this needs a corpus whose rows carry trainable assistant "
                "turns, not a change to the filters that dropped them."
            )
        super().__init__(
            f"rendering {source} produced {counts.written} trainable rows, below "
            f"the floor of {minimum}, and {cause}. {counts.drop_accounting()}. "
            f"{resolution}"
        )


class EmptySpeculatorsDatasetError(Eagle3Error):
    """No captured record survived conversion into the Speculators contract.

    Raised when nothing was written and no *one* filter dominates the loss --
    including when the dominant bucket is ``dropped_untrainable``, which has no
    dedicated remedy to print because there is no knob that makes a row without
    an assistant turn trainable.  ``counts`` is optional only because one older
    call site predates the reconciliation; when it is supplied the message
    carries the full accounting, so "nothing converted" is never the *whole*
    answer where a per-bucket breakdown exists.
    """

    def __init__(
        self, source: Path, counts: RenderedRowCounts | None = None
    ) -> None:
        self.source = source
        self.counts = counts
        message = (
            f"no captured trace in {source} converted to a Speculators conversation"
        )
        if counts is not None:
            message += f". {counts.drop_accounting()}"
        super().__init__(message)


def _snapshot_rows(snapshot: TraceSnapshot) -> list[Mapping[str, Any]]:
    """Every record in the snapshot, in capture order.

    Capture order is request order: the trace store appends one line per
    completed response, so line order is the order the server answered.  That
    ordering is what makes the attestation meaningful -- a prefix turn may only
    be matched against a turn generated in an EARLIER row, and shuffling would
    let a row attest itself.
    """
    rows: list[Mapping[str, Any]] = []
    with snapshot.path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise Eagle3Error(
                    f"{snapshot.path} line {number} is not valid JSON"
                ) from error
            if not isinstance(record, Mapping):
                raise Eagle3Error(f"{snapshot.path} line {number} is not a JSON object")
            rows.append(record)
    return rows


def _full_capture_rows(snapshot: TraceSnapshot) -> list[Mapping[str, Any]]:
    """The complete capture the snapshot was drawn from, as attestation evidence.

    The snapshot handed to training is NOT the raw capture: a held-out split
    removes roughly a fifth of the rows, and it removes them from the middle of
    sessions rather than from the end. Attesting those survivors against only
    themselves therefore orphans every prefix turn whose originating row was held
    out, so a genuine self-play corpus reports hundreds of "foreign" turns. GPU
    job 376291 measured exactly that: 0 unmatched of 2460 against the full
    capture, 378 unmatched of 1974 against the training subset.

    The store lives at ``<home>/traces/traces.jsonl`` while the snapshot sits at
    ``<home>/runs/<id>/trace-snapshot/traces.jsonl``. If that relationship ever
    stops holding, this raises rather than quietly falling back to attesting the
    subset against itself -- the fallback would look like a working check while
    being one that cannot pass.
    """
    try:
        home = snapshot.path.parents[3]
    except IndexError:
        raise Eagle3Error(
            f"cannot locate the full trace capture from snapshot {snapshot.path}: "
            "expected it under <home>/runs/<id>/trace-snapshot/. Self-play "
            "attestation needs the complete capture as evidence, because the "
            "training snapshot has had held-out rows removed from the middle of "
            "sessions."
        ) from None
    capture = home / "traces" / "traces.jsonl"
    if not capture.is_file():
        raise Eagle3Error(
            f"self-play attestation needs the full trace capture at {capture}, "
            "which does not exist. Attesting the training snapshot against "
            "itself would fail on genuine self-play traffic, so this refuses "
            "rather than guessing."
        )
    rows: list[Mapping[str, Any]] = []
    lines = capture.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise Eagle3Error(f"{capture} line {number} is not valid JSON") from error
        if isinstance(record, Mapping):
            rows.append(record)
    return rows


def _relabel_self_play_turns(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retag this row's client-supplied ASSISTANT turns as generated.

    Only assistant turns are touched.  A ``user``, ``tool`` or ``system`` turn is
    client-supplied as a matter of fact rather than of provenance accounting, and
    relabelling one would corrupt the loss mask the Speculators loader derives
    from role boundaries.
    """
    updated = dict(record)
    messages = updated.get("messages")
    if not isinstance(messages, list):
        return updated
    rewritten: list[Any] = []
    for message in messages:
        if (
            isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and message.get("provenance_tag") == "client_supplied"
        ):
            rewritten.append({**message, "provenance_tag": "generated"})
        else:
            rewritten.append(message)
    updated["messages"] = rewritten
    return updated


def _raise_for_shortfall(
    source: Path, counts: RenderedRowCounts, minimum: int
) -> None:
    """Raise the failure that describes this rendering pass, or return.

    A separate question from rendering: the loop in
    :func:`_render_speculators_dataset` decides what each row is, and this
    decides which of four failures too few surviving rows amounts to.  The
    reasoning below is what makes that an attribution rather than a check.
    """
    # Checked before the empty-dataset error so a corpus a filter emptied is
    # reported as that filter's doing, not as the generic "nothing converted"
    # that predates the filters and names the wrong cause.
    #
    # Which filter is blamed is decided by which one dropped more rows, not by
    # which is checked first.  Fixed order made truncation the reported cause
    # of every shortfall it contributed anything to: job 375414 rendered 0 of
    # 409 rows with 36 truncated and 373 client-supplied, and was told to raise
    # the token cap.
    #
    # The two-way comparison that replaced that fixed order was the same bug one
    # bucket narrower.  It weighed ``dropped_client_supplied`` against
    # ``dropped_truncated`` and ignored ``dropped_untrainable`` entirely, so a
    # corpus losing 90 rows to untrainability, 6 to truncation and 4 to
    # authorship still reported truncation and still prescribed the token cap --
    # a remedy for 6 of the 100 missing rows.  And on an exact tie it always
    # answered truncation, where by construction neither cause dominates.
    #
    # So: attribute across ALL buckets, and only when one genuinely dominates.
    # A bucket with a dedicated remedy gets its own failure; anything else --
    # untrainability, which no knob can undo, or a tie, which has no single
    # cause -- gets the full accounting and no invented prescription.
    if counts.written < minimum and (
        counts.dropped_untrainable
        or counts.dropped_truncated
        or counts.dropped_client_supplied
    ):
        dominant = counts.dominant_drop_bucket()
        if dominant == "dropped_client_supplied":
            raise ClientSupervisedCorpusError(source, counts, minimum)
        if dominant == "dropped_truncated":
            raise TruncationFilteredCorpusError(source, counts, minimum)
        # Nothing was written at all, so the long-standing "nothing converted"
        # failure is still the truest headline -- it now carries the accounting
        # behind it rather than leaving the breakdown unsaid.
        if not counts.written:
            raise EmptySpeculatorsDatasetError(source, counts)
        raise UnattributedCorpusShortfallError(source, counts, minimum)
    if not counts.written:
        raise EmptySpeculatorsDatasetError(source)


def _render_speculators_dataset(
    snapshot: TraceSnapshot,
    destination: Path,
    *,
    guard: AbortCheck,
    started: float,
    timeout: float,
    policy: TruncatedRowPolicy = TruncatedRowPolicy.KEEP,
    minimum_rows: int = 1,
    trust_untagged_assistant_messages: bool = False,
    trust_self_play_assistant_turns: bool = False,
) -> RenderedRowCounts:
    """Rewrite a leased trace snapshot into the Speculators loader contract.

    SpeedLM captures a top-level ``messages`` key while the Speculators loader
    reads a top-level ``conversations`` key, so handing the snapshot over verbatim
    builds an empty dataset without reporting an error. Records that carry no
    trainable assistant turn are dropped, and an entirely empty result fails loudly
    rather than reaching training as a silently empty dataset.

    Rows whose completion was truncated are dropped under
    :attr:`TruncatedRowPolicy.DROP`; see :data:`TRUNCATED_FINISH_REASONS` for
    why a truncated row is worse than a missing one.  *minimum_rows* is applied
    only when that filter actually removed something, so it bounds the damage
    this filter can do without imposing a floor on corpora that are small for
    unrelated reasons.

    Rows carrying an assistant turn this verifier did not produce are dropped
    outright.  The Speculators loader builds its loss mask by matching assistant
    spans in the *rendered* text and has no per-turn input, so there is no way
    to render a client-supplied assistant turn as context while withholding loss
    from it -- and the turn cannot simply be deleted either, because it is part
    of the prompt the verifier actually saw.  Dropping the row is therefore the
    only representable answer, and it is fail-closed.  *minimum_rows* bounds
    this filter's damage the same way it bounds the truncation filter's.
    """
    read = 0
    written = 0
    dropped_untrainable = 0
    dropped_truncated = 0
    truncated_seen = 0
    dropped_client_supplied = 0
    client_supplied_turns_seen = 0

    # Self-play trust is EARNED here, before a single row is rendered.
    #
    # ``trust_untagged_assistant_messages`` cannot serve this case twice over:
    # it relabels only turns whose tag is ``None`` (see ``rows.py``), while the
    # gateway tags every replayed prefix turn ``"client_supplied"`` outright, and
    # it is an unverified promise from the operator besides.  This flag instead
    # runs an attestation over the whole snapshot in capture order and relabels
    # only if it passes.
    #
    # Failure is LOUD and terminal.  Falling back to dropping the rows would
    # reproduce today's "3 trainable rows against a floor of 32" failure behind a
    # message about corpus size, hiding the fact that the operator's premise --
    # that this traffic is the verifier's own output -- was simply false.
    if trust_self_play_assistant_turns:
        attestation = self_play_attestation(
            _snapshot_rows(snapshot),
            reference_rows=_full_capture_rows(snapshot),
        )
        if not attestation.attested:
            raise Eagle3Error(
                "trust_self_play_assistant_turns is set, but this corpus is not "
                f"self-play traffic: {attestation.detail}. Rows carrying another "
                "model's assistant turns must not be trained on all-assistant; "
                "either point this run at genuine self-play capture or unset the "
                "flag and accept the client-supplied drop."
            )
    try:
        with (
            snapshot.path.open("r", encoding="utf-8") as source,
            destination.open("x", encoding="utf-8") as output,
        ):
            for number, line in enumerate(source, start=1):
                check_deadline(started, timeout, "Speculators row rendering")
                check_abort(guard, "Speculators row rendering")
                if not line.strip():
                    continue
                location = f"{snapshot.path} line {number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Eagle3Error(f"{location} is not valid JSON") from error
                if not isinstance(record, Mapping):
                    raise Eagle3Error(f"{location} is not a JSON object")
                read += 1
                if trust_self_play_assistant_turns:
                    # Safe as a blanket rewrite precisely because the attestation
                    # above already proved EVERY client-supplied assistant turn in
                    # this snapshot reproduces a turn this server generated
                    # earlier. Relabelling per row without that proof would be the
                    # unverified assertion this flag exists to replace.
                    record = _relabel_self_play_turns(record)
                # ``truncated_seen`` is evidence about the input, not an
                # exclusive drop bucket.  Count it before any filter can
                # ``continue``.  The old order checked truncation only after
                # conversion and authorship, so a truncated replayed row was
                # correctly charged to ``dropped_client_supplied`` yet vanished
                # from the supposedly corpus-wide truncation counter.  Drop
                # buckets remain exclusive and still reconcile exactly; evidence
                # counters are allowed to overlap because causes do.
                truncated = _is_truncated(record)
                if truncated:
                    truncated_seen += 1
                converted = _speculators_record(record)
                if converted is None:
                    dropped_untrainable += 1
                    continue
                unowned = _unowned_assistant_turns(
                    record, trust_untagged=trust_untagged_assistant_messages
                )
                if unowned:
                    client_supplied_turns_seen += unowned
                    dropped_client_supplied += 1
                    continue
                if truncated and policy is TruncatedRowPolicy.DROP:
                    dropped_truncated += 1
                    continue
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1
    except OSError as error:
        raise Eagle3Error(
            f"cannot render Speculators rows from {snapshot.path}: {error}"
        ) from error
    counts = RenderedRowCounts(
        read=read,
        written=written,
        dropped_untrainable=dropped_untrainable,
        dropped_truncated=dropped_truncated,
        truncated_seen=truncated_seen,
        policy=policy,
        dropped_client_supplied=dropped_client_supplied,
        client_supplied_turns_seen=client_supplied_turns_seen,
    )
    logger.info(
        "rendered %d of %d captured rows (%d untrainable, %d truncated dropped, "
        "%d truncated seen, %d client-supplied dropped, %d client-supplied turns, "
        "policy=%s)",
        written,
        read,
        dropped_untrainable,
        dropped_truncated,
        truncated_seen,
        dropped_client_supplied,
        client_supplied_turns_seen,
        policy.value,
    )
    # The floors below fire only when this filter left too FEW rows.  A corpus
    # it merely reshaped clears them and then trains, promotes and reports like
    # any other corpus while every multi-turn row is gone.  Warn on the drop
    # FRACTION instead, carrying the same diagnosis the hard failure carries, so
    # the survivable case and the fatal case read alike.
    if counts.client_supplied_drop_fraction >= CLIENT_SUPPLIED_DROP_ALERT_FRACTION:
        logger.warning(
            "client-authorship filter dropped a material fraction of the "
            "corpus: %s",
            counts.client_supplied_diagnosis(),
        )
    _raise_for_shortfall(snapshot.path, counts, minimum_rows)
    return counts
