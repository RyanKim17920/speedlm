"""Replay a frozen benchmark suite against an OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final

from speedlm.config import SamplingConfig
from speedlm.gate.suite import BenchmarkSuite, FrozenContext

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReplayError(RuntimeError):
    """Raised when a replay run encounters a critical error."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RequestResult:
    """Result of a single suite-request replay."""

    context_hash: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_text: str
    valid: bool
    error: str = ""
    #: The generated sequence as the endpoint tokenised it, in order.
    #:
    #: Populated only when the request asked for logprobs -- see
    #: :func:`_send_request`.  Empty means "not captured", which is different
    #: from "the model emitted nothing"; a caller comparing two generations
    #: must fall back to characters rather than treat an empty tuple as a
    #: zero-length match.
    output_tokens: tuple[str, ...] = ()
    #: vLLM's ``finish_reason`` verbatim (``"stop"``, ``"length"``, ...).
    #: Recorded so a divergence at the very end of a bounded generation can be
    #: told apart from one that ended naturally.
    finish_reason: str = ""
    #: The reasoning channel a thinking model fills instead of ``content``.
    #:
    #: Reasoning models spend most of a bounded budget inside ``<think>``, and
    #: an OpenAI-compatible server that parses that block files it under
    #: ``message.reasoning`` (vLLM 0.25.x) or ``message.reasoning_content``,
    #: leaving ``content`` null until the block closes.  Captured because it is
    #: the *only* direct evidence that a response whose ``content`` is empty was
    #: nonetheless a complete, healthy generation.
    reasoning_text: str = ""
    #: How many tool calls the assistant message dispatched.
    #:
    #: The third channel a model can put its work in, and the one the replay
    #: path used to be blind to.  An OpenAI-shaped tool dispatch is
    #: ``content: null`` plus a populated ``tool_calls`` array plus
    #: ``finish_reason: "tool_calls"`` -- a complete, healthy generation with
    #: nothing in either text channel.  Recorded, and read by
    #: :func:`_validity_error`, because this gate replays agentic traffic *with*
    #: the captured tool schemas (see :func:`_send_request`), so this response
    #: shape is one the suite actively provokes.
    tool_call_count: int = 0

    @property
    def invalid(self) -> bool:
        return not self.valid

    @property
    def generated_text(self) -> str:
        """Everything the model emitted, reasoning channel included.

        Tool-call arguments are deliberately absent: they are structured JSON
        the server may re-serialise, not a token stream, so folding them in
        would compare two renderings rather than two generations.
        :attr:`tool_call_count` records that the dispatch happened.

        ``response_text`` alone is not the generation for a thinking model: at
        the caps this gate replays under, it is routinely empty while hundreds
        of tokens sit in ``reasoning_text``.  A character-basis comparison over
        ``response_text`` would therefore compare ``""`` against ``""`` and
        report agreement it never checked.
        """
        return self.reasoning_text + self.response_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_hash": self.context_hash,
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "response_text": self.response_text,
            "valid": self.valid,
            "error": self.error,
            "output_tokens": list(self.output_tokens),
            "finish_reason": self.finish_reason,
            "reasoning_text": self.reasoning_text,
            "tool_call_count": self.tool_call_count,
        }


@dataclass(frozen=True, slots=True)
class RunResults:
    """Results from a single replay run (one pass over all contexts)."""

    results: tuple[RequestResult, ...]
    total_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    valid_count: int
    invalid_count: int
    invalid_rate: float
    #: Valid responses whose finish reason said the output cap ended them, i.e.
    #: it is in :data:`TRUNCATED_FINISH_REASONS`.
    truncated_count: int = 0
    #: Valid responses whose finish reason said the *model* ended them, i.e. it
    #: is in :data:`NATURAL_STOP_FINISH_REASONS`.
    #:
    #: Counted, not inferred as "reported minus truncated".  A finish reason
    #: this build does not recognise falls into neither counter -- see
    #: :func:`classify_finish_reason` for why that is the fail-closed choice.
    natural_stop_count: int = 0

    @property
    def finish_reason_count(self) -> int:
        """Responses whose finish reason this build could classify.

        The denominator of :attr:`truncation_rate`, and it is deliberately
        *not* ``len(results)``.  A request that failed before the model
        answered (HTTP error, empty ``choices``, transport exception) carries
        ``""``, and folding those into the denominator would make a run look
        less truncated the more of it failed.  It is equally deliberately not
        "everything that carried a non-empty string": an unrecognised spelling
        is excluded on exactly the same footing as a blank one, because the
        gate cannot say which of the two buckets it belongs in.

        Zero here means the endpoint reported nothing this build understands,
        which is a different fact from "nothing was truncated" -- see
        :class:`speedlm.gate.decide.TruncationRegime`, which reads zero as
        ``UNTESTABLE``.
        """
        return self.truncated_count + self.natural_stop_count

    @property
    def truncation_rate(self) -> float:
        """Fraction of *classified* generations that hit the output cap.

        ``0.0`` when nothing was classifiable, which is why callers must
        consult :attr:`finish_reason_count` before reading this as a
        measurement.
        """
        if self.finish_reason_count <= 0:
            return 0.0
        return self.truncated_count / self.finish_reason_count

    @property
    def avg_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return self.total_latency_s / len(self.results)

    @property
    def output_tok_per_sec(self) -> float:
        if self.total_latency_s == 0:
            return 0.0
        return self.total_completion_tokens / self.total_latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_latency_s": self.total_latency_s,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "invalid_rate": self.invalid_rate,
            "finish_reason_count": self.finish_reason_count,
            "truncated_count": self.truncated_count,
            "natural_stop_count": self.natural_stop_count,
            "truncation_rate": self.truncation_rate,
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Aggregated results from N replay runs."""

    run_results: tuple[RunResults, ...]
    num_runs: int
    suite_hash: str
    #: Identifies the single :func:`replay_suite` invocation that produced every
    #: run in ``run_results``.  Two results carrying the *same* non-empty id
    #: were collected back-to-back against one live engine, with no restart,
    #: no weight reload and no cache teardown between them; two results
    #: carrying *different* non-empty ids were not.
    #:
    #: This exists because the gate's divergence criterion compares a measured
    #: pair against a control pair, and that comparison is only a test if both
    #: pairs span the same boundaries.  A control collected inside one
    #: invocation cannot bound the variation of a measurement that spans two,
    #: and without this stamp the decider has no way to notice.  See
    #: :func:`speedlm.gate.decide.decide_promotion`.
    #:
    #: Empty means *unstamped*, which the decider must read as "provenance
    #: unknown", never as "same session": a :class:`ReplayResult` rebuilt from
    #: slices of another one (as the runner's correctness split does) carries no
    #: claim about how its runs were collected.
    session_id: str = ""

    @property
    def avg_invalid_rate(self) -> float:
        if not self.run_results:
            return 0.0
        return sum(r.invalid_rate for r in self.run_results) / len(self.run_results)

    @property
    def avg_output_tok_per_sec(self) -> float:
        if not self.run_results:
            return 0.0
        return sum(r.output_tok_per_sec for r in self.run_results) / len(self.run_results)

    @property
    def total_finish_reason_count(self) -> int:
        """Classified finish reasons across every repeat, pooled."""
        return sum(r.finish_reason_count for r in self.run_results)

    @property
    def total_natural_stops(self) -> int:
        """Generations that ended on the model's own terms, pooled.

        Pooled rather than averaged because the question it settles is a
        count: whether this arm produced *any* evidence of where the model
        stops when the harness is not stopping it.  A per-repeat mean would
        round that evidence away.
        """
        return sum(r.natural_stop_count for r in self.run_results)

    @property
    def avg_truncation_rate(self) -> float:
        """Fraction of reported generations that hit the output cap, pooled.

        Pooled over reported responses rather than averaged over repeats, so
        that a repeat in which most requests errored cannot weigh as much as a
        clean one.  ``0.0`` when nothing reported a finish reason; read
        :attr:`total_finish_reason_count` first.
        """
        reported = self.total_finish_reason_count
        if reported <= 0:
            return 0.0
        return sum(r.truncated_count for r in self.run_results) / reported

    @property
    def any_high_invalid_rate(self, threshold: float = 0.1) -> bool:
        return any(r.invalid_rate > threshold for r in self.run_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_results": [r.to_dict() for r in self.run_results],
            "num_runs": self.num_runs,
            "suite_hash": self.suite_hash,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

#: ``finish_reason`` an OpenAI-compatible server reports when a generation
#: stopped because it exhausted the request's ``max_tokens``, as opposed to
#: because the model emitted its own stop token.  This is the one signal that
#: separates a deliberate truncation from a broken generation.
FINISH_REASON_LENGTH: Final = "length"

#: Every ``finish_reason`` spelling that means "the cap ended this", not just
#: the OpenAI chat one above.
#:
#: Mirrors ``speedlm.training.backends.eagle3.TRUNCATED_FINISH_REASONS`` and
#: ``speedlm.traces.store.TRUNCATED_FINISH_REASONS`` -- restated rather than
#: imported for the same reason the trace store restates it (the gate must not
#: depend on a training backend), and pinned against both by
#: ``tests/test_gate_replay.py::test_the_gate_truncation_vocabulary_matches_the_training_filter``
#: so the three cannot drift apart unnoticed.
#:
#: The gate needs the full set, not ``FINISH_REASON_LENGTH`` alone.  ``length``
#: is the chat-completions spelling; ``incomplete`` is the Responses API's
#: spelling of the identical event, and this project's own gateway emits it
#: (``gateway/responses.py``).  Counting only ``length`` meant a server that
#: says ``incomplete`` reported *zero* truncations for a wholly-capped run and
#: classified ``BOUNDED`` -- the strongest possible "the cap was not binding"
#: reading of the strongest possible saturation.  That is precisely the silent
#: masking :class:`speedlm.gate.decide.TruncationRegime` exists to prevent, so
#: the gate would have been generalizing itself into a vacuous guard on any
#: server that is not the one it was written against.
TRUNCATED_FINISH_REASONS: Final = frozenset({"length", "incomplete"})


def is_truncated_finish_reason(value: str | None) -> bool:
    """Whether *value* means the output cap ended the generation.

    Normalised exactly as ``speedlm.traces.store._is_truncated_completion``
    normalises it -- ``strip().lower()`` -- so a server spelling it ``Length``
    is not silently read as a natural stop.  A positive test rather than "not
    ``stop``": an unknown value is not evidence of truncation, and
    ``tool_calls`` is a perfectly complete stop.
    """
    return isinstance(value, str) and value.strip().lower() in TRUNCATED_FINISH_REASONS


#: Every ``finish_reason`` spelling that means "the *model* ended this".
#:
#: Drawn from what this repository already knows servers emit, not invented:
#:
#: * ``stop`` -- the OpenAI chat value for the model emitting a stop token.
#: * ``tool_calls`` -- an OpenAI-shaped tool dispatch.  A complete generation:
#:   the model chose to hand off rather than keep decoding.  This gate replays
#:   agentic contexts with their captured tool schemas (:func:`_send_request`),
#:   so it provokes this value itself.
#: * ``function_call`` -- the legacy single-function spelling of the same event,
#:   still emitted by OpenAI-compatible servers with older-style tool support.
#: * ``content_filter`` -- generation ended by a moderation hook.  Not a healthy
#:   answer, but unambiguously *not* the output cap, which is the only question
#:   this vocabulary is asked.
#: * ``completed`` -- the Responses API's spelling.  That path files the
#:   response *status* into this same field (``gateway/responses.py``), where a
#:   finished response reads ``completed`` and a truncated one reads
#:   ``incomplete`` -- the exact counterpart already carried in
#:   :data:`TRUNCATED_FINISH_REASONS`.  This project's own gateway produces it.
#: * ``repetition`` -- vLLM's own terminal reason for its n-gram repetition
#:   detector firing.  Verified against the pinned build rather than assumed:
#:   ``vllm/v1/engine/__init__.py`` declares
#:   ``FINISH_REASON_STRINGS = ("stop", "length", "abort", "error",
#:   "repetition")`` above the comment "These are possible values of
#:   ``RequestOutput.finish_reason``, so form part of the external API", and
#:   ``FinishReason`` documents it as "repetitive token pattern detected
#:   (hallucination)".  ``vllm/v1/core/sched/utils.py`` sets
#:   ``RequestStatus.FINISHED_REPETITION`` when ``check_sequence_repetition``
#:   trips, and ``vllm/v1/request.py`` maps that status to
#:   ``FinishReason.REPETITION``.  Nothing on that path raises, so it arrives
#:   on an ordinary 200 with a completed body.
#:
#:   It belongs here because of the *only* question this vocabulary is asked:
#:   did the output cap bind?  It did not -- the detector ended the generation
#:   early, strictly before ``max_tokens``.  Counting it as truncation would
#:   invent cap pressure that never happened; leaving it unknown makes an
#:   all-``repetition`` arm ``UNTESTABLE`` and rejects it with
#:   ``TRUNCATION_UNMEASURED``, a false veto on a gate that has no rollback.
#:   Like ``content_filter``, it is not a *healthy* answer -- but health is the
#:   ``invalid_rate`` guard's question, not this one's.
#:
#:   Its two siblings in that tuple are deliberately **not** here.  ``error`` is
#:   converted to an exception by ``_raise_if_error`` in
#:   ``vllm/entrypoints/generate/base/serving.py`` ("Invariant: always converted
#:   to 500 Internal Server Error"), so this gate never sees it on a 200 at all.
#:   ``abort`` means the client hung up, so there is no completed response to
#:   classify -- and if one ever did arrive, "we cancelled it" is not evidence
#:   the cap was slack.  Both stay unknown, which is fail-closed.
NATURAL_STOP_FINISH_REASONS: Final = frozenset(
    {
        "stop",
        "tool_calls",
        "function_call",
        "content_filter",
        "completed",
        "repetition",
    }
)


#: The three buckets :func:`classify_finish_reason` sorts a finish reason into.
FINISH_REASON_TRUNCATED: Final = "truncated"
FINISH_REASON_NATURAL: Final = "natural"
FINISH_REASON_UNKNOWN: Final = "unknown"


def is_natural_stop_finish_reason(value: str | None) -> bool:
    """Whether *value* means the model, not the cap, ended the generation."""
    return (
        isinstance(value, str)
        and value.strip().lower() in NATURAL_STOP_FINISH_REASONS
    )


def classify_finish_reason(value: str | None) -> str:
    """Sort one finish reason into ``"truncated"``, ``"natural"``, ``"unknown"``.

    Three buckets, and the third is the point.  A natural stop used to be
    *inferred*: anything non-blank that was not in
    :data:`TRUNCATED_FINISH_REASONS` was counted as the model stopping by
    itself.  That makes an unrecognised spelling -- a server this build has
    never seen, a value renamed upstream, a typo in a proxy -- into positive
    evidence that the cap was not binding, which is precisely backwards.  A
    wholly capped run behind such a server reports zero truncations against a
    full denominator and classifies ``BOUNDED``: the strongest possible "the
    cap did not bind" reading of the strongest possible saturation, which is
    the silent masking :class:`speedlm.gate.decide.TruncationRegime` exists to
    prevent.

    So an unknown value counts in *neither* bucket and is excluded from the
    denominator, on exactly the footing a blank or absent reason already had.
    An arm whose responses all carry an unrecognised reason therefore reports a
    classified count of zero, classifies ``UNTESTABLE``, and is rejected with
    ``TRUNCATION_UNMEASURED`` -- the failure mode is "the gate says it could not
    measure", never "the gate says the cap was fine".  Fail-closed.

    **This deliberately reverses the argument recorded in**
    ``speedlm.training.backends.eagle3.TRUNCATED_FINISH_REASONS``, **and both
    are correct where they stand -- do not "reconcile" them by deleting one.**
    That module keeps a positive truncation list and argues explicitly against
    an inverted test, because there a row whose finish reason is unknown must
    still be *kept*: the filter has exactly two outcomes, keep or drop, and an
    unknown value falling into "drop" would silently discard corpora archived
    before the field existed.  Keeping is that filter's safe default.

    The gate has a third state the training filter does not: ``UNTESTABLE``,
    i.e. "this run cannot answer the question", which rejects rather than
    promotes.  Because refusing to classify is available *and* is the
    conservative outcome, the gate's safe default is to refuse -- where the
    filter's is to keep.  Same evidence, opposite defaults, because the two
    have different action spaces.  The shared constant is the truncated
    vocabulary, which is pinned across all three copies by
    ``tests/test_gate_replay.py`` and stays identical.
    """
    if is_truncated_finish_reason(value):
        return FINISH_REASON_TRUNCATED
    if is_natural_stop_finish_reason(value):
        return FINISH_REASON_NATURAL
    return FINISH_REASON_UNKNOWN

#: Field names a server may file a thinking model's ``<think>`` block under,
#: in the order they are consulted.  vLLM 0.25.x uses ``reasoning``; the
#: ``reasoning_content`` spelling is what several other OpenAI-compatible
#: servers (and older vLLM reasoning parsers) emit; ``thinking`` is the
#: third spelling in the wild.
#:
#: This list is duplicated in :mod:`speedlm.gateway.sse` and
#: :mod:`speedlm.gateway.capture`, and it had already drifted: this copy was
#: missing ``thinking``, so a server using that spelling was captured by the
#: gateway as a reasoning response but read by the gate as an empty one --
#: the exact shape that inflates ``invalid_rate`` and fails a healthy
#: candidate.  The three copies are now identical in *content*; the order
#: here matches ``gateway.sse`` so the gate resolves a response the same way
#: the capture path did.  ``test_gate_replay`` asserts the three agree, which
#: is the containable half of the fix -- see the report for the
#: shared-constant requirement that would remove the duplication outright.
_REASONING_FIELDS: Final = ("reasoning_content", "reasoning", "thinking")

#: Recorded on a response the engine answered without generating anything at
#: all.  This is the failure ``high_invalid_rate`` exists to catch: zero
#: generated tokens means the throughput and acceptance numbers measured over
#: that sample are meaningless.
_ERROR_NO_TOKENS: Final = "No generated tokens"

#: Recorded on a response that generated tokens and then surfaced none of them
#: while claiming to have stopped of its own accord.  Also a broken engine --
#: a healthy one that stops on its own has something to show for it.
_ERROR_EMPTY_TEXT: Final = "Empty response text"


def _extract_reasoning(message: dict[str, Any]) -> str:
    """Recover the reasoning channel, whichever name the server used for it."""
    for field in _REASONING_FIELDS:
        value = message.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_tool_calls(message: dict[str, Any]) -> int:
    """Count the tool calls an assistant message dispatched.

    Read the same way :mod:`speedlm.gateway.sse` reads them off a
    non-streaming body -- a list of objects under ``tool_calls`` -- so the gate
    recognises the responses the capture path recognised.  A malformed value is
    counted as zero rather than raising: this is one field of one response, and
    losing the whole replay over it is the failure mode the ``finish_reason``
    coercion above already had to be fixed for.

    "Object" is not enough on its own, though, and that gap re-hollowed the
    guard it was written to serve.  This count is the *only* thing standing
    between "the model surfaced no output" and "the model dispatched a tool"
    in :func:`_validity_error`, so ``{}`` counting as a dispatch means a
    response with ``content: null``, ``tool_calls: [{}]`` and
    ``finish_reason: "tool_calls"`` reads as a healthy natural stop -- an
    engine emitting nothing at all, scored as if it had answered.  The comment
    above claimed malformed values were counted as zero; a malformed *dict* was
    not.

    So require what makes a tool call an actual dispatch rather than an empty
    envelope: a ``function`` object carrying a non-empty ``name``.  That is the
    field ``sse._ToolCall.to_dict`` always emits and the only part a client
    could route on -- ``id`` is dropped there when the server never sent one and
    ``type`` is defaulted, so neither can be required without rejecting
    responses the capture path accepts.  Anything less is not output.

    The deprecated singular ``message.function_call`` shape is counted by the
    same rule.  That is not optional compatibility trivia: ``function_call`` is
    still an explicitly recognised natural finish reason above.  Recognising
    the terminal state while ignoring the only output channel that accompanies
    it makes a real legacy dispatch invalid and inflates ``invalid_rate`` -- the
    exact false-negative this function exists to prevent.
    """
    raw = message.get("tool_calls")
    listed = (
        sum(1 for call in raw if _is_dispatched_tool_call(call))
        if isinstance(raw, list)
        else 0
    )
    legacy = message.get("function_call")
    return listed + (1 if _is_named_function(legacy) else 0)


def _is_named_function(function: object) -> bool:
    """Whether a function payload carries the name a client routes on."""
    if not isinstance(function, dict):
        return False
    name = function.get("name")
    return isinstance(name, str) and bool(name.strip())


def _is_dispatched_tool_call(call: object) -> bool:
    """Whether one ``tool_calls`` entry names a function to actually call."""
    if not isinstance(call, dict):
        return False
    return _is_named_function(call.get("function"))


def _validity_error(
    *,
    completion_tokens: int,
    text: str,
    reasoning: str,
    output_tokens: tuple[str, ...],
    tool_calls: int,
    finish_reason: str,
) -> str:
    """Classify one 200-OK response: ``""`` when usable, else why it is not.

    "Valid" used to mean "``choices[0].message.content`` is a non-empty
    string", which asks a question neither replay pass needs answered.  The
    throughput/acceptance pass never reads the text -- it needs tokens/second
    and it needs the engine's Prometheus acceptance counters to move.  The
    correctness pass compares token streams recovered from logprobs, which do
    not care what the text renders as.  Only the predicate cared, and on a
    reasoning model it was wrong most of the time: SLURM 369147 replayed
    Qwen3-8B at ``benchmark_max_tokens=512``, 76 of 103 held-out contexts spent
    the whole budget inside ``<think>``, and every one came back with
    ``content: null``, ``finish_reason: "length"`` and
    ``completion_tokens: 512``.  The gate rejected on ``invalid_rate 0.7379``
    while its own correctness pass reported zero divergences.  gpt-oss-20b
    (SLURM 369148) produces the identical shape, so this is a property of
    reasoning models, not of one tokenizer.

    What replaces it is "the engine did real work", which is what the threshold
    is actually protecting.  Ordered so the genuinely broken cases still fail:

    1. Nothing generated and nothing surfaced -> invalid.  ``usage`` is the
       cleanest signal available and it was non-zero for every one of the 512
       captured responses in that run, so a zero here is a real anomaly.
       Surfaced output is accepted as a substitute because a server may omit
       ``usage`` entirely.
    2. Anything surfaced -- content, reasoning, captured tokens, *or a
       dispatched tool call* -> valid.
    3. Nothing surfaced but the cap was hit -> valid.  The tokens exist and
       were counted; which channel the server filed them under does not change
       what a throughput sample measures.
    4. Nothing surfaced and the model claims it stopped on its own -> invalid.
       That is an engine returning empty responses, which is exactly the
       measurement-is-worthless case the threshold guards.

    ``tool_calls`` is the third channel and was the missing one.  A tool
    dispatch is ``content: null``, a populated ``tool_calls`` array and
    ``finish_reason: "tool_calls"`` -- an entirely normal, healthy OpenAI-shaped
    reply that both preceding tests read as empty, so it was recorded as
    ``_ERROR_EMPTY_TEXT``.  That is not hypothetical here: this gate replays
    agentic contexts *with* their captured tool schemas (see
    :func:`_send_request`), so the suite provokes exactly this shape.  The
    consequences compounded -- on agentic traffic the arm could trip
    ``HIGH_INVALID_RATE`` outright, and even below that threshold each
    misclassified response was dropped from the finish-reason counts, stripping
    genuine natural stops out of a ``MIXED`` run until it read ``SATURATED`` (or
    ``UNTESTABLE``).  Validity accounting and truncation accounting were
    contradicting each other on the same response.
    """
    surfaced = (
        bool(text) or bool(reasoning) or bool(output_tokens) or tool_calls > 0
    )
    if completion_tokens <= 0 and not surfaced:
        return _ERROR_NO_TOKENS
    if surfaced:
        return ""
    # Any spelling of "the cap ended this", not just the chat-completions one.
    # A Responses-API server reports the identical event as ``incomplete``, and
    # testing only ``length`` here marked its truncated-empty responses invalid.
    # That is not merely a mislabel: those responses then fail the validity
    # filter, drop out of the truncation counts entirely, and the run rejects
    # as ``HIGH_INVALID_RATE`` -- so ``TRUNCATION_SATURATED`` could never fire
    # for the very response shape it was written to catch.
    if is_truncated_finish_reason(finish_reason):
        return ""
    return _ERROR_EMPTY_TEXT


def _extract_output_tokens(choice: dict[str, Any]) -> tuple[str, ...]:
    """Recover the generated token sequence from an OpenAI-shaped choice.

    The OpenAI-compatible surface vLLM serves has no way to return raw token
    *ids* per request: ``--return-tokens-as-token-ids`` is a server-launch flag,
    and the gate replays against an engine it did not launch with that flag.
    What the endpoint does expose per request is ``logprobs.content``, one entry
    per generated token carrying that token's decoded piece.  That is a genuine
    token-level alignment -- the segmentation is the model's own, not a
    re-tokenisation of the response string -- which is what a first-divergence
    index needs.  (If the server *was* started with
    ``--return-tokens-as-token-ids`` these pieces are literally ``token_id:N``
    strings, and the comparison becomes an exact id comparison for free.)

    Returns an empty tuple when logprobs were not requested or not returned, so
    the caller can tell "not captured" from "captured, and empty".
    """
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return ()
    content = logprobs.get("content")
    if not isinstance(content, list):
        return ()
    tokens: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            return ()
        token = entry.get("token")
        if not isinstance(token, str):
            return ()
        tokens.append(token)
    return tuple(tokens)


async def _send_request(
    client: Any,
    ctx: FrozenContext,
    sampling: SamplingConfig,
    model: str,
    *,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> RequestResult:
    """Send a single context to the OpenAI-compatible endpoint.

    Args:
        max_tokens: Hard cap on generated tokens, or ``None`` to let the
            generation run to the served model's length limit.  Both of the
            gate's passes set it, at different values: the correctness pass at
            ``correctness_max_tokens`` and the throughput/acceptance pass at
            ``benchmark_max_tokens``.  An earlier revision of this docstring
            said the throughput pass never sets it, on the theory that bounding
            output would change the statistic the throughput threshold is
            calibrated against.  That was wrong twice over -- ``None`` is not
            "unbounded" but "bounded by ``max_model_len`` minus the prompt",
            which is model-dependent, and the throughput statistic is an
            arm-to-arm ratio under a shared cap, so a common cap cancels.  See
            :attr:`speedlm.config.IdleTuningConfig.benchmark_max_tokens`.
        capture_tokens: Ask for per-token logprobs so the response carries the
            model's own tokenisation of its output.
    """
    import httpx
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in ctx.messages],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
    }
    if ctx.tools:
        # Agentic traffic is captured with the tool schemas the caller offered.
        # Replaying without them sends a strictly different prompt -- the
        # template renders no tool block, so the model cannot dispatch and the
        # gate measures a request production never served. Sent only when the
        # context has tools, so chat replays stay byte-identical.
        payload["tools"] = [dict(tool) for tool in ctx.tools]
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if capture_tokens:
        # ``top_logprobs`` is deliberately omitted: the gate needs the chosen
        # token per position, not the distribution around it, and asking for
        # alternatives multiplies the response size by the requested width.
        payload["logprobs"] = True
    start = time.monotonic()
    try:
        resp = await client.post("v1/chat/completions", json=payload)
        latency = time.monotonic() - start

        if resp.status_code >= 400:
            return RequestResult(
                context_hash=ctx.context_hash,
                latency_s=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_text="",
                valid=False,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            return RequestResult(
                context_hash=ctx.context_hash,
                latency_s=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_text="",
                valid=False,
                error="Empty choices array",
            )

        choice = choices[0]
        message = choice.get("message", {})
        raw_text = message.get("content")
        text = raw_text if isinstance(raw_text, str) else ""
        reasoning = _extract_reasoning(message)
        tool_calls = _extract_tool_calls(message)
        output_tokens = _extract_output_tokens(choice)
        # A finish reason that is not a string is a reason *not given*, and is
        # normalised to ``""`` here -- at ingestion -- exactly as a missing one
        # already was.  ``or ""`` alone accepted any truthy JSON value, so a
        # server answering ``finish_reason: 7`` built a ``RequestResult`` whose
        # ``str``-annotated field held an int.  Nothing failed at that point:
        # the counting block in ``_run_single`` later called ``.strip()`` on it
        # and raised ``AttributeError`` *outside* this function's ``except``
        # boundary, so one malformed response killed the entire replay instead
        # of being recorded as one response that gave no reason.  Guarding here
        # rather than at the strip site keeps the annotation true and leaves the
        # rest of the pipeline free of defensive checks.
        raw_finish_reason = choice.get("finish_reason")
        finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else ""
        usage = body.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        error = _validity_error(
            completion_tokens=ct,
            text=text,
            reasoning=reasoning,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            response_text=text,
            valid=not error,
            error=error,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            reasoning_text=reasoning,
            tool_call_count=tool_calls,
        )

    except httpx.HTTPError as exc:
        latency = time.monotonic() - start
        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_text="",
            valid=False,
            error=f"HTTPError: {exc}",
        )
    except Exception as exc:
        latency = time.monotonic() - start
        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_text="",
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_single(
    client: Any,
    suite: BenchmarkSuite,
    sampling: SamplingConfig,
    model: str,
    concurrency: int,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> RunResults:
    """Execute one full pass over the suite, up to *concurrency* in flight.

    Concurrency changes only how fast the pass runs, never what it contains.
    Every context in ``suite.contexts`` is sent exactly once, and the recorded
    ``results`` stay in suite order regardless of completion order:
    :func:`asyncio.gather` returns results positionally, so the per-request
    array remains alignable with the suite that produced it.  The aggregates
    below are order-independent sums, so they are unaffected either way.

    At ``concurrency == 1`` this is byte-for-byte the sequential pass it
    replaced -- a semaphore of one admits one waiter at a time, in the order
    the tasks were created.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(ctx: FrozenContext) -> RequestResult:
        async with semaphore:
            return await _send_request(
                client,
                ctx,
                sampling,
                model,
                max_tokens=max_tokens,
                capture_tokens=capture_tokens,
            )

    results: list[RequestResult] = list(
        await asyncio.gather(*(bounded(ctx) for ctx in suite.contexts))
    )

    total_latency = sum(r.latency_s for r in results)
    total_pt = sum(r.prompt_tokens for r in results)
    total_ct = sum(r.completion_tokens for r in results)
    valid = sum(1 for r in results if r.valid)
    invalid = len(results) - valid
    # Only *valid* responses may speak to who chose the generation length.
    # An invalid one is a response the run failed to obtain, and its finish
    # reason describes the failure, not the model: 99 healthy ``length``
    # responses plus a single broken empty ``stop`` would otherwise report one
    # natural stop, classify ``MIXED`` instead of ``SATURATED``, and promote --
    # while ``invalid_rate`` sat at 0.01, far under its own threshold.  One
    # failed request must not be able to vouch for a wholly-capped run.
    #
    # Each reason is sorted into one of three buckets and only two of them are
    # counted -- see :func:`classify_finish_reason`.  A whitespace-only value
    # lands in ``unknown`` and so enters neither, which is the same treatment a
    # blank or absent reason gets and the reason no separate ``.strip()`` guard
    # is needed here: a reason that says nothing, and a reason this build cannot
    # read, are both reasons the gate must not draw a conclusion from.
    truncated = 0
    natural = 0
    for r in results:
        if not r.valid:
            continue
        bucket = classify_finish_reason(r.finish_reason)
        if bucket == FINISH_REASON_TRUNCATED:
            truncated += 1
        elif bucket == FINISH_REASON_NATURAL:
            natural += 1

    return RunResults(
        results=tuple(results),
        total_latency_s=total_latency,
        total_prompt_tokens=total_pt,
        total_completion_tokens=total_ct,
        valid_count=valid,
        invalid_count=invalid,
        invalid_rate=invalid / len(results) if results else 0.0,
        truncated_count=truncated,
        natural_stop_count=natural,
    )


async def replay_suite(
    suite: BenchmarkSuite,
    endpoint_url: str,
    sampling: SamplingConfig,
    *,
    repeats: int = 1,
    timeout: float = 120.0,
    model: str = "auto",
    concurrency: int = 1,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> ReplayResult:
    """Replay suite against an OpenAI-compatible endpoint N times.

    Args:
        suite: The frozen benchmark suite.
        endpoint_url: Base URL of the endpoint.
        sampling: Sampling parameters (temperature, top_p, seed).
        repeats: Number of full passes over the suite.
        timeout: Per-request timeout in seconds.
        model: Served model name to request.
        concurrency: Requests kept in flight within one pass.  Repeats stay
            strictly sequential -- a repeat is the unit the gate's per-repeat
            statistic is computed over, so overlapping two of them would fold
            two samples into one measurement.
        max_tokens: Per-request output cap, or ``None`` for the served model's
            own limit.  Bounding output changes what throughput means, so the
            gate sets this only on its correctness pass.
        capture_tokens: Request per-token logprobs so each result carries the
            model's own tokenisation, which is what
            :func:`speedlm.gate.decide.first_divergence` aligns on.

    Returns:
        Aggregated :class:`ReplayResult` with per-run data.

    Raises:
        ReplayError: If repeats < 1, concurrency < 1, max_tokens < 1, or the
            suite is empty.
    """
    import httpx

    if repeats < 1:
        raise ReplayError(f"repeats must be >= 1, got {repeats}")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ReplayError(f"concurrency must be an integer >= 1, got {concurrency!r}")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        raise ReplayError(f"max_tokens must be None or an integer >= 1, got {max_tokens!r}")
    if not suite.contexts:
        raise ReplayError("Cannot replay empty suite")

    runs: list[RunResults] = []

    async with httpx.AsyncClient(
        base_url=endpoint_url,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        # The pool must not become the narrower limit than the semaphore, or
        # the configured degree would silently not be the degree achieved.
        limits=httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        ),
    ) as client:
        for _ in range(repeats):
            run = await _run_single(
                client,
                suite,
                sampling,
                model,
                concurrency,
                max_tokens=max_tokens,
                capture_tokens=capture_tokens,
            )
            runs.append(run)

    return ReplayResult(
        run_results=tuple(runs),
        num_runs=len(runs),
        suite_hash=suite.suite_hash,
        # Minted per invocation, not per run: every repeat above ran against the
        # same live engine, and that is exactly what this id asserts.
        session_id=uuid.uuid4().hex,
    )
