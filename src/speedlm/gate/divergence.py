"""The output-correctness criterion the promotion gate rejects on.

Split out of :mod:`speedlm.gate.decide`, which had grown to hold two separate
arguments: whether a candidate head is *faster*, and whether it is emitting the
same tokens at all.  Only the second one lives here.  It is the part with its
own statistical apparatus -- a measured noise floor, an exact test against it,
and a control-free positional test that survives when the floor is not a valid
null -- and the reasoning behind each of those is what the length of this module
is.  :mod:`speedlm.gate.decide` re-exports every public name below, so this
split moved no import site.

Nothing here decides a promotion.  :func:`evaluate_divergence` returns the
evidence and the single rejection Boolean it implies; the gate sequences that
against its other guards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from speedlm.gate.replay import ReplayResult, RequestResult

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DecisionError(ValueError):
    """Raised when the decision cannot be computed."""



#: Significance level the whole output-correctness criterion is run at.
#:
#: **Why the criterion is a significance test at all.**  Speculative decoding is
#: lossless by construction: the verifier accepts a drafted token only when it
#: matches what the target model would itself have emitted, so *both* arms of
#: this gate -- stock draft and candidate draft alike -- are sampling the same
#: target distribution.  A better or worse draft head changes how many tokens
#: clear the verifier per forward pass, i.e. speed.  It cannot change the token
#: stream.  Any token-level disagreement between the two arms is therefore
#: evidence about the *engine*, not about the head: vLLM's target forward pass
#: is not bitwise reproducible across differing batch shapes, kernel configs are
#: selected by nearest-``M`` lookup, block sizes and split-``k`` branch on ``M``,
#: and at temperature 0 acceptance is exact argmax equality with no tolerance
#: band -- so a shifted reduction order flips near-ties.  ``VLLM_BATCH_INVARIANT``
#: exists precisely to force the invariant kernel subset and is off by default.
#:
#: The old criterion rejected on *any single* early divergence.  That is a
#: hair trigger on a stochastic process.  With an intrinsic per-context
#: divergence rate of just 1% and a 128-token correctness cap, the chance that
#: at least one of 410 contexts happens to flip inside the first 16 tokens is
#: ``1 - (1 - 0.01 * 16/128) ** 410`` = 40%.  At the rate actually measured on
#: Qwen3-8B (1.46%) it is 53%; on gpt-oss-20b, whose MoE path diverged on 56%
#: of contexts, it is indistinguishable from 1.  Every archived gpt-oss gate
#: rejected, and no stock-versus-stock control was ever run to say what any of
#: those numbers should have been compared against.
#:
#: **What replaces it.**  The gate now measures its own noise floor in the same
#: run -- the stock arm replays the correctness suite twice and the two passes
#: are compared against each other -- and rejects only when the
#: candidate-versus-stock divergence count *significantly exceeds that floor*
#: under a one-sided Fisher exact test.  Nothing about the floor is assumed, so
#: the criterion needs no per-model or per-engine tuning: a MoE target under
#: CUDA graphs at batch 64 measures its own large floor and is judged against
#: it, exactly as a dense target under eager at batch 1 measures its own small
#: one.  A genuinely broken head -- a mis-wired draft-to-target token map, a
#: vocabulary mismatch, corrupted weights -- diverges on essentially every
#: context at essentially the first token, which no engine noise floor reaches,
#: so detection is unaffected.
#:
#: 0.01 rather than the reflexive 0.05: this gate has no rollback behind it, so
#: a false *promote* is the expensive error, and a looser alpha buys detection
#: power cheaply.  It is not smaller than that because the exact test is
#: discrete -- at the five-context suites the simulation harness uses, a total
#: corruption attains ``1/C(10,5)`` = 0.0040, and an alpha below that would make
#: small suites structurally unable to reject anything at all.
DIVERGENCE_ALPHA: Final[float] = 0.01

#: Statistics the divergence criterion tests, and therefore the Bonferroni
#: divisor applied to :data:`DIVERGENCE_ALPHA`.
#:
#: Two: the *total* divergence count and the *early* subset.  Total has the most
#: events and so the most power against a head that is merely wrong more often.
#: Early targets the shape a genuine corruption takes -- disagreement at the
#: very first tokens -- and keeps power when the total rate is already saturated
#: by engine noise, as it is on a MoE target.  Both are tested against their own
#: separately measured floor, so neither can fire on noise.
DIVERGENCE_STATISTICS: Final[int] = 2


#: The level each divergence statistic is individually compared against, i.e.
#: :data:`DIVERGENCE_ALPHA` spread over :data:`DIVERGENCE_STATISTICS`.
#:
#: Named because it was previously spelled out at both of its use sites -- the
#: test in :func:`evaluate_divergence` and the ``divergence_alpha`` column of
#: the decision record that reports what that test used.  Those two must agree
#: by construction, not by inspection.
DIVERGENCE_PER_STATISTIC_ALPHA: Final[float] = DIVERGENCE_ALPHA / DIVERGENCE_STATISTICS


class DivergenceBasis(Enum):
    """What two generations were aligned on when they were compared.

    ``TOKEN`` is the model's own segmentation, recovered from the endpoint's
    per-token logprobs; an index is then a token offset.  ``CHARACTER`` is the
    fallback when logprobs were unavailable on either side, and an index is a
    character offset into the response text -- coarser, and systematically
    *larger* than the token offset it stands in for, so it is recorded rather
    than silently mixed in with token offsets.
    """

    TOKEN = "token"
    CHARACTER = "character"


@dataclass(frozen=True, slots=True)
class ContextDivergence:
    """Where one held-out context's two generations first parted.

    Persisted for every diverging context.  Before this existed the run
    directory kept only ``decision.json`` and ``gate-metrics/*.prom.gz``, so a
    rejection reading ``output_mismatches: 85`` was unfalsifiable after the
    fact: there was no way to tell a drafter that broke at token 3 from float
    noise at token 900.
    """

    context_hash: str
    repeat_index: int
    #: Offset of the first position at which the two generations differ, in
    #: units of ``basis``.  Always >= 0; a context that never diverges is not
    #: recorded at all.
    first_divergence_index: int
    basis: str
    stock_length: int
    candidate_length: int
    #: True when the divergence is early enough to gate against, i.e.
    #: ``first_divergence_index < promotion.min_divergence_token_index``.
    #:
    #: This flag classifies a divergence; on its own it no longer *disqualifies*
    #: one.  See :data:`DIVERGENCE_ALPHA` for what the gate now compares it
    #: against.
    early: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_hash": self.context_hash,
            "repeat_index": self.repeat_index,
            "first_divergence_index": self.first_divergence_index,
            "basis": self.basis,
            "stock_length": self.stock_length,
            "candidate_length": self.candidate_length,
            "early": self.early,
        }



def first_divergence(
    stock: RequestResult,
    candidate: RequestResult,
) -> tuple[int | None, DivergenceBasis, int, int]:
    """Locate where two generations of the same context first differ.

    This replaces whole-string equality, which asks the wrong question.  Two
    greedy generations of 1600 tokens that agree for 1500 of them are, by any
    behavioural standard, the same answer; two that part at token 3 are not.
    Equality collapses both to ``True`` and gates on the collapse.

    Alignment prefers the model's own tokenisation (``output_tokens``, from the
    endpoint's per-token logprobs) and falls back to characters when either
    side did not capture it.  The basis is returned rather than assumed,
    because a character offset is not comparable to a token offset.

    The character fallback aligns on ``generated_text``, not ``response_text``.
    A reasoning model bounded at ``correctness_max_tokens`` routinely never
    closes its ``<think>`` block, which leaves ``response_text`` empty on
    *both* arms; comparing those two empty strings returns ``None`` --
    "identical" -- having compared nothing at all.  Folding the reasoning
    channel in means the fallback compares the text that was actually
    generated.

    Returns:
        ``(index, basis, stock_length, candidate_length)`` where *index* is the
        first differing position, or ``None`` when one sequence is a prefix of
        the other *and* they are the same length -- i.e. they are identical.
        A shorter-but-otherwise-identical generation diverges at the end of the
        shorter sequence, because stopping early is itself a difference.
    """
    if stock.output_tokens and candidate.output_tokens:
        basis = DivergenceBasis.TOKEN
        left: Sequence[str] = stock.output_tokens
        right: Sequence[str] = candidate.output_tokens
    else:
        basis = DivergenceBasis.CHARACTER
        left = stock.generated_text
        right = candidate.generated_text

    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index, basis, len(left), len(right)
    if len(left) != len(right):
        return min(len(left), len(right)), basis, len(left), len(right)
    return None, basis, len(left), len(right)


def _collect_divergences(
    stock: ReplayResult,
    candidate: ReplayResult,
    *,
    min_divergence_index: int,
) -> tuple[tuple[ContextDivergence, ...], int]:
    """Locate, classify and record every context whose generations parted.

    Returns ``(divergences, trials)``.  *trials* -- the number of context pairs
    actually compared -- is returned rather than re-derived by the caller
    because it is the denominator of every rate the criterion computes, and a
    count of events without the count of opportunities is not a rate.
    """
    found: list[ContextDivergence] = []
    trials = 0
    for repeat_index in range(min(stock.num_runs, candidate.num_runs)):
        s_run = stock.run_results[repeat_index]
        c_run = candidate.run_results[repeat_index]
        for s_req, c_req in zip(s_run.results, c_run.results, strict=True):
            trials += 1
            index, basis, s_len, c_len = first_divergence(s_req, c_req)
            if index is None:
                continue
            found.append(
                ContextDivergence(
                    context_hash=s_req.context_hash,
                    repeat_index=repeat_index,
                    first_divergence_index=index,
                    basis=basis.value,
                    stock_length=s_len,
                    candidate_length=c_len,
                    early=index < min_divergence_index,
                )
            )
    return tuple(found), trials


def divergence_excess_p_value(
    observed_events: int,
    observed_trials: int,
    control_events: int,
    control_trials: int,
) -> float:
    """One-sided Fisher exact p-value for "the candidate diverges *more*".

    The null hypothesis is the one speculative decoding's losslessness makes the
    right one: both comparisons -- candidate-versus-stock and stock-versus-stock
    -- are draws from the *same* per-context divergence hazard, namely the
    engine's.  The alternative is that the candidate arm's hazard is strictly
    larger, which is the only thing a broken head can produce.

    Conditioning on the observed event total, the number landing in the
    candidate comparison is hypergeometric, so the upper tail is exact -- no
    normal approximation, no continuity correction, and no minimum expected-cell
    count to violate.  That matters here because the interesting regimes are
    both extremes at once: a handful of events out of 410 on a dense target, and
    a near-saturated 230 out of 410 on a MoE one.

    Returns 1.0 when nothing diverged anywhere, or when there were no trials --
    "no evidence", never "significant".
    """
    for name, value in (
        ("observed_events", observed_events),
        ("observed_trials", observed_trials),
        ("control_events", control_events),
        ("control_trials", control_trials),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DecisionError(f"{name} must be a non-negative integer, got {value!r}")
    if observed_events > observed_trials or control_events > control_trials:
        raise DecisionError(
            "divergence events cannot exceed trials: "
            f"{observed_events}/{observed_trials} observed, "
            f"{control_events}/{control_trials} control"
        )

    total_events = observed_events + control_events
    total_trials = observed_trials + control_trials
    if total_events == 0 or total_trials == 0:
        return 1.0
    denominator = math.comb(total_trials, total_events)
    numerator = sum(
        math.comb(observed_trials, i) * math.comb(control_trials, total_events - i)
        for i in range(observed_events, min(observed_trials, total_events) + 1)
    )
    return min(1.0, numerator / denominator)


def _binomial_upper_tail(events: int, trials: int, rate: float) -> float:
    """``P(X >= events)`` for ``X ~ Binomial(trials, rate)``, exactly.

    Summed in log space because the gate's denominators run to hundreds of
    trials, where ``rate ** k`` underflows to zero long before the binomial
    coefficient it is multiplied by overflows a float.
    """
    if events <= 0:
        return 1.0
    if events > trials:
        return 0.0
    if rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return 1.0
    log_p = math.log(rate)
    log_q = math.log1p(-rate)
    log_n_fact = math.lgamma(trials + 1)
    terms = [
        math.exp(
            log_n_fact
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * log_p
            + (trials - k) * log_q
        )
        for k in range(events, trials + 1)
    ]
    return min(1.0, math.fsum(terms))


def divergence_position_p_value(
    early_events: int,
    total_events: int,
    trials: int,
    *,
    min_divergence_index: int,
    max_tokens: int,
    control_early_rate: float = 0.0,
) -> float:
    """One-sided p-value for "these divergences are *front-loaded*".

    This is the divergence statistic that does not need a control, and it is
    the one that survives the discovery that the gate's control is not a valid
    null for the comparison it is used against (see
    :func:`decide_promotion`).

    The null is the one greedy speculative decoding actually licenses.
    Verification accepts a drafted token only on exact ``argmax`` equality with
    the verifier -- no tolerance band, no probability, no RNG -- so a *sound*
    engine emits the verifier's own greedy trajectory whatever the draft head
    proposes.  The two arms can therefore only part where floating-point noise
    flips an ``argmax`` at a near-tie, and near-ties are not concentrated at any
    particular offset: the per-token flip hazard is flat.  Under a flat hazard
    the first-divergence offset is memoryless, and the share of contexts parting
    inside the first ``min_divergence_index`` tokens is pinned by the share
    parting over the remaining window.

    The alternative is what a genuinely broken head produces.  If verification
    is bypassed, or a tolerance band is opened, or the served head is not the
    head that was measured, the emitted text stops being the verifier's
    trajectory *immediately* -- the divergences pile up at the front of the
    window and the flat-hazard null is violated by orders of magnitude.

    The hazard is calibrated from this run's own late window rather than
    assumed, which is what makes the statistic valid across models: an engine
    whose arms part in 2% of contexts and one whose arms part in 51% are
    scored against their own hazards, not against a shared constant.

    Args:
        early_events: Contexts that parted before ``min_divergence_index``.
        total_events: Contexts that parted at any offset.
        trials: Context comparisons attempted -- the denominator.
        min_divergence_index: Offset separating "early" from "late".
        max_tokens: Output cap of the correctness pass, i.e. the width of the
            window in which a divergence could have been seen at all.
        control_early_rate: Early divergences per trial the *engine* produced
            replaying stock against stock, when a control ran.  It raises the
            null rate and can only ever make this test harder to reject.

            That asymmetry is the point.  The gate's control is collected
            inside a single engine incarnation while the measurement straddles
            a restart, so it is a *lower bound* on the true floor -- which
            makes it unusable as evidence that the candidate is at fault, and
            perfectly usable as evidence that it is not.  Letting a control
            exonerate but never condemn is what keeps a floor this design
            cannot measure from being read as though it had.

    Returns:
        ``P(early >= early_events)`` under the flat-hazard null.  ``1.0``
        whenever the run carries no usable evidence -- no divergences, no
        trials, a window the threshold saturates, or a late window that
        produced nothing to calibrate against.  "No evidence" is never
        "significant".
    """
    for name, value in (
        ("early_events", early_events),
        ("total_events", total_events),
        ("trials", trials),
        ("min_divergence_index", min_divergence_index),
        ("max_tokens", max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DecisionError(f"{name} must be a non-negative integer, got {value!r}")
    if not 0.0 <= control_early_rate <= 1.0:
        raise DecisionError(
            f"control_early_rate must lie in [0, 1], got {control_early_rate!r}"
        )
    if early_events > total_events:
        raise DecisionError(
            f"early divergences cannot exceed the total: {early_events} > {total_events}"
        )
    if total_events > trials:
        raise DecisionError(
            f"divergences cannot exceed trials: {total_events} > {trials}"
        )
    if early_events == 0 or trials == 0:
        return 1.0
    # A threshold at or beyond the cap makes every divergence "early" by
    # construction; there is no late window left to calibrate against and no
    # contrast left to test.  ``divergence_criterion`` already names this
    # configuration; here it simply carries no evidence.
    if min_divergence_index >= max_tokens:
        return 1.0

    at_risk = trials - early_events
    late_events = total_events - early_events
    if at_risk <= 0 or late_events <= 0:
        # Nothing survived to the late window, or nothing parted in it.  Fall
        # back to the small-hazard limit of the same null, where a flat hazard
        # puts exactly ``min_divergence_index / max_tokens`` of its mass early.
        # This is the branch a fully bypassed verifier lands in, and it must
        # still be able to reject.
        null_rate = min_divergence_index / max_tokens
    elif late_events >= at_risk:
        # Every context still alive at the threshold parted afterwards, so the
        # calibrated hazard is unbounded and no early count is surprising.
        return 1.0
    else:
        late_survival = 1.0 - late_events / at_risk
        per_token_survival = late_survival ** (
            1.0 / (max_tokens - min_divergence_index)
        )
        null_rate = 1.0 - per_token_survival**min_divergence_index
    return _binomial_upper_tail(
        early_events, trials, max(null_rate, control_early_rate)
    )


@dataclass(frozen=True, slots=True)
class DivergenceEvidence:
    """Everything the output-correctness criterion establishes about one run.

    Assembled by :func:`evaluate_divergence` and consumed whole by
    :func:`decide_promotion`, which reads it as one verdict
    (:attr:`rejects`) plus a block of fields that land verbatim on the
    :class:`Decision`.  Nothing here is re-derived downstream: the counts, the
    p-values and the rejection they imply are computed once, together, from the
    same comparisons.
    """

    divergences: tuple[ContextDivergence, ...]
    trials: int
    control_divergences: tuple[ContextDivergence, ...]
    control_trials: int
    control_available: bool
    control_comparable: bool
    #: Correctness passes the evidence rests on -- ``min(num_runs)`` over the
    #: pair actually compared.
    correctness_repeats: int
    #: Early divergences keyed by the correctness repeat they were found in.
    #: This is the ``output_mismatches`` column of ``per_repeat``.
    early_by_repeat: dict[int, int]
    total_early: int
    total_p: float
    early_p: float
    position_p: float
    #: Whether any gating channel fired.  See :func:`evaluate_divergence`.
    rejects: bool


def evaluate_divergence(
    stock_replay: ReplayResult,
    candidate_replay: ReplayResult,
    *,
    stock_correctness: ReplayResult | None,
    candidate_correctness: ReplayResult | None,
    stock_correctness_control: ReplayResult | None,
    min_divergence_index: int,
    correctness_max_tokens: int | None,
    max_output_mismatches: int,
) -> DivergenceEvidence:
    """Run the whole output-correctness criterion over one gate's replays.

    Compared on its own replay when the caller ran one, so that the question
    "do these two heads produce the same answer" is asked of a short,
    single-stream, bounded generation rather than of the batched throughput
    pass, whose whole purpose is to vary batch composition.
    """

    stock_corr = stock_correctness if stock_correctness is not None else stock_replay
    candidate_corr = (
        candidate_correctness if candidate_correctness is not None else candidate_replay
    )
    divergences, divergence_trials = _collect_divergences(
        stock_corr,
        candidate_corr,
        min_divergence_index=min_divergence_index,
    )
    # The engine's own noise floor: the stock arm against a second pass of
    # itself, scored by the identical procedure so the two counts are
    # commensurable.  Absent one, assume a floor of zero over an equal number of
    # trials -- the strictest assumption that is still a test.
    if stock_correctness_control is not None:
        control_divergences, control_trials = _collect_divergences(
            stock_corr,
            stock_correctness_control,
            min_divergence_index=min_divergence_index,
        )
        control_available = True
    else:
        control_divergences = ()
        control_trials = divergence_trials
        control_available = False

    # Is that floor a valid null for the comparison it is used against?
    #
    # The measured pair is stock-arm-versus-candidate-arm, and those two arms
    # are served by two different engine incarnations -- the runner restarts
    # vLLM to change the draft head -- so the pair straddles a weight reload, a
    # KV and prefix cache rebuild, and a fresh kernel-autotune state.  The
    # control pair is stock-versus-stock inside a *single* incarnation, which
    # replays a bit-identical computation and therefore reports zero however
    # noisy the engine is across restarts.  Testing a cross-incarnation count
    # against a same-incarnation floor of zero is not a test: it rejects on the
    # first divergence, whatever produced it.
    #
    # ``ReplayResult.session_id`` is what makes the topology visible.  Distinct
    # non-empty ids mean two different replay invocations; an empty id is a
    # result that carries no claim about how it was collected and must not be
    # read as agreement.  The control is comparable only when the measured pair
    # spans invocations *and* the control pair spans invocations too.
    control_session = (
        stock_correctness_control.session_id if stock_correctness_control else ""
    )
    control_comparable = bool(
        control_available
        and stock_corr.session_id
        and candidate_corr.session_id
        and control_session
        and stock_corr.session_id != candidate_corr.session_id
        and stock_corr.session_id != control_session
    )

    # The number of passes the divergence evidence above actually rests on.
    # ``_collect_divergences`` compares ``min(num_runs)`` pairs, so that is the
    # count -- and it is what bounds which ``per_repeat`` rows can carry a
    # non-zero ``output_mismatches``.
    correctness_repeats = min(stock_corr.num_runs, candidate_corr.num_runs)
    early_by_repeat: dict[int, int] = {}
    for d in divergences:
        if d.early:
            early_by_repeat[d.repeat_index] = early_by_repeat.get(d.repeat_index, 0) + 1
    total_early = sum(early_by_repeat.values())
    total_any = len(divergences)
    control_early = sum(1 for d in control_divergences if d.early)
    control_any = len(control_divergences)

    # Each statistic against its own floor, at
    # :data:`DIVERGENCE_PER_STATISTIC_ALPHA` -- Bonferroni over the two, because
    # rejecting on "either is significant" tests twice.
    total_p = divergence_excess_p_value(
        total_any, divergence_trials, control_any, control_trials
    )
    early_p = divergence_excess_p_value(
        total_early, divergence_trials, control_early, control_trials
    )
    # The control-free channel.  It asks whether the divergences sit where a
    # sound engine's floating-point noise would put them, and it needs no floor
    # to do it -- see :func:`divergence_position_p_value`.
    #
    # The window is the correctness pass's own cap when the caller declared one
    # -- that is the pair ``divergence_criterion`` is classified from, so the
    # two readings stay consistent.  A caller that declared none still leaves
    # the window observable: no comparison could have found a divergence past
    # the longest generation it actually compared.
    divergence_window = correctness_max_tokens or max(
        (max(d.stock_length, d.candidate_length) for d in divergences),
        default=0,
    )
    position_p = divergence_position_p_value(
        total_early,
        total_any,
        divergence_trials,
        min_divergence_index=min_divergence_index,
        max_tokens=divergence_window,
        control_early_rate=(
            control_early / control_trials if control_trials > 0 else 0.0
        ),
    )

    # The allowance is a precondition, not an alternative: a statistic fires
    # only when it clears *both* the caller's outright tolerance and its own
    # null.
    #
    # The position channel always gates.  Its null holds for any model and any
    # inference configuration -- a perfectly deterministic engine and a wildly
    # nondeterministic one are each scored against their own measured hazard --
    # and it is the channel a genuinely broken head trips, because bypassed or
    # loosened verification moves the divergences to the front of the window.
    #
    # The two rate channels gate only against a comparable control.  Without
    # one their null is a floor of zero collected under strictly easier
    # conditions than the measurement, and a test against that floor rejects
    # any nonzero count -- including the count that draft-independent
    # floating-point noise produces on every engine that is not batch-invariant.
    # They stay computed and recorded either way, so the evidence survives even
    # where it does not decide.
    position_rejects = (
        total_early > max_output_mismatches and position_p < DIVERGENCE_PER_STATISTIC_ALPHA
    )
    # Unanimity needs no distributional null.  A floor is a rate and a rate
    # cannot exceed one, so "the candidate parted on every context while the
    # engine's own control did not" is an excess statement that stands on the
    # two rates alone -- no hazard model, no significance test, no assumption
    # about how the two passes were collected.  It is also the observation the
    # position channel cannot reach: where the threshold saturates the window
    # every divergence is early by construction, leaving the flat-hazard null
    # with no late window to calibrate against.  So this gates whether or not
    # the control is comparable -- but it still defers to a control that
    # reproduced the same unanimity, which is the engine speaking, not the head.
    unanimous_rejects = (
        divergence_trials > 0
        and total_any == divergence_trials
        and total_any > max_output_mismatches
        and control_any < control_trials
    )
    rate_rejects = control_comparable and (
        (total_any > max_output_mismatches and total_p < DIVERGENCE_PER_STATISTIC_ALPHA)
        or (total_early > max_output_mismatches and early_p < DIVERGENCE_PER_STATISTIC_ALPHA)
    )

    return DivergenceEvidence(
        divergences=divergences,
        trials=divergence_trials,
        control_divergences=control_divergences,
        control_trials=control_trials,
        control_available=control_available,
        control_comparable=control_comparable,
        correctness_repeats=correctness_repeats,
        early_by_repeat=early_by_repeat,
        total_early=total_early,
        total_p=total_p,
        early_p=early_p,
        position_p=position_p,
        rejects=position_rejects or unanimous_rejects or rate_rejects,
    )
