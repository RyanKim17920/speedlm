"""Concrete orchestration for the stock-versus-candidate benchmark gate.

The runner owns no vLLM process and performs no direct metrics I/O.  Process
switching, metrics scraping, and replay execution are injected behind small
protocols so the orchestration can be exercised without a GPU or network.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.decide import (
    UNRECORDED_EXECUTION_MODE,
    VETO_REASON_NON_STATIONARY,
    Decision,
    DivergenceSampling,
    EngineExecution,
    MeasurementBlock,
    ThroughputStationarity,
    Verdict,
    decide_promotion,
)
from speedlm.gate.metrics import (
    CounterResetError,
    MetricsDelta,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)
from speedlm.gate.replay import ReplayResult, RunResults, replay_suite
from speedlm.gate.suite import (
    BenchmarkSuite,
    SuiteError,
    build_suite,
    load_suite,
    persist_suite,
)
from speedlm.traces.store import TraceRecord

if TYPE_CHECKING:
    from speedlm.tuner.orchestrator import GateFailure, GateResult

AbortCheck = Callable[[], bool]
DraftReference = Path | str
Clock = Callable[[], float]
TrainingHashes = (
    set[str]
    | frozenset[str]
    | Callable[[], set[str] | frozenset[str]]
)
SuiteDirectory = Path | Callable[[], Path]
#: The stock arm's draft, or a callable that resolves it when the gate runs.
#:
#: The callable form exists because the stock arm's identity is not a property
#: of the process, it is a property of the moment: every promotion replaces the
#: draft that serves live traffic, and the arm that names itself "stock" has to
#: name *that* draft or the delta it reports is not marginal improvement over
#: the incumbent.  A bare reference stays supported, and is what a test that
#: benchmarks one fixed pair of drafts should pass.
StockDraft = DraftReference | Callable[[], DraftReference]
_T = TypeVar("_T")

#: Held-out requests kept in flight per arm while a suite pass runs.
#:
#: Until this existed the replay issued one request at a time -- vLLM logged
#: ``Running: 1 reqs`` for the entire benchmark on job 368959, and a single
#: 103-context warmup pass took over 1720s, which is what exhausted that run's
#: benchmark deadline before a measurement was ever taken.  The served engine
#: is not the constraint: the gate replays against the managed ``vllm serve``
#: child built by :func:`speedlm.gateway.process.build_vllm_argv`, which never
#: sets ``--max-num-seqs``, so the engine runs vLLM's default scheduler width
#: and can absorb many more than one in-flight request.  (The
#: ``max_num_seqs=1`` in :mod:`speedlm.training.backends.eagle3` belongs to the
#: offline hidden-state extraction engine, which is not on the gate's serving
#: path.)
#:
#: Eight matches
#: :attr:`speedlm.config.IdleTuningConfig.extraction_concurrency`, the degree
#: the codebase already uses when driving this same engine family.  It is a
#: coincidence of value, not a dependency: that field configures the training
#: side's offline extraction engine and is never read here.
#:
#: The gating statistic -- see
#: :data:`speedlm.gate.decide.GATING_THROUGHPUT_STATISTIC` -- divides completion
#: tokens by the *sum* of per-request latencies, so its absolute value depends
#: on how much batching the engine is doing.  Both arms are always replayed at
#: the same degree, which is what the arm-to-arm delta needs; the consequence
#: is that absolute tok/s figures are only comparable across runs that used the
#: same concurrency, which is why the value is recorded in the gate's metrics.
DEFAULT_REPLAY_CONCURRENCY: Final = 8

#: In-flight requests during the gate's output-correctness pass.
#:
#: Pinned to one, and deliberately *not* derived from
#: ``tuning.benchmark_concurrency``.  Bitwise agreement between two greedy
#: generations is a property of the batch composition they were generated
#: under: vLLM's target forward pass is not bitwise reproducible across
#: differing batch shapes, and commit cbaff80 measured the per-token divergence
#: hazard rising roughly 12x when replay concurrency went from 1 to 8.  A
#: correctness check run at the throughput pass's concurrency is therefore
#: measuring the scheduler, not the drafter.  The throughput pass keeps its
#: concurrency; the two jobs are simply no longer done by the same pass.
CORRECTNESS_REPLAY_CONCURRENCY: Final = 1

#: Output cap for the correctness pass when the caller supplies none.  Mirrors
#: :attr:`speedlm.config.IdleTuningConfig.correctness_max_tokens`, which
#: carries the justification.
DEFAULT_CORRECTNESS_MAX_TOKENS: Final = 128

#: Output cap for the throughput/acceptance pass when the caller supplies none.
#: Mirrors :attr:`speedlm.config.IdleTuningConfig.benchmark_max_tokens`, which
#: carries the justification and the acceptance-bias disclosure.
DEFAULT_BENCHMARK_MAX_TOKENS: Final = 512

#: Suite passes the correctness check scores per arm.
#:
#: One.  Raising it does not fix the false-reject problem it looks like it
#: should: every extra pass is another fresh draw against the engine's
#: divergence hazard, so under an any-occurrence rule more repeats strictly
#: *increase* the false-reject rate.  What fixes it is knowing what the hazard
#: is, which is what :data:`CORRECTNESS_CONTROL_REPEATS` buys.
CORRECTNESS_REPEATS: Final = 1

#: Extra correctness passes the *stock* arm makes, to measure the engine's own
#: divergence rate against itself.
#:
#: One, and it is the whole fix.  Speculative decoding is lossless -- the
#: verifier only accepts a drafted token the target would have emitted anyway --
#: so both arms sample the same distribution and a token-level disagreement
#: between them says nothing about the draft head.  It says the engine is not
#: bitwise reproducible, which on vLLM it is not: kernel configs are chosen by
#: nearest-``M`` lookup, block sizes and split-``k`` branch on ``M``, and greedy
#: acceptance is exact argmax equality with no tolerance band, so a shifted
#: reduction order flips near-ties.
#:
#: Until this existed the gate had no idea what rate that produced.  It never
#: once replayed one arm against itself, so ``output_mismatch`` was a count with
#: no denominator: every archived gpt-oss run diverged on roughly half its
#: contexts and every archived Qwen run on a few percent, and there was no way
#: to tell either apart from a broken head.  Running the stock arm's correctness
#: pass twice on the same engine, same draft, same sampling, same cap and same
#: concurrency makes every divergence between those two passes engine noise *by
#: construction*, and gives :func:`speedlm.gate.decide.decide_promotion` a floor
#: measured under the exact conditions the candidate is judged under.
#:
#: The cost is one bounded, single-stream suite pass per gate -- the correctness
#: cap is an order of magnitude below the throughput cap, so this is a small
#: fraction of a cycle.
#:
#: It is deliberately the *stock* arm that repeats: it is the incumbent, it is
#: the reference side of every comparison, and it exists on every run including
#: the ones where the candidate engine fails to come up.  The floor it measures
#: is same-process, so it does not capture whatever additional variation comes
#: from the two arms running in separate engine processes with separately
#: autotuned kernel caches; that residue is unmeasured and makes the floor a
#: lower bound, i.e. the criterion errs toward rejecting rather than promoting.
CORRECTNESS_CONTROL_REPEATS: Final = 1

#: Contiguous measurement blocks each arm's scored repeats are split into.
#:
#: One block per arm means "run every stock repeat, then every candidate
#: repeat", which is what this gate did until job badba71 showed what that
#: costs.  In that run the stock arm's replay throughput read 96.7, 96.9, 96.3,
#: 105.4, 104.8 tok/s -- a step of roughly 9% between its third and fourth
#: scored repeat, holding to the end of the job -- while the candidate arm,
#: which had already finished, was flat at 98.3, 98.3, 97.7, 97.9, 97.8.  The
#: engine did bit-identical work either side of the step (the same 12356 engine
#: iterations and 206324 generation tokens as its own first repeat, the same
#: acceptance, prefill time down by the same 8.7% as decode time), so nothing
#: about either draft changed; the machine got faster and stayed faster.  The
#: gate nonetheless attributed all of it to stock, and the arm means it
#: published (100.0 vs 98.0) have the *opposite sign* to the three repeats where
#: both arms saw the same machine (96.65 vs 98.10).  With the arms run one after
#: the other there is no measurement that can tell those apart: arm identity and
#: wall-clock are the same axis.
#:
#: Two blocks per arm makes them different axes.  The blocks are issued in
#: mirrored rounds -- ``A B`` then ``B A``, i.e. the ABBA design -- so the two
#: arms occupy the same average position in time and any drift that is linear
#: in time cancels to first order instead of landing on one arm.  A *step* like
#: badba71's does not cancel, but it stops being invisible: whichever arm
#: straddles it now shows the step inside its own per-repeat column, which is
#: exactly what ``*_throughput_flat_from_repeat`` measures and what
#: :data:`PROMOTION_REQUIRES_STATIONARITY` refuses to promote on.  It also makes
#: repeat ``i`` of one arm roughly contemporaneous with repeat ``i`` of the
#: other, which is what :class:`speedlm.gate.decide.Decision`'s per-repeat rows
#: have always been shaped like and have never until now actually been.
#:
#: **The cost is real and is paid in engine activations.**  Every block boundary
#: is a draft change, and a draft change restarts vLLM: 2 blocks per arm is 4
#: activations where 1 block was 2, and because each block must open its
#: measurement window warm (see ``warmup_repeats``) it is also 4 warmup passes
#: where 1 block was 2.  Measured on job badba71 (Qwen3-8B): a restart is ~100s
#: and a suite pass ~174s, so +2 restarts and +2 warmups is ~548s on a 1043s
#: benchmark phase.  On gpt-oss-20b (job 369162) a restart is ~90s and a pass
#: ~301s, so ~782s on 1808s.  Requiring block 0 to restart adds one activation
#: when the configured first arm is already serving -- production's
#: candidate-first case after CANDIDATE_STARTING: about 100s on Qwen and 90s on
#: gpt-oss from those measurements.  There is one
#: further restart on top: mirrored rounds end on whichever arm *started*, so with
#: ``benchmark_candidate_arm_first`` the benchmark now ends on the candidate and
#: a rejecting cycle pays the rollback restart that one-block candidate-first
#: order avoided (~100s).  Call it +50-60% of the benchmark phase.
#:
#: That is the price of the measurement answering its question at all.  The
#: alternative on the evidence above is not a cheaper answer, it is a number
#: whose sign is set by whether the machine happened to change speed during the
#: second arm.  Set to 1 to restore the sequential design.
DEFAULT_ARM_BLOCKS: Final = 1

#: Blocks per arm the composed production gate runs; see
#: :data:`DEFAULT_ARM_BLOCKS` for why it is not 1 and what it costs.  Two is the
#: smallest number that balances the arms in time; more blocks buy no additional
#: balance and cost two more activations each.
INTERLEAVED_ARM_BLOCKS: Final = 2

#: Whether a drifting throughput measurement may promote.
#:
#: It may not.  A promotion is irreversible here -- the gate is the only
#: safeguard and nothing behind it rolls back -- so the question a promote has
#: to answer is not "was the delta positive" but "was the delta measured".  On
#: job badba71 the arm-to-arm delta was -2.03% against a standard error of
#: 2.08%, and the whole of the apparent stock advantage was one arm's mid-run
#: step; a gate that treated that as a measurement would have been deciding on
#: the machine's clock, and would have got the *sign* wrong.
#:
#: The condition is that both arms' ``*_throughput_flat_from_repeat`` is ``0``:
#: every scored repeat of both arms lies inside a window whose own trend is
#: below its own residual noise (see
#: :data:`speedlm.gate.decide.FLAT_TREND_T_STATISTIC`).  ``k > 0`` means repeats
#: ``0..k-1`` were still moving and are nonetheless inside the reported mean;
#: ``None`` means the arm never settled inside the window at all, which is what
#: badba71's stock arm recorded.  Both are answered by raising
#: ``tuning.warmup_repeats`` or ``benchmark_repeats``, which is why the veto
#: names the index rather than just refusing.
#:
#: This deliberately errs toward rejecting: a stationarity veto can only turn a
#: promote into a no-op cycle, never a reject into a promote.  It is enforced
#: here rather than in :func:`speedlm.gate.decide.decide_promotion` because the
#: stationarity of a measurement is a property of how the measurement was taken,
#: which is this module's job; ``decide`` keeps reporting what the numbers say
#: and the record keeps both answers.
PROMOTION_REQUIRES_STATIONARITY: Final = True

#: ``GateResult.reason`` when the numbers cleared every threshold but the
#: measurement they came from was still drifting.  Alias of
#: :data:`speedlm.gate.decide.VETO_REASON_NON_STATIONARY`, which the persisted
#: decision uses for the same fact: the two must be one string, or the record
#: and the cycle log can name the same veto differently.
NON_STATIONARY_REASON: Final = VETO_REASON_NON_STATIONARY

#: Sampling the correctness/divergence pass is pinned to, whatever
#: ``config.sampling`` serves under.
#:
#: The output-correctness criterion compares the two arms for exact equality,
#: and its nulls hold only where speculative decoding is output-preserving --
#: greedy verification, where acceptance is exact ``argmax`` equality with no
#: tolerance band and no RNG.  ``config.sampling`` exists to mirror production
#: traffic and may be anything; inheriting it here would let the gate publish a
#: p-value measuring the sampler.  See :class:`speedlm.gate.decide.DivergenceSampling`.
#:
#: Forcing greedy rather than refusing to run keeps the criterion alive: it is
#: the gate's only guard against a mis-wired or bypassed verifier, and a broken
#: head is broken at temperature 0 too.  The seed is carried through unchanged
#: -- it is not a stochasticity knob under greedy decoding, and pinning it
#: would silently change the prefix-cache behaviour a run was configured for.
DIVERGENCE_TEMPERATURE: Final = 0.0
DIVERGENCE_TOP_P: Final = 1.0

#: ``|shift| / SE(shift)`` above which the arm-to-arm delta is held to have
#: moved during the run rather than jittered.
#:
#: The statistic is a two-sample t on the *paired* per-repeat throughput delta,
#: maximised over every way of splitting the repeat sequence into a leading and
#: a trailing part -- see :func:`_delta_shift`.  Pairing is what makes it the
#: right question under an interleaved schedule: drift the two arms saw alike is
#: common-mode and cancels out of the paired column, so what is left is drift
#: that landed on one arm and not the other, which is precisely the error that
#: makes an arm-to-arm delta meaningless.
#:
#: Four, calibrated from both directions on real data.  Job badba71 -- the run
#: this whole design is a response to -- reads ``t = 50.2`` on a shift of
#: 8.40 pp, so the case that must be caught is caught by a factor of twelve.
#: Against that, a Monte Carlo over stationary Gaussian repeat columns at the
#: per-repeat dispersion the clean archived runs actually show (jobs
#: 369161/369162 report delta standard errors of 0.607% and 0.640% over five
#: repeats, i.e. a per-repeat sd of about 1.4 pp) puts the false-veto rate at
#: 7.5%, falling to 0.3% at 0.6 pp.  A false veto costs one deferred promotion;
#: the failure it prevents is an irreversible one.
#:
#: A least-squares *trend* was tried first and rejected on the same data: it
#: assumes the drift is a ramp, and badba71's was a step, which leaves residuals
#: so large that the trend reads ``t = 3.0`` -- indistinguishable from noise on
#: any bar that clean runs also clear.  Maximising a split statistic instead
#: costs nothing on a genuine ramp (some split still separates its ends) and
#: does not blind the gate to the shape the hardware actually produced.
STATIONARITY_SPLIT_T_STATISTIC: Final = 4.0

#: Scored repeats below which stationarity is not tested at all.
#:
#: Four, because the split statistic pools ``n - 2`` residual degrees of freedom
#: and at ``n = 3`` that is one: the estimate of the noise it divides by is
#: itself a single number, and the test fires or does not fire on which side of
#: a coin the one residual landed.  Measured on this repo's own simulation
#: harness -- three repeats against an engine whose per-repeat throughput jitters
#: by 15-30% because each request is a 4 ms sleep -- the paired column produced
#: ``t`` of 2.2 and 3.2 on runs that differ only in scheduling noise.  A veto
#: driven by that is not a safeguard.
#:
#: Below this the record says ``testable: false`` and the promote is allowed to
#: stand on the thresholds alone, which is what the gate did before this existed.
#: Production runs five (``IdleTuningConfig.benchmark_repeats``, which has its
#: own argument for why it must not be cut), so the veto is live where it
#: matters; a caller that configures three is asking for a measurement too short
#: to check.
MIN_STATIONARITY_REPEATS: Final = 4


def _delta_shift(values: Sequence[float]) -> tuple[float, float]:
    """Largest resolvable step in a per-repeat column: ``(shift, |t|)``.

    Splits the column at every interior point into a leading and a trailing
    part, and returns the split whose pooled two-sample ``t`` is largest,
    together with the difference of means at that split (leading minus
    trailing), in the column's own units.

    Reported as a pair because neither number alone is a decision: a shift with
    no ``t`` behind it is noise, and a ``t`` on a shift too small to matter is a
    tightly-measured irrelevance.  See
    :data:`STATIONARITY_SPLIT_T_STATISTIC` for how the two are used together.
    """
    n = len(values)
    best = (0.0, 0.0)
    for split in range(1, n):
        leading, trailing = values[:split], values[split:]
        mean_leading = sum(leading) / len(leading)
        mean_trailing = sum(trailing) / len(trailing)
        shift = mean_leading - mean_trailing
        residual = sum((v - mean_leading) ** 2 for v in leading) + sum(
            (v - mean_trailing) ** 2 for v in trailing
        )
        pooled = residual / (n - 2)
        if pooled <= 0.0:
            # No residual to studentise against.  A column that never moved is
            # flat; one that moved with zero within-part spread is a clean step.
            statistic = 0.0 if shift == 0.0 else float("inf")
        else:
            statistic = abs(shift) / math.sqrt(
                pooled * (1.0 / len(leading) + 1.0 / len(trailing))
            )
        if statistic > best[1]:
            best = (shift, statistic)
    return best


def _block_schedule(first_arm: str, *, blocks: int, repeats: int) -> tuple[
    tuple[str, int], ...
]:
    """Order the two arms' measurement blocks so drift is common-mode.

    Returns ``(arm, scored_repeats)`` pairs in the order they are to be run.
    ``blocks == 1`` reproduces the sequential design exactly: the whole of
    *first_arm*, then the whole of the other.

    Above one, blocks are issued in mirrored rounds -- round 0 is
    ``(first, other)``, round 1 is ``(other, first)``, and so on -- which is the
    ABBA design.  With an even number of blocks and an even repeat count the two
    arms have identical average positions in the pass sequence, so a drift that
    is linear in time contributes identically to both arms and cancels out of
    the delta.  With an odd repeat count it very nearly does: at
    ``repeats=5, blocks=2`` the block sizes are 3 and 2, giving pass-index
    centroids of 4.0 and 5.0 against 2.0 and 7.0 for the sequential design --
    the arms' separation in time falls from 5 pass-indices to 1, so the residual
    exposure to a linear drift is a fifth of what it was.

    Raises:
        ValueError: If *blocks* is not a positive integer, or exceeds *repeats*
            so that some block would score nothing.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("arm_blocks must be an integer >= 1")
    if blocks > repeats:
        raise ValueError(
            f"arm_blocks ({blocks}) cannot exceed repeats ({repeats}): "
            "every block must score at least one repeat"
        )
    other = "stock" if first_arm == "candidate" else "candidate"
    base, remainder = divmod(repeats, blocks)
    sizes = [base + (1 if i < remainder else 0) for i in range(blocks)]
    schedule: list[tuple[str, int]] = []
    for round_index, size in enumerate(sizes):
        order = (first_arm, other) if round_index % 2 == 0 else (other, first_arm)
        schedule.extend((arm, size) for arm in order)
    return tuple(schedule)


def _validated_concurrency(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


class TraceSource(Protocol):
    """Readable trace-store surface needed to freeze a suite."""

    def iter_records(self) -> Iterator[TraceRecord]: ...


class DraftEndpoint(Protocol):
    """Control surface for selecting the draft served by one endpoint."""

    @property
    def url(self) -> str: ...

    def activate(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
        allow_engine_reuse: bool = True,
    ) -> bool:
        """Make *draft* the served draft, and report whether that cost a restart.

        An endpoint that can tell what is already running may satisfy the call
        without restarting -- but only when *allow_engine_reuse* is set.  The
        gate clears it for every scored measurement block.  A block that
        inherits an engine launched before the benchmark did not open from the
        same lifecycle state as its peer, and the difference lands on whichever
        arm the schedule starts with.
        See :data:`DEFAULT_ARM_BLOCKS`.

        Returns:
            True if the engine was restarted, False if a running engine was
            reused.  The gate checks this rather than trusting the flag: the
            invariant is a property of what happened, not of what was asked.
        """
        ...


class MetricsSource(Protocol):
    """Raw Prometheus metrics source for the active endpoint."""

    def scrape(
        self,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> str: ...


class ReplayExecutor(Protocol):
    """Synchronous boundary around replaying a frozen suite."""

    def replay(
        self,
        suite: BenchmarkSuite,
        endpoint_url: str,
        sampling: SamplingConfig,
        *,
        repeats: int,
        timeout_seconds: float,
        should_abort: AbortCheck,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        capture_tokens: bool = False,
    ) -> ReplayResult: ...


class BenchmarkAborted(RuntimeError):
    """Raised internally when serving activity preempts a benchmark."""


class BenchmarkTimedOut(TimeoutError):
    """Raised internally when the benchmark's whole-run deadline expires."""


class HttpReplayExecutor:
    """Run the existing async HTTP replay implementation behind a protocol."""

    _POLL_SECONDS = 0.05

    def __init__(
        self,
        *,
        model: str = "auto",
        concurrency: int = DEFAULT_REPLAY_CONCURRENCY,
    ) -> None:
        if not model:
            raise ValueError("replay model must be non-empty")
        self._model = model
        self._concurrency = _validated_concurrency(concurrency, "replay concurrency")

    @property
    def concurrency(self) -> int:
        """Requests this executor keeps in flight within one suite pass."""
        return self._concurrency

    def replay(
        self,
        suite: BenchmarkSuite,
        endpoint_url: str,
        sampling: SamplingConfig,
        *,
        repeats: int,
        timeout_seconds: float,
        should_abort: AbortCheck,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        capture_tokens: bool = False,
    ) -> ReplayResult:
        """Replay *suite*, optionally overriding this executor's own degree.

        ``concurrency`` overrides :attr:`concurrency` for this call only, which
        is how the gate runs its correctness pass single-stream without giving
        up the throughput pass's batching.
        """
        degree = (
            self._concurrency
            if concurrency is None
            else _validated_concurrency(concurrency, "replay concurrency")
        )

        async def run_with_preemption() -> ReplayResult:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_seconds
            task = asyncio.create_task(
                replay_suite(
                    suite,
                    endpoint_url,
                    sampling,
                    repeats=repeats,
                    timeout=timeout_seconds,
                    model=self._model,
                    concurrency=degree,
                    max_tokens=max_tokens,
                    capture_tokens=capture_tokens,
                )
            )
            try:
                while not task.done():
                    if should_abort():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        raise BenchmarkAborted("benchmark replay was preempted")
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        raise BenchmarkTimedOut("benchmark replay timed out")
                    await asyncio.wait(
                        {task},
                        timeout=min(self._POLL_SECONDS, remaining),
                    )
                return await task
            finally:
                if not task.done():
                    task.cancel()

        return asyncio.run(run_with_preemption())


@dataclass(frozen=True, slots=True)
class _ArmMeasurement:
    """Everything one arm contributes to the decision."""

    #: The scored suite passes, concatenated into one result so downstream code
    #: still sees "N repeats of this arm" regardless of how they were issued.
    replay: ReplayResult
    #: One metric window per scored repeat, in repeat order.
    repeat_deltas: tuple[MetricsDelta, ...]
    #: The window spanning every scored repeat, i.e. first scrape to last.
    #: This is the same quantity the gate used to report as *the* delta, so the
    #: diagnostic Prometheus throughput keeps its meaning across the change.
    pooled_delta: MetricsDelta
    #: The bounded, single-stream output-correctness pass.
    correctness: ReplayResult
    #: A second such pass by the *same* arm, when one was run.  Compared against
    #: ``correctness`` it measures the engine's divergence rate against itself;
    #: see :data:`CORRECTNESS_CONTROL_REPEATS`.
    correctness_control: ReplayResult | None = None
    #: The unscored warmup pass this arm ran before each of its blocks, in block
    #: order.  One entry per block, ``None`` where warmup was disabled.
    warmups: tuple[ReplayResult | None, ...] = ()


@dataclass
class _ArmBuilder:
    """One arm's measurement while its blocks are still being collected.

    Mutable on purpose: the blocks of an arm are interleaved with the other
    arm's, so an arm's evidence arrives in pieces separated in time and cannot
    be assembled by a single function call the way the sequential design's could.
    """

    #: One tuple of snapshots per block: the block's opening scrape followed by
    #: one scrape after each of its scored repeats.  Never spliced across
    #: blocks -- the engine restarts in between and the counters restart with it.
    chains: list[tuple[MetricsSnapshot, ...]] = field(default_factory=list)
    runs: list[RunResults] = field(default_factory=list)
    warmups: list[ReplayResult | None] = field(default_factory=list)
    correctness: ReplayResult | None = None
    correctness_control: ReplayResult | None = None
    blocks_run: int = 0
    repeats_scored: int = 0

    def build(self, suite_hash: str) -> _ArmMeasurement:
        repeat_deltas = tuple(
            _safe_delta(before, after)
            for chain in self.chains
            for before, after in zip(chain[:-1], chain[1:], strict=True)
        )
        # A reset inside any repeat window invalidates the arm, even when the
        # endpoints of the pooled window happen to look monotone across it.
        # Sampling more finely must not make the gate blinder.
        pooled = (
            _reset_delta(self.chains[0][0], self.chains[-1][-1])
            if any(d.reset_detected for d in repeat_deltas)
            else _pool_chains(self.chains)
        )
        assert self.correctness is not None
        return _ArmMeasurement(
            replay=ReplayResult(
                run_results=tuple(self.runs),
                num_runs=len(self.runs),
                suite_hash=suite_hash,
            ),
            repeat_deltas=repeat_deltas,
            pooled_delta=pooled,
            correctness=self.correctness,
            correctness_control=self.correctness_control,
            warmups=tuple(self.warmups),
        )


class BenchmarkGateRunner:
    """Build/load a suite, measure both draft arms, and decide promotion."""

    def __init__(
        self,
        *,
        config: SpeedLMConfig,
        trace_source: TraceSource,
        suite_dir: SuiteDirectory,
        stock_draft: StockDraft,
        endpoint: DraftEndpoint,
        metrics_source: MetricsSource,
        replay_executor: ReplayExecutor | None = None,
        replay_concurrency: int = DEFAULT_REPLAY_CONCURRENCY,
        repeats: int = 3,
        warmup_repeats: int = 1,
        arm_blocks: int = DEFAULT_ARM_BLOCKS,
        correctness_max_tokens: int = DEFAULT_CORRECTNESS_MAX_TOKENS,
        benchmark_max_tokens: int = DEFAULT_BENCHMARK_MAX_TOKENS,
        held_out_fraction: float = 0.2,
        training_context_hashes: TrainingHashes | None = None,
        candidate_arm_first: bool = False,
        engine_execution: EngineExecution | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
            raise ValueError("repeats must be an integer >= 3")
        if (
            isinstance(warmup_repeats, bool)
            or not isinstance(warmup_repeats, int)
            or warmup_repeats < 0
        ):
            raise ValueError("warmup_repeats must be an integer >= 0")
        # Validated eagerly rather than at the first benchmark: an arm_blocks
        # the schedule cannot honour is a construction error, and discovering
        # it only once a cycle has already spent its training budget is the
        # expensive way to find out.
        _block_schedule("stock", blocks=arm_blocks, repeats=repeats)
        if (
            isinstance(correctness_max_tokens, bool)
            or not isinstance(correctness_max_tokens, int)
            or correctness_max_tokens < 1
        ):
            raise ValueError("correctness_max_tokens must be an integer >= 1")
        if (
            isinstance(benchmark_max_tokens, bool)
            or not isinstance(benchmark_max_tokens, int)
            or benchmark_max_tokens < 1
        ):
            raise ValueError("benchmark_max_tokens must be an integer >= 1")
        if (
            isinstance(held_out_fraction, bool)
            or not isinstance(held_out_fraction, (int, float))
            or not 0 <= held_out_fraction <= 1
        ):
            raise ValueError("held_out_fraction must be a number in [0, 1]")
        self._config = config
        self._trace_source = trace_source
        self._suite_dir = suite_dir
        # Held unresolved on purpose; see :data:`StockDraft` and
        # :meth:`_resolve_stock_draft`.
        self._stock_draft = stock_draft
        self._endpoint = endpoint
        self._metrics_source = metrics_source
        _validated_concurrency(replay_concurrency, "replay_concurrency")
        self._replay_executor = replay_executor or HttpReplayExecutor(
            model=config.alias,
            concurrency=replay_concurrency,
        )
        # Report what the executor actually does, not what was asked for: an
        # injected executor owns its own degree and a metrics field that
        # disagrees with the traffic it describes is worse than no field.
        executor_concurrency = getattr(self._replay_executor, "concurrency", None)
        self._replay_concurrency = (
            executor_concurrency
            if isinstance(executor_concurrency, int)
            and not isinstance(executor_concurrency, bool)
            and executor_concurrency >= 1
            else replay_concurrency
        )
        self._repeats = repeats
        self._warmup_repeats = warmup_repeats
        self._arm_blocks = arm_blocks
        self._correctness_max_tokens = correctness_max_tokens
        self._benchmark_max_tokens = benchmark_max_tokens
        self._held_out_fraction = float(held_out_fraction)
        self._training_context_hashes = training_context_hashes
        # Defaults to the historical stock-first order so that constructing a
        # runner directly keeps measuring what it always did.  Production wires
        # this from ``tuning.benchmark_candidate_arm_first``, which defaults to
        # true and carries the rationale and the bias disclosure.
        if not isinstance(candidate_arm_first, bool):
            raise ValueError("candidate_arm_first must be a bool")
        self._candidate_arm_first = candidate_arm_first
        # How the engine both arms replay against is executing.  The runner
        # owns no vLLM process and cannot discover this -- see the module
        # docstring -- so it is injected by whoever built the argv.  ``None``
        # is honest and stays honest: the decision then records
        # ``engine_execution_mode: unrecorded`` rather than assuming eager.
        if engine_execution is not None and not isinstance(
            engine_execution, EngineExecution
        ):
            raise ValueError("engine_execution must be an EngineExecution or None")
        self._engine_execution = engine_execution
        self._clock = clock

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> GateResult:
        """Benchmark stock and candidate under one deadline.

        Abort and timeout are fail-closed outcomes without a decision.  A
        completed rejection, including a counter-reset rejection, carries the
        real :class:`Decision` returned by :func:`decide_promotion`.
        """
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number")

        deadline = self._clock() + float(timeout_seconds)

        def checkpoint(stage: str) -> None:
            if should_abort():
                raise BenchmarkAborted(f"benchmark aborted {stage}")
            if self._clock() >= deadline:
                raise BenchmarkTimedOut(f"benchmark timed out {stage}")

        def remaining(stage: str) -> float:
            checkpoint(stage)
            value = deadline - self._clock()
            if value <= 0:
                raise BenchmarkTimedOut(f"benchmark timed out {stage}")
            return value

        def stage(name: str, operation: Callable[[float], _T]) -> _T:
            checkpoint(f"before {name}")
            result = operation(remaining(f"during {name}"))
            checkpoint(f"after {name}")
            return result

        # Every scrape's raw exposition is kept verbatim.  Acceptance is a
        # counter delta, so a reader who doubts a reported rate can only
        # reconcile it against the absolute counters that produced it, and a
        # parsed-and-discarded body makes that question unanswerable without a
        # re-run.
        bodies: dict[str, str] = {}

        def scrape(name: str, label: str) -> MetricsSnapshot:
            def operation(available: float) -> MetricsSnapshot:
                snapshot, text = self._scrape(available, should_abort)
                bodies[label] = text
                return snapshot

            return stage(name, operation)

        try:
            suite = stage("suite preparation", lambda _timeout: self._load_or_build_suite())
            stage(
                "suite leakage check",
                lambda _timeout: self._check_suite_leakage(suite),
            )

            def activate(
                arm: str,
                draft: DraftReference,
                *,
                allow_engine_reuse: bool = True,
            ) -> bool:
                return stage(
                    f"{arm} activation",
                    lambda available: self._endpoint.activate(
                        draft,
                        timeout_seconds=available,
                        should_abort=should_abort,
                        allow_engine_reuse=allow_engine_reuse,
                    ),
                )

            # Resolved once, here, rather than at construction: what counts as
            # "stock" is whatever is serving when this benchmark runs, and both
            # arms of one benchmark must agree on it even if a promotion lands
            # between the two activations.
            stock_draft = self._resolve_stock_draft()
            drafts: dict[str, DraftReference] = {
                "stock": stock_draft,
                "candidate": candidate_draft,
            }

            # Arm order balances wall-clock position only.  It never changes
            # engine lifecycle: every scored block restarts below, including a
            # candidate-first block after CANDIDATE_STARTING built that draft.
            first_arm = "candidate" if self._candidate_arm_first else "stock"
            schedule = _block_schedule(
                first_arm, blocks=self._arm_blocks, repeats=self._repeats
            )

            arms: dict[str, _ArmBuilder] = {
                "stock": _ArmBuilder(),
                "candidate": _ArmBuilder(),
            }
            blocks_run: list[MeasurementBlock] = []
            for block_index, (arm, block_repeats) in enumerate(schedule):
                builder = arms[arm]
                # No scored-block exemption: CANDIDATE_STARTING may already have
                # built block 0's candidate engine, but reusing only that engine
                # makes one arm's lifecycle systematically different.  Paying
                # one more activation makes the invariant independent of arm
                # order, block count, model, and inference configuration.
                restarted = activate(arm, drafts[arm], allow_engine_reuse=False)
                # Checked, not assumed.  The flag is a request to the endpoint;
                # this is the endpoint's answer, and the invariant is a
                # property of the answer.  Fail-closed: the benchmark yields no
                # decision rather than a decision whose arms were measured
                # under different engine lifecycles.
                if not restarted:
                    raise BenchmarkAborted(
                        f"{arm} block {block_index} opened on an engine it did "
                        "not restart: every scored measurement window must open "
                        "from activate, warm, then score"
                    )
                blocks_run.append(
                    MeasurementBlock(
                        arm=arm, repeats=block_repeats, restarted=restarted
                    )
                )
                # Warm the freshly restarted engine before the measurement
                # window opens.  Activation restarts vLLM, so the first suite
                # pass pays one-time JIT compilation of the speculative-decoding
                # kernels; that cost belongs to neither arm's steady state.  The
                # warmup runs *before* the pre-scrape, which keeps it out of the
                # Prometheus window as well as out of ``per_repeat``.  It is
                # what keeps the two arms comparable when the arm order is
                # reversed -- each arm pays its own cold start -- and, once
                # there is more than one block per arm, what keeps every
                # measurement window of *either* arm opening from the same
                # engine lifecycle state: activate, warm, then score.
                builder.warmups.append(
                    stage(
                        f"{arm} warmup",
                        lambda available: self._warmup(suite, available, should_abort),
                    )
                )
                builder.blocks_run += 1
                self._measure_block(
                    arm,
                    suite,
                    stage,
                    scrape,
                    should_abort,
                    builder=builder,
                    block_repeats=block_repeats,
                    final=builder.blocks_run == self._arm_blocks,
                    # Only the reference arm measures the noise floor; see
                    # :data:`CORRECTNESS_CONTROL_REPEATS`.
                    control=arm == "stock",
                )

            stock_arm = arms["stock"].build(suite.suite_hash)
            candidate_arm = arms["candidate"].build(suite.suite_hash)

            decision = stage(
                "promotion decision",
                lambda _timeout: decide_promotion(
                    stock_arm.pooled_delta,
                    candidate_arm.pooled_delta,
                    stock_arm.replay,
                    candidate_arm.replay,
                    self._config.promotion,
                    warmup_repeats=_warmup_runs(
                        stock_arm.warmups, candidate_arm.warmups
                    ),
                    stock_repeat_metrics=stock_arm.repeat_deltas,
                    candidate_repeat_metrics=candidate_arm.repeat_deltas,
                    stock_correctness=stock_arm.correctness,
                    candidate_correctness=candidate_arm.correctness,
                    stock_correctness_control=stock_arm.correctness_control,
                    # Measurement context.  These were already published on
                    # ``GateResult.metrics``, which nothing persists, so a
                    # decision.json could not say what cap, degree, suite or
                    # baseline produced the numbers it reports.
                    benchmark_max_tokens=self._benchmark_max_tokens,
                    replay_concurrency=self._replay_concurrency,
                    correctness_max_tokens=self._correctness_max_tokens,
                    num_contexts=len(suite.contexts),
                    stock_draft=str(stock_draft),
                    engine_execution=self._engine_execution,
                ),
            )
            # Whether the throughput columns the decision was computed from were
            # stationary.  See :data:`PROMOTION_REQUIRES_STATIONARITY`; it is
            # computed unconditionally so the record carries the answer even
            # when the verdict was never going to be a promotion.
            stationarity = _stationarity(
                decision, abs(self._config.promotion.min_throughput_delta_pct)
            )
            # Onto the decision, which is the only thing written to disk.  Both
            # are properties of how the measurement was taken rather than of
            # the comparison, so they are attached here rather than passed
            # through ``decide_promotion`` -- and the stationarity verdict is
            # computed *from* the decision, so it could not be.
            decision = replace(
                decision,
                arm_blocks=self._arm_blocks,
                block_schedule=tuple(blocks_run),
                throughput_stationarity=stationarity,
                divergence_sampling=self._divergence_sampling_record(),
            )
            # Read off the record rather than recomputed here.  The veto used to
            # live only in this local, so ``decision.json`` said ``promote`` for
            # a cycle that rolled back; ``final_verdict`` is now the one place
            # the outcome is derived, and it is persisted.
            promoted = decision.final_verdict is Verdict.PROMOTE

            # The last block of the schedule leaves *its* arm serving.  A
            # promotion is the one case that needs the candidate back: the
            # orchestrator promotes by flipping the durable pointer and waking,
            # never by restarting, so whatever is running when this returns is
            # what serves live traffic.  A non-promotion -- a rejection, or a
            # promote-shaped decision this runner vetoed -- wants stock, and is
            # left to the cycle's own restore path, exactly as the stock-first
            # order has always been.  (One block per arm plus
            # ``candidate_arm_first`` is the case where that restore is free
            # because the schedule already ends on stock; mirrored rounds end on
            # whichever arm started, which is the rollback restart
            # :data:`DEFAULT_ARM_BLOCKS` costs out.)
            #
            # It is inside the try on purpose.  If the candidate cannot be put
            # back, the abort/timeout handlers below report a non-passing gate,
            # which is the fail-closed answer: a candidate that cannot be made
            # to serve must not be promoted.
            if promoted and schedule[-1][0] != "candidate":
                activate("candidate", candidate_draft)
        except BenchmarkAborted as exc:
            from speedlm.tuner.orchestrator import GateFailure  # noqa: PLC0415

            return _gate_result(
                passed=False,
                reason=str(exc),
                metrics={"aborted": True},
                metrics_bodies=bodies,
                failure=GateFailure.ABORTED,
            )
        except (BenchmarkTimedOut, TimeoutError) as exc:
            from speedlm.tuner.orchestrator import GateFailure  # noqa: PLC0415

            return _gate_result(
                passed=False,
                reason=str(exc) or "benchmark timed out",
                metrics={"timed_out": True},
                metrics_bodies=bodies,
                failure=GateFailure.TIMED_OUT,
            )

        return _gate_result(
            passed=promoted,
            # A vetoed promotion must not report the reason the *numbers* gave,
            # or the cycle log would say "gate rejected: both thresholds met".
            # ``final_reason`` is that rule, moved onto the record so the cycle
            # log and ``decision.json`` cannot disagree about the same veto.
            reason=decision.final_reason,
            metrics={
                "suite_hash": suite.suite_hash,
                "engine_execution": _engine_execution_dict(
                    self._engine_execution
                ),
                "num_contexts": len(suite.contexts),
                # The baseline this comparison was against, as resolved when
                # the benchmark ran.  It moves on every promotion.
                "stock_draft": str(stock_draft),
                # Observed, not configured: a report that says three repeats
                # ran must mean three repeats ran.
                "num_repeats": min(
                    stock_arm.replay.num_runs, candidate_arm.replay.num_runs
                ),
                "requested_repeats": self._repeats,
                # Both arms replay at this degree.  Absolute tok/s figures are
                # only comparable across runs that shared it.
                "replay_concurrency": self._replay_concurrency,
                # Both arms replay under this cap.  It bounds the acceptance
                # window as well as the wall clock, so a reader comparing
                # acceptance deltas across runs needs to know it matched.
                "benchmark_max_tokens": self._benchmark_max_tokens,
                # The correctness pass is a different measurement at a
                # different degree; recording both stops a reader assuming the
                # single number above described every request the gate sent.
                "correctness": {
                    "concurrency": CORRECTNESS_REPLAY_CONCURRENCY,
                    "max_tokens": self._correctness_max_tokens,
                    "repeats": CORRECTNESS_REPEATS,
                    # Extra stock-arm passes that measure the engine's own
                    # divergence rate; see ``CORRECTNESS_CONTROL_REPEATS``.
                    "control_repeats": CORRECTNESS_CONTROL_REPEATS,
                    "control_arm": "stock",
                },
                "stock_runs": stock_arm.replay.num_runs,
                "candidate_runs": candidate_arm.replay.num_runs,
                # Unscored, but not invisible: the warmup pass is reported so a
                # reader can see what was excluded and how slow it was.
                "warmup": {
                    "requested_repeats": self._warmup_repeats,
                    "stock": _warmup_dict(stock_arm.warmups),
                    "candidate": _warmup_dict(candidate_arm.warmups),
                },
                # How the scored repeats were interleaved, and where each arm's
                # blocks landed in the pass order.  Without it a reader cannot
                # tell an arm-to-arm delta that survived a drift correction from
                # one that never faced a drift.
                "schedule": {
                    "arm_blocks": self._arm_blocks,
                    "blocks": [block.to_dict() for block in blocks_run],
                },
                # Whether the columns the delta came from had stopped moving,
                # and whether that was allowed to matter.  See
                # :data:`PROMOTION_REQUIRES_STATIONARITY`.
                "throughput_stationarity": stationarity.to_dict(),
                "stock": _delta_dict(stock_arm.pooled_delta),
                "candidate": _delta_dict(candidate_arm.pooled_delta),
                # Per-repeat windows, published beside the pooled one so the
                # acceptance vector in ``decision.json`` is reconcilable
                # against the counters that produced it.
                "stock_per_repeat": [_delta_dict(d) for d in stock_arm.repeat_deltas],
                "candidate_per_repeat": [
                    _delta_dict(d) for d in candidate_arm.repeat_deltas
                ],
            },
            metrics_bodies=bodies,
            decision=decision,
        )

    def estimated_benchmark_seconds(self) -> float | None:
        """Deadline this benchmark needs, sized from the work it will do.

        The orchestrator reads this to replace a fixed benchmark deadline with
        one derived from the actual held-out suite; ``None`` means "cannot
        tell", and the caller falls back to its configured hard ceiling.

        Freezing the suite here is deliberate and cheap: ``_load_or_build_suite``
        persists it, so the pass that follows loads exactly the same suite this
        estimate was sized against rather than a differently-sized one.
        """
        try:
            suite = self._load_or_build_suite()
        except Exception:
            return None
        from speedlm.tuner.orchestrator import derive_benchmark_timeout  # noqa: PLC0415

        return derive_benchmark_timeout(
            num_contexts=len(suite.contexts),
            repeats=self._repeats,
            # Every block opens its own measurement window and so pays its own
            # warmup: the arms run ``arm_blocks`` warmups each, not one.  The
            # engine restart at each block boundary is still uncharged here, as
            # activation always has been -- ``BENCHMARK_SAFETY_FACTOR`` and the
            # fixed overhead are what cover it.
            warmup_repeats=self._warmup_repeats * self._arm_blocks,
            concurrency=self._replay_concurrency,
            # Charged per arm, and only the stock arm runs the control -- so
            # this over-books by one bounded pass.  Deliberate: a deadline that
            # is one correctness pass too generous costs nothing, and one that
            # is a pass too tight kills the run before it decides.
            correctness_repeats=CORRECTNESS_REPEATS + CORRECTNESS_CONTROL_REPEATS,
            benchmark_max_tokens=self._benchmark_max_tokens,
            correctness_max_tokens=self._correctness_max_tokens,
        )

    def _measure_block(
        self,
        arm: str,
        suite: BenchmarkSuite,
        stage: Callable[[str, Callable[[float], object]], object],
        scrape: Callable[[str, str], MetricsSnapshot],
        should_abort: AbortCheck,
        *,
        builder: _ArmBuilder,
        block_repeats: int,
        final: bool,
        control: bool = False,
    ) -> None:
        """Run one contiguous block of one arm's scored repeats.

        The scrape schedule is one before the block's first repeat and one after
        each repeat -- ``repeats + 1`` scrapes per block rather than 2, which
        costs no extra suite passes and no extra generations.  Consecutive pairs
        give a real per-repeat acceptance sample; a block's first and last give
        the window it contributes to the pooled measurement.  Before this, the
        repeat loop lived inside a single ``replay(repeats=N)`` call with the
        scrapes outside it, so there was exactly one acceptance measurement per
        arm no matter how many repeats ran, and it was stamped into all N
        per-repeat rows.

        Each block scrapes its own opening snapshot on purpose.  A block
        boundary is a draft change and therefore an engine restart, so the
        counters an earlier block ended on are gone; differencing across the
        boundary would report a counter reset for a measurement that had nothing
        wrong with it.  The per-repeat ladder is unbroken *within* each block,
        which is the level acceptance is sampled at, and the arm's pooled window
        is the sum of its blocks rather than one first-to-last difference -- see
        :func:`_pool_chains`.

        The correctness pass runs after the arm's *last* block, and only then:
        it is bounded and single-stream, so folding it into a measurement window
        would contaminate both the acceptance counters and the decode-time
        throughput with traffic that is not the workload being measured, and
        running it once per block would multiply a cost that buys nothing.

        With ``control`` set the arm makes
        ``CORRECTNESS_REPEATS + CORRECTNESS_CONTROL_REPEATS`` correctness passes
        in one call rather than ``CORRECTNESS_REPEATS``, and the trailing passes
        are kept separately as the noise-floor control.  They are issued back to
        back on the one engine on purpose: anything that differs between the two
        passes is nondeterminism the same engine produced under the same
        settings, which is exactly the quantity the criterion needs.
        """
        first_index = builder.repeats_scored
        # ``{arm}-before`` and ``{arm}-after`` keep naming the ends of the arm's
        # whole measurement, so a one-block schedule labels its scrapes exactly
        # as it always did and archived runs stay comparable.
        opening = (
            f"{arm}-before"
            if first_index == 0
            else f"{arm}-before-repeat-{first_index}"
        )
        snapshots = [scrape(f"{arm} metrics pre-scrape", opening)]
        for offset in range(block_repeats):
            index = first_index + offset
            replay = stage(
                f"{arm} replay repeat {index}",
                lambda available: self._replay(suite, available, should_abort, repeats=1),
            )
            assert isinstance(replay, ReplayResult)
            builder.runs.extend(replay.run_results)
            label = (
                f"{arm}-after"
                if final and offset == block_repeats - 1
                else f"{arm}-after-repeat-{index}"
            )
            snapshots.append(scrape(f"{arm} metrics scrape after repeat {index}", label))
        builder.repeats_scored += block_repeats
        builder.chains.append(tuple(snapshots))

        if not final:
            return

        correctness_runs = stage(
            f"{arm} correctness pass",
            lambda available: self._correctness_replay(
                suite, available, should_abort, control=control
            ),
        )
        assert isinstance(correctness_runs, ReplayResult)
        builder.correctness, builder.correctness_control = _split_correctness(
            correctness_runs
        )

    def _resolve_stock_draft(self) -> DraftReference:
        """The draft the stock arm must run, as of *now*.

        The gate is the only safeguard and there is no post-promotion rollback,
        so the baseline it measures against has to be the draft that is
        currently serving.  Resolving a frozen reference captured at
        construction time made every cycle after the first compare the
        candidate against the *original* head: the reported delta was then
        cumulative improvement over that head rather than marginal improvement
        over the incumbent, and a candidate strictly worse than what was
        already serving could still show a positive delta and be promoted.

        Raises:
            ValueError: If the reference resolves to something that cannot name
                a draft.  Fail-closed: an unresolvable baseline fails the cycle
                rather than silently benchmarking against a stale one.
        """
        configured = self._stock_draft
        draft = configured() if callable(configured) else configured
        if not isinstance(draft, (str, Path)) or not str(draft):
            raise ValueError(
                "stock draft must resolve to a non-empty path or model id, "
                f"got {draft!r}"
            )
        return draft

    def _load_or_build_suite(self) -> BenchmarkSuite:
        suite_dir = self._suite_dir() if callable(self._suite_dir) else self._suite_dir
        manifest = suite_dir / "suite_manifest.json"
        if manifest.exists():
            return load_suite(suite_dir)
        suite = build_suite(
            tuple(self._trace_source.iter_records()),
            held_out_fraction=self._held_out_fraction,
        )
        persist_suite(suite, suite_dir)
        return suite

    def _check_suite_leakage(self, suite: BenchmarkSuite) -> None:
        if self._training_context_hashes is None:
            raise SuiteError(
                "Cannot prove benchmark suite is held out: "
                "training context hashes were not provided"
            )
        configured = self._training_context_hashes
        hashes = configured() if callable(configured) else configured
        overlaps = suite.check_leakage(set(hashes))
        if overlaps:
            preview = ", ".join(overlaps[:3])
            suffix = "" if len(overlaps) <= 3 else f", ... ({len(overlaps)} total)"
            raise SuiteError(
                "Training/benchmark context leakage detected: "
                f"{preview}{suffix}"
            )

    def _scrape(
        self,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> tuple[MetricsSnapshot, str]:
        text = self._metrics_source.scrape(
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )
        return parse_metrics(text), text

    def _replay(
        self,
        suite: BenchmarkSuite,
        timeout_seconds: float,
        should_abort: AbortCheck,
        *,
        repeats: int | None = None,
    ) -> ReplayResult:
        """One throughput/acceptance pass, capped and batched.

        The cap is sent explicitly rather than left to the server.  Omitting it
        did not produce an unbounded pass; it produced one bounded by
        ``max_model_len`` minus the prompt, which is a different length for
        every served model and truncated 25.6% of gpt-oss requests against 3.5%
        of Qwen's.  See
        :attr:`speedlm.config.IdleTuningConfig.benchmark_max_tokens` for the
        measurements and for what capping does and does not bias.
        """
        return self._replay_executor.replay(
            suite,
            self._endpoint.url,
            self._config.sampling,
            repeats=self._repeats if repeats is None else repeats,
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
            max_tokens=self._benchmark_max_tokens,
        )

    def _correctness_replay(
        self,
        suite: BenchmarkSuite,
        timeout_seconds: float,
        should_abort: AbortCheck,
        *,
        control: bool = False,
    ) -> ReplayResult:
        """One bounded, single-stream pass whose only job is output agreement.

        The concurrency is pinned here rather than passed through from config
        on purpose: ``benchmark_concurrency`` is a throughput knob, and letting
        it reach this pass is exactly the conflation that made job 369005's
        equality check unsound.

        ``control`` adds :data:`CORRECTNESS_CONTROL_REPEATS` further passes so
        the arm can be compared against itself.  They go through this same call,
        under this same cap and this same concurrency, because a floor measured
        under different conditions from the measurement is not that
        measurement's floor.

        The sampling is pinned greedy here for the same reason the concurrency
        is pinned: ``config.sampling`` is a *serving* knob, and letting it reach
        this pass is the conflation that would make the equality check
        meaningless.  See :data:`DIVERGENCE_TEMPERATURE`.
        """
        return self._replay_executor.replay(
            suite,
            self._endpoint.url,
            self._divergence_sampling(),
            repeats=(
                CORRECTNESS_REPEATS + CORRECTNESS_CONTROL_REPEATS
                if control
                else CORRECTNESS_REPEATS
            ),
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
            concurrency=CORRECTNESS_REPLAY_CONCURRENCY,
            max_tokens=self._correctness_max_tokens,
            capture_tokens=True,
        )

    def _divergence_sampling(self) -> SamplingConfig:
        """The configured sampling, forced greedy.  See :data:`DIVERGENCE_TEMPERATURE`."""
        return replace(
            self._config.sampling,
            temperature=DIVERGENCE_TEMPERATURE,
            top_p=DIVERGENCE_TOP_P,
        )

    def _divergence_sampling_record(self) -> DivergenceSampling:
        """What the two passes replayed under, for the persisted decision."""
        used = self._divergence_sampling()
        configured = self._config.sampling
        return DivergenceSampling(
            temperature=used.temperature,
            top_p=used.top_p,
            configured_temperature=configured.temperature,
            configured_top_p=configured.top_p,
        )

    def _warmup(
        self,
        suite: BenchmarkSuite,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> ReplayResult | None:
        if self._warmup_repeats <= 0:
            return None
        return self._replay(
            suite,
            timeout_seconds,
            should_abort,
            repeats=self._warmup_repeats,
        )


GateRunner = BenchmarkGateRunner


def _split_correctness(
    result: ReplayResult,
) -> tuple[ReplayResult, ReplayResult | None]:
    """Divide a correctness replay into its scored passes and its control.

    The first :data:`CORRECTNESS_REPEATS` passes are the arm's own correctness
    evidence; anything beyond them is the noise-floor control.  An arm that ran
    no extra passes returns ``None`` for the control rather than an empty
    :class:`~speedlm.gate.replay.ReplayResult`, so "no control was run" cannot be
    mistaken downstream for "a control ran and found nothing".
    """
    scored = result.run_results[:CORRECTNESS_REPEATS]
    extra = result.run_results[CORRECTNESS_REPEATS:]
    measured = ReplayResult(
        run_results=scored,
        num_runs=len(scored),
        suite_hash=result.suite_hash,
    )
    if not extra:
        return measured, None
    return measured, ReplayResult(
        run_results=extra,
        num_runs=len(extra),
        suite_hash=result.suite_hash,
    )


#: Counters :func:`_pool_chains` adds up across an arm's blocks.  Extensive
#: quantities only: rates are re-derived from the totals, never averaged, so a
#: block that scored more repeats weighs more, exactly as it would have inside
#: one first-to-last window.
_POOLED_COUNTERS: Final = (
    "generated_tokens",
    "decode_time_seconds",
    "drafted_tokens",
    "accepted_tokens",
    "num_drafts",
)


def _pool_chains(
    chains: Sequence[tuple[MetricsSnapshot, ...]],
) -> MetricsDelta:
    """The arm's whole measurement window, summed over its blocks.

    A one-block arm reproduces :func:`speedlm.gate.metrics.compute_delta` on its
    two endpoints bit for bit -- the same subtractions in the same order -- so
    the sequential design's pooled figures are unchanged.

    More than one block cannot be differenced end to end: the blocks are
    separated by an engine restart, so ``last - first`` is a counter reset, not
    a measurement.  Summing the per-block differences is the same quantity the
    single window reported (total work over total decode time) and is the only
    form of it that survives the restart.
    """
    totals = dict.fromkeys(_POOLED_COUNTERS, 0.0)
    has_draft = True
    for chain in chains:
        before, after = chain[0], chain[-1]
        for name in _POOLED_COUNTERS:
            delta = getattr(after, name) - getattr(before, name)
            if delta < 0:
                return _reset_delta(chains[0][0], chains[-1][-1])
            totals[name] += delta
        has_draft = has_draft and after.has_draft_counters

    drafted = totals["drafted_tokens"]
    accepted = totals["accepted_tokens"]
    drafts = totals["num_drafts"]
    generated = totals["generated_tokens"]
    decode_seconds = totals["decode_time_seconds"]
    acceptance_available = has_draft and drafted > 0
    return MetricsDelta(
        reset_detected=False,
        acceptance_available=acceptance_available,
        drafted_tokens=drafted,
        accepted_tokens=accepted,
        acceptance_rate=accepted / drafted if acceptance_available else 0.0,
        mean_accepted_length=(
            1.0 + accepted / drafts if acceptance_available and drafts > 0 else 0.0
        ),
        tpot_ms=(decode_seconds / generated) * 1000.0 if generated > 0 else 0.0,
        output_tok_per_sec=generated / decode_seconds if decode_seconds > 0 else 0.0,
    )


def _stationarity(
    decision: Decision, materiality_pct: float
) -> ThroughputStationarity:
    """Whether the arm-to-arm throughput delta held still during the run.

    The column tested is the *paired* per-repeat delta, ``100 x (candidate -
    stock) / stock`` repeat by repeat, not either arm's own throughput.  Under
    the interleaved schedule repeat ``i`` of one arm is roughly contemporaneous
    with repeat ``i`` of the other, so machine drift both arms saw cancels out
    of this column and only drift that landed on one arm survives -- which is
    the error that makes an arm-to-arm delta meaningless, and the only one worth
    vetoing on.  The per-arm ``*_throughput_flat_from_repeat`` indices stay in
    the record beside it: they answer the different, and still useful, question
    of whether ``warmup_repeats`` was large enough.

    Proven non-stationary means the largest split shift is *both* resolvable
    above the column's own noise (:data:`STATIONARITY_SPLIT_T_STATISTIC`) and at
    least ``materiality_pct`` percentage points wide.  Both conditions are needed:
    without the first, a tightly-measured irrelevance vetoes; without the
    second, a large ``t`` on a fraction of a point does.  The materiality bar is
    the magnitude of ``PromotionConfig.min_throughput_delta_pct`` -- the width of
    the guard the gate is deciding with -- so a delta that moved by more than
    the whole guard during the run cannot support a decision made with it.

    A material shift below the t bar is recorded as
    ``material_shift_unresolved``: it is neither called stationary nor used as
    proof of drift.  Below :data:`MIN_STATIONARITY_REPEATS` repeats the answer is
    ``untestable`` and nothing is vetoed; see that constant.
    """
    repeats = decision.per_repeat
    paired = [
        100.0 * (row.candidate_tok_per_sec - row.stock_tok_per_sec)
        / row.stock_tok_per_sec
        for row in repeats
        if row.stock_tok_per_sec > 0
    ]
    testable = (
        len(paired) == len(repeats) and len(paired) >= MIN_STATIONARITY_REPEATS
    )
    shift, statistic = _delta_shift(paired) if testable else (None, None)
    return ThroughputStationarity(
        testable=testable,
        min_repeats=MIN_STATIONARITY_REPEATS,
        # Percentage points the arm-to-arm delta moved between the run's leading
        # and trailing repeats, at the split that separates them best.
        delta_shift_pct=shift,
        delta_shift_t_statistic=statistic,
        min_shift_t_statistic=STATIONARITY_SPLIT_T_STATISTIC,
        materiality_pct=materiality_pct,
        # Kept beside the verdict because they answer the warmup question that
        # the paired column deliberately cancels out of itself.
        stock_flat_from_repeat=decision.stock_throughput_flat_from_repeat,
        candidate_flat_from_repeat=(
            decision.candidate_throughput_flat_from_repeat
        ),
        stock_trend_pct_per_repeat=(
            decision.stock_throughput_trend_pct_per_repeat
        ),
        candidate_trend_pct_per_repeat=(
            decision.candidate_throughput_trend_pct_per_repeat
        ),
        # True now means what it says: the sample was testable and no material
        # shift was observed.  A material but unresolved shift is a distinct
        # status and remains below the calibrated veto threshold.
        stationary=(
            testable
            and shift is not None
            and abs(shift) < materiality_pct
        ),
        required_for_promotion=PROMOTION_REQUIRES_STATIONARITY,
    )


def _reset_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    """The shape a window takes once a counter reset has invalidated it.

    Every derived quantity is zeroed rather than guessed: a window spanning a
    restart has no defensible throughput or acceptance, and reporting one would
    make an unmeasurable run look measured.
    """
    return MetricsDelta(
        reset_detected=True,
        acceptance_available=before.has_draft_counters and after.has_draft_counters,
        drafted_tokens=0.0,
        accepted_tokens=0.0,
        acceptance_rate=0.0,
        mean_accepted_length=0.0,
        tpot_ms=0.0,
        output_tok_per_sec=0.0,
    )


def _safe_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    try:
        return compute_delta(before, after)
    except CounterResetError:
        return _reset_delta(before, after)


def _warmup_runs(
    stock: Sequence[ReplayResult | None],
    candidate: Sequence[ReplayResult | None],
) -> int:
    """Unscored passes each measurement window actually opened warm on.

    The minimum over every block of both arms, not the total: the quantity the
    decision records is "how many passes preceded a scored window", and an arm
    with two blocks of one warmup each opened both of its windows on one pass,
    not two.  A single missing warmup anywhere makes the answer zero, because
    that is the window whose numbers the reader has to distrust.
    """
    results = [*stock, *candidate]
    if not results or any(result is None for result in results):
        return 0
    return min(result.num_runs for result in results if result is not None)


def _warmup_dict(results: Sequence[ReplayResult | None]) -> dict[str, object] | None:
    """The arm's warmup passes across every block, or ``None`` if it ran none.

    Blocks are concatenated rather than reported separately so the shape is the
    same whether the arm ran one block or several; ``num_runs`` is then the
    arm's total unscored passes and ``tok_per_sec`` has one entry per pass, in
    the order they ran.
    """
    measured = [result for result in results if result is not None]
    if not measured:
        return None
    return {
        "num_runs": sum(result.num_runs for result in measured),
        "tok_per_sec": [
            run.output_tok_per_sec
            for result in measured
            for run in result.run_results
        ],
    }


def _engine_execution_dict(
    execution: EngineExecution | None,
) -> dict[str, object]:
    """Describe the engine regime, including when it is not known.

    Always returns a dict with ``execution_mode`` populated -- an omitted key
    would be indistinguishable from a run that forgot to record it, whereas
    :data:`speedlm.gate.decide.UNRECORDED_EXECUTION_MODE` says so.
    """

    if execution is None:
        return {
            "execution_mode": UNRECORDED_EXECUTION_MODE,
            "enforce_eager": None,
            "enable_chunked_prefill": None,
            "enable_prefix_caching": None,
            "max_num_seqs": None,
        }
    return {
        "execution_mode": execution.execution_mode,
        "enforce_eager": execution.enforce_eager,
        "enable_chunked_prefill": execution.enable_chunked_prefill,
        "enable_prefix_caching": execution.enable_prefix_caching,
        "max_num_seqs": execution.max_num_seqs,
    }


def _delta_dict(delta: MetricsDelta) -> dict[str, object]:
    return {
        "reset_detected": delta.reset_detected,
        "acceptance_available": delta.acceptance_available,
        "drafted_tokens": delta.drafted_tokens,
        "accepted_tokens": delta.accepted_tokens,
        "acceptance_rate": delta.acceptance_rate,
        "mean_accepted_length": delta.mean_accepted_length,
        "tpot_ms": delta.tpot_ms,
        "output_tok_per_sec": delta.output_tok_per_sec,
    }


def _gate_result(
    *,
    passed: bool,
    reason: str,
    metrics: dict[str, object],
    metrics_bodies: dict[str, str] | None = None,
    decision: Decision | None = None,
    failure: GateFailure | None = None,
) -> GateResult:
    # Importing GateResult at module load time would cycle while the
    # orchestrator imports speedlm.gate.decide through this package.
    from speedlm.tuner.orchestrator import GateResult

    return GateResult(
        passed=passed,
        reason=reason,
        metrics=metrics,
        metrics_bodies=dict(metrics_bodies or {}),
        decision=decision,
        failure=failure,
    )
