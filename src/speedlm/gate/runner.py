"""Concrete orchestration for the stock-versus-candidate benchmark gate.

The runner owns no vLLM process and performs no direct metrics I/O.  Process
switching, metrics scraping, and replay execution are injected behind small
protocols so the orchestration can be exercised without a GPU or network.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.decide import Decision, Verdict, decide_promotion
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

#: Suite passes the correctness check makes per arm.
#:
#: One.  The check is a predicate about the head, not a statistic about the
#: machine, so repeating it buys resolution on a quantity that has none -- and
#: every extra pass is a fresh draw against the benign-divergence hazard, i.e.
#: it strictly *increases* the false-reject rate without improving detection.
CORRECTNESS_REPEATS: Final = 1


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
    ) -> None: ...


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
        correctness_max_tokens: int = DEFAULT_CORRECTNESS_MAX_TOKENS,
        benchmark_max_tokens: int = DEFAULT_BENCHMARK_MAX_TOKENS,
        held_out_fraction: float = 0.2,
        training_context_hashes: TrainingHashes | None = None,
        candidate_arm_first: bool = False,
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

            def activate(arm: str, draft: DraftReference) -> None:
                stage(
                    f"{arm} activation",
                    lambda available: self._endpoint.activate(
                        draft,
                        timeout_seconds=available,
                        should_abort=should_abort,
                    ),
                )

            def run_arm(
                arm: str, draft: DraftReference
            ) -> tuple[ReplayResult | None, _ArmMeasurement]:
                activate(arm, draft)
                # Warm the freshly restarted engine before the measurement
                # window opens.  Activation restarts vLLM, so the first suite
                # pass pays one-time JIT compilation of the speculative-decoding
                # kernels; that cost belongs to neither arm's steady state.  The
                # warmup runs *before* the pre-scrape, which keeps it out of the
                # Prometheus window as well as out of ``per_repeat``.  It is
                # also what keeps the two arms comparable when the arm order is
                # reversed: each arm pays its own cold start.
                warmup = stage(
                    f"{arm} warmup",
                    lambda available: self._warmup(suite, available, should_abort),
                )
                return warmup, self._measure_arm(arm, suite, stage, scrape, should_abort)

            # Resolved once, here, rather than at construction: what counts as
            # "stock" is whatever is serving when this benchmark runs, and both
            # arms of one benchmark must agree on it even if a promotion lands
            # between the two activations.
            stock_draft = self._resolve_stock_draft()

            # Running the candidate first reuses the engine that the cycle's
            # CANDIDATE_STARTING phase already built, and leaves the benchmark
            # ending on stock -- which is what a rejection wants serving anyway.
            # See ``tuning.benchmark_candidate_arm_first``.
            if self._candidate_arm_first:
                candidate_warmup, candidate_arm = run_arm("candidate", candidate_draft)
                stock_warmup, stock_arm = run_arm("stock", stock_draft)
            else:
                stock_warmup, stock_arm = run_arm("stock", stock_draft)
                candidate_warmup, candidate_arm = run_arm("candidate", candidate_draft)

            decision = stage(
                "promotion decision",
                lambda _timeout: decide_promotion(
                    stock_arm.pooled_delta,
                    candidate_arm.pooled_delta,
                    stock_arm.replay,
                    candidate_arm.replay,
                    self._config.promotion,
                    warmup_repeats=_warmup_runs(stock_warmup, candidate_warmup),
                    stock_repeat_metrics=stock_arm.repeat_deltas,
                    candidate_repeat_metrics=candidate_arm.repeat_deltas,
                    stock_correctness=stock_arm.correctness,
                    candidate_correctness=candidate_arm.correctness,
                    # Measurement context.  These were already published on
                    # ``GateResult.metrics``, which nothing persists, so a
                    # decision.json could not say what cap, degree, suite or
                    # baseline produced the numbers it reports.
                    benchmark_max_tokens=self._benchmark_max_tokens,
                    replay_concurrency=self._replay_concurrency,
                    correctness_max_tokens=self._correctness_max_tokens,
                    num_contexts=len(suite.contexts),
                    stock_draft=str(stock_draft),
                ),
            )
            # Candidate-first ends the benchmark on the *stock* draft, which is
            # exactly what a rejection wants left serving -- that is where the
            # rollback restart goes away.  A promotion is the one case that
            # still needs the candidate back: the orchestrator promotes by
            # flipping the durable pointer and waking, never by restarting, so
            # whatever is running when this returns is what serves live traffic.
            #
            # It is inside the try on purpose.  If the candidate cannot be put
            # back, the abort/timeout handlers below report a non-passing gate,
            # which is the fail-closed answer: a candidate that cannot be made
            # to serve must not be promoted.
            if self._candidate_arm_first and decision.verdict is Verdict.PROMOTE:
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
            passed=decision.verdict is Verdict.PROMOTE,
            reason=decision.reason.value,
            metrics={
                "suite_hash": suite.suite_hash,
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
                },
                "stock_runs": stock_arm.replay.num_runs,
                "candidate_runs": candidate_arm.replay.num_runs,
                # Unscored, but not invisible: the warmup pass is reported so a
                # reader can see what was excluded and how slow it was.
                "warmup": {
                    "requested_repeats": self._warmup_repeats,
                    "stock": _warmup_dict(stock_warmup),
                    "candidate": _warmup_dict(candidate_warmup),
                },
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
            warmup_repeats=self._warmup_repeats,
            concurrency=self._replay_concurrency,
            correctness_repeats=CORRECTNESS_REPEATS,
            benchmark_max_tokens=self._benchmark_max_tokens,
            correctness_max_tokens=self._correctness_max_tokens,
        )

    def _measure_arm(
        self,
        arm: str,
        suite: BenchmarkSuite,
        stage: Callable[[str, Callable[[float], object]], object],
        scrape: Callable[[str, str], MetricsSnapshot],
        should_abort: AbortCheck,
    ) -> _ArmMeasurement:
        """Run one arm's scored repeats and its correctness pass.

        The scrape schedule is one before the first repeat and one after each
        repeat -- ``repeats + 1`` scrapes rather than 2, which costs no extra
        suite passes and no extra generations.  Consecutive pairs give a real
        per-repeat acceptance sample; the first and last give the pooled window
        the gate used to report.  Before this, the repeat loop lived inside a
        single ``replay(repeats=N)`` call with the scrapes outside it, so there
        was exactly one acceptance measurement per arm no matter how many
        repeats ran, and it was stamped into all N per-repeat rows.

        The correctness pass runs *after* the last scrape, deliberately: it is
        bounded and single-stream, so folding it into the measurement window
        would contaminate both the acceptance counters and the decode-time
        throughput with traffic that is not the workload being measured.
        """
        snapshots = [scrape(f"{arm} metrics pre-scrape", f"{arm}-before")]
        runs: list[RunResults] = []
        for index in range(self._repeats):
            replay = stage(
                f"{arm} replay repeat {index}",
                lambda available: self._replay(suite, available, should_abort, repeats=1),
            )
            assert isinstance(replay, ReplayResult)
            runs.extend(replay.run_results)
            label = (
                f"{arm}-after"
                if index == self._repeats - 1
                else f"{arm}-after-repeat-{index}"
            )
            snapshots.append(scrape(f"{arm} metrics scrape after repeat {index}", label))

        correctness = stage(
            f"{arm} correctness pass",
            lambda available: self._correctness_replay(suite, available, should_abort),
        )
        assert isinstance(correctness, ReplayResult)

        repeat_deltas = tuple(
            _safe_delta(before, after)
            for before, after in zip(snapshots[:-1], snapshots[1:], strict=True)
        )
        # A reset inside any repeat window invalidates the arm, even when the
        # endpoints of the pooled window happen to look monotone across it.
        # Sampling more finely must not make the gate blinder.
        pooled = (
            _reset_delta(snapshots[0], snapshots[-1])
            if any(d.reset_detected for d in repeat_deltas)
            else _safe_delta(snapshots[0], snapshots[-1])
        )
        return _ArmMeasurement(
            replay=ReplayResult(
                run_results=tuple(runs),
                num_runs=len(runs),
                suite_hash=suite.suite_hash,
            ),
            repeat_deltas=repeat_deltas,
            pooled_delta=pooled,
            correctness=correctness,
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
    ) -> ReplayResult:
        """One bounded, single-stream pass whose only job is output agreement.

        The concurrency is pinned here rather than passed through from config
        on purpose: ``benchmark_concurrency`` is a throughput knob, and letting
        it reach this pass is exactly the conflation that made job 369005's
        equality check unsound.
        """
        return self._replay_executor.replay(
            suite,
            self._endpoint.url,
            self._config.sampling,
            repeats=CORRECTNESS_REPEATS,
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
            concurrency=CORRECTNESS_REPLAY_CONCURRENCY,
            max_tokens=self._correctness_max_tokens,
            capture_tokens=True,
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
    stock: ReplayResult | None,
    candidate: ReplayResult | None,
) -> int:
    """Warmup passes both arms actually completed, as observed."""
    if stock is None or candidate is None:
        return 0
    return min(stock.num_runs, candidate.num_runs)


def _warmup_dict(result: ReplayResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "num_runs": result.num_runs,
        "tok_per_sec": [run.output_tok_per_sec for run in result.run_results],
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
