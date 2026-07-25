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
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.decide import Decision, Verdict, decide_promotion
from speedlm.gate.metrics import (
    CounterResetError,
    MetricsDelta,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)
from speedlm.gate.replay import ReplayResult, replay_suite
from speedlm.gate.suite import (
    BenchmarkSuite,
    SuiteError,
    build_suite,
    load_suite,
    persist_suite,
)
from speedlm.traces.store import TraceRecord

if TYPE_CHECKING:
    from speedlm.tuner.orchestrator import GateResult

AbortCheck = Callable[[], bool]
DraftReference = Path | str
Clock = Callable[[], float]
_T = TypeVar("_T")


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
    ) -> ReplayResult: ...


class BenchmarkAborted(RuntimeError):
    """Raised internally when serving activity preempts a benchmark."""


class BenchmarkTimedOut(TimeoutError):
    """Raised internally when the benchmark's whole-run deadline expires."""


class HttpReplayExecutor:
    """Run the existing async HTTP replay implementation behind a protocol."""

    _POLL_SECONDS = 0.05

    def replay(
        self,
        suite: BenchmarkSuite,
        endpoint_url: str,
        sampling: SamplingConfig,
        *,
        repeats: int,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> ReplayResult:
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


class BenchmarkGateRunner:
    """Build/load a suite, measure both draft arms, and decide promotion."""

    def __init__(
        self,
        *,
        config: SpeedLMConfig,
        trace_source: TraceSource,
        suite_dir: Path,
        stock_draft: DraftReference,
        endpoint: DraftEndpoint,
        metrics_source: MetricsSource,
        replay_executor: ReplayExecutor | None = None,
        repeats: int = 3,
        held_out_fraction: float = 0.2,
        training_context_hashes: set[str] | frozenset[str] | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
            raise ValueError("repeats must be an integer >= 3")
        if (
            isinstance(held_out_fraction, bool)
            or not isinstance(held_out_fraction, (int, float))
            or not 0 <= held_out_fraction <= 1
        ):
            raise ValueError("held_out_fraction must be a number in [0, 1]")
        self._config = config
        self._trace_source = trace_source
        self._suite_dir = suite_dir
        self._stock_draft = stock_draft
        self._endpoint = endpoint
        self._metrics_source = metrics_source
        self._replay_executor = replay_executor or HttpReplayExecutor()
        self._repeats = repeats
        self._held_out_fraction = float(held_out_fraction)
        self._training_context_hashes = (
            None
            if training_context_hashes is None
            else frozenset(training_context_hashes)
        )
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

        try:
            suite = stage("suite preparation", lambda _timeout: self._load_or_build_suite())
            stage(
                "suite leakage check",
                lambda _timeout: self._check_suite_leakage(suite),
            )

            stage(
                "stock activation",
                lambda available: self._endpoint.activate(
                    self._stock_draft,
                    timeout_seconds=available,
                    should_abort=should_abort,
                ),
            )
            stock_before = stage(
                "stock metrics pre-scrape",
                lambda available: self._scrape(available, should_abort),
            )
            stock_replay = stage(
                "stock replay",
                lambda available: self._replay(suite, available, should_abort),
            )
            stock_after = stage(
                "stock metrics post-scrape",
                lambda available: self._scrape(available, should_abort),
            )
            stock_delta = stage(
                "stock metric delta",
                lambda _timeout: _safe_delta(stock_before, stock_after),
            )

            stage(
                "candidate activation",
                lambda available: self._endpoint.activate(
                    candidate_draft,
                    timeout_seconds=available,
                    should_abort=should_abort,
                ),
            )
            candidate_before = stage(
                "candidate metrics pre-scrape",
                lambda available: self._scrape(available, should_abort),
            )
            candidate_replay = stage(
                "candidate replay",
                lambda available: self._replay(suite, available, should_abort),
            )
            candidate_after = stage(
                "candidate metrics post-scrape",
                lambda available: self._scrape(available, should_abort),
            )
            candidate_delta = stage(
                "candidate metric delta",
                lambda _timeout: _safe_delta(candidate_before, candidate_after),
            )

            decision = stage(
                "promotion decision",
                lambda _timeout: decide_promotion(
                    stock_delta,
                    candidate_delta,
                    stock_replay,
                    candidate_replay,
                    self._config.promotion,
                ),
            )
        except BenchmarkAborted as exc:
            return _gate_result(
                passed=False,
                reason=str(exc),
                metrics={"aborted": True},
            )
        except (BenchmarkTimedOut, TimeoutError) as exc:
            return _gate_result(
                passed=False,
                reason=str(exc) or "benchmark timed out",
                metrics={"timed_out": True},
            )

        return _gate_result(
            passed=decision.verdict is Verdict.PROMOTE,
            reason=decision.reason.value,
            metrics={
                "suite_hash": suite.suite_hash,
                "num_contexts": len(suite.contexts),
                "num_repeats": self._repeats,
                "stock": _delta_dict(stock_delta),
                "candidate": _delta_dict(candidate_delta),
            },
            decision=decision,
        )

    def _load_or_build_suite(self) -> BenchmarkSuite:
        manifest = self._suite_dir / "suite_manifest.json"
        if manifest.exists():
            return load_suite(self._suite_dir)
        suite = build_suite(
            tuple(self._trace_source.iter_records()),
            held_out_fraction=self._held_out_fraction,
        )
        persist_suite(suite, self._suite_dir)
        return suite

    def _check_suite_leakage(self, suite: BenchmarkSuite) -> None:
        if self._training_context_hashes is None:
            raise SuiteError(
                "Cannot prove benchmark suite is held out: "
                "training context hashes were not provided"
            )
        overlaps = suite.check_leakage(set(self._training_context_hashes))
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
    ) -> MetricsSnapshot:
        text = self._metrics_source.scrape(
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )
        return parse_metrics(text)

    def _replay(
        self,
        suite: BenchmarkSuite,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> ReplayResult:
        return self._replay_executor.replay(
            suite,
            self._endpoint.url,
            self._config.sampling,
            repeats=self._repeats,
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )


GateRunner = BenchmarkGateRunner


def _safe_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    try:
        return compute_delta(before, after)
    except CounterResetError:
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
    decision: Decision | None = None,
) -> GateResult:
    # Importing GateResult at module load time would cycle while the
    # orchestrator imports speedlm.gate.decide through this package.
    from speedlm.tuner.orchestrator import GateResult

    return GateResult(
        passed=passed,
        reason=reason,
        metrics=metrics,
        decision=decision,
    )
