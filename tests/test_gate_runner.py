"""CPU-only tests for the concrete benchmark-gate runner."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.decide import (
    CUDA_GRAPH_EXECUTION_MODE,
    EAGER_EXECUTION_MODE,
    UNRECORDED_EXECUTION_MODE,
    EngineExecution,
    Reason,
    Verdict,
)
from speedlm.gate.replay import ReplayResult, RequestResult, RunResults
from speedlm.gate.runner import (
    CORRECTNESS_CONTROL_REPEATS,
    CORRECTNESS_REPEATS,
    CORRECTNESS_REPLAY_CONCURRENCY,
    DEFAULT_BENCHMARK_MAX_TOKENS,
    DEFAULT_CORRECTNESS_MAX_TOKENS,
    DEFAULT_REPLAY_CONCURRENCY,
    BenchmarkGateRunner,
    HttpReplayExecutor,
)
from speedlm.gate.suite import BenchmarkSuite, FrozenContext, SuiteError
from speedlm.traces.store import TraceRecord
from speedlm.tuner.orchestrator import GateFailure, derive_benchmark_timeout


@dataclass
class FakeTraceSource:
    records: tuple[TraceRecord, ...]
    reads: int = 0

    def iter_records(self) -> Iterator[TraceRecord]:
        self.reads += 1
        yield from self.records


@dataclass
class FakeEndpoint:
    url: str = "http://not-used.test/"
    activations: list[Path | str] = field(default_factory=list)

    def activate(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0
        self.activations.append(draft)


@dataclass
class FakeMetricsSource:
    scrapes: list[str]
    events: list[str] | None = None

    def scrape(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> str:
        assert timeout_seconds > 0
        if self.events is not None:
            self.events.append("scrape")
        return self.scrapes.pop(0)


@dataclass
class FakeReplayExecutor:
    abort_after_first: list[bool] | None = None
    advance_after_first: Callable[[], None] | None = None
    #: Wall-clock seconds for each individual suite pass, popped in order
    #: across every replay call.  ``None`` means every pass takes 1.0s.
    run_latencies: list[float] | None = None
    events: list[str] | None = None
    calls: int = 0
    seen_repeats: list[int] = field(default_factory=list)
    seen_suite_ids: list[int] = field(default_factory=list)
    #: ``(concurrency, max_tokens, capture_tokens)`` per call, in call order.
    seen_options: list[tuple[int | None, int | None, bool]] = field(default_factory=list)
    #: Text each request should return, keyed by whether the call is the
    #: correctness pass.  Lets a test make the arms disagree on the correctness
    #: pass while agreeing on the throughput pass, which is the real shape.
    response_text: str = "same deterministic output"
    correctness_tokens: tuple[str, ...] | None = None
    #: ``(correctness_call_index, run_index, context_index) -> tokens``.
    #:
    #: The runner makes one correctness *call* per arm, and the stock arm's call
    #: carries more than one run because it also measures the noise floor.  A
    #: test that wants to model "the engine disagrees with itself here and the
    #: candidate disagrees there" needs to address those three axes separately,
    #: which a single ``correctness_tokens`` tuple cannot.
    correctness_streams: (
        Callable[[int, int, int], tuple[str, ...]] | None
    ) = None
    #: Correctness calls seen so far, i.e. the first axis above.
    correctness_calls: int = 0

    def replay(
        self,
        suite: BenchmarkSuite,
        endpoint_url: str,
        sampling: SamplingConfig,
        *,
        repeats: int,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
        concurrency: int | None = None,
        max_tokens: int | None = None,
        capture_tokens: bool = False,
    ) -> ReplayResult:
        assert endpoint_url == "http://not-used.test/"
        assert timeout_seconds > 0
        assert repeats >= 1
        assert sampling.temperature == 0.0
        assert sampling.top_p == 1.0
        assert sampling.seed == 0
        self.calls += 1
        self.seen_repeats.append(repeats)
        self.seen_suite_ids.append(id(suite))
        self.seen_options.append((concurrency, max_tokens, capture_tokens))
        if self.events is not None:
            self.events.append(
                f"correctness:{concurrency}" if capture_tokens else f"replay:{repeats}"
            )

        correctness_call = self.correctness_calls
        if capture_tokens:
            self.correctness_calls += 1

        runs: list[RunResults] = []
        for run_index in range(repeats):
            latency = (
                1.0 if self.run_latencies is None else self.run_latencies.pop(0)
            )
            requests: list[RequestResult] = []
            for ctx_index, ctx in enumerate(suite.contexts):
                if not capture_tokens:
                    tokens: tuple[str, ...] = ()
                elif self.correctness_streams is not None:
                    tokens = self.correctness_streams(
                        correctness_call, run_index, ctx_index
                    )
                else:
                    tokens = self.correctness_tokens or ()
                requests.append(
                    RequestResult(
                        context_hash=ctx.context_hash,
                        latency_s=latency / len(suite.contexts),
                        prompt_tokens=4,
                        completion_tokens=10,
                        total_tokens=14,
                        response_text=self.response_text,
                        valid=True,
                        output_tokens=tokens,
                    )
                )
            runs.append(
                RunResults(
                    results=tuple(requests),
                    total_latency_s=latency,
                    total_prompt_tokens=4 * len(requests),
                    total_completion_tokens=10 * len(requests),
                    valid_count=len(requests),
                    invalid_count=0,
                    invalid_rate=0.0,
                )
            )
        result = ReplayResult(
            run_results=tuple(runs),
            num_runs=repeats,
            suite_hash=suite.suite_hash,
        )
        if self.calls == 1 and self.abort_after_first is not None:
            self.abort_after_first[0] = True
        if self.calls == 1 and self.advance_after_first is not None:
            self.advance_after_first()
        return result


@dataclass
class FakeClock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _trace() -> TraceRecord:
    return TraceRecord(
        id="trace-1",
        timestamp=1.0,
        model="model",
        messages=({"role": "user", "content": "hello"},),
        tool_calls=(),
        temperature=0.9,
        top_p=0.5,
        seed=99,
        prompt_tokens=4,
        completion_tokens=2,
    )


def _snapshot(
    *,
    generated: float,
    elapsed_ns: float,
    accepted: float,
    rejected: float,
) -> str:
    drafted = accepted + rejected
    return "\n".join(
        (
            f'vllm:generation_tokens_total{{engine="0"}} {generated}',
            'vllm:prompt_tokens_total{engine="0"} 100',
            f'vllm:request_decode_time_seconds_sum{{engine="0"}} {elapsed_ns / 1e9}',
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {drafted}',
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}',
            f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafted}',
        )
    )


def _arm_ladder(
    *,
    start: tuple[float, float, float, float],
    totals: tuple[float, float, float, float],
    repeats: int,
    fractions: tuple[float, ...] | None = None,
    accepted_fractions: tuple[float, ...] | None = None,
) -> list[str]:
    """One scrape before the arm's first repeat and one after each repeat.

    The gate samples ``/metrics`` between repeats so acceptance is measured per
    repeat rather than once per arm, so a fake metrics source has to supply a
    ladder rather than two endpoints.  ``fractions`` says how much of *totals*
    has accrued by each rung, which is how a test injects real per-repeat
    acceptance variance; the default spreads the totals evenly, reproducing the
    old two-endpoint behaviour exactly at the pooled level.
    """
    if fractions is None:
        fractions = tuple((i + 1) / repeats for i in range(repeats))
    assert len(fractions) == repeats
    assert fractions[-1] == 1.0
    if accepted_fractions is None:
        accepted_fractions = fractions
    assert len(accepted_fractions) == repeats
    assert accepted_fractions[-1] == 1.0
    rungs = list(zip([0.0, *fractions], [0.0, *accepted_fractions], strict=True))
    return [
        _snapshot(
            generated=start[0] + totals[0] * f,
            elapsed_ns=start[1] + totals[1] * f,
            # Accrues on its own schedule so a test can give the arm real
            # per-repeat acceptance variance without touching its totals.
            accepted=start[2] + totals[2] * a,
            rejected=start[3] + totals[3] * f,
        )
        for f, a in rungs
    ]


def _normal_scrapes(
    *,
    candidate_generated: float = 120,
    candidate_elapsed_ns: float = 800_000_000,
    candidate_accepted: float = 80,
    candidate_rejected: float = 20,
    repeats: int = 3,
    stock_fractions: tuple[float, ...] | None = None,
    stock_accepted_fractions: tuple[float, ...] | None = None,
    candidate_fractions: tuple[float, ...] | None = None,
) -> list[str]:
    return [
        *_arm_ladder(
            start=(100, 1_000_000_000, 10, 10),
            totals=(100, 1_000_000_000, 60, 40),
            repeats=repeats,
            fractions=stock_fractions,
            accepted_fractions=stock_accepted_fractions,
        ),
        *_arm_ladder(
            start=(1_000, 10_000_000_000, 100, 100),
            totals=(
                candidate_generated,
                candidate_elapsed_ns,
                candidate_accepted,
                candidate_rejected,
            ),
            repeats=repeats,
            fractions=candidate_fractions,
        ),
    ]


def _runner(
    tmp_path: Path,
    *,
    scrapes: list[str],
    replay: FakeReplayExecutor | None = None,
    trace_source: FakeTraceSource | None = None,
    training_context_hashes: (
        set[str]
        | frozenset[str]
        | Callable[[], set[str] | frozenset[str]]
        | None
    ) = frozenset(),
    clock: Callable[[], float] | None = None,
    events: list[str] | None = None,
    candidate_arm_first: bool = False,
) -> tuple[BenchmarkGateRunner, FakeEndpoint, FakeReplayExecutor, FakeTraceSource]:
    endpoint = FakeEndpoint()
    replay_executor = replay or FakeReplayExecutor()
    if events is not None:
        replay_executor.events = events
    traces = trace_source or FakeTraceSource((_trace(),))
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=traces,
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=endpoint,
        metrics_source=FakeMetricsSource(scrapes, events=events),
        replay_executor=replay_executor,
        training_context_hashes=training_context_hashes,
        candidate_arm_first=candidate_arm_first,
        clock=clock or FakeClock(),
    )
    return runner, endpoint, replay_executor, traces


def test_passing_candidate_promotes_with_real_decision(tmp_path: Path) -> None:
    runner, endpoint, replay, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is True
    assert result.decision is not None
    assert result.decision.verdict is Verdict.PROMOTE
    assert result.decision.reason is Reason.BOTH_THRESHOLDS_MET
    assert endpoint.activations == ["stock", tmp_path / "candidate"]
    # Per arm: one warmup pass, three scored passes issued one at a time so
    # a scrape can sit between them, and one correctness pass.
    assert replay.calls == 10
    assert replay.seen_repeats == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    assert len(set(replay.seen_suite_ids)) == 1


def test_failing_candidate_rejects_with_measured_decision(tmp_path: Path) -> None:
    scrapes = _normal_scrapes(
        candidate_generated=90,
        candidate_elapsed_ns=1_000_000_000,
        candidate_accepted=50,
        candidate_rejected=50,
    )
    runner, _, _, _ = _runner(tmp_path, scrapes=scrapes)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert result.decision is not None
    assert result.decision.verdict is Verdict.REJECT
    assert result.decision.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD
    assert result.decision.acceptance_delta_pp == pytest.approx(-10.0)


def test_counter_reset_is_an_invalid_measurement(tmp_path: Path) -> None:
    scrapes = _normal_scrapes()
    scrapes[1] = _snapshot(
        generated=5,
        elapsed_ns=2_000_000_000,
        accepted=70,
        rejected=50,
    )
    runner, _, _, _ = _runner(tmp_path, scrapes=scrapes)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert result.decision is not None
    assert result.decision.reason is Reason.COUNTER_RESET
    # The reset zeroes the *Prometheus* window, which is the diagnostic
    # statistic.  The replay-derived gating figures survive it, so they keep
    # reporting what the arms actually did.
    assert result.decision.stock_prometheus_decode_tok_per_sec == 0.0
    assert result.decision.stock_avg_tok_per_sec > 0.0
    assert result.metrics["stock"]["reset_detected"] is True


def test_abort_mid_run_returns_promptly_without_verdict(tmp_path: Path) -> None:
    aborted = [False]
    replay = FakeReplayExecutor(abort_after_first=aborted)
    runner, endpoint, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: aborted[0],
    )

    assert result.passed is False
    assert result.decision is None
    assert result.metrics == {"aborted": True}
    assert replay.calls == 1
    assert endpoint.activations == ["stock"]


def test_whole_run_timeout_is_honored_without_verdict(tmp_path: Path) -> None:
    clock = FakeClock()
    replay = FakeReplayExecutor(advance_after_first=lambda: clock.advance(6))
    runner, endpoint, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
        clock=clock,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=5,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert result.decision is None
    assert result.metrics == {"timed_out": True}
    assert replay.calls == 1
    assert endpoint.activations == ["stock"]


def test_decision_repeat_provenance_is_internally_consistent(tmp_path: Path) -> None:
    runner, _, _, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert result.decision.num_repeats == len(result.decision.per_repeat)


def test_existing_frozen_suite_is_loaded_without_rereading_traces(tmp_path: Path) -> None:
    traces = FakeTraceSource((_trace(),))
    first, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        trace_source=traces,
    )
    first.benchmark(
        tmp_path / "candidate-a",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    second, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        trace_source=traces,
    )
    result = second.benchmark(
        tmp_path / "candidate-b",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert traces.reads == 1


def test_gate_refuses_to_run_without_training_provenance(tmp_path: Path) -> None:
    runner, endpoint, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        training_context_hashes=None,
    )

    with pytest.raises(SuiteError, match="Cannot prove benchmark suite is held out"):
        runner.benchmark(
            tmp_path / "candidate",
            timeout_seconds=30,
            should_abort=lambda: False,
        )

    assert endpoint.activations == []
    assert replay.calls == 0


def test_gate_fails_loudly_on_training_suite_overlap(tmp_path: Path) -> None:
    context_hash = FrozenContext.from_trace(_trace()).context_hash
    runner, endpoint, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        training_context_hashes={context_hash},
    )

    with pytest.raises(SuiteError, match="context leakage detected"):
        runner.benchmark(
            tmp_path / "candidate",
            timeout_seconds=30,
            should_abort=lambda: False,
        )

    assert endpoint.activations == []
    assert replay.calls == 0


def test_gate_resolves_callable_training_hashes_for_each_benchmark(
    tmp_path: Path,
) -> None:
    context_hash = FrozenContext.from_trace(_trace()).context_hash
    current_hashes: set[str] = set()
    provider_calls = 0

    def training_hashes() -> set[str]:
        nonlocal provider_calls
        provider_calls += 1
        return set(current_hashes)

    runner, endpoint, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        training_context_hashes=training_hashes,
    )
    assert provider_calls == 0

    first = runner.benchmark(
        tmp_path / "candidate-a",
        timeout_seconds=30,
        should_abort=lambda: False,
    )
    current_hashes.add(context_hash)

    with pytest.raises(SuiteError, match="context leakage detected"):
        runner.benchmark(
            tmp_path / "candidate-b",
            timeout_seconds=30,
            should_abort=lambda: False,
        )

    assert first.decision is not None
    assert provider_calls == 2
    assert endpoint.activations == ["stock", tmp_path / "candidate-a"]
    assert replay.calls == 10


def test_warmup_pass_runs_before_the_measurement_window(tmp_path: Path) -> None:
    events: list[str] = []
    runner, _, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        events=events,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    # Each arm warms up first, and only then does the metrics window open.
    per_arm = [
        "replay:1",  # warmup, outside the window
        "scrape",  # window opens
        "replay:1",
        "scrape",
        "replay:1",
        "scrape",
        "replay:1",
        "scrape",  # window closes
        "correctness:1",  # bounded, single-stream, outside the window
    ]
    assert events == per_arm * 2
    # Stock issues its correctness call with two runs -- the scored pass and the
    # noise-floor control -- so its call is the only one asking for 2.
    assert replay.seen_repeats == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]


def test_warmup_pass_is_excluded_from_the_scored_repeats(tmp_path: Path) -> None:
    # One slow cold pass per arm (JIT compilation), then steady state.  If the
    # warmup were scored, the 2.5 tok/s pass would show up in ``per_repeat``.
    replay = FakeReplayExecutor(
        # Stock: warmup, three scored passes, then two correctness passes
        # (scored + noise-floor control).  Candidate: warmup, three, one.
        run_latencies=[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
    )
    runner, _, replay, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    decision = result.decision
    assert decision.num_repeats == 3
    assert len(decision.per_repeat) == decision.num_repeats
    assert [r.stock_tok_per_sec for r in decision.per_repeat] == [10.0, 10.0, 10.0]
    assert [r.candidate_tok_per_sec for r in decision.per_repeat] == [10.0, 10.0, 10.0]


def test_warmup_measurement_stays_visible_in_the_report(tmp_path: Path) -> None:
    replay = FakeReplayExecutor(
        # Stock: warmup, three scored passes, then two correctness passes
        # (scored + noise-floor control).  Candidate: warmup, three, one.
        run_latencies=[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
    )
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    warmup = result.metrics["warmup"]
    assert isinstance(warmup, dict)
    assert warmup["requested_repeats"] == 1
    assert warmup["stock"] == {"num_runs": 1, "tok_per_sec": [2.5]}
    assert warmup["candidate"] == {"num_runs": 1, "tok_per_sec": [2.5]}


def test_warmup_can_be_disabled(tmp_path: Path) -> None:
    endpoint = FakeEndpoint()
    replay_executor = FakeReplayExecutor()
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=endpoint,
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=replay_executor,
        warmup_repeats=0,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert replay_executor.seen_repeats == [1, 1, 1, 2, 1, 1, 1, 1]
    warmup = result.metrics["warmup"]
    assert isinstance(warmup, dict)
    assert warmup == {"requested_repeats": 0, "stock": None, "candidate": None}


def test_warmup_repeats_must_be_a_non_negative_integer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_repeats must be an integer >= 0"):
        BenchmarkGateRunner(
            config=SpeedLMConfig(model="model"),
            trace_source=FakeTraceSource((_trace(),)),
            suite_dir=tmp_path / "suite",
            stock_draft="stock",
            endpoint=FakeEndpoint(),
            metrics_source=FakeMetricsSource([]),
            replay_executor=FakeReplayExecutor(),
            warmup_repeats=-1,
            training_context_hashes=frozenset(),
        )


def test_decision_records_the_warmup_it_excluded(tmp_path: Path) -> None:
    runner, _, _, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert result.decision.warmup_repeats == 1
    assert result.decision.to_dict()["warmup_repeats"] == 1
    assert result.decision.to_dict()["num_repeats"] == 3


def test_every_scrape_body_is_returned_verbatim(tmp_path: Path) -> None:
    scrapes = _normal_scrapes()
    expected = list(scrapes)
    runner, _, _, _ = _runner(tmp_path, scrapes=scrapes)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    # One scrape opens each arm's window and one closes it after every
    # repeat, so the archive keeps every window the acceptance vector was
    # derived from -- not just the two endpoints of the pooled one.
    assert dict(result.metrics_bodies) == {
        "stock-before": expected[0],
        "stock-after-repeat-0": expected[1],
        "stock-after-repeat-1": expected[2],
        "stock-after": expected[3],
        "candidate-before": expected[4],
        "candidate-after-repeat-0": expected[5],
        "candidate-after-repeat-1": expected[6],
        "candidate-after": expected[7],
    }


def test_scrape_bodies_survive_an_abort(tmp_path: Path) -> None:
    scrapes = _normal_scrapes()
    expected = list(scrapes)
    runner, _, replay, _ = _runner(tmp_path, scrapes=scrapes)

    # Abort after the scored stock replay, i.e. once one scrape has happened.
    def should_abort() -> bool:
        return replay.calls >= 2

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=should_abort,
    )

    assert result.decision is None
    assert dict(result.metrics_bodies) == {"stock-before": expected[0]}


# ---------------------------------------------------------------------------
# Replay concurrency
# ---------------------------------------------------------------------------


def test_default_replay_executor_is_concurrent() -> None:
    """The default executor must not fall back to one request at a time."""
    assert DEFAULT_REPLAY_CONCURRENCY > 1
    assert HttpReplayExecutor().concurrency == DEFAULT_REPLAY_CONCURRENCY


def test_replay_concurrency_reaches_the_default_executor(tmp_path: Path) -> None:
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_concurrency=4,
        training_context_hashes=frozenset(),
    )

    executor = runner._replay_executor
    assert isinstance(executor, HttpReplayExecutor)
    assert executor.concurrency == 4


@pytest.mark.parametrize("concurrency", [0, -1, True])
def test_invalid_replay_concurrency_is_rejected(
    tmp_path: Path,
    concurrency: object,
) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        HttpReplayExecutor(concurrency=concurrency)


def test_metrics_record_the_concurrency_both_arms_ran_at(tmp_path: Path) -> None:
    """Absolute tok/s is only comparable across runs at the same degree."""
    runner, _, _, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.metrics["replay_concurrency"] == DEFAULT_REPLAY_CONCURRENCY


def test_executor_without_a_declared_degree_reports_the_configured_one(
    tmp_path: Path,
) -> None:
    """Report what ran where the executor says so, the request otherwise."""
    runner, _, _, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    # FakeReplayExecutor exposes no ``concurrency``, so the configured default
    # stands in rather than a fabricated number.
    assert result.metrics["replay_concurrency"] == DEFAULT_REPLAY_CONCURRENCY


# ---------------------------------------------------------------------------
# Typed gate failures
# ---------------------------------------------------------------------------


def test_abort_carries_a_typed_failure(tmp_path: Path) -> None:
    aborted = [False]
    replay = FakeReplayExecutor(abort_after_first=aborted)
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: aborted[0],
    )

    assert result.failure is GateFailure.ABORTED


def test_timeout_carries_a_typed_failure(tmp_path: Path) -> None:
    clock = FakeClock()
    replay = FakeReplayExecutor(advance_after_first=lambda: clock.advance(6))
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
        clock=clock,
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=5,
        should_abort=lambda: False,
    )

    assert result.failure is GateFailure.TIMED_OUT


def test_a_measured_rejection_carries_no_failure(tmp_path: Path) -> None:
    """A rejection is a result; only a non-measurement is a failure."""
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(candidate_accepted=10, candidate_rejected=90),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert result.decision is not None
    assert result.failure is None


# ---------------------------------------------------------------------------
# Derived benchmark deadline
# ---------------------------------------------------------------------------


def test_estimated_benchmark_seconds_scales_with_the_suite(tmp_path: Path) -> None:
    runner, _, _, _ = _runner(tmp_path, scrapes=_normal_scrapes())

    estimate = runner.estimated_benchmark_seconds()

    assert estimate is not None
    assert estimate == derive_benchmark_timeout(
        num_contexts=1,
        repeats=3,
        warmup_repeats=1,
        concurrency=DEFAULT_REPLAY_CONCURRENCY,
        correctness_repeats=CORRECTNESS_REPEATS + CORRECTNESS_CONTROL_REPEATS,
        benchmark_max_tokens=DEFAULT_BENCHMARK_MAX_TOKENS,
        correctness_max_tokens=DEFAULT_CORRECTNESS_MAX_TOKENS,
    )


def test_estimated_benchmark_seconds_is_none_when_no_suite_can_be_built(
    tmp_path: Path,
) -> None:
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        trace_source=FakeTraceSource(()),
    )

    assert runner.estimated_benchmark_seconds() is None


# ---------------------------------------------------------------------------
# Output-correctness pass
# ---------------------------------------------------------------------------


def test_correctness_pass_runs_single_stream_whatever_the_benchmark_degree(
    tmp_path: Path,
) -> None:
    """Concurrency is a throughput knob and must not reach the equality check.

    Job 369005 ran the equality check inside the concurrency-8 throughput pass,
    where the per-token divergence hazard is ~12x what it is single-stream, and
    rejected a candidate on 82-87 "mismatches" that were the scheduler's doing.
    """
    replay_executor = FakeReplayExecutor()
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=replay_executor,
        replay_concurrency=32,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    correctness_calls = [
        options for options in replay_executor.seen_options if options[2]
    ]
    assert correctness_calls == [
        (CORRECTNESS_REPLAY_CONCURRENCY, DEFAULT_CORRECTNESS_MAX_TOKENS, True)
    ] * 2
    assert CORRECTNESS_REPLAY_CONCURRENCY == 1
    # Every other pass left the degree to the executor and carried the
    # throughput cap, which is a different number from the correctness cap.
    assert [o for o in replay_executor.seen_options if not o[2]] == [
        (None, DEFAULT_BENCHMARK_MAX_TOKENS, False)
    ] * 8
    assert result.metrics["correctness"] == {
        "control_repeats": 1,
        "control_arm": "stock",
        "concurrency": 1,
        "max_tokens": DEFAULT_CORRECTNESS_MAX_TOKENS,
        "repeats": CORRECTNESS_REPEATS,
    }


def test_correctness_max_tokens_is_configurable(tmp_path: Path) -> None:
    replay_executor = FakeReplayExecutor()
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=replay_executor,
        correctness_max_tokens=64,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert {o[1] for o in replay_executor.seen_options if o[2]} == {64}


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_correctness_max_tokens_is_rejected(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="correctness_max_tokens"):
        BenchmarkGateRunner(
            config=SpeedLMConfig(model="model"),
            trace_source=FakeTraceSource((_trace(),)),
            suite_dir=tmp_path / "suite",
            stock_draft="stock",
            endpoint=FakeEndpoint(),
            metrics_source=FakeMetricsSource([]),
            replay_executor=FakeReplayExecutor(),
            correctness_max_tokens=value,
            training_context_hashes=frozenset(),
        )


# ---------------------------------------------------------------------------
# Throughput/acceptance output cap
# ---------------------------------------------------------------------------


def test_the_throughput_pass_carries_an_explicit_output_cap(tmp_path: Path) -> None:
    """Job 369005's throughput pass sent no cap and ran to the model-len bound.

    That is not "uncapped", it is capped by the served model: gpt-oss averaged
    2091 completion tokens with 25.6% truncated at length, Qwen 1602 with 3.5%.
    An explicit uniform cap replaces a model-dependent implicit one.
    """
    replay_executor = FakeReplayExecutor()
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay_executor,
    )

    runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    throughput_caps = {o[1] for o in replay_executor.seen_options if not o[2]}
    assert throughput_caps == {DEFAULT_BENCHMARK_MAX_TOKENS}
    assert None not in throughput_caps


def test_benchmark_max_tokens_is_configurable_and_reported(tmp_path: Path) -> None:
    replay_executor = FakeReplayExecutor()
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=replay_executor,
        benchmark_max_tokens=256,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert {o[1] for o in replay_executor.seen_options if not o[2]} == {256}
    assert result.metrics["benchmark_max_tokens"] == 256


def test_the_two_passes_do_not_share_one_cap(tmp_path: Path) -> None:
    """The correctness cap must not leak into the throughput pass, or back."""
    replay_executor = FakeReplayExecutor()
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=replay_executor,
        benchmark_max_tokens=256,
        correctness_max_tokens=64,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert {o[1] for o in replay_executor.seen_options if not o[2]} == {256}
    assert {o[1] for o in replay_executor.seen_options if o[2]} == {64}


def test_the_gate_config_cap_is_what_the_composed_runner_sends() -> None:
    """The knob operators set is the number that reaches the payload."""
    config = SpeedLMConfig.from_dict(
        {"model": "org/model", "tuning": {"benchmark_max_tokens": 300}}
    )

    assert config.tuning.benchmark_max_tokens == 300
    assert config.to_dict()["tuning"]["benchmark_max_tokens"] == 300


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_benchmark_max_tokens_is_rejected(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="benchmark_max_tokens"):
        BenchmarkGateRunner(
            config=SpeedLMConfig(model="model"),
            trace_source=FakeTraceSource((_trace(),)),
            suite_dir=tmp_path / "suite",
            stock_draft="stock",
            endpoint=FakeEndpoint(),
            metrics_source=FakeMetricsSource([]),
            replay_executor=FakeReplayExecutor(),
            benchmark_max_tokens=value,
            training_context_hashes=frozenset(),
        )


#: Traces that freeze into a forty-context held-out suite at the runner's
#: default ``held_out_fraction`` of 0.2.  The divergence criterion compares a
#: rate against a rate, so a one-context suite cannot exercise it at all.
_DIVERGENCE_SUITE_TRACES = 200

_BASE_TOKENS = tuple(f"t{i}" for i in range(64))


def _diverging_traces() -> FakeTraceSource:
    return FakeTraceSource(
        tuple(
            TraceRecord(
                id=f"trace-{i}",
                timestamp=float(i),
                model="model",
                messages=({"role": "user", "content": f"hello {i}"},),
                tool_calls=(),
                temperature=0.9,
                top_p=0.5,
                seed=99,
                prompt_tokens=4,
                completion_tokens=2,
            )
            for i in range(_DIVERGENCE_SUITE_TRACES)
        )
    )


def _correctness_streams(
    *,
    candidate_at: int | None = None,
    candidate_contexts: int = 10**6,
    control_at: int | None = None,
    control_contexts: int = 10**6,
) -> Callable[[int, int, int], tuple[str, ...]]:
    """Token streams for the two correctness calls the runner makes.

    Call 0 is the stock arm.  Its run 0 is the scored pass and its run 1 is the
    noise-floor control -- the same engine replaying the same suite a second
    time -- so ``control_at`` is where *the engine* disagrees with itself.  Call
    1 is the candidate arm, and ``candidate_at`` is where it disagrees with
    stock.
    """

    def _part(at: int, tail: str) -> tuple[str, ...]:
        return _BASE_TOKENS[:at] + (tail,) * (len(_BASE_TOKENS) - at)

    def factory(call: int, run: int, ctx: int) -> tuple[str, ...]:
        if call == 0 and run == 0:
            return _BASE_TOKENS
        if call == 0:
            if control_at is None or ctx >= control_contexts:
                return _BASE_TOKENS
            return _part(control_at, "NOISE")
        if candidate_at is None or ctx >= candidate_contexts:
            return _BASE_TOKENS
        return _part(candidate_at, "OTHER")

    return factory


def _divergence_runner(
    tmp_path: Path, streams: Callable[[int, int, int], tuple[str, ...]]
) -> BenchmarkGateRunner:
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=FakeReplayExecutor(correctness_streams=streams),
        trace_source=_diverging_traces(),
    )
    return runner


def test_an_early_divergence_on_the_correctness_pass_rejects(tmp_path: Path) -> None:
    """Every context parts at token 2 while the engine agrees with itself."""
    runner = _divergence_runner(
        tmp_path, _correctness_streams(candidate_at=2)
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert result.decision is not None
    assert result.decision.reason is Reason.OUTPUT_MISMATCH
    assert result.decision.output_early_divergences == 40
    assert result.decision.control_early_divergences == 0
    assert {
        d.first_divergence_index for d in result.decision.output_divergences
    } == {2}
    assert result.decision.to_dict()["output_divergences"][0]["basis"] == "token"


def test_a_divergence_the_engine_also_makes_against_itself_does_not_reject(
    tmp_path: Path,
) -> None:
    """The false reject this whole control pass exists to stop.

    Byte-identical evidence to the test above -- forty contexts parting at token
    2, well inside ``min_divergence_token_index`` -- but here the stock arm
    replayed against *itself* parts in exactly the same places.  Speculative
    decoding is lossless, so that is the only thing this measurement can be:
    the engine, not the head.  The old rule could not tell the two apart
    because it never ran the second pass.
    """
    runner = _divergence_runner(
        tmp_path, _correctness_streams(candidate_at=2, control_at=2)
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    decision = result.decision
    assert decision.output_early_divergences == 40
    assert decision.control_early_divergences == 40
    assert decision.verdict is Verdict.PROMOTE
    assert decision.reason is Reason.BOTH_THRESHOLDS_MET
    assert result.passed is True


def test_the_stock_arm_measures_the_noise_floor_and_the_record_says_so(
    tmp_path: Path,
) -> None:
    """The control has to be real, on the stock arm, and auditable."""
    replay = FakeReplayExecutor(
        correctness_streams=_correctness_streams(control_at=30, control_contexts=7)
    )
    runner, _, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        replay=replay,
        trace_source=_diverging_traces(),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    decision = result.decision
    # Exactly one arm ran an extra correctness pass, and it was stock -- the
    # first arm, and the reference side of every comparison.
    assert replay.seen_repeats == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    assert result.metrics["correctness"]["control_arm"] == "stock"
    assert decision.divergence_control_available is True
    assert decision.divergence_trials == 40
    assert decision.control_trials == 40
    assert decision.control_total_divergences == 7
    assert decision.control_divergence_rate == pytest.approx(7 / 40)
    # The candidate agreed with stock everywhere, so the engine is measurably
    # noisier than the head is, and nothing gates.
    assert decision.output_total_divergences == 0
    assert decision.verdict is Verdict.PROMOTE


# ---------------------------------------------------------------------------
# Acceptance is sampled per repeat
# ---------------------------------------------------------------------------


def test_acceptance_is_measured_once_per_repeat_not_once_per_arm(
    tmp_path: Path,
) -> None:
    """The per-repeat rows must carry three measurements, not one repeated.

    Job 369005 reported ``0.40302518489174166`` bit-identically in all five
    rows because the whole repeat loop sat inside a single ``replay`` call with
    the two scrapes outside it.
    """
    # Front-load the stock arm's accepted tokens so the three repeat windows
    # genuinely differ, and keep the pooled totals unchanged.
    scrapes = _normal_scrapes(stock_accepted_fractions=(0.5, 0.75, 1.0))
    runner, _, _, _ = _runner(tmp_path, scrapes=scrapes)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    rates = [r.stock_acceptance_rate for r in result.decision.per_repeat]
    assert len(rates) == 3
    assert len(set(rates)) == 3
    # ... and the published mean really is the mean of that column.
    assert result.decision.stock_avg_acceptance == pytest.approx(
        sum(rates) / len(rates)
    )
    assert result.decision.stock_acceptance_stdev > 0.0
    # The raw per-repeat windows land in the metrics beside the pooled one.
    per_repeat_windows = result.metrics["stock_per_repeat"]
    assert isinstance(per_repeat_windows, list)
    assert len(per_repeat_windows) == 3


def test_a_reset_inside_one_repeat_window_invalidates_the_arm(
    tmp_path: Path,
) -> None:
    """Sampling more finely must not make the gate blinder to a restart."""
    scrapes = _normal_scrapes()
    # Counters go backwards inside repeat 1 and recover by the last scrape, so
    # the pooled window on its own looks perfectly monotone.
    scrapes[1] = _snapshot(
        generated=5,
        elapsed_ns=1_100_000_000,
        accepted=1,
        rejected=1,
    )
    runner, _, _, _ = _runner(tmp_path, scrapes=scrapes)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert result.decision.reason is Reason.COUNTER_RESET


def _candidate_first_scrapes(**kwargs: object) -> list[str]:
    """``_normal_scrapes`` with the two arm ladders swapped.

    The metrics source hands out scrapes in call order, so measuring the
    candidate first means it must be fed the candidate's ladder first.
    """
    scrapes = _normal_scrapes(**kwargs)  # type: ignore[arg-type]
    half = len(scrapes) // 2
    return [*scrapes[half:], *scrapes[:half]]


def test_candidate_first_reuses_the_running_engine_and_ends_on_stock(
    tmp_path: Path,
) -> None:
    """A rejection must leave stock serving, so the rollback has nothing to do."""
    candidate = tmp_path / "candidate"
    runner, endpoint, _, _ = _runner(
        tmp_path,
        scrapes=_candidate_first_scrapes(
            candidate_generated=90,
            candidate_elapsed_ns=1_000_000_000,
            candidate_accepted=50,
            candidate_rejected=50,
        ),
        candidate_arm_first=True,
    )

    result = runner.benchmark(
        candidate,
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is False
    assert endpoint.activations == [candidate, "stock"]


def test_candidate_first_puts_the_candidate_back_when_the_gate_passes(
    tmp_path: Path,
) -> None:
    """The orchestrator promotes by waking, not restarting, so it must be running."""
    candidate = tmp_path / "candidate"
    runner, endpoint, _, _ = _runner(
        tmp_path,
        scrapes=_candidate_first_scrapes(),
        candidate_arm_first=True,
    )

    result = runner.benchmark(
        candidate,
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is True
    assert endpoint.activations == [candidate, "stock", candidate]


def test_stock_first_order_is_still_available(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    runner, endpoint, _, _ = _runner(
        tmp_path,
        scrapes=_normal_scrapes(),
        candidate_arm_first=False,
    )

    result = runner.benchmark(
        candidate,
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.passed is True
    # No trailing re-activation: this order already ends on the candidate.
    assert endpoint.activations == ["stock", candidate]


def test_candidate_arm_first_must_be_a_bool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate_arm_first must be a bool"):
        _runner(
            tmp_path,
            scrapes=_normal_scrapes(),
            candidate_arm_first="yes",  # type: ignore[arg-type]
        )


# --- measurement context on the persisted decision --------------------------


def test_the_decision_records_what_the_measurement_was_made_under(
    tmp_path: Path,
) -> None:
    """``decision.json`` is the only file ``speedlm gain`` reads.

    Every knob asserted here changes what the reported statistics are worth
    without changing their names -- the cap alone is documented as shifting the
    acceptance delta by ~+0.1 pp -- and all of them lived only on
    ``GateResult.metrics``, which nothing persists.  A decision that omits them
    is not comparable to another run's.
    """
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=FakeReplayExecutor(),
        replay_concurrency=4,
        benchmark_max_tokens=256,
        correctness_max_tokens=64,
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    record = result.decision.to_dict()
    assert record["benchmark_max_tokens"] == 256
    assert record["replay_concurrency"] == 4
    assert record["correctness_max_tokens"] == 64
    assert record["num_contexts"] == result.metrics["num_contexts"]
    assert record["suite_hash"] == result.metrics["suite_hash"]
    # The baseline the delta is *against*, which moves on every promotion.
    assert record["stock_draft"] == "stock"


def test_the_decision_records_the_baseline_the_gate_actually_resolved(
    tmp_path: Path,
) -> None:
    endpoint = FakeEndpoint()
    baselines = iter(("first-baseline", "second-baseline"))
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft=lambda: next(baselines),
        endpoint=endpoint,
        metrics_source=FakeMetricsSource(_normal_scrapes() + _normal_scrapes()),
        replay_executor=FakeReplayExecutor(),
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    first = runner.benchmark(
        tmp_path / "candidate", timeout_seconds=30, should_abort=lambda: False
    )
    second = runner.benchmark(
        tmp_path / "candidate", timeout_seconds=30, should_abort=lambda: False
    )

    assert first.decision is not None and second.decision is not None
    assert first.decision.stock_draft == "first-baseline"
    assert second.decision.stock_draft == "second-baseline"
    assert endpoint.activations[0] == "first-baseline"
    assert "second-baseline" in endpoint.activations


@pytest.mark.parametrize("resolved", [None, "", 7])
def test_an_unresolvable_stock_draft_fails_closed(
    tmp_path: Path,
    resolved: object,
) -> None:
    """No baseline is not a comparison, so it must not become a benchmark."""
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft=lambda: resolved,  # type: ignore[return-value]
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=FakeReplayExecutor(),
        training_context_hashes=frozenset(),
        clock=FakeClock(),
    )

    with pytest.raises(ValueError, match="stock draft"):
        runner.benchmark(
            tmp_path / "candidate",
            timeout_seconds=30,
            should_abort=lambda: False,
        )


# ---------------------------------------------------------------------------
# The measurement records its own engine regime
# ---------------------------------------------------------------------------
def _regime_runner(
    tmp_path: Path, execution: EngineExecution | None
) -> BenchmarkGateRunner:
    return BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=FakeTraceSource((_trace(),)),
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=FakeEndpoint(),
        metrics_source=FakeMetricsSource(_normal_scrapes()),
        replay_executor=FakeReplayExecutor(),
        training_context_hashes=frozenset(),
        engine_execution=execution,
        clock=FakeClock(),
    )


def test_the_decision_records_the_engine_regime_it_was_measured_under(
    tmp_path: Path,
) -> None:
    """Eager and CUDA-graph runs are not comparable numbers.

    They are a large throughput difference on the same weights and, for a MoE
    target, not even bitwise equivalent -- which also moves the divergence
    floor a run is judged against.
    """
    runner = _regime_runner(
        tmp_path,
        EngineExecution(
            enforce_eager=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            max_num_seqs=64,
        ),
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    record = result.decision.to_dict()
    assert record["engine_execution_mode"] == CUDA_GRAPH_EXECUTION_MODE
    assert record["engine_enforce_eager"] is False
    assert record["engine_enable_chunked_prefill"] is True
    assert record["engine_enable_prefix_caching"] is False
    assert record["engine_max_num_seqs"] == 64
    # The replay degree was already recorded and stays recorded: the regime is
    # the engine *and* the traffic it was driven at.
    assert record["replay_concurrency"] == DEFAULT_REPLAY_CONCURRENCY


def test_an_eager_run_is_distinguishable_from_a_graph_run_in_the_record(
    tmp_path: Path,
) -> None:
    eager = _regime_runner(tmp_path / "a", EngineExecution(enforce_eager=True))
    graphed = _regime_runner(tmp_path / "b", EngineExecution(enforce_eager=False))

    eager_record = eager.benchmark(
        tmp_path / "candidate", timeout_seconds=30, should_abort=lambda: False
    ).decision
    graph_record = graphed.benchmark(
        tmp_path / "candidate", timeout_seconds=30, should_abort=lambda: False
    ).decision

    assert eager_record is not None and graph_record is not None
    assert eager_record.to_dict()["engine_execution_mode"] == EAGER_EXECUTION_MODE
    assert (
        graph_record.to_dict()["engine_execution_mode"] == CUDA_GRAPH_EXECUTION_MODE
    )


def test_a_runner_told_nothing_records_unrecorded_not_eager(
    tmp_path: Path,
) -> None:
    """The runner owns no vLLM process, so silence must stay silence."""
    runner = _regime_runner(tmp_path, None)

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.decision is not None
    assert (
        result.decision.to_dict()["engine_execution_mode"] == UNRECORDED_EXECUTION_MODE
    )
    assert result.metrics["engine_execution"] == {
        "execution_mode": UNRECORDED_EXECUTION_MODE,
        "enforce_eager": None,
        "enable_chunked_prefill": None,
        "enable_prefix_caching": None,
        "max_num_seqs": None,
    }


def test_the_engine_regime_is_also_on_the_gate_metrics(tmp_path: Path) -> None:
    runner = _regime_runner(
        tmp_path, EngineExecution(enforce_eager=True, max_num_seqs=32)
    )

    result = runner.benchmark(
        tmp_path / "candidate",
        timeout_seconds=30,
        should_abort=lambda: False,
    )

    assert result.metrics["engine_execution"] == {
        "execution_mode": EAGER_EXECUTION_MODE,
        "enforce_eager": True,
        "enable_chunked_prefill": None,
        "enable_prefix_caching": None,
        "max_num_seqs": 32,
    }


def test_a_bogus_engine_description_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="engine_execution"):
        _regime_runner(tmp_path, "eager")  # type: ignore[arg-type]
