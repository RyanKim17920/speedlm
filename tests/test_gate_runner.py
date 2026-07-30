"""CPU-only tests for the concrete benchmark-gate runner."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.decide import Reason, Verdict
from speedlm.gate.replay import ReplayResult, RequestResult, RunResults
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gate.suite import BenchmarkSuite, FrozenContext, SuiteError
from speedlm.traces.store import TraceRecord


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

    def scrape(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> str:
        assert timeout_seconds > 0
        return self.scrapes.pop(0)


@dataclass
class FakeReplayExecutor:
    abort_after_first: list[bool] | None = None
    advance_after_first: Callable[[], None] | None = None
    calls: int = 0
    seen_suite_ids: list[int] = field(default_factory=list)

    def replay(
        self,
        suite: BenchmarkSuite,
        endpoint_url: str,
        sampling: SamplingConfig,
        *,
        repeats: int,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> ReplayResult:
        assert endpoint_url == "http://not-used.test/"
        assert timeout_seconds > 0
        assert repeats == 3
        assert sampling.temperature == 0.0
        assert sampling.top_p == 1.0
        assert sampling.seed == 0
        self.calls += 1
        self.seen_suite_ids.append(id(suite))

        request = RequestResult(
            context_hash=suite.contexts[0].context_hash,
            latency_s=1.0,
            prompt_tokens=4,
            completion_tokens=10,
            total_tokens=14,
            response_text="same deterministic output",
            valid=True,
        )
        run = RunResults(
            results=(request,),
            total_latency_s=1.0,
            total_prompt_tokens=4,
            total_completion_tokens=10,
            valid_count=1,
            invalid_count=0,
            invalid_rate=0.0,
        )
        result = ReplayResult(
            run_results=(run,) * repeats,
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


def _normal_scrapes(
    *,
    candidate_generated: float = 120,
    candidate_elapsed_ns: float = 800_000_000,
    candidate_accepted: float = 80,
    candidate_rejected: float = 20,
) -> list[str]:
    return [
        _snapshot(
            generated=100,
            elapsed_ns=1_000_000_000,
            accepted=10,
            rejected=10,
        ),
        _snapshot(
            generated=200,
            elapsed_ns=2_000_000_000,
            accepted=70,
            rejected=50,
        ),
        _snapshot(
            generated=1_000,
            elapsed_ns=10_000_000_000,
            accepted=100,
            rejected=100,
        ),
        _snapshot(
            generated=1_000 + candidate_generated,
            elapsed_ns=10_000_000_000 + candidate_elapsed_ns,
            accepted=100 + candidate_accepted,
            rejected=100 + candidate_rejected,
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
) -> tuple[BenchmarkGateRunner, FakeEndpoint, FakeReplayExecutor, FakeTraceSource]:
    endpoint = FakeEndpoint()
    replay_executor = replay or FakeReplayExecutor()
    traces = trace_source or FakeTraceSource((_trace(),))
    runner = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=traces,
        suite_dir=tmp_path / "suite",
        stock_draft="stock",
        endpoint=endpoint,
        metrics_source=FakeMetricsSource(scrapes),
        replay_executor=replay_executor,
        training_context_hashes=training_context_hashes,
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
    assert replay.calls == 2
    assert replay.seen_suite_ids[0] == replay.seen_suite_ids[1]


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
    assert result.decision.stock_avg_tok_per_sec == 0.0
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
    assert replay.calls == 2
