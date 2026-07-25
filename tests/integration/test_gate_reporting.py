from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from speedlm.config import SamplingConfig, SpeedLMConfig
from speedlm.gate.replay import ReplayResult, RequestResult, RunResults
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gate.suite import BenchmarkSuite
from speedlm.gateway.activity import ActivityTracker
from speedlm.report import GainStatus, build_gain_report
from speedlm.traces.store import TraceRecord, TraceStore
from speedlm.training.base import BackendInfo
from speedlm.tuner.artifacts import ArtifactRegistry
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import CycleOutcome, TunerOrchestrator
from speedlm.tuner.state import TunerStateMachine


@dataclass(frozen=True, slots=True)
class _Snapshot:
    content_hash: str = "integration-traces"


@dataclass(frozen=True, slots=True)
class _Prepared:
    snapshot: _Snapshot = field(default_factory=_Snapshot)


@dataclass
class _Backend:
    def describe(self) -> BackendInfo:
        return BackendInfo("verifier", "draft", "stock-draft", {"steps": 1})

    def prepare(
        self,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> _Prepared:
        del work_dir
        assert not should_abort()
        return _Prepared()

    def extract(
        self,
        prepared: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> object:
        del prepared, work_dir
        assert not should_abort()
        return object()

    def train(
        self,
        extracted: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> object:
        del extracted, work_dir
        assert not should_abort()
        return object()

    def materialize(
        self,
        trained: Any,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> Path:
        del trained
        assert not should_abort()
        draft = work_dir / "draft"
        draft.mkdir()
        (draft / "weights.bin").write_bytes(b"candidate")
        return draft

    def validate(
        self,
        artifact: Any,
        *,
        should_abort: Callable[[], bool],
    ) -> None:
        assert Path(artifact).is_dir()
        assert not should_abort()


@dataclass
class _Runtime:
    calls: list[str] = field(default_factory=list)

    def quiesce(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.calls.append("quiesce")

    def sleep(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.calls.append("sleep")

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert draft_directory.is_dir()
        assert timeout_seconds > 0 and not should_abort()
        self.calls.append("candidate")

    def restore(self, active_draft: Path | str, *, timeout_seconds: float) -> None:
        del active_draft
        assert timeout_seconds > 0
        self.calls.append("restore")

    def wake(self, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.calls.append("wake")


@dataclass
class _Endpoint:
    url: str = "http://fake-endpoint"
    activations: list[Path | str] = field(default_factory=list)

    def activate(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.activations.append(draft)


@dataclass
class _Metrics:
    scrapes: list[str]

    def scrape(
        self,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> str:
        assert timeout_seconds > 0 and not should_abort()
        return self.scrapes.pop(0)


@dataclass
class _Replay:
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
        assert endpoint_url == "http://fake-endpoint"
        assert sampling == SamplingConfig()
        assert repeats == 3
        assert timeout_seconds > 0 and not should_abort()
        request = RequestResult(
            context_hash=suite.contexts[0].context_hash,
            latency_s=1.0,
            prompt_tokens=4,
            completion_tokens=10,
            total_tokens=14,
            response_text="stable output",
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
        return ReplayResult(
            run_results=(run,) * repeats,
            num_runs=repeats,
            suite_hash=suite.suite_hash,
        )


def _metrics_snapshot(
    generated: float,
    elapsed_ns: float,
    accepted: float,
    rejected: float,
) -> str:
    return "\n".join(
        (
            f"generated_tokens_total {generated}",
            "prompt_tokens_total 100",
            f"time_per_output_token_ns_sum {elapsed_ns}",
            f"vllm:speculated_tokens_total {accepted + rejected}",
            f"vllm:accepted_tokens_total {accepted}",
            f"vllm:accept_token_count_total {accepted}",
            f"vllm:reject_token_count_total {rejected}",
        )
    )


def test_gate_decision_persistence_is_reported_as_measured_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    traces = TraceStore(tmp_path / "traces" / "traces.jsonl")
    traces.append(
        TraceRecord(
            id="gate-trace",
            timestamp=1.0,
            model="model",
            messages=(
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "answer"},
            ),
            tool_calls=(),
            temperature=0.0,
            top_p=1.0,
            seed=0,
            prompt_tokens=4,
            completion_tokens=2,
        )
    )
    endpoint = _Endpoint()
    metrics = _Metrics(
        [
            _metrics_snapshot(100, 1_000_000_000, 10, 10),
            _metrics_snapshot(200, 2_000_000_000, 70, 50),
            _metrics_snapshot(1_000, 10_000_000_000, 100, 100),
            _metrics_snapshot(1_120, 10_800_000_000, 180, 120),
        ]
    )
    gate = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=traces,
        suite_dir=tmp_path / "runs" / "suite",
        stock_draft="stock-draft",
        endpoint=endpoint,
        metrics_source=metrics,
        replay_executor=_Replay(),
        training_context_hashes=frozenset(),
        clock=lambda: 10.0,
    )
    now = [0.0]
    activity = ActivityTracker(clock=lambda: now[0])
    now[0] = 10.0
    runtime = _Runtime()
    orchestrator = TunerOrchestrator(
        state=TunerStateMachine(tmp_path / "runs"),
        idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: now[0]),
        backend=_Backend(),
        artifacts=ArtifactRegistry(tmp_path / "runs"),
        runtime=runtime,
        gate=gate,
        work_root=tmp_path / "runs",
        run_id_factory=lambda: "gain-cycle",
    )

    result = orchestrator.run_once()
    report = build_gain_report()
    rendered = report.render_text().lower()

    assert result.outcome is CycleOutcome.PROMOTED
    assert result.decision_path == tmp_path / "runs" / "gain-cycle" / "decision.json"
    assert result.decision_path.is_file()
    assert endpoint.activations[0] == "stock-draft"
    assert endpoint.activations[1] != endpoint.activations[0]
    assert report.status is GainStatus.MEASURED
    assert report.source_path == result.decision_path
    assert "not measured" not in rendered
    assert "tok/s" in rendered
