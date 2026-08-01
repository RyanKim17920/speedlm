from __future__ import annotations

import json
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
from speedlm.report import load_decision
from speedlm.traces.store import TraceRecord, TraceStore
from speedlm.training.base import BackendInfo
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.composition import active_draft_reference
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    GateResult,
    TunerOrchestrator,
)
from speedlm.tuner.state import TunerState, TunerStateMachine


@dataclass(frozen=True, slots=True)
class _Snapshot:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    snapshot: _Snapshot


@dataclass
class _Backend:
    payload: bytes

    def describe(self) -> BackendInfo:
        return BackendInfo(
            verifier_model="verifier",
            draft_model="draft",
            from_pretrained="stock-draft",
            training_params={"steps": 1},
        )

    def prepare(
        self,
        work_dir: Path,
        *,
        should_abort: Callable[[], bool],
    ) -> _Prepared:
        del work_dir
        assert not should_abort()
        return _Prepared(_Snapshot(self.payload.hex()))

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
        (draft / "weights.bin").write_bytes(self.payload)
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
class _Gate:
    passed: bool

    def benchmark(
        self,
        candidate_draft: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> GateResult:
        assert candidate_draft.is_dir()
        assert timeout_seconds > 0 and not should_abort()
        return GateResult(
            passed=self.passed,
            reason="thresholds met" if self.passed else "candidate regressed",
        )


@dataclass
class _Runtime:
    activity: ActivityTracker
    clock_value: list[float]
    traffic_on_sleep: bool = False
    serving: Path | str | None = None
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
        if self.traffic_on_sleep:
            self.clock_value[0] += 1.0
            self.activity.begin()
            self.activity.end()

    def start_candidate(
        self,
        draft_directory: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        assert timeout_seconds > 0 and not should_abort()
        self.serving = draft_directory
        self.calls.append("start_candidate")

    def restore(self, active_draft: Path | str, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.serving = active_draft
        self.calls.append("restore")

    def wake(self, *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.calls.append("wake")


def _idle_pair() -> tuple[list[float], ActivityTracker, IdleDetector]:
    now = [0.0]
    activity = ActivityTracker(clock=lambda: now[0])
    now[0] = 10.0
    idle = IdleDetector(activity, threshold_seconds=5.0, clock=lambda: now[0])
    return now, activity, idle


def test_full_tuner_promotes_then_rejection_preserves_active_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    runs = tmp_path / "runs"
    state = TunerStateMachine(runs / "state")
    artifacts = ArtifactRegistry(runs)
    now, activity, idle = _idle_pair()
    runtime = _Runtime(activity, now)

    promoted = TunerOrchestrator(
        state=state,
        idle=idle,
        backend=_Backend(b"promoted-candidate"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(True),
        work_root=runs,
        run_id_factory=lambda: "promote-cycle",
    ).run_once()

    assert promoted.outcome is CycleOutcome.PROMOTED
    assert state.state is TunerState.READY
    assert promoted.artifact_id is not None
    pointer_after_promotion = json.loads(
        artifacts.active_path.read_text(encoding="utf-8")
    )
    assert pointer_after_promotion["artifact_id"] == promoted.artifact_id
    active_bytes = artifacts.active_path.read_bytes()

    rejected = TunerOrchestrator(
        state=state,
        idle=idle,
        backend=_Backend(b"rejected-candidate"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(False),
        work_root=runs,
        run_id_factory=lambda: "reject-cycle",
    ).run_once()

    assert rejected.outcome is CycleOutcome.REJECTED
    assert rejected.artifact_id != promoted.artifact_id
    assert state.state is TunerState.READY
    assert artifacts.active_path.read_bytes() == active_bytes
    assert artifacts.active_pointer() is not None
    assert artifacts.active_pointer().artifact_id == promoted.artifact_id  # type: ignore[union-attr]


def test_mid_cycle_traffic_preempts_and_restores_previous_active_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    runs = tmp_path / "runs"
    artifacts = ArtifactRegistry(runs)
    baseline_source = tmp_path / "baseline"
    baseline_source.mkdir()
    (baseline_source / "weights.bin").write_bytes(b"known-good")
    baseline = artifacts.publish(
        baseline_source,
        ArtifactSpec(
            verifier_model="verifier",
            draft_model="draft",
            base_draft="stock-draft",
            trace_hash="baseline-traces",
            training_params={"steps": 1},
        ),
    )
    artifacts.promote(baseline.artifact_id, gate_passed=True)
    active_before = artifacts.active_path.read_bytes()
    now, activity, idle = _idle_pair()
    runtime = _Runtime(
        activity,
        now,
        traffic_on_sleep=True,
        serving=baseline.path,
    )

    result = TunerOrchestrator(
        state=TunerStateMachine(runs / "state"),
        idle=idle,
        backend=_Backend(b"must-not-train"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(True),
        work_root=runs,
        run_id_factory=lambda: "preempted-cycle",
    ).run_once()

    assert result.outcome is CycleOutcome.PREEMPTED
    assert artifacts.active_path.read_bytes() == active_before
    assert runtime.serving == baseline.path
    assert runtime.calls == ["quiesce", "sleep", "restore", "wake"]


# ---------------------------------------------------------------------------
# Multi-cycle gate baseline
# ---------------------------------------------------------------------------


@dataclass
class _Endpoint:
    """Records which draft each benchmark arm asked to have activated."""

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
        concurrency: int | None = None,
        max_tokens: int | None = None,
        capture_tokens: bool = False,
    ) -> ReplayResult:
        assert endpoint_url == "http://fake-endpoint"
        assert sampling == SamplingConfig()
        assert repeats == 1
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


#: One promoting arm pair: four scrapes per arm -- one opening the window and
#: one after each of the three scored repeats -- in the order the arms run.
#: ``candidate_arm_first`` is on, so the candidate's block comes first, and it
#: accepts 0.80 of its drafted tokens against the stock arm's 0.60.
def _promoting_scrapes() -> list[str]:
    return [
        _metrics_snapshot(1_000, 10_000_000_000, 100, 100),
        _metrics_snapshot(1_040, 10_266_666_666, 126, 106),
        _metrics_snapshot(1_080, 10_533_333_333, 153, 113),
        _metrics_snapshot(1_120, 10_800_000_000, 180, 120),
        _metrics_snapshot(100, 1_000_000_000, 10, 10),
        _metrics_snapshot(133, 1_333_333_333, 30, 23),
        _metrics_snapshot(166, 1_666_666_666, 50, 36),
        _metrics_snapshot(200, 2_000_000_000, 70, 50),
    ]


def _gate_trace_store(root: Path) -> TraceStore:
    traces = TraceStore(root / "traces" / "traces.jsonl")
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
    return traces


def test_second_cycle_benchmarks_against_the_promoted_draft_not_the_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate's stock arm must track promotions across cycles.

    Regression for a gate that captured its stock draft once, at composition
    time.  With no post-promotion rollback anywhere in the system the gate is
    the only safeguard, and a frozen baseline makes every cycle after the first
    report cumulative gain over the *original* head rather than marginal gain
    over the draft that is actually serving -- so a candidate worse than the
    incumbent can still show a positive delta and be promoted.

    Two cycles, promotion in the first, is the minimum that can see it: cycle
    one legitimately benchmarks against the warm-start draft, and only cycle
    two can disagree about what "stock" means.
    """
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    runs = tmp_path / "runs"
    artifacts = ArtifactRegistry(runs)
    endpoint = _Endpoint()
    metrics = _Metrics(_promoting_scrapes() + _promoting_scrapes())
    gate = BenchmarkGateRunner(
        config=SpeedLMConfig(model="model"),
        trace_source=_gate_trace_store(tmp_path),
        suite_dir=tmp_path / "suite",
        # Exactly the wiring ``create_production_tuner`` uses.
        stock_draft=lambda: active_draft_reference(artifacts, "stock-draft"),
        endpoint=endpoint,
        metrics_source=metrics,
        replay_executor=_Replay(),
        training_context_hashes=frozenset(),
        # Production's default arm order: the candidate arm runs first and the
        # stock arm second, so the stock activation is the one to look at.
        candidate_arm_first=True,
        clock=lambda: 10.0,
    )
    now, activity, idle = _idle_pair()
    runtime = _Runtime(activity, now)

    def cycle(payload: bytes, run_id: str) -> Any:
        return TunerOrchestrator(
            state=TunerStateMachine(runs / "state"),
            idle=idle,
            backend=_Backend(payload),
            artifacts=artifacts,
            runtime=runtime,
            gate=gate,
            work_root=runs,
            run_id_factory=lambda: run_id,
        ).run_once()

    first = cycle(b"first-candidate", "cycle-one")
    assert first.outcome is CycleOutcome.PROMOTED
    assert first.artifact_id is not None
    promoted = artifacts.active()
    assert promoted is not None and promoted.artifact_id == first.artifact_id

    second = cycle(b"second-candidate", "cycle-two")
    assert second.artifact_id not in (None, first.artifact_id)

    # Candidate first, then stock, then -- on a promotion -- the candidate is
    # put back.  Cycle one has nothing promoted yet, so its baseline is the
    # warm-start draft; cycle two's baseline must be cycle one's promotion.
    first_candidate, first_stock = endpoint.activations[0], endpoint.activations[1]
    second_candidate, second_stock = endpoint.activations[3], endpoint.activations[4]
    assert first_stock == "stock-draft"
    assert second_stock == promoted.path
    assert second_stock != first_stock
    assert second_candidate not in (first_candidate, second_stock)

    # ...and the persisted decision has to say which baseline it was against,
    # or the two cycles' deltas are indistinguishable after the fact.
    assert second.decision_path is not None
    assert load_decision(second.decision_path).stock_draft == str(promoted.path)
