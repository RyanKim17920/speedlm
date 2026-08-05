from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from draft_weights import write_draft_config, write_draft_weights

from speedlm.config import IdleTuningConfig, SamplingConfig, SpeedLMConfig
from speedlm.gate.replay import ReplayResult, RequestResult, RunResults
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gate.suite import BenchmarkSuite
from speedlm.gateway.activity import ActivityTracker
from speedlm.report import load_decision
from speedlm.traces.store import TraceRecord, TraceStats, TraceStore
from speedlm.training.base import BackendInfo
from speedlm.training.masking import MaskPolicy
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
from speedlm.tuner.composition import (
    active_draft_reference,
    promotion_chain_depth,
    warm_start_reference,
)
from speedlm.tuner.eagle3 import (
    Eagle3Adapter,
    Eagle3Config,
    TraceSnapshot,
    TrainingResult,
)
from speedlm.tuner.idle import IdleDetector
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    GateResult,
    TunerOrchestrator,
)
from speedlm.tuner.service import TunerService
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
    #: Restores that raise before one is allowed to succeed.
    fail_restores: int = 0
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
        self.calls.append("restore")
        if self.fail_restores > 0:
            self.fail_restores -= 1
            # Deliberately does *not* move ``serving``: a restore that could not
            # respawn leaves the engine on whatever the cycle last loaded.
            raise RuntimeError("simulated launch failure for the rollback")
        self.serving = active_draft

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


@dataclass
class _StaticTraces:
    """A trace-buffer stub whose watermark the test moves by hand."""

    count: int

    def stats(self) -> TraceStats:
        return TraceStats(count=self.count, tokens=self.count * 16, oldest=0.0, newest=1.0)

    def prune(self) -> int:
        return 0


def test_the_scheduler_re_attempts_a_restore_that_could_not_respawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole loop, not either half of it.

    ``tests/test_tuner_orchestrator.py`` proves the orchestrator records the
    condition and ``tests/test_tuner_service.py`` proves the scheduler acts on
    it, but each does so against a stub of the other.  This wires the real
    :class:`TunerOrchestrator` into the real :class:`TunerService` and asserts
    the property that actually matters: an abandoned candidate does not keep
    answering live traffic, and it does not have to wait out the 600s retry
    cooldown that ``PREEMPTED`` arms.
    """
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

    now, activity, idle = _idle_pair()
    runtime = _Runtime(
        activity,
        now,
        traffic_on_sleep=True,
        serving=baseline.path,
        fail_restores=1,
    )
    state = TunerStateMachine(runs / "state")
    orchestrator = TunerOrchestrator(
        state=state,
        idle=idle,
        backend=_Backend(b"must-not-train"),
        artifacts=artifacts,
        runtime=runtime,
        gate=_Gate(True),
        work_root=runs,
        run_id_factory=lambda: "unrestored-cycle",
    )

    clock = [0.0]
    service = TunerService(
        SpeedLMConfig(
            model="verifier",
            idle_threshold_seconds=0.01,
            tuning=IdleTuningConfig(
                idle_confirmations=1,
                # The exact interaction under test: a preemption arms this, and
                # the restore re-attempt must not be behind it.
                retry_cooldown_seconds=600.0,
                serving_recovery_interval_seconds=0.0,
            ),
        ),
        activity=ActivityTracker(clock=lambda: clock[0]),
        traces=_StaticTraces(64),
        orchestrator_factory=lambda _activity: orchestrator,
        enabled=True,
        min_trace_records=2,
        min_corpus_records=2,
        poll_interval_seconds=0.005,
        clock=lambda: clock[0],
        status_path=runs / "scheduler.json",
    )

    clock[0] = 100.0
    service._poll_once()

    # The cycle was preempted and its rollback could not respawn -- reported as
    # such, and as an ordinary PREEMPTED outcome, which is the point: nothing
    # about the outcome distinguishes this cycle.
    status = json.loads((runs / "scheduler.json").read_text(encoding="utf-8"))
    assert status["last_result"]["outcome"] == CycleOutcome.PREEMPTED.value
    assert status["last_result"]["serving_restored"] is False

    # So the recovery cannot have been keyed off the outcome.  The same poll
    # re-attempted the restore and it took -- 600 seconds earlier than the
    # cooldown that this very outcome armed would have allowed.
    assert runtime.calls == [
        "quiesce",
        "sleep",
        "restore",
        "wake",
        "restore",
    ]
    assert runtime.serving == baseline.path
    assert orchestrator.serving_unrestored is False
    assert state.state is TunerState.READY
    assert status["serving_unrestored"] is False


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
        allow_engine_reuse: bool = True,
    ) -> bool:
        assert timeout_seconds > 0 and not should_abort()
        self.activations.append(draft)
        return True


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
        # Only the correctness call may ask for more than one pass, and only
        # on the stock arm, where the extra run measures the engine's own
        # divergence rate against itself.
        assert repeats == 1 or capture_tokens
        assert repeats <= 2
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


# ---------------------------------------------------------------------------
# Compounding warm start
# ---------------------------------------------------------------------------


@dataclass
class _RecordingTrainer:
    """Records the ``--from-pretrained`` base each cycle actually trained on."""

    bases: list[str] = field(default_factory=list)

    def train(
        self,
        hidden_states_path: Path,
        destination: Path,
        *,
        from_pretrained: str,
        training_params: Mapping[str, object],
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> TrainingResult:
        del hidden_states_path, training_params
        assert timeout_seconds > 0 and not should_abort()
        self.bases.append(from_pretrained)
        checkpoint = destination / "checkpoint_best"
        checkpoint.mkdir(parents=True)
        return TrainingResult(checkpoint_best=checkpoint, returncode=0)


@dataclass
class _StubStages:
    """Every EAGLE-3 effect except training, which is what the test watches."""

    payload: bytes

    def lease_snapshot(
        self,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> TraceSnapshot:
        assert timeout_seconds > 0 and not should_abort()
        destination.mkdir(parents=True, exist_ok=True)
        return TraceSnapshot(path=destination, content_hash=self.payload.hex())

    def render_rows(
        self,
        snapshot: TraceSnapshot,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        del snapshot
        assert timeout_seconds > 0 and not should_abort()
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def extract_hidden_states(
        self,
        rows_path: Path,
        destination: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        del rows_path, verifier_model
        assert timeout_seconds > 0 and not should_abort()
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def materialize(
        self,
        checkpoint_best: Path,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> Path:
        del checkpoint_best
        assert timeout_seconds > 0 and not should_abort()
        destination.mkdir(parents=True, exist_ok=True)
        # A real container seeded per cycle: the adapter now parses the
        # safetensors header, requires the trained-head tensor set, and
        # refuses a candidate whose weights match its baseline byte for byte.
        write_draft_weights(destination, seed=self.payload[-1])
        # Materialization rewrites the draft's declared speculative_tokens to
        # the depth the cycle trained at, so a stand-in head has to carry the
        # Speculators config block that declaration lives in.
        write_draft_config(destination)
        (destination / "weights.bin").write_bytes(self.payload)
        return destination

    def validate(
        self,
        draft_directory: Path,
        *,
        verifier_model: str,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        del verifier_model
        assert draft_directory.is_dir()
        assert timeout_seconds > 0 and not should_abort()


STOCK_DRAFT = "acme/stock-speculator"


def _adapter(
    payload: bytes,
    trainer: _RecordingTrainer,
    resolver: Callable[[], str] | None,
) -> Eagle3Adapter:
    stages = _StubStages(payload)
    return Eagle3Adapter(
        Eagle3Config(
            verifier_model="acme/verifier",
            draft_model=STOCK_DRAFT,
            from_pretrained=STOCK_DRAFT,
            mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
        ),
        leaser=stages,
        renderer=stages,
        extractor=stages,
        trainer=trainer,
        materializer=stages,
        validator=stages,
        warm_start_resolver=resolver,
    )


def _compounding_cycles(
    tmp_path: Path,
    verdicts: Sequence[bool],
    *,
    max_chain_depth: int | None = None,
    compounding: bool = True,
) -> tuple[_RecordingTrainer, ArtifactRegistry, list[Any]]:
    """Drive consecutive real cycles through the real EAGLE-3 adapter."""
    runs = tmp_path / "runs"
    artifacts = ArtifactRegistry(runs)
    trainer = _RecordingTrainer()
    now, activity, idle = _idle_pair()
    runtime = _Runtime(activity, now)
    resolver = (
        (
            lambda: warm_start_reference(
                artifacts,
                STOCK_DRAFT,
                max_chain_depth=max_chain_depth,
            )
        )
        if compounding
        else None
    )
    results: list[Any] = []
    for index, passed in enumerate(verdicts):
        results.append(
            TunerOrchestrator(
                state=TunerStateMachine(runs / "state"),
                idle=idle,
                backend=_adapter(f"weights-{index}".encode(), trainer, resolver),
                artifacts=artifacts,
                runtime=runtime,
                gate=_Gate(passed),
                work_root=runs,
                run_id_factory=lambda index=index: f"cycle-{index}",
            ).run_once()
        )
    return trainer, artifacts, results


def test_each_cycle_warm_starts_from_the_artifact_that_is_currently_serving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Learning has to accumulate, which means cycle N+1 builds on cycle N.

    Regression for a warm start captured once, at composition time, from the
    profile's stock speculator.  Frozen, every cycle re-ran the same one-shot
    fine-tune of the original head and threw the previous cycle's promoted
    artifact away, so the compounding that is the whole premise of idle tuning
    could never happen -- and never had.

    Three cycles is the minimum that separates the two things a warm start must
    track: promotion (cycle two must move to cycle one's artifact) and the
    *absence* of one (cycle three must stay on it after a rejection, not fall
    back to stock and not follow the rejected candidate).
    """
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    trainer, artifacts, results = _compounding_cycles(tmp_path, (True, False, True))

    assert [result.outcome for result in results] == [
        CycleOutcome.PROMOTED,
        CycleOutcome.REJECTED,
        CycleOutcome.PROMOTED,
    ]
    first, rejected, third = (result.artifact_id for result in results)
    first_path = artifacts.get(first).path

    # Cycle one has nothing promoted, so it legitimately starts from stock.
    # Cycle two must start from cycle one's promotion, and cycle three -- whose
    # predecessor was rejected -- must still start from the incumbent, which is
    # cycle one's artifact and not the rejected candidate.
    assert trainer.bases == [STOCK_DRAFT, str(first_path), str(first_path)]
    assert str(artifacts.get(rejected).path) not in trainer.bases

    # ...and the chain has to be reconstructable from the artifacts alone, or a
    # promotion's ancestry is unknowable after the fact.  The manifest must
    # carry the *resolved* base, not the configured one.
    assert artifacts.get(first).manifest.base_draft == STOCK_DRAFT
    assert artifacts.get(third).manifest.base_draft == str(first_path)
    assert promotion_chain_depth(first_path) == 1
    assert promotion_chain_depth(artifacts.get(third).path) == 2
    assert promotion_chain_depth(STOCK_DRAFT) == 0


def test_compounding_can_be_switched_off_and_every_cycle_returns_to_stock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tuning.compounding_warm_start = false`` is the archived-run behaviour."""
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    trainer, _artifacts, results = _compounding_cycles(
        tmp_path,
        (True, True),
        compounding=False,
    )

    assert [result.outcome for result in results] == [
        CycleOutcome.PROMOTED,
        CycleOutcome.PROMOTED,
    ]
    assert trainer.bases == [STOCK_DRAFT, STOCK_DRAFT]


def test_a_chain_bound_re_baselines_to_stock_instead_of_compounding_forever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only lever against drift, since the gate cannot measure it.

    ``warm_start_max_chain_depth=1`` means "an incumbent that is itself trained
    is already at the bound", so the second cycle re-baselines.  It promotes
    anyway -- the bound moves where training *starts*, never what the gate
    decides -- and the third cycle's incumbent is still one deep, so it
    re-baselines too.
    """
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    trainer, artifacts, results = _compounding_cycles(
        tmp_path,
        (True, True, True),
        max_chain_depth=1,
    )

    assert [result.outcome for result in results] == [CycleOutcome.PROMOTED] * 3
    assert trainer.bases == [STOCK_DRAFT, STOCK_DRAFT, STOCK_DRAFT]
    # Every artifact is one link deep, which is exactly what the bound buys:
    # the chain never grows, so a promotion cannot inherit an ancestor's drift.
    for result in results:
        assert result.artifact_id is not None
        assert promotion_chain_depth(artifacts.get(result.artifact_id).path) == 1
