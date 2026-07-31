"""Validation loss pre-filter: parse, manifest storage, and orchestrator integration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.config import ConfigError, ValLossPreFilterConfig
from speedlm.training.backends.eagle3 import _parse_val_loss
from speedlm.tuner.artifacts import ArtifactError, ArtifactManifest, ArtifactSpec
from speedlm.tuner.eagle3 import TrainingResult
from speedlm.tuner.orchestrator import (
    CycleOutcome,
    CycleResult,
    GateResult,
)

# ---------------------------------------------------------------------------
# _parse_val_loss
# ---------------------------------------------------------------------------


class TestParseValLoss:
    def test_parses_loss_epoch_correctly(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            json.dumps({"loss_epoch": 2.345, "epoch": 1}),
            encoding="utf-8",
        )
        result = _parse_val_loss(checkpoint)
        assert result == 2.345

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        result = _parse_val_loss(checkpoint)
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            "not json at all{{{", encoding="utf-8"
        )
        result = _parse_val_loss(checkpoint)
        assert result is None

    def test_json_without_loss_epoch_returns_none(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            json.dumps({"epoch": 1}), encoding="utf-8"
        )
        result = _parse_val_loss(checkpoint)
        assert result is None

    def test_loss_epoch_is_string_returns_none(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            json.dumps({"loss_epoch": "two point five"}), encoding="utf-8"
        )
        result = _parse_val_loss(checkpoint)
        assert result is None

    def test_loss_epoch_is_boolean_returns_none(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            json.dumps({"loss_epoch": True}), encoding="utf-8"
        )
        result = _parse_val_loss(checkpoint)
        assert result is None

    def test_loss_epoch_integer_is_coerced_to_float(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint_best"
        checkpoint.mkdir()
        (checkpoint / "val_metrics.json").write_text(
            json.dumps({"loss_epoch": 3}), encoding="utf-8"
        )
        result = _parse_val_loss(checkpoint)
        assert result == 3.0
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TrainingResult carries val_loss
# ---------------------------------------------------------------------------


class TestTrainingResultValLoss:
    def test_val_loss_default_none(self) -> None:
        result = TrainingResult(Path("/chk"), 0)
        assert result.val_loss is None

    def test_val_loss_settable(self) -> None:
        result = TrainingResult(Path("/chk"), 0, val_loss=2.5)
        assert result.val_loss == 2.5


# ---------------------------------------------------------------------------
# ArtifactSpec and ArtifactManifest val_loss
# ---------------------------------------------------------------------------


class TestArtifactSpecValLoss:
    def test_val_loss_optional(self) -> None:
        spec = ArtifactSpec(
            verifier_model="v",
            draft_model="d",
            base_draft="b",
            trace_hash="h",
            training_params={},
        )
        assert spec.val_loss is None

    def test_val_loss_settable(self) -> None:
        spec = ArtifactSpec(
            verifier_model="v",
            draft_model="d",
            base_draft="b",
            trace_hash="h",
            training_params={},
            val_loss=1.5,
        )
        assert spec.val_loss == 1.5


class TestArtifactManifestValLoss:
    def test_to_dict_includes_val_loss(self) -> None:
        manifest = ArtifactManifest(
            artifact_id="a" * 64,
            verifier_model="v",
            draft_model="d",
            base_draft="b",
            trace_hash="h",
            training_params={},
            created_at=1000.0,
            val_loss=2.5,
        )
        d = manifest.to_dict()
        assert d["val_loss"] == 2.5

    def test_to_dict_val_loss_none(self) -> None:
        manifest = ArtifactManifest(
            artifact_id="a" * 64,
            verifier_model="v",
            draft_model="d",
            base_draft="b",
            trace_hash="h",
            training_params={},
            created_at=1000.0,
        )
        d = manifest.to_dict()
        assert d["val_loss"] is None

    def test_from_dict_without_val_loss(self) -> None:
        raw = {
            "artifact_id": "a" * 64,
            "verifier_model": "v",
            "draft_model": "d",
            "base_draft": "b",
            "trace_hash": "h",
            "training_params": {},
            "created_at": 1000.0,
        }
        manifest = ArtifactManifest.from_dict(raw)
        assert manifest.val_loss is None

    def test_from_dict_with_val_loss(self) -> None:
        raw = {
            "artifact_id": "a" * 64,
            "verifier_model": "v",
            "draft_model": "d",
            "base_draft": "b",
            "trace_hash": "h",
            "training_params": {},
            "created_at": 1000.0,
            "val_loss": 3.14,
        }
        manifest = ArtifactManifest.from_dict(raw)
        assert manifest.val_loss == 3.14

    def test_from_dict_with_val_loss_null(self) -> None:
        raw = {
            "artifact_id": "a" * 64,
            "verifier_model": "v",
            "draft_model": "d",
            "base_draft": "b",
            "trace_hash": "h",
            "training_params": {},
            "created_at": 1000.0,
            "val_loss": None,
        }
        manifest = ArtifactManifest.from_dict(raw)
        assert manifest.val_loss is None

    def test_from_dict_rejects_bad_val_loss_type(self) -> None:
        raw = {
            "artifact_id": "a" * 64,
            "verifier_model": "v",
            "draft_model": "d",
            "base_draft": "b",
            "trace_hash": "h",
            "training_params": {},
            "created_at": 1000.0,
            "val_loss": "not_a_number",
        }
        with pytest.raises(ArtifactError):
            ArtifactManifest.from_dict(raw)

    def test_from_dict_round_trip(self) -> None:
        original = ArtifactManifest(
            artifact_id="b" * 64,
            verifier_model="v",
            draft_model="d",
            base_draft="b",
            trace_hash="h",
            training_params={"k": 1},
            created_at=2000.0,
            val_loss=1.23,
        )
        restored = ArtifactManifest.from_dict(original.to_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# ValLossPreFilterConfig
# ---------------------------------------------------------------------------


class TestValLossPreFilterConfig:
    def test_defaults(self) -> None:
        cfg = ValLossPreFilterConfig()
        assert cfg.enabled is True
        assert cfg.min_improvement == 0.01

    def test_disabled(self) -> None:
        cfg = ValLossPreFilterConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_threshold(self) -> None:
        cfg = ValLossPreFilterConfig(min_improvement=0.05)
        assert cfg.min_improvement == 0.05

    def test_negative_min_improvement_rejected(self) -> None:
        with pytest.raises(ConfigError):
            ValLossPreFilterConfig(min_improvement=-0.1)


# ---------------------------------------------------------------------------
# CycleOutcome and CycleResult
# ---------------------------------------------------------------------------


class TestCycleOutcomeAndResult:
    def test_val_loss_not_improved_outcome_exists(self) -> None:
        assert CycleOutcome.VAL_LOSS_NOT_IMPROVED.value == "val_loss_not_improved"

    def test_cycle_result_val_loss(self) -> None:
        result = CycleResult(
            outcome=CycleOutcome.VAL_LOSS_NOT_IMPROVED,
            val_loss=2.5,
        )
        assert result.val_loss == 2.5

    def test_cycle_result_val_loss_default_none(self) -> None:
        result = CycleResult(outcome=CycleOutcome.FAILED)
        assert result.val_loss is None


# ---------------------------------------------------------------------------
# Orchestrator pre-filter integration
# ---------------------------------------------------------------------------

# We import the orchestrator test fixtures from test_tuner_orchestrator.py
# patterns and build minimal integration tests here.


def _build_fake_backend_with_val_loss(
    val_loss: float | None = None,
):
    """Build a FakeBackend that returns a TrainingResult with val_loss."""
    from speedlm.training.base import BackendInfo
    from speedlm.tuner.eagle3 import (
        PreparedData,
        TraceSnapshot,
        TrainingResult,
    )

    @dataclass
    class FakeBackend:
        val_loss: float | None = None
        calls: list[str] = field(default_factory=list)

        def describe(self) -> BackendInfo:
            return BackendInfo(
                verifier_model="fake/verifier",
                draft_model="fake/draft",
                from_pretrained="fake/base-draft",
                training_params={"steps": 2},
            )

        def prepare(
            self,
            work_dir: Path,
            *,
            should_abort: Callable[[], bool],
        ) -> PreparedData:
            self.calls.append("prepare")
            snapshot_path = work_dir / "trace-snapshot"
            snapshot_path.mkdir()
            rows_path = work_dir / "rows.jsonl"
            rows_path.write_text("{}\n", encoding="utf-8")
            return PreparedData(
                snapshot=TraceSnapshot(snapshot_path, "trace-hash"),
                rows_path=rows_path,
            )

        def extract(
            self,
            prepared: PreparedData,
            work_dir: Path,
            *,
            should_abort: Callable[[], bool],
        ) -> Path:
            self.calls.append("extract")
            hidden = work_dir / "hidden.pt"
            hidden.write_bytes(b"hidden")
            return hidden

        def train(
            self,
            extracted: Path,
            work_dir: Path,
            *,
            should_abort: Callable[[], bool],
        ) -> TrainingResult:
            self.calls.append("train")
            checkpoint = work_dir / "checkpoint_best"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"trained")
            return TrainingResult(checkpoint, 0, val_loss=self.val_loss)

        def materialize(
            self,
            trained: TrainingResult,
            work_dir: Path,
            *,
            should_abort: Callable[[], bool],
        ) -> Path:
            self.calls.append("materialize")
            draft = work_dir / "draft-model"
            draft.mkdir()
            (draft / "config.json").write_text(
                '{"model_type":"fake"}', encoding="utf-8"
            )
            (draft / "weights.bin").write_bytes(
                (trained.checkpoint_best / "weights.bin").read_bytes()
            )
            return draft

        def validate(
            self,
            artifact: Path,
            *,
            should_abort: Callable[[], bool],
        ) -> None:
            self.calls.append("validate")

    return FakeBackend(val_loss=val_loss)


class TestPreFilterIntegration:
    """Integration tests for the val_loss pre-filter in the orchestrator.

    These tests verify that the pre-filter correctly:
    - Skips the benchmark when val_loss did not improve
    - Falls through to the benchmark when val_loss is None
    - Always benchmarks when the pre-filter is disabled
    - Is skipped when there is no incumbent (no active artifact)
    """

    def test_prefilter_skips_benchmark_when_loss_not_improved(
        self, tmp_path: Path
    ) -> None:
        """When candidate val_loss is worse than incumbent, skip benchmark."""
        from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
        from speedlm.tuner.idle import IdleDetector
        from speedlm.tuner.orchestrator import TunerOrchestrator
        from speedlm.tuner.state import TunerStateMachine

        backend = _build_fake_backend_with_val_loss(val_loss=3.0)
        # Incumbent has val_loss=2.0 (better)
        artifacts = ArtifactRegistry(tmp_path / "registry")
        baseline_source = tmp_path / "baseline"
        baseline_source.mkdir()
        (baseline_source / "w.bin").write_bytes(b"baseline")
        baseline = artifacts.publish(
            baseline_source,
            ArtifactSpec(
                verifier_model="v",
                draft_model="d",
                base_draft="b",
                trace_hash="h",
                training_params={},
                val_loss=2.0,
            ),
        )
        artifacts.promote(baseline.artifact_id, gate_passed=True)

        activity = type("FA", (), {"in_flight": 0, "last_activity": 0.0})()

        @dataclass
        class FakeRuntime:
            calls: list[str] = field(default_factory=list)

            def quiesce(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("quiesce")

            def sleep(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("sleep")

            def start_candidate(
                self,
                draft_directory: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> None:
                self.calls.append("start_candidate")

            def restore(
                self, active_draft: Path | str, *, timeout_seconds: float
            ) -> None:
                self.calls.append("restore")

            def wake(self, *, timeout_seconds: float) -> None:
                self.calls.append("wake")

        @dataclass
        class FakeGate:
            called: bool = False

            def benchmark(
                self,
                candidate_draft: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> GateResult:
                self.called = True
                return GateResult(True, "passed")

        fake_gate = FakeGate()
        fake_runtime = FakeRuntime()
        state = TunerStateMachine(tmp_path / "state")

        orchestrator = TunerOrchestrator(
            state=state,
            idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
            backend=backend,
            artifacts=artifacts,
            runtime=fake_runtime,
            gate=fake_gate,
            work_root=tmp_path / "work",
            run_id_factory=lambda: "run-1",
            val_loss_prefilter=ValLossPreFilterConfig(
                enabled=True, min_improvement=0.01
            ),
        )

        result = orchestrator.run_once()

        # The pre-filter should have skipped the benchmark
        assert result.outcome is CycleOutcome.VAL_LOSS_NOT_IMPROVED
        assert result.val_loss == 3.0
        # Benchmark should NOT have been called
        assert not fake_gate.called
        # start_candidate should NOT have been called
        assert "start_candidate" not in fake_runtime.calls
        # Backend should have prepared, extracted, trained, materialized, validated
        assert backend.calls == ["prepare", "extract", "train", "materialize", "validate"]

    def test_prefilter_falls_through_when_val_loss_none(
        self, tmp_path: Path
    ) -> None:
        """When val_loss is None (unavailable), fall through to benchmark."""
        from speedlm.tuner.artifacts import ArtifactRegistry
        from speedlm.tuner.idle import IdleDetector
        from speedlm.tuner.orchestrator import TunerOrchestrator
        from speedlm.tuner.state import TunerStateMachine

        backend = _build_fake_backend_with_val_loss(val_loss=None)
        artifacts = ArtifactRegistry(tmp_path / "registry")

        activity = type("FA", (), {"in_flight": 0, "last_activity": 0.0})()

        @dataclass
        class FakeRuntime:
            calls: list[str] = field(default_factory=list)

            def quiesce(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("quiesce")

            def sleep(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("sleep")

            def start_candidate(
                self,
                draft_directory: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> None:
                self.calls.append("start_candidate")

            def restore(
                self, active_draft: Path | str, *, timeout_seconds: float
            ) -> None:
                self.calls.append("restore")

            def wake(self, *, timeout_seconds: float) -> None:
                self.calls.append("wake")

        @dataclass
        class FakeGate:
            called: bool = False

            def benchmark(
                self,
                candidate_draft: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> GateResult:
                self.called = True
                return GateResult(True, "passed")

        fake_gate = FakeGate()
        fake_runtime = FakeRuntime()
        state = TunerStateMachine(tmp_path / "state")

        orchestrator = TunerOrchestrator(
            state=state,
            idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
            backend=backend,
            artifacts=artifacts,
            runtime=fake_runtime,
            gate=fake_gate,
            work_root=tmp_path / "work",
            run_id_factory=lambda: "run-1",
            val_loss_prefilter=ValLossPreFilterConfig(
                enabled=True, min_improvement=0.01
            ),
        )

        result = orchestrator.run_once()

        # val_loss is None, so pre-filter should fall through to benchmark
        assert fake_gate.called
        assert result.outcome is CycleOutcome.PROMOTED

    def test_prefilter_disabled_always_benchmarks(
        self, tmp_path: Path
    ) -> None:
        """When pre-filter is disabled, always benchmark regardless of val_loss."""
        from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
        from speedlm.tuner.idle import IdleDetector
        from speedlm.tuner.orchestrator import TunerOrchestrator
        from speedlm.tuner.state import TunerStateMachine

        # Candidate val_loss is MUCH worse, but pre-filter is disabled
        backend = _build_fake_backend_with_val_loss(val_loss=10.0)
        artifacts = ArtifactRegistry(tmp_path / "registry")
        baseline_source = tmp_path / "baseline"
        baseline_source.mkdir()
        (baseline_source / "w.bin").write_bytes(b"baseline")
        baseline = artifacts.publish(
            baseline_source,
            ArtifactSpec(
                verifier_model="v",
                draft_model="d",
                base_draft="b",
                trace_hash="h",
                training_params={},
                val_loss=1.0,
            ),
        )
        artifacts.promote(baseline.artifact_id, gate_passed=True)

        activity = type("FA", (), {"in_flight": 0, "last_activity": 0.0})()

        @dataclass
        class FakeRuntime:
            calls: list[str] = field(default_factory=list)

            def quiesce(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("quiesce")

            def sleep(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("sleep")

            def start_candidate(
                self,
                draft_directory: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> None:
                self.calls.append("start_candidate")

            def restore(
                self, active_draft: Path | str, *, timeout_seconds: float
            ) -> None:
                self.calls.append("restore")

            def wake(self, *, timeout_seconds: float) -> None:
                self.calls.append("wake")

        @dataclass
        class FakeGate:
            called: bool = False

            def benchmark(
                self,
                candidate_draft: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> GateResult:
                self.called = True
                return GateResult(True, "passed")

        fake_gate = FakeGate()
        fake_runtime = FakeRuntime()
        state = TunerStateMachine(tmp_path / "state")

        orchestrator = TunerOrchestrator(
            state=state,
            idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
            backend=backend,
            artifacts=artifacts,
            runtime=fake_runtime,
            gate=fake_gate,
            work_root=tmp_path / "work",
            run_id_factory=lambda: "run-1",
            val_loss_prefilter=ValLossPreFilterConfig(enabled=False),
        )

        result = orchestrator.run_once()

        assert fake_gate.called
        assert result.outcome is CycleOutcome.PROMOTED

    def test_prefilter_benchmarks_when_no_incumbent(
        self, tmp_path: Path
    ) -> None:
        """When there is no active artifact (first cycle), benchmark normally."""
        from speedlm.tuner.artifacts import ArtifactRegistry
        from speedlm.tuner.idle import IdleDetector
        from speedlm.tuner.orchestrator import TunerOrchestrator
        from speedlm.tuner.state import TunerStateMachine

        backend = _build_fake_backend_with_val_loss(val_loss=2.5)
        artifacts = ArtifactRegistry(tmp_path / "registry")
        # No baseline published — no incumbent

        activity = type("FA", (), {"in_flight": 0, "last_activity": 0.0})()

        @dataclass
        class FakeRuntime:
            calls: list[str] = field(default_factory=list)

            def quiesce(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("quiesce")

            def sleep(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("sleep")

            def start_candidate(
                self,
                draft_directory: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> None:
                self.calls.append("start_candidate")

            def restore(
                self, active_draft: Path | str, *, timeout_seconds: float
            ) -> None:
                self.calls.append("restore")

            def wake(self, *, timeout_seconds: float) -> None:
                self.calls.append("wake")

        @dataclass
        class FakeGate:
            called: bool = False

            def benchmark(
                self,
                candidate_draft: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> GateResult:
                self.called = True
                return GateResult(True, "passed")

        fake_gate = FakeGate()
        fake_runtime = FakeRuntime()
        state = TunerStateMachine(tmp_path / "state")

        orchestrator = TunerOrchestrator(
            state=state,
            idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
            backend=backend,
            artifacts=artifacts,
            runtime=fake_runtime,
            gate=fake_gate,
            work_root=tmp_path / "work",
            run_id_factory=lambda: "run-1",
            val_loss_prefilter=ValLossPreFilterConfig(
                enabled=True, min_improvement=0.01
            ),
        )

        result = orchestrator.run_once()

        assert fake_gate.called
        assert result.outcome is CycleOutcome.PROMOTED

    def test_prefilter_skips_when_incumbent_val_loss_none(
        self, tmp_path: Path
    ) -> None:
        """When incumbent has no val_loss, fall through to benchmark (fail open)."""
        from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec
        from speedlm.tuner.idle import IdleDetector
        from speedlm.tuner.orchestrator import TunerOrchestrator
        from speedlm.tuner.state import TunerStateMachine

        backend = _build_fake_backend_with_val_loss(val_loss=2.5)
        artifacts = ArtifactRegistry(tmp_path / "registry")
        # Incumbent has NO val_loss
        baseline_source = tmp_path / "baseline"
        baseline_source.mkdir()
        (baseline_source / "w.bin").write_bytes(b"baseline")
        baseline = artifacts.publish(
            baseline_source,
            ArtifactSpec(
                verifier_model="v",
                draft_model="d",
                base_draft="b",
                trace_hash="h",
                training_params={},
                val_loss=None,
            ),
        )
        artifacts.promote(baseline.artifact_id, gate_passed=True)

        activity = type("FA", (), {"in_flight": 0, "last_activity": 0.0})()

        @dataclass
        class FakeRuntime:
            calls: list[str] = field(default_factory=list)

            def quiesce(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("quiesce")

            def sleep(
                self, *, timeout_seconds: float, should_abort: Callable[[], bool]
            ) -> None:
                self.calls.append("sleep")

            def start_candidate(
                self,
                draft_directory: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> None:
                self.calls.append("start_candidate")

            def restore(
                self, active_draft: Path | str, *, timeout_seconds: float
            ) -> None:
                self.calls.append("restore")

            def wake(self, *, timeout_seconds: float) -> None:
                self.calls.append("wake")

        @dataclass
        class FakeGate:
            called: bool = False

            def benchmark(
                self,
                candidate_draft: Path,
                *,
                timeout_seconds: float,
                should_abort: Callable[[], bool],
            ) -> GateResult:
                self.called = True
                return GateResult(True, "passed")

        fake_gate = FakeGate()
        fake_runtime = FakeRuntime()
        state = TunerStateMachine(tmp_path / "state")

        orchestrator = TunerOrchestrator(
            state=state,
            idle=IdleDetector(activity, threshold_seconds=5.0, clock=lambda: 10.0),
            backend=backend,
            artifacts=artifacts,
            runtime=fake_runtime,
            gate=fake_gate,
            work_root=tmp_path / "work",
            run_id_factory=lambda: "run-1",
            val_loss_prefilter=ValLossPreFilterConfig(
                enabled=True, min_improvement=0.01
            ),
        )

        result = orchestrator.run_once()

        # Incumbent has no val_loss, so pre-filter falls through
        assert fake_gate.called
        assert result.outcome is CycleOutcome.PROMOTED