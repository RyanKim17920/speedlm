"""Production composition for one opt-in idle-tuning service."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

from speedlm.config import SpeedLMConfig
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.capture import CaptureManager
from speedlm.gateway.control import (
    AdmissionGate,
    ControlAborted,
    GPUMemoryPrecondition,
    NvidiaSmiMemoryProbe,
    RuntimeController,
)
from speedlm.gateway.process import LOOPBACK_HOST, build_vllm_argv
from speedlm.gateway.supervisor import ThreadsafeProcessControl
from speedlm.gateway.vllm_http import VLLMControlClient
from speedlm.profiles import (
    ModelProfile,
    canonical_verifier_reference,
    resolve_profile,
)
from speedlm.storage import ensure_layout
from speedlm.traces.store import TraceStore
from speedlm.training import check_prepared_dataset
from speedlm.training.backends.eagle3 import (
    Eagle3Backend,
    SpeculatorsPipelineConfig,
)
from speedlm.training.split import HeldOutTraceSnapshotLeaser
from speedlm.tuner.artifacts import ArtifactRegistry
from speedlm.tuner.service import TunerService, create_tuner_service
from speedlm.tuner.state import TunerStateMachine

DraftReference = Path | str
AbortCheck = Callable[[], bool]


class ProductionTuningError(RuntimeError):
    """Raised before serving when an opt-in production contract is incomplete."""


@dataclass(frozen=True, slots=True)
class TuningLaunchPlan:
    profile: ModelProfile
    active_draft: DraftReference
    argv_factory: Callable[[DraftReference], list[str]]


def build_tuning_launch_plan(
    config: SpeedLMConfig,
    *,
    passthrough: Sequence[str],
    child_port: int,
    home: Path,
) -> TuningLaunchPlan:
    """Validate a trainable profile and build generation-safe vLLM argv."""
    profile = resolve_profile(
        {"model": config.model, "profile": config.profile},
        served_model=config.model,
        home=home,
    )
    if canonical_verifier_reference(
        profile.verifier_model
    ) != canonical_verifier_reference(config.model):
        raise ProductionTuningError(
            f"profile {profile.name!r} verifier {profile.verifier_model!r} does not "
            f"match served model {config.model!r}"
        )
    if profile.speculative_method != "eagle3":
        raise ProductionTuningError(
            f"idle tuning backend {profile.speculative_method!r} is not yet "
            "production-validated"
        )
    if profile.draft_model is None:
        raise ProductionTuningError(
            f"profile {profile.name!r} has no warm-start draft"
        )
    if _has_option(passthrough, "--speculative-config"):
        raise ProductionTuningError(
            "--speculative-config is owned by idle tuning; configure the model profile"
        )
    _validate_training_environment(config)

    layout = ensure_layout(home)
    artifacts = ArtifactRegistry(layout.runs_dir)
    active = artifacts.active()
    if active is not None and (
        active.manifest.verifier_model != profile.verifier_model
        or active.manifest.draft_model != profile.draft_model
    ):
        raise ProductionTuningError(
            f"active artifact {active.artifact_id} belongs to verifier/draft "
            f"{active.manifest.verifier_model!r}/{active.manifest.draft_model!r}, "
            f"not {profile.verifier_model!r}/{profile.draft_model!r}"
        )
    active_draft: DraftReference = (
        active.path if active is not None else profile.draft_model
    )
    base_passthrough = [
        argument
        for argument in passthrough
        if argument != "--enable-sleep-mode"
    ]
    alias_supplied, supplied_alias = _option_value(
        base_passthrough,
        "--served-model-name",
    )
    if alias_supplied and supplied_alias != config.alias:
        raise ProductionTuningError(
            f"--served-model-name must match configured model alias "
            f"{config.alias!r}, got {supplied_alias!r}"
        )
    if config.model_alias and not alias_supplied:
        base_passthrough.extend(("--served-model-name", config.model_alias))

    def argv_factory(draft: DraftReference) -> list[str]:
        speculative = profile.speculative_config()
        speculative["model"] = str(draft)
        return build_vllm_argv(
            config.model,
            [
                *base_passthrough,
                "--enable-sleep-mode",
                "--speculative-config",
                json.dumps(speculative, separators=(",", ":"), sort_keys=True),
            ],
            host=LOOPBACK_HOST,
            port=child_port,
        )

    return TuningLaunchPlan(
        profile=profile,
        active_draft=active_draft,
        argv_factory=argv_factory,
    )


def create_production_tuner(
    config: SpeedLMConfig,
    *,
    profile: ModelProfile,
    active_draft: DraftReference,
    activity: ActivityTracker,
    admission: AdmissionGate,
    traces: TraceStore,
    capture: CaptureManager,
    process: ThreadsafeProcessControl,
    http: VLLMControlClient,
    child_url: str,
    loop: asyncio.AbstractEventLoop,
    home: Path,
) -> TunerService:
    """Compose every concrete collaborator used by the background service."""
    if profile.speculative_method != "eagle3" or profile.draft_model is None:
        raise ProductionTuningError(
            f"profile {profile.name!r} is not supported by the EAGLE-3 backend"
        )
    tuning = config.tuning
    layout = ensure_layout(home)
    state = TunerStateMachine(layout.runs_dir)
    artifacts = ArtifactRegistry(layout.runs_dir)
    split = HeldOutTraceSnapshotLeaser(
        traces,
        held_out_fraction=tuning.held_out_fraction,
        scratch_quota_bytes=tuning.scratch_quota_bytes,
        training_window_records=tuning.training_window_records,
    )
    pipeline = SpeculatorsPipelineConfig(
        prepared_validator_script=_validator_script(config),
        row_count=None,
        speculators_repo=Path(_required(tuning.speculators_repo, "speculators_repo")),
        training_python=Path(_required(tuning.training_python, "training_python")),
        vllm_python=(
            Path(tuning.vllm_python) if tuning.vllm_python is not None else None
        ),
        verifier_model=profile.verifier_model,
        warm_start_model=profile.draft_model,
        target_layer_ids=profile.target_layer_ids or (),
        sequence_length=min(tuning.sequence_length, profile.max_seq_len),
        learning_rate=tuning.learning_rate,
        epochs=tuning.epochs,
        port=tuning.training_port,
        concurrency=tuning.concurrency,
        scratch_quota_bytes=tuning.scratch_quota_bytes,
    )
    backend = Eagle3Backend.from_speculators(
        pipeline,
        trace_leaser=split,
    )
    runtime = RuntimeController(
        activity=activity,
        admission=admission,
        http=http,
        process=process,
        active_draft=active_draft,
        capture_barrier=_CaptureBarrier(capture, loop),
        # The hidden-state engine started right after sleep is launched with
        # exactly this utilization, so it is also exactly what must be free.
        gpu_memory=GPUMemoryPrecondition(
            probe=NvidiaSmiMemoryProbe(),
            required_fraction=pipeline.gpu_memory_utilization,
        ),
    )
    endpoint = _DraftEndpoint(
        url=child_url,
        process=process,
        http=http,
    )
    gate = BenchmarkGateRunner(
        config=config,
        trace_source=traces,
        suite_dir=lambda: split.suite_dir,
        stock_draft=active_draft,
        endpoint=endpoint,
        metrics_source=_MetricsSource(http),
        repeats=tuning.benchmark_repeats,
        held_out_fraction=tuning.held_out_fraction,
        training_context_hashes=lambda: split.training_context_hashes,
    )
    return create_tuner_service(
        config,
        activity=activity,
        traces=traces,
        backend=backend,
        gate=gate,
        runtime=runtime,
        enabled=True,
        min_trace_records=tuning.min_trace_records,
        poll_interval_seconds=tuning.poll_interval_seconds,
        home=home,
        state=state,
        artifacts=artifacts,
    )


class _CaptureBarrier:
    def __init__(
        self,
        capture: CaptureManager,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._capture = capture
        self._loop = loop

    def __call__(self, timeout_seconds: float, should_abort: AbortCheck) -> None:
        future = asyncio.run_coroutine_threadsafe(self._capture.drain(), self._loop)
        deadline = self._loop.time() + timeout_seconds
        try:
            while True:
                if should_abort():
                    future.cancel()
                    raise ControlAborted("capture barrier aborted")
                remaining = deadline - self._loop.time()
                if remaining <= 0:
                    future.cancel()
                    raise TimeoutError("capture barrier timed out")
                try:
                    future.result(timeout=min(0.05, remaining))
                    return
                except FutureTimeout:
                    continue
        except BaseException:
            if not future.done():
                future.cancel()
            raise


@dataclass(slots=True)
class _DraftEndpoint:
    url: str
    process: ThreadsafeProcessControl
    http: VLLMControlClient

    def activate(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        if should_abort():
            raise ControlAborted("draft activation aborted")
        self.process.restart(draft, timeout_seconds=timeout_seconds)
        self.http.wait_ready(
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )


class _MetricsSource:
    def __init__(self, http: VLLMControlClient) -> None:
        self._http = http

    def scrape(
        self,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> str:
        if should_abort():
            raise ControlAborted("metrics scrape aborted")
        result = self._http.read_metrics(timeout_seconds=timeout_seconds)
        if should_abort():
            raise ControlAborted("metrics scrape aborted")
        return result


def _validator_script(config: SpeedLMConfig) -> Path:
    configured = config.tuning.prepared_validator_script
    if configured is not None:
        return Path(configured)
    module_path = Path(check_prepared_dataset.__file__)
    if not module_path.is_file():
        raise ProductionTuningError("packaged prepared-dataset validator is missing")
    return module_path


def _validate_training_environment(config: SpeedLMConfig) -> None:
    tuning = config.tuning
    repo = Path(_required(tuning.speculators_repo, "speculators_repo"))
    python = Path(_required(tuning.training_python, "training_python"))
    required = (
        repo / "scripts" / "prepare_data.py",
        repo / "scripts" / "launch_vllm.py",
        repo / "scripts" / "data_generation_offline.py",
        repo / "scripts" / "train.py",
        python,
        _validator_script(config),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProductionTuningError(
            "idle tuning dependencies are missing: " + ", ".join(missing)
        )


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ProductionTuningError(
            f"tuning.{name} must be configured when idle tuning is enabled"
        )
    return value


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _option_value(
    arguments: Sequence[str],
    option: str,
) -> tuple[bool, str | None]:
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            return True, argument.partition("=")[2] or None
        if argument == option:
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                return True, arguments[index + 1]
            return True, None
    return False, None


__all__ = [
    "ProductionTuningError",
    "TuningLaunchPlan",
    "build_tuning_launch_plan",
    "create_production_tuner",
]
