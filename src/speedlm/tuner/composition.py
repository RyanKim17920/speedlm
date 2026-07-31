"""Production composition for one opt-in idle-tuning service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
    resolve_target_layer_ids,
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

logger = logging.getLogger(__name__)

DraftReference = Path | str
AbortCheck = Callable[[], bool]


class ProductionTuningError(RuntimeError):
    """Raised before serving when an opt-in production contract is incomplete."""


def resolve_verifier_revision(
    verifier_model: str,
    configured: str | None,
    *,
    resolver: Callable[[str], str | None] | None = None,
) -> str | None:
    """Return the immutable revision to pin the verifier to, or ``None``.

    Nothing pinned the verifier before this: the pipeline left
    ``verifier_revision`` at ``None`` and the backend passed the bare model
    string through, so an upstream Hub change silently altered both what was
    trained and what it was benchmarked against, with no artifact field that
    would show it had happened.

    Resolution is *best effort*, deliberately.  Pinning is a reproducibility
    improvement, not a safety-critical precondition, so it must never be able
    to stop the service from starting.  An unresolvable revision costs one
    provenance field -- recorded as a null ``verifier_revision`` in the
    artifact manifest, so a reader can see the cycle was unpinned rather than
    assume it was pinned -- and nothing else.

    Resolution prefers the local Hub cache over any network call.  The cache
    ref is authoritative for what this host will actually load, it is readable
    without importing ``huggingface_hub`` (which is part of the out-of-band GPU
    stack, not a declared dependency of this wrapper), and it works under
    ``HF_HUB_OFFLINE``.  A Hub lookup is only the fallback, and is imported
    lazily so a host without the heavy stack still starts.  A verifier given
    as a local filesystem path is exempt: the path is its own pin.

    Args:
        verifier_model: Hub repo id or local path from the resolved profile.
        configured: ``tuning.verifier_revision``, when the operator pinned one
            explicitly -- for example to reproduce an archived run.
        resolver: Seam for tests; defaults to cache-first resolution.
    """
    if configured is not None:
        return configured
    if Path(verifier_model).exists():
        return None
    lookup = resolver if resolver is not None else _resolve_revision
    try:
        revision = lookup(verifier_model)
    except Exception:
        logger.warning(
            "could not resolve a revision for verifier %r; the cycle will run "
            "unpinned and the manifest will record a null verifier_revision",
            verifier_model,
            exc_info=True,
        )
        return None
    if not isinstance(revision, str) or not revision:
        logger.warning(
            "verifier %r resolved to no revision; the cycle will run unpinned "
            "and the manifest will record a null verifier_revision",
            verifier_model,
        )
        return None
    logger.info("pinned verifier %r to revision %s", verifier_model, revision)
    return revision


def _resolve_revision(verifier_model: str) -> str | None:
    """Resolve from the local Hub cache first, then fall back to the Hub."""
    cached = cached_hub_revision(verifier_model)
    if cached is not None:
        return cached
    return _hub_revision(verifier_model)


def cached_hub_revision(repo_id: str, *, ref: str = "main") -> str | None:
    """Return the commit a locally cached Hub ref points at, if any.

    This reads ``<cache>/models--<org>--<name>/refs/<ref>`` directly rather
    than calling into ``huggingface_hub``.  The layout is the on-disk contract
    every cached download already writes, and reading it keeps the startup
    path free of an undeclared import.
    """
    folder = "models--" + repo_id.replace("/", "--")
    for root in _hub_cache_roots():
        try:
            revision = (root / folder / "refs" / ref).read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            continue
        if revision:
            return revision
    return None


def _hub_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(roots)


def _hub_revision(verifier_model: str) -> str | None:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError:
        # The heavy runtime stack is installed out-of-band on the GPU host and
        # is deliberately not a declared dependency (see pyproject.toml), so
        # its absence is an ordinary configuration, not a startup failure.
        return None
    return str(HfApi().model_info(verifier_model).sha)


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
        verifier_revision=resolve_verifier_revision(
            profile.verifier_model, tuning.verifier_revision
        ),
        warm_start_model=profile.draft_model,
        target_layer_ids=profile.target_layer_ids or (
            resolve_target_layer_ids(
                explicit=None,
                num_hidden_layers=profile.num_hidden_layers,
                drafter_aux_count=None,
            )
            if profile.num_hidden_layers is not None
            else ()
        ),
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
        min_corpus_records=tuning.min_corpus_records,
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
    "cached_hub_revision",
    "create_production_tuner",
    "resolve_verifier_revision",
]
