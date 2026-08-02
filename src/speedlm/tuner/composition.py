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
from typing import Final

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
from speedlm.gateway.vllm_http import VLLMControlClient, VLLMDraftSwapClient
from speedlm.profiles import (
    ModelProfile,
    cached_snapshot_dir,
    canonical_verifier_reference,
    drafter_declared_speculative_tokens,
    resolve_profile,
    resolve_speculative_tokens,
    resolve_target_layer_ids,
    validate_training_depth,
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
from speedlm.tuner.eagle3 import Eagle3Adapter
from speedlm.tuner.service import TunerService, create_tuner_service
from speedlm.tuner.state import TunerStateMachine

logger = logging.getLogger(__name__)

DraftReference = Path | str
AbortCheck = Callable[[], bool]

WORKER_EXTENSION_OPTION: Final = "--worker-extension-cls"
#: vLLM resolves this with ``qualname.rsplit(".", 1)`` then ``getattr``, so the
#: dotted ``pkg.mod.Class`` form is the only one that works -- ``pkg.mod:Class``
#: would try to import ``pkg`` and look up an attribute named ``mod:Class``.
DRAFT_SWAP_WORKER_EXTENSION_CLS: Final = (
    "speedlm.gateway.draft_swap.CombinedWorkerExtension"
)


class ProductionTuningError(RuntimeError):
    """Raised before serving when an opt-in production contract is incomplete."""


@dataclass(slots=True)
class _PublishedWeightGuard:
    """Deferred binding of the publish-time weight assertion.

    :class:`~speedlm.tuner.artifacts.ArtifactRegistry` needs its
    ``before_publish`` hook at construction, but the adapter that knows which
    weights were trained is built from a pipeline config that the registry
    itself feeds (the warm-start resolver reads the registry).  Rather than
    reorder the two and rely on a closure over a not-yet-assigned local, the
    hook is this object and the adapter is bound into it a few lines later.

    Unbound is a programming error, not a permission to skip the check: a
    guard that silently passed when nothing was wired would reproduce exactly
    the "ruled out by inference" gap it exists to close.
    """

    _adapter: Eagle3Adapter | None = None

    def bind(self, adapter: Eagle3Adapter) -> None:
        self._adapter = adapter

    def __call__(self, published: Path) -> None:
        if self._adapter is None:
            raise ProductionTuningError(
                "publish-time weight guard was never bound to an adapter"
            )
        self._adapter.assert_published_weights(published)


def declared_draft_depth(draft_model: str) -> int | None:
    """Return the chain depth *draft_model*'s cached config declares, if any.

    Best effort, and deliberately so: this is provenance, not a precondition.
    A drafter absent from the local cache, or one whose config cannot be read,
    yields ``None`` and the profile's serving depth stands on its own.
    """
    directory = cached_snapshot_dir(draft_model)
    if directory is None:
        return None
    try:
        raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return drafter_declared_speculative_tokens(raw)


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


def active_draft_reference(
    artifacts: ArtifactRegistry,
    fallback: DraftReference,
) -> DraftReference:
    """The draft serving live traffic *now*, or *fallback* if none was promoted.

    The registry is the durable source of truth and it moves: every promotion
    rewrites the active pointer.  Anything that needs to name the incumbent --
    notably the gate's stock arm -- has to ask at the moment it needs the
    answer, which is what this exists to make easy to do correctly.  It mirrors
    :meth:`speedlm.tuner.orchestrator.TunerOrchestrator._active_draft`, whose
    fallback is the backend's ``from_pretrained``, i.e. the same warm-start
    draft this one falls back to.
    """
    active = artifacts.active()
    if active is not None:
        return active.path
    return fallback


#: Hard stop on a ``base_draft`` walk, so a corrupted or hand-edited manifest
#: chain cannot spin.  Any real chain is bounded by the number of promotions
#: this installation has ever made.
_MAX_CHAIN_WALK: Final = 1_000


def promotion_chain_depth(reference: DraftReference) -> int:
    """How many trained artifacts deep *reference* sits above the stock drafter.

    ``0`` for the profile's stock drafter (a Hub repo id, or any directory that
    is not a published artifact), ``1`` for a head trained from it, and so on.

    The chain is read from the artifacts themselves rather than from the active
    pointer's ``history``.  ``history`` records every promotion, including ones
    made while compounding was switched off, so it counts *promotions* and not
    *ancestry*; ``base_draft`` records what a cycle actually trained from and is
    therefore the only field that answers this question.

    The manifest is read directly, without
    :meth:`~speedlm.tuner.artifacts.ArtifactRegistry.get`'s content
    verification: this feeds a policy decision about where to warm-start, not a
    trust decision about what to load, and re-hashing every ancestor would cost
    gigabytes of reads per cycle.  An unreadable link ends the walk, which
    biases the depth *down* -- toward compounding rather than toward an
    unrequested re-baseline.
    """
    depth = 0
    seen: set[str] = set()
    current = str(reference)
    while depth < _MAX_CHAIN_WALK:
        if current in seen:
            # A manifest chain that revisits a node is corrupt, not infinite.
            return depth
        seen.add(current)
        try:
            raw = json.loads(
                (Path(current) / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return depth
        if not isinstance(raw, dict):
            return depth
        base = raw.get("base_draft")
        if not isinstance(base, str) or not base:
            # A readable artifact whose ancestry is unusable still counts
            # itself; only its ancestors are lost.
            return depth + 1
        depth += 1
        current = base
    logger.warning(
        "stopped walking the warm-start chain at %d links; treating it as "
        "unbounded",
        _MAX_CHAIN_WALK,
    )
    return depth


def warm_start_reference(
    artifacts: ArtifactRegistry,
    stock: str,
    *,
    max_chain_depth: int | None = None,
) -> str:
    """The checkpoint the *next* cycle should train from.

    This is the training-side twin of :func:`active_draft_reference`, and it
    exists for the same reason: the incumbent is a durable pointer that moves,
    so a value captured at composition time is wrong from the second cycle on.
    Frozen, every cycle re-ran the same one-shot fine-tune of the *stock*
    speculator and threw the previous cycle's promoted head away -- so learning
    could not accumulate, which is the entire premise of idle tuning.

    Falls back to *stock* when the registry has no active artifact: the first
    cycle of a fresh installation, and any state a rollback leaves with nothing
    promoted.

    ``max_chain_depth`` bounds compounding.  There is no post-promotion
    rollback anywhere in this system, so a chain accumulates its own mistakes
    as readily as its own gains; past the bound a cycle deliberately
    re-baselines to *stock* and lets the gate compare that head against the
    deep incumbent.  ``None``, the default, does not bound it -- see
    :attr:`speedlm.config.IdleTuningConfig.warm_start_max_chain_depth` for why
    an invented bound would be worse than none.

    **What this cannot protect against.**  Warm-starting repeatedly from a head
    fine-tuned on one traffic slice walks the drafter toward that slice.  The
    gate does not see it: its held-out suite is split from the *same* captured
    traffic (:class:`~speedlm.training.split.HeldOutTraceSnapshotLeaser`), so a
    head that has become excellent on this deployment's traffic and worse in
    general passes every arm of it.  Speculative decoding is lossless, so the
    cost is throughput on unlike traffic and never wrong answers -- but it is a
    real cost, it compounds, and nothing in this process measures it.
    """
    active = artifacts.active()
    if active is None:
        return stock
    if max_chain_depth is not None:
        depth = promotion_chain_depth(active.path)
        if depth >= max_chain_depth:
            logger.info(
                "warm-start chain for artifact %s is %d deep, at or past the "
                "configured bound of %d; re-baselining this cycle to the stock "
                "drafter %s",
                active.artifact_id,
                depth,
                max_chain_depth,
                stock,
            )
            return stock
    return str(active.path)


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
    hot_swap_enabled = config.tuning.draft_hot_swap_enabled
    if hot_swap_enabled and _has_option(passthrough, WORKER_EXTENSION_OPTION):
        # vLLM takes exactly one worker extension class and mixes it into the
        # worker's bases; a second one has nowhere to go. Refusing here means
        # the operator learns at startup validation instead of watching every
        # engine launch fail with an attribute-collision assert.
        raise ProductionTuningError(
            f"tuning.draft_hot_swap_enabled needs {WORKER_EXTENSION_OPTION} "
            f"{DRAFT_SWAP_WORKER_EXTENSION_CLS}, but vLLM accepts exactly one "
            f"worker extension class and {WORKER_EXTENSION_OPTION} was already "
            "passed through; drop the passthrough option (its capabilities are "
            "already in the combined class) or disable "
            "tuning.draft_hot_swap_enabled"
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
                *(
                    (WORKER_EXTENSION_OPTION, DRAFT_SWAP_WORKER_EXTENSION_CLS)
                    if hot_swap_enabled
                    else ()
                ),
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
    #: The profile's stock speculator, bound once here so the warm-start
    #: resolver below closes over a narrowed ``str`` rather than re-deriving it.
    stock_draft_model = profile.draft_model
    tuning = config.tuning
    #: The one depth this profile trains *and* serves.
    #:
    #: The profile's ``num_speculative_tokens`` is what vLLM is told to
    #: speculate, so it is the truth; ``resolve_speculative_tokens`` states
    #: that precedence in one place rather than leaving the trainer to fall
    #: through to a default of 3.  gpt-oss-20b serves 5 and used to train 3.
    stock_declared_depth = declared_draft_depth(stock_draft_model)
    training_depth = resolve_speculative_tokens(
        explicit=profile.num_speculative_tokens,
        drafter_declared=stock_declared_depth,
    )
    if stock_declared_depth is not None and stock_declared_depth != training_depth:
        # Not a fault: both stock RedHatAI EAGLE-3 drafters declare 3, and
        # deepening that head is what a tuning cycle is *for*.  It is logged
        # because the cycle is then genuinely training past the depth its warm
        # start was fitted at, and that should be visible rather than inferred.
        logger.info(
            "profile %r serves %d speculative tokens while its stock drafter %r "
            "declares %d; this cycle trains the deeper chain",
            profile.name,
            training_depth,
            stock_draft_model,
            stock_declared_depth,
        )
    validate_training_depth(
        profile_name=profile.name,
        serving_tokens=profile.num_speculative_tokens,
        training_steps=training_depth,
    )
    layout = ensure_layout(home)
    state = TunerStateMachine(layout.runs_dir)
    publish_guard = _PublishedWeightGuard()
    artifacts = ArtifactRegistry(layout.runs_dir, before_publish=publish_guard)
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
        warm_start_model=stock_draft_model,
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
        num_speculative_steps=training_depth,
        learning_rate=tuning.learning_rate,
        epochs=tuning.epochs,
        port=tuning.training_port,
        concurrency=tuning.extraction_concurrency,
        # The two halves of the same knob.  ``concurrency`` is how many
        # requests ``data_generation_offline.py`` keeps in flight;
        # ``max_num_seqs`` is how many the hidden-state engine will schedule at
        # once, and it was never wired -- so the engine ran at the dataclass
        # default of 1 and every prefill the client issued serialized behind
        # the previous one.  On job 369040 that made 408 extraction prefills a
        # strictly sequential queue against a client driving four.  Setting the
        # scheduler width to the offered load is what makes the configured
        # concurrency mean anything; it cannot exceed what the client sends, so
        # it adds no memory pressure beyond the requests already in flight.
        max_num_seqs=tuning.extraction_concurrency,
        scratch_quota_bytes=tuning.scratch_quota_bytes,
    )
    backend = Eagle3Backend.from_speculators(
        pipeline,
        trace_leaser=split,
        # Resolved per cycle, not captured here.  The profile's stock speculator
        # -- the same string ``pipeline.warm_start_model`` carries -- stays the
        # fallback; this is what lets cycle N+1 build on cycle N's promotion
        # instead of re-running the same from-stock fine-tune forever.  See
        # ``warm_start_reference`` for the bound and for the drift it explicitly
        # does not cover.
        warm_start_resolver=(
            (
                lambda: warm_start_reference(
                    artifacts,
                    stock_draft_model,
                    max_chain_depth=tuning.warm_start_max_chain_depth,
                )
            )
            if tuning.compounding_warm_start
            else None
        ),
    )
    publish_guard.bind(backend)
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
        # Off by default: without the worker extension registered, the swap RPC
        # has no handler, so wiring a client in would only manufacture failures.
        draft_swap_http=(
            VLLMDraftSwapClient(http) if tuning.draft_hot_swap_enabled else None
        ),
        restore_fast_path_timeout_seconds=tuning.restore_fast_path_timeout_seconds,
    )
    endpoint = _DraftEndpoint(
        url=child_url,
        process=process,
        http=http,
        runtime=runtime,
    )
    gate = BenchmarkGateRunner(
        config=config,
        trace_source=traces,
        suite_dir=lambda: split.suite_dir,
        # Resolved per benchmark, not captured here.  A frozen reference made
        # every cycle after the first measure the candidate against the draft
        # that was active when the *process started* rather than against the
        # one actually serving, so the reported delta was cumulative gain over
        # the original head instead of marginal gain over the incumbent -- and
        # the gate is the only safeguard, with no rollback behind it.
        stock_draft=lambda: active_draft_reference(artifacts, active_draft),
        endpoint=endpoint,
        metrics_source=_MetricsSource(http),
        repeats=tuning.benchmark_repeats,
        # Was the runner's own default of 1 and unreachable from config, which
        # left the arms' cold start unmeasurable in production; see
        # ``IdleTuningConfig.warmup_repeats``.  The default is unchanged.
        warmup_repeats=tuning.warmup_repeats,
        replay_concurrency=tuning.benchmark_concurrency,
        correctness_max_tokens=tuning.correctness_max_tokens,
        benchmark_max_tokens=tuning.benchmark_max_tokens,
        held_out_fraction=tuning.held_out_fraction,
        training_context_hashes=lambda: split.training_context_hashes,
        candidate_arm_first=tuning.benchmark_candidate_arm_first,
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
    """Select the draft one vLLM child serves, on behalf of the gate.

    This restarts the *same* child that :class:`RuntimeController` manages, so
    the two must not keep independent ideas of what is running.  ``runtime`` is
    both the source of truth consulted before a restart and the object told
    about one afterwards; leaving it ``None`` restores the unconditional
    restart behaviour and is what a test that owns no controller should do.
    """

    url: str
    process: ThreadsafeProcessControl
    http: VLLMControlClient
    runtime: RuntimeController | None = None

    def activate(
        self,
        draft: DraftReference,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck,
    ) -> None:
        if should_abort():
            raise ControlAborted("draft activation aborted")
        # The benchmark's first arm asks for a draft the cycle has *already*
        # started -- CANDIDATE_STARTING spends a full engine launch on it -- so
        # activating unconditionally threw that engine away and paid a second
        # launch for the identical configuration.  Readiness is still confirmed
        # on the cheap path, so the arm never begins against an engine that
        # cannot answer.
        if self.runtime is not None and self.runtime.matches_running_draft(draft):
            self.http.wait_ready(
                timeout_seconds=timeout_seconds,
                should_abort=should_abort,
            )
            logger.info("draft %s is already serving; skipping engine restart", draft)
            return
        self.process.restart(draft, timeout_seconds=timeout_seconds)
        self.http.wait_ready(
            timeout_seconds=timeout_seconds,
            should_abort=should_abort,
        )
        # Only after the replacement child is ready: an unfinished restart must
        # not be recorded as the engine's identity.
        if self.runtime is not None:
            self.runtime.note_external_restart(draft)


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
    "DRAFT_SWAP_WORKER_EXTENSION_CLS",
    "WORKER_EXTENSION_OPTION",
    "ProductionTuningError",
    "TuningLaunchPlan",
    "active_draft_reference",
    "build_tuning_launch_plan",
    "cached_hub_revision",
    "create_production_tuner",
    "promotion_chain_depth",
    "resolve_verifier_revision",
    "warm_start_reference",
]
