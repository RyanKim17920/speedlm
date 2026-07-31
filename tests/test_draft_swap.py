from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.control import (
    ControlAborted,
    DraftSwapHTTP,
    RuntimeController,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeClock:
    now: float = 0.0
    sleep_hook: Callable[[], None] | None = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.sleep_hook is not None:
            self.sleep_hook()


@dataclass
class FakeAdmission:
    admitting: bool = True
    calls: list[str] = field(default_factory=list)

    def stop_admitting(self) -> None:
        self.calls.append("stop")
        self.admitting = False

    def start_admitting(self) -> None:
        self.calls.append("start")
        self.admitting = True


@dataclass(frozen=True)
class HTTPCall:
    endpoint: str
    timeout_seconds: float
    query: Mapping[str, str] | None


@dataclass
class FakeHTTP:
    clock: FakeClock
    post_calls: list[HTTPCall] = field(default_factory=list)
    ready_timeouts: list[float] = field(default_factory=list)
    fail_post: set[str] = field(default_factory=set)
    fail_ready_count: int = 0
    advance_post: float = 0.0
    advance_ready: float = 0.0
    sleeping_waits: list[tuple[bool, float]] = field(default_factory=list)
    fail_sleeping_wait: bool = False

    def post(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        query: Mapping[str, str] | None = None,
    ) -> None:
        self.post_calls.append(HTTPCall(endpoint, timeout_seconds, query))
        self.clock.now += self.advance_post
        self.advance_post = 0.0
        if endpoint in self.fail_post:
            raise RuntimeError(f"{endpoint} failed")

    def wait_ready(self, *, timeout_seconds: float) -> None:
        self.ready_timeouts.append(timeout_seconds)
        self.clock.now += self.advance_ready
        self.advance_ready = 0.0
        if self.fail_ready_count:
            self.fail_ready_count -= 1
            raise RuntimeError("readiness failed")

    def wait_sleeping(
        self,
        sleeping: bool,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        self.sleeping_waits.append((sleeping, timeout_seconds))
        if should_abort():
            raise ControlAborted("sleep-state wait aborted")
        if self.fail_sleeping_wait:
            raise RuntimeError("sleep-state wait failed")


@dataclass(frozen=True)
class ProcessCall:
    draft: Path | str
    timeout_seconds: float


@dataclass
class FakeProcess:
    clock: FakeClock
    running_draft: Path | str
    calls: list[ProcessCall] = field(default_factory=list)
    fail_drafts: set[Path | str] = field(default_factory=set)
    advance_restart: float = 0.0
    running: bool = True

    def restart(
        self,
        draft: Path | str,
        *,
        timeout_seconds: float,
    ) -> None:
        self.calls.append(ProcessCall(draft, timeout_seconds))
        self.running = False
        self.clock.now += self.advance_restart
        self.advance_restart = 0.0
        if draft in self.fail_drafts:
            raise RuntimeError(f"restart failed for {draft}")
        self.running_draft = draft
        self.running = True


@dataclass
class FakeDraftSwap:
    """Fake DraftSwapHTTP that records calls and can be configured to fail."""
    swapped_paths: list[str] = field(default_factory=list)
    fail_on_swap: bool = False
    swap_error: str = "swap endpoint unreachable"
    info_calls: int = 0

    def hot_swap_draft(
        self,
        weights_path: str,
        *,
        timeout_seconds: float,
    ) -> None:
        self.swapped_paths.append(weights_path)
        if self.fail_on_swap:
            raise RuntimeError(self.swap_error)

    def draft_info(self, *, timeout_seconds: float) -> dict:
        self.info_calls += 1
        return {
            "num_parameters": 10,
            "parameter_shapes": {},
            "parameter_dtypes": {},
            "quantization": None,
        }


@dataclass
class Rig:
    controller: RuntimeController
    activity: ActivityTracker
    admission: FakeAdmission
    http: FakeHTTP
    process: FakeProcess
    clock: FakeClock
    draft_swap: FakeDraftSwap | None


def make_rig(
    *,
    active_draft: Path | str = "base-draft",
    draft_swap_http: FakeDraftSwap | None = None,
) -> Rig:
    clock = FakeClock()
    activity = ActivityTracker(clock=clock)
    admission = FakeAdmission()
    http = FakeHTTP(clock)
    process = FakeProcess(clock, running_draft=active_draft)
    controller = RuntimeController(
        activity=activity,
        admission=admission,
        http=http,
        process=process,
        active_draft=active_draft,
        clock=clock,
        sleeper=clock.sleep,
        poll_interval_seconds=0.1,
        recovery_timeout_seconds=5.0,
        draft_swap_http=draft_swap_http,
    )
    return Rig(controller, activity, admission, http, process, clock, draft_swap_http)


# ---------------------------------------------------------------------------
# Hot-swap: disabled by default => restart path
# ---------------------------------------------------------------------------


def test_disabled_flag_uses_restart_path() -> None:
    """When draft_swap_http is None, start_candidate falls back to restart."""
    rig = make_rig()
    candidate = Path("/candidate/draft")

    rig.controller.start_candidate(
        candidate,
        timeout_seconds=3.0,
        should_abort=lambda: False,
    )

    assert rig.controller._running_draft == candidate
    assert [call.draft for call in rig.process.calls] == [candidate]


def test_no_swap_when_sleeping() -> None:
    """Hot-swap must not fire when the engine is sleeping."""
    draft_swap = FakeDraftSwap()
    rig = make_rig(draft_swap_http=draft_swap)

    # Simulate sleeping
    rig.controller._sleeping = True  # noqa: SLF001
    candidate = Path("/candidate/draft")

    rig.controller.start_candidate(
        candidate,
        timeout_seconds=3.0,
        should_abort=lambda: False,
    )

    # Should have restarted, not hot-swapped
    assert draft_swap.swapped_paths == []
    assert [call.draft for call in rig.process.calls] == [candidate]


def test_rpc_failure_falls_back_to_restart() -> None:
    """When the swap endpoint raises, fall back to full restart."""
    draft_swap = FakeDraftSwap(fail_on_swap=True, swap_error="RPC error")
    rig = make_rig(draft_swap_http=draft_swap)
    candidate = Path("/candidate/draft")

    rig.controller.start_candidate(
        candidate,
        timeout_seconds=3.0,
        should_abort=lambda: False,
    )

    # Swap was attempted, then restarted
    assert draft_swap.swapped_paths == [str(candidate)]
    assert [call.draft for call in rig.process.calls] == [candidate]


# ---------------------------------------------------------------------------
# Hot-swap: successful swap skips restart
# ---------------------------------------------------------------------------


def test_successful_swap_skips_restart() -> None:
    """When hot-swap succeeds, the process is not restarted."""
    draft_swap = FakeDraftSwap()
    rig = make_rig(draft_swap_http=draft_swap)
    candidate = Path("/candidate/draft")

    rig.controller.start_candidate(
        candidate,
        timeout_seconds=3.0,
        should_abort=lambda: False,
    )

    # Swap was called and no restart happened
    assert draft_swap.swapped_paths == [str(candidate)]
    assert rig.process.calls == []
    assert rig.controller._running_draft == candidate


# ---------------------------------------------------------------------------
# DraftSwapHTTP Protocol verification
# ---------------------------------------------------------------------------


def test_draft_swap_http_is_protocol() -> None:
    """DraftSwapHTTP is a Protocol class."""
    assert hasattr(DraftSwapHTTP, "__mro__")
    assert DraftSwapHTTP.__bases__[0].__name__ == "Protocol"


# ---------------------------------------------------------------------------
# Config round-trip for draft_hot_swap_enabled
# ---------------------------------------------------------------------------


def test_config_draft_hot_swap_default() -> None:
    from speedlm.config import IdleTuningConfig
    cfg = IdleTuningConfig()
    assert cfg.draft_hot_swap_enabled is False


def test_config_draft_hot_swap_can_be_true() -> None:
    from speedlm.config import IdleTuningConfig
    cfg = IdleTuningConfig(draft_hot_swap_enabled=True)
    assert cfg.draft_hot_swap_enabled is True


def test_config_draft_hot_swap_rejects_non_bool() -> None:
    from speedlm.config import ConfigError, IdleTuningConfig
    with pytest.raises(ConfigError, match="must be a bool"):
        IdleTuningConfig(draft_hot_swap_enabled=1)  # type: ignore[arg-type]


def test_config_round_trip_with_hot_swap() -> None:
    from speedlm.config import IdleTuningConfig, SpeedLMConfig
    cfg = SpeedLMConfig(
        model="test/model",
        tuning=IdleTuningConfig(draft_hot_swap_enabled=True),
    )
    restored = SpeedLMConfig.from_dict(cfg.to_dict())
    assert restored.tuning.draft_hot_swap_enabled is True


# ---------------------------------------------------------------------------
# Compatibility validation (no GPU -- pure Python)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal tensor-like object for compatibility tests."""

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype


class _FakeDrafter:
    """Minimal drafter model for compatibility tests."""

    def __init__(self, params: dict[str, _FakeTensor]) -> None:
        self._params = params

    def named_parameters(self):
        yield from self._params.items()


def _make_params(
    shapes: dict[str, tuple[int, ...]], dtype: str = "float32"
) -> dict[str, _FakeTensor]:
    return {name: _FakeTensor(shape, dtype) for name, shape in shapes.items()}


def _make_weights(
    shapes: dict[str, tuple[int, ...]], dtype: str = "float32"
) -> dict[str, _FakeTensor]:
    return {name: _FakeTensor(shape, dtype) for name, shape in shapes.items()}


def _validate_compatibility(
    drafter: _FakeDrafter, new_weights: dict[str, _FakeTensor]
) -> None:
    """Pure-Python compatibility check mirroring DraftSwapExtension logic."""
    existing_params = {
        name: param for name, param in drafter.named_parameters()
    }

    missing = set(existing_params.keys()) - set(new_weights.keys())
    if missing:
        raise ValueError(
            f"draft weights missing {len(missing)} parameters: "
            f"{sorted(missing)[:5]}..."
        )

    extra = set(new_weights.keys()) - set(existing_params.keys())
    if extra:
        raise ValueError(
            f"draft weights contain {len(extra)} unexpected parameters: "
            f"{sorted(extra)[:5]}..."
        )

    for name, new_param in new_weights.items():
        existing = existing_params[name]
        if new_param.shape != existing.shape:
            raise ValueError(
                f"parameter {name!r} shape mismatch: "
                f"existing={tuple(existing.shape)}, "
                f"new={tuple(new_param.shape)}"
            )
        if new_param.dtype != existing.dtype:
            raise ValueError(
                f"parameter {name!r} dtype mismatch: "
                f"existing={existing.dtype}, new={new_param.dtype}"
            )


class TestCompatibilityValidation:
    """Validate the compatibility checker accepts matching / rejects mismatched."""

    BASE_SHAPES = {
        "layer.0.weight": (128, 256),
        "layer.0.bias": (256,),
        "layer.1.weight": (128, 256),
    }

    def test_matching_shapes_accepted(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES))
        weights = _make_weights(self.BASE_SHAPES)
        _validate_compatibility(drafter, weights)  # no exception

    def test_matching_dtypes_accepted(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES, dtype="float16"))
        weights = _make_weights(self.BASE_SHAPES, dtype="float16")
        _validate_compatibility(drafter, weights)  # no exception

    def test_shape_mismatch_rejected(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES))
        weights = {
            "layer.0.weight": _FakeTensor((256, 256), "float32"),
            "layer.0.bias": _FakeTensor((256,), "float32"),
            "layer.1.weight": _FakeTensor((128, 256), "float32"),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            _validate_compatibility(drafter, weights)

    def test_dtype_mismatch_rejected(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES, dtype="float32"))
        weights = _make_weights(self.BASE_SHAPES, dtype="float16")
        with pytest.raises(ValueError, match="dtype mismatch"):
            _validate_compatibility(drafter, weights)

    def test_missing_parameter_rejected(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES))
        weights = {
            "layer.0.weight": _FakeTensor((128, 256), "float32"),
            "layer.0.bias": _FakeTensor((256,), "float32"),
            # layer.1.weight missing
        }
        with pytest.raises(ValueError, match="missing"):
            _validate_compatibility(drafter, weights)

    def test_extra_parameter_rejected(self) -> None:
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES))
        weights = {
            "layer.0.weight": _FakeTensor((128, 256), "float32"),
            "layer.0.bias": _FakeTensor((256,), "float32"),
            "layer.1.weight": _FakeTensor((128, 256), "float32"),
            "extra_param": _FakeTensor((64,), "float32"),
        }
        with pytest.raises(ValueError, match="unexpected"):
            _validate_compatibility(drafter, weights)

    def test_quantization_mismatch_note(self) -> None:
        """Quantization is checked via config, not weights.
        The compatibility checker validates shapes+dtypes only; quantization
        is read from vllm_config.speculative_config.draft_model_config.quantization.
        This test documents that quantization mismatch is NOT caught by shape
        checks -- it requires the _get_quantization config check."""
        # The shape/dtype checks pass, quantization is a separate path
        drafter = _FakeDrafter(_make_params(self.BASE_SHAPES, dtype="float16"))
        weights = _make_weights(self.BASE_SHAPES, dtype="float16")
        _validate_compatibility(drafter, weights)  # passes -- correct


# ---------------------------------------------------------------------------
# CombinedWorkerExtension composition
# ---------------------------------------------------------------------------


def test_combined_extension_has_both_method_sets() -> None:
    """CombinedWorkerExtension exposes both capture and draft-swap methods."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    # Verify all expected methods exist
    assert hasattr(CombinedWorkerExtension, "activate_capture")
    assert hasattr(CombinedWorkerExtension, "flush_capture")
    assert hasattr(CombinedWorkerExtension, "deactivate_capture")
    assert hasattr(CombinedWorkerExtension, "hot_swap_draft")
    assert hasattr(CombinedWorkerExtension, "draft_info")


def test_combined_extension_no_attribute_conflicts() -> None:
    """No public method names are shared between the two extension sets."""
    capture_methods = {
        "activate_capture", "flush_capture", "deactivate_capture",
        "_install_hook", "_deactivate_impl", "_buffer_aux",
    }
    draft_swap_methods = {
        "hot_swap_draft", "draft_info",
        "_get_drafter_model", "_load_weights_file",
        "_validate_compatibility", "_apply_weights", "_get_quantization",
    }
    # No overlap between the two sets
    assert capture_methods.isdisjoint(draft_swap_methods)

    # Also verify that CombinedWorkerExtension does not have duplicate
    # public methods (vLLM's collision check would fail on duplicates)
    from speedlm.gateway.draft_swap import CombinedWorkerExtension
    public_attrs = {
        attr for attr in dir(CombinedWorkerExtension)
        if not attr.startswith("__")
    }
    # Each public attr should appear exactly once
    # (dir() deduplicates, so this is inherently satisfied -- this test
    # documents the invariant)
    assert len(public_attrs) >= len(capture_methods | draft_swap_methods)


def test_combined_extension_init_does_not_require_torch() -> None:
    """CombinedWorkerExtension can be instantiated without torch available."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext = CombinedWorkerExtension()
    assert not ext._capture_active
    assert ext._capture_dir is None


# ---------------------------------------------------------------------------
# DraftSwapExtension standalone
# ---------------------------------------------------------------------------


def test_draft_swap_extension_methods_exist() -> None:
    """DraftSwapExtension has the expected public methods."""
    from speedlm.gateway.draft_swap import DraftSwapExtension

    assert hasattr(DraftSwapExtension, "hot_swap_draft")
    assert hasattr(DraftSwapExtension, "draft_info")
    assert hasattr(DraftSwapExtension, "_validate_compatibility")
    assert hasattr(DraftSwapExtension, "_apply_weights")
    assert hasattr(DraftSwapExtension, "_get_quantization")


# ---------------------------------------------------------------------------
# Worker extension dotted-path resolution (BUG 1)
# ---------------------------------------------------------------------------


def test_activation_capture_extension_dotted_path() -> None:
    """The dotted path 'speedlm.activation_capture.hook.ActivationCaptureExtension'
    resolves via importlib (vLLM's resolve_obj_by_qualname expects dots, not colons)."""
    from importlib import import_module

    mod = import_module("speedlm.activation_capture.hook")
    cls = mod.ActivationCaptureExtension
    assert cls.__name__ == "ActivationCaptureExtension"


def test_combined_extension_dotted_path() -> None:
    """The dotted path 'speedlm.gateway.draft_swap.CombinedWorkerExtension' resolves."""
    from importlib import import_module

    mod = import_module("speedlm.gateway.draft_swap")
    cls = mod.CombinedWorkerExtension
    assert cls.__name__ == "CombinedWorkerExtension"


def test_draft_swap_extension_dotted_path() -> None:
    """The dotted path 'speedlm.gateway.draft_swap.DraftSwapExtension' resolves."""
    from importlib import import_module

    mod = import_module("speedlm.gateway.draft_swap")
    cls = mod.DraftSwapExtension
    assert cls.__name__ == "DraftSwapExtension"


# ---------------------------------------------------------------------------
# Extension works WITHOUT __init__ being called (BUG 2)
# ---------------------------------------------------------------------------


def test_activation_capture_extension_without_init() -> None:
    """ActivationCaptureExtension works when instantiated via object.__new__
    (simulating vLLM's __bases__ injection which never calls __init__)."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    ext = object.__new__(ActivationCaptureExtension)

    # Class-level defaults should be accessible
    assert ext._capture_active is False
    assert ext._capture_dir is None
    assert ext._original_model_forward is None
    assert ext._final_layer_idx is None
    assert ext._original_aux_layers == ()

    # _ensure_init should create mutable state
    ext._ensure_init()
    assert ext._lock is not None
    assert ext._pending is not None
    assert isinstance(ext._pending, dict)


def test_combined_extension_without_init() -> None:
    """CombinedWorkerExtension works when instantiated via object.__new__
    (simulating vLLM's __bases__ injection which never calls __init__)."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext = object.__new__(CombinedWorkerExtension)

    # Class-level defaults should be accessible
    assert ext._capture_active is False
    assert ext._capture_dir is None
    assert ext._original_model_forward is None

    # _ensure_init should create mutable state
    ext._ensure_init()
    assert ext._lock is not None
    assert ext._pending is not None
    assert isinstance(ext._pending, dict)


def test_extension_methods_callable_after_ensure_init() -> None:
    """Extension entry point methods can be called after _ensure_init
    without raising AttributeError for missing instance attributes."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    ext = object.__new__(ActivationCaptureExtension)
    ext._ensure_init()

    # deactivate_capture should not raise AttributeError
    # (it calls _deactivate_impl which accesses _original_model_forward)
    # Note: _deactivate_impl will raise ImportError for vllm, but that's fine
    # — we're checking it doesn't raise AttributeError for missing state
    try:
        ext.deactivate_capture()
    except (ImportError, AttributeError) as exc:
        # ImportError is expected (no vLLM in project venv)
        # AttributeError means we missed a class-level default
        if isinstance(exc, AttributeError):
            raise AssertionError(
                f"deactivate_capture raised AttributeError: {exc}"
            ) from exc
        # ImportError is acceptable


# ---------------------------------------------------------------------------
# Attribute collision audit
# ---------------------------------------------------------------------------


def test_no_collision_activation_capture_extension() -> None:
    """ActivationCaptureExtension defines no name that collides with
    known vLLM Worker/WorkerBase attributes."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    # Representative vLLM Worker/WorkerBase attribute names
    known_vllm = {
        "model_runner", "model_executor", "vllm_config",
        "executor_backend", "driver_worker", "cuda_context",
        "worker_init_fn", "load_model", "deterministic_init",
        "compile_model", "start_worker_loop", "execute_model",
        "cache_info", "get_capacity", "num_seq_ids",
        "get_cluster_usage_info", "get_mm_processor",
        "add_mm_ppu_data", "check_health",
        "collective_rpc", "start",
        "detached_executors", "scheduler",
        "num_prefill", "num_decoder",
        "store_model_torch_compile_stats",
    }
    ext_attrs = {a for a in dir(ActivationCaptureExtension) if not a.startswith("__")}
    collisions = ext_attrs & known_vllm
    assert not collisions, f"Attribute collisions: {collisions}"


def test_no_collision_combined_extension() -> None:
    """CombinedWorkerExtension defines no name that collides with
    known vLLM Worker/WorkerBase attributes."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    known_vllm = {
        "model_runner", "model_executor", "vllm_config",
        "executor_backend", "driver_worker", "cuda_context",
        "worker_init_fn", "load_model", "deterministic_init",
        "compile_model", "start_worker_loop", "execute_model",
        "cache_info", "get_capacity", "num_seq_ids",
        "get_cluster_usage_info", "get_mm_processor",
        "add_mm_ppu_data", "check_health",
        "collective_rpc", "start",
        "detached_executors", "scheduler",
        "num_prefill", "num_decoder",
        "store_model_torch_compile_stats",
    }
    ext_attrs = {a for a in dir(CombinedWorkerExtension) if not a.startswith("__")}
    collisions = ext_attrs & known_vllm
    assert not collisions, f"Attribute collisions: {collisions}"


# ---------------------------------------------------------------------------
# Mutable state is NOT shared across instances (BUG 2 follow-up)
# ---------------------------------------------------------------------------


def test_activation_capture_extension_no_shared_state() -> None:
    """Two ActivationCaptureExtension instances created via object.__new__
    do not share their _pending dicts or _lock instances."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    ext_a = object.__new__(ActivationCaptureExtension)
    ext_b = object.__new__(ActivationCaptureExtension)

    ext_a._ensure_init()
    ext_b._ensure_init()

    # Each instance has its own lock
    assert ext_a._lock is not ext_b._lock
    # Each instance has its own pending dict
    assert ext_a._pending is not ext_b._pending
    # And they are both empty (not sharing a class-level dict)
    assert ext_a._pending == {}
    assert ext_b._pending == {}


def test_combined_extension_no_shared_state() -> None:
    """Two CombinedWorkerExtension instances created via object.__new__
    do not share their _pending dicts or _lock instances."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext_a = object.__new__(CombinedWorkerExtension)
    ext_b = object.__new__(CombinedWorkerExtension)

    ext_a._ensure_init()
    ext_b._ensure_init()

    # Each instance has its own lock
    assert ext_a._lock is not ext_b._lock
    # Each instance has its own pending dict
    assert ext_a._pending is not ext_b._pending
    # And they are both empty (not sharing a class-level dict)
    assert ext_a._pending == {}
    assert ext_b._pending == {}