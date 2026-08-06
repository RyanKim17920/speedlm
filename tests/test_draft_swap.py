from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.control import (
    ControlAborted,
    DraftSwapHTTP,
    RuntimeController,
)


@pytest.fixture(autouse=True)
def _isolated_capture_session():
    """Capture state is process-wide, so tests must clear it themselves.

    See ``test_activation_capture_state_is_one_session_per_process`` for why it
    is process-wide.  Unreset process-wide state bleeds, and buffered rows
    bleeding between tests is exactly the mislabelling these tests guard.
    """
    from speedlm.activation_capture.hook import reset_session

    reset_session()
    yield
    reset_session()


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
    canary_timeouts: list[float] = field(default_factory=list)
    fail_canary: bool = False

    def canary(self, *, timeout_seconds: float) -> None:
        self.canary_timeouts.append(timeout_seconds)
        if self.fail_canary:
            raise RuntimeError("canary failed")

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
# Worker-side fakes: a torch-free stand-in for an nn.Module tree
# ---------------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, kind: str) -> None:
        self.type = kind

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"device({self.type})"


class _FakeTensor:
    """Minimal tensor-like object: shape, dtype, and a device we can flip."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str = "float32",
        device: str = "cuda",
        values: list[int] | None = None,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self.device = _FakeDevice(device)
        self._values = values

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    def tolist(self) -> list[int]:
        """Only index maps carry values; anything else has none to give.

        Deliberately raises rather than synthesizing zeros: an all-zeros
        ``d2t`` is a *corruption* the validator must reject, so a fake that
        invented one would make every vocab-mapping test a test of the fake.
        """
        if self._values is None:
            raise TypeError(f"_FakeTensor{self._shape} carries no values")
        return list(self._values)


#: Draft/verifier vocabulary sizes the fakes in this module agree on.
#: ``_FakeHFConfig`` declares them and ``DRAFT_CONFIG`` repeats them, so the
#: index maps built below are the ones a real candidate for this fake server
#: would carry.
_FAKE_DRAFT_VOCAB = 32_000
_FAKE_VERIFIER_VOCAB = 151_936


def _valid_d2t(
    draft_vocab: int = _FAKE_DRAFT_VOCAB, stride: int = 4
) -> list[int]:
    """A well-formed ``d2t``: draft id ``i`` maps to target ``i * stride``.

    Stored as the *delta* ``i * stride - i``, which is the on-disk convention
    (``speculators/train/vocab_mapping.py``:
    ``draft_to_target = selected_ids - arange(draft_vocab_size)``).  Strided
    rather than identity so the map is a genuine reduction with a non-zero
    delta -- an identity map is a state the validator must reject.
    """
    return [index * stride - index for index in range(draft_vocab)]


def _valid_t2d(
    draft_vocab: int = _FAKE_DRAFT_VOCAB,
    verifier_vocab: int = _FAKE_VERIFIER_VOCAB,
    stride: int = 4,
) -> list[int]:
    """The ``t2d`` mask matching :func:`_valid_d2t`: true at each target id."""
    mask = [0] * verifier_vocab
    for index in range(draft_vocab):
        mask[index * stride] = 1
    return mask


def _vocab_map_tensors(
    d2t: list[int] | None = None, t2d: list[int] | None = None
) -> dict[str, _FakeTensor]:
    """The two index-map entries of a candidate checkpoint."""
    d2t = _valid_d2t() if d2t is None else d2t
    t2d = _valid_t2d() if t2d is None else t2d
    return {
        "d2t": _FakeTensor((len(d2t),), dtype="int64", values=d2t),
        "t2d": _FakeTensor((len(t2d),), dtype="bool", values=t2d),
    }


class _FakeModule:
    """Stand-in for ``torch.nn.Module`` supporting the traversal APIs used.

    Child modules are held in a dict but exposed as attributes so that the
    ``delattr``/``setattr`` detach dance in ``_detached_submodules`` works
    exactly as it would on a real module.
    """

    def __init__(self, **params: _FakeTensor) -> None:
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_params", dict(params))
        object.__setattr__(self, "_buffers", {})

    # -- attribute protocol --

    def __getattr__(self, name: str):
        children = object.__getattribute__(self, "_children")
        if name in children:
            return children[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, _FakeModule):
            self._children[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._children:
            del self._children[name]
            return
        object.__delattr__(self, name)

    # -- module protocol --

    def add_buffer(self, name: str, tensor: _FakeTensor) -> None:
        self._buffers[name] = tensor

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def _named_tensors(self, attr: str, prefix: str = ""):
        for name, tensor in getattr(self, attr).items():
            yield (f"{prefix}.{name}" if prefix else name), tensor
        for name, child in self._children.items():
            yield from child._named_tensors(attr, f"{prefix}.{name}" if prefix else name)

    def named_parameters(self):
        yield from self._named_tensors("_params")

    def named_buffers(self):
        yield from self._named_tensors("_buffers")

    def get_submodule(self, path: str) -> _FakeModule:
        module = self
        for part in path.split("."):
            module = module._children[part]
        return module


class _FakeDrafter(_FakeModule):
    """A drafter whose ``load_weights`` materializes the names it is given."""

    def __init__(self, **params: _FakeTensor) -> None:
        super().__init__(**params)
        object.__setattr__(self, "load_calls", [])
        object.__setattr__(self, "materialize", True)

    def load_weights(self, weights):
        names = [name for name, _ in weights]
        self.load_calls.append(names)
        if self.materialize:
            for _, tensor in self.named_parameters():
                tensor.device = _FakeDevice("cuda")
            for _, tensor in self.named_buffers():
                tensor.device = _FakeDevice("cuda")
        return set(names)


@dataclass
class _FakeDraftModelConfig:
    hf_config: object
    quantization: str | None = None


@dataclass
class _FakeSpeculativeConfig:
    draft_model_config: _FakeDraftModelConfig


@dataclass
class _FakeVllmConfig:
    speculative_config: _FakeSpeculativeConfig
    model_config: object = None


@dataclass
class _FakeHFConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 1
    draft_vocab_size: int | None = 32000
    dtype: str = "bfloat16"


#: The two runner topologies vLLM actually ships.  ``gpu_worker.py:384-398``
#: picks between them on ``vllm_config.use_v2_model_runner``, and the runner
#: attribute holding the proposer is renamed across the split:
#: V1 ``GPUModelRunnerV1.drafter`` (``v1/worker/gpu_model_runner.py:571-634``)
#: vs V2 ``GPUModelRunnerV2.speculator`` (``v1/worker/gpu/model_runner.py:194``).
#: Fakes must model *one* of these at a time -- a fake carrying both would let
#: a lookup that only knows one name pass, which is exactly how the first GPU
#: run shipped broken.
RUNNER_TOPOLOGIES: tuple[str, ...] = ("drafter", "speculator")


#: The drafter/verifier *weight-sharing* topologies vLLM can produce, keyed by
#: which drafter submodule paths end up bound to the verifier's own object.
#:
#: Sharing is decided per checkpoint, not per architecture.  Both runner
#: generations gate it on "does the checkpoint ship its own copy, and is that
#: copy weight-identical to the verifier's" -- V2 in
#: ``v1/worker/gpu/spec_decode/eagle/utils.py:12-25`` (``_should_share``,
#: applied at ``:60-65`` and ``:69-74``), V1 in
#: ``llm_base_proposer.py:1432-1461`` and ``:1508-1547``.  Both rebind under
#: the same paths, so the runner generation does not change this axis.
#:
#: ``neither`` is the shipped case: ``RedHatAI/Qwen3-8B-speculator.eagle3``
#: carries an ``embed_tokens`` that differs from Qwen3-8B's and a
#: ``(32000, 4096)`` head against the verifier's ``(151936, 4096)``, so vLLM
#: declines both and the drafter owns all 16 of its checkpoint tensors.
#: Fakes must model each of these separately -- a fake that only ever shares
#: makes a name-based lookup look correct, which is how the hard-coded
#: expectation shipped.
SHARING_TOPOLOGIES: dict[str, tuple[str, ...]] = {
    "both": ("lm_head", "model.embed_tokens"),
    "embed_only": ("model.embed_tokens",),
    "head_only": ("lm_head",),
    "neither": (),
}


class _FakeRunner:
    """A model runner exposing the proposer under exactly one attribute name.

    Accessing the *other* topology's attribute raises ``AttributeError``, the
    same as the real runner classes, so a lookup hard-coded to one name cannot
    silently satisfy itself against the wrong runner generation.
    """

    def __init__(
        self,
        model: object = None,
        proposer: object = None,
        *,
        runner_attr: str = "speculator",
    ) -> None:
        if runner_attr not in RUNNER_TOPOLOGIES:
            raise ValueError(f"unknown runner topology {runner_attr!r}")
        self.model = model
        self._runner_attr = runner_attr
        if proposer is not None:
            setattr(self, runner_attr, proposer)


class _FakeProposer:
    def __init__(self, model: object) -> None:
        self.model = model


DRAFT_CONFIG = {
    "speculators_model_type": "eagle3",
    "draft_vocab_size": 32000,
    "dtype": "bfloat16",
    "transformer_layer_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 1,
        "vocab_size": 151936,
        "dtype": "bfloat16",
    },
}


def _write_draft_dir(
    root: Path,
    *,
    shards: Sequence[str] = ("model.safetensors",),
    config: Mapping[str, object] | None = None,
    index: Mapping[str, object] | None = None,
) -> Path:
    """Materialize a candidate draft directory on disk."""
    directory = root / "candidate"
    directory.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for shard in shards:
        (directory / shard).write_bytes(b"")
    if index is not None:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
    return directory


def _install_fake_vllm_reload(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Install a fake ``vllm...reload.layerwise`` that mimics the real one.

    ``initialize_layerwise_reload`` strips every parameter of every module in
    ``model.modules()`` onto the meta device (that is what
    ``restore_layer_on_meta`` does); ``finalize_layerwise_reload`` is a no-op.
    This is precisely the mechanism that corrupts the verifier when the walk
    is not scoped, so the fake makes it observable without a GPU.
    """
    calls: dict[str, list] = {"init": [], "finalize": []}

    def initialize_layerwise_reload(model) -> None:
        calls["init"].append(model)
        for module in model.modules():
            for tensor in list(module._params.values()) + list(module._buffers.values()):
                tensor.device = _FakeDevice("meta")

    def finalize_layerwise_reload(model, model_config) -> None:
        calls["finalize"].append((model, model_config))

    layerwise = types.ModuleType("vllm.model_executor.model_loader.reload.layerwise")
    layerwise.initialize_layerwise_reload = initialize_layerwise_reload
    layerwise.finalize_layerwise_reload = finalize_layerwise_reload

    chain = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
    ]
    parent = None
    for dotted in chain:
        module = sys.modules.get(dotted) or types.ModuleType(dotted)
        monkeypatch.setitem(sys.modules, dotted, module)
        if parent is not None:
            monkeypatch.setattr(parent, dotted.rsplit(".", 1)[1], module, raising=False)
        parent = module
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader.reload.layerwise", layerwise
    )
    monkeypatch.setattr(parent, "layerwise", layerwise, raising=False)
    return calls


def _install_fake_safetensors(
    monkeypatch: pytest.MonkeyPatch, contents: Mapping[str, Mapping[str, _FakeTensor]]
) -> None:
    """Install a ``safetensors.safe_open`` backed by an in-memory registry.

    The real handle is a context manager that is NOT iterable -- ``.keys()``
    is the only enumeration entry point -- so the fake reproduces that shape
    to keep the production code honest about it.
    """

    class _Handle:
        def __init__(self, tensors: Mapping[str, _FakeTensor]) -> None:
            self._tensors = tensors

        def __enter__(self) -> _Handle:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def __iter__(self):
            raise TypeError("safe_open handles are not iterable")

        def keys(self):
            return list(self._tensors)

        def get_tensor(self, name: str) -> _FakeTensor:
            return self._tensors[name]

    def safe_open(path: str, framework: str = "pt", **_: object) -> _Handle:
        return _Handle(contents[Path(path).name])

    module = types.ModuleType("safetensors")
    module.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", module)


def _make_extension(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drafter: _FakeDrafter,
    target: _FakeModule | None = None,
    hf_config: object | None = None,
    runner_attr: str = "speculator",
):
    """Build a CombinedWorkerExtension wired to fake runner/config objects.

    Defaults to the **V2** (``speculator``) topology because that is what a
    current vLLM build actually selects; ``TestDrafterLookup`` pins both.
    """
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext = object.__new__(CombinedWorkerExtension)
    runner = _FakeRunner(
        model=target, proposer=_FakeProposer(drafter), runner_attr=runner_attr
    )
    object.__setattr__(ext, "model_runner", runner)
    object.__setattr__(
        ext,
        "vllm_config",
        _FakeVllmConfig(
            speculative_config=_FakeSpeculativeConfig(
                draft_model_config=_FakeDraftModelConfig(
                    hf_config=hf_config if hf_config is not None else _FakeHFConfig()
                )
            )
        ),
    )
    return ext


# ---------------------------------------------------------------------------
# TASK B -- directory -> safetensors shard resolution
# ---------------------------------------------------------------------------


class TestShardResolution:
    def test_single_shard_directory(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        assert [p.name for p in resolve_safetensors_shards(directory)] == [
            "model.safetensors"
        ]

    def test_multiple_shards_resolve_in_natural_order(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        directory = _write_draft_dir(
            tmp_path,
            shards=(
                "model-00010-of-00010.safetensors",
                "model-00002-of-00010.safetensors",
                "model-00001-of-00010.safetensors",
            ),
            config=DRAFT_CONFIG,
        )
        assert [p.name for p in resolve_safetensors_shards(directory)] == [
            "model-00001-of-00010.safetensors",
            "model-00002-of-00010.safetensors",
            "model-00010-of-00010.safetensors",
        ]

    def test_index_filters_consolidated_duplicate(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        directory = _write_draft_dir(
            tmp_path,
            shards=(
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "consolidated.safetensors",
            ),
            config=DRAFT_CONFIG,
            index={
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            },
        )
        assert [p.name for p in resolve_safetensors_shards(directory)] == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_directory_without_safetensors_fails_clearly(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        directory = _write_draft_dir(tmp_path, shards=(), config=DRAFT_CONFIG)
        (directory / "pytorch_model.bin").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="no \\*.safetensors weight shards"):
            resolve_safetensors_shards(directory)

    def test_missing_directory_fails_clearly(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_safetensors_shards(tmp_path / "nope")

    def test_bare_safetensors_file_is_accepted(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import resolve_safetensors_shards

        shard = tmp_path / "model.safetensors"
        shard.write_bytes(b"")
        assert resolve_safetensors_shards(shard) == [shard]

    def test_merges_tensors_across_shards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(
            tmp_path,
            shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            config=DRAFT_CONFIG,
        )
        _install_fake_safetensors(
            monkeypatch,
            {
                "model-00001-of-00002.safetensors": {"a": _FakeTensor((2,))},
                "model-00002-of-00002.safetensors": {"b": _FakeTensor((2,))},
            },
        )
        ext = _make_extension(monkeypatch, drafter=_FakeDrafter())
        assert sorted(ext._load_weights_file(str(directory))) == ["a", "b"]

    def test_duplicate_tensor_across_shards_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(
            tmp_path,
            shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            config=DRAFT_CONFIG,
        )
        _install_fake_safetensors(
            monkeypatch,
            {
                "model-00001-of-00002.safetensors": {"a": _FakeTensor((2,))},
                "model-00002-of-00002.safetensors": {"a": _FakeTensor((2,))},
            },
        )
        ext = _make_extension(monkeypatch, drafter=_FakeDrafter())
        with pytest.raises(ValueError, match="more than one shard"):
            ext._load_weights_file(str(directory))


# ---------------------------------------------------------------------------
# TASK C -- fusion-safe compatibility validation
# ---------------------------------------------------------------------------


class TestCompatibilityValidation:
    """The checker must read config.json, not diff fused tensor names."""

    def _drafter(self, *, with_mapping: bool = True) -> _FakeDrafter:
        #: Names deliberately mirror the FUSED vLLM layout (qkv_proj /
        #: gate_up_proj) while the checkpoint below uses the split HF layout.
        #: The old name set-diff reported both "missing" and "extra" here.
        drafter = _FakeDrafter()
        inner = _FakeModule()
        layer = _FakeModule(
            **{
                "self_attn.qkv_proj.weight": _FakeTensor((12288, 4096)),
                "mlp.gate_up_proj.weight": _FakeTensor((24576, 4096)),
            }
        )
        inner.layers = layer
        drafter.model = inner
        if with_mapping:
            drafter.add_buffer("draft_id_to_target_id", _FakeTensor((32000,)))
        return drafter

    HF_CHECKPOINT = {
        "midlayer.self_attn.q_proj.weight": _FakeTensor((4096, 4096)),
        "midlayer.self_attn.k_proj.weight": _FakeTensor((1024, 4096)),
        "midlayer.self_attn.v_proj.weight": _FakeTensor((1024, 4096)),
        "midlayer.mlp.gate_proj.weight": _FakeTensor((12288, 4096)),
        "midlayer.mlp.up_proj.weight": _FakeTensor((12288, 4096)),
        **_vocab_map_tensors(),
    }

    def test_split_vs_fused_names_are_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A byte-identical drafter with split HF names must NOT be rejected."""
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_vocab_size_mismatch_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["vocab_size"] = 32000
        directory = _write_draft_dir(tmp_path, config=config)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        with pytest.raises(ValueError, match="draft vocab_size mismatch"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_hidden_size_mismatch_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["hidden_size"] = 2048
        directory = _write_draft_dir(tmp_path, config=config)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        with pytest.raises(ValueError, match="draft hidden_size mismatch"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_layer_count_mismatch_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["num_hidden_layers"] = 2
        directory = _write_draft_dir(tmp_path, config=config)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        with pytest.raises(ValueError, match="draft num_hidden_layers mismatch"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_dtype_mismatch_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["dtype"] = "float16"
        config["dtype"] = "float16"
        directory = _write_draft_dir(tmp_path, config=config)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        with pytest.raises(ValueError, match="draft dtype mismatch"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_missing_vocab_mapping_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter(with_mapping=True)
        ext = _make_extension(monkeypatch, drafter=drafter)
        weights = {
            k: v for k, v in self.HF_CHECKPOINT.items() if k not in {"d2t", "t2d"}
        }
        with pytest.raises(ValueError, match="missing vocab mappings"):
            ext._validate_compatibility(drafter, weights, directory)

    def test_vocab_mapping_not_required_without_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter(with_mapping=False)
        ext = _make_extension(monkeypatch, drafter=drafter)
        weights = {
            k: v for k, v in self.HF_CHECKPOINT.items() if k not in {"d2t", "t2d"}
        }
        ext._validate_compatibility(drafter, weights, directory)

    # -- vocab mapping *contents*, not merely presence ----------------------
    #
    # The check above establishes that d2t/t2d exist.  That was the only check
    # this repo had, and it is satisfied by a d2t holding anything at all --
    # including all zeros, which is the identity map vLLM silently falls back
    # to when d2t is missing entirely (llama_eagle3.py:304-307, :419-420).
    # vLLM applies the map as base + d2t with no clamp and no assert
    # (:347-357), where a negative target wraps under PyTorch advanced
    # indexing rather than raising, so nothing downstream catches either case.

    def _swap_with(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        d2t: list[int] | None = None,
        t2d: list[int] | None = None,
        hf_config=None,
        model_config: object = None,
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter(with_mapping=True)
        ext = _make_extension(monkeypatch, drafter=drafter, hf_config=hf_config)
        if model_config is not None:
            object.__setattr__(ext.vllm_config, "model_config", model_config)
        weights = dict(self.HF_CHECKPOINT) | _vocab_map_tensors(d2t=d2t, t2d=t2d)
        ext._validate_compatibility(drafter, weights, directory)

    def test_negative_d2t_target_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The silent case: a negative index wraps instead of raising."""
        d2t = _valid_d2t()
        d2t[7] = -100  # draft id 7 -> target -93
        #: Both halves are load-bearing.  "map outside" pins *which* check
        #: fired: a negative target also trips the t2d cross-check further
        #: down (a negative index reads t2d from the end), so matching only on
        #: the offending entry would let a disabled bound check still pass.
        with pytest.raises(ValueError, match=r"map outside.*d2t\[7\]=-100 -> -93"):
            self._swap_with(tmp_path, monkeypatch, d2t=d2t)

    def test_oversized_d2t_target_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d2t = _valid_d2t()
        d2t[9] = _FAKE_VERIFIER_VOCAB
        with pytest.raises(ValueError, match="map outside"):
            self._swap_with(tmp_path, monkeypatch, d2t=d2t)

    def test_all_zeros_d2t_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zeroing every d2t entry left the whole suite green before this."""
        with pytest.raises(ValueError, match="identity map"):
            self._swap_with(
                tmp_path,
                monkeypatch,
                d2t=[0] * _FAKE_DRAFT_VOCAB,
                t2d=[1] * _FAKE_DRAFT_VOCAB + [0] * (_FAKE_VERIFIER_VOCAB - _FAKE_DRAFT_VOCAB),
            )

    def test_t2d_disagreeing_with_d2t_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Popcount-preserving disagreement: one of the two maps is stale."""
        t2d = _valid_t2d()
        t2d[0] = 0
        t2d[1] = 1
        with pytest.raises(ValueError, match="not marked draftable"):
            self._swap_with(tmp_path, monkeypatch, t2d=t2d)

    def test_a_valid_mapping_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative control: the guard must not reject a good candidate."""
        self._swap_with(tmp_path, monkeypatch)

    def test_the_verifier_vocab_comes_from_the_running_verifiers_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``model_config.hf_config`` wins over the drafter's copy of the fact.

        The drafter's ``vocab_size`` is what it believed about the verifier
        when it was *built*.  The bound must be the verifier actually running,
        or the check validates the map against itself.  Here the drafter still
        claims 151,936 while the live verifier declares 40,000, and a d2t
        target of 45,000 must be rejected on the smaller, real bound.
        """
        #: 45,001 rather than a round number: it is out of range against the
        #: live 40,000 verifier, but *in* range against the drafter's stale
        #: 151,936 -- and it is deliberately not a multiple of the fixture's
        #: stride, so falling back to the stale bound produces a different
        #: message (a t2d disagreement) rather than the same one by accident.
        #: Matching on "map outside" is what makes this test able to fail.
        d2t = _valid_d2t()
        d2t[10] = 45_001 - 10
        with pytest.raises(ValueError, match=r"map outside.*-> 45001"):
            self._swap_with(
                tmp_path,
                monkeypatch,
                d2t=d2t,
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(vocab_size=40_000)
                ),
            )

    def test_a_multimodal_verifier_config_is_unwrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``text_config.vocab_size`` is the text vocabulary d2t indexes."""
        d2t = _valid_d2t()
        d2t[10] = 45_001 - 10
        with pytest.raises(ValueError, match=r"map outside.*-> 45001"):
            self._swap_with(
                tmp_path,
                monkeypatch,
                d2t=d2t,
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(
                        vocab_size=999_999,
                        text_config=types.SimpleNamespace(vocab_size=40_000),
                    )
                ),
            )

    def test_an_undecodable_d2t_is_rejected_not_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter(with_mapping=True)
        ext = _make_extension(monkeypatch, drafter=drafter)
        weights = dict(self.HF_CHECKPOINT) | {"d2t": object()}
        with pytest.raises(ValueError, match="not a readable tensor"):
            ext._validate_compatibility(drafter, weights, directory)

    def test_missing_config_json_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=None)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter)
        with pytest.raises(FileNotFoundError, match="no config.json"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_unreadable_running_config_refuses_swap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        drafter = self._drafter()
        ext = _make_extension(monkeypatch, drafter=drafter, hf_config=None)
        object.__setattr__(ext, "vllm_config", None)
        with pytest.raises(RuntimeError, match="refusing to hot-swap"):
            ext._validate_compatibility(drafter, self.HF_CHECKPOINT, directory)

    def test_speculators_config_flattening_matches_vllm(self, tmp_path: Path) -> None:
        """transformer_layer_config wins over top-level, as vLLM flattens it."""
        from speedlm.gateway.draft_swap import DraftConfigSummary, read_draft_config

        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        assert read_draft_config(directory) == DraftConfigSummary(
            vocab_size=151936,
            hidden_size=4096,
            num_hidden_layers=1,
            draft_vocab_size=32000,
            dtype="bfloat16",
        )

    def test_torch_dtype_fallback(self, tmp_path: Path) -> None:
        from speedlm.gateway.draft_swap import read_draft_config

        config = {"torch_dtype": "torch.float16", "transformer_layer_config": {}}
        directory = _write_draft_dir(tmp_path, config=config)
        assert read_draft_config(directory).dtype == "float16"


# ---------------------------------------------------------------------------
# TASK D -- scoped reload: the verifier must survive the swap
# ---------------------------------------------------------------------------


class TestScopedReload:
    def _rig(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        shared: Sequence[str] = SHARING_TOPOLOGIES["both"],
    ):
        """A drafter/verifier pair wired to the sharing topology *shared*.

        See :data:`SHARING_TOPOLOGIES`.  Paths named in *shared* are bound to
        the verifier's module object by identity; the rest get the drafter's
        own, distinct module, exactly as vLLM leaves them when the
        weight-equality gate declines to share.
        """
        target = _FakeModule()
        target_inner = _FakeModule()
        target_embed = _FakeModule(weight=_FakeTensor((151936, 4096)))
        target_head = _FakeModule(weight=_FakeTensor((151936, 4096)))
        target_inner.embed_tokens = target_embed
        target.model = target_inner
        target.lm_head = target_head

        drafter = _FakeDrafter()
        draft_inner = _FakeModule()
        draft_inner.layers = _FakeModule(weight=_FakeTensor((4096, 4096)))
        draft_inner.embed_tokens = (
            target_embed
            if "model.embed_tokens" in shared
            #: Distinct object, SAME attribute name -- an identity check
            #: separates these two, a name check cannot.
            else _FakeModule(weight=_FakeTensor((151936, 4096)))
        )
        drafter.model = draft_inner
        drafter.lm_head = (
            target_head
            if "lm_head" in shared
            #: Reduced draft vocabulary: the shape alone makes vLLM's
            #: ``torch.equal`` gate False, so the drafter keeps its own head.
            else _FakeModule(weight=_FakeTensor((32000, 4096)))
        )

        calls = _install_fake_vllm_reload(monkeypatch)
        ext = _make_extension(monkeypatch, drafter=drafter, target=target)
        return ext, drafter, target, target_embed, target_head, calls

    @pytest.mark.parametrize("shared", SHARING_TOPOLOGIES.values(), ids=list(SHARING_TOPOLOGIES))
    def test_target_owned_modules_track_the_live_topology(
        self, monkeypatch: pytest.MonkeyPatch, shared: tuple[str, ...]
    ) -> None:
        """Only the modules actually bound to the verifier count as owned.

        The regression this pins: a name-based lookup that always claims
        ``embed_tokens``/``lm_head`` is tautologically right against a fake
        that only ever shares them, and wrong on every checkpoint vLLM
        declines to share -- which is what the shipped drafter is.
        """
        ext, drafter, _target, _embed, _head, _calls = self._rig(
            monkeypatch, shared=shared
        )
        assert sorted(ext._target_owned_submodules(drafter)) == sorted(shared)

    def test_unshared_topology_loads_the_candidate_embed_and_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drafter owning its weights must receive them, not have them dropped.

        Dropping here would leave the drafter's own embedding and head at
        their pre-swap values while every other tensor advanced -- a silently
        half-swapped model that still answers.
        """
        ext, drafter, _target, _embed, _head, _calls = self._rig(
            monkeypatch, shared=SHARING_TOPOLOGIES["neither"]
        )
        weights = {
            "midlayer.weight": _FakeTensor((4096, 4096)),
            "embed_tokens.weight": _FakeTensor((151936, 4096)),
            "lm_head.weight": _FakeTensor((32000, 4096)),
        }

        count = ext._apply_weights(drafter, weights)

        assert drafter.load_calls == [list(weights)]
        assert count == 3

    def test_partially_shared_topology_drops_only_the_shared_leaf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shipped EAGLE-3 shape: embedding shared, reduced-vocab head not."""
        ext, drafter, _target, _embed, _head, _calls = self._rig(
            monkeypatch, shared=SHARING_TOPOLOGIES["embed_only"]
        )
        weights = {
            "midlayer.weight": _FakeTensor((4096, 4096)),
            "embed_tokens.weight": _FakeTensor((151936, 4096)),
            "lm_head.weight": _FakeTensor((32000, 4096)),
        }

        count = ext._apply_weights(drafter, weights)

        assert drafter.load_calls == [["midlayer.weight", "lm_head.weight"]]
        assert count == 2

    @pytest.mark.parametrize("shared", SHARING_TOPOLOGIES.values(), ids=list(SHARING_TOPOLOGIES))
    def test_verifier_never_left_on_meta_under_any_topology(
        self, monkeypatch: pytest.MonkeyPatch, shared: tuple[str, ...]
    ) -> None:
        """The topology-independent post-condition, checked on every topology."""
        ext, drafter, target, _embed, _head, _calls = self._rig(
            monkeypatch, shared=shared
        )
        ext._apply_weights(
            drafter,
            {
                "midlayer.weight": _FakeTensor((4096, 4096)),
                "embed_tokens.weight": _FakeTensor((151936, 4096)),
                "lm_head.weight": _FakeTensor((32000, 4096)),
            },
        )
        assert [n for n, t in target.named_parameters() if t.device.type == "meta"] == []

    def test_shared_modules_reattached_after_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, embed, head, _calls = self._rig(monkeypatch)
        ext._apply_weights(drafter, {"midlayer.weight": _FakeTensor((4096, 4096))})
        assert drafter.model.embed_tokens is embed
        assert drafter.lm_head is head

    def test_shared_modules_reattached_when_load_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, embed, head, _calls = self._rig(monkeypatch)

        def boom(weights):
            raise RuntimeError("loader exploded")

        object.__setattr__(drafter, "load_weights", boom)
        with pytest.raises(RuntimeError, match="loader exploded"):
            ext._apply_weights(drafter, {"midlayer.weight": _FakeTensor((4096, 4096))})
        assert drafter.model.embed_tokens is embed
        assert drafter.lm_head is head

    def test_target_bound_candidate_tensors_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, _embed, _head, _calls = self._rig(monkeypatch)
        weights = {
            "midlayer.weight": _FakeTensor((4096, 4096)),
            "embed_tokens.weight": _FakeTensor((151936, 4096)),
            "lm_head.weight": _FakeTensor((151936, 4096)),
        }
        ext._apply_weights(drafter, weights)
        assert drafter.load_calls == [["midlayer.weight"]]

    def test_meta_device_leftovers_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, _embed, _head, _calls = self._rig(monkeypatch)
        #: A loader that accepts the weights but materializes nothing is
        #: exactly the silent-corruption case the post-condition exists for.
        object.__setattr__(drafter, "materialize", False)
        with pytest.raises(RuntimeError, match="meta device"):
            ext._apply_weights(drafter, {"midlayer.weight": _FakeTensor((4096, 4096))})

    def test_finalize_receives_the_draft_model_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, _embed, _head, calls = self._rig(monkeypatch)
        ext._apply_weights(drafter, {"midlayer.weight": _FakeTensor((4096, 4096))})
        _model, model_config = calls["finalize"][0]
        assert model_config is ext.vllm_config.speculative_config.draft_model_config

    def test_zero_loaded_parameters_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, drafter, _target, _embed, _head, _calls = self._rig(monkeypatch)
        with pytest.raises(RuntimeError, match="loaded zero parameters"):
            ext._apply_weights(drafter, {})

    def test_load_weights_returning_none_still_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eagle3LlamaForCausalLM.load_weights returns None (llama_eagle3.py:381)."""
        ext, drafter, _target, _embed, _head, _calls = self._rig(monkeypatch)
        real = drafter.load_weights

        def returns_none(weights):
            real(weights)
            return None

        object.__setattr__(drafter, "load_weights", returns_none)
        count = ext._apply_weights(
            drafter,
            {"a.weight": _FakeTensor((4,)), "b.weight": _FakeTensor((4,))},
        )
        assert count == 2


# ---------------------------------------------------------------------------
# End-to-end null swap through the public RPC
# ---------------------------------------------------------------------------


class TestHotSwapDraftRPC:
    def _rig(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        config=None,
        runner_attr: str = "speculator",
    ):
        directory = _write_draft_dir(
            tmp_path, config=DRAFT_CONFIG if config is None else config
        )
        _install_fake_safetensors(
            monkeypatch,
            {
                "model.safetensors": {
                    "midlayer.self_attn.q_proj.weight": _FakeTensor((4096, 4096)),
                    "midlayer.self_attn.k_proj.weight": _FakeTensor((1024, 4096)),
                    "midlayer.mlp.gate_proj.weight": _FakeTensor((12288, 4096)),
                    **_vocab_map_tensors(),
                }
            },
        )
        calls = _install_fake_vllm_reload(monkeypatch)

        drafter = _FakeDrafter()
        inner = _FakeModule()
        inner.layers = _FakeModule(weight=_FakeTensor((4096, 4096)))
        drafter.model = inner
        drafter.add_buffer("draft_id_to_target_id", _FakeTensor((32000,)))
        ext = _make_extension(monkeypatch, drafter=drafter, runner_attr=runner_attr)
        return ext, drafter, directory, calls

    def test_null_swap_loads_every_parameter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swapping a directory identical to the running drafter loads all of it."""
        ext, drafter, directory, _calls = self._rig(tmp_path, monkeypatch)

        result = ext.hot_swap_draft(str(directory))

        assert result == {
            "swapped": True,
            "parameters_loaded": 5,
            #: This rig has no verifier wired in at all, so nothing can be
            #: shared and nothing may be dropped.
            "target_owned_submodules": [],
            "dropped_tensors": [],
        }
        assert drafter.load_calls == [
            [
                "midlayer.self_attn.q_proj.weight",
                "midlayer.self_attn.k_proj.weight",
                "midlayer.mlp.gate_proj.weight",
                "d2t",
                "t2d",
            ]
        ]
        assert all(t.device.type != "meta" for _, t in drafter.named_parameters())

    def test_return_value_is_json_serializable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, _drafter, directory, _calls = self._rig(tmp_path, monkeypatch)
        assert json.loads(json.dumps(ext.hot_swap_draft(str(directory)))) == {
            "swapped": True,
            "parameters_loaded": 5,
            "target_owned_submodules": [],
            "dropped_tensors": [],
        }

    def test_incompatible_vocab_rejected_before_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["vocab_size"] = 32000
        ext, drafter, directory, calls = self._rig(tmp_path, monkeypatch, config=config)

        with pytest.raises(ValueError, match="draft vocab_size mismatch"):
            ext.hot_swap_draft(str(directory))

        assert calls["init"] == []
        assert drafter.load_calls == []
        assert all(t.device.type != "meta" for _, t in drafter.named_parameters())

    def test_incompatible_hidden_size_rejected_before_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = json.loads(json.dumps(DRAFT_CONFIG))
        config["transformer_layer_config"]["hidden_size"] = 1024
        ext, drafter, directory, calls = self._rig(tmp_path, monkeypatch, config=config)

        with pytest.raises(ValueError, match="draft hidden_size mismatch"):
            ext.hot_swap_draft(str(directory))

        assert calls["init"] == []
        assert drafter.load_calls == []

    def test_no_drafter_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from speedlm.gateway.draft_swap import CombinedWorkerExtension

        ext = object.__new__(CombinedWorkerExtension)
        object.__setattr__(ext, "model_runner", _FakeRunner())
        with pytest.raises(RuntimeError, match="no drafter model found"):
            ext.hot_swap_draft("/nowhere")

    def test_v1_topology_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same swap succeeds against a V1 (``drafter``) runner."""
        ext, drafter, directory, _calls = self._rig(
            tmp_path, monkeypatch, runner_attr="drafter"
        )
        assert ext.hot_swap_draft(str(directory)) == {
            "swapped": True,
            "parameters_loaded": 5,
            "target_owned_submodules": [],
            "dropped_tensors": [],
        }
        assert drafter.load_calls != []

    def test_draft_info_is_json_serializable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ext, _drafter, _directory, _calls = self._rig(tmp_path, monkeypatch)
        info = ext.draft_info()
        assert json.loads(json.dumps(info))["draft_config"]["hidden_size"] == 4096
        assert info["num_parameters"] == 1

    def test_cudagraph_wrapper_is_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gpu_model_runner.py:5359 wraps drafter.model in a non-nn.Module."""
        from speedlm.gateway.draft_swap import CombinedWorkerExtension

        drafter = _FakeDrafter()

        class _Wrapper:
            def __init__(self, runnable: object) -> None:
                self.runnable = runnable

            def unwrap(self) -> object:
                return self.runnable

        ext = object.__new__(CombinedWorkerExtension)
        object.__setattr__(
            ext, "model_runner", _FakeRunner(proposer=_FakeProposer(_Wrapper(drafter)))
        )
        assert ext._get_drafter_model() is drafter


# ---------------------------------------------------------------------------
# Reported sharing topology -- what the E2E derives its expectation from
# ---------------------------------------------------------------------------


#: Tensor names of a candidate shaped like the shipped EAGLE-3 checkpoint:
#: fused-source projections, the vocab mapping pair, plus an embedding and a
#: head whose fate depends entirely on the sharing topology.
CANDIDATE_TENSORS: tuple[str, ...] = (
    "midlayer.self_attn.q_proj.weight",
    "embed_tokens.weight",
    "lm_head.weight",
    "d2t",
    "t2d",
)


class TestReportedTopology:
    """The swap must publish the topology it acted on, and act on what it published.

    The E2E cannot hard-code how many tensors a swap should load -- that
    number is a function of the live sharing topology.  It derives the
    expectation from ``draft_info``'s ``target_owned_submodules`` instead, so
    that report has to be (a) accurate and (b) the same one ``hot_swap_draft``
    applied.  These tests pin both halves without a GPU.
    """

    def _rig(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        shared: Sequence[str],
    ):
        directory = _write_draft_dir(tmp_path, config=DRAFT_CONFIG)
        _install_fake_safetensors(
            monkeypatch,
            {
                "model.safetensors": {
                    name: _FakeTensor((4096, 4096))
                    for name in CANDIDATE_TENSORS
                    if name not in _vocab_map_tensors()
                }
                | _vocab_map_tensors()
            },
        )
        _install_fake_vllm_reload(monkeypatch)

        target = _FakeModule()
        target_inner = _FakeModule()
        target_embed = _FakeModule(weight=_FakeTensor((151936, 4096)))
        target_head = _FakeModule(weight=_FakeTensor((151936, 4096)))
        target_inner.embed_tokens = target_embed
        target.model = target_inner
        target.lm_head = target_head

        drafter = _FakeDrafter()
        draft_inner = _FakeModule()
        draft_inner.layers = _FakeModule(weight=_FakeTensor((4096, 4096)))
        draft_inner.embed_tokens = (
            target_embed
            if "model.embed_tokens" in shared
            else _FakeModule(weight=_FakeTensor((151936, 4096)))
        )
        drafter.model = draft_inner
        drafter.lm_head = (
            target_head
            if "lm_head" in shared
            else _FakeModule(weight=_FakeTensor((32000, 4096)))
        )
        drafter.add_buffer("draft_id_to_target_id", _FakeTensor((32000,)))

        ext = _make_extension(monkeypatch, drafter=drafter, target=target)
        return ext, drafter, directory

    @pytest.mark.parametrize(
        "shared", SHARING_TOPOLOGIES.values(), ids=list(SHARING_TOPOLOGIES)
    )
    def test_draft_info_reports_the_live_topology(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: tuple[str, ...]
    ) -> None:
        ext, _drafter, _directory = self._rig(tmp_path, monkeypatch, shared=shared)
        info = ext.draft_info()
        assert info["target_owned_submodules"] == sorted(shared)
        #: Still JSON-serializable -- it crosses collective_rpc.
        assert json.loads(json.dumps(info))["target_owned_submodules"] == sorted(shared)

    @pytest.mark.parametrize(
        "shared", SHARING_TOPOLOGIES.values(), ids=list(SHARING_TOPOLOGIES)
    )
    def test_swap_report_reconciles_the_candidate_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: tuple[str, ...]
    ) -> None:
        """``candidates - dropped == parameters_loaded``, on every topology.

        This is precisely the arithmetic the E2E performs, so a topology the
        production code mishandles fails here first, without a GPU.
        """
        ext, _drafter, directory = self._rig(tmp_path, monkeypatch, shared=shared)
        info = ext.draft_info()

        swap = ext.hot_swap_draft(str(directory))

        leaves = {path.rpartition(".")[2] for path in info["target_owned_submodules"]}
        expected = [
            name
            for name in CANDIDATE_TENSORS
            if not any(leaf in name for leaf in leaves)
        ]
        assert swap["target_owned_submodules"] == info["target_owned_submodules"]
        assert sorted(swap["dropped_tensors"]) == sorted(
            set(CANDIDATE_TENSORS) - set(expected)
        )
        assert swap["parameters_loaded"] == len(expected)
        assert len(CANDIDATE_TENSORS) - len(swap["dropped_tensors"]) == (
            swap["parameters_loaded"]
        )

    def test_unshared_topology_loads_every_candidate_tensor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shipped case, stated as an absolute rather than a relation.

        ``RedHatAI/Qwen3-8B-speculator.eagle3`` shares nothing with Qwen3-8B,
        so all of its tensors -- embedding and head included -- are the
        drafter's own and must load.  Asserting a count of
        ``len(candidates) - 2`` here is the bug this test exists to catch.
        """
        ext, drafter, directory = self._rig(
            tmp_path, monkeypatch, shared=SHARING_TOPOLOGIES["neither"]
        )
        swap = ext.hot_swap_draft(str(directory))
        assert swap["parameters_loaded"] == len(CANDIDATE_TENSORS)
        assert swap["dropped_tensors"] == []
        assert drafter.load_calls == [list(CANDIDATE_TENSORS)]


# ---------------------------------------------------------------------------
# Runner topology
# ---------------------------------------------------------------------------


class TestDrafterLookup:
    """Pin the *real* vLLM object graph the drafter lookup has to walk.

    The first GPU run of hot-swap failed with a bare "no drafter model found"
    while every unit test was green, because the fake modelled only the V1
    ``runner.drafter`` topology and the engine had built a V2 runner exposing
    ``runner.speculator``.  These tests assert both topologies resolve, and
    that the constant listing them stays in sync with the lookup.
    """

    @staticmethod
    def _ext(runner: object):
        from speedlm.gateway.draft_swap import CombinedWorkerExtension

        ext = object.__new__(CombinedWorkerExtension)
        object.__setattr__(ext, "model_runner", runner)
        return ext

    @pytest.mark.parametrize("runner_attr", RUNNER_TOPOLOGIES)
    def test_resolves_under_every_shipped_topology(self, runner_attr: str) -> None:
        drafter = _FakeDrafter()
        target = _FakeModule()
        ext = self._ext(
            _FakeRunner(
                model=target,
                proposer=_FakeProposer(drafter),
                runner_attr=runner_attr,
            )
        )
        assert ext._get_drafter_model() is drafter
        assert ext._get_target_model() is target

    @pytest.mark.parametrize("runner_attr", RUNNER_TOPOLOGIES)
    def test_fake_exposes_only_its_own_topology(self, runner_attr: str) -> None:
        """The fake must not carry both names -- that is what hid the bug."""
        runner = _FakeRunner(proposer=_FakeProposer(_FakeDrafter()), runner_attr=runner_attr)
        other = [a for a in RUNNER_TOPOLOGIES if a != runner_attr]
        assert hasattr(runner, runner_attr)
        assert all(not hasattr(runner, a) for a in other)

    def test_constant_covers_both_runner_generations(self) -> None:
        """``DRAFTER_ATTRS`` is the contract the fakes are parametrized over."""
        from speedlm.gateway.draft_swap import DRAFTER_ATTRS

        assert set(DRAFTER_ATTRS) == set(RUNNER_TOPOLOGIES)

    @pytest.mark.parametrize("runner_attr", RUNNER_TOPOLOGIES)
    def test_runner_preferred_by_having_a_proposer(self, runner_attr: str) -> None:
        """A bare ``model_runner`` loses to a ``model_executor`` with a drafter."""
        from speedlm.gateway.draft_swap import CombinedWorkerExtension

        drafter = _FakeDrafter()
        ext = object.__new__(CombinedWorkerExtension)
        object.__setattr__(ext, "model_runner", _FakeRunner())
        object.__setattr__(
            ext,
            "model_executor",
            _FakeRunner(proposer=_FakeProposer(drafter), runner_attr=runner_attr),
        )
        assert ext._get_drafter_model() is drafter

    def test_missing_drafter_error_names_what_it_tried(self) -> None:
        """The failure must be self-explaining, not a bare sentinel string."""
        ext = self._ext(_FakeRunner())

        with pytest.raises(RuntimeError) as excinfo:
            ext.draft_info()

        message = str(excinfo.value)
        assert "no drafter model found" in message
        #: Both the runner attribute walked and both proposer names tried.
        assert "model_runner" in message
        assert "model_executor" in message
        for attr in RUNNER_TOPOLOGIES:
            assert attr in message
        #: ...and the concrete type that *was* found, so the next failure says
        #: whether the runner was absent or merely had no speculative config.
        assert "_FakeRunner" in message

    def test_error_reports_a_proposer_whose_model_is_unset(self) -> None:
        """A lazily-constructed proposer must not look like an absent one."""
        ext = self._ext(_FakeRunner(proposer=_FakeProposer(None)))

        with pytest.raises(RuntimeError) as excinfo:
            ext.draft_info()

        message = str(excinfo.value)
        assert "_FakeProposer(.model=None)" in message

    def test_materialization_report_shares_the_diagnostic(self) -> None:
        ext = self._ext(_FakeRunner())
        with pytest.raises(RuntimeError, match="Tried runner attributes"):
            ext.draft_materialization_report()


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
        "_intercept_aux", "_declare_aux_layers", "_buffer_aux",
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
    assert ext._declared is False
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
    assert ext._declared is False

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
    # (it clears the session buffer, which _ensure_init has to have created)
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


#: Capture state is deliberately PROCESS-wide, not per-instance.
#:
#: It used to be per-instance, and that was fine while everything ran on the
#: extension: arming installed the hook and extended the aux layers, so one
#: object owned the whole lifecycle.  It stopped being fine when the aux-layer
#: set had to be declared before the engine compiles.  That declaration runs
#: inside ``load_model``, from a wrapper installed at module import -- long
#: before any extension instance exists and with only the runner in hand.  A
#: per-instance buffer would leave that wrapper writing into state the armed
#: extension never reads, and capture would silently produce nothing.
#:
#: There is exactly one worker per process (TP fans out across processes), so
#: one session per process is also the truthful model of the runtime.
def _capture_instances(cls, count: int = 2) -> list:
    return [object.__new__(cls) for _ in range(count)]


def test_activation_capture_state_is_one_session_per_process() -> None:
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    ext_a, ext_b = _capture_instances(ActivationCaptureExtension)

    ext_a._ensure_init()
    ext_b._ensure_init()

    assert ext_a._lock is ext_b._lock
    assert ext_a._pending is ext_b._pending
    assert ext_a._pending == {}

    #: The point of sharing: a write through one view is visible through the
    #: other, which is what lets the import-time ``load_model`` wrapper and the
    #: armed extension agree on what was declared.
    ext_a._original_aux_layers = (2, 18, 33)
    assert ext_b._original_aux_layers == (2, 18, 33)


def test_combined_extension_shares_the_same_session() -> None:
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext_a, ext_b = _capture_instances(CombinedWorkerExtension)

    ext_a._ensure_init()
    ext_b._ensure_init()

    assert ext_a._lock is ext_b._lock
    assert ext_a._pending is ext_b._pending
    assert ext_a._pending == {}


# ---------------------------------------------------------------------------
# TASK A / E -- the combined class really inherits, and cannot drift
# ---------------------------------------------------------------------------


#: The non-dunder attributes vLLM's collision check sees on a worker class.
#: Sourced from ``dir(Worker)`` semantics in ``worker_base.py:266-275``.
_WORKER_ATTRS = frozenset(
    {
        "model_runner", "model_executor", "vllm_config", "device", "rank",
        "local_rank", "distributed_init_method", "parallel_config", "cache_config",
        "load_model", "init_device", "execute_model", "determine_available_memory",
        "get_kv_cache_spec", "initialize_from_config", "compile_or_warm_up_model",
        "check_health", "profile", "sleep", "wake_up", "get_model", "apply_model",
        "add_lora", "remove_lora", "list_loras", "pin_lora", "save_sharded_state",
        "save_tensorized_model", "initialize_cache", "get_cache_block_size_bytes",
        "reload_weights", "update_config", "shutdown",
    }
)


def test_combined_extension_actually_inherits() -> None:
    """The composite is a real subclass, not a hand-copied duplicate."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import (
        CombinedWorkerExtension,
        DraftSwapExtension,
    )

    assert CombinedWorkerExtension.__bases__ == (
        ActivationCaptureExtension,
        DraftSwapExtension,
    )
    #: Nothing is redefined on the composite itself, so the two bases are the
    #: single source of truth and cannot silently drift apart again.
    assert [k for k in CombinedWorkerExtension.__dict__ if not k.startswith("__")] == []


def test_combined_extension_has_exactly_one_implementation_per_method() -> None:
    """No attribute is defined on more than one class in the MRO."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import (
        CombinedWorkerExtension,
        DraftSwapExtension,
    )

    owners: dict[str, list[str]] = {}
    for cls in (CombinedWorkerExtension, ActivationCaptureExtension, DraftSwapExtension):
        for name in cls.__dict__:
            if not name.startswith("__"):
                owners.setdefault(name, []).append(cls.__name__)

    duplicated = {name: where for name, where in owners.items() if len(where) > 1}
    assert duplicated == {}


def test_combined_extension_passes_vllm_collision_rule() -> None:
    """Reproduce worker_base.py:266-275 verbatim against a stand-in worker."""
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    worker_class = type("StandInWorker", (), dict.fromkeys(_WORKER_ATTRS, None))

    extended_calls = []
    for attr in dir(CombinedWorkerExtension):
        if attr.startswith("__"):
            continue
        assert not hasattr(worker_class, attr), (
            f"Worker class already has an attribute {attr}, which conflicts "
            f"with the worker extension class."
        )
        if callable(getattr(CombinedWorkerExtension, attr)):
            extended_calls.append(attr)

    assert "hot_swap_draft" in extended_calls
    assert "draft_info" in extended_calls
    assert "activate_capture" in extended_calls


def test_injected_attributes_are_annotations_only() -> None:
    """model_runner/model_executor/vllm_config must not reach dir().

    They are the worker's own attributes; if the extension assigned them
    instead of merely annotating them, vLLM's assertion would abort startup.
    """
    from speedlm.gateway.draft_swap import CombinedWorkerExtension, DraftSwapExtension

    for cls in (CombinedWorkerExtension, DraftSwapExtension):
        exposed = {a for a in dir(cls) if not a.startswith("__")}
        assert exposed.isdisjoint({"model_runner", "model_executor", "vllm_config"})


def test_combined_extension_passes_real_vllm_collision_rule() -> None:
    """Same check, but against the installed vLLM Worker when available."""
    pytest.importorskip("torch")
    gpu_worker = pytest.importorskip("vllm.v1.worker.gpu_worker")

    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    worker_class = gpu_worker.Worker
    collisions = [
        attr
        for attr in dir(CombinedWorkerExtension)
        if not attr.startswith("__") and hasattr(worker_class, attr)
    ]
    assert collisions == []


# ---------------------------------------------------------------------------
# TASK A -- activation-capture behaviour survives the merge unchanged
# ---------------------------------------------------------------------------


_CAPTURE_METHODS = (
    "activate_capture",
    "flush_capture",
    "deactivate_capture",
    "capture_info",
    "runner_info",
    "_intercept_aux",
    "_declare_aux_layers",
    "_buffer_aux",
    "_ensure_init",
    "_get_lock",
    "_get_pending",
)


def test_capture_methods_are_inherited_not_reimplemented() -> None:
    """Every capture method on the composite IS ActivationCaptureExtension's.

    This is the anti-drift guard: previously the composite carried a
    hand-copied copy that had already lost ``capture_info``, the aux-layer
    declaration, the ``.meta.json`` sidecar write, the aux-list truncation and
    the empty-list guard.
    """
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    for name in _CAPTURE_METHODS:
        assert hasattr(CombinedWorkerExtension, name), name
        assert getattr(CombinedWorkerExtension, name) is getattr(
            ActivationCaptureExtension, name
        ), name


def test_capture_class_defaults_are_inherited() -> None:
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext = object.__new__(CombinedWorkerExtension)
    assert ext._capture_active is False
    assert ext._capture_dir is None
    assert ext._declared is False
    assert ext._final_layer_idx is None
    assert ext._original_aux_layers == ()


def test_capture_info_present_and_matches_base() -> None:
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    combined = object.__new__(CombinedWorkerExtension)
    base = object.__new__(ActivationCaptureExtension)
    for ext in (combined, base):
        ext._ensure_init()
        ext._final_layer_idx = 36
        ext._original_aux_layers = (2, 18, 33)

    assert combined.capture_info() == base.capture_info()
    assert combined.capture_info() == {
        "final_layer_idx": 36,
        "original_aux_layers": [2, 18, 33],
        "declared": False,
        "armed": False,
        "unavailable": None,
    }


class _AuxTensor:
    """Torch-free stand-in supporting ``.detach().cpu()``."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def detach(self) -> _AuxTensor:
        return self

    def cpu(self) -> _AuxTensor:
        return self


def _aux_runner(layers: tuple[int, ...], num_hidden_layers: int = 36) -> _FakeRunner:
    inner = _FakeModule()
    inner.aux_hidden_state_layers = layers
    top = _FakeModule()
    top.model = inner

    def set_aux_hidden_state_layers(new: tuple[int, ...]) -> None:
        inner.aux_hidden_state_layers = new

    top.set_aux_hidden_state_layers = set_aux_hidden_state_layers

    hf_config = types.SimpleNamespace(num_hidden_layers=num_hidden_layers)
    runner = _FakeRunner(model=top)
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config)
    )
    return runner


def test_buffer_aux_keys_by_true_layer_id() -> None:
    """The copy keyed buffers positionally (0,1,2); the base keys by layer id."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    keyed = {}
    for cls in (CombinedWorkerExtension, ActivationCaptureExtension):
        ext = object.__new__(cls)
        ext._ensure_init()
        ext._capture_active = True
        ext.model_runner = _aux_runner((2, 18, 33))
        ext._buffer_aux([_AuxTensor("a"), _AuxTensor("b"), _AuxTensor("c")])
        keyed[cls.__name__] = sorted(ext._get_pending())

    assert keyed["CombinedWorkerExtension"] == [2, 18, 33]
    assert keyed["CombinedWorkerExtension"] == keyed["ActivationCaptureExtension"]


def test_declare_aux_layers_available_on_combined() -> None:
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    ext = object.__new__(CombinedWorkerExtension)
    ext._ensure_init()
    ext.model_runner = _aux_runner((2, 18, 33), num_hidden_layers=36)

    ext._declare_aux_layers()

    assert ext._original_aux_layers == (2, 18, 33)
    assert ext._final_layer_idx == 36
    assert ext.model_runner.model.model.aux_hidden_state_layers == (2, 18, 33, 36)


def test_flush_capture_writes_meta_sidecar_source_is_shared() -> None:
    """The ``.meta.json`` sidecar lives in the single shared implementation."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    assert (
        CombinedWorkerExtension.flush_capture is ActivationCaptureExtension.flush_capture
    )
    source = ActivationCaptureExtension.flush_capture.__code__.co_consts
    assert ".meta.json" in source


def test_interception_keeps_truncation_and_empty_guard() -> None:
    """The copy dropped both; they must survive in the shared source.

    Asserted behaviourally rather than by bytecode introspection: an earlier
    version scanned ``_install_hook.__code__`` for the guard's message, which
    pinned the truncation to one particular *location* instead of its effect,
    and went red the moment the shared logic moved -- first into
    ``_intercept_aux`` to serve both runner generations, then onto the session
    so the import-time wrapper could reach it.
    """
    from speedlm.activation_capture.hook import ActivationCaptureExtension
    from speedlm.gateway.draft_swap import CombinedWorkerExtension

    assert (
        CombinedWorkerExtension._intercept_aux
        is ActivationCaptureExtension._intercept_aux
    )

    ext = object.__new__(CombinedWorkerExtension)
    ext._ensure_init()
    ext.model_runner = _aux_runner((2, 18, 33), num_hidden_layers=36)
    ext._declare_aux_layers()
    ext._capture_active = True

    # Truncation: the appended 4th entry is buffered, then removed in place.
    aux = [_AuxTensor(t) for t in ("a", "b", "c", "final")]
    ext._intercept_aux(aux)
    assert len(aux) == 3
    assert sorted(ext._get_pending()) == [2, 18, 33, 36]

    # Empty guard: an empty list would make the drafter's torch.cat explode.
    with pytest.raises(RuntimeError, match="empty before drafter"):
        ext._intercept_aux([])