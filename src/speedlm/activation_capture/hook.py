"""Worker extension for serving-time activation capture.

This module provides a vLLM worker extension that captures aux hidden states
from a live EAGLE-3 serving engine without modifying vLLM source files.

**Chosen mechanism: declare the final aux-layer set before the engine
compiles, then arm/disarm only the buffering.**

Why it has to be this way
-------------------------

The obvious design -- extend ``aux_hidden_state_layers`` from three layers to
four at *arm* time -- is only correct on an eager engine, and eager is not the
production default.

``EagleModelMixin._maybe_add_hidden_state`` decides what to emit with a plain
membership test (``interfaces.py:1336``)::

    if layer_idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)

That test is traced once.  vLLM compiles and CUDA-graph-captures the target
forward during ``Worker.compile_or_warm_up_model`` (``gpu_worker.py:724``,
``capture_model`` at ``:762``) and then only ever *replays* it.  Mutating the
tuple afterwards changes the attribute but not the captured graph, so the
model advertises four aux layers while the replayed forward keeps emitting
three.  The labelling guard in :meth:`_buffer_aux` then -- correctly -- refuses
to key four buffers off three tensors, ``collective_rpc`` propagates the
error, and ``EngineCore`` dies.  Reproduced on GPU as job 370798.

So the aux-layer set is declared **once, before compilation**, and arming
toggles only whether the states are buffered.  Arming changes no shape and no
graph, works identically under CUDA graphs and eager, and needs no restart.

Where "before compilation" is
-----------------------------

vLLM never calls a worker extension's ``__init__``; it appends the class to
the worker class's ``__bases__`` (``worker_base.py:279``) and asserts that the
extension defines *no* attribute the worker already has (``:271``).  Overriding
``Worker.load_model`` or ``Worker.init_device`` from the extension is
therefore impossible by construction -- an appended base cannot win the MRO,
and merely declaring the name aborts worker start-up.

The one thing that does run early is the **import**.  vLLM resolves
``--worker-extension-cls`` with ``resolve_obj_by_qualname`` at
``worker_base.py:262``, inside ``init_worker`` -- before ``init_device``
(``gpu_worker.py:279``), before ``load_model`` (``:406``), and long before
``compile_or_warm_up_model`` (``:724``).  Importing this module therefore
happens strictly earlier than any compilation, and :func:`install_bootstrap`
runs at import to wrap the runner class:

* ``load_model`` -- after vLLM sets the eagle3 default tuple
  (``gpu_model_runner.py:5402`` / ``gpu/spec_decode/eagle/eagle3_utils.py:32``)
  we append the final decoder layer.  Still inside ``load_model``, so it
  precedes ``determine_available_memory``'s profile forward, the warm-up and
  the graph capture.
* the interception point -- installed permanently so the extra aux entry is
  stripped on *every* forward, armed or not.

The strip is not optional
-------------------------

The runner concatenates the *entire* aux list before the drafter's ``fc``
(``gpu_model_runner.py:5118-5121``,
``gpu/spec_decode/autoregressive/speculator.py:168-172``)::

    target_hidden_states = torch.cat(
        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)

There is no slice, no length check and no positional unpack anywhere on that
path, and the drafter's ``fc`` width is frozen at construction from its own
config (``llama_eagle3.py:175-214``, defaulting to three aux states).  A
four-entry list is therefore not "harmlessly ignored" -- it is a hard
``RuntimeError: mat1 and mat2 shapes cannot be multiplied (N x 4H and 3H x H)``.
The extension buffers all four entries and then truncates the *same list
object* back to the canonical three, so the drafter is bit-for-bit unaffected.

Because the declaration now happens before warm-up, the strip has to be in
place before warm-up too -- V2's ``_dummy_run`` really does call
``speculator.propose(..., aux_hidden_states=...)``
(``gpu/model_runner.py:577-619``), so an unstripped list would crash during
graph capture rather than on the first request.

Two runner generations
----------------------

``gpu_worker.py:384-398`` picks the runner class on
``vllm_config.use_v2_model_runner``, and the interception point differs:

* V1 ``vllm.v1.worker.gpu_model_runner.GPUModelRunner`` returns
  ``(hidden_states, aux_hidden_states)`` from ``_model_forward`` (``:3783``),
  which the runner unpacks at ``:4364``.
* V2 ``vllm.v1.worker.gpu.model_runner.GPUModelRunner`` has **no**
  ``_model_forward``.  It unpacks inline in ``execute_model``
  (``gpu/model_runner.py:1325-1331``) and parks the list on
  ``execute_model_state`` (``:1341-1348``).  Two different consumers read it
  back: ``sample_tokens`` (``:1369``) on the serving path and ``_dummy_run``
  (``:577``) during warm-up.  Intercepting ``execute_model`` covers both;
  intercepting ``sample_tokens`` -- as this hook used to -- misses warm-up
  entirely, which only became fatal once the declaration moved before capture.

Both interception points are resolved from the *live* runner class rather than
an import, because both modules ship side by side and a hard-coded V1 patch
installs silently on a V2 build.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    # Only executed by static type checkers. mypy has per-module overrides for
    # torch/safetensors/vllm (see pyproject.toml [[tool.mypy.overrides]]) so
    # this does not require torch to be installed in the project venv.
    from torch import Tensor

try:
    #: Inside a worker only vLLM's own logger tree has handlers and an INFO
    #: level; a bare ``speedlm.*`` logger drops every record on the floor.  Job
    #: 370927 was diagnosed without a single line from this module in
    #: ``vllm.log`` for exactly that reason -- whether the declaration ran at
    #: all had to be inferred.  Route through vLLM's logger when there is one.
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(__name__)
except Exception:  # noqa: BLE001 -- the project venv has no vLLM
    logger = logging.getLogger(__name__)


#: Runner methods that see ``aux_hidden_states`` before the drafter consumes
#: them, in resolution order.  ``execute_model`` exists on BOTH generations, so
#: the V1-only ``_model_forward`` must be probed first or a V1 runner would be
#: hooked through the V2 path.  The order is load-bearing; see the module
#: docstring for the line references.
HOOK_POINTS: tuple[str, ...] = ("_model_forward", "execute_model")

#: Set to ``1`` to keep importing this module from patching vLLM's runner
#: classes.  Only useful for tests that import the module in a process where
#: vLLM happens to be importable but no engine will ever run.
BOOTSTRAP_OPT_OUT_ENV = "SPEEDLM_CAPTURE_NO_BOOTSTRAP"

#: The runner classes to patch, newest first.  Each is optional: the two
#: generations ship side by side, but a given vLLM build may drop either.
_RUNNER_TARGETS: tuple[tuple[str, str], ...] = (
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
)


def _resolve_inner_model(model: Any) -> Any:
    """Resolve the model that carries ``aux_hidden_state_layers``.

    The attribute lives on the *inner* model (the one inheriting
    ``EagleModelMixin``), not the top-level ``...ForCausalLM``.  Resolved the
    same way vLLM does in ``SupportsEagle3.set_aux_hidden_state_layers``
    (``interfaces.py:1387-1400``): try ``get_language_model()`` /
    ``.language_model``, then access ``.model``.
    """
    parent_ref = model
    if hasattr(model, "get_language_model"):
        parent_ref = model.get_language_model()
    elif hasattr(model, "language_model"):
        parent_ref = model.language_model
    return parent_ref.model


#: Key under which the declared aux-layer set is recorded in vLLM's
#: ``additional_config``.  See :func:`_register_compile_cache_factor`.
COMPILE_CACHE_FACTOR_KEY = "speedlm_capture_aux_hidden_state_layers"


def _register_compile_cache_factor(runner: Any, declared: tuple[int, ...]) -> bool:
    """Fold ``declared`` into vLLM's torch.compile cache key.

    Declaring the extra aux layer before compilation is necessary but NOT
    sufficient, and job 370927 is the proof: the declaration landed at the
    right moment and the engine still ran a three-aux-layer forward, because
    vLLM never compiled anything.  It found a cache hit and loaded a graph
    compiled ninety minutes earlier by the *pre-fix* run (job 370798), which
    had traced three aux layers.  The attribute said four, the replayed graph
    emitted three, and the labelling guard refused -- the original production
    error, reproduced verbatim by a fix that was itself correct.

    The cause is that the compile cache key is
    ``[env_hash, config_hash, code_hash, compiler_hash]``
    (``vllm/compilation/backends.py:1054-1063``) and *none* of the four depends
    on the live ``aux_hidden_state_layers`` tuple.  vLLM hashes only the
    config-declared ``eagle_aux_hidden_state_layer_ids``
    (``vllm/config/speculative.py:298-317``) -- precisely because the layer set
    "affects the computation graph".  Our layer is appended imperatively, so it
    is invisible to that hash and to the AOT key derived from the same
    ``config_hash`` (``vllm/compilation/caching.py:565-581``).

    Recording the set in ``additional_config``, which *is* hashed
    (``vllm/config/vllm.py:473-483``), gives a four-layer engine its own cache
    namespace.  A graph traced with three aux layers can then never be replayed
    by an engine that declared four, in either direction.

    Called from :meth:`_CaptureSession.declare`, i.e. inside ``load_model`` and
    therefore before the first ``compute_hash`` call, which happens at the
    first compile.

    Returns:
        True if the factor was registered.  False means the hazard is still
        live -- ``additional_config`` was not a plain dict, or no config was
        reachable -- and the caller must rely on the runtime stale-graph guard
        in :meth:`_CaptureSession.intercept`.
    """
    vllm_config = getattr(runner, "vllm_config", None)
    additional = getattr(vllm_config, "additional_config", None)
    #: Only a plain dict may be mutated: the field is typed
    #: ``dict | SupportsHash`` and a SupportsHash object computes its own hash
    #: from fields we must not invent.
    if not isinstance(additional, dict):
        logger.warning(
            "cannot record the declared aux layers %s in additional_config "
            "(got %r); vLLM's torch.compile cache key will not distinguish "
            "this engine from one compiled without the final aux layer, so a "
            "warm cache may replay a stale graph",
            declared, type(additional).__name__,
        )
        return False

    additional[COMPILE_CACHE_FACTOR_KEY] = list(declared)
    logger.info(
        "Recorded declared aux layers %s under additional_config[%r] so the "
        "torch.compile cache key distinguishes this graph",
        declared, COMPILE_CACHE_FACTOR_KEY,
    )
    return True


def _resolve_hook_point(runner_cls: Any) -> str:
    """Return the interception attribute to wrap on ``runner_cls``.

    Raises:
        RuntimeError: if the class exposes no known interception point.
    """
    for name in HOOK_POINTS:
        if hasattr(runner_cls, name):
            return name
    raise RuntimeError(
        f"model runner {runner_cls.__module__}.{runner_cls.__qualname__} "
        f"exposes none of {HOOK_POINTS}; activation capture cannot "
        f"intercept aux_hidden_states on this vLLM build"
    )


def _extract_aux(runner: Any, attr: str, result: Any) -> Any:
    """Pull the aux list out of an intercepted call.

    The list is found in a different place per generation but is in both cases
    the *same object* the drafter will later concatenate, so truncating it here
    is visible downstream.
    """
    if attr == "_model_forward":
        #: V1: second element of the return value (unpacked at
        #: ``gpu_model_runner.py:4364``).
        if isinstance(result, tuple) and len(result) >= 2:
            return result[1]
        return None
    #: V2: parked on ``execute_model_state`` (``gpu/model_runner.py:1341-1348``)
    #: and read back by both ``sample_tokens`` (``:1369``) and ``_dummy_run``
    #: (``:577``).
    state = getattr(runner, "execute_model_state", None)
    return getattr(state, "aux_hidden_states", None)


class _AsyncTransfer(NamedTuple):
    """One fused activation batch moving into a pinned host buffer."""

    layer_indices: tuple[int, ...]
    host: Any
    ready: Any
    source: Any
    pool_key: tuple[Any, ...]


class _CaptureSession:
    """Process-wide capture state.

    Module scope rather than instance state because the three pieces that must
    agree are created at three different times in three different scopes: the
    runner-class patches are installed at *import* (before ``load_model``), the
    aux-layer declaration happens *inside* ``load_model`` with only the runner
    in hand, and arming arrives much later over ``collective_rpc`` on the
    worker instance.  vLLM never calls the extension's ``__init__``, so there
    is no instance to hang shared state off anyway.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Return to a pristine, un-bootstrapped state (tests only)."""
        #: ``reset_session`` is test-only, but an async transfer may still be
        #: writing into a pinned slot when teardown reaches it.  Wait before
        #: dropping the queue and its last references; otherwise CUDA can keep
        #: writing into storage Python believes is reclaimable.
        lock = getattr(self, "lock", None)
        transfers = getattr(self, "transfers", None)
        if lock is not None and transfers:
            with lock:
                self.drain_transfers(wait=True)

        self.active = False
        self.capture_dir: str | None = None
        self.pending: dict[int, list] | None = None
        self.lock: threading.Lock | None = None
        #: CUDA transfers awaiting an event before their pinned buffers may be
        #: read.  FIFO order is the capture row order.
        self.transfers: deque[_AsyncTransfer] | None = None
        #: Recycled pinned destinations, keyed by fused shape and dtype.  The
        #: pool grows only to the maximum number of overlapping transfers, not
        #: to the number of captured forwards.
        self.pinned_pool: dict[tuple[Any, ...], list[Any]] | None = None
        self.transfer_streams: dict[Any, Any] | None = None
        self.async_transfer_disabled = False

        #: Declaration state, all set inside ``load_model``.
        self.declared = False
        self.runner: Any = None
        self.inner_model: Any = None
        self.final_layer_idx: int | None = None
        self.original_aux_layers: tuple[int, ...] = ()
        #: Why capture is impossible on this engine, when it is.
        self.unavailable: str | None = None

        #: Installed patches, for symmetry and for tests: (cls, attr, original).
        self.patches: list[tuple[Any, str, Any]] = []
        self.patched_class: Any = None
        self.patched_attr: str | None = None
        self.installed_wrapper: Any = None

    # -- lazy mutable state --

    def ensure_init(self) -> None:
        if self.lock is None:
            self.lock = threading.Lock()
        if self.pending is None:
            self.pending = {}
        if self.transfers is None:
            self.transfers = deque()
        if self.pinned_pool is None:
            self.pinned_pool = {}
        if self.transfer_streams is None:
            self.transfer_streams = {}

    # -- declaration (runs inside load_model, before compile) --

    def declare(self, runner: Any, *, force: bool = False) -> None:
        """Declare the FULL aux-layer set on the live model.

        Called from the ``load_model`` wrapper, i.e. after vLLM has installed
        the eagle3 default tuple and before any compilation, memory profiling
        or graph capture.  Appending here means the traced forward bakes in the
        final count, so arming later cannot desynchronise the graph from the
        attribute.

        Args:
            runner: the live model runner.
            force: re-declare even if a declaration already happened.  Only
                used by tests driving the declaration by hand.

        Raises:
            RuntimeError: if the layer count cannot be determined.  A silent
                skip would leave a graph baked with three layers and no way to
                capture the final one, which is the defect this replaces.
        """
        if self.declared and not force:
            return

        #: ``use_aux_hidden_state_outputs`` gates the whole aux path
        #: (``gpu_model_runner.py:5384``).  Without eagle3 the model returns a
        #: bare tensor and the runner asserts on it, so declaring layers would
        #: break a perfectly good engine rather than enable capture.
        if getattr(runner, "use_aux_hidden_state_outputs", True) is False:
            self.unavailable = (
                "the engine is not collecting eagle3 aux hidden states "
                "(use_aux_hidden_state_outputs is False), so there is nothing "
                "to capture"
            )
            self.declared = True
            return

        model = runner.model
        inner_model = _resolve_inner_model(model)
        current_layers = tuple(inner_model.aux_hidden_state_layers)

        hf_config = runner.vllm_config.model_config.hf_config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError(
                "could not determine num_hidden_layers; "
                "cannot extend aux layers for final-layer capture"
            )

        self.runner = runner
        self.inner_model = inner_model
        final_idx = num_layers  # 1-based index used by vLLM's aux collection

        if final_idx in current_layers:
            #: The engine was configured this way itself -- offline extraction
            #: sets ``eagle_aux_hidden_state_layer_ids`` and the drafter's
            #: ``fc`` is sized for all of them (``llama_eagle3.py:187`` reads
            #: ``len(layer_ids)``).  The drafter genuinely consumes every
            #: entry, so nothing may be stripped.
            self.original_aux_layers = current_layers
            self.final_layer_idx = None
            self.declared = True
            logger.info(
                "Aux layers %s already include final layer %d; "
                "declaring nothing and stripping nothing",
                current_layers, final_idx,
            )
            return

        self.original_aux_layers = current_layers
        self.final_layer_idx = final_idx
        extended = current_layers + (final_idx,)
        model.set_aux_hidden_state_layers(extended)
        #: Before the first compile, and therefore before the first
        #: ``VllmConfig.compute_hash()``: declaring the layers is worthless if
        #: vLLM then replays a cached graph traced with a different set.
        _register_compile_cache_factor(runner, extended)
        self.declared = True
        logger.info(
            "Declared aux layers %s -> %s (added final layer %d) before "
            "compilation; the drafter will still be handed %d",
            current_layers, extended, final_idx, len(current_layers),
        )

    # -- interception (every forward, armed or not) --

    def intercept(self, runner: Any, aux_hidden_states: Any) -> None:
        """Buffer ``aux_hidden_states`` then strip the declared final layer.

        Args:
            runner: the live runner, used only as a fallback when the
                declaration did not cache the inner model.
            aux_hidden_states: the runner's aux list, or any non-list value
                (``None`` on non-last PP ranks / non-eagle steps), ignored.
        """
        if not isinstance(aux_hidden_states, list):
            return

        expected = len(self.original_aux_layers)

        if len(aux_hidden_states) == 0:
            #: Hard guard: the drafter will crash on an empty list (torch.cat
            #: of nothing).  Only *our* problem once we declared the extra aux
            #: layer -- an engine that was already collecting nothing is not
            #: something the capture extension broke, so leave it to the runner.
            if expected > 0:
                raise RuntimeError(
                    "aux_hidden_states is empty before drafter; "
                    "activation capture extension left the "
                    "engine in a broken state"
                )
            return

        #: Stale-graph detection.  We declared one layer more than the drafter
        #: needs, so a healthy graph emits ``expected + 1``.  Getting exactly
        #: ``expected`` back means the forward being executed was never traced
        #: with our declaration -- a replayed compile-cache entry, the failure
        #: job 370927 hit.  Noting it on the FIRST forward, armed or not, is
        #: what lets a caller find out from a disarmed warm-up pass -- the
        #: labelling guard in ``buffer`` only fires once ARMED, and there it
        #: propagates out of ``execute_model`` and kills EngineCore, which is
        #: how a diagnosable cache-staleness bug became a dead engine and zero
        #: artifacts.  The guard still fires for an already-armed engine; this
        #: only makes the same fact visible earlier and more cheaply.
        if (
            self.final_layer_idx is not None
            and self.unavailable is None
            and len(aux_hidden_states) == expected
        ):
            self._note_stale_graph(len(aux_hidden_states))

        if self.active:
            self.buffer(runner, aux_hidden_states)

        #: Truncate AFTER buffering so the drafter sees exactly the layers its
        #: ``fc`` was built for.  Unconditional -- not gated on ``active`` --
        #: because the declaration is unconditional: the forward emits the
        #: extra state on every pass from warm-up onwards.
        if expected > 0 and len(aux_hidden_states) > expected:
            del aux_hidden_states[expected:]

    def _note_stale_graph(self, produced: int) -> None:
        """Record that the running graph predates our declaration.

        Diagnosis only: it logs and sets :attr:`unavailable`, which makes
        ``capture_info`` and ``activate_capture`` report the cause.  It
        deliberately does NOT disarm and does NOT restore the layer tuple, so
        an engine that is *already* armed still hits the labelling guard in
        :meth:`buffer` and refuses the request rather than silently writing
        rows keyed off the wrong layers.  The point is that a caller now learns
        this from a disarmed warm-up forward, before arming, instead of from a
        dead EngineCore.

        Must not raise: the engine is mid-forward.
        """
        reason = (
            f"the compiled forward produced {produced} aux hidden states, "
            f"but the aux layers {self.original_aux_layers} plus final layer "
            f"{self.final_layer_idx} were declared before compilation; the "
            f"engine is replaying a torch.compile/CUDA-graph artifact that was "
            f"traced without the final layer (a stale compile cache under "
            f"VLLM_CACHE_ROOT).  Capture is disabled for this engine; rerun "
            f"with VLLM_DISABLE_COMPILE_CACHE=1 or a cleared cache"
        )
        logger.error("%s", reason)
        self.unavailable = reason

    def _append_host_locked(
        self, layer_indices: tuple[int, ...], host_tensors: list[Tensor]
    ) -> None:
        """Append one completed forward while preserving its live labels."""
        assert self.pending is not None
        for layer_idx, tensor in zip(layer_indices, host_tensors, strict=True):
            self.pending.setdefault(layer_idx, []).append(tensor)

    def _finalize_transfer_locked(self, transfer: _AsyncTransfer) -> None:
        """Move a completed pinned slot into ordinary, durable host memory."""
        import torch

        #: The pinned slot is a ring resource and will be overwritten.  Clone it
        #: only after its event has completed, then retain unbound views of the
        #: pageable clone in ``pending``.  The views keep their shared base alive.
        pageable = transfer.host.clone()
        host_tensors = list(torch.unbind(pageable))
        if len(host_tensors) != len(transfer.layer_indices):
            raise RuntimeError(
                "completed activation transfer changed the number of aux "
                "tensors; refusing to buffer rows with ambiguous labels"
            )
        self._append_host_locked(transfer.layer_indices, host_tensors)
        assert self.pinned_pool is not None
        self.pinned_pool.setdefault(transfer.pool_key, []).append(transfer.host)

    def drain_transfers(self, *, wait: bool) -> None:
        """Finalize ready CUDA transfers, optionally waiting for every one.

        The caller holds :attr:`lock`.  Non-waiting drains run on the serving
        path and only recycle already-complete ring slots.  Waiting drains are
        reserved for flush/reset/deactivation, where host access is required.
        """
        assert self.transfers is not None
        while self.transfers:
            transfer = self.transfers[0]
            if wait:
                transfer.ready.synchronize()
            elif not transfer.ready.query():
                break
            self._finalize_transfer_locked(transfer)
            self.transfers.popleft()

    def _try_async_transfer(
        self,
        layer_indices: tuple[int, ...],
        aux_hidden_states: list[Tensor],
    ) -> bool:
        """Enqueue one fused D2H copy without synchronizing the serving thread."""
        first = aux_hidden_states[0]
        device = getattr(first, "device", None)
        if (
            self.async_transfer_disabled
            or getattr(device, "type", None) != "cuda"
            or len(aux_hidden_states) < 2
            or not _is_fusable(aux_hidden_states)
        ):
            return False

        import torch

        assert self.transfers is not None
        assert self.pinned_pool is not None
        assert self.transfer_streams is not None

        stream = None
        host = None
        pool_key: tuple[Any, ...] | None = None
        submitted = False
        try:
            #: ``stack`` snapshots vLLM's graph-owned output buffers on the
            #: current compute stream.  The independent allocation is essential:
            #: the next graph replay may overwrite those static outputs while the
            #: copy stream is still draining this step.
            fused = torch.stack(aux_hidden_states).detach()
            pool_key = (tuple(fused.shape), fused.dtype)
            slots = self.pinned_pool.setdefault(pool_key, [])
            host = (
                slots.pop()
                if slots
                else torch.empty_like(fused, device="cpu", pin_memory=True)
            )

            stream = self.transfer_streams.get(device)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self.transfer_streams[device] = stream

            ready = torch.cuda.Event()
            current_stream = torch.cuda.current_stream(device=device)
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                host.copy_(fused, non_blocking=True)
                submitted = True
                ready.record(stream)

            self.transfers.append(
                _AsyncTransfer(layer_indices, host, ready, fused, pool_key)
            )
            return True
        except Exception as exc:
            #: Before submission, a pinned allocation/stream setup failure can
            #: safely degrade to the synchronous path.  After submission, first
            #: finish the DMA: releasing a destination that CUDA may still write
            #: would trade performance for silent corruption.
            if submitted:
                try:
                    assert stream is not None
                    stream.synchronize()
                except Exception as sync_exc:
                    raise RuntimeError(
                        "asynchronous activation transfer failed after DMA was "
                        "submitted and its completion could not be established"
                    ) from sync_exc
            if host is not None and pool_key is not None:
                self.pinned_pool.setdefault(pool_key, []).append(host)
            self.async_transfer_disabled = True
            logger.warning(
                "asynchronous pinned activation transfer failed; using "
                "synchronous host copies for the rest of this capture",
                exc_info=exc,
            )
            return False

    def buffer(self, runner: Any, aux_hidden_states: list[Tensor]) -> None:
        """Buffer aux hidden states into the pending dict, keyed by layer index.

        Each tensor has shape ``(num_scheduled_tokens, H)``.
        """
        self.ensure_init()

        # Use actual layer indices from the model's aux_hidden_state_layers.
        # These are the indices the runner configured via
        # set_aux_hidden_state_layers (interfaces.py:1326-1327).
        #: There used to be a ``except Exception:`` here that fell back to
        #: ``range(len(aux_hidden_states))``.  That fallback is not a
        #: degradation, it is a silent corruption: for aux layers
        #: ``(2, 18, 33, 36)`` it would key the buffer ``0, 1, 2, 3``, and every
        #: downstream consumer -- the bf16 tolerance, which is
        #: ``2^-8 * (layer_id + 4)``, and the residual-stream depth the capture
        #: is compared against -- would then be reading the wrong layer with no
        #: signal at all.  Capture is opt-in and only reaches here while active,
        #: so failing the request is strictly better than writing mislabelled
        #: activations into a training cache.
        try:
            inner_model = self.inner_model
            if inner_model is None:
                inner_model = _resolve_inner_model(runner.model)
                self.inner_model = inner_model
            layer_indices: tuple[int, ...] = inner_model.aux_hidden_state_layers
        except Exception as exc:
            raise RuntimeError(
                "cannot read aux_hidden_state_layers from the running model, "
                "so captured activations cannot be labelled with their true "
                "layer indices; refusing to buffer positionally-keyed rows"
            ) from exc

        #: The guard the whole redesign exists to keep satisfiable.  It still
        #: refuses on a genuine mismatch -- a graph baked with a different
        #: count than the model advertises means the rows cannot be labelled,
        #: and mislabelled training rows are worse than a failed request.
        if len(layer_indices) != len(aux_hidden_states):
            raise RuntimeError(
                f"the model reported {len(layer_indices)} aux layers "
                f"{layer_indices} but the forward produced "
                f"{len(aux_hidden_states)} aux hidden states; the layer "
                f"labelling cannot be trusted"
            )

        assert self.lock is not None and self.pending is not None
        with self.lock:
            assert self.transfers is not None
            self.drain_transfers(wait=False)
            if self._try_async_transfer(layer_indices, aux_hidden_states):
                return
            self._append_host_locked(layer_indices, _to_host(aux_hidden_states))

    # -- patch installation --

    def install(self, runner_cls: Any) -> str:
        """Wrap ``load_model`` and the interception point on ``runner_cls``.

        Idempotent per class: re-installing would nest wrappers and strip
        twice.

        Returns:
            The interception attribute that was wrapped.
        """
        attr = _resolve_hook_point(runner_cls)
        session = self

        for patched_cls, patched_attr, _ in self.patches:
            if patched_cls is runner_cls and patched_attr == attr:
                return attr

        original_load = runner_cls.load_model

        def _wrapped_load_model(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_load(self_ref, *args, **kwargs)
            #: Declaration must not be swallowed: a load that "succeeded" while
            #: leaving three layers baked into the graph is exactly the state
            #: that made capture unarmable.
            session.declare(self_ref)
            return result

        original_intercept = getattr(runner_cls, attr)

        def _wrapped_intercept(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_intercept(self_ref, *args, **kwargs)
            session.intercept(self_ref, _extract_aux(self_ref, attr, result))
            return result

        runner_cls.load_model = _wrapped_load_model
        setattr(runner_cls, attr, _wrapped_intercept)
        self.patches.append((runner_cls, "load_model", original_load))
        self.patches.append((runner_cls, attr, original_intercept))
        self.patched_class = runner_cls
        self.patched_attr = attr
        self.installed_wrapper = _wrapped_intercept
        logger.info(
            "Installed activation-capture bootstrap on %s.%s "
            "(declaration in load_model, interception in %s)",
            runner_cls.__module__, runner_cls.__qualname__, attr,
        )
        return attr

    def uninstall(self) -> None:
        """Remove every patch this session installed (tests / teardown)."""
        for runner_cls, attr, original in reversed(self.patches):
            try:
                setattr(runner_cls, attr, original)
            except Exception:  # noqa: BLE001 -- teardown must not mask errors
                logger.exception("Error restoring %s.%s", runner_cls, attr)
        self.patches = []
        self.patched_class = None
        self.patched_attr = None
        self.installed_wrapper = None


#: The one session per worker process.
_SESSION = _CaptureSession()


def get_session() -> _CaptureSession:
    """Return the process-wide capture session."""
    return _SESSION


def reset_session() -> None:
    """Uninstall patches and clear all capture state.  Tests only."""
    _SESSION.uninstall()
    _SESSION.reset()


def install_bootstrap(runner_classes: Any = None) -> list[str]:
    """Declare-before-compile bootstrap: patch the vLLM runner classes.

    Called at import, which vLLM performs inside ``init_worker``
    (``worker_base.py:262``) -- strictly before ``init_device``, ``load_model``
    and ``compile_or_warm_up_model``.  That ordering is the whole point: it is
    the earliest moment code of ours can run in the worker process, and it is
    the only one available, because an appended worker-extension base can
    neither override ``Worker.load_model`` nor even declare the name
    (``worker_base.py:271`` asserts the extension shares no attribute with the
    worker).

    Args:
        runner_classes: explicit classes to patch.  Defaults to whichever of
            vLLM's two runner generations import successfully.

    Returns:
        The interception attribute wrapped on each class, in order.
    """
    if runner_classes is None:
        runner_classes = []
        for module_name, cls_name in _RUNNER_TARGETS:
            try:
                module = __import__(module_name, fromlist=[cls_name])
            except Exception:  # noqa: BLE001 -- generation may not ship
                logger.debug("runner module %s not importable", module_name)
                continue
            runner_cls = getattr(module, cls_name, None)
            if runner_cls is not None:
                runner_classes.append(runner_cls)

    installed = []
    for runner_cls in runner_classes:
        installed.append(_SESSION.install(runner_cls))
    return installed


def _bootstrap_on_import() -> None:
    """Run :func:`install_bootstrap` at import, best-effort.

    An import failure here means vLLM is not installed (the project venv runs
    the unit tests), which is not an error.  A *patch* failure is logged rather
    than raised so that importing the module never breaks an engine that was
    not going to capture anything -- ``activate_capture`` refuses loudly later.
    """
    if os.environ.get(BOOTSTRAP_OPT_OUT_ENV) == "1":
        logger.info("%s=1; skipping capture bootstrap", BOOTSTRAP_OPT_OUT_ENV)
        return
    try:
        install_bootstrap()
    except Exception:  # noqa: BLE001
        logger.exception("activation-capture bootstrap failed")


# -- device-to-host transfer ------------------------------------------------


def _is_fusable(aux_hidden_states: list[Tensor]) -> bool:
    """True when all aux tensors can go home in a single stacked copy."""
    first = aux_hidden_states[0]
    shape = first.shape
    dtype = first.dtype
    device = first.device
    return all(
        tensor.shape == shape
        and tensor.dtype == dtype
        and tensor.device == device
        for tensor in aux_hidden_states[1:]
    )


def _to_host(aux_hidden_states: list[Tensor]) -> list[Tensor]:
    """Copy one forward pass' aux states to the host in ONE transfer.

    ``Tensor.cpu()`` enqueues a ``cudaMemcpyAsync`` and then drains the
    stream, so calling it per aux layer costs four pipeline stalls per
    forward pass -- on the serving path, once for every decode step.  The
    four aux tensors always share a shape (``num_scheduled_tokens``, H),
    a dtype and a device, so stacking them on-device first turns those
    four stalls into one.  The device-side stack is a copy at HBM
    bandwidth of a buffer far smaller than the forward's own transient
    activations; the host-side byte count is unchanged.

    Slicing the fused host tensor back out per layer costs nothing (the
    slices are views) and drops the per-step host allocations from four
    to one.

    Falls back to the per-layer copy whenever the tensors are not uniform
    or the stack is refused: a slower capture is always preferable to a
    lost or mislabelled row.
    """
    try:
        if len(aux_hidden_states) > 1 and _is_fusable(aux_hidden_states):
            import torch

            # One ``detach`` on the stacked result rather than one per
            # layer: stacking cannot smuggle in a graph that detaching
            # afterwards would not drop.
            fused = torch.stack(aux_hidden_states).detach().cpu()
            return list(torch.unbind(fused))
    except Exception:
        logger.warning(
            "fused device-to-host capture copy failed; falling back "
            "to one transfer per aux layer",
            exc_info=True,
        )
    return [tensor.detach().cpu() for tensor in aux_hidden_states]


# ---------------------------------------------------------------------------
# Worker extension
# ---------------------------------------------------------------------------


class _SessionAttr:
    """Expose one :class:`_CaptureSession` field as an extension attribute.

    A *data* descriptor on purpose.  vLLM mixes this class into the worker by
    appending it to ``__bases__``, so the extension sits last in the MRO -- but
    a data descriptor found anywhere on the MRO still beats the instance
    ``__dict__``.  Without that, ``self._capture_active = True`` inside
    ``activate_capture`` would write a worker-instance attribute that shadowed
    the session the import-time wrappers actually read, and arming would appear
    to succeed while buffering nothing.
    """

    def __init__(self, field: str) -> None:
        self.field = field

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(_SESSION, self.field)

    def __set__(self, obj: Any, value: Any) -> None:
        setattr(_SESSION, self.field, value)


class ActivationCaptureExtension:
    """vLLM worker extension for capturing aux hidden states at serving time.

    Register via ``--worker-extension-cls
    speedlm.activation_capture.hook.ActivationCaptureExtension``.

    Registering it is what imports this module inside the worker, which is what
    installs the declare-before-compile bootstrap (see the module docstring).
    By the time any of the methods below are reachable over ``collective_rpc``
    the engine has already compiled and captured graphs with the final aux
    count baked in, so arming and disarming are pure buffering switches: no
    shape changes, no re-tracing, no restart.

    **Important:** the extension's public methods are called via
    ``collective_rpc`` from the driver process.  They are NOT regular worker
    methods.

    **Note:** vLLM injects this class via ``worker_class.__bases__`` injection
    and NEVER calls ``__init__``.  Every mutable field below is a
    :class:`_SessionAttr` view onto the process-wide session, so the class
    functions without ``__init__`` *and* without the instance/import-time split
    that instance attributes would create.
    """

    _capture_active = _SessionAttr("active")
    _capture_dir = _SessionAttr("capture_dir")
    _final_layer_idx = _SessionAttr("final_layer_idx")
    _original_aux_layers = _SessionAttr("original_aux_layers")
    _inner_model = _SessionAttr("inner_model")
    _pending = _SessionAttr("pending")
    _lock = _SessionAttr("lock")
    _patched_class = _SessionAttr("patched_class")
    _patched_attr = _SessionAttr("patched_attr")
    _installed_wrapper = _SessionAttr("installed_wrapper")
    _declared = _SessionAttr("declared")
    _unavailable = _SessionAttr("unavailable")

    def _ensure_init(self) -> None:
        """Lazy initialization for the session's mutable state."""
        _SESSION.ensure_init()

    def _get_lock(self) -> threading.Lock:
        """Return the session lock, initializing it lazily."""
        _SESSION.ensure_init()
        assert _SESSION.lock is not None
        return _SESSION.lock

    def _get_pending(self) -> dict[int, list]:
        """Return all captured rows after pending CUDA transfers complete."""
        _SESSION.ensure_init()
        with self._get_lock():
            _SESSION.drain_transfers(wait=True)
            assert _SESSION.pending is not None
            return _SESSION.pending

    # -- collective_rpc handlers --

    def activate_capture(self, capture_dir: str) -> None:
        """Arm buffering.  Called via collective_rpc from the driver process.

        Deliberately does *not* touch ``aux_hidden_state_layers`` and does not
        install anything: both happened before the engine compiled.  Arming is
        therefore safe on a CUDA-graph-capturing engine, which is the
        production default and which the previous arm-time-extension design
        could not survive.

        Raises:
            RuntimeError: if the declare-before-compile bootstrap never ran, or
                ran and found nothing to capture.  Arming a capture that would
                silently miss the final layer is worse than refusing.
        """
        if not _SESSION.declared:
            raise RuntimeError(
                "the activation-capture bootstrap never declared the aux "
                "layers, so the compiled forward does not produce the final "
                "layer and arming would capture the wrong thing; the "
                "extension module must be imported before the worker loads "
                "the model (which --worker-extension-cls guarantees)"
            )
        if _SESSION.unavailable is not None:
            raise RuntimeError(
                f"activation capture is unavailable: {_SESSION.unavailable}"
            )

        _SESSION.ensure_init()
        if _SESSION.active:
            logger.warning("capture already active; resetting")

        with self._get_lock():
            _SESSION.drain_transfers(wait=True)
            _SESSION.pending = {}
            _SESSION.async_transfer_disabled = False

        _SESSION.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)
        _SESSION.active = True
        logger.info("Activation capture armed, output dir: %s", capture_dir)

    def flush_capture(self) -> str:
        """Flush buffered activations to disk.

        Called via collective_rpc from the driver process.
        Returns the path to the written safetensors file.
        """
        self._ensure_init()
        if not _SESSION.active or _SESSION.capture_dir is None:
            #: An engine that :meth:`_CaptureSession._note_stale_graph` found
            #: unable to capture is "not active" for a specific reason.
            #: Reporting the bare sentence sends the reader back to the engine
            #: log to find out why -- exactly the detour job 370927 forced.
            if _SESSION.unavailable is not None:
                raise RuntimeError(
                    f"capture is not active: {_SESSION.unavailable}"
                )
            raise RuntimeError("capture is not active")

        with self._get_lock():
            #: This is the only serving operation that must wait for capture
            #: DMA.  Until here, CUDA events are queried but never synchronized
            #: on the request thread.
            _SESSION.drain_transfers(wait=True)
            assert _SESSION.pending is not None
            pending = _SESSION.pending
            _SESSION.pending = {}

        if not pending:
            logger.warning("flush_capture called with no buffered data")

        import torch  # lazy: only available at runtime inside the vLLM venv

        # Stack tensors per layer
        saved: dict[str, Tensor] = {}
        for lidx in sorted(pending.keys()):
            layer_tensors = pending[lidx]
            if len(layer_tensors) == 1:
                stacked = layer_tensors[0]
            else:
                stacked = torch.cat(layer_tensors, dim=0)
            saved[f"layer_{lidx}"] = stacked

        # Write to safetensors with flock for async safety
        path = os.path.join(_SESSION.capture_dir, "captured.safetensors")
        lock_path = path + ".lock"

        # Create lock file before writing
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            from safetensors.torch import save_file

            save_file(saved, path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        # Remove lock file after writer finishes so readers can proceed
        os.remove(lock_path)

        # Write metadata alongside the captured file so the caller can
        # distinguish drafter-input layers from the appended final layer.
        meta_path = path + ".meta.json"
        meta = {
            "final_layer_idx": _SESSION.final_layer_idx,
            "original_aux_layers": list(_SESSION.original_aux_layers),
        }
        with open(meta_path, "w") as mf:
            json.dump(meta, mf)

        logger.info("Flushed %d layer activations to %s", len(saved), path)
        return path

    def runner_info(self) -> dict:
        """Report which model runner generation this worker actually loaded.

        Called via collective_rpc from the driver process.

        This exists so a test can *prove* which runner it exercised rather
        than assuming.  vLLM picks the generation from
        ``VllmConfig.use_v2_model_runner`` (``vllm/config/vllm.py:519-522``),
        which honours ``VLLM_USE_V2_MODEL_RUNNER`` when set but otherwise
        derives the answer from the model architecture, the speculative
        method and Triton availability -- and silently falls back to V1 when
        V2 does not support the config (``vllm/config/vllm.py:546-553``).
        A run that merely *requested* a generation therefore cannot be assumed
        to have *got* it, and a V1-only capture hook has already shipped
        undetected twice for exactly that reason.

        Three independent signals are returned so a disagreement between them
        is visible rather than papered over:

        * ``runner_class`` -- the live runner's ``module.QualName``.  The two
          generations live in different modules that ship side by side.
        * ``hook_point`` -- the attribute the hook resolved against the live
          class, or ``None`` when neither is present.
        * ``config_use_v2`` -- what the worker's own ``VllmConfig`` says, when
          reachable.  This is the value vLLM itself branched on.

        ``generation`` is derived from ``hook_point`` because that is the axis
        the capture hook actually depends on: V1 exposes ``_model_forward``,
        V2 does not (see :data:`HOOK_POINTS`).

        Returns:
            Dict with keys ``generation`` (``"v1"``/``"v2"``/``"unknown"``),
            ``runner_class`` (str), ``hook_point`` (str or None) and
            ``config_use_v2`` (bool or None).
        """
        runner = self.model_runner  # type: ignore[attr-defined]
        runner_cls = type(runner)

        hook_point: str | None = None
        for name in HOOK_POINTS:
            if hasattr(runner_cls, name):
                hook_point = name
                break

        #: ``_model_forward`` is V1-only; ``execute_model`` exists on both, so
        #: resolving in HOOK_POINTS order makes this an exact discriminator.
        if hook_point == "_model_forward":
            generation = "v1"
        elif hook_point is not None:
            generation = "v2"
        else:
            generation = "unknown"

        #: Best-effort: the config may hang off the worker or the runner
        #: depending on the vLLM build, and the property can raise on a
        #: partially-built config.  A missing third signal must not break the
        #: two that matter.
        config_use_v2: bool | None = None
        for holder in (self, runner):
            vllm_config = getattr(holder, "vllm_config", None)
            if vllm_config is None:
                continue
            try:
                config_use_v2 = bool(vllm_config.use_v2_model_runner)
            except Exception:  # noqa: BLE001 -- diagnostic only, never fatal
                config_use_v2 = None
            break

        return {
            "generation": generation,
            "runner_class": f"{runner_cls.__module__}.{runner_cls.__qualname__}",
            "hook_point": hook_point,
            "config_use_v2": config_use_v2,
        }

    def capture_info(self) -> dict:
        """Return metadata about the capture session.

        Called via collective_rpc from the driver process.  Returns the final
        layer index and the drafter-visible (pre-declaration) aux layer tuple
        so the caller can correctly split drafter-input layers from the
        appended final regression-target layer.

        ``declared``/``armed`` are reported too: on a graph-capturing engine
        the declaration is the thing that had to happen before compile, and a
        caller that cannot see it would be back to guessing.

        Returns:
            Dict with keys ``final_layer_idx`` (int or None),
            ``original_aux_layers`` (list[int]), ``declared`` (bool),
            ``armed`` (bool) and ``unavailable`` (str or None).
        """
        return {
            "final_layer_idx": _SESSION.final_layer_idx,
            "original_aux_layers": list(_SESSION.original_aux_layers),
            "declared": _SESSION.declared,
            "armed": _SESSION.active,
            "unavailable": _SESSION.unavailable,
        }

    def deactivate_capture(self) -> None:
        """Disarm buffering.  Called via collective_rpc from the driver.

        Deliberately leaves the declared aux-layer set and the interception
        wrappers in place.  Restoring the three-layer tuple would desynchronise
        the attribute from the already-captured four-layer graph -- the exact
        failure this design removes -- and re-arming would then be impossible
        without an engine restart.  Disarmed cost is one surplus
        ``hidden + residual`` add per forward plus a list truncation.
        """
        _SESSION.ensure_init()
        _SESSION.active = False
        with self._get_lock():
            #: A pinned ring slot cannot be released or reused while CUDA may
            #: still be writing it.  Deactivation is off the serving hot path,
            #: so finish outstanding work before discarding captured rows.
            _SESSION.drain_transfers(wait=True)
            _SESSION.pending = {}
        _SESSION.capture_dir = None
        logger.info("Activation capture disarmed")

    # -- internal (kept as methods: exercised directly by the unit tests) --

    @staticmethod
    def _resolve_inner_model(model: Any) -> Any:
        """See :func:`_resolve_inner_model`."""
        return _resolve_inner_model(model)

    @staticmethod
    def _is_fusable(aux_hidden_states: list[Tensor]) -> bool:
        """See :func:`_is_fusable`."""
        return _is_fusable(aux_hidden_states)

    def _to_host(self, aux_hidden_states: list[Tensor]) -> list[Tensor]:
        """See :func:`_to_host`."""
        return _to_host(aux_hidden_states)

    def _resolve_hook_point(self) -> tuple[Any, str]:
        """Resolve the runner class to patch and which method to wrap."""
        runner_cls = type(self.model_runner)  # type: ignore[attr-defined]
        return runner_cls, _resolve_hook_point(runner_cls)

    def _declare_aux_layers(self) -> None:
        """Declare the full aux-layer set against ``self.model_runner``.

        The production path never calls this -- the ``load_model`` wrapper
        does, before compilation.  It exists so a caller holding only the
        worker can drive the declaration explicitly.
        """
        _SESSION.declare(self.model_runner, force=True)  # type: ignore[attr-defined]

    def _intercept_aux(self, aux_hidden_states: Any) -> None:
        """See :meth:`_CaptureSession.intercept`."""
        _SESSION.intercept(
            getattr(self, "model_runner", None), aux_hidden_states
        )

    def _buffer_aux(self, aux_hidden_states: list[Tensor]) -> None:
        """Buffer aux hidden states, if armed."""
        if not _SESSION.active:
            return
        _SESSION.buffer(
            getattr(self, "model_runner", None), aux_hidden_states
        )


_bootstrap_on_import()
