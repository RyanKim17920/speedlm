"""Worker extension for serving-time activation capture.

This module provides a vLLM worker extension that captures aux hidden states
from a live EAGLE-3 serving engine without modifying vLLM source files.

**Chosen mechanism: runtime monkeypatch of ``_model_forward`` via
``worker_extension_cls``.**

Why this approach:

1. **``worker_extension_cls`` is a supported entry point.** It is a documented
   ``vllm serve`` flag that injects custom methods into the worker class at
   runtime. No vLLM files on disk are modified.

2. **Monkeypatching the runner runs inside the worker process**, where
   ``aux_hidden_states`` is directly available after the model forward. This
   avoids the need to reconstruct the runner's row-slicing semantics at the
   layer level (a layer-level forward hook sees the padded buffer, not the
   ``num_scheduled_tokens`` slice).

3. **The capture point is after the model unpacks aux_hidden_states**
   (V1 ``gpu_model_runner.py:4362-4368``, V2 ``gpu/model_runner.py:1327-1349``),
   so the slicing to ``num_scheduled_tokens`` is already handled by the runner.

4. **No vLLM patch required.** The extension is a standalone class that the
   caller registers via ``--worker-extension-cls``. It only touches private
   vLLM attributes at runtime (inevitable given vLLM exposes no supported hook
   for this).

**Final-layer pre-norm capture.** The extension CAN collect the final decoder
layer as a 4th aux layer by calling ``set_aux_hidden_state_layers`` with an
extended tuple. The drafter's ``fc`` width is fixed from the drafter's own
config at construction time (``llama_eagle3.py:176-187``) and is unaffected.
However, the concatenation at ``gpu_model_runner.py:5119-5121`` would produce
a 4*H tensor for a 3*H drafter. The extension therefore removes the extra
entry from the ``aux_hidden_states`` list AFTER buffering it, so the drafter
sees only the 3 canonical layers.

**Two runner generations.** ``gpu_worker.py:384-398`` picks the runner class on
``vllm_config.use_v2_model_runner``, and the two generations expose different
interception points:

* V1 ``vllm.v1.worker.gpu_model_runner.GPUModelRunner`` has ``_model_forward``
  (``:3783``), which returns ``(hidden_states, aux_hidden_states)``.
* V2 ``vllm.v1.worker.gpu.model_runner.GPUModelRunner`` has NO
  ``_model_forward`` at all.  It unpacks the model output inline
  (``gpu/model_runner.py:1327-1330``), parks it on ``execute_model_state``
  (``:1341-1349``), and ``sample_tokens`` (``:1358``) reads it back
  (``:1369``) and hands it straight to ``speculator.propose`` (``:1466``).

The hook therefore resolves the interception point from the *live* runner
object (``type(self.model_runner)``) rather than importing a hard-coded class.
Importing the V1 module always succeeds -- both modules ship side by side --
so a hard-coded V1 patch installs silently, buffers nothing, and (fatally)
never strips the appended 4th entry, which reaches the drafter's ``fc`` as
``RuntimeError: mat1 and mat2 shapes cannot be multiplied (N x 4H and 3H x H)``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only executed by static type checkers. mypy has per-module overrides for
    # torch/safetensors/vllm (see pyproject.toml [[tool.mypy.overrides]]) so
    # this does not require torch to be installed in the project venv.
    from torch import Tensor

logger = logging.getLogger(__name__)


#: Runner methods that see ``aux_hidden_states`` before the drafter consumes
#: them, in resolution order.  ``sample_tokens`` exists on BOTH generations, so
#: the V1-only ``_model_forward`` must be probed first or a V1 runner would be
#: hooked through the V2 path.  The order is load-bearing; see the module
#: docstring for the line references.
HOOK_POINTS: tuple[str, ...] = ("_model_forward", "sample_tokens")


# ---------------------------------------------------------------------------
# Worker extension
# ---------------------------------------------------------------------------


class ActivationCaptureExtension:
    """vLLM worker extension for capturing aux hidden states at serving time.

    Register via ``--worker-extension-cls
    speedlm.activation_capture.hook.ActivationCaptureExtension``.

    The extension monkeypatches the model runner's ``_model_forward`` method
    after the model is loaded, intercepting ``aux_hidden_states`` before they
    are consumed by the drafter.

    **Important:** the extension's public methods are called via ``collective_rpc``
    from the driver process. They are NOT regular worker methods.

    **Note:** vLLM injects this class via ``worker_class.__bases__`` injection
    and NEVER calls ``__init__``.  All instance state uses class-level defaults
    or lazy initialization (via ``_ensure_init``) to function without ``__init__``.
    """

    # Class-level defaults — safe because they are the initial "no capture"
    # state and immutable types.  Mutable state is lazy-initialized below.
    _capture_active = False
    _capture_dir: str | None = None
    _original_model_forward: Any = None
    _final_layer_idx: int | None = None
    _original_aux_layers: tuple[int, ...] = ()

    #: Where the monkeypatch actually landed.  Recorded so ``_deactivate_impl``
    #: restores the same attribute on the same class it patched, instead of
    #: guessing at a hard-coded one.
    _patched_class: Any = None
    _patched_attr: str | None = None
    _installed_wrapper: Any = None

    # These must be per-instance (mutable) — lazy-initialized on first use.
    _pending: dict[int, list] | None = None
    _lock: threading.Lock | None = None

    def _ensure_init(self) -> None:
        """Lazy initialization for mutable per-instance state.

        vLLM never calls ``__init__`` (it appends the class to the worker's
        base tuple).  This method guarantees the mutable defaults exist.
        """
        if self._lock is None:
            self._lock = threading.Lock()
        if self._pending is None:
            self._pending = {}

    def _get_lock(self) -> threading.Lock:
        """Return the per-instance lock, initializing it lazily."""
        self._ensure_init()
        assert self._lock is not None  # guaranteed by _ensure_init
        return self._lock

    def _get_pending(self) -> dict[int, list]:
        """Return the per-instance pending dict, initializing it lazily."""
        self._ensure_init()
        assert self._pending is not None  # guaranteed by _ensure_init
        return self._pending

    # -- collective_rpc handlers --

    def activate_capture(self, capture_dir: str) -> None:
        """Enable capture and install the monkeypatch.

        Called via collective_rpc from the driver process.

        Also adds the final decoder layer to the aux collection list so that
        its pre-norm output is captured alongside the canonical EAGLE-3 aux
        layers. The extra entry is sliced off before reaching the drafter.
        """
        self._ensure_init()
        if self._capture_active:
            logger.warning("capture already active; resetting")
            self._deactivate_impl()

        self._capture_active = True
        self._capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)

        # Add the final decoder layer to aux collection.
        # self is mixed into the worker class, so we have model_runner.
        self._extend_aux_layers()

        # Install monkeypatch on _model_forward
        self._install_hook()
        logger.info("Activation capture activated, output dir: %s", capture_dir)

    def flush_capture(self) -> str:
        """Flush buffered activations to disk.

        Called via collective_rpc from the driver process.
        Returns the path to the written safetensors file.
        """
        self._ensure_init()
        if not self._capture_active or self._capture_dir is None:
            raise RuntimeError("capture is not active")

        with self._get_lock():
            pending = self._get_pending()
            self._pending = {}

        if not pending:
            logger.warning("flush_capture called with no buffered data")

        import torch  # lazy: only available at runtime inside the vLLM venv

        # Group by layer index
        by_layer: dict[int, list[Tensor]] = {}
        for layer_idx, tensors in pending.items():
            by_layer[layer_idx] = tensors

        # Stack tensors per layer
        saved: dict[str, Tensor] = {}
        for lidx in sorted(by_layer.keys()):
            layer_tensors = by_layer[lidx]
            if len(layer_tensors) == 1:
                stacked = layer_tensors[0]
            else:
                stacked = torch.cat(layer_tensors, dim=0)
            saved[f"layer_{lidx}"] = stacked

        # Write to safetensors with flock for async safety
        path = os.path.join(self._capture_dir, "captured.safetensors")
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
            "final_layer_idx": self._final_layer_idx,
            "original_aux_layers": list(self._original_aux_layers),
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
        self._ensure_init()
        runner = self.model_runner  # type: ignore[attr-defined]
        runner_cls = type(runner)

        hook_point: str | None = None
        for name in HOOK_POINTS:
            if hasattr(runner_cls, name):
                hook_point = name
                break

        #: ``_model_forward`` is V1-only; ``sample_tokens`` exists on both, so
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
        """Return metadata about the active capture session.

        Called via collective_rpc from the driver process.  Returns the final
        layer index and the original (pre-extension) aux layer tuple so the
        caller can correctly split drafter-input layers from the appended
        final regression-target layer.

        Returns:
            Dict with keys ``final_layer_idx`` (int or None) and
            ``original_aux_layers`` (list[int]).
        """
        self._ensure_init()
        return {
            "final_layer_idx": self._final_layer_idx,
            "original_aux_layers": list(self._original_aux_layers),
        }

    def deactivate_capture(self) -> None:
        """Deactivate capture and remove hooks.

        Called via collective_rpc from the driver process.
        """
        self._ensure_init()
        self._capture_active = False
        self._deactivate_impl()
        logger.info("Activation capture deactivated")

    # -- internal --

    def _intercept_aux(self, aux_hidden_states: Any) -> None:
        """Buffer ``aux_hidden_states`` then strip the appended final layer.

        Shared by both runner generations: whatever the interception point,
        the list object handed to the drafter is the same object we mutate
        here, so the truncation is visible downstream.

        Args:
            aux_hidden_states: the runner's aux list, or any non-list value
                (``None`` on non-last PP ranks / non-eagle steps), which is
                ignored.
        """
        if not isinstance(aux_hidden_states, list):
            return

        # Only strip when the extension actually succeeded (detected by
        # _original_aux_layers being non-empty).
        expected = len(self._original_aux_layers)

        if len(aux_hidden_states) == 0:
            # Hard guard: the drafter will crash on an empty list (torch.cat
            # of nothing).  Only *our* problem once we extended the aux list —
            # an engine that was already collecting nothing is not something
            # the capture extension broke, so leave it to the runner.
            if expected > 0:
                raise RuntimeError(
                    "aux_hidden_states is empty before drafter; "
                    "activation capture extension left the "
                    "engine in a broken state"
                )
            return

        self._buffer_aux(aux_hidden_states)

        # Remove the extra entry AFTER buffering so the drafter sees exactly
        # the canonical layers.  The drafter's fc expects num_aux * H (the
        # runner concatenates all entries -- V1 gpu_model_runner.py:5119-5121,
        # V2 via speculator.propose).
        if expected > 0 and len(aux_hidden_states) > expected:
            del aux_hidden_states[expected:]

    def _resolve_hook_point(self) -> tuple[Any, str]:
        """Resolve the runner class to patch and which method to wrap.

        Resolution is against the *live* runner object, never an imported
        class: both runner modules ship side by side, so importing the V1 one
        succeeds even when the engine is running V2 and the patch would land
        on a class nobody instantiates.

        Returns:
            ``(runner_class, method_name)``.

        Raises:
            RuntimeError: if the runner exposes no known interception point.
        """
        runner = self.model_runner  # type: ignore[attr-defined]
        runner_cls = type(runner)
        for name in HOOK_POINTS:
            if hasattr(runner_cls, name):
                return runner_cls, name
        raise RuntimeError(
            f"model runner {runner_cls.__module__}.{runner_cls.__qualname__} "
            f"exposes none of {HOOK_POINTS}; activation capture cannot "
            f"intercept aux_hidden_states on this vLLM build"
        )

    def _install_hook(self) -> None:
        """Monkeypatch the live runner class to intercept aux hidden states."""
        try:
            runner_cls, attr = self._resolve_hook_point()
            original = getattr(runner_cls, attr)
            ext = self  # capture extension self for closure

            if attr == "_model_forward":
                #: V1: the aux list is the second element of the return value
                #: (gpu_model_runner.py:4362-4368 unpacks it right after).
                def _wrapped(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
                    result = original(self_ref, *args, **kwargs)
                    if isinstance(result, tuple) and len(result) >= 2:
                        ext._intercept_aux(result[1])
                    return result
            else:
                #: V2: ``execute_model`` already parked the aux list on
                #: ``execute_model_state`` (gpu/model_runner.py:1341-1349) and
                #: ``sample_tokens`` reads it back (:1369) before handing it to
                #: ``speculator.propose`` (:1466).  Intercept on the way IN, so
                #: the truncation is in place before propose() sees the list.
                def _wrapped(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
                    state = getattr(self_ref, "execute_model_state", None)
                    if state is not None:
                        ext._intercept_aux(getattr(state, "aux_hidden_states", None))
                    return original(self_ref, *args, **kwargs)

            self._original_model_forward = original
            self._patched_class = runner_cls
            self._patched_attr = attr
            self._installed_wrapper = _wrapped
            setattr(runner_cls, attr, _wrapped)
            logger.info(
                "Installed activation-capture hook on %s.%s",
                runner_cls.__qualname__, attr,
            )
        except Exception:
            logger.exception("Failed to install activation capture hook")
            raise

    def _deactivate_impl(self) -> None:
        """Remove the monkeypatch and undo the aux-layer extension."""
        try:
            runner_cls = self._patched_class
            attr = self._patched_attr
            if runner_cls is not None and attr is not None:
                current = getattr(runner_cls, attr, None)
                if (current is not None
                        and self._installed_wrapper is not None
                        and current is self._installed_wrapper
                        and self._original_model_forward is not None):
                    setattr(runner_cls, attr, self._original_model_forward)
        except Exception:
            logger.exception("Error removing activation capture hook")
        finally:
            self._patched_class = None
            self._patched_attr = None
            self._installed_wrapper = None
            self._original_model_forward = None

        self._restore_aux_layers()

    @staticmethod
    def _resolve_inner_model(model: Any) -> Any:
        """Resolve the model that carries ``aux_hidden_state_layers``.

        The attribute lives on the *inner* model (the one inheriting
        ``EagleModelMixin``), not the top-level ``...ForCausalLM``.  Resolved
        the same way vLLM does in ``SupportsEagle3.set_aux_hidden_state_layers``
        (``interfaces.py:1395-1406``): try ``get_language_model()`` /
        ``.language_model``, then access ``.model``.
        """
        parent_ref = model
        if hasattr(model, "get_language_model"):
            parent_ref = model.get_language_model()
        elif hasattr(model, "language_model"):
            parent_ref = model.language_model
        return parent_ref.model

    def _restore_aux_layers(self) -> None:
        """Undo :meth:`_extend_aux_layers`, if it extended anything.

        Deactivation has to put the aux list back or a later
        ``activate_capture`` would extend an already-extended list.  Idempotent
        and best-effort: a failure here must not mask the caller's own error,
        and the engine is still serviceable (the strip is driven by
        ``_original_aux_layers``, which is re-derived on the next activation).
        """
        if self._final_layer_idx is None:
            return
        try:
            runner = self.model_runner  # type: ignore[attr-defined]
            runner.model.set_aux_hidden_state_layers(self._original_aux_layers)
            logger.info(
                "Restored aux layers to %s (removed final layer %d)",
                self._original_aux_layers, self._final_layer_idx,
            )
        except Exception:
            logger.exception("Error restoring aux hidden state layers")
        finally:
            self._final_layer_idx = None

    def _extend_aux_layers(self) -> None:
        """Add the final decoder layer to the aux collection list.

        This lets the serving engine capture the final layer's pre-norm
        output as a 4th aux entry.  The runner's ``aux_hidden_state_layers``
        is set by ``set_aux_hidden_state_layers`` (gpu_model_runner.py:5402)
        landing on the model's ``EagleModelMixin`` (interfaces.py:1326-1327).

        The ``aux_hidden_state_layers`` attribute lives on the *inner* model
        (the one that inherits ``EagleModelMixin``), not the top-level model
        (e.g. ``GptOssForCausalLM``).  We resolve the inner model the same
        way vLLM does in ``SupportsEagle3.set_aux_hidden_state_layers``
        (interfaces.py:1395-1406): try ``get_language_model()`` /
        ``.language_model``, then access ``.model``.

        The drafter's ``fc`` width is fixed from the drafter's own config at
        construction time (``llama_eagle3.py:176-187``) and is unaffected.
        We remove the extra entry after buffering (in the _model_forward hook)
        so the concatenation at ``gpu_model_runner.py:5119-5121`` produces a
        3*H tensor for the 3*H drafter fc.

        Raises:
            RuntimeError: if the inner model cannot be resolved or lacks the
                aux_hidden_state_layers attribute.  The caller
                (``activate_capture``) should NOT catch this — a failed
                extension means the pre-norm capture is broken and serving
                must not proceed.
        """
        runner = self.model_runner  # type: ignore[attr-defined]
        model = runner.model
        inner_model = self._resolve_inner_model(model)

        current_layers = inner_model.aux_hidden_state_layers

        # Read the total layer count from the model config
        hf_config = runner.vllm_config.model_config.hf_config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError(
                "could not determine num_hidden_layers; "
                "cannot extend aux layers for final-layer capture"
            )

        final_idx = num_layers  # 1-based index used by vLLM's aux collection
        if final_idx in current_layers:
            #: Already present.  Two very different causes, and conflating them
            #: is fatal: either the engine was configured that way (offline
            #: extraction -- the drafter genuinely consumes all of them), or a
            #: prior activation already extended and was not rolled back.  In
            #: the latter case recording the EXTENDED tuple as "original" makes
            #: ``_intercept_aux`` stop stripping, and the drafter's fc gets
            #: 4*H.  ``_final_layer_idx`` distinguishes the two.
            if self._final_layer_idx != final_idx:
                self._original_aux_layers = current_layers
            return

        self._original_aux_layers = current_layers
        self._final_layer_idx = final_idx
        extended = current_layers + (final_idx,)
        model.set_aux_hidden_state_layers(extended)
        logger.info(
            "Extended aux layers from %s to %s (added final layer %d)",
            current_layers, extended, final_idx,
        )

    def _buffer_aux(self, aux_hidden_states: list[Tensor]) -> None:
        """Buffer aux hidden states into the pending dict.

        aux_hidden_states is a list of tensors, one per collected layer,
        in layer-index order. Each tensor has shape (num_scheduled_tokens, H).
        """
        self._ensure_init()
        if not self._capture_active:
            return

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
            runner = self.model_runner  # type: ignore[attr-defined]
            inner_model = self._resolve_inner_model(runner.model)
            layer_indices: tuple[int, ...] = inner_model.aux_hidden_state_layers
        except Exception as exc:
            raise RuntimeError(
                "cannot read aux_hidden_state_layers from the running model, "
                "so captured activations cannot be labelled with their true "
                "layer indices; refusing to buffer positionally-keyed rows"
            ) from exc

        if len(layer_indices) != len(aux_hidden_states):
            raise RuntimeError(
                f"the model reported {len(layer_indices)} aux layers "
                f"{layer_indices} but the forward produced "
                f"{len(aux_hidden_states)} aux hidden states; the layer "
                f"labelling cannot be trusted"
            )

        with self._get_lock():
            pending = self._get_pending()
            for i, tensor in enumerate(aux_hidden_states):
                cpu_tensor = tensor.detach().cpu()
                key = layer_indices[i]
                if key not in pending:
                    pending[key] = []
                pending[key].append(cpu_tensor)