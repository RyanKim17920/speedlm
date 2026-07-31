"""Worker extension for serving-time activation capture.

This module provides a vLLM worker extension that captures aux hidden states
from a live EAGLE-3 serving engine without modifying vLLM source files.

**Chosen mechanism: runtime monkeypatch of ``_model_forward`` via
``worker_extension_cls``.**

Why this approach:

1. **``worker_extension_cls`` is a supported entry point.** It is a documented
   ``vllm serve`` flag that injects custom methods into the worker class at
   runtime. No vLLM files on disk are modified.

2. **Monkeypatching ``_model_forward`` runs inside the worker process**, where
   ``aux_hidden_states`` is directly available after the model forward. This
   avoids the need to reconstruct the runner's row-slicing semantics at the
   layer level (a layer-level forward hook sees the padded buffer, not the
   ``num_scheduled_tokens`` slice).

3. **The capture point is after the model unpacks aux_hidden_states**
   (``gpu_model_runner.py:4362-4368``), so the slicing to
   ``num_scheduled_tokens`` is already handled by the runner.

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
"""

from __future__ import annotations

import fcntl
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
        logger.info("Flushed %d layer activations to %s", len(saved), path)
        return path

    def deactivate_capture(self) -> None:
        """Deactivate capture and remove hooks.

        Called via collective_rpc from the driver process.
        """
        self._ensure_init()
        self._capture_active = False
        self._deactivate_impl()
        logger.info("Activation capture deactivated")

    # -- internal --

    def _install_hook(self) -> None:
        """Monkeypatch the model runner's _model_forward to intercept aux states."""
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            original = GPUModelRunner._model_forward
            ext = self  # capture extension self for closure

            def _wrapped_forward(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self_ref, *args, **kwargs)

                # The result tuple for eagle3 includes aux_hidden_states.
                # gpu_model_runner.py:4362-4368 unpacks:
                #   hidden_states, aux_hidden_states = model_output
                # We capture aux_hidden_states from the model output.
                if isinstance(result, tuple) and len(result) >= 2:
                    aux_hidden_states = result[1]
                    if isinstance(aux_hidden_states, list) and len(aux_hidden_states) > 0:
                        ext._buffer_aux(aux_hidden_states)
                        # If we added the final layer (4 entries total), remove
                        # it AFTER buffering so the drafter sees exactly 3.
                        # The drafter's fc expects num_aux * H (gpu_model_runner.py
                        # concatenates all entries at :5119-5121).
                        expected = len(ext._original_aux_layers)
                        if len(aux_hidden_states) > expected:
                            del aux_hidden_states[expected:]
                return result

            self._original_model_forward = original
            GPUModelRunner._model_forward = _wrapped_forward
        except Exception:
            logger.exception("Failed to install _model_forward hook")
            raise

    def _deactivate_impl(self) -> None:
        """Remove the monkeypatched _model_forward."""
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            if hasattr(GPUModelRunner, "_model_forward"):
                original = getattr(GPUModelRunner, "_model_forward", None)
                if (original is not None
                        and self._original_model_forward is not None
                        and original.__name__ == "_wrapped_forward"):
                    GPUModelRunner._model_forward = self._original_model_forward
        except Exception:
            logger.exception("Error removing _model_forward hook")

    def _extend_aux_layers(self) -> None:
        """Add the final decoder layer to the aux collection list.

        This lets the serving engine capture the final layer's pre-norm
        output as a 4th aux entry.  The runner's ``aux_hidden_state_layers``
        is set by ``set_aux_hidden_state_layers`` (gpu_model_runner.py:5402)
        landing on the model's ``EagleModelMixin`` (interfaces.py:1326-1327).

        The drafter's ``fc`` width is fixed from the drafter's own config at
        construction time (``llama_eagle3.py:176-187``) and is unaffected.
        We remove the extra entry after buffering (in the _model_forward hook)
        so the concatenation at ``gpu_model_runner.py:5119-5121`` produces a
        3*H tensor for the 3*H drafter fc.
        """
        try:
            runner = self.model_runner  # type: ignore[attr-defined]
            model = runner.model
            current_layers = model.aux_hidden_state_layers

            # Read the total layer count from the model config
            hf_config = runner.vllm_config.model_config.hf_config
            num_layers = getattr(hf_config, "num_hidden_layers", None)
            if num_layers is None:
                logger.warning(
                    "could not determine num_hidden_layers; "
                    "skipping final-layer aux extension"
                )
                return

            final_idx = num_layers  # 1-based index used by vLLM's aux collection
            if final_idx in current_layers:
                # Already present (e.g., offline extraction config)
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
        except Exception:
            logger.exception(
                "Failed to extend aux layers; final-layer capture unavailable"
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
        try:
            runner = self.model_runner  # type: ignore[attr-defined]
            layer_indices: tuple[int, ...] = runner.model.aux_hidden_state_layers
        except Exception:
            # Fallback to positional indices if we can't reach the model
            layer_indices = tuple(range(len(aux_hidden_states)))

        with self._get_lock():
            pending = self._get_pending()
            for i, tensor in enumerate(aux_hidden_states):
                cpu_tensor = tensor.detach().cpu()
                key = layer_indices[i] if i < len(layer_indices) else i
                if key not in pending:
                    pending[key] = []
                pending[key].append(cpu_tensor)