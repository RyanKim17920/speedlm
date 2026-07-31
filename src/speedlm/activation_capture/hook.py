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

**What CANNOT be done without a vLLM patch:**

- Collecting the **final decoder layer's pre-norm output** as a 4th aux layer
  in the serving engine. The ``set_aux_hidden_state_layers`` call controls
  which layers are collected, but the default config only includes the 3
  EAGLE-3 aux layers. Adding the final layer requires either:
  (a) patching ``set_aux_hidden_state_layers`` to accept an arbitrary list, or
  (b) patching the runner to append the final layer index.

  Without this, we can only compare the 3 aux layers (which the drafter uses).
  The pre-norm/post-norm question for the *final layer specifically* therefore
  requires a vLLM patch to answer empirically. The prototype will report this
  as a known limitation when the final layer is absent.
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
    speedlm.activation_capture.hook:ActivationCaptureExtension``.

    The extension monkeypatches the model runner's ``_model_forward`` method
    after the model is loaded, intercepting ``aux_hidden_states`` before they
    are consumed by the drafter.

    **Important:** the extension's public methods are called via ``collective_rpc``
    from the driver process. They are NOT regular worker methods.
    """

    def __init__(self) -> None:
        self._capture_active = False
        self._capture_dir: str | None = None
        self._pending: dict[int, list[Tensor]] = {}
        self._lock = threading.Lock()
        self._original_model_forward: Any = None

    # -- collective_rpc handlers --

    def activate_capture(self, capture_dir: str) -> None:
        """Enable capture and install the monkeypatch.

        Called via collective_rpc from the driver process.
        """
        if self._capture_active:
            logger.warning("capture already active; resetting")
            self._deactivate_impl()

        self._capture_active = True
        self._capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)

        # Install monkeypatch on _model_forward
        self._install_hook()
        logger.info("Activation capture activated, output dir: %s", capture_dir)

    def flush_capture(self) -> str:
        """Flush buffered activations to disk.

        Called via collective_rpc from the driver process.
        Returns the path to the written safetensors file.
        """
        if not self._capture_active or self._capture_dir is None:
            raise RuntimeError("capture is not active")

        with self._lock:
            pending = self._pending
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
        self._capture_active = False
        self._deactivate_impl()
        logger.info("Activation capture deactivated")

    # -- internal --

    def _install_hook(self) -> None:
        """Monkeypatch the model runner's _model_forward to intercept aux states."""
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            original = GPUModelRunner._model_forward

            def _wrapped_forward(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self_ref, *args, **kwargs)

                # The result tuple for eagle3 includes aux_hidden_states.
                # gpu_model_runner.py:4362-4368 unpacks:
                #   hidden_states, aux_hidden_states = model_output
                # We capture aux_hidden_states from the runner's instance.
                # After _model_forward returns, aux_hidden_states is available
                # as a local, but we can also read it from the model output.

                # We need to capture from the runner's perspective.
                # The simplest approach: capture from the model output itself.
                # If the model returned (hidden_states, aux_hidden_states),
                # aux_hidden_states is in result.
                if isinstance(result, tuple) and len(result) >= 2:
                    aux_hidden_states = result[1]
                    if isinstance(aux_hidden_states, list) and len(aux_hidden_states) > 0:
                        self._buffer_aux(aux_hidden_states)
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

    def _buffer_aux(self, aux_hidden_states: list[Tensor]) -> None:
        """Buffer aux hidden states into the pending dict.

        aux_hidden_states is a list of tensors, one per collected layer,
        in layer-index order. Each tensor has shape (num_scheduled_tokens, H).
        """
        if not self._capture_active:
            return

        # We need the layer indices. The runner sets them via
        # set_aux_hidden_state_layers. We can read them from the model.
        # For now, buffer by position — the caller knows the layer list.
        with self._lock:
            for i, tensor in enumerate(aux_hidden_states):
                # Detach and pin for async CPU transfer
                cpu_tensor = tensor.detach().cpu()
                # Key by sequential index; the caller maps to layer indices
                key = i
                if key not in self._pending:
                    self._pending[key] = []
                self._pending[key].append(cpu_tensor)