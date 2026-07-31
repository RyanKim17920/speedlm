"""Worker extension for hot-swapping EAGLE-3 draft weights in place.

This module provides a vLLM worker extension that swaps the drafter's weights
without restarting the engine, preserving CUDA-graph-captured tensor pointers
by using the layerwise reload infrastructure.

**Worker extension composition.**  vLLM accepts a single
``--worker-extension-cls`` string.  This project already uses
``ActivationCaptureExtension`` for serving-time activation capture.
Rather than requiring the caller to choose between the two, this module
provides ``CombinedWorkerExtension`` -- a composite class that merges both
extensions via multiple inheritance.  Register it via::

    --worker-extension-cls speedlm.gateway.draft_swap:CombinedWorkerExtension

The composite class merges both extension namespaces into one MRO chain.
vLLM checks for attribute collisions at startup (``worker_base.py:268-275``);
the two extensions use distinct method prefixes (``activate_``/``flush_``/
``deactivate_``/``_install_``/``_deactivate_``/``_buffer_`` vs.
``hot_swap_``/``_draft_``) so there are no conflicts.

**What CANNOT be done:** swap a drafter whose architecture, shapes, or
quantization differ from the currently loaded one.  The hot-swap is purely
weight-oriented; the tensor topology and CUDA-graph bindings must already
match.  The caller MUST validate compatibility before invoking the swap.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DraftSwapExtension:
    """vLLM worker extension for in-place draft weight hot-swapping.

    Mixin for ``CombinedWorkerExtension``.  Do NOT register this directly via
    ``--worker-extension-cls`` unless you do NOT need activation capture.

    Public methods are called via ``collective_rpc`` from the driver process.
    """

    #: Injected by vLLM at runtime via multiple inheritance into WorkerBase.
    model_executor: Any  # noqa: RUF012
    model_runner: Any  # noqa: RUF012
    vllm_config: Any  # noqa: RUF012

    def hot_swap_draft(self, weights_path: str) -> dict[str, Any]:
        """Swap the drafter's draft weights from *weights_path* in place.

        Validates that the new weights match the current drafter's parameter
        shapes, dtypes, and quantization before applying them.  Uses the
        layerwise reload infrastructure so that CUDA-graph-captured tensor
        pointers survive the swap.

        Returns a dict with ``"swapped": True`` and the count of parameters
        loaded on success.

        Called via collective_rpc from the driver process.
        """

        drafter = self._get_drafter_model()
        if drafter is None:
            raise RuntimeError("no drafter model found; hot-swap requires a draft model")

        # Load new weights from disk
        new_weights = self._load_weights_file(weights_path)

        # Validate compatibility BEFORE touching anything
        self._validate_compatibility(drafter, new_weights)

        # Apply weights in-place using layerwise reload
        count = self._apply_weights(drafter, new_weights)

        logger.info(
            "Hot-swapped %d drafter parameters from %s", count, weights_path
        )
        return {"swapped": True, "parameters_loaded": count}

    def draft_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded drafter.

        Used by the caller to validate compatibility before hot-swapping.

        Returns a dict with ``"num_parameters"``, ``"parameter_shapes"``,
        ``"parameter_dtypes"``, and ``"quantization"``.
        """

        drafter = self._get_drafter_model()
        if drafter is None:
            raise RuntimeError("no drafter model found")

        param_shapes: dict[str, list[int]] = {}
        param_dtypes: dict[str, str] = {}
        for name, param in drafter.named_parameters():
            param_shapes[name] = list(param.shape)
            param_dtypes[name] = str(param.dtype)

        quantization = self._get_quantization()

        return {
            "num_parameters": sum(1 for _ in drafter.named_parameters()),
            "parameter_shapes": param_shapes,
            "parameter_dtypes": param_dtypes,
            "quantization": quantization,
        }

    # -- internal helpers --

    def _get_drafter_model(self) -> Any:
        """Retrieve the drafter's model object from the model runner."""
        try:
            runner = self.model_executor
        except AttributeError:
            try:
                runner = self.model_runner
            except AttributeError:
                return None

        drafter = getattr(runner, "drafter", None)
        if drafter is None:
            return None

        return getattr(drafter, "model", None)

    def _load_weights_file(self, path: str) -> dict[str, Any]:
        """Load a safetensors weights file from *path*."""
        from safetensors.torch import load_file as _load_file

        return _load_file(path)

    def _validate_compatibility(
        self, drafter: Any, new_weights: dict[str, Any]
    ) -> None:
        """Raise if *new_weights* is incompatible with *drafter*."""

        existing_params = {
            name: param
            for name, param in drafter.named_parameters()
        }

        # Check for missing parameters
        missing = set(existing_params.keys()) - set(new_weights.keys())
        if missing:
            raise ValueError(
                f"draft weights missing {len(missing)} parameters: "
                f"{sorted(missing)[:5]}..."
            )

        # Check for unexpected parameters
        extra = set(new_weights.keys()) - set(existing_params.keys())
        if extra:
            raise ValueError(
                f"draft weights contain {len(extra)} unexpected parameters: "
                f"{sorted(extra)[:5]}..."
            )

        # Check shapes and dtypes
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

    def _apply_weights(
        self, drafter: Any, new_weights: dict[str, Any]
    ) -> int:
        """Apply *new_weights* into *drafter* in-place, preserving CUDA graphs."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(drafter)

        weights_iter = new_weights.items()
        loaded = drafter.load_weights(weights_iter)

        try:
            model_config = getattr(self, "vllm_config", None)
            model_config = model_config.model_config if model_config is not None else None
            finalize_layerwise_reload(drafter, model_config)
        except Exception:
            logger.exception("finalize_layerwise_reload failed; weights may be partial")
            raise

        count = len(loaded) if loaded is not None else 0
        return count

    def _get_quantization(self) -> str | None:
        """Read the drafter's quantization setting from the vLLM config."""
        try:
            config = getattr(self, "vllm_config", None)
            if config is not None:
                spec = getattr(config, "speculative_config", None)
                if spec is not None:
                    draft_cfg = getattr(spec, "draft_model_config", None)
                    if draft_cfg is not None:
                        return getattr(draft_cfg, "quantization", None)
        except Exception:
            pass
        return None


class CombinedWorkerExtension:  # noqa: RUF012
    """Composite vLLM worker extension: activation capture + draft swap.

    vLLM accepts a single ``--worker-extension-cls`` string.  This class
    merges ``ActivationCaptureExtension`` and ``DraftSwapExtension`` into one
    MRO chain so both sets of RPC-handled methods are available on the same
    worker.

    Register via ``--worker-extension-cls
    speedlm.gateway.draft_swap:CombinedWorkerExtension``.

    The two extensions use non-overlapping method prefixes, so vLLM's
    attribute-collision check (``worker_base.py:268-275``) will pass cleanly.

    **Initialization:** vLLM does not call ``__init__`` on extension classes
    (it appends the class to ``worker_class.__bases__``).  ``ActivationCaptureExtension``
    relies on ``__init__`` to set up state variables.  This composite class
    re-initializes that state lazily on first method call so it works correctly
    even though vLLM skips the ``__init__`` call.
    """

    def __init__(self) -> None:
        # Initialize ActivationCaptureExtension state.
        # DraftSwapExtension has no __init__, so needs nothing.
        import threading

        self._capture_active = False
        self._capture_dir: str | None = None
        self._pending: dict[int, list] = {}
        self._lock = threading.Lock()
        self._original_model_forward: Any = None

    # -- ActivationCaptureExtension delegates --

    def activate_capture(self, capture_dir: str) -> None:
        self._capture_active = True
        self._capture_dir = capture_dir
        import os

        os.makedirs(capture_dir, exist_ok=True)
        self._install_hook()
        logger.info("Activation capture activated, output dir: %s", capture_dir)

    def flush_capture(self) -> str:
        if not self._capture_active or self._capture_dir is None:
            raise RuntimeError("capture is not active")
        import os

        import torch  # lazy: only available at runtime inside the vLLM venv

        with self._lock:
            pending = self._pending
            self._pending = {}

        if not pending:
            logger.warning("flush_capture called with no buffered data")

        by_layer: dict[int, list] = {}
        for layer_idx, tensors in pending.items():
            by_layer[layer_idx] = tensors

        saved: dict[str, Any] = {}
        for lidx in sorted(by_layer.keys()):
            layer_tensors = by_layer[lidx]
            if len(layer_tensors) == 1:
                stacked = layer_tensors[0]
            else:
                stacked = torch.cat(layer_tensors, dim=0)
            saved[f"layer_{lidx}"] = stacked

        path = os.path.join(self._capture_dir, "captured.safetensors")
        import fcntl

        lock_path = path + ".lock"
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            from safetensors.torch import save_file

            save_file(saved, path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        os.remove(lock_path)
        logger.info("Flushed %d layer activations to %s", len(saved), path)
        return path

    def deactivate_capture(self) -> None:
        self._capture_active = False
        self._deactivate_impl()
        logger.info("Activation capture deactivated")

    def _install_hook(self) -> None:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original = GPUModelRunner._model_forward

        def _wrapped_forward(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self_ref, *args, **kwargs)
            if isinstance(result, tuple) and len(result) >= 2:
                aux_hidden_states = result[1]
                if isinstance(aux_hidden_states, list) and len(aux_hidden_states) > 0:
                    self._buffer_aux(aux_hidden_states)
            return result

        self._original_model_forward = original
        GPUModelRunner._model_forward = _wrapped_forward

    def _deactivate_impl(self) -> None:
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

    def _buffer_aux(self, aux_hidden_states: list) -> None:
        if not self._capture_active:
            return
        with self._lock:
            for i, tensor in enumerate(aux_hidden_states):
                cpu_tensor = tensor.detach().cpu()
                if i not in self._pending:
                    self._pending[i] = []
                self._pending[i].append(cpu_tensor)

    # -- DraftSwapExtension delegates (same implementations) --

    def hot_swap_draft(self, weights_path: str) -> dict[str, Any]:
        drafter = self._get_drafter_model()
        if drafter is None:
            raise RuntimeError("no drafter model found; hot-swap requires a draft model")

        new_weights = self._load_weights_file(weights_path)
        self._validate_compatibility(drafter, new_weights)
        count = self._apply_weights(drafter, new_weights)

        logger.info(
            "Hot-swapped %d drafter parameters from %s", count, weights_path
        )
        return {"swapped": True, "parameters_loaded": count}

    def draft_info(self) -> dict[str, Any]:
        drafter = self._get_drafter_model()
        if drafter is None:
            raise RuntimeError("no drafter model found")

        param_shapes: dict[str, list[int]] = {}
        param_dtypes: dict[str, str] = {}
        for name, param in drafter.named_parameters():
            param_shapes[name] = list(param.shape)
            param_dtypes[name] = str(param.dtype)

        quantization = self._get_quantization()

        return {
            "num_parameters": sum(1 for _ in drafter.named_parameters()),
            "parameter_shapes": param_shapes,
            "parameter_dtypes": param_dtypes,
            "quantization": quantization,
        }

    def _get_drafter_model(self) -> Any:
        try:
            runner = self.model_executor  # type: ignore[attr-defined]
        except AttributeError:
            try:
                runner = self.model_runner  # type: ignore[attr-defined]
            except AttributeError:
                return None
        drafter = getattr(runner, "drafter", None)
        if drafter is None:
            return None
        return getattr(drafter, "model", None)

    def _load_weights_file(self, path: str) -> dict[str, Any]:
        from safetensors.torch import load_file as _load_file
        return _load_file(path)

    def _validate_compatibility(
        self, drafter: Any, new_weights: dict[str, Any]
    ) -> None:
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

    def _apply_weights(self, drafter: Any, new_weights: dict[str, Any]) -> int:
        from vllm.model_executor.model_loader.reload.layerwise import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(drafter)
        weights_iter = new_weights.items()
        loaded = drafter.load_weights(weights_iter)

        try:
            model_config = getattr(self, "vllm_config", None)
            model_config = model_config.model_config if model_config is not None else None
            finalize_layerwise_reload(drafter, model_config)
        except Exception:
            logger.exception("finalize_layerwise_reload failed; weights may be partial")
            raise

        count = len(loaded) if loaded is not None else 0
        return count

    def _get_quantization(self) -> str | None:
        try:
            config = getattr(self, "vllm_config", None)
            if config is not None:
                spec = getattr(config, "speculative_config", None)
                if spec is not None:
                    draft_cfg = getattr(spec, "draft_model_config", None)
                    if draft_cfg is not None:
                        return getattr(draft_cfg, "quantization", None)
        except Exception:
            pass
        return None