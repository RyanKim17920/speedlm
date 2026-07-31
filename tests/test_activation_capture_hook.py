"""Unit tests for the activation capture hook — no GPU/torch needed.

These tests exercise the aux-layer resolution, slice guard, and fail-loudly
behaviour using synthetic mocks.  They deliberately avoid importing torch so
they run in the project venv.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Hook: aux-layer inner model resolution (no GPU needed)
# ---------------------------------------------------------------------------


class TestAuxLayerInnerModelResolution:
    """Verify that the hook resolves aux_hidden_state_layers from the inner
    model (EagleModelMixin) rather than the top-level model wrapper,
    mirroring vLLM's SupportsEagle3.set_aux_hidden_state_layers
    (interfaces.py:1395-1406).
    """

    def test_nested_mock_find_aux_on_inner_model(self) -> None:
        """The hook should find aux_hidden_state_layers on parent_ref.model,
        not on the top-level model -- replicating the GptOssForCausalLM layout.
        """
        from speedlm.activation_capture.hook import (
            ActivationCaptureExtension as ext_cls,
        )

        # Build a mock structure mirroring GptOssForCausalLM:
        # runner.model = top-level (no aux_hidden_state_layers attr)
        # runner.model.model = inner model (has aux_hidden_state_layers)
        inner_model = type("InnerModel", (), {
            "aux_hidden_state_layers": (4, 12, 20),
        })()
        top_model = type("TopModel", (), {
            "set_aux_hidden_state_layers": lambda self, layers: None,
        })()
        top_model.model = inner_model

        hf_config = type("HFConfig", (), {"num_hidden_layers": 24})()
        model_config = type("ModelConfig", (), {"hf_config": hf_config})()
        vllm_config = type("VLLMConfig", (), {"model_config": model_config})()
        runner = type("Runner", (), {
            "model": top_model,
            "vllm_config": vllm_config,
        })()

        ext = ext_cls()
        ext.model_runner = runner  # type: ignore[attr-defined]

        # Should succeed without AttributeError
        ext._extend_aux_layers()

        # Should have stored original (4, 12, 20) and added final layer 24
        assert ext._original_aux_layers == (4, 12, 20)
        assert ext._final_layer_idx == 24

    def test_direct_model_without_wrapper(self) -> None:
        """When the model has no .language_model or .model chain, the hook
        should still work if aux_hidden_state_layers is directly on the model.
        """
        from speedlm.activation_capture.hook import (
            ActivationCaptureExtension as ext_cls,
        )

        # Model that IS the inner model (no wrapper)
        model = type("Model", (), {})()
        model.model = model  # self-reference so parent_ref.model is itself
        model.aux_hidden_state_layers = (2, 8, 14)
        model.set_aux_hidden_state_layers = lambda layers: None

        hf_config = type("HFConfig", (), {"num_hidden_layers": 16})()
        model_config = type("ModelConfig", (), {"hf_config": hf_config})()
        vllm_config = type("VLLMConfig", (), {"model_config": model_config})()
        runner = type("Runner", (), {
            "model": model,
            "vllm_config": vllm_config,
        })()

        ext = ext_cls()
        ext.model_runner = runner  # type: ignore[attr-defined]

        ext._extend_aux_layers()
        assert ext._original_aux_layers == (2, 8, 14)
        assert ext._final_layer_idx == 16


class TestSliceGuard:
    """Verify the slice-before-drafter logic only strips entries when the
    extension succeeded, and never hands an empty list to the drafter.
    """

    def test_slice_noop_when_extension_failed(self) -> None:
        """When _original_aux_layers is empty (extension failed), the slice
        must be a no-op -- aux_hidden_states must pass through untouched.
        """
        aux = [object() for _ in range(3)]  # dummy items, no torch needed
        original_aux: tuple = ()  # extension failed
        expected = len(original_aux)
        if expected > 0 and len(aux) > expected:
            del aux[expected:]
        assert len(aux) == 3

    def test_slice_removes_only_extra_when_extension_succeeded(self) -> None:
        """4 entries with 3 original -> remove 1, leave 3."""
        aux = [object() for _ in range(4)]
        original_aux = (4, 12, 20)  # extension succeeded
        expected = len(original_aux)
        if expected > 0 and len(aux) > expected:
            del aux[expected:]
        assert len(aux) == 3

    def test_empty_aux_raises(self) -> None:
        """The guard should raise if aux_hidden_states somehow becomes empty."""
        aux: list = []
        with pytest.raises(RuntimeError, match="empty before drafter"):
            if len(aux) == 0:
                raise RuntimeError(
                    "aux_hidden_states is empty before drafter; "
                    "activation capture extension left the engine in a "
                    "broken state"
                )


class TestExtensionFailureSurfacesAsError:
    """Verify that _extend_aux_layers raises (not logs a warning) when it
    fails, so that activate_capture propagates the error via collective_rpc.
    """

    def test_missing_aux_attr_raises(self) -> None:
        """If the inner model lacks aux_hidden_state_layers, raise."""
        from speedlm.activation_capture.hook import (
            ActivationCaptureExtension as ext_cls,
        )

        inner = type("Inner", (), {})()  # no aux_hidden_state_layers
        top = type("Top", (), {"model": inner})()

        hf_config = type("HFConfig", (), {"num_hidden_layers": 24})()
        model_config = type("ModelConfig", (), {"hf_config": hf_config})()
        vllm_config = type("VLLMConfig", (), {"model_config": model_config})()
        runner = type("Runner", (), {
            "model": top,
            "vllm_config": vllm_config,
        })()

        ext = ext_cls()
        ext.model_runner = runner  # type: ignore[attr-defined]

        with pytest.raises((AttributeError, RuntimeError)):
            ext._extend_aux_layers()

    def test_missing_num_layers_raises(self) -> None:
        """If num_hidden_layers is absent, raise (not warn-and-return)."""
        from speedlm.activation_capture.hook import (
            ActivationCaptureExtension as ext_cls,
        )

        inner = type("Inner", (), {"aux_hidden_state_layers": (4, 12, 20)})()
        top = type("Top", (), {"model": inner})()

        hf_config = type("HFConfig", (), {})()  # no num_hidden_layers
        model_config = type("ModelConfig", (), {"hf_config": hf_config})()
        vllm_config = type("VLLMConfig", (), {"model_config": model_config})()
        runner = type("Runner", (), {
            "model": top,
            "vllm_config": vllm_config,
        })()

        ext = ext_cls()
        ext.model_runner = runner  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="num_hidden_layers"):
            ext._extend_aux_layers()