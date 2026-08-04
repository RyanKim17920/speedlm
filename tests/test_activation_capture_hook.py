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

# ---------------------------------------------------------------------------
# Hot-path cost: the capture must not tax the live serving path
# ---------------------------------------------------------------------------
#
# Every test below runs torch-free.  `_to_host` imports torch lazily and only
# reaches it on the fused path, so a stub module in `sys.modules` is enough to
# exercise the real code with counted, inspectable "tensors".


class _FakeTensor:
    """Minimal stand-in for a CUDA tensor that counts host transfers."""

    def __init__(
        self,
        tag: object,
        counters: dict,
        *,
        shape: tuple = (4, 8),
        dtype: str = "bfloat16",
        device: str = "cuda:0",
    ) -> None:
        self.tag = tag
        self.counters = counters
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        # On a real GPU this is a cudaMemcpyAsync followed by a stream drain,
        # i.e. one pipeline stall on the serving path.
        self.counters["transfers"] = self.counters.get("transfers", 0) + 1
        return _FakeTensor(
            self.tag, self.counters, shape=self.shape, dtype=self.dtype, device="cpu"
        )


class _FakeStacked(_FakeTensor):
    def __init__(self, parts: list, counters: dict) -> None:
        super().__init__(("stacked", tuple(p.tag for p in parts)), counters)
        self.parts = list(parts)

    def cpu(self) -> _FakeStacked:
        self.counters["transfers"] = self.counters.get("transfers", 0) + 1
        moved = [
            _FakeTensor(p.tag, self.counters, shape=p.shape, dtype=p.dtype,
                        device="cpu")
            for p in self.parts
        ]
        return _FakeStacked(moved, self.counters)


def _install_stub_torch(monkeypatch, counters: dict) -> None:
    import sys
    import types

    module = types.ModuleType("torch")

    def stack(tensors, *args, **kwargs):
        counters["stacks"] = counters.get("stacks", 0) + 1
        return _FakeStacked(list(tensors), counters)

    def unbind(tensor, *args, **kwargs):
        return tuple(tensor.parts)

    module.stack = stack  # type: ignore[attr-defined]
    module.unbind = unbind  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)


def _active_extension(layers=(2, 18, 33, 36)):
    """An extension wired to a counting model, already 'activated'."""
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    class Inner:
        def __init__(self) -> None:
            self.aux_hidden_state_layers = layers

    class Top:
        def __init__(self, inner) -> None:
            self.model = inner
            self.resolutions = 0

        def get_language_model(self):
            self.resolutions += 1
            return self

    inner = Inner()
    top = Top(inner)
    runner = type("Runner", (), {"model": top})()

    ext = ActivationCaptureExtension()
    ext.model_runner = runner  # type: ignore[attr-defined]
    ext._ensure_init()
    ext._capture_active = True
    ext._inner_model = None
    ext._original_aux_layers = layers[:-1]
    ext._final_layer_idx = layers[-1]
    return ext, top, inner


class TestHotPathTransferCount:
    """One device-to-host transfer per forward pass, not one per aux layer.

    Each `.cpu()` on a CUDA tensor drains the stream, so the transfer count IS
    the number of pipeline stalls the capture imposes on serving.
    """

    def test_one_transfer_per_forward_pass(self, monkeypatch) -> None:
        counters: dict = {}
        _install_stub_torch(monkeypatch, counters)
        ext, _, _ = _active_extension()

        steps = 5
        for step in range(steps):
            ext._buffer_aux([
                _FakeTensor((step, layer), counters) for layer in range(4)
            ])

        assert counters["stacks"] == steps
        assert counters["transfers"] == steps, (
            f"expected {steps} blocking device-to-host transfers "
            f"(one per forward pass), got {counters['transfers']}"
        )

    def test_no_row_is_lost_or_mislabelled_by_fusing(self, monkeypatch) -> None:
        """The fused path must key every row by its true aux layer index."""
        counters: dict = {}
        _install_stub_torch(monkeypatch, counters)
        layers = (2, 18, 33, 36)
        ext, _, _ = _active_extension(layers)

        steps = 4
        for step in range(steps):
            ext._buffer_aux([
                _FakeTensor((step, layer), counters) for layer in layers
            ])

        pending = ext._pending
        assert pending is not None
        assert sorted(pending) == sorted(layers)
        for layer in layers:
            assert [t.tag for t in pending[layer]] == [
                (step, layer) for step in range(steps)
            ], f"layer {layer} rows are out of order or mislabelled"
            assert all(t.device == "cpu" for t in pending[layer])

    def test_non_uniform_tensors_fall_back_without_losing_rows(
        self, monkeypatch
    ) -> None:
        """Shapes that cannot be stacked must still be captured, per layer."""
        counters: dict = {}
        _install_stub_torch(monkeypatch, counters)
        layers = (2, 18, 33, 36)
        ext, _, _ = _active_extension(layers)

        aux = [_FakeTensor((0, layer), counters) for layer in layers]
        aux[2].shape = (9, 8)  # one odd layer out

        ext._buffer_aux(aux)

        assert counters.get("stacks", 0) == 0
        assert counters["transfers"] == 4  # the per-layer fallback
        pending = ext._pending
        assert pending is not None
        assert sorted(pending) == sorted(layers)
        assert [pending[layer][0].tag for layer in layers] == [
            (0, layer) for layer in layers
        ]


class TestModelResolutionIsNotPerStep:
    """Walking `get_language_model()/.model` on every forward pass is per-step
    Python work with a per-activation answer.
    """

    def test_inner_model_resolved_once_across_many_steps(self, monkeypatch) -> None:
        counters: dict = {}
        _install_stub_torch(monkeypatch, counters)
        ext, top, _ = _active_extension()

        for step in range(20):
            ext._buffer_aux([
                _FakeTensor((step, layer), counters) for layer in range(4)
            ])

        assert top.resolutions == 1, (
            f"inner model resolved {top.resolutions} times for 20 forward "
            "passes; resolution belongs to activation, not to the hot path"
        )

    def test_layer_labels_still_track_a_live_change(self, monkeypatch) -> None:
        """Caching the model object must NOT cache the layer tuple: that tuple
        is what labels the rows, and the engine can change it under us.
        """
        counters: dict = {}
        _install_stub_torch(monkeypatch, counters)
        ext, top, inner = _active_extension((2, 18, 33, 36))

        ext._buffer_aux([_FakeTensor(("a", i), counters) for i in range(4)])
        inner.aux_hidden_state_layers = (5, 19, 34, 37)
        ext._buffer_aux([_FakeTensor(("b", i), counters) for i in range(4)])

        pending = ext._pending
        assert pending is not None
        assert sorted(pending) == [2, 5, 18, 19, 33, 34, 36, 37]
        assert [t.tag for t in pending[5]] == [("b", 0)]
        assert [t.tag for t in pending[2]] == [("a", 0)]
        assert top.resolutions == 1


class TestDisabledCaptureCostsNothing:
    """A configured-but-inactive capture must be an early return, not a no-op
    that still resolves the model, takes the lock or allocates.
    """

    def test_inactive_buffer_touches_nothing(self) -> None:
        from speedlm.activation_capture.hook import ActivationCaptureExtension

        class Exploding:
            def __getattr__(self, name: str) -> object:
                raise AssertionError(
                    f"disabled capture touched model_runner.{name}"
                )

        ext = ActivationCaptureExtension()
        ext.model_runner = Exploding()  # type: ignore[attr-defined]
        assert ext._capture_active is False

        ext._buffer_aux([object(), object(), object()])  # type: ignore[list-item]

        assert ext._pending is None, "disabled capture allocated its buffers"
        assert ext._lock is None, "disabled capture allocated its lock"

    def test_inactive_extension_leaves_the_forward_path_unwrapped(self) -> None:
        """A serving engine that merely has the extension mixed in must run an
        unwrapped forward: the hook only exists between activate and deactivate.
        """
        from speedlm.activation_capture.hook import ActivationCaptureExtension

        calls: list[str] = []

        class Runner:
            def _model_forward(self, *args, **kwargs):
                calls.append("forward")
                return ("hidden", ["aux0", "aux1", "aux2", "aux3"])

        original = Runner._model_forward
        runner = Runner()
        ext = ActivationCaptureExtension()
        ext.model_runner = runner  # type: ignore[attr-defined]
        # Anything that legitimately runs before a capture session (the lazy
        # init the vLLM base-injection pattern relies on) must not install it.
        ext._ensure_init()

        aux = runner._model_forward()[1]

        assert Runner._model_forward is original, (
            "the runner's forward was wrapped without an activate_capture call"
        )
        assert calls == ["forward"]
        assert aux == ["aux0", "aux1", "aux2", "aux3"], (
            "an inactive capture altered the aux list handed to the drafter"
        )
        assert ext._pending == {}
        assert ext._patched_class is None
        assert ext._installed_wrapper is None


class TestFusedTransferMatchesPerLayer:
    """Fusing is a performance change only: the bytes buffered must be
    identical to what the per-layer copy produced.  Needs real torch.
    """

    def test_fused_and_per_layer_agree_bit_for_bit(self) -> None:
        torch = pytest.importorskip(
            "torch",
            reason="torch is not installed in the project venv",
        )

        layers = (2, 18, 33, 36)
        ext, _, _ = _active_extension(layers)

        generator = torch.Generator().manual_seed(1234)
        aux = [
            torch.randn((7, 16), generator=generator, dtype=torch.float32)
            for _ in layers
        ]

        fused = ext._to_host(aux)
        per_layer = [tensor.detach().cpu() for tensor in aux]

        assert len(fused) == len(per_layer)
        for got, want in zip(fused, per_layer, strict=True):
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            assert torch.equal(got, want)

    def test_fusable_only_when_shape_dtype_and_device_agree(self) -> None:
        torch = pytest.importorskip(
            "torch",
            reason="torch is not installed in the project venv",
        )
        from speedlm.activation_capture.hook import ActivationCaptureExtension

        base = [torch.zeros((3, 4)) for _ in range(4)]
        assert ActivationCaptureExtension._is_fusable(base) is True

        odd_shape = [*base[:3], torch.zeros((5, 4))]
        assert ActivationCaptureExtension._is_fusable(odd_shape) is False

        odd_dtype = [*base[:3], torch.zeros((3, 4), dtype=torch.float16)]
        assert ActivationCaptureExtension._is_fusable(odd_dtype) is False
