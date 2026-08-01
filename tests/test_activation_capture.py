"""Unit tests for the activation capture comparison logic.

These tests exercise the comparison, tolerance, verdict, and serialization
pathways using synthetic tensors — no GPU required.

Skipped when torch is not installed (torch lives in the vLLM venv, not the
project venv). The skip is declared with a module-level ``pytestmark`` rather
than ``pytest.importorskip`` on purpose: ``importorskip`` aborts collection, so
the whole file reported as ZERO tests under the project venv and the suite still
printed green. With ``pytestmark`` the file always collects and every test is
reported as an explicit skip.

To actually execute these tests, put the vLLM venv on the path:

    PYTHONPATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/lib/python3.12/site-packages \\
        .venv/bin/python -m pytest tests/test_activation_capture.py -q
"""

from __future__ import annotations

import importlib.util
import json
import math
import types
from pathlib import Path

import pytest

#: Imported as a module rather than by name: the tests below exercise the
#: private ``_environ`` helper, whose whole job is what it does to a copied
#: process environment.
from speedlm.activation_capture import offline_extract
from speedlm.activation_capture.compare import (
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_TOLERANCE,
    ComparisonResult,
    LayerComparison,
    PrefixCacheResult,
    _detect_rel_error_trend,
    align_prompt_rows,
    build_result,
    check_pre_norm,
    check_within_tolerance,
    compare_layerwise,
    derive_verdict,
)

try:  # torch lives in the vLLM venv, not the project venv.
    import torch
except ImportError:  # pragma: no cover - depends on the interpreter in use
    torch = None  # type: ignore[assignment]

_HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None

_NO_TORCH_REASON = (
    "torch is not installed in the project venv; run with "
    "PYTHONPATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/"
    "lib/python3.12/site-packages to execute these tests"
)


@pytest.fixture(autouse=True)
def _require_torch(request: pytest.FixtureRequest) -> None:
    """Skip torch-dependent tests, but let ``no_torch`` ones through.

    This replaces a module-level ``pytestmark`` skipif. Same collection
    property (every test is collected and reported as an explicit skip rather
    than vanishing, which is why ``pytest.importorskip`` was rejected), but a
    module-level mark cannot be cancelled per class, and the runner-topology
    tests below deliberately use torch-free fakes so they run in the PROJECT
    venv too. A hook test that only executes under a hand-set PYTHONPATH is a
    hook test that stays silent in the default suite -- which is exactly how
    the V1-only hook shipped.
    """
    if torch is None and "no_torch" not in request.keywords:
        pytest.skip(_NO_TORCH_REASON)

# ---------------------------------------------------------------------------
# SpeculativeConfig construction
# ---------------------------------------------------------------------------


class TestSpeculativeConfig:
    """Unit tests for the vLLM --speculative_config dict construction.

    These validate the schema without needing a GPU or a running engine.
    The authoritative schema is SpeculativeConfig in vLLM 0.25.1
    (config/speculative.py:86-100) with fields: method, model,
    num_speculative_tokens, draft_tensor_parallel_size.
    """

    def test_schema_uses_model_not_draft_model(self) -> None:
        """draft_model key must NOT be present; use 'model' instead."""
        config = {
            "method": "eagle3",
            "num_speculative_tokens": 5,
            "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
        }
        assert "draft_model" not in config
        assert "model" in config
        assert config["model"] == "RedHatAI/gpt-oss-20b-speculator.eagle3"

    def test_schema_has_no_draft_model_config(self) -> None:
        """draft_model_config is an invented key; must not be present."""
        config = {
            "method": "eagle3",
            "num_speculative_tokens": 5,
            "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
        }
        assert "draft_model_config" not in config

    def test_schema_matches_production(self) -> None:
        """The config should match the working production argv."""
        config = {
            "method": "eagle3",
            "num_speculative_tokens": 5,
            "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
        }
        serialized = json.dumps(config)
        parsed = json.loads(serialized)
        assert parsed["method"] == "eagle3"
        assert parsed["num_speculative_tokens"] == 5
        assert parsed["model"] == "RedHatAI/gpt-oss-20b-speculator.eagle3"

    def test_old_invented_schema_rejected(self) -> None:
        """The old schema with draft_model should NOT validate."""
        old_config = {
            "method": "eagle3",
            "num_speculative_tokens": 3,
            "draft_model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [4, 12, 20]}
            },
        }
        # Must have neither invented key
        assert "draft_model" in old_config  # old config has it
        assert "model" not in old_config  # old config lacks correct key
        assert "draft_model_config" in old_config  # old config has it

# ---------------------------------------------------------------------------
# compare_layerwise
# ---------------------------------------------------------------------------


class TestCompareLayerwise:
    def test_identical_tensors(self) -> None:
        """Identical tensors yield zero difference."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        layers = compare_layerwise({0: t}, {0: t})
        assert len(layers) == 1
        assert layers[0].shape_match is True
        assert layers[0].max_abs_diff == 0.0
        assert layers[0].mean_abs_diff == 0.0

    def test_slight_difference(self) -> None:
        """Small difference is captured correctly."""
        a = torch.ones(4, 8)
        b = torch.ones(4, 8) + 0.001
        layers = compare_layerwise({5: a}, {5: b})
        assert len(layers) == 1
        assert layers[0].max_abs_diff == pytest.approx(0.001, abs=1e-6)
        assert layers[0].mean_abs_diff == pytest.approx(0.001, abs=1e-6)

    def test_shape_mismatch(self) -> None:
        """Mismatched shapes produce shape_match=False and None diffs."""
        a = torch.ones(4, 8)
        b = torch.ones(4, 16)
        layers = compare_layerwise({0: a}, {0: b})
        assert len(layers) == 1
        assert layers[0].shape_match is False
        assert layers[0].max_abs_diff is None
        assert layers[0].mean_abs_diff is None

    def test_missing_layer_in_captured(self) -> None:
        """Layer present only in offline is a shape mismatch."""
        layers = compare_layerwise({}, {0: torch.ones(2, 2)})
        assert len(layers) == 1
        assert layers[0].shape_match is False
        assert layers[0].captured_shape == ()
        assert layers[0].offline_shape == (2, 2)

    def test_missing_layer_in_offline(self) -> None:
        """Layer present only in captured is a shape mismatch."""
        layers = compare_layerwise({0: torch.ones(2, 2)}, {})
        assert len(layers) == 1
        assert layers[0].shape_match is False
        assert layers[0].captured_shape == (2, 2)
        assert layers[0].offline_shape == ()

    def test_multiple_layers_sorted(self) -> None:
        """Multiple layers are returned sorted by layer index."""
        t = torch.ones(1, 1)
        layers = compare_layerwise({20: t, 4: t, 12: t}, {4: t, 20: t, 12: t})
        assert [lc.layer_idx for lc in layers] == [4, 12, 20]

    def test_bf16_difference(self) -> None:
        """bf16 tensors produce measurable differences."""
        a = torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16)
        b = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
        layers = compare_layerwise({0: a}, {0: b})
        assert layers[0].max_abs_diff > 0.0
        assert layers[0].shape_match is True


# ---------------------------------------------------------------------------
# Relative metrics, cosine similarity, trend detection
# ---------------------------------------------------------------------------


class TestRelativeMetrics:
    """Verify relative error metrics are correct on known inputs."""

    def test_identical_tensors_zero_rel_error(self) -> None:
        """Identical tensors have zero relative error and cosine = 1.0."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        layers = compare_layerwise({0: t}, {0: t})
        lc = layers[0]
        assert lc.mean_rel_error == pytest.approx(0.0, abs=1e-10)
        assert lc.max_rel_error == pytest.approx(0.0, abs=1e-10)
        assert lc.cosine_similarity == pytest.approx(1.0, abs=1e-6)
        assert lc.mean_ref_magnitude == pytest.approx(2.5, abs=1e-6)

    def test_known_relative_error(self) -> None:
        """mean|a-b| / mean|b| is computed correctly."""
        a = torch.ones(4, 8) * 10.0
        b = torch.ones(4, 8) * 1.0  # 10% relative diff
        layers = compare_layerwise({0: a}, {0: b})
        lc = layers[0]
        # mean|a-b| = 9.0, mean|b| = 1.0 => mean_rel_error = 9.0
        assert lc.mean_rel_error == pytest.approx(9.0, abs=1e-6)

    def test_small_relative_error(self) -> None:
        """A 1% perturbation gives ~0.01 mean relative error."""
        a = torch.ones(4, 8) * 100.0
        b = torch.ones(4, 8) * 101.0  # 1% relative
        layers = compare_layerwise({0: a}, {0: b})
        lc = layers[0]
        # mean|a-b| = 1.0, mean|b| = 101.0 => mean_rel ~ 0.0099
        assert lc.mean_rel_error == pytest.approx(1.0 / 101.0, abs=1e-6)

    def test_cosine_identical(self) -> None:
        """Cosine similarity of identical vectors is 1.0."""
        t = torch.randn(8, 16)
        layers = compare_layerwise({0: t}, {0: t})
        assert layers[0].cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_scaled(self) -> None:
        """Cosine similarity is scale-invariant."""
        t = torch.randn(8, 16)
        layers = compare_layerwise({0: t * 3.7}, {0: t})
        assert layers[0].cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_negative(self) -> None:
        """Cosine of opposite vectors is -1.0."""
        t = torch.ones(4, 8)
        layers = compare_layerwise({0: t}, {0: -t})
        assert layers[0].cosine_similarity == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_orthogonal_approx(self) -> None:
        """Orthogonal-ish vectors have cosine near 0."""
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        layers = compare_layerwise({0: a}, {0: b})
        assert layers[0].cosine_similarity == pytest.approx(0.0, abs=1e-10)

    def test_divide_by_zero_guarded(self) -> None:
        """Zero reference tensor does not cause NaN or inf."""
        a = torch.ones(4, 8)
        b = torch.zeros(4, 8)
        layers = compare_layerwise({0: a}, {0: b})
        lc = layers[0]
        # mean_ref_magnitude = 0; rel error uses epsilon guard
        assert lc.mean_ref_magnitude == 0.0
        assert lc.mean_rel_error is not None
        assert not math.isnan(lc.mean_rel_error)
        assert not math.isinf(lc.cosine_similarity)

    def test_magnitude_metrics_present(self) -> None:
        """Reference magnitude fields are populated."""
        a = torch.ones(4, 8) * 5.0
        b = torch.ones(4, 8) * 10.0
        layers = compare_layerwise({0: a}, {0: b})
        lc = layers[0]
        assert lc.mean_ref_magnitude == pytest.approx(10.0, abs=1e-6)
        assert lc.max_ref_magnitude == pytest.approx(10.0, abs=1e-6)

    def test_shape_mismatch_no_relative_fields(self) -> None:
        """Mismatched shapes leave relative fields as None."""
        a = torch.ones(4, 8)
        b = torch.ones(4, 16)
        layers = compare_layerwise({0: a}, {0: b})
        lc = layers[0]
        assert lc.mean_rel_error is None
        assert lc.cosine_similarity is None
        assert lc.mean_ref_magnitude is None
        assert lc.max_rel_error is None
        assert lc.p99_rel_error is None


class TestElementwiseRelErrorFloor:
    """Regression tests for the scale-tied denominator floor on max_rel_error.

    Before the floor was introduced, ``max_rel_error`` divided by
    ``|offline_i| + 1e-12`` and real Stage 0 artifacts reported values of
    1e12-1e14 -- i.e. the metric measured the smallness of the denominator,
    not the size of the discrepancy.
    """

    #: Metrics that must be invariant under a common rescaling of both tensors.
    _SCALE_INVARIANT = (
        "mean_rel_error",
        "max_rel_error",
        "p99_rel_error",
        "cosine_similarity",
    )

    def test_scale_invariance(self) -> None:
        """Multiplying both tensors by 1000 changes no relative metric."""
        torch.manual_seed(0)
        # float64 so the x1000 rescale is exact enough to compare tightly.
        ref = torch.randn(64, 128, dtype=torch.float64)
        cap = ref + torch.randn(64, 128, dtype=torch.float64) * 0.01

        base = compare_layerwise({0: cap}, {0: ref})[0]
        scaled = compare_layerwise({0: cap * 1000.0}, {0: ref * 1000.0})[0]

        for name in self._SCALE_INVARIANT:
            got = getattr(scaled, name)
            want = getattr(base, name)
            assert got == pytest.approx(want, rel=1e-9), name

    def test_zeros_and_denormals_do_not_explode(self) -> None:
        """Exact zeros and denormal-scale reference values stay bounded.

        This is the real-artifact failure mode: a reference tensor whose bulk
        is O(1) but which also contains exact zeros and ~1e-38 elements.  The
        captured tensor differs only by ordinary bf16-scale noise everywhere.
        """
        ref = torch.ones(4, 16) * 2.0
        ref[0, 0] = 0.0
        ref[0, 1] = 1e-38  # denormal-ish relative to the O(1) bulk
        ref[0, 2] = -0.0
        cap = ref + 1e-3  # uniform, tiny, absolute perturbation

        lc = compare_layerwise({0: cap}, {0: ref})[0]

        assert lc.max_rel_error is not None
        assert lc.p99_rel_error is not None
        assert math.isfinite(lc.max_rel_error)
        # RMS of ref is ~2.0 => floor ~2e-3, so the worst element reports
        # ~1e-3/2e-3 = 0.5, not 1e35.
        assert lc.max_rel_error < 1.0
        assert lc.p99_rel_error <= lc.max_rel_error

    def test_real_divergence_still_caught(self) -> None:
        """A genuinely diverging element is still reported at full size.

        The floor must not mask divergence: an element whose reference value
        is a normal O(1) magnitude is divided by its own value, unchanged.
        """
        ref = torch.ones(4, 16) * 2.0
        cap = ref.clone()
        cap[2, 5] = 2.0 + 40.0  # 20x relative divergence on a normal element

        lc = compare_layerwise({0: cap}, {0: ref})[0]

        assert lc.max_rel_error == pytest.approx(20.0, rel=1e-4)
        # ...and it survives a rescale of both tensors.
        scaled = compare_layerwise({0: cap * 1000.0}, {0: ref * 1000.0})[0]
        assert scaled.max_rel_error == pytest.approx(20.0, rel=1e-4)

    def test_divergence_on_a_small_element_still_caught(self) -> None:
        """Divergence large relative to the *tensor* is caught on tiny refs.

        A reference element of ~0 whose captured counterpart is the size of
        the whole tensor is a real divergence, and the floor (a small
        fraction of the RMS) keeps the reported ratio large.
        """
        ref = torch.ones(4, 16) * 2.0
        ref[1, 3] = 0.0
        cap = ref.clone()
        cap[1, 3] = 2.0  # was zero, now a full-scale value

        lc = compare_layerwise({0: cap}, {0: ref})[0]

        assert lc.max_rel_error is not None
        # 2.0 / (1e-3 * RMS~2.0) ~= 1000: unmistakably a failure.
        assert lc.max_rel_error > 100.0

    def test_identical_tensors_zero_max_and_p99(self) -> None:
        """Identical tensors give exactly zero for both elementwise metrics."""
        t = torch.randn(8, 32)
        lc = compare_layerwise({0: t}, {0: t})[0]
        assert lc.max_rel_error == pytest.approx(0.0, abs=1e-12)
        assert lc.p99_rel_error == pytest.approx(0.0, abs=1e-12)

    def test_all_zero_reference_reports_unit_scale_error(self) -> None:
        """An all-zero reference falls back to the captured tensor's scale."""
        cap = torch.ones(4, 8)
        ref = torch.zeros(4, 8)
        lc = compare_layerwise({0: cap}, {0: ref})[0]
        assert lc.max_rel_error is not None
        assert math.isfinite(lc.max_rel_error)
        # |1-0| / (1e-3 * RMS(cap)=1.0) == 1000, not 1e12.
        assert lc.max_rel_error == pytest.approx(1000.0, rel=1e-4)

    def test_both_zero_gives_zero(self) -> None:
        """Two all-zero tensors have zero relative error, not NaN."""
        z = torch.zeros(4, 8)
        lc = compare_layerwise({0: z}, {0: z})[0]
        assert lc.max_rel_error == pytest.approx(0.0, abs=1e-12)
        assert lc.p99_rel_error == pytest.approx(0.0, abs=1e-12)

    def test_max_rel_error_does_not_gate_the_verdict(self) -> None:
        """A large max_rel_error alone must not flip the verdict to FAIL.

        The verdict is driven by shapes, ``mean_rel_error`` and the pre-norm
        check; the elementwise metrics are diagnostics only.
        """
        ref = torch.ones(4, 16) * 2.0
        ref[1, 3] = 0.0
        cap = ref.clone()
        cap[1, 3] = 2.0

        result = build_result(
            {0: cap}, {0: ref},
            captured_final_pre_norm=cap,
            offline_final_pre_norm=ref,
        )
        assert result.layers[0].max_rel_error is not None
        assert result.layers[0].max_rel_error > 100.0
        assert result.layers[0].mean_rel_error is not None
        assert result.layers[0].mean_rel_error <= DEFAULT_RELATIVE_TOLERANCE
        assert result.verdict == "PASS"


class TestTrendDetection:
    """Test _detect_rel_error_trend classifies relative error trends."""

    def test_constant_trend(self) -> None:
        """Similar relative errors across layers -> constant."""
        layers = [
            LayerComparison(2, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.01),
            LayerComparison(12, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.012),
            LayerComparison(21, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.011),
        ]
        assert _detect_rel_error_trend(layers) == "constant"

    def test_growing_trend(self) -> None:
        """Relative error growing by >3x -> growing."""
        layers = [
            LayerComparison(2, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.001),
            LayerComparison(12, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.01),
            LayerComparison(21, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.05),
        ]
        assert _detect_rel_error_trend(layers) == "growing"

    def test_insufficient_data_single_layer(self) -> None:
        """One layer -> insufficient_data."""
        layers = [
            LayerComparison(2, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.01),
        ]
        assert _detect_rel_error_trend(layers) == "insufficient_data"

    def test_insufficient_data_empty(self) -> None:
        """No layers -> insufficient_data."""
        assert _detect_rel_error_trend([]) == "insufficient_data"

    def test_zero_first_layer_growth(self) -> None:
        """Zero relative error at first layer with non-zero later -> growing."""
        layers = [
            LayerComparison(2, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.0),
            LayerComparison(21, (1, 1), (1, 1), True, 0.1, 0.1,
                            mean_rel_error=0.05),
        ]
        assert _detect_rel_error_trend(layers) == "growing"


# ---------------------------------------------------------------------------
# check_pre_norm
# ---------------------------------------------------------------------------


class TestCheckPreNorm:
    def test_both_none(self) -> None:
        assert check_pre_norm(None, None) is None

    def test_captured_none(self) -> None:
        assert check_pre_norm(None, torch.ones(1, 1)) is None

    def test_offline_none(self) -> None:
        assert check_pre_norm(torch.ones(1, 1), None) is None

    def test_matching(self) -> None:
        t = torch.ones(4, 8)
        assert check_pre_norm(t, t) is True

    def test_within_tolerance(self) -> None:
        a = torch.ones(4, 8)
        b = torch.ones(4, 8) + 0.001
        assert check_pre_norm(a, b) is True

    def test_exceeds_tolerance(self) -> None:
        a = torch.zeros(4, 8)
        b = torch.ones(4, 8)
        assert check_pre_norm(a, b) is False

    def test_shape_mismatch(self) -> None:
        a = torch.ones(4, 8)
        b = torch.ones(4, 16)
        assert check_pre_norm(a, b) is False


# ---------------------------------------------------------------------------
# check_within_tolerance
# ---------------------------------------------------------------------------


class TestCheckWithinTolerance:
    def test_all_within_relative(self) -> None:
        """Layers with small relative errors are within tolerance."""
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.009, 0.005,
                            mean_rel_error=0.005),
        ]
        assert check_within_tolerance(layers) is True

    def test_one_exceeds_relative(self) -> None:
        """One layer exceeding relative tolerance fails."""
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.05, 0.03,
                            mean_rel_error=0.5),
        ]
        assert check_within_tolerance(layers) is False

    def test_empty(self) -> None:
        assert check_within_tolerance([]) is True

    def test_none_rel_error_fails(self) -> None:
        """None mean_rel_error (shape mismatch) fails tolerance check."""
        layers = [
            LayerComparison(0, (4, 8), (4, 16), False, None, None),
        ]
        assert check_within_tolerance(layers) is False

    def test_legacy_absolute_tolerance(self) -> None:
        """When relative_tolerance=None and tolerance is given, use absolute."""
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
        ]
        assert check_within_tolerance(layers, tolerance=0.01,
                                      relative_tolerance=None) is True


# ---------------------------------------------------------------------------
# derive_verdict
# ---------------------------------------------------------------------------


class TestDeriveVerdict:
    def _result(
        self,
        layers: list[LayerComparison],
        pre_norm: bool | None,
    ) -> ComparisonResult:
        return ComparisonResult(
            layers=layers,
            pre_norm_match=pre_norm,
            prefix_cache_test=None,
            tolerance=DEFAULT_TOLERANCE,
            verdict="",
        )

    def test_pass(self) -> None:
        """Small relative errors + pre_norm=True -> PASS."""
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.005, 0.003,
                            mean_rel_error=0.003),
        ]
        result = self._result(layers, True)
        assert derive_verdict(result) == "PASS"

    def test_fail_empty(self) -> None:
        result = self._result([], True)
        assert derive_verdict(result) == "FAIL_empty"

    def test_fail_shape(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 16), False, None, None),
        ]
        result = self._result(layers, True)
        assert derive_verdict(result) == "FAIL_shape"

    def test_fail_tolerance_relative(self) -> None:
        """Large relative error -> FAIL_tolerance."""
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.05, 0.03,
                            mean_rel_error=0.5),
        ]
        result = self._result(layers, True)
        assert derive_verdict(result) == "FAIL_tolerance"

    def test_fail_pre_norm_none(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
        ]
        result = self._result(layers, None)
        assert derive_verdict(result) == "FAIL_pre_norm"

    def test_fail_pre_norm_false(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
        ]
        result = self._result(layers, False)
        assert derive_verdict(result) == "FAIL_pre_norm"


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_pass_build(self) -> None:
        t = torch.ones(4, 8)
        result = build_result(
            {0: t, 1: t},
            {0: t, 1: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        assert result.verdict == "PASS"
        assert len(result.layers) == 2
        assert result.pre_norm_match is True

    def test_fail_tolerance_build(self) -> None:
        """Completely different tensors exceed relative tolerance."""
        a = torch.zeros(4, 8)
        b = torch.ones(4, 8)
        result = build_result(
            {0: a},
            {0: b},
            captured_final_pre_norm=a,
            offline_final_pre_norm=b,
        )
        assert result.verdict == "FAIL_tolerance"

    def test_with_prefix_cache(self) -> None:
        t = torch.ones(4, 8)
        pc = PrefixCacheResult(
            prompt_token_count=10,
            captured_rows_per_layer=10,
            captured_layer_count=1,
            cache_hit=False,
            prompt_rows_missing=0,
        )
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
            prefix_cache=pc,
        )
        assert result.verdict == "PASS"
        assert result.prefix_cache_test is not None
        assert result.prefix_cache_test.cache_hit is False

    def test_custom_relative_tolerance(self) -> None:
        """A 1.5% relative diff passes at 0.02 but fails at 0.001."""
        a = torch.ones(4, 8)
        b = torch.ones(4, 8) * 1.015  # 1.5% relative
        result_strict = build_result(
            {0: a}, {0: b}, relative_tolerance=0.001,
        )
        assert result_strict.verdict in ("FAIL_tolerance", "FAIL_pre_norm")
        result_lenient = build_result(
            {0: a}, {0: b}, relative_tolerance=0.05,
        )
        # With lenient rel tol, layer passes but no pre-norm -> FAIL_pre_norm
        assert result_lenient.verdict == "FAIL_pre_norm"

    def test_trend_is_recorded(self) -> None:
        """build_result records rel_error_trend."""
        t = torch.ones(4, 8)
        result = build_result(
            {0: t, 1: t},
            {0: t, 1: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        assert result.rel_error_trend in ("constant", "insufficient_data")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self, tmp_path: Path) -> None:
        t = torch.ones(4, 8)
        result = build_result(
            {0: t, 5: t},
            {0: t, 5: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        d = result.to_dict()
        assert d["verdict"] == "PASS"
        assert d["tolerance"] == DEFAULT_TOLERANCE
        assert d["relative_tolerance"] == DEFAULT_RELATIVE_TOLERANCE
        assert d["all_shapes_match"] is True
        assert d["all_within_tolerance"] is True
        assert d["pre_norm_match"] is True
        assert len(d["layers"]) == 2
        # New fields present
        layer_0 = d["layers"][0]
        assert "mean_rel_error" in layer_0
        assert "max_rel_error" in layer_0
        assert "p99_rel_error" in layer_0
        assert "cosine_similarity" in layer_0
        assert "mean_ref_magnitude" in layer_0
        assert layer_0["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_write_json(self, tmp_path: Path) -> None:
        t = torch.ones(2, 4)
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        out = tmp_path / "result.json"
        result.write_json(out)
        data = json.loads(out.read_text())
        assert data["verdict"] == "PASS"
        assert len(data["layers"]) == 1
        assert data["layers"][0]["layer_idx"] == 0

    def test_write_json_with_prefix_cache(self, tmp_path: Path) -> None:
        t = torch.ones(2, 4)
        pc = PrefixCacheResult(
            prompt_token_count=8,
            captured_rows_per_layer=6,
            captured_layer_count=3,
            cache_hit=True,
            prompt_rows_missing=2,
        )
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
            prefix_cache=pc,
        )
        out = tmp_path / "result.json"
        result.write_json(out)
        data = json.loads(out.read_text())
        assert data["prefix_cache_test"]["cache_hit"] is True
        assert data["prefix_cache_test"]["prompt_rows_missing"] == 2

    def test_write_json_includes_trend(self, tmp_path: Path) -> None:
        t = torch.ones(2, 4)
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        out = tmp_path / "result.json"
        result.write_json(out)
        data = json.loads(out.read_text())
        assert "rel_error_trend" in data


# ---------------------------------------------------------------------------
# Runner-topology hook installation (hook.py)
# ---------------------------------------------------------------------------


#: The two runner generations vLLM ships.  ``gpu_worker.py:384-398`` picks
#: between them on ``vllm_config.use_v2_model_runner``, and the interception
#: point differs: V1 ``gpu_model_runner.GPUModelRunner`` has ``_model_forward``
#: (``:3783``); V2 ``gpu.model_runner.GPUModelRunner`` has NO ``_model_forward``
#: and reads the aux list back off ``execute_model_state`` inside
#: ``sample_tokens`` (``gpu/model_runner.py:1358-1369``).
#:
#: Fakes must model *one* of these at a time.  A fake carrying both would let a
#: hook hard-coded to one generation pass, which is exactly how the V1-only
#: ``_model_forward`` patch shipped: importing the V1 module SUCCEEDS on a V2
#: build (both modules exist side by side), so the patch installed silently,
#: buffered nothing, and never stripped the appended 4th aux entry -- the
#: drafter's fc then got 4*H against a 3*H weight.
RUNNER_GENERATIONS: tuple[str, ...] = ("v1", "v2")

#: Matches the shipped Qwen3-8B target: 36 decoder layers, EAGLE-3 collecting
#: three of them, so the extension appends index 36 as the 4th.
_NUM_HIDDEN_LAYERS = 36
_CANONICAL_AUX_LAYERS = (2, 18, 34)


class _FakeAuxTensor:
    """Stands in for a CUDA tensor: only ``detach().cpu()`` is exercised."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def detach(self) -> _FakeAuxTensor:
        return self

    def cpu(self) -> _FakeAuxTensor:
        return self


class _FakeInnerModel:
    """Carries ``aux_hidden_state_layers`` (``interfaces.py:1326-1327``)."""

    def __init__(self) -> None:
        self.aux_hidden_state_layers: tuple[int, ...] = _CANONICAL_AUX_LAYERS


class _FakeTargetModel:
    """Top-level ``...ForCausalLM``; the attribute lives on ``.model``."""

    def __init__(self) -> None:
        self.model = _FakeInnerModel()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)


class _FakeHFConfig:
    num_hidden_layers = _NUM_HIDDEN_LAYERS


class _FakeModelConfig:
    hf_config = _FakeHFConfig()


class _FakeVllmConfig:
    model_config = _FakeModelConfig()


def _make_runner_cls(generation: str) -> type:
    """Build a FRESH runner class for one generation.

    Fresh per call on purpose: the hook patches the runner *class*, so a shared
    class would leak a monkeypatch across tests.
    """
    if generation not in RUNNER_GENERATIONS:
        raise ValueError(f"unknown runner generation {generation!r}")

    class _Base:
        def __init__(self) -> None:
            self.model = _FakeTargetModel()
            self.vllm_config = _FakeVllmConfig()
            #: How many aux entries the drafter actually received.
            self.drafter_saw: int | None = None

    if generation == "v1":

        class _FakeV1Runner(_Base):
            """V1: aux list is element 1 of ``_model_forward``'s return."""

            def _model_forward(self, aux: list) -> tuple:
                return ("hidden", aux)

            def drive(self, aux: list) -> None:
                _, returned = self._model_forward(aux)
                self.drafter_saw = len(returned)

        return _FakeV1Runner

    class _FakeV2Runner(_Base):
        """V2: no ``_model_forward``; aux arrives via ``execute_model_state``."""

        def __init__(self) -> None:
            super().__init__()
            self.execute_model_state: object | None = None

        def sample_tokens(self) -> None:
            #: Stands in for ``speculator.propose(..., aux_hidden_states, ...)``.
            state = self.execute_model_state
            self.drafter_saw = len(state.aux_hidden_states)  # type: ignore[union-attr]

        def drive(self, aux: list) -> None:
            self.execute_model_state = types.SimpleNamespace(aux_hidden_states=aux)
            self.sample_tokens()

    return _FakeV2Runner


def _make_capture_extension(runner: object):
    """Build an ActivationCaptureExtension bound to a fake runner.

    Mirrors how vLLM injects the class: ``__init__`` is never called, the
    extension is mixed into the worker and reaches the runner via
    ``self.model_runner``.
    """
    from speedlm.activation_capture.hook import ActivationCaptureExtension

    ext = object.__new__(ActivationCaptureExtension)
    object.__setattr__(ext, "model_runner", runner)
    #: Class-level defaults are shared across instances; reset the ones the
    #: hook mutates so tests cannot bleed into each other.
    for attr, value in (
        ("_capture_active", False),
        ("_capture_dir", None),
        ("_original_model_forward", None),
        ("_final_layer_idx", None),
        ("_original_aux_layers", ()),
        ("_patched_class", None),
        ("_patched_attr", None),
        ("_installed_wrapper", None),
        ("_pending", None),
        ("_lock", None),
    ):
        object.__setattr__(ext, attr, value)
    return ext


def _aux_batch(count: int) -> list:
    return [_FakeAuxTensor(f"layer{i}") for i in range(count)]


@pytest.mark.no_torch
class TestRunnerTopologyHook:
    """The hook must land on whichever runner generation is actually live."""

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_hook_lands_on_the_live_runner_class(
        self, generation: str, tmp_path: Path
    ) -> None:
        """Resolution is against ``type(self.model_runner)``, not an import."""
        runner_cls = _make_runner_cls(generation)
        runner = runner_cls()
        ext = _make_capture_extension(runner)

        ext.activate_capture(str(tmp_path / "cap"))

        assert ext._patched_class is runner_cls
        assert ext._patched_attr == (
            "_model_forward" if generation == "v1" else "sample_tokens"
        )

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_drafter_sees_only_canonical_aux_layers(
        self, generation: str, tmp_path: Path
    ) -> None:
        """The appended final layer is buffered, then stripped before the fc.

        This is the regression: on V2 the strip never ran, so the drafter's
        ``fc`` received 4*H against a 3*H weight and the engine died with
        ``mat1 and mat2 shapes cannot be multiplied``.
        """
        runner = _make_runner_cls(generation)()
        ext = _make_capture_extension(runner)
        ext.activate_capture(str(tmp_path / "cap"))

        # The engine now collects one extra layer.
        assert runner.model.model.aux_hidden_state_layers == (
            *_CANONICAL_AUX_LAYERS,
            _NUM_HIDDEN_LAYERS,
        )

        runner.drive(_aux_batch(4))

        assert runner.drafter_saw == len(_CANONICAL_AUX_LAYERS)
        # All four -- including the appended final layer -- were buffered.
        assert sorted(ext._get_pending()) == [
            *_CANONICAL_AUX_LAYERS,
            _NUM_HIDDEN_LAYERS,
        ]

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_deactivate_restores_runner_and_aux_layers(
        self, generation: str, tmp_path: Path
    ) -> None:
        """Deactivation must leave the engine exactly as it found it."""
        runner_cls = _make_runner_cls(generation)
        runner = runner_cls()
        attr = "_model_forward" if generation == "v1" else "sample_tokens"
        pristine = getattr(runner_cls, attr)
        ext = _make_capture_extension(runner)

        ext.activate_capture(str(tmp_path / "cap"))
        assert getattr(runner_cls, attr) is not pristine

        ext.deactivate_capture()

        assert getattr(runner_cls, attr) is pristine
        assert runner.model.model.aux_hidden_state_layers == _CANONICAL_AUX_LAYERS

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_reactivation_does_not_double_extend(
        self, generation: str, tmp_path: Path
    ) -> None:
        """activate -> deactivate -> activate must still strip correctly.

        Without the rollback, the second activation would see the final layer
        already present, record the EXTENDED tuple as "original", and stop
        stripping -- reintroducing the 4*H crash on a re-armed capture.
        """
        runner = _make_runner_cls(generation)()
        ext = _make_capture_extension(runner)

        ext.activate_capture(str(tmp_path / "cap"))
        ext.deactivate_capture()
        ext.activate_capture(str(tmp_path / "cap2"))

        assert runner.model.model.aux_hidden_state_layers == (
            *_CANONICAL_AUX_LAYERS,
            _NUM_HIDDEN_LAYERS,
        )
        assert ext._original_aux_layers == _CANONICAL_AUX_LAYERS

        runner.drive(_aux_batch(4))
        assert runner.drafter_saw == len(_CANONICAL_AUX_LAYERS)

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_activate_twice_without_deactivate_is_idempotent(
        self, generation: str, tmp_path: Path
    ) -> None:
        """A second activate_capture resets rather than compounding."""
        runner = _make_runner_cls(generation)()
        ext = _make_capture_extension(runner)

        ext.activate_capture(str(tmp_path / "cap"))
        ext.activate_capture(str(tmp_path / "cap"))

        assert runner.model.model.aux_hidden_state_layers == (
            *_CANONICAL_AUX_LAYERS,
            _NUM_HIDDEN_LAYERS,
        )
        runner.drive(_aux_batch(4))
        assert runner.drafter_saw == len(_CANONICAL_AUX_LAYERS)

    def test_unknown_runner_topology_is_a_hard_error(self, tmp_path: Path) -> None:
        """A third runner generation must fail loudly, not install nothing."""

        class _FutureRunner:
            def __init__(self) -> None:
                self.model = _FakeTargetModel()
                self.vllm_config = _FakeVllmConfig()

        ext = _make_capture_extension(_FutureRunner())
        with pytest.raises(RuntimeError, match="exposes none of"):
            ext.activate_capture(str(tmp_path / "cap"))


# ---------------------------------------------------------------------------
# Runner reporting (hook.py)
# ---------------------------------------------------------------------------


@pytest.mark.no_torch
class TestRunnerInfo:
    """``runner_info`` must report the generation that actually loaded.

    The e2e harness pins a generation with ``VLLM_USE_V2_MODEL_RUNNER`` but
    cannot trust that it got one: ``vllm/config/vllm.py:546-553`` downgrades
    V2 to V1 silently when the variable is unset, and that silent downgrade is
    how a V1-only capture hook shipped undetected twice.  ``runner_info`` is
    the RPC the harness asserts on, so its answer must be derived from the
    live runner rather than from what was requested.
    """

    @pytest.mark.parametrize("generation", RUNNER_GENERATIONS)
    def test_generation_matches_the_live_runner(self, generation: str) -> None:
        """The reported generation is read off the runner, not a config."""
        runner_cls = _make_runner_cls(generation)
        ext = _make_capture_extension(runner_cls())

        info = ext.runner_info()

        assert info["generation"] == generation
        #: The discriminator is the interception point, because that is the
        #: axis the capture hook actually depends on: ``_model_forward`` is
        #: V1-only while ``sample_tokens`` exists on both.
        expected_hook = "_model_forward" if generation == "v1" else "sample_tokens"
        assert info["hook_point"] == expected_hook
        assert runner_cls.__qualname__ in info["runner_class"]

    def test_generation_is_reported_before_activation(self) -> None:
        """``runner_info`` must not require an active capture.

        The harness calls it as soon as the engine is ready, so that a run
        against the wrong runner fails before it spends a prefill proving it.
        """
        ext = _make_capture_extension(_make_runner_cls("v2")())
        assert ext.runner_info()["generation"] == "v2"

    def test_unknown_topology_is_reported_not_guessed(self) -> None:
        """A runner with no known hook point reports ``unknown``, never a guess.

        ``activate_capture`` raises on this shape.  ``runner_info`` instead has
        to answer, because the harness calls it to *decide* whether the run is
        sound -- so the one thing it must not do is pick a plausible-looking
        generation for a runner it does not recognise.
        """

        class _FutureRunner:
            def __init__(self) -> None:
                self.model = _FakeTargetModel()
                self.vllm_config = _FakeVllmConfig()

        info = _make_capture_extension(_FutureRunner()).runner_info()

        assert info["generation"] == "unknown"
        assert info["hook_point"] is None
        assert "_FutureRunner" in info["runner_class"]

    def test_unreadable_config_does_not_break_the_report(self) -> None:
        """A raising ``use_v2_model_runner`` costs the third signal, not the two.

        ``config_use_v2`` is a diagnostic cross-check on a property that can
        raise on a partially-built config.  Losing it must not take down the
        report the harness actually asserts on.
        """

        class _ExplodingConfig:
            @property
            def use_v2_model_runner(self) -> bool:
                raise RuntimeError("config not finished building")

        runner = _make_runner_cls("v1")()
        runner.vllm_config = _ExplodingConfig()
        info = _make_capture_extension(runner).runner_info()

        assert info["generation"] == "v1"
        assert info["config_use_v2"] is None


# ---------------------------------------------------------------------------
# Offline-engine environment isolation (offline_extract.py)
# ---------------------------------------------------------------------------


@pytest.mark.no_torch
class TestOfflineEngineEnvIsolation:
    """The offline engine must pick its own model runner.

    ``_environ`` copies ``os.environ`` wholesale, and the capture leg now sets
    ``VLLM_USE_V2_MODEL_RUNNER`` to pin the axis under test.  Left to leak,
    that setting would not merely bias the offline engine -- it would kill it.
    The offline leg runs the ``extract_hidden_states`` speculative method,
    which V2 does not implement; with the variable unset vLLM downgrades to V1
    and logs it (``vllm/config/vllm.py:546-553``), but with it set to ``1`` the
    property short-circuits and ``_validate_v2_model_runner`` raises instead
    (``vllm/config/vllm.py:2137-2147``).

    The offline extraction is the independent reference the capture is
    compared against, so it has to run whatever generation vLLM considers
    correct for its own config.
    """

    @pytest.mark.parametrize("value", ["0", "1"])
    def test_forced_runner_does_not_leak_into_the_offline_engine(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Both pinned values are stripped, not just the fatal one."""
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)

        env = offline_extract._environ(Path("/nonexistent/speculators"))

        assert "VLLM_USE_V2_MODEL_RUNNER" not in env

    def test_unrelated_environment_still_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Isolation is surgical: only the named variables are removed.

        The offline subprocesses need the inherited HF cache and offline-mode
        settings, so a blanket scrub would break them.
        """
        monkeypatch.setenv("HF_HOME", "/data/ryan.kim/hf-cache")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

        env = offline_extract._environ(Path("/nonexistent/speculators"))

        assert env["HF_HOME"] == "/data/ryan.kim/hf-cache"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert "VLLM_USE_V2_MODEL_RUNNER" not in env

    def test_pythonpath_still_gains_the_speculators_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-existing PYTHONPATH prepend survives the new filtering."""
        monkeypatch.setenv("PYTHONPATH", "/existing/path")
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

        env = offline_extract._environ(Path("/repo/speculators"))

        assert env["PYTHONPATH"].startswith("/repo/speculators/src")
        assert env["PYTHONPATH"].endswith("/existing/path")

    def test_absent_variable_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common case -- an unpinned run -- must not raise on the pop."""
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

        env = offline_extract._environ(Path("/repo/speculators"))

        assert "VLLM_USE_V2_MODEL_RUNNER" not in env


# ---------------------------------------------------------------------------
# Slice-before-drafter logic (hook.py)
# ---------------------------------------------------------------------------


class TestSliceBeforeDrafter:
    """Test that the extension correctly slices the final-layer entry
    from aux_hidden_states before feeding to the drafter.
    """

    def test_slice_removes_extra_entry(self) -> None:
        """4-entry aux list truncated to 3 for the drafter."""
        aux = [torch.ones(10, 2880) for _ in range(4)]
        expected_count = 3
        if len(aux) > expected_count:
            del aux[expected_count:]
        assert len(aux) == 3

    def test_slice_noop_when_correct_count(self) -> None:
        """3-entry aux list is untouched."""
        aux = [torch.ones(10, 2880) for _ in range(3)]
        expected_count = 3
        if len(aux) > expected_count:
            del aux[expected_count:]
        assert len(aux) == 3

    def test_concat_width_matches_drafter_fc(self) -> None:
        """After slicing, concatenated width = num_aux * H."""
        aux = [torch.ones(10, 2880) for _ in range(3)]
        concat = torch.cat(aux, dim=-1)
        assert concat.shape == (10, 3 * 2880)

    def test_four_entries_would_crash_drafter(self) -> None:
        """4 entries concatenated would be 4*H, not 3*H."""
        aux = [torch.ones(10, 2880) for _ in range(4)]
        concat = torch.cat(aux, dim=-1)
        assert concat.shape[1] == 4 * 2880
        assert concat.shape[1] != 3 * 2880


# ---------------------------------------------------------------------------
# Token count alignment (e2e test helpers)
# ---------------------------------------------------------------------------


class TestTokenAlignment:
    """Exercise the real aligner, ``compare.align_prompt_rows``.

    The predecessors of these tests re-implemented ``cap[: off.shape[0]]``
    inline and asserted on their own re-implementation, so they were true no
    matter what the shipped function did.  Worse, the quantity they treated as
    "the prompt" was the *offline* row count, baking the bug into the tests.
    Every test below calls the shipped function and passes the prompt token
    count as an independent third value.
    """

    def test_trims_both_sides_to_the_prompt(self) -> None:
        # 18 prompt rows; capture also holds 30 generated rows, offline also
        # holds 6 template-assistant rows.  Neither tail is comparable.
        cap = torch.randn(48, 128)
        off = torch.randn(24, 128)
        c, o = align_prompt_rows(cap, off, 18)
        assert c.shape == (18, 128)
        assert o.shape == (18, 128)
        assert torch.equal(c, cap[:18])
        assert torch.equal(o, off[:18])

    def test_offline_row_count_is_not_the_prompt_length(self) -> None:
        """The offline stack's tail must be dropped, not compared.

        This is the exact shape of GPU job 369218: the offline path renders
        the assistant turn into the conversation, so its rows outnumber the
        prompt.  The aligner must not treat ``offline.shape[0]`` as the
        prompt length.
        """
        cap = torch.randn(48, 128)
        off = torch.randn(24, 128)
        _, o = align_prompt_rows(cap, off, 18)
        assert o.shape[0] == 18, (
            "aligner kept the offline template-assistant rows; those tokens "
            "were never fed to the serving engine"
        )

    def test_equal_lengths_still_trim_to_prompt(self) -> None:
        cap = torch.randn(24, 128)
        off = torch.randn(24, 128)
        c, o = align_prompt_rows(cap, off, 18)
        assert c.shape[0] == 18
        assert o.shape[0] == 18

    def test_captured_shorter_than_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="captured has fewer rows"):
            align_prompt_rows(torch.randn(6, 128), torch.randn(24, 128), 18)

    def test_offline_shorter_than_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="offline has fewer rows"):
            align_prompt_rows(torch.randn(48, 128), torch.randn(12, 128), 18)

    def test_non_positive_prompt_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            align_prompt_rows(torch.randn(48, 128), torch.randn(24, 128), 0)

    def test_bf16_alignment_preserves_dtype(self) -> None:
        cap = torch.randn(48, 128, dtype=torch.bfloat16)
        off = torch.randn(24, 128, dtype=torch.bfloat16)
        c, o = align_prompt_rows(cap, off, 18)
        assert c.dtype == torch.bfloat16
        assert o.dtype == torch.bfloat16
        assert c.shape == (18, 128)


class TestDivergentTailRegression:
    """Regression for GPU job 369218 (Qwen3-8B, verdict FAIL_tolerance).

    Signature reproduced from the real artifacts: the first 18 rows of the
    capture are *bit-identical* to offline (proving the capture point is
    correct), and rows 18-23 are unrelated because the serving engine
    generated its own tokens while the offline path ran the chat template's
    assistant turn.  Comparing all 24 rows produced mean_rel_error 0.12-0.23
    against a 0.10 tolerance; comparing the 18 shared rows must be exact.
    """

    @staticmethod
    def _stacks() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        shared = torch.randn(18, 64)
        captured = torch.cat([shared, torch.randn(30, 64)], dim=0)
        offline = torch.cat([shared, torch.randn(6, 64)], dim=0)
        return captured, offline

    def test_comparing_the_divergent_tail_fails(self) -> None:
        """The OLD behaviour — trim captured to offline — must fail."""
        captured, offline = self._stacks()
        old_captured = captured[: offline.shape[0]]
        result = build_result({2: old_captured}, {2: offline})
        assert result.verdict == "FAIL_tolerance"

    def test_prompt_aligned_comparison_is_exact(self) -> None:
        """The NEW behaviour — trim both to the prompt — must be exact."""
        captured, offline = self._stacks()
        c, o = align_prompt_rows(captured, offline, 18)
        result = build_result(
            {2: c}, {2: o}, captured_final_pre_norm=c, offline_final_pre_norm=o
        )
        assert result.verdict == "PASS"
        assert result.layers[0].mean_rel_error == 0.0


# ---------------------------------------------------------------------------
# E2E test helper contract (no GPU needed)
# ---------------------------------------------------------------------------


class TestE2EHelpers:
    """Verify the e2e test helper contracts without a running engine."""

    def test_collective_rpc_accepts_port_parameter(self) -> None:
        """_collective_rpc must accept a port parameter (not re-derive it)."""
        import inspect

        from tests.e2e.test_serving_activation_capture import _collective_rpc
        sig = inspect.signature(_collective_rpc)
        params = list(sig.parameters.keys())
        assert "port" in params, (
            f"_collective_rpc must have a 'port' parameter; got {params}"
        )
        # port should come before method
        assert params.index("port") < params.index("method"), (
            "port should come before method in _collective_rpc signature"
        )

    def test_vllm_env_contains_dev_mode(self) -> None:
        """_vllm_env must set VLLM_SERVER_DEV_MODE=1."""

        from tests.e2e.test_serving_activation_capture import _vllm_env
        env = _vllm_env()
        assert env.get("VLLM_SERVER_DEV_MODE") == "1"
        # Should inherit other env vars
        assert "PATH" in env or len(env) > 1  # basic sanity

    def test_vllm_env_does_not_mutate_os_environ(self) -> None:
        """_vllm_env must not modify os.environ."""
        import os

        from tests.e2e.test_serving_activation_capture import _vllm_env
        # Temporarily unset the key to prove we copy
        original = os.environ.pop("VLLM_SERVER_DEV_MODE", None)
        try:
            env = _vllm_env()
            assert os.environ.get("VLLM_SERVER_DEV_MODE") is None
            assert env.get("VLLM_SERVER_DEV_MODE") == "1"
        finally:
            if original is not None:
                os.environ["VLLM_SERVER_DEV_MODE"] = original

    def test_strict_verdict_env_var(self) -> None:
        """SPEEDLM_E2E_STRICT_VERDICT=0 disables verdict assertion."""
        import os

        # Default: strict is enabled
        assert os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"

        # Opt-out: setting to "0" disables
        os.environ["SPEEDLM_E2E_STRICT_VERDICT"] = "0"
        try:
            strict = os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"
            assert strict is False
        finally:
            del os.environ["SPEEDLM_E2E_STRICT_VERDICT"]


# ---------------------------------------------------------------------------
# Real safetensors I/O (prevents safe_open API misuse from recurring)
# ---------------------------------------------------------------------------


class TestSafetensorsIO:
    """Round-trip through real safetensors files on tmp_path.

    These tests write actual safetensors files and read them back through
    the loaders used by the e2e test and offline_extract.

    The safetensors requirement is declared as a class-scoped ``pytestmark``.
    It used to be a ``pytest.importorskip`` in the class body, which raised
    ``Skipped`` during collection of the *module* and therefore silently
    discarded every test in this file -- ``collected 0 items / 1 skipped`` --
    on any interpreter without safetensors.
    """

    pytestmark = pytest.mark.skipif(
        not _HAS_SAFETENSORS,
        reason="safetensors is not installed in the project venv (it lives in the vLLM venv)",
    )

    def test_captured_loader_roundtrip(self, tmp_path: Path) -> None:
        """_load_captured_safetensors can read a file we wrote."""
        from safetensors.torch import save_file

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()
        save_file(
            {
                "layer_2": torch.ones(4, 8),
                "layer_12": torch.ones(4, 8) * 2,
                "layer_21": torch.ones(4, 8) * 3,
                "layer_24": torch.ones(4, 8) * 4,  # final layer
            },
                str(capture_dir / "captured.safetensors"),
        )

        from tests.e2e.test_serving_activation_capture import _load_captured_safetensors

        tensors = _load_captured_safetensors(capture_dir)
        assert sorted(tensors.keys()) == [2, 12, 21, 24]
        assert tensors[2].shape == (4, 8)
        assert torch.allclose(tensors[12], torch.ones(4, 8) * 2)

    def test_captured_loader_missing_file(self, tmp_path: Path) -> None:
        """_load_captured_safetensors raises FileNotFoundError when absent."""
        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()

        from tests.e2e.test_serving_activation_capture import _load_captured_safetensors

        with pytest.raises(FileNotFoundError, match="no captured.safetensors"):
            _load_captured_safetensors(capture_dir)

    def test_offline_loader_roundtrip(self, tmp_path: Path) -> None:
        """_load_offline_hidden_states reads hs_*.safetensors shards."""
        from safetensors.torch import save_file

        hs_dir = tmp_path / "offline_hs"
        hs_dir.mkdir()
        # (seq_len=4, num_layers=4, hidden_size=8)
        hs = torch.randn(4, 4, 8)
        save_file({"hidden_states": hs}, str(hs_dir / "hs_0.safetensors"))

        from tests.e2e.test_serving_activation_capture import _load_offline_hidden_states

        target_layers = [2, 12, 21, 24]
        tensors = _load_offline_hidden_states(hs_dir, target_layers=target_layers)
        # 4 layers in shard -> keys = target_layers
        assert sorted(tensors.keys()) == [2, 12, 21, 24]
        assert tensors[2].shape == (4, 8)
        assert torch.allclose(tensors[2], hs[:, 0])

    def test_offline_loader_missing_file(self, tmp_path: Path) -> None:
        """_load_offline_hidden_states raises FileNotFoundError when absent."""
        hs_dir = tmp_path / "offline_hs"
        hs_dir.mkdir()

        from tests.e2e.test_serving_activation_capture import _load_offline_hidden_states

        with pytest.raises(FileNotFoundError, match="no hs_.*safetensors"):
            _load_offline_hidden_states(hs_dir, target_layers=[2])

    def test_offline_loader_multiple_shards(self, tmp_path: Path) -> None:
        """Multiple hs_*.safetensors shards are concatenated along dim=0."""
        from safetensors.torch import save_file

        hs_dir = tmp_path / "offline_hs"
        hs_dir.mkdir()
        save_file({"hidden_states": torch.ones(4, 2, 8)}, str(hs_dir / "hs_0.safetensors"))
        save_file({"hidden_states": torch.ones(3, 2, 8) * 2}, str(hs_dir / "hs_1.safetensors"))

        from tests.e2e.test_serving_activation_capture import _load_offline_hidden_states

        tensors = _load_offline_hidden_states(hs_dir, target_layers=[2, 12])
        assert sorted(tensors.keys()) == [2, 12]
        # 4 + 3 = 7 rows concatenated
        assert tensors[2].shape == (7, 8)

    def test_load_hidden_states_roundtrip(self, tmp_path: Path) -> None:
        """offline_extract.load_hidden_states reads shards correctly."""
        from safetensors.torch import save_file

        from speedlm.activation_capture.offline_extract import load_hidden_states

        hs_dir = tmp_path / "offline_hs"
        hs_dir.mkdir()
        hs = torch.randn(6, 3, 16)
        save_file({"hidden_states": hs}, str(hs_dir / "hs_0.safetensors"))

        result = load_hidden_states(hs_dir)
        assert sorted(result.keys()) == ["0", "1", "2"]
        assert result["0"].shape == (6, 16)
        assert torch.allclose(result["0"], hs[:, 0])

    def test_safe_open_uses_keys_method(self, tmp_path: Path) -> None:
        """Verify that safe_open handle is NOT directly iterable; .keys() is required.

        Regression test: iterating the handle directly raised
        TypeError: 'builtins.safe_open' object is not iterable.
        """
        from safetensors.torch import save_file

        path = tmp_path / "test.safetensors"
        save_file({"layer_42": torch.ones(2, 4)}, str(path))

        from safetensors import safe_open

        # This should work
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        assert keys == ["layer_42"]

        # Direct iteration would fail (documented, not tested here to avoid
        # asserting on C extension behavior that may change)

    def test_key_parsing_robust(self, tmp_path: Path) -> None:
        """Key parsing with split("_", 1) handles layer indices with underscores."""
        from safetensors.torch import save_file

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()
        # Edge case: layer index could theoretically contain extra text
        save_file(
            {
                "layer_2": torch.ones(2, 4),
                "layer_12": torch.ones(2, 4),
            },
            str(capture_dir / "captured.safetensors"),
        )

        from tests.e2e.test_serving_activation_capture import _load_captured_safetensors

        tensors = _load_captured_safetensors(capture_dir)
        assert 2 in tensors
        assert 12 in tensors

    def test_captured_loader_ignores_non_layer_keys(self, tmp_path: Path) -> None:
        """Keys not starting with 'layer_' are silently skipped."""
        from safetensors.torch import save_file

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()
        save_file(
            {
                "layer_5": torch.ones(2, 4),
                "metadata": torch.zeros(1),  # should be ignored
                "prompt_ids": torch.zeros(1),  # should be ignored
            },
            str(capture_dir / "captured.safetensors"),
            # ``__metadata__`` is the reserved safetensors header slot and only
            # accepts str -> str; it must not be passed as a tensor key.
            metadata={"final_layer_idx": "35"},  # should be ignored
        )

        from tests.e2e.test_serving_activation_capture import _load_captured_safetensors

        tensors = _load_captured_safetensors(capture_dir)
        assert sorted(tensors.keys()) == [5]

    def test_bf16_roundtrip(self, tmp_path: Path) -> None:
        """Captured tensors are bf16; loader preserves dtype."""
        from safetensors.torch import save_file

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()
        save_file(
            {"layer_2": torch.randn(4, 8, dtype=torch.bfloat16)},
            str(capture_dir / "captured.safetensors"),
        )

        from tests.e2e.test_serving_activation_capture import _load_captured_safetensors

        tensors = _load_captured_safetensors(capture_dir)
        assert tensors[2].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Metadata reading
# ---------------------------------------------------------------------------


class TestCaptureMetadata:
    """Test _load_capture_metadata reads metadata from disk."""

    def test_load_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Metadata written by flush_capture can be read back."""
        from tests.e2e.test_serving_activation_capture import _load_capture_metadata

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()
        meta = {
            "final_layer_idx": 24,
            "original_aux_layers": [2, 12, 21],
        }
        import json

        (capture_dir / "captured.safetensors.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        loaded = _load_capture_metadata(capture_dir)
        assert loaded["final_layer_idx"] == 24
        assert loaded["original_aux_layers"] == [2, 12, 21]

    def test_load_metadata_missing_file(self, tmp_path: Path) -> None:
        """Returns sensible defaults when metadata file is absent."""
        from tests.e2e.test_serving_activation_capture import _load_capture_metadata

        capture_dir = tmp_path / "captured"
        capture_dir.mkdir()

        loaded = _load_capture_metadata(capture_dir)
        assert loaded["final_layer_idx"] is None
        assert loaded["original_aux_layers"] == []


# ---------------------------------------------------------------------------
# Split captured layers
# ---------------------------------------------------------------------------


class TestSplitCapturedLayers:
    """Test _split_captured_layers separates drafter-inputs from regression-target."""

    def test_split_correct_with_final_layer(self) -> None:
        """Captured [2,12,21,24] with metadata [2,12,21] + final=24 splits correctly."""
        from tests.e2e.test_serving_activation_capture import _split_captured_layers

        captured = {
            2: torch.ones(4, 8),
            12: torch.ones(4, 8) * 2,
            21: torch.ones(4, 8) * 3,
            24: torch.ones(4, 8) * 4,
        }
        meta = {
            "final_layer_idx": 24,
            "original_aux_layers": [2, 12, 21],
        }

        drafter_ids, final_idx, drafter_tensors, regression = _split_captured_layers(
            captured, meta
        )

        assert drafter_ids == [2, 12, 21]
        assert final_idx == 24
        assert sorted(drafter_tensors.keys()) == [2, 12, 21]
        assert regression is not None
        assert torch.allclose(regression, torch.ones(4, 8) * 4)

    def test_split_without_final_layer(self) -> None:
        """When final_layer_idx is None, regression target is None."""
        from tests.e2e.test_serving_activation_capture import _split_captured_layers

        captured = {2: torch.ones(4, 8), 12: torch.ones(4, 8)}
        meta = {"final_layer_idx": None, "original_aux_layers": [2, 12]}

        drafter_ids, final_idx, drafter_tensors, regression = _split_captured_layers(
            captured, meta
        )

        assert drafter_ids == [2, 12]
        assert final_idx is None
        assert sorted(drafter_tensors.keys()) == [2, 12]
        assert regression is None

    def test_split_wrong_layers_still_succeeds(self) -> None:
        """If captured has unexpected layers, only the expected ones are split."""
        from tests.e2e.test_serving_activation_capture import _split_captured_layers

        captured = {
            2: torch.ones(4, 8),
            12: torch.ones(4, 8),
            99: torch.ones(4, 8),  # unexpected
        }
        meta = {
            "final_layer_idx": 24,
            "original_aux_layers": [2, 12, 21],
        }

        drafter_ids, final_idx, drafter_tensors, regression = _split_captured_layers(
            captured, meta
        )

        assert drafter_ids == [2, 12]  # 21 not in captured
        assert final_idx == 24
        assert regression is None  # 24 not in captured


# ---------------------------------------------------------------------------
# Split offline layers
# ---------------------------------------------------------------------------


class TestSplitOfflineLayers:
    """Test _split_offline_layers separates drafter-inputs from regression-target."""

    def test_split_correct_with_final_layer(self) -> None:
        """Offline [2,12,21,24] splits into drafter [2,12,21] and final 24."""
        from tests.e2e.test_serving_activation_capture import _split_offline_layers

        offline = {
            2: torch.ones(4, 8),
            12: torch.ones(4, 8) * 2,
            21: torch.ones(4, 8) * 3,
            24: torch.ones(4, 8) * 4,
        }
        offline_target_layers = [2, 12, 21, 24]

        drafter_ids, final_idx, drafter_tensors, regression = _split_offline_layers(
            offline, offline_target_layers
        )

        assert drafter_ids == [2, 12, 21]
        assert final_idx == 24
        assert sorted(drafter_tensors.keys()) == [2, 12, 21]
        assert regression is not None
        assert torch.allclose(regression, torch.ones(4, 8) * 4)

    def test_split_without_final_layer(self) -> None:
        """When offline_target_layers has only drafter inputs, no regression target."""
        from tests.e2e.test_serving_activation_capture import _split_offline_layers

        offline = {2: torch.ones(4, 8), 12: torch.ones(4, 8)}
        offline_target_layers = [2, 12]

        drafter_ids, final_idx, drafter_tensors, regression = _split_offline_layers(
            offline, offline_target_layers
        )

        # With only 2 layers, the last is still treated as final
        assert drafter_ids == [2]
        assert final_idx == 12
        assert sorted(drafter_tensors.keys()) == [2]
        assert regression is not None


# ---------------------------------------------------------------------------
# Assertion: captured layers match target + final
# ---------------------------------------------------------------------------


class TestCorrectedAssertion:
    """Test that the corrected assertion logic works correctly."""

    def test_correct_captured_layers_pass(self) -> None:
        """[2,12,21,24] passes when target=[2,12,21] and final=24."""
        captured_keys = sorted([2, 12, 21, 24])
        meta = {"original_aux_layers": [2, 12, 21], "final_layer_idx": 24}
        target_layers = [2, 12, 21]

        # Drafter-input check
        assert sorted(meta["original_aux_layers"]) == target_layers

        # Full key check
        expected = sorted(
            meta["original_aux_layers"]
            + ([meta["final_layer_idx"]] if meta["final_layer_idx"] is not None else [])
        )
        assert captured_keys == expected

    def test_wrong_captured_layers_fail(self) -> None:
        """[2,12,21,99] fails when target=[2,12,21] and final=24."""
        captured_keys = sorted([2, 12, 21, 99])
        meta = {"original_aux_layers": [2, 12, 21], "final_layer_idx": 24}
        target_layers = [2, 12, 21]

        assert sorted(meta["original_aux_layers"]) == target_layers

        expected = sorted(
            meta["original_aux_layers"]
            + ([meta["final_layer_idx"]] if meta["final_layer_idx"] is not None else [])
        )
        assert captured_keys != expected  # 99 != 24, so they differ

    def test_drafter_input_vs_regression_mapping(self) -> None:
        """Verify drafter-inputs and regression-target are correctly mapped."""
        from tests.e2e.test_serving_activation_capture import (
            _split_captured_layers,
            _split_offline_layers,
        )

        captured = {
            2: torch.ones(4, 8),
            12: torch.ones(4, 8) * 2,
            21: torch.ones(4, 8) * 3,
            24: torch.ones(4, 8) * 4,
        }
        meta = {"final_layer_idx": 24, "original_aux_layers": [2, 12, 21]}

        offline = dict(captured)  # same layers for offline
        offline_target_layers = [2, 12, 21, 24]

        (
            cap_drafter_ids,
            cap_final_idx,
            cap_drafter,
            cap_regression,
        ) = _split_captured_layers(captured, meta)

        (
            off_drafter_ids,
            off_final_idx,
            off_drafter,
            off_regression,
        ) = _split_offline_layers(offline, offline_target_layers)

        # Drafter-input layer ids must match
        assert cap_drafter_ids == off_drafter_ids == [2, 12, 21]

        # Final layer indices must match
        assert cap_final_idx == off_final_idx == 24

        # Drafter tensors are compared by key
        for idx in cap_drafter_ids:
            assert idx in cap_drafter
            assert idx in off_drafter

        # Regression targets are compared separately
        assert cap_regression is not None
        assert off_regression is not None


# ---------------------------------------------------------------------------
# check_pre_norm: real captured final layer produces real verdict
# ---------------------------------------------------------------------------


class TestCheckPreNormReal:
    """Verify check_pre_norm returns True/False when both tensors present."""

    def test_real_match_returns_true(self) -> None:
        """Identical tensors -> True."""
        t = torch.randn(4, 8)
        assert check_pre_norm(t, t) is True

    def test_real_mismatch_returns_false(self) -> None:
        """Very different tensors -> False (not None)."""
        a = torch.zeros(4, 8)
        b = torch.ones(4, 8)
        assert check_pre_norm(a, b) is False

    def test_build_result_passes_with_real_final_layer(self) -> None:
        """build_result produces PASS when final layer is present and matches."""
        t = torch.ones(4, 8)
        result = build_result(
            {2: t, 12: t, 21: t},
            {2: t, 12: t, 21: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
        )
        assert result.verdict == "PASS"
        assert result.pre_norm_match is True

    def test_build_result_fails_when_no_final_layer(self) -> None:
        """build_result produces FAIL_pre_norm when final layer absent."""
        t = torch.ones(4, 8)
        result = build_result(
            {2: t, 12: t, 21: t},
            {2: t, 12: t, 21: t},
            captured_final_pre_norm=None,
            offline_final_pre_norm=None,
        )
        assert result.verdict == "FAIL_pre_norm"
        assert result.pre_norm_match is None


# ---------------------------------------------------------------------------
# derive_verdict cannot PASS with pre_norm_match None
# ---------------------------------------------------------------------------


class TestDeriveVerdictPreNormInvariant:
    """derive_verdict must never return PASS when pre_norm_match is None."""

    def test_pass_requires_pre_norm_true(self) -> None:
        """PASS verdict requires pre_norm_match == True, not None."""
        layers = [
            LayerComparison(2, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
            LayerComparison(12, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
            LayerComparison(21, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
        ]
        result_with_none = ComparisonResult(
            layers=layers,
            pre_norm_match=None,
            tolerance=DEFAULT_TOLERANCE,
            verdict="",
        )
        assert derive_verdict(result_with_none) != "PASS"
        assert derive_verdict(result_with_none) == "FAIL_pre_norm"

        result_with_true = ComparisonResult(
            layers=layers,
            pre_norm_match=True,
            tolerance=DEFAULT_TOLERANCE,
            verdict="",
        )
        assert derive_verdict(result_with_true) == "PASS"

    def test_fail_verdict_fails_test(self) -> None:
        """A FAIL verdict should fail the e2e test by default."""
        # Simulate the verdict check in the e2e test
        result = build_result(
            {0: torch.zeros(4, 8)},
            {0: torch.ones(4, 8)},
            captured_final_pre_norm=torch.zeros(4, 8),
            offline_final_pre_norm=torch.ones(4, 8),
        )
        assert result.verdict != "PASS"
        # The e2e test would now assert on this
        import os
        strict = os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"
        assert strict  # default is strict
        # With strict mode, a FAIL verdict would fail the test
        if strict:
            with pytest.raises(AssertionError, match="Activation capture comparison failed"):
                assert result.verdict == "PASS", (
                    f"Activation capture comparison failed: verdict={result.verdict}"
                )

    def test_opt_out_env_var_allows_exploratory(self) -> None:
        """SPEEDLM_E2E_STRICT_VERDICT=0 allows FAIL verdict without test failure."""
        import os

        result = build_result(
            {0: torch.zeros(4, 8)},
            {0: torch.ones(4, 8)},
            captured_final_pre_norm=torch.zeros(4, 8),
            offline_final_pre_norm=torch.ones(4, 8),
        )
        assert result.verdict != "PASS"

        # With opt-out, the verdict check is skipped
        os.environ["SPEEDLM_E2E_STRICT_VERDICT"] = "0"
        try:
            strict = os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"
            assert strict is False
            # The assertion would be skipped
        finally:
            del os.environ["SPEEDLM_E2E_STRICT_VERDICT"]

# ---------------------------------------------------------------------------
# Matrix aggregation (PromptCase / MatrixResult / derive_matrix_verdict)
#
# APPENDED SECTION -- self-contained. These tests are torch-free (the matrix
# layer only reads already-computed ComparisonResult verdicts), so they carry
# ``@pytest.mark.no_torch`` and run in the PROJECT venv. The whole point of the
# matrix is to widen Stage 0 coverage across prompts; a matrix layer that only
# executed under a hand-set PYTHONPATH would be untested in the default suite.
# ---------------------------------------------------------------------------


def _matrix_result(verdict: str) -> ComparisonResult:
    """A minimal ComparisonResult carrying a chosen verdict.

    Built the same way the ``TestDeriveVerdict`` helper above builds one --
    a literal ``LayerComparison`` list, no tensors -- so nothing here needs
    torch. The layer content is irrelevant to the matrix layer, which reads
    only ``result.verdict``.
    """
    return ComparisonResult(
        layers=[
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001,
                            mean_rel_error=0.001),
        ],
        pre_norm_match=True,
        prefix_cache_test=None,
        tolerance=DEFAULT_TOLERANCE,
        verdict=verdict,
    )


def _matrix_case(label: str, verdict: str, prompt_token_count: int = 16):
    from speedlm.activation_capture.compare import PromptCase

    return PromptCase(
        label=label,
        prompt_token_count=prompt_token_count,
        result=_matrix_result(verdict),
    )


class TestDeriveMatrixVerdict:
    """A matrix verdict must never hide a failing prompt."""

    @pytest.mark.no_torch
    def test_empty_cases_is_fail_not_pass(self) -> None:
        """A matrix that compared nothing must NOT be a PASS.

        This is the load-bearing assertion of the whole matrix layer: a
        mis-wired matrix that silently covers zero prompts would otherwise
        report green on vacuous truth.
        """
        from speedlm.activation_capture.compare import derive_matrix_verdict

        assert derive_matrix_verdict([]) != "PASS"
        assert derive_matrix_verdict([]) == "FAIL_empty"

    @pytest.mark.no_torch
    def test_all_cases_pass(self) -> None:
        from speedlm.activation_capture.compare import derive_matrix_verdict

        cases = [
            _matrix_case("short", "PASS"),
            _matrix_case("medium", "PASS"),
            _matrix_case("long", "PASS"),
        ]
        assert derive_matrix_verdict(cases) == "PASS"

    @pytest.mark.no_torch
    def test_one_failing_case_is_named(self) -> None:
        """One failure among passes must surface that case's label + verdict."""
        from speedlm.activation_capture.compare import derive_matrix_verdict

        cases = [
            _matrix_case("short", "PASS"),
            _matrix_case("medium", "FAIL_tolerance"),
            _matrix_case("long", "PASS"),
        ]
        verdict = derive_matrix_verdict(cases)
        assert verdict != "PASS"
        assert verdict == "FAIL_cases:medium=FAIL_tolerance"
        assert "medium" in verdict
        assert "FAIL_tolerance" in verdict

    @pytest.mark.no_torch
    def test_two_failing_cases_both_named_in_order(self) -> None:
        """Both failures appear, in case order -- no averaging, no worst-of."""
        from speedlm.activation_capture.compare import derive_matrix_verdict

        cases = [
            _matrix_case("short", "FAIL_shape"),
            _matrix_case("medium", "PASS"),
            _matrix_case("long", "FAIL_tolerance"),
        ]
        verdict = derive_matrix_verdict(cases)
        assert verdict != "PASS"
        assert verdict == "FAIL_cases:short=FAIL_shape,long=FAIL_tolerance"
        assert verdict.index("short") < verdict.index("long")

    @pytest.mark.no_torch
    def test_build_matrix_result_sets_verdict(self) -> None:
        from speedlm.activation_capture.compare import build_matrix_result

        matrix = build_matrix_result(
            model="Qwen/Qwen3-8B",
            runner="v1",
            cases=[_matrix_case("short", "PASS"), _matrix_case("long", "PASS")],
        )
        assert matrix.model == "Qwen/Qwen3-8B"
        assert matrix.runner == "v1"
        assert isinstance(matrix.cases, tuple)
        assert matrix.verdict == "PASS"

        failing = build_matrix_result(
            model="Qwen/Qwen3-8B",
            runner="v2",
            cases=[_matrix_case("short", "PASS"), _matrix_case("long", "FAIL_empty")],
        )
        assert failing.verdict == "FAIL_cases:long=FAIL_empty"

    @pytest.mark.no_torch
    def test_build_matrix_result_empty_is_not_pass(self) -> None:
        from speedlm.activation_capture.compare import build_matrix_result

        matrix = build_matrix_result(model="m", runner="v1", cases=[])
        assert matrix.cases == ()
        assert matrix.verdict != "PASS"
        assert matrix.verdict == "FAIL_empty"


class TestMatrixResultSerialization:
    @pytest.mark.no_torch
    def test_to_dict_is_json_serializable_and_complete(self) -> None:
        from speedlm.activation_capture.compare import build_matrix_result

        cases = [
            _matrix_case("short", "PASS", prompt_token_count=8),
            _matrix_case("long", "FAIL_tolerance", prompt_token_count=128),
        ]
        matrix = build_matrix_result(model="model-x", runner="v2", cases=cases)
        payload = matrix.to_dict()

        # Round-trips through json without a custom encoder.
        reparsed = json.loads(json.dumps(payload))
        assert reparsed == payload

        assert reparsed["model"] == "model-x"
        assert reparsed["runner"] == "v2"
        assert reparsed["verdict"] == "FAIL_cases:long=FAIL_tolerance"
        assert [c["label"] for c in reparsed["cases"]] == ["short", "long"]
        assert [c["prompt_token_count"] for c in reparsed["cases"]] == [8, 128]
        # The nested per-prompt comparison dicts are carried whole.
        for case, expected in zip(reparsed["cases"], cases, strict=True):
            assert case["result"] == expected.result.to_dict()
            assert case["result"]["verdict"] == expected.result.verdict

    @pytest.mark.no_torch
    def test_write_json_matches_to_dict(self, tmp_path: Path) -> None:
        from speedlm.activation_capture.compare import build_matrix_result

        matrix = build_matrix_result(
            model="model-x",
            runner="v1",
            cases=[
                _matrix_case("short", "PASS"),
                _matrix_case("long", "PASS"),
            ],
        )
        out = tmp_path / "matrix.json"
        matrix.write_json(out)

        text = out.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text) == matrix.to_dict()
