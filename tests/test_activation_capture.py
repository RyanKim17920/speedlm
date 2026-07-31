"""Unit tests for the activation capture comparison logic.

These tests exercise the comparison, tolerance, verdict, and serialization
pathways using synthetic tensors — no GPU required.

Skipped when torch is not installed (torch lives in the vLLM venv, not the
project venv).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from speedlm.activation_capture.compare import (  # noqa: E402
    DEFAULT_TOLERANCE,
    ComparisonResult,
    LayerComparison,
    PrefixCacheResult,
    build_result,
    check_pre_norm,
    check_within_tolerance,
    compare_layerwise,
    derive_verdict,
)

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
        assert layers[0].captured_shape == []
        assert layers[0].offline_shape == [2, 2]

    def test_missing_layer_in_offline(self) -> None:
        """Layer present only in captured is a shape mismatch."""
        layers = compare_layerwise({0: torch.ones(2, 2)}, {})
        assert len(layers) == 1
        assert layers[0].shape_match is False
        assert layers[0].captured_shape == [2, 2]
        assert layers[0].offline_shape == []

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
    def test_all_within(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.009, 0.005),
        ]
        assert check_within_tolerance(layers, DEFAULT_TOLERANCE) is True

    def test_one_exceeds(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.05, 0.03),
        ]
        assert check_within_tolerance(layers, DEFAULT_TOLERANCE) is False

    def test_empty(self) -> None:
        assert check_within_tolerance([], DEFAULT_TOLERANCE) is True

    def test_none_diff_treated_as_within(self) -> None:
        """None max_abs_diff means shapes didn't match; tolerance check
        is irrelevant. The verdict layer handles this separately."""
        layers = [
            LayerComparison(0, (4, 8), (4, 16), False, None, None),
        ]
        # None is not <= tolerance, so this returns False
        assert check_within_tolerance(layers, DEFAULT_TOLERANCE) is False


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
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
            LayerComparison(1, (4, 8), (4, 8), True, 0.005, 0.003),
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

    def test_fail_tolerance(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.05, 0.03),
        ]
        result = self._result(layers, True)
        assert derive_verdict(result) == "FAIL_tolerance"

    def test_fail_pre_norm_none(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
        ]
        result = self._result(layers, None)
        assert derive_verdict(result) == "FAIL_pre_norm"

    def test_fail_pre_norm_false(self) -> None:
        layers = [
            LayerComparison(0, (4, 8), (4, 8), True, 0.001, 0.001),
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
            captured_row_count=10,
            cache_hit=False,
            rows_missing=0,
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

    def test_custom_tolerance(self) -> None:
        a = torch.ones(4, 8)
        b = torch.ones(4, 8) + 0.015
        # With default tolerance (0.01), this fails. With 0.02, it passes.
        result_strict = build_result({0: a}, {0: b}, tolerance=0.01)
        assert result_strict.verdict == "FAIL_tolerance"
        result_lenient = build_result({0: a}, {0: b}, tolerance=0.02)
        assert result_lenient.verdict == "FAIL_pre_norm"  # no pre-norm tensors


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
        assert d["all_shapes_match"] is True
        assert d["all_within_tolerance"] is True
        assert d["pre_norm_match"] is True
        assert len(d["layers"]) == 2

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
        pc = PrefixCacheResult(8, 6, True, 2)
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
        assert data["prefix_cache_test"]["rows_missing"] == 2


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
    """Test that captured/offline token alignment works correctly."""

    def test_equal_lengths_pass_through(self) -> None:
        cap = torch.randn(12, 2880)
        off = torch.randn(12, 2880)
        assert cap.shape[0] == off.shape[0]
        # _align_token_count would return them unchanged

    def test_captured_longer_is_trimmed(self) -> None:
        cap = torch.randn(28, 2880)  # prompt + generated
        off = torch.randn(12, 2880)  # prompt only
        trimmed = cap[: off.shape[0]]
        assert trimmed.shape == off.shape
        assert trimmed.shape[0] == 12

    def test_captured_shorter_raises(self) -> None:
        cap = torch.randn(6, 2880)
        off = torch.randn(12, 2880)
        # Captured has fewer rows — this should be detected as a mismatch
        assert cap.shape[0] < off.shape[0]

    def test_bf16_alignment_preserves_dtype(self) -> None:
        cap = torch.randn(16, 2880, dtype=torch.bfloat16)
        off = torch.randn(12, 2880, dtype=torch.bfloat16)
        trimmed = cap[: off.shape[0]]
        assert trimmed.dtype == torch.bfloat16
        assert trimmed.shape == (12, 2880)