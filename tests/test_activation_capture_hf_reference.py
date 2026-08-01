"""Unit tests for the independent HuggingFace fp32 reference comparison.

These pin the logic that decides whether a captured aux layer *is* the
quantity it claims to be: the derived bf16 tolerance, the neighbour
discrimination check, the pre-norm confirmation, and the verdict they feed.

The tolerance derivation is pure arithmetic and is tested without torch. The
comparison functions need tensors, so they carry the same module-level
``pytestmark`` skip as ``tests/test_activation_capture.py`` — declared as a
``pytestmark`` rather than an ``importorskip`` so the file always collects and
each test is reported as an explicit skip rather than vanishing.

To execute the tensor tests, put the vLLM venv on the path:

    PYTHONPATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/lib/python3.12/site-packages \\
        .venv/bin/python -m pytest tests/test_activation_capture_hf_reference.py -q
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from speedlm.activation_capture.compare import build_result
from speedlm.activation_capture.hf_reference import (
    BF16_DEPTH_OFFSET,
    BF16_UNIT_ROUNDOFF,
    DISCRIMINATION_MARGIN,
    FP32_BYTES_PER_PARAMETER,
    REFERENCE_DEVICE_HEADROOM_FRACTION,
    HFReferenceLayer,
    HFReferenceResult,
    ReferenceUnavailable,
    _checkpoint_quantization,
    _derive_reference_verdict,
    assert_reference_fits,
    bf16_relative_tolerance,
    checkpoint_parameter_count,
    compare_to_hf_reference,
    cosine_similarity,
    reference_fp32_bytes,
    reference_residual_stream,
    select_reference_device,
)

try:  # torch lives in the vLLM venv, not the project venv.
    import torch
except ImportError:  # pragma: no cover - depends on the interpreter in use
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(
    torch is None,
    reason=(
        "torch is not installed in the project venv; run with PYTHONPATH="
        "/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/lib/python3.12/"
        "site-packages to execute these tests"
    ),
)


# ---------------------------------------------------------------------------
# Tolerance derivation (no torch)
# ---------------------------------------------------------------------------


class TestBf16Tolerance:
    def test_unit_roundoff_is_two_to_the_minus_eight(self) -> None:
        """bf16 has 7 stored significand bits plus 1 implicit bit."""
        assert BF16_UNIT_ROUNDOFF == 2.0**-8

    def test_tolerance_grows_linearly_with_depth(self) -> None:
        """One bf16 materialization of the residual stream per decoder layer."""
        step = bf16_relative_tolerance(11) - bf16_relative_tolerance(10)
        assert step == pytest.approx(BF16_UNIT_ROUNDOFF)

    def test_layer_zero_is_the_depth_offset_alone(self) -> None:
        assert bf16_relative_tolerance(0) == pytest.approx(
            BF16_UNIT_ROUNDOFF * BF16_DEPTH_OFFSET
        )

    @pytest.mark.parametrize(
        ("layer", "expected"),
        [(2, 0.0234375), (18, 0.0859375), (33, 0.14453125), (36, 0.15625)],
    )
    def test_qwen3_8b_layers(self, layer: int, expected: float) -> None:
        """The values quoted in the module docstring must stay true.

        #: These are Qwen3-8B's actual aux layers ``[2, 18, 33]`` plus the
        #: appended final layer 36, as resolved from the drafter's
        #: ``eagle_aux_hidden_state_layer_ids`` at runtime.  The list previously
        #: asserted here, ``[2, 12, 21]``, was gpt-oss-20b's and was attributed
        #: to Qwen3-8B by mistake; it pinned a wrong worked example into the
        #: module docstring.  Nothing in the production path was affected --
        #: :func:`bf16_relative_tolerance` is called with the resolved layer id
        #: (``hf_reference.py`` ``for aux_idx in sorted(captured)``), never with
        #: a positional index -- but the documented example was wrong.
        """
        assert bf16_relative_tolerance(layer) == pytest.approx(expected)

    def test_tolerance_uses_the_layer_id_not_a_positional_index(self) -> None:
        """Guard the distinction the wrong-list bug made easy to miss.

        For aux layers ``[2, 18, 33, 36]`` the positional indices are
        ``0, 1, 2, 3``.  Feeding positions instead of ids would collapse every
        tolerance into the 1.6e-2..2.7e-2 band -- far too tight at depth 33 --
        and the comparison would silently target the wrong residual depth.
        """
        aux_layer_ids = [2, 18, 33, 36]
        by_id = [bf16_relative_tolerance(k) for k in aux_layer_ids]
        by_position = [
            bf16_relative_tolerance(i) for i in range(len(aux_layer_ids))
        ]
        assert by_id != by_position
        assert by_id == pytest.approx(
            [0.0234375, 0.0859375, 0.14453125, 0.15625]
        )
        assert max(by_position) < min(by_id[1:])

    def test_stays_far_below_a_wrong_quantity(self) -> None:
        """Even the deepest bound leaves an order of magnitude to O(1) error."""
        assert bf16_relative_tolerance(36) < 0.2

    def test_negative_layer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            bf16_relative_tolerance(-1)


class TestSelectReferenceDevice:
    def test_no_device_means_cpu(self) -> None:
        assert (
            select_reference_device(num_parameters=8_000_000_000, free_device_bytes=None)
            == "cpu"
        )

    def test_ample_vram_means_cuda(self) -> None:
        #: 8B params fp32 = 32 GB raw; 70 GB free clears the 1.35x headroom.
        assert (
            select_reference_device(
                num_parameters=8_000_000_000, free_device_bytes=70 * 1024**3
            )
            == "cuda"
        )

    def test_insufficient_vram_falls_back_to_cpu_not_oom(self) -> None:
        assert (
            select_reference_device(
                num_parameters=8_000_000_000, free_device_bytes=20 * 1024**3
            )
            == "cpu"
        )

    def test_headroom_is_actually_applied(self) -> None:
        """Raw weight bytes alone must not be enough; activations need room."""
        raw = 8_000_000_000 * 4
        assert (
            select_reference_device(
                num_parameters=8_000_000_000, free_device_bytes=raw
            )
            == "cpu"
        )


# ---------------------------------------------------------------------------
# Quantized checkpoints (no torch)
# ---------------------------------------------------------------------------


def _write_config(directory: Any, payload: Any) -> str:
    """Write ``config.json`` into *directory* and return the directory path."""
    path = directory / "config.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(directory)


class TestCheckpointQuantization:
    """``""`` must mean "unquantized", i.e. the load path that changes nothing."""

    def test_mxfp4_config_is_reported(self, tmp_path: Any) -> None:
        #: The real shape of ``openai/gpt-oss-20b``'s config.json: a
        #: quantization_config with no top-level dtype anywhere, which is why
        #: the dtype probe alone cannot tell it from an fp32 checkpoint.
        model_dir = _write_config(
            tmp_path,
            {
                "num_hidden_layers": 24,
                "quantization_config": {
                    "quant_method": "mxfp4",
                    "modules_to_not_convert": [
                        "model.layers.*.self_attn",
                        "model.layers.*.mlp.router",
                        "model.embed_tokens",
                        "lm_head",
                    ],
                },
            },
        )
        assert _checkpoint_quantization(model_dir) == "mxfp4"

    def test_the_method_is_lowercased(self, tmp_path: Any) -> None:
        model_dir = _write_config(
            tmp_path, {"quantization_config": {"quant_method": "MXFP4"}}
        )
        assert _checkpoint_quantization(model_dir) == "mxfp4"

    def test_nested_text_config_is_searched(self, tmp_path: Any) -> None:
        model_dir = _write_config(
            tmp_path,
            {"text_config": {"quantization_config": {"quant_method": "awq"}}},
        )
        assert _checkpoint_quantization(model_dir) == "awq"

    def test_plain_config_is_empty(self, tmp_path: Any) -> None:
        model_dir = _write_config(
            tmp_path, {"dtype": "bfloat16", "num_hidden_layers": 36}
        )
        assert _checkpoint_quantization(model_dir) == ""

    def test_missing_file_is_empty(self, tmp_path: Any) -> None:
        assert _checkpoint_quantization(str(tmp_path / "absent")) == ""

    def test_malformed_json_is_empty(self, tmp_path: Any) -> None:
        """Doubt must resolve to the load path that has always been taken."""
        model_dir = _write_config(tmp_path, "{not json at all")
        assert _checkpoint_quantization(model_dir) == ""

    def test_a_non_string_method_is_empty(self, tmp_path: Any) -> None:
        model_dir = _write_config(
            tmp_path, {"quantization_config": {"quant_method": 4}}
        )
        assert _checkpoint_quantization(model_dir) == ""


class TestCheckpointParameterCount:
    def test_index_estimate_is_half_the_byte_count(self, tmp_path: Any) -> None:
        """bf16 checkpoints: two bytes per parameter."""
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 16_384_000_000}}),
            encoding="utf-8",
        )
        assert checkpoint_parameter_count(str(tmp_path)) == 8_192_000_000

    def test_missing_index_is_none(self, tmp_path: Any) -> None:
        assert checkpoint_parameter_count(str(tmp_path)) is None

    def test_malformed_index_is_none(self, tmp_path: Any) -> None:
        (tmp_path / "model.safetensors.index.json").write_text(
            "{broken", encoding="utf-8"
        )
        assert checkpoint_parameter_count(str(tmp_path)) is None

    def test_nonpositive_total_size_is_none(self, tmp_path: Any) -> None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}}), encoding="utf-8"
        )
        assert checkpoint_parameter_count(str(tmp_path)) is None

    def test_mxfp4_counts_two_elements_per_stored_block_byte(
        self, tmp_path: Any
    ) -> None:
        """The arithmetic the index estimate gets wrong by 3.04x on gpt-oss-20b.

        A real (tiny) safetensors file is written rather than a mock, because
        the claim under test is about the ``get_slice(...).get_shape()`` surface
        and the ``_blocks`` / ``_scales`` naming convention — a mock would only
        restate this test's own assumptions back to it.

        ``_blocks`` is uint8 storage holding TWO e2m1 nibbles per byte, so it
        contributes ``2x`` its element count.  ``_scales`` is a per-block e8m0
        exponent consumed by the unpacking, so it contributes ``0``.
        """
        torch_mod = pytest.importorskip("torch")
        safetensors_torch = pytest.importorskip("safetensors.torch")

        tensors = {
            #: 4 x 8 = 32 stored bytes -> 64 dequantized elements.
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch_mod.zeros(
                4, 8, dtype=torch_mod.uint8
            ),
            #: 4 x 2 = 8 block exponents -> 0 parameters.
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch_mod.zeros(
                4, 2, dtype=torch_mod.uint8
            ),
            #: Untouched by the quantizer -> 3 x 5 = 15 parameters.
            "model.layers.0.self_attn.q_proj.weight": torch_mod.zeros(
                3, 5, dtype=torch_mod.float32
            ),
        }
        safetensors_torch.save_file(
            tensors, str(tmp_path / "model-00001-of-00001.safetensors")
        )
        _write_config(tmp_path, {"quantization_config": {"quant_method": "mxfp4"}})

        assert checkpoint_parameter_count(str(tmp_path)) == 2 * 32 + 0 + 15

    def test_mxfp4_ignores_the_index_estimate_entirely(
        self, tmp_path: Any
    ) -> None:
        """An index present alongside mxfp4 shards must not win.

        This is the trap: gpt-oss-20b HAS an index, its ``total_size // 2``
        answers 6,880,658,452 against a true 20,914,757,184, and acting on the
        undercount picks ``cuda`` for an 83.66 GB fp32 copy.
        """
        torch_mod = pytest.importorskip("torch")
        safetensors_torch = pytest.importorskip("safetensors.torch")

        safetensors_torch.save_file(
            {"w_blocks": torch_mod.zeros(10, 10, dtype=torch_mod.uint8)},
            str(tmp_path / "model-00001-of-00001.safetensors"),
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 999_999_999}}), encoding="utf-8"
        )
        _write_config(tmp_path, {"quantization_config": {"quant_method": "mxfp4"}})

        assert checkpoint_parameter_count(str(tmp_path)) == 200

    def test_mxfp4_without_shards_is_none(self, tmp_path: Any) -> None:
        pytest.importorskip("safetensors")
        _write_config(tmp_path, {"quantization_config": {"quant_method": "mxfp4"}})
        assert checkpoint_parameter_count(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# fp32 memory feasibility (no torch)
# ---------------------------------------------------------------------------

#: gpt-oss-20b, measured by summing the real safetensors shards: 11,956,805,184
#: raw stored elements dequantize to this, whose fp32 copy is 83.66 GB.
_GPT_OSS_20B_DEQUANTIZED_PARAMS = 20_914_757_184


class TestReferenceFp32Bytes:
    def test_four_bytes_per_parameter(self) -> None:
        assert FP32_BYTES_PER_PARAMETER == 4
        assert reference_fp32_bytes(1_000) == 4_000

    def test_gpt_oss_20b_is_eighty_three_gigabytes(self) -> None:
        """The number the refusal message has to be able to quote."""
        assert (
            reference_fp32_bytes(_GPT_OSS_20B_DEQUANTIZED_PARAMS) / 1e9
            == pytest.approx(83.66, abs=0.01)
        )

    def test_zero_is_zero(self) -> None:
        assert reference_fp32_bytes(0) == 0

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            reference_fp32_bytes(-1)


class TestAssertReferenceFits:
    """A reference that cannot be computed must raise, not degrade or vanish."""

    def test_a_fitting_cuda_reference_passes(self) -> None:
        assert_reference_fits(
            "/models/qwen3-8b",
            num_parameters=8_000_000_000,
            free_device_bytes=70 * 1024**3,
            free_host_bytes=60 * 1024**3,
            device="cuda",
        )

    def test_a_fitting_cpu_reference_passes(self) -> None:
        assert_reference_fits(
            "/models/qwen3-8b",
            num_parameters=8_000_000_000,
            free_device_bytes=None,
            free_host_bytes=60 * 1024**3,
            device="cpu",
        )

    def test_cuda_refusal_shows_the_arithmetic(self) -> None:
        with pytest.raises(ReferenceUnavailable) as excinfo:
            assert_reference_fits(
                "/models/gpt-oss-20b",
                num_parameters=_GPT_OSS_20B_DEQUANTIZED_PARAMS,
                free_device_bytes=79 * 1024**3,
                free_host_bytes=63 * 1000**3,
                device="cuda",
            )
        message = str(excinfo.value)
        assert "/models/gpt-oss-20b" in message
        assert "20,914,757,184 params x 4 B" in message
        assert "83.66 GB" in message
        assert "device=cuda" in message
        #: The remedy must be named, not implied.
        assert "larger-memory node" in message
        assert excinfo.value.required_bytes == 83_659_028_736

    def test_cpu_refusal_shows_the_arithmetic(self) -> None:
        """CPU is not a fallback for this checkpoint; it is a second way to die."""
        with pytest.raises(ReferenceUnavailable) as excinfo:
            assert_reference_fits(
                "/models/gpt-oss-20b",
                num_parameters=_GPT_OSS_20B_DEQUANTIZED_PARAMS,
                free_device_bytes=None,
                free_host_bytes=63_000_000_000,
                device="cpu",
            )
        message = str(excinfo.value)
        assert "20,914,757,184 params x 4 B" in message
        assert "83.66 GB" in message
        assert "device=cpu has 63.0 GB free" in message
        assert excinfo.value.available_bytes == 63_000_000_000

    def test_a_forced_device_is_still_checked(self) -> None:
        """An operator forcing cuda gets a refusal, not an OOM kill."""
        with pytest.raises(ReferenceUnavailable, match="cannot run"):
            assert_reference_fits(
                "/models/gpt-oss-20b",
                num_parameters=_GPT_OSS_20B_DEQUANTIZED_PARAMS,
                free_device_bytes=80 * 1024**3,
                free_host_bytes=10**15,
                device="cuda",
            )

    def test_cuda_applies_the_same_headroom_as_the_device_choice(self) -> None:
        """Raw weight bytes alone must not admit what select_reference_device rejects.

        The two must answer "does it fit" the same way; a device chosen under
        one budget and admitted under another is exactly the OOM this refusal
        exists to prevent.
        """
        params = 8_000_000_000
        raw = reference_fp32_bytes(params)
        assert (
            select_reference_device(
                num_parameters=params, free_device_bytes=raw
            )
            == "cpu"
        )
        with pytest.raises(ReferenceUnavailable):
            assert_reference_fits(
                "/models/qwen3-8b",
                num_parameters=params,
                free_device_bytes=raw,
                free_host_bytes=10**15,
                device="cuda",
            )
        #: ...and it is admitted once the headroom is actually there.
        assert_reference_fits(
            "/models/qwen3-8b",
            num_parameters=params,
            free_device_bytes=int(raw * REFERENCE_DEVICE_HEADROOM_FRACTION),
            free_host_bytes=10**15,
            device="cuda",
        )

    def test_unreadable_free_memory_refuses_rather_than_guesses(self) -> None:
        with pytest.raises(ReferenceUnavailable, match="could not be read"):
            assert_reference_fits(
                "/models/qwen3-8b",
                num_parameters=8_000_000_000,
                free_device_bytes=None,
                free_host_bytes=None,
                device="cpu",
            )

    def test_an_unknown_device_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 'cuda' or 'cpu'"):
            assert_reference_fits(
                "/models/qwen3-8b",
                num_parameters=1,
                free_device_bytes=10**12,
                free_host_bytes=10**12,
                device="mps",
            )


# ---------------------------------------------------------------------------
# Verdict derivation (no torch)
# ---------------------------------------------------------------------------


def _layer(
    idx: int,
    *,
    within: bool = True,
    identified: bool = True,
) -> HFReferenceLayer:
    return HFReferenceLayer(
        aux_layer_idx=idx,
        reference_index=idx,
        rows=8,
        mean_rel_error=0.001,
        cosine_similarity=0.999,
        tolerance=bf16_relative_tolerance(idx),
        within_tolerance=within,
        neighbour_rel_errors={str(idx): 0.001},
        best_match_index=idx if identified else idx + 1,
        discrimination_ratio=10.0 if identified else 1.0,
        identified=identified,
    )


class TestReferenceVerdict:
    def test_empty_is_not_a_pass(self) -> None:
        assert _derive_reference_verdict([], None) == "FAIL_empty"

    def test_all_good_passes(self) -> None:
        assert _derive_reference_verdict([_layer(2), _layer(12)], True) == "PASS"

    def test_out_of_tolerance_fails(self) -> None:
        assert (
            _derive_reference_verdict([_layer(2), _layer(12, within=False)], True)
            == "FAIL_tolerance"
        )

    def test_unidentified_layer_fails_even_within_tolerance(self) -> None:
        """Tolerance alone is not identification; an off-by-one must still fail."""
        assert (
            _derive_reference_verdict([_layer(12, identified=False)], True)
            == "FAIL_identification"
        )

    def test_post_norm_capture_fails(self) -> None:
        assert _derive_reference_verdict([_layer(2)], False) == "FAIL_pre_norm"

    def test_absent_final_layer_does_not_fail(self) -> None:
        """A run that never captured the final layer has nothing to confirm."""
        assert _derive_reference_verdict([_layer(2)], None) == "PASS"


# ---------------------------------------------------------------------------
# Comparison against a synthetic reference stream (needs torch)
# ---------------------------------------------------------------------------


def _stream(depth: int, rows: int = 6, hidden: int = 16) -> list:
    """A synthetic residual stream where each depth is a distinguishable step.

    Consecutive entries differ by a fixed-magnitude increment, so an off-by-one
    reference index produces a relative error far above the bf16 tolerance —
    which is what the discrimination check must detect.
    """
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(rows, hidden, generator=generator)
    return [base + k * torch.ones(rows, hidden) for k in range(depth + 1)]


@requires_torch
class TestCompareToHFReference:
    def test_exact_match_identifies_the_right_layer(self) -> None:
        stream = _stream(5)
        captured = {2: stream[2].clone(), 4: stream[4].clone()}
        result = compare_to_hf_reference(
            captured,
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.verdict == "PASS"
        assert [layer.aux_layer_idx for layer in result.layers] == [2, 4]
        assert all(layer.best_match_index == layer.aux_layer_idx for layer in result.layers)

    def test_bf16_rounding_still_passes(self) -> None:
        """A real capture is bf16; that must be within the derived tolerance."""
        stream = _stream(5)
        captured = {2: stream[2].to(torch.bfloat16).float()}
        result = compare_to_hf_reference(
            captured,
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.verdict == "PASS"
        assert result.layers[0].mean_rel_error <= result.layers[0].tolerance

    def test_off_by_one_layer_is_caught(self) -> None:
        """A capture that is really layer 3 must not pass as layer 2.

        This is the check the vLLM-vs-vLLM transport comparison cannot make:
        both of its sides would carry the same off-by-one and agree exactly.
        """
        stream = _stream(5)
        captured = {2: stream[3].clone()}
        result = compare_to_hf_reference(
            captured,
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.layers[0].best_match_index == 3
        assert not result.layers[0].identified
        assert result.verdict != "PASS"

    def test_neighbour_errors_are_reported_for_both_sides(self) -> None:
        stream = _stream(5)
        result = compare_to_hf_reference(
            {2: stream[2].clone()},
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert set(result.layers[0].neighbour_rel_errors) == {"1", "2", "3"}

    def test_edge_layer_has_only_one_neighbour(self) -> None:
        stream = _stream(5)
        result = compare_to_hf_reference(
            {0: stream[0].clone()},
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert set(result.layers[0].neighbour_rel_errors) == {"0", "1"}
        assert result.layers[0].identified

    def test_pre_norm_capture_is_confirmed(self) -> None:
        stream = _stream(5)
        #: A plausible post-norm tensor: same direction, unit-ish scale.
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {5: stream[5].clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.final_layer_prenorm_confirmed is True
        assert result.verdict == "PASS"

    def test_post_norm_capture_is_rejected(self) -> None:
        """The regression target must be pre-norm; a normalized one must fail."""
        stream = _stream(5)
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {5: post_norm.clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.final_layer_prenorm_confirmed is False
        assert result.verdict != "PASS"

    def test_rows_beyond_the_prompt_are_ignored(self) -> None:
        """Only prompt rows ran the same tokens on both sides."""
        stream = _stream(5, rows=6)
        captured = {2: torch.cat([stream[2], torch.full((4, 16), 99.0)], dim=0)}
        result = compare_to_hf_reference(
            captured,
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.verdict == "PASS"
        assert result.layers[0].rows == 6

    def test_too_few_captured_rows_raises(self) -> None:
        stream = _stream(5, rows=6)
        with pytest.raises(ValueError, match="fewer than the prompt"):
            compare_to_hf_reference(
                {2: stream[2][:3]},
                stream,
                stream[5],
                prompt_token_count=6,
                final_layer_idx=None,
                device="cpu",
                dtype="torch.float32",
            )

    def test_aux_id_outside_the_stream_raises(self) -> None:
        stream = _stream(5)
        with pytest.raises(ValueError, match="no counterpart"):
            compare_to_hf_reference(
                {9: stream[2]},
                stream,
                stream[5],
                prompt_token_count=6,
                final_layer_idx=None,
                device="cpu",
                dtype="torch.float32",
            )

    def test_zero_prompt_tokens_raises(self) -> None:
        stream = _stream(5)
        with pytest.raises(ValueError, match="must be positive"):
            compare_to_hf_reference(
                {2: stream[2]},
                stream,
                stream[5],
                prompt_token_count=0,
                final_layer_idx=None,
                device="cpu",
                dtype="torch.float32",
            )


# ---------------------------------------------------------------------------
# Two-sided discrimination for the FINAL aux layer (needs torch)
# ---------------------------------------------------------------------------


@requires_torch
class TestFinalLayerTwoSidedDiscrimination:
    """The final aux layer's argmin must have a rival on the far side too.

    ``k+1`` does not exist in the residual stream for ``k == L``, so the
    integer candidate set clips it away and only ``k-1`` competes.  The real
    other side is ``post_norm_final`` — the only quantity the model produces
    downstream of layer ``L`` — which used to be spent solely on the separate
    ``final_layer_prenorm_confirmed`` check.
    """

    def test_post_norm_capture_is_not_identified(self) -> None:
        """The regression this change buys.

        The dangerous post-norm tensor is not a wildly different one — it is a
        *mild rescaling* of layer ``L``, which is what an RMSNorm whose learned
        gains sit near one actually produces.  Against ``k-1`` alone such a
        capture looks excellent: it clears the 3x margin comfortably, because
        the only competitor is a whole decoder layer away.  Nothing above it
        exists in the stream to object, so the one-sided argmin identified it
        as layer ``L``.

        With ``post_norm_final`` in the rival set its true source competes, and
        wins outright.
        """
        stream = _stream(5)
        #: Close enough to layer 5 to beat layer 4 easily; that is the trap.
        post_norm = stream[5] * 1.02
        result = compare_to_hf_reference(
            {5: post_norm.clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        layer = result.layers[0]
        rivals = dict(layer.rival_errors)

        #: What the OLD one-sided check saw: k-1 as the sole rival, clearing
        #: the margin. Pinned so the regression cannot be quietly reintroduced.
        assert rivals["4"] / layer.mean_rel_error > DISCRIMINATION_MARGIN

        #: What the two-sided check sees: the capture IS the post-norm tensor,
        #: so that rival is an exact match and beats the claimed layer.
        assert rivals["post_norm"] < layer.mean_rel_error
        assert not layer.identified
        assert layer.discrimination_ratio < DISCRIMINATION_MARGIN
        assert result.verdict != "PASS"

    def test_true_final_layer_is_still_identified(self) -> None:
        """Adding a rival must not cost a genuine capture its identification."""
        stream = _stream(5)
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {5: stream[5].clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        layer = result.layers[0]
        assert "post_norm" in dict(layer.rival_errors)
        assert layer.identified
        assert layer.two_sided
        assert result.verdict == "PASS"

    def test_bf16_final_layer_still_beats_the_post_norm_rival(self) -> None:
        """A realistic bf16 capture, not just a bit-exact one, must survive."""
        stream = _stream(5)
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {5: stream[5].to(torch.bfloat16).float()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        assert result.layers[0].identified
        assert result.verdict == "PASS"

    def test_rival_errors_and_two_sided_for_final_vs_middle(self) -> None:
        """Labels differ by position; both ends of a real stack are bracketed."""
        stream = _stream(5)
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {2: stream[2].clone(), 5: stream[5].clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        middle, final = result.layers
        assert set(dict(middle.rival_errors)) == {"1", "3"}
        assert middle.two_sided
        #: The final layer's upper rival is the post-norm tensor, not "6".
        assert set(dict(final.rival_errors)) == {"4", "post_norm"}
        assert final.two_sided

    def test_layer_zero_is_reported_as_one_sided(self) -> None:
        """Layer 0 has no lower rival; the flag must say so rather than lie."""
        stream = _stream(5)
        result = compare_to_hf_reference(
            {0: stream[0].clone()},
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
        )
        assert set(dict(result.layers[0].rival_errors)) == {"1"}
        assert not result.layers[0].two_sided

    def test_rival_errors_survive_serialization(self) -> None:
        stream = _stream(5)
        post_norm = stream[5] / stream[5].norm(dim=-1, keepdim=True)
        result = compare_to_hf_reference(
            {5: stream[5].clone()},
            stream,
            post_norm,
            prompt_token_count=6,
            final_layer_idx=5,
            device="cpu",
            dtype="torch.float32",
        )
        payload = json.loads(json.dumps(result.to_dict()))
        entry = payload["layers"][0]
        assert entry["two_sided"] is True
        assert [label for label, _error in entry["rival_errors"]] == [
            "4",
            "post_norm",
        ]


# ---------------------------------------------------------------------------
# Why the rival is post_norm and not k-2 (no torch)
# ---------------------------------------------------------------------------


class TestKMinusTwoIsANoOp:
    """Documents why ``k-2`` was rejected as the final layer's second rival.

    The stated worry is an error metric that grows monotonically as you move
    away from the true layer *in the direction of the final layer*.  Under
    exactly that assumption ``err(k-2) > err(k-1)``, so ``k-2`` can never be
    ``min(rivals)`` and the pass/fail outcome is bit-identical with or without
    it.  A rival that cannot change the verdict is not discrimination, which is
    why the genuinely downstream quantity (``post_norm``) is used instead.
    """

    #: A monotonic profile around a true layer 36: error grows with distance.
    _MONOTONIC = {34: 0.42, 35: 0.21, 36: 0.005}

    def test_adding_k_minus_two_does_not_change_min_rivals(self) -> None:
        own = self._MONOTONIC[36]
        one_sided = [self._MONOTONIC[35]]
        with_k_minus_two = [self._MONOTONIC[35], self._MONOTONIC[34]]
        assert min(with_k_minus_two) == min(one_sided)
        assert (
            min(with_k_minus_two) / own == min(one_sided) / own
        )

    def test_the_post_norm_rival_can_change_min_rivals(self) -> None:
        """The contrast: a downstream rival is the one that can actually bite."""
        own = self._MONOTONIC[36]
        one_sided = [self._MONOTONIC[35]]
        with_post_norm = [self._MONOTONIC[35], 0.0005]
        assert min(with_post_norm) < min(one_sided)
        assert min(with_post_norm) / own < DISCRIMINATION_MARGIN


# ---------------------------------------------------------------------------
# Prompt identity (no torch)
# ---------------------------------------------------------------------------


def _result(**overrides: Any) -> HFReferenceResult:
    kwargs: dict[str, Any] = {
        "device": "cpu",
        "dtype": "torch.float32",
        "prompt_token_count": 8,
        "layers": [_layer(2)],
        "final_layer_idx": None,
        "final_layer_prenorm_confirmed": None,
        "verdict": "PASS",
    }
    kwargs.update(overrides)
    return HFReferenceResult(**kwargs)


class TestPromptLabel:
    """A multi-prompt matrix whose rows cannot be attributed is unusable."""

    def test_defaults_to_none_so_existing_callers_are_unchanged(self) -> None:
        assert _result().prompt_label is None

    def test_round_trips_into_the_serialized_dict(self) -> None:
        payload = json.loads(json.dumps(_result(prompt_label="short_en").to_dict()))
        assert payload["prompt_label"] == "short_en"

    def test_absent_label_serializes_as_null_not_missing(self) -> None:
        """A missing key and a null mean different things to a consumer."""
        payload = json.loads(json.dumps(_result().to_dict()))
        assert "prompt_label" in payload
        assert payload["prompt_label"] is None


@requires_torch
class TestPromptLabelThreading:
    def test_compare_threads_the_label_through(self) -> None:
        stream = _stream(5)
        result = compare_to_hf_reference(
            {2: stream[2].clone()},
            stream,
            stream[5],
            prompt_token_count=6,
            final_layer_idx=None,
            device="cpu",
            dtype="torch.float32",
            prompt_label="prompt_3",
        )
        assert result.prompt_label == "prompt_3"
        assert json.loads(json.dumps(result.to_dict()))["prompt_label"] == "prompt_3"


# ---------------------------------------------------------------------------
# Reference model reuse across prompts (no torch -- torch is faked)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """The narrow surface ``_forward_residual_stream`` touches on a tensor."""

    def __getitem__(self, index: Any) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self


class _FakeHandle:
    def __init__(self, layer: _FakeLayer, hook: Any) -> None:
        self._layer = layer
        self._hook = hook

    def remove(self) -> None:
        self._layer.hooks.remove(self._hook)


class _FakeLayer:
    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> _FakeHandle:
        self.hooks.append(hook)
        return _FakeHandle(self, hook)


class _FakeModel:
    """Counts forwards and fires the last layer's hook like a real module."""

    def __init__(self, num_layers: int = 3) -> None:
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.model = types.SimpleNamespace(layers=self.layers)
        self.forwards = 0

    def eval(self) -> None:
        return None

    def to(self, device: str) -> None:
        return None

    def float(self) -> None:
        return None

    def __call__(self, **_kwargs: Any) -> Any:
        self.forwards += 1
        last = self.layers[-1]
        for hook in list(last.hooks):
            hook(last, (), _FakeTensor())
        return types.SimpleNamespace(
            hidden_states=tuple(_FakeTensor() for _ in range(len(self.layers) + 1))
        )


class _LoadCounter:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls = 0

    def from_pretrained(self, *_args: Any, **_kwargs: Any) -> _FakeModel:
        self.calls += 1
        return self.model


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> _LoadCounter:
    """Install fake ``torch`` and ``transformers`` modules for the lazy imports.

    ``reference_residual_stream`` imports both inside the function body, so
    swapping ``sys.modules`` is enough and the test runs in the PROJECT venv
    where neither package is installed. That matters: the point of this test is
    to count *loads*, and a test that only runs under a hand-set PYTHONPATH
    would not be counting them in the default suite.
    """
    model = _FakeModel()
    loader = _LoadCounter(model)
    float32 = object()
    fake_torch = types.SimpleNamespace(
        float32=float32,
        long=object(),
        tensor=lambda *_a, **_k: _FakeTensor(),
        no_grad=lambda: _NullContext(),
        cuda=types.SimpleNamespace(empty_cache=lambda: None),
    )
    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=loader)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return loader


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> bool:
        return False


class TestReferenceModelReuse:
    """A multi-prompt Stage 0 must not reload ~30 GiB of fp32 weights per prompt."""

    def test_without_model_the_reference_is_loaded_once(
        self, fake_hf: _LoadCounter, tmp_path: Any
    ) -> None:
        stream, _post_norm, dtype_name = reference_residual_stream(
            str(tmp_path), [1, 2, 3]
        )
        assert fake_hf.calls == 1
        assert fake_hf.model.forwards == 1
        #: L + 1 entries: L from the HF tuple plus the hooked pre-norm final.
        assert len(stream) == len(fake_hf.model.layers) + 1
        assert dtype_name == "torch.float32"

    def test_passing_model_performs_zero_loads(
        self, fake_hf: _LoadCounter, tmp_path: Any
    ) -> None:
        """The whole point of the reuse path."""
        reference_residual_stream(str(tmp_path), [1, 2, 3], model=fake_hf.model)
        assert fake_hf.calls == 0
        assert fake_hf.model.forwards == 1

    def test_reuse_across_several_prompts_loads_nothing(
        self, fake_hf: _LoadCounter, tmp_path: Any
    ) -> None:
        for _ in range(4):
            reference_residual_stream(str(tmp_path), [1, 2, 3], model=fake_hf.model)
        assert fake_hf.calls == 0
        assert fake_hf.model.forwards == 4

    def test_the_hook_is_removed_on_every_forward(
        self, fake_hf: _LoadCounter, tmp_path: Any
    ) -> None:
        """A hook left installed would accumulate and corrupt later prompts."""
        for _ in range(3):
            reference_residual_stream(str(tmp_path), [1, 2, 3], model=fake_hf.model)
        assert fake_hf.model.layers[-1].hooks == []

    def test_a_short_hidden_state_tuple_is_rejected(
        self, fake_hf: _LoadCounter, tmp_path: Any
    ) -> None:
        """The index mapping is asserted on BOTH paths, not just the load one."""
        with pytest.raises(AssertionError, match="hidden states"):
            reference_residual_stream(
                str(tmp_path), [1, 2, 3], model=_ShortModel()
            )


class _ShortModel(_FakeModel):
    """A model whose forward returns too few hidden states.

    The ``L + 1`` assertion is what pins the aux-layer index mapping, so it has
    to hold on the reuse path too — that path skips the loader, not the checks.
    """

    def __call__(self, **_kwargs: Any) -> Any:
        last = self.layers[-1]
        for hook in list(last.hooks):
            hook(last, (), _FakeTensor())
        return types.SimpleNamespace(hidden_states=(_FakeTensor(),))


# ---------------------------------------------------------------------------
# Integration with the overall Stage 0 verdict (needs torch)
# ---------------------------------------------------------------------------


def _reference(verdict: str) -> HFReferenceResult:
    return HFReferenceResult(
        device="cuda",
        dtype="torch.float32",
        prompt_token_count=8,
        layers=[_layer(2)],
        final_layer_idx=36,
        final_layer_prenorm_confirmed=True,
        verdict=verdict,
    )


@requires_torch
class TestComparisonResultIntegration:
    def test_absent_reference_does_not_change_the_verdict(self) -> None:
        """A None reference must not flip pre-existing callers to FAIL."""
        t = torch.ones(4, 8)
        result = build_result(
            {0: t}, {0: t}, captured_final_pre_norm=t, offline_final_pre_norm=t
        )
        assert result.verdict == "PASS"
        assert result.hf_reference is None

    def test_failing_reference_fails_the_whole_result(self) -> None:
        """A bit-perfect transport must not pass when the identity check fails."""
        t = torch.ones(4, 8)
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
            hf_reference=_reference("FAIL_identification"),
        )
        assert result.verdict == "FAIL_hf_reference"

    def test_passing_reference_passes(self) -> None:
        t = torch.ones(4, 8)
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
            hf_reference=_reference("PASS"),
        )
        assert result.verdict == "PASS"

    def test_reference_is_serialized(self) -> None:
        t = torch.ones(4, 8)
        result = build_result(
            {0: t},
            {0: t},
            captured_final_pre_norm=t,
            offline_final_pre_norm=t,
            hf_reference=_reference("PASS"),
        )
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["hf_reference"]["verdict"] == "PASS"
        assert payload["hf_reference"]["device"] == "cuda"
        assert payload["hf_reference"]["discrimination_margin"] == DISCRIMINATION_MARGIN
        assert payload["hf_reference"]["layers"][0]["aux_layer_idx"] == 2


# ---------------------------------------------------------------------------
# Tolerance budget (no torch)
# ---------------------------------------------------------------------------


class TestToleranceBudgetUsed:
    """The tolerance gate is a ceiling; the budget field is what says so."""

    def test_budget_is_error_over_tolerance(self) -> None:
        layer = _layer(2)
        assert layer.tolerance_budget_used == pytest.approx(
            layer.mean_rel_error / layer.tolerance, rel=1e-9
        )

    def test_a_passing_layer_can_use_a_small_fraction_of_the_budget(self) -> None:
        """A pass is not a tight result — this is the whole point of the field.

        Job 369256's aux layer 2 measured 0.00511 against a derived 0.0234.
        """
        layer = HFReferenceLayer(
            aux_layer_idx=2,
            reference_index=2,
            rows=18,
            mean_rel_error=0.00511,
            cosine_similarity=0.9999,
            tolerance=bf16_relative_tolerance(2),
            within_tolerance=True,
            neighbour_rel_errors={"1": 0.416, "2": 0.00511, "3": 0.416},
            best_match_index=2,
            discrimination_ratio=81.46,
            identified=True,
        )
        assert layer.within_tolerance
        assert layer.tolerance_budget_used == pytest.approx(0.218, abs=0.01)

    def test_budget_is_serialized(self) -> None:
        result = HFReferenceResult(
            device="cpu",
            dtype="torch.float32",
            prompt_token_count=8,
            layers=[_layer(2)],
            final_layer_idx=None,
            final_layer_prenorm_confirmed=None,
            verdict="PASS",
        )
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["layers"][0]["tolerance_budget_used"] == pytest.approx(
            _layer(2).tolerance_budget_used, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Cosine similarity accumulator (needs torch)
# ---------------------------------------------------------------------------


@requires_torch
class TestCosineSimilarity:
    """Regression tests for the >1.0 cosines job 369256 wrote to disk.

    The transport leg compares two views of ONE tensor, so its inputs are
    bit-identical and the only correct cosine is exactly 1.0.  A float32
    accumulator returned 1.0000027857 / 1.0000187356 / 1.0000228824 instead.
    """

    def _residual_like(self, rows: int) -> torch.Tensor:
        #: Shaped and scaled like a real residual stream: bf16 storage, a few
        #: thousand hidden units, and a magnitude range wide enough that the
        #: squares overflow float32's useful precision when summed.
        generator = torch.Generator().manual_seed(0)
        t = torch.randn(rows, 2880, generator=generator) * 20.0
        t[0, 0] = 2.0e4
        return t.bfloat16()

    @pytest.mark.parametrize("rows", [18, 165, 213])
    def test_bitwise_identical_is_exactly_one(self, rows: int) -> None:
        t = self._residual_like(rows)
        other = t.clone()
        assert torch.equal(t, other)
        assert cosine_similarity(t, other) == 1.0

    def test_never_exceeds_one(self) -> None:
        """The clamp makes an impossible value unrepresentable, not just rare."""
        t = self._residual_like(213)
        assert cosine_similarity(t, t.clone()) <= 1.0

    def test_anti_parallel_is_exactly_minus_one(self) -> None:
        t = self._residual_like(18)
        assert cosine_similarity(t, -t) == -1.0

    def test_orthogonal_is_zero(self) -> None:
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-12)

    def test_zero_reference_does_not_divide_by_zero(self) -> None:
        a = torch.ones(4, 8)
        b = torch.zeros(4, 8)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_a_real_difference_still_shows(self) -> None:
        """The clamp must not flatten a genuine mismatch to 1.0."""
        a = self._residual_like(18)
        b = a.float().clone()
        b[:, :1440] *= -1.0
        assert cosine_similarity(a, b) < 0.9
