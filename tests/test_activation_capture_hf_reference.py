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

import pytest

from speedlm.activation_capture.compare import build_result
from speedlm.activation_capture.hf_reference import (
    BF16_DEPTH_OFFSET,
    BF16_UNIT_ROUNDOFF,
    DISCRIMINATION_MARGIN,
    HFReferenceLayer,
    HFReferenceResult,
    _derive_reference_verdict,
    bf16_relative_tolerance,
    compare_to_hf_reference,
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
