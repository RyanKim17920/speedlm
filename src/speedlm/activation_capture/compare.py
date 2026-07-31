"""Elementwise comparison of captured vs. offline activation tensors.

This module implements the comparison logic for Stage 0 of the serving-time
activation capture prototype. It compares two stacks of per-layer tensors
(captured at serving time vs. extracted offline) and produces a machine-readable
verdict.

Tolerance rationale:
Both paths run the same model weights through the same vLLM kernel selection
on the same GPU. The dominant sources of numerical difference are:

1. **Batch composition** (prefill vs. decode kernel): serving captures a single-
   token decode or a short prefill batch; offline extraction typically re-runs
   as a larger prefill. Different ``BLOCK_SIZE_M`` choices in fused MoE kernels
   can produce different floating-point reduction orders.

2. **CUDA kernel ordering**: the order in which kernels launch can differ
   between the two engine instances, affecting accumulated rounding.

3. **dtype**: both paths use bf16, so no cross-dtype conversion is expected.

A per-element absolute tolerance of **1e-2** is chosen as a conservative bound.
It is loose enough to absorb batch-composition differences but tight enough to
flag a genuinely wrong quantity (e.g., post-norm vs. pre-norm). The tolerance
is documented rather than hidden behind an assertion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only executed by static type checkers. mypy has per-module overrides for
    # torch/safetensors/vllm (see pyproject.toml [[tool.mypy.overrides]]) so
    # this does not require torch to be installed in the project venv.
    from torch import Tensor

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

#: Per-element absolute tolerance for bf16 activation comparison.
#: See module docstring for derivation.
DEFAULT_TOLERANCE: float = 1e-2


# ---------------------------------------------------------------------------
# Per-layer result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerComparison:
    """Result of comparing a single layer's activations."""

    layer_idx: int
    captured_shape: tuple[int, ...]
    offline_shape: tuple[int, ...]
    shape_match: bool
    max_abs_diff: float | None  # None if shapes differ
    mean_abs_diff: float | None  # None if shapes differ


# ---------------------------------------------------------------------------
# Prefix-cache coverage test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixCacheResult:
    """Result of the prefix-cache coverage measurement."""

    prompt_token_count: int
    captured_row_count: int
    cache_hit: bool
    rows_missing: int


# ---------------------------------------------------------------------------
# Overall result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Machine-readable comparison between captured and offline activations."""

    layers: list[LayerComparison]
    pre_norm_match: bool | None  # None if final-layer data absent
    prefix_cache_test: PrefixCacheResult | None = None
    tolerance: float = DEFAULT_TOLERANCE
    verdict: str = ""  # Set by derive_verdict

    # -- derived helpers --

    @property
    def all_shapes_match(self) -> bool:
        return all(lc.shape_match for lc in self.layers)

    @property
    def all_within_tolerance(self) -> bool:
        return all(
            lc.max_abs_diff is not None and lc.max_abs_diff <= self.tolerance
            for lc in self.layers
        )

    # -- serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "layers": [
                {
                    "layer_idx": lc.layer_idx,
                    "captured_shape": list(lc.captured_shape),
                    "offline_shape": list(lc.offline_shape),
                    "shape_match": lc.shape_match,
                    "max_abs_diff": lc.max_abs_diff,
                    "mean_abs_diff": lc.mean_abs_diff,
                }
                for lc in self.layers
            ],
            "pre_norm_match": self.pre_norm_match,
            "prefix_cache_test": (
                asdict(self.prefix_cache_test)
                if self.prefix_cache_test is not None
                else None
            ),
            "tolerance": self.tolerance,
            "verdict": self.verdict,
            "all_shapes_match": self.all_shapes_match,
            "all_within_tolerance": self.all_within_tolerance,
        }

    def write_json(self, path: Path) -> None:
        """Write the result as a pretty-printed JSON file."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_layerwise(
    captured: dict[int, Tensor],
    offline: dict[int, Tensor],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[LayerComparison]:
    """Compare captured and offline activations per layer.

    Args:
        captured: mapping of layer index to tensor (shape: seq x hidden).
        offline: same schema from the offline extraction path.
        tolerance: absolute per-element tolerance (stored but not used here;
            the caller checks tolerance via :func:`check_within_tolerance`).

    Returns:
        List of per-layer comparisons, sorted by layer index.
    """
    del tolerance  # tolerance is checked at result level
    all_indices = sorted(set(captured.keys()) | set(offline.keys()))
    results: list[LayerComparison] = []
    for idx in all_indices:
        c = captured.get(idx)
        o = offline.get(idx)
        if c is None or o is None:
            c_shape = list(c.shape) if c is not None else []
            o_shape = list(o.shape) if o is not None else []
            results.append(
                LayerComparison(
                    layer_idx=idx,
                    captured_shape=tuple(c_shape),
                    offline_shape=tuple(o_shape),
                    shape_match=False,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                )
            )
            continue
        shape_match = c.shape == o.shape
        if shape_match:
            diff = (c - o).abs()
            results.append(
                LayerComparison(
                    layer_idx=idx,
                    captured_shape=c.shape,
                    offline_shape=o.shape,
                    shape_match=True,
                    max_abs_diff=float(diff.max()),
                    mean_abs_diff=float(diff.mean()),
                )
            )
        else:
            results.append(
                LayerComparison(
                    layer_idx=idx,
                    captured_shape=c.shape,
                    offline_shape=o.shape,
                    shape_match=False,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                )
            )
    return results


def check_pre_norm(
    captured_final: Tensor | None,
    offline_final: Tensor | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool | None:
    """Check whether the final-layer pre-norm capture matches offline.

    Returns ``None`` if either tensor is absent (i.e., the final layer could
    not be collected).
    """
    if captured_final is None or offline_final is None:
        return None
    if captured_final.shape != offline_final.shape:
        return False
    diff = (captured_final - offline_final).abs()
    return float(diff.max()) <= tolerance


def check_within_tolerance(
    layers: list[LayerComparison], tolerance: float
) -> bool:
    """Return True if every layer with a numeric diff is within tolerance."""
    return all(
        lc.max_abs_diff is not None and lc.max_abs_diff <= tolerance
        for lc in layers
    )


def derive_verdict(result: ComparisonResult) -> str:
    """Derive a PASS/FAIL verdict from a completed ComparisonResult.

    Verdict rules:
    - **PASS**: all shapes match AND all layers within tolerance AND
      pre-norm match is confirmed (not None and True).
    - **FAIL_empty**: no layers compared (nothing to judge).
    - **FAIL_shape**: one or more layers have mismatched shapes.
    - **FAIL_tolerance**: shapes match but numerical difference exceeds tolerance.
    - **FAIL_pre_norm**: pre-norm comparison failed or could not be confirmed.
    """
    if not result.layers:
        return "FAIL_empty"
    if not result.all_shapes_match:
        return "FAIL_shape"
    if not result.all_within_tolerance:
        return "FAIL_tolerance"
    if result.pre_norm_match is None:
        return "FAIL_pre_norm"
    if not result.pre_norm_match:
        return "FAIL_pre_norm"
    return "PASS"


def build_result(
    captured: dict[int, Tensor],
    offline: dict[int, Tensor],
    *,
    captured_final_pre_norm: Tensor | None = None,
    offline_final_pre_norm: Tensor | None = None,
    prefix_cache: PrefixCacheResult | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ComparisonResult:
    """Build a complete ComparisonResult from raw tensor dicts.

    This is the primary entry point for constructing a comparison result.
    """
    layers = compare_layerwise(captured, offline, tolerance=tolerance)
    pre_norm = check_pre_norm(
        captured_final_pre_norm, offline_final_pre_norm, tolerance=tolerance
    )
    result = ComparisonResult(
        layers=layers,
        pre_norm_match=pre_norm,
        prefix_cache_test=prefix_cache,
        tolerance=tolerance,
        verdict="",
    )
    return ComparisonResult(
        layers=result.layers,
        pre_norm_match=result.pre_norm_match,
        prefix_cache_test=result.prefix_cache_test,
        tolerance=result.tolerance,
        verdict=derive_verdict(result),
    )