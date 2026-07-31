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

**Relative tolerance (primary)**
bf16 has ~7-8 bits of mantissa, giving a per-op relative error near 1e-2.
Across 20+ decoder layers with residual accumulation, error can compound to a
few percent. A relative tolerance of **0.10 (10 %)** is chosen as a generous
but defensible bound: it is well above the bf16 noise floor for a single
operation yet tight enough to catch a genuinely different quantity (e.g.,
post-norm vs. pre-norm, or a shifted residual).  The relative metric
``mean_rel_error`` (mean|a-b| / mean|b|) drives the verdict; the absolute
tolerance is retained for reference but no longer gates PASS/FAIL.

**Trend detection**
If relative error is roughly *constant* across layers, the divergence is
proportional to activation magnitude -- consistent with numerical noise on the
same underlying quantity.  If relative error *grows* with depth, the divergence
is systematic and suggests the captured tensor may not be the same quantity.
The trend is reported as one of: ``"constant"``, ``"growing"``, or
``"insufficient_data"``.
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
# Tolerances
# ---------------------------------------------------------------------------

#: Per-element absolute tolerance for bf16 activation comparison (reference
#: only; the verdict now uses relative tolerance).
DEFAULT_TOLERANCE: float = 1e-2

#: Relative tolerance for bf16 activation comparison (PRIMARY verdict driver).
#:
#: Rationale: bf16 provides ~7-8 bits of mantissa (~1e-2 per-op relative
#: error).  Across 20+ layers with residual-stream accumulation, a few percent
#: is expected.  10% is well above the noise floor yet tight enough to catch a
#: genuinely different quantity (e.g., post-norm vs. pre-norm).
DEFAULT_RELATIVE_TOLERANCE: float = 0.10

#: Small constant to avoid division by zero in relative error computation.
_EPSILON: float = 1e-12


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

    # Relative error metrics (only set when shape_match is True)
    mean_ref_magnitude: float | None = None  # mean |offline|
    max_ref_magnitude: float | None = None   # max |offline|
    mean_rel_error: float | None = None      # mean|cap-off| / mean|off|
    max_rel_error: float | None = None       # max(|cap-off| / (|off|+eps))
    cosine_similarity: float | None = None   # cosine(captured, offline)


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
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    verdict: str = ""  # Set by derive_verdict
    rel_error_trend: str = ""  # "constant" | "growing" | "insufficient_data"

    # -- derived helpers --

    @property
    def all_shapes_match(self) -> bool:
        return all(lc.shape_match for lc in self.layers)

    @property
    def all_within_tolerance(self) -> bool:
        """Check relative tolerance (primary) across all layers."""
        return all(
            lc.mean_rel_error is not None
            and lc.mean_rel_error <= self.relative_tolerance
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
                    "mean_ref_magnitude": lc.mean_ref_magnitude,
                    "max_ref_magnitude": lc.max_ref_magnitude,
                    "mean_rel_error": lc.mean_rel_error,
                    "max_rel_error": lc.max_rel_error,
                    "cosine_similarity": lc.cosine_similarity,
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
            "relative_tolerance": self.relative_tolerance,
            "verdict": self.verdict,
            "rel_error_trend": self.rel_error_trend,
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
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_rel_error_trend(
    layers: list[LayerComparison],
) -> str:
    """Detect whether relative error is constant or growing across layers.

    Compares the mean relative error of the earliest and latest layers that
    have numeric data.  If the ratio exceeds 3 (i.e., the last layer's error
    is more than 3x the first), the trend is ``"growing"``.  Otherwise it is
    ``"constant"``.

    Returns ``"insufficient_data"`` when fewer than 2 layers have relative
    error values.
    """
    rel_errors = [
        lc.mean_rel_error
        for lc in layers
        if lc.mean_rel_error is not None
    ]
    if len(rel_errors) < 2:
        return "insufficient_data"
    first = rel_errors[0]
    last = rel_errors[-1]
    if first < _EPSILON:
        # First layer is near-zero; any non-zero later value is growth.
        return "growing" if last > _EPSILON else "constant"
    ratio = last / first
    return "growing" if ratio > 3.0 else "constant"


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
            # Reference magnitude (offline tensor) for context.
            o_abs = o.abs()
            mean_ref_mag = float(o_abs.mean())
            max_ref_mag = float(o_abs.max())

            # Relative error: mean|a-b| / mean|b|  (epsilon-guarded).
            mean_rel = float(diff.mean()) / (mean_ref_mag + _EPSILON)

            # Elementwise relative error with epsilon guard.
            elem_rel = diff / (o_abs + _EPSILON)
            max_rel = float(elem_rel.max())

            # Cosine similarity (scale-invariant, cast to float32 for
            # precision in the dot product).
            c_flat = c.flatten().float()
            o_flat = o.flatten().float()
            dot = float((c_flat * o_flat).sum())
            norm_c = float(c_flat.norm())
            norm_o = float(o_flat.norm())
            cosine = dot / (norm_c * norm_o + _EPSILON)

            results.append(
                LayerComparison(
                    layer_idx=idx,
                    captured_shape=c.shape,
                    offline_shape=o.shape,
                    shape_match=True,
                    max_abs_diff=float(diff.max()),
                    mean_abs_diff=float(diff.mean()),
                    mean_ref_magnitude=mean_ref_mag,
                    max_ref_magnitude=max_ref_mag,
                    mean_rel_error=mean_rel,
                    max_rel_error=max_rel,
                    cosine_similarity=cosine,
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
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> bool | None:
    """Check whether the final-layer pre-norm capture matches offline.

    Uses *relative* error as the primary check (consistent with the layerwise
    verdict).  Falls back to absolute tolerance only when the reference
    magnitude is essentially zero.

    Returns ``None`` if either tensor is absent (i.e., the final layer could
    not be collected).
    """
    if captured_final is None or offline_final is None:
        return None
    if captured_final.shape != offline_final.shape:
        return False
    diff = (captured_final - offline_final).abs()
    mean_ref = float(offline_final.abs().mean())
    if mean_ref < _EPSILON:
        # Reference is near-zero; use absolute tolerance.
        return float(diff.max()) <= tolerance
    return float(diff.mean()) / (mean_ref + _EPSILON) <= relative_tolerance


def check_within_tolerance(
    layers: list[LayerComparison],
    tolerance: float | None = None,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> bool:
    """Return True if every layer with a numeric diff is within tolerance.

    Uses relative error by default.  The *tolerance* parameter is retained
    for backward compatibility but is only used when *relative_tolerance*
    is explicitly passed as ``None``.
    """
    if relative_tolerance is None and tolerance is not None:
        # Legacy: only absolute tolerance given.
        return all(
            lc.max_abs_diff is not None and lc.max_abs_diff <= tolerance
            for lc in layers
        )
    return all(
        lc.mean_rel_error is not None and lc.mean_rel_error <= relative_tolerance
        for lc in layers
    )


def derive_verdict(result: ComparisonResult) -> str:
    """Derive a PASS/FAIL verdict from a completed ComparisonResult.

    Verdict rules (relative tolerance is primary):
    - **PASS**: all shapes match AND all layers within *relative* tolerance
      AND pre-norm match is confirmed (not None and True).
    - **FAIL_empty**: no layers compared (nothing to judge).
    - **FAIL_shape**: one or more layers have mismatched shapes.
    - **FAIL_tolerance**: shapes match but relative error exceeds tolerance.
      The ``rel_error_trend`` field indicates whether the divergence looks
      like proportional noise ("constant") or a systematic drift ("growing").
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
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> ComparisonResult:
    """Build a complete ComparisonResult from raw tensor dicts.

    This is the primary entry point for constructing a comparison result.
    """
    layers = compare_layerwise(captured, offline, tolerance=tolerance)
    pre_norm = check_pre_norm(
        captured_final_pre_norm, offline_final_pre_norm,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    trend = _detect_rel_error_trend(layers)
    result = ComparisonResult(
        layers=layers,
        pre_norm_match=pre_norm,
        prefix_cache_test=prefix_cache,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        verdict="",
        rel_error_trend=trend,
    )
    return ComparisonResult(
        layers=result.layers,
        pre_norm_match=result.pre_norm_match,
        prefix_cache_test=result.prefix_cache_test,
        tolerance=result.tolerance,
        relative_tolerance=result.relative_tolerance,
        verdict=derive_verdict(result),
        rel_error_trend=result.rel_error_trend,
    )