"""Elementwise comparison of captured vs. offline activation tensors.

This module implements the comparison logic for Stage 0 of the serving-time
activation capture prototype. It compares two stacks of per-layer tensors
(captured at serving time vs. extracted offline) and produces a machine-readable
verdict.

**WHAT THIS COMPARISON DOES AND DOES NOT ESTABLISH.**  The captured and offline
stacks are *not* two independent derivations of the aux hidden states.  Both
originate from the single tensor that vLLM's
``EagleModelMixin._maybe_add_hidden_state`` appends at
``model_executor/models/interfaces.py:1337``; the serving hook takes that list
object directly out of ``_model_forward``, while the offline path reads the same
variable one branch later at ``v1/worker/gpu_model_runner.py:5035`` and merely
transports it (stack, same-dtype buffer assignment, integer-indexed KV scatter,
the inverse gather, a pinned same-dtype D2H copy, ``save_file``) with no
floating-point arithmetic and no dtype cast at any step.  The offline "draft
model" has no weights (``models/extract_hidden_states.py:392-394`` returns
``set()``) and computes nothing (``CacheOnlyAttentionImpl.forward`` at
``:229-231`` is ``pass``).

So this comparison is a **transport check**: it catches slot-mapping errors,
layer misordering, truncation, row misalignment and prefix-cache row loss.  A
result of exactly ``0.0`` is the *expected* outcome, not a surprise, and it does
not establish that the captured tensor is the quantity the trainer wants.  The
independent evidence for that lives in
:mod:`speedlm.activation_capture.hf_reference`, which re-derives the same
quantity with HuggingFace transformers in float32 — a different implementation,
a different dtype, no shared code path.  Read the two together; neither alone
settles Stage 0.

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
The 0.10 bound below was sized for a *two-forward* comparison — two engines
each recomputing the activations, where kernel-order and batch-composition
differences are real.  This test does not perform two forwards (see the note
above), so on the paths it actually exercises the achievable value is 0.0 and
the bound is slack by construction.  It is retained deliberately: it is the
right bound if the transports ever stop being bit-identical (e.g. an upstream
change introduces a cast or a fused-MoE reduction-order difference), and
tightening it to 0.0 would turn any future benign vLLM change into a red test
without adding information.  **Do not read a pass at this tolerance as
evidence about numerical agreement between derivations** — for that, see
``hf_reference.bf16_relative_tolerance``, which derives a real, depth-dependent
bf16-vs-fp32 bound.

The original rationale, which still describes the sources of difference in the
two-forward case: bf16 has ~7-8 bits of mantissa, giving a per-op relative
error near 1e-2; across 20+ decoder layers with residual accumulation, error
can compound to a few percent; 10% sits well above the bf16 noise floor for a
single operation yet below the O(1) error of a genuinely different quantity
(e.g. post-norm vs. pre-norm, or a shifted residual).  The relative metric
``mean_rel_error`` (mean|a-b| / mean|b|) drives the verdict; the absolute
tolerance is retained for reference but no longer gates PASS/FAIL.

**Elementwise relative error (``max_rel_error`` / ``p99_rel_error``)**
These are diagnostics only -- they never gate PASS/FAIL.  They divide by
``max(|offline_i|, floor)`` where ``floor`` is a small fraction of the
reference tensor's RMS, *not* by ``|offline_i| + epsilon``.  A bare epsilon
makes the ratio unbounded on the near-zero elements that fill a residual
stream, which is why earlier artifacts reported ``max_rel_error`` values of
1e12-1e14 next to a ``mean_rel_error`` of 0.035.  See
``_REL_ERROR_FLOOR_FRACTION``.

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

from speedlm.activation_capture.hf_reference import (
    HFReferenceResult,
    cosine_similarity,
)

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

#: Fraction of the reference tensor's RMS used as a *floor* on the denominator
#: of the elementwise relative error.
#:
#: WHY A BARE EPSILON FAILS: dividing by ``|ref| + 1e-12`` makes the ratio
#: unbounded as ``|ref| -> 0``.  A residual-stream activation tensor contains
#: many elements that are essentially zero, so a perfectly ordinary bf16
#: rounding difference of ~1e-3 divided by a reference element of ~1e-15 yields
#: a "relative error" of ~1e12.  That is exactly what real Stage 0 artifacts
#: show: ``max_rel_error`` in the 1e12-1e14 range while ``mean_rel_error`` is
#: ~0.035 and cosine similarity is ~0.99.  The metric was measuring the
#: smallness of the denominator, not the badness of the difference.
#:
#: The physical model is that error does NOT propagate per-element: each
#: activation is the output of a reduction (matmul / residual add) over the
#: whole hidden dimension, so its rounding noise is bounded by the *tensor's*
#: magnitude, not by that element's own magnitude.  The correct denominator is
#: therefore ``max(|ref_i|, floor)`` with ``floor`` proportional to a magnitude
#: statistic of the reference tensor.  RMS is used (rather than mean|ref| or a
#: high percentile) because it is the L2 scale that matmul reductions actually
#: accumulate against, and because it is scale-equivariant: scaling both
#: tensors by k scales the floor by k, leaving every relative metric unchanged.
#:
#: 1e-3 is deliberately permissive: only elements below 0.1 % of the tensor RMS
#: are floored, so a genuinely diverging element of any meaningful magnitude is
#: still divided by its own value and still reported at full size.
_REL_ERROR_FLOOR_FRACTION: float = 1e-3

#: Percentile reported alongside ``max_rel_error``.  Even with a denominator
#: floor, ``max`` over millions of elements is an extreme-order statistic; the
#: p99 is the robust "how bad is a bad element, typically" number.
_REL_ERROR_PERCENTILE: float = 0.99


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
    #: max(|cap-off| / max(|off|, floor)) where floor is tied to the reference
    #: tensor's RMS -- see _REL_ERROR_FLOOR_FRACTION.
    max_rel_error: float | None = None
    #: p99 of the same floored elementwise ratio; robust companion to
    #: ``max_rel_error``, which is a single extreme order statistic.
    p99_rel_error: float | None = None
    #: cosine(captured, offline), from
    #: :func:`speedlm.activation_capture.hf_reference.cosine_similarity`:
    #: accumulated in float64 and clamped to ``[-1, 1]``, so a bit-identical
    #: pair reports exactly ``1.0``.  Corroborating only -- it never gates the
    #: verdict, and it stays above 0.99 for tensors that differ by far more
    #: than tolerance.
    cosine_similarity: float | None = None


# ---------------------------------------------------------------------------
# Prefix-cache coverage test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixCacheResult:
    """Result of the prefix-cache coverage measurement.

    Every field here must be **measured**.  The first version of this type was
    populated with hardcoded literals (``cache_hit=True``, ``rows_missing=0``)
    by a test with no assertions, which recorded an assumption and reported it
    as a finding.  If a caller cannot measure a field, it must not construct
    this type.

    The row count is **per layer**, not summed over layers.  The old
    ``captured_row_count`` summed ``rows x layers``, which is not comparable to
    ``prompt_token_count`` and made the missing-row count meaningless; the
    field was renamed rather than redefined so no stale caller keeps the old
    meaning.

    ``rows_missing`` was renamed to ``prompt_rows_missing`` for the same reason.
    It used to be computed as ``prompt_token_count - captured_rows_per_layer``,
    which is wrong whenever the capture also holds decode rows: job 369256
    recorded ``rows_missing: 96`` from ``165 - 69`` on a request whose 69
    captured rows were 21 prompt rows plus 48 decode rows.  The true number of
    prompt positions that produced no activation row was 144 — the cold row
    count (213) minus the warm one (69), which is exactly the engine's own
    prefix-cache hit-token count.  No verdict moved (the sharper
    ``hit_tokens == expected_hit_tokens`` assertion carried the proof) but the
    headline number was wrong by 48, so the name no longer survives.
    """

    #: The engine's own ``usage.prompt_tokens`` for the request under test.
    prompt_token_count: int
    #: Rows captured **for a single layer**.  One row per *forwarded* token
    #: position, which is the prompt rows the engine actually recomputed plus
    #: any decode rows the same capture window collected.  It is therefore
    #: comparable to ``prompt_token_count`` only when the caller has already
    #: trimmed the stack to the prompt range — do not subtract the two.
    captured_rows_per_layer: int
    #: Number of layers captured.  ``captured_rows_per_layer *
    #: captured_layer_count`` is the total tensor row count.
    captured_layer_count: int
    #: Whether the engine actually served this request from its prefix cache,
    #: read from ``vllm:prefix_cache_hits_total`` on ``/metrics`` — never
    #: inferred from the fact that the same prompt was sent twice.
    cache_hit: bool
    #: Prompt positions that produced no activation row.  Section 6.1's hazard
    #: is real exactly when this is positive on a cache hit.
    #:
    #: **Must be measured against a cold baseline**, as
    #: ``cold_rows_per_layer - captured_rows_per_layer`` with both requests
    #: sending the same prompt and generating the same number of decode rows.
    #: Deriving it from ``prompt_token_count`` instead folds the decode rows
    #: into a prompt-row count; that is the bug this field is named after.
    prompt_rows_missing: int


# ---------------------------------------------------------------------------
# Overall result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Machine-readable comparison between captured and offline activations."""

    layers: list[LayerComparison]
    pre_norm_match: bool | None  # None if final-layer data absent
    prefix_cache_test: PrefixCacheResult | None = None
    #: Verdict of the independent HuggingFace fp32 re-derivation.  ``None``
    #: means the reference was NOT run, which is not a pass -- the caller is
    #: responsible for asserting that it ran.  It is not gated here because a
    #: default-None field must not change the verdict of every existing
    #: caller; see ``derive_verdict``.
    hf_reference: HFReferenceResult | None = None
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
                    "p99_rel_error": lc.p99_rel_error,
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
            "hf_reference": (
                self.hf_reference.to_dict()
                if self.hf_reference is not None
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


def _at_least_float32(t: Tensor) -> Tensor:
    """Promote a narrow dtype to float32, leaving float32/float64 untouched.

    bf16 cannot represent the squared values of large activations (the real
    artifacts contain elements above 4e4), so the promotion is not optional.
    ``element_size()`` is used instead of a dtype comparison so that this
    module keeps its runtime-torch-free import surface.
    """
    return t.float() if t.element_size() < 4 else t


def _rms(t: Tensor) -> float:
    """Root-mean-square magnitude of a tensor, in at least float32."""
    return float(_at_least_float32(t).pow(2).mean().sqrt())


def _percentile(t: Tensor, q: float) -> float:
    """Return the *q* quantile of a tensor (nearest-rank, 0 <= q <= 1).

    ``torch.quantile`` refuses inputs above ~16M elements, which real capture
    tensors exceed, so this uses ``kthvalue`` on the flattened tensor instead.
    """
    flat = t.flatten()
    n = int(flat.numel())
    if n == 0:
        return 0.0
    # Nearest-rank: the smallest value at or above the q fraction of the data.
    k = min(n, max(1, int(q * n + 0.5)))
    return float(flat.kthvalue(k).values)


def _elementwise_rel_error(diff: Tensor, ref_abs: Tensor, captured: Tensor) -> Tensor:
    """Elementwise |captured-offline| / max(|offline|, scale-tied floor).

    The floor is ``_REL_ERROR_FLOOR_FRACTION * RMS(offline)``.  See the
    constant's docstring for why a bare additive epsilon produces the
    1e12-1e14 garbage values seen in real Stage 0 artifacts.

    When the reference tensor is *identically* zero its RMS gives no usable
    scale, so the captured tensor's RMS is used instead: that reports a total
    divergence as a relative error of ~1.0 (still far above the 0.10
    tolerance) rather than as an arbitrary 1/epsilon blow-up.  If both tensors
    are zero every difference is zero and the ratio is 0 regardless.
    """
    scale = _rms(ref_abs)
    if scale <= 0.0:
        scale = _rms(captured)
    floor = max(_REL_ERROR_FLOOR_FRACTION * scale, _EPSILON)
    return _at_least_float32(diff) / _at_least_float32(ref_abs).clamp_min(floor)


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

            # Elementwise relative error with a scale-tied denominator floor.
            elem_rel = _elementwise_rel_error(diff, o_abs, c)
            max_rel = float(elem_rel.max())
            p99_rel = _percentile(elem_rel, _REL_ERROR_PERCENTILE)

            # Cosine similarity (scale-invariant).  Delegated to
            # hf_reference so that both legs of Stage 0 report the metric on
            # identical terms -- float64 accumulation, clamped to [-1, 1].
            # This module used to inline a float32 version, which returned
            # values ABOVE 1.0 on bit-identical tensors; see that function's
            # docstring for the measurement.
            cosine = cosine_similarity(c, o)

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
                    p99_rel_error=p99_rel,
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


def align_prompt_rows(
    captured: Tensor,
    offline: Tensor,
    num_prompt_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Trim both stacks to the first *num_prompt_tokens* rows.

    **The two paths do not run the same token sequence past the prompt.**

    * Serving capture records every scheduled token: the ``num_prompt_tokens``
      prompt rows produced by the prefill, followed by one row per token the
      model *itself sampled*.
    * Offline extraction runs the fully-rendered conversation through
      ``prepare_data.py``, which requires an assistant turn.  Its rows are the
      prompt rows followed by the rows of the *template's* assistant turn.

    Only the leading ``num_prompt_tokens`` rows are the same tokens on both
    sides.  Everything after that is "what the model said" vs. "what the
    template said" -- two different token sequences, whose hidden states have
    no reason to agree and which therefore must never be compared.

    Trimming only the *captured* side down to ``offline.shape[0]`` (the old
    behaviour) does exactly that: it silently pads the comparison out with
    mismatched-token rows.  How many such rows there are is a property of the
    verifier's chat template, not of the capture, which is why the same code
    passed on one model and failed on another.  gpt-oss-20b renders an empty
    assistant turn to nothing; Qwen3 renders it to
    ``<think>\\n\\n</think>\\n\\n<|im_end|>\\n`` -- six extra rows, a quarter of
    a 24-row comparison, every one of them garbage.

    Args:
        captured: serving-capture rows, ``(>= num_prompt_tokens, H)``.
        offline: offline-extraction rows, ``(>= num_prompt_tokens, H)``.
        num_prompt_tokens: number of tokens in the prompt actually fed to the
            serving engine (its ``usage.prompt_tokens``).  This is *not*
            ``offline.shape[0]``.

    Returns:
        ``(captured[:n], offline[:n])`` with ``n = num_prompt_tokens``.

    Raises:
        ValueError: if *num_prompt_tokens* is not positive, or if either
            tensor has fewer than *num_prompt_tokens* rows.
    """
    if num_prompt_tokens <= 0:
        raise ValueError(
            f"num_prompt_tokens must be positive, got {num_prompt_tokens}"
        )
    if captured.shape[0] < num_prompt_tokens:
        raise ValueError(
            f"captured has fewer rows ({captured.shape[0]}) than the prompt "
            f"({num_prompt_tokens}) — cannot align; a prefix-cache hit or a "
            f"dropped prefill would look like this"
        )
    if offline.shape[0] < num_prompt_tokens:
        raise ValueError(
            f"offline has fewer rows ({offline.shape[0]}) than the prompt "
            f"({num_prompt_tokens}) — cannot align; the offline path did not "
            f"re-run the full prompt"
        )
    return captured[:num_prompt_tokens], offline[:num_prompt_tokens]


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

    **This is a tolerance check, not a structural one.**  Both sides collect
    the final decoder layer at the same point (``hidden_states + residual``,
    before the next norm — vLLM ``model_executor/models/interfaces.py:1337``),
    so this never distinguishes pre-norm from post-norm.  It is the layerwise
    relative-error test applied to one extra layer, and it goes ``False`` for
    every reason that test does.  Do not read ``pre_norm_match=False`` as
    evidence of a normalization-point bug.

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
    - **FAIL_hf_reference**: the independent HuggingFace fp32 re-derivation was
      run and did not pass.  A ``hf_reference`` of ``None`` means it was not
      run and is deliberately NOT a failure here: gating on absence would flip
      the verdict of every caller that predates the reference.  The e2e harness
      asserts separately that the reference actually ran.
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
    if result.hf_reference is not None and not result.hf_reference.passed:
        return "FAIL_hf_reference"
    return "PASS"


def build_result(
    captured: dict[int, Tensor],
    offline: dict[int, Tensor],
    *,
    captured_final_pre_norm: Tensor | None = None,
    offline_final_pre_norm: Tensor | None = None,
    prefix_cache: PrefixCacheResult | None = None,
    hf_reference: HFReferenceResult | None = None,
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
        hf_reference=hf_reference,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        verdict="",
        rel_error_trend=trend,
    )
    return ComparisonResult(
        layers=result.layers,
        pre_norm_match=result.pre_norm_match,
        prefix_cache_test=result.prefix_cache_test,
        hf_reference=result.hf_reference,
        tolerance=result.tolerance,
        relative_tolerance=result.relative_tolerance,
        verdict=derive_verdict(result),
        rel_error_trend=result.rel_error_trend,
    )