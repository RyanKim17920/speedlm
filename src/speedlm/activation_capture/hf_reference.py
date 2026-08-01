"""Independent HuggingFace reference for the Stage 0 activation comparison.

Why this module exists
======================

The original Stage 0 check compared the serving-time capture against vLLM's
own "offline extraction" path and reported ``mean_rel_error == 0.0`` with
``max_abs_diff == 0.0`` on every layer.  That result is real, but it does not
mean what it looks like it means: **the two paths are not two derivations of
the same quantity, they are two transports of one tensor object.**

Verified against the pinned vLLM 0.25.1 checkout at
``/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/lib/python3.12/
site-packages/vllm``:

* Both paths originate at ``model_executor/models/interfaces.py:1337``,
  ``value = hidden_states + residual``, appended by
  ``EagleModelMixin._maybe_add_hidden_state`` (``:1329``).  The runner unpacks
  that list at ``v1/worker/gpu_model_runner.py:4364``.
* The serving capture wraps ``_model_forward`` and takes ``result[1]`` —
  literally the same list object (``hook.py`` ``_install_hook`` /
  ``_intercept_aux``) — then ``.detach().cpu()`` and ``save_file``.
* The offline path reads the *same variable* one branch later at
  ``gpu_model_runner.py:5035``
  (``[h[:num_scheduled_tokens] for h in aux_hidden_states]``) and transports
  it: ``torch.stack`` → assignment into a same-dtype buffer
  (``v1/spec_decode/extract_hidden_states.py:132,136``) → integer-indexed KV
  scatter (``model_executor/models/extract_hidden_states.py:81-90``, guarded
  by dtype-equality asserts at ``:220,223``) → the inverse gather
  (``distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector
  .py:35-42``) → pinned same-dtype D2H copy (``:393-396``) → ``save_file``.
  **There is no floating-point arithmetic and no dtype cast anywhere on that
  chain.**
* The offline "draft model" computes nothing:
  ``extract_hidden_states.py:392-394`` ``load_weights`` returns ``set()`` and
  ``CacheOnlyAttentionImpl.forward`` at ``:229-231`` is a bare ``pass``.

So a ``0.0`` answers "is one tensor round-tripped losslessly by two different
copy chains", not "do two derivations agree".  It can catch slot-mapping,
layer-ordering, truncation and off-by-one-row bugs — it cannot catch a wrong
*quantity*, because there is only one quantity.

This module supplies the missing independent leg: the same token ids run
through **HuggingFace transformers in float32**, an implementation that shares
no kernel, no dtype and no code path with vLLM.

The layer-index mapping (proved, not assumed)
=============================================

vLLM's aux collection, ``model_executor/models/qwen2.py:415-421`` (``Qwen3Model``
subclasses ``Qwen2Model`` — ``qwen3.py:260``)::

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(positions, hidden_states, residual)
        self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

and ``interfaces.py:1337`` stores ``hidden_states + residual``.  vLLM's decoder
layers carry the residual out-of-band, so ``hidden_states + residual`` *is* the
reconstructed residual stream.  Therefore:

* aux id ``0`` — called with ``residual is None`` before the loop — is the raw
  embedding output.
* aux id ``k`` (``1 <= k <= L``) is the residual stream **after** decoder layer
  ``k-1``, i.e. the value that enters decoder layer ``k``'s ``input_layernorm``.
* ``qwen2.py:429`` applies ``self.norm`` only to the main return value, *after*
  the aux list is complete, so aux states are never normalized.

HuggingFace transformers 5.10.4 collects ``output_hidden_states`` through
``utils/output_capturing.py``.  ``install_output_capuring_hook`` (``:99-118``)
prepends the first layer's *input* (``args[0]``) and then appends each decoder
layer's output; ``capture_outputs`` (``:205``) then, because
``tie_last_hidden_states`` defaults to ``True``, **overwrites the last entry
with ``outputs.last_hidden_state``** (``:264-266``) — which
``modeling_qwen3.py:434`` has already passed through ``self.norm``.  So::

    hf.hidden_states[0]      = embeddings                     (pre-layer-0)
    hf.hidden_states[k]      = input to decoder layer k       (1 <= k <= L-1)
    hf.hidden_states[L]      = FINAL RMSNorm APPLIED          (post-norm!)

Hence the mapping used here:

* **vLLM aux id k == hf.hidden_states[k] exactly, for 0 <= k <= L-1.**
* **vLLM aux id L (the final decoder layer the capture hook appends) has NO
  counterpart in the HF tuple** — the tie replaces it with the post-norm
  tensor.  It is obtained instead from a forward hook on
  ``model.model.layers[L-1]``, whose output is the pre-norm residual stream.

Both halves of that claim are *checked at runtime* rather than trusted:
:func:`reference_residual_stream` asserts the tuple length is ``L+1``, and
:func:`compare_to_hf_reference` reports
``final_layer_prenorm_confirmed`` — the post-norm tensor must differ from the
hooked pre-norm one by far more than tolerance, which is what makes the hook
necessary in the first place.

Independence is further established structurally by the **neighbour
discrimination check**: each captured aux layer is compared not only to its
claimed reference index but to ``k-1`` and ``k+1`` as well, and the claimed
index must be the strict argmin by a factor of
:data:`DISCRIMINATION_MARGIN`.  That check does not depend on the tolerance
below being correctly derived, and it is what a lossless round-trip cannot
fake: an off-by-one layer or a post-norm substitution moves the argmin.

For the **final** aux layer ``k == L`` there is no ``k+1`` in the stream, so
the rival on that side is the post-norm tensor — the only quantity the model
produces downstream of layer ``L``.  Without it that layer's argmin would be
one-sided and a post-norm capture would win it uncontested; the post-norm
tensor is therefore a *rival* in the argmin as well as the subject of the
separate ``final_layer_prenorm_confirmed`` check.  ``k-2`` is deliberately not
used as a substitute: the concern is an error metric that grows monotonically
away from the true layer, and under that assumption ``err(k-2) > err(k-1)``, so
``min(rivals)`` and the verdict would be unchanged.

Tolerance derivation
====================

Unlike the vLLM-vs-vLLM comparison, exact equality is **not** achievable here:
vLLM serves in bfloat16 while this reference runs in float32.  A real
tolerance is therefore required, and it is derived rather than picked:

bfloat16 carries a 7-bit stored significand plus one implicit bit, so its unit
roundoff is ``u = 2^-8 = 3.90625e-3`` (:data:`BF16_UNIT_ROUNDOFF`).  The
residual stream is materialized in bfloat16 exactly once per decoder layer, so
after ``k`` layers the accumulated representation error, in the worst case
where every rounding aligns, is bounded by ``k * u`` relative to the stream's
own magnitude.  A small constant offset :data:`BF16_DEPTH_OFFSET` accounts for
the roundings that feed the stream before any layer has run (embedding lookup)
and for the handful of bf16-rounded reductions inside the first layer::

    tolerance(k) = BF16_UNIT_ROUNDOFF * (k + BF16_DEPTH_OFFSET)

The worst-case *linear* accumulation is used deliberately in preference to the
``sqrt(k) * u`` random-walk bound: vLLM's kernel selection and reduction order
are not statistically independent of HuggingFace's, so the errors cannot be
assumed to cancel.  For Qwen3-8B (``L = 36``, aux layers ``[2, 18, 33]`` as
resolved from the drafter at runtime) this gives 2.3e-2, 8.6e-2 and 1.45e-1,
and 1.6e-1 for the appended final layer —
tighter than the flat 0.10 in :mod:`speedlm.activation_capture.compare` at
shallow depth, looser at the bottom of the stack, and in every case one to two
orders of magnitude below the O(1) error a genuinely wrong quantity produces.

Which of the two checks is sharp
================================

**The discrimination check is the sharp instrument; the tolerance gate is a
ceiling.**  They are not two views of the same evidence and they must not be
quoted as if they were:

* ``within_tolerance`` compares a measured error against a *worst-case* bound
  that assumes every per-layer rounding aligns.  Real roundoff does not align,
  so the measurement sits far inside the bound.  Job 369256 used only 21.8 %,
  11.2 %, 10.7 % and 11.3 % of the budget at aux layers 2, 18, 33 and 36 (see
  :attr:`HFReferenceLayer.tolerance_budget_used`) — capture fidelity could
  degrade roughly ninefold and this gate would still pass.  It is a
  sanity floor, not a precision result.
* ``identified`` compares the claimed layer against its own immediate
  neighbours and demands a :data:`DISCRIMINATION_MARGIN`-fold separation.  The
  same run cleared that 3x margin by factors of 81.46, 23.14, 24.03 and 50.02.
  This is the check that fails if the capture is off by one layer, post-norm
  rather than pre-norm, or a different quantity altogether — and it does not
  depend on the tolerance above being correctly derived.

The tolerance is deliberately *not* tightened to the measured values: those
come from 18 prompt rows of a single prompt on a single model, and a bound
fitted to them would flake on any other prompt, model or vLLM build.  Making
the slack legible via ``tolerance_budget_used`` is the honest alternative to
fitting the bound to one sample.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only executed by static type checkers.  mypy has per-module overrides for
    # torch/safetensors/vllm/transformers (see pyproject.toml), so this module
    # keeps a torch-free import surface for the project venv.
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

#: Unit roundoff of bfloat16: 7 stored significand bits + 1 implicit bit, so
#: ``2^-8``.  This is the per-materialization relative representation error of
#: the residual stream in the serving engine.
BF16_UNIT_ROUNDOFF: float = 2.0**-8

#: Constant depth offset added to the layer index in :func:`bf16_relative_tolerance`.
#: It covers the roundings that occur before any decoder layer has run (the
#: bf16 embedding lookup) plus the small fixed number of bf16-rounded
#: reductions — qkv projection, output projection, gate/up and down
#: projections — that feed a single residual-stream update.  Four is the
#: order of that count, not a fitted value.
BF16_DEPTH_OFFSET: int = 4

#: Factor by which the correct reference index must beat its nearest
#: neighbour in the discrimination check.  Adjacent residual-stream layers are
#: genuinely similar, so a bare argmin is too weak a signal; requiring a 3x
#: separation makes "the capture is layer k, not k+-1" a real claim.
DISCRIMINATION_MARGIN: float = 3.0

#: Guards division by zero in the relative-error and cosine computations.
_EPSILON: float = 1e-12


def bf16_relative_tolerance(aux_layer_idx: int) -> float:
    """Return the derived bf16-vs-fp32 relative tolerance for one aux layer.

    See the module docstring for the derivation.  ``aux_layer_idx`` is vLLM's
    aux layer id, which equals the number of decoder layers the residual
    stream has passed through.

    Raises:
        ValueError: if *aux_layer_idx* is negative.
    """
    if aux_layer_idx < 0:
        raise ValueError(f"aux_layer_idx must be >= 0, got {aux_layer_idx}")
    return BF16_UNIT_ROUNDOFF * (aux_layer_idx + BF16_DEPTH_OFFSET)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ReferenceUnavailable(RuntimeError):
    """The fp32 reference cannot be computed for this checkpoint here.

    Raised instead of returning ``None`` or silently downgrading the reference
    to a narrower dtype.  This leg exists to answer "is the captured tensor the
    right *quantity*"; a reference that quietly stopped being an independent
    fp32 recomputation answers nothing while still reporting numbers, which is
    strictly worse than not running.  Every instance carries an actionable
    message: the cause, the arithmetic behind it, and the concrete remedy.

    Args:
        message: the actionable text, as described above.
        model_dir: the checkpoint the refusal is about, when known.
        required_bytes: the fp32 footprint that could not be met, when the
            refusal is a memory refusal.
        available_bytes: the budget that was actually free, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        model_dir: str | None = None,
        required_bytes: int | None = None,
        available_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        #: The checkpoint this refusal is about, or ``None``.
        self.model_dir = model_dir
        #: fp32 bytes the reference would have needed, or ``None``.
        self.required_bytes = required_bytes
        #: Bytes actually available on the chosen device, or ``None``.
        self.available_bytes = available_bytes


@dataclass(frozen=True)
class HFReferenceLayer:
    """Comparison of one captured aux layer against the fp32 HF reference."""

    #: vLLM aux layer id.
    aux_layer_idx: int
    #: Index into the reference residual stream this layer is claimed to be.
    #: Equal to ``aux_layer_idx`` by the mapping proved in the module docstring.
    reference_index: int
    #: Rows compared (the prompt range only).
    rows: int
    #: ``mean|captured - reference| / mean|reference|`` against the claimed index.
    mean_rel_error: float
    #: Cosine similarity against the claimed index; corroborating only.
    cosine_similarity: float
    #: Derived bf16 tolerance for this depth.
    tolerance: float
    #: ``mean_rel_error <= tolerance``.
    within_tolerance: bool
    #: ``{reference index: mean_rel_error}`` for the claimed index and its
    #: immediate neighbours, keyed by index as a string for JSON round-trip.
    neighbour_rel_errors: dict[str, float]
    #: Index with the smallest relative error across ``neighbour_rel_errors``.
    best_match_index: int
    #: Runner-up error divided by the claimed index's error.  ``inf`` when the
    #: claimed index matched exactly (only possible if fp32 and bf16 agree bit
    #: for bit, which they do not in practice).
    discrimination_ratio: float
    #: ``best_match_index == reference_index`` and
    #: ``discrimination_ratio >= DISCRIMINATION_MARGIN``.
    identified: bool
    #: Every rival this layer was actually discriminated against, as
    #: ``(label, mean_rel_error)`` in the order they were considered.  Labels
    #: are stringified reference indices (``"35"``) except for the post-norm
    #: rival, labelled ``"post_norm"``.  Recorded because
    #: :attr:`discrimination_ratio` collapses the whole set to one number, and
    #: an artifact that only reports the ratio cannot be re-audited for *which*
    #: quantities the claim was made against.  Empty tuple only when there was
    #: nothing to compete with (a single-entry reference stream).
    rival_errors: tuple[tuple[str, float], ...] = ()
    #: ``True`` when the rival set brackets the claimed layer on both sides.
    #: For an interior layer that is ``k-1`` and ``k+1``.  For the FINAL layer
    #: ``k+1`` does not exist as a residual-stream entry, so the other side is
    #: the post-norm tensor — the only quantity in the model downstream of
    #: layer ``k``.  Without it the final layer's argmin is one-sided and a
    #: post-norm capture can win it by default, which is exactly the confusion
    #: this module exists to rule out.
    two_sided: bool = False

    @property
    def tolerance_budget_used(self) -> float:
        """``mean_rel_error / tolerance`` -- how much of the bound was spent.

        Recorded so that ``within_tolerance: true`` cannot be misread as a
        tight result.  :func:`bf16_relative_tolerance` is a **worst-case
        ceiling**: it assumes every per-layer bf16 rounding aligns in the same
        direction, which real roundoff does not do.  Job 369256 measured
        0.218 / 0.112 / 0.107 / 0.113 at aux layers 2 / 18 / 33 / 36 -- i.e.
        capture fidelity could degrade by roughly 9x and ``within_tolerance``
        would still report ``true``.

        The sharp check is :attr:`identified`, not this one: it compares the
        claimed layer against its own neighbours, and the same run cleared its
        3x margin by factors of 81 / 23 / 24 / 50.  Read the two together and
        weight the discrimination ratio.

        Deliberately a property rather than a stored field: it is a pure
        function of two fields that are already measured, so making it storable
        would create a way for a caller to record a value inconsistent with
        them.
        """
        return self.mean_rel_error / (self.tolerance + _EPSILON)


@dataclass(frozen=True)
class HFReferenceResult:
    """Verdict of the independent HuggingFace fp32 reference comparison."""

    #: Device the reference forward ran on (``"cuda"`` / ``"cpu"``).
    device: str
    #: dtype of the reference forward; always ``"torch.float32"`` in practice.
    dtype: str
    #: Rows compared on every layer.
    prompt_token_count: int
    #: Per-aux-layer comparisons, in ascending aux layer id.
    layers: list[HFReferenceLayer]
    #: The appended final decoder layer, when it was captured.
    final_layer_idx: int | None
    #: ``True`` when HF's ``hidden_states[L]`` was shown to be materially
    #: different from the hooked pre-norm value, i.e. the final RMSNorm really
    #: is applied there and the hook was necessary.  ``None`` when the final
    #: layer was not captured.
    final_layer_prenorm_confirmed: bool | None
    #: ``"PASS"``, or one of ``"FAIL_empty"`` / ``"FAIL_tolerance"`` /
    #: ``"FAIL_identification"`` / ``"FAIL_pre_norm"``.
    verdict: str
    #: Which prompt this verdict is about, when the caller ran more than one.
    #: ``None`` for a single-prompt run, which is why it defaults: a result
    #: without a label is a result that predates the multi-prompt matrix, not a
    #: result about an unknown prompt.  Carried so that a row of an N-prompt
    #: matrix can be attributed after the fact — an unlabelled matrix is a set
    #: of numbers nobody can re-run.
    prompt_label: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "device": self.device,
            "dtype": self.dtype,
            "prompt_label": self.prompt_label,
            "prompt_token_count": self.prompt_token_count,
            "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
            "bf16_depth_offset": BF16_DEPTH_OFFSET,
            "discrimination_margin": DISCRIMINATION_MARGIN,
            "final_layer_idx": self.final_layer_idx,
            "final_layer_prenorm_confirmed": self.final_layer_prenorm_confirmed,
            "verdict": self.verdict,
            "layers": [
                {
                    "aux_layer_idx": layer.aux_layer_idx,
                    "reference_index": layer.reference_index,
                    "rows": layer.rows,
                    "mean_rel_error": layer.mean_rel_error,
                    "cosine_similarity": layer.cosine_similarity,
                    "tolerance": layer.tolerance,
                    "within_tolerance": layer.within_tolerance,
                    "tolerance_budget_used": layer.tolerance_budget_used,
                    "neighbour_rel_errors": layer.neighbour_rel_errors,
                    "best_match_index": layer.best_match_index,
                    "discrimination_ratio": layer.discrimination_ratio,
                    "identified": layer.identified,
                    #: A list of pairs rather than a mapping so the order the
                    #: rivals were considered survives the JSON round-trip.
                    "rival_errors": [
                        [label, error] for label, error in layer.rival_errors
                    ],
                    "two_sided": layer.two_sided,
                }
                for layer in self.layers
            ],
        }


# ---------------------------------------------------------------------------
# Numerics (torch is imported lazily inside each function)
# ---------------------------------------------------------------------------


def _as_float32(tensor: Tensor) -> Tensor:
    """Promote a narrow dtype to float32; leave float32/float64 untouched.

    ``element_size()`` is used rather than a dtype comparison so this module
    needs no runtime torch import to be type-checked.
    """
    return tensor.float() if tensor.element_size() < 4 else tensor


def mean_relative_error(captured: Tensor, reference: Tensor) -> float:
    """``mean|captured - reference| / mean|reference|``, computed in float32.

    Matches the aggregate metric
    :mod:`speedlm.activation_capture.compare` uses for the vLLM-vs-vLLM
    comparison, so the two numbers are directly comparable.

    Raises:
        ValueError: if the two tensors have different shapes.
    """
    if captured.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: captured {tuple(captured.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    cap = _as_float32(captured)
    ref = _as_float32(reference)
    return float((cap - ref).abs().mean()) / (float(ref.abs().mean()) + _EPSILON)


def _as_float64(tensor: Tensor) -> Tensor:
    """Promote anything narrower than float64 to float64.

    ``element_size()`` is used rather than a dtype comparison so this module
    needs no runtime torch import to be type-checked.
    """
    return tensor.double() if tensor.element_size() < 8 else tensor


def cosine_similarity(captured: Tensor, reference: Tensor) -> float:
    """Flattened cosine similarity, accumulated in float64 and clamped.

    **Why float64 and not float32.**  The dot product and the two norms are
    three *separate* reductions over the same elements, each with its own
    rounding and its own summation order, so ``dot / (|a| |b|)`` does not
    telescope back to 1 even when the two tensors are bit-for-bit identical.
    With a float32 accumulator over a residual-stream tensor -- ~5e4 elements
    whose magnitudes reach 2e4, so the squares reach 4e8 -- that residue is
    O(1e-5).  Real Stage 0 artifacts show it: job 369256's transport leg
    reported cosines of 1.0000027857, 1.0000187356 and 1.0000228824 on tensors
    for which ``torch.equal`` was ``True`` and every abs/rel error was exactly
    ``0.0``.  A cosine above 1.0 is arithmetically impossible for real vectors,
    so those digits were pure accumulator noise.

    Promoting to float64 removes it: the same inputs return exactly ``1.0``.
    Note that ``_as_float32`` was *not* the bug -- bf16 was already being
    promoted -- float32 simply is not wide enough for this reduction.

    The result is clamped to ``[-1, 1]``.  Clamping cannot mask a real signal
    (no true cosine lies outside that interval) and it makes the impossible
    value unrepresentable rather than merely unlikely.

    **This metric is corroborating only.**  It is dominated by the largest
    elements of the tensor and stays above 0.99 for tensors that differ by far
    more than tolerance, so it must never be read as a precision measurement.
    ``mean_rel_error`` and the neighbour-discrimination ratio carry the verdict.
    """
    cap = _as_float64(captured).flatten()
    ref = _as_float64(reference).flatten()
    cosine = float(cap.dot(ref)) / (
        float(cap.norm()) * float(ref.norm()) + _EPSILON
    )
    return min(1.0, max(-1.0, cosine))


# ---------------------------------------------------------------------------
# Reference forward
# ---------------------------------------------------------------------------


#: fp32 bytes per parameter.
FP32_BYTES_PER_PARAMETER: int = 4

#: Multiplier over raw fp32 weight bytes covering activations, the RoPE cache
#: and allocator fragmentation.  Named rather than repeated so
#: :func:`select_reference_device` and :func:`assert_reference_fits` cannot
#: drift into answering "does it fit" two different ways — a device chosen
#: under one budget and admitted under another is exactly the OOM this module
#: is meant to turn into a legible refusal.
REFERENCE_DEVICE_HEADROOM_FRACTION: float = 1.35


def select_reference_device(
    *,
    num_parameters: int,
    free_device_bytes: int | None,
    headroom_fraction: float = REFERENCE_DEVICE_HEADROOM_FRACTION,
) -> str:
    """Choose ``"cuda"`` or ``"cpu"`` for the fp32 reference forward.

    **Why GPU is preferred.** A float32 copy of Qwen3-8B is
    ``8.19e9 * 4 B ~= 30.5 GiB``.  On CPU that is a 30 GiB host allocation and
    a prefill that is one to two orders of magnitude slower than on device —
    a true reference, but one that lengthens an already long GPU job for no
    numerical gain.  On device it is a few seconds.  The Stage 0 harness
    already tears the capture engine down and blocks on
    ``_wait_for_gpu_memory_release`` before the offline engine starts, so the
    machinery to hand the whole card to a third consumer already exists and is
    simply reused.  A smaller model was rejected outright: the question is
    whether *this deployment's* capture is the right quantity, and a different
    model does not answer it.

    CPU remains the automatic fallback when the device cannot hold the fp32
    copy, so a smaller card degrades to slow-but-correct rather than to OOM.

    Args:
        num_parameters: parameter count of the reference model.
        free_device_bytes: free VRAM, or ``None`` when no device is available.
        headroom_fraction: multiplier over raw weight bytes covering
            activations, the RoPE cache and allocator fragmentation.

    Returns:
        ``"cuda"`` or ``"cpu"``.
    """
    if free_device_bytes is None:
        return "cpu"
    required = int(
        num_parameters * FP32_BYTES_PER_PARAMETER * headroom_fraction
    )
    return "cuda" if free_device_bytes >= required else "cpu"


def reference_fp32_bytes(num_parameters: int) -> int:
    """Return the raw fp32 weight footprint of *num_parameters* parameters.

    Raises:
        ValueError: if *num_parameters* is negative.
    """
    if num_parameters < 0:
        raise ValueError(f"num_parameters must be >= 0, got {num_parameters}")
    return num_parameters * FP32_BYTES_PER_PARAMETER


def assert_reference_fits(
    model_dir: str,
    *,
    num_parameters: int,
    free_device_bytes: int | None,
    free_host_bytes: int | None,
    device: str,
) -> None:
    """Refuse, legibly, when the fp32 reference cannot fit on *device*.

    **Why this raises rather than degrades.**  The two silent alternatives are
    both worse than a refusal.  Downgrading to bf16 would keep producing
    numbers while destroying the one property that makes this leg evidence —
    it would no longer be an independent fp32 recomputation, it would be a
    second bf16 computation, and the comparison would report a tolerance it no
    longer earns.  Returning ``None`` would make the whole leg vanish into the
    ``hf_reference is None`` branch, which the e2e harness treats as "the leg
    did not run"; a checkpoint that *cannot* be referenced on this node is a
    different fact from a leg that was switched off, and it must be said out
    loud.

    **Why the arithmetic is in the message.**  The alternative failure mode is
    an OOM kill or a raw allocator error, neither of which tells the reader
    whether the checkpoint is too big, the node too small, or the parameter
    count miscounted.  Printing ``params x 4 B`` against the measured free
    bytes makes all three distinguishable from the failure text alone.

    Args:
        model_dir: the checkpoint, quoted in the message.
        num_parameters: **dequantized** parameter count, from
            :func:`checkpoint_parameter_count`.
        free_device_bytes: free VRAM, or ``None`` when no device is available.
        free_host_bytes: free host RAM, or ``None`` when it could not be read.
        device: the device already chosen, ``"cuda"`` or ``"cpu"``.

    Raises:
        ReferenceUnavailable: when the fp32 footprint exceeds the chosen
            device's budget.
        ValueError: if *device* is neither ``"cuda"`` nor ``"cpu"``.
    """
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")

    required = reference_fp32_bytes(num_parameters)
    #: On device the budget carries the same headroom the device choice used,
    #: so an operator forcing ``cuda`` is judged against the identical bar the
    #: automatic path would have applied.  On host the raw footprint is
    #: compared directly: the fp32 staging copy is the dominant term there and
    #: there is no allocator reserve to model.
    if device == "cuda":
        budgeted = int(required * REFERENCE_DEVICE_HEADROOM_FRACTION)
        available = free_device_bytes
    else:
        budgeted = required
        available = free_host_bytes
    if available is None:
        raise ReferenceUnavailable(
            f"fp32 reference for {model_dir} needs "
            f"{num_parameters:,} params x {FP32_BYTES_PER_PARAMETER} B = "
            f"{required / 1e9:.2f} GB, but the free memory on device={device} "
            f"could not be read, so the fit cannot be checked; refusing rather "
            f"than risking an OOM kill that would report nothing. Set "
            f"SPEEDLM_E2E_HF_REFERENCE_DEVICE explicitly on a node whose free "
            f"memory is readable, or run the reference on a larger-memory node.",
            model_dir=model_dir,
            required_bytes=required,
        )
    if budgeted > available:
        raise ReferenceUnavailable(
            f"fp32 reference for {model_dir} needs "
            f"{num_parameters:,} params x {FP32_BYTES_PER_PARAMETER} B = "
            f"{required / 1e9:.2f} GB"
            + (
                f" ({budgeted / 1e9:.2f} GB with the "
                f"{REFERENCE_DEVICE_HEADROOM_FRACTION}x activation headroom)"
                if device == "cuda"
                else ""
            )
            + f", but device={device} has {available / 1e9:.1f} GB free; the "
            f"fp32 leg cannot run for this checkpoint on this node. Shard it "
            f"across GPUs (that is what `accelerate` would actually be for) or "
            f"run the reference on a larger-memory node.",
            model_dir=model_dir,
            required_bytes=required,
            available_bytes=available,
        )


#: Checkpoint dtypes that a float32 load widens *exactly*, so the weights may be
#: materialized in the narrow dtype first and upcast on device.  Anything not
#: listed here is loaded straight to float32, because the shortcut would round.
_EXACTLY_WIDENED_TO_FP32: tuple[str, ...] = ("bfloat16", "float16")


def _checkpoint_dtype(model_dir: str) -> str:
    """Return the checkpoint's declared dtype, or ``""`` when unreadable.

    Returning ``""`` on any doubt keeps :func:`reference_residual_stream` on the
    conservative float32 load path: a wrong guess here would silently round the
    reference's weights, which is the one thing this module must never do.
    """
    import json
    from pathlib import Path

    path = Path(model_dir) / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for section in (raw, raw.get("text_config")):
        if isinstance(section, dict):
            for key in ("dtype", "torch_dtype"):
                value = section.get(key)
                if isinstance(value, str):
                    return value
    return ""


def _checkpoint_quantization(model_dir: str) -> str:
    """Return the checkpoint's declared quant method, or ``""`` when absent.

    Reads ``quantization_config.quant_method`` from ``config.json`` and
    lowercases it.  Mirrors :func:`_checkpoint_dtype` exactly, including its
    "return ``''`` on any doubt" rationale: ``""`` means *unquantized*, which
    routes :func:`loaded_reference_model` down the plain float32 load it has
    always taken.  A wrong *positive* answer here would swap in a quantizer's
    own dequantization path, so the doubt case must be the one that changes
    nothing.

    Verified on ``openai/gpt-oss-20b``: its ``quantization_config`` is
    ``{"quant_method": "mxfp4", "modules_to_not_convert": [...]}`` and there is
    no top-level or ``text_config`` ``dtype``/``torch_dtype`` key at all — which
    is precisely why the dtype probe alone cannot tell this checkpoint apart
    from an ordinary fp32 one.
    """
    import json
    from pathlib import Path

    path = Path(model_dir) / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for section in (raw, raw.get("text_config")):
        if isinstance(section, dict):
            config = section.get("quantization_config")
            if isinstance(config, dict):
                method = config.get("quant_method")
                if isinstance(method, str):
                    return method.lower()
    return ""


def checkpoint_parameter_count(model_dir: str) -> int | None:
    """Count the checkpoint's **dequantized** parameters, or ``None`` on doubt.

    Public because the number is not a detail of one caller: the e2e harness
    uses it to pick a device and this module uses it to refuse, and those two
    must not disagree about how big the model is.

    **Unquantized checkpoints** keep the cheap estimate: ``metadata.total_size``
    from ``model.safetensors.index.json`` is the on-disk byte count of a bf16
    checkpoint, so ``total_size // 2`` is the parameter count.

    **mxfp4 checkpoints break that estimate badly, and silently.**  Measured on
    ``openai/gpt-oss-20b`` by summing the real safetensors shards: the
    shards store ``11,956,805,184`` raw elements in ``13,761,316,904`` bytes,
    but each ``*_blocks`` byte packs **two** e2m1 nibbles and each ``*_scales``
    entry is a block exponent rather than a parameter, so the count that
    actually gets materialized in memory is ``20,914,757,184``.  The index
    estimate gives ``13,761,316,904 // 2 = 6,880,658,452`` — an **undercount of
    3.04x**.  Acting on it would answer "cuda" for a model whose fp32 copy is
    83.66 GB and OOM an 80 GiB H100.

    So for mxfp4 the shards are opened and the elements counted directly:
    ``*_blocks`` contributes ``2x`` its element count, ``*_scales`` contributes
    ``0``, everything else contributes its own element count.

    Args:
        model_dir: resolved local snapshot directory for the verifier.

    Returns:
        The dequantized parameter count, or ``None`` when the checkpoint's
        metadata could not be read — which keeps the caller conservative
        rather than guessing.
    """
    import json
    from pathlib import Path

    directory = Path(model_dir)
    if _checkpoint_quantization(model_dir) == "mxfp4":
        return _mxfp4_parameter_count(directory)

    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        raw = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    total = raw.get("metadata", {}).get("total_size")
    if not isinstance(total, int) or total <= 0:
        return None
    return total // 2


def _mxfp4_parameter_count(directory: Path) -> int | None:
    """Sum the dequantized element count of an mxfp4 checkpoint's shards.

    Only the tensor *shapes* are read, via ``get_slice(k).get_shape()``, so
    this never materializes a weight.

    Returns:
        The dequantized parameter count, or ``None`` when no shard could be
        read.
    """
    #: safetensors ships in the vLLM venv, not the project venv; kept lazy so
    #: this module's import surface stays torch-free.
    from safetensors import safe_open

    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        return None
    total = 0
    for shard in shards:
        try:
            with safe_open(str(shard), framework="pt") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors API
                    shape = handle.get_slice(key).get_shape()
                    elements = 1
                    for dim in shape:
                        elements *= int(dim)
                    if key.endswith("_scales"):
                        #: A per-block e8m0 exponent, not a parameter: it is
                        #: consumed by the unpacking and does not survive into
                        #: the dequantized model.
                        continue
                    if key.endswith("_blocks"):
                        #: uint8 storage holding TWO e2m1 nibbles per byte.
                        total += 2 * elements
                    else:
                        total += elements
        except (OSError, ValueError):
            return None
    return total


@contextmanager
def loaded_reference_model(model_dir: str, *, device: str = "cpu") -> Iterator[Any]:
    """Load the fp32 reference once and free it deterministically on exit.

    Split out of :func:`reference_residual_stream` so that a caller running
    several prompts pays the load exactly once.  The load is the dominant cost
    of this leg — a float32 Qwen3-8B is ~30 GiB of weights to materialize and
    move — while the forward itself is seconds, so a per-prompt reload turns an
    N-prompt Stage 0 matrix into N full loads for no numerical gain.

    Teardown lives in the ``finally`` branch rather than at the end of the body
    so the weights are released even when the caller's forward raises: on
    ``"cuda"`` the allocator would otherwise keep the whole fp32 copy reserved
    for the rest of the process, and the Stage 0 harness hands the same card on
    to the next consumer.

    Args:
        model_dir: resolved local snapshot directory for the verifier.
        device: ``"cuda"`` or ``"cpu"``; see :func:`select_reference_device`.

    Yields:
        The loaded ``AutoModelForCausalLM``, in eval mode, on *device*, with
        float32 arithmetic.

    Raises:
        ReferenceUnavailable: if the checkpoint declares a quantizer this
            module has not been shown to dequantize losslessly, or if the load
            fails on a missing optional dependency.
    """
    import torch  # lazy: only present in the vLLM venv

    # transformers ships in the vLLM venv, not the project venv; the inline
    # ignore matches the existing convention in
    # ``speedlm.training.rows.load_tokenizer_snapshot``.
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
    )

    #: ``attn_implementation="eager"`` is deliberate: the reference must be the
    #: textbook computation, not whichever fused attention kernel happens to be
    #: installed.  A fused kernel would reintroduce exactly the kind of shared
    #: implementation detail this leg exists to avoid.
    #:
    #: ``accelerate`` is not installed in the vLLM venv, so there is no
    #: ``device_map`` path: weights are materialized on the host and then moved.
    #: Loading straight to float32 would stage ~30 GiB on the host for an 8B
    #: model.  When the checkpoint is bf16/fp16, float32 is an *exact* widening
    #: of it, so the narrow dtype is loaded (half the host staging) and upcast
    #: on device -- bit-identical weights, lower peak host memory.  The
    #: arithmetic is still float32, which is the whole point.
    checkpoint_dtype = _checkpoint_dtype(model_dir)
    load_dtype = (
        getattr(torch, checkpoint_dtype)
        if device == "cuda" and checkpoint_dtype in _EXACTLY_WIDENED_TO_FP32
        else torch.float32
    )

    #: ``quantization_config=Mxfp4Config(dequantize=True)`` is set for
    #: INDEPENDENCE, not for convenience.  Left to itself the loader decides
    #: at runtime: with ``accelerate`` installed control reaches the capability
    #: gates (``transformers/quantizers/quantizer_mxfp4.py:91-147``), and
    #: because ``kernels`` is absent and the checkpoint is pre-quantized it
    #: logs "we will default to dequantizing the model to bf16" and sets
    #: ``dequantize=True`` itself.  Same weights — but implicitly, and
    #: contingent on ``kernels`` never being installed.  If ``kernels`` WERE
    #: installed the loader would instead run the native mxfp4 triton kernels,
    #: i.e. the *same quantized arithmetic vLLM runs*, and this leg would stop
    #: being an independent derivation at all.  Setting it explicitly PINS the
    #: load to transformers' own ``convert_moe_packed_tensors`` unpacking.
    #: That it also bypasses the ``accelerate`` requirement
    #: (``quantizer_mxfp4.py:72-76`` returns before the import check when
    #: ``dequantize`` is set) is a side effect, not the reason.
    #:
    #: The dequantization is bf16 yet the reference stays genuinely fp32:
    #: an mxfp4 element is an e2m1 significand (<=2 significant bits) times an
    #: e8m0 power-of-two block scale, so the product carries at most 2
    #: significant bits and bf16 holds 8.  Verified empirically:
    #: ``convert_moe_packed_tensors(b, s, dtype=bfloat16).float()`` is
    #: ``torch.equal`` to the fp32 dequantization, max abs diff 0.0.  bf16 then
    #: widened to fp32 is therefore bit-identical to fp32 throughout.
    quantization = _checkpoint_quantization(model_dir)
    load_kwargs: dict[str, Any] = {}
    if quantization == "mxfp4":
        #: The per-module mypy override for ``transformers`` (see pyproject)
        #: already covers this import; no inline ignore is needed here.
        from transformers import Mxfp4Config

        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    elif quantization:
        raise ReferenceUnavailable(
            f"checkpoint {model_dir} declares quant_method={quantization!r}, "
            f"which this reference has not been shown to dequantize losslessly. "
            f"Only 'mxfp4' is handled, and only because its e2m1-times-e8m0 "
            f"unpacking was proved bit-exact into fp32. An unrecognised "
            f"quantizer may dequantize lossily, and a lossy reference is not a "
            f"reference — it would report a tolerance it does not earn. Either "
            f"add {quantization!r} here with the same bit-exactness argument, "
            f"or run the reference leg against an unquantized snapshot of this "
            f"model.",
            model_dir=model_dir,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=load_dtype,
            attn_implementation="eager",
            **load_kwargs,
        )
    except ImportError as error:
        raise ReferenceUnavailable(
            f"loading the fp32 reference for {model_dir} failed on a missing "
            f"optional dependency: {error}. The usual candidate is "
            f"`accelerate`, which transformers' mxfp4 quantizer demands before "
            f"it reaches its capability gates "
            f"(quantizer_mxfp4.py:72-76). It can be installed with "
            f"`/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/pip "
            f"install accelerate`, but do NOT expect that to be sufficient or "
            f"even the right fix: with `kernels` absent the quantizer then just "
            f"defaults to dequantizing to bf16 anyway, i.e. it reaches the "
            f"SAME weights this module already pins explicitly via "
            f"Mxfp4Config(dequantize=True), while making the choice implicit "
            f"and contingent on `kernels` never appearing. Diagnose what "
            f"actually failed to import before installing anything.",
            model_dir=model_dir,
        ) from error
    model.eval()
    model.to(device)
    #: The upcast must run for quantized checkpoints too, and the dtype probe
    #: alone will not trigger it.  gpt-oss-20b's config.json declares no
    #: ``dtype``/``torch_dtype`` at all, so ``_checkpoint_dtype`` returns
    #: ``""``, ``load_dtype`` is already ``torch.float32``, and the
    #: ``load_dtype is not torch.float32`` test below is False.  But
    #: ``dequantize_convertops`` (``transformers/integrations/mxfp4.py:546``)
    #: calls ``convert_moe_packed_tensors`` with no dtype and that function
    #: defaults to ``torch.bfloat16`` (``mxfp4.py:337``) — so the MoE expert
    #: weights come back bf16 regardless of the requested dtype.  Without
    #: forcing the upcast they would sit at bf16 inside a nominally fp32 model
    #: and this leg would silently stop being an fp32 reference.
    if load_dtype is not torch.float32 or quantization:
        model.float()

    try:
        yield model
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def _forward_residual_stream(
    model: Any,
    token_ids: list[int],
    device: str,
) -> tuple[list[Tensor], Tensor, str]:
    """Prefill *token_ids* on an already-loaded *model* and extract the stream.

    The single copy of the forward logic: both the load-and-forward path and
    the reuse path of :func:`reference_residual_stream` call this, so the hook
    installation, the hidden-state count assertion and the index mapping cannot
    drift apart between them.

    Args:
        model: a model as yielded by :func:`loaded_reference_model`.
        token_ids: the exact prompt token ids the serving engine prefilled.
        device: the device *model* already lives on.

    Returns:
        As :func:`reference_residual_stream`.

    Raises:
        AssertionError: if HF returns an unexpected number of hidden states,
            or the hook did not fire — either would silently invalidate the
            index mapping the whole comparison rests on.
    """
    import torch  # lazy: only present in the vLLM venv

    inner = model.model
    layers = inner.layers
    num_layers = len(layers)

    #: The pre-norm output of the last decoder layer.  HF discards it from the
    #: public tuple, so it is taken straight off the module.
    captured_final: list[Tensor] = []

    def _grab_final(_module: Any, _args: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        captured_final.append(tensor)

    handle = layers[num_layers - 1].register_forward_hook(_grab_final)
    try:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    hidden = list(out.hidden_states)
    assert len(hidden) == num_layers + 1, (
        f"expected {num_layers + 1} HF hidden states for a {num_layers}-layer "
        f"model, got {len(hidden)}; the aux-layer index mapping in "
        f"speedlm.activation_capture.hf_reference is no longer valid for this "
        f"transformers version"
    )
    assert captured_final, (
        "the forward hook on the last decoder layer never fired; the pre-norm "
        "final-layer reference could not be produced"
    )

    #: Squeeze the batch dimension and move to CPU so the comparison never
    #: depends on where the reference happened to run.
    stream = [hidden[k][0].detach().cpu() for k in range(num_layers)]
    stream.append(captured_final[-1][0].detach().cpu())
    post_norm_final = hidden[num_layers][0].detach().cpu()

    return stream, post_norm_final, "torch.float32"


def reference_residual_stream(
    model_dir: str,
    token_ids: list[int],
    *,
    device: str = "cpu",
    model: Any | None = None,
) -> tuple[list[Tensor], Tensor, str]:
    """Run *token_ids* through HuggingFace transformers in float32.

    Returns the full **residual stream** — the quantity vLLM's aux collection
    stores — indexed so that entry ``k`` is directly comparable to vLLM aux
    layer id ``k`` for every ``k`` in ``0..L``:

    * entries ``0..L-1`` come from ``output_hidden_states`` unchanged;
    * entry ``L`` comes from a forward hook on ``model.layers[L-1]``, because
      HuggingFace overwrites the last tuple entry with the **post-norm**
      ``last_hidden_state`` (``transformers/utils/output_capturing.py:264-266``).

    **Why the reuse path exists.**  The default behaviour — load, forward, free
    — is correct for one prompt and wasteful for several: a Stage 0 matrix over
    N prompts would materialize and free a ~30 GiB float32 copy of an 8B model N
    times, which dominates the runtime of this leg while changing nothing about
    the numbers.  Passing an already-loaded *model* (from
    :func:`loaded_reference_model`) amortizes that over the whole matrix.  Both
    paths run the identical :func:`_forward_residual_stream` body, so the reuse
    path cannot quietly diverge from the one-shot path it replaces.

    Args:
        model_dir: resolved local snapshot directory for the verifier.  Still
            required on the reuse path so the two call shapes stay
            interchangeable at the call site.
        token_ids: the exact prompt token ids the serving engine prefilled.
        device: ``"cuda"`` or ``"cpu"``; see :func:`select_reference_device`.
        model: an already-loaded reference model to forward on.  When given,
            no load and no teardown happen here — the caller's
            :func:`loaded_reference_model` scope owns the weights.

    Returns:
        ``(residual_stream, post_norm_final, dtype_name)`` where
        ``residual_stream`` has ``L + 1`` entries of shape ``(len(token_ids),
        hidden_size)`` on CPU, and ``post_norm_final`` is HF's own
        ``hidden_states[-1]`` retained so the caller can prove the final
        RMSNorm really is applied there.

    Raises:
        AssertionError: if HF returns an unexpected number of hidden states,
            or the hook did not fire — either would silently invalidate the
            index mapping the whole comparison rests on.
    """
    if model is not None:
        return _forward_residual_stream(model, token_ids, device)
    with loaded_reference_model(model_dir, device=device) as loaded:
        return _forward_residual_stream(loaded, token_ids, device)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_to_hf_reference(
    captured: dict[int, Tensor],
    residual_stream: list[Tensor],
    post_norm_final: Tensor,
    *,
    prompt_token_count: int,
    final_layer_idx: int | None,
    device: str,
    dtype: str,
    prompt_label: str | None = None,
) -> HFReferenceResult:
    """Compare captured aux layers against the fp32 HF residual stream.

    Every captured layer is trimmed to *prompt_token_count* rows — the only
    range both sides ran the same tokens over — and then subjected to two
    independent checks:

    1. **Tolerance.** ``mean_rel_error`` against ``residual_stream[k]`` must be
       within :func:`bf16_relative_tolerance` for that depth.
    2. **Identification.** ``residual_stream[k]`` must beat both neighbours by
       at least :data:`DISCRIMINATION_MARGIN`.  This is the check that a
       lossless round-trip cannot pass by construction, and it does not depend
       on (1)'s tolerance being correctly derived.

    For the **final** aux layer the second neighbour is *post_norm_final*
    rather than ``k+1``, which does not exist in the stream; see the inline
    note at the rival-set construction for why that, and not ``k-2``, is the
    real other side.

    When *final_layer_idx* is captured, the post-norm tensor is additionally
    required to be materially *worse* than the hooked pre-norm one, which
    proves the capture is the pre-norm quantity the trainer expects rather
    than the model's normalized output.

    Args:
        captured: ``{aux layer id: tensor}`` from the serving capture.
        residual_stream: from :func:`reference_residual_stream`.
        post_norm_final: from :func:`reference_residual_stream`.
        prompt_token_count: the engine's own ``usage.prompt_tokens``.
        final_layer_idx: the appended final decoder layer, or ``None``.
        device: recorded verbatim into the result.
        dtype: recorded verbatim into the result.
        prompt_label: identifies which prompt this verdict is about, so a
            multi-prompt matrix stays attributable.  ``None`` for a
            single-prompt run.

    Raises:
        ValueError: if *prompt_token_count* is not positive, or a captured
            layer has fewer rows than the prompt, or an aux layer id has no
            counterpart in *residual_stream*.
    """
    if prompt_token_count <= 0:
        raise ValueError(
            f"prompt_token_count must be positive, got {prompt_token_count}"
        )

    layers: list[HFReferenceLayer] = []
    last_index = len(residual_stream) - 1
    for aux_idx in sorted(captured):
        if not 0 <= aux_idx <= last_index:
            raise ValueError(
                f"aux layer id {aux_idx} has no counterpart in a reference "
                f"residual stream of {len(residual_stream)} entries (0..{last_index})"
            )
        tensor = captured[aux_idx]
        if tensor.shape[0] < prompt_token_count:
            raise ValueError(
                f"captured layer {aux_idx} has {tensor.shape[0]} rows, fewer "
                f"than the prompt's {prompt_token_count}; cannot compare"
            )
        cap = tensor[:prompt_token_count]

        #: The claimed index plus whichever neighbours exist.  A layer at
        #: either end of the stack simply has one fewer competitor.
        candidates = [
            idx
            for idx in (aux_idx - 1, aux_idx, aux_idx + 1)
            if 0 <= idx <= last_index
        ]
        errors = {
            idx: mean_relative_error(cap, residual_stream[idx][:prompt_token_count])
            for idx in candidates
        }
        own = errors[aux_idx]
        rival_errors: list[tuple[str, float]] = [
            (str(idx), err) for idx, err in errors.items() if idx != aux_idx
        ]

        #: The final layer has no ``k+1`` *inside* the residual stream, so the
        #: integer candidate set above clips it away and the argmin becomes
        #: one-sided: only ``k-1`` competes, and a capture that is really the
        #: model's post-norm output wins by default.
        #:
        #: The genuine two-sided rival already exists — ``post_norm_final`` is
        #: the ONLY tensor downstream of layer ``k``, i.e. the true "k+1" on the
        #: other side — it was simply spent on the separate
        #: ``final_layer_prenorm_confirmed`` check below and never entered the
        #: argmin that sets ``identified``.  It is entered here.
        #:
        #: Adding ``k-2`` instead was rejected: the stated worry is an error
        #: metric that grows monotonically as you move away from the true layer
        #: *towards* the end of the stack, and under exactly that assumption
        #: ``err(k-2) > err(k-1)``, so ``min(rivals)`` — and therefore the
        #: pass/fail outcome — is bit-identical with or without it.  A rival
        #: that cannot change the verdict is not discrimination.
        is_final_layer = aux_idx == last_index
        if is_final_layer:
            rival_errors.append(
                (
                    "post_norm",
                    mean_relative_error(cap, post_norm_final[:prompt_token_count]),
                )
            )

        rivals = [err for _label, err in rival_errors]
        best_index = min(errors, key=lambda idx: errors[idx])
        if own <= _EPSILON:
            ratio = float("inf")
        elif rivals:
            #: Over the *enlarged* set, so this can only shrink.  That is the
            #: point: the extra rival may reveal a separation that was never
            #: there, it may never manufacture one.
            ratio = min(rivals) / own
        else:
            #: A single-entry stream leaves nothing to discriminate against;
            #: report 0.0 so ``identified`` is False rather than vacuously True.
            ratio = 0.0

        #: Bracketed on both sides.  ``aux_idx - 1`` supplies the lower side;
        #: the upper side is ``aux_idx + 1`` for an interior layer and the
        #: post-norm tensor for the final one.  Layer 0 of a real stack has no
        #: lower side and is honestly reported as one-sided.
        two_sided = (aux_idx - 1) in errors and (
            (aux_idx + 1) in errors or is_final_layer
        )
        tolerance = bf16_relative_tolerance(aux_idx)
        layers.append(
            HFReferenceLayer(
                aux_layer_idx=aux_idx,
                reference_index=aux_idx,
                rows=prompt_token_count,
                mean_rel_error=own,
                cosine_similarity=cosine_similarity(
                    cap, residual_stream[aux_idx][:prompt_token_count]
                ),
                tolerance=tolerance,
                within_tolerance=own <= tolerance,
                neighbour_rel_errors={str(k): v for k, v in errors.items()},
                best_match_index=best_index,
                discrimination_ratio=ratio,
                #: ``min(rivals) < own`` already drives ``ratio`` below 1 and so
                #: below the margin whenever a rival — including the post-norm
                #: one — is the strict minimum.  The explicit ``own <= min``
                #: term states that independently of the ratio arithmetic, so a
                #: future change to ``ratio`` cannot quietly let a strictly
                #: better rival through.
                identified=(
                    best_index == aux_idx
                    and (not rivals or own <= min(rivals))
                    and ratio >= DISCRIMINATION_MARGIN
                ),
                rival_errors=tuple(rival_errors),
                two_sided=two_sided,
            )
        )

    prenorm_confirmed: bool | None = None
    if final_layer_idx is not None and final_layer_idx in captured:
        cap_final = captured[final_layer_idx][:prompt_token_count]
        pre_error = mean_relative_error(
            cap_final, residual_stream[final_layer_idx][:prompt_token_count]
        )
        post_error = mean_relative_error(
            cap_final, post_norm_final[:prompt_token_count]
        )
        #: The capture is pre-norm iff it is much closer to the hooked pre-norm
        #: value than to HF's normalized output.  Reusing DISCRIMINATION_MARGIN
        #: keeps one notion of "materially closer" in this module.
        prenorm_confirmed = post_error >= DISCRIMINATION_MARGIN * max(
            pre_error, _EPSILON
        )

    return HFReferenceResult(
        device=device,
        dtype=dtype,
        prompt_token_count=prompt_token_count,
        layers=layers,
        final_layer_idx=final_layer_idx,
        final_layer_prenorm_confirmed=prenorm_confirmed,
        verdict=_derive_reference_verdict(layers, prenorm_confirmed),
        prompt_label=prompt_label,
    )


def _derive_reference_verdict(
    layers: list[HFReferenceLayer],
    prenorm_confirmed: bool | None,
) -> str:
    """Derive the reference verdict; see :class:`HFReferenceResult`."""
    if not layers:
        return "FAIL_empty"
    if not all(layer.within_tolerance for layer in layers):
        return "FAIL_tolerance"
    if not all(layer.identified for layer in layers):
        return "FAIL_identification"
    if prenorm_confirmed is False:
        return "FAIL_pre_norm"
    return "PASS"
