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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only executed by static type checkers.  mypy has per-module overrides for
    # torch/safetensors/vllm/transformers (see pyproject.toml), so this module
    # keeps a torch-free import surface for the project venv.
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

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "device": self.device,
            "dtype": self.dtype,
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
                    "neighbour_rel_errors": layer.neighbour_rel_errors,
                    "best_match_index": layer.best_match_index,
                    "discrimination_ratio": layer.discrimination_ratio,
                    "identified": layer.identified,
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


def cosine_similarity(captured: Tensor, reference: Tensor) -> float:
    """Flattened cosine similarity, computed in float32."""
    cap = _as_float32(captured).flatten()
    ref = _as_float32(reference).flatten()
    return float((cap * ref).sum()) / (
        float(cap.norm()) * float(ref.norm()) + _EPSILON
    )


# ---------------------------------------------------------------------------
# Reference forward
# ---------------------------------------------------------------------------


def select_reference_device(
    *,
    num_parameters: int,
    free_device_bytes: int | None,
    headroom_fraction: float = 1.35,
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
    required = int(num_parameters * 4 * headroom_fraction)
    return "cuda" if free_device_bytes >= required else "cpu"


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


def reference_residual_stream(
    model_dir: str,
    token_ids: list[int],
    *,
    device: str = "cpu",
) -> tuple[list[Tensor], Tensor, str]:
    """Run *token_ids* through HuggingFace transformers in float32.

    Returns the full **residual stream** — the quantity vLLM's aux collection
    stores — indexed so that entry ``k`` is directly comparable to vLLM aux
    layer id ``k`` for every ``k`` in ``0..L``:

    * entries ``0..L-1`` come from ``output_hidden_states`` unchanged;
    * entry ``L`` comes from a forward hook on ``model.layers[L-1]``, because
      HuggingFace overwrites the last tuple entry with the **post-norm**
      ``last_hidden_state`` (``transformers/utils/output_capturing.py:264-266``).

    Args:
        model_dir: resolved local snapshot directory for the verifier.
        token_ids: the exact prompt token ids the serving engine prefilled.
        device: ``"cuda"`` or ``"cpu"``; see :func:`select_reference_device`.

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
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=load_dtype,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    if load_dtype is not torch.float32:
        model.float()

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

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return stream, post_norm_final, "torch.float32"


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
        rivals = [err for idx, err in errors.items() if idx != aux_idx]
        best_index = min(errors, key=lambda idx: errors[idx])
        if own <= _EPSILON:
            ratio = float("inf")
        elif rivals:
            ratio = min(rivals) / own
        else:
            #: A single-entry stream leaves nothing to discriminate against;
            #: report 0.0 so ``identified`` is False rather than vacuously True.
            ratio = 0.0
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
                identified=(
                    best_index == aux_idx and ratio >= DISCRIMINATION_MARGIN
                ),
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
