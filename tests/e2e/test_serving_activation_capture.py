"""Stage 0 kill-condition prototype: serving-time activation capture.

Two comparisons run here, and they answer different questions.  Conflating them
is the mistake this docstring exists to prevent.

**Leg 1 — captured vs. vLLM's offline extraction (a TRANSPORT check).**  These
two are *not* independent derivations.  Both take the tensor that
``EagleModelMixin._maybe_add_hidden_state`` appends at vLLM
``model_executor/models/interfaces.py:1337``; the serving hook lifts that list
object straight out of ``_model_forward`` while the offline path reads the same
variable one branch later at ``v1/worker/gpu_model_runner.py:5035`` and merely
copies it (stack, same-dtype buffer assignment, integer-indexed KV scatter, the
inverse gather, a pinned same-dtype D2H copy, ``save_file``) — no
floating-point arithmetic and no dtype cast anywhere, and the offline "draft
model" has no weights (``models/extract_hidden_states.py:392-394``) and a
``pass`` forward (``:229-231``).  Bit-identical is therefore the *expected*
outcome, and a ``mean_rel_error`` of exactly 0.0 says the round trip is
lossless — it says nothing about whether the tensor is the right quantity.
What this leg does catch, and what nothing else here catches: slot-mapping
errors, layer misordering, truncation, row misalignment, prefix-cache row loss.

**Leg 2 — captured vs. HuggingFace transformers in float32 (an IDENTITY
check).**  The same prompt token ids are re-run through an implementation that
shares no kernel, no dtype and no code path with vLLM, and each captured aux
layer must both land within a derived bf16-vs-fp32 tolerance *and* be the
strict nearest match among its neighbouring residual-stream depths.  See
:mod:`speedlm.activation_capture.hf_reference` for the layer-index mapping
proof and the tolerance derivation.  This is the leg that can distinguish
"pre-norm layer 21" from "post-norm layer 21" or "layer 20".

**Of Leg 2's two checks, the neighbour discrimination is the sharp one.**  The
derived tolerance is a worst-case ceiling that assumes every per-layer bf16
rounding aligns, so a passing measurement sits far inside it: job 369256 spent
21.8 %, 11.2 %, 10.7 % and 11.3 % of the budget at aux layers 2, 18, 33 and 36
(recorded per layer as ``tolerance_budget_used``), meaning capture fidelity
could degrade roughly ninefold and ``within_tolerance`` would still say
``true``.  The discrimination ratios from the same run — 81.46, 23.14, 24.03
and 50.02 against a 3.0 margin — are what actually pin the identity.  Read
``identified``, not ``within_tolerance``, as the result.  Note also that layer
36 is the last layer, so its discrimination is **one-sided**: only neighbour 35
exists to beat.

Required environment:
* ``SPEEDLM_E2E_ACTIVATION_CAPTURE=1`` — opt-in GPU gate
* ``SPEEDLM_E2E_VERIFIER_MODEL`` — HuggingFace model ID or local path
* ``SPEEDLM_E2E_DRAFTER_MODEL`` — HuggingFace model ID or local path
* ``SPEEDLM_E2E_ARTIFACT_DIR`` — durable artifact root
* ``SPEEDLM_E2E_VLLM_PYTHON`` — path to vLLM python (default: vLLM venv)

Optional environment:
* ``SPEEDLM_E2E_READY_TIMEOUT`` — engine readiness cap in seconds (default: 360)
* ``SPEEDLM_E2E_PORT`` — vLLM serve port (default: auto-assigned free port)
* ``SPEEDLM_E2E_TARGET_LAYER_IDS`` — JSON array, e.g. ``[2, 18, 33]``.  Only
  needed to override the layers derived from the models under test.
* ``SPEEDLM_E2E_PROMPT`` — override the prompt entirely.  Wins over
  ``SPEEDLM_E2E_PROMPT_SET`` and collapses the matrix to a single case
  labelled ``custom``.
* ``SPEEDLM_E2E_PROMPT_SET`` — ``full`` (default; the four standard prompts of
  :data:`PROMPT_MATRIX`), ``minimal`` (only ``medium``, i.e. the
  pre-broadening single-prompt behaviour, for a fast smoke), or ``injection``
  (only ``control_token_injection``, the diagnostic case documented on that
  entry, which is *expected to fail* on a Harmony-templated verifier).  An
  unrecognised value is a hard error rather than a silent fallback.
* ``SPEEDLM_E2E_EXPECTED_RUNNER`` — ``auto`` (default), ``v1`` or ``v2``.
  Asserts which vLLM model-runner generation the serving engine *actually*
  loaded, as reported by the worker itself.  Pair it with
  ``VLLM_USE_V2_MODEL_RUNNER``, which is what forces the choice; see
  :func:`_assert_runner` for why requesting a generation is not evidence of
  having got one.
* ``SPEEDLM_E2E_HF_REFERENCE`` — set to ``0`` to skip the independent
  HuggingFace fp32 reference leg.  Default is on; skipping is recorded in
  ``result.json`` as ``hf_reference: null`` so a skipped run cannot be mistaken
  for a passing one.
* ``SPEEDLM_E2E_HF_REFERENCE_DEVICE`` — force ``cuda`` or ``cpu`` for the
  reference forward.  Default: ``cuda`` when the freed device can hold the
  fp32 copy, else ``cpu``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from speedlm.activation_capture.compare import (
    MatrixResult,
    PrefixCacheResult,
    PromptCase,
    align_prompt_rows,
    build_matrix_result,
    build_result,
)
from speedlm.activation_capture.hf_reference import (
    HFReferenceResult,
    assert_reference_fits,
    checkpoint_parameter_count,
    compare_to_hf_reference,
    loaded_reference_model,
    reference_residual_stream,
    select_reference_device,
)
from speedlm.activation_capture.offline_extract import (
    extract as _run_offline_extract,
)
from speedlm.gateway.control import (
    GPUMemoryPrecondition,
    NvidiaSmiMemoryProbe,
)
from speedlm.profiles import (
    ProfileError,
    resolve_profile,
    resolve_target_layer_ids,
)

# torch and safetensors live in the vLLM venv, not the project venv. These are
# imported defensively rather than via a module-level ``pytest.importorskip``:
# importorskip aborts collection, so the whole file reported as zero tests on
# the project venv -- a silent pass, not a skip. The skip is declared as a
# ``pytestmark`` below so both tests always collect and are reported.
try:
    import torch
except ImportError:  # pragma: no cover - depends on the interpreter in use
    torch = None  # type: ignore[assignment]

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - depends on the interpreter in use
    safe_open = None  # type: ignore[assignment]

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        torch is None or safe_open is None,
        reason=(
            "torch/safetensors are not installed in the project venv; run with "
            "PYTHONPATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/"
            "lib/python3.12/site-packages to execute these tests"
        ),
    ),
]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM_PYTHON = Path(
    os.environ.get("SPEEDLM_E2E_VLLM_PYTHON", str(VLLM_VENV / "bin" / "python"))
)
SPECULATORS_REPO = Path("/admin/home/ryan.kim/speedlm/.preflight/speculators")

# Short fixed prompt for the experiment
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."

#: vLLM's default KV block size on CUDA (``V/config/cache.py:47``,
#: ``DEFAULT_BLOCK_SIZE: ClassVar[int] = 16``; FlashAttention keeps 16 on CUDA,
#: ``V/v1/attention/backends/flash_attn.py:81-85``).  The prefix-cache test does
#: not pass ``--block-size``, so this is what it gets.
PREFIX_CACHE_BLOCK_SIZE: Final[int] = 16

#: Blocks that can never be reported as a prefix-cache hit, however long the
#: prompt is:
#:
#: * one for the trailing partial block, which is never hashed at all --
#:   ``V/v1/core/kv_cache_utils.py:709-712`` breaks out of the hashing loop with
#:   "We only hash full blocks", and ``V/v1/core/kv_cache_manager.py:231`` caps
#:   the lookup at ``request.num_tokens - 1`` so the last token is always
#:   recomputed for its logits;
#: * one more because eagle-family speculative decoding drops the last matched
#:   block -- ``V/v1/core/single_type_kv_cache_manager.py:602-605``,
#:   ``if drop_eagle_block and computed_blocks[0]: computed.pop()``, reached
#:   because ``eagle3`` is in ``SpeculativeConfig.use_eagle()``
#:   (``V/config/speculative.py:1238-1242``).
#:
#: So the hit length in blocks is ``(N - 1) // 16 - 1``.
PREFIX_CACHE_UNHITTABLE_BLOCKS: Final[int] = 2

#: How many blocks must actually hit for the test to be exercising hazard 6.1
#: rather than measuring the floor.  Four is arbitrary but deliberate: it puts
#: the prompt several blocks past the point where a hit becomes possible at all,
#: so the measurement does not sit on a cliff edge.
PREFIX_CACHE_MIN_HIT_BLOCKS: Final[int] = 4

#: Prompt long enough that a repeat of it spans several cacheable blocks.
#:
#: **This length is the point of the prompt.**  The previous version of
#: ``test_prefix_cache_coverage`` reused :data:`DEFAULT_PROMPT`, which renders to
#: 18 tokens through Qwen3's chat template.  With a 16-token block that is one
#: hashable block, the lookup is capped at 17 tokens (one block), and eagle3
#: pops that single matched block -- so the measured result was
#: ``queries 18 -> 36, hits 0 -> 0``.  Zero hits was *correct engine behaviour*,
#: not a capture bug and not a misconfiguration; the prompt was simply too short
#: for a hit to be representable.  At 165 tokens this renders to 10 hashable
#: blocks, 9 of which are hittable, which the hazard can actually be observed on.
PREFIX_CACHE_PROMPT: Final[str] = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 4


def _min_prefix_cache_prompt_tokens() -> int:
    """Shortest prompt that can show :data:`PREFIX_CACHE_MIN_HIT_BLOCKS` hits.

    Derived from the block arithmetic above rather than written down as a
    number, so that changing the block size or the required margin cannot leave
    a stale literal behind.
    """
    blocks = PREFIX_CACHE_MIN_HIT_BLOCKS + PREFIX_CACHE_UNHITTABLE_BLOCKS
    return blocks * PREFIX_CACHE_BLOCK_SIZE + 1


def _expected_prefix_cache_hit_tokens(prompt_token_count: int) -> int:
    """Hit tokens vLLM should report for a re-sent *prompt_token_count* prompt.

    ``(N - 1) // block_size`` blocks are scanned
    (``V/v1/core/single_type_kv_cache_manager.py:590``), the last matched one is
    dropped for eagle (``:602-605``), and the hit is reported in tokens
    (``V/v1/metrics/stats.py:115-142``, "the number of tokens that were
    queried").  Clamped at zero: for a short enough prompt the correct answer is
    genuinely no hit.
    """
    scanned = (prompt_token_count - 1) // PREFIX_CACHE_BLOCK_SIZE
    return max(0, scanned - 1) * PREFIX_CACHE_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Prompt matrix
# ---------------------------------------------------------------------------
#
# Stage 0 used to run exactly one 18-token prompt.  One prompt of one length on
# one template is a single point, and three of the four failure modes this file
# exists to catch are *length- or rendering-dependent*: an off-by-one in the
# prompt/template split, a row loss that only appears once the prompt spans more
# than one KV block, and a divergence between the chat template this test
# renders locally and the one vLLM applied.  The matrix below adds one prompt per
# regime and keeps the original one unchanged, so the broadening is strictly
# additive and the historical datapoint stays directly comparable.
#
# **Every prompt here must fit inside ``--max-model-len 512``**, which is what
# both engines in this file are launched with.  There is no separate budget for
# the chat template: 512 is the total the engine will accept, prompt plus
# rendered template plus the 16 sampled tokens.  ``multi_block`` is the only one
# anywhere near the bound and is kept comfortably under it (see below).


#: Repeated unit for the ``multi_block`` prompt.  A pangram is used so the
#: tokenizer cannot collapse the repeat into a handful of merged tokens.
_MULTI_BLOCK_UNIT: Final[str] = "The quick brown fox jumps over the lazy dog. "

#: Deliberately *pessimistic* lower bound on how many tokens
#: :data:`_MULTI_BLOCK_UNIT` renders to.  It is nine words plus punctuation, and
#: :data:`DEFAULT_PROMPT` — the same sentence — measures 18 tokens *including*
#: Qwen3's chat-template wrapper, so ten is a bound the content alone clears on
#: any BPE vocabulary.  Under-estimating here makes the prompt longer than
#: required, which is harmless; over-estimating would make it too short to span
#: the blocks it is named after, which is the failure this constant prevents.
_MULTI_BLOCK_MIN_TOKENS_PER_UNIT: Final[int] = 10

#: Repeats needed to clear :func:`_min_prefix_cache_prompt_tokens`.  Derived
#: from the same block arithmetic ``test_prefix_cache_coverage`` uses rather
#: than written down as a token count, so changing
#: :data:`PREFIX_CACHE_BLOCK_SIZE` or :data:`PREFIX_CACHE_MIN_HIT_BLOCKS`
#: re-sizes this prompt too instead of leaving a stale literal behind.
#: At the current constants that is ``ceil(97 / 10) = 10`` repeats, ~110 tokens
#: of content — nine blocks past the point where multi-block behaviour becomes
#: representable at all, and roughly a fifth of the 512-token model length.
_MULTI_BLOCK_REPEATS: Final[int] = -(
    -_min_prefix_cache_prompt_tokens() // _MULTI_BLOCK_MIN_TOKENS_PER_UNIT
)

#: The four prompts, as ``(label, text)``.  Order is the order they are run and
#: the order they appear in ``result.json``; ``medium`` is deliberately not
#: first, so a matrix that silently ran only its first case cannot accidentally
#: reproduce the old single-prompt result and look unchanged.
PROMPT_MATRIX: Final[tuple[tuple[str, str], ...]] = (
    #: **short** — the chat template's own tokens outnumber the content's.
    #: "Hi." is two or three tokens against a wrapper of a dozen or more, so an
    #: off-by-one in the prompt/template split is proportionally largest here:
    #: one misplaced row is several percent of the comparison rather than the
    #: fraction of a percent it would be on a long prompt.  The same absolute
    #: bug that hides inside tolerance on ``multi_block`` is loud on this one.
    ("short", "Hi."),
    #: **medium** — EXACTLY :data:`DEFAULT_PROMPT`, referenced rather than
    #: retyped.  This is the prompt job 369256 measured (discrimination ratios
    #: 81.46 / 23.14 / 24.03 / 50.02 at aux layers 2 / 18 / 33 / 36), so keeping
    #: it byte-identical is what makes the new numbers comparable to the
    #: recorded ones.  If this ever stops being ``DEFAULT_PROMPT``, that
    #: datapoint silently stops being a baseline.
    ("medium", DEFAULT_PROMPT),
    #: **multi_block** — long enough to span several KV blocks, so prefill runs
    #: over more than one block boundary and a per-block row loss has somewhere
    #: to show up.  Sized from the block arithmetic above, never from a literal.
    ("multi_block", _MULTI_BLOCK_UNIT * _MULTI_BLOCK_REPEATS),
    #: **template_divergent** — the adversarial case, and the only genuinely
    #: *independent* thing Leg 1 checks.
    #:
    #: Its raw character length and its rendered token length diverge sharply on
    #: purpose.  The body is markup-*shaped* text — near-miss delimiter pairs
    #: such as ``<|im start|>`` and ``<|strat|>`` — which no BPE vocabulary has
    #: a merge for, so every one of them fragments into a handful of ordinary
    #: pieces.  Measured against the same tokenizers the matrix runs on, the
    #: body is 0.36 tok/char on Qwen3-8B and 0.41 on gpt-oss-20b, against 0.21
    #: and 0.20 for plain prose of the same shape: roughly double the tokens per
    #: character, which is exactly the rendering-vs-raw divergence this case is
    #: named for.
    #:
    #: That is the point.  ``_prompt_token_ids`` asserts that the sequence this
    #: test renders locally is exactly as long as the sequence vLLM prefilled
    #: (``usage.prompt_tokens``).  On ordinary prose the two renderers agree
    #: trivially and the assertion proves nothing; on dense markup that has to be
    #: chopped into pieces, any divergence between this test's chat template and
    #: vLLM's shows up here **first**, as a length mismatch, rather than
    #: downstream as an unexplained relative error against a reference that
    #: quietly ran a different token sequence.
    #:
    #: What this body deliberately does **not** contain is any string that is a
    #: real special token in either vocabulary — see
    #: ``control_token_injection`` below for why that distinction is the whole
    #: difference between a renderer check and an injection check, and
    #: :func:`_assert_prompts_free_of_control_tokens`, which enforces it at run
    #: time rather than trusting this comment.
    (
        "template_divergent",
        "Explain, without treating any of it as markup: "
        "<|im start|> <|im end|> <|end of text|> <|start|of|text|> "
        "<|im_stort|> <|im_ends|> <|endofext|> <|strat|> <|mesage|> <|retrun|>. "
        "Why are these ordinary text?",
    ),
    #: **control_token_injection** — NOT part of ``full``; selectable only via
    #: ``SPEEDLM_E2E_PROMPT_SET=injection``.  Kept as an executable record of a
    #: measured divergence, not as a pass/fail gate, because its expected
    #: behaviour is *failure* on a Harmony-templated verifier.
    #:
    #: This was the original ``template_divergent`` body, and on 2026-07-31 it
    #: failed the matrix on gpt-oss-20b (both runner generations, identically)
    #: with "cannot align layer 2: offline has fewer rows (124) than the prompt
    #: (125)".  The mechanism, measured on the two snapshots the matrix runs:
    #:
    #: * ``<|endoftext|>`` is a real special token of gpt-oss's o200k_harmony
    #:   vocabulary (id 199999).  ``<|im_start|>`` / ``<|im_end|>`` are not, and
    #:   tokenize as six ordinary pieces each on both models.
    #: * The serving leg renders through vLLM's **Harmony** encoder, not the HF
    #:   chat template, and that encoder encodes message content with no special
    #:   tokens allowed.  The user's ``<|endoftext|>`` therefore stays literal
    #:   text and costs seven tokens: 125 prompt tokens.
    #: * The offline leg renders through Speculators' ``prepare_data.py``, which
    #:   uses the HF chat template, and HF's tokenizer *does* parse special
    #:   tokens out of ordinary text.  The same ``<|endoftext|>`` collapses to
    #:   the single id 199999: 120 tokens for the user turn plus four for the
    #:   assistant turn the conversation renderer appends — 124 rows.
    #:
    #: Both counts are internally correct; they disagree because two different
    #: renderers ran, and they can only disagree when the user's own content
    #: contains a string that *is* a control token.  Harmony's behaviour is the
    #: defensible one — user content must never contribute control tokens — so
    #: nothing here is "fixed" by making the legs agree on this input.  Qwen3-8B
    #: passes it only because both legs go through the same HF template, which
    #: means the injection lands identically on both sides rather than not
    #: landing at all.
    (
        "control_token_injection",
        "Explain what <|im_start|>user and <|im_end|> mean when they appear "
        "verbatim inside a message body, e.g. <|im_start|>system<|im_end|>, "
        "and why <|endoftext|> is not the same thing.",
    ),
)

#: The label whose body is allowed to contain real control tokens.  Everything
#: else in :data:`PROMPT_MATRIX` is checked against the verifier's vocabulary
#: before the engine is asked to render it.
INJECTION_PROMPT_LABEL: Final[str] = "control_token_injection"

#: The four prompts a normal run covers, in run order.  Written out rather than
#: derived from :data:`PROMPT_MATRIX`, because the matrix now also holds a case
#: that must *not* run by default.
_STANDARD_PROMPT_LABELS: Final[tuple[str, ...]] = (
    "short",
    "medium",
    "multi_block",
    "template_divergent",
)

#: Named subsets selectable with ``SPEEDLM_E2E_PROMPT_SET``.  ``minimal`` is the
#: exact pre-matrix behaviour — one prompt, ``DEFAULT_PROMPT`` — retained as a
#: fast smoke; it is a *named* subset rather than "whatever runs quickly" so a
#: run that used it says so in the artifact.
PROMPT_SETS: Final[Mapping[str, tuple[str, ...]]] = {
    "full": _STANDARD_PROMPT_LABELS,
    "minimal": ("medium",),
    "injection": (INJECTION_PROMPT_LABEL,),
}

#: Every matrix entry must be reachable from some named set.  Without this a
#: prompt added to :data:`PROMPT_MATRIX` but forgotten in :data:`PROMPT_SETS`
#: would be dead text that no run can ever select, which is the same coverage
#: loss the matrix was introduced to remove — just quieter.
assert {label for label, _text in PROMPT_MATRIX} == {
    label for labels in PROMPT_SETS.values() for label in labels
}

#: Label given to a ``SPEEDLM_E2E_PROMPT`` override.  Not one of the matrix
#: labels, so an artifact can never confuse an operator's ad-hoc prompt with a
#: standard case whose numbers are supposed to be comparable across runs.
CUSTOM_PROMPT_LABEL: Final[str] = "custom"


def _prompt_set() -> tuple[tuple[str, str], ...]:
    """Return the ``(label, text)`` prompts this run must cover.

    ``SPEEDLM_E2E_PROMPT`` wins outright, preserving the pre-existing override:
    an operator who names a prompt gets that prompt and nothing else, labelled
    :data:`CUSTOM_PROMPT_LABEL`.

    Otherwise ``SPEEDLM_E2E_PROMPT_SET`` selects a named subset of
    :data:`PROMPT_MATRIX`, defaulting to ``full``.  An unrecognised value raises
    rather than falling back: a typo'd set name that quietly degraded to one
    prompt would shrink coverage while still reporting PASS, which is the
    failure mode the matrix exists to remove.
    """
    override = os.environ.get("SPEEDLM_E2E_PROMPT")
    if override:
        return ((CUSTOM_PROMPT_LABEL, override),)

    name = os.environ.get("SPEEDLM_E2E_PROMPT_SET", "full")
    labels = PROMPT_SETS.get(name)
    if labels is None:
        raise AssertionError(
            f"SPEEDLM_E2E_PROMPT_SET={name!r} is not a known prompt set; valid "
            f"values are {sorted(PROMPT_SETS)}.  Set SPEEDLM_E2E_PROMPT to run "
            f"a one-off prompt instead."
        )
    by_label = dict(PROMPT_MATRIX)
    return tuple((label, by_label[label]) for label in labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_environment() -> tuple[str, str, Path]:
    """Check required environment and return (verifier, drafter, artifact_root)."""
    if os.environ.get("SPEEDLM_E2E_ACTIVATION_CAPTURE") != "1":
        pytest.skip("set SPEEDLM_E2E_ACTIVATION_CAPTURE=1 in an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), (
        "activation capture E2E must run inside a SLURM allocation"
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), (
        "SLURM allocation did not expose a GPU"
    )

    verifier = os.environ.get("SPEEDLM_E2E_VERIFIER_MODEL")
    if not verifier:
        raise AssertionError("SPEEDLM_E2E_VERIFIER_MODEL is required")
    drafter = os.environ.get("SPEEDLM_E2E_DRAFTER_MODEL")
    if not drafter:
        raise AssertionError("SPEEDLM_E2E_DRAFTER_MODEL is required")

    artifact_root = os.environ.get("SPEEDLM_E2E_ARTIFACT_DIR")
    if not artifact_root:
        raise AssertionError("SPEEDLM_E2E_ARTIFACT_DIR is required")
    artifact_path = Path(artifact_root)
    assert artifact_path.exists(), f"artifact dir does not exist: {artifact_path}"

    return verifier, drafter, artifact_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ready_timeout() -> float:
    raw = os.environ.get("SPEEDLM_E2E_READY_TIMEOUT", "360")
    try:
        return float(raw)
    except ValueError as exc:
        raise AssertionError("SPEEDLM_E2E_READY_TIMEOUT must be a number") from exc


#: Keys a Speculators/EAGLE-3 drafter config may use to declare its aux layers.
#: ``eagle_aux_hidden_state_layer_ids`` pins the indices outright;
#: ``num_aux_hidden_states`` pins only the arity, which is what
#: ``fc_input_size`` is built from.  Neither drafter this deployment uses
#: states the indices: RedHatAI/Qwen3-8B-speculator.eagle3 omits the key
#: entirely and RedHatAI/gpt-oss-20b-speculator.eagle3 carries it as null.
_AUX_LAYER_IDS_KEY: Final = "eagle_aux_hidden_state_layer_ids"
_AUX_COUNT_KEY: Final = "num_aux_hidden_states"
#: Nested sections a drafter config may bury the declaration under.
_AUX_CONFIG_SECTIONS: Final = ("speculators_config", "eagle_config")


def _resolve_model_dir(model: str, *, override: str | None = None) -> Path:
    """Resolve a repo id (or path) to the cached snapshot directory on disk.

    These runs set ``HF_HUB_OFFLINE=1``, so there is no download fallback: the
    snapshot must already be in the cache under ``HF_HOME``.  This mirrors
    ``_resolve_drafter_dir`` in the hot-swap E2E rather than inventing a
    second resolution rule.
    """
    if override:
        path = Path(override)
        assert (path / "config.json").is_file(), (
            f"model directory override has no config.json: {path}"
        )
        return path

    direct = Path(model)
    if (direct / "config.json").is_file():
        return direct

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "models--" + model.replace("/", "--")
    repository = hf_home / "hub" / slug
    snapshots = sorted((repository / "snapshots").glob("*"))
    usable = [path for path in snapshots if (path / "config.json").is_file()]
    assert usable, (
        f"cannot resolve {model!r} to a cached snapshot under {repository}; "
        f"the run is offline, so the snapshot must already be in HF_HOME"
    )
    return usable[-1]


def _read_model_config(model_dir: Path) -> Mapping[str, Any]:
    """Read and parse ``config.json`` from a resolved snapshot directory."""
    path = model_dir / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"cannot read model config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AssertionError(f"model config is not a JSON object: {path}")
    return raw


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _drafter_aux_declaration(
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...] | None, int | None]:
    """Return the drafter's ``(declared aux layer ids, declared aux count)``.

    Both are ``None`` when the drafter declares neither, which is the normal
    case for the published RedHatAI speculators.  A declared id list also
    fixes the count, so the count is taken from its length in that case.
    """
    sections: list[Mapping[str, Any]] = [config]
    sections.extend(
        section
        for name in _AUX_CONFIG_SECTIONS
        if isinstance(section := config.get(name), Mapping)
    )

    declared_ids: tuple[int, ...] | None = None
    for section in sections:
        value = section.get(_AUX_LAYER_IDS_KEY)
        if isinstance(value, list) and value:
            if not all(
                not isinstance(entry, bool) and isinstance(entry, int)
                for entry in value
            ):
                raise AssertionError(
                    f"{_AUX_LAYER_IDS_KEY} must be a list of integers, got {value!r}"
                )
            declared_ids = tuple(int(entry) for entry in value)
            break

    declared_count: int | None = None
    for section in sections:
        declared_count = _positive_int(section.get(_AUX_COUNT_KEY))
        if declared_count is not None:
            break

    if declared_ids is not None and declared_count is None:
        declared_count = len(declared_ids)
    return declared_ids, declared_count


def _verifier_num_hidden_layers(config: Mapping[str, Any]) -> int | None:
    """Read the verifier's decoder depth, honouring a ``text_config`` nest."""
    text_config = config.get("text_config")
    for section in (config, text_config):
        if isinstance(section, Mapping):
            depth = _positive_int(section.get("num_hidden_layers"))
            if depth is not None:
                return depth
    return None


def _target_layer_ids(verifier: str, drafter: str) -> list[int]:
    """Derive the target aux layer IDs for the offline extraction path.

    The serving engine derives its aux layers from the drafter's declaration
    and the verifier's depth; offline extraction must key on the same IDs or
    the elementwise comparison is meaningless.  A constant default cannot do
    that -- it is model-specific, and the previous one ([4, 12, 20]) matched
    neither drafter this deployment runs (job 369214).

    Resolution order, mirroring ``profiles.resolve_target_layer_ids``:

    1. ``SPEEDLM_E2E_TARGET_LAYER_IDS`` -- the operator's explicit override.
    2. The drafter's own ``eagle_aux_hidden_state_layer_ids``, when it pins a
       real list.
    3. ``profiles.resolve_target_layer_ids`` over the profile's pin, the
       verifier's ``num_hidden_layers`` read off disk, and the drafter's
       declared arity -- i.e. exactly the production resolution path.

    This reads the *inputs* the engine reads (the two on-disk configs), never
    the engine's own report.  The engine's captured ``original_aux_layers``
    therefore remains an independent value, and the assertion comparing the
    two still fails whenever vLLM's in-engine derivation and SpeedLM's
    resolution disagree.
    """
    raw = os.environ.get("SPEEDLM_E2E_TARGET_LAYER_IDS")
    if raw:
        try:
            override = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssertionError("SPEEDLM_E2E_TARGET_LAYER_IDS must be JSON") from exc
        if not isinstance(override, list) or not override:
            raise AssertionError(
                "SPEEDLM_E2E_TARGET_LAYER_IDS must be a non-empty JSON array"
            )
        return sorted(int(entry) for entry in override)

    drafter_config = _read_model_config(
        _resolve_model_dir(
            drafter,
            override=os.environ.get("SPEEDLM_E2E_DRAFTER_DIR"),
        )
    )
    declared_ids, declared_count = _drafter_aux_declaration(drafter_config)
    if declared_ids is not None:
        return sorted(declared_ids)

    num_hidden_layers = _verifier_num_hidden_layers(
        _read_model_config(_resolve_model_dir(verifier))
    )

    profile = None
    try:
        profile = resolve_profile(served_model=verifier)
    except ProfileError:
        #: An unprofiled verifier is legitimate here -- the derivation only
        #: needs the depth, which was read off disk above.
        profile = None
    if num_hidden_layers is None and profile is not None:
        num_hidden_layers = profile.num_hidden_layers

    resolved = resolve_target_layer_ids(
        explicit=profile.target_layer_ids if profile is not None else None,
        num_hidden_layers=num_hidden_layers,
        drafter_aux_count=declared_count,
    )
    return sorted(resolved)


def _create_artifact_dir(root: Path) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = root / f"activation-capture-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vllm_env() -> dict[str, str]:
    """Return a copy of os.environ with VLLM_SERVER_DEV_MODE enabled.

    vLLM only registers the /collective_rpc dev endpoint when
    VLLM_SERVER_DEV_MODE is truthy.
    """
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def _read_log_tail(log_path: Path, lines: int = 100) -> str:
    """Return the last *lines* of a log file, or a short fallback."""
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        all_lines = raw.splitlines()
        tail = all_lines[-lines:]
        return "\n".join(tail)
    except FileNotFoundError:
        return "(log file not found)"


def _wait_for_ready(
    url: str,
    process: subprocess.Popen[bytes],
    timeout: float,
    *,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            rc = process.poll()
            if rc is not None:
                log_tail = _read_log_tail(log_path)
                raise AssertionError(
                    f"vLLM exited before readiness with code {rc}\n"
                    f"--- vLLM log (last 100 lines) ---\n"
                    f"{log_tail}\n"
                    f"--- end of vLLM log ---"
                )
            try:
                resp = client.get(f"{url}/health")
                if 200 <= resp.status_code < 300:
                    return
                last_error = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.5)
    log_tail = _read_log_tail(log_path)
    raise AssertionError(
        f"vLLM did not become ready within {timeout}s; {last_error}\n"
        f"--- vLLM log (last 100 lines) ---\n"
        f"{log_tail}\n"
        f"--- end of vLLM log ---"
    )


def _get_served_model_id(url: str) -> str:
    """Query /v1/models and return the first served model id.

    vLLM registers the model under its resolved snapshot path (e.g.
    /data/.../snapshots/<commit>), not the friendly repo id.  Using the
    wrong id causes a 404 — not a 400 — so we always resolve it here.

    Also verifies that /v1/chat/completions is routable by inspecting the
    OpenAPI schema; raises with a clear message if the endpoint is missing.
    """
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        resp = client.get(f"{url}/v1/models")
        resp.raise_for_status()
        body = resp.json()
        model_ids = [m["id"] for m in body.get("data", [])]
    if not model_ids:
        raise AssertionError(
            "/v1/models returned no served models — the engine may not have "
            "finished loading"
        )

    # Verify /v1/chat/completions is routable via the OpenAPI schema.
    # Do NOT use HEAD — /v1/chat/completions is POST-only and returns 405
    # for HEAD, which would be a false "route missing" signal.
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        openapi = client.get(f"{url}/openapi.json").json()
        paths = openapi.get("paths", {})
    if "/v1/chat/completions" not in paths:
        routes = sorted(
            path for path in paths if not path.startswith("/collective")
        )
        raise AssertionError(
            f"deployment does not serve /v1/chat/completions; "
            f"available routes: {routes}"
        ) from None

    return model_ids[0]


def _send_prompt(
    url: str, prompt: str, *, served_model_id: str
) -> tuple[str, int]:
    """Send a single chat completion request.

    Uses /v1/chat/completions with a messages array so that vLLM applies
    the model's chat template — matching the offline extraction path, which
    also applies the chat template via prepare_data.py / apply_chat_template.

    ``served_model_id`` must be the exact id from /v1/models (the resolved
    snapshot path), not a friendly repo id like "openai/gpt-oss-20b".

    Returns:
        ``(output_text, prompt_token_count)``.  ``prompt_token_count`` is the
        engine's own ``usage.prompt_tokens``: the number of rows the prefill
        produced, and the only row range whose tokens the offline path also
        ran.  It is emphatically NOT the offline tensor's row count — see
        :func:`speedlm.activation_capture.compare.align_prompt_rows`.
    """
    with httpx.Client(timeout=120.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": served_model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
            },
        )
        if resp.status_code == 404:
            # Self-diagnosing: show what we sent vs. what is served.
            try:
                with httpx.Client(timeout=10.0, trust_env=False) as probe:
                    models_resp = probe.get(f"{url}/v1/models")
                    models_resp.raise_for_status()
                    served = [m["id"] for m in models_resp.json().get("data", [])]
            except Exception:
                served = ["(unable to query /v1/models)"]
            raise AssertionError(
                f"404 from /v1/chat/completions — model id mismatch. "
                f"Sent model={served_model_id!r}, served model ids={served}"
            ) from None
        resp.raise_for_status()
        data = resp.json()
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise AssertionError(
            "chat completion response carried no usable usage.prompt_tokens "
            f"({prompt_tokens!r}); the comparison cannot know how many "
            "captured rows are prompt rows and would silently compare "
            "generated tokens against template tokens"
        )
    return data["choices"][0]["message"]["content"], prompt_tokens


def _collective_rpc_results(
    vllm_proc: subprocess.Popen[bytes], port: int, method: str, *args: object
) -> list[Any]:
    """Issue a collective_rpc call and RETURN the per-worker return values.

    :func:`_collective_rpc` throws the workers' answers away, which is right for
    the commands that have no answer (``activate_capture``, ``flush_capture``)
    and wrong for the queries that do.  Rather than duplicate the error-checking
    here, this is the single implementation and ``_collective_rpc`` is a thin
    discard-the-result wrapper over it, so the two can never drift apart on what
    counts as a worker-side failure.

    vLLM's dev router (``vllm/entrypoints/serve/dev/rpc/api_router.py:48-54``)
    passes ``None``, ``dict`` and ``list`` returns through verbatim and
    stringifies everything else, so a handler that returns a dict — as
    ``ActivationCaptureExtension.runner_info`` does — arrives as a real dict and
    needs no re-parsing.

    Returns:
        One entry per worker, in worker order.  An empty list when the engine
        answered 200 with no body at all (``:46-47``, the ``results is None``
        branch) — the caller must decide whether "no worker answered" is
        acceptable for its method, because for a *query* it never is.
    """
    url = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/collective_rpc",
            json={"method": method, "args": [str(a) for a in args]},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"collective_rpc {method} failed: {resp.status_code} {resp.text}"
            )
        if not resp.content:
            return []
        # vLLM returns {"results": [...]} on success.  Each entry is the
        # worker's return value (None, a dict/list, or str(result)).  If a
        # worker-side method raised, the entry may contain error information.
        body = resp.json()
    results = body.get("results")
    if not isinstance(results, list):
        return []
    for i, result in enumerate(results):
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"collective_rpc {method} worker {i} error: {result['error']}"
            )
    return results


def _collective_rpc(
    vllm_proc: subprocess.Popen[bytes], port: int, method: str, *args: object
) -> None:
    """Issue a collective_rpc call to the vLLM engine via the debug endpoint.

    vLLM exposes a /collective_rpc endpoint that forwards to all workers.
    The caller must pass the actual port the engine is listening on.

    Signature and behaviour are unchanged; use
    :func:`_collective_rpc_results` when the workers' return values matter.
    """
    _collective_rpc_results(vllm_proc, port, method, *args)


#: Values ``SPEEDLM_E2E_EXPECTED_RUNNER`` accepts.  ``auto`` means "record
#: whichever runner vLLM chose", not "skip the check": ``_assert_runner`` still
#: refuses an ``unknown`` generation under ``auto``.
_EXPECTED_RUNNERS: Final[tuple[str, ...]] = ("auto", "v1", "v2")


def _expected_runner() -> str:
    """Return the model-runner generation this run demands, or ``"auto"``."""
    value = os.environ.get("SPEEDLM_E2E_EXPECTED_RUNNER", "auto")
    assert value in _EXPECTED_RUNNERS, (
        f"SPEEDLM_E2E_EXPECTED_RUNNER={value!r} is not one of "
        f"{list(_EXPECTED_RUNNERS)}"
    )
    return value


def _assert_runner(vllm_proc: subprocess.Popen[bytes], port: int) -> str:
    """Prove which vLLM model-runner generation the engine actually loaded.

    **This exists because the runner axis has been assumed twice and been wrong
    both times.**  A V1-only capture hook and a V1-only drafter lookup each
    shipped undetected, because every run that exercised them merely *requested*
    a generation and nothing ever checked what it got.  Requesting is not
    getting, and the gap is not hypothetical — it is how vLLM is written:

    * ``vllm/config/vllm.py:519-522`` — ``use_v2_model_runner`` returns
      ``envs.VLLM_USE_V2_MODEL_RUNNER`` verbatim when it is set, before any
      capability check, so the variable really is a forcing knob;
    * ``vllm/envs.py:264`` and ``:1867-1868`` — that variable is declared and
      parsed through ``maybe_convert_bool``;
    * ``vllm/config/vllm.py:546-553`` — when it is **unset**, vLLM may silently
      downgrade V2 to V1 and only *log* the fact, which no assertion reads;
    * ``vllm/config/vllm.py:2137-2147`` — when it is **set to 1** on an
      unsupported config, vLLM raises instead of downgrading, so a forced run
      that starts at all really is running what it asked for;
    * ``vllm/v1/worker/gpu_worker.py:384-398`` — the worker selects the runner
      class off that property.

    So the answer is knowable, and it is read from the worker itself
    (``ActivationCaptureExtension.runner_info``) rather than inferred from the
    environment the harness set.

    Returns:
        The observed generation (``"v1"`` or ``"v2"``), for recording into
        :attr:`MatrixResult.runner` so the artifact says which runner ran even
        when the run did not pin one.
    """
    infos = [
        info
        for info in _collective_rpc_results(vllm_proc, port, "runner_info")
        if isinstance(info, dict)
    ]
    assert infos, (
        "collective_rpc runner_info returned no worker results, so this run "
        "cannot say which model runner it exercised; every downstream number "
        "would be attributed to a runner nobody verified"
    )
    logger.info("vLLM runner_info (per worker): %s", infos)

    generations = {str(info.get("generation")) for info in infos}
    assert len(generations) == 1, (
        f"workers disagree about the model runner generation: {infos}.  A "
        f"split engine cannot be summarized by one runner label, and the "
        f"capture hook only intercepts the generation it was written for"
    )
    observed = generations.pop()

    #: ``unknown`` means the hook resolved *no* interception point against the
    #: live runner class, i.e. nothing was captured from a known location.  The
    #: whole capture is unsound at that point, so this fails ahead of the
    #: expected-runner comparison and regardless of it.
    assert observed != "unknown", (
        f"the capture hook found no interception point on runner class "
        f"{infos[0].get('runner_class')!r}, so it does not know where — or "
        f"whether — it captured anything.  Every tensor this run produced is "
        f"of unproven provenance; the comparison below would be measuring the "
        f"transport of an unidentified quantity"
    )

    expected = _expected_runner()
    if expected != "auto":
        assert observed == expected, (
            f"SPEEDLM_E2E_EXPECTED_RUNNER={expected!r} but the engine loaded "
            f"the {observed!r} model runner (runner_class="
            f"{infos[0].get('runner_class')!r}, hook_point="
            f"{infos[0].get('hook_point')!r}, config_use_v2="
            f"{infos[0].get('config_use_v2')!r}).  VLLM_USE_V2_MODEL_RUNNER is "
            f"what forces this choice; the request was ignored or downgraded "
            f"(vllm/config/vllm.py:546-553 downgrades silently when the "
            f"variable is unset).  This run did NOT exercise the runner it "
            f"claims to have exercised"
        )
    return observed


def _load_captured_safetensors(capture_dir: Path) -> dict[int, torch.Tensor]:
    """Load the captured.safetensors file and return {layer_idx: tensor}."""
    path = capture_dir / "captured.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"no captured.safetensors in {capture_dir}")
    tensors: dict[int, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118  (safe_open handle, not a dict)  # noqa: SIM118 (safe_open handle, not a dict)
            if key.startswith("layer_"):
                idx = int(key.split("_", 1)[1])
                tensors[idx] = f.get_tensor(key)
    return tensors


def _load_capture_metadata(capture_dir: Path) -> dict:
    """Load the capture metadata JSON written alongside captured.safetensors.

    Returns the raw metadata dict (``final_layer_idx``, ``original_aux_layers``).
    Falls back to ``{"final_layer_idx": None, "original_aux_layers": []}`` if
    the metadata file is absent (e.g., for legacy captures that predate this
    feature).
    """
    meta_path = capture_dir / "captured.safetensors.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"final_layer_idx": None, "original_aux_layers": []}


def _split_captured_layers(
    captured: dict[int, torch.Tensor],
    metadata: dict,
) -> tuple[list[int], int | None, dict[int, torch.Tensor], torch.Tensor | None]:
    """Split captured tensors into drafter-input and regression-target groups.

    Uses the metadata from the hook (``original_aux_layers`` and
    ``final_layer_idx``) to distinguish:

    - **Drafter-input layers**: the layers the drafter model consumes (e.g.
      [2, 12, 21]).  These correspond to ``original_aux_layers`` and are
      compared against the offline path's ``target_layers``.

    - **Regression-target layer**: the final decoder layer (e.g. 24) appended
      by the hook so the pre-norm regression target can be captured.  This is
      ``final_layer_idx`` and is compared against the offline path's
      ``hidden_states[:, -1]``.

    Returns:
        Tuple of (drafter_input_layer_ids, final_layer_idx,
        drafter_input_tensors, regression_target_tensor).
    """
    final_layer_idx = metadata.get("final_layer_idx")
    original_aux = metadata.get("original_aux_layers", [])

    # Drafter-input tensors are those whose keys match original_aux_layers.
    drafter_input_tensors: dict[int, torch.Tensor] = {
        k: v for k, v in captured.items() if k in original_aux
    }
    drafter_input_ids = sorted(drafter_input_tensors.keys())

    # Regression target is the final layer, if present.
    regression_target: torch.Tensor | None = None
    if final_layer_idx is not None and final_layer_idx in captured:
        regression_target = captured[final_layer_idx]

    return drafter_input_ids, final_layer_idx, drafter_input_tensors, regression_target


def _load_offline_hidden_states(
    hs_dir: Path, *, target_layers: list[int]
) -> dict[int, torch.Tensor]:
    """Load offline hs_*.safetensors shards into {layer_idx: tensor}.

    The offline path writes shape (seq_len, num_layers, hidden_size).
    We split the layer dimension and map positional index *i* to
    ``target_layers[i]`` so the keys match the serving capture's
    actual layer indices (which come from the engine's drafter config).
    """
    shards = sorted(hs_dir.glob("hs_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no hs_*.safetensors in {hs_dir}")

    layers: dict[int, list[torch.Tensor]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            hs = f.get_tensor("hidden_states")
        # hs shape: (seq_len, num_layers, hidden_size)
        for i in range(hs.shape[1]):
            key = target_layers[i] if i < len(target_layers) else i
            layers.setdefault(key, []).append(hs[:, i])

    merged: dict[int, torch.Tensor] = {}
    for idx in sorted(layers.keys()):
        parts = layers[idx]
        merged[idx] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return merged


def _align_token_count(
    captured: torch.Tensor,
    offline: torch.Tensor,
    prompt_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align captured and offline tensors to the prompt token range.

    Thin wrapper over
    :func:`speedlm.activation_capture.compare.align_prompt_rows`, which owns
    the (unit-tested) semantics.  Both sides are trimmed to
    ``prompt_token_count``; see that function for why trimming only the
    captured side to ``offline.shape[0]`` is wrong.
    """
    return align_prompt_rows(captured, offline, prompt_token_count)


def _split_offline_layers(
    offline: dict[int, torch.Tensor],
    offline_target_layers: list[int],
) -> tuple[list[int], int | None, dict[int, torch.Tensor], torch.Tensor | None]:
    """Split offline tensors into drafter-input and regression-target groups.

    The ``offline_target_layers`` list contains the drafter-input layers
    followed by the final layer index (if included).  The last entry is the
    regression-target layer; all preceding entries are drafter inputs.

    Returns:
        Tuple of (drafter_input_layer_ids, final_layer_idx,
        drafter_input_tensors, regression_target_tensor).
    """
    if not offline_target_layers:
        return [], None, {}, None

    # The last entry in offline_target_layers is the final (regression target)
    # layer; all preceding entries are drafter inputs.
    final_layer_idx = offline_target_layers[-1]
    drafter_input_ids = offline_target_layers[:-1]

    drafter_input_tensors: dict[int, torch.Tensor] = {
        k: v for k, v in offline.items() if k in drafter_input_ids
    }

    regression_target: torch.Tensor | None = None
    if final_layer_idx in offline:
        regression_target = offline[final_layer_idx]

    return sorted(drafter_input_ids), final_layer_idx, drafter_input_tensors, regression_target


def _wait_for_gpu_memory_release(
    gpu_memory_fraction: float,
    *,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> None:
    """Block until GPU device memory is released enough for the next engine.

    Uses ``nvidia-smi`` to poll the real driver — not a fixed sleep — so we
    only proceed once the previous engine has actually freed its allocations.

    Args:
        gpu_memory_fraction: The fraction of total device memory the next
            engine will request via ``--gpu-memory-utilization``.
    """
    probe = NvidiaSmiMemoryProbe()
    precondition = GPUMemoryPrecondition(
        probe=probe,
        required_fraction=gpu_memory_fraction,
        timeout_seconds=timeout,
        poll_interval_seconds=poll_interval,
    )
    deadline = time.monotonic() + timeout
    shortfall = precondition.shortfall()
    while shortfall is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory was not released within {timeout}s: {shortfall}"
            )
        time.sleep(poll_interval)
        shortfall = precondition.shortfall()
    logger.info("GPU memory released; proceeding with offline extraction")


# ---------------------------------------------------------------------------
# Independent HuggingFace fp32 reference (Leg 2)
# ---------------------------------------------------------------------------


def _flatten_token_ids(encoded: Any) -> list[int]:
    """Coerce a tokenizer's chat-template output into a flat ``list[int]``.

    Accepts the three shapes ``apply_chat_template(tokenize=True)`` has
    returned across transformers versions:

    * a ``BatchEncoding``/mapping with an ``input_ids`` entry (5.x default,
      since ``return_dict`` became ``True``);
    * a batched ``[[id, ...]]`` sequence (one row, because one conversation is
      passed);
    * a flat ``[id, ...]`` sequence (pre-5.x).

    Anything else raises rather than silently producing a wrong sequence: the
    whole point of this helper's caller is that the reference forward must run
    exactly the tokens the engine prefilled.
    """
    #: BatchEncoding is a Mapping, so this also covers a plain dict.  Tensors
    #: are not requested (no return_tensors), so the value is a nested list.
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):  #: torch.Tensor / numpy.ndarray
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise TypeError(
            f"apply_chat_template returned {type(encoded).__name__} with no "
            f"usable token ids; this test cannot verify that the reference "
            f"forward runs the same tokens the engine prefilled"
        )
    #: One conversation in means at most one row out; a batched return is
    #: unwrapped, but a batch of >1 means the call was not what we think it is.
    if isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise TypeError(
                f"apply_chat_template returned {len(encoded)} batched rows for "
                f"a single conversation; refusing to guess which one the "
                f"engine prefilled"
            )
        encoded = encoded[0]
    return [int(t) for t in encoded]


def _prompt_token_ids(
    verifier: str, prompt: str, *, expected_count: int
) -> list[int]:
    """Render *prompt* through the verifier's chat template and tokenize it.

    The reference forward is only independent evidence if it runs **the same
    tokens** the engine prefilled.  That is asserted, not assumed: the rendered
    length must equal the engine's own ``usage.prompt_tokens``.  A mismatch
    means the template this test renders and the one vLLM applied are not the
    same, and every downstream number would be comparing different sequences.

    #: transformers 5.x flipped ``apply_chat_template``'s ``return_dict``
    #: default to ``True`` (``tokenization_utils_base.py:3004``), so
    #: ``tokenize=True`` returns a ``BatchEncoding`` and iterating it yields the
    #: key strings ``'input_ids'``/``'attention_mask'`` rather than token ids.
    #: Pre-5.x returns a flat ``list[int]``.  Both shapes are unwrapped here
    #: rather than pinning a version, because the vLLM venv's transformers is
    #: not under this repo's control.
    """
    from transformers import AutoTokenizer

    model_dir = _resolve_model_dir(verifier)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = _flatten_token_ids(encoded)
    assert len(ids) == expected_count, (
        f"locally rendered prompt is {len(ids)} tokens but the engine reported "
        f"usage.prompt_tokens={expected_count}; the HF reference would be "
        f"running a different token sequence than the capture, which makes the "
        f"comparison meaningless.  The known cause of this on gpt-oss is a "
        f"control token inside the prompt body, which vLLM's Harmony encoder "
        f"keeps as text and this tokenizer parses out — see "
        f"{INJECTION_PROMPT_LABEL!r} in PROMPT_MATRIX.  "
        f"Rendered: {tokenizer.decode(ids)!r}"
    )
    return ids


def _special_token_ids(tokenizer: Any) -> set[int]:
    """Return every id *tokenizer* treats as a control/special token.

    ``all_special_ids`` covers only the roles the tokenizer config names (bos,
    eos, pad, ...).  ``added_tokens_decoder`` is what actually decides whether a
    string found in ordinary text is parsed out as one token, so both are
    unioned: gpt-oss's ``<|endoftext|>`` reaches the prompt through the second
    of the two, not the first.
    """
    ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    ids |= {
        int(token_id)
        for token_id, token in decoder.items()
        if getattr(token, "special", False)
    }
    return ids


def _assert_prompts_free_of_control_tokens(
    verifier: str, prompt_set: tuple[tuple[str, str], ...]
) -> None:
    """Fail before the engine starts if a prompt body carries a control token.

    ``template_divergent`` exists to check that two *renderers* agree, and that
    check is only meaningful on input both renderers are obliged to treat the
    same way.  A body containing a literal special token is not such an input:
    vLLM's Harmony encoder keeps it as text while HF's tokenizer parses it out
    as the token itself, so the two legs run different sequences by design, and
    the resulting length mismatch says nothing about the capture.  See
    ``control_token_injection`` in :data:`PROMPT_MATRIX` for the measurement.

    "Special" is decided against the verifier's own vocabulary rather than
    against a hard-coded list of markup, so a future model that promotes one of
    these strings to a real token is caught here — as a named, pre-flight
    failure — instead of resurfacing as an unexplained off-by-N mid-run.

    The ``control_token_injection`` case is exempt: carrying control tokens is
    the entire content of that case.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_resolve_model_dir(verifier)))
    special = _special_token_ids(tokenizer)
    for label, prompt in prompt_set:
        if label == INJECTION_PROMPT_LABEL:
            continue
        body = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        found = sorted({int(token) for token in body} & special)
        assert not found, (
            f"prompt {label!r} contains control token(s) "
            f"{tokenizer.convert_ids_to_tokens(found)} in its body, which "
            f"{verifier}'s vocabulary parses out of ordinary text.  The serving "
            f"and offline legs render such a body through different encoders "
            f"and will disagree on its length for reasons that have nothing to "
            f"do with the capture; move the case to "
            f"{INJECTION_PROMPT_LABEL!r} instead"
        )


def _free_device_bytes() -> int | None:
    """Return free VRAM on the current device, or ``None`` if there is none."""
    if torch is None or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


@dataclass(frozen=True)
class _HFReferenceSetup:
    """Everything the reference leg decides **once**, before any prompt runs.

    Split out of :func:`_run_hf_reference` so that an N-prompt matrix pays the
    device selection, the parameter-count read and — above all — the model load
    exactly once.  The load dominates this leg by orders of magnitude (a float32
    copy of an 8B verifier is ~30 GiB to materialize and move; the forward
    itself is seconds), so a per-prompt reload would make the matrix N times
    slower while changing not one number it reports.
    """

    #: Resolved snapshot directory for the verifier.
    model_dir: Path
    #: ``"cuda"`` or ``"cpu"`` for the reference forward.
    device: str
    #: Decoder depth, read off disk.  Held only to be logged: the *checked*
    #: version of this claim is the ``L+1`` hidden-state assertion inside
    #: ``hf_reference._forward_residual_stream``.
    num_hidden_layers: int
    #: Estimated parameter count, or ``None`` when the index was unreadable.
    num_parameters: int | None


def _prepare_hf_reference(verifier: str) -> _HFReferenceSetup:
    """Choose the device for the reference leg and free the GPU for it.

    Runs **after** both vLLM engines are gone, reusing
    :func:`_wait_for_gpu_memory_release` — the same nvidia-smi-polling
    machinery the offline phase already uses — so the fp32 copy gets the whole
    card rather than competing with an engine for it.  See
    ``hf_reference.select_reference_device`` for why GPU-after-teardown was
    chosen over CPU-fp32 or a smaller model, and why CPU remains the automatic
    fallback.

    Called once per run, not once per prompt: the wait is a property of the
    engines having been torn down, which happens once.
    """
    _wait_for_gpu_memory_release(gpu_memory_fraction=0.5)

    model_dir = _resolve_model_dir(verifier)
    config = _read_model_config(model_dir)
    num_hidden_layers = _verifier_num_hidden_layers(config)
    assert num_hidden_layers is not None, (
        f"cannot read num_hidden_layers from {model_dir}/config.json; the "
        f"reference forward cannot validate its own layer-index mapping"
    )
    #: Parameter count is not in config.json.  Estimating it from the config
    #: would be another unverified assumption, so the safetensors index is read
    #: instead: total_size is the on-disk byte count of a bf16 checkpoint, so
    #: params ~= total_size / 2.  Falls back to CPU if the index is absent.
    num_parameters = _checkpoint_parameter_count(model_dir)

    free_device_bytes = _free_device_bytes()
    forced = os.environ.get("SPEEDLM_E2E_HF_REFERENCE_DEVICE")
    if forced in ("cuda", "cpu"):
        device = forced
    elif num_parameters is None:
        device = "cpu"
    else:
        device = select_reference_device(
            num_parameters=num_parameters,
            free_device_bytes=free_device_bytes,
        )
    #: The fit check runs on the CHOSEN device, including when an operator
    #: forced it: forcing ``cuda`` for a checkpoint whose fp32 copy does not
    #: fit must produce a legible refusal naming the arithmetic, not an OOM
    #: kill that says nothing.  It also catches the case no device choice can
    #: rescue — an mxfp4 20B whose fp32 copy is 83.66 GB exceeds both an 80 GiB
    #: H100 and this node's ~63 GB of free host RAM, so ``cpu`` is not a
    #: fallback, it is a second way to die.
    if num_parameters is not None:
        assert_reference_fits(
            str(model_dir),
            num_parameters=num_parameters,
            free_device_bytes=free_device_bytes,
            free_host_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            device=device,
        )
    logger.info(
        "HF fp32 reference: %d params, device=%s, %d layers",
        num_parameters or -1, device, num_hidden_layers,
    )
    return _HFReferenceSetup(
        model_dir=model_dir,
        device=device,
        num_hidden_layers=num_hidden_layers,
        num_parameters=num_parameters,
    )


def _run_hf_reference(
    verifier: str,
    prompt: str,
    captured: dict[int, torch.Tensor],
    *,
    prompt_token_count: int,
    final_layer_idx: int | None,
    setup: _HFReferenceSetup,
    model: Any,
    prompt_label: str,
) -> HFReferenceResult:
    """Re-derive one prompt's captured activations with HF transformers, fp32.

    The per-prompt half of the reference leg: render the prompt to token ids,
    forward them on the **already-loaded** *model*, and compare.  Everything
    that does not vary with the prompt lives in :func:`_prepare_hf_reference`
    and :func:`loaded_reference_model`.

    *model* is passed straight through to ``reference_residual_stream``'s reuse
    path, which runs the identical forward body as the load-and-free path, so
    amortizing the load cannot quietly change what is measured.

    *prompt_label* is recorded on the result: a row of an N-prompt matrix that
    cannot be attributed to a prompt is a number nobody can re-run.
    """
    token_ids = _prompt_token_ids(
        verifier, prompt, expected_count=prompt_token_count
    )
    logger.info(
        "HF fp32 reference: prompt %r, %d tokens", prompt_label, len(token_ids)
    )

    stream, post_norm_final, dtype_name = reference_residual_stream(
        str(setup.model_dir), token_ids, device=setup.device, model=model
    )
    return compare_to_hf_reference(
        captured,
        stream,
        post_norm_final,
        prompt_token_count=prompt_token_count,
        final_layer_idx=final_layer_idx,
        device=setup.device,
        dtype=dtype_name,
        prompt_label=prompt_label,
    )


def _checkpoint_parameter_count(model_dir: Path) -> int | None:
    """Delegate to :func:`hf_reference.checkpoint_parameter_count`.

    The estimate used to live here: ``metadata.total_size`` from the
    safetensors index is the checkpoint's byte count, these verifiers ship in
    bf16, so two bytes per parameter.  Returning ``None`` on any doubt keeps
    the device choice conservative (CPU) rather than guessing high and OOMing.

    **The quantized case is why it moved into the module.**  That estimate is
    wrong for an mxfp4 checkpoint, and wrong in the dangerous direction: each
    stored ``*_blocks`` byte holds two e2m1 nibbles, so gpt-oss-20b's
    ``13,761,316,904`` on-disk bytes dequantize to ``20,914,757,184``
    parameters while ``total_size // 2`` answers ``6,880,658,452`` — a 3.04x
    **undercount** that would pick ``cuda`` for an 83.66 GB fp32 copy and OOM
    an 80 GiB H100.  The corrected count needs the shard shapes, and it is
    needed by ``hf_reference`` itself (for
    :func:`assert_reference_fits`) as well as here, so a single
    implementation lives there and this stays a thin adapter.
    """
    return checkpoint_parameter_count(str(model_dir))


# ---------------------------------------------------------------------------
# Prefix-cache measurement helpers
# ---------------------------------------------------------------------------


def _prefix_cache_counters(url: str) -> tuple[float, float]:
    """Read ``(hits, queries)`` from the engine's Prometheus ``/metrics``.

    The counters are ``vllm:prefix_cache_hits`` and
    ``vllm:prefix_cache_queries`` (registered at vLLM
    ``v1/metrics/loggers.py:547-564``, exported with Prometheus' ``_total``
    suffix, in units of blocks).  This is the engine's own accounting of
    whether a request was served from cache — the only way to *measure*
    ``cache_hit`` rather than infer it from "we sent the same prompt twice".

    Returns:
        ``(hits, queries)``.  Both are ``0.0`` when the counters are absent,
        which the caller must treat as "no hit observed", never as success.
    """
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        resp = client.get(f"{url}/metrics")
        resp.raise_for_status()
        body = resp.text

    def _sum(metric: str) -> float:
        total = 0.0
        for line in body.splitlines():
            if line.startswith("#") or not line.startswith(metric):
                continue
            head, _, value = line.rpartition(" ")
            #: Guard against a prefix collision (e.g. ``..._hits_total`` vs a
            #: hypothetical ``..._hits_total_bucket``): the metric name must be
            #: followed by a label brace or whitespace, nothing else.
            name = head.split("{", 1)[0].strip()
            if name != metric:
                continue
            try:
                total += float(value)
            except ValueError:
                continue
        return total

    return _sum("vllm:prefix_cache_hits_total"), _sum(
        "vllm:prefix_cache_queries_total"
    )


def _rows_per_layer(captured: dict[int, torch.Tensor]) -> int:
    """Return the row count shared by every captured layer.

    Summing rows across layers (the old ``captured_row_count``) yields
    ``rows x layers``, which is not comparable to a prompt token count.  Every
    layer is collected at the same positions in the same forward, so they must
    agree; a disagreement is itself a capture bug and is raised rather than
    averaged away.
    """
    assert captured, "no captured layers"
    counts = {idx: int(t.shape[0]) for idx, t in captured.items()}
    distinct = set(counts.values())
    assert len(distinct) == 1, (
        f"captured layers disagree on row count: {counts}; every aux layer is "
        f"collected at the same token positions in the same forward, so this "
        f"is a capture bug, not a measurement to be summarized"
    )
    return distinct.pop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CapturedPrompt:
    """One prompt's serving-time capture, as read back off disk."""

    #: matrix label; becomes :attr:`PromptCase.label`
    label: str
    #: the prompt text that was sent
    text: str
    #: per-prompt capture directory, so one prompt's flush cannot be read as
    #: another's
    capture_dir: Path
    #: the engine's own ``usage.prompt_tokens`` for this prompt
    prompt_token_count: int
    #: ``{layer id: tensor}`` from ``captured.safetensors``
    tensors: dict[int, torch.Tensor]
    #: the hook's metadata sidecar for this flush
    metadata: dict[str, Any]
    #: the appended final decoder layer, or ``None``
    final_layer_idx: int | None


@dataclass(frozen=True)
class _AlignedPrompt:
    """One prompt after Leg 1's split/align, ready for the reference leg."""

    capture: _CapturedPrompt
    aligned_captured: dict[int, torch.Tensor]
    aligned_offline: dict[int, torch.Tensor]
    captured_final: torch.Tensor | None
    offline_final: torch.Tensor | None
    prefix_cache: PrefixCacheResult


def test_stage0_activation_capture() -> None:
    """Full Stage 0 experiment: a prompt MATRIX with capture, compared offline.

    This test:
    1. Starts a vLLM engine with the EAGLE-3 speculator and capture extension.
    2. Proves which model-runner generation that engine actually loaded.
    3. Sends every prompt in the selected set through that ONE engine,
       flushing each prompt's activations to its own directory.
    4. Tears down the capture engine.
    5. Waits for GPU memory to be released.
    6. Runs the offline extraction on each prompt, on its own port.
    7. Compares the two tensor stacks elementwise per layer, per prompt (Leg 1
       — transport; bit-identical is the expected result, see module docstring).
    8. Re-derives the same activations with HuggingFace transformers in fp32
       and checks tolerance *and* neighbour discrimination, per prompt (Leg 2 —
       identity), loading the fp32 model ONCE for the whole matrix.
    9. Writes a JSON matrix result with a per-case and aggregate verdict.

    Only step 8 can distinguish a correct capture from a well-transported
    wrong quantity.  Step 7 returning 0.0 is not evidence of correctness.

    **Why a matrix and not a prompt.**  The single 18-token prompt this test
    used to run could not see a length-dependent bug at all: an off-by-one in
    the prompt/template split, a row loss at a KV-block boundary, and a
    divergence between this test's chat-template rendering and vLLM's are all
    invisible at one length on one template.  See :data:`PROMPT_MATRIX` for what
    each prompt buys.  The engine is started once and reused across prompts —
    the engine lifecycle, not the comparison, is what costs time here.

    **Every assertion is per case.**  Widening coverage is worthless if the
    aggregate can hide a member, so nothing is checked "for the run": the
    drafter-layer match, the captured-key set, the row-count check and the
    reference-ran check all run inside the per-prompt loop, and
    ``derive_matrix_verdict`` names every failing case rather than reducing
    them to a worst case.
    """
    verifier, drafter, artifact_root = _require_environment()
    prompt_set = _prompt_set()
    #: Pre-flight, before an engine is paid for: a prompt whose body carries a
    #: real control token cannot be compared across the two legs at all.
    _assert_prompts_free_of_control_tokens(verifier, prompt_set)
    #: Derived per model from the two on-disk configs — nothing here assumes a
    #: depth or an architecture.  Qwen3-8B resolves to 36 layers with aux
    #: ``(2, 18, 33)``; gpt-oss-20b resolves to 24 layers with aux
    #: ``(2, 12, 21)``.  Both come out of the same resolution path, which is
    #: why switching ``SPEEDLM_E2E_VERIFIER_MODEL`` needs no change here.
    target_layers = _target_layer_ids(verifier, drafter)
    artifact_dir = _create_artifact_dir(artifact_root)
    port = int(os.environ.get("SPEEDLM_E2E_PORT", _free_port()))
    timeout = _ready_timeout()

    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": 5,
        "model": drafter,
    }

    capture_root = artifact_dir / "captured"
    capture_root.mkdir(exist_ok=True)

    vllm_log = artifact_dir / "vllm.log"
    log_handle = vllm_log.open("wb")
    vllm_proc = subprocess.Popen(
        [
            str(VLLM_PYTHON),
            "-m", "vllm.entrypoints.cli.main",
            "serve",
            verifier,
            "--speculative_config",
            json.dumps(speculative_config),
            "--worker-extension-cls",
            "speedlm.activation_capture.hook.ActivationCaptureExtension",
            "--port",
            str(port),
            #: Every prompt in :data:`PROMPT_MATRIX` must render inside this
            #: bound, template and sampled tokens included.
            "--max-model-len",
            "512",
            "--max-num-seqs",
            "8",
            #: Deliberately use vLLM's default compiled/CUDA-graphed execution.
            #: The extension declares all four aux layers before compilation;
            #: arming after readiness must capture from that baked graph.  The
            #: transport and independent fp32 fidelity assertions below remain
            #: unchanged and therefore validate the graph-produced tensors.
            "--gpu-memory-utilization",
            "0.5",
            "--no-enable-prefix-caching",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_vllm_env(),
    )

    # Phase 1: ONE capture engine, every prompt — serve, capture, tear down.
    #
    # The engine is started once for the whole matrix.  Loading the verifier and
    # the speculator is what makes this test slow; a prompt is a request.
    url = f"http://127.0.0.1:{port}"
    captures: list[_CapturedPrompt] = []
    try:
        _wait_for_ready(url, vllm_proc, timeout, log_path=vllm_log)

        served_model_id = _get_served_model_id(url)

        # Asserted once, immediately: the runner is a property of the engine,
        # not of a prompt, and every prompt below inherits this answer.
        observed_runner = _assert_runner(vllm_proc, port)

        for label, prompt in prompt_set:
            #: Per-prompt directory: ``flush_capture`` drains the buffer, but
            #: writing every prompt to one directory would still have each
            #: flush overwrite the last one's ``captured.safetensors``.
            capture_dir = capture_root / label
            capture_dir.mkdir(exist_ok=True)

            #: No deactivate RPC between prompts: ``activate_capture`` already
            #: resets an active capture itself — ``hook.py`` logs "capture
            #: already active; resetting" and clears the pending buffer before
            #: re-arming — so the second and later prompts start from the same
            #: clean state as the first.
            _collective_rpc(
                vllm_proc, port, "activate_capture", str(capture_dir)
            )
            _, prompt_token_count = _send_prompt(
                url, prompt, served_model_id=served_model_id
            )
            logger.info(
                "Prompt %r: engine reported %d prompt tokens",
                label, prompt_token_count,
            )
            # Small pause to let the hook buffer finish
            time.sleep(0.5)
            _collective_rpc(vllm_proc, port, "flush_capture")

            # Load captured tensors
            captured_tensors = _load_captured_safetensors(capture_dir)
            logger.info(
                "Prompt %r captured layers: %s",
                label, sorted(captured_tensors.keys()),
            )

            # Load metadata written by the hook during flush_capture so we can
            # correctly distinguish drafter-input layers from the appended final
            # regression-target layer.
            meta = _load_capture_metadata(capture_dir)
            final_layer_idx = meta["final_layer_idx"]
            original_aux = meta["original_aux_layers"]

            # Verify the drafter-input layers match the test's expectations.
            assert sorted(original_aux) == target_layers, (
                f"[{label}] Captured drafter-input layers "
                f"{sorted(original_aux)} do not match "
                f"target_layers {target_layers} — offline extraction will use "
                f"wrong keys.  target_layers were derived from the drafter/"
                f"verifier configs on disk, so a mismatch means vLLM's in-engine "
                f"derivation disagrees with speedlm.profiles.resolve_target_"
                f"layer_ids.  Set SPEEDLM_E2E_TARGET_LAYER_IDS to override."
            )

            # The captured keys must be exactly original_aux + final_layer_idx.
            _extra = [final_layer_idx] if final_layer_idx is not None else []
            expected_captured = sorted(original_aux + _extra)
            actual_captured = sorted(captured_tensors.keys())
            assert actual_captured == expected_captured, (
                f"[{label}] Captured layer keys {actual_captured} do not match "
                f"expected {expected_captured} (original_aux={original_aux}, "
                f"final_layer_idx={final_layer_idx})"
            )

            captures.append(
                _CapturedPrompt(
                    label=label,
                    text=prompt,
                    capture_dir=capture_dir,
                    prompt_token_count=prompt_token_count,
                    tensors=captured_tensors,
                    metadata=meta,
                    final_layer_idx=final_layer_idx,
                )
            )

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()

    # Phase 2: wait for GPU memory release before launching offline engines.
    # The capture engine held GPU memory; the offline engine must not start
    # until the device memory is actually returned, otherwise both engines
    # compete for the same GPU and the second crashes.
    _wait_for_gpu_memory_release(gpu_memory_fraction=0.5)

    # Phase 3 + 4: per prompt, extract offline and align against the capture.
    #
    # These run in one loop because phase 4 consumes exactly what phase 3
    # produces for that prompt, and keeping them together means a prompt's
    # offline stack is never in scope while another prompt's alignment runs.
    aligned_prompts: list[_AlignedPrompt] = []
    for index, capture in enumerate(captures):
        label = capture.label
        prompt_token_count = capture.prompt_token_count
        final_layer_idx = capture.final_layer_idx

        # Phase 3: offline extraction — now the GPU is free.
        offline_target_layers = list(target_layers)
        if final_layer_idx is not None:
            offline_target_layers.append(final_layer_idx)
        offline_dir = artifact_dir / "offline" / label
        #: Each prompt's offline engine gets its own port.  ``port + 1000`` was
        #: already the "different port" for the single-prompt case; the ``+
        #: index`` term is what keeps the matrix's engines from colliding with
        #: each other.  They are strictly sequential (the previous one is gone
        #: before the next starts), so this only has to be collision-free, not
        #: simultaneously bindable.
        offline_port = port + 1000 + index
        hs_dir = _run_offline_extract(
            verifier,
            capture.text,
            offline_target_layers,
            offline_dir,
            port=offline_port,
        )
        offline_tensors = _load_offline_hidden_states(
            hs_dir, target_layers=offline_target_layers
        )
        logger.info(
            "Prompt %r offline layers: %s", label, sorted(offline_tensors.keys())
        )

        # Phase 4: Align and compare — using explicit split to avoid comparing
        # drafter-input layers against the regression target or vice versa.

        # Split captured tensors into drafter-inputs and regression-target.
        (
            captured_drafter_ids,
            captured_final_idx,
            captured_drafter,
            captured_regression,
        ) = _split_captured_layers(capture.tensors, capture.metadata)

        # Split offline tensors into drafter-inputs and regression-target.
        (
            offline_drafter_ids,
            offline_final_idx,
            offline_drafter,
            offline_regression,
        ) = _split_offline_layers(offline_tensors, offline_target_layers)

        # The drafter-input layer ids must match between captured and offline.
        assert captured_drafter_ids == offline_drafter_ids, (
            f"[{label}] Drafter-input layer mismatch: captured "
            f"{captured_drafter_ids} vs offline {offline_drafter_ids}"
        )

        # The final layer indices must also match.
        assert captured_final_idx == offline_final_idx, (
            f"[{label}] Final layer index mismatch: captured "
            f"{captured_final_idx} vs offline {offline_final_idx}"
        )

        # The comparable row range is the PROMPT, and only the prompt.  The
        # engine's own usage.prompt_tokens is the authority; the offline row
        # count is not — it also covers the assistant turn that prepare_data.py
        # renders into the conversation, whose tokens the serving engine never
        # saw.  See ``align_prompt_rows``.
        offline_rows = next(iter(offline_drafter.values())).shape[0]
        logger.info(
            "Prompt %r: aligning on %d prompt rows (offline stack has %d rows; "
            "the extra %d are the template's assistant turn and are NOT "
            "comparable)",
            label, prompt_token_count, offline_rows,
            offline_rows - prompt_token_count,
        )

        # Align drafter-input layers (same key set on both sides)
        aligned_captured: dict[int, torch.Tensor] = {}
        aligned_offline: dict[int, torch.Tensor] = {}
        for idx in offline_drafter_ids:
            cap = captured_drafter[idx]
            off = offline_drafter[idx]
            try:
                c_aligned, o_aligned = _align_token_count(
                    cap, off, prompt_token_count
                )
            except ValueError as exc:
                raise AssertionError(
                    f"[{label}] cannot align layer {idx}: {exc}"
                ) from exc
            aligned_captured[idx] = c_aligned
            aligned_offline[idx] = o_aligned

        # Align the regression-target (final layer) separately
        captured_final: torch.Tensor | None = None
        offline_final: torch.Tensor | None = None
        if captured_regression is not None and offline_regression is not None:
            try:
                captured_final, offline_final = _align_token_count(
                    captured_regression, offline_regression, prompt_token_count,
                )
            except ValueError as exc:
                raise AssertionError(
                    f"[{label}] cannot align final layer: {exc}"
                ) from exc

        # This engine ran with --no-enable-prefix-caching, so cache_hit is False
        # by construction rather than by measurement -- and it is verified as
        # such: every prompt row must have produced an activation row.  The
        # measured cache-hit case lives in test_prefix_cache_coverage.
        captured_rows = _rows_per_layer(aligned_captured)
        assert captured_rows == prompt_token_count, (
            f"[{label}] prefix caching is disabled on this engine, so all "
            f"{prompt_token_count} prompt positions must have been forwarded, "
            f"but only {captured_rows} rows were captured per layer"
        )
        prefix_cache = PrefixCacheResult(
            prompt_token_count=prompt_token_count,
            captured_rows_per_layer=captured_rows,
            captured_layer_count=len(aligned_captured),
            cache_hit=False,
            #: ``captured_rows`` is counted on the *aligned* stack, which has
            #: already been trimmed to the prompt range, so here — and only
            #: here — subtracting it from ``prompt_token_count`` really does
            #: count prompt rows.  The assertion directly above pins it to zero.
            prompt_rows_missing=prompt_token_count - captured_rows,
        )

        aligned_prompts.append(
            _AlignedPrompt(
                capture=capture,
                aligned_captured=aligned_captured,
                aligned_offline=aligned_offline,
                captured_final=captured_final,
                offline_final=offline_final,
                prefix_cache=prefix_cache,
            )
        )

    # Phase 5: independent HuggingFace fp32 re-derivation (Leg 2).
    #
    # Leg 1 above is a transport check between two views of ONE tensor (see
    # the module docstring).  This is the only part of the test that runs a
    # second, independent forward, and therefore the only part that can tell
    # a correct capture from a well-transported wrong quantity.
    #
    # The fp32 model is loaded ONCE for the whole matrix.  That is the entire
    # reason ``loaded_reference_model`` exists: the load is ~30 GiB of weights
    # to materialize and move for an 8B verifier, while each prompt's forward is
    # seconds, so a per-prompt reload would multiply this leg's cost by the
    # number of prompts and change none of its numbers.
    hf_references: dict[str, HFReferenceResult] = {}
    reference_enabled = os.environ.get("SPEEDLM_E2E_HF_REFERENCE", "1") != "0"
    if reference_enabled:
        setup = _prepare_hf_reference(verifier)
        with loaded_reference_model(
            str(setup.model_dir), device=setup.device
        ) as reference_model:
            for aligned in aligned_prompts:
                label = aligned.capture.label
                hf_result = _run_hf_reference(
                    verifier,
                    aligned.capture.text,
                    aligned.capture.tensors,
                    prompt_token_count=aligned.capture.prompt_token_count,
                    final_layer_idx=aligned.capture.final_layer_idx,
                    setup=setup,
                    model=reference_model,
                    prompt_label=label,
                )
                hf_references[label] = hf_result
                logger.info(
                    "Prompt %r HF fp32 reference verdict=%s device=%s layers=%s",
                    label,
                    hf_result.verdict,
                    hf_result.device,
                    [
                        (layer.aux_layer_idx, layer.mean_rel_error, layer.tolerance)
                        for layer in hf_result.layers
                    ],
                )

    # Phase 6: Build one case per prompt, then the matrix verdict.
    cases: list[PromptCase] = []
    for aligned in aligned_prompts:
        label = aligned.capture.label
        result = build_result(
            aligned.aligned_captured,
            aligned.aligned_offline,
            captured_final_pre_norm=aligned.captured_final,
            offline_final_pre_norm=aligned.offline_final,
            prefix_cache=aligned.prefix_cache,
            hf_reference=hf_references.get(label),
        )
        cases.append(
            PromptCase(
                label=label,
                prompt_token_count=aligned.capture.prompt_token_count,
                result=result,
            )
        )
        logger.info(
            "Prompt %r (%d tokens) verdict=%s trend=%s",
            label, aligned.capture.prompt_token_count,
            result.verdict, result.rel_error_trend,
        )

    #: ``observed_runner``, not the requested one: under ``auto`` the artifact
    #: must still record which generation actually ran, or a later reader cannot
    #: tell a V2 result from a V1 one.
    matrix: MatrixResult = build_matrix_result(
        model=served_model_id,
        runner=observed_runner,
        cases=cases,
    )
    result_path = artifact_dir / "result.json"
    matrix.write_json(result_path)
    logger.info(
        "Matrix result written to %s — runner=%s verdict: %s",
        result_path, matrix.runner, matrix.verdict,
    )

    # By default, a FAIL verdict fails the test so that regressions are caught.
    # Set SPEEDLM_E2E_STRICT_VERDICT=0 to skip this assertion for exploratory runs.
    strict = os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"
    if strict:
        # A skipped reference leg must never read as a pass.  ``build_result``
        # deliberately does not fail on ``hf_reference is None`` (that would
        # flip the verdict for every pre-existing caller), so the harness that
        # asked for the leg is the one that insists it ran.  Checked PER CASE:
        # a reference that ran for one prompt says nothing about the others.
        for case in cases:
            assert not reference_enabled or case.result.hf_reference is not None, (
                f"[{case.label}] SPEEDLM_E2E_HF_REFERENCE is enabled but no "
                f"reference result was produced; the run proves only that one "
                f"tensor round-trips two ways, not that it is the right tensor"
            )
        #: Every selected prompt must have produced a case.  A prompt that was
        #: skipped — by an early ``continue``, a swallowed exception, or a
        #: mis-wired loop — would otherwise shrink coverage silently while the
        #: remaining cases still reported PASS.
        assert len(cases) == len(prompt_set), (
            f"{len(prompt_set)} prompts were selected "
            f"({[label for label, _text in prompt_set]}) but only "
            f"{len(cases)} produced a case "
            f"({[case.label for case in cases]}); a prompt that silently ran "
            f"no comparison is a coverage regression, not a pass"
        )
        #: Surfaced per case rather than only as the aggregate label: the
        #: aggregate names *which* cases failed, this names what each of them
        #: reported, so the failure text alone says whether the divergence is a
        #: transport problem (Leg 1 verdict), an identity problem (the reference
        #: verdict) or a drift (the trend) — without opening the artifact.
        case_summary = [
            (
                case.label,
                case.result.verdict,
                case.result.rel_error_trend,
                (
                    case.result.hf_reference.verdict
                    if case.result.hf_reference is not None
                    else "not run"
                ),
            )
            for case in cases
        ]
        assert matrix.verdict == "PASS", (
            f"Activation capture matrix failed: verdict={matrix.verdict}, "
            f"runner={matrix.runner}, "
            f"cases (label, verdict, trend, hf_reference)={case_summary}. "
            f"Full result at {result_path}"
        )


# ---------------------------------------------------------------------------
# Prefix-cache coverage measurement
# ---------------------------------------------------------------------------


def test_prefix_cache_coverage() -> None:
    """Prove that a prefix-cache hit leaves activation rows missing.

    This is the empirical check for hazard 6.1 in
    ``docs/serving-time-activation-capture.md``: on a cache hit the matched
    tokens are never forwarded, so no aux hidden state is produced for them and
    a naive capture silently under-covers the prompt.

    **What changed and why.**  The first version of this test contained no
    ``assert`` at all.  It wrote ``cache_hit: true`` and ``rows_missing: 0``
    into its result file as *hardcoded literals* and reported them as findings —
    a test that recorded an assumption and could not fail, including in the case
    where prefix caching silently did not engage.  Every field is now measured:

    * ``cache_hit`` comes from the engine's own
      ``vllm:prefix_cache_hits_total`` counter (vLLM
      ``v1/metrics/loggers.py:558-564``), sampled before and after the second
      request.  Sending the same prompt twice is what *should* cause a hit; it
      is not evidence that one occurred.
    * ``captured_rows_per_layer`` is the per-layer row count, not the old
      ``rows x layers`` sum.
    * ``prompt_rows_missing`` is the cold row count minus the warm one.

    **Why the missing-row count is not** ``prompt_token_count -
    captured_rows_per_layer``.  It was, until job 369256 wrote
    ``rows_missing: 96`` for a 165-token prompt whose warm capture held 69 rows
    per layer.  Those 69 rows are 21 prompt rows plus 48 decode rows, so the
    subtraction credited 48 decode rows as prompt rows and understated the loss
    by exactly that much; the true figure is 144, the cold count (213) minus
    the warm one (69).  The decode rows cancel in the cold-vs-warm difference
    because both requests send the same prompt and generate the same
    continuation, and that premise is not assumed — it is asserted below, by
    requiring the difference to equal the engine's own hit-token count.  The
    old field name is gone rather than redefined so nothing downstream can keep
    quoting the wrong number.

    **On the apparent conflict with the main test.**  ``result.json`` from
    ``test_stage0_activation_capture`` reports ``cache_hit: false`` while this
    test reports true.  Both are correct and they are not in conflict: the main
    test launches its engine with ``--no-enable-prefix-caching``, so a hit is
    impossible there and the absence of one is asserted; this test leaves prefix
    caching at its default (on) precisely so a hit can occur.  The old
    ``cache_hit: true`` here happened to name the right answer for the wrong
    reason — it was never read off the engine.

    **Why the prompt is not** :data:`DEFAULT_PROMPT`.  The first measured run of
    this test (job 369236) reported ``queries 18 -> 36`` with ``hits 0 -> 0``
    and identical cold/warm row counts.  That is not a failure of the hazard and
    not a misconfiguration — it is arithmetic.  ``DEFAULT_PROMPT`` renders to 18
    tokens; vLLM hashes only full 16-token blocks
    (``V/v1/core/kv_cache_utils.py:709-712``), caps the lookup at
    ``num_tokens - 1`` so the final token is always recomputed
    (``V/v1/core/kv_cache_manager.py:231``), and drops the last matched block
    outright under eagle-family speculation
    (``V/v1/core/single_type_kv_cache_manager.py:602-605``, reached because
    ``eagle3`` is in ``use_eagle()``, ``V/config/speculative.py:1238-1242``).
    An 18-token prompt yields exactly one hashable block, and that block is the
    one eagle drops, so **zero hits is the correct result and no prompt of that
    length can ever produce another**.  :data:`PREFIX_CACHE_PROMPT` is sized to
    span several blocks so the hazard is representable at all, and the required
    length is asserted below against the engine's own token count rather than
    assumed.
    """
    verifier, drafter, artifact_root = _require_environment()
    #: Deliberately NOT SPEEDLM_E2E_PROMPT: an overridden short prompt would
    #: silently reduce this back to the unmeasurable case described above.
    prompt = PREFIX_CACHE_PROMPT
    artifact_dir = _create_artifact_dir(artifact_root)
    port = int(os.environ.get("SPEEDLM_E2E_PORT", _free_port()))
    timeout = _ready_timeout()

    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": 5,
        "model": drafter,
    }

    capture_dir = artifact_dir / "captured"
    capture_dir.mkdir(exist_ok=True)

    vllm_log = artifact_dir / "vllm.log"
    log_handle = vllm_log.open("wb")
    vllm_proc = subprocess.Popen(
        [
            str(VLLM_PYTHON),
            "-m", "vllm.entrypoints.cli.main",
            "serve",
            verifier,
            "--speculative_config",
            json.dumps(speculative_config),
            "--worker-extension-cls",
            "speedlm.activation_capture.hook.ActivationCaptureExtension",
            "--port",
            str(port),
            "--max-model-len",
            "512",
            "--max-num-seqs",
            "8",
            #: Keep graph mode here too: both cold and prefix-cache-hit captures
            #: must replay the graph baked with the full aux-layer set.
            "--gpu-memory-utilization",
            "0.5",
            # Prefix caching ENABLED (default)
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_vllm_env(),
    )

    try:
        url = f"http://127.0.0.1:{port}"
        _wait_for_ready(url, vllm_proc, timeout, log_path=vllm_log)

        served_model_id = _get_served_model_id(url)

        # -- Request 1: cold.  Flushed to its own directory so its rows are
        # not conflated with request 2's; flush_capture drains the buffer, so
        # the second flush contains only the second request's rows.
        first_dir = capture_dir / "first"
        first_dir.mkdir(exist_ok=True)
        _collective_rpc(vllm_proc, port, "activate_capture", str(first_dir))
        _, first_prompt_tokens = _send_prompt(
            url, prompt, served_model_id=served_model_id
        )
        time.sleep(0.5)
        _collective_rpc(vllm_proc, port, "flush_capture")
        first_captured = _load_captured_safetensors(first_dir)
        first_rows = _rows_per_layer(first_captured)

        # -- Request 2: identical prompt, expected to hit the prefix cache.
        second_dir = capture_dir / "second"
        second_dir.mkdir(exist_ok=True)
        _collective_rpc(vllm_proc, port, "activate_capture", str(second_dir))
        hits_before, queries_before = _prefix_cache_counters(url)
        _, prompt_token_count = _send_prompt(
            url, prompt, served_model_id=served_model_id
        )
        time.sleep(0.5)
        hits_after, queries_after = _prefix_cache_counters(url)
        _collective_rpc(vllm_proc, port, "flush_capture")
        second_captured = _load_captured_safetensors(second_dir)
        second_rows = _rows_per_layer(second_captured)

        hit_tokens = hits_after - hits_before
        cache_hit = hit_tokens > 0
        min_prompt_tokens = _min_prefix_cache_prompt_tokens()
        expected_hit_tokens = _expected_prefix_cache_hit_tokens(
            prompt_token_count
        )
        #: Prompt rows the warm request never forwarded, measured against the
        #: cold request rather than against ``prompt_token_count``.  Both
        #: captures hold prompt rows *plus* decode rows, so the decode rows
        #: cancel in the difference and do not cancel in
        #: ``prompt_token_count - second_rows``.  Job 369256 recorded the
        #: latter as ``rows_missing: 96`` (165 - 69) when the true figure was
        #: 144 (213 - 69): 48 decode rows counted as present prompt rows.
        prompt_rows_missing = first_rows - second_rows
        result = PrefixCacheResult(
            prompt_token_count=prompt_token_count,
            captured_rows_per_layer=second_rows,
            captured_layer_count=len(second_captured),
            cache_hit=cache_hit,
            prompt_rows_missing=prompt_rows_missing,
        )
        result_path = artifact_dir / "prefix_cache_result.json"
        # Write before asserting: a failing check must not discard the
        # measurement that produced it.
        result_path.write_text(
            json.dumps({
                "prompt_token_count": result.prompt_token_count,
                "captured_rows_per_layer": result.captured_rows_per_layer,
                "captured_layer_count": result.captured_layer_count,
                "cache_hit": result.cache_hit,
                "prompt_rows_missing": result.prompt_rows_missing,
                "first_request_prompt_tokens": first_prompt_tokens,
                "first_request_rows_per_layer": first_rows,
                "prefix_cache_hits_before": hits_before,
                "prefix_cache_hits_after": hits_after,
                "prefix_cache_queries_before": queries_before,
                "prefix_cache_queries_after": queries_after,
                "prefix_cache_hit_tokens": hit_tokens,
                "prefix_cache_expected_hit_tokens": expected_hit_tokens,
                "prefix_cache_block_size": PREFIX_CACHE_BLOCK_SIZE,
                "min_prompt_tokens_for_a_hit": min_prompt_tokens,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Prefix cache result: %s", result_path)

        #: Checked before the hit assertions, because a prompt below this bound
        #: makes "no hit" the correct answer and the rest of this test
        #: meaningless.  Job 369236 failed exactly here, with 18 tokens.
        assert prompt_token_count >= min_prompt_tokens, (
            f"the prompt rendered to {prompt_token_count} tokens but at least "
            f"{min_prompt_tokens} are needed before "
            f"{PREFIX_CACHE_MIN_HIT_BLOCKS} blocks of "
            f"{PREFIX_CACHE_BLOCK_SIZE} tokens can be reported as a prefix-cache "
            f"hit (the trailing partial block is never hashed and eagle drops "
            f"the last matched block).  Below that bound zero hits is correct "
            f"engine behaviour and this test cannot observe hazard 6.1 at all.  "
            f"Result at {result_path}"
        )

        assert queries_after > queries_before, (
            f"the engine recorded no prefix-cache queries across the second "
            f"request ({queries_before} -> {queries_after}); prefix caching is "
            f"not engaged at all, so this test measured nothing.  Result at "
            f"{result_path}"
        )
        assert cache_hit, (
            f"the second identical request did not hit the prefix cache "
            f"(vllm:prefix_cache_hits_total {hits_before} -> {hits_after}); "
            f"hazard 6.1 cannot be demonstrated without a hit.  Result at "
            f"{result_path}"
        )
        #: The hit size is fully determined by vLLM's block arithmetic, so it is
        #: asserted exactly rather than just "greater than zero".  A mismatch
        #: means one of the premises this test is built on has moved — the
        #: block size is no longer 16, eagle no longer drops the last matched
        #: block, or the last-token recompute cap is gone — and the derived
        #: minimum prompt length above would be wrong too.
        assert hit_tokens == expected_hit_tokens, (
            f"the engine reported {hit_tokens} prefix-cache hit tokens for a "
            f"{prompt_token_count}-token repeat, but vLLM's block arithmetic "
            f"predicts {expected_hit_tokens} "
            f"(((N-1)//{PREFIX_CACHE_BLOCK_SIZE})-1 blocks, dropping the "
            f"unhashed trailing block and eagle's last matched block).  One of "
            f"those premises no longer holds on this build.  Result at "
            f"{result_path}"
        )
        assert first_prompt_tokens == prompt_token_count, (
            f"the two requests were not the same length "
            f"({first_prompt_tokens} vs {prompt_token_count} prompt tokens); "
            f"their row counts are not comparable"
        )
        assert second_rows < first_rows, (
            f"a prefix-cache hit did not reduce the captured row count "
            f"({first_rows} rows cold vs {second_rows} rows warm).  Either the "
            f"hazard in doc section 6.1 does not hold on this build, or the "
            f"capture is picking up rows the forward did not produce.  Result "
            f"at {result_path}"
        )
        assert result.prompt_rows_missing > 0, (
            f"the warm request captured {second_rows} rows per layer against "
            f"{first_rows} cold, i.e. no prompt row is missing -- which "
            f"contradicts the measured cache hit.  Result at {result_path}"
        )
        #: The sharp form of the assertion above, and the reason
        #: ``prompt_rows_missing`` is measured cold-vs-warm at all: every prompt
        #: row the engine skipped is a row it reported as a prefix-cache hit,
        #: so the two independently-sourced numbers -- one counted off the
        #: safetensors, one read off ``/metrics`` -- must agree exactly.  If
        #: they diverge, either the capture is emitting rows the forward did
        #: not produce, or the two requests generated different numbers of
        #: decode rows and the cold-vs-warm difference is no longer a pure
        #: prompt-row count.
        assert result.prompt_rows_missing == hit_tokens, (
            f"{result.prompt_rows_missing} prompt rows went missing "
            f"({first_rows} cold - {second_rows} warm) but the engine reported "
            f"{hit_tokens} prefix-cache hit tokens; every skipped prompt row "
            f"must be a cache hit and vice versa.  Either the capture is "
            f"emitting rows the forward did not produce, or the two requests "
            f"did not generate the same number of decode rows.  Result at "
            f"{result_path}"
        )

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()
