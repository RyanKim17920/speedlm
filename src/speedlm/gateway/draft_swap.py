"""Worker extension for hot-swapping EAGLE-3 draft weights in place.

This module provides a vLLM worker extension that swaps the drafter's weights
without restarting the engine, preserving CUDA-graph-captured tensor pointers
by using the layerwise reload infrastructure.

**Worker extension composition.**  vLLM accepts a single
``--worker-extension-cls`` string.  This project already uses
``ActivationCaptureExtension`` for serving-time activation capture.
Rather than requiring the caller to choose between the two, this module
provides ``CombinedWorkerExtension`` -- a subclass of *both*
``ActivationCaptureExtension`` and ``DraftSwapExtension``.  Register it via::

    --worker-extension-cls speedlm.gateway.draft_swap.CombinedWorkerExtension

vLLM checks for attribute collisions at startup
(``vllm/v1/worker/worker_base.py:261-286``): it iterates ``dir(extension)``,
skips only names starting with ``__``, and asserts ``not hasattr(worker_class,
attr)`` for the rest before doing ``worker_class.__bases__ += (extension,)``.
The two mixins use disjoint names (``activate_``/``flush_``/``deactivate_``/
``_install_``/``_deactivate_``/``_buffer_``/``_extend_``/``capture_`` vs.
``hot_swap_``/``draft_``/``_apply_``/``_load_``/``_validate_``/``_target_``/
``_stranded_``/``_get_drafter``/``_get_quantization``/``_find_proposer``/
``_require_drafter_model``), except for the
deliberately shared
lazy-init helpers (``_ensure_init``/``_get_lock``/``_get_pending``) which are
inherited from a single definition in ``ActivationCaptureExtension`` and so
appear exactly once in ``dir()``.

**What CANNOT be done:** swap a drafter whose architecture, shapes, or
quantization differ from the currently loaded one.  The hot-swap is purely
weight-oriented; the tensor topology and CUDA-graph bindings must already
match.  ``_validate_compatibility`` rejects such a candidate *before* any
mutation happens.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from speedlm.activation_capture.hook import ActivationCaptureExtension

logger = logging.getLogger(__name__)


#: Checkpoint tensor names carrying the EAGLE-3 draft<->target vocabulary
#: mapping.  vLLM renames ``d2t`` to ``draft_id_to_target_id`` and drops
#: ``t2d`` outright (``llama_eagle3.py:386-390``), but the project's own
#: publisher requires both (``training/backends/eagle3.py`` ``_VALIDATE_DRAFT``),
#: so a candidate missing either one did not come out of our pipeline.
DRAFT_VOCAB_MAPPING_KEYS: Final[tuple[str, ...]] = ("d2t", "t2d")

#: HF shard index emitted alongside multi-shard checkpoints.  When present it
#: is authoritative: a directory may hold *both* per-shard files and a
#: consolidated copy, and loading both double-counts tensors.  vLLM applies the
#: same filter in ``default_loader.py:216-233``.
SAFETENSORS_INDEX_FILE: Final[str] = "model.safetensors.index.json"

#: Worker attributes that may hold the model runner, in preference order.
RUNNER_ATTRS: Final[tuple[str, ...]] = ("model_runner", "model_executor")

#: Runner attributes under which vLLM exposes the speculative proposer.  The
#: name depends on which model runner the worker built
#: (``gpu_worker.py:384-398`` branches on ``vllm_config.use_v2_model_runner``):
#:
#: * V1 ``GPUModelRunnerV1`` -> ``runner.drafter``, an ``EagleProposer``
#:   (``v1/worker/gpu_model_runner.py:571-634``).
#: * V2 ``GPUModelRunnerV2`` -> ``runner.speculator``, an ``EagleSpeculator``
#:   (``v1/worker/gpu/model_runner.py:189-194`` via ``init_speculator``).
#:
#: Both hold the draft ``nn.Module`` on ``.model``, assigned during
#: ``load_model`` (V2: ``v1/worker/gpu/spec_decode/speculator.py:152-160``),
#: so only the *runner-side* attribute name differs.  Checking both keeps a
#: single build working against either runner.
DRAFTER_ATTRS: Final[tuple[str, ...]] = ("drafter", "speculator")

#: Config fields that must match between the running drafter and a candidate
#: for the in-place swap to be tensor-topology compatible.  Names are read
#: from ``config.json`` rather than from checkpoint tensor names because vLLM
#: FUSES ``q/k/v_proj`` into ``qkv_proj`` and ``gate/up_proj`` into
#: ``gate_up_proj``, so a raw name set-diff against ``named_parameters()``
#: reports both "missing" and "extra" params for a byte-identical drafter.
DRAFT_SHAPE_FIELDS: Final[tuple[str, ...]] = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "draft_vocab_size",
)


# ---------------------------------------------------------------------------
# Candidate directory inspection (torch-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DraftConfigSummary:
    """The subset of a drafter's config that governs swap compatibility.

    Every field is optional because a candidate config may legitimately omit
    one (e.g. ``draft_vocab_size`` is ``None`` when the drafter shares the
    verifier's vocabulary).  A ``None`` on either side is treated as "unknown"
    and skipped rather than as a mismatch -- the checks that matter
    (``vocab_size``/``hidden_size``) are always present in both.
    """

    vocab_size: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    draft_vocab_size: int | None = None
    dtype: str | None = None


def _as_int(value: object) -> int | None:
    """Coerce a config value to ``int``, returning ``None`` when unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _index_map_values(
    tensor: Any, name: str, directory: Path
) -> list[int] | None:
    """Flatten a 1-D index-map tensor to a list of Python ints.

    ``None`` in, ``None`` out, so an absent ``t2d`` stays "not checked" rather
    than becoming an empty map that would fail every cross-check.  Anything
    else that cannot be read as a flat sequence of integers is an error and
    not a skip: a ``d2t`` this cannot decode is a ``d2t`` nothing downstream
    can validate, and silently waving it through is how the presence-only
    check came to be the only check.
    """
    if tensor is None:
        return None
    tolist = getattr(tensor, "tolist", None)
    if not callable(tolist):
        raise ValueError(
            f"candidate {name!r} is not a readable tensor "
            f"({type(tensor).__name__}); its vocabulary mapping cannot be "
            f"validated ({directory})"
        )
    values = tolist()
    if not isinstance(values, list) or any(
        isinstance(value, (list, tuple)) for value in values
    ):
        raise ValueError(
            f"candidate {name!r} must be a 1-D index map, got a nested "
            f"sequence ({directory})"
        )
    return [int(value) for value in values]


def _normalize_dtype(value: object) -> str | None:
    """Reduce ``torch.bfloat16`` / ``"bfloat16"`` / ``torch.dtype`` to a name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.rsplit(".", 1)[-1].lower()


def _natural_sort_key(path: Path) -> list[object]:
    """Order ``model-00002-of-00010.safetensors`` after ``...-00001-...``.

    Mirrors vLLM's ``weight_utils._natural_sort_key`` so that shard merge
    order matches what the engine's own loader would have used.
    """
    return [
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", path.name)
    ]


def resolve_safetensors_shards(directory: Path) -> list[Path]:
    """Resolve a candidate draft *directory* to its safetensors shard files.

    ``hot_swap_draft`` receives a directory path (that is what the tuner
    publishes and what ``RuntimeController._try_hot_swap`` passes), while
    ``safetensors.torch.load_file`` takes exactly one file.  This bridges the
    two the same way vLLM's ``DefaultModelLoader._prepare_weights`` does: a
    non-recursive ``*.safetensors`` glob, natural-sorted, filtered through the
    shard index when one exists.

    A single ``.safetensors`` file is accepted directly so the RPC stays
    usable against a file path.

    Raises:
        FileNotFoundError: if the path does not exist, or the directory holds
            no ``*.safetensors`` file at all.
        ValueError: if a file path with a non-safetensors suffix is given.
    """
    if directory.is_file():
        if directory.suffix != ".safetensors":
            raise ValueError(
                f"draft weights path is a file but not safetensors: {directory}"
            )
        return [directory]

    if not directory.is_dir():
        raise FileNotFoundError(f"draft weights path does not exist: {directory}")

    shards = sorted(directory.glob("*.safetensors"), key=_natural_sort_key)
    if not shards:
        raise FileNotFoundError(
            f"no *.safetensors weight shards in draft directory {directory}; "
            f"found instead: {sorted(p.name for p in directory.iterdir())[:10]}"
        )

    index_path = directory / SAFETENSORS_INDEX_FILE
    if index_path.is_file():
        indexed = _indexed_shard_names(index_path)
        if indexed:
            filtered = [shard for shard in shards if shard.name in indexed]
            if not filtered:
                raise FileNotFoundError(
                    f"{SAFETENSORS_INDEX_FILE} in {directory} references "
                    f"{sorted(indexed)[:5]} but none of those files are present"
                )
            shards = filtered

    return shards


def _indexed_shard_names(index_path: Path) -> set[str]:
    """Read the shard file names listed in an HF safetensors index."""
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable %s: %r", index_path, exc)
        return set()
    weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
    if not isinstance(weight_map, dict):
        return set()
    return {str(value) for value in weight_map.values()}


def read_draft_config(directory: Path) -> DraftConfigSummary:
    """Summarize the candidate draft directory's ``config.json``.

    Speculators-format EAGLE-3 configs nest the transformer dimensions under
    ``transformer_layer_config`` and keep the EAGLE-specific fields at the top
    level; vLLM flattens them the same way in
    ``transformers_utils/configs/speculators/base.py:59-64``.  This reproduces
    that flattening so the summary is directly comparable against the running
    drafter's ``hf_config``.

    Raises:
        FileNotFoundError: if the directory has no ``config.json``.
        ValueError: if the config cannot be parsed or is not a JSON object.
    """
    path = directory / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate draft has no config.json: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate draft config {path}: {exc!r}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"candidate draft config is not a JSON object: {path}")

    nested = raw.get("transformer_layer_config")
    layer: Mapping[str, object] = nested if isinstance(nested, dict) else {}

    def pick(key: str) -> object:
        return layer[key] if key in layer else raw.get(key)

    dtype = pick("dtype")
    if dtype is None:
        dtype = pick("torch_dtype")

    return DraftConfigSummary(
        vocab_size=_as_int(pick("vocab_size")),
        hidden_size=_as_int(pick("hidden_size")),
        num_hidden_layers=_as_int(pick("num_hidden_layers")),
        draft_vocab_size=_as_int(raw.get("draft_vocab_size")),
        dtype=_normalize_dtype(dtype),
    )


def summarize_hf_config(hf_config: Any) -> DraftConfigSummary:
    """Summarize the *running* drafter's already-flattened ``hf_config``."""
    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        dtype = getattr(hf_config, "torch_dtype", None)
    return DraftConfigSummary(
        vocab_size=_as_int(getattr(hf_config, "vocab_size", None)),
        hidden_size=_as_int(getattr(hf_config, "hidden_size", None)),
        num_hidden_layers=_as_int(getattr(hf_config, "num_hidden_layers", None)),
        draft_vocab_size=_as_int(getattr(hf_config, "draft_vocab_size", None)),
        dtype=_normalize_dtype(dtype),
    )


# ---------------------------------------------------------------------------
# Worker extension
# ---------------------------------------------------------------------------


class DraftSwapExtension:
    """vLLM worker extension for in-place draft weight hot-swapping.

    Base class for :class:`CombinedWorkerExtension`.  Registering this one
    directly via ``--worker-extension-cls`` is supported and gives draft
    hot-swap *without* activation capture.

    Public methods are called via ``collective_rpc`` from the driver process,
    so every argument and return value must be JSON-serializable
    (``/collective_rpc`` passes string args only).

    **Note:** vLLM never calls ``__init__`` on an extension -- it appends the
    class to ``worker_class.__bases__``.  This class therefore holds no
    instance state at all; the attributes below are bare annotations (not
    assignments) so they do not appear in ``dir()`` and cannot trip vLLM's
    attribute-collision assertion against the real worker attributes.
    """

    #: Injected by vLLM at runtime via ``__bases__`` injection into WorkerBase.
    model_executor: Any
    model_runner: Any
    vllm_config: Any

    # -- collective_rpc handlers --

    def hot_swap_draft(self, weights_path: str) -> dict[str, Any]:
        """Swap the drafter's weights from the directory *weights_path*.

        *weights_path* is a candidate draft **directory** (the shape the tuner
        publishes).  Shard resolution, merging, and compatibility validation
        all happen here, worker-side.

        Validation runs to completion before anything is mutated, so a
        rejected candidate leaves the running drafter untouched.  The apply
        step uses the layerwise reload infrastructure so CUDA-graph-captured
        tensor pointers survive, and asserts afterwards that neither the
        drafter nor the verifier was left stranded on the meta device.

        Returns:
            ``{"swapped": True, "parameters_loaded": int,
            "target_owned_submodules": [...], "dropped_tensors": [...]}``.
            The last two report the *live* drafter/verifier sharing topology
            that governed the payload, because how many tensors legitimately
            load depends on it (see :meth:`_target_owned_submodules`) and a
            bare count cannot be interpreted without it.

        Raises:
            RuntimeError: no drafter is loaded, or the swap left tensors on
                the meta device.
            FileNotFoundError / ValueError: the candidate is unusable or
                incompatible (raised before any mutation).
        """
        directory = Path(weights_path)

        drafter = self._require_drafter_model()

        #: Load first so the vocab-mapping check can see the real tensor keys.
        new_weights = self._load_weights_file(weights_path)

        #: Validate compatibility BEFORE touching anything.
        self._validate_compatibility(drafter, new_weights, directory)

        owned = self._target_owned_submodules(drafter)
        dropped = self._target_owned_tensor_names(new_weights, owned)
        count = self._apply_weights(drafter, new_weights, owned=owned)

        logger.info("Hot-swapped %d drafter parameters from %s", count, weights_path)
        return {
            "swapped": True,
            "parameters_loaded": count,
            "target_owned_submodules": sorted(owned),
            "dropped_tensors": dropped,
        }

    def draft_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded drafter.

        Used by the caller to validate compatibility before hot-swapping.
        All values are JSON-serializable.
        """
        drafter = self._require_drafter_model()

        param_shapes: dict[str, list[int]] = {}
        param_dtypes: dict[str, str] = {}
        count = 0
        for name, param in drafter.named_parameters():
            param_shapes[name] = list(param.shape)
            param_dtypes[name] = str(param.dtype)
            count += 1

        return {
            "num_parameters": count,
            "parameter_shapes": param_shapes,
            "parameter_dtypes": param_dtypes,
            "quantization": self._get_quantization(),
            "draft_config": asdict(summarize_hf_config(self._draft_hf_config())),
            #: The live sharing topology.  vLLM decides per checkpoint whether
            #: the drafter's embedding/head are the verifier's objects or the
            #: drafter's own copies (see :meth:`_target_owned_submodules`), and
            #: that decision changes how many candidate tensors a swap may
            #: legitimately load.  Exposing it lets a caller *derive* the
            #: expected payload size instead of assuming one topology.
            "target_owned_submodules": sorted(self._target_owned_submodules(drafter)),
        }

    # -- model lookup --

    def _get_runner(self) -> Any:
        """Return the model runner, preferring whichever attribute has a drafter."""
        for attr in RUNNER_ATTRS:
            runner = getattr(self, attr, None)
            if runner is not None and self._find_proposer(runner) is not None:
                return runner
        for attr in RUNNER_ATTRS:
            runner = getattr(self, attr, None)
            if runner is not None:
                return runner
        return None

    @staticmethod
    def _find_proposer(runner: Any) -> Any:
        """Return *runner*'s speculative proposer object, or ``None``.

        See :data:`DRAFTER_ATTRS` -- the V1 and V2 model runners name this
        attribute differently, so both are probed.
        """
        for attr in DRAFTER_ATTRS:
            proposer = getattr(runner, attr, None)
            if proposer is not None:
                return proposer
        return None

    @staticmethod
    def _unwrap(model: Any) -> Any:
        """Strip vLLM's CUDA-graph wrappers off a model reference.

        ``gpu_model_runner.py:5359-5362`` replaces ``drafter.model`` with a
        ``BreakableCUDAGraphWrapper``, which is *not* an ``nn.Module`` -- it
        only forwards reads through ``__getattr__``.  Writes (which
        ``initialize_layerwise_reload`` performs, e.g. ``_do_torchao_reload``)
        would land on the wrapper instead of the model, so unwrap first.
        """
        for _ in range(8):
            unwrap = getattr(model, "unwrap", None)
            if not callable(unwrap):
                break
            inner = unwrap()
            if inner is None or inner is model:
                break
            model = inner
        return model

    def _get_drafter_model(self) -> Any:
        """Retrieve the drafter's (unwrapped) model object from the runner."""
        runner = self._get_runner()
        if runner is None:
            return None
        proposer = self._find_proposer(runner)
        if proposer is None:
            return None
        return self._unwrap(getattr(proposer, "model", None))

    def _drafter_lookup_report(self) -> str:
        """Describe what the drafter lookup walked, for a diagnostic failure.

        A bare "no drafter model found" cannot distinguish "the engine has no
        speculative config" from "the drafter hangs off an attribute this
        build does not know about" -- the exact ambiguity that made the first
        GPU run of hot-swap unreadable.  This renders the object graph that
        *was* found so the next failure names its own cause.
        """
        parts: list[str] = []
        for attr in RUNNER_ATTRS:
            runner = getattr(self, attr, None)
            if runner is None:
                parts.append(f"self.{attr}=<absent>")
                continue
            found: list[str] = []
            for name in DRAFTER_ATTRS:
                proposer = getattr(runner, name, None)
                if proposer is None:
                    continue
                model = getattr(proposer, "model", None)
                model_desc = "None" if model is None else type(model).__name__
                found.append(
                    f"{name}={type(proposer).__name__}(.model={model_desc})"
                )
            detail = ", ".join(found) if found else "none of " + "/".join(DRAFTER_ATTRS)
            parts.append(f"self.{attr}={type(runner).__name__} -> {detail}")
        return "; ".join(parts)

    def _require_drafter_model(self) -> Any:
        """Return the drafter model, or raise a self-explaining ``RuntimeError``."""
        drafter = self._get_drafter_model()
        if drafter is None:
            raise RuntimeError(
                "no drafter model found; hot-swap requires an engine started "
                f"with a speculative config. Tried runner attributes "
                f"{list(RUNNER_ATTRS)} and drafter attributes "
                f"{list(DRAFTER_ATTRS)}; found: {self._drafter_lookup_report()}"
            )
        return drafter

    def _get_target_model(self) -> Any:
        """Retrieve the verifier's (unwrapped) model object from the runner."""
        runner = self._get_runner()
        if runner is None:
            return None
        return self._unwrap(getattr(runner, "model", None))

    # -- weight loading --

    def _load_weights_file(self, path: str) -> dict[str, Any]:
        """Load and merge every safetensors shard under the draft *path*.

        ``safetensors.safe_open`` handles is NOT iterable -- ``.keys()`` is the
        only enumeration entry point (this mirrors
        ``activation_capture/offline_extract.py`` and the ``_VALIDATE_DRAFT``
        script in ``training/backends/eagle3.py``).
        """
        from safetensors import safe_open

        shards = resolve_safetensors_shards(Path(path))

        merged: dict[str, Any] = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118 -- safe_open is not iterable
                    if name in merged:
                        raise ValueError(
                            f"tensor {name!r} appears in more than one shard under "
                            f"{path}; refusing to guess which copy is current"
                        )
                    merged[name] = handle.get_tensor(name)

        if not merged:
            raise ValueError(
                f"draft weight shards under {path} contain no tensors: "
                f"{[s.name for s in shards]}"
            )
        return merged

    # -- compatibility --

    def _draft_hf_config(self) -> Any:
        """Return the running drafter's HF config, or ``None`` if unavailable."""
        config = getattr(self, "vllm_config", None)
        spec = getattr(config, "speculative_config", None)
        draft_cfg = getattr(spec, "draft_model_config", None)
        return getattr(draft_cfg, "hf_config", None)

    def _draft_model_config(self) -> Any:
        """Return the ModelConfig ``finalize_layerwise_reload`` should use.

        The drafter's own config is correct here: ``finalize_layerwise_reload``
        only consults it when re-finalizing attention layers
        (``reload/layerwise.py:279-281``), and those are the *draft* model's
        attention layers.  Falls back to the target config so an older vLLM
        without ``draft_model_config`` still finalizes.
        """
        config = getattr(self, "vllm_config", None)
        spec = getattr(config, "speculative_config", None)
        draft_cfg = getattr(spec, "draft_model_config", None)
        if draft_cfg is not None:
            return draft_cfg
        return getattr(config, "model_config", None)

    @staticmethod
    def _requires_vocab_mapping(drafter: Any) -> bool:
        """Whether the running drafter carries a draft->target vocab mapping.

        Ground truth is the live buffer rather than the config: vLLM only
        creates ``draft_id_to_target_id`` when the draft vocabulary is a
        subset of the verifier's.
        """
        named_buffers = getattr(drafter, "named_buffers", None)
        if not callable(named_buffers):
            return False
        return any("draft_id_to_target_id" in name for name, _ in named_buffers())

    def _validate_compatibility(
        self,
        drafter: Any,
        new_weights: Mapping[str, Any],
        directory: Path,
    ) -> None:
        """Raise if the candidate under *directory* cannot replace *drafter*.

        This deliberately does NOT set-diff checkpoint tensor names against
        ``drafter.named_parameters()``.  vLLM fuses ``q/k/v_proj`` into
        ``qkv_proj`` and ``gate/up_proj`` into ``gate_up_proj`` while HF
        checkpoints store them separately, and the EAGLE proposer *may* delete
        the drafter's ``embed_tokens``/``lm_head`` and rebind the verifier's
        (conditionally -- see :meth:`_target_owned_submodules`).  A name
        set-diff therefore reports both "missing" and "extra" parameters for a
        byte-identical drafter.  ``drafter.load_weights()`` already resolves
        fusion correctly, so what is validated here is what the loader cannot
        recover from: differing tensor topology, read from ``config.json``.
        """
        running_config = self._draft_hf_config()
        if running_config is None:
            raise RuntimeError(
                "cannot read the running drafter's config "
                "(vllm_config.speculative_config.draft_model_config.hf_config); "
                "refusing to hot-swap unvalidated weights"
            )

        current = summarize_hf_config(running_config)
        candidate = read_draft_config(directory)

        for field_name in DRAFT_SHAPE_FIELDS:
            want = getattr(current, field_name)
            got = getattr(candidate, field_name)
            if want is None or got is None:
                continue
            if want != got:
                raise ValueError(
                    f"draft {field_name} mismatch: running={want}, candidate={got} "
                    f"({directory})"
                )

        if (
            current.dtype is not None
            and candidate.dtype is not None
            and current.dtype != candidate.dtype
        ):
            raise ValueError(
                f"draft dtype mismatch: running={current.dtype}, "
                f"candidate={candidate.dtype} ({directory})"
            )

        if self._requires_vocab_mapping(drafter):
            missing = [key for key in DRAFT_VOCAB_MAPPING_KEYS if key not in new_weights]
            if missing:
                raise ValueError(
                    f"candidate draft is missing vocab mappings {missing}; the running "
                    f"drafter has a draft_id_to_target_id buffer and needs them "
                    f"({directory})"
                )
            self._validate_vocab_mapping(new_weights, directory, current)

    def _verifier_vocab_size(self, current: DraftConfigSummary) -> int:
        """Return the running *verifier's* vocabulary size.

        Read from ``vllm_config.model_config.hf_config`` -- the verifier's own
        ``config.json`` as the running server loaded it -- and never from a
        constant, because the failure being guarded is precisely a map
        calibrated against one vocabulary being applied to another.

        Falls back to the drafter's ``vocab_size``, which speculators
        populates from the verifier's config at build time
        (``model.py:47`` ``self.verifier_vocab_size = tl_config.vocab_size``).
        That is a weaker witness -- it is the drafter's *copy* of the fact --
        but it is still a config-derived runtime value, and having it means a
        vLLM build that does not expose ``model_config`` degrades to a real
        check rather than to no check.

        Raises:
            ValueError: if neither source yields a usable size.  An unknown
                verifier vocabulary is not permission to swap unvalidated
                index maps into a running server.
        """
        config = getattr(self, "vllm_config", None)
        model_config = getattr(config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        for candidate in (text_config, hf_config):
            size = _as_int(getattr(candidate, "vocab_size", None))
            if size is not None and size > 0:
                return size
        if current.vocab_size is not None and current.vocab_size > 0:
            return current.vocab_size
        raise ValueError(
            "cannot read the verifier's vocab_size from "
            "vllm_config.model_config.hf_config or from the running drafter's "
            "config; refusing to swap in an unvalidated draft vocab mapping"
        )

    def _validate_vocab_mapping(
        self,
        new_weights: Mapping[str, Any],
        directory: Path,
        current: DraftConfigSummary,
    ) -> None:
        """Raise unless every ``i + d2t[i]`` lands inside the verifier's vocab.

        The presence check above only established that ``d2t``/``t2d`` exist.
        That is the check this repo shipped, and it is satisfied by a d2t of
        the right shape holding anything at all -- including all zeros, the
        identity map that mistranslates every drafted token.  vLLM will not
        catch it either: it applies the map as ``base + d2t`` with no clamp and
        no assert (``llama_eagle3.py:347-357``), where a *negative* target
        wraps silently under PyTorch advanced indexing rather than raising.

        The rule itself lives in :func:`speedlm.tuner.eagle3.check_vocab_mapping`
        so the swap-time and publish-time checks cannot drift apart.  Imported
        lazily: the tuner module is login-node code and has no business in the
        worker's import graph until a swap actually happens.
        """
        from speedlm.tuner.eagle3 import DraftVocabMappingError, check_vocab_mapping

        d2t = _index_map_values(new_weights["d2t"], "d2t", directory)
        if d2t is None:
            # The key is present (checked above) but holds ``None``.  Not a
            # shape this pipeline produces, and not a reason to skip: an
            # absent d2t is the identity-map corruption in another costume.
            raise ValueError(f"candidate draft's d2t is None ({directory})")
        t2d = _index_map_values(new_weights.get("t2d"), "t2d", directory)
        try:
            report = check_vocab_mapping(
                d2t,
                t2d,
                verifier_vocab_size=self._verifier_vocab_size(current),
                source=str(directory),
            )
        except DraftVocabMappingError as exc:
            # Re-raised as ValueError so it joins the other pre-mutation
            # rejections in this method: ``hot_swap_draft`` documents
            # ValueError as "candidate is incompatible, nothing was touched",
            # and a caller should not have to import a tuner exception to
            # handle a rejected swap.
            raise ValueError(str(exc)) from exc
        logger.info(
            "Candidate draft vocab mapping validated: %d draft ids -> targets "
            "[%d, %d] within verifier vocab_size %d (%s)",
            report["draft_vocab_size"],
            report["min_target"],
            report["max_target"],
            report["verifier_vocab_size"],
            directory,
        )

    # -- apply --

    def _target_owned_submodules(self, drafter: Any) -> dict[str, Any]:
        """Map drafter submodule paths that are *the verifier's* modules.

        The EAGLE proposer *may* share the verifier's ``embed_tokens`` and
        ``lm_head`` with the drafter by object identity.  When it does, they
        are reachable from ``drafter.modules()`` but they are NOT the
        drafter's to reload: ``initialize_layerwise_reload`` walks every module
        and calls ``restore_layer_on_meta`` on each
        (``reload/layerwise.py:100-117``), which would move the *running
        verifier's* embedding onto the meta device and never restore it.

        **Sharing is conditional, not structural.**  Both runner generations
        gate it on the same two questions -- does the checkpoint carry its own
        copy, and if so is that copy weight-identical to the verifier's:

        * V1 ``EagleProposer``: ``llm_base_proposer.py:1432-1461``
          (``has_own_embed_tokens`` + ``torch.equal``) and ``:1508-1547``
          (``has_own_lm_head`` + ``torch.equal``).
        * V2 ``EagleSpeculator``: ``v1/worker/gpu/spec_decode/eagle/utils.py``
          ``_should_share`` (``:12-25``), applied to ``embed_tokens`` at
          ``:60-65`` and ``lm_head`` at ``:69-74``.  It rebinds under the *same*
          attribute paths V1 uses -- ``drafter.model.embed_tokens`` and
          ``drafter.lm_head`` -- so nothing about the V2 topology needs special
          casing here.

        The flags themselves are set from the checkpoint's tensor names during
        loading (``model_executor/models/utils.py:915-934``).  So a drafter
        published with a *distinct* embedding or a reduced-vocabulary head --
        e.g. an EAGLE-3 checkpoint with ``draft_vocab_size`` smaller than the
        verifier's ``vocab_size``, which makes the head shapes differ and
        ``torch.equal`` therefore ``False`` -- genuinely owns those weights,
        and this correctly returns ``{}`` for it.  That is why the answer is
        read from live object identity rather than from a fixed name list: the
        same drafter architecture yields a different answer per checkpoint.

        Only top-most matches are returned; detaching a parent detaches its
        children with it.
        """
        target = self._get_target_model()
        if target is None or not callable(getattr(target, "modules", None)):
            return {}
        if not callable(getattr(drafter, "named_modules", None)):
            return {}

        target_ids = {id(module) for module in target.modules()}
        owned: dict[str, Any] = {}
        for name, module in drafter.named_modules():
            if name and id(module) in target_ids:
                owned[name] = module

        topmost = {
            name: module
            for name, module in owned.items()
            if not any(name.startswith(f"{other}.") for other in owned if other != name)
        }
        #: Logged unconditionally, including the empty case.  vLLM logs its
        #: sharing decision on V1 but not on V2 (``eagle/utils.py`` is silent),
        #: so without this line an absent "dropping ..." warning is ambiguous
        #: between "nothing is shared" and "the lookup failed" -- an ambiguity
        #: that already cost one GPU run.
        logger.info(
            "Drafter shares %d submodule(s) with the verifier by identity: %s",
            len(topmost),
            sorted(topmost),
        )
        return topmost

    @staticmethod
    def _target_owned_tensor_names(
        new_weights: Mapping[str, Any], owned: Mapping[str, Any]
    ) -> list[str]:
        """Candidate tensor names that would land in a verifier-owned module.

        The single definition of the drop rule, shared by
        :meth:`_drop_target_owned_weights` (which applies it) and
        :meth:`hot_swap_draft` (which reports it), so the reported set cannot
        drift from the applied one.
        """
        if not owned:
            return []
        leaves = {path.rpartition(".")[2] for path in owned}
        return sorted(
            name for name in new_weights if any(leaf in name for leaf in leaves)
        )

    @contextmanager
    def _detached_submodules(
        self, drafter: Any, owned: Mapping[str, Any]
    ) -> Iterator[None]:
        """Temporarily unbind *owned* submodules from *drafter*.

        This is how target-owned modules are excluded from
        ``initialize_layerwise_reload``'s ``model.modules()`` walk -- the vLLM
        helper takes no exclusion argument, so the only way to scope it is to
        make the modules unreachable for the duration.
        """
        detached: list[tuple[Any, str, Any]] = []
        try:
            for path in sorted(owned):
                parent_path, _, attr = path.rpartition(".")
                holder = drafter.get_submodule(parent_path) if parent_path else drafter
                delattr(holder, attr)
                detached.append((holder, attr, owned[path]))
            yield
        finally:
            for holder, attr, module in reversed(detached):
                setattr(holder, attr, module)

    @classmethod
    def _drop_target_owned_weights(
        cls, new_weights: Mapping[str, Any], owned: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Drop candidate tensors destined for verifier-owned modules.

        Loading them would write the candidate's embedding/head into the
        *running verifier*.  vLLM applies the same substring exclusion via
        ``AutoWeightsLoader(skip_substrs=...)`` in
        ``llama_eagle3.py:419-432``.

        When nothing is shared (``owned`` empty) the drafter owns its own
        embedding/head and the candidate's copies are exactly what must be
        loaded, so the payload passes through untouched.
        """
        dropped = set(cls._target_owned_tensor_names(new_weights, owned))
        if not dropped:
            return dict(new_weights)

        logger.warning(
            "Dropping %d candidate tensors bound to verifier-shared modules %s: %s",
            len(dropped),
            sorted(path.rpartition(".")[2] for path in owned),
            sorted(dropped)[:5],
        )
        return {
            name: tensor for name, tensor in new_weights.items() if name not in dropped
        }

    def _stranded_on_meta(self, drafter: Any) -> dict[str, list[str]]:
        """Return ``{"draft": [...], "target": [...]}`` of meta-device tensors.

        Walks parameters *and* buffers of both the drafter and the verifier.
        The verifier is included because the two share modules by object
        identity, so a mis-scoped layerwise reload strands the *target's*
        embedding rather than anything the drafter owns.
        """
        stranded: dict[str, list[str]] = {"draft": [], "target": []}
        for label, model in (("draft", drafter), ("target", self._get_target_model())):
            if model is None:
                continue
            for accessor in ("named_parameters", "named_buffers"):
                enumerate_tensors = getattr(model, accessor, None)
                if not callable(enumerate_tensors):
                    continue
                for name, tensor in enumerate_tensors():
                    device = getattr(tensor, "device", None)
                    if getattr(device, "type", None) == "meta":
                        stranded[label].append(name)
        return stranded

    def draft_materialization_report(self) -> dict[str, Any]:
        """Read-only report of tensors left on the meta device.

        Purely observational test-support accessor: it mutates nothing and is
        safe to call at any time.  ``_assert_materialized`` runs the same walk
        as a post-condition inside ``hot_swap_draft``, but its only observable
        effect is raising, so an E2E caller cannot distinguish "swap kept every
        tensor materialized" from "the post-condition never ran".  This exposes
        the walk as data so the caller can assert on it directly, and take a
        pre-swap baseline to compare against.

        Returns:
            ``{"stranded": {"draft": [...], "target": [...]},
            "stranded_total": int}`` -- all JSON-serializable.
        """
        drafter = self._require_drafter_model()
        stranded = self._stranded_on_meta(drafter)
        return {
            "stranded": stranded,
            "stranded_total": sum(len(names) for names in stranded.values()),
        }

    def _assert_materialized(self, drafter: Any) -> None:
        """Raise if the swap left drafter or verifier tensors on meta.

        ``restore_layer_on_meta`` strips a layer's parameters and re-registers
        meta-device placeholders; only a successful load materializes them
        again.  Anything still on meta afterwards is a silently broken model,
        so this is the post-condition that turns a partial swap into a loud
        failure the caller can fall back from.
        """
        by_label = self._stranded_on_meta(drafter)
        stranded = [
            f"{label}.{name}" for label, names in by_label.items() for name in names
        ]

        if stranded:
            raise RuntimeError(
                f"draft hot-swap left {len(stranded)} tensors on the meta device "
                f"(model is unusable): {sorted(stranded)[:5]}"
            )

    def _apply_weights(
        self,
        drafter: Any,
        new_weights: Mapping[str, Any],
        *,
        owned: Mapping[str, Any] | None = None,
    ) -> int:
        """Apply *new_weights* into *drafter* in-place, preserving CUDA graphs.

        *owned* is the verifier-shared submodule map from
        :meth:`_target_owned_submodules`; it is accepted as an argument so
        :meth:`hot_swap_draft` can report the same topology it acted on rather
        than recomputing (and possibly re-deriving a different) one.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        if owned is None:
            owned = self._target_owned_submodules(drafter)
        payload = self._drop_target_owned_weights(new_weights, owned)
        model_config = self._draft_model_config()

        with self._detached_submodules(drafter, owned):
            initialize_layerwise_reload(drafter)
            try:
                loaded = drafter.load_weights(payload.items())
            finally:
                try:
                    finalize_layerwise_reload(drafter, model_config)
                except Exception:
                    logger.exception(
                        "finalize_layerwise_reload failed; weights may be partial"
                    )
                    raise

        self._assert_materialized(drafter)

        if isinstance(loaded, (set, frozenset, list, tuple, dict)):
            count = len(loaded)
        else:
            #: ``Eagle3LlamaForCausalLM.load_weights`` (``llama_eagle3.py:381-432``)
            #: discards the ``AutoWeightsLoader`` result and returns ``None``, so
            #: there is no loaded-name set to count.  Fall back to the number of
            #: candidate tensors handed to the loader; ``_assert_materialized``
            #: above is what proves they actually landed.
            count = len(payload)

        if count == 0:
            raise RuntimeError(
                "draft hot-swap loaded zero parameters; refusing to report success"
            )
        return count

    def _get_quantization(self) -> str | None:
        """Read the drafter's quantization setting from the vLLM config."""
        config = getattr(self, "vllm_config", None)
        spec = getattr(config, "speculative_config", None)
        draft_cfg = getattr(spec, "draft_model_config", None)
        quantization = getattr(draft_cfg, "quantization", None)
        return str(quantization) if quantization is not None else None


class CombinedWorkerExtension(ActivationCaptureExtension, DraftSwapExtension):
    """Composite vLLM worker extension: activation capture + draft swap.

    vLLM accepts a single ``--worker-extension-cls`` string, so this class
    exists purely to merge :class:`~speedlm.activation_capture.hook.
    ActivationCaptureExtension` and :class:`DraftSwapExtension` into one MRO.
    It defines no members of its own -- every method has exactly one
    implementation, in its owning base, so the two cannot drift apart.

    Register via ``--worker-extension-cls
    speedlm.gateway.draft_swap.CombinedWorkerExtension``.

    vLLM's attribute-collision check (``worker_base.py:261-286``) iterates
    ``dir()`` of this class, which flattens the MRO, so the shared lazy-init
    helpers appear once and the two disjoint method sets never collide with
    each other.  The lazy class-default + ``_ensure_init()`` pattern is
    inherited unchanged, which matters because vLLM appends this class to
    ``worker_class.__bases__`` and never calls ``__init__``.
    """
