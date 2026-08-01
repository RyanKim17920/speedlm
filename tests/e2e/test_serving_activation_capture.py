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
* ``SPEEDLM_E2E_PROMPT`` — override the fixed prompt (default: a short
  English sentence)
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
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from speedlm.activation_capture.compare import (
    PrefixCacheResult,
    align_prompt_rows,
    build_result,
)
from speedlm.activation_capture.hf_reference import (
    HFReferenceResult,
    compare_to_hf_reference,
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


def _collective_rpc(
    vllm_proc: subprocess.Popen[bytes], port: int, method: str, *args: object
) -> None:
    """Issue a collective_rpc call to the vLLM engine via the debug endpoint.

    vLLM exposes a /collective_rpc endpoint that forwards to all workers.
    The caller must pass the actual port the engine is listening on.
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
        # vLLM returns {"results": [...]} on success.  Each entry is the
        # worker's return value (None, a dict/list, or str(result)).  If a
        # worker-side method raised, the entry may contain error information.
        body = resp.json()
        if "results" in body:
            for i, result in enumerate(body["results"]):
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(
                        f"collective_rpc {method} worker {i} error: {result['error']}"
                    )


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
        f"comparison meaningless.  Rendered: {tokenizer.decode(ids)!r}"
    )
    return ids


def _free_device_bytes() -> int | None:
    """Return free VRAM on the current device, or ``None`` if there is none."""
    if torch is None or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _run_hf_reference(
    verifier: str,
    prompt: str,
    captured: dict[int, torch.Tensor],
    *,
    prompt_token_count: int,
    final_layer_idx: int | None,
) -> HFReferenceResult:
    """Re-derive the captured activations with HuggingFace transformers, fp32.

    Runs **after** both vLLM engines are gone, reusing
    :func:`_wait_for_gpu_memory_release` — the same nvidia-smi-polling
    machinery the offline phase already uses — so the fp32 copy gets the whole
    card rather than competing with an engine for it.  See
    ``hf_reference.select_reference_device`` for why GPU-after-teardown was
    chosen over CPU-fp32 or a smaller model, and why CPU remains the automatic
    fallback.
    """
    token_ids = _prompt_token_ids(
        verifier, prompt, expected_count=prompt_token_count
    )

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

    forced = os.environ.get("SPEEDLM_E2E_HF_REFERENCE_DEVICE")
    if forced in ("cuda", "cpu"):
        device = forced
    elif num_parameters is None:
        device = "cpu"
    else:
        device = select_reference_device(
            num_parameters=num_parameters,
            free_device_bytes=_free_device_bytes(),
        )
    logger.info(
        "HF fp32 reference: %d params, device=%s, %d prompt tokens, %d layers",
        num_parameters or -1, device, len(token_ids), num_hidden_layers,
    )

    stream, post_norm_final, dtype_name = reference_residual_stream(
        str(model_dir), token_ids, device=device
    )
    return compare_to_hf_reference(
        captured,
        stream,
        post_norm_final,
        prompt_token_count=prompt_token_count,
        final_layer_idx=final_layer_idx,
        device=device,
        dtype=dtype_name,
    )


def _checkpoint_parameter_count(model_dir: Path) -> int | None:
    """Estimate the parameter count from the safetensors index, or ``None``.

    ``metadata.total_size`` is the checkpoint's byte count; these verifiers ship
    in bf16, so two bytes per parameter.  Returning ``None`` on any doubt keeps
    the device choice conservative (CPU) rather than guessing high and OOMing.
    """
    index = model_dir / "model.safetensors.index.json"
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


def test_stage0_activation_capture() -> None:
    """Full Stage 0 experiment: serve one prompt with capture, compare offline.

    This test:
    1. Starts a vLLM engine with the EAGLE-3 speculator and capture extension.
    2. Sends a single fixed prompt.
    3. Flushes captured activations to disk.
    4. Tears down the capture engine.
    5. Waits for GPU memory to be released.
    6. Runs the offline extraction on the same prompt.
    7. Compares the two tensor stacks elementwise per layer (Leg 1 —
       transport; bit-identical is the expected result, see module docstring).
    8. Re-derives the same activations with HuggingFace transformers in fp32
       and checks tolerance *and* neighbour discrimination (Leg 2 — identity).
    9. Writes a JSON result with PASS/FAIL verdict.

    Only step 8 can distinguish a correct capture from a well-transported
    wrong quantity.  Step 7 returning 0.0 is not evidence of correctness.
    """
    verifier, drafter, artifact_root = _require_environment()
    prompt = os.environ.get("SPEEDLM_E2E_PROMPT", DEFAULT_PROMPT)
    target_layers = _target_layer_ids(verifier, drafter)
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
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.5",
            "--no-enable-prefix-caching",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_vllm_env(),
    )

    # Phase 1: capture engine — serve, capture, and tear down.
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_ready(url, vllm_proc, timeout, log_path=vllm_log)

        served_model_id = _get_served_model_id(url)

        # Step 2: Activate capture via collective_rpc, send prompt, flush
        _collective_rpc(vllm_proc, port, "activate_capture", str(capture_dir))
        _, prompt_token_count = _send_prompt(
            url, prompt, served_model_id=served_model_id
        )
        logger.info("Engine reported %d prompt tokens", prompt_token_count)
        # Small pause to let the hook buffer finish
        time.sleep(0.5)
        _collective_rpc(vllm_proc, port, "flush_capture")

        # Step 3: Load captured tensors
        captured_tensors = _load_captured_safetensors(capture_dir)
        logger.info("Captured layers: %s", sorted(captured_tensors.keys()))

        # Load metadata written by the hook during flush_capture so we can
        # correctly distinguish drafter-input layers from the appended final
        # regression-target layer.
        meta = _load_capture_metadata(capture_dir)
        final_layer_idx = meta["final_layer_idx"]
        original_aux = meta["original_aux_layers"]

        # Verify the drafter-input layers match the test's expectations.
        assert sorted(original_aux) == target_layers, (
            f"Captured drafter-input layers {sorted(original_aux)} do not match "
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
            f"Captured layer keys {actual_captured} do not match expected "
            f"{expected_captured} (original_aux={original_aux}, "
            f"final_layer_idx={final_layer_idx})"
        )

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()

    # Phase 2: wait for GPU memory release before launching offline engine.
    # The capture engine held GPU memory; the offline engine must not start
    # until the device memory is actually returned, otherwise both engines
    # compete for the same GPU and the second crashes.
    _wait_for_gpu_memory_release(gpu_memory_fraction=0.5)

    # Phase 3: offline extraction — now the GPU is free.
    offline_target_layers = list(target_layers)
    if final_layer_idx is not None:
        offline_target_layers.append(final_layer_idx)
    offline_dir = artifact_dir / "offline"
    hs_dir = _run_offline_extract(
        verifier,
        prompt,
        offline_target_layers,
        offline_dir,
        port=port + 1000,  # Different port
    )
    offline_tensors = _load_offline_hidden_states(hs_dir, target_layers=offline_target_layers)
    logger.info("Offline layers: %s", sorted(offline_tensors.keys()))

    # Phase 4: Align and compare — using explicit split to avoid comparing
    # drafter-input layers against the regression target or vice versa.

# Split captured tensors into drafter-inputs and regression-target.
    (
        captured_drafter_ids,
        captured_final_idx,
        captured_drafter,
        captured_regression,
    ) = _split_captured_layers(captured_tensors, meta)

    # Split offline tensors into drafter-inputs and regression-target.
    (
        offline_drafter_ids,
        offline_final_idx,
        offline_drafter,
        offline_regression,
    ) = _split_offline_layers(offline_tensors, offline_target_layers)

    # The drafter-input layer ids must match between captured and offline.
    assert captured_drafter_ids == offline_drafter_ids, (
        f"Drafter-input layer mismatch: captured {captured_drafter_ids} "
        f"vs offline {offline_drafter_ids}"
    )

    # The final layer indices must also match.
    assert captured_final_idx == offline_final_idx, (
        f"Final layer index mismatch: captured {captured_final_idx} "
        f"vs offline {offline_final_idx}"
    )

    # The comparable row range is the PROMPT, and only the prompt.  The
    # engine's own usage.prompt_tokens is the authority; the offline row count
    # is not — it also covers the assistant turn that prepare_data.py renders
    # into the conversation, whose tokens the serving engine never saw.  See
    # ``align_prompt_rows``.
    offline_rows = next(iter(offline_drafter.values())).shape[0]
    logger.info(
        "Aligning on %d prompt rows (offline stack has %d rows; the extra "
        "%d are the template's assistant turn and are NOT comparable)",
        prompt_token_count, offline_rows, offline_rows - prompt_token_count,
    )

    # Align drafter-input layers (same key set on both sides)
    aligned_captured: dict[int, torch.Tensor] = {}
    aligned_offline: dict[int, torch.Tensor] = {}
    for idx in offline_drafter_ids:
        cap = captured_drafter[idx]
        off = offline_drafter[idx]
        try:
            c_aligned, o_aligned = _align_token_count(cap, off, prompt_token_count)
        except ValueError as exc:
            raise AssertionError(f"cannot align layer {idx}: {exc}") from exc
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
            raise AssertionError(f"cannot align final layer: {exc}") from exc
    # Phase 5: independent HuggingFace fp32 re-derivation (Leg 2).
    #
    # Leg 1 above is a transport check between two views of ONE tensor (see
    # the module docstring).  This is the only part of the test that runs a
    # second, independent forward, and therefore the only part that can tell
    # a correct capture from a well-transported wrong quantity.
    hf_reference: HFReferenceResult | None = None
    reference_enabled = os.environ.get("SPEEDLM_E2E_HF_REFERENCE", "1") != "0"
    if reference_enabled:
        hf_reference = _run_hf_reference(
            verifier,
            prompt,
            captured_tensors,
            prompt_token_count=prompt_token_count,
            final_layer_idx=final_layer_idx,
        )
        logger.info(
            "HF fp32 reference verdict=%s device=%s layers=%s",
            hf_reference.verdict,
            hf_reference.device,
            [
                (layer.aux_layer_idx, layer.mean_rel_error, layer.tolerance)
                for layer in hf_reference.layers
            ],
        )

    # Phase 6: Build verdict.
    #
    # This engine ran with --no-enable-prefix-caching, so cache_hit is False by
    # construction rather than by measurement -- and it is verified as such:
    # every prompt row must have produced an activation row.  The measured
    # cache-hit case lives in test_prefix_cache_coverage.
    captured_rows = _rows_per_layer(aligned_captured)
    assert captured_rows == prompt_token_count, (
        f"prefix caching is disabled on this engine, so all "
        f"{prompt_token_count} prompt positions must have been forwarded, but "
        f"only {captured_rows} rows were captured per layer"
    )
    prefix_cache = PrefixCacheResult(
        prompt_token_count=prompt_token_count,
        captured_rows_per_layer=captured_rows,
        captured_layer_count=len(aligned_captured),
        cache_hit=False,
        rows_missing=prompt_token_count - captured_rows,
    )

    result = build_result(
        aligned_captured,
        aligned_offline,
        captured_final_pre_norm=captured_final,
        offline_final_pre_norm=offline_final,
        prefix_cache=prefix_cache,
        hf_reference=hf_reference,
    )

    result_path = artifact_dir / "result.json"
    result.write_json(result_path)
    logger.info("Result written to %s — verdict: %s", result_path, result.verdict)

    # By default, a FAIL verdict fails the test so that regressions are caught.
    # Set SPEEDLM_E2E_STRICT_VERDICT=0 to skip this assertion for exploratory runs.
    strict = os.environ.get("SPEEDLM_E2E_STRICT_VERDICT", "1") != "0"
    if strict:
        # A skipped reference leg must never read as a pass.  ``build_result``
        # deliberately does not fail on ``hf_reference is None`` (that would
        # flip the verdict for every pre-existing caller), so the harness that
        # asked for the leg is the one that insists it ran.
        assert not reference_enabled or hf_reference is not None, (
            "SPEEDLM_E2E_HF_REFERENCE is enabled but no reference result was "
            "produced; the run proves only that one tensor round-trips two "
            "ways, not that it is the right tensor"
        )
        assert result.verdict == "PASS", (
            f"Activation capture comparison failed: verdict={result.verdict}, "
            f"rel_error_trend={result.rel_error_trend}, "
            f"hf_reference={(hf_reference.verdict if hf_reference else 'not run')}. "
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
      ``rows x layers`` sum, so it is comparable to ``prompt_token_count``.
    * ``rows_missing`` is the difference between them.

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
            "--enforce-eager",
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
        result = PrefixCacheResult(
            prompt_token_count=prompt_token_count,
            captured_rows_per_layer=second_rows,
            captured_layer_count=len(second_captured),
            cache_hit=cache_hit,
            rows_missing=prompt_token_count - second_rows,
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
                "rows_missing": result.rows_missing,
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
        assert result.rows_missing > 0, (
            f"captured {second_rows} rows for a {prompt_token_count}-token "
            f"prompt on a cache hit, i.e. nothing is missing -- which "
            f"contradicts the measured cache hit.  Result at {result_path}"
        )

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()