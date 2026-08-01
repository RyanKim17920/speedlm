"""GPU E2E: draft hot-swap against a live vLLM engine.

This is the first test that exercises ``CombinedWorkerExtension`` on real
hardware.  Unit tests can prove the torch-free helpers
(``resolve_safetensors_shards``, ``read_draft_config``) but they cannot prove
any of the things that actually broke: that vLLM merges the two mixins into
one worker, that ``_get_drafter_model`` resolves a real drafter through
``model_runner.drafter.model``, that the layerwise reload leaves nothing on
the meta device, or that the engine still decodes speculatively afterwards.

Everything runs against **one** engine process, in order, so the GPU is paid
for once:

1. Engine starts with ``CombinedWorkerExtension`` -- the injection log line
   must name both a swap method and a capture method (proves the inheritance
   merge took effect, not just that the module imported).
2. ``draft_info`` returns non-empty parameter shapes, and nothing is on the
   meta device *before* any swap -- so a clean post-swap reading is not just
   a broken probe.
3. NULL SWAP: swap in the weights that are already loaded.  Reports
   ``swapped=True`` with a parameter count that matches the candidate payload,
   nothing is on the meta device afterwards -- drafter *or* target -- and a
   greedy canary completion afterwards is token-identical to before.
4. Speculative-decode counters advance after the swap.
5. INCOMPATIBLE swap (doctored ``hidden_size``) is rejected with the running
   drafter untouched, and a canary still succeeds.
6. Activation capture still works on the same engine after a swap.

**Deliberately out of scope.**  This test does not swap to a genuinely
different *trained* drafter and compare its logits against a from-scratch
engine restarted on that drafter.  That is the assertion that would prove the
swapped weights are the ones being proposed from -- it needs a second trained
draft artifact, which does not exist yet.  Until then the null swap isolates
the reload machinery: any behavioural change it produces is a bug, because the
weights did not change.

Required environment:

* ``SPEEDLM_E2E_DRAFT_HOT_SWAP=1`` -- opt-in GPU gate
* ``SPEEDLM_E2E_VERIFIER_MODEL`` -- HuggingFace model ID or local path
* ``SPEEDLM_E2E_DRAFTER_MODEL`` -- HuggingFace model ID or local path
* ``SPEEDLM_E2E_ARTIFACT_DIR`` -- durable artifact root (must be on /data)

Optional environment:

* ``SPEEDLM_E2E_DRAFTER_DIR`` -- resolved snapshot directory of the drafter.
  Only needed when ``SPEEDLM_E2E_DRAFTER_MODEL`` is a repo id that cannot be
  resolved from ``HF_HOME``.
* ``SPEEDLM_E2E_VLLM_PYTHON`` -- path to vLLM python (default: vLLM venv)
* ``SPEEDLM_E2E_READY_TIMEOUT`` -- engine readiness cap in seconds (default 900)
* ``SPEEDLM_E2E_PORT`` -- vLLM serve port (default: auto-assigned free port)
* ``SPEEDLM_E2E_PROMPT`` -- override the canary prompt

This module deliberately imports **no torch and no safetensors**.  The safetensors
header is a length-prefixed JSON blob, which is all this test needs to read, and
a module-level ``importorskip`` would make the whole file collect as zero tests
on any interpreter without torch -- a silent pass, not a skip.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from speedlm.gate.metrics import parse_metrics

pytestmark = pytest.mark.e2e

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM_PYTHON = Path(
    os.environ.get("SPEEDLM_E2E_VLLM_PYTHON", str(VLLM_VENV / "bin" / "python"))
)

#: vLLM resolves ``--worker-extension-cls`` with ``resolve_obj_by_qualname``,
#: which splits on the LAST dot.  A ``pkg.mod:Class`` spec does not resolve.
WORKER_EXTENSION_CLS = "speedlm.gateway.draft_swap.CombinedWorkerExtension"

#: Methods that must appear in vLLM's injection log line.  One comes from each
#: mixin, so seeing both proves the MRO merge -- seeing only one would mean
#: vLLM injected a single base and the composition silently lost a feature.
REQUIRED_SWAP_CALL = "hot_swap_draft"
REQUIRED_CAPTURE_CALL = "activate_capture"

#: Substrings of the drafter submodules that the EAGLE proposer shares with the
#: verifier by object identity.  ``_drop_target_owned_weights`` filters
#: candidate tensors by exactly this rule, so the expected
#: ``parameters_loaded`` is the candidate tensor count minus these.
TARGET_OWNED_LEAVES = ("embed_tokens", "lm_head")

DEFAULT_PROMPT = "Explain in one sentence why the sky appears blue."

#: Enough tokens that speculative decoding gets several proposal rounds, but
#: short enough that a greedy canary stays cheap.
CANARY_MAX_TOKENS = 48


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _require_environment() -> tuple[str, str, Path]:
    """Check the GPU gate and return (verifier, drafter, artifact_root)."""
    if os.environ.get("SPEEDLM_E2E_DRAFT_HOT_SWAP") != "1":
        pytest.skip("set SPEEDLM_E2E_DRAFT_HOT_SWAP=1 in an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), (
        "draft hot-swap E2E must run inside a SLURM allocation"
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


def _resolve_drafter_dir(drafter: str) -> Path:
    """Resolve the drafter to an on-disk directory holding its checkpoint.

    ``hot_swap_draft`` takes a **directory**, not a repo id, so a repo id has
    to be turned into its HF snapshot path here.  ``HF_HUB_OFFLINE=1`` means
    there is no download fallback: if the snapshot is not cached the test must
    say so rather than hang.
    """
    override = os.environ.get("SPEEDLM_E2E_DRAFTER_DIR")
    if override:
        path = Path(override)
        assert (path / "config.json").is_file(), (
            f"SPEEDLM_E2E_DRAFTER_DIR has no config.json: {path}"
        )
        return path

    direct = Path(drafter)
    if (direct / "config.json").is_file():
        return direct

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "models--" + drafter.replace("/", "--")
    snapshots = sorted((hf_home / "hub" / slug / "snapshots").glob("*"))
    usable = [s for s in snapshots if (s / "config.json").is_file()]
    assert usable, (
        f"cannot resolve drafter {drafter!r} to a cached snapshot under "
        f"{hf_home / 'hub' / slug}; set SPEEDLM_E2E_DRAFTER_DIR explicitly"
    )
    return usable[-1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ready_timeout() -> float:
    raw = os.environ.get("SPEEDLM_E2E_READY_TIMEOUT", "900")
    try:
        return float(raw)
    except ValueError as exc:
        raise AssertionError("SPEEDLM_E2E_READY_TIMEOUT must be a number") from exc


def _create_artifact_dir(root: Path) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = root / f"draft-hot-swap-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vllm_env() -> dict[str, str]:
    """Environment for the engine, with the dev RPC endpoint enabled.

    vLLM only mounts ``/collective_rpc`` when ``VLLM_SERVER_DEV_MODE`` is
    truthy; without it every RPC in this test 404s.
    """
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


def _read_log_tail(log_path: Path, lines: int = 100) -> str:
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(raw.splitlines()[-lines:])
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
                raise AssertionError(
                    f"vLLM exited before readiness with code {rc}\n"
                    f"--- vLLM log (last 100 lines) ---\n"
                    f"{_read_log_tail(log_path)}\n"
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
    raise AssertionError(
        f"vLLM did not become ready within {timeout}s; {last_error}\n"
        f"--- vLLM log (last 100 lines) ---\n"
        f"{_read_log_tail(log_path)}\n"
        f"--- end of vLLM log ---"
    )


def _get_served_model_id(url: str) -> str:
    """Return the id vLLM actually serves the verifier under.

    vLLM registers the model by its resolved snapshot path, not the friendly
    repo id.  Sending the repo id gets a 404, not a 400, so the id is always
    read back from ``/v1/models`` rather than assumed.
    """
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        resp = client.get(f"{url}/v1/models")
        resp.raise_for_status()
        model_ids = [m["id"] for m in resp.json().get("data", [])]
    if not model_ids:
        raise AssertionError(
            "/v1/models returned no served models — the engine may not have "
            "finished loading"
        )
    return model_ids[0]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _collective_rpc(port: int, method: str, *args: object) -> list[Any]:
    """Call *method* on every worker and return the per-worker results.

    ``/collective_rpc`` accepts **string** args only (the endpoint documents
    this: "only serialized string args/kwargs are passed"), so every argument
    is stringified here.  Worker return values come back as
    ``{"results": [...]}``; dicts and lists survive as JSON, anything else is
    stringified by vLLM.
    """
    url = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=600.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/collective_rpc",
            json={"method": method, "args": [str(a) for a in args]},
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"collective_rpc {method} failed: {resp.status_code} {resp.text}"
        )
    body = resp.json() if resp.content else {}
    results = body.get("results", []) if isinstance(body, dict) else []
    for i, result in enumerate(results):
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"collective_rpc {method} worker {i} error: {result['error']}"
            )
    return list(results)


def _collective_rpc_one(port: int, method: str, *args: object) -> dict[str, Any]:
    """Call *method* and return the single worker's dict result.

    The test runs tensor-parallel size 1, so more than one result means the
    topology is not what the assertions assume and the test should say so
    rather than silently read worker 0.
    """
    results = _collective_rpc(port, method, *args)
    assert len(results) == 1, (
        f"{method} returned {len(results)} worker results; this test assumes "
        f"tensor-parallel size 1"
    )
    result = results[0]
    assert isinstance(result, dict), (
        f"{method} returned {type(result).__name__}, expected a dict: {result!r}"
    )
    return result


def _collective_rpc_expect_failure(port: int, method: str, *args: object) -> str:
    """Call *method* expecting it to raise worker-side; return the response body.

    A worker-side exception propagates out of ``engine_client.collective_rpc``
    and FastAPI turns it into a non-2xx response.  The body may or may not
    carry the original message depending on how vLLM wraps it, so the caller
    corroborates against the engine log.
    """
    url = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=600.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/collective_rpc",
            json={"method": method, "args": [str(a) for a in args]},
        )
    assert resp.status_code != 200, (
        f"{method} unexpectedly SUCCEEDED with {args!r}: {resp.text[:500]}"
    )
    return resp.text


def _canary(url: str, prompt: str, *, served_model_id: str) -> tuple[str, list[str]]:
    """Run one greedy completion and return (text, token strings).

    Temperature 0 with a fixed seed and ``top_p=1`` makes the decode
    deterministic, which is what makes "identical before and after the swap" a
    meaningful claim rather than a coincidence.  ``logprobs`` is requested so
    the comparison can be made at the *token* level -- two different token
    sequences can detokenize to the same string, and that difference is exactly
    the kind of drift a broken reload would produce.
    """
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": served_model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": CANARY_MAX_TOKENS,
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
                "logprobs": True,
                "top_logprobs": 0,
            },
        )
        if resp.status_code == 404:
            with httpx.Client(timeout=10.0, trust_env=False) as probe:
                served = [
                    m["id"] for m in probe.get(f"{url}/v1/models").json().get("data", [])
                ]
            raise AssertionError(
                f"404 from /v1/chat/completions — model id mismatch. "
                f"Sent model={served_model_id!r}, served model ids={served}"
            ) from None
        resp.raise_for_status()
        choice = resp.json()["choices"][0]

    text = choice["message"]["content"] or ""
    logprobs = choice.get("logprobs")
    assert logprobs and logprobs.get("content"), (
        "canary response carried no logprobs.content — token-level identity "
        "cannot be checked; the deployment may not support logprobs on "
        "/v1/chat/completions"
    )
    tokens = [entry["token"] for entry in logprobs["content"]]
    return text, tokens


def _metrics(url: str) -> Any:
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        resp = client.get(f"{url}/metrics")
        resp.raise_for_status()
    return parse_metrics(resp.text)


# ---------------------------------------------------------------------------
# safetensors header reading (no safetensors dependency)
# ---------------------------------------------------------------------------


def _safetensors_tensor_names(path: Path) -> list[str]:
    """Return the tensor names in a safetensors file, reading only its header.

    The format is a little-endian ``uint64`` header length followed by that
    many bytes of JSON.  Reading it directly keeps this module importable in
    the project venv (which has no torch and no safetensors), so the file
    collects as a real, skippable test instead of collapsing to zero tests
    behind a module-level ``importorskip``.
    """
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return [name for name in header if name != "__metadata__"]


def _candidate_tensor_names(directory: Path) -> list[str]:
    """Names of every tensor the loader would read from *directory*.

    Mirrors ``resolve_safetensors_shards``: natural-sorted ``*.safetensors``,
    filtered through the shard index when one is present (a directory can hold
    both sharded and consolidated copies, and counting both double-counts).
    """
    shards = sorted(directory.glob("*.safetensors"))
    assert shards, f"no *.safetensors under {directory}"

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
            "weight_map", {}
        )
        indexed = {str(v) for v in weight_map.values()}
        if indexed:
            shards = [s for s in shards if s.name in indexed]

    names: list[str] = []
    for shard in shards:
        names.extend(_safetensors_tensor_names(shard))
    return names


def _expected_parameters_loaded(directory: Path) -> int:
    """Payload size ``_apply_weights`` should report for *directory*.

    Deliberately NOT the drafter's ``named_parameters()`` count: vLLM fuses
    ``q/k/v_proj`` into ``qkv_proj`` and ``gate/up_proj`` into
    ``gate_up_proj``, so a byte-identical checkpoint always has more tensors
    than the live model has parameters.  What the swap reports is the number of
    candidate tensors handed to the loader after the verifier-owned
    embedding/head tensors are dropped, and that is what is asserted.
    """
    names = _candidate_tensor_names(directory)
    return sum(
        1 for name in names if not any(leaf in name for leaf in TARGET_OWNED_LEAVES)
    )


# ---------------------------------------------------------------------------
# Incompatible candidate construction
# ---------------------------------------------------------------------------


def _write_incompatible_draft(dest: Path, running: dict[str, Any]) -> Path:
    """Build a cheap draft directory that must be rejected on shape grounds.

    Constructed from the *running* drafter's own config summary with
    ``hidden_size`` perturbed, so the only reason it can be rejected is the
    perturbation -- nothing else differs.  No second model is downloaded and no
    real weights are copied: ``hot_swap_draft`` loads the candidate tensors
    before validating, so the directory needs *a* safetensors file, but a
    one-element placeholder is enough to reach (and fail) the shape check.
    """
    dest.mkdir(parents=True, exist_ok=True)

    hidden = running.get("hidden_size")
    assert isinstance(hidden, int) and hidden > 0, (
        f"running drafter reported no usable hidden_size: {running!r}"
    )
    config = {
        "vocab_size": running.get("vocab_size"),
        "hidden_size": hidden + 64,
        "num_hidden_layers": running.get("num_hidden_layers"),
        "draft_vocab_size": running.get("draft_vocab_size"),
        "dtype": running.get("dtype"),
    }
    (dest / "config.json").write_text(
        json.dumps({k: v for k, v in config.items() if v is not None}, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_placeholder_safetensors(dest / "model.safetensors")
    return dest


def _write_placeholder_safetensors(path: Path) -> None:
    """Write a minimal valid one-tensor safetensors file.

    Emitted by hand rather than via ``safetensors.torch.save_file`` so this
    module keeps no torch/safetensors import.
    """
    header = {
        "placeholder": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    blob = json.dumps(header).encode("utf-8")
    # The header must be 8-byte aligned; pad with spaces, which JSON ignores.
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\x00\x00\x00\x00")


# ---------------------------------------------------------------------------
# Injection log line
# ---------------------------------------------------------------------------


def _injected_rpc_calls(log_path: Path) -> list[str]:
    """Parse the method list out of vLLM's extension-injection log line.

    ``worker_base.py`` logs ``"Injected %s into %s for extended collective_rpc
    calls %s"`` with the list of callable attributes it found on the extension
    *after* resolving it.  That list is the ground truth for what the MRO merge
    produced -- an import-time check of the class would prove nothing about
    what vLLM actually spliced into the worker.
    """
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        if "for extended collective_rpc calls" not in line:
            continue
        _, _, tail = line.partition("for extended collective_rpc calls")
        start = tail.find("[")
        end = tail.rfind("]")
        if start == -1 or end == -1:
            continue
        return [
            item.strip().strip("'\"")
            for item in tail[start + 1 : end].split(",")
            if item.strip()
        ]
    raise AssertionError(
        "vLLM never logged the worker-extension injection line; "
        "--worker-extension-cls was not applied.\n"
        f"--- vLLM log (last 100 lines) ---\n{_read_log_tail(log_path)}"
    )


# ---------------------------------------------------------------------------
# Capture artifacts
# ---------------------------------------------------------------------------


def _assert_capture_artifacts(capture_dir: Path) -> dict[str, Any]:
    """Assert a flushed capture is present, non-empty and layer-id keyed."""
    tensor_path = capture_dir / "captured.safetensors"
    assert tensor_path.is_file(), (
        f"no captured.safetensors under {capture_dir} after flush_capture — "
        f"activation capture stopped working once draft swap was composed in"
    )
    assert tensor_path.stat().st_size > 0, f"{tensor_path} is empty"

    meta_path = capture_dir / "captured.safetensors.meta.json"
    assert meta_path.is_file(), (
        f"no captured.safetensors.meta.json sidecar under {capture_dir}; "
        f"without it the captured keys cannot be interpreted as layer ids"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    original_aux = meta.get("original_aux_layers") or []
    final_layer_idx = meta.get("final_layer_idx")
    assert original_aux, f"capture metadata reports no aux layers: {meta!r}"

    names = _safetensors_tensor_names(tensor_path)
    assert names, f"{tensor_path} contains no tensors"
    for name in names:
        assert name.startswith("layer_"), (
            f"captured tensor {name!r} is not layer-id keyed; the offline "
            f"comparison path keys strictly on layer_<id>"
        )

    captured_ids = sorted(int(name.split("_", 1)[1]) for name in names)
    extra = [final_layer_idx] if final_layer_idx is not None else []
    expected_ids = sorted(list(original_aux) + extra)
    assert captured_ids == expected_ids, (
        f"captured layer ids {captured_ids} do not match "
        f"original_aux_layers + final_layer_idx {expected_ids}"
    )
    return meta


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_draft_hot_swap_on_live_engine() -> None:
    """Drive a real engine through a null swap, a rejected swap, and a capture.

    One engine, six phases (some split into lettered sub-steps), in order.
    Each phase targets a way the shipped code could fail while every unit
    test stayed green; see the module docstring for the mapping.
    """
    verifier, drafter, artifact_root = _require_environment()
    drafter_dir = _resolve_drafter_dir(drafter)
    prompt = os.environ.get("SPEEDLM_E2E_PROMPT", DEFAULT_PROMPT)
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
    incompatible_dir = artifact_dir / "incompatible-draft"

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
            WORKER_EXTENSION_CLS,
            "--port",
            str(port),
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            "8",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.7",
            # Prefix caching off: a cache hit would let the post-swap canary
            # replay the pre-swap decode without re-running the drafter, which
            # would make token identity vacuous.
            "--no-enable-prefix-caching",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_vllm_env(),
    )

    url = f"http://127.0.0.1:{port}"
    report: dict[str, Any] = {
        "verifier": verifier,
        "drafter": drafter,
        "drafter_dir": str(drafter_dir),
        "worker_extension_cls": WORKER_EXTENSION_CLS,
    }

    try:
        _wait_for_ready(url, vllm_proc, timeout, log_path=vllm_log)
        served_model_id = _get_served_model_id(url)
        report["served_model_id"] = served_model_id

        # -- Phase 1: the composed extension really was injected -------------
        # Failure mode: CombinedWorkerExtension "inherits" from both mixins on
        # paper but vLLM only splices one method set into the worker, so half
        # the RPCs 500 at runtime while every import-time test passes.
        injected = _injected_rpc_calls(vllm_log)
        report["injected_rpc_calls"] = injected
        assert REQUIRED_SWAP_CALL in injected, (
            f"vLLM injected {injected} — {REQUIRED_SWAP_CALL!r} is missing, so "
            f"DraftSwapExtension did not make it into the worker MRO"
        )
        assert REQUIRED_CAPTURE_CALL in injected, (
            f"vLLM injected {injected} — {REQUIRED_CAPTURE_CALL!r} is missing, so "
            f"ActivationCaptureExtension did not make it into the worker MRO"
        )

        # -- Phase 2a: a real drafter is reachable ---------------------------
        # Failure mode: _get_drafter_model walks model_runner.drafter.model and
        # silently returns None (or an unwrapped CUDA-graph wrapper with no
        # parameters), which no unit test can detect because there is no real
        # runner to walk.
        info_before = _collective_rpc_one(port, "draft_info")
        shapes_before = info_before["parameter_shapes"]
        assert info_before["num_parameters"] > 0, (
            f"draft_info reports zero parameters: {info_before!r}"
        )
        assert shapes_before, "draft_info returned an empty parameter_shapes map"
        assert all(shape for shape in shapes_before.values()), (
            f"draft_info returned rank-0/empty shapes: "
            f"{[n for n, s in shapes_before.items() if not s]}"
        )
        report["draft_info_before"] = {
            "num_parameters": info_before["num_parameters"],
            "draft_config": info_before.get("draft_config"),
            "quantization": info_before.get("quantization"),
        }

        # -- Phase 2b: nothing is on meta before any swap --------------------
        # Baseline meta-device walk: proves the probe reports "clean" on a
        # healthy engine, so a clean post-swap reading is not just a broken probe.
        meta_before = _collective_rpc_one(port, "draft_materialization_report")
        assert meta_before["stranded_total"] == 0, (
            f"tensors were already on the meta device BEFORE any swap: "
            f"{meta_before['stranded']}"
        )

        # -- Phase 3a: canary before the swap --------------------------------
        text_before, tokens_before = _canary(
            url, prompt, served_model_id=served_model_id
        )
        assert tokens_before, "pre-swap canary generated no tokens"
        metrics_before = _metrics(url)

        # -- Phase 3b: NULL SWAP ---------------------------------------------
        # Failure mode: the reload machinery itself is broken.  Swapping in the
        # weights that are already loaded means any behavioural difference is
        # attributable to the machinery and nothing else.
        expected_loaded = _expected_parameters_loaded(drafter_dir)
        swap = _collective_rpc_one(port, "hot_swap_draft", str(drafter_dir))
        assert swap.get("swapped") is True, f"null swap did not report success: {swap!r}"
        assert swap.get("parameters_loaded") == expected_loaded, (
            f"null swap loaded {swap.get('parameters_loaded')} parameters but the "
            f"candidate payload under {drafter_dir} has {expected_loaded} tensors "
            f"after dropping verifier-owned {list(TARGET_OWNED_LEAVES)}"
        )
        report["null_swap"] = swap

        # -- Phase 3c: nothing stranded on meta after the swap ---------------
        # Failure mode: the layerwise reload walked the drafter's *shared*
        # submodules and left the running verifier's embedding on the meta
        # device — the model then answers, wrongly, until it crashes.
        meta_after = _collective_rpc_one(port, "draft_materialization_report")
        assert meta_after["stranded_total"] == 0, (
            f"draft hot-swap stranded tensors on the meta device: "
            f"{meta_after['stranded']}"
        )
        assert not meta_after["stranded"]["target"], (
            f"the VERIFIER was left on the meta device by a draft-only swap: "
            f"{meta_after['stranded']['target']}"
        )

        # -- Phase 3d: canary after the swap must be token-identical ---------
        text_after, tokens_after = _canary(url, prompt, served_model_id=served_model_id)
        assert tokens_after == tokens_before, (
            f"null swap changed the greedy decode.  The weights did not change, "
            f"so this is the reload machinery corrupting the drafter.\n"
            f"before ({len(tokens_before)} tokens): {tokens_before}\n"
            f"after  ({len(tokens_after)} tokens): {tokens_after}"
        )
        assert text_after == text_before, (
            f"null swap changed the decoded text despite identical tokens:\n"
            f"before: {text_before!r}\nafter:  {text_after!r}"
        )
        report["canary_tokens"] = tokens_before

        # -- Phase 4: the drafter still participates -------------------------
        # Failure mode: the swap "succeeds" but the drafter proposes nothing,
        # so the engine silently falls back to plain autoregressive decoding —
        # correct output, zero speedup, no error anywhere.
        metrics_after = _metrics(url)
        assert metrics_after.has_draft_counters, (
            "/metrics exposes no vllm:spec_decode_* counters after the swap — "
            "speculative decoding is not running at all"
        )
        drafted_delta = metrics_after.drafted_tokens - metrics_before.drafted_tokens
        drafts_delta = metrics_after.num_drafts - metrics_before.num_drafts
        assert drafted_delta > 0 and drafts_delta > 0, (
            f"no speculative activity during the post-swap canary "
            f"(drafted_tokens delta={drafted_delta}, num_drafts delta={drafts_delta}); "
            f"the drafter failed closed after the swap"
        )
        report["post_swap_spec_metrics"] = {
            "drafted_tokens_delta": drafted_delta,
            "num_drafts_delta": drafts_delta,
            "accepted_tokens_delta": (
                metrics_after.accepted_tokens - metrics_before.accepted_tokens
            ),
        }

        # -- Phase 5: an incompatible candidate is rejected, cleanly ---------
        # Failure mode: validation runs after the mutation (or not at all), so
        # a bad candidate half-loads and bricks a live engine.
        _write_incompatible_draft(incompatible_dir, info_before["draft_config"])
        body = _collective_rpc_expect_failure(
            port, "hot_swap_draft", str(incompatible_dir)
        )
        report["incompatible_rejection_body"] = body[:2000]
        log_text = vllm_log.read_text(encoding="utf-8", errors="replace")
        assert "hidden_size mismatch" in body or "hidden_size mismatch" in log_text, (
            "the incompatible candidate was rejected, but not for the doctored "
            f"hidden_size — response body: {body[:500]}"
        )

        info_rejected = _collective_rpc_one(port, "draft_info")
        assert info_rejected["parameter_shapes"] == shapes_before, (
            "the rejected candidate mutated the running drafter's parameter "
            "shapes; validation did not happen before mutation"
        )
        meta_rejected = _collective_rpc_one(port, "draft_materialization_report")
        assert meta_rejected["stranded_total"] == 0, (
            f"the rejected candidate left tensors on the meta device: "
            f"{meta_rejected['stranded']}"
        )
        text_rejected, tokens_rejected = _canary(
            url, prompt, served_model_id=served_model_id
        )
        assert tokens_rejected == tokens_before, (
            f"the engine no longer decodes identically after a REJECTED swap; "
            f"the rollback story is broken.\n"
            f"before: {tokens_before}\nafter rejection: {tokens_rejected}"
        )

        # -- Phase 6: activation capture still works post-swap ---------------
        # Failure mode: composing the two mixins regresses the shipped capture
        # feature (shared lazy-init state, or the swap invalidating the hooks).
        _collective_rpc(port, "activate_capture", str(capture_dir))
        _canary(url, prompt, served_model_id=served_model_id)
        # Let the hook's buffer drain before flushing.
        time.sleep(0.5)
        _collective_rpc(port, "flush_capture")
        capture_meta = _assert_capture_artifacts(capture_dir)
        report["capture_meta"] = capture_meta

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()
        result_path = artifact_dir / "result.json"
        result_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        logger.info("Draft hot-swap result written to %s", result_path)
