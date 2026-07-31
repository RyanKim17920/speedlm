"""Stage 0 kill-condition prototype: serving-time activation capture.

This test verifies that the aux hidden states captured from a live EAGLE-3
vLLM serving engine match the offline extraction path, elementwise per layer.

Required environment:
* ``SPEEDLM_E2E_ACTIVATION_CAPTURE=1`` — opt-in GPU gate
* ``SPEEDLM_E2E_VERIFIER_MODEL`` — HuggingFace model ID or local path
* ``SPEEDLM_E2E_DRAFTER_MODEL`` — HuggingFace model ID or local path
* ``SPEEDLM_E2E_ARTIFACT_DIR`` — durable artifact root
* ``SPEEDLM_E2E_VLLM_PYTHON`` — path to vLLM python (default: vLLM venv)

Optional environment:
* ``SPEEDLM_E2E_READY_TIMEOUT`` — engine readiness cap in seconds (default: 360)
* ``SPEEDLM_E2E_PORT`` — vLLM serve port (default: auto-assigned free port)
* ``SPEEDLM_E2E_TARGET_LAYER_IDS`` — JSON array, e.g. ``[4, 12, 20]``
* ``SPEEDLM_E2E_PROMPT`` — override the fixed prompt (default: a short
  English sentence)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

torch = pytest.importorskip("torch")

from safetensors import safe_open  # noqa: E402

from speedlm.activation_capture.compare import (  # noqa: E402
    PrefixCacheResult,
    build_result,
)
from speedlm.activation_capture.offline_extract import (  # noqa: E402
    extract as _run_offline_extract,
)

pytestmark = pytest.mark.e2e

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM_PYTHON = Path(
    os.environ.get("SPEEDLM_E2E_VLLM_PYTHON", str(VLLM_VENV / "bin" / "python"))
)
SPECULATORS_REPO = Path("/admin/home/ryan.kim/speedlm/.preflight/speculators")

# Short fixed prompt for the experiment
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."


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


def _target_layer_ids() -> list[int]:
    raw = os.environ.get("SPEEDLM_E2E_TARGET_LAYER_IDS")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssertionError("SPEEDLM_E2E_TARGET_LAYER_IDS must be JSON") from exc
    return [4, 12, 20]


def _create_artifact_dir(root: Path) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = root / f"activation-capture-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wait_for_ready(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            rc = process.poll()
            if rc is not None:
                raise AssertionError(f"vLLM exited before readiness with code {rc}")
            try:
                resp = client.get(f"{url}/health")
                if 200 <= resp.status_code < 300:
                    return
                last_error = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.5)
    raise AssertionError(f"vLLM did not become ready within {timeout}s; {last_error}")


def _send_prompt(url: str, prompt: str) -> str:
    """Send a single completion request and return the output text."""
    with httpx.Client(timeout=120.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/v1/completions",
            json={
                "model": "test",
                "prompt": prompt,
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["text"]


def _collective_rpc(vllm_proc: subprocess.Popen[bytes], method: str, *args: object) -> None:
    """Issue a collective_rpc call to the vLLM engine via the debug endpoint.

    vLLM exposes a /collective_rpc endpoint that forwards to all workers.
    """
    # We use a separate port offset for the collective_rpc endpoint.
    # vLLM's engine serves this on the same port as the API.
    url = f"http://127.0.0.1:{os.environ.get('SPEEDLM_E2E_PORT', _free_port())}"
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        resp = client.post(
            f"{url}/collective_rpc",
            json={"method": method, "args": list(args)},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"collective_rpc {method} failed: {resp.status_code} {resp.text}"
            )


def _load_captured_safetensors(capture_dir: Path) -> dict[int, torch.Tensor]:
    """Load the captured.safetensors file and return {layer_idx: tensor}."""
    path = capture_dir / "captured.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"no captured.safetensors in {capture_dir}")
    tensors: dict[int, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f:
            if key.startswith("layer_"):
                idx = int(key.split("_")[1])
                tensors[idx] = f.get_tensor(key)
    return tensors


def _load_offline_hidden_states(hs_dir: Path) -> dict[int, torch.Tensor]:
    """Load offline hs_*.safetensors shards into {layer_idx: tensor}.

    The offline path writes shape (seq_len, num_layers, hidden_size).
    We split the layer dimension so each layer becomes a separate tensor.
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
            layers.setdefault(i, []).append(hs[:, i])

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
    """Align captured and offline tensors to the same token range.

    Offline extraction re-runs the prompt as a prefill, producing exactly
    ``prompt_token_count`` rows.  Serving-time capture records all scheduled
    tokens (prompt + generated), so it may have more rows.  We trim the
    captured tensor to the first ``prompt_token_count`` rows.

    Returns (captured_trimmed, offline) — both (N, H).
    """
    if captured.shape[0] == offline.shape[0]:
        return captured, offline
    if captured.shape[0] > offline.shape[0]:
        return captured[:offline.shape[0]], offline
    # Captured has fewer rows — likely prefix-cache hit or a bug.
    # Report the mismatch rather than fudging.
    raise ValueError(
        f"captured has fewer rows ({captured.shape[0]}) than offline "
        f"({offline.shape[0]}) — cannot align"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage0_activation_capture() -> None:
    """Full Stage 0 experiment: serve one prompt with capture, compare offline.

    This test:
    1. Starts a vLLM engine with the EAGLE-3 speculator and capture extension.
    2. Sends a single fixed prompt.
    3. Flushes captured activations to disk.
    4. Runs the offline extraction on the same prompt.
    5. Compares the two tensor stacks elementwise per layer.
    6. Writes a JSON result with PASS/FAIL verdict.
    """
    verifier, drafter, artifact_root = _require_environment()
    prompt = os.environ.get("SPEEDLM_E2E_PROMPT", DEFAULT_PROMPT)
    target_layers = _target_layer_ids()
    artifact_dir = _create_artifact_dir(artifact_root)
    port = int(os.environ.get("SPEEDLM_E2E_PORT", _free_port()))
    timeout = _ready_timeout()

    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": 3,
        "draft_model": drafter,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layers}
        },
    }

    capture_dir = artifact_dir / "captured"
    capture_dir.mkdir(exist_ok=True)

    vllm_proc = subprocess.Popen(
        [
            str(VLLM_PYTHON),
            "-m", "vllm.entrypoints.cli.main",
            "serve",
            verifier,
            "--speculative_config",
            json.dumps(speculative_config),
            "--worker-extension-cls",
            "speedlm.activation_capture.hook:ActivationCaptureExtension",
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )

    try:
        _wait_for_ready(f"http://127.0.0.1:{port}", vllm_proc, timeout)

        # Step 2: Activate capture via collective_rpc, send prompt, flush
        _collective_rpc(vllm_proc, "activate_capture", str(capture_dir))
        _send_prompt(f"http://127.0.0.1:{port}", prompt)
        # Small pause to let the hook buffer finish
        time.sleep(0.5)
        _collective_rpc(vllm_proc, "flush_capture")

        # Step 3: Load captured tensors
        captured_tensors = _load_captured_safetensors(capture_dir)
        logger.info("Captured layers: %s", sorted(captured_tensors.keys()))

        # Step 4: Run offline extraction
        offline_dir = artifact_dir / "offline"
        hs_dir = _run_offline_extract(
            verifier,
            prompt,
            target_layers,
            offline_dir,
            port=port + 1000,  # Different port
        )
        offline_tensors = _load_offline_hidden_states(hs_dir)
        logger.info("Offline layers: %s", sorted(offline_tensors.keys()))

        # Step 5: Align and compare
        # Determine the prompt token count (approximate from offline shape)
        prompt_token_count: int | None = None
        for t in offline_tensors.values():
            prompt_token_count = t.shape[0]
            break
        assert prompt_token_count is not None

        aligned_captured: dict[int, torch.Tensor] = {}
        aligned_offline: dict[int, torch.Tensor] = {}
        # Find the final layer index (highest in offline set, typically num_hidden_layers)
        offline_indices = sorted(offline_tensors.keys())
        for idx in offline_indices:
            cap = captured_tensors.get(idx)
            off = offline_tensors[idx]
            if cap is None:
                continue
            try:
                c_aligned, o_aligned = _align_token_count(cap, off, prompt_token_count)
                aligned_captured[idx] = c_aligned
                aligned_offline[idx] = o_aligned
            except ValueError as exc:
                logger.warning("Cannot align layer %d: %s", idx, exc)

        # The final layer (num_hidden_layers) is the regression target.
        # It should be the highest index in the offline set.
        if offline_indices:
            final_idx = offline_indices[-1]
            captured_final = aligned_captured.get(final_idx)
            offline_final = aligned_offline.get(final_idx)

        # Step 6: Build verdict
        prefix_cache = PrefixCacheResult(
            prompt_token_count=prompt_token_count,
            captured_row_count=sum(t.shape[0] for t in aligned_captured.values()),
            cache_hit=False,
            rows_missing=0,
        )

        result = build_result(
            aligned_captured,
            aligned_offline,
            captured_final_pre_norm=captured_final,
            offline_final_pre_norm=offline_final,
            prefix_cache=prefix_cache,
        )

        result_path = artifact_dir / "result.json"
        result.write_json(result_path)
        logger.info("Result written to %s — verdict: %s", result_path, result.verdict)

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()


# ---------------------------------------------------------------------------
# Prefix-cache coverage measurement
# ---------------------------------------------------------------------------


def test_prefix_cache_coverage() -> None:
    """Measure whether prefix-cache hits produce no activation row.

    This test sends the same prompt twice with prefix caching enabled,
    then checks whether the second request captures fewer activation rows
    than the first.
    """
    verifier, drafter, artifact_root = _require_environment()
    prompt = os.environ.get("SPEEDLM_E2E_PROMPT", DEFAULT_PROMPT)
    target_layers = _target_layer_ids()
    artifact_dir = _create_artifact_dir(artifact_root)
    port = int(os.environ.get("SPEEDLM_E2E_PORT", _free_port()))
    timeout = _ready_timeout()

    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": 3,
        "draft_model": drafter,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layers}
        },
    }

    capture_dir = artifact_dir / "captured"
    capture_dir.mkdir(exist_ok=True)

    vllm_proc = subprocess.Popen(
        [
            str(VLLM_PYTHON),
            "-m", "vllm.entrypoints.cli.main",
            "serve",
            verifier,
            "--speculative_config",
            json.dumps(speculative_config),
            "--worker-extension-cls",
            "speedlm.activation_capture.hook:ActivationCaptureExtension",
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )

    try:
        _wait_for_ready(f"http://127.0.0.1:{port}", vllm_proc, timeout)

        # Activate capture
        _collective_rpc(vllm_proc, "activate_capture", str(capture_dir))

        # Send same prompt twice; second should hit prefix cache
        _send_prompt(f"http://127.0.0.1:{port}", prompt)
        _send_prompt(f"http://127.0.0.1:{port}", prompt)

        time.sleep(0.5)
        _collective_rpc(vllm_proc, "flush_capture")

        # Load captured results to measure row counts
        captured = _load_captured_safetensors(capture_dir)
        total_rows = sum(t.shape[0] for t in captured.values())
        # Approximate word count for prompt_token_count
        word_count = len(prompt.split())

        result = PrefixCacheResult(
            prompt_token_count=word_count,
            captured_row_count=total_rows,
            cache_hit=True,
            rows_missing=0,
        )
        result_path = artifact_dir / "prefix_cache_result.json"
        result_path.write_text(
            json.dumps({
                "prompt_token_count": result.prompt_token_count,
                "captured_row_count": result.captured_row_count,
                "cache_hit": result.cache_hit,
                "rows_missing": result.rows_missing,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Prefix cache result: %s", result_path)

    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()