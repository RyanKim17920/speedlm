"""Offline hidden-state extraction for Stage 0 verification.

Drives the Speculators ``launch_vllm.py`` + ``data_generation_offline.py``
pipeline on a single prompt so the results can be compared against the
serving-time capture.

This module keeps torch/vllm imports lazy so it can be imported in the
project venv (which lacks torch).  All heavy lifting happens in subprocesses
that run under the vLLM venv.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

#: Default path to the vLLM venv Python.
_DEFAULT_VLLM_PYTHON = Path(
    os.environ.get(
        "SPEEDLM_E2E_VLLM_PYTHON",
        "/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python",
    )
)

#: Default path to the Speculators checkout.
_DEFAULT_SPECULATORS_REPO = Path(
    os.environ.get(
        "SPEEDLM_SPECULATORS_REPO",
        "/admin/home/ryan.kim/speedlm/.preflight/speculators",
    )
)

#: Default GPU memory utilization fraction for the offline extraction engine.
#: This fraction is applied to each device's *total* memory so the engine
#: allocates at most ``total * gpu_memory_utilization`` bytes.  A value of
#: 0.5 ensures the offline engine can run alongside a recently-teardown
#: capture engine without hitting CUDA OOM, while surviving both smaller
#: cards and larger verifier models.
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.5


#: Environment variables that must NOT leak from the caller into the offline
#: extraction engine.
#:
#: ``VLLM_USE_V2_MODEL_RUNNER`` forces the runner generation
#: (``vllm/config/vllm.py:519-522`` returns it verbatim before any of vLLM's
#: own capability checks run).  The serving-time capture e2e sets it to pin
#: the axis it is testing, and ``_environ`` copies ``os.environ`` wholesale,
#: so without this filter the setting would follow the caller into the
#: offline engine too.
#:
#: That would be fatal rather than merely wrong.  The offline engine runs the
#: ``extract_hidden_states`` speculative method, which V2 does not implement.
#: Left to itself vLLM notices and downgrades, logging "Model Runner V2 does
#: not yet support speculative method 'extract_hidden_states'; using the V1
#: model runner instead" (``vllm/config/vllm.py:546-553``).  But that
#: graceful fallback is only reachable when the env var is *unset*: with it
#: set to ``1`` the property short-circuits and ``_validate_v2_model_runner``
#: raises ``ValueError`` instead (``vllm/config/vllm.py:2137-2147``).
#:
#: The offline leg is the independent reference the capture is compared
#: against, so it must run whatever generation vLLM considers correct for
#: *its own* config -- never whatever the capture leg was pinned to.
_ISOLATED_ENV_VARS: tuple[str, ...] = ("VLLM_USE_V2_MODEL_RUNNER",)


def _environ(speculators_repo: Path) -> dict[str, str]:
    """Build the environment for Speculators subprocesses.

    Inherits the caller's environment except for :data:`_ISOLATED_ENV_VARS`,
    which are stripped so the offline engine picks its own model runner.
    """
    env = dict(os.environ)
    for name in _ISOLATED_ENV_VARS:
        env.pop(name, None)
    source = str(speculators_repo / "src")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not previous else f"{source}{os.pathsep}{previous}"
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


def _persist_captured(
    output_dir: Path,
    name: str,
    result: subprocess.CompletedProcess[bytes],
) -> None:
    """Write a captured child's streams beside the server log.

    ``capture_output=True`` holds both streams in memory and the callers below
    surface only stderr, and only on a non-zero exit -- so stdout was always
    discarded, and a run that failed for any other reason left nothing.  The
    server leg already keeps a durable log; these two now do too.

    Best effort: losing diagnostics must not be what fails an extraction.
    """
    for stream, payload in (("stdout", result.stdout), ("stderr", result.stderr)):
        with contextlib.suppress(OSError):
            (output_dir / f"{name}.{stream}.log").write_bytes(payload or b"")


def _wait_for_health(
    url: str,
    process: subprocess.Popen[bytes],
    timeout: float,
    *,
    poll: float = 0.5,
    log_path: Path | None = None,
) -> None:
    """Block until the vLLM engine reports healthy.

    Polls the subprocess each iteration so an already-dead engine
    short-circuits immediately rather than burning the full timeout.
    If *log_path* is given, its tail is appended to the timeout message.
    """
    import urllib.error  # lazy stdlib
    import urllib.request

    deadline = time.monotonic() + timeout
    last_err: str = "not attempted"
    while time.monotonic() < deadline:
        rc = process.poll()
        if rc is not None:
            detail = f"vLLM exited with code {rc}"
            if log_path is not None:
                detail += "\n--- vLLM log (last 100 lines) ---\n"
                detail += _read_log_tail(log_path)
                detail += "\n--- end of vLLM log ---"
            raise RuntimeError(detail)
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    return
                last_err = f"HTTP {resp.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_err = repr(exc)
        time.sleep(poll)
    detail = f"vLLM did not become ready within {timeout}s; {last_err}"
    if log_path is not None:
        detail += "\n--- vLLM log (last 100 lines) ---\n"
        detail += _read_log_tail(log_path)
        detail += "\n--- end of vLLM log ---"
    raise TimeoutError(detail)


def _write_conversation_jsonl(
    path: Path, prompt: str, *, assistant_response: str | None = None
) -> None:
    """Write a single-row speculators-conversations.jsonl for *prompt*.

    Emits rows in the same schema as the production renderer
    (``_speculators_record`` in ``eagle3.py``): a ``conversations`` list
    carrying ``role`` / ``content`` dicts, with at least one assistant turn.
    ``prepare_data.py`` drops records that lack an assistant turn, so omitting
    one would produce a silently empty dataset.
    """
    turns = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response or ""},
    ]
    row = {"conversations": turns}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def extract(
    verifier_model: str,
    prompt: str,
    target_layers: list[int],
    output_dir: Path,
    *,
    vllm_python: Path | None = None,
    speculators_repo: Path | None = None,
    port: int = 8131,
    ready_timeout: float = 300.0,
    extraction_timeout: float = 600.0,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> Path:
    """Run the full offline extraction pipeline on a single prompt.

    Steps:
    1. Write a one-row preprocessed dataset.
    2. Launch a vLLM hidden-state server (via ``launch_vllm.py``).
    3. Run ``data_generation_offline.py`` against the server.
    4. Tear down the server.

    Args:
        gpu_memory_utilization: Fraction of each GPU device's total memory
            that the offline engine may use.  Defaults to
            ``DEFAULT_GPU_MEMORY_UTILIZATION`` (0.5).

    Returns the path to the directory containing the ``hs_*.safetensors``
    shards.
    """
    vpython = vllm_python or _DEFAULT_VLLM_PYTHON
    repo = speculators_repo or _DEFAULT_SPECULATORS_REPO
    env = _environ(repo)

    hs_dir = output_dir / "offline_hs"
    hs_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: write dataset
    ds_dir = output_dir / "offline_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ds_dir / "speculators-conversations.jsonl"
    _write_conversation_jsonl(jsonl_path, prompt)

    # Step 2: prepare the prepared dataset via prepare_data.py
    prepared_dir = output_dir / "offline_prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepare_result = subprocess.run(
        [
            str(vpython),
            str(repo / "scripts" / "prepare_data.py"),
            "--model",
            verifier_model,
            "--data",
            str(jsonl_path),
            "--output",
            str(prepared_dir),
            "--seq-length",
            "512",
            "--seed",
            "0",
            "--num-preprocessing-workers",
            "0",
            "--overwrite",
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        timeout=extraction_timeout,
    )
    _persist_captured(output_dir, "offline_prepare", prepare_result)
    if prepare_result.returncode != 0:
        stderr = prepare_result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"prepare_data.py failed (exit {prepare_result.returncode}):\n{stderr}"
        )

    # Guard: if prepare_data.py produced zero rows, fail early with the
    # rendered row so the caller can see what was sent vs. what survived.
    prepared_files = sorted(prepared_dir.iterdir())
    if not prepared_files:
        rendered = jsonl_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"prepare_data.py produced an empty dataset (0 rows) in {prepared_dir}.\n"
            f"Rendered input row: {rendered}"
        )

    # Step 3: launch hidden-state server with output captured to a log file
    server_log = output_dir / "offline_vllm.log"
    # Opening for write truncates, so the ordinary debug loop -- run it again
    # and watch -- destroyed the log of the run that actually failed.  Keep the
    # previous attempt beside the current one; one generation is enough for
    # "it worked last time" and cannot grow without bound.
    if server_log.exists():
        server_log.replace(output_dir / "offline_vllm.previous.log")
    log_handle = server_log.open("wb")
    server = subprocess.Popen(
        [
            str(vpython),
            str(repo / "scripts" / "launch_vllm.py"),
            verifier_model,
            "--hidden-states-path",
            str(hs_dir),
            "--target-layer-ids",
            *[str(layer_id) for layer_id in target_layers],
            "--",
            "--port",
            str(port),
            "--max-model-len",
            "512",
            "--max-num-seqs",
            "8",
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ],
        cwd=str(repo),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_health(
            f"http://127.0.0.1:{port}/health",
            server,
            ready_timeout,
            log_path=server_log,
        )

        # Step 4: run data_generation_offline.py
        gen_result = subprocess.run(
            [
                str(vpython),
                str(repo / "scripts" / "data_generation_offline.py"),
                "--endpoint",
                f"http://127.0.0.1:{port}/v1",
                "--preprocessed-data",
                str(prepared_dir),
                "--output",
                str(hs_dir),
                "--max-samples",
                "1",
                "--concurrency",
                "1",
                "--validate-outputs",
                "--fail-on-error",
            ],
            cwd=str(repo),
            env=env,
            capture_output=True,
            timeout=extraction_timeout,
        )
        _persist_captured(output_dir, "offline_generation", gen_result)
        if gen_result.returncode != 0:
            stderr = gen_result.stderr.decode(errors="replace")
            raise RuntimeError(
                f"data_generation_offline.py failed (exit {gen_result.returncode}):\n"
                f"{stderr}"
            )
        return hs_dir

    finally:
        try:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        finally:
            # Nested so a terminate() that raises cannot leave the server log
            # unflushed -- that log is the only record of the server's side.
            log_handle.close()


def load_hidden_states(
    hs_dir: Path,
) -> dict[str, Any]:
    """Load the hidden-state shards from *hs_dir* and return the merged tensor.

    The offline pipeline writes one or more ``hs_*.safetensors`` files, each
    containing a ``hidden_states`` tensor of shape
    ``(sequence_length, num_layers, hidden_size)``.

    Returns a dict mapping layer index to a CPU tensor of shape
    ``(sequence_length, hidden_size)``.
    """
    import torch  # lazy — only in vLLM venv

    shards = sorted(hs_dir.glob("hs_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no hs_*.safetensors in {hs_dir}")

    from safetensors import safe_open

    layers: dict[int, list[torch.Tensor]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            hs = f.get_tensor("hidden_states")
        # hs shape: (seq_len, num_layers, hidden_size)
        for i in range(hs.shape[1]):
            if i not in layers:
                layers[i] = []
            layers[i].append(hs[:, i])

    merged: dict[str, torch.Tensor] = {}
    for idx in sorted(layers.keys()):
        tensors = layers[idx]
        if len(tensors) == 1:
            merged[str(idx)] = tensors[0].cpu()
        else:
            merged[str(idx)] = torch.cat(tensors, dim=0).cpu()

    return merged