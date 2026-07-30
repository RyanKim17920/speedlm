"""Literal GPU E2E for the production idle-tuning lifecycle.

The test is intentionally model- and vendor-agnostic.  The operator supplies a
complete SpeedLM config (including an EAGLE-3 profile and Speculators paths);
the test derives the served model, alias, trace threshold, and sampling values
from that config.

Required environment:

* ``SPEEDLM_E2E_IDLE_TUNING=1``
* ``SPEEDLM_E2E_TUNING_CONFIG=/absolute/path/to/config.json``
* ``SPEEDLM_E2E_ARTIFACT_DIR=/durable/artifact/root``

Optional environment:

* ``SPEEDLM_E2E_TUNING_PROFILE=/path/to/custom-profile.json``
* ``SPEEDLM_E2E_VLLM_ARGS='["--max-model-len", "4096"]'``
* ``SPEEDLM_E2E_READY_TIMEOUT=900``
* ``SPEEDLM_E2E_TUNING_TIMEOUT=7200``
* ``SPEEDLM_E2E_REQUEST_TIMEOUT=1200``

One invocation creates an isolated ``SPEEDLM_HOME`` under the artifact root,
so old traces, active artifacts, and scheduler state cannot satisfy assertions.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from speedlm.config import SpeedLMConfig, load_config

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PREEMPTIBLE_STATES = frozenset(
    {
        "EXTRACTING",
        "TRAINING",
        "CANDIDATE_STARTING",
        "BENCHMARKING",
    }
)
COMPLETE_OUTCOMES = frozenset({"promoted", "rejected"})
FAILED_OUTCOMES = frozenset({"failed", "final_assistant_mask_error"})

JsonObject = dict[str, Any]


def _require_environment() -> tuple[Path, Path, Path | None]:
    if os.environ.get("SPEEDLM_E2E_IDLE_TUNING") != "1":
        pytest.skip("set SPEEDLM_E2E_IDLE_TUNING=1 in an accelerator job")

    config = _required_file("SPEEDLM_E2E_TUNING_CONFIG")
    artifact_root = _required_path("SPEEDLM_E2E_ARTIFACT_DIR")
    profile_raw = os.environ.get("SPEEDLM_E2E_TUNING_PROFILE")
    profile = None if profile_raw is None else Path(profile_raw).expanduser().resolve()
    if profile is not None:
        assert profile.is_file(), f"SPEEDLM_E2E_TUNING_PROFILE is not a file: {profile}"

    assert shutil.which("vllm"), "vllm must be available on PATH"
    return config, artifact_root, profile


def _required_file(name: str) -> Path:
    path = _required_path(name)
    assert path.is_file(), f"{name} is not a file: {path}"
    return path


def _required_path(name: str) -> Path:
    raw = os.environ.get(name)
    assert raw, f"{name} is required"
    return Path(raw).expanduser().resolve()


def _timeout(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be numeric, got {raw!r}") from exc
    assert value > 0, f"{name} must be positive, got {raw!r}"
    return value


def _vllm_args() -> list[str]:
    raw = os.environ.get("SPEEDLM_E2E_VLLM_ARGS", "[]")
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError("SPEEDLM_E2E_VLLM_ARGS must be a JSON array") from exc
    assert isinstance(value, list) and all(isinstance(item, str) for item in value), (
        "SPEEDLM_E2E_VLLM_ARGS must be a JSON array of strings"
    )
    return value


def _unique_artifact_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1_000):
        suffix = "" if attempt == 1 else f"-run{attempt}"
        candidate = root / f"live-idle-tuning{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError(f"could not allocate an artifact directory under {root}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_object(path: Path) -> JsonObject | None:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # The production writer uses atomic replacement.  Keep this tolerance
        # for network filesystems whose client cache may briefly expose stale
        # metadata during replacement.
        return None
    assert isinstance(value, dict), f"{path} must contain a JSON object"
    return value


def _wait_until(
    description: str,
    predicate: Callable[[], object | None],
    *,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise AssertionError(
                f"speedlm exited with code {returncode} while waiting for {description}"
            )
        result = predicate()
        if result is not None and result is not False:
            return result
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:g}s waiting for {description}")


def _wait_for_gateway(
    url: str,
    *,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> None:
    last_error = "not attempted"

    def ready() -> bool:
        nonlocal last_error
        try:
            response = httpx.get(f"{url}/health", timeout=2.0, trust_env=False)
        except httpx.HTTPError as exc:
            last_error = repr(exc)
            return False
        if 200 <= response.status_code < 300:
            return True
        last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        return False

    try:
        _wait_until("gateway readiness", ready, process=process, timeout=timeout)
    except AssertionError as exc:
        raise AssertionError(f"{exc}; last error: {last_error}") from exc


def _post_chat(
    gateway_url: str,
    config: SpeedLMConfig,
    prompt: str,
    *,
    timeout: float,
) -> tuple[JsonObject, float]:
    payload = {
        "model": config.alias,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "seed": config.sampling.seed,
        "max_tokens": 32,
    }
    started = time.monotonic()
    response = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        trust_env=False,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    body: object = response.json()
    assert isinstance(body, dict), body
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, body
    usage = body.get("usage")
    assert isinstance(usage, dict), body
    assert isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
    assert (
        isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
    )
    return body, elapsed


def _trace_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    except FileNotFoundError:
        return 0


def _scheduler_result_after(
    path: Path,
    *,
    after: float,
    accepted: frozenset[str],
) -> JsonObject | None:
    scheduler = _read_object(path)
    if scheduler is None:
        return None
    error = scheduler.get("last_error")
    result = scheduler.get("last_result")
    result_at = scheduler.get("last_result_at")
    if result is None:
        if error is not None:
            raise AssertionError(f"idle tuner failed outside a cycle: {error}")
        return None
    assert isinstance(result, dict), scheduler
    outcome = result.get("outcome")
    if outcome in FAILED_OUTCOMES:
        raise AssertionError(f"idle tuning cycle failed: {result}")
    if (
        outcome in accepted
        and isinstance(result_at, (int, float))
        and not isinstance(result_at, bool)
        and float(result_at) > after
    ):
        return result
    return None


def _copy_profile(profile: Path | None, home: Path) -> None:
    if profile is None:
        return
    profiles = home / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile, profiles / profile.name)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descendant_pids(root_pid: int) -> set[int]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    children: dict[int, set[int]] = {}
    for line in completed.stdout.splitlines():
        raw_pid, raw_parent = line.split()
        children.setdefault(int(raw_parent), set()).add(int(raw_pid))
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), set()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _assert_processes_gone(pids: set[int], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    pytest.fail(f"orphaned child processes remain: {sorted(alive)}")


def test_live_idle_tuning_preempts_then_completes() -> None:
    config_path, artifact_root, profile = _require_environment()
    config = load_config(config_path)
    artifact_dir = _unique_artifact_dir(artifact_root)
    home = artifact_dir / "speedlm_home"
    home.mkdir()
    _copy_profile(profile, home)

    ready_timeout = _timeout("SPEEDLM_E2E_READY_TIMEOUT", 900.0)
    tuning_timeout = _timeout("SPEEDLM_E2E_TUNING_TIMEOUT", 7_200.0)
    request_timeout = _timeout("SPEEDLM_E2E_REQUEST_TIMEOUT", 1_200.0)
    port = _free_port()
    gateway_url = f"http://127.0.0.1:{port}"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    command = [
        sys.executable,
        "-m",
        "speedlm.cli",
        "--home",
        str(home),
        "vllm",
        "serve",
        config.model,
        "--config",
        str(config_path),
        "--enable-idle-tuning",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *_vllm_args(),
    ]
    _write_json(
        artifact_dir / "invocation.json",
        {
            "argv": command,
            "config": str(config_path),
            "custom_profile": str(profile) if profile is not None else None,
            "model": config.model,
            "model_alias": config.alias,
            "min_trace_records": config.tuning.min_trace_records,
            "hostname": socket.gethostname(),
        },
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SPEEDLM_HOME": str(home),
        }
    )
    log_handle = gateway_log.open("wb")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    observed_pids: set[int] = set()
    try:
        _wait_for_gateway(gateway_url, process=process, timeout=ready_timeout)
        observed_pids.update(_descendant_pids(process.pid))
        assert observed_pids, "speedlm did not launch a vLLM child"

        traces_path = home / "traces" / "traces.jsonl"
        for index in range(config.tuning.min_trace_records):
            body, _ = _post_chat(
                gateway_url,
                config,
                (
                    "This is idle-tuning seed request "
                    f"{index + 1}/{config.tuning.min_trace_records}. "
                    "Reply with one short sentence."
                ),
                timeout=request_timeout,
            )
            _write_json(artifact_dir / f"seed-response-{index + 1:04d}.json", body)

        _wait_until(
            "all seed traces to be captured",
            lambda: (
                _trace_count(traces_path)
                if _trace_count(traces_path) >= config.tuning.min_trace_records
                else None
            ),
            process=process,
            timeout=60.0,
        )

        state_path = home / "runs" / "state.json"
        active_state = _wait_until(
            "vLLM to sleep and the first cycle to become preemptible",
            lambda: (
                state
                if (state := _read_object(state_path)) is not None
                and state.get("state") in ACTIVE_PREEMPTIBLE_STATES
                else None
            ),
            process=process,
            timeout=tuning_timeout,
        )
        assert isinstance(active_state, dict)
        preempt_started_at = time.time()
        queued_body, queued_seconds = _post_chat(
            gateway_url,
            config,
            "Preempt the sleeping idle tuner, restore serving, and reply with READY.",
            timeout=request_timeout,
        )
        _write_json(artifact_dir / "queued-response.json", queued_body)
        _write_json(
            artifact_dir / "preemption-observation.json",
            {
                "state_when_submitted": active_state.get("state"),
                "queued_request_seconds": queued_seconds,
            },
        )

        scheduler_path = home / "runs" / "scheduler.json"
        preempted = _wait_until(
            "the first cycle to report transparent preemption",
            lambda: _scheduler_result_after(
                scheduler_path,
                after=preempt_started_at,
                accepted=frozenset({"preempted"}),
            ),
            process=process,
            timeout=request_timeout,
        )
        assert isinstance(preempted, dict)

        expected_traces = config.tuning.min_trace_records + 1
        _wait_until(
            "the queued request to be captured",
            lambda: (
                count
                if (count := _trace_count(traces_path)) >= expected_traces
                else None
            ),
            process=process,
            timeout=60.0,
        )

        terminal_after = time.time()
        terminal = _wait_until(
            "the next trace watermark to complete training and its held-out gate",
            lambda: _scheduler_result_after(
                scheduler_path,
                after=terminal_after,
                accepted=COMPLETE_OUTCOMES,
            ),
            process=process,
            timeout=tuning_timeout,
        )
        assert isinstance(terminal, dict)
        artifact_id = terminal.get("artifact_id")
        decision_path = terminal.get("decision_path")
        assert isinstance(artifact_id, str) and artifact_id, terminal
        assert isinstance(decision_path, str) and Path(decision_path).is_file(), terminal
        assert (home / "runs" / "artifacts" / artifact_id / "manifest.json").is_file()
        if terminal["outcome"] == "promoted":
            active = _read_object(home / "runs" / "active.json")
            assert active is not None and active.get("artifact_id") == artifact_id

        scheduler = _read_object(scheduler_path)
        state = _read_object(state_path)
        assert scheduler is not None
        assert state is not None and state.get("state") == "READY", state
        _write_json(artifact_dir / "terminal-scheduler.json", scheduler)
        _write_json(artifact_dir / "terminal-state.json", state)
        events = home / "runs" / "events.jsonl"
        assert events.is_file() and events.stat().st_size > 0
    finally:
        if process.poll() is None:
            observed_pids.update(_descendant_pids(process.pid))
            os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=90.0)
        except subprocess.TimeoutExpired:
            observed_pids.update(_descendant_pids(process.pid))
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=30.0)
        log_handle.close()
        _write_json(
            artifact_dir / "shutdown.json",
            {
                "gateway_pid": process.pid,
                "observed_descendant_pids": sorted(observed_pids),
                "gateway_returncode": returncode,
            },
        )
        _assert_processes_gone(observed_pids)
