"""Live protocol/harness matrix for SpeedLM's raw exchange capture contract.

The test contains no model-name conditionals. Select the served model and every
vLLM passthrough argument through ``SPEEDLM_CAPTURE_MODEL`` and
``SPEEDLM_CAPTURE_VLLM_ARGS`` (a JSON string array), so the same test can cover
Qwen, GPT-OSS, Mistral, and future OpenAI-compatible models.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEEDLM = REPO_ROOT / ".venv" / "bin" / "speedlm"
DEFAULT_VLLM_VENV = Path(
    "/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm"
)

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResponseIdentity:
    marker: str
    response_id: str
    choice_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawObservation:
    identity: ResponseIdentity
    request_body: bytes
    response_body: bytes


def _require_live_e2e() -> tuple[str, list[str], Path]:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), (
        "capture harness E2E must run inside a SLURM allocation"
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible_devices and visible_devices != "-1", (
        "SLURM allocation did not expose a GPU"
    )
    assert SPEEDLM.is_file(), f"missing project CLI: {SPEEDLM}"

    model = os.environ.get("SPEEDLM_CAPTURE_MODEL")
    assert model, "SPEEDLM_CAPTURE_MODEL is required"
    raw_args = os.environ.get("SPEEDLM_CAPTURE_VLLM_ARGS", "[]")
    try:
        vllm_args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "SPEEDLM_CAPTURE_VLLM_ARGS must be a JSON string array"
        ) from exc
    assert isinstance(vllm_args, list) and all(
        isinstance(argument, str) for argument in vllm_args
    ), "SPEEDLM_CAPTURE_VLLM_ARGS must be a JSON string array"

    artifact_root = os.environ.get("SPEEDLM_CAPTURE_ARTIFACT_DIR")
    assert artifact_root, "SPEEDLM_CAPTURE_ARTIFACT_DIR is required"
    return model, vllm_args, Path(artifact_root)


def _float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be a number, got {raw!r}") from exc
    assert value > 0, f"{name} must be positive, got {raw!r}"
    return value


def _int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be an integer, got {raw!r}") from exc
    assert value > 0, f"{name} must be positive, got {raw!r}"
    return value


def _unique_artifact_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stage = os.environ.get("SPEEDLM_CAPTURE_STAGE", "capture-harness-matrix")
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"-run{attempt}"
        candidate = root / f"{stage}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError(f"could not allocate artifact directory below {root}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_gateway_socket(
    port: int,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
) -> None:
    """Wait without making an HTTP request that would enter the raw ledger."""
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise AssertionError(
                f"speedlm exited before readiness with code {returncode}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:
            last_error = repr(exc)
        time.sleep(0.25)
    raise AssertionError(
        f"gateway did not listen within {timeout:g}s; last error: {last_error}"
    )


def _chat_payload(
    model: str,
    marker: str,
    *,
    stream: bool,
    max_tokens: int,
    n: int = 1,
) -> JsonObject:
    payload: JsonObject = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Unique capture marker: {marker}\n"
                    "Respond with one short ordinary word."
                ),
            }
        ],
        "temperature": 0.2 if n > 1 else 0.0,
        "seed": 731,
        "max_tokens": max_tokens,
        "n": n,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _identity_from_response(
    marker: str,
    response_body: bytes,
    *,
    stream: bool,
) -> ResponseIdentity:
    response_ids: set[str] = set()
    choice_indexes: set[int] = set()
    if stream:
        saw_done = False
        for raw_line in response_body.splitlines():
            if not raw_line.startswith(b"data:"):
                continue
            data = raw_line[5:].strip()
            if data == b"[DONE]":
                saw_done = True
                continue
            if not data:
                continue
            event = json.loads(data)
            response_id = event.get("id")
            if isinstance(response_id, str) and response_id:
                response_ids.add(response_id)
            for choice in event.get("choices", []):
                index = choice.get("index") if isinstance(choice, dict) else None
                if isinstance(index, int) and not isinstance(index, bool):
                    choice_indexes.add(index)
        assert saw_done, f"{marker}: streaming response omitted [DONE]"
    else:
        body = json.loads(response_body)
        response_id = body.get("id")
        if isinstance(response_id, str) and response_id:
            response_ids.add(response_id)
        for choice in body.get("choices", []):
            index = choice.get("index") if isinstance(choice, dict) else None
            if isinstance(index, int) and not isinstance(index, bool):
                choice_indexes.add(index)

    assert len(response_ids) == 1, f"{marker}: response IDs were {response_ids}"
    assert choice_indexes, f"{marker}: response contained no choice indexes"
    return ResponseIdentity(
        marker=marker,
        response_id=next(iter(response_ids)),
        choice_indexes=tuple(sorted(choice_indexes)),
    )


def _raw_httpx_chat(
    client: httpx.Client,
    url: str,
    payload: JsonObject,
    marker: str,
) -> RawObservation:
    request_body = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    request = client.build_request(
        "POST",
        url,
        content=request_body,
        headers={"content-type": "application/json"},
    )
    response = client.send(request, stream=True)
    try:
        assert response.status_code == 200, response.read().decode(
            "utf-8",
            errors="replace",
        )
        response_body = b"".join(response.iter_raw())
    finally:
        response.close()
    identity = _identity_from_response(
        marker,
        response_body,
        stream=payload.get("stream") is True,
    )
    return RawObservation(identity, request_body, response_body)


SDK_SCRIPT = r"""
import asyncio
import json
import sys

from openai import AsyncOpenAI, OpenAI

base_url, model, raw_cases, raw_max_tokens = sys.argv[1:]
cases = json.loads(raw_cases)
max_tokens = int(raw_max_tokens)


def payload(marker):
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                f"Unique capture marker: {marker}\n"
                "Respond with one short ordinary word."
            ),
        }],
        "temperature": 0.0,
        "seed": 731,
        "max_tokens": max_tokens,
    }


def identity(marker, response_id, indexes):
    return {
        "marker": marker,
        "response_id": response_id,
        "choice_indexes": sorted(set(indexes)),
    }


results = []
with OpenAI(
    api_key="speedlm-e2e",
    base_url=base_url,
    max_retries=0,
    timeout=120.0,
) as client:
    marker = cases["sync_nonstream"]
    response = client.chat.completions.create(**payload(marker))
    results.append(identity(marker, response.id, [item.index for item in response.choices]))

    marker = cases["sync_stream"]
    stream = client.chat.completions.create(
        **payload(marker),
        stream=True,
        stream_options={"include_usage": True},
    )
    response_ids = set()
    indexes = set()
    for chunk in stream:
        if chunk.id:
            response_ids.add(chunk.id)
        indexes.update(item.index for item in chunk.choices)
    assert len(response_ids) == 1
    results.append(identity(marker, next(iter(response_ids)), indexes))


async def async_case(client, marker, stream):
    if not stream:
        response = await client.chat.completions.create(**payload(marker))
        return identity(
            marker,
            response.id,
            [item.index for item in response.choices],
        )
    response_stream = await client.chat.completions.create(
        **payload(marker),
        stream=True,
        stream_options={"include_usage": True},
    )
    response_ids = set()
    indexes = set()
    async for chunk in response_stream:
        if chunk.id:
            response_ids.add(chunk.id)
        indexes.update(item.index for item in chunk.choices)
    assert len(response_ids) == 1
    return identity(marker, next(iter(response_ids)), indexes)


async def run_async():
    async with AsyncOpenAI(
        api_key="speedlm-e2e",
        base_url=base_url,
        max_retries=0,
        timeout=120.0,
    ) as client:
        markers = cases["async_concurrent"]
        return await asyncio.gather(
            *(
                async_case(client, marker, index % 2 == 1)
                for index, marker in enumerate(markers)
            )
        )


results.extend(asyncio.run(run_async()))
print(json.dumps(results, separators=(",", ":"), sort_keys=True))
"""


def _run_sdk_matrix(
    *,
    sdk_python: Path,
    gateway_url: str,
    model: str,
    cases: JsonObject,
    max_tokens: int,
    artifact_dir: Path,
) -> list[ResponseIdentity]:
    assert sdk_python.is_file(), f"missing SDK Python: {sdk_python}"
    completed = subprocess.run(
        [
            str(sdk_python),
            "-c",
            SDK_SCRIPT,
            f"{gateway_url}/v1",
            model,
            json.dumps(cases, separators=(",", ":"), sort_keys=True),
            str(max_tokens),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        },
        text=True,
        capture_output=True,
        timeout=_float_env("SPEEDLM_CAPTURE_SDK_TIMEOUT", "300"),
        check=False,
    )
    (artifact_dir / "sdk-client.log").write_text(
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    assert completed.returncode == 0, (
        f"OpenAI SDK matrix failed:\n{completed.stdout}\n{completed.stderr}"
    )
    try:
        raw_results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"OpenAI SDK emitted non-JSON output: {completed.stdout!r}"
        ) from exc
    assert isinstance(raw_results, list)
    results = [
        ResponseIdentity(
            marker=item["marker"],
            response_id=item["response_id"],
            choice_indexes=tuple(item["choice_indexes"]),
        )
        for item in raw_results
    ]
    assert all(result.response_id for result in results)
    assert all(result.choice_indexes for result in results)
    return results


def _wait_for_manifests(
    exchanges_dir: Path,
    *,
    expected: int,
    timeout: float,
) -> list[JsonObject]:
    deadline = time.monotonic() + timeout
    last: list[JsonObject] = []
    while time.monotonic() < deadline:
        last = []
        if exchanges_dir.is_dir():
            for manifest_path in sorted(exchanges_dir.glob("*/manifest.json")):
                last.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        if len(last) == expected and all(
            manifest.get("state") == "complete" for manifest in last
        ):
            return last
        time.sleep(0.1)
    raise AssertionError(
        f"expected {expected} complete raw exchanges, got "
        f"{[(item.get('exchange_id'), item.get('state')) for item in last]}"
    )


def _wait_for_traces(
    path: Path,
    *,
    expected: int,
    timeout: float,
) -> list[JsonObject]:
    deadline = time.monotonic() + timeout
    records: list[JsonObject] = []
    while time.monotonic() < deadline:
        if path.is_file():
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len(records) == expected:
            return records
        time.sleep(0.1)
    raise AssertionError(f"expected {expected} traces at {path}, got {len(records)}")


def _marker_from_request(payload: JsonObject) -> str:
    messages = payload.get("messages")
    assert isinstance(messages, list) and messages
    message = messages[-1]
    assert isinstance(message, dict)
    content = message.get("content")
    assert isinstance(content, str)
    prefix = "Unique capture marker: "
    matching = [
        line.removeprefix(prefix)
        for line in content.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matching) == 1
    return matching[0]


def _body_path(
    exchanges_dir: Path,
    manifest: JsonObject,
    side: str,
) -> Path:
    exchange_id = manifest["exchange_id"]
    metadata = manifest[side]
    assert isinstance(exchange_id, str)
    assert isinstance(metadata, dict)
    body_file = metadata["body_file"]
    assert isinstance(body_file, str)
    return exchanges_dir / exchange_id / body_file


def _assert_raw_and_trace_bijection(
    *,
    exchanges_dir: Path,
    manifests: list[JsonObject],
    traces: list[JsonObject],
    observations: dict[str, ResponseIdentity],
    direct_raw: dict[str, RawObservation],
) -> None:
    expected_markers = set(observations)
    manifests_by_marker: dict[str, JsonObject] = {}
    identity_by_exchange: dict[str, ResponseIdentity] = {}

    for manifest in manifests:
        exchange_id = manifest["exchange_id"]
        request_body = _body_path(exchanges_dir, manifest, "request").read_bytes()
        response_body = _body_path(exchanges_dir, manifest, "response").read_bytes()
        request_payload = json.loads(request_body)
        marker = _marker_from_request(request_payload)
        assert marker not in manifests_by_marker, f"duplicate raw request marker: {marker}"
        manifests_by_marker[marker] = manifest

        assert manifest["method"] == "POST"
        assert manifest["path"] == "/v1/chat/completions"
        assert manifest["failure_reason"] is None
        assert manifest["request"]["complete"] is True
        assert manifest["response"]["complete"] is True
        assert manifest["response"]["status"] == 200
        for side, body in (("request", request_body), ("response", response_body)):
            assert manifest[side]["bytes"] == len(body)
            assert manifest[side]["sha256"] == hashlib.sha256(body).hexdigest()

        identity = _identity_from_response(
            marker,
            response_body,
            stream=request_payload.get("stream") is True,
        )
        assert identity == observations[marker]
        identity_by_exchange[exchange_id] = identity

        if marker in direct_raw:
            assert request_body == direct_raw[marker].request_body
            assert response_body == direct_raw[marker].response_body

    assert set(manifests_by_marker) == expected_markers

    traces_by_exchange: dict[str, list[JsonObject]] = {}
    for trace in traces:
        exchange_id = trace.get("exchange_id")
        assert isinstance(exchange_id, str) and exchange_id in identity_by_exchange
        traces_by_exchange.setdefault(exchange_id, []).append(trace)

    assert set(traces_by_exchange) == set(identity_by_exchange)
    actual_trace_ids = [trace["id"] for trace in traces]
    assert len(actual_trace_ids) == len(set(actual_trace_ids))
    for exchange_id, identity in identity_by_exchange.items():
        expected_ids = (
            {identity.response_id}
            if len(identity.choice_indexes) == 1
            else {
                f"{identity.response_id}:choice:{index}"
                for index in identity.choice_indexes
            }
        )
        assert {
            trace["id"] for trace in traces_by_exchange[exchange_id]
        } == expected_ids


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
        parent = pending.pop()
        for child in children.get(parent, set()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _assert_processes_gone(pids: set[int], *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    alive: set[int] = set()
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    pytest.fail(f"orphaned child processes remain: {sorted(alive)}")


def test_live_capture_across_raw_httpx_and_openai_sdks() -> None:
    model, vllm_args, artifact_root = _require_live_e2e()
    artifact_dir = _unique_artifact_dir(artifact_root)
    home = artifact_dir / "speedlm-home"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    gateway_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    max_tokens = _int_env("SPEEDLM_CAPTURE_MAX_TOKENS", "32")
    run_token = f"{int(time.time() * 1_000_000):x}"
    direct_cases = {
        "raw-nonstream": f"{run_token}:raw-httpx:nonstream",
        "raw-stream": f"{run_token}:raw-httpx:stream",
    }
    sdk_cases: JsonObject = {
        "sync_nonstream": f"{run_token}:openai-sync:nonstream",
        "sync_stream": f"{run_token}:openai-sync:stream",
        "async_concurrent": [
            f"{run_token}:openai-async:concurrent-{index}"
            for index in range(4)
        ],
    }
    expected_markers = {
        *direct_cases.values(),
        sdk_cases["sync_nonstream"],
        sdk_cases["sync_stream"],
        *sdk_cases["async_concurrent"],
    }
    assert len(expected_markers) == 8

    command = [
        str(SPEEDLM),
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        *vllm_args,
    ]
    (artifact_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": command,
                "model": model,
                "vllm_args": vllm_args,
                "expected_markers": sorted(expected_markers),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    vllm_venv = Path(
        os.environ.get("SPEEDLM_CAPTURE_VLLM_VENV", str(DEFAULT_VLLM_VENV))
    )
    vllm_executable = vllm_venv / "bin" / "vllm"
    assert vllm_executable.is_file(), f"missing vLLM CLI: {vllm_executable}"
    env.update(
        {
            "SPEEDLM_HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PATH": f"{vllm_venv / 'bin'}:{env.get('PATH', '')}",
        }
    )
    log_handle = gateway_log.open("wb")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    observed_pids: set[int] = set()
    try:
        _wait_for_gateway_socket(
            gateway_port,
            process,
            timeout=_float_env("SPEEDLM_CAPTURE_READY_TIMEOUT", "900"),
        )
        observed_pids = _descendant_pids(process.pid)
        assert observed_pids, "speedlm did not have a live vLLM child"

        chat_url = f"{gateway_url}/v1/chat/completions"
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            raw_nonstream = _raw_httpx_chat(
                client,
                chat_url,
                _chat_payload(
                    model,
                    direct_cases["raw-nonstream"],
                    stream=False,
                    max_tokens=max_tokens,
                    n=2,
                ),
                direct_cases["raw-nonstream"],
            )
            raw_stream = _raw_httpx_chat(
                client,
                chat_url,
                _chat_payload(
                    model,
                    direct_cases["raw-stream"],
                    stream=True,
                    max_tokens=max_tokens,
                ),
                direct_cases["raw-stream"],
            )
        direct_raw = {
            observation.identity.marker: observation
            for observation in (raw_nonstream, raw_stream)
        }

        sdk_python = Path(
            os.environ.get(
                "SPEEDLM_CAPTURE_SDK_PYTHON",
                str(vllm_venv / "bin" / "python"),
            )
        )
        sdk_observations = _run_sdk_matrix(
            sdk_python=sdk_python,
            gateway_url=gateway_url,
            model=model,
            cases=sdk_cases,
            max_tokens=max_tokens,
            artifact_dir=artifact_dir,
        )
        observations = {
            identity.marker: identity
            for identity in (
                raw_nonstream.identity,
                raw_stream.identity,
                *sdk_observations,
            )
        }
        assert set(observations) == expected_markers

        manifests = _wait_for_manifests(
            home / "exchanges",
            expected=len(expected_markers),
            timeout=_float_env("SPEEDLM_CAPTURE_DRAIN_TIMEOUT", "60"),
        )
        expected_traces = sum(
            len(identity.choice_indexes) for identity in observations.values()
        )
        traces = _wait_for_traces(
            home / "traces" / "traces.jsonl",
            expected=expected_traces,
            timeout=_float_env("SPEEDLM_CAPTURE_DRAIN_TIMEOUT", "60"),
        )
        _assert_raw_and_trace_bijection(
            exchanges_dir=home / "exchanges",
            manifests=manifests,
            traces=traces,
            observations=observations,
            direct_raw=direct_raw,
        )
        (artifact_dir / "manifests.json").write_text(
            json.dumps(manifests, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "traces.json").write_text(
            json.dumps(traces, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10)
        log_handle.close()
        (artifact_dir / "shutdown.json").write_text(
            json.dumps(
                {
                    "gateway_pid": process.pid,
                    "observed_descendant_pids": sorted(observed_pids),
                    "returncode": returncode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _assert_processes_gone(observed_pids)
