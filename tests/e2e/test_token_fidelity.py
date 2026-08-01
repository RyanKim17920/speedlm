from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
def _resolve_speedlm() -> Path:
    """Locate the speedlm CLI.

    The console script lives in an installed venv, never in the source tree.
    Under a snapshot run (scripts/make_snapshot_run.sh) REPO_ROOT is a
    read-only `git archive` extract with no .venv, so fall back to PATH.
    Snapshot provenance is still enforced by PYTHONPATH, which the child
    process inherits via os.environ.copy().
    """
    local = REPO_ROOT / ".venv" / "bin" / "speedlm"
    if local.exists():
        return local
    found = shutil.which("speedlm")
    return Path(found) if found is not None else local


SPEEDLM = _resolve_speedlm()
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM = VLLM_VENV / "bin" / "vllm"
MODEL = "Qwen/Qwen3.5-2B"
ADD_GENERATION_PROMPT = True


def _require_live_e2e() -> None:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside an allocated GPU job")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("live E2E must run inside a SLURM allocation")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("SLURM allocation did not expose a GPU")
    if not SPEEDLM.is_file():
        raise RuntimeError(f"missing project CLI: {SPEEDLM}")
    if not VLLM.is_file():
        raise RuntimeError(f"missing vLLM CLI: {VLLM}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wait_for_gateway(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"speedlm exited before readiness with code {returncode}"
                )
            try:
                response = client.get(f"{url}/health")
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.25)
    raise RuntimeError(
        f"gateway did not become ready within {timeout}s; last error: {last_error}"
    )


def _upstream_url(gateway_log: Path, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    pattern = re.compile(r"launching vLLM on 127\.0\.0\.1:(\d+)")
    while time.monotonic() < deadline:
        log = gateway_log.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(log)
        if match is not None:
            return f"http://127.0.0.1:{match.group(1)}"
        time.sleep(0.1)
    raise RuntimeError(f"could not find child vLLM port in {gateway_log}")


def _post_chat(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    response = client.post(url, json=payload)
    artifact.write_text(response.text + "\n", encoding="utf-8")
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"chat response was not an object: {body!r}")
    return body


def _tokenize(
    client: httpx.Client,
    upstream_url: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "add_generation_prompt": ADD_GENERATION_PROMPT,
    }
    if tools is not None:
        payload["tools"] = tools
    response = client.post(f"{upstream_url}/tokenize", json=payload)
    response.raise_for_status()
    body = response.json()
    count = body.get("count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise RuntimeError(f"/tokenize returned an invalid count: {body!r}")
    return count


def _wait_for_traces(
    path: Path,
    response_ids: set[str],
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stored_ids = {
                record.get("id") for record in records if isinstance(record, dict)
            }
            if response_ids <= stored_ids:
                return records
        time.sleep(0.1)
    actual = path.read_text(encoding="utf-8") if path.exists() else "<missing>"
    raise RuntimeError(f"traces did not contain {sorted(response_ids)}:\n{actual}")


def _delta(reconstructed: int, reported: int) -> dict[str, float | int | None]:
    signed = reconstructed - reported
    return {
        "signed": signed,
        "absolute": abs(signed),
        "relative": abs(signed) / reported if reported else None,
    }


def _summary(report: dict[str, Any]) -> str:
    lines = [
        "Token fidelity measurement (no fidelity thresholds asserted)",
        (
            "type       reported(prompt/completion)  "
            "reconstructed(prompt/completion)  abs_delta(prompt/completion)  "
            "relative_delta(prompt/completion)"
        ),
    ]
    for item in report["requests"]:
        reported = item["reported"]
        reconstructed = item["reconstructed"]
        delta = item["delta"]
        prompt_relative = delta["prompt_tokens"]["relative"]
        completion_relative = delta["completion_tokens"]["relative"]
        lines.append(
            f"{item['request_type']:<11}"
            f"{reported['prompt_tokens']:>8}/{reported['completion_tokens']:<12}"
            f"{reconstructed['prompt_tokens']:>8}/"
            f"{reconstructed['completion_tokens']:<17}"
            f"{delta['prompt_tokens']['absolute']:>8}/"
            f"{delta['completion_tokens']['absolute']:<20}"
            f"{prompt_relative if prompt_relative is not None else 'n/a':>8}/"
            f"{completion_relative if completion_relative is not None else 'n/a'}"
        )
    return "\n".join(lines) + "\n"


def _descendant_pids(root_pid: int) -> set[int]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    children: dict[int, set[int]] = {}
    for line in completed.stdout.splitlines():
        raw_pid, raw_ppid = line.split()
        children.setdefault(int(raw_ppid), set()).add(int(raw_pid))
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, set()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _wait_for_processes_gone(pids: set[int], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    raise RuntimeError(f"orphaned child processes remain: {sorted(alive)}")


@pytest.mark.e2e
def test_trace_token_fidelity_measurement() -> None:
    """Measure trace reconstruction fidelity; do not enforce a fidelity threshold."""
    _require_live_e2e()
    artifact_root_raw = os.environ.get("SPEEDLM_E2E_ARTIFACT_DIR")
    if not artifact_root_raw:
        raise RuntimeError("SPEEDLM_E2E_ARTIFACT_DIR is required")
    stage = os.environ.get("SPEEDLM_FIDELITY_STAGE", "token-fidelity-qwen")
    artifact_dir = Path(artifact_root_raw) / stage
    artifact_dir.mkdir(parents=True, exist_ok=False)
    home = artifact_dir / "speedlm_home"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    gateway_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    raw_vllm_args = os.environ.get("SPEEDLM_E2E_VLLM_ARGS")
    if raw_vllm_args is None:
        vllm_args = [
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ]
    else:
        vllm_args = json.loads(raw_vllm_args)
        assert isinstance(vllm_args, list) and all(
            isinstance(argument, str) for argument in vllm_args
        ), "SPEEDLM_E2E_VLLM_ARGS must be a JSON array of strings"
    command = [
        str(SPEEDLM),
        "vllm",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        *vllm_args,
    ]
    (artifact_dir / "command.txt").write_text(
        " ".join(command) + "\n"
        f"node: {socket.gethostname()}\n"
        f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}\n"
        f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n"
        f"vllm_executable: {VLLM}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "SPEEDLM_HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "PATH": f"{VLLM_VENV / 'bin'}:{env['PATH']}",
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
        _wait_for_gateway(gateway_url, process, timeout=360.0)
        observed_pids = _descendant_pids(process.pid)
        upstream_url = _upstream_url(gateway_log)
        common = {
            "model": MODEL,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": 48,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        requests: list[tuple[str, dict[str, Any]]] = [
            (
                "plain",
                {
                    **common,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Answer with exactly one short sentence about Paris.",
                        }
                    ],
                },
            ),
            (
                "multi-turn",
                {
                    **common,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Be concise and preserve conversational context.",
                        },
                        {"role": "user", "content": "My favorite color is cobalt blue."},
                        {
                            "role": "assistant",
                            "content": "I will remember that your favorite color is cobalt blue.",
                        },
                        {
                            "role": "user",
                            "content": "What color did I say? Answer in a short phrase.",
                        },
                    ],
                },
            ),
            (
                "tool-call",
                {
                    **common,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Use the tool to look up the weather in Paris.",
                        }
                    ],
                    "tools": tools,
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "lookup_weather"},
                    },
                },
            ),
        ]

        responses: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            for request_type, payload in requests:
                body = _post_chat(
                    client,
                    f"{gateway_url}/v1/chat/completions",
                    payload,
                    artifact_dir / f"{request_type}-response.json",
                )
                response_id = body.get("id")
                if not isinstance(response_id, str) or not response_id:
                    raise RuntimeError(f"{request_type} response has no id: {body!r}")
                responses[response_id] = (request_type, payload, body)

            traces_path = home / "traces" / "traces.jsonl"
            traces = _wait_for_traces(traces_path, set(responses))
            (artifact_dir / "traces.jsonl").write_text(
                traces_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            trace_by_id = {
                trace.get("id"): trace for trace in traces if isinstance(trace, dict)
            }
            measurements: list[dict[str, Any]] = []
            for response_id, (request_type, payload, response) in responses.items():
                trace = trace_by_id.get(response_id)
                if not isinstance(trace, dict):
                    raise RuntimeError(f"trace {response_id} was not an object")
                messages = trace.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise RuntimeError(
                        f"trace {response_id} has no reconstructed assistant turn"
                    )
                request_tools = payload.get("tools")
                prompt_count = _tokenize(
                    client,
                    upstream_url,
                    messages=messages[:-1],
                    tools=request_tools,
                )
                conversation_count = _tokenize(
                    client,
                    upstream_url,
                    messages=messages,
                    tools=request_tools,
                )
                reconstructed_completion = conversation_count - prompt_count
                usage = response.get("usage")
                if not isinstance(usage, dict):
                    raise RuntimeError(f"response {response_id} has no usage object")
                reported_prompt = usage.get("prompt_tokens")
                reported_completion = usage.get("completion_tokens")
                if (
                    isinstance(reported_prompt, bool)
                    or not isinstance(reported_prompt, int)
                    or isinstance(reported_completion, bool)
                    or not isinstance(reported_completion, int)
                ):
                    raise RuntimeError(
                        f"response {response_id} has invalid token usage: {usage!r}"
                    )
                measurements.append(
                    {
                        "request_type": request_type,
                        "response_id": response_id,
                        "reported": {
                            "prompt_tokens": reported_prompt,
                            "completion_tokens": reported_completion,
                            "total_tokens": reported_prompt + reported_completion,
                        },
                        "reconstructed": {
                            "prompt_tokens": prompt_count,
                            "completion_tokens": reconstructed_completion,
                            "total_tokens": conversation_count,
                        },
                        "delta": {
                            "prompt_tokens": _delta(prompt_count, reported_prompt),
                            "completion_tokens": _delta(
                                reconstructed_completion, reported_completion
                            ),
                            "total_tokens": _delta(
                                conversation_count,
                                reported_prompt + reported_completion,
                            ),
                        },
                        "tokenize_inputs": {
                            "prompt_message_count": len(messages) - 1,
                            "conversation_message_count": len(messages),
                            "tool_schema_count": len(request_tools or []),
                            "add_generation_prompt": ADD_GENERATION_PROMPT,
                        },
                    }
                )

        measurements.sort(key=lambda item: item["request_type"])
        report = {
            "model": MODEL,
            "measurement_only": True,
            "fidelity_threshold_asserted": False,
            "method": (
                "Tokenize trace.messages[:-1] and trace.messages through the child "
                "vLLM /tokenize endpoint with identical tools and "
                "add_generation_prompt; reconstructed completion is the count "
                "difference."
            ),
            "requests": measurements,
            "by_request_type": {
                item["request_type"]: {
                    "reported": item["reported"],
                    "reconstructed": item["reconstructed"],
                    "delta": item["delta"],
                }
                for item in measurements
            },
        }
        _write_json(artifact_dir / "token-fidelity.json", report)
        human_summary = _summary(report)
        (artifact_dir / "token-fidelity-summary.txt").write_text(
            human_summary,
            encoding="utf-8",
        )
        print(human_summary, end="")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10)
        log_handle.close()
        (artifact_dir / "shutdown.txt").write_text(
            f"gateway_pid: {process.pid}\n"
            f"observed_descendant_pids: {sorted(observed_pids)}\n"
            f"gateway_returncode: {returncode}\n",
            encoding="utf-8",
        )
        _wait_for_processes_gone(observed_pids)
