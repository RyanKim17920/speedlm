from __future__ import annotations

import json
import os
import re
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
SPEEDLM = REPO_ROOT / ".venv" / "bin" / "speedlm"
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM = VLLM_VENV / "bin" / "vllm"


def _require_live_e2e() -> None:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), "live E2E must run inside a SLURM allocation"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "SLURM allocation did not expose a GPU"
    assert SPEEDLM.is_file(), f"missing project CLI: {SPEEDLM}"
    assert VLLM.is_file(), f"missing vLLM CLI: {VLLM}"


def _unique_artifact_dir(root: Path, stage: str, limit: int = 999) -> Path:
    """Create and return a fresh artifact directory for this run.

    Uses ``root/stage`` when it is free, otherwise the first free
    ``root/stage-runN``. ``mkdir(exist_ok=False)`` is kept so two concurrent
    runs can never blend into one directory and a previous run's artifacts are
    never overwritten -- but a rerun no longer fails outright.
    """
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, limit + 1):
        candidate = root / (stage if attempt == 1 else f"{stage}-run{attempt}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError(f"could not allocate a free artifact directory under {root / stage}")


def _ready_timeout() -> float:
    """Gateway readiness cap in seconds, overridable for slow cold starts.

    Some models pay a large one-time cost before serving (e.g. JIT-compiled
    attention kernels), so this is an environment knob rather than a constant.
    """
    raw = os.environ.get("SPEEDLM_E2E_READY_TIMEOUT", "360")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise AssertionError(f"SPEEDLM_E2E_READY_TIMEOUT must be a number, got {raw!r}") from exc
    assert timeout > 0, f"SPEEDLM_E2E_READY_TIMEOUT must be positive, got {raw!r}"
    return timeout


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
                raise AssertionError(f"speedlm exited before readiness with code {returncode}")
            try:
                response = client.get(f"{url}/health")
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.25)
    raise AssertionError(
        f"gateway did not become ready within {timeout}s; last error: {last_error}"
    )


def _post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    response = client.post(url, json=payload)
    elapsed = time.monotonic() - started
    artifact.write_text(
        "\n".join(
            [
                f"status: {response.status_code}",
                f"elapsed_seconds: {elapsed:.6f}",
                f"headers: {json.dumps(dict(response.headers), sort_keys=True)}",
                f"body: {response.text}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert isinstance(body["id"], str) and body["id"]
    assert isinstance(body["choices"], list) and body["choices"]
    assert isinstance(body["usage"]["prompt_tokens"], int)
    assert body["usage"]["prompt_tokens"] > 0
    assert isinstance(body["usage"]["completion_tokens"], int)
    assert body["usage"]["completion_tokens"] > 0
    return body


def _stream_chat(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    artifact: Path,
) -> tuple[dict[str, Any], str]:
    raw_chunks: list[tuple[float, bytes]] = []
    events: list[dict[str, Any]] = []
    saw_done = False
    line_buffer = bytearray()
    started = time.monotonic()
    with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200, response.read().decode("utf-8", errors="replace")
        assert response.headers["content-type"].startswith("text/event-stream")
        for chunk in response.iter_raw():
            elapsed = time.monotonic() - started
            raw_chunks.append((elapsed, chunk))
            line_buffer.extend(chunk)
            while b"\n" in line_buffer:
                raw_line, _, remainder = line_buffer.partition(b"\n")
                line_buffer[:] = remainder
                line = raw_line.rstrip(b"\r").decode("utf-8", errors="strict")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                elif data:
                    events.append(json.loads(data))
    assert not line_buffer.strip(), f"unterminated SSE bytes: {bytes(line_buffer)!r}"

    artifact.write_text(
        "".join(
            f"[+{elapsed:.6f}s] {chunk.decode('utf-8', errors='replace')}"
            for elapsed, chunk in raw_chunks
        ),
        encoding="utf-8",
    )
    assert saw_done, "stream ended without [DONE]"
    assert len(raw_chunks) >= 2, "response was not delivered incrementally"
    assert len(events) >= 2, "expected multiple SSE JSON events"
    assert any(chunk for _, chunk in raw_chunks[:-1] if b"data:" in chunk)

    content = "".join(
        choice.get("delta", {}).get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
    )
    reasoning_content = "".join(
        delta_value
        for event in events
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
        for delta in [choice.get("delta")]
        if isinstance(delta, dict)
        for field in ("reasoning_content", "reasoning", "thinking")
        for delta_value in [delta.get(field)]
        if isinstance(delta_value, str)
    )
    usage_events = [event["usage"] for event in events if isinstance(event.get("usage"), dict)]
    assert content or reasoning_content, (
        "stream contained neither assistant content nor reasoning"
    )
    assert usage_events, "stream_options.include_usage did not produce usage"
    usage = usage_events[-1]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    summary = {
        "id": next(event["id"] for event in events if isinstance(event.get("id"), str)),
        "model": next(event["model"] for event in events if isinstance(event.get("model"), str)),
        "content": content,
        "reasoning_content": reasoning_content or None,
        "usage": usage,
        "raw_chunk_count": len(raw_chunks),
        "event_count": len(events),
        "first_chunk_seconds": raw_chunks[0][0],
        "last_chunk_seconds": raw_chunks[-1][0],
    }
    return summary, content


def _wait_for_traces(path: Path, expected: int, timeout: float = 15.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) >= expected:
                return records
        time.sleep(0.1)
    actual = path.read_text(encoding="utf-8") if path.exists() else "<missing>"
    raise AssertionError(f"expected {expected} traces at {path}, got:\n{actual}")


def _run_cli(home: Path, args: list[str], artifact: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SPEEDLM_HOME"] = str(home)
    completed = subprocess.run(
        [str(SPEEDLM), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    artifact.write_text(completed.stdout, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout
    return completed


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


def _assert_processes_gone(pids: set[int], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    alive: set[int] = set()
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    pytest.fail(f"orphaned child processes remain: {sorted(alive)}")


def _capture_metrics(gateway_log: Path, artifact: Path) -> None:
    log = gateway_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"launching vLLM on 127\.0\.0\.1:(\d+)", log)
    assert match, "could not find child vLLM port in gateway log"
    response = httpx.get(
        f"http://127.0.0.1:{match.group(1)}/metrics",
        timeout=10.0,
        trust_env=False,
    )
    assert response.status_code == 200
    matching = [
        line
        for line in response.text.splitlines()
        if any(word in line.lower() for word in ("speculat", "draft", "accept"))
    ]
    artifact.write_text("\n".join(matching) + "\n", encoding="utf-8")


@pytest.mark.e2e
def test_speedlm_against_live_vllm() -> None:
    _require_live_e2e()
    model = os.environ.get("SPEEDLM_E2E_MODEL", "Qwen/Qwen3.5-2B")
    stage = os.environ.get("SPEEDLM_E2E_STAGE", "stage1-qwen")
    passthrough = json.loads(os.environ.get("SPEEDLM_E2E_VLLM_ARGS", "[]"))
    assert isinstance(passthrough, list) and all(isinstance(arg, str) for arg in passthrough)

    artifact_root_raw = os.environ.get("SPEEDLM_E2E_ARTIFACT_DIR")
    assert artifact_root_raw, "SPEEDLM_E2E_ARTIFACT_DIR is required"
    artifact_dir = _unique_artifact_dir(Path(artifact_root_raw), stage)
    print(f"stage artifact directory: {artifact_dir}")
    ready_timeout = _ready_timeout()
    home = artifact_dir / "speedlm_home"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    gateway_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    command = [
        str(SPEEDLM),
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        *passthrough,
    ]
    (artifact_dir / "command.txt").write_text(
        " ".join(command) + "\n"
        f"node: {socket.gethostname()}\n"
        f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}\n"
        f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n"
        f"vllm_executable: {VLLM}\n"
        f"vllm_shebang: {VLLM.read_text(encoding='utf-8').splitlines()[0]}\n",
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
        _wait_for_gateway(gateway_url, process, timeout=ready_timeout)
        observed_pids = _descendant_pids(process.pid)
        assert observed_pids, "speedlm did not have a live vLLM child"

        chat_url = f"{gateway_url}/v1/chat/completions"
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            nonstream_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with a short friendly greeting."}
                ],
                "temperature": 0.2,
                "top_p": 0.85,
                "seed": 123,
                "max_tokens": 32,
            }
            nonstream = _post_json(
                client,
                chat_url,
                nonstream_payload,
                artifact_dir / "nonstream-response.txt",
            )
            nonstream_message = nonstream["choices"][0]["message"]
            nonstream_content = nonstream_message.get("content")
            nonstream_reasoning = next(
                (
                    nonstream_message[field]
                    for field in ("reasoning_content", "reasoning", "thinking")
                    if isinstance(nonstream_message.get(field), str)
                ),
                None,
            )
            assert (
                isinstance(nonstream_content, str) and nonstream_content.strip()
            ) or (
                isinstance(nonstream_reasoning, str) and nonstream_reasoning.strip()
            )

            stream_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write four short numbered facts about the Moon. "
                            "Put each fact on its own line."
                        ),
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": 456,
                "max_tokens": 96,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            stream, stream_content = _stream_chat(
                client,
                chat_url,
                stream_payload,
                artifact_dir / "stream-response.txt",
            )
            _write_json(artifact_dir / "stream-summary.json", stream)

            tool_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the temperature in Paris? Use the provided tool.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_temperature",
                            "description": "Get the current temperature for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_temperature"},
                },
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 789,
                "max_tokens": 64,
            }
            tool = _post_json(
                client,
                chat_url,
                tool_payload,
                artifact_dir / "tool-response.txt",
            )
            tool_calls = tool["choices"][0]["message"].get("tool_calls")
            assert isinstance(tool_calls, list) and tool_calls
            assert tool_calls[0]["function"]["name"] == "get_temperature"
            json.loads(tool_calls[0]["function"]["arguments"])

        traces_path = home / "traces" / "traces.jsonl"
        traces = _wait_for_traces(traces_path, 3)
        assert len(traces) == 3
        trace_by_id = {trace["id"]: trace for trace in traces}
        nonstream_trace = trace_by_id[nonstream["id"]]
        stream_trace = trace_by_id[stream["id"]]
        tool_trace = trace_by_id[tool["id"]]

        assert nonstream_trace["messages"][-1]["content"] == nonstream_content
        assert stream_trace["messages"][-1]["content"] == stream_content
        if nonstream_reasoning:
            assert (
                nonstream_trace["messages"][-1]["reasoning_content"]
                == nonstream_reasoning
            )
        if stream["reasoning_content"]:
            assert (
                stream_trace["messages"][-1]["reasoning_content"]
                == stream["reasoning_content"]
            )
        assert tool_trace["tool_calls"] == tool_calls
        for trace, payload, usage in (
            (nonstream_trace, nonstream_payload, nonstream["usage"]),
            (stream_trace, stream_payload, stream["usage"]),
            (tool_trace, tool_payload, tool["usage"]),
        ):
            assert trace["prompt_tokens"] == usage["prompt_tokens"]
            assert trace["completion_tokens"] == usage["completion_tokens"]
            assert trace["temperature"] == payload["temperature"]
            assert trace["top_p"] == payload["top_p"]
            assert trace["seed"] == payload["seed"]

        (artifact_dir / "traces.jsonl").write_text(
            traces_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        status = _run_cli(home, ["status", "--json"], artifact_dir / "speedlm-status.json")
        status_body = json.loads(status.stdout)
        assert status_body["gateway"]["state"] == "running"
        assert status_body["gateway"]["pid"] == process.pid
        assert status_body["gateway"]["child_pid"] in observed_pids
        assert status_body["traces"]["count"] == len(traces)
        expected_tokens = sum(
            trace["prompt_tokens"] + trace["completion_tokens"] for trace in traces
        )
        assert status_body["traces"]["tokens"] == expected_tokens

        stats = _run_cli(home, ["traces", "stats"], artifact_dir / "speedlm-traces-stats.txt")
        assert f"count    : {len(traces)}" in stats.stdout
        assert f"tokens   : {expected_tokens}" in stats.stdout

        if os.environ.get("SPEEDLM_E2E_CAPTURE_METRICS") == "1":
            _capture_metrics(gateway_log, artifact_dir / "acceptance-metrics.txt")
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
        _assert_processes_gone(observed_pids)
