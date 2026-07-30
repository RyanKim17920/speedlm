"""Live, offline E2E matrix for SpeedLM's backend/template generality claim.

One invocation runs exactly one cell selected by ``SPEEDLM_MATRIX_CELL``.  The
test intentionally verifies the claimed contracts rather than adapting its
expectations to the current implementation: a cell that serves successfully
but loses its profile or template metadata is a useful falsification result.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest

from speedlm.training.rows import training_row_from_trace  # type: ignore[import-untyped]
from speedlm.training.templates.chatml import ChatMLTemplate  # type: ignore[import-untyped]
from speedlm.training.templates.harmony import HarmonyTemplate  # type: ignore[import-untyped]

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEEDLM = REPO_ROOT / ".venv" / "bin" / "speedlm"
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM = VLLM_VENV / "bin" / "vllm"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "log_artifacts" / "model-matrix"

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class MatrixCell:
    name: str
    model: str
    speculative_method: str
    profile_name: str
    template_kind: str
    draft_model: str | None
    num_speculative_tokens: int
    vllm_args: tuple[str, ...]
    supports_tools: bool = True
    user_profile: bool = False

    @property
    def speculative_config(self) -> JsonObject | None:
        if self.speculative_method == "none":
            return None
        config: JsonObject = {
            "method": self.speculative_method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.draft_model is not None:
            config["model"] = self.draft_model
        if self.speculative_method == "ngram":
            config.update(prompt_lookup_min=2, prompt_lookup_max=5)
        return config

    def expected_profile(self) -> JsonObject:
        return {
            "name": self.profile_name,
            "verifier_model": self.model,
            "draft_model": self.draft_model,
            "speculative_method": self.speculative_method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "chat_template_kind": self.template_kind,
        }


COMMON_QWEN_ARGS = (
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.90",
    "--enforce-eager",
    "--gdn-prefill-backend",
    "triton",
)

MATRIX: Mapping[str, MatrixCell] = {
    "gpt-oss-20b-eagle3": MatrixCell(
        name="gpt-oss-20b-eagle3",
        model="openai/gpt-oss-20b",
        speculative_method="eagle3",
        profile_name="gpt-oss-20b-eagle3",
        template_kind="harmony",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        num_speculative_tokens=5,
        vllm_args=(
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
            "--enforce-eager",
        ),
    ),
    "qwen3.5-9b-mtp": MatrixCell(
        name="qwen3.5-9b-mtp",
        model="Qwen/Qwen3.5-9B",
        speculative_method="mtp",
        profile_name="qwen3.5-9b-mtp",
        template_kind="chatml",
        draft_model=None,
        num_speculative_tokens=3,
        vllm_args=COMMON_QWEN_ARGS,
    ),
    "qwen3.5-2b-none": MatrixCell(
        name="qwen3.5-2b-none",
        model="Qwen/Qwen3.5-2B",
        speculative_method="none",
        profile_name="qwen3.5-2b-none",
        template_kind="chatml",
        draft_model=None,
        num_speculative_tokens=0,
        vllm_args=COMMON_QWEN_ARGS,
    ),
    "qwen3.5-2b-ngram": MatrixCell(
        name="qwen3.5-2b-ngram",
        model="Qwen/Qwen3.5-2B",
        speculative_method="ngram",
        profile_name="qwen3.5-2b-ngram",
        template_kind="chatml",
        draft_model=None,
        num_speculative_tokens=5,
        vllm_args=COMMON_QWEN_ARGS,
        user_profile=True,
    ),
}

PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)

TOOLS: list[JsonObject] = [
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
]


def _require_live_matrix() -> MatrixCell:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside a GPU job")
    cell_name = os.environ.get("SPEEDLM_MATRIX_CELL")
    assert cell_name, (
        "SPEEDLM_MATRIX_CELL is required; choose one of: " + ", ".join(MATRIX)
    )
    assert cell_name in MATRIX, (
        f"unknown SPEEDLM_MATRIX_CELL={cell_name!r}; choose one of: {', '.join(MATRIX)}"
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible_devices and visible_devices != "-1", (
        "CUDA_VISIBLE_DEVICES must expose the GPU assigned to this cell"
    )
    assert SPEEDLM.is_file(), f"missing project CLI: {SPEEDLM}"
    assert VLLM.is_file(), f"missing vLLM CLI: {VLLM}"
    return MATRIX[cell_name]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_object(value: object, location: str) -> JsonObject:
    if not isinstance(value, dict):
        raise AssertionError(f"{location} must be a JSON object, got {type(value).__name__}")
    return value


def _wait_for_gateway(
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise AssertionError(
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
    raise AssertionError(
        f"gateway did not become ready within {timeout}s; last error: {last_error}"
    )


def _upstream_url(gateway_log: Path, *, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    pattern = re.compile(r"launching vLLM on 127\.0\.0\.1:(\d+)")
    while time.monotonic() < deadline:
        log = gateway_log.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(log)
        if match is not None:
            return f"http://127.0.0.1:{match.group(1)}"
        time.sleep(0.1)
    raise AssertionError(f"could not discover child vLLM port from {gateway_log}")


def _post_chat(
    client: httpx.Client,
    url: str,
    payload: JsonObject,
    artifact: Path,
) -> JsonObject:
    started = time.monotonic()
    response = client.post(url, json=payload)
    elapsed = time.monotonic() - started
    try:
        parsed: object = response.json()
    except json.JSONDecodeError:
        parsed = response.text
    _write_json(
        artifact,
        {
            "request": payload,
            "status": response.status_code,
            "elapsed_seconds": elapsed,
            "headers": dict(response.headers),
            "response": parsed,
        },
    )
    assert response.status_code == 200, response.text
    body = _json_object(parsed, "chat response")
    assert body.get("object") == "chat.completion", body
    assert isinstance(body.get("id"), str) and body["id"], body
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, body
    usage = body.get("usage")
    assert isinstance(usage, dict), body
    assert isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
    assert (
        isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
    )
    return body


def _stream_chat(
    client: httpx.Client,
    url: str,
    payload: JsonObject,
    raw_artifact: Path,
    summary_artifact: Path,
) -> JsonObject:
    chunks: list[tuple[float, bytes]] = []
    events: list[JsonObject] = []
    saw_done = False
    line_buffer = bytearray()
    started = time.monotonic()
    with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200, response.read().decode(
            "utf-8", errors="replace"
        )
        assert response.headers.get("content-type", "").startswith("text/event-stream")
        for chunk in response.iter_raw():
            chunks.append((time.monotonic() - started, chunk))
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
                    events.append(_json_object(json.loads(data), "SSE event"))
    raw_artifact.write_text(
        "".join(
            f"[+{elapsed:.6f}s] {chunk.decode('utf-8', errors='replace')}"
            for elapsed, chunk in chunks
        ),
        encoding="utf-8",
    )
    assert not line_buffer.strip(), f"unterminated SSE bytes: {bytes(line_buffer)!r}"
    assert saw_done, "stream ended without [DONE]"
    assert len(chunks) >= 2, "stream was delivered as fewer than two wire chunks"
    assert len(events) >= 2, "stream contained fewer than two JSON events"

    content = "".join(
        delta_content
        for event in events
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
        for delta in [choice.get("delta")]
        if isinstance(delta, dict)
        for delta_content in [delta.get("content")]
        if isinstance(delta_content, str)
    )
    reasoning_content = "".join(
        delta_reasoning
        for event in events
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
        for delta in [choice.get("delta")]
        if isinstance(delta, dict)
        for field in ("reasoning_content", "reasoning", "thinking")
        for delta_reasoning in [delta.get(field)]
        if isinstance(delta_reasoning, str)
    )
    usage_events = [
        event["usage"] for event in events if isinstance(event.get("usage"), dict)
    ]
    assert content.strip() or reasoning_content.strip(), (
        "stream contained neither assistant content nor reasoning"
    )
    assert usage_events, "stream_options.include_usage did not produce usage"
    usage = _json_object(usage_events[-1], "stream usage")
    assert isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
    assert (
        isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
    )
    response_id = next(
        (event["id"] for event in events if isinstance(event.get("id"), str)),
        None,
    )
    model = next(
        (event["model"] for event in events if isinstance(event.get("model"), str)),
        None,
    )
    assert isinstance(response_id, str) and response_id
    summary: JsonObject = {
        "id": response_id,
        "model": model,
        "content": content,
        "reasoning_content": reasoning_content or None,
        "usage": usage,
        "wire_chunk_count": len(chunks),
        "event_count": len(events),
        "first_chunk_seconds": chunks[0][0],
        "last_chunk_seconds": chunks[-1][0],
    }
    _write_json(summary_artifact, {"request": payload, "response": summary})
    return summary


def _wait_for_traces(
    path: Path,
    response_ids: set[str],
    *,
    timeout: float = 20.0,
) -> dict[str, JsonObject]:
    deadline = time.monotonic() + timeout
    records: list[JsonObject] = []
    while time.monotonic() < deadline:
        if path.exists():
            records = [
                _json_object(json.loads(line), "trace")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_id = {
                record["id"]: record
                for record in records
                if isinstance(record.get("id"), str)
            }
            if response_ids <= set(by_id):
                return {response_id: by_id[response_id] for response_id in response_ids}
        time.sleep(0.1)
    available: list[str] = []
    for record in records:
        record_id = record.get("id")
        if isinstance(record_id, str):
            available.append(record_id)
    available.sort()
    raise AssertionError(
        f"traces at {path} did not contain {sorted(response_ids)}; available={available}"
    )


def _run_cli_json(
    home: Path,
    args: list[str],
    artifact: Path,
) -> tuple[JsonObject, int]:
    env = os.environ.copy()
    env["SPEEDLM_HOME"] = str(home)
    completed = subprocess.run(
        [str(SPEEDLM), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    try:
        body = _json_object(json.loads(completed.stdout), f"speedlm {' '.join(args)}")
    except (json.JSONDecodeError, AssertionError) as exc:
        _write_json(
            artifact,
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "parse_error": str(exc),
            },
        )
        raise AssertionError(
            f"speedlm {' '.join(args)} did not emit JSON; "
            f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}"
        ) from exc
    _write_json(
        artifact,
        {
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "report": body,
        },
    )
    return body, completed.returncode


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


def _assert_processes_gone(pids: set[int], *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    alive: set[int] = set()
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    raise AssertionError(f"orphaned child processes remain: {sorted(alive)}")


def _scrape_metrics(upstream_url: str, artifact: Path) -> str:
    response = httpx.get(
        f"{upstream_url}/metrics",
        timeout=10.0,
        trust_env=False,
    )
    assert response.status_code == 200, response.text
    artifact.write_text(response.text, encoding="utf-8")
    return response.text


def _interesting_counters(prometheus: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in prometheus.splitlines():
        match = PROMETHEUS_SAMPLE.match(raw_line.strip())
        if match is None:
            continue
        name = match.group("name")
        if not any(word in name.lower() for word in ("spec", "draft", "accept")):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def _write_metrics_summary(
    before: str,
    after: str,
    artifact: Path,
) -> None:
    before_values = _interesting_counters(before)
    after_values = _interesting_counters(after)
    names = sorted(set(before_values) | set(after_values))
    _write_json(
        artifact,
        {
            "before": before_values,
            "after": after_values,
            "delta": {
                name: after_values.get(name, 0.0) - before_values.get(name, 0.0)
                for name in names
            },
            "matching_counter_count": len(names),
        },
    )


def _record_check(
    checks: JsonObject,
    failures: list[str],
    name: str,
    function: Callable[[], object],
) -> object | None:
    try:
        value = function()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        checks[name] = {"passed": False, "error": message}
        failures.append(f"{name}: {message}")
        return None
    checks[name] = {"passed": True}
    return value


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _configure_home(cell: MatrixCell, home: Path, artifact_dir: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config: JsonObject = {"model": cell.model}
    if cell.speculative_method != "none":
        config["profile"] = cell.profile_name
    _write_json(home / "config.json", config)
    _write_json(artifact_dir / "expected-profile.json", cell.expected_profile())

    if cell.user_profile:
        profiles_dir = home / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            profiles_dir / f"{cell.profile_name}.json",
            {
                **cell.expected_profile(),
                "target_layer_ids": None,
                "max_seq_len": 4096,
                "trainable": False,
            },
        )


def _verify_trace(
    trace: JsonObject,
    response: JsonObject,
    payload: JsonObject,
    *,
    expected_content: str | None,
) -> None:
    messages = trace.get("messages")
    _require(isinstance(messages, list) and messages, "trace messages are missing")
    assert isinstance(messages, list)
    _require(messages[0].get("role") == "user", "trace must begin with the user turn")
    _require(messages[-1].get("role") == "assistant", "trace must end with assistant")
    if expected_content is not None:
        _require(
            messages[-1].get("content") == expected_content,
            "trace assistant content differs from the response",
        )
    response_reasoning: object = response.get("reasoning_content")
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            for field in ("reasoning_content", "reasoning", "thinking"):
                if isinstance(message.get(field), str):
                    response_reasoning = message[field]
                    break
    if isinstance(response_reasoning, str) and response_reasoning:
        _require(
            messages[-1].get("reasoning_content") == response_reasoning,
            "trace assistant reasoning differs from the response",
        )
    usage = response.get("usage")
    _require(isinstance(usage, dict), "response usage is missing")
    assert isinstance(usage, dict)
    _require(
        trace.get("prompt_tokens") == usage.get("prompt_tokens"),
        "trace prompt token count differs from response usage",
    )
    _require(
        trace.get("completion_tokens") == usage.get("completion_tokens"),
        "trace completion token count differs from response usage",
    )
    for key in ("temperature", "top_p", "seed"):
        _require(trace.get(key) == payload.get(key), f"trace lost request field {key!r}")


def _verify_template(cell: MatrixCell, trace: JsonObject, artifact: Path) -> None:
    row = training_row_from_trace(trace)
    if cell.template_kind == "harmony":
        template = HarmonyTemplate()
        expected_markers = ("<|start|>user<|message|>", "<|start|>assistant")
    elif cell.template_kind == "chatml":
        template = ChatMLTemplate()
        expected_markers = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    else:
        raise AssertionError(f"matrix has no verifier for template {cell.template_kind!r}")
    rendered = template.render(row.conversation, tools=row.tools)
    spans = template.assistant_spans(rendered)
    artifact.write_text(rendered, encoding="utf-8")
    for marker in expected_markers:
        _require(marker in rendered, f"rendered {cell.template_kind} trace lacks {marker!r}")
    _require(spans, f"rendered {cell.template_kind} trace has no assistant spans")
    _require(
        any(rendered[span.start : span.end].strip() for span in spans),
        f"rendered {cell.template_kind} trace has no non-empty assistant span",
    )


def _verify_tool_trace_is_self_contained(trace: JsonObject) -> None:
    _require(trace.get("tools") == TOOLS, "captured trace lost the request's tool schemas")
    row = training_row_from_trace(trace)
    _require(row.tools == tuple(TOOLS), "training row lost the captured tool schemas")


def _verify_status(
    cell: MatrixCell,
    status: JsonObject,
    *,
    gateway_pid: int,
    descendant_pids: set[int],
) -> None:
    gateway = _json_object(status.get("gateway"), "status.gateway")
    _require(gateway.get("state") == "running", "status does not report a running gateway")
    _require(gateway.get("pid") == gateway_pid, "status gateway pid is incorrect")
    _require(
        gateway.get("child_pid") in descendant_pids,
        "status child pid is not a live vLLM descendant",
    )
    _require(gateway.get("model") == cell.model, "status gateway model is incorrect")

    profile_value = status.get("profile")
    if profile_value is None:
        models = status.get("models")
        if isinstance(models, dict):
            profile_value = models.get("profile")
    profile = _json_object(profile_value, "status profile")
    for key, expected in cell.expected_profile().items():
        _require(
            profile.get(key) == expected,
            f"status profile field {key!r}: expected {expected!r}, got {profile.get(key)!r}",
        )


def _verify_doctor_profile(cell: MatrixCell, doctor: JsonObject) -> None:
    checks = doctor.get("checks")
    _require(isinstance(checks, list), "doctor checks are missing")
    assert isinstance(checks, list)
    model_pair = next(
        (
            check
            for check in checks
            if isinstance(check, dict) and check.get("name") == "model_pair"
        ),
        None,
    )
    _require(isinstance(model_pair, dict), "doctor omitted the model_pair check")
    assert isinstance(model_pair, dict)
    _require(
        model_pair.get("status") in {"PASS", "WARN"},
        f"doctor rejected the expected profile: {model_pair.get('detail')}",
    )
    data = _json_object(model_pair.get("data"), "doctor model_pair.data")
    expected = {
        "profile": cell.profile_name,
        "verifier": cell.model,
        "draft": cell.draft_model,
        "method": cell.speculative_method,
    }
    for key, value in expected.items():
        _require(
            data.get(key) == value,
            f"doctor profile field {key!r}: expected {value!r}, got {data.get(key)!r}",
        )


def _traffic_prompt(cell: MatrixCell) -> str:
    if cell.speculative_method == "ngram":
        return (
            "alpha beta gamma delta epsilon alpha beta gamma delta epsilon "
            "alpha beta gamma delta epsilon. Continue the repeated sequence once."
        )
    return "Reply with one short sentence explaining why independent tests are useful."


@pytest.mark.e2e
def test_speedlm_model_matrix_cell() -> None:
    cell = _require_live_matrix()
    artifact_root = Path(
        os.environ.get("SPEEDLM_MATRIX_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_ROOT))
    )
    artifact_dir = artifact_root / cell.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    home = artifact_dir / "speedlm_home"
    _configure_home(cell, home, artifact_dir)

    gateway_log = artifact_dir / "gateway-and-vllm.log"
    gateway_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    command = [
        str(SPEEDLM),
        "vllm",
        "serve",
        cell.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        *cell.vllm_args,
    ]
    speculative_config = cell.speculative_config
    if speculative_config is not None:
        command.extend(
            ["--speculative-config", json.dumps(speculative_config, separators=(",", ":"))]
        )

    (artifact_dir / "command.txt").write_text(
        "\n".join(
            [
                f"command: {shlex.join(command)}",
                f"cell: {cell.name}",
                f"node: {socket.gethostname()}",
                f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}",
                f"vllm_executable: {VLLM}",
                "",
            ]
        ),
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
    startup_timeout = float(os.environ.get("SPEEDLM_MATRIX_STARTUP_TIMEOUT", "900"))
    request_timeout = float(os.environ.get("SPEEDLM_MATRIX_REQUEST_TIMEOUT", "240"))
    checks: JsonObject = {}
    failures: list[str] = []
    results: JsonObject = {
        "cell": cell.name,
        "expected_profile": cell.expected_profile(),
        "checks": checks,
    }

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
    descendants: set[int] = set()
    responses: dict[str, JsonObject] = {}
    payloads: dict[str, JsonObject] = {}
    metrics_before: str | None = None
    metrics_after: str | None = None
    ready = False
    try:
        ready_result = _record_check(
            checks,
            failures,
            "gateway_startup",
            lambda: _wait_for_gateway(
                gateway_url,
                process,
                timeout=startup_timeout,
            ),
        )
        ready = checks["gateway_startup"]["passed"]
        del ready_result
        if ready:
            descendants = _descendant_pids(process.pid)
            _record_check(
                checks,
                failures,
                "vllm_child",
                lambda: _require(bool(descendants), "speedlm has no live vLLM child"),
            )

            upstream_url_value = _record_check(
                checks,
                failures,
                "upstream_discovery",
                lambda: _upstream_url(gateway_log),
            )
            upstream_url = (
                upstream_url_value if isinstance(upstream_url_value, str) else None
            )
            if upstream_url is not None:
                before_value = _record_check(
                    checks,
                    failures,
                    "metrics_before",
                    lambda: _scrape_metrics(
                        upstream_url, artifact_dir / "metrics-before.prom"
                    ),
                )
                if isinstance(before_value, str):
                    metrics_before = before_value

            chat_url = f"{gateway_url}/v1/chat/completions"
            with httpx.Client(timeout=request_timeout, trust_env=False) as client:
                nonstream_payload: JsonObject = {
                    "model": cell.model,
                    "messages": [{"role": "user", "content": _traffic_prompt(cell)}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 101,
                    "max_tokens": 48,
                }
                payloads["nonstream"] = nonstream_payload
                nonstream_value = _record_check(
                    checks,
                    failures,
                    "nonstream_chat",
                    lambda: _post_chat(
                        client,
                        chat_url,
                        nonstream_payload,
                        artifact_dir / "nonstream-response.json",
                    ),
                )
                if isinstance(nonstream_value, dict):
                    responses["nonstream"] = nonstream_value

                stream_payload: JsonObject = {
                    "model": cell.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Give two short numbered facts about the Moon.",
                        }
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 202,
                    "max_tokens": 64,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                payloads["stream"] = stream_payload
                stream_value = _record_check(
                    checks,
                    failures,
                    "stream_chat",
                    lambda: _stream_chat(
                        client,
                        chat_url,
                        stream_payload,
                        artifact_dir / "stream-response.sse",
                        artifact_dir / "stream-summary.json",
                    ),
                )
                if isinstance(stream_value, dict):
                    responses["stream"] = stream_value

                if cell.supports_tools:
                    tool_payload: JsonObject = {
                        "model": cell.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "What is the temperature in Paris? "
                                    "Use the provided tool."
                                ),
                            }
                        ],
                        "tools": TOOLS,
                        "tool_choice": {
                            "type": "function",
                            "function": {"name": "get_temperature"},
                        },
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 303,
                        "max_tokens": 64,
                    }
                    payloads["tool"] = tool_payload

                    def request_tool() -> JsonObject:
                        body = _post_chat(
                            client,
                            chat_url,
                            tool_payload,
                            artifact_dir / "tool-response.json",
                        )
                        choices = body.get("choices")
                        assert isinstance(choices, list) and choices
                        message = _json_object(choices[0].get("message"), "tool message")
                        tool_calls = message.get("tool_calls")
                        _require(
                            isinstance(tool_calls, list) and tool_calls,
                            "model returned no tool calls",
                        )
                        assert isinstance(tool_calls, list)
                        function = _json_object(
                            tool_calls[0].get("function"), "tool call function"
                        )
                        _require(
                            function.get("name") == "get_temperature",
                            "model called the wrong tool",
                        )
                        arguments = function.get("arguments")
                        _require(isinstance(arguments, str), "tool arguments are not JSON text")
                        assert isinstance(arguments, str)
                        _json_object(json.loads(arguments), "tool arguments")
                        return body

                    tool_value = _record_check(
                        checks,
                        failures,
                        "tool_chat",
                        request_tool,
                    )
                    if isinstance(tool_value, dict):
                        responses["tool"] = tool_value

            response_ids = {
                response_id
                for response in responses.values()
                for response_id in [response.get("id")]
                if isinstance(response_id, str)
            }
            traces_path = home / "traces" / "traces.jsonl"
            traces_value = _record_check(
                checks,
                failures,
                "trace_capture",
                lambda: _wait_for_traces(traces_path, response_ids),
            )
            traces = traces_value if isinstance(traces_value, dict) else {}
            if traces:
                selected = [traces[response_id] for response_id in sorted(traces)]
                (artifact_dir / "traces-current-run.jsonl").write_text(
                    "".join(json.dumps(trace, sort_keys=True) + "\n" for trace in selected),
                    encoding="utf-8",
                )
                for request_name, response in responses.items():
                    response_id = response.get("id")
                    if not isinstance(response_id, str) or response_id not in traces:
                        continue
                    trace = traces[response_id]
                    choices = response.get("choices")
                    expected_content: str | None = None
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message")
                        if isinstance(message, dict) and isinstance(
                            message.get("content"), str
                        ):
                            expected_content = message["content"]
                    elif request_name == "stream":
                        content = response.get("content")
                        if isinstance(content, str):
                            expected_content = content
                    _record_check(
                        checks,
                        failures,
                        f"trace_{request_name}",
                        partial(
                            _verify_trace,
                            trace,
                            response,
                            payloads[request_name],
                            expected_content=expected_content,
                        ),
                    )

                nonstream_response = responses.get("nonstream")
                if nonstream_response is not None:
                    nonstream_id = nonstream_response.get("id")
                    if isinstance(nonstream_id, str) and nonstream_id in traces:
                        _record_check(
                            checks,
                            failures,
                            "template_structure",
                            lambda: _verify_template(
                                cell,
                                traces[nonstream_id],
                                artifact_dir / "rendered-template.txt",
                            ),
                        )
                tool_response = responses.get("tool")
                if tool_response is not None:
                    tool_id = tool_response.get("id")
                    if isinstance(tool_id, str) and tool_id in traces:
                        _record_check(
                            checks,
                            failures,
                            "tool_trace_self_contained",
                            lambda: _verify_tool_trace_is_self_contained(
                                traces[tool_id]
                            ),
                        )

            status_value = _record_check(
                checks,
                failures,
                "status_command",
                lambda: _run_cli_json(
                    home,
                    ["status", "--json"],
                    artifact_dir / "speedlm-status.json",
                ),
            )
            if isinstance(status_value, tuple):
                status, status_returncode = status_value
                _record_check(
                    checks,
                    failures,
                    "status_exit_code",
                    lambda: _require(
                        status_returncode == 0,
                        f"speedlm status exited {status_returncode}",
                    ),
                )
                _record_check(
                    checks,
                    failures,
                    "status_profile",
                    lambda: _verify_status(
                        cell,
                        status,
                        gateway_pid=process.pid,
                        descendant_pids=descendants,
                    ),
                )

            doctor_value = _record_check(
                checks,
                failures,
                "doctor_command",
                lambda: _run_cli_json(
                    home,
                    ["doctor", "--json"],
                    artifact_dir / "speedlm-doctor.json",
                ),
            )
            if isinstance(doctor_value, tuple):
                doctor, doctor_returncode = doctor_value
                results["doctor_returncode"] = doctor_returncode
                _record_check(
                    checks,
                    failures,
                    "doctor_profile",
                    lambda: _verify_doctor_profile(cell, doctor),
                )

            if upstream_url is not None:
                after_value = _record_check(
                    checks,
                    failures,
                    "metrics_after",
                    lambda: _scrape_metrics(
                        upstream_url, artifact_dir / "metrics-after.prom"
                    ),
                )
                if isinstance(after_value, str):
                    metrics_after = after_value
            if metrics_before is not None and metrics_after is not None:
                _write_metrics_summary(
                    metrics_before,
                    metrics_after,
                    artifact_dir / "acceptance-metrics.json",
                )
    finally:
        if not descendants and process.poll() is None:
            descendants = _descendant_pids(process.pid)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10)
        log_handle.close()
        (artifact_dir / "shutdown.txt").write_text(
            "\n".join(
                [
                    f"gateway_pid: {process.pid}",
                    f"observed_descendant_pids: {sorted(descendants)}",
                    f"gateway_returncode: {returncode}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        try:
            _assert_processes_gone(descendants)
        except AssertionError as exc:
            checks["process_cleanup"] = {"passed": False, "error": str(exc)}
            failures.append(f"process_cleanup: {exc}")
        else:
            checks["process_cleanup"] = {"passed": True}

    results["ready"] = ready
    results["failures"] = failures
    results["passed"] = not failures
    _write_json(artifact_dir / "verification.json", results)
    if failures:
        pytest.fail(
            f"matrix cell {cell.name!r} falsified {len(failures)} contract(s):\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
