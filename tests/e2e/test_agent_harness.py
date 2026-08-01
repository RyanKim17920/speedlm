"""Full-circle live agent test: gateway -> tools -> traces -> training masks."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
DEFAULT_MODEL = "openai/gpt-oss-20b"
SUCCESS_MARKER = "SPEEDLM_AGENT_SUCCESS"
# Deliberately unguessable: the agent cannot produce `total` without actually
# reading input.json, so a passing run proves the tool round trip happened.
INPUT_VALUES = [317, 1289, 7, 4096]
RESULT = {
    "total": sum(INPUT_VALUES),
    "count": len(INPUT_VALUES),
    "status": "verified",
}
JsonObject = dict[str, Any]

AGENT_TOOLS: list[JsonObject] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to read.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a file in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to write.",
                    },
                    "content": {"type": "string", "description": "Complete file content."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]

TASK_PROMPT = f"""
Work only in the current directory. You must complete these steps with tools:
1. Call read_file to inspect input.json; do not guess its contents.
2. Compute the sum and count of its "values" array.
3. Call write_file to create result.json containing a JSON object with exactly
   the keys "total", "count", and "status"; status must be "verified".
4. After the file is written, reply with exactly {SUCCESS_MARKER}.
Do not use shell commands and do not merely describe the result.
""".strip()

QWEN_SYSTEM_PROMPT = """
You are a deterministic file-task agent. Follow the user's requested tool sequence.
Use only the tools made available. Use workspace-relative paths. Keep the final reply exact.
""".strip()


@dataclass(frozen=True, slots=True)
class AgentRun:
    kind: str
    command: tuple[str, ...]
    returncode: int
    output: str
    discovered: Mapping[str, str]
    final_text: str
    advertised_tools: tuple[str, ...]
    summary: Mapping[str, Any]


def _require_live_e2e() -> None:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), "live E2E must run inside a SLURM allocation"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible and visible != "-1", "SLURM allocation did not expose a GPU"
    assert SPEEDLM.is_file(), f"missing project CLI: {SPEEDLM}"
    assert VLLM.is_file(), f"missing vLLM CLI: {VLLM}"


def _float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be a number, got {raw!r}") from exc
    assert value > 0, f"{name} must be positive, got {raw!r}"
    return value


def _unique_artifact_dir(root: Path, stage: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"-run{attempt}"
        candidate = root / f"{stage}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError(f"could not allocate an artifact directory below {root}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.extend(
        [
            Path("/admin/home/ryan.kim/.cache/huggingface/hub"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    return tuple(dict.fromkeys(roots))


def _model_dir_name(model: str) -> str:
    return "models--" + model.replace("/", "--")


def _snapshot_for_model(model: str) -> Path:
    checked: list[Path] = []
    for cache_root in _model_cache_roots():
        model_root = cache_root / _model_dir_name(model)
        checked.append(model_root)
        main_ref = model_root / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = model_root / "snapshots" / revision
            if (snapshot / "tokenizer.json").is_file():
                return snapshot
        snapshots = model_root / "snapshots"
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir(), reverse=True):
                if (snapshot / "tokenizer.json").is_file():
                    return snapshot
    raise AssertionError(
        f"no complete cached snapshot for {model!r}; checked "
        + ", ".join(str(path) for path in checked)
    )


def _default_vllm_args(model: str) -> list[str]:
    args = [
        "--max-model-len",
        os.environ.get("SPEEDLM_AGENT_MAX_MODEL_LEN", "16384"),
        "--gpu-memory-utilization",
        os.environ.get("SPEEDLM_AGENT_GPU_MEMORY_UTILIZATION", "0.90"),
        "--enforce-eager",
    ]
    if model.startswith("Qwen/Qwen3.5"):
        args.extend(["--gdn-prefill-backend", "triton"])
    return args


def _vllm_args(model: str) -> list[str]:
    raw = os.environ.get("SPEEDLM_AGENT_VLLM_ARGS")
    if raw is None:
        return _default_vllm_args(model)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError("SPEEDLM_AGENT_VLLM_ARGS must be a JSON array") from exc
    assert isinstance(parsed, list) and all(isinstance(item, str) for item in parsed)
    return parsed


def _wait_for_gateway(
    gateway_url: str,
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
                raise AssertionError(f"speedlm exited before readiness with code {returncode}")
            try:
                response = client.get(f"{gateway_url}/health")
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.25)
    raise AssertionError(
        f"gateway did not become ready within {timeout}s; last error: {last_error}"
    )


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


def _discover_agent_clis() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for name in ("qwen-cli", "qwen-code", "qwen", "aider", "opencode", "claude", "codex"):
        path = shutil.which(name)
        if path is not None:
            discovered[name] = path
    return discovered


def _select_agent(discovered: Mapping[str, str]) -> tuple[str, str | None]:
    requested = os.environ.get("SPEEDLM_AGENT_CLI")
    if requested == "scripted":
        return "scripted", None
    if requested:
        path = shutil.which(requested)
        if path is None and Path(requested).is_file():
            path = str(Path(requested).resolve())
        assert path is not None, f"SPEEDLM_AGENT_CLI is not executable: {requested}"
        assert Path(path).name in {"qwen-cli", "qwen-code", "qwen"}, (
            "only Qwen Code exposes the required headless OpenAI tool interface; "
            "set SPEEDLM_AGENT_CLI=scripted for the fallback"
        )
        return "qwen", path
    for name in ("qwen-cli", "qwen-code", "qwen"):
        if name in discovered:
            return "qwen", discovered[name]
    return "scripted", None


def _parse_qwen_events(output: str) -> list[JsonObject]:
    """Extract the ``--output-format json`` event array from qwen's stdout.

    qwen prints human-readable warnings before the JSON array, so the payload
    cannot be parsed with a bare ``json.loads`` of the whole stream.
    """
    for index, char in enumerate(output):
        if char != "[":
            continue
        try:
            value = json.loads(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _qwen_summary(events: Sequence[JsonObject]) -> tuple[str, tuple[str, ...], JsonObject]:
    """Return ``(final_text, advertised_tools, summary)`` from qwen JSON events."""
    final_text = ""
    advertised: tuple[str, ...] = ()
    summary: JsonObject = {
        "events": len(events),
        "api_requests": 0,
        "api_errors": 0,
        "tool_calls": 0,
        "tool_calls_by_name": {},
        "num_turns": 0,
        "is_error": None,
        "subtype": None,
    }
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            tools = event.get("tools")
            if isinstance(tools, list):
                advertised = tuple(item for item in tools if isinstance(item, str))
        if event.get("type") != "result":
            continue
        result = event.get("result")
        if isinstance(result, str) and result.strip():
            final_text = result
        summary["is_error"] = event.get("is_error")
        summary["subtype"] = event.get("subtype")
        summary["num_turns"] = event.get("num_turns", 0)
        stats = event.get("stats")
        if not isinstance(stats, dict):
            continue
        tools_stats = stats.get("tools")
        if isinstance(tools_stats, dict):
            summary["tool_calls"] = tools_stats.get("totalCalls", 0)
            by_name = tools_stats.get("byName")
            if isinstance(by_name, dict):
                summary["tool_calls_by_name"] = {
                    name: entry.get("totalCalls", entry) if isinstance(entry, dict) else entry
                    for name, entry in by_name.items()
                }
        models = stats.get("models")
        if isinstance(models, dict):
            for entry in models.values():
                api = entry.get("api") if isinstance(entry, dict) else None
                if isinstance(api, dict):
                    summary["api_requests"] += int(api.get("totalRequests", 0) or 0)
                    summary["api_errors"] += int(api.get("totalErrors", 0) or 0)
    return final_text, advertised, summary


def _run_qwen_agent(
    executable: str,
    *,
    gateway_url: str,
    model: str,
    workspace: Path,
    artifact_dir: Path,
    discovered: Mapping[str, str],
) -> AgentRun:
    # NOTE: `--bare` is deliberately opt-in. qwen-code 0.19.x drops
    # `argv.coreTools`/settings-derived allow lists in bare mode
    # (`resolvedCoreTools = [...bareMode ? [] : argv.coreTools ?? []]`), so
    # `--core-tools read_file write_file` silently did nothing: the 2026-07 live
    # run advertised `read_file, edit, notebook_edit, run_shell_command` and had
    # no `write_file` at all, making the task literally impossible. We instead
    # isolate HOME so no user settings leak in, and let --core-tools apply.
    bare = os.environ.get("SPEEDLM_AGENT_QWEN_BARE", "0") == "1"
    agent_home = artifact_dir / "agent_home"
    agent_home.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        *(["--bare"] if bare else []),
        "--auth-type",
        "openai",
        "--openai-api-key",
        "speedlm-e2e",
        "--openai-base-url",
        f"{gateway_url}/v1",
        "--model",
        model,
        "--system-prompt",
        QWEN_SYSTEM_PROMPT,
        "--approval-mode",
        "yolo",
        "--core-tools",
        "read_file",
        "write_file",
        "--allowed-tools",
        "read_file",
        "write_file",
        "--exclude-tools",
        "run_shell_command",
        "notebook_edit",
        "--output-format",
        "json",
        "--max-session-turns",
        "8",
        "--max-tool-calls",
        "6",
        "--max-wall-time",
        os.environ.get("SPEEDLM_AGENT_TIMEOUT", "6m"),
        TASK_PROMPT,
    ]
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "speedlm-e2e",
            "OPENAI_BASE_URL": f"{gateway_url}/v1",
            "OPENAI_MODEL": model,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            # Hermetic settings/context discovery without losing --core-tools.
            "HOME": str(agent_home),
            "XDG_CONFIG_HOME": str(agent_home / "config"),
            "XDG_CACHE_HOME": str(agent_home / "cache"),
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
            # Qwen Code otherwise assumes a 65,536-token output budget for an
            # unknown OpenAI-compatible model alias, which exceeds many valid
            # vLLM context windows before the first agent turn can run.
            "QWEN_CODE_MAX_OUTPUT_TOKENS": os.environ.get(
                "SPEEDLM_AGENT_MAX_OUTPUT_TOKENS",
                "1024",
            ),
        }
    )
    timeout = _float_env("SPEEDLM_AGENT_SUBPROCESS_TIMEOUT", "420")
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        (artifact_dir / "agent-output.txt").write_text(partial, encoding="utf-8")
        raise AssertionError(
            f"qwen agent exceeded SPEEDLM_AGENT_SUBPROCESS_TIMEOUT={timeout}s; "
            f"last {len(partial)} bytes of output written to "
            f"{artifact_dir / 'agent-output.txt'}"
        ) from exc
    (artifact_dir / "agent-output.txt").write_text(completed.stdout, encoding="utf-8")
    events = _parse_qwen_events(completed.stdout)
    _write_json(artifact_dir / "agent-events.json", events)
    final_text, advertised, summary = _qwen_summary(events)
    return AgentRun(
        kind="qwen",
        command=tuple(command),
        returncode=completed.returncode,
        output=completed.stdout,
        discovered=dict(discovered),
        final_text=final_text,
        advertised_tools=advertised,
        summary=summary,
    )


def _chat(
    client: httpx.Client,
    *,
    gateway_url: str,
    model: str,
    messages: list[JsonObject],
    tool_choice: object,
) -> JsonObject:
    payload: JsonObject = {
        "model": model,
        "messages": messages,
        "tools": AGENT_TOOLS,
        "tool_choice": tool_choice,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 4242,
        "max_tokens": 256,
    }
    response = client.post(f"{gateway_url}/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict), body
    usage = body.get("usage")
    assert isinstance(usage, dict), body
    assert isinstance(usage.get("prompt_tokens"), int), body
    assert isinstance(usage.get("completion_tokens"), int), body
    return body


def _first_message(response: Mapping[str, Any]) -> JsonObject:
    choices = response.get("choices")
    assert isinstance(choices, list) and choices, response
    message = choices[0].get("message")
    assert isinstance(message, dict), response
    return dict(message)


def _one_tool_call(message: Mapping[str, Any], expected_name: str) -> JsonObject:
    calls = message.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 1, message
    call = calls[0]
    assert isinstance(call, dict), message
    function = call.get("function")
    assert isinstance(function, dict), message
    assert function.get("name") == expected_name, message
    arguments = function.get("arguments")
    assert isinstance(arguments, str), message
    parsed = json.loads(arguments)
    assert isinstance(parsed, dict), message
    return dict(call)


def _safe_workspace_path(workspace: Path, raw_path: object, expected_name: str) -> Path:
    assert isinstance(raw_path, str) and raw_path, f"tool path is invalid: {raw_path!r}"
    candidate = (workspace / raw_path).resolve()
    assert candidate.parent == workspace.resolve(), f"tool escaped workspace: {raw_path!r}"
    assert candidate.name == expected_name, f"tool used unexpected path: {raw_path!r}"
    return candidate


def _scripted_agent(
    *,
    gateway_url: str,
    model: str,
    workspace: Path,
    artifact_dir: Path,
    discovered: Mapping[str, str],
) -> AgentRun:
    messages: list[JsonObject] = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROMPT},
    ]
    transcript: list[JsonObject] = []
    with httpx.Client(timeout=240.0, trust_env=False) as client:
        read_response = _chat(
            client,
            gateway_url=gateway_url,
            model=model,
            messages=messages,
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )
        transcript.append(read_response)
        read_message = _first_message(read_response)
        read_call = _one_tool_call(read_message, "read_file")
        read_args = json.loads(read_call["function"]["arguments"])
        input_path = _safe_workspace_path(workspace, read_args.get("path"), "input.json")
        messages.extend(
            [
                read_message,
                {
                    "role": "tool",
                    "tool_call_id": read_call["id"],
                    "name": "read_file",
                    "content": input_path.read_text(encoding="utf-8"),
                },
            ]
        )

        write_response = _chat(
            client,
            gateway_url=gateway_url,
            model=model,
            messages=messages,
            tool_choice={"type": "function", "function": {"name": "write_file"}},
        )
        transcript.append(write_response)
        write_message = _first_message(write_response)
        write_call = _one_tool_call(write_message, "write_file")
        write_args = json.loads(write_call["function"]["arguments"])
        result_path = _safe_workspace_path(workspace, write_args.get("path"), "result.json")
        content = write_args.get("content")
        assert isinstance(content, str), write_args
        parsed_content = json.loads(content)
        assert parsed_content == RESULT, parsed_content
        result_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        messages.extend(
            [
                write_message,
                {
                    "role": "tool",
                    "tool_call_id": write_call["id"],
                    "name": "write_file",
                    "content": "Wrote result.json successfully.",
                },
            ]
        )

        final_response = _chat(
            client,
            gateway_url=gateway_url,
            model=model,
            messages=messages,
            tool_choice="none",
        )
        transcript.append(final_response)
        final_message = _first_message(final_response)
        output = final_message.get("content")
        assert isinstance(output, str), final_message
    _write_json(artifact_dir / "scripted-agent-transcript.json", transcript)
    (artifact_dir / "agent-output.txt").write_text(output + "\n", encoding="utf-8")
    return AgentRun(
        kind="scripted",
        command=(),
        returncode=0,
        output=output,
        discovered=dict(discovered),
        final_text=output,
        advertised_tools=tuple(tool["function"]["name"] for tool in AGENT_TOOLS),
        summary={
            "events": len(transcript),
            "api_requests": len(transcript),
            "api_errors": 0,
            "tool_calls": 2,
            "tool_calls_by_name": {"read_file": 1, "write_file": 1},
            "num_turns": len(transcript),
            "is_error": False,
            "subtype": "scripted",
        },
    )


def _run_agent(
    *,
    gateway_url: str,
    model: str,
    workspace: Path,
    artifact_dir: Path,
) -> AgentRun:
    discovered = _discover_agent_clis()
    kind, executable = _select_agent(discovered)
    if kind == "qwen":
        assert executable is not None
        run = _run_qwen_agent(
            executable,
            gateway_url=gateway_url,
            model=model,
            workspace=workspace,
            artifact_dir=artifact_dir,
            discovered=discovered,
        )
    else:
        run = _scripted_agent(
            gateway_url=gateway_url,
            model=model,
            workspace=workspace,
            artifact_dir=artifact_dir,
            discovered=discovered,
        )
    _write_json(
        artifact_dir / "agent-selection.json",
        {
            "selected": run.kind,
            "discovered": run.discovered,
            "command": list(run.command),
            "returncode": run.returncode,
            "advertised_tools": list(run.advertised_tools),
            "summary": dict(run.summary),
        },
    )
    return run


def _load_traces(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        assert isinstance(value, dict), f"trace line {line_number} is not an object"
        records.append(value)
    return records


def _wait_for_traces(path: Path, *, timeout: float = 30.0) -> tuple[list[JsonObject], bool]:
    """Poll ``path`` until the multi-turn tool session is fully captured.

    Returns ``(records, complete)`` instead of raising so the caller can attach
    the agent-side evidence that says *why* a run fell short.
    """
    deadline = time.monotonic() + timeout
    records: list[JsonObject] = []
    last_count = -1
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        if path.is_file():
            records = _load_traces(path)
            if len(records) != last_count:
                last_count = len(records)
                stable_since = time.monotonic()
            terminal = any(
                isinstance(trace.get("messages"), list)
                and trace["messages"]
                and isinstance(trace["messages"][-1].get("content"), str)
                and SUCCESS_MARKER in trace["messages"][-1]["content"]
                for trace in records
            )
            if len(records) >= 3 and terminal and time.monotonic() - stable_since >= 0.5:
                return records, True
        time.sleep(0.1)
    return records, False


def _trace_shape(traces: Sequence[JsonObject]) -> list[str]:
    """One human-readable line per captured trace, for failure reports."""
    lines: list[str] = []
    for index, trace in enumerate(traces):
        messages = trace.get("messages")
        roles = (
            [str(message.get("role")) for message in messages] if isinstance(messages, list) else []
        )
        calls = trace.get("tool_calls")
        names = (
            [call.get("function", {}).get("name") for call in calls if isinstance(call, dict)]
            if isinstance(calls, list)
            else []
        )
        lines.append(
            f"  trace[{index}] messages={roles} tool_calls={names} "
            f"finish_reason={trace.get('finish_reason')!r} "
            f"token_count_source={trace.get('token_count_source')!r}"
        )
    return lines


def _tail(path: Path, *, limit: int = 4000) -> str:
    if not path.is_file():
        return "<missing>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:] if len(text) > limit else text


def _failure_report(
    reason: str,
    *,
    agent_run: AgentRun,
    traces: Sequence[JsonObject],
    traces_path: Path,
    gateway_log: Path,
    workspace: Path,
    vllm_argv: Sequence[str],
) -> str:
    """Tell the next operator which layer broke, not just that something did."""
    tool_calls = agent_run.summary.get("tool_calls", 0)
    captured_calls = sum(
        len(trace.get("tool_calls") or []) for trace in traces if isinstance(trace, dict)
    )
    if not traces:
        verdict = (
            "NO TRACES CAPTURED -- the proxy never wrote a trace; capture or the "
            "gateway is broken (check the gateway log below)."
        )
    elif tool_calls == 0 and captured_calls == 0:
        verdict = (
            "AGENT NEVER CALLED A TOOL -- capture worked but the model emitted no "
            "tool_calls. Check that vLLM was started with --enable-auto-tool-choice "
            "and a --tool-call-parser (and --reasoning-parser for gpt-oss, without "
            "which message.content comes back empty)."
        )
    elif captured_calls == 0:
        verdict = (
            "AGENT CALLED TOOLS BUT CAPTURE MISSED THEM -- the CLI reports tool "
            "calls while the traces record none; the capture path is broken."
        )
    else:
        verdict = "TOOL CALLING WORKED -- the failure is downstream of tool use."
    return "\n".join(
        [
            f"agent harness failure: {reason}",
            f"verdict: {verdict}",
            f"agent kind: {agent_run.kind} returncode={agent_run.returncode}",
            f"agent advertised tools: {list(agent_run.advertised_tools)}",
            f"agent summary: {json.dumps(dict(agent_run.summary), sort_keys=True)}",
            f"vllm args: {shlex.join(str(item) for item in vllm_argv)}",
            f"captured traces: {len(traces)} at {traces_path}",
            *_trace_shape(traces),
            f"workspace contents: {sorted(p.name for p in workspace.iterdir())}",
            "--- agent final text ---",
            agent_run.final_text or "<empty>",
            "--- agent stdout (tail) ---",
            agent_run.output[-4000:] or "<empty>",
            "--- gateway + vLLM log (tail) ---",
            _tail(gateway_log),
        ]
    )


def _call_details(message: Mapping[str, Any]) -> list[tuple[str, str, JsonObject]]:
    details: list[tuple[str, str, JsonObject]] = []
    calls = message.get("tool_calls", [])
    assert isinstance(calls, list), message
    for call in calls:
        assert isinstance(call, dict) and call.get("type") == "function", call
        call_id = call.get("id")
        function = call.get("function")
        assert isinstance(call_id, str) and call_id, call
        assert isinstance(function, dict), call
        name = function.get("name")
        arguments = function.get("arguments")
        assert isinstance(name, str) and name, call
        assert isinstance(arguments, str), call
        parsed = json.loads(arguments)
        assert isinstance(parsed, dict), call
        details.append((call_id, name, parsed))
    return details


def _validate_trace_completeness(
    traces: Sequence[JsonObject],
    *,
    model: str,
    advertised_tools: Sequence[str] = (),
) -> JsonObject:
    required = {
        "id",
        "timestamp",
        "model",
        "messages",
        "tool_calls",
        "temperature",
        "top_p",
        "seed",
        "prompt_tokens",
        "completion_tokens",
        "token_count_source",
        "finish_reason",
        "stop_reason",
    }
    ids: set[str] = set()
    calls: dict[str, str] = {}
    observed_tool_results: set[str] = set()
    generated_call_names: list[str] = []
    terminal_count = 0

    for trace_index, trace in enumerate(traces):
        missing = required - trace.keys()
        assert not missing, f"trace {trace_index} is missing {sorted(missing)}"
        trace_id = trace["id"]
        assert isinstance(trace_id, str) and trace_id and trace_id not in ids
        ids.add(trace_id)
        assert trace["model"] == model
        assert isinstance(trace["timestamp"], (int, float)) and trace["timestamp"] > 0
        assert trace["token_count_source"] == "measured"
        assert isinstance(trace["prompt_tokens"], int) and trace["prompt_tokens"] > 0
        assert isinstance(trace["completion_tokens"], int) and trace["completion_tokens"] > 0
        messages = trace["messages"]
        assert isinstance(messages, list) and messages
        assert messages[-1].get("role") == "assistant"
        assert messages[-1].get("provenance_tag") == "generated"
        assert all(
            message.get("provenance_tag") == "client_supplied"
            for message in messages[:-1]
        )
        for message in messages:
            assert isinstance(message, dict)
            assert isinstance(message.get("role"), str) and message["role"]
            assert "content" in message
            for call_id, name, _ in _call_details(message):
                previous_name = calls.setdefault(call_id, name)
                assert previous_name == name, f"tool call {call_id!r} changed names"
                if message is messages[-1]:
                    generated_call_names.append(name)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                assert isinstance(call_id, str) and call_id in calls, message
                name = message.get("name")
                if name is not None:
                    assert name == calls[call_id], message
                observed_tool_results.add(call_id)
        top_level_calls = trace["tool_calls"]
        assistant_calls = messages[-1].get("tool_calls", [])
        assert top_level_calls == assistant_calls
        if assistant_calls:
            assert trace["finish_reason"] == "tool_calls", trace
        else:
            assert isinstance(trace["finish_reason"], str) and trace["finish_reason"]
        content = messages[-1].get("content")
        if isinstance(content, str) and SUCCESS_MARKER in content:
            terminal_count += 1

    tool_names = {tool["function"]["name"] for tool in AGENT_TOOLS}
    # The real CLI owns its own tool registry; anything it advertised is a
    # legitimate name, anything else is a hallucination.
    allowed_names = tool_names | set(advertised_tools)
    assert {"read_file", "write_file"} <= set(generated_call_names), (
        "the agent did not exercise both the read and the write tool; "
        f"observed {sorted(set(generated_call_names))}"
    )
    assert set(generated_call_names) <= allowed_names, (
        f"agent invoked tools it was never offered: "
        f"{sorted(set(generated_call_names) - allowed_names)}"
    )
    assert len(calls) >= 2, calls
    assert calls.keys() <= observed_tool_results, (
        f"captured tool calls lack later tool results: {calls.keys() - observed_tool_results}"
    )
    assert terminal_count >= 1, "no terminal success assistant turn was captured"
    return {
        "trace_count": len(traces),
        "unique_ids": len(ids),
        "tool_call_count": len(calls),
        "tool_result_count": len(observed_tool_results),
        "tool_names": sorted(set(generated_call_names)),
        "terminal_success_traces": terminal_count,
        "all_required_fields_present": True,
        "complete_tool_lifecycle": True,
        "training_tool_schema_source": "harness agent adapter",
    }


@pytest.mark.e2e
def test_agent_harness_full_circle() -> None:
    _require_live_e2e()
    model = os.environ.get("SPEEDLM_AGENT_MODEL", DEFAULT_MODEL)
    snapshot = _snapshot_for_model(model)
    artifact_root = Path(
        os.environ.get(
            "SPEEDLM_AGENT_ARTIFACT_DIR",
            str(REPO_ROOT / "log_artifacts" / "agent-harness"),
        )
    )
    stage = os.environ.get("SPEEDLM_AGENT_STAGE", model.replace("/", "--"))
    artifact_dir = _unique_artifact_dir(artifact_root, stage)
    print(f"agent harness artifact directory: {artifact_dir}")

    home = artifact_dir / "speedlm_home"
    workspace = artifact_dir / "agent_workspace"
    workspace.mkdir()
    _write_json(workspace / "input.json", {"values": INPUT_VALUES})
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
        *_vllm_args(model),
    ]
    (artifact_dir / "command.txt").write_text(
        "\n".join(
            [
                f"command: {shlex.join(command)}",
                f"node: {socket.gethostname()}",
                f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}",
                f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}",
                f"vllm_executable: {VLLM}",
                f"tokenizer_snapshot: {snapshot}",
                "parser_args: selected by SpeedLM runtime discovery",
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
    try:
        _wait_for_gateway(
            gateway_url,
            process,
            timeout=_float_env("SPEEDLM_AGENT_STARTUP_TIMEOUT", "900"),
        )
        descendants = _descendant_pids(process.pid)
        assert descendants, "speedlm did not have a live vLLM child"

        agent_run = _run_agent(
            gateway_url=gateway_url,
            model=model,
            workspace=workspace,
            artifact_dir=artifact_dir,
        )
        traces_path = home / "traces" / "traces.jsonl"

        def _fail(reason: str) -> str:
            traces_so_far = _load_traces(traces_path) if traces_path.is_file() else []
            report = _failure_report(
                reason,
                agent_run=agent_run,
                traces=traces_so_far,
                traces_path=traces_path,
                gateway_log=gateway_log,
                workspace=workspace,
                vllm_argv=command,
            )
            (artifact_dir / "failure-report.txt").write_text(report, encoding="utf-8")
            return report

        assert agent_run.returncode == 0, _fail(
            f"agent CLI exited with code {agent_run.returncode}"
        )
        assert agent_run.summary.get("tool_calls", 0) >= 2, _fail(
            "agent finished without making at least two tool calls -- the "
            "multi-turn tool-calling session this test exists to measure never "
            "happened"
        )
        marker_text = agent_run.final_text or agent_run.output
        assert SUCCESS_MARKER in marker_text, _fail(
            f"agent never emitted the {SUCCESS_MARKER} marker"
        )
        result_path = workspace / "result.json"
        assert result_path.is_file(), _fail(f"agent did not create {result_path}")
        assert json.loads(result_path.read_text(encoding="utf-8")) == RESULT, _fail(
            "result.json does not match the value derived from input.json"
        )

        traces, complete = _wait_for_traces(traces_path)
        assert complete, _fail(
            "the captured trace stream never reached >=3 traces ending in a "
            "terminal success turn"
        )
        (artifact_dir / "traces.jsonl").write_text(
            traces_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        completeness = _validate_trace_completeness(
            traces,
            model=model,
            advertised_tools=agent_run.advertised_tools,
        )
        _write_json(artifact_dir / "trace-completeness.json", completeness)
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
            f"observed_descendant_pids: {sorted(descendants)}\n"
            f"gateway_returncode: {returncode}\n",
            encoding="utf-8",
        )
        _assert_processes_gone(descendants)
