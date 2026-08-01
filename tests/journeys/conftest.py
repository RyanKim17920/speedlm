from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = REPOSITORY_ROOT / ".venv" / "bin"
SPEEDLM = VENV_BIN / "speedlm"


@dataclass(frozen=True)
class CliResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


@dataclass
class GatewaySession:
    process: subprocess.Popen[str]
    url: str
    home: Path
    stdout: str = ""
    stderr: str = ""


@pytest.fixture(scope="session", autouse=True)
def actual_speedlm_binary_exists() -> None:
    assert SPEEDLM.is_file(), f"project CLI is not installed at {SPEEDLM}"
    assert os.access(SPEEDLM, os.X_OK), f"project CLI is not executable at {SPEEDLM}"


@pytest.fixture()
def speedlm_home(tmp_path: Path) -> Path:
    return tmp_path / "fresh-speedlm-home"


def cli_environment(
    home: Path,
    *,
    path_prefix: Path | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["SPEEDLM_HOME"] = str(home)
    env["PYTHONUNBUFFERED"] = "1"
    path_parts = []
    if path_prefix is not None:
        path_parts.append(str(path_prefix))
    path_parts.append(str(VENV_BIN))
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_parts)
    if overrides is not None:
        env.update(overrides)
    return env


def run_cli(
    home: Path,
    *args: str,
    path_prefix: Path | None = None,
    timeout: float = 30.0,
) -> CliResult:
    completed = subprocess.run(
        [str(SPEEDLM), *args],
        cwd=REPOSITORY_ROOT,
        env=cli_environment(home, path_prefix=path_prefix),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CliResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def assert_clean_cli_result(
    result: CliResult,
    *,
    expected_codes: Sequence[int] = (0,),
) -> None:
    command = " ".join(result.args)
    assert result.returncode in expected_codes, (
        f"speedlm {command} exited {result.returncode}\n{result.output}"
    )
    assert result.output.strip(), f"speedlm {command} produced no output"
    assert "Traceback (most recent call last)" not in result.output


#: The child ``vllm serve`` stand-in the journeys launch on PATH.
#:
#: Routing is on the *parsed* path, not on the raw request target.  The gateway
#: proxy builds its upstream URL with ``httpx.URL.copy_with(query=b"")``, which
#: renders an empty query as a trailing ``?`` -- so a proxied ``GET /health``
#: arrives here as ``GET /health?``.  Real vLLM is an ASGI app and routes it
#: correctly; a ``BaseHTTPRequestHandler`` that compares ``self.path`` to
#: ``"/health"`` does not, and answers 404 to every request the gateway
#: forwards.  That, not any sandbox restriction, is why these journeys were
#: skipped.
_FAKE_VLLM_SOURCE = '''#!/usr/bin/env python3
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def option(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]


host = option("--host")
port = int(option("--port"))
model = sys.argv[2]

# Cumulative, monotonic, and reset only by restarting this process -- the same
# contract real vLLM counters have.
counters = {"prompt_tokens": 0.0, "generation_tokens": 0.0, "decode_seconds": 0.0,
            "drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
sleeping = [False]


def exposition():
    labels = '{engine="0",model_name="%s"}' % model
    return "".join(
        "# TYPE %s %s\\n%s%s %s\\n" % (name, kind, name, labels, value)
        for name, kind, value in (
            ("vllm:prompt_tokens_total", "counter", counters["prompt_tokens"]),
            ("vllm:generation_tokens_total", "counter", counters["generation_tokens"]),
            ("vllm:request_decode_time_seconds_sum", "histogram", counters["decode_seconds"]),
            ("vllm:spec_decode_num_drafts_total", "counter", counters["drafts"]),
            ("vllm:spec_decode_num_draft_tokens_total", "counter", counters["draft_tokens"]),
            ("vllm:spec_decode_num_accepted_tokens_total", "counter", counters["accepted_tokens"]),
            ("vllm:num_requests_running", "gauge", 0.0),
            ("vllm:num_requests_waiting", "gauge", 0.0),
        )
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    @property
    def route(self):
        # See _FAKE_VLLM_SOURCE: the gateway forwards "/health?", not "/health".
        return urlsplit(self.path).path

    def send_json(self, status, payload):
        self.send_body(status, "application/json", json.dumps(payload).encode())

    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.route
        if route == "/health":
            self.send_json(200 if not sleeping[0] else 503, {"status": "ok"})
        elif route == "/metrics":
            self.send_body(200, "text/plain; version=0.0.4", exposition().encode())
        elif route == "/is_sleeping":
            self.send_json(200, {"is_sleeping": sleeping[0]})
        elif route == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        route = self.route
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length)) if length else {}
        if route == "/sleep":
            sleeping[0] = True
            self.send_json(200, {"status": "ok"})
            return
        if route == "/wake_up":
            sleeping[0] = False
            self.send_json(200, {"status": "ok"})
            return
        if route == "/collective_rpc":
            self.send_json(200, {"results": [None]})
            return
        if route != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        counters["prompt_tokens"] += 4
        counters["generation_tokens"] += 2
        counters["decode_seconds"] += 0.01
        counters["drafts"] += 1
        counters["draft_tokens"] += 4
        counters["accepted_tokens"] += 2
        self.send_json(
            200,
            {
                "id": "chatcmpl-journey",
                "created": 1700000000,
                "model": request.get("model", model),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )


signal.signal(signal.SIGTERM, lambda _signum, _frame: os._exit(0))
server = ThreadingHTTPServer((host, port), Handler)
server.serve_forever()
'''


@pytest.fixture()
def fake_vllm_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    executable = bin_dir / "vllm"
    executable.write_text(_FAKE_VLLM_SOURCE, encoding="utf-8")
    executable.chmod(0o755)
    return bin_dir


def reserve_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:
        # Kept as a *precise*, self-diagnosing skip rather than a blanket one:
        # loopback binding works on this host, and the journeys that were once
        # skipped for it were actually failing for an unrelated fixture bug.
        pytest.skip(f"host forbids binding a loopback socket ({exc}); serve journeys need one")


def request_json(
    url: str,
    *,
    payload: Mapping[str, object] | None = None,
    timeout: float = 2.0,
) -> tuple[int, dict[str, object]]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
        return response.status, body


def wait_until_ready(session: GatewaySession, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "gateway did not answer"
    while time.monotonic() < deadline:
        if session.process.poll() is not None:
            stdout, stderr = session.process.communicate()
            pytest.fail(
                f"serve exited {session.process.returncode} before readiness\n{stdout}{stderr}"
            )
        try:
            status, _ = request_json(f"{session.url}/health", timeout=0.5)
            if status == 200:
                return
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.05)
    pytest.fail(f"gateway was not ready within {timeout:g}s: {last_error}")


@contextmanager
def running_gateway(
    home: Path,
    fake_bin: Path,
    *,
    enable_idle_tuning: bool = False,
) -> Iterator[GatewaySession]:
    port = reserve_port()
    args = [
        str(SPEEDLM),
        "vllm",
        "serve",
        "journey-model",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if enable_idle_tuning:
        args.append("--enable-idle-tuning")
    process = subprocess.Popen(
        args,
        cwd=REPOSITORY_ROOT,
        env=cli_environment(home, path_prefix=fake_bin),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session = GatewaySession(
        process=process,
        url=f"http://127.0.0.1:{port}",
        home=home,
    )
    try:
        wait_until_ready(session)
        yield session
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            session.stdout, session.stderr = process.communicate(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            session.stdout, session.stderr = process.communicate(timeout=5.0)


def send_chat(session: GatewaySession) -> dict[str, object]:
    status, body = request_json(
        f"{session.url}/v1/chat/completions",
        payload={
            "model": "journey-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        },
    )
    assert status == 200
    return body


def wait_for_trace(home: Path, *, count: int = 1, timeout: float = 10.0) -> CliResult:
    deadline = time.monotonic() + timeout
    last = run_cli(home, "traces", "stats")
    while f"count    : {count}" not in last.stdout and time.monotonic() < deadline:
        time.sleep(0.05)
        last = run_cli(home, "traces", "stats")
    return last
