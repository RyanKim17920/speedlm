"""Benchmark SpeedLM proxy overhead against its own vLLM child.

Run this live-GPU test with ``tests/e2e/run_overhead_bench.sh``.  It starts
exactly one SpeedLM gateway (and therefore one vLLM child), then sends the
same deterministic workloads first to the child and then through the gateway.

Capture is intentionally enabled for every gateway request.  A trace-count
check after measurement verifies that the reported gateway numbers include
the normal asynchronous capture path.

Two ordering rules apply to everything after the measured phases:

* Measurement artifacts are written *before* any post-run assertion, so a
  failing check can never discard an H100 benchmark's data.
* Capture completeness is exact. Every accepted gateway response must have
  exactly one corresponding trace, including under peak concurrency.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

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

MODEL_DEFAULT = "Qwen/Qwen3.5-2B"
MAX_TOKENS = 64
WARMUP_COUNT = 3
LATENCY_REPEATS = 8
CONCURRENCY_REPEATS = 4
CONCURRENCY_LEVELS = (1, 4, 16, 32)
REQUEST_TIMEOUT_SECONDS = 300.0

MIN_CAPTURE_RATIO = 1.0

PROMPT = (
    "Explain how a reverse proxy forwards an inference request to a model "
    "server. Give a detailed, deterministic answer in numbered points."
)


def _require_live_e2e() -> None:
    if os.environ.get("SPEEDLM_E2E") != "1":
        pytest.skip("set SPEEDLM_E2E=1 inside an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), "must run inside a SLURM allocation"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "SLURM allocation has no GPU"
    assert SPEEDLM.is_file(), f"missing project CLI: {SPEEDLM}"
    assert VLLM.is_file(), f"missing vLLM CLI: {VLLM}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile (including for small samples)."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wait_for_http(
    url: str,
    process: subprocess.Popen[bytes],
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
                response = client.get(url)
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.25)
    raise AssertionError(
        f"server at {url} was not ready within {timeout}s; last error: {last_error}"
    )


def _wait_for_trace_count(path: Path, expected: int, timeout: float = 60.0) -> int:
    """Wait outside timed regions for asynchronous capture writes to finish.

    Returns the observed trace count.  This deliberately never raises: capture
    completeness is judged (and reported) by the caller *after* the measurement
    artifacts have been written, so a shortfall cannot destroy a benchmark run.
    """
    deadline = time.monotonic() + timeout
    actual = 0
    while time.monotonic() < deadline:
        if path.exists():
            actual = sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if actual >= expected:
                return actual
        time.sleep(0.1)
    return actual


def _payload(model: str, seed: int, *, stream: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "top_p": 0.85,
        "seed": seed,
        "max_tokens": MAX_TOKENS,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage")
    if not isinstance(usage, dict) or not isinstance(
        usage.get("completion_tokens"), int
    ):
        raise AssertionError(f"response has no completion-token usage: {body}")
    return body


def _measure_nonstream(
    client: httpx.Client,
    url: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for payload in payloads:
        started = time.perf_counter()
        body = _checked_json(client.post(url, json=payload))
        elapsed = time.perf_counter() - started
        completion_tokens = int(body["usage"]["completion_tokens"])
        samples.append(
            {
                "seed": payload["seed"],
                "e2e_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "throughput_tok_per_sec": completion_tokens / elapsed,
            }
        )
    return samples


def _measure_streaming(
    client: httpx.Client,
    url: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure first token and gaps between token-bearing SSE events.

    vLLM normally emits one generated token per content-bearing SSE event.
    The raw interval distribution is retained in the JSON artifact.  If an
    HTTP layer coalesces events, near-zero gaps faithfully expose that delivery
    behavior; completion-token throughput still uses authoritative API usage.
    """
    samples: list[dict[str, Any]] = []
    for payload in payloads:
        started = time.perf_counter()
        first_token_at: float | None = None
        previous_token_at: float | None = None
        inter_token_seconds: list[float] = []
        token_event_count = 0
        completion_tokens: int | None = None
        saw_done = False

        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                received_at = time.perf_counter()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    continue

                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int
                ):
                    completion_tokens = int(usage["completion_tokens"])

                token_bearing = False
                for choice in event.get("choices", []):
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    token_bearing = token_bearing or any(
                        isinstance(delta.get(field), str) and bool(delta[field])
                        for field in ("content", "reasoning_content")
                    )
                if not token_bearing:
                    continue

                token_event_count += 1
                if first_token_at is None:
                    first_token_at = received_at
                if previous_token_at is not None:
                    inter_token_seconds.append(received_at - previous_token_at)
                previous_token_at = received_at

        elapsed = time.perf_counter() - started
        if not saw_done:
            raise AssertionError("stream ended without [DONE]")
        if first_token_at is None:
            raise AssertionError("stream contained no token-bearing SSE event")
        if completion_tokens is None or completion_tokens <= 0:
            raise AssertionError("stream did not report positive completion-token usage")
        if not inter_token_seconds:
            raise AssertionError("stream contained fewer than two token-bearing SSE events")

        samples.append(
            {
                "seed": payload["seed"],
                "ttft_seconds": first_token_at - started,
                "e2e_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "token_event_count": token_event_count,
                "inter_token_seconds": inter_token_seconds,
                "throughput_tok_per_sec": completion_tokens / elapsed,
            }
        )
    return samples


async def _measure_concurrent_async(
    base_url: str,
    payload_batches: list[list[dict[str, Any]]],
    concurrency: int,
) -> list[dict[str, Any]]:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    samples: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        for repeat, payloads in enumerate(payload_batches):
            # Build every coroutine before starting the batch timer so client-side
            # request construction is consistent between direct and gateway runs.
            requests = [
                client.post(f"{base_url}/v1/chat/completions", json=payload)
                for payload in payloads
            ]
            started = time.perf_counter()
            responses = await asyncio.gather(*requests)
            elapsed = time.perf_counter() - started
            bodies = [_checked_json(response) for response in responses]
            total_tokens = sum(
                int(body["usage"]["completion_tokens"]) for body in bodies
            )
            samples.append(
                {
                    "repeat": repeat,
                    "seeds": [payload["seed"] for payload in payloads],
                    "concurrency": concurrency,
                    "elapsed_seconds": elapsed,
                    "total_completion_tokens": total_tokens,
                    "throughput_tok_per_sec": total_tokens / elapsed,
                }
            )
    return samples


def _measure_concurrent(
    base_url: str,
    payload_batches: list[list[dict[str, Any]]],
    concurrency: int,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _measure_concurrent_async(base_url, payload_batches, concurrency)
    )


def _comparison(
    direct_values: list[float],
    gateway_values: list[float],
    *,
    worse_when: Literal["higher", "lower"],
) -> dict[str, Any]:
    """Summarize absolute values and overhead, with positive meaning worse."""
    direct = {
        "median": _percentile(direct_values, 50),
        "p95": _percentile(direct_values, 95),
    }
    gateway = {
        "median": _percentile(gateway_values, 50),
        "p95": _percentile(gateway_values, 95),
    }
    absolute: dict[str, float] = {}
    percentage: dict[str, float] = {}
    for statistic in ("median", "p95"):
        if worse_when == "higher":
            difference = gateway[statistic] - direct[statistic]
        else:
            difference = direct[statistic] - gateway[statistic]
        absolute[statistic] = difference
        percentage[statistic] = (
            difference / direct[statistic] * 100.0
            if direct[statistic] > 0
            else 0.0
        )
    return {
        "direct": direct,
        "gateway": gateway,
        "absolute_overhead": absolute,
        "percentage_overhead": percentage,
    }


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


def _latencies(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [float(sample[key]) for sample in samples]


def _throughputs(samples: list[dict[str, Any]]) -> list[float]:
    return _latencies(samples, "throughput_tok_per_sec")


def _inter_token_latencies(samples: list[dict[str, Any]]) -> list[float]:
    return [
        float(interval)
        for sample in samples
        for interval in sample["inter_token_seconds"]
    ]


def _format_latency_row(label: str, stats: dict[str, Any]) -> str:
    direct = stats["direct"]
    gateway = stats["gateway"]
    absolute = stats["absolute_overhead"]
    percentage = stats["percentage_overhead"]
    return (
        f"{label:<18}"
        f"{direct['median'] * 1000:>11.2f}"
        f"{direct['p95'] * 1000:>11.2f}"
        f"{gateway['median'] * 1000:>11.2f}"
        f"{gateway['p95'] * 1000:>11.2f}"
        f"{absolute['median'] * 1000:>11.2f}"
        f"{absolute['p95'] * 1000:>11.2f}"
        f"{percentage['median']:>10.2f}%"
        f"{percentage['p95']:>9.2f}%"
    )


def _format_throughput_row(label: str, stats: dict[str, Any]) -> str:
    direct = stats["direct"]
    gateway = stats["gateway"]
    absolute = stats["absolute_overhead"]
    percentage = stats["percentage_overhead"]
    return (
        f"{label:<18}"
        f"{direct['median']:>11.2f}"
        f"{direct['p95']:>11.2f}"
        f"{gateway['median']:>11.2f}"
        f"{gateway['p95']:>11.2f}"
        f"{absolute['median']:>11.2f}"
        f"{absolute['p95']:>11.2f}"
        f"{percentage['median']:>10.2f}%"
        f"{percentage['p95']:>9.2f}%"
    )


def _summary_text(results: dict[str, Any]) -> str:
    metadata = results["metadata"]
    header = (
        f"{'Metric':<18}"
        f"{'Direct med':>11}"
        f"{'Direct p95':>11}"
        f"{'Proxy med':>11}"
        f"{'Proxy p95':>11}"
        f"{'Abs med':>11}"
        f"{'Abs p95':>11}"
        f"{'OH med':>11}"
        f"{'OH p95':>10}"
    )
    lines = [
        "SpeedLM proxy-overhead benchmark",
        f"Model: {metadata['model']}",
        (
            f"Warmups/path: {metadata['warmup_requests_per_path']}; "
            f"latency repeats: {metadata['latency_repeats']}; "
            f"concurrency repeats: {metadata['concurrency_repeats']}"
        ),
        (
            f"Gateway capture: enabled; "
            f"{metadata['captured_trace_count']}/"
            f"{metadata['expected_captured_trace_count']} traces "
            f"({metadata['capture_ratio'] * 100:.2f}% of requests), "
            f"missing {metadata['missing_captured_traces']} "
            f"(tolerance: >= {metadata['min_capture_ratio'] * 100:.2f}%)"
        ),
        "",
        "Latency (milliseconds; positive overhead means slower)",
        header,
        "-" * len(header),
        _format_latency_row("TTFT", results["ttft"]),
        _format_latency_row("Nonstream E2E", results["nonstream_e2e"]),
        _format_latency_row("Stream E2E", results["stream_e2e"]),
        _format_latency_row("Inter-token", results["inter_token_latency"]),
        "",
        "Throughput (completion tok/s; positive overhead means throughput loss)",
        header,
        "-" * len(header),
        _format_throughput_row(
            "Sequential", results["sequential_throughput"]
        ),
    ]
    for concurrency in metadata["concurrency_levels"]:
        lines.append(
            _format_throughput_row(
                f"Concurrent {concurrency}",
                results["concurrent_throughput"][str(concurrency)],
            )
        )
    lines.extend(
        [
            "",
            "Concurrency median gateway/direct ratios",
            "  N      ratio",
        ]
    )
    for concurrency in metadata["concurrency_levels"]:
        ratio = results["concurrent_throughput"][str(concurrency)][
            "gateway_to_direct_median_ratio"
        ]
        lines.append(f"{concurrency:>3}    {ratio:>7.4f}")
    lines.extend(
        [
            "",
            "Median and p95 use linear interpolation. Absolute overhead is "
            "gateway-direct for latency and direct-gateway for throughput.",
            "The performance assertion is that every concurrency level retains "
            "at least 50% of direct median throughput.",
            "The capture assertion requires an exact one-to-one count for "
            "every accepted gateway request.",
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.e2e
def test_proxy_overhead_benchmark() -> None:
    _require_live_e2e()

    model = os.environ.get("SPEEDLM_E2E_MODEL", MODEL_DEFAULT)
    stage = os.environ.get("SPEEDLM_E2E_STAGE", "proxy-overhead")
    passthrough = json.loads(os.environ.get("SPEEDLM_E2E_VLLM_ARGS", "[]"))
    assert isinstance(passthrough, list) and all(
        isinstance(argument, str) for argument in passthrough
    )

    artifact_root_raw = os.environ.get("SPEEDLM_E2E_ARTIFACT_DIR")
    assert artifact_root_raw, "SPEEDLM_E2E_ARTIFACT_DIR is required"
    artifact_dir = Path(artifact_root_raw) / stage
    artifact_dir.mkdir(parents=True, exist_ok=False)

    speedlm_home = artifact_dir / "speedlm_home"
    traces_path = speedlm_home / "traces" / "traces.jsonl"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    gateway_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"

    # Keep this explicit: omitting it can trigger a >900s FlashInfer JIT compile.
    command = [
        str(SPEEDLM),
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        "--gdn-prefill-backend",
        "triton",
        *passthrough,
    ]
    (artifact_dir / "command.txt").write_text(
        " ".join(command)
        + "\n"
        + f"node: {socket.gethostname()}\n"
        + f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}\n"
        + f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "SPEEDLM_HOME": str(speedlm_home),
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
        _wait_for_http(f"{gateway_url}/health", process, timeout=360.0)
        observed_pids = _descendant_pids(process.pid)
        assert observed_pids, "speedlm did not have a live vLLM child"

        gateway_log_text = gateway_log.read_text(
            encoding="utf-8", errors="replace"
        )
        port_match = re.search(
            r"launching vLLM on 127\.0\.0\.1:(\d+)",
            gateway_log_text,
        )
        assert port_match, "could not find child vLLM port in gateway log"
        vllm_port = int(port_match.group(1))
        vllm_url = f"http://127.0.0.1:{vllm_port}"
        direct_chat_url = f"{vllm_url}/v1/chat/completions"
        gateway_chat_url = f"{gateway_url}/v1/chat/completions"
        _wait_for_http(f"{vllm_url}/health", process, timeout=10.0)

        client_timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)

        # Warm up each path three times and exclude all warmups from timing.
        warmup_payloads = [
            _payload(model, seed=100 + index) for index in range(WARMUP_COUNT)
        ]
        with httpx.Client(timeout=client_timeout, trust_env=False) as client:
            for payload in warmup_payloads:
                _checked_json(client.post(direct_chat_url, json=payload))
            for payload in warmup_payloads:
                _checked_json(client.post(gateway_chat_url, json=payload))
        # Avoid carrying asynchronous warmup capture I/O into the direct
        # baseline.
        _wait_for_trace_count(traces_path, WARMUP_COUNT, timeout=30.0)

        nonstream_payloads = [
            _payload(model, seed=1_000 + index)
            for index in range(LATENCY_REPEATS)
        ]
        stream_payloads = [
            _payload(model, seed=2_000 + index, stream=True)
            for index in range(LATENCY_REPEATS)
        ]
        concurrent_batches = {
            concurrency: [
                [
                    _payload(
                        model,
                        seed=10_000 + concurrency * 1_000 + repeat * 100 + index,
                    )
                    for index in range(concurrency)
                ]
                for repeat in range(CONCURRENCY_REPEATS)
            ]
            for concurrency in CONCURRENCY_LEVELS
        }

        benchmark_started = time.perf_counter()

        # Phase A: every measured workload goes directly to the one vLLM child.
        with httpx.Client(timeout=client_timeout, trust_env=False) as client:
            direct_nonstream = _measure_nonstream(
                client, direct_chat_url, nonstream_payloads
            )
            direct_stream = _measure_streaming(
                client, direct_chat_url, stream_payloads
            )
        direct_concurrent = {
            str(concurrency): _measure_concurrent(
                vllm_url,
                concurrent_batches[concurrency],
                concurrency,
            )
            for concurrency in CONCURRENCY_LEVELS
        }

        # Phase B: repeat the exact payloads through the SpeedLM gateway.
        with httpx.Client(timeout=client_timeout, trust_env=False) as client:
            gateway_nonstream = _measure_nonstream(
                client, gateway_chat_url, nonstream_payloads
            )
            gateway_stream = _measure_streaming(
                client, gateway_chat_url, stream_payloads
            )
        gateway_concurrent = {
            str(concurrency): _measure_concurrent(
                gateway_url,
                concurrent_batches[concurrency],
                concurrency,
            )
            for concurrency in CONCURRENCY_LEVELS
        }

        benchmark_seconds = time.perf_counter() - benchmark_started

        raw_samples = {
            "direct": {
                "nonstream": direct_nonstream,
                "stream": direct_stream,
                "concurrent": direct_concurrent,
            },
            "gateway": {
                "nonstream": gateway_nonstream,
                "stream": gateway_stream,
                "concurrent": gateway_concurrent,
            },
        }
        # Land the raw measurements first: nothing below this line -- summary
        # formatting, capture accounting, or any assertion -- may discard them.
        _write_json(artifact_dir / "proxy_overhead_raw_samples.json", raw_samples)

        expected_captures = (
            WARMUP_COUNT
            + LATENCY_REPEATS
            + LATENCY_REPEATS
            + CONCURRENCY_REPEATS * sum(CONCURRENCY_LEVELS)
        )
        captured_trace_count = _wait_for_trace_count(
            traces_path,
            expected_captures,
        )
        missing_captures = max(expected_captures - captured_trace_count, 0)
        capture_ratio = (
            captured_trace_count / expected_captures if expected_captures else 1.0
        )

        results: dict[str, Any] = {
            "metadata": {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "warmup_requests_per_path": WARMUP_COUNT,
                "latency_repeats": LATENCY_REPEATS,
                "concurrency_repeats": CONCURRENCY_REPEATS,
                "concurrency_levels": list(CONCURRENCY_LEVELS),
                "gateway_capture_enabled": True,
                "captured_trace_count": captured_trace_count,
                "expected_captured_trace_count": expected_captures,
                "missing_captured_traces": missing_captures,
                "capture_ratio": capture_ratio,
                "min_capture_ratio": MIN_CAPTURE_RATIO,
                "benchmark_seconds": benchmark_seconds,
                "direct_vllm_port": vllm_port,
                "gateway_port": gateway_port,
                "percentile_method": "linear interpolation",
            },
            "ttft": _comparison(
                _latencies(direct_stream, "ttft_seconds"),
                _latencies(gateway_stream, "ttft_seconds"),
                worse_when="higher",
            ),
            "nonstream_e2e": _comparison(
                _latencies(direct_nonstream, "e2e_seconds"),
                _latencies(gateway_nonstream, "e2e_seconds"),
                worse_when="higher",
            ),
            "stream_e2e": _comparison(
                _latencies(direct_stream, "e2e_seconds"),
                _latencies(gateway_stream, "e2e_seconds"),
                worse_when="higher",
            ),
            "inter_token_latency": _comparison(
                _inter_token_latencies(direct_stream),
                _inter_token_latencies(gateway_stream),
                worse_when="higher",
            ),
            "sequential_throughput": _comparison(
                _throughputs(direct_nonstream),
                _throughputs(gateway_nonstream),
                worse_when="lower",
            ),
            "concurrent_throughput": {},
            "raw_samples": raw_samples,
        }

        for concurrency in CONCURRENCY_LEVELS:
            key = str(concurrency)
            comparison = _comparison(
                _throughputs(direct_concurrent[key]),
                _throughputs(gateway_concurrent[key]),
                worse_when="lower",
            )
            direct_median = comparison["direct"]["median"]
            gateway_median = comparison["gateway"]["median"]
            comparison["gateway_to_direct_median_ratio"] = (
                gateway_median / direct_median if direct_median > 0 else 0.0
            )
            results["concurrent_throughput"][key] = comparison

        # Emit every artifact before any post-run assertion.  A benchmark that
        # discards its own measurements on a trailing check is worse than none.
        _write_json(artifact_dir / "proxy_overhead.json", results)
        summary = _summary_text(results)
        (artifact_dir / "proxy_overhead_summary.txt").write_text(
            summary,
            encoding="utf-8",
        )
        print(summary, end="")

        assert captured_trace_count == expected_captures, (
            f"capture is not an exact bijection: {captured_trace_count}/"
            f"{expected_captures} traces, missing {missing_captures}"
        )

        # Performance must fail only on the requested serialization signal.
        for concurrency in CONCURRENCY_LEVELS:
            stats = results["concurrent_throughput"][str(concurrency)]
            ratio = stats["gateway_to_direct_median_ratio"]
            assert ratio >= 0.5, (
                f"possible proxy serialization at concurrency={concurrency}: "
                f"gateway median throughput "
                f"{stats['gateway']['median']:.2f} tok/s is {ratio:.2%} of "
                f"direct {stats['direct']['median']:.2f} tok/s"
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
        (artifact_dir / "shutdown.txt").write_text(
            f"gateway_pid: {process.pid}\n"
            f"observed_descendant_pids: {sorted(observed_pids)}\n"
            f"gateway_returncode: {returncode}\n",
            encoding="utf-8",
        )
        _assert_processes_gone(observed_pids)
