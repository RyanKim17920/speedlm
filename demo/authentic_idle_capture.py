#!/usr/bin/env python3
"""Capture a real two-GPU vLLM-vs-SpeedLM idle lifecycle.

The output is a pair of timestamped JSONL timelines plus the unfiltered server
logs.  Every lifecycle event is observed from a running process, an HTTP
response, SpeedLM's events.jsonl, or the trainer's stdout log.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-8B"
REPO = Path(__file__).resolve().parents[1]
VLLM = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/vllm")
SPEEDLM = REPO / ".venv/bin/speedlm"
SEED_TRACES = Path("/data/ryan.kim/speedlm-runs/cyclev2-run1/speedlm_home/traces/traces.jsonl")
BASE_CONFIG = REPO / "demo/demo-qwen8b.json"
LOSS = re.compile(r"train/loss=([0-9.]+)")


class Capture:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.handles = {
            side: (out / f"{side}-timeline.jsonl").open("w", encoding="utf-8")
            for side in ("normal", "speedlm")
        }

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def emit(self, side: str, kind: str, text: str, **data: Any) -> None:
        payload = {
            "t": round(time.time() - self.started, 3),
            "timestamp": time.time(),
            "side": side,
            "kind": kind,
            "text": text,
            **data,
        }
        handle = self.handles[side]
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def wait_http(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness: code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server was not ready after {timeout:.0f}s: {url}")


def chat(url: str) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Reply with exactly this text and nothing else: "
                    "SpeedLM serves the same vLLM API"
                ),
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 48,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def response_text(payload: dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"]["content"]).strip()


def stop_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=30)


def write_config(out: Path, base_port: int) -> tuple[Path, dict[str, int]]:
    config = json.loads(BASE_CONFIG.read_text())
    ports = {
        "normal": base_port,
        "target": base_port + 1,
        "wrapper": base_port + 2,
        "training": base_port + 3,
    }
    config["target"] = {"host": "127.0.0.1", "port": ports["target"]}
    config["wrapper"] = {"host": "127.0.0.1", "port": ports["wrapper"]}
    config["tuning"]["training_port"] = ports["training"]
    path = out / "capture-config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path, ports


def spawn(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    handle = log_path.open("wb")
    try:
        process = subprocess.Popen(
            argv,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        handle.close()
        raise
    process._capture_log_handle = handle  # type: ignore[attr-defined]
    return process


def close_log(process: subprocess.Popen[bytes]) -> None:
    handle = getattr(process, "_capture_log_handle", None)
    if handle is not None:
        handle.close()


def latest_idle_line(log: Path) -> str:
    if not log.exists():
        return "server process remains running"
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if "Avg prompt throughput" in line and "Running: 0 reqs" in line:
            match = line.find("Avg prompt throughput")
            return line[match:].strip()
    return "server process remains running · no client traffic"


def wait_for_training(
    capture: Capture,
    home: Path,
    normal_process: subprocess.Popen[bytes],
    normal_log: Path,
    timeout: float,
) -> None:
    events = home / "runs/events.jsonl"
    seen_sequences: set[int] = set()
    training_seen = False
    deadline = time.monotonic() + timeout
    last_idle_observation = 0.0

    while time.monotonic() < deadline:
        if normal_process.poll() is not None:
            raise RuntimeError("normal vLLM exited while idle")

        if events.exists():
            for line in events.read_text(errors="replace").splitlines():
                event = json.loads(line)
                sequence = int(event["sequence"])
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
                to_state = str(event["to"])
                capture.emit(
                    "speedlm",
                    "state",
                    f"{event.get('from') or 'START'} → {to_state} · {event['reason']}",
                    state=to_state,
                    source="events.jsonl",
                    source_timestamp=event["timestamp"],
                )
                if to_state in {"QUIESCING", "SLEEPING", "EXTRACTING", "TRAINING"}:
                    capture.emit(
                        "normal",
                        "idle",
                        latest_idle_line(normal_log),
                        process_alive=True,
                        client_requests_since_stop=0,
                        paired_with_speedlm_state=to_state,
                    )
                if to_state == "TRAINING":
                    training_seen = True

        now = time.monotonic()
        if now - last_idle_observation >= 10:
            capture.emit(
                "normal",
                "idle",
                latest_idle_line(normal_log),
                process_alive=True,
                client_requests_since_stop=0,
            )
            last_idle_observation = now

        if training_seen:
            logs = sorted((home / "runs").glob("*/training-logs/stdout.log"))
            if logs:
                text = logs[-1].read_text(errors="replace")
                match = LOSS.search(text)
                if match:
                    capture.emit(
                        "speedlm",
                        "optimizer",
                        f"optimizer step · train/loss={match.group(1)}",
                        source=str(logs[-1]),
                    )
                    capture.emit(
                        "normal",
                        "idle",
                        latest_idle_line(normal_log),
                        process_alive=True,
                        client_requests_since_stop=0,
                        trainer_started=False,
                    )
                    return
        time.sleep(1)
    raise TimeoutError("SpeedLM did not reach a real optimizer step before timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=2700)
    args = parser.parse_args()

    visible = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]
    if len(visible) < 2:
        raise SystemExit("authentic capture requires two allocated GPUs")
    if not SEED_TRACES.is_file():
        raise SystemExit(f"missing seed corpus: {SEED_TRACES}")

    capture = Capture(args.out)
    normal_process: subprocess.Popen[bytes] | None = None
    speed_process: subprocess.Popen[bytes] | None = None
    try:
        job_id = int(os.environ.get("SLURM_JOB_ID", "400000"))
        config_path, ports = write_config(args.out, 20000 + (job_id % 20000))
        home = args.out / "speedlm-home"
        traces = home / "traces"
        traces.mkdir(parents=True)
        shutil.copyfile(SEED_TRACES, traces / "traces.jsonl")

        common_env = os.environ.copy()
        common_env.update(
            {
                "HF_HOME": "/data/ryan.kim/hf-cache",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        vllm_bin = str(VLLM)
        normal_argv = [
            vllm_bin,
            "serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(ports["normal"]),
            "--max-model-len",
            "24576",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ]
        speed_argv = [
            str(SPEEDLM),
            "vllm",
            "serve",
            MODEL,
            "--config",
            str(config_path),
            "--enable-idle-tuning",
            "--max-model-len",
            "24576",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ]

        normal_log = args.out / "normal-server.log"
        speed_log = args.out / "speedlm-server.log"
        capture.emit("normal", "command", " ".join(normal_argv), argv=normal_argv)
        normal_env = common_env | {"CUDA_VISIBLE_DEVICES": visible[0]}
        normal_process = spawn(normal_argv, normal_env, normal_log)
        capture.emit("normal", "process", "launching real vLLM server", pid=normal_process.pid)
        wait_http(f"http://127.0.0.1:{ports['normal']}/health", normal_process, timeout=1200)
        capture.emit("normal", "ready", "vLLM server READY · OpenAI-compatible API")

        capture.emit("speedlm", "command", " ".join(speed_argv), argv=speed_argv)
        speed_env = common_env | {
            "CUDA_VISIBLE_DEVICES": visible[1],
            "SPEEDLM_HOME": str(home),
            "PATH": f"{VLLM.parent}:{common_env['PATH']}",
        }
        speed_process = spawn(speed_argv, speed_env, speed_log)
        capture.emit(
            "speedlm", "process", "launching real SpeedLM vLLM server", pid=speed_process.pid
        )
        wait_http(f"http://127.0.0.1:{ports['wrapper']}/health", speed_process, timeout=1200)
        capture.emit("speedlm", "ready", "SpeedLM server READY · same OpenAI-compatible API")

        prompt = "Reply exactly: SpeedLM serves the same vLLM API"
        capture.emit("normal", "request", prompt)
        capture.emit("speedlm", "request", prompt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            normal_future = pool.submit(chat, f"http://127.0.0.1:{ports['normal']}")
            speed_future = pool.submit(chat, f"http://127.0.0.1:{ports['wrapper']}")
            normal_response = normal_future.result()
            speed_response = speed_future.result()

        normal_text = response_text(normal_response)
        speed_text = response_text(speed_response)
        capture.emit(
            "normal",
            "response",
            normal_text,
            usage=normal_response.get("usage", {}),
        )
        capture.emit(
            "speedlm",
            "response",
            speed_text,
            usage=speed_response.get("usage", {}),
        )
        identical = normal_text == speed_text
        capture.emit("normal", "comparison", "IDENTICAL OUTPUT" if identical else "OUTPUT MISMATCH")
        capture.emit(
            "speedlm", "comparison", "IDENTICAL OUTPUT" if identical else "OUTPUT MISMATCH"
        )
        if not identical:
            raise RuntimeError("the two real servers did not return identical greedy output")

        stopped_at = time.time()
        capture.emit(
            "normal",
            "traffic_stopped",
            "incoming traffic stopped · server remains running",
            stopped_at=stopped_at,
        )
        capture.emit(
            "speedlm",
            "traffic_stopped",
            "incoming traffic stopped · idle timer started",
            stopped_at=stopped_at,
        )
        wait_for_training(capture, home, normal_process, normal_log, args.timeout)
        capture.emit("normal", "complete", "capture complete · vLLM remained idle")
        capture.emit("speedlm", "complete", "capture complete · idle training verified")
        (args.out / "SUCCESS").write_text("authentic two-GPU capture completed\n")
    except BaseException as error:
        capture.emit("normal", "error", f"capture failed · {error}")
        capture.emit("speedlm", "error", f"capture failed · {error}")
        raise
    finally:
        if speed_process is not None:
            stop_group(speed_process)
            close_log(speed_process)
        if normal_process is not None:
            stop_group(normal_process)
            close_log(normal_process)
        capture.close()


if __name__ == "__main__":
    main()
