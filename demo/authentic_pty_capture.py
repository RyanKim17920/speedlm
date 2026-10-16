#!/usr/bin/env python3
"""Record a real vLLM/SpeedLM comparison in four real pseudo-terminals.

Two PTYs run the servers and therefore contain the commands and the server
processes' literal stdout/stderr.  After both health endpoints are ready, two
more PTYs run the same curl request and contain the literal HTTP responses.
The SpeedLM home is preconfigured before the recording, so the command shown to
the operator needs only the public opt-in flag: ``--enable-idle-tuning``.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"
RECORDER = REPO / "demo/session_record.py"
BASE_CONFIG = REPO / "demo/demo-qwen8b.json"
SEED_TRACES = Path("/data/ryan.kim/speedlm-runs/cyclev2-run1/speedlm_home/traces/traces.jsonl")
ANSI = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def wait_http(url: str, processes: tuple[subprocess.Popen[bytes], ...]) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"PTY recorder exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready: {url}")


def write_script(path: Path, *, prompt: str, steps: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "typing_delay": 0.003,
                "settle": 1.0,
                "prompt": prompt,
                "steps": steps,
            },
            indent=2,
        )
        + "\n"
    )


def record(
    script: Path, cast: Path, env: dict[str, str], *, cols: int, rows: int
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            str(PYTHON),
            str(RECORDER),
            "--script",
            str(script),
            "--out",
            str(cast),
            "--cwd",
            str(REPO),
            "--cols",
            str(cols),
            "--rows",
            str(rows),
        ],
        cwd=REPO,
        env=env,
    )


def cast_text(path: Path) -> str:
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            event = json.loads(line)
            if len(event) >= 3 and event[1] == "o":
                chunks.append(str(event[2]))
    return ANSI.sub("", "".join(chunks)).replace("\r", "")


def response_content(path: Path) -> str:
    payload = json.loads(path.read_text())
    return str(payload["choices"][0]["message"]["content"]).strip()


def wait_for_training(home: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        events = home / "runs/events.jsonl"
        if events.exists():
            states = [json.loads(line).get("to") for line in events.read_text().splitlines()]
            if "TRAINING" in states:
                return
        time.sleep(1)
    raise TimeoutError("SpeedLM did not reach TRAINING")


def main() -> None:
    out = Path(os.environ["CAPTURE_DIR"])
    out.mkdir(parents=True, exist_ok=False)
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) < 2:
        raise SystemExit("capture requires two allocated GPUs")

    home = out / "speedlm-home"
    (home / "traces").mkdir(parents=True)
    shutil.copyfile(SEED_TRACES, home / "traces/traces.jsonl")

    config = json.loads(BASE_CONFIG.read_text())
    # The flag, not the config, enables tuning in this proof.  The config only
    # supplies the installation-specific trainer paths and tuning policy.
    config["tuning_enabled"] = False
    (home / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    request = {
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
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    (out / "request.json").write_text(json.dumps(request, separators=(",", ":")) + "\n")

    write_script(
        out / "normal-server.json",
        prompt="\\[\\033[1;36m\\]normal-vllm\\[\\033[0m\\]$ ",
        steps=[
            {
                "command": "vllm serve Qwen/Qwen3-8B",
                "pause_after": 600,
            }
        ],
    )
    write_script(
        out / "speedlm-server.json",
        prompt="\\[\\033[1;36m\\]speedlm-vllm\\[\\033[0m\\]$ ",
        steps=[
            {
                "command": "speedlm vllm serve Qwen/Qwen3-8B --enable-idle-tuning",
                "pause_after": 600,
            }
        ],
    )

    curl_common = "-H 'Content-Type: application/json' --data-binary @\"$CAPTURE_DIR/request.json\""
    write_script(
        out / "normal-client.json",
        prompt="\\[\\033[1;36m\\]normal-client\\[\\033[0m\\]$ ",
        steps=[
            {"command": 'cat "$CAPTURE_DIR/request.json"', "pause_after": 1.0},
            {
                "command": (
                    "curl -sS http://127.0.0.1:8000/v1/chat/completions "
                    f'{curl_common} --output "$CAPTURE_DIR/normal-response.json" '
                    '&& cat "$CAPTURE_DIR/normal-response.json"'
                ),
                "pause_after": 2.0,
            },
        ],
    )
    write_script(
        out / "speedlm-client.json",
        prompt="\\[\\033[1;36m\\]speedlm-client\\[\\033[0m\\]$ ",
        steps=[
            {"command": 'cat "$CAPTURE_DIR/request.json"', "pause_after": 1.0},
            {
                "command": (
                    "curl -sS http://127.0.0.1:8150/v1/chat/completions "
                    f'{curl_common} --output "$CAPTURE_DIR/speedlm-response.json" '
                    '&& cat "$CAPTURE_DIR/speedlm-response.json"'
                ),
                "pause_after": 2.0,
            },
        ],
    )

    common = os.environ.copy()
    common.update(
        {
            "HF_HOME": "/data/ryan.kim/hf-cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "CAPTURE_DIR": str(out),
            "SPEEDLM_HOME": str(home),
        }
    )
    vllm_bin = "/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin"
    normal_env = common | {
        "CUDA_VISIBLE_DEVICES": visible[0],
        "PATH": f"{vllm_bin}:{common['PATH']}",
    }
    speed_env = common | {
        "CUDA_VISIBLE_DEVICES": visible[1],
        "PATH": f"{REPO / '.venv/bin'}:{vllm_bin}:{common['PATH']}",
    }

    normal_server = record(
        out / "normal-server.json", out / "normal-server.cast", normal_env, cols=100, rows=28
    )
    speed_server = record(
        out / "speedlm-server.json", out / "speedlm-server.cast", speed_env, cols=100, rows=28
    )
    servers = (normal_server, speed_server)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            normal_ready = pool.submit(wait_http, "http://127.0.0.1:8000/health", servers)
            speed_ready = pool.submit(wait_http, "http://127.0.0.1:8150/health", servers)
            normal_ready.result()
            speed_ready.result()

        client_env = common | {"PATH": f"/usr/bin:{common['PATH']}"}
        normal_client = record(
            out / "normal-client.json", out / "normal-client.cast", client_env, cols=88, rows=12
        )
        speed_client = record(
            out / "speedlm-client.json", out / "speedlm-client.cast", client_env, cols=88, rows=12
        )
        if normal_client.wait(timeout=120) != 0 or speed_client.wait(timeout=120) != 0:
            raise RuntimeError("a curl PTY failed")

        normal_text = response_content(out / "normal-response.json")
        speedlm_text = response_content(out / "speedlm-response.json")
        if normal_text != speedlm_text:
            raise RuntimeError(f"response mismatch: {normal_text!r} != {speedlm_text!r}")
        wait_for_training(home)
        (out / "VERIFIED.json").write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "normal_url": "http://127.0.0.1:8000",
                    "speedlm_url": "http://127.0.0.1:8150",
                    "identical_response": normal_text,
                    "normal_cast_contains_command": "vllm serve Qwen/Qwen3-8B"
                    in cast_text(out / "normal-server.cast")
                    if (out / "normal-server.cast").exists()
                    else False,
                    "speedlm_cast_contains_command": (
                        "speedlm vllm serve Qwen/Qwen3-8B --enable-idle-tuning"
                    )
                    in cast_text(out / "speedlm-server.cast")
                    if (out / "speedlm-server.cast").exists()
                    else False,
                    "training_state_observed": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for process in servers:
            try:
                process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=60)

    # Server casts are only finalized when their PTY sessions close, so verify
    # their literal commands after both recorders have flushed them.
    verified = json.loads((out / "VERIFIED.json").read_text())
    verified["normal_cast_contains_command"] = "vllm serve Qwen/Qwen3-8B" in cast_text(
        out / "normal-server.cast"
    )
    verified["speedlm_cast_contains_command"] = (
        "speedlm vllm serve Qwen/Qwen3-8B --enable-idle-tuning"
        in cast_text(out / "speedlm-server.cast")
    )
    verified["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    (out / "VERIFIED.json").write_text(json.dumps(verified, indent=2) + "\n")


if __name__ == "__main__":
    main()
