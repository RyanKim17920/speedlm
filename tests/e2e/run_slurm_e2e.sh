#!/usr/bin/env bash
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
artifacts="${SPEEDLM_E2E_ARTIFACT_DIR:-$repo/log_artifacts}"
mode="${1:-all}"
project_python="$repo/.venv/bin/python"

cd "$repo"
mkdir -p "$artifacts"

# Create "$1", or the first free "$1-runN" if it already exists, and echo the
# path that was created. Reruns therefore never blend into (or overwrite) a
# previous run's artifacts; each run gets its own directory.
unique_dir() {
  local base="$1"
  if mkdir "$base" 2>/dev/null; then
    printf '%s\n' "$base"
    return 0
  fi
  local n=2
  while (( n <= 999 )); do
    if mkdir "$base-run$n" 2>/dev/null; then
      printf '%s\n' "$base-run$n"
      return 0
    fi
    n=$((n + 1))
  done
  echo "could not allocate a free artifact directory under $base" >&2
  return 1
}

# Same idea for a single file: echo "$1$2", or the first free "$1-runN$2".
unique_file() {
  local base="$1" ext="$2"
  if [[ ! -e "$base$ext" ]]; then
    printf '%s\n' "$base$ext"
    return 0
  fi
  local n=2
  while (( n <= 999 )); do
    if [[ ! -e "$base-run$n$ext" ]]; then
      printf '%s\n' "$base-run$n$ext"
      return 0
    fi
    n=$((n + 1))
  done
  echo "could not allocate a free artifact file under $base$ext" >&2
  return 1
}

echo "hostname: $(hostname -f)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:?missing SLURM node list}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?no allocated GPU visible}"
nvidia-smi
"$project_python" --version
/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python -c \
  'import vllm; print(f"vllm {vllm.__version__}")'

run_stage1() {
  export SPEEDLM_E2E=1
  export SPEEDLM_E2E_MODEL=Qwen/Qwen3.5-2B
  export SPEEDLM_E2E_STAGE=stage1-qwen
  export SPEEDLM_FIDELITY_STAGE=stage1-token-fidelity-qwen
  export SPEEDLM_E2E_ARTIFACT_DIR="$artifacts"

  # vLLM passthrough args for stage 1 (Qwen/Qwen3.5-2B).
  #
  # --gdn-prefill-backend triton is load-bearing, not an optimization.
  # Qwen3.5-2B uses GDN linear attention. On an H100 (SM90) vLLM picks the
  # FlashInfer GDN prefill kernel, which is JIT-compiled on first use. vLLM
  # warns about this itself, in
  # vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:235:
  #   "FlashInfer GDN prefill is JIT-compiled; first run may take a while.
  #    Set --gdn-prefill-backend triton to skip JIT."
  # Measured cost of ignoring it (log_artifacts/stage1-qwen/gateway-and-vllm.log):
  # 10m33s of complete log silence between the last progress line at 05:05:48
  # ("Enabled custom fusions: norm_quant, act_quant") and the harness readiness
  # SIGTERM at 05:16:21. Weights load in ~2.2s; the JIT is the entire delay.
  # That burned four H100 runs. The Triton/FLA GDN kernels ship prebuilt, so
  # selecting them removes the JIT entirely.
  #
  # Flag verified against the pinned vLLM 0.25.1 in
  # /admin/home/ryan.kim/speedlm/.preflight/venvs/vllm:
  #   vllm/engine/arg_utils.py:1586 "--gdn-prefill-backend"
  #   choices: flashinfer | triton | cutedsl
  #
  # Override SPEEDLM_E2E_VLLM_ARGS (a JSON array of strings) to serve stage 1
  # with different vLLM flags without editing this file.
  export SPEEDLM_E2E_VLLM_ARGS="${SPEEDLM_E2E_VLLM_ARGS:-$(cat <<'JSON'
[
  "--max-model-len", "4096",
  "--gpu-memory-utilization", "0.85",
  "--enforce-eager",
  "--gdn-prefill-backend", "triton"
]
JSON
  )}"

  # Seconds to wait for the gateway to answer /health. Raise this (rather than
  # editing the file) when a model needs an unusually long cold start.
  export SPEEDLM_E2E_READY_TIMEOUT="${SPEEDLM_E2E_READY_TIMEOUT:-360}"

  unset SPEEDLM_E2E_CAPTURE_METRICS

  local live_log fidelity_log
  live_log="$(unique_file "$artifacts/stage1-pytest-terminal" .txt)"
  fidelity_log="$(unique_file "$artifacts/stage1-token-fidelity-terminal" .txt)"

  "$project_python" -m pytest tests/e2e/test_live_vllm.py -q -s 2>&1 |
    tee "$live_log"
  "$project_python" -m pytest tests/e2e/test_token_fidelity.py -q -s 2>&1 |
    tee "$fidelity_log"
}

run_stage2() (
  local stage_artifacts
  stage_artifacts="$(unique_dir "$artifacts/stage2-gpt-oss-eagle3")" || return 1
  local stage_home="$stage_artifacts/speedlm_home"
  local gateway_log="$stage_artifacts/gateway-and-vllm.log"
  local gateway_port
  gateway_port="$("$project_python" -c \
    'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  local gateway_url="http://127.0.0.1:$gateway_port"
  local model="openai/gpt-oss-20b"
  local draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3"
  local speculative_config
  speculative_config="{\"model\":\"$draft_model\",\"method\":\"eagle3\",\"num_speculative_tokens\":5}"

  export SPEEDLM_HOME="$stage_home"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export PYTHONUNBUFFERED=1
  export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"

  local command=(
    "$repo/.venv/bin/speedlm" vllm serve "$model"
    --host 127.0.0.1
    --port "$gateway_port"
    --max-model-len 4096
    --gpu-memory-utilization 0.90
    --enforce-eager
    --speculative-config "$speculative_config"
  )
  {
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\nnode: %s\n' "$(hostname -f)"
    printf 'SLURM_JOB_ID: %s\n' "$SLURM_JOB_ID"
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
  } > "$stage_artifacts/command.txt"

  setsid "${command[@]}" > "$gateway_log" 2>&1 &
  local gateway_pid=$!
  local upstream_port=""
  cleanup_stage2() {
    if kill -0 "$gateway_pid" 2>/dev/null; then
      kill -TERM -- "-$gateway_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        if ! kill -0 "$gateway_pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "$gateway_pid" 2>/dev/null; then
        kill -KILL -- "-$gateway_pid" 2>/dev/null || true
      fi
    fi
    wait "$gateway_pid" 2>/dev/null || true
  }
  trap cleanup_stage2 EXIT

  local ready=0
  for _ in $(seq 1 300); do
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      echo "stage 2 gateway exited before readiness" >&2
      return 1
    fi
    if curl --fail --silent --max-time 2 "$gateway_url/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "stage 2 gateway did not become ready within 600 seconds" >&2
    return 1
  fi

  for _ in $(seq 1 100); do
    upstream_port="$(sed -n \
      's/.*launching vLLM on 127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' \
      "$gateway_log" | tail -n 1)"
    if [[ -n "$upstream_port" ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ -z "$upstream_port" ]]; then
    echo "could not discover child vLLM metrics port" >&2
    return 1
  fi

  curl --fail --silent --show-error \
    "http://127.0.0.1:$upstream_port/metrics" \
    --output "$stage_artifacts/metrics-before.prom"

  local prompts=(
    "Explain in detail why deterministic testing is useful for distributed systems."
    "Write a numbered list of twelve practical ways to improve Python service reliability."
    "Describe the lifecycle of an HTTP request through a reverse proxy and model server."
    "Compare speculative decoding with ordinary autoregressive decoding in four paragraphs."
  )
  local request_index=0
  for prompt in "${prompts[@]}"; do
    request_index=$((request_index + 1))
    "$project_python" - "$gateway_url" "$model" "$prompt" \
      "$stage_artifacts/traffic-$request_index.json" <<'PY'
import json
import sys
import urllib.request

url, model, prompt, output_path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": 128,
}
request = urllib.request.Request(
    f"{url}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    body = json.load(response)
with open(output_path, "w", encoding="utf-8") as artifact:
    json.dump({"request": payload, "response": body}, artifact, indent=2, sort_keys=True)
    artifact.write("\n")
PY
  done

  curl --fail --silent --show-error \
    "http://127.0.0.1:$upstream_port/metrics" \
    --output "$stage_artifacts/metrics-after.prom"

  "$project_python" - \
    "$stage_artifacts/metrics-before.prom" \
    "$stage_artifacts/metrics-after.prom" \
    "$stage_artifacts/acceptance-summary.json" \
    "$stage_artifacts/acceptance-summary.txt" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

before_path, after_path, json_path, text_path = map(Path, sys.argv[1:])
line_re = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


def counters(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = line_re.match(raw_line.strip())
        if match is None:
            continue
        name, raw_value = match.groups()
        if not name.startswith("vllm:spec_decode_"):
            continue
        totals[name] = totals.get(name, 0.0) + float(raw_value)
    return totals


before = counters(before_path)
after = counters(after_path)
names = sorted(set(before) | set(after))
deltas = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in names}


def find(suffix: str) -> float:
    return sum(value for name, value in deltas.items() if name.endswith(suffix))


drafts = find("spec_decode_num_drafts_total")
draft_tokens = find("spec_decode_num_draft_tokens_total")
accepted_tokens = find("spec_decode_num_accepted_tokens_total")
required_suffixes = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)
counters_present = {
    suffix: any(name.endswith(suffix) for name in after)
    for suffix in required_suffixes
}
per_position = {
    name: value
    for name, value in deltas.items()
    if "spec_decode_num_accepted_tokens_per_pos" in name
}
summary = {
    "model": "openai/gpt-oss-20b",
    "draft_model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
    "method": "eagle3",
    "num_speculative_tokens": 5,
    "sampling": {"temperature": 0, "top_p": 1, "seed": 0},
    "traffic_requests": 4,
    "raw_metrics": {
        "before": str(before_path),
        "after": str(after_path),
    },
    "counter_before": before,
    "counter_after": after,
    "counter_delta": deltas,
    "parsed": {
        "required_counters_present": counters_present,
        "speculative_traffic_observed": drafts > 0 and draft_tokens > 0,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "mean_acceptance_length_including_bonus": (
            1 + accepted_tokens / drafts if drafts else None
        ),
        "accepted_tokens_per_position": per_position,
    },
}
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
parsed = summary["parsed"]
human = "\n".join(
    [
        "Stage 2 speculative decoding measurement",
        f"model: {summary['model']}",
        f"draft model: {summary['draft_model']}",
        "sampling: temperature=0, top_p=1, seed=0",
        f"required counters present: {parsed['required_counters_present']}",
        f"speculative traffic observed: {parsed['speculative_traffic_observed']}",
        f"drafts: {parsed['drafts']}",
        f"draft tokens: {parsed['draft_tokens']}",
        f"accepted tokens: {parsed['accepted_tokens']}",
        f"acceptance rate: {parsed['acceptance_rate']}",
        (
            "mean acceptance length including bonus: "
            f"{parsed['mean_acceptance_length_including_bonus']}"
        ),
    ]
) + "\n"
text_path.write_text(human, encoding="utf-8")
print(human, end="")
if not all(counters_present.values()):
    raise SystemExit("required vLLM speculative decoding counters were absent")
if not parsed["speculative_traffic_observed"]:
    raise SystemExit("traffic produced no speculative decoding drafts")
PY

  {
    printf 'gateway_pid: %s\n' "$gateway_pid"
    printf 'vllm_port: %s\n' "$upstream_port"
    printf 'traffic_requests: %s\n' "${#prompts[@]}"
  } > "$stage_artifacts/run-summary.txt"
)

stage2_terminal="$(unique_file "$artifacts/stage2-terminal" .txt)"

case "$mode" in
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2 2>&1 | tee "$stage2_terminal"
    ;;
  all)
    run_stage1
    run_stage2 2>&1 | tee "$stage2_terminal"
    ;;
  *)
    echo "usage: $0 [stage1|stage2|all]" >&2
    exit 2
    ;;
esac
