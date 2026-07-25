#!/usr/bin/env bash
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
project_python="$repo/.venv/bin/python"
vllm_python=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python
artifact_root="${SPEEDLM_AGENT_ARTIFACT_DIR:-$repo/log_artifacts/agent-harness}"

: "${SLURM_JOB_ID:?run inside a SLURM GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?the allocation did not expose a GPU}"

cd "$repo"
mkdir -p "$artifact_root"

export SPEEDLM_E2E=1
export SPEEDLM_AGENT_ARTIFACT_DIR="$artifact_root"
export SPEEDLM_AGENT_MODEL="${SPEEDLM_AGENT_MODEL:-openai/gpt-oss-20b}"
export SPEEDLM_AGENT_STARTUP_TIMEOUT="${SPEEDLM_AGENT_STARTUP_TIMEOUT:-900}"
export SPEEDLM_AGENT_SUBPROCESS_TIMEOUT="${SPEEDLM_AGENT_SUBPROCESS_TIMEOUT:-420}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: $SLURM_JOB_ID"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "model: $SPEEDLM_AGENT_MODEL"
  echo "agent override: ${SPEEDLM_AGENT_CLI:-auto-discover}"
  "$project_python" --version
  "$vllm_python" -c 'import importlib.metadata; print("vllm", importlib.metadata.version("vllm"))'
  for candidate in qwen-cli qwen-code qwen aider opencode claude codex; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf 'agent CLI %s: %s\n' "$candidate" "$(command -v "$candidate")"
    fi
  done
} | tee "$artifact_root/runner-environment.txt"

"$project_python" -m pytest tests/e2e/test_agent_harness.py -q -s 2>&1 |
  tee "$artifact_root/pytest-terminal.txt"
