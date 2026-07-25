#!/usr/bin/env bash
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
artifacts="${SPEEDLM_E2E_ARTIFACT_DIR:-$repo/log_artifacts}"
project_python="$repo/.venv/bin/python"

cd "$repo"
mkdir -p "$artifacts"

echo "hostname: $(hostname -f)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:?missing SLURM node list}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?no allocated GPU visible}"
nvidia-smi
"$project_python" --version
/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python -c \
  'import vllm; print(f"vllm {vllm.__version__}")'

export SPEEDLM_E2E=1
export SPEEDLM_E2E_MODEL=Qwen/Qwen3.5-2B
export SPEEDLM_E2E_STAGE="${SPEEDLM_E2E_STAGE:-proxy-overhead}"
export SPEEDLM_E2E_ARTIFACT_DIR="$artifacts"
export SPEEDLM_E2E_VLLM_ARGS='[
  "--max-model-len", "4096",
  "--gpu-memory-utilization", "0.85",
  "--enforce-eager"
]'

"$project_python" -m pytest tests/e2e/test_proxy_overhead.py -q -s 2>&1 |
  tee "$artifacts/$SPEEDLM_E2E_STAGE-terminal.txt"
