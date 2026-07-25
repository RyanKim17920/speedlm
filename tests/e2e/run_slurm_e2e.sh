#!/usr/bin/env bash
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
artifacts="$repo/log_artifacts"
mode="${1:-all}"
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

run_stage1() {
  export SPEEDLM_E2E=1
  export SPEEDLM_E2E_MODEL=Qwen/Qwen3.5-2B
  export SPEEDLM_E2E_STAGE=stage1-qwen
  export SPEEDLM_E2E_ARTIFACT_DIR="$artifacts"
  export SPEEDLM_E2E_VLLM_ARGS='[
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--enforce-eager"
  ]'
  unset SPEEDLM_E2E_CAPTURE_METRICS
  "$project_python" -m pytest tests/e2e/test_live_vllm.py -q -s 2>&1 |
    tee "$artifacts/stage1-pytest-terminal.txt"
}

run_stage2() {
  export SPEEDLM_E2E=1
  export SPEEDLM_E2E_MODEL=openai/gpt-oss-20b
  export SPEEDLM_E2E_STAGE=stage2-gpt-oss-eagle3
  export SPEEDLM_E2E_ARTIFACT_DIR="$artifacts"
  export SPEEDLM_E2E_CAPTURE_METRICS=1
  export SPEEDLM_E2E_VLLM_ARGS='[
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.90",
    "--enforce-eager",
    "--speculative-config",
    "{\"model\":\"RedHatAI/gpt-oss-20b-speculator.eagle3\",\"method\":\"eagle3\",\"num_speculative_tokens\":5}"
  ]'
  "$project_python" -m pytest tests/e2e/test_live_vllm.py -q -s 2>&1 |
    tee "$artifacts/stage2-pytest-terminal.txt"
}

case "$mode" in
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  all)
    run_stage1
    run_stage2
    ;;
  *)
    echo "usage: $0 [stage1|stage2|all]" >&2
    exit 2
    ;;
esac
