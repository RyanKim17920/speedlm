#!/usr/bin/env bash
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
project_python="$repo/.venv/bin/python"
artifact_root="${SPEEDLM_MATRIX_ARTIFACT_DIR:-$repo/log_artifacts/model-matrix}"
cell="${1:-${SPEEDLM_MATRIX_CELL:-}}"

case "$cell" in
  gpt-oss-20b-eagle3|qwen3.5-9b-mtp|qwen3.5-2b-none|qwen3.5-2b-ngram)
    ;;
  "")
    echo "SPEEDLM_MATRIX_CELL (or the first argument) is required" >&2
    exit 2
    ;;
  *)
    echo "unknown matrix cell: $cell" >&2
    exit 2
    ;;
esac

: "${CUDA_VISIBLE_DEVICES:?assign one GPU to CUDA_VISIBLE_DEVICES}"

cd "$repo"
mkdir -p "$artifact_root/$cell"

export SPEEDLM_E2E=1
export SPEEDLM_MATRIX_CELL="$cell"
export SPEEDLM_MATRIX_ARTIFACT_DIR="$artifact_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"

{
  echo "cell: $cell"
  echo "hostname: $(hostname -f)"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  "$project_python" --version
} | tee "$artifact_root/$cell/runner-environment.txt"

"$project_python" -m pytest tests/e2e/test_model_matrix.py -q -s 2>&1 |
  tee "$artifact_root/$cell/pytest-terminal.txt"
