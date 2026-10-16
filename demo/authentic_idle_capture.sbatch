#!/bin/bash
#SBATCH --job-name=speedlm-authentic-capture
#SBATCH --partition=n
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-captures/slurm-%j.out
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
capture_root=/data/ryan.kim/speedlm-captures
capture_dir="$capture_root/run-${SLURM_JOB_ID}"

mkdir -p "$capture_root"
cd "$repo"

echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "gpus=$CUDA_VISIBLE_DEVICES"
echo "commit=$(git rev-parse HEAD)"
nvidia-smi

exec "$repo/.venv/bin/python" "$repo/demo/authentic_idle_capture.py" \
  --out "$capture_dir" \
  --timeout 2700
