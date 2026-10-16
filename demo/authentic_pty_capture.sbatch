#!/bin/bash
#SBATCH --job-name=speedlm-authentic-pty
#SBATCH --partition=n
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-captures/pty-slurm-%j.out
set -euo pipefail

repo=/admin/home/ryan.kim/speedlm-fr
capture_root=/data/ryan.kim/speedlm-captures
export CAPTURE_DIR="$capture_root/pty-run-${SLURM_JOB_ID}"

mkdir -p "$capture_root"
cd "$repo"

echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "gpus=$CUDA_VISIBLE_DEVICES"
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

exec "$repo/.venv/bin/python" "$repo/demo/authentic_pty_capture.py"
