#!/bin/bash
#SBATCH --job-name=speedlm-demo-capture
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-demo-capture-%j.out
set -euo pipefail

# Record the token-by-token timelines behind the stock-vs-tuned demo video.
#
# This is NOT a measurement job. Job 378546 already measured the thing -- the
# gated result is +0.2989 accepted length / +9.94% tok/s on the session-disjoint
# suite, pooled over 5 repeats x 100 contexts x 2 blocks per arm. This job runs
# ONE sequential pass over the watchable contexts of that same suite, which is
# far too little to promote on and exactly enough to watch. If the numbers this
# job prints disagree with the decision, the decision is right and this is noise.
#
# Both arms run sequentially in this one allocation on this one GPU. Racing two
# engines on two GPUs would confound the draft head with silicon and thermals;
# racing them on one GPU would have them contend for it. Sequential is slower and
# is the only ordering whose wall clock means anything.
#
# The stock arm runs first and pays for a discarded warmup request, so the cold
# engine's CUDA-graph and cache costs are not charged to whichever arm went first.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=/data/ryan.kim/speedlm-runs/regate-unseen-run1
source_run=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/speedlm_home/runs/7343677008da41f995f46e6a714036b4
out_dir=${DEMO_OUT_DIR:-/data/ryan.kim/speedlm-runs/demo-video-run1}
# 49 is every context in the suite whose reference continuation is long enough to
# be worth watching; the other 51 held-out turns are bare tool calls of ~4 tokens.
num_contexts=${DEMO_NUM_CONTEXTS:-49}

mkdir -p "$out_dir"
cd "$repo"

# Same environment the gate job used. HF_HOME in particular is not the login
# shell's default, and pointing it elsewhere would resolve a different snapshot
# of the stock speculator than the one the decision was measured against.
export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?allocation exposed no GPU}"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "source run: $source_run"
} | tee "$out_dir/provenance.txt"
nvidia-smi || true

[[ -d "$source_run/draft-model" ]] || { echo "missing candidate draft" >&2; exit 1; }
[[ -f "$run_dir/regate_wiring.json" ]] || { echo "missing argv template" >&2; exit 1; }

# The engine argv is lifted from the gate job's own wiring snapshot rather than
# retyped, so --enforce-eager and --max-model-len cannot drift away from the
# settings the recorded decision was measured under.
"$repo/.venv/bin/python" scripts/demo_capture.py \
  --suite-dir "$run_dir/unseen-suite" \
  --selection "$run_dir/unseen-suite/selection.json" \
  --trajectories /data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/traffic/trajectories \
  --wiring "$run_dir/regate_wiring.json" \
  --stock-draft RedHatAI/Qwen3-8B-speculator.eagle3 \
  --candidate-draft "$source_run/draft-model" \
  --out-dir "$out_dir" \
  --num-contexts "$num_contexts" \
  --max-tokens 512 \
  2>&1 | tee "$out_dir/capture.log"

echo "=== comparison ==="
cat "$out_dir/comparison.json"
echo "capture directory: $out_dir"
