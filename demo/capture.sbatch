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

# ---------------------------------------------------------------------------
# CAPTURE_RUN_ROOT: root of the training run whose draft head is the candidate.
#   Must contain speedlm_home/runs/<hash>/draft-model.
#   Default preserves the original run for backward compatibility.
# ---------------------------------------------------------------------------
capture_run_root=${CAPTURE_RUN_ROOT:-/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5}

# ---------------------------------------------------------------------------
# CAPTURE_GATE_DIR: directory written by the regate job that holds the held-out
#   suite (unseen-suite/) and regate_wiring.json.
#
#   These MUST come from the gate that evaluated THIS candidate head. The suite
#   is session-disjoint relative to the training window of that specific run;
#   reusing a gate dir from a different run would compare against a different
#   held-out split, invalidating the comparison the video asserts. If cyclev2-run1
#   was evaluated by a new regate job, set CAPTURE_GATE_DIR to that job's output
#   directory. If the same regate-unseen-run1 suite was used to evaluate cyclev2
#   (i.e., it is session-disjoint for that run too), this default is correct.
# ---------------------------------------------------------------------------
capture_gate_dir=${CAPTURE_GATE_DIR:-/data/ryan.kim/speedlm-runs/regate-unseen-run1}

# ---------------------------------------------------------------------------
# Discover source_run: exactly one run directory under the training root must
#   contain a draft-model. Zero or multiple matches are both hard failures --
#   picking arbitrarily would silently race a different head than the video
#   claims. This mirrors the discovery+ambiguity pattern used in session_fast.json.
# ---------------------------------------------------------------------------
_draft_matches=()
for _d in "$capture_run_root"/speedlm_home/runs/*/draft-model; do
    [[ -d "$_d" ]] && _draft_matches+=("$_d")
done

if [[ ${#_draft_matches[@]} -eq 0 ]]; then
    echo "ERROR: no draft-model found under $capture_run_root/speedlm_home/runs/*/" >&2
    echo "  (training may still be running, or CAPTURE_RUN_ROOT is wrong)" >&2
    exit 1
elif [[ ${#_draft_matches[@]} -gt 1 ]]; then
    echo "ERROR: ambiguous -- ${#_draft_matches[@]} draft-model directories found" \
         "under $capture_run_root/speedlm_home/runs/*/:" >&2
    printf "  %s\n" "${_draft_matches[@]}" >&2
    echo "  Resolve to a single completed run before capturing." >&2
    exit 1
fi
source_run=$(dirname "${_draft_matches[0]}")

# ---------------------------------------------------------------------------
# out_dir: derive from the training root's basename so successive captures of
#   different runs land in different directories.
#   DEMO_OUT_DIR overrides the default entirely.
# ---------------------------------------------------------------------------
_run_basename=$(basename "$capture_run_root")
_default_out_dir=/data/ryan.kim/speedlm-runs/demo-video-${_run_basename}
out_dir=${DEMO_OUT_DIR:-$_default_out_dir}

# Refuse to overwrite a non-empty output directory -- clobbering a prior
# capture silently would be as bad as picking the wrong head.
if [[ -d "$out_dir" ]] && [[ -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
    echo "ERROR: output directory already exists and is non-empty: $out_dir" >&2
    echo "  Set DEMO_OUT_DIR to a fresh path, or remove the directory first." >&2
    exit 1
fi

run_dir=$capture_gate_dir

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
  echo "capture_run_root: $capture_run_root"
  echo "capture_gate_dir: $capture_gate_dir"
  echo "source run: $source_run"
} | tee "$out_dir/provenance.txt"
nvidia-smi || true

[[ -d "$source_run/draft-model" ]] || { echo "missing candidate draft" >&2; exit 1; }
[[ -f "$run_dir/regate_wiring.json" ]] || { echo "missing argv template" >&2; exit 1; }

# The engine argv is lifted from the gate job's own wiring snapshot rather than
# retyped, so --enforce-eager and --max-model-len cannot drift away from the
# settings the recorded decision was measured under.
"$repo/.venv/bin/python" demo/capture.py \
  --suite-dir "$run_dir/unseen-suite" \
  --selection "$run_dir/unseen-suite/selection.json" \
  --trajectories "$capture_run_root/traffic/trajectories" \
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
