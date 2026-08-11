#!/bin/bash
#SBATCH --job-name=speedlm-regate-unseen
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/regate-unseen-run1/slurm-%j.out
set -euo pipefail

# Re-gate run5's ALREADY-TRAINED Eagle3 draft head against contexts drawn only
# from agent sessions that contributed ZERO records to its training window.
#
# run5 gated at +0.65 accepted length / +32% tok/s, but its held-out split was
# row-level and agentic records are nested: record N of a session contains
# record N-1's messages. Most of that suite therefore came from sessions the
# head had also trained on, so the number could be recall rather than
# generalisation. This job re-runs the SAME gate -- same stock baseline, same
# block schedule, same thresholds, same decision writer -- on a suite built from
# 100 contexts belonging to 61 sessions none of whose records were ever leased.
#
# Nothing is trained here. The candidate is read off run5's run directory, which
# is opened READ-ONLY: this job writes only under its own run_dir.
#
# TWO gates run in this one allocation, and the pair is the measurement:
#
#   1. UNSEEN  -- the session-disjoint suite built below.
#   2. CONTROL -- run5's ORIGINAL held-out suite, re-gated here.
#
# The control is what makes the comparison honest. Comparing the unseen result
# against run5's recorded +0.6525 would confound leakage with everything that
# differs between two allocations: node, thermal state, engine build, driver
# version. Running both suites against the same head on the same GPU in the same
# job leaves the SUITE as the only thing that changed. If the control reproduces
# ~+0.65 while the unseen suite collapses toward the +0.05 threshold, leakage
# explains the original result; if both hold, the gain is real.
#
# Note that both suites pass the gate's own leakage assertion, which is exactly
# the point: that assertion is row-level, and the original suite is
# session-contaminated while satisfying it.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=/data/ryan.kim/speedlm-runs/regate-unseen-run1
source_run=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/speedlm_home/runs/7343677008da41f995f46e6a714036b4
traces=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/speedlm_home/traces/traces.jsonl
suite_dir="$run_dir/unseen-suite"
config="$repo/configs/agentenv-qwen8b.json"

mkdir -p "$run_dir"
cd "$repo"

export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"

# A FRESH home, deliberately empty of promoted artifacts. The launch plan
# resolves the stock arm from the active artifact, so a home carrying a
# promotion would silently baseline against that promotion instead of against
# RedHatAI/Qwen3-8B-speculator.eagle3 -- the driver refuses when the resolved
# stock arm disagrees with the original decision.json, and an empty home is what
# makes it agree.
export SPEEDLM_HOME="$run_dir/speedlm_home"
mkdir -p "$SPEEDLM_HOME"

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?allocation exposed no GPU}"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "config: $config"
  echo "source run: $source_run"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

for required in "$traces" "$source_run/decision.json" "$source_run/held-out/suite_manifest.json"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done
[[ -d "$source_run/draft-model" ]] || { echo "missing candidate draft" >&2; exit 1; }

# --- 1. build the unseen-session suite ------------------------------------
# Rebuilt here rather than copied in, so the suite this job gates against is
# derived from the trace corpus on this machine and its leakage proof is
# recomputed. The script fails closed if the recovered lease does not hold
# exactly the 412 training rows the run5 gateway log recorded, or if any chosen
# context or session touches the training window.
"$repo/.venv/bin/python" scripts/build_unseen_session_suite.py \
  --traces "$traces" \
  --original-suite-dir "$source_run/held-out" \
  --out "$suite_dir" \
  --target 100 \
  --training-window-records 512 \
  --expect-training-records 412 \
  2>&1 | tee "$run_dir/build-suite.log"

# --- 2. dry run: everything except the GPU --------------------------------
# Cheap, and it is what turns a wiring mistake into a failure before the engine
# is ever launched: config, profile, argv, suite load, stock-arm agreement with
# the original decision, and the gate's own leakage check.
"$repo/.venv/bin/python" scripts/regate_existing_draft.py \
  --config "$config" \
  --suite-dir "$suite_dir" \
  --training-hashes "$suite_dir/training_context_hashes.json" \
  --original-decision "$source_run/decision.json" \
  --candidate-draft "$source_run/draft-model" \
  --run-dir "$run_dir" \
  --home "$SPEEDLM_HOME" \
  --repeats 5 \
  --dry-run \
  -- \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee "$run_dir/dry-run.log"

# --- 3. the gate ----------------------------------------------------------
# The engine flags below MUST match run5's, or the two decisions are not
# comparable: --enforce-eager is what makes both runs record
# engine_execution_mode "eager", and --max-model-len 24576 is what lets the
# ~20.7k-token agentic prompts through at all.
#
# Budget: 4 measurement blocks, each preceded by a real engine restart the gate
# refuses to skip, plus one warmup pass per block and 5 scored repeats of 100
# contexts per arm. run5's gate phase took ~18 min of replay; the restarts and
# model loads dominate the rest. --timeout-seconds is the gate's own deadline
# and is set below the SLURM wall clock so a stall yields a logged timeout
# rather than a killed job.
"$repo/.venv/bin/python" scripts/regate_existing_draft.py \
  --config "$config" \
  --suite-dir "$suite_dir" \
  --training-hashes "$suite_dir/training_context_hashes.json" \
  --original-decision "$source_run/decision.json" \
  --candidate-draft "$source_run/draft-model" \
  --run-dir "$run_dir" \
  --home "$SPEEDLM_HOME" \
  --repeats 5 \
  --timeout-seconds 5400 \
  -- \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee "$run_dir/regate.log"

# --- 4. the control gate: run5's ORIGINAL suite, same head, same allocation ---
# The original suite is copied rather than gated in place, so run5's run
# directory stays read-only and this job's inputs are all under its own run_dir.
#
# The same training-hash file is passed deliberately. Those 342 hashes are the
# rows run5 actually trained on, and the original suite was held out from them
# at the row level, so this arm passes the identical leakage assertion the
# unseen arm passes -- while containing contexts from sessions that DID
# contribute training rows.
#
# Failure here must not discard the primary result, which is already written by
# this point, so the summary below runs either way.
control_dir="$run_dir/control"
control_suite="$control_dir/suite"
mkdir -p "$control_suite"
cp "$source_run/held-out/suite_manifest.json" "$control_suite/"
cp "$source_run/held-out/suite_contexts.jsonl" "$control_suite/" || true
export SPEEDLM_HOME="$control_dir/speedlm_home"
mkdir -p "$SPEEDLM_HOME"

control_status=ok
"$repo/.venv/bin/python" scripts/regate_existing_draft.py \
  --config "$config" \
  --suite-dir "$control_suite" \
  --training-hashes "$suite_dir/training_context_hashes.json" \
  --original-decision "$source_run/decision.json" \
  --candidate-draft "$source_run/draft-model" \
  --run-dir "$control_dir" \
  --home "$SPEEDLM_HOME" \
  --repeats 5 \
  --timeout-seconds 5400 \
  -- \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee "$control_dir/regate.log" || control_status=failed

echo "=== UNSEEN-session decision (primary) ==="
cat "$run_dir/decision.json" || echo "(no decision written)"
echo "=== CONTROL: run5's original suite, re-gated here (status: $control_status) ==="
cat "$control_dir/decision.json" || echo "(no control decision written)"
echo "=== run5's recorded decision, for reference ==="
cat "$source_run/decision.json"
echo "run directory: $run_dir"
