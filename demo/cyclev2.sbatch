#!/bin/bash
#SBATCH --job-name=speedlm-cyclev2
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-cyclev2-%j.out
set -euo pipefail

# Idle-tuning cycle on the v2 corpus (bigcorpus-v2, 8 shards, ~17,000 records,
# 12 task families, 2,400 distinct prompts, jobs 383875-383882).
#
# This script:
#   1. Merges whatever v2 shard trace stores exist (partial capture is tolerated;
#      see the floor check below).
#   2. Seeds the WHOLE merged file into $SPEEDLM_HOME/traces/ without truncation.
#   3. Fires one idle-tuning cycle (Eagle3 draft head, Qwen3-8B).
#
# Modelled on demo/bigcycle.sbatch; every hard-won detail is preserved.
#
# --- EPOCH COUNT: 3 (not 8) -----------------------------------------------
# Evidence from docs/speedup-ceiling.md and the bigcycle-run1 result:
#   - 8 epochs drove val full_acc_0 to 0.988 but on-policy acceptance FELL
#     to 0.549, BELOW the 3-epoch head's 0.612. The model memorised the
#     training distribution and the on-policy metric regressed. The 8-epoch
#     head was rolled back for throughput_not_stationary, but the acceptance
#     regression makes epochs the wrong scaling axis regardless.
#   - The scaling axis is DATA, not passes over it. v2 has ~17,000 records
#     (~4.5x the ~3,758-record bigcorpus-run1 seed), which provides the
#     regularisation that extra epochs cannot. Fewer passes reduce overfit.
#   - 3 is the "middle ground with evidence": enough gradient steps to
#     learn the distribution, not so many as to memorise it. cycle1e showed
#     1 epoch is probably not enough; 8 is provably too many. 3 is the
#     default that should have been used from the start.
#
# --- SCRATCH QUOTA: 103,079,215,104 bytes (96 GiB) -------------------------
# The training_window_records stays at 1024: scratch, not data, is the binding
# constraint (eagle3.py MAX_SCRATCH_BYTES = 96 GiB). Hidden-state shards are:
#     tokens x (num_aux_layers + 1) x hidden_size x 2
#   = tokens x (31 + 1) x 4096 x 2  [Qwen3-8B: 32 aux layers, hidden 4096]
#   = tokens x 32,768 B
# Measured mean tokens/row for this traffic: 2,277 => 74.6 MB/row.
# Worst-case shard occupancy at window = 1024 rows:
#     1,024 x 74,612,736 + 1 GiB headroom
#   = 76,403,441,664 + 1,073,741,824
#   = 77,477,183,488 B = 72.16 GiB  -- FITS under 96 GiB with 23.84 GiB margin
# (1,280 rows would be 89.95 GiB, only 6.05 GiB margin; 1,536 exceeds 96 GiB.)
# The quota is set to the 96 GiB ceiling (= 103,079,215,104 B).  That is
# fail-closed but not a reservation: actual occupancy is ~55 GB because the
# pre-flight split reserves 287 of 1,024 as held-out and leases ~737 for
# training.  The v2 corpus is larger but the window and model are unchanged,
# so the quota is unchanged.
#
# --- GATE SETTINGS (mirrored from regate-big-run2, job 380736) --------------
# decision.json values used to produce the definitive +0.3457 measurement:
#   num_contexts        : 287     (session-disjoint held-out, held_out_fraction=0.2)
#   num_repeats         : 8       (benchmark_repeats in config)
#   warmup_repeats      : 3       (in config; not scored, warms up engines)
#   benchmark_max_tokens: 512
#   num_speculative_tokens: 3     (from profile qwen3-8b-eagle3, not a config key)
# These are set in demo/cyclev2-qwen8b.json so the result is directly comparable.
#
# --- K = 3 (not 5) ----------------------------------------------------------
# regate-k5-run1 (docs/speedup-ceiling.md, section 4) measured K=5 on the
# same head: accepted-length delta grew +0.2989 -> +0.4648, but candidate
# absolute throughput FELL 143.65 -> 132.14 tok/s (-8.0%) because every
# drafted token costs a draft forward pass whether or not it survives.
# K=5 headline "+20%" is relative to a stock arm K=5 damaged even more (-16%).
# In absolute terms K=5 makes serving slower. K=3 is at or near the optimum.
#
# --- READINESS CHECK: log-grep on the port from config, NOT hardcoded ------
# Hardcoding 8150 (as in demo/cycle1e.sbatch) breaks silently if wrapper.port
# is changed in the config. Port is extracted from the JSON at runtime below.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=${CYCLEV2_RUN_DIR:-/data/ryan.kim/speedlm-runs/cyclev2-run1}
config="$repo/demo/cyclev2-qwen8b.json"
model="Qwen/Qwen3-8B"

# Shard glob — covers all 8 v2 shards (jobs 383875-383882).
# merge_traces.py --allow-missing skips shards that failed or were preempted.
shard_glob='/data/ryan.kim/speedlm-runs/bigcorpus-v2-shard*/speedlm_home/traces/traces.jsonl'
merged_traces=/data/ryan.kim/speedlm-runs/bigcorpus-v2-merged/traces.jsonl

# Minimum merged record count before we refuse to start.
# Justification: v2 targets ~17,000 records across 8 shards (~2,125/shard).
# 2,000 records (< 1 average shard) is a weak signal that capture failed
# almost entirely, and is still above tuning.min_corpus_records=1024, so the
# engine would not idle-loop -- but training on < 2,000 records from a 17k
# corpus means something went badly wrong.  2,000 is the floor; prefer >= 8,000.
min_records=2000

mkdir -p "$run_dir"
cd "$repo"

export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"
export SPEEDLM_HOME="$run_dir/speedlm_home"
mkdir -p "$SPEEDLM_HOME"

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?allocation exposed no GPU}"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "config: $config"
  echo "run_dir: $run_dir"
  echo "SPEEDLM_HOME: $SPEEDLM_HOME"
  echo "merged_traces: $merged_traces"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

# --- free-space guard -----------------------------------------------------
# The cycle may occupy up to tuning.scratch_quota_bytes (96 GiB) of /data.
# Refuse to start rather than discover mid-run that the disk is full.
avail_kib=$(df -Pk /data | awk 'NR==2 {print $4}')
required_kib=$((120 * 1024 * 1024))   # 120 GiB: 72 GiB measured need + margin
if (( avail_kib < required_kib )); then
  echo "/data has $((avail_kib / 1024 / 1024)) GiB free, need >= 120 GiB" >&2
  exit 1
fi
echo "/data free: $((avail_kib / 1024 / 1024)) GiB"

# --- merge v2 shard trace stores ------------------------------------------
# Merge all shard stores that exist; skip missing ones (partial capture is OK).
# merge_traces.py dedupes on record id -- the upstream completion id from vLLM
# -- so overlapping shard seed windows are safe and double-counting is impossible.
#
# Count shards found before merging so we can report and warn if low.
shard_count=0
for f in $(eval "ls $shard_glob 2>/dev/null || true"); do
  (( shard_count++ )) || true
done
echo "shard trace stores found: $shard_count / 8"

if (( shard_count == 0 )); then
  echo "ERROR: no v2 shard stores found -- is capture complete?" >&2
  echo "  looked for: $shard_glob" >&2
  exit 1
fi

if (( shard_count < 8 )); then
  echo "WARNING: only $shard_count of 8 shards found; some may have been preempted" >&2
fi

vllm_py="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python"
mkdir -p "$(dirname "$merged_traces")"

"$vllm_py" "$repo/demo/merge_traces.py" \
  --out "$merged_traces" \
  --glob "$shard_glob" \
  --allow-missing

merged_count=$(wc -l < "$merged_traces")
echo "merged record count: $merged_count"
echo "shard_count: $shard_count"    >> "$run_dir/provenance.txt"
echo "merged_records: $merged_count" >> "$run_dir/provenance.txt"

if (( merged_count < min_records )); then
  echo "ERROR: merged only $merged_count records, below floor of $min_records" >&2
  echo "  ($shard_count shards captured; expected ~$((17000 * shard_count / 8)) from $shard_count shards)" >&2
  exit 1
fi
echo "corpus floor check passed: $merged_count >= $min_records"

# --- seed the trace store from the WHOLE merged corpus --------------------
# Copy the WHOLE merged file, never a tail.
#
# tuning.trust_self_play_assistant_turns=true, and self_play_attestation
# (src/speedlm/training/provenance.py) proves a captured assistant turn is
# self-generated by matching it byte-for-byte against a `generated` turn from
# an EARLIER row.  Evidence lives in earlier rows; truncating destroys it.
# A tailed seed would leave 83%+ of rows untrainable (observed on prior runs).
#
# Seeding the whole file costs nothing at training time: training_window_records=1024
# means split.py deserilaizes only the newest 1024 rows for training, while
# speculators_corpus.py reads the full file separately as attestation evidence.
#
# buffer.max_age_days=3650 is load-bearing: at the default value of 1.0 these
# records would be pruned on the first write and the cycle would never start.
mkdir -p "$SPEEDLM_HOME/traces"
[[ -r "$merged_traces" ]] || { echo "merged traces not readable: $merged_traces" >&2; exit 1; }
cp "$merged_traces" "$SPEEDLM_HOME/traces/traces.jsonl"
seeded=$(wc -l < "$SPEEDLM_HOME/traces/traces.jsonl")
echo "seeded $seeded trace records into $SPEEDLM_HOME/traces/traces.jsonl"
echo "seed_records: $seeded" >> "$run_dir/provenance.txt"

if (( seeded < 1024 )); then
  echo "seeded $seeded records, below tuning.min_corpus_records=1024" >&2
  exit 1
fi

# --- extract the gateway port from the config (do not hardcode) -----------
# Reading the port from the config means this script stays correct if
# wrapper.port is ever changed. demo/cycle1e.sbatch hardcoded 8150, which
# is exactly the bug this replaces.
gateway_port=$("$vllm_py" -c "import json,sys; c=json.load(open('$config')); print(c['wrapper']['port'])")
ready_marker="Uvicorn running on http://127.0.0.1:${gateway_port}"
echo "gateway will listen on port $gateway_port (from config)"

gateway_log="$run_dir/gateway-and-vllm.log"

# --tool-call-parser hermes is load-bearing: QWEN_3_8B_EAGLE3_PROFILE pins
# qwen3_xml, but job 376274 proved that wrong on hardware. Qwen3-8B emits
# Hermes-format calls; qwen3_xml parser silently yields no tool_calls and
# every trajectory dies at turn one.
#
# --max-model-len 24576: agentic prompts here reach ~20.7k tokens.
setsid "$repo/.venv/bin/speedlm" vllm serve "$model" \
  --config "$config" \
  --enable-idle-tuning \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$gateway_log" 2>&1 &
gateway_pid=$!

# --- cleanup trap ---------------------------------------------------------
# setsid puts the gateway in its own process group. Grandchildren (Speculators
# trainer, vLLM engines) do not die with the parent. Trap TERM/INT in addition
# to EXIT, and finish with a group-wide SIGKILL so a SLURM time-limit kill
# costs nothing beyond the run itself.
cleanup() {
  trap - EXIT TERM INT
  if kill -0 "$gateway_pid" 2>/dev/null; then
    kill -TERM -- "-$gateway_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$gateway_pid" 2>/dev/null || break
      sleep 1
    done
  fi
  kill -KILL -- "-$gateway_pid" 2>/dev/null || true
  wait "$gateway_pid" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# --- wait for readiness WITHOUT generating gateway traffic ----------------
# DO NOT poll the gateway with curl or any HTTP request.
#
# src/speedlm/gateway/app.py has no /health special case: every route goes
# through proxy.handle, whose first act is ActivityTracker.try_begin()
# (activity.py:48), stamping last_activity BEFORE checking admission.
# A health probe is ordinary traffic. With idle_threshold_seconds=5.0 the
# tuner arms shortly after startup; a probe's stamp preempts the tuner mid-
# cycle with ControlAborted: vLLM sleep-state wait aborted (job 379744).
#
# The readiness signal is the gateway's own log line, emitted by cli.py after
# uvicorn starts and BEFORE service.start() arms the tuner. Grepping a file
# cannot touch the activity tracker by construction.
ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited before becoming ready; see $gateway_log" >&2
    tail -n 60 "$gateway_log" >&2
    exit 1
  fi
  if grep -qF "$ready_marker" "$gateway_log" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "gateway not ready within 1800s" >&2; exit 1; }
echo "READY: gateway listening (log marker: $ready_marker)"

# --- wait for the cycle ---------------------------------------------------
# No traffic is driven from here. idle_threshold_seconds=5.0 and
# idle_confirmations=2 mean the tuner arms on its own once the gateway goes
# quiet, and the seeded corpus is already above min_corpus_records.
#
# TunerService emits "idle tuning cycle completed: <outcome>" once the gate
# has decided (src/speedlm/tuner/service.py:674).
echo "waiting for the idle-tuning cycle (log: $gateway_log)"
tail -f "$gateway_log" &
tail_pid=$!

completed=0
while :; do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited during tuning; see $gateway_log" >&2
    break
  fi
  if grep -q "idle tuning cycle completed" "$gateway_log" 2>/dev/null; then
    completed=1
    break
  fi
  sleep 10
done
kill "$tail_pid" 2>/dev/null || true

echo "=== cycle outcome ==="
grep "idle tuning cycle completed" "$gateway_log" || echo "(cycle did not complete)"
echo "=== lease ==="
grep "leased .* training record" "$gateway_log" || true
echo "=== decisions ==="
find "$SPEEDLM_HOME" -name 'decision*.json' -exec echo '--- {}' \; -exec cat {} \; 2>/dev/null | head -120 || true

mkdir -p "$run_dir/results"
find "$SPEEDLM_HOME" -name '*.json' -size -10M -exec cp --parents {} "$run_dir/results/" \; 2>/dev/null || true
tail -n 400 "$gateway_log" > "$run_dir/gateway-tail.log" || true
echo "run directory: $run_dir"

[[ "$completed" == 1 ]] || exit 1
exit 0
