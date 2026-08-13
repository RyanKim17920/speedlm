#!/bin/bash
#SBATCH --job-name=speedlm-bigcycle
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-bigcycle-%j.out
set -euo pipefail

# A REAL idle-tuning cycle at the largest size this machine can provision.
#
# This is demo/cycle.sbatch with the size knobs opened up: it seeds from the
# BIG corpus captured by job 380039 (~4600 records) instead of run5's 1065, and
# runs demo/bigcycle-qwen8b.json -- a 1536-record training window, 8 epochs,
# and run5's real gate settings (5 benchmark repeats, 512-token benchmark cap).
# Everything hard-won in cycle.sbatch is kept verbatim: the log-grep readiness
# check, the full-corpus seed, the group-wide cleanup trap.
#
# Why 1536 and not 2048/3072: scratch, not data, is the binding constraint.
# eagle3.py's MAX_SCRATCH_BYTES is 96 GiB and the quota is fail-closed
# (_check -> scratch_usage(work_dir) > scratch_quota_bytes on every stage tick).
# Hidden-state shards are `tokens x (num_aux_layers + 1) x hidden_size x 2` =
# tokens x 32,768 B for Qwen3-8B, and the measured mean row of this corpus is
# 2,277 tokens = 74.6 MB.  A 1536 window leases ~1,120 rows and renders ~1,010,
# i.e. ~75 GB of shards -- comfortably inside 92 GiB.  A 2048 window would
# render ~1,500 rows (~112 GiB) and abort mid-extraction, and 3072 is rejected
# outright by derive_scratch_quota_bytes.  That function is now model-aware --
# it takes hidden_size / num_aux_layers / tokens_per_row and reproduces the
# expression above, instead of the flat 32 MiB/row it used to charge, which was
# fitted on gpt-oss-20b and under-estimated this traffic by ~2.3x.  The quota
# here is still set from the measured token distribution rather than from the
# function's defaults, because those default to 4,096 tokens/row (~1.5x this
# corpus's mean) and would price the full 1536-record ceiling above 96 GiB.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=${BIGCYCLE_RUN_DIR:-/data/ryan.kim/speedlm-runs/bigcycle-run1}
config="$repo/demo/bigcycle-qwen8b.json"
model="Qwen/Qwen3-8B"
seed_traces=/data/ryan.kim/speedlm-runs/bigcorpus-run1/speedlm_home/traces/traces.jsonl

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
  echo "seed_traces: $seed_traces"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

# --- free-space guard -----------------------------------------------------
# The cycle may occupy up to tuning.scratch_quota_bytes (92 GiB) of /data at
# once.  /data is a shared 55 TB filesystem that has run above 90% full; a
# cycle that dies at hour three because someone else filled the disk wastes the
# whole allocation, so refuse to start rather than discover it late.
avail_kib=$(df -Pk /data | awk 'NR==2 {print $4}')
required_kib=$((120 * 1024 * 1024))   # 120 GiB: 92 GiB quota + working margin
if (( avail_kib < required_kib )); then
  echo "/data has $((avail_kib / 1024 / 1024)) GiB free, need >= 120 GiB" >&2
  exit 1
fi
echo "/data free: $((avail_kib / 1024 / 1024)) GiB"

# --- seed the trace store from the BIG captured corpus --------------------
# Same reasoning as demo/cycle.sbatch, and it matters more here, not less.
# Seed the WHOLE capture, never a tail: bigcycle-qwen8b.json sets
# tuning.trust_self_play_assistant_turns=true, and what makes a captured
# assistant turn legal supervision is not that flag but
# src/speedlm/training/provenance.py:self_play_attestation, which requires every
# `client_supplied` assistant turn to be byte-identical to a `generated` turn
# from an EARLIER row.  The evidence that a turn is self-play lives in the rows
# BEFORE it, so a tail keeps the turns and discards their evidence and the rows
# become untrainable.
#
# Seeding the whole ~4600-record corpus costs nothing at training time: the
# training window is still only the newest 1536 -- split.py:_select_window seeks
# by offset and deserializes just the tail -- while
# speculators_corpus.py:_full_capture_rows reads the complete
# <home>/traces/traces.jsonl separately, purely as attestation evidence.
#
# buffer.max_age_days is 3650 in the config and that is load-bearing: the trace
# store age-prunes on append and these records are days old.  At the production
# value of 1.0 they would be pruned on the first write and the cycle would sit
# below the corpus floor forever with nothing to train.
mkdir -p "$SPEEDLM_HOME/traces"
[[ -r "$seed_traces" ]] || { echo "seed traces not readable: $seed_traces" >&2; exit 1; }
cp "$seed_traces" "$SPEEDLM_HOME/traces/traces.jsonl"
seeded=$(wc -l < "$SPEEDLM_HOME/traces/traces.jsonl")
echo "seeded $seeded trace records from $seed_traces"
echo "seed_records: $seeded" >> "$run_dir/provenance.txt"

# tuning.min_corpus_records is 1024.  Refuse rather than idle forever if the
# capture job was still warming up when this was submitted.
if (( seeded < 1024 )); then
  echo "seeded $seeded records, below tuning.min_corpus_records=1024" >&2
  exit 1
fi

gateway_log="$run_dir/gateway-and-vllm.log"

# --tool-call-parser hermes is load-bearing and deliberately OVERRIDES the
# profile. QWEN_3_8B_EAGLE3_PROFILE pins qwen3_xml, and job 376274 proved that
# wrong on hardware: vLLM received tool_call_parser='qwen3_xml' and Qwen3-8B then
# emitted Hermes-format calls -- exactly the start token hermes_tool_parser.py:36
# looks for -- inside message content with an empty tool_calls array. Every
# trajectory died at turn one. The mismatch is silent by construction: a parser
# that matches nothing yields no tool calls, which is indistinguishable from a
# model that simply chose not to call one.
#
# --max-model-len 24576 because agentic prompts here reach ~20.7k tokens.
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
# `setsid` puts the gateway in its OWN process group, and a tuning cycle spawns
# grandchildren (the Speculators trainer, and vLLM engines it restarts) that do
# not die with their parent.  Trapping EXIT alone is not enough: a SIGTERM at
# the SLURM time limit kills the script without ever running cleanup, leaving an
# orphaned vLLM pinning the GPU until the allocation is reaped.  Trapping
# TERM/INT as well, and finishing with a group-wide SIGKILL, makes "the run ran
# out of time" cost nothing more than the run.
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

# --- wait for readiness WITHOUT generating gateway traffic ------------------
# DO NOT poll the gateway.  This is the trap that killed job 379744.
# 8150 is wrapper.host/wrapper.port -- the gateway -- and
# src/speedlm/gateway/app.py has NO /health special case: the only route is the
# catch-all `route_all` (app.py:106-112), which hands every request straight to
# `proxy.handle`, whose first act is `admission.try_begin()` ->
# `ActivityTracker.try_begin()` (activity.py:48), stamping `last_activity`
# BEFORE it even checks whether admission is open.  So `curl 8150/health` is
# ORDINARY TRAFFIC.  There is no free probe on 8150: `/health` is explicitly
# allowed and proxied (proxy.py:647), and a blocked path like `/metrics` 404s
# but still stamps the tracker on the way out.
#
# Nor is there an upstream port to poll instead.  The config's target.port 8050
# is inert -- config.py's TargetConfig is only ever serialized, never dialed --
# and cli.py:485 gives vLLM a kernel-assigned ephemeral port via
# reserve_loopback_port().  In the tuning path that port is never even logged.
#
# That is fatal rather than untidy, because with idle_threshold_seconds=5.0 the
# tuner arms while the gateway is still starting: a cycle is already in SLEEPING
# by the time vLLM finishes loading and the first curl gets a 200.  The
# readiness poll's own request then trips the tuner's PreemptionGuard, which
# reads that traffic as a client arriving mid-cycle and aborts with
# `ControlAborted: vLLM sleep-state wait aborted`.
#
# So the readiness signal is the gateway's own startup log line instead.  It
# costs zero requests, and it is emitted at exactly the right moment: cli.py:650
# starts uvicorn only after the supervisor has the upstream vLLM up, and
# cli.py:677 waits for `server.started` before `service.start()` arms the tuner.
# Grepping a file cannot touch the activity tracker by construction.
ready_marker="Uvicorn running on http://127.0.0.1:8150"
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
# No traffic is driven from here.  idle_threshold_seconds=5.0 and
# idle_confirmations=2 mean the tuner arms on its own a few seconds after the
# gateway goes quiet, and the seeded corpus is already above min_corpus_records,
# so the cycle fires without anything prompting it.  This loop only watches.
#
# Completion is detected on the tuner's own log line -- TunerService emits
# "idle tuning cycle completed: <outcome>" (src/speedlm/tuner/service.py:674)
# once the gate has decided, whichever way it decided.
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
