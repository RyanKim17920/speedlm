#!/bin/bash
#SBATCH --job-name=speedlm-demo-cycle
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-demo-cycle-%j.out
set -euo pipefail

# A REAL idle-tuning cycle, shrunk to fit an hour on one GPU -- for filming.
#
# Nothing here is faked or stubbed: real vLLM, real Eagle3 head training with
# real optimizer steps, real benchmark gate.  The only differences from
# production are size (demo/demo-qwen8b.json: 96-record corpus, 1 epoch,
# 3 benchmark repeats -- the validator's HARD FLOOR -- 0 warmup, 128/64-token
# benchmark and correctness caps) and where the corpus comes from.
#
# The corpus comes from run5's captured traffic rather than fresh rollouts.
# scripts/agentenv_run.sbatch spends ~53 minutes driving 180 agentic
# trajectories to fill the buffer; those trajectories are already on disk from
# run5, they are genuine captured agent sessions from this same gateway and
# model, and regenerating them would consume the entire demo window to produce
# data no more real than what we already have.  So the demo replays them and
# spends its hour on the part worth filming: the tuning cycle itself.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=${DEMO_RUN_DIR:-/data/ryan.kim/speedlm-runs/demo-cycle-run1}
config="$repo/demo/demo-qwen8b.json"
model="Qwen/Qwen3-8B"
seed_traces=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/speedlm_home/traces/traces.jsonl

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
  echo "seed_traces: $seed_traces"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

# --- seed the trace store from REAL captured traffic ----------------------
# These are real agent sessions run5 captured through this same gateway against
# this same model -- not synthetic prompts, not a fixture.  The cycle therefore
# trains the Eagle3 head on genuine traffic, which is the whole claim the demo
# is making.
#
# Seed the WHOLE capture, not a tail.  An earlier revision seeded `tail -n 200`
# on the reasoning that 200 clears min_corpus_records=48 with 4x margin, and
# that was self-defeating.  The demo config sets
# tuning.trust_self_play_assistant_turns=true, and the thing that makes a
# captured assistant turn legal supervision is not the operator's say-so, it is
# src/speedlm/training/provenance.py:self_play_attestation, which requires every
# `client_supplied` assistant turn to be byte-identical to a `generated` turn
# from an EARLIER row.  The evidence that a turn is self-play lives in the rows
# BEFORE it.  A tail keeps the turns and throws away their evidence, so nothing
# can attest: run1 (job 379744) logged "166 of 200 captured requests (83.0%)
# carry an assistant turn this deployment did not generate ... 180 records in
# total are untrainable", leaving ~20 trainable rows against
# training_window_records=48.
#
# Seeding all 1065 records costs nothing at training time.  The training window
# is still only the newest 96 -- split.py:_select_window seeks by offset and
# deserializes just the tail, so cycle cost does not scale with corpus size --
# while speculators_corpus.py:_full_capture_rows reads the complete
# <home>/traces/traces.jsonl separately, purely as attestation evidence.  Big
# corpus, same-size training window, restored attestability.
#
# buffer.max_age_days is 3650 in the demo config and that is load-bearing, not
# cosmetic: the trace store age-prunes on append, and these records are days
# old.  At the production value of 1.0 they would be pruned on the first write
# and the cycle would sit forever below the corpus floor with nothing to train.
mkdir -p "$SPEEDLM_HOME/traces"
[[ -r "$seed_traces" ]] || { echo "seed traces not readable: $seed_traces" >&2; exit 1; }
cp "$seed_traces" "$SPEEDLM_HOME/traces/traces.jsonl"
echo "seeded $(wc -l < "$SPEEDLM_HOME/traces/traces.jsonl") trace records from $seed_traces"

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
# not die with their parent.  scripts/agentenv_run.sbatch traps EXIT only, so a
# SIGTERM at the SLURM time limit -- which for a one-hour demo job is the
# LIKELY exit path, not an edge case -- kills the script without ever running
# cleanup, leaving an orphaned vLLM pinning the GPU until the allocation is
# reaped.  Trapping TERM/INT as well, and finishing with a group-wide SIGKILL,
# makes "the demo ran out of time" cost nothing more than the demo.
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
# DO NOT poll the gateway.  This is the trap that killed run1 (job 379744).
# 8150 is wrapper.host/wrapper.port from demo/demo-qwen8b.json -- the gateway
# -- and src/speedlm/gateway/app.py has NO /health special case: the only route
# is the catch-all `route_all` (app.py:106-112), which hands every request
# straight to `proxy.handle`, whose first act is `admission.try_begin()` ->
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
# That is fatal here rather than merely untidy, because with
# idle_threshold_seconds=5.0 the tuner arms while the gateway is still starting:
# a cycle is already in SLEEPING by the time vLLM finishes loading and the first
# curl gets a 200.  The readiness poll's own request then trips the tuner's
# PreemptionGuard, which reads that traffic as a client arriving mid-cycle and
# aborts with `ControlAborted: vLLM sleep-state wait aborted`.  run1 died there,
# in SLEEPING, never reaching EXTRACTING or TRAINING.  The failure is nasty
# because the probe looks read-only and is not: nothing in `curl /health`
# suggests it is the thing cancelling the run.
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
echo "=== decisions ==="
find "$SPEEDLM_HOME" -name 'decision*.json' -exec echo '--- {}' \; -exec cat {} \; 2>/dev/null | head -120 || true

mkdir -p "$run_dir/results"
find "$SPEEDLM_HOME" -name '*.json' -size -10M -exec cp --parents {} "$run_dir/results/" \; 2>/dev/null || true
tail -n 400 "$gateway_log" > "$run_dir/gateway-tail.log" || true
echo "run directory: $run_dir"

[[ "$completed" == 1 ]] || exit 1
exit 0
