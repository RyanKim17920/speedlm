#!/bin/bash
#SBATCH --job-name=speedlm-bigcorpus
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-bigcorpus-%j.out
set -euo pipefail

# CAPTURE ONLY. This job serves Qwen3-8B through the SpeedLM gateway with idle
# tuning DISABLED and drives real agentic self-play traffic through it, purely to
# grow the trace corpus. A later job does the tuning cycle over what lands here.
#
# Idle tuning is off in two independent ways, because either one alone would be
# enough and neither is obvious from the outside:
#   1. `--enable-idle-tuning` is NOT passed. cli.py's only call site reads
#      `getattr(s_args, "enable_idle_tuning", False)`, so an absent flag arrives
#      as False, not None -- and False WINS over the config's tuning_enabled.
#   2. demo/traffic-qwen8b.json sets "tuning_enabled": false anyway.
# The untuned gateway path still passes the TraceStore into create_app, so every
# request is captured exactly as it would be with tuning on.
#
# buffer.max_age_days is 3650 so the seeded run5 records (captured 2026-08-10)
# are not age-pruned out from under us, and max_tokens is 200M so a corpus of
# ~5k records at ~3.0k tokens each (~15M) is nowhere near the eviction ceiling.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=${TRAFFIC_RUN_DIR:-/data/ryan.kim/speedlm-runs/bigcorpus-run1}
config="$repo/demo/traffic-qwen8b.json"
model="Qwen/Qwen3-8B"
seed_store=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/speedlm_home/traces/traces.jsonl

mkdir -p "$run_dir"
cd "$repo"

export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"
export SPEEDLM_HOME="$run_dir/speedlm_home"
mkdir -p "$SPEEDLM_HOME/traces"

# --- seed the store from a known-good corpus ------------------------------
# We ADD to run5's 1065 records rather than starting from zero, so the tuning
# cycle downstream sees one continuous corpus.
if [[ ! -s "$SPEEDLM_HOME/traces/traces.jsonl" ]]; then
  cp "$seed_store" "$SPEEDLM_HOME/traces/traces.jsonl"
fi
seed_records=$(wc -l < "$SPEEDLM_HOME/traces/traces.jsonl")

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?allocation exposed no GPU}"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "config: $config"
  echo "run_dir: $run_dir"
  echo "SPEEDLM_HOME: $SPEEDLM_HOME"
  echo "seed_source: $seed_store"
  echo "seed_records: $seed_records"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

gateway_log="$run_dir/gateway-and-vllm.log"

# --tool-call-parser hermes deliberately OVERRIDES the profile's qwen3_xml.
# Job 376274 proved qwen3_xml wrong on hardware: Qwen3-8B emits Hermes-format
# calls, the xml parser matched nothing, and every trajectory died at turn one
# with an empty tool_calls array -- indistinguishable from a model that simply
# chose not to call a tool. profiles.py applies the user value LAST, so vLLM
# receives hermes exactly once.
#
# --max-model-len 24576 because agentic prompts here reach ~20.7k tokens.
setsid "$repo/.venv/bin/speedlm" vllm serve "$model" \
  --config "$config" \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$gateway_log" 2>&1 &
gateway_pid=$!

cleanup() {
  if kill -0 "$gateway_pid" 2>/dev/null; then
    kill -TERM -- "-$gateway_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$gateway_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$gateway_pid" 2>/dev/null || true
  fi
  wait "$gateway_pid" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

base_url="http://127.0.0.1:8150"

# --- readiness WITHOUT generating traffic ---------------------------------
# A /health poll goes through the gateway and is itself a request; run 3 polled
# for 20 minutes and salted the corpus with hundreds of empty exchanges. Grep
# the log instead. The wrapper port is in the match string on purpose: the child
# vLLM runs its own uvicorn on a different loopback port and logs the same line.
ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited before becoming ready; see $gateway_log" >&2
    tail -n 60 "$gateway_log" >&2
    exit 1
  fi
  if grep -q "Uvicorn running on http://127.0.0.1:8150" "$gateway_log" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "gateway not ready within 1800s" >&2; tail -n 60 "$gateway_log" >&2; exit 1; }
echo "gateway ready at $base_url (log match, zero traffic generated)"

# --- traffic --------------------------------------------------------------
# Sizing, from run5 measured on this same hardware (job 376xxx, traffic window
# 22:13:52 -> 23:06:40 = 52.8 min):
#   1065 records / 180 trajectories = 5.92 records per trajectory
#   1065 records / 52.8 min         = 20.2 records per minute
#   3213336 tokens / 1065 records   = 3017 tokens per record
# We want to roughly TRIPLE the corpus, i.e. add ~3500 records on top of 1065:
#   3500 / 5.92 = 591 trajectories -> 591 / 6 families = 98.5 seeds -> use 100
#   100 seeds x 6 families = 600 trajectories x 5.92 = ~3550 new records
#   ~3550 / 20.2 = 176 min = 2.9 h; even 30% slower than run5 lands at ~3.8 h,
#   inside the 4 h --stop-after-seconds cap and well inside the 8 h wall.
# Final corpus: 1065 + ~3550 = ~4600 records (~13.9M tokens, 7% of max_tokens).
"$repo/.venv/bin/python" scripts/run_agentic_traffic.py \
  --base-url "$base_url" \
  --model "$model" \
  --out "$run_dir/traffic" \
  --seeds 100 \
  --max-turns 24 \
  --max-output-tokens 1024 \
  --request-timeout 600 \
  --trajectory-wall-clock 1500 \
  --stop-after-seconds 14400 \
  --python "$repo/.venv/bin/python" \
  2>&1 | tee "$run_dir/traffic.log"

echo "traffic complete"

tail -n 400 "$gateway_log" > "$run_dir/gateway-tail.log" || true

echo "=== traffic summary ==="
head -60 "$run_dir/traffic/summary.json" 2>/dev/null || echo "(no traffic summary)"

echo "=== corpus ==="
echo "seeded with $seed_records records from $seed_store"
"$repo/.venv/bin/speedlm" traces stats --store "$SPEEDLM_HOME/traces/traces.jsonl" || true
echo "run directory: $run_dir"
