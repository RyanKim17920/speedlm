#!/bin/bash
#SBATCH --job-name=speedlm-bigcorpus-v2
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/slurm-bigcorpus-v2-%j.out
set -euo pipefail

# CAPTURE ONLY (v2 catalog). Serves Qwen3-8B through the SpeedLM gateway with
# idle tuning DISABLED and drives real agentic self-play traffic to grow the
# trace corpus. The v2 catalog has 12 task families (up from 6), with
# per-seed instruction generation, so 200 seeds x 12 families = 2,400 distinct
# prompts per full shard window instead of 600.
#
# Idle tuning is off in two independent ways, identical to traffic.sbatch:
#   1. `--enable-idle-tuning` is NOT passed.
#   2. demo/traffic-qwen8b.json sets "tuning_enabled": false.
# The untuned gateway path still passes the TraceStore into create_app, so every
# request is captured exactly as it would be with tuning on.
#
# buffer.max_age_days is 3650 so records are not age-pruned out from under us,
# and max_tokens is 200M so a shard of ~2k records at ~3.0k tokens each is
# nowhere near the eviction ceiling.
#
# ---------------------------------------------------------------------------
# SHARDING
# ---------------------------------------------------------------------------
# N copies of this script run CONCURRENTLY on N GPUs and are merged afterwards
# by demo/merge_traces.py. Everything that two shards could fight over is
# derived from $SHARD:
#
#   run_dir / SPEEDLM_HOME / traffic out   -- per-shard directory
#   wrapper.port, target.port, training_port -- per-shard, via a rewritten config
#   seed window                            -- per-shard, DISJOINT
#
# The ports matter even though each shard asks for its own GPU: SLURM will
# happily place two 1-GPU jobs on the same 8-GPU node, and both would then bind
# 127.0.0.1:8150. The seed window matters more subtly -- catalog.instance(seed)
# is a PURE FUNCTION of the seed, so two shards both starting at seed 0 would
# plant byte-identical workspaces and generate a duplicate corpus, not a bigger
# one.
#
# Seed range decisions:
#   bigcorpus-run1  used seeds [0, 100)
#     (traffic/summary.json: seeds=100, no seed_start -> default 0)
#   bigcorpus-shard0 used seeds [100, 160)  (seed_start=100+0*200=100, seeds=60)
#   bigcorpus-shard1 used seeds [300, 360)  (seed_start=100+1*200=300, seeds=60)
#   ... up to shard 9: [1900, 1960)
#
# V2 starts at seed 2000 to be unambiguously clear of any v1 shard window.
#
# Each shard starts from an EMPTY store.
#
# ---------------------------------------------------------------------------
# SIZING ARITHMETIC
# ---------------------------------------------------------------------------
# Measured from bigcorpus-run1 (v1 catalog, 6 families):
#   3,758 records total, 11.7 records/min, 5.92 records/trajectory
#
# V2 catalog has 12 families (double).  --seeds N means N seeds x 12 families
# = 12N trajectories per shard.
#
# Target: ~12,000 - 15,000 records across all v2 shards.
#
# With 5 shards x 40 seeds each:
#   40 seeds x 12 families = 480 trajectories per shard
#   480 x 5.92 = ~2,842 records per shard
#   2,842 x 5 shards  = ~14,210 total       <-- IN TARGET RANGE
#   2,842 / 11.7 = 243 min = 4.05 h per shard.  Inside 8 h wall.
#
# With 8 shards x 40 seeds each:
#   Same 4.05 h per shard.
#   2,842 x 8 = ~22,736 total               <-- ABOVE TARGET (generous margin)
#
# Decision: 8 shards x 30 seeds.
#   30 seeds x 12 families = 360 trajectories per shard
#   360 x 5.92 = ~2,131 records per shard
#   2,131 x 8 = ~17,050 total
#   2,131 / 11.7 = 182 min = 3.0 h per shard.  Comfortable in 8 h wall.
#
# Merged with old corpus (optional): 3,758 + 17,050 = ~20,800 records (~330 MB).
# ---------------------------------------------------------------------------

repo=/admin/home/ryan.kim/speedlm-fr
shard=${SHARD:-0}
run_dir=${TRAFFIC_RUN_DIR:-/data/ryan.kim/speedlm-runs/bigcorpus-v2-shard${shard}}
base_config="$repo/demo/traffic-qwen8b.json"
model="Qwen/Qwen3-8B"

# --- per-shard ports -------------------------------------------------------
# shard 0..9: same scheme as traffic.sbatch.
#   wrapper 8150..8240, target 8050..8140, training 8181..8271
# The ranges never cross each other. qwen-serve-low-pri on n-2 uses its own
# ports (a different SLURM job), so no collision on 127.0.0.1.
if (( shard < 0 || shard > 9 )); then
  echo "SHARD must be 0..9 (got $shard); the port scheme collides beyond that" >&2
  exit 1
fi
wrapper_port=$((8150 + shard * 10))
target_port=$((8050 + shard * 10))
training_port=$((8181 + shard * 10))

# --- per-shard seed window -------------------------------------------------
# V2 starts at seed 2000. 30 seeds per shard, with 170-unused padding so a
# follow-up pass can extend a shard in place without colliding with its neighbour.
seed_start=$((2000 + shard * 200))
seeds=${TRAFFIC_SEEDS:-30}

mkdir -p "$run_dir"
cd "$repo"

export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PATH="/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH"
export SPEEDLM_HOME="$run_dir/speedlm_home"
mkdir -p "$SPEEDLM_HOME/traces"

# --- rewrite the config with this shard's ports ----------------------------
config="$run_dir/traffic-qwen8b-shard${shard}.json"
"$repo/.venv/bin/python" - "$base_config" "$config" \
    "$wrapper_port" "$target_port" "$training_port" <<'PY'
import json
import sys

src, dst, wrapper_port, target_port, training_port = sys.argv[1:6]
with open(src) as handle:
    cfg = json.load(handle)
# Fail loudly rather than silently adding a key the loader would reject.
for section, key in (("wrapper", "port"), ("target", "port"), ("tuning", "training_port")):
    if key not in cfg[section]:
        raise SystemExit(f"{section}.{key} missing from {src}; port scheme is stale")
cfg["wrapper"]["port"] = int(wrapper_port)
cfg["target"]["port"] = int(target_port)
cfg["tuning"]["training_port"] = int(training_port)
with open(dst, "w") as handle:
    json.dump(cfg, handle, indent=2)
    handle.write("\n")
print(f"wrote {dst}: wrapper={wrapper_port} target={target_port} training={training_port}")
PY

{
  echo "hostname: $(hostname -f)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:?must run inside a SLURM allocation}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:?allocation exposed no GPU}"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "shard: $shard"
  echo "config: $config"
  echo "run_dir: $run_dir"
  echo "SPEEDLM_HOME: $SPEEDLM_HOME"
  echo "wrapper_port: $wrapper_port"
  echo "target_port: $target_port"
  echo "training_port: $training_port"
  echo "seed_window: [$seed_start, $((seed_start + seeds)))"
  echo "seed_store: none (empty store; merged later by demo/merge_traces.py)"
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

gateway_log="$run_dir/gateway-and-vllm.log"

# --tool-call-parser hermes deliberately OVERRIDES the profile's qwen3_xml.
# Job 376274 proved qwen3_xml wrong on hardware: Qwen3-8B emits Hermes-format
# calls, the xml parser matched nothing, and every trajectory died at turn one
# with an empty tool_calls array.
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

base_url="http://127.0.0.1:${wrapper_port}"

# --- readiness WITHOUT generating traffic ---------------------------------
# A /health poll goes through the gateway and is itself a request; run 3 polled
# for 20 minutes and salted the corpus with hundreds of empty exchanges. Grep
# the log instead. The wrapper port is in the match string on purpose: the child
# vLLM runs its own uvicorn on a different loopback port and logs the same line.
# NOTE: we use ${wrapper_port} (dynamic, per-shard) -- NOT a hardcoded port.
#   demo/cycle1e.sbatch has a latent bug where it hardcodes 8150, which only
#   works for shard 0 and silently matches the wrong uvicorn for other shards.
ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited before becoming ready; see $gateway_log" >&2
    tail -n 60 "$gateway_log" >&2
    exit 1
  fi
  if grep -q "Uvicorn running on http://127.0.0.1:${wrapper_port}" "$gateway_log" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "gateway not ready within 1800s" >&2; tail -n 60 "$gateway_log" >&2; exit 1; }
echo "gateway ready at $base_url (log match, zero traffic generated)"

# --- traffic --------------------------------------------------------------
"$repo/.venv/bin/python" scripts/run_agentic_traffic.py \
  --base-url "$base_url" \
  --model "$model" \
  --out "$run_dir/traffic" \
  --seeds "$seeds" \
  --seed-start "$seed_start" \
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

echo "=== corpus (this shard only; merge with demo/merge_traces.py) ==="
"$repo/.venv/bin/speedlm" traces stats --store "$SPEEDLM_HOME/traces/traces.jsonl" || true
echo "run directory: $run_dir"
