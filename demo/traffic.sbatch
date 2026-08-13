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
# one. bigcorpus-run1 already consumed seeds 0-99; shard windows start at 100.
#
# Each shard starts from an EMPTY store. Seeding every shard from run5 would
# have merged N copies of the same 1065 records. merge_traces.py folds run5 and
# bigcorpus-run1 back in exactly once, de-duplicated on the record id.

repo=/admin/home/ryan.kim/speedlm-fr
shard=${SHARD:-0}
run_dir=${TRAFFIC_RUN_DIR:-/data/ryan.kim/speedlm-runs/bigcorpus-shard${shard}}
base_config="$repo/demo/traffic-qwen8b.json"
model="Qwen/Qwen3-8B"

# --- per-shard ports -------------------------------------------------------
# shard 0..9 only: target tops out at 8140, one below the wrapper base of 8150,
# so the two ranges never cross.
if (( shard < 0 || shard > 9 )); then
  echo "SHARD must be 0..9 (got $shard); the port scheme collides beyond that" >&2
  exit 1
fi
wrapper_port=$((8150 + shard * 10))
target_port=$((8050 + shard * 10))
training_port=$((8181 + shard * 10))

# --- per-shard seed window -------------------------------------------------
# 200 wide but only 60 used, so a follow-up pass can extend a shard in place
# without ever colliding with its neighbour.
seed_start=$((100 + shard * 200))
seeds=${TRAFFIC_SEEDS:-60}

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

base_url="http://127.0.0.1:${wrapper_port}"

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
  if grep -q "Uvicorn running on http://127.0.0.1:${wrapper_port}" "$gateway_log" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "gateway not ready within 1800s" >&2; tail -n 60 "$gateway_log" >&2; exit 1; }
echo "gateway ready at $base_url (log match, zero traffic generated)"

# --- traffic --------------------------------------------------------------
# Sizing, from bigcorpus-run1 measured on this same hardware (3758 records
# total, 2693 of them newly generated in ~3.9 h => 11.7 records/min), and from
# run5's shape (5.92 records per trajectory, ~3.0k tokens per record):
#   target 200 MB at the observed 15.8 KB/record => ~12,950 records
#   already have 3,758 => need ~9,200 more => 5 shards x ~1,840
#   1,840 / 5.92 = 311 trajectories / 6 families = 52 seeds -> use 60 for slack
#   60 seeds x 6 families = 360 trajectories x 5.92 = ~2,130 records per shard
#   ~2,130 / 11.7 = 182 min = 3.0 h, inside the 4 h --stop-after-seconds cap and
#   well inside the 8 h wall.
# Merged: 3,758 + 5 x ~2,130 = ~14,400 records (~227 MB), overshooting 200 MB on
# purpose so trajectory failures do not land us short.
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
