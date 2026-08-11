#!/bin/bash
#SBATCH --job-name=speedlm-agentenv
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5/slurm-%j.out
set -euo pipefail

# Serve Qwen3-8B with idle tuning, drive REAL agentic self-play traffic through
# the gateway, and let the tuner train the Eagle3 draft head on what it captured.
#
# The three previous agentic attempts all died in training-corpus admission, not
# in serving. The last one dropped 373 of 409 rows because they carried assistant
# turns the verifier "did not produce" -- except it had produced them, seconds
# earlier, in the same session. `trust_self_play_assistant_turns` is the fix, and
# it earns its trust rather than asserting it: the cycle attests the snapshot and
# fails loudly if the traffic is not genuinely self-play.

repo=/admin/home/ryan.kim/speedlm-fr
run_dir=/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5
config="$repo/configs/agentenv-qwen8b.json"
model="Qwen/Qwen3-8B"

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
} | tee "$run_dir/provenance.txt"
nvidia-smi || true

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
# The override is safe: profiles.py applies the user-supplied value LAST and sets
# source='user-supplied', which makes cli.py skip appending the profile's value,
# so vLLM receives hermes exactly once.
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
trap cleanup EXIT

base_url="http://127.0.0.1:8150"
ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited before becoming ready; see $gateway_log" >&2
    tail -n 60 "$gateway_log" >&2
    exit 1
  fi
  if curl --fail --silent --max-time 3 "$base_url/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "gateway not ready within 1800s" >&2; exit 1; }
echo "gateway ready at $base_url"

# --- smoke: prove ONE tool call dispatches before committing to 90 trajectories.
# Job 376274 burned an allocation discovering a parser mismatch on trajectory one
# and then repeating it ninety times. A single graded task costs about a minute
# and turns that into an immediate, legible failure.
"$repo/.venv/bin/python" scripts/run_agentic_traffic.py \
  --base-url "$base_url" \
  --model "$model" \
  --out "$run_dir/smoke" \
  --seeds 1 \
  --families call-chain-trace \
  --max-turns 10 \
  --python "$repo/.venv/bin/python" \
  2>&1 | tee "$run_dir/smoke.log"

echo "smoke passed: tool calls dispatch. Starting full traffic."

# --- traffic -------------------------------------------------------------
# 30 seeds x 6 families = 180 trajectories. Run 2 got 539 records from 90,
# of which 43 stopped at no_tool_call; doubling the seeds targets ~1000 records.
# configs/agentenv-qwen8b.json sets min_corpus_records=512 and
# training_window_records=512, so ~1000 clears the admission floor with roughly
# 2x margin -- the window is filled from a surplus rather than from every record
# the buffer happens to hold.
"$repo/.venv/bin/python" scripts/run_agentic_traffic.py \
  --base-url "$base_url" \
  --model "$model" \
  --out "$run_dir/traffic" \
  --seeds 30 \
  --max-turns 24 \
  --max-output-tokens 1024 \
  --request-timeout 600 \
  --trajectory-wall-clock 1500 \
  --stop-after-seconds 14400 \
  --python "$repo/.venv/bin/python" \
  2>&1 | tee "$run_dir/traffic.log"

echo "traffic complete; waiting for the idle-tuning cycle"

deadline=$(( SECONDS + 25200 ))
while (( SECONDS < deadline )); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "gateway exited during tuning; see $gateway_log" >&2
    break
  fi
  if find "$SPEEDLM_HOME" -name 'decision*.json' 2>/dev/null | grep -q .; then
    echo "decision artifact found"
    break
  fi
  sleep 30
done

mkdir -p "$run_dir/results"
find "$SPEEDLM_HOME" -name '*.json' -size -10M -exec cp --parents {} "$run_dir/results/" \; 2>/dev/null || true
tail -n 400 "$gateway_log" > "$run_dir/gateway-tail.log" || true

echo "=== traffic summary ==="
head -60 "$run_dir/traffic/summary.json" 2>/dev/null || echo "(no traffic summary)"
echo "=== decisions ==="
find "$SPEEDLM_HOME" -name 'decision*.json' -exec echo '--- {}' \; -exec cat {} \; 2>/dev/null | head -120 || true
echo "run directory: $run_dir"
