#!/bin/bash
# Materialize an immutable snapshot of the repo at a commit and emit an sbatch
# that runs an e2e flavor from that snapshot instead of the live working tree.
#
# WHY THIS EXISTS
# ---------------
# Every hand-written job.sbatch under /data/ryan.kim/speedlm-runs/ does
# `cd /admin/home/ryan.kim/speedlm-fr` and runs pytest from the LIVE tree.  The
# `git rev-parse HEAD` line those scripts echo is a LABEL, not a guarantee: an
# edit landing while the job runs changes the code the job executes, and the
# recorded commit becomes a lie.  That has repeatedly forced development to halt
# for the duration of a GPU run.  This script removes the coupling.
#
# HOW ISOLATION IS ACHIEVED  (verified, see docs/e2e-harness.md "Snapshot runs")
# ------------------------------------------------------------------------------
# `git archive` extracts a detached copy with no .git and no link back to the
# repo.  That alone is NOT enough, and neither is `git worktree add`:
#
#     /admin/home/ryan.kim/speedlm-fr/.venv is an EDITABLE install.  Its
#     site-packages contains _editable_impl_speedlm.pth holding the single line
#     `/admin/home/ryan.kim/speedlm-fr/src`.  .pth files are processed by `site`
#     while it walks site-packages, so that path is APPENDED to sys.path.  Any
#     interpreter using that venv therefore resolves `import speedlm` back to
#     the live tree no matter which directory it was launched from.
#
# PYTHONPATH is the fix, and it was confirmed empirically rather than assumed.
# PYTHONPATH entries are inserted by `site` BEFORE site-packages is scanned, so
# they precede the .pth entry.  Measured sys.path indices with
# PYTHONPATH=<snapshot>/src under the project venv:
#
#     index 1 : <snapshot>/src              <- PYTHONPATH, wins
#     index 5 : .../.venv/lib/.../site-packages
#     index 6 : /admin/home/ryan.kim/speedlm-fr/src   <- the editable .pth
#
# The snapshot shadows the live tree by five positions.  Belt and braces, the
# emitted sbatch ASSERTS this at runtime and exits non-zero before touching a
# GPU if `speedlm.__file__` does not live under the snapshot.
#
# Subprocesses are covered too: every engine spawn in the e2e tests builds its
# environment with `os.environ.copy()` (test_serving_draft_hot_swap.py:219,
# test_serving_activation_capture.py:687, test_live_idle_tuning.py:859), so the
# vLLM worker that loads `--worker-extension-cls speedlm...` inherits PYTHONPATH
# and resolves the extension from the snapshot as well.
#
# USAGE
# -----
#   scripts/make_snapshot_run.sh --flavor idle-tuning \
#       --tuning-config /data/ryan.kim/speedlm-runs/<run>/config.json
#
#   scripts/make_snapshot_run.sh --flavor hot-swap
#   scripts/make_snapshot_run.sh --flavor activation-capture
#
# Run `--help` for the full option list.  This script NEVER submits anything; it
# prints the sbatch path and the command to submit it.

set -euo pipefail

REPO=/admin/home/ryan.kim/speedlm-fr
SNAPSHOT_ROOT=/data/ryan.kim/speedlm-snapshots
RUN_ROOT=/data/ryan.kim/speedlm-runs
VLLM_ENV=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm
PYLIBS_PYTEST=/data/ryan.kim/pylibs/pytest
CORPUS=/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl

# The gateway flavors default SPEEDLM_*_VLLM_ARGS to "[]", i.e. no bounds at
# all, which on a shared GPU lets vLLM size its KV cache for the whole card.
# Keep enough headroom for the gpt-oss-20b verifier plus its eagle3 drafter;
# 0.75 completed the full idle-tuning run while the unbounded launch OOMed.
DEFAULT_GATEWAY_VLLM_ARGS='[ "--max-model-len", "4096", "--gpu-memory-utilization", "0.75", "--enforce-eager" ]'

usage() {
    cat <<'EOF'
make_snapshot_run.sh -- pin a GPU run to an immutable snapshot of the repo.

Required:
  --flavor F            idle-tuning | activation-capture | capture-overhead
                        | hot-swap | live-vllm | proxy-overhead
                        | token-fidelity | model-matrix | capture-matrix
                        | agent-harness | config-matrix

Optional:
  --commit REF          commit/ref to snapshot            (default: HEAD)
  --run-name NAME       run directory name under
                        /data/ryan.kim/speedlm-runs       (default: <flavor>-snap-<UTC>)
  --run-root PATH       parent directory for run artifacts
                        (default: /data/ryan.kim/speedlm-runs)
  --time HH:MM:SS       SBATCH --time                     (default: per flavor)
  --tuning-timeout SEC  SPEEDLM_E2E_TUNING_TIMEOUT       (idle-tuning only;
                        default: wall time minus 15 minutes)
  --partition P         SBATCH --partition                (default: n)
  --cpus N              SBATCH --cpus-per-task            (default: 16)
  --tuning-config PATH  SPEEDLM_E2E_TUNING_CONFIG         (idle-tuning only, required)
  --tuning-profile PATH SPEEDLM_E2E_TUNING_PROFILE        (idle-tuning, optional)
  --verifier ID         SPEEDLM_E2E_VERIFIER_MODEL        (capture/capture-
                        overhead/hot-swap)
  --drafter ID          reference draft (config-matrix), or
                        SPEEDLM_E2E_DRAFTER_MODEL (capture/capture-overhead/
                        hot-swap)
  --candidate-drafter ID
                        candidate draft for config-matrix (required)
  --inject-ms MS        SPEEDLM_E2E_CAPTURE_OVERHEAD_INJECT_MS
                        (capture-overhead only, optional).  Adds MS
                        milliseconds to every capture-ON request, so the
                        paired-difference machinery can be shown to detect a
                        regression on real hardware.  A value above the test's
                        100 ms TTFT bound MUST make the run fail; if it does
                        not, the measurement is not measuring.
  --inject-percent P    SPEEDLM_CONFIG_MATRIX_INJECT_SLOWDOWN_PERCENT
                        (config-matrix only, optional). Applies a synthetic P%
                        candidate slowdown to the regression decision while
                        preserving raw measurements; a value above the 5%
                        default budget proves that the harness can fail.
  --drafter-dir PATH    SPEEDLM_E2E_DRAFTER_DIR           (hot-swap, optional)
  --runner R            auto | v1 | v2                    (default: auto)
                        v1/v2 force VLLM_USE_V2_MODEL_RUNNER=0/1; auto lets
                        vLLM decide.  A non-auto runner is also folded into the
                        default run name so cells do not collide.
  --prompt-set NAME     SPEEDLM_E2E_PROMPT_SET            (capture, optional)
  --target-layer-ids C  SPEEDLM_E2E_TARGET_LAYER_IDS      (capture, optional, csv)
  --hf-reference 0|1    SPEEDLM_E2E_HF_REFERENCE          (capture, optional)
  --strict-verdict 0|1  SPEEDLM_E2E_STRICT_VERDICT        (capture, optional)
  --pytest-k EXPR       pytest -k EXPR                    (optional; without it
                        the whole test file runs, i.e. Stage 0 AND the
                        prefix-cache test together).  For config-matrix this
                        is required and must be one exact matrix cell name.
  --corpus PATH         SPEEDLM_E2E_PROMPT_CORPUS         (idle-tuning/config-matrix)
  --no-corpus           omit the corpus; test falls back to synthetic prompts
  --model ID            served model                      (default: per flavor)
  --matrix-cell NAME    SPEEDLM_MATRIX_CELL               (model-matrix only, required)
  --hf-home PATH        HF_HOME                           (default: /data/ryan.kim/hf-cache)
  --vllm-args JSON      a JSON array of strings, exported as the flavor's own
                        vLLM-args variable (SPEEDLM_E2E_VLLM_ARGS, or the
                        CAPTURE/AGENT equivalent)
  --force               re-extract the snapshot even if it already exists
  -h | --help           this message

The snapshot lands in /data/ryan.kim/speedlm-snapshots/<full-sha>/ and is made
read-only.  Re-running for the same commit reuses it (content is identical by
construction), so snapshots are cheap: the tree is ~2.5 MB.

STAGE 0 RUNNER x MODEL MATRIX
----------------------------
Both gpt-oss models are already in the default HF_HOME=/data/ryan.kim/hf-cache
(openai/gpt-oss-20b has 24 hidden layers), so the four cells below need no
extra plumbing beyond --verifier/--drafter/--runner:

  # qwen3-8b x v1
  scripts/make_snapshot_run.sh --flavor activation-capture --runner v1 \
      --pytest-k test_stage0_activation_capture

  # qwen3-8b x v2
  scripts/make_snapshot_run.sh --flavor activation-capture --runner v2 \
      --pytest-k test_stage0_activation_capture

  # gpt-oss-20b x v1
  scripts/make_snapshot_run.sh --flavor activation-capture --runner v1 \
      --verifier openai/gpt-oss-20b \
      --drafter RedHatAI/gpt-oss-20b-speculator.eagle3 \
      --run-name capture-gptoss20b-v1 \
      --pytest-k test_stage0_activation_capture

  # gpt-oss-20b x v2
  scripts/make_snapshot_run.sh --flavor activation-capture --runner v2 \
      --verifier openai/gpt-oss-20b \
      --drafter RedHatAI/gpt-oss-20b-speculator.eagle3 \
      --run-name capture-gptoss20b-v2 \
      --pytest-k test_stage0_activation_capture

The two qwen cells need no --run-name: --runner is folded into the default,
giving activation-capture-v1-snap-<UTC> and activation-capture-v2-snap-<UTC>.
The gpt-oss cells share that default with the qwen ones, so name them.

Nothing is submitted.  The script prints the sbatch to run.
EOF
}

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
flavor=""
commit_ref="HEAD"
run_name=""
sbatch_time=""
tuning_timeout=""
partition="n"
cpus=16
tuning_config=""
tuning_profile=""
verifier=""
drafter=""
candidate_drafter=""
drafter_dir=""
inject_ms=""
inject_percent=""
runner="auto"
prompt_set=""
target_layer_ids=""
hf_reference=""
strict_verdict=""
pytest_k=""
config_matrix_cell=""
corpus="$CORPUS"
model=""
matrix_cell=""
hf_home=/data/ryan.kim/hf-cache
hf_home_set=0
vllm_args=""
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flavor)         flavor="$2"; shift 2 ;;
        --commit)         commit_ref="$2"; shift 2 ;;
        --run-name)       run_name="$2"; shift 2 ;;
        --run-root)       RUN_ROOT="$2"; shift 2 ;;
        --time)           sbatch_time="$2"; shift 2 ;;
        --tuning-timeout) tuning_timeout="$2"; shift 2 ;;
        --partition)      partition="$2"; shift 2 ;;
        --cpus)           cpus="$2"; shift 2 ;;
        --tuning-config)  tuning_config="$2"; shift 2 ;;
        --tuning-profile) tuning_profile="$2"; shift 2 ;;
        --verifier)       verifier="$2"; shift 2 ;;
        --drafter)        drafter="$2"; shift 2 ;;
        --candidate-drafter) candidate_drafter="$2"; shift 2 ;;
        --drafter-dir)    drafter_dir="$2"; shift 2 ;;
        --inject-ms)      inject_ms="$2"; shift 2 ;;
        --inject-percent) inject_percent="$2"; shift 2 ;;
        --runner)         runner="$2"; shift 2 ;;
        --prompt-set)     prompt_set="$2"; shift 2 ;;
        --target-layer-ids) target_layer_ids="$2"; shift 2 ;;
        --hf-reference)   hf_reference="$2"; shift 2 ;;
        --strict-verdict) strict_verdict="$2"; shift 2 ;;
        --pytest-k)       pytest_k="$2"; shift 2 ;;
        --corpus)         corpus="$2"; shift 2 ;;
        --no-corpus)      corpus=""; shift ;;
        --model)          model="$2"; shift 2 ;;
        --matrix-cell)    matrix_cell="$2"; shift 2 ;;
        --hf-home)        hf_home="$2"; hf_home_set=1; shift 2 ;;
        --vllm-args)      vllm_args="$2"; shift 2 ;;
        --force)          force=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$flavor" ]] || { echo "error: --flavor is required" >&2; usage >&2; exit 2; }

case "$runner" in
    auto|v1|v2) ;;
    *) echo "error: invalid --runner '$runner' (want auto|v1|v2)" >&2; exit 2 ;;
esac

# --inject-ms is a fault injection into ONE test's ON arm.  Reject it anywhere
# else rather than exporting a variable nothing reads: a silently-ignored
# fault injection would look like "the harness cannot detect this regression".
if [[ -n "$inject_ms" ]]; then
    [[ "$flavor" == "capture-overhead" ]] || {
        echo "error: --inject-ms is valid only for --flavor capture-overhead" >&2
        exit 2
    }
    [[ "$inject_ms" =~ ^[0-9]+$ ]] || {
        echo "error: --inject-ms must be a non-negative integer number of milliseconds" >&2
        exit 2
    }
fi

if [[ -n "$candidate_drafter" && "$flavor" != "config-matrix" ]]; then
    echo "error: --candidate-drafter is valid only for --flavor config-matrix" >&2
    exit 2
fi

if [[ -n "$inject_percent" ]]; then
    [[ "$flavor" == "config-matrix" ]] || {
        echo "error: --inject-percent is valid only for --flavor config-matrix" >&2
        exit 2
    }
    [[ "$inject_percent" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
        echo "error: --inject-percent must be a non-negative number" >&2
        exit 2
    }
fi

# --------------------------------------------------------------------------
# Per-flavor wiring.
#
# The interpreter differs by flavor and it is NOT a free choice:
#   * test_live_idle_tuning.py drives the gateway over HTTP and imports no
#     torch, so it runs under the project venv (which has pytest but NO torch).
#   * test_serving_activation_capture.py imports torch and safetensors directly
#     (its pytestmark skipif at :63 turns the whole file into a skip without
#     them), so it MUST run under the vLLM venv -- which in turn has no pytest
#     of its own, hence /data/ryan.kim/pylibs/pytest on PYTHONPATH.
#   * test_serving_draft_hot_swap.py parses safetensors headers by hand and
#     needs no torch in-process, but it is kept on the vLLM venv to match the
#     existing hotswap-smoke-TEMPLATE convention and the engine it drives.
#   * the six gateway flavors below (live-vllm, proxy-overhead, token-fidelity,
#     model-matrix, capture-matrix, agent-harness) drive `speedlm vllm serve`
#     over HTTP and import no torch in-process -- model-matrix imports only
#     speedlm.training.rows and the chatml/harmony templates, which are
#     torch-free -- so they run under the project venv.  They DO need the
#     `speedlm` console script on PATH: it lives in an installed venv, never in
#     the snapshot, and each test falls back to shutil.which() when
#     REPO_ROOT/.venv is absent (which it always is under a snapshot).  Running
#     the CLI from the live venv does not break provenance: the child inherits
#     PYTHONPATH, so `import speedlm` still resolves to the snapshot.
#
# Every one of these six also uses its own artifact-dir variable, hence
# `artifact_var` rather than a hardcoded SPEEDLM_E2E_ARTIFACT_DIR.
# --------------------------------------------------------------------------
artifact_var="SPEEDLM_E2E_ARTIFACT_DIR"
vllm_args_var="SPEEDLM_E2E_VLLM_ARGS"
extra_path=""

case "$flavor" in
    idle-tuning)
        test_path="tests/e2e/test_live_idle_tuning.py"
        gate_var="SPEEDLM_E2E_IDLE_TUNING"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        job_suffix="idle"
        default_time="12:00:00"
        [[ -n "$tuning_config" ]] || {
            echo "error: --tuning-config is required for --flavor idle-tuning" >&2
            echo "       (test_live_idle_tuning.py:226 hard-fails without SPEEDLM_E2E_TUNING_CONFIG)" >&2
            exit 2
        }
        ;;
    activation-capture)
        test_path="tests/e2e/test_serving_activation_capture.py"
        gate_var="SPEEDLM_E2E_ACTIVATION_CAPTURE"
        interpreter="$VLLM_ENV/bin/python"
        extra_pythonpath=":$PYLIBS_PYTEST"
        job_suffix="capture"
        default_time="03:00:00"
        : "${verifier:=Qwen/Qwen3-8B}"
        : "${drafter:=RedHatAI/Qwen3-8B-speculator.eagle3}"
        corpus=""   # this flavor does not read SPEEDLM_E2E_PROMPT_CORPUS
        ;;
    capture-overhead)
        # Measures what activation capture costs the serving path (TTFT, tok/s)
        # by toggling the hook on ONE engine.  Wired exactly like
        # activation-capture: same worker extension, same verifier/drafter env
        # vars (exported by the shared case arm below), same vLLM-venv
        # interpreter + PYLIBS_PYTEST.  It takes NO vLLM args from the launcher
        # for the same reason activation-capture takes none -- the test spawns
        # the engine itself and hardcodes its own argv and its own bounds, so
        # the gateway defaults would not reach the engine anyway.  That argv
        # includes --enforce-eager: capture arms by mutating the model's
        # aux_hidden_state_layers at runtime, which a compiled/CUDA-graphed
        # forward ignores, and the resulting 4-vs-3 mismatch kills EngineCore
        # (SLURM 370798).  See the module docstring for what that costs the
        # interpretation of the numbers.
        #
        # ~246 streaming requests at 128 output tokens on top of one model load.
        test_path="tests/e2e/test_serving_activation_capture_overhead.py"
        gate_var="SPEEDLM_E2E_CAPTURE_OVERHEAD"
        interpreter="$VLLM_ENV/bin/python"
        extra_pythonpath=":$PYLIBS_PYTEST"
        job_suffix="capoverhead"
        default_time="02:00:00"
        : "${verifier:=Qwen/Qwen3-8B}"
        : "${drafter:=RedHatAI/Qwen3-8B-speculator.eagle3}"
        corpus=""   # this flavor does not read SPEEDLM_E2E_PROMPT_CORPUS
        ;;
    hot-swap)
        test_path="tests/e2e/test_serving_draft_hot_swap.py"
        gate_var="SPEEDLM_E2E_DRAFT_HOT_SWAP"
        interpreter="$VLLM_ENV/bin/python"
        extra_pythonpath=":$PYLIBS_PYTEST"
        job_suffix="hotswap"
        default_time="02:00:00"
        : "${verifier:=Qwen/Qwen3-8B}"
        : "${drafter:=RedHatAI/Qwen3-8B-speculator.eagle3}"
        corpus=""   # this flavor does not read SPEEDLM_E2E_PROMPT_CORPUS
        ;;
    live-vllm)
        test_path="tests/e2e/test_live_vllm.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        job_suffix="livevllm"
        # One model load plus three chat requests (non-stream, stream, tools),
        # a trace drain and two CLI calls.  Load dominates.
        default_time="01:30:00"
        : "${model:=Qwen/Qwen3-8B}"
        corpus=""
        ;;
    proxy-overhead)
        test_path="tests/e2e/test_proxy_overhead.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        job_suffix="overhead"
        # ~470 requests at 64 output tokens each (warmup, 8 latency repeats
        # streaming and non-streaming on both paths, then 4 repeats over
        # concurrency 1/4/16/32 direct and via the gateway) on top of the load.
        default_time="02:00:00"
        : "${model:=Qwen/Qwen3-8B}"
        corpus=""
        ;;
    token-fidelity)
        test_path="tests/e2e/test_token_fidelity.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        job_suffix="fidelity"
        # Three chat requests plus six /tokenize round trips.  Load dominates.
        default_time="01:30:00"
        # test_token_fidelity.py:56 hardcodes MODEL = "Qwen/Qwen3.5-2B" with no
        # env override, and that model is NOT in /data/ryan.kim/hf-cache -- the
        # only local copy is under ob-cache.  Point HF_HOME there unless the
        # caller said otherwise, or the offline load fails instantly.
        [[ $hf_home_set -eq 1 ]] || hf_home=/data/ryan.kim/ob-cache
        model=""
        corpus=""
        ;;
    model-matrix)
        test_path="tests/e2e/test_model_matrix.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        artifact_var="SPEEDLM_MATRIX_ARTIFACT_DIR"
        job_suffix="matrix"
        # Two or three requests; the cell's startup timeout defaults to 900s
        # and a 20B eagle3 cell is the slow case.
        default_time="01:30:00"
        [[ -n "$matrix_cell" ]] || {
            echo "error: --matrix-cell is required for --flavor model-matrix" >&2
            echo "       (test_model_matrix.py:183 hard-fails without SPEEDLM_MATRIX_CELL)" >&2
            echo "       cells: gpt-oss-20b-eagle3 qwen3.5-9b-mtp qwen3.5-2b-none qwen3.5-2b-ngram" >&2
            exit 2
        }
        # The cell pins its own model; --model is not consulted here.
        model=""
        corpus=""
        ;;
    capture-matrix)
        test_path="tests/e2e/test_capture_harness_matrix.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        artifact_var="SPEEDLM_CAPTURE_ARTIFACT_DIR"
        job_suffix="capmatrix"
        # Eight exchanges over raw httpx and a separate OpenAI SDK subprocess,
        # then a manifest and trace drain.  Load dominates.
        default_time="01:30:00"
        : "${model:=Qwen/Qwen3-8B}"
        corpus=""
        ;;
    agent-harness)
        test_path="tests/e2e/test_agent_harness.py"
        gate_var="SPEEDLM_E2E"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        artifact_var="SPEEDLM_AGENT_ARTIFACT_DIR"
        job_suffix="agent"
        # A three-turn scripted tool-calling conversation, capped by
        # SPEEDLM_AGENT_SUBPROCESS_TIMEOUT=420 on top of the load.
        default_time="01:30:00"
        : "${model:=openai/gpt-oss-20b}"
        corpus=""
        ;;
    config-matrix)
        test_path="tests/e2e/test_inference_configuration_matrix.py"
        gate_var="SPEEDLM_E2E_CONFIG_MATRIX"
        interpreter="$REPO/.venv/bin/python"
        extra_pythonpath=""
        extra_path=":$REPO/.venv/bin"
        artifact_var="SPEEDLM_CONFIG_MATRIX_ARTIFACT_DIR"
        job_suffix="configmatrix"
        # Twelve cells and four fresh engine lifecycles per cell. Startup time
        # dominates, and the long-context/concurrency-32 cells are intentionally
        # substantial enough to exercise graph capture and KV-cache pressure.
        default_time="12:00:00"
        : "${model:=Qwen/Qwen3-8B}"
        : "${drafter:=RedHatAI/Qwen3-8B-speculator.eagle3}"
        [[ -n "$candidate_drafter" ]] || {
            echo "error: --candidate-drafter is required for --flavor config-matrix" >&2
            exit 2
        }
        [[ -n "$corpus" ]] || {
            echo "error: config-matrix requires --corpus (do not use --no-corpus)" >&2
            exit 2
        }
        [[ -n "$pytest_k" ]] || {
            echo "error: config-matrix requires --pytest-k with one exact cell name" >&2
            exit 2
        }
        case "$pytest_k" in
            eager-c1-short|eager-c1-long|eager-c8-short|eager-c8-long|eager-c32-short|eager-c32-long|cuda_graphs-c1-short|cuda_graphs-c1-long|cuda_graphs-c8-short|cuda_graphs-c8-long|cuda_graphs-c32-short|cuda_graphs-c32-long)
                config_matrix_cell="$pytest_k"
                ;;
            *)
                echo "error: invalid config-matrix --pytest-k cell '$pytest_k'" >&2
                echo "       want {eager,cuda_graphs}-c{1,8,32}-{short,long}" >&2
                exit 2
                ;;
        esac
        # Hyphens are operators in pytest's -k expression language.  The
        # collection plugin below adds this underscore spelling as a keyword
        # to the one live test item after narrowing its MATRIX global.
        pytest_k="${config_matrix_cell//-/_}"
        ;;
    *)
        echo "error: unknown flavor '$flavor' (want idle-tuning|activation-capture|capture-overhead|hot-swap|live-vllm|proxy-overhead|token-fidelity|model-matrix|capture-matrix|agent-harness|config-matrix)" >&2
        exit 2
        ;;
esac

sbatch_time="${sbatch_time:-$default_time}"

# Keep the idle-tuning test's own deadline inside the allocation.  Slurm time
# accepts several formats; this launcher accepts HH:MM:SS and D-HH:MM:SS and
# normalizes them so both deadlines can be printed and compared before an
# allocation exists.
slurm_time_seconds() {
    local value="$1" days=0 hours minutes seconds
    if [[ "$value" =~ ^([0-9]+)-([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        days="${BASH_REMATCH[1]}"
        hours="${BASH_REMATCH[2]}"
        minutes="${BASH_REMATCH[3]}"
        seconds="${BASH_REMATCH[4]}"
    elif [[ "$value" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        hours="${BASH_REMATCH[1]}"
        minutes="${BASH_REMATCH[2]}"
        seconds="${BASH_REMATCH[3]}"
    else
        echo "error: invalid --time '$value' (want HH:MM:SS or D-HH:MM:SS)" >&2
        return 2
    fi
    (( 10#$minutes < 60 && 10#$seconds < 60 )) || {
        echo "error: invalid --time '$value' (minutes and seconds must be < 60)" >&2
        return 2
    }
    echo $((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

wall_time_seconds="$(slurm_time_seconds "$sbatch_time")"
if [[ "$flavor" == "idle-tuning" ]]; then
    timeout_margin_seconds=900
    if [[ -z "$tuning_timeout" ]]; then
        (( wall_time_seconds > timeout_margin_seconds )) || {
            echo "error: idle-tuning wall time must exceed the 900-second timeout margin" >&2
            exit 2
        }
        tuning_timeout=$((wall_time_seconds - timeout_margin_seconds))
    fi
    [[ "$tuning_timeout" =~ ^[0-9]+$ ]] || {
        echo "error: --tuning-timeout must be an integer number of seconds" >&2
        exit 2
    }
    tuning_timeout=$((10#$tuning_timeout))
    (( tuning_timeout > 0 && tuning_timeout < wall_time_seconds )) || {
        echo "error: tuning timeout ($tuning_timeout seconds) must be positive and less than Slurm wall time ($wall_time_seconds seconds)" >&2
        exit 2
    }
elif [[ -n "$tuning_timeout" ]]; then
    echo "error: --tuning-timeout is valid only for --flavor idle-tuning" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Resolve the commit to a full sha.  A ref is ambiguous over time; the snapshot
# directory is named for the sha so the mapping run -> code is one-to-one.
# --------------------------------------------------------------------------
cd "$REPO"
sha="$(git rev-parse --verify "${commit_ref}^{commit}")"
short_sha="${sha:0:12}"
snapshot="$SNAPSHOT_ROOT/$sha"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
# A runner sweep generates cells that differ ONLY by --runner, and two of them
# started in the same second would collide on the "run directory already exists"
# refusal below.  Fold a non-auto runner into the default name; an explicit
# --run-name is left exactly as given.
runner_tag=""
[[ "$runner" == "auto" ]] || runner_tag="-$runner"
run_name="${run_name:-${flavor}${runner_tag}-snap-${stamp}}"
run_dir="$RUN_ROOT/$run_name"

if [[ -e "$run_dir" ]]; then
    echo "error: run directory already exists: $run_dir" >&2
    exit 1
fi

# Record dirtiness as provenance, not just as terminal output.  git archive
# serializes the COMMIT, so every tracked modification and untracked file is
# absent from the snapshot.
source_tree_dirty=0
if [[ -n "$(git status --porcelain)" ]]; then
    source_tree_dirty=1
    echo "WARNING: working tree has uncommitted changes." >&2
    echo "         The snapshot contains commit $short_sha ONLY; local edits are NOT included." >&2
fi

# --------------------------------------------------------------------------
# Materialize the snapshot.
#
# `git archive` is preferred over `git worktree add --detach` here:
#   - no .git, no administrative link, so the main repo cannot be perturbed and
#     `git worktree list` stays clean even if a snapshot outlives its job;
#   - the result can be chmod'd read-only, which `git worktree` tolerates badly;
#   - a worktree does not isolate anyway (see the header: the editable .pth is
#     what binds imports to the live tree, not the checkout location).
# --------------------------------------------------------------------------
marker="$snapshot/.snapshot-complete"
if [[ -f "$marker" && $force -eq 0 ]]; then
    echo "reusing existing snapshot: $snapshot"
else
    if [[ -d "$snapshot" ]]; then
        chmod -R u+w "$snapshot"
        rm -rf "$snapshot"
    fi
    mkdir -p "$snapshot"
    git archive "$sha" | tar -x -C "$snapshot"
    {
        echo "commit=$sha"
        echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "created_by=${USER:-unknown}@$(hostname -f)"
        echo "source_repo=$REPO"
    } > "$marker"
    # Read-only: an accidental edit here should fail, not silently alter a run.
    chmod -R a-w "$snapshot"
    echo "created snapshot: $snapshot"
fi

[[ -d "$snapshot/src/speedlm" ]] || { echo "error: snapshot has no src/speedlm" >&2; exit 1; }
[[ -f "$snapshot/$test_path" ]] || { echo "error: snapshot has no $test_path" >&2; exit 1; }

mkdir -p "$run_dir/results"

# The configuration matrix is implemented as one live pytest item that loops
# over MATRIX, so pytest -k cannot select an internal cell by itself.  Keep the
# test immutable and add a run-local collection plugin that narrows MATRIX
# before the test executes, then tags the item with the keyword derived from
# --pytest-k.  The launcher that generates this plugin is in the pinned
# snapshot, and the plugin itself lives with the run outputs.
if [[ -n "$config_matrix_cell" ]]; then
    cat > "$run_dir/speedlm_config_matrix_cell.py" <<'PYEOF'
import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    cell_name = os.environ["SPEEDLM_CONFIG_MATRIX_CELL"]
    keyword = cell_name.replace("-", "_")
    matched = 0
    for item in items:
        if item.name != "test_speculative_inference_configuration_matrix":
            continue
        selected = tuple(cell for cell in item.module.MATRIX if cell.name == cell_name)
        if len(selected) != 1:
            raise pytest.UsageError(
                f"configuration matrix cell {cell_name!r} matched {len(selected)} cells"
            )
        item.module.MATRIX = selected
        item.extra_keyword_matches.add(keyword)
        matched += 1
    if matched != 1:
        raise pytest.UsageError(
            f"configuration matrix selector found {matched} live test items"
        )
PYEOF
fi

cat > "$run_dir/snapshot-provenance.txt" <<EOF
snapshot_commit=$sha
snapshot_source_ref=$commit_ref
source_repo=$REPO
source_tree_dirty_at_generation=$source_tree_dirty
generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
note=git archive contains the snapshot commit only; working-tree changes are excluded
EOF

# --------------------------------------------------------------------------
# Emit the sbatch.
#
# #SBATCH --output is read by slurm before any shell runs, so the path must be
# spelled out literally -- it cannot be built from a shell variable.  That is
# why the heredoc below is unquoted and interpolates at generation time.
# --------------------------------------------------------------------------
sbatch_path="$run_dir/job.sbatch"
pythonpath="$snapshot/src${extra_pythonpath}"
pytest_plugin_arg=""
if [[ -n "$config_matrix_cell" ]]; then
    pythonpath="$run_dir:$pythonpath"
    pytest_plugin_arg=" -p speedlm_config_matrix_cell"
fi

{
cat <<EOF
#!/bin/bash
#SBATCH --job-name=speedlm-snap-$job_suffix
#SBATCH --partition=$partition
#SBATCH --gres=gpu:1
#SBATCH --time=$sbatch_time
#SBATCH --cpus-per-task=$cpus
#SBATCH --output=$run_dir/slurm-%j.out
set -euo pipefail

# GENERATED by scripts/make_snapshot_run.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
# Do not hand-edit: regenerate instead, so the snapshot and the script agree.
#
# This job does NOT read /admin/home/ryan.kim/speedlm-fr.  All code comes from
# the read-only snapshot below, so edits to the live tree while this job runs
# cannot change what executes.
snapshot=$snapshot
run_dir=$run_dir
commit=$sha
interpreter=$interpreter
slurm_wall_time=$sbatch_time
slurm_wall_time_seconds=$wall_time_seconds
source_tree_dirty_at_generation=$source_tree_dirty

cd "\$snapshot"

export PATH="$VLLM_ENV/bin$extra_path:\$PATH"
export HF_HOME=$hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Do not inherit result-relaxing switches from the submission shell.  The two
# activation-capture controls are re-exported below only when their launcher
# options were supplied.  The runner override is likewise owned by --runner.
unset SPEEDLM_E2E_ALLOW_UNMEASURED_GATE \
    SPEEDLM_E2E_STRICT_VERDICT \
    SPEEDLM_E2E_HF_REFERENCE \
    VLLM_USE_V2_MODEL_RUNNER

# The snapshot must beat the project venv's editable install.  PYTHONPATH is
# consulted before site-packages, so <snapshot>/src lands at sys.path[1] while
# the _editable_impl_speedlm.pth entry pointing at the live tree lands after
# site-packages.  Asserted below, not trusted.
export PYTHONPATH="$pythonpath"
# The snapshot is read-only; do not attempt to litter it with .pyc files.
export PYTHONDONTWRITEBYTECODE=1

export $gate_var=1
export $artifact_var="\$run_dir/results"
mkdir -p "\$$artifact_var"
EOF

case "$flavor" in
    idle-tuning)
        : "${vllm_args:=$DEFAULT_GATEWAY_VLLM_ARGS}"
        cat <<EOF
export SPEEDLM_E2E_TUNING_CONFIG="$tuning_config"
EOF
        [[ -n "$tuning_profile" ]] && echo "export SPEEDLM_E2E_TUNING_PROFILE=\"$tuning_profile\""
        [[ -n "$corpus" ]] && echo "export SPEEDLM_E2E_PROMPT_CORPUS=$corpus"
        cat <<EOF
export SPEEDLM_E2E_READY_TIMEOUT=1800
export SPEEDLM_E2E_TUNING_TIMEOUT=$tuning_timeout
export SPEEDLM_E2E_REQUEST_TIMEOUT=1800
EOF
        ;;
    capture-overhead)
        # Only the model pair and the readiness timeout.  The Stage 0 matrix
        # knobs (prompt set, target layer ids, HF reference, strict verdict) and
        # the runner override belong to the correctness test and are NOT emitted
        # here: this test reads none of them, and exporting variables a test
        # ignores is how a run ends up believed to be configured in a way it
        # was not.
        cat <<EOF
export SPEEDLM_E2E_VERIFIER_MODEL=$verifier
export SPEEDLM_E2E_DRAFTER_MODEL=$drafter
EOF
        echo "export SPEEDLM_E2E_READY_TIMEOUT=1800"
        # Fault injection, in milliseconds, into the ON arm only.  Emitted only
        # when asked for; this is how the paired-difference machinery is proven
        # able to fail on real hardware.
        if [[ -n "$inject_ms" ]]; then
            echo "export SPEEDLM_E2E_CAPTURE_OVERHEAD_INJECT_MS=$inject_ms"
        fi
        ;;
    activation-capture|hot-swap)
        cat <<EOF
export SPEEDLM_E2E_VERIFIER_MODEL=$verifier
export SPEEDLM_E2E_DRAFTER_MODEL=$drafter
EOF
        [[ -n "$drafter_dir" ]] && echo "export SPEEDLM_E2E_DRAFTER_DIR=$drafter_dir"
        echo "export SPEEDLM_E2E_READY_TIMEOUT=1800"
        # Stage 0 matrix knobs.  Each is emitted ONLY when the operator set it,
        # so the unset case keeps the test's own defaults.
        [[ -n "$prompt_set" ]] && echo "export SPEEDLM_E2E_PROMPT_SET=$prompt_set"
        [[ -n "$target_layer_ids" ]] && echo "export SPEEDLM_E2E_TARGET_LAYER_IDS=$target_layer_ids"
        [[ -n "$hf_reference" ]] && echo "export SPEEDLM_E2E_HF_REFERENCE=$hf_reference"
        [[ -n "$strict_verdict" ]] && echo "export SPEEDLM_E2E_STRICT_VERDICT=$strict_verdict"
        # The GPU model runner generation is forceable from the environment.
        # VllmConfig.use_v2_model_runner (vllm/config/vllm.py:519-522) reads
        # envs.VLLM_USE_V2_MODEL_RUNNER FIRST and returns it verbatim when it is
        # not None, short-circuiting every auto-detection heuristic.  The var is
        # declared at vllm/envs.py:264 and parsed at vllm/envs.py:1867-1868 with
        # maybe_convert_bool(os.getenv(...)), so 0 and 1 both work and leaving it
        # unset means "auto".  EXPECTED_RUNNER is emitted unconditionally so the
        # test can assert at runtime that it got the runner that was requested.
        case "$runner" in
            v1) echo "export VLLM_USE_V2_MODEL_RUNNER=0" ;;
            v2) echo "export VLLM_USE_V2_MODEL_RUNNER=1" ;;
        esac
        echo "export SPEEDLM_E2E_EXPECTED_RUNNER=\"$runner\""
        ;;
    live-vllm)
        echo "export SPEEDLM_E2E_MODEL=$model"
        echo "export SPEEDLM_E2E_STAGE=${model//\//--}"
        echo "export SPEEDLM_E2E_READY_TIMEOUT=1800"
        : "${vllm_args:=$DEFAULT_GATEWAY_VLLM_ARGS}"
        ;;
    proxy-overhead)
        echo "export SPEEDLM_E2E_MODEL=$model"
        echo "export SPEEDLM_E2E_STAGE=proxy-overhead"
        : "${vllm_args:=$DEFAULT_GATEWAY_VLLM_ARGS}"
        ;;
    token-fidelity)
        echo "export SPEEDLM_FIDELITY_STAGE=token-fidelity"
        : "${vllm_args:=$DEFAULT_GATEWAY_VLLM_ARGS}"
        ;;
    model-matrix)
        echo "export SPEEDLM_MATRIX_CELL=$matrix_cell"
        echo "export SPEEDLM_MATRIX_STARTUP_TIMEOUT=1800"
        echo "export SPEEDLM_MATRIX_REQUEST_TIMEOUT=240"
        # The cell carries its own vLLM args; do not override them.
        ;;
    capture-matrix)
        echo "export SPEEDLM_CAPTURE_MODEL=$model"
        echo "export SPEEDLM_CAPTURE_STAGE=${model//\//--}"
        echo "export SPEEDLM_CAPTURE_READY_TIMEOUT=1800"
        echo "export SPEEDLM_CAPTURE_VLLM_VENV=$VLLM_ENV"
        # The OpenAI-SDK half of the matrix runs in its own interpreter; the
        # project venv has no `openai`, the vLLM venv does.
        echo "export SPEEDLM_CAPTURE_SDK_PYTHON=$VLLM_ENV/bin/python"
        vllm_args_var="SPEEDLM_CAPTURE_VLLM_ARGS"
        : "${vllm_args:=$DEFAULT_GATEWAY_VLLM_ARGS}"
        ;;
    agent-harness)
        echo "export SPEEDLM_AGENT_MODEL=$model"
        echo "export SPEEDLM_AGENT_STAGE=${model//\//--}"
        echo "export SPEEDLM_AGENT_CLI=scripted"
        echo "export SPEEDLM_AGENT_STARTUP_TIMEOUT=1800"
        echo "export SPEEDLM_AGENT_SUBPROCESS_TIMEOUT=420"
        # test_agent_harness.py:209 builds per-model defaults already.
        vllm_args_var="SPEEDLM_AGENT_VLLM_ARGS"
        ;;
    config-matrix)
        echo "export SPEEDLM_CONFIG_MATRIX_MODEL=$model"
        echo "export SPEEDLM_CONFIG_MATRIX_REFERENCE_DRAFT=$drafter"
        echo "export SPEEDLM_CONFIG_MATRIX_CANDIDATE_DRAFT=$candidate_drafter"
        echo "export SPEEDLM_E2E_PROMPT_CORPUS=$corpus"
        echo "export SPEEDLM_CONFIG_MATRIX_STARTUP_TIMEOUT=1800"
        echo "export SPEEDLM_CONFIG_MATRIX_REQUEST_TIMEOUT=600"
        echo "export SPEEDLM_CONFIG_MATRIX_CELL=$config_matrix_cell"
        if [[ -n "$inject_percent" ]]; then
            echo "export SPEEDLM_CONFIG_MATRIX_INJECT_SLOWDOWN_PERCENT=$inject_percent"
        fi
        ;;
esac

if [[ -n "$vllm_args" ]]; then
    echo "export $vllm_args_var='$vllm_args'"
fi
if [[ "$flavor" == "idle-tuning" ]]; then
    echo 'echo "vllm_args=$SPEEDLM_E2E_VLLM_ARGS"'
fi

# Without -k the whole test file runs, i.e. Stage 0 and the prefix-cache test
# together.  -k is what lets one runner/model cell be run on its own.
pytest_k_arg=""
[[ -n "$pytest_k" ]] && pytest_k_arg=" -k '$pytest_k'"

cat <<EOF

echo "started_at=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job=\$SLURM_JOB_ID host=\$(hostname -f)"
echo "commit=\$commit"
echo "snapshot=\$snapshot"
echo "source_tree_dirty_at_generation=\$source_tree_dirty_at_generation"
echo "slurm_wall_time=\$slurm_wall_time (\$slurm_wall_time_seconds seconds)"
if [[ -n "$tuning_timeout" ]]; then
    echo "pytest_tuning_timeout=$tuning_timeout seconds"
fi
echo "interpreter=\$interpreter"
echo "PYTHONPATH=\$PYTHONPATH"
# Never pipe nvidia-smi into head under \`set -o pipefail\`: head exits early,
# nvidia-smi takes SIGPIPE, and the non-zero status kills the whole job.
nvidia-smi || true

# ------------------------------------------------------------------------
# Provenance gate.  Fail here, cheaply and loudly, rather than burning GPU
# hours on code that silently came from the live working tree.
# ------------------------------------------------------------------------
"\$interpreter" - <<'PYEOF'
import os, sys, pathlib
snapshot = pathlib.Path(os.environ["SNAPSHOT_DIR"]).resolve()
import speedlm
actual = pathlib.Path(speedlm.__file__).resolve()
print(f"speedlm.__file__ = {actual}")
if not actual.is_relative_to(snapshot):
    print("FATAL: speedlm did not resolve to the snapshot.", file=sys.stderr)
    print(f"  expected under: {snapshot}", file=sys.stderr)
    print(f"  actually got:   {actual}", file=sys.stderr)
    print("  sys.path:", file=sys.stderr)
    for i, entry in enumerate(sys.path):
        print(f"    [{i}] {entry}", file=sys.stderr)
    raise SystemExit(1)
print("provenance OK: imports resolve to the snapshot")
PYEOF

"\$interpreter" -m pytest -o addopts='' -p no:cacheprovider$pytest_plugin_arg -q -ra -s \\
    "\$snapshot/$test_path"$pytest_k_arg

echo "finished_at=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
} > "$sbatch_path"

# SNAPSHOT_DIR is what the provenance gate reads; export it alongside the rest.
# Inserted after generation so the heredoc above stays readable.
python3 - "$sbatch_path" "$snapshot" <<'PYEOF'
import sys, pathlib
path, snapshot = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
anchor = 'export PYTHONDONTWRITEBYTECODE=1\n'
assert anchor in text, "generator drift: anchor line not found"
text = text.replace(anchor, anchor + f'export SNAPSHOT_DIR="{snapshot}"\n', 1)
path.write_text(text)
PYEOF

chmod +x "$sbatch_path"

cat <<EOF

flavor      $flavor
commit      $sha
snapshot    $snapshot   (read-only)
run dir     $run_dir
sbatch      $sbatch_path
interpreter $interpreter
PYTHONPATH  $pythonpath

Nothing was submitted.  To submit:

    sbatch $sbatch_path
EOF
