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
# environment with `os.environ.copy()` (test_serving_draft_hot_swap.py:198,
# test_serving_activation_capture.py:162, test_live_idle_tuning.py:565), so the
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

usage() {
    cat <<'EOF'
make_snapshot_run.sh -- pin a GPU run to an immutable snapshot of the repo.

Required:
  --flavor F            idle-tuning | activation-capture | hot-swap

Optional:
  --commit REF          commit/ref to snapshot            (default: HEAD)
  --run-name NAME       run directory name under
                        /data/ryan.kim/speedlm-runs       (default: <flavor>-snap-<UTC>)
  --time HH:MM:SS       SBATCH --time                     (default: per flavor)
  --partition P         SBATCH --partition                (default: n)
  --cpus N              SBATCH --cpus-per-task            (default: 16)
  --tuning-config PATH  SPEEDLM_E2E_TUNING_CONFIG         (idle-tuning only, required)
  --tuning-profile PATH SPEEDLM_E2E_TUNING_PROFILE        (idle-tuning, optional)
  --verifier ID         SPEEDLM_E2E_VERIFIER_MODEL        (capture/hot-swap)
  --drafter ID          SPEEDLM_E2E_DRAFTER_MODEL         (capture/hot-swap)
  --drafter-dir PATH    SPEEDLM_E2E_DRAFTER_DIR           (hot-swap, optional)
  --corpus PATH         SPEEDLM_E2E_PROMPT_CORPUS         (idle-tuning)
  --no-corpus           omit the corpus; test falls back to synthetic prompts
  --vllm-args JSON      SPEEDLM_E2E_VLLM_ARGS, a JSON array of strings
  --force               re-extract the snapshot even if it already exists
  -h | --help           this message

The snapshot lands in /data/ryan.kim/speedlm-snapshots/<full-sha>/ and is made
read-only.  Re-running for the same commit reuses it (content is identical by
construction), so snapshots are cheap: the tree is ~2.5 MB.

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
partition="n"
cpus=16
tuning_config=""
tuning_profile=""
verifier=""
drafter=""
drafter_dir=""
corpus="$CORPUS"
vllm_args=""
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flavor)         flavor="$2"; shift 2 ;;
        --commit)         commit_ref="$2"; shift 2 ;;
        --run-name)       run_name="$2"; shift 2 ;;
        --time)           sbatch_time="$2"; shift 2 ;;
        --partition)      partition="$2"; shift 2 ;;
        --cpus)           cpus="$2"; shift 2 ;;
        --tuning-config)  tuning_config="$2"; shift 2 ;;
        --tuning-profile) tuning_profile="$2"; shift 2 ;;
        --verifier)       verifier="$2"; shift 2 ;;
        --drafter)        drafter="$2"; shift 2 ;;
        --drafter-dir)    drafter_dir="$2"; shift 2 ;;
        --corpus)         corpus="$2"; shift 2 ;;
        --no-corpus)      corpus=""; shift ;;
        --vllm-args)      vllm_args="$2"; shift 2 ;;
        --force)          force=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$flavor" ]] || { echo "error: --flavor is required" >&2; usage >&2; exit 2; }

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
# --------------------------------------------------------------------------
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
            echo "       (test_live_idle_tuning.py:166 hard-fails without SPEEDLM_E2E_TUNING_CONFIG)" >&2
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
    *)
        echo "error: unknown flavor '$flavor' (want idle-tuning|activation-capture|hot-swap)" >&2
        exit 2
        ;;
esac

sbatch_time="${sbatch_time:-$default_time}"

# --------------------------------------------------------------------------
# Resolve the commit to a full sha.  A ref is ambiguous over time; the snapshot
# directory is named for the sha so the mapping run -> code is one-to-one.
# --------------------------------------------------------------------------
cd "$REPO"
sha="$(git rev-parse --verify "${commit_ref}^{commit}")"
short_sha="${sha:0:12}"
snapshot="$SNAPSHOT_ROOT/$sha"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${run_name:-${flavor}-snap-${stamp}}"
run_dir="$RUN_ROOT/$run_name"

if [[ -e "$run_dir" ]]; then
    echo "error: run directory already exists: $run_dir" >&2
    exit 1
fi

# Warn loudly if the tree is dirty.  git archive serializes the COMMIT, so any
# uncommitted change is silently absent from the snapshot -- exactly the kind of
# surprise this script exists to prevent.
if ! git diff --quiet HEAD -- 2>/dev/null || [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
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

# --------------------------------------------------------------------------
# Emit the sbatch.
#
# #SBATCH --output is read by slurm before any shell runs, so the path must be
# spelled out literally -- it cannot be built from a shell variable.  That is
# why the heredoc below is unquoted and interpolates at generation time.
# --------------------------------------------------------------------------
sbatch_path="$run_dir/job.sbatch"
pythonpath="$snapshot/src${extra_pythonpath}"

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

cd "\$snapshot"

export PATH="$VLLM_ENV/bin:\$PATH"
export HF_HOME=/data/ryan.kim/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# The snapshot must beat the project venv's editable install.  PYTHONPATH is
# consulted before site-packages, so <snapshot>/src lands at sys.path[1] while
# the _editable_impl_speedlm.pth entry pointing at the live tree lands after
# site-packages.  Asserted below, not trusted.
export PYTHONPATH="$pythonpath"
# The snapshot is read-only; do not attempt to litter it with .pyc files.
export PYTHONDONTWRITEBYTECODE=1

export $gate_var=1
export SPEEDLM_E2E_ARTIFACT_DIR="\$run_dir/results"
mkdir -p "\$SPEEDLM_E2E_ARTIFACT_DIR"
EOF

case "$flavor" in
    idle-tuning)
        cat <<EOF
export SPEEDLM_E2E_TUNING_CONFIG="$tuning_config"
EOF
        [[ -n "$tuning_profile" ]] && echo "export SPEEDLM_E2E_TUNING_PROFILE=\"$tuning_profile\""
        [[ -n "$corpus" ]] && echo "export SPEEDLM_E2E_PROMPT_CORPUS=$corpus"
        cat <<'EOF'
export SPEEDLM_E2E_READY_TIMEOUT=1800
export SPEEDLM_E2E_TUNING_TIMEOUT=36000
export SPEEDLM_E2E_REQUEST_TIMEOUT=1800
EOF
        ;;
    activation-capture|hot-swap)
        cat <<EOF
export SPEEDLM_E2E_VERIFIER_MODEL=$verifier
export SPEEDLM_E2E_DRAFTER_MODEL=$drafter
EOF
        [[ -n "$drafter_dir" ]] && echo "export SPEEDLM_E2E_DRAFTER_DIR=$drafter_dir"
        echo "export SPEEDLM_E2E_READY_TIMEOUT=1800"
        ;;
esac

if [[ -n "$vllm_args" ]]; then
    echo "export SPEEDLM_E2E_VLLM_ARGS='$vllm_args'"
fi

cat <<EOF

echo "started_at=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job=\$SLURM_JOB_ID host=\$(hostname -f)"
echo "commit=\$commit"
echo "snapshot=\$snapshot"
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

"\$interpreter" -m pytest -o addopts='' -p no:cacheprovider -q -s \\
    "\$snapshot/$test_path"

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
