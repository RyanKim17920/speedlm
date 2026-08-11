# Launching the agentic self-play idle-tuning run

This is the operator's page for `scripts/agentenv_run.sbatch`: what it does, in
what order, where its output lands, and how to watch it. Every note here exists
because a prior allocation died on it.

## Launch

    cd /admin/home/ryan.kim/speedlm-fr
    sbatch scripts/agentenv_run.sbatch

That is the whole launch. The script needs no arguments: the model
(`Qwen/Qwen3-8B`), the config (`configs/agentenv-qwen8b.json`), the run
directory and the cache location are all pinned inside it. In particular the HF
cache is set by the script itself with

    export HF_HOME=/data/ryan.kim/hf-cache

alongside `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. There is no
`--hf-home` or `--tuning-config` to pass: `speedlm vllm serve` takes a
positional model plus `--host`, `--port`, `--config`, `--profile` and
`--enable-idle-tuning` (`src/speedlm/cli.py`), and the global `--home`. Those two
flags belong to `scripts/make_snapshot_run.sh` and `scripts/speedbench`, neither
of which this workload invokes.

Before launching a *new* run, change `run_dir` and the `#SBATCH --output` path
in the script together -- both name `agentenv-qwen8b-run5` and a re-run would
otherwise overwrite the previous run's artifacts.

## What the job does, in order

1. **Provenance.** Writes hostname, `SLURM_JOB_ID`, `CUDA_VISIBLE_DEVICES`, the
   repo commit and the config path to `$run_dir/provenance.txt`, then
   `nvidia-smi`. `SLURM_JOB_ID` and `CUDA_VISIBLE_DEVICES` use `${VAR:?...}`, so
   running the script outside an allocation fails immediately rather than
   quietly measuring the wrong thing.
2. **Serve.** Launches `.venv/bin/speedlm vllm serve Qwen/Qwen3-8B` under
   `setsid` (so cleanup can signal the whole process group) with
   `--config configs/agentenv-qwen8b.json --enable-idle-tuning
   --max-model-len 24576 --gpu-memory-utilization 0.85 --enforce-eager
   --enable-auto-tool-choice --tool-call-parser hermes`. All output goes to
   `$run_dir/gateway-and-vllm.log`. An `EXIT` trap TERMs then KILLs the group.
3. **Readiness.** Polls `http://127.0.0.1:8150/health` every 2s for up to 900
   attempts (30m), aborting early with a log tail if the gateway process dies
   first. Port 8150 is the wrapper port from the config.
4. **Smoke gate.** Runs `scripts/run_agentic_traffic.py` with `--seeds 1
   --families call-chain-trace --max-turns 10` into `$run_dir/smoke`, logging to
   `$run_dir/smoke.log`. One graded task, about a minute, and the script is
   `set -e`, so a failure aborts before the full traffic run. Job 376274 burned
   an allocation discovering a tool-parser mismatch on trajectory one and then
   repeating it ninety times; this gate is that lesson.
5. **Traffic.** Runs the same script with `--seeds 30` across all six families
   (`bugfix-localize`, `feature-implement`, `log-triage`, `refactor-rename`,
   `schema-migrate`, `call-chain-trace`) = **180 trajectories**, `--max-turns 24`,
   `--max-output-tokens 1024`, `--stop-after-seconds 14400`, into
   `$run_dir/traffic` with `$run_dir/traffic.log`.
6. **Wait for a decision.** Polls for up to 7h, every 30s, for a
   `decision*.json` under `$SPEEDLM_HOME`, breaking early if the gateway dies.
7. **Collect.** Copies every JSON under `$SPEEDLM_HOME` smaller than 10 MB into
   `$run_dir/results/`, writes `$run_dir/gateway-tail.log`, and prints the
   traffic summary and the decision artifacts to the SLURM log.

## Where things land

    $run_dir  = /data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5
      slurm-<jobid>.out        job stdout (the #SBATCH --output path)
      provenance.txt           host, job id, GPU, commit, config
      gateway-and-vllm.log     everything the server and tuner printed
      gateway-tail.log         last 400 lines of the above
      smoke/ smoke.log         the one-task gate
      traffic/summary.json     totals: instances, turns, tool calls, prompt sizes
      traffic.log
      speedlm_home/runs/<id>/decision.json    the verdict
      results/                 JSON artifacts copied out of speedlm_home

`SPEEDLM_HOME` is `$run_dir/speedlm_home`, so the run is fully self-contained --
nothing is written to a shared speedlm home.

## Monitoring

    scripts/agentenv_monitor.sh <jobid> /data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5

Optional third argument is the ledger path; it defaults to
`/data/ryan.kim/speedlm-runs/agentenv-ledger.jsonl`. The monitor polls `squeue`
every 40s (1200 polls, about 13.3h against the job's 12h limit), reports
in-flight `idle tuning cycle failed` lines without exiting on them (the cycle
retries after a cooldown), and exits as soon as a decision artifact appears at
`$run_dir/speedlm_home/runs/*/decision.json`. When the job leaves the queue it
requires the same terminal state from `sacct` on two consecutive samples before
recording anything, so a lagging accounting record is not mistaken for a
verdict.

Every terminal outcome appends one JSON line to the ledger -- job id, run dir,
outcome, the decision fields, and the traffic totals. An N=1 promotion is not a
result; the ledger is what turns a sequence of runs into one. If there is no
decision, the monitor records *why* (no `speedlm_home`, no cycle run directory,
cycle directories with no verdict, or an unparseable `decision.json`) rather
than reporting a bare absence.

## Reference outcome

Two runs of this script reached a verdict, both **promote**, both
`reason: both_thresholds_met`:

| job | run dir | accepted_length_delta | throughput | acceptance |
|---|---|---|---|---|
| 377142 | `agentenv-qwen8b-run4` | **+0.6921** (2.151 -> 2.844) | **+24.9%** (111.3 -> 139.0 tok/s) | +23.07 pp |
| 377760 | `agentenv-qwen8b-run5` | **+0.6525** (2.185 -> 2.837) | **+32.3%** (113.9 -> 150.7 tok/s) | +21.75 pp |

Both measured 100 contexts x 5 repeats with bounded truncation on both arms.
Traffic in both: 180 instances attempted, 171-173 completed, ~1000 tool calls,
mean ~6.1 turns per trajectory. A run that lands far outside this envelope is a
run to distrust before believing.

## Sizing

`--max-model-len 24576` is set for headroom on agentic prompts. The two promoted
runs measured `max_final_prompt_tokens` of 13358 (377142) and 11863 (377760),
with medians near 2.6k, so 24576 is comfortable rather than tight -- do not
lower it toward the observed maximum, because a truncated agentic prompt fails
in the corpus-admission stage rather than at serve time.

`configs/agentenv-qwen8b.json` sets `tuning.min_corpus_records: 512`, so the
traffic generator must produce at least 512 captured records before a cycle will
fire. A trajectory contributes one record per assistant turn; 180 trajectories
at ~6 turns each yields roughly 1000, which clears the floor with about 2x
margin. `training_window_records` is also 512.

## What killed the earlier agentic runs

None of these reached a gate decision. All three failed in *training corpus
admission* -- serving, capture and traffic generation worked every time
(`serving_restored: True` in each failure record). The failure moved later each
run, so this was a progression rather than a repeat:

| run | wall clock | died on |
|---|---|---|
| `agentic-mixed-qwen` (375376) | 1m45s | `--workload` and `--corpus` both set: two rival seed sources |
| `agentic-mixed-qwen2` (375414) | 57m58s | truncated rows: 33 of 409 truncated, **3 trainable rows** against a floor of 32 |
| `agentic-mixed-qwen3` (375525) | 57m13s | **373 of 409 rows (91.2%) carried an assistant turn this verifier did not produce** (4080 such turns), leaving 3 |

The third is the real blocker, and it is what `trust_self_play_assistant_turns`
exists to solve.

## Why `trust_untagged_assistant_messages` does not solve it

That flag relabels only assistant turns whose `provenance_tag` **is None**
(`src/speedlm/training/rows.py:177`). Gateway-captured traffic tags every
prefix assistant turn `"client_supplied"` explicitly
(`src/speedlm/gateway/capture.py:473`), so tagged rows are dropped regardless.
It is also an unverified assertion -- the operator promises the corpus is the
verifier's own output and nothing checks it, which is why an earlier attempt to
unblock agentic training this way was reverted on safety grounds.

`tuning.trust_self_play_assistant_turns` (set in `configs/agentenv-qwen8b.json`)
replaces the promise with a measurement:
`speedlm.training.provenance.self_play_attestation`
(`src/speedlm/training/provenance.py:155`) requires every `client_supplied`
assistant turn to be byte-identical to a `generated` turn from an **earlier**
row, and the cycle fails loudly if it is not. Order matters: matching against
the whole batch would let a row attest itself. Trust is earned per run, recorded
in the artifacts, and falsifiable.

## Other things not to get wrong

* `--tool-call-parser hermes` deliberately overrides the profile.
  `QWEN_3_8B_EAGLE3_PROFILE` pins `qwen3_xml`, and job 376274 proved that wrong
  on hardware: Qwen3-8B emitted Hermes-format calls inside message content with
  an empty `tool_calls` array, and every trajectory died at turn one. The
  mismatch is silent by construction -- a parser that matches nothing is
  indistinguishable from a model that chose not to call a tool.
* Pass `--workload` **or** `--corpus`, never both -- that pairing killed
  `agentic-mixed-qwen`. This flavor generates its own traffic and needs neither.
* Do not pass `--skip-preflight`. The preflight gate is the thing that refuses
  configurations which would measure nothing, and it costs no GPU time.
