# Resuming the diverse-traffic demo video

State as of 2026-08-22.

## Artifact tree deleted — 2026-08-21

The entire run artifact tree under `/data/ryan.kim/speedlm-runs/` (142 GB) was
deleted in an approved cleanup on 2026-08-21. See
`/data/ryan.kim/.cleanup_manifest_20260821.txt`. Everything listed below is gone:

- `bigcorpus-v2-shard{0..7}/` — the v2 corpus (3,777 records, 707 trajectories)
- `bigcorpus-run1/`, `agentenv-qwen8b-run5/` — v1 corpora
- `bigcycle-run1/`, `cyclev2-run1/` — training runs
- `regate-big-run1/`, `regate-big-run2/`, `regate-unseen-run1/` — gates + decision.json
- `demo-video-run2/` — race capture timelines
- `demo-versions/` — rendered video intermediates

**The video cannot be regenerated from existing artifacts.** Every input
(corpus traces, training logs, gate decisions, capture timelines) is gone.

**What survives** — the four rendered videos, recovered from scratch and now at
`/data/ryan.kim/speedlm-demo/`:

| file | duration | description |
|---|---|---|
| `speedlm-v2-diverse.mp4` | 106.53s | the finished demo (montage + walkthrough + side-by-side) |
| `traffic-montage-v3.mp4` | 36.17s | montage beat, built from the 707-prompt corpus |
| `versionA-split.mp4` | 45.01s | terminal walkthrough + Remotion charts |
| `race-8x.mp4` | 30.90s | stock-vs-tuned race (trimmed to 25.40s in the final cut) |

All four verified by full decode, zero errors.

## What was achieved — durable record

- **Corpus.** The v2 corpus at `bigcorpus-v2-shard{0..7}/` held 3,777 records
  over 707 trajectories, 12 families, **707 distinct opening prompts**.
  v1 (`bigcorpus-run1`) was 3,758 records / 601 trajectories / **6** prompts.
  The two are size-matched to 0.5%, so a difference in the trained head is
  attributable to diversity rather than volume.
- **Training + gate.** `cyclev2.sbatch` trained 3 epochs, K=3, with a gate
  mirroring `regate-big-run2`. SLURM jobs 383985 (train+gate) and 383987 (race
  capture, chained on `--dependency=afterok:383985`) were the last submissions.
- **Video assembly.** Five stages: `montage` -> `terminal` -> `remotion` ->
  `race` -> `assemble`. Every on-screen number derived from the `--decision`
  file; `tests/test_no_hardcoded_run_paths.py` fails if a run directory gets
  baked into anything that feeds the screen.

## Full rebuild procedure (from scratch)

Rebuilding requires re-running every GPU stage. This is not a quick re-render —
it is a multi-day effort on a cluster that preempted this work four times.
Each step below produces artifacts that were part of the deleted tree.

**1. Traffic capture (GPU, 8 shards)**

   Run `demo/traffic-v2.sbatch` 8 times (one per shard). Each shard produces a
   `traffic/trajectories/` directory. This was the original corpus capture that
   filled `bigcorpus-v2-shard{0..7}/`.

**2. Merge traces (CPU)**

   `demo/merge_traces.py` to merge all shard `traces.jsonl` files. Shards use
   `mkdir -p` with no `rm`, so re-running the same shard indices would add
   near-duplicate content; `merge_traces.py --drop-duplicate-content` exists
   for that case.

**3. Training + gate (GPU)**

   Run `demo/cyclev2.sbatch` — trains on the merged corpus and runs the
   regression gate. Produces `cyclev2-run1/` with training logs, traces, and
   a `decision.json`. Idempotent: re-merges the shards itself.

**4. Race capture (GPU)**

   Run `demo/capture.sbatch` — runs `capture.py` over the held-out suite, both
   arms (stock vs. tuned) sequentially in one GPU allocation. Produces
   `demo-video-cyclev2-run1/` with `timeline-stock.jsonl` and
   `timeline-candidate.jsonl`.

**5. Build video (CPU)**

   ```bash
   .venv/bin/python demo/build_video.py \
       --corpus <shard0..7>/traffic/trajectories \
       --capture-dir <race_capture_dir> \
       --decision <cyclev2 decision.json> \
       --corroborating <corroborating_gate>/decision.json \
       --training-run <cyclev2_run_dir> \
       --out <output.mp4> \
       --work-dir <scratch>
   ```

   Use `--skip montage` to reuse an existing montage clip. Stages:
   `montage` -> `terminal` -> `remotion` -> `race` -> `assemble`.

## Reading the result

`docs/speedup-ceiling.md` fits `delta = -0.187 + 0.0822*ln(rows)` over the
size-scaling data. v1 trained 652 rendered rows from a 423-record remainder and
gated **+0.3457**. v2's remainder is 541 (28% more), so **row count alone
predicts roughly +0.366**. Diversity has paid off only if v2 lands materially
above that, not merely above +0.3457.

The "Reading the result" section originally pointed to
`cyclev2-run1/.../stage-logs/training-row-rendering/stderr.log` for the real
rendered-row count — that path is now gone. On a future rerun, check the same
log to get the actual `"Loaded N samples"` count rather than trusting the
expansion assumption.

Throughput will almost certainly be vetoed as non-stationary again; four gates,
four vetoes on this shared cluster. Quote accepted length.

## Two known defects, deliberately not fixed here

1. **`feature-implement` is broken.** `--max-output-tokens 1024` truncates
   Qwen3's thinking block before it emits a tool call, so every trajectory ends
   at ~1.9 turns with zero tool calls and 0% solve. Inherited from v1, but the
   longer generated prompts tip it over. Needs 3072+ and a re-capture of that
   family. `perf-hotspot` looks similar in solve rate but is fine — 7.5 turns of
   real tool use against an honestly hard grader.

2. **79.9% of records are untrainable.** Self-play attestation cannot attribute
   thinking assistant turns, so eagle3 drops them. v1 was worse (82.8%), so
   this is pre-existing and the merge did not cause it (`merge_traces.py:175`
   sorts by `(timestamp, id)`, preserving causal order). Setting the backend's
   `trust_untagged_assistant_messages` would recover ~3,236 rows and is
   *probably* legitimate — this traffic really is self-play, driven by
   `scripts/run_agentic_traffic.py` against the same served model. But that flag
   sits upstream of the training path with no post-promotion rollback, so it
   wants an explicit attestation, not an assumption. Left off here so the v1/v2
   comparison stays like-for-like.

## Preemption history

6 of 8 capture shards were preempted off n-5 and n-4 about 40 minutes in. The
corpus was frozen at 3,777 rather than chasing 17,000. The preemption history
is recorded here for context on why rebuilds are risky on this cluster.
