# Resuming the diverse-traffic demo video

State as of 2026-08-16. Commits: `d6fd8ec`, `b34a07c`, `98c7c9c`.

## What is already done

- **Corpus.** `/data/ryan.kim/speedlm-runs/bigcorpus-v2-shard{0..7}/` holds 3,777
  records over 707 trajectories, 12 families, **707 distinct opening prompts**.
  v1 (`bigcorpus-run1`) was 3,758 records / 601 trajectories / **6** prompts.
  The two are size-matched to 0.5%, so a difference in the trained head is
  attributable to diversity rather than volume.
- **Montage beat.** Already rendered from the v2 corpus. Re-render with:

      ARGS=""; for s in 0 1 2 3 4 5 6 7; do
        ARGS="$ARGS --corpus /data/ryan.kim/speedlm-runs/bigcorpus-v2-shard$s/traffic/trajectories"
      done
      .venv/bin/python demo/montage.py $ARGS --out <out>/traffic-montage-v3.mp4

  36 cards / 36.2s at the default 3 per family. `--per-family 2` gives 24 cards
  (~24s) if you want the original ~87s total runtime back.

## In flight

| job | what | note |
|---|---|---|
| 383985 | `demo/cyclev2.sbatch` — train + gate on the v2 corpus | 3 epochs, K=3, gate mirrors regate-big-run2 |
| 383987 | `demo/capture.sbatch` — race timelines vs the new head | chained `--dependency=afterok:383985` |

If 383985 died, resubmit `sbatch demo/cyclev2.sbatch`; it re-merges the shards
itself and is idempotent. Then re-chain 383987.

## Finishing the video

Once 383985 has written a `decision.json` and 383987 has written timelines:

    .venv/bin/python demo/build_video.py \
      $ARGS \
      --capture-dir /data/ryan.kim/speedlm-runs/demo-video-cyclev2-run1 \
      --decision <cyclev2-run1 decision.json> \
      --corroborating regate-big-run2=/data/ryan.kim/speedlm-runs/regate-big-run2/decision.json \
      --training-run /data/ryan.kim/speedlm-runs/cyclev2-run1 \
      --out /data/ryan.kim/speedlm-runs/demo-versions/speedlm-v2.mp4 \
      --work-dir <scratch>

Stages: `montage` -> `terminal` -> `remotion` -> `race` -> `assemble`. Use
`--skip montage` to reuse the clip above. Every on-screen number derives from
the `--decision` file; `tests/test_no_hardcoded_run_paths.py` fails if a run
directory gets baked back into anything that feeds the screen.

## Reading the result

`docs/speedup-ceiling.md` fits `delta = -0.187 + 0.0822*ln(rows)` over the
size-scaling data. v1 trained 652 rendered rows from a 423-record remainder and
gated **+0.3457**. v2's remainder is 541 (28% more), so **row count alone
predicts roughly +0.366**. Diversity has paid off only if v2 lands materially
above that, not merely above +0.3457. Take the real rendered-row count from
`cyclev2-run1/.../stage-logs/training-row-rendering/stderr.log` ("Loaded N
samples") rather than trusting the expansion assumption.

Throughput will almost certainly be vetoed as non-stationary again; four gates,
four vetoes on this shared cluster. Quote accepted length.

## Two known defects, deliberately not fixed here

1. **`feature-implement` is broken.** `--max-output-tokens 1024` truncates
   Qwen3's `<think>` block before it emits a tool call, so every trajectory ends
   at ~1.9 turns with zero tool calls and 0% solve. Inherited from v1, but the
   longer generated prompts tip it over. Needs 3072+ and a re-capture of that
   family. `perf-hotspot` looks similar in solve rate but is fine — 7.5 turns of
   real tool use against an honestly hard grader.

2. **79.9% of records are untrainable.** Self-play attestation cannot attribute
   `<think>` assistant turns, so eagle3 drops them. v1 was worse (82.8%), so
   this is pre-existing and the merge did not cause it (`merge_traces.py:175`
   sorts by `(timestamp, id)`, preserving causal order). Setting the backend's
   `trust_untagged_assistant_messages` would recover ~3,236 rows and is
   *probably* legitimate — this traffic really is self-play, driven by
   `scripts/run_agentic_traffic.py` against the same served model. But that flag
   sits upstream of the training path with no post-promotion rollback, so it
   wants an explicit attestation, not an assumption. Left off here so the v1/v2
   comparison stays like-for-like.

## Preemption

6 of 8 capture shards were preempted off n-5 and n-4 about 40 minutes in. The
corpus was frozen at 3,777 rather than chasing 17,000. Shards append (`mkdir -p`,
no `rm`), so re-running the same shard indices would add near-duplicate content;
`merge_traces.py --drop-duplicate-content` exists if you ever do that.
