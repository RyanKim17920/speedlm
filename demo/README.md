# demo/

Everything used to produce the SpeedLM demo video. The finished video
(`speedlm-final.mp4`, 87s, three source clips concatenated) can be rebuilt
end-to-end from a new set of run artifacts with a single command:

```bash
.venv/bin/python demo/build_video.py \
    --corpus /data/ryan.kim/speedlm-runs/bigcorpus-run1/traffic/trajectories \
    --capture-dir /data/ryan.kim/speedlm-runs/demo-video-run2 \
    --decision /data/ryan.kim/speedlm-runs/regate-big-run2/decision.json \
    --corroborating /data/ryan.kim/speedlm-runs/regate-big-run1/decision.json \
    --training-run /data/ryan.kim/speedlm-runs/bigcycle-run1 \
    --out /data/ryan.kim/speedlm-runs/demo-versions/speedlm-rebuilt.mp4 \
    --work-dir /data/ryan.kim/speedlm-runs/demo-build
```

Use `--skip <stage>` (repeatable) to reuse an existing intermediate when only
some stages have changed.  Stages: `montage`, `terminal`, `remotion`, `race`,
`assemble`.


## Pipeline overview

The three source clips are assembled in one `filter_complex` re-encode pass
(which also converts Remotion's `yuvj420p` output to `yuv420p`):

```
[0] traffic-montage.mp4   whole clip, no trim     16.97 s   demo/montage.py
[1] versionA-split.mp4    whole clip, no trim     44.83 s   session_record.py
                                                             session_render.py
                                                             extract_series.py
                                                             npx remotion render
[2] race-8x.mp4           trim 1.5 s – 26.9 s    25.40 s   demo/render.py
                          ─────────────────────   ───────
                                                  87.20 s   speedlm-final.mp4
```

`libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart`, no
audio.  `versionA-split.mp4` is `yuvj420p` (Remotion output) and carries a
silent AAC track; the `filter_complex` pass normalises both.


## What each file does

| file | what it is | GPU? |
| --- | --- | --- |
| `build_video.py` | **one-command rebuild** — drives all five stages in order, validates inputs up-front, verifies Root.tsx frame count, checks final duration | no |
| `montage.py` | flip through real captured agent trajectories, one card per instance; renders the traffic intro clip | no |
| `corpus_card.py` | standalone corpus summary card (totals, per-family, suite info) | no |
| `merge_traces.py` | merge multiple `traces.jsonl` files from different traffic runs | no |
| `render.py` | replay two captured drafting timelines side-by-side at `--speed 8`; derives every on-screen number from `--decision` at render time | no |
| `session_record.py` | drives a real `bash` on a real PTY from a JSON step script, records asciicast v2 | no |
| `session_render.py` | rasterises a `.cast` to 1920×1080 30fps H.264 via `pyte` + PIL + ffmpeg | no |
| `session_fast.json` | step script for the fast terminal segment (typing_delay 0.0025 s/char, 44.83 s rendered) | n/a |
| `session_cycle.json` | step script that tails a *live* run's `events.jsonl` and trainer `stdout.log` | n/a |
| `session_cycle1e.json` | variant for the 1-epoch config | n/a |
| `session_live.json` | alternative step script for narrating a running gateway | n/a |
| `session_live_verdict.json` | short verdict-only step script | n/a |
| `session_speedlm.json` | alternative step script for a finished run (post-hoc) | n/a |
| `remotion/extract_series.py` | reads training logs + gate decision.json, writes `remotion/src/data.json`; anchors every chart datum to the frame where the terminal printed it | no |
| `remotion/src/Root.tsx` | Remotion composition; `DURATION_IN_FRAMES` (line 14) **must match** the newly recorded terminal clip's frame count — `build_video.py` enforces this and aborts with the correct value if it does not | n/a |
| `bigcycle.sbatch` | SLURM job: runs a real `speedlm vllm serve ... --enable-idle-tuning` gateway and waits for one full idle-tuning cycle | **yes** — 1 GPU |
| `bigcycle-qwen8b.json` | SpeedLM config used by `bigcycle.sbatch` | n/a |
| `cycle.sbatch` | earlier demo-sized cycle job (96-record corpus, 1 epoch) | **yes** — 1 GPU |
| `cycle1e.sbatch` | 1-epoch variant cycle job | **yes** — 1 GPU |
| `cycle1e-qwen8b.json` | SpeedLM config for `cycle1e.sbatch` | n/a |
| `traffic.sbatch` | SLURM job: runs agentic traffic through the gateway via `scripts/run_agentic_traffic.py` (which imports `tests/e2e/agentenv/`) | **yes** — 1 GPU |
| `traffic-v2.sbatch` | revised traffic job (updated routing / concurrency) | **yes** — 1 GPU |
| `traffic-qwen8b.json` | SpeedLM config for traffic jobs | n/a |
| `capture.sbatch` | SLURM job: runs `capture.py` over the held-out suite, both arms sequentially in one GPU allocation | **yes** — 1 GPU |
| `capture.py` | starts a real vLLM engine per arm and records a token-by-token JSONL timeline; `--dry-run` is GPU-free | **yes** |
| `demo-qwen8b.json` | demo-sized SpeedLM config (96-record corpus, 1 epoch) | n/a |


## Stage details

### montage

`demo/montage.py` reads one or more `traffic/trajectories` directories and
renders one card per genuinely distinct agent instance.  Asserts that each
selected card carries a distinct opening prompt (the property that
`phrasing.py` provides); the corpus must have been captured with the variant
instructions for this assertion to pass.

```bash
.venv/bin/python demo/montage.py \
    --corpus /data/ryan.kim/speedlm-runs/bigcorpus-run1/traffic/trajectories \
    --traces /data/ryan.kim/speedlm-runs/bigcycle-run1/speedlm_home/traces/traces.jsonl \
    --out /data/ryan.kim/speedlm-runs/demo-versions/traffic-montage-v3.mp4
```

### terminal

Two sub-steps: record a PTY session, then render it.

```bash
.venv/bin/python demo/session_record.py \
    --script demo/session_fast.json \
    --out    /data/ryan.kim/speedlm-runs/demo-fast/session_fast_v5.cast \
    --cwd    /admin/home/ryan.kim/speedlm-fr \
    --cols 100 --rows 30

.venv/bin/python demo/session_render.py \
    /data/ryan.kim/speedlm-runs/demo-fast/session_fast_v5.cast \
    /data/ryan.kim/speedlm-runs/demo-fast/session_fast_v5.mp4 \
    --timing-out /data/ryan.kim/speedlm-runs/demo-fast/timing_v5.json
```

`--timing-out` writes the piecewise-linear cast→video clock mapping that
`extract_series.py` needs to anchor chart reveals to the exact frame where the
terminal printed each value.

### remotion

Three sub-steps.

**1. extract_series.py** reads the training log and gate decision, writes
`demo/remotion/src/data.json`.  Must be re-run whenever the cast or timing
sidecar changes, or the chart reveals will be anchored to the wrong frames.

```bash
.venv/bin/python demo/remotion/extract_series.py \
    --run-root /data/ryan.kim/speedlm-runs/bigcycle-run1 \
    --decision /data/ryan.kim/speedlm-runs/regate-big-run2/decision.json \
    --cast   /data/ryan.kim/speedlm-runs/demo-fast/session_fast_v5.cast \
    --timing /data/ryan.kim/speedlm-runs/demo-fast/timing_v5.json
```

**2.** Copy `terminal.mp4` to `demo/remotion/public/terminal.mp4`.

**3. npx remotion render** (`build_video.py` runs this automatically):

```bash
cd demo/remotion
npx remotion render SpeedLMDemo /data/ryan.kim/speedlm-runs/demo-versions/versionA-split-v3.mp4 \
    --codec=h264 --crf=20
```

**Root.tsx frame count.**  `demo/remotion/src/Root.tsx` hardcodes
`DURATION_IN_FRAMES` (currently 1344).  **This must match the newly recorded
terminal clip's frame count** or the Remotion render will be truncated or
padded.  `build_video.py` checks this automatically and aborts with the
correct value.  Update it by hand if running Remotion directly:

```
# terminal duration in seconds × 30 fps, rounded to nearest integer
# Example: 44.83 s × 30 = 1344.9 → 1345 frames
```

### race

```bash
.venv/bin/python demo/render.py \
    --capture-dir /data/ryan.kim/speedlm-runs/demo-video-run2 \
    --decision    /data/ryan.kim/speedlm-runs/regate-big-run2/decision.json \
    --corroborating /data/ryan.kim/speedlm-runs/regate-big-run1/decision.json \
    --out  /data/ryan.kim/speedlm-runs/demo-versions/race-8x.mp4 \
    --speed 8
```

`--decision` is required; every displayed gate figure is derived from it at
render time — nothing is hardcoded.  `--corroborating` (repeatable) supplies
the "reproduced N times" line.

### assemble

`build_video.py` runs a single `ffmpeg filter_complex` pass:

- clip [0] (montage): whole clip, `format=yuv420p`
- clip [1] (remotion): whole clip, `format=yuv420p` (converts from `yuvj420p`)
- clip [2] (race): `trim=start=1.5:end=26.9`, `setpts=PTS-STARTPTS`,
  `format=yuv420p`

then `concat=n=3`, `libx264 -preset medium -crf 20 -pix_fmt yuv420p
-movflags +faststart`, audio dropped.

After encoding, `build_video.py` ffprobes the output and reports the duration
vs the arithmetic sum; a discrepancy > 0.5 s is flagged.


## Artifacts

Under `/data/ryan.kim/speedlm-runs/`:

```
demo-versions/
  speedlm-final.mp4        # 87.17 s, the shipped final cut
  traffic-montage-v2.mp4   # montage clip (16.97 s)
  versionA-split-v2.mp4    # remotion terminal+chart clip (44.83 s)
  race-8x.mp4              # race clip (30.9 s, trimmed to 25.4 s in assembly)

demo-fast/
  session_fast_v4.cast     # PTY recording (asciicast v2)
  session_fast_v4.mp4      # rendered terminal (44.83 s)
  timing_v4.json           # cast→video clock mapping

bigcorpus-run1/
  traffic/trajectories/    # 601 agent trajectories (bigcorpus corpus)
  speedlm_home/
    traces/traces.jsonl
    runs/e7004c4c.../
      training-logs/stdout.log
      decision.json

demo-video-run2/           # stock-vs-tuned capture
  capture_manifest.json
  timeline-stock.jsonl
  timeline-candidate.jsonl

regate-big-run2/
  decision.json            # definitive gate decision (287-context suite, 8 repeats)

regate-big-run1/
  decision.json            # corroborating gate (same head, earlier run)
```


## Honesty properties

**The terminal is a real PTY running real commands.**  `session_record.py`
calls `pty.fork()` to hand a genuine `bash` a real terminal, then only writes
keystrokes in and reads bytes out.  Nothing fabricates output or replays a
canned transcript — if a command fails, the failure is what gets recorded.
Narration is typed as real `# ...` shell input.

**Dead-air compression is marked on screen.**  `session_render.py --max-gap`
(default 2.0 s) shrinks any stretch with no output down to that limit and
draws a `⏩ <real time> elapsed` pill for every compressed gap.

**`--speed` draws a persistent badge.**  `render.py --speed 8` scales only
the shared replay clock; no chunk is dropped and every displayed number (elapsed,
tokens, tok/s, accepted length) stays in the recording's own real seconds.  At
`--speed 1.0` no badge is drawn.

**Gate figures are derived at render time.**  `render.py` requires `--decision`
and reads every displayed measurement — accepted-length delta, SE, percentage
headline, throughput range, acceptance-rate pp — from the gate's own
`decision.json`.  Nothing is hardcoded.

**Chart anchors are real terminal timestamps.**  `extract_series.py` matches
each training/validation datum against the exact cast event where the terminal
printed it, then maps that to a video frame via the timing sidecar.  Points
the terminal never printed inherit the next printed point's frame.


## traffic.sbatch / traffic-v2.sbatch

These SLURM jobs run `scripts/run_agentic_traffic.py`, which imports
`tests/e2e/agentenv/` to drive real agentic tasks through the gateway.
`traffic-v2.sbatch` is the current version (updated routing and concurrency).
They produce `traffic/trajectories/` directories consumed by `montage.py`.

`traffic.sbatch` is not a leaf — it has a dependency on `agentenv/` (the
per-task sandbox drivers) and `scripts/run_agentic_traffic.py` (the harness).
