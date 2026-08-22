# demo/

Everything used to produce the SpeedLM demo video.

## Current state

The finished video and its intermediate clips survive at
`/data/ryan.kim/speedlm-demo/` (see table below). The original run artifact tree
under `/data/ryan.kim/speedlm-runs/` was **deleted on 2026-08-21** in an approved
cleanup, so the video **cannot be regenerated** without re-running the GPU stages
listed below.

| file | size | description |
|---|---|---|
| `speedlm-v2-diverse.mp4` | 106.53s | the finished demo (montage + walkthrough + side-by-side) |
| `traffic-montage-v3.mp4` | 36.17s | montage beat, built from the 707-prompt corpus |
| `versionA-split.mp4` | 45.01s | terminal walkthrough + Remotion charts |
| `race-8x.mp4` | 30.90s | stock-vs-tuned race (trimmed to 25.40s in the final cut) |

All four verified by full decode, zero errors.

## Pipeline overview

The three source clips are assembled in one `filter_complex` re-encode pass
(which also converts Remotion's `yuvj420p` output to `yuv420p`):

```
[0] traffic-montage-v3.mp4  whole clip, no trim     36.17 s   demo/montage.py             (CPU)
[1] versionA-split.mp4      whole clip, no trim     45.01 s   session_record.py (CPU)
                                                     session_render.py (CPU)
                                                     extract_series.py (CPU)
                                                     npx remotion render (CPU)
[2] race-8x.mp4             trim 1.5 s - 26.9 s    25.40 s   demo/render.py              (CPU)
                            ─────────────────────   ───────
                                                    106.58 s  speedlm-v2-diverse.mp4
```

`libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart`, no
audio.  `versionA-split.mp4` is `yuvj420p` (Remotion output) and carries a
silent AAC track; the `filter_complex` pass normalises both.

## Rebuilding from scratch

A full rebuild requires GPU stages that produce the artifacts now gone. The
original `build_video.py` command referenced paths under
`/data/ryan.kim/speedlm-runs/` that no longer exist. To rebuild:

**GPU stages (require 1 GPU each):**
1. **Traffic capture** -- `demo/traffic-v2.sbatch` x8 shards. Produces
   `traffic/trajectories/` directories consumed by `montage.py`.
2. **Training + gate** -- `demo/cyclev2.sbatch`. Trains on the merged corpus
   and runs the regression gate. Produces training logs, traces, and
   `decision.json`.
3. **Race capture** -- `demo/capture.sbatch`. Runs `capture.py` over the
   held-out suite, both arms sequentially. Produces
   `timeline-stock.jsonl` and `timeline-candidate.jsonl`.

**CPU stages (no GPU needed):**
4. **Montage** -- `demo/montage.py` over the captured trajectories.
5. **Terminal** -- `demo/session_record.py` + `demo/session_render.py`.
6. **Remotion** -- `demo/remotion/extract_series.py` + `npx remotion render`.
7. **Race** -- `demo/render.py` over the capture timelines.
8. **Assemble** -- `demo/build_video.py` (orchestrates all five stages and
   runs the final `ffmpeg filter_complex` pass).

## What each file does

| file | what it is | GPU? |
| --- | --- | --- |
| `build_video.py` | **orchestrator** — drives all five stages in order, validates inputs up-front, verifies Root.tsx frame count, checks final duration | no |
| `montage.py` | flip through real captured agent trajectories, one card per instance; renders the traffic intro clip | no |
| `corpus_card.py` | standalone corpus summary card (totals, per-family, suite info) | no |
| `merge_traces.py` | merge multiple `traces.jsonl` files from different traffic runs | no |
| `render.py` | replay two captured drafting timelines side-by-side at `--speed 8`; derives every on-screen number from `--decision` at render time | no |
| `session_record.py` | drives a real `bash` on a real PTY from a JSON step script, records asciicast v2 | no |
| `session_render.py` | rasterises a `.cast` to 1920x1080 30fps H.264 via `pyte` + PIL + ffmpeg | no |
| `session_fast.json` | step script for the fast terminal segment (typing_delay 0.0025 s/char, 44.83 s rendered) | n/a |
| `session_cycle.json` | step script that tails a *live* run's `events.jsonl` and trainer `stdout.log` | n/a |
| `session_cycle1e.json` | variant for the 1-epoch config | n/a |
| `session_live.json` | alternative step script for narrating a running gateway | n/a |
| `session_live_verdict.json` | short verdict-only step script | n/a |
| `session_speedlm.json` | alternative step script for a finished run (post-hoc) | n/a |
| `remotion/extract_series.py` | reads training logs + gate decision.json, writes `remotion/src/data.json`; anchors every chart datum to the frame where the terminal printed it | no |
| `remotion/src/Root.tsx` | Remotion composition; `DURATION_IN_FRAMES` (line 14) **must match** the newly recorded terminal clip's frame count — `build_video.py` enforces this and aborts with the correct value if it does not | n/a |
| `bigcycle.sbatch` | SLURM job: runs a real `speedlm vllm serve ... --enable-idle-tuning` gateway and waits for one full idle-tuning cycle | **yes** -- 1 GPU |
| `bigcycle-qwen8b.json` | SpeedLM config used by `bigcycle.sbatch` | n/a |
| `cyclev2.sbatch` | SLURM job: train + gate on the v2 corpus (3 epochs, K=3) | **yes** -- 1 GPU |
| `cyclev2-qwen8b.json` | SpeedLM config used by `cyclev2.sbatch` | n/a |
| `cycle.sbatch` | earlier demo-sized cycle job (96-record corpus, 1 epoch) | **yes** -- 1 GPU |
| `cycle1e.sbatch` | 1-epoch variant cycle job | **yes** -- 1 GPU |
| `cycle1e-qwen8b.json` | SpeedLM config for `cycle1e.sbatch` | n/a |
| `traffic.sbatch` | SLURM job: runs agentic traffic through the gateway via `scripts/run_agentic_traffic.py` (which imports `tests/e2e/agentenv/`) | **yes** -- 1 GPU |
| `traffic-v2.sbatch` | revised traffic job (updated routing / concurrency) | **yes** -- 1 GPU |
| `traffic-qwen8b.json` | SpeedLM config for traffic jobs | n/a |
| `capture.sbatch` | SLURM job: runs `capture.py` over the held-out suite, both arms sequentially in one GPU allocation | **yes** -- 1 GPU |
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
    --corpus <shard0>/traffic/trajectories <shard1>/traffic/trajectories ... \
    --traces <training_run>/speedlm_home/traces/traces.jsonl \
    --out <output>/traffic-montage-v3.mp4
```

36 cards / 36.2s at the default 3 per family. `--per-family 2` gives 24 cards
(~24s) for a shorter cut.

### terminal

Two sub-steps: record a PTY session, then render it.

```bash
.venv/bin/python demo/session_record.py \
    --script demo/session_fast.json \
    --out    <scratch>/session_fast.cast \
    --cwd    /admin/home/ryan.kim/speedlm-fr \
    --cols 100 --rows 30

.venv/bin/python demo/session_render.py \
    <scratch>/session_fast.cast \
    <scratch>/session_fast.mp4 \
    --timing-out <scratch>/timing.json
```

`--timing-out` writes the piecewise-linear cast-to-video clock mapping that
`extract_series.py` needs to anchor chart reveals to the exact frame where the
terminal printed each value.

### remotion

Three sub-steps.

**1. extract_series.py** reads the training log and gate decision, writes
`demo/remotion/src/data.json`.  Must be re-run whenever the cast or timing
sidecar changes, or the chart reveals will be anchored to the wrong frames.

```bash
.venv/bin/python demo/remotion/extract_series.py \
    --run-root <training_run_dir> \
    --decision <gate_dir>/decision.json \
    --cast   <scratch>/session_fast.cast \
    --timing <scratch>/timing.json
```

**2.** Copy `terminal.mp4` to `demo/remotion/public/terminal.mp4`.

**3. npx remotion render** (`build_video.py` runs this automatically):

```bash
cd demo/remotion
npx remotion render SpeedLMDemo <output>/versionA-split.mp4 \
    --codec=h264 --crf=20
```

**Root.tsx frame count.**  `demo/remotion/src/Root.tsx` hardcodes
`DURATION_IN_FRAMES` (currently 1344).  **This must match the newly recorded
terminal clip's frame count** or the Remotion render will be truncated or
padded.  `build_video.py` checks this automatically and aborts with the
correct value.  Update it by hand if running Remotion directly:

```
# terminal duration in seconds x 30 fps, rounded to nearest integer
# Example: 44.83 s x 30 = 1344.9 -> 1345 frames
```

### race

```bash
.venv/bin/python demo/render.py \
    --capture-dir <race_capture_dir> \
    --decision    <gate_dir>/decision.json \
    --corroborating <corroborating_gate>/decision.json \
    --out  <output>/race-8x.mp4 \
    --speed 8
```

`--decision` is required; every displayed gate figure is derived from it at
render time -- nothing is hardcoded.  `--corroborating` (repeatable) supplies
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

## Honesty properties

**The terminal is a real PTY running real commands.**  `session_record.py`
calls `pty.fork()` to hand a genuine `bash` a real terminal, then only writes
keystrokes in and reads bytes out.  Nothing fabricates output or replays a
canned transcript -- if a command fails, the failure is what gets recorded.
Narration is typed as real `# ...` shell input.

**Dead-air compression is marked on screen.**  `session_render.py --max-gap`
(default 2.0 s) shrinks any stretch with no output down to that limit and
draws a fast-forward pill for every compressed gap.

**`--speed` draws a persistent badge.**  `render.py --speed 8` scales only
the shared replay clock; no chunk is dropped and every displayed number (elapsed,
tokens, tok/s, accepted length) stays in the recording's own real seconds.  At
`--speed 1.0` no badge is drawn.

**Gate figures are derived at render time.**  `render.py` requires `--decision`
and reads every displayed measurement -- accepted-length delta, SE, percentage
headline, throughput range, acceptance-rate pp -- from the gate's own
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

`traffic.sbatch` is not a leaf -- it has a dependency on `agentenv/` (the
per-task sandbox drivers) and `scripts/run_agentic_traffic.py` (the harness).
