# demo/

Everything used to produce the SpeedLM demo video. This directory is a leaf:
nothing under `src/`, `tests/`, or `scripts/` imports or references anything here,
and `demo/demo-qwen8b.json` is the only config it needs.

The finished video is two segments concatenated, in this order:

1. **the live tuning cycle** — a real terminal watching a real SpeedLM gateway
   arm itself, train an Eagle3 head, benchmark it, and gate it (`cycle.sbatch` +
   `session_record.py` + `session_render.py`)
2. **stock vs tuned drafting, side by side** — the two heads replayed against
   each other on one shared clock (`capture.sbatch` / `capture.py` + `render.py`)


## What each file does

| file | what it is | GPU? |
| --- | --- | --- |
| `cycle.sbatch` | SLURM job: runs a real `speedlm vllm serve` gateway with `--enable-idle-tuning`, seeded with run5's captured traffic, and waits for one full idle-tuning cycle to complete | **yes** — 1 GPU, 2h |
| `session_record.py` | drives a real `bash` on a real PTY from a JSON step script, records asciicast v2 | no |
| `session_render.py` | rasterises a `.cast` to 1920x1080 30fps H.264 via `pyte` + PIL + ffmpeg | no |
| `session_cycle.json` | the step script for the cycle segment — watches `events.jsonl`, the trainer's `stdout.log`, and finally `decision.json` on the *live* run | n/a |
| `session_speedlm.json` | an alternative step script narrating an already-finished run (post-hoc; not used for the shipped cycle segment) | n/a |
| `capture.sbatch` | SLURM job: runs `capture.py` over the held-out suite, both arms sequentially in one allocation on one GPU | **yes** — 1 GPU, 2h |
| `capture.py` | starts a real vLLM engine per arm and records a token-by-token JSONL timeline for each request | **yes** (`--dry-run` is GPU-free) |
| `render.py` | replays the two timelines side by side and encodes the mp4 | no |
| `demo-qwen8b.json` | the demo-sized SpeedLM config used by `cycle.sbatch` | n/a |

Rendering is pure CPU: `imageio_ffmpeg` + PIL, and `session_render.py` additionally
needs `pyte` and DejaVu Sans Mono (`fonts-dejavu-core`). Only `cycle.sbatch`,
`capture.sbatch`, and `capture.py` touch a GPU.


## Artifacts

Under `/data/ryan.kim/speedlm-runs/`:

```
demo-cycle-run10/          # the live tuning cycle
  provenance.txt           # host, SLURM_JOB_ID 380018, commit, seed traces
  gateway-and-vllm.log     # full gateway + vLLM log
  gateway-tail.log         # last 400 lines
  results/                 # every JSON under SPEEDLM_HOME, incl. decision.json
  speedlm_home/            # the run's SPEEDLM_HOME
  speedlm-cycle.cast       # the raw PTY recording (1215.2s real)
  speedlm-cycle.mp4        # 149.97s, 1920x1080 30fps

demo-video-run2/           # the stock-vs-tuned capture
  provenance.txt           # host, SLURM_JOB_ID 378951, commit
  capture.log
  capture_manifest.json
  comparison.json          # the totals of this recording
  engine-stock.log / engine-candidate.log
  timeline-stock.jsonl / timeline-candidate.jsonl
  speedlm-drafting.mp4     # 1x
  speedlm-drafting-4x.mp4  # 46.33s, the shipped 4x cut

speedlm-demo.mp4           # the two segments concatenated, 196.30s
```


## Regenerating

### Segment 1 — the live tuning cycle

`cycle.sbatch` is the thing being filmed; the recorder attaches to it while it runs.

```bash
# 1. submit the real cycle (1 GPU, ~30 min of cycle inside a 2h wall limit)
sbatch demo/cycle.sbatch
# writes /data/ryan.kim/speedlm-runs/demo-cycle-run10/ and
#        /data/ryan.kim/speedlm-runs/slurm-demo-cycle-<jobid>.out

# 2. once the job is RUNNING, record the terminal that watches it.
#    session_cycle.json tails the live events.jsonl and the live trainer log,
#    so this must be started while the cycle is still in flight.
.venv/bin/python demo/session_record.py \
    --script demo/session_cycle.json \
    --out    /data/ryan.kim/speedlm-runs/demo-cycle-run10/speedlm-cycle.cast \
    --cwd    /admin/home/ryan.kim/speedlm-fr \
    --cols 100 --rows 30

# 3. render (CPU only, no GPU, no SLURM)
.venv/bin/python demo/session_render.py \
    /data/ryan.kim/speedlm-runs/demo-cycle-run10/speedlm-cycle.cast \
    /data/ryan.kim/speedlm-runs/demo-cycle-run10/speedlm-cycle.mp4 \
    --max-gap 3.0 \
    --timing-out /data/ryan.kim/speedlm-runs/demo-cycle-run10/speedlm-cycle.timing.json
```

`--max-gap 3.0` is what produced the shipped 149.97s cut from a 1215.2s recording
(16 gaps compressed). `--max-gap 0` disables compression and plays the recording
at its real pace.

`--timing-out` writes a JSON sidecar stating how the clock was bent: piecewise-linear
`[cast_t, video_t]` breakpoints plus the compressed gaps and their real durations.
Only the renderer knows that mapping, and without it nothing can put an overlay on the
frame where a line appeared. `demo/remotion/extract_series.py` reads it (together with
the `.cast`) to stamp every chart datum with that frame; the flag is additive, and
omitting it changes nothing about the video.

### Segment 2 — stock vs tuned, side by side

```bash
# 1. capture both arms (1 GPU, both arms sequentially in one allocation)
DEMO_OUT_DIR=/data/ryan.kim/speedlm-runs/demo-video-run2 sbatch demo/capture.sbatch

# 2. render at 4x (CPU only)
.venv/bin/python demo/render.py \
    --capture-dir /data/ryan.kim/speedlm-runs/demo-video-run2 \
    --out         /data/ryan.kim/speedlm-runs/demo-video-run2/speedlm-drafting-4x.mp4 \
    --speed 4
```

`capture.py --dry-run` resolves the suite, selection, and engine argv without
allocating a GPU, which is the cheap way to check the wiring before submitting.

### Assembling the final video

Both segments are already 1920x1080, 30 fps, H.264 High, yuv420p, tbn 15360, no
audio — so they concatenate losslessly with a stream copy, no re-encode:

```bash
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
printf "file '%s'\nfile '%s'\n" \
    /data/ryan.kim/speedlm-runs/demo-cycle-run10/speedlm-cycle.mp4 \
    /data/ryan.kim/speedlm-runs/demo-video-run2/speedlm-drafting-4x.mp4 > /tmp/concat.txt
"$FF" -y -f concat -safe 0 -i /tmp/concat.txt -c copy -movflags +faststart \
    /data/ryan.kim/speedlm-runs/speedlm-demo.mp4
```


## Honesty properties

These are the properties that make the video defensible, and where they live.

**The terminal is a real PTY running real commands.** `session_record.py`
calls `pty.fork()` to hand a genuine `bash` a real terminal, then only writes
keystrokes in and reads bytes out. Nothing fabricates output, replays a canned
transcript, or re-flows text a program did not emit — if a command fails, the
failure is what gets recorded. Narration is typed as real `# ...` shell input
(the shell echoes and discards it), so there is no post-hoc caption layer.

**Dead-air compression is marked on screen with the real elapsed time.**
`session_render.py --max-gap` shrinks any stretch with no output longer than
N seconds down to N seconds, and every compressed gap draws a marker pill in the
bottom-right reading `⏩ <real time> elapsed` (e.g. `⏩ 8m12s elapsed`), using the
*real* wall time that was removed. The renderer also prints a summary line to
stderr stating the gap count, the total time removed, and the marker text.
`--max-gap 0` turns compression off entirely.

**`--speed` draws a persistent speed badge.** `render.py --speed` scales the
shared replay clock only: no chunk is dropped, neither arm is re-timed, and every
number on screen — elapsed, tokens, tok/s, accepted length — stays in the
recording's own real seconds. Because a real 123.4s timer running out in 31s of
video would otherwise read as a bug, every replay frame carries a badge reading
`{speed}x SPEED` (`4x SPEED` in the shipped cut). At `--speed 1.0` no badge is
drawn.

**The demo does not invent its own headline number.** `cycle.sbatch` runs a real
cycle at demo size (see `demo-qwen8b.json`: 96-record corpus, 1 epoch, 3 benchmark
repeats — the validator's hard floor — 0 warmup, 128/64-token benchmark and
correctness caps). `capture.sbatch` runs *one* sequential pass over the watchable
contexts, which is far too little to promote on and exactly enough to watch; its
captions quote the gated numbers alongside its own and say which is which.


## The numbers, from the artifacts

From `demo-cycle-run10/.../decision.json` — the gate the video shows, 36 contexts
x 3 repeats, `max_tokens=128`, stock draft `RedHatAI/Qwen3-8B-speculator.eagle3`:

| | candidate | stock | delta | threshold |
| --- | --- | --- | --- | --- |
| accepted length | 2.4637 | 2.3143 | +0.1494 | >= 0.05 |
| acceptance rate | 0.4879 | 0.4381 | +4.98 pp | >= 1.0 pp |
| tok/s (replay) | 129.34 | 123.25 | +4.94% | >= -2.0% |
| tok/s (prometheus) | 141.97 | 134.72 | +5.38% | — |

`verdict: promote`, `reason: both_thresholds_met`,
`acceptance_criterion: mean_accepted_length_delta`,
output divergences 7/36 (p = 0.0057).

From `demo-video-run2/comparison.json` — this recording's own totals, 49 contexts,
16625 completion tokens per arm:

| | stock | candidate |
| --- | --- | --- |
| wall seconds | 124.045 | 110.505 |
| wall tok/s | 134.02 | 150.45 |
| mean accepted length | 2.3890 | 2.7387 |
| mean engine tok/s | 142.36 | 163.49 |

`wall_speedup_pct: 10.92`, `accepted_length_delta: 0.3497`,
`engine_tok_per_sec_delta_pct: 14.84`.

The same file carries `gated_result_reference`, which is the number to cite:
**+0.2989 accepted length, +9.94% tok/s**, from `docs/agentic-selfplay-result.md`
(job 378546, unseen session-disjoint suite, 5 repeats x 100 contexts x 2 blocks
per arm).
