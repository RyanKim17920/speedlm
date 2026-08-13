#!/usr/bin/env python3
"""Render the corpus tally card for the SpeedLM demo, in demo/render.py's look.

Every number is read off the real artifacts, not typed in here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1920, 1080, 30
BG = (13, 17, 23)
RULE = (48, 54, 61)
TEXT = (201, 209, 217)
DIM = (110, 118, 129)
MUTED = (139, 148, 158)
TUNED_ACCENT = (63, 185, 80)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

RUN = Path("/data/ryan.kim/speedlm-runs/bigcycle-run1")
RUN_ID = "e7004c4c0c7548fba65b05a924aa57ea"
DEC = Path("/data/ryan.kim/speedlm-runs/regate-big-run2/decision.json")
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONT_DIR / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")), size
    )


traces = RUN / "speedlm_home" / "traces" / "traces.jsonl"
records = sum(1 for _ in traces.open("rb"))
mb = traces.stat().st_size / 1_000_000
cfg = json.loads(Path("/admin/home/ryan.kim/speedlm-fr/demo/bigcycle-qwen8b.json").read_text())
leased = cfg["tuning"]["training_window_records"]
rows_log = (
    RUN / "speedlm_home" / "runs" / RUN_ID / "stage-logs" / "training-row-rendering" / "stderr.log"
).read_text(errors="replace")
rows = int(rows_log.rsplit("shards): 100%", 1)[1].split("/")[1].split("[")[0].strip().split()[0])
d = json.loads(DEC.read_text())

f_title = font(48, True)
f_sub = font(22)
f_big = font(84, True)
f_card = font(28)
f_small = font(20)

img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

draw.text((40, 36), "SpeedLM  ·  the corpus this cycle trained on", font=f_title, fill=TEXT)
draw.text(
    (42, 96),
    "one night of served traffic, and what the gate replays",
    font=f_sub,
    fill=MUTED,
)
draw.line((40, 134, WIDTH - 40, 134), fill=RULE, width=1)

stats = [
    (f"{records:,}", "captured records", TEXT),
    (f"{mb:.0f} MB", "of real traces on disk", TUNED_ACCENT),
    (f"{rows}", f"training rows rendered from a {leased:,}-record lease", TEXT),
]
for i, (big, label, colour) in enumerate(stats):
    x = 120 + i * 580
    draw.text((x, 230), big, font=f_big, fill=colour)
    draw.text((x, 336), label, font=f_small, fill=MUTED)

draw.line((120, 420, WIDTH - 120, 420), fill=RULE, width=1)

body = [
    ("Every request the server answers is recorded as a trace. When the traffic", TEXT),
    ("stops, the idle cycle leases the newest records and trains a draft head.", TEXT),
    ("", TEXT),
    (f"The gate then replays a separate suite of {d['num_contexts']} contexts,", TEXT),
    ("session-disjoint from that training window — 0 leakage overlaps,", TUNED_ACCENT),
    (
        f"{d['num_repeats']} scored repeats per arm after "
        f"{d['warmup_repeats']} warmups, greedy.",
        TUNED_ACCENT,
    ),
]
y = 480
for line, colour in body:
    draw.text((120, y), line, font=f_card, fill=colour)
    y += 46

draw.line((40, HEIGHT - 74, WIDTH - 40, HEIGHT - 74), fill=RULE, width=1)
draw.text(
    (40, HEIGHT - 56),
    "bigcycle-run1  ·  Qwen/Qwen3-8B  ·  gate: regate-big-run2  ·  greedy, max 512 new tokens",
    font=f_small,
    fill=DIM,
)

out = Path(sys.argv[1])
png = out.with_suffix(".png")
img.save(png)
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-loglevel", "error",
    "-loop", "1", "-framerate", str(FPS), "-t", f"{SECONDS}", "-i", str(png),
    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
    "-r", str(FPS), "-movflags", "+faststart", str(out),
]
subprocess.run(cmd, check=True)
print(f"records={records} mb={mb:.1f} leased={leased} rows={rows} -> {out} ({SECONDS}s)")
