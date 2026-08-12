#!/usr/bin/env python3
"""Replay two captured drafting timelines side by side and encode them to video.

Input is what ``scripts/demo_capture.py`` wrote: one JSONL timeline per arm, each
request carrying every streamed chunk and the offset at which the client saw it.
This script puts the two arms on a single shared clock and plays them, so the
tuned arm visibly pulls ahead over the run rather than the viewer being asked to
compare two numbers in a table.

What the clock means matters, so it is worth being precise. Each arm's timeline
is its requests laid back to back, using the measured wall time of each request
and dropping the gaps between them. Those gaps are the harness scraping
``/metrics``, which is instrumentation rather than serving; both arms pay it
identically, and leaving it in would inflate both totals with time the engine
spent idle. Everything inside a request -- time to first token, and the arrival
of every chunk after it -- is played back exactly as measured.

The video is deliberately captioned with the GATED numbers, not this recording's
numbers. One sequential pass over a few dozen contexts is an illustration; the
promotion decision came from 5 repeats x 100 contexts x 2 blocks per arm.

Usage:

    python scripts/demo_render.py \
        --capture-dir /data/.../demo-video-run1 \
        --out         /data/.../demo-video-run1/speedlm-drafting.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 1920, 1080
FPS = 30

BG = (13, 17, 23)
PANEL = (22, 27, 34)
RULE = (48, 54, 61)
TEXT = (201, 209, 217)
DIM = (110, 118, 129)
MUTED = (139, 148, 158)
STOCK_ACCENT = (210, 153, 34)
TUNED_ACCENT = (63, 185, 80)
WARN = (248, 81, 73)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
INTRO_SECONDS = 5.0
OUTRO_SECONDS = 9.0
TAIL_SECONDS = 1.5  # hold on the last frame of the replay before the summary


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


# ---------------------------------------------------------------------------
# Timeline model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    t: float          # absolute seconds on the arm's own clock
    text: str
    reasoning: bool


@dataclass
class Request:
    order: int
    family: str
    context_hash: str
    turn_depth: int
    start: float      # absolute seconds on the arm's own clock
    end: float
    tokens: int
    accepted_length: float
    finish_reason: str
    chunks: list[Chunk]
    text: str


@dataclass
class Arm:
    name: str
    label: str
    draft: str
    accent: tuple[int, int, int]
    requests: list[Request]

    def __post_init__(self) -> None:
        # Precomputed so state_at can bisect instead of rebuilding this list on
        # every one of the several thousand frames.
        self.starts = [r.start for r in self.requests]
        self.tokens_before = []
        running = 0
        for request in self.requests:
            self.tokens_before.append(running)
            running += request.tokens
        # Progress measured in characters emitted, paired with the time each
        # prefix was complete.  Characters rather than chunks: the two arms emit
        # identical text but slice it into SSE deltas differently (a faster arm
        # batches more tokens per chunk), so chunk index is not comparable across
        # arms and character count is exactly comparable.
        self.chunk_times: list[float] = []
        self.chars_at: list[int] = []
        chars = 0
        for request in self.requests:
            for chunk in request.chunks:
                chars += len(chunk.text)
                self.chunk_times.append(chunk.t)
                self.chars_at.append(chars)
        self.total_chars = chars

    @property
    def duration(self) -> float:
        return self.requests[-1].end if self.requests else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.requests)

    @property
    def mean_accepted_length(self) -> float:
        measured = [r.accepted_length for r in self.requests if r.accepted_length > 0]
        return sum(measured) / len(measured) if measured else 0.0


def load_arm(
    path: Path, name: str, label: str, draft: str, accent: tuple[int, int, int]
) -> Arm:
    """Lay one arm's requests back to back on a single clock."""
    requests: list[Request] = []
    cursor = 0.0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            start = cursor
            chunks = [
                Chunk(t=start + float(c["t"]), text=c["text"], reasoning=bool(c["reasoning"]))
                for c in row["tokens"]
            ]
            end = start + float(row["wall_s"])
            requests.append(
                Request(
                    order=int(row["order"]),
                    family=str(row["family"]),
                    context_hash=str(row["context_hash"]),
                    turn_depth=int(row["turn_depth"]),
                    start=start,
                    end=end,
                    tokens=int(row["completion_tokens"]),
                    accepted_length=float(row["accepted_length"]),
                    finish_reason=str(row["finish_reason"]),
                    chunks=chunks,
                    text=str(row["text"]),
                )
            )
            cursor = end
    if not requests:
        raise SystemExit(f"{path} contains no requests")
    return Arm(name=name, label=label, draft=draft, accent=accent, requests=requests)


@dataclass
class ArmFrameState:
    """Everything one pane needs to draw itself at one instant."""

    index: int              # 0-based request currently on screen
    done: bool
    elapsed: float
    finished_at: float | None
    tokens_done: int        # tokens completed across all requests so far
    chars_done: int         # characters of output emitted so far, across all requests
    reasoning: str
    content: str
    request: Request


def state_at(arm: Arm, t: float) -> ArmFrameState:
    """Resolve a pane's state at absolute time ``t``.

    Uses bisect over request start times rather than a scan, so rendering stays
    linear in frames rather than quadratic in frames x requests.
    """
    idx = max(0, bisect_right(arm.starts, t) - 1)
    request = arm.requests[idx]
    done = t >= arm.duration

    if done:
        request = arm.requests[-1]
        idx = len(arm.requests) - 1
        tokens_done = arm.total_tokens
    else:
        tokens_done = arm.tokens_before[idx]
        # Tokens inside the in-flight request are pro-rated by how much of the
        # request's text has arrived.  Counting SSE chunks instead would bias the
        # readout between arms: both emit identical text but the faster arm packs
        # more tokens into each chunk, so it would appear to have emitted fewer.
        seen = sum(len(c.text) for c in request.chunks if c.t <= t)
        total = sum(len(c.text) for c in request.chunks)
        tokens_done += int(request.tokens * seen / total) if total else 0

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in request.chunks:
        if chunk.t > t and not done:
            break
        (reasoning_parts if chunk.reasoning else content_parts).append(chunk.text)

    return ArmFrameState(
        index=idx,
        done=done,
        elapsed=min(t, arm.duration),
        finished_at=arm.duration if done else None,
        tokens_done=tokens_done,
        chars_done=(
            arm.total_chars
            if done
            else (arm.chars_at[k - 1] if (k := bisect_right(arm.chunk_times, t)) else 0)
        ),
        reasoning="".join(reasoning_parts),
        content="".join(content_parts),
        request=request,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def wrap_tail(reasoning: str, content: str, columns: int, rows: int) -> list[tuple[str, bool]]:
    """Wrap the visible text and keep only the last ``rows`` lines.

    Returns (line, is_reasoning) so the renderer can dim the chain-of-thought.
    ~89% of what this corpus drafts is reasoning monologue, and showing it in the
    same colour as the answer would misrepresent what the speedup is speeding up.
    """
    lines: list[tuple[str, bool]] = []
    for blob, is_reasoning in ((reasoning, True), (content, False)):
        for raw in blob.split("\n"):
            if not raw:
                lines.append(("", is_reasoning))
                continue
            for piece in textwrap.wrap(
                raw, width=columns, replace_whitespace=False, drop_whitespace=False
            ) or [""]:
                lines.append((piece, is_reasoning))
    return lines[-rows:]


class Renderer:
    def __init__(self, stock: Arm, tuned: Arm, manifest: dict[str, Any]) -> None:
        self.stock = stock
        self.tuned = tuned
        self.manifest = manifest
        # Greedy decoding should make the two arms emit the same text: a draft
        # head changes how many tokens are proposed per step, not which tokens
        # survive verification.  Counting the agreement rather than asserting it
        # is what lets the summary card claim it, and would expose the arms
        # having quietly drifted apart into an unfair comparison.
        self.identical = sum(
            1
            for a, b in zip(stock.requests, tuned.requests, strict=True)
            if a.text == b.text
        )
        self.f_title = font(38, bold=True)
        self.f_sub = font(19)
        self.f_pane = font(23, bold=True)
        self.f_stat = font(21, bold=True)
        self.f_stat_label = font(15)
        self.f_body = font(15)
        self.f_foot = font(17)
        self.f_big = font(64, bold=True)
        self.f_card = font(26)
        self.f_card_bold = font(26, bold=True)

        # Pane geometry, computed once: two equal columns with a gutter.
        self.pane_top = 148
        self.pane_bottom = HEIGHT - 96
        self.pane_w = (WIDTH - 3 * 40) // 2
        self.pane_x = (40, 40 + self.pane_w + 40)

        # The context label sits at pane_top + 126 and is 18px tall, so the body
        # has to clear pane_top + 144 or a long first line strikes through it.
        self.body_top = self.pane_top + 154
        self.line_h = 19
        self.body_rows = (self.pane_bottom - 24 - self.body_top) // self.line_h
        # DejaVu Sans Mono advance is 0.602 em.
        self.columns = int((self.pane_w - 44) / (15 * 0.602))

    # -- chrome ------------------------------------------------------------

    def frame(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        return image, ImageDraw.Draw(image)

    def header(self, draw: ImageDraw.ImageDraw, subtitle: str) -> None:
        draw.text((40, 38), "SpeedLM  ·  idle-tuned Eagle3 draft head vs stock",
                  font=self.f_title, fill=TEXT)
        draw.text((42, 92), subtitle, font=self.f_sub, fill=MUTED)
        draw.line((40, 130, WIDTH - 40, 130), fill=RULE, width=1)

    def footer(self, draw: ImageDraw.ImageDraw, text: str) -> None:
        draw.line((40, HEIGHT - 74, WIDTH - 40, HEIGHT - 74), fill=RULE, width=1)
        draw.text((42, HEIGHT - 58), text, font=self.f_foot, fill=DIM)

    # -- panes -------------------------------------------------------------

    def pane(
        self,
        draw: ImageDraw.ImageDraw,
        arm: Arm,
        state: ArmFrameState,
        x: int,
        total_contexts: int,
    ) -> None:
        w, top, bottom = self.pane_w, self.pane_top, self.pane_bottom
        draw.rectangle((x, top, x + w, bottom), fill=PANEL)
        draw.rectangle((x, top, x + w, top + 4), fill=arm.accent)

        draw.text((x + 22, top + 22), arm.label, font=self.f_pane, fill=arm.accent)
        draw.text((x + 22, top + 52), arm.draft, font=self.f_stat_label, fill=DIM)

        # Live statistics row.  Rate is tokens over elapsed on this arm's clock,
        # which is the rate a user waiting on the stream actually experiences.
        rate = state.tokens_done / state.elapsed if state.elapsed > 0 else 0.0
        stats = [
            ("elapsed", f"{state.elapsed:6.1f}s"),
            ("tokens", f"{state.tokens_done:5d}"),
            ("tok/s", f"{rate:5.1f}"),
            ("accepted len", f"{state.request.accepted_length:4.2f}"),
        ]
        col = x + 22
        for label, value in stats:
            draw.text((col, top + 78), label, font=self.f_stat_label, fill=DIM)
            draw.text((col, top + 96), value, font=self.f_stat, fill=TEXT)
            col += 152

        label = (
            f"context {min(state.index + 1, total_contexts)}/{total_contexts}"
            f"   {state.request.family}   depth {state.request.turn_depth}"
        )
        draw.text((x + 22, top + 126), label, font=self.f_stat_label, fill=MUTED)

        # Progress bar over the whole workload.
        bar_y = bottom - 16
        draw.rectangle((x + 22, bar_y, x + w - 22, bar_y + 6), fill=RULE)
        # Characters rather than the request index: the index only moves once per
        # request, so it reads 0% for the whole of the first (longest) context and
        # understates progress by up to a full request everywhere else, whereas
        # characters advance with every token.  Both arms emit identical text, so
        # the same denominator makes the two bars directly comparable.
        # An arm that emitted nothing has no denominator, so it stays empty until
        # it finishes rather than dividing by zero.
        if state.done:
            fraction = 1.0
        elif arm.total_chars:
            fraction = min(1.0, max(0.0, state.chars_done / arm.total_chars))
        else:
            fraction = 0.0
        if fraction > 0:
            draw.rectangle(
                (x + 22, bar_y, x + 22 + int((w - 44) * fraction), bar_y + 6), fill=arm.accent
            )

        body_bottom = bar_y - 14
        rows = max(1, (body_bottom - self.body_top) // self.line_h)
        y = self.body_top
        for line, is_reasoning in wrap_tail(state.reasoning, state.content, self.columns, rows):
            draw.text((x + 22, y), line, font=self.f_body,
                      fill=DIM if is_reasoning else TEXT)
            y += self.line_h

        if state.done and state.finished_at is not None:
            badge = f"  FINISHED  {state.finished_at:.1f}s  "
            box = draw.textbbox((0, 0), badge, font=self.f_stat)
            bx, by = x + w - (box[2] - box[0]) - 26, top + 22
            draw.rectangle(
                (bx - 6, by - 6, bx + (box[2] - box[0]) + 6, by + (box[3] - box[1]) + 10),
                fill=arm.accent,
            )
            draw.text((bx, by), badge, font=self.f_stat, fill=BG)

    def lead(
        self,
        draw: ImageDraw.ImageDraw,
        stock_state: ArmFrameState,
        tuned_state: ArmFrameState,
        total_contexts: int,
    ) -> None:
        """Show how far ahead the tuned arm is, in contexts and in seconds.

        The seconds figure is the honest one to quote: it is how long the stock
        arm still needs to reach the context the tuned arm is on right now, read
        straight off the stock arm's own recorded timeline.
        """
        if tuned_state.done and not stock_state.done:
            seconds = self.stock.duration - stock_state.elapsed
            text = f"IDLE-TUNED FINISHED  ·  STOCK STILL HAS {seconds:.1f}s TO GO"
        else:
            # How long the stock arm still needs to reach the words the tuned arm
            # has already emitted.  Only meaningful while the two arms agree on
            # the text, which self.identical checks.
            if self.identical != len(self.stock.requests):
                return  # the arms diverged; "the same output" would be a lie
            emitted = tuned_state.chars_done
            if emitted == 0 or emitted >= self.stock.total_chars:
                return
            index = bisect_left(self.stock.chars_at, emitted)
            seconds = self.stock.chunk_times[index] - stock_state.elapsed
            if seconds < 0.3:
                return
            text = f"STOCK IS {seconds:.1f}s BEHIND THE SAME OUTPUT"
        # Right-aligned rather than centred: the subtitle is left-aligned and
        # long enough to run into the middle of the header.
        # box[2] rather than box[2] - box[0]: the left bearing is part of where
        # the glyphs land, so subtracting it would push the text past the margin.
        box = draw.textbbox((0, 0), text, font=self.f_sub)
        draw.text((WIDTH - 40 - box[2], 92), text, font=self.f_sub, fill=TUNED_ACCENT)

    # -- cards -------------------------------------------------------------

    def intro(self) -> Image.Image:
        image, draw = self.frame()
        self.header(draw, "same GPU · same prompts · same greedy sampling · arms run sequentially")
        lines = [
            ("", TEXT),
            ("A draft head proposes tokens; the target model verifies them.", TEXT),
            ("Accept more of each draft and the same answer arrives sooner.", TEXT),
            ("", TEXT),
            ("LEFT   stock speculator, straight off the shelf", STOCK_ACCENT),
            ("RIGHT  the same speculator after idle tuning on captured agent traffic",
             TUNED_ACCENT),
            ("", TEXT),
            (f"{len(self.stock.requests)} held-out agent contexts, replayed one at a time "
             "through both.", MUTED),
            ("Every context comes from an agent session the tuned head never trained on.",
             MUTED),
            ("", TEXT),
            ("Gated result: +0.2989 accepted length, +9.94% tok/s.", TUNED_ACCENT),
        ]
        y = 300
        for text, colour in lines:
            draw.text((120, y), text, font=self.f_card, fill=colour)
            y += 46
        self.footer(draw, "docs/agentic-selfplay-result.md · job 378546 · session-disjoint suite")
        return image

    def outro(self) -> Image.Image:
        image, draw = self.frame()
        self.header(draw, "totals over the replayed workload")

        stock_s, tuned_s = self.stock.duration, self.tuned.duration
        saved = stock_s - tuned_s
        pct = (saved / stock_s * 100.0) if stock_s > 0 else 0.0

        draw.text((120, 210), f"time to finish the same {len(self.stock.requests)} contexts",
                  font=self.f_card, fill=MUTED)
        draw.text((120, 260), f"{stock_s:.1f}s", font=self.f_big, fill=STOCK_ACCENT)
        draw.text((360, 288), "stock", font=self.f_card, fill=MUTED)
        draw.text((640, 260), f"{tuned_s:.1f}s", font=self.f_big, fill=TUNED_ACCENT)
        draw.text((880, 288), "idle-tuned", font=self.f_card, fill=MUTED)
        draw.text((1220, 268), f"{saved:.1f}s faster  ({pct:.1f}%)",
                  font=self.f_card_bold, fill=TUNED_ACCENT)

        rows = [
            ("output tokens",
             f"{self.stock.total_tokens}", f"{self.tuned.total_tokens}"),
            ("tokens / second (wall clock)",
             f"{self.stock.total_tokens / stock_s:.1f}" if stock_s else "-",
             f"{self.tuned.total_tokens / tuned_s:.1f}" if tuned_s else "-"),
            ("mean accepted length (engine)",
             f"{self.stock.mean_accepted_length:.3f}", f"{self.tuned.mean_accepted_length:.3f}"),
            ("identical output text",
             f"{self.identical}/{len(self.stock.requests)}",
             f"{self.identical}/{len(self.tuned.requests)}"),
        ]
        y = 420
        draw.line((120, y - 14, WIDTH - 120, y - 14), fill=RULE, width=1)
        for label, left, right in rows:
            draw.text((120, y), label, font=self.f_card, fill=TEXT)
            draw.text((1080, y), left, font=self.f_card_bold, fill=STOCK_ACCENT)
            draw.text((1400, y), right, font=self.f_card_bold, fill=TUNED_ACCENT)
            y += 56

        y += 40
        draw.line((120, y - 20, WIDTH - 120, y - 20), fill=RULE, width=1)
        for text, colour in (
            ("This replay is one sequential pass, shown because it is watchable.", MUTED),
            ("The number to cite is the gated one: 5 repeats x 100 contexts x 2 blocks per arm,",
             MUTED),
            ("+0.2989 accepted length (SE 0.0046) and +9.94% tok/s, verdict promote.",
             TUNED_ACCENT),
        ):
            draw.text((120, y), text, font=self.f_card, fill=colour)
            y += 44

        self.footer(
            draw,
            "docs/agentic-selfplay-result.md · job 378546 · unseen-session suite, 61 sessions",
        )
        return image

    # -- the replay --------------------------------------------------------

    def replay_frames(self) -> Iterator[Image.Image]:
        total = max(self.stock.duration, self.tuned.duration) + TAIL_SECONDS
        n_contexts = len(self.stock.requests)
        subtitle = (
            f"{n_contexts} held-out agent contexts · greedy · "
            f"{self.manifest.get('model', '')} · one GPU, sequential"
        )
        frames = int(total * FPS)
        for i in range(frames):
            t = i / FPS
            image, draw = self.frame()
            self.header(draw, subtitle)
            stock_state = state_at(self.stock, t)
            tuned_state = state_at(self.tuned, t)
            self.pane(draw, self.stock, stock_state, self.pane_x[0], n_contexts)
            self.pane(draw, self.tuned, tuned_state, self.pane_x[1], n_contexts)
            self.lead(draw, stock_state, tuned_state, n_contexts)
            self.footer(
                draw,
                "gated result: +0.2989 accepted length / +9.94% tok/s on the session-disjoint suite"
                "   ·   inter-request instrumentation gaps removed from both arms",
            )
            yield image


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode(frames: Iterator[Image.Image], out: Path, crf: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    argv = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        # yuv420p is what makes the file play in browsers and QuickTime rather
        # than only in ffplay.
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out),
    ]
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    assert process.stdin is not None
    written = 0
    try:
        for image in frames:
            process.stdin.write(image.tobytes())
            written += 1
            if written % (FPS * 10) == 0:
                print(f"  {written} frames ({written / FPS:.0f}s)", flush=True)
    finally:
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise SystemExit(f"ffmpeg failed:\n{stderr[-4000:]}")
    print(f"  {written} frames total ({written / FPS:.1f}s)", flush=True)


def repeat(image: Image.Image, seconds: float) -> Iterator[Image.Image]:
    for _ in range(int(seconds * FPS)):
        yield image


def chain(*parts: Iterator[Image.Image]) -> Iterator[Image.Image]:
    for part in parts:
        yield from part


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--frames-only", type=int, default=0,
                        help="render only the first N replay frames as PNGs, for a quick look")
    args = parser.parse_args(argv)

    capture: Path = args.capture_dir
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    stock = load_arm(capture / "timeline-stock.jsonl", "stock", "STOCK",
                     manifest["stock_draft"], STOCK_ACCENT)
    tuned = load_arm(capture / "timeline-candidate.jsonl", "candidate", "IDLE-TUNED",
                     Path(manifest["candidate_draft"]).parent.name + "/draft-model", TUNED_ACCENT)

    if len(stock.requests) != len(tuned.requests):
        raise SystemExit(
            f"arms replayed different workloads: {len(stock.requests)} vs {len(tuned.requests)}"
        )
    mismatched = [
        (a.context_hash, b.context_hash)
        for a, b in zip(stock.requests, tuned.requests, strict=True)
        if a.context_hash != b.context_hash
    ]
    if mismatched:
        raise SystemExit(f"arms replayed different contexts, first: {mismatched[0]}")

    renderer = Renderer(stock, tuned, manifest)
    print(f"stock  {stock.duration:.1f}s  {stock.total_tokens} tok  "
          f"accepted {stock.mean_accepted_length:.3f}")
    print(f"tuned  {tuned.duration:.1f}s  {tuned.total_tokens} tok  "
          f"accepted {tuned.mean_accepted_length:.3f}")

    if args.frames_only:
        out_dir = args.out.parent / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        renderer.intro().save(out_dir / "intro.png")
        renderer.outro().save(out_dir / "outro.png")
        for i, image in enumerate(renderer.replay_frames()):
            if i >= args.frames_only:
                break
            if i % 30 == 0:
                image.save(out_dir / f"replay-{i:05d}.png")
        print(f"wrote sample frames to {out_dir}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encode(
        chain(
            repeat(renderer.intro(), INTRO_SECONDS),
            renderer.replay_frames(),
            repeat(renderer.outro(), OUTRO_SECONDS),
        ),
        args.out,
        args.crf,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
