#!/usr/bin/env python3
"""Replay two captured drafting timelines side by side and encode them to video.

Input is what ``demo/capture.py`` wrote: one JSONL timeline per arm, each
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

The totals the video shows are this recording's own, but every caption quotes the
GATED numbers alongside them and says which is which. One sequential pass over a
few dozen contexts is an illustration; the promotion decision came from 5 repeats
x 100 contexts x 2 blocks per arm.

``--speed`` exists because this segment closes a longer demo and has to land in
under a minute.  It scales the shared replay clock rather than the recording: no
chunk is dropped and neither arm is re-timed, so the gap between the arms keeps
its true shape.  Everything the panes print -- elapsed, tokens, tok/s, accepted
length -- and every total on the summary card stays in the recording's own real
seconds; only the wall time you spend watching it changes.  A badge on every
replay frame says so, since otherwise a real 123.4s timer running out in 31s of
video would read as a bug.

Usage:

    python demo/render.py \
        --capture-dir /data/.../demo-video-run1 \
        --out         /data/.../demo-video-run1/speedlm-drafting.mp4 \
        --speed       4
"""

from __future__ import annotations

import argparse
import itertools
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


@dataclass
class Request:
    family: str
    context_hash: str
    turn_depth: int
    start: float      # absolute seconds on the arm's own clock
    end: float
    tokens: int
    accepted_length: float
    chunks: list[Chunk]
    text: str
    total_chars: int  # len of all chunk text, precomputed: state_at needs it every frame


@dataclass
class Arm:
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
    def mean_accepted_length(self) -> float | None:
        # None, not 0.0, when the engine never advanced its speculative counters
        # over any request in this arm.  A measured zero and an unmeasured value
        # must not render the same: this figure is printed on the summary card
        # directly beside the gated +0.3457, so a reader who saw 0.000 would take
        # it as a refutation of the gated result rather than as the absence of a
        # measurement.  The capture side writes null for exactly this reason, and
        # flattening it back to a number here would undo that.
        measured = [r.accepted_length for r in self.requests if r.accepted_length > 0]
        return sum(measured) / len(measured) if measured else None


def load_arm(
    path: Path, label: str, draft: str, accent: tuple[int, int, int]
) -> Arm:
    """Lay one arm's requests back to back on a single clock."""
    requests: list[Request] = []
    cursor = 0.0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            start = cursor
            chunks = [
                Chunk(t=start + float(c["t"]), text=c["text"])
                for c in row["tokens"]
            ]
            end = start + float(row["wall_s"])
            requests.append(
                Request(
                    family=str(row["family"]),
                    context_hash=str(row["context_hash"]),
                    turn_depth=int(row["turn_depth"]),
                    start=start,
                    end=end,
                    tokens=int(row["completion_tokens"]),
                    accepted_length=float(row["accepted_length"]),
                    chunks=chunks,
                    text=str(row["text"]),
                    total_chars=sum(len(c.text) for c in chunks),
                )
            )
            cursor = end
    if not requests:
        raise SystemExit(f"{path} contains no requests")
    return Arm(label=label, draft=draft, accent=accent, requests=requests)


@dataclass
class ArmFrameState:
    """Everything one pane needs to draw itself at one instant."""

    index: int              # 0-based request currently on screen
    done: bool
    elapsed: float
    tokens_done: int        # tokens completed across all requests so far
    chars_done: int         # characters of output emitted so far, across all requests
    text: str
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
        total = request.total_chars
        tokens_done += int(request.tokens * seen / total) if total else 0

    parts: list[str] = []
    for chunk in request.chunks:
        if chunk.t > t and not done:
            break
        parts.append(chunk.text)

    return ArmFrameState(
        index=idx,
        done=done,
        elapsed=min(t, arm.duration),
        tokens_done=tokens_done,
        chars_done=(
            arm.total_chars
            if done
            else (arm.chars_at[k - 1] if (k := bisect_right(arm.chunk_times, t)) else 0)
        ),
        text="".join(parts),
        request=request,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def accepted_text(value: float | None) -> str:
    """Render an accepted length, showing an unmeasured one as ``n/a``.

    Kept in one place so the card, the pane and the stdout summary cannot drift
    into disagreeing about what a missing engine measurement looks like.
    """
    return "n/a" if value is None else f"{value:.3f}"


def wrap_tail(text: str, columns: int, rows: int) -> list[str]:
    """Wrap ``text`` to ``columns`` and return only its last ``rows`` lines."""
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(
            raw, width=columns, replace_whitespace=False, drop_whitespace=False
        ) or [""])
    return lines[-rows:]


class Renderer:
    def __init__(
        self,
        stock: Arm,
        tuned: Arm,
        manifest: dict[str, Any],
        speed: float = 1.0,
    ) -> None:
        self.stock = stock
        self.tuned = tuned
        self.manifest = manifest
        # How many seconds of recorded time the replay clock advances per second
        # of video.  This scales the clock only: no recorded chunk is dropped and
        # no arm's timeline is re-timed, so the two arms keep their exact relative
        # shape and every token the capture recorded still appears on screen.
        self.speed = speed
        # Greedy decoding should make the two arms emit the same text: a draft
        # head changes how many tokens are proposed per step, not which tokens
        # survive verification.  Counting the agreement rather than asserting it
        # is what lets the summary card report it as a measurement, and it is
        # what lead() checks before claiming the stock arm is behind "the same
        # output": arms that had quietly drifted apart would show up here first.
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
        # The body stops 14px above the progress bar, which itself sits 16px above
        # the pane floor.  Fixed for every pane and every frame, so it is resolved
        # here rather than per frame.
        self.body_rows = max(1, (self.pane_bottom - 30 - self.body_top) // self.line_h)
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
        # The capture leaves accepted_length at zero when the engine's speculative
        # counters did not advance for this request, and a real mean accepted
        # length is never below one.  So zero here means unmeasured, and printing
        # it as 0.00 beside the gated figure in the footer would read as a
        # measured collapse rather than as a missing measurement.
        accepted = state.request.accepted_length
        stats = [
            ("elapsed", f"{state.elapsed:6.1f}s"),
            ("tokens", f"{state.tokens_done:5d}"),
            ("tok/s", f"{rate:5.1f}"),
            ("accepted len", f"{accepted:4.2f}" if accepted > 0 else " n/a"),
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

        y = self.body_top
        for line in wrap_tail(state.text, self.columns, self.body_rows):
            draw.text((x + 22, y), line, font=self.f_body, fill=TEXT)
            y += self.line_h

        if state.done:
            badge = f"  FINISHED  {arm.duration:.1f}s  "
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
    ) -> None:
        """Show how far ahead the tuned arm is, in seconds.

        Both figures are read straight off the stock arm's own recorded timeline
        rather than extrapolated from a rate.  Once the tuned arm has finished it
        is the wall time the stock arm still has left to run; before that it is
        how much longer the stock arm needs to catch up to the text the tuned arm
        has already emitted, matched by character count rather than by request,
        so the caption keeps moving inside a long context instead of stepping
        once per request.
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

    def speed_badge(self, draw: ImageDraw.ImageDraw) -> None:
        """Stamp the replay clock's scale factor on the frame.

        This is drawn on every replay frame, not just the first few, because the
        panes show each arm's own elapsed wall clock and those numbers are real
        seconds.  A viewer who joined halfway through and could not see that the
        replay clock is scaled would read a pane ticking to 123.4s over 31s of
        video as the timers being wrong, or worse, take the video's own running
        time as the measured result.  The badge is what keeps the real seconds on
        screen legible as real seconds.

        It sits at the right end of the footer row: the header's right side is
        already taken by the lead readout, and each pane's top right is where the
        FINISHED badge lands, so this is the one piece of chrome-free margin that
        is present on every frame.
        """
        text = f"{self.speed:g}x SPEED"
        # box[2] rather than box[2] - box[0], for the same reason lead() does it:
        # the left bearing is part of where the glyphs land, so subtracting it
        # would push the pill past the right margin.
        box = draw.textbbox((0, 0), text, font=self.f_stat)
        w, h = box[2], box[3] - box[1]
        x = WIDTH - 52 - w
        y = HEIGHT - 58
        draw.rectangle((x - 12, y - 7, x + w + 12, y + h + 9), fill=TUNED_ACCENT)
        draw.text((x, y), text, font=self.f_stat, fill=BG)

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
            ("The mechanism's definitive gated result, on a later and larger corpus:", MUTED),
        ]
        y = 300
        for text, colour in lines:
            draw.text((120, y), text, font=self.f_card, fill=colour)
            y += 46

        # The headline gets the card's largest type. Everything qualifying it is
        # true and stays on the card, one size down: the point of the card is
        # that a viewer leaves with the drafting number, not with a verdict.
        draw.text((120, y + 4), "+15.0%", font=self.f_big, fill=TUNED_ACCENT)
        draw.text((384, y + 14), "more tokens accepted per verifier step",
                  font=self.f_card_bold, fill=TEXT)
        draw.text((384, y + 52), "2.3051 -> 2.6507 tokens/step  ·  +0.3457, SE 0.0029",
                  font=self.f_card, fill=MUTED)
        draw.text((120, y + 108),
                  "That run's gate vetoed its throughput channel as non-stationary; the "
                  "accepted-length channel passed.",
                  font=self.f_sub, fill=MUTED)
        self.footer(
            draw,
            "replay: capture job 378546, an earlier idle-tuned head  ·  "
            "gate figures: regate-big-run2, 287-context session-disjoint suite",
        )
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
             accepted_text(self.stock.mean_accepted_length),
             accepted_text(self.tuned.mean_accepted_length)),
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

        y += 24
        draw.line((120, y - 20, WIDTH - 120, y - 20), fill=RULE, width=1)

        # The closing beat lands on the drafting result, so it gets the largest
        # type on the card and the last word. The replay totals above it are this
        # clip's own arithmetic; this is the measured, reproduced number.
        draw.text((120, y + 2), "+15.0%", font=self.f_big, fill=TUNED_ACCENT)
        draw.text((384, y + 10), "more tokens accepted per verifier step",
                  font=self.f_card_bold, fill=TEXT)
        draw.text((384, y + 48),
                  "2.3051 -> 2.6507 tokens/step  ·  +0.3457, SE 0.0029  ·  +11.52pp acceptance",
                  font=self.f_card, fill=MUTED)
        y += 96

        for text, colour in (
            ("Reproduced three times on this head: +0.3416 / +0.3402 / +0.3457, each on a"
             " 287-context", MUTED),
            ("session-disjoint suite. The number to cite is regate-big-run2: 8 repeats x 287"
             " contexts.", MUTED),
            ("Wall-clock throughput there ran +12.9% to +20.7% across repeats -- the tuned arm held"
             " 143-147", MUTED),
            ("tok/s while the shared-node baseline drifted 127->121, so we quote the drafting"
             " metric.", MUTED),
            ("That gate vetoed the throughput channel as non-stationary (final verdict: reject);"
             " the", MUTED),
            ("accepted-length channel passed at +0.3457. This replay is one sequential pass of an"
             " earlier", MUTED),
            ("head, shown because it is watchable.", MUTED),
        ):
            draw.text((120, y), text, font=self.f_sub, fill=colour)
            y += 30

        self.footer(
            draw,
            "replay: capture job 378546, an earlier idle-tuned head  ·  "
            "gate figures: regate-big-run2, 287-context session-disjoint suite",
        )
        return image

    # -- the replay --------------------------------------------------------

    def replay_frames(self) -> Iterator[Image.Image]:
        replay_seconds = max(self.stock.duration, self.tuned.duration)
        n_contexts = len(self.stock.requests)
        subtitle = (
            f"{n_contexts} held-out agent contexts · greedy · "
            f"{self.manifest.get('model', '')} · one GPU, sequential"
        )
        # The replay is shortened by advancing the shared clock self.speed seconds
        # per second of video, never by skipping frames or resampling the arms:
        # each frame still asks state_at for the exact state at its own instant,
        # so every recorded chunk is still rendered, just sooner.  The trailing
        # hold keeps its real duration -- it is a static frame like the cards, and
        # a 1.5s beat before the summary is already the shortest it can usefully be.
        frames = int(replay_seconds / self.speed * FPS)
        for i in range(frames + int(TAIL_SECONDS * FPS)):
            # Clamped so the hold sits on the true end of the replay rather than
            # running the clock past it; state_at saturates there in any case.
            t = min(i / FPS * self.speed, replay_seconds)
            image, draw = self.frame()
            self.header(draw, subtitle)
            stock_state = state_at(self.stock, t)
            tuned_state = state_at(self.tuned, t)
            self.pane(draw, self.stock, stock_state, self.pane_x[0], n_contexts)
            self.pane(draw, self.tuned, tuned_state, self.pane_x[1], n_contexts)
            self.lead(draw, stock_state, tuned_state)
            self.footer(
                draw,
                "+15.0% more tokens accepted per verifier step "
                "(+0.3457, SE 0.0029) on the 287-context session-disjoint suite"
                "  ·  instrumentation gaps removed from both arms",
            )
            # Only when the clock is actually scaled: a "1x SPEED" pill on an
            # unscaled render would be chrome that tells the viewer nothing.
            if self.speed != 1.0:
                self.speed_badge(draw)
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


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay the drafting segment this many times faster; the "
                             "intro and outro cards keep their real duration, and the "
                             "numbers on screen stay real measured seconds")
    parser.add_argument("--frames-only", type=int, default=0,
                        help="render only the first N replay frames as PNGs, for a quick look")
    args = parser.parse_args(argv)
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    capture: Path = args.capture_dir
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    stock = load_arm(capture / "timeline-stock.jsonl", "STOCK",
                     manifest["stock_draft"], STOCK_ACCENT)
    tuned = load_arm(capture / "timeline-candidate.jsonl", "IDLE-TUNED",
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

    renderer = Renderer(stock, tuned, manifest, speed=args.speed)
    print(f"stock  {stock.duration:.1f}s  {stock.total_tokens} tok  "
          f"accepted {accepted_text(stock.mean_accepted_length)}")
    print(f"tuned  {tuned.duration:.1f}s  {tuned.total_tokens} tok  "
          f"accepted {accepted_text(tuned.mean_accepted_length)}")

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
        itertools.chain(
            itertools.repeat(renderer.intro(), int(INTRO_SECONDS * FPS)),
            renderer.replay_frames(),
            itertools.repeat(renderer.outro(), int(OUTRO_SECONDS * FPS)),
        ),
        args.out,
        args.crf,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
