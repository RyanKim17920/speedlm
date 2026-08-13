#!/usr/bin/env python3
"""Replay captured agent traffic through a single pane, very fast.

This is the "here is us giving it prompts" beat of the demo, and it runs
*before* the idle/train section.  It answers one question -- what is the raw
material? -- so it deliberately shows only one arm.  The stock timeline is the
one it plays: this segment is about the traffic, not about the comparison, and
the stock arm is the traffic as an ordinary server would have served it.

Everything on screen is recorded, not staged:

  * the response text and its arrival times come from
    ``timeline-stock.jsonl``, exactly as ``demo/capture.py`` wrote them;
  * the token counts are the engine's ``completion_tokens`` per request, and
    the running tally pro-rates the in-flight request by the fraction of its
    characters that have arrived (the same rule ``render.py`` uses, for the
    same reason: chunk boundaries are an artefact of SSE batching);
  * the prompt is the real opening user turn of the captured agent session,
    looked up in the frozen gate suite by ``context_hash``.

The timeline holds responses but not prompts, so the prompts come from the
suite that produced the capture.  ``PROMPT_LINES`` of the wrapped opening user
message are shown and the rest is elided with a marker that says how many lines
were dropped -- the contexts are agent sessions tens of turns deep with 45k
character prompts, so the alternative to truncating is showing nothing legible.
The on-screen label says the prompt is the opening turn, because the request
actually served was that turn plus every tool call after it; ``turn depth`` next
to it is how many messages were really in the request.

The running tally is the point of the segment: contexts, tokens generated and
elapsed serving time, all in the recording's own real seconds.  ``--speed``
scales only the replay clock -- no chunk is dropped and nothing is re-timed --
so the badge stays on every frame to keep those real seconds legible as real
seconds while the video runs 8x faster.

Usage:

    python demo/montage.py \
        --capture-dir /data/.../demo-video-run2 \
        --suite-dir   /data/.../regate-unseen-run1/unseen-suite \
        --out         /data/.../demo-video-run2/traffic-montage.mp4 \
        --speed       8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render import (  # noqa: E402  (needs the sys.path line above)
    BG,
    DIM,
    FPS,
    HEIGHT,
    MUTED,
    PANEL,
    RULE,
    STOCK_ACCENT,
    TEXT,
    TUNED_ACCENT,
    WIDTH,
    Arm,
    encode,
    font,
    load_arm,
    state_at,
    wrap_tail,
)

# How many wrapped lines of the opening user turn are shown before eliding.
PROMPT_LINES = 12
# Static hold on the summary card at the end of the montage.
SUMMARY_SECONDS = 3.0
# Hold on the final replay frame before the summary, so the last response is
# readable rather than flashing past at the replay speed.
TAIL_SECONDS = 0.8


def opening_prompt(context) -> str:
    """The first user message of a captured agent session.

    Later turns of these sessions are tool results, so the opening user message
    is the only part a viewer would recognise as "the prompt we gave it".
    """
    for message in context.messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def wrap_prompt(text: str, columns: int, rows: int) -> list[str]:
    """Wrap a prompt and keep its first ``rows`` lines, marking what was cut."""
    lines: list[str] = []
    for raw in text.strip().split("\n"):
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=columns) or [""])
    if len(lines) <= rows:
        return lines
    dropped = len(lines) - (rows - 1)
    return lines[: rows - 1] + [f"... {dropped} more lines of prompt"]


class Montage:
    def __init__(
        self,
        arm: Arm,
        prompts: dict[str, str],
        prompt_tokens: dict[str, int],
        manifest: dict,
        speed: float,
    ) -> None:
        self.arm = arm
        self.prompts = prompts
        # Kept beside the arm rather than pushed into render.Request: the
        # side-by-side renderer has no use for it and its Request is shared.
        self.prompt_tokens = prompt_tokens
        self.manifest = manifest
        self.speed = speed

        self.f_title = font(38, bold=True)
        self.f_sub = font(19)
        self.f_tally = font(58, bold=True)
        self.f_tally_label = font(17)
        self.f_head = font(20, bold=True)
        self.f_meta = font(16)
        self.f_body = font(17)
        self.f_stat = font(21, bold=True)
        self.f_big = font(72, bold=True)
        self.f_card = font(26)

        # Geometry: one row of tallies, then two columns -- prompt on the left,
        # the response streaming on the right.
        self.tally_top = 150
        self.pane_top = 296
        self.pane_bottom = HEIGHT - 96
        self.left_x, self.left_w = 40, 700
        self.right_x = self.left_x + self.left_w + 32
        self.right_w = WIDTH - 40 - self.right_x

        # The meta line sits at pane_top + 52 and the section label 30px above
        # the body, so the body has to clear pane_top + 112 or the two collide.
        self.body_top = self.pane_top + 118
        self.line_h = 22
        self.rows = max(1, (self.pane_bottom - 24 - self.body_top) // self.line_h)
        # DejaVu Sans Mono advance is 0.602 em.
        self.left_cols = int((self.left_w - 44) / (17 * 0.602))
        self.right_cols = int((self.right_w - 44) / (17 * 0.602))

    # -- chrome ------------------------------------------------------------

    def frame(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        return image, ImageDraw.Draw(image)

    def header(self, draw: ImageDraw.ImageDraw, subtitle: str) -> None:
        draw.text((40, 38), "SpeedLM  ·  the traffic we serve",
                  font=self.f_title, fill=TEXT)
        draw.text((42, 92), subtitle, font=self.f_sub, fill=MUTED)
        draw.line((40, 130, WIDTH - 40, 130), fill=RULE, width=1)

    def footer(self, draw: ImageDraw.ImageDraw, text: str) -> None:
        draw.line((40, HEIGHT - 74, WIDTH - 40, HEIGHT - 74), fill=RULE, width=1)
        draw.text((42, HEIGHT - 58), text, font=self.f_body, fill=DIM)

    def speed_badge(self, draw: ImageDraw.ImageDraw) -> None:
        """Stamp the replay clock's scale on every frame.

        The tallies are the recording's own real seconds and real token counts.
        Without this, a 123s timer emptying in 15s of video reads as a bug.
        """
        text = f"{self.speed:g}x SPEED"
        box = draw.textbbox((0, 0), text, font=self.f_stat)
        w, h = box[2], box[3] - box[1]
        x, y = WIDTH - 52 - w, HEIGHT - 58
        draw.rectangle((x - 12, y - 7, x + w + 12, y + h + 9), fill=TUNED_ACCENT)
        draw.text((x, y), text, font=self.f_stat, fill=BG)

    # -- the running tally -------------------------------------------------

    def tally(
        self,
        draw: ImageDraw.ImageDraw,
        contexts_done: int,
        tokens: int,
        elapsed: float,
        fraction: float,
    ) -> None:
        top = self.tally_top
        draw.rectangle((40, top, WIDTH - 40, top + 118), fill=PANEL)
        cells = [
            ("contexts served", f"{contexts_done}/{len(self.arm.requests)}", TEXT),
            ("tokens generated", f"{tokens:,}", TUNED_ACCENT),
            ("serving time (real)", f"{elapsed:.1f}s", TEXT),
        ]
        x = 76
        for label, value, colour in cells:
            draw.text((x, top + 18), label, font=self.f_tally_label, fill=DIM)
            draw.text((x, top + 40), value, font=self.f_tally, fill=colour)
            x += 470
        draw.text((x + 30, top + 34),
                  "every one of these is a real recorded",
                  font=self.f_meta, fill=DIM)
        draw.text((x + 30, top + 58),
                  "request against the served model",
                  font=self.f_meta, fill=DIM)

        bar_y = top + 108
        draw.rectangle((40, bar_y, WIDTH - 40, bar_y + 6), fill=RULE)
        if fraction > 0:
            draw.rectangle(
                (40, bar_y, 40 + int((WIDTH - 80) * fraction), bar_y + 6),
                fill=TUNED_ACCENT,
            )

    # -- the two columns ---------------------------------------------------

    def prompt_pane(self, draw: ImageDraw.ImageDraw, request) -> None:
        x, w = self.left_x, self.left_w
        draw.rectangle((x, self.pane_top, x + w, self.pane_bottom), fill=PANEL)
        draw.rectangle((x, self.pane_top, x + w, self.pane_top + 4), fill=STOCK_ACCENT)

        draw.text((x + 22, self.pane_top + 22), request.family.upper(),
                  font=self.f_head, fill=STOCK_ACCENT)
        draw.text((x + 22, self.pane_top + 52),
                  f"turn depth {request.turn_depth}   ·   prompt "
                  f"{self.prompt_tokens.get(request.context_hash, 0):,} tok",
                  font=self.f_meta, fill=DIM)
        draw.text((x + 22, self.body_top - 30), "USER PROMPT  (opening turn)",
                  font=self.f_meta, fill=MUTED)

        text = self.prompts.get(request.context_hash, "")
        y = self.body_top
        for line in wrap_prompt(text, self.left_cols, self.rows):
            draw.text((x + 22, y), line, font=self.f_body, fill=TEXT)
            y += self.line_h

    def response_pane(self, draw: ImageDraw.ImageDraw, state) -> None:
        x, w = self.right_x, self.right_w
        draw.rectangle((x, self.pane_top, x + w, self.pane_bottom), fill=PANEL)
        draw.rectangle((x, self.pane_top, x + w, self.pane_top + 4), fill=TUNED_ACCENT)

        draw.text((x + 22, self.pane_top + 22), "MODEL RESPONSE",
                  font=self.f_head, fill=TUNED_ACCENT)
        draw.text((x + 22, self.pane_top + 52),
                  f"{state.request.tokens} tok  ·  {state.request.end - state.request.start:.2f}s "
                  f"on the wall  ·  streamed exactly as recorded",
                  font=self.f_meta, fill=DIM)
        draw.text((x + 22, self.body_top - 30), "STREAMING",
                  font=self.f_meta, fill=MUTED)

        y = self.body_top
        for line in wrap_tail(state.text, self.right_cols, self.rows):
            draw.text((x + 22, y), line, font=self.f_body, fill=TEXT)
            y += self.line_h

    # -- cards -------------------------------------------------------------

    def summary(self) -> Image.Image:
        image, draw = self.frame()
        self.header(draw, "what that traffic adds up to")

        n = len(self.arm.requests)
        tokens = self.arm.total_tokens
        seconds = self.arm.duration

        draw.text((120, 230), f"{n}", font=self.f_big, fill=TEXT)
        draw.text((120, 320), "agent contexts", font=self.f_card, fill=MUTED)
        draw.text((640, 230), f"{tokens:,}", font=self.f_big, fill=TUNED_ACCENT)
        draw.text((640, 320), "tokens generated", font=self.f_card, fill=MUTED)
        draw.text((1300, 230), f"{seconds:.0f}s", font=self.f_big, fill=TEXT)
        draw.text((1300, 320), "of real serving", font=self.f_card, fill=MUTED)

        y = 430
        draw.line((120, y - 20, WIDTH - 120, y - 20), fill=RULE, width=1)
        families: dict[str, int] = {}
        for request in self.arm.requests:
            families[request.family] = families.get(request.family, 0) + 1
        # Wrapped rather than one line: six families at font 26 overrun the
        # right margin, and the last one would be silently clipped off frame.
        summary = "   ·   ".join(
            f"{name} {count}"
            for name, count in sorted(families.items(), key=lambda kv: -kv[1])
        )
        family_lines = textwrap.wrap(summary, width=96) or [""]
        for text, colour in (
            *((line, MUTED) for line in family_lines),
            ("", TEXT),
            ("Traffic like this is the corpus. Every request the server answers is", TEXT),
            ("recorded as a trace, and the idle cycle trains the draft head on it.", TEXT),
            ("", TEXT),
            ("These particular contexts are the held-out, session-disjoint slice:", MUTED),
            ("they are what the gate replays, so the draft head never trains on them.",
             TUNED_ACCENT),
        ):
            draw.text((120, y), text, font=self.f_card, fill=colour)
            y += 46

        self.footer(draw, self.provenance())
        return image

    def provenance(self) -> str:
        return (
            f"capture job {self.manifest.get('slurm_job_id', '?')}  ·  "
            f"{self.manifest.get('model', '')}  ·  greedy, max 512 new tokens  ·  "
            "unseen-session suite (100 contexts)"
        )

    # -- the replay --------------------------------------------------------

    def replay_frames(self) -> Iterator[Image.Image]:
        n = len(self.arm.requests)
        subtitle = (
            f"{n} recorded agent contexts replayed through the served model  ·  "
            f"{self.manifest.get('model', '')}  ·  greedy"
        )
        seconds = self.arm.duration
        frames = int(seconds / self.speed * FPS)
        for i in range(frames + int(TAIL_SECONDS * FPS)):
            t = min(i / FPS * self.speed, seconds)
            state = state_at(self.arm, t)
            image, draw = self.frame()
            self.header(draw, subtitle)
            self.tally(
                draw,
                contexts_done=(n if state.done else state.index + 1),
                tokens=state.tokens_done,
                elapsed=state.elapsed,
                fraction=(
                    1.0
                    if state.done
                    else (state.chars_done / self.arm.total_chars
                          if self.arm.total_chars else 0.0)
                ),
            )
            self.prompt_pane(draw, state.request)
            self.response_pane(draw, state)
            self.footer(draw, self.provenance())
            self.speed_badge(draw)
            yield image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--speed", type=float, default=8.0,
                        help="advance the replay clock this many recorded seconds per "
                             "second of video; the tallies stay in real recorded seconds")
    parser.add_argument("--frames-only", type=int, default=0,
                        help="render only the first N replay frames as PNGs")
    args = parser.parse_args(argv)
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    from speedlm.gate.suite import load_suite

    manifest = json.loads((args.capture_dir / "capture_manifest.json").read_text())
    arm = load_arm(args.capture_dir / "timeline-stock.jsonl", "TRAFFIC",
                   manifest["stock_draft"], STOCK_ACCENT)

    suite = load_suite(args.suite_dir)
    prompts = {c.context_hash: opening_prompt(c) for c in suite.contexts}
    missing = [r.context_hash for r in arm.requests if not prompts.get(r.context_hash)]
    if missing:
        # Refusing rather than drawing an empty prompt pane: a montage whose
        # point is "these are the real prompts" must not show a blank one.
        raise SystemExit(
            f"{len(missing)} replayed contexts have no prompt in {args.suite_dir}, "
            f"first {missing[0]}"
        )

    prompt_tokens = {
        row["context_hash"]: int(row["prompt_tokens"])
        for row in (
            json.loads(line)
            for line in (args.capture_dir / "timeline-stock.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    montage = Montage(arm, prompts, prompt_tokens, manifest, speed=args.speed)
    print(f"{len(arm.requests)} contexts  {arm.total_tokens} tokens  "
          f"{arm.duration:.1f}s recorded  ->  "
          f"{arm.duration / args.speed + TAIL_SECONDS + SUMMARY_SECONDS:.1f}s of video")

    if args.frames_only:
        out_dir = args.out.parent / "montage-frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        montage.summary().save(out_dir / "summary.png")
        for i, image in enumerate(montage.replay_frames()):
            if i >= args.frames_only:
                break
            if i % 30 == 0:
                image.save(out_dir / f"montage-{i:05d}.png")
        print(f"wrote sample frames to {out_dir}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encode(
        itertools.chain(
            montage.replay_frames(),
            itertools.repeat(montage.summary(), int(SUMMARY_SECONDS * FPS)),
        ),
        args.out,
        args.crf,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
