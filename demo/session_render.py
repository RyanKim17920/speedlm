#!/usr/bin/env python3
"""Rasterise an asciicast v2 recording into an mp4.

Input is what ``demo/session_record.py`` wrote: the literal byte stream a real
shell emitted on a real pseudo-terminal, with the real times at which it emitted
it.  This script does not decide what the terminal said -- it replays those
bytes through ``pyte``, a VT100/xterm emulator, and draws whatever character
grid the emulator ends up in.  The honesty property is worth stating precisely,
because it is the whole reason to do it this way: the *content* is the shell's,
and only the *rasterisation* is ours.  Fonts, colours, cell metrics and frame
timing are our choices; the characters, their attributes and their positions are
the terminal's.

Colour handling, since it is the part most likely to be quietly wrong:

  * The 16 ANSI colours arrive from pyte as names ("red", "brightcyan", and
    "brown", which is pyte's name for ANSI yellow) and are looked up in
    ``PALETTE`` below.  The palette is a choice, like a terminal's colour scheme.
  * 256-colour (``ESC[38;5;N``) and truecolour (``ESC[38;2;R;G;B``) are both
    surfaced by pyte 0.8.2 as a six-hex-digit string, already resolved to RGB in
    the 256-colour case, and are used verbatim.  Verified against pyte 0.8.2:
    ``38;5;196`` arrives as ``ff0000`` and ``38;2;255;128;0`` as ``ff8000``.
  * Bold on a named colour is drawn as that colour's bright variant, which is
    what a real terminal does.  Bold on a hex colour keeps the exact hex, since
    there is no defined "brighter" for an arbitrary RGB triple.

Time is the one thing this script is allowed to edit, and it says so on screen.
A real tuning cycle spends most of its wall clock inside a model load that emits
nothing at all, so replaying it at 1:1 would be twenty minutes of a still frame.
``--max-gap`` shrinks each stretch of dead air to a couple of seconds and
``--speed`` scales what is left.  Neither one drops or reorders a single
recorded byte -- every event still plays, in order; only the silence between
events gets shorter.  Whenever a gap is compressed the renderer draws a marker
naming the real wall time that elapsed there, because a screencast that quietly
cuts eight minutes of model loading is claiming the pipeline is faster than it
is, and that is the sort of dishonesty a demo video is most tempting to commit.

Known limitations, both inherited from the emulator rather than introduced here:

  * No alternate-screen support.  pyte 0.8.2 does not implement the
    ``ESC[?1049h`` / ``?1047`` / ``?47`` buffer switch; it ignores the sequence
    rather than failing, so a TUI's output lands in the normal buffer and mixes
    with the scrollback instead of taking over the screen.  Nothing hangs or
    crashes, but the frame is not what the TUI intended, so this script warns
    once when it sees such a sequence.  Keep full-screen TUIs out of the script.
  * CJK and emoji occupy two cells.  pyte stores the glyph in the first cell and
    an empty string in the second; a monospace font draws it one cell wide, so
    such text can bleed into or leave a gap beside its neighbour.

Usage:

    python demo/session_render.py session.cast session.mp4 \\
        [--fps 30] [--font-size 28] [--width 1920] [--height 1080] \\
        [--max-gap 2.0] [--speed 1.0] [--timing-out session.timing.json]

``--max-gap 0`` disables gap compression entirely and plays the recording at its
real pace.  The default is 2.0 seconds, because unbounded dead air is never what
anyone actually wants in a screencast.
"""

from __future__ import annotations

import argparse
import json
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pyte
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSansMono.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSansMono-Bold.ttf"

# A dark 16-colour scheme.  The keys are exactly the colour names pyte emits,
# including "brown", which is what pyte calls ANSI yellow.
PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (46, 52, 64),
    "red": (191, 97, 106),
    "green": (163, 190, 140),
    "brown": (235, 203, 139),
    "blue": (110, 155, 214),
    "magenta": (180, 142, 173),
    "cyan": (136, 192, 208),
    "white": (216, 222, 233),
    "brightblack": (108, 117, 133),
    "brightred": (219, 122, 130),
    "brightgreen": (185, 209, 163),
    "brightbrown": (245, 222, 170),
    "brightblue": (140, 180, 232),
    "brightmagenta": (203, 168, 197),
    "brightcyan": (166, 216, 228),
    "brightwhite": (240, 244, 250),
}
DEFAULT_FG = (216, 222, 233)
DEFAULT_BG = (30, 34, 42)
CURSOR_COLOR = (200, 208, 220)
MARGIN = 24

# The compressed-gap marker.  It is deliberately dim and small -- it is a
# footnote about the edit, not a thing the viewer should be reading instead of
# the terminal -- but it sits on its own filled plate so it stays legible no
# matter which characters it happens to land on top of.
MARKER_FG = (150, 160, 176)
MARKER_BG = (44, 50, 62)
MARKER_BORDER = (72, 80, 96)
MARKER_PAD = 10
# U+23E9 is the "fast forward" glyph one would like to use.  DejaVu Sans Mono
# does not ship most of the Miscellaneous Technical block, and a missing glyph
# renders as a hollow .notdef box that reads as a rendering bug rather than as a
# deliberate mark, so the ASCII form is the fallback.
MARKER_GLYPH = "⏩"
MARKER_GLYPH_ASCII = ">>"

BRIGHTEN = {
    name: f"bright{name}"
    for name in ("black", "red", "green", "brown", "blue", "magenta", "cyan", "white")
}

# Private-mode sequences that switch to the alternate screen.  pyte ignores
# them, which is graceful but wrong-looking, so their presence is worth a
# warning rather than silence.
ALT_SCREEN_MARKERS = ("\x1b[?1049h", "\x1b[?1047h", "\x1b[?47h")

_HEX = set(string.hexdigits)


def resolve_color(
    name: str | None, *, bold: bool, default: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Map one pyte colour token to RGB.

    pyte hands back either a palette name, the literal string "default", or a
    six-hex-digit string for 256-colour and truecolour SGR.  Anything else is a
    token this renderer does not know, and falling back to the default is better
    than guessing a colour that was never asked for.
    """
    if not name or name == "default":
        return default
    if name in PALETTE:
        return PALETTE[BRIGHTEN[name]] if bold and name in BRIGHTEN else PALETTE[name]
    if len(name) == 6 and set(name) <= _HEX:
        return (int(name[0:2], 16), int(name[2:4], 16), int(name[4:6], 16))
    return default


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def load_events(path: Path) -> tuple[dict, list[tuple[float, str]]]:
    """Read an asciicast v2 file into (header, [(time, output text), ...])."""
    with path.open(encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        events = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t, kind, data = json.loads(line)
            # Only "o" (output) events exist in our recordings; input and resize
            # events are part of the format but nothing here emits them.
            if kind == "o":
                events.append((float(t), data))
    if header.get("version") != 2:
        raise SystemExit(f"{path}: not an asciicast v2 file")
    return header, events


# ---------------------------------------------------------------------------
# Timeline editing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """One stretch of dead air that was compressed.

    ``start`` and ``end`` are on the *rendered* clock, which is what the frame
    loop counts in; ``real`` is the wall time the recording actually spent
    there, which is what the marker has to say out loud.  ``cast_start`` and
    ``cast_end`` are the same stretch on the recording's own clock, kept so the
    timing sidecar can state the mapping in both directions rather than only in
    the one the frame loop happens to need.
    """

    start: float
    end: float
    real: float
    cast_start: float
    cast_end: float


def format_duration(seconds: float) -> str:
    """Render a duration the way a human reads a stopwatch, not a float."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def compress_timeline(
    events: list[tuple[float, str]], max_gap: float, speed: float
) -> tuple[list[tuple[float, str]], list[Gap]]:
    """Remap event times so no silence lasts longer than ``max_gap``, then scale.

    This is a pure remap: the returned list has exactly the same events in
    exactly the same order, only with earlier timestamps.  Doing it as a remap
    rather than by skipping events is what keeps the property that every
    recorded byte still reaches the emulator -- a compressed gap changes when
    the next chunk is fed, never whether it is fed.

    ``max_gap <= 0`` means "leave the pacing alone", so the whole compression
    pass collapses to the identity and only ``speed`` applies.
    """
    shifted: list[tuple[float, str]] = []
    gaps: list[Gap] = []
    removed = 0.0
    prev = 0.0
    for t, data in events:
        # The stretch before the first event counts too: a recording that opens
        # with thirty seconds of nothing is dead air like any other.
        idle = t - prev
        if max_gap > 0 and idle > max_gap:
            start = prev - removed
            gaps.append(
                Gap(
                    start=start / speed,
                    end=(start + max_gap) / speed,
                    real=idle,
                    cast_start=prev,
                    cast_end=t,
                )
            )
            removed += idle - max_gap
        shifted.append(((t - removed) / speed, data))
        prev = t
    return shifted, gaps


def timing_sidecar(
    *,
    cast: Path,
    out: Path,
    gaps: list[Gap],
    fps: int,
    max_gap: float,
    speed: float,
    tail: float,
    total_cast_seconds: float,
    total_video_seconds: float,
    n_frames: int,
) -> dict:
    """Describe the cast-clock -> video-clock mapping this render applied.

    Why this exists at all: anything that wants to draw *over* the video at the
    moment a particular line hit the terminal -- a chart that fills in as the
    numbers scroll past, a caption pinned to an event -- has to know where that
    moment landed, and after gap compression the two clocks are no longer
    related by a constant.  Only the renderer knows how the clock was bent, so
    only the renderer can say; a consumer guessing from ratios of the two
    durations will be wrong by however much dead air happened to sit before the
    event.  Emitting the mapping is what keeps such an overlay a measurement
    instead of a guess.

    The mapping is exactly piecewise linear, so ``breakpoints`` -- ``[cast_t,
    video_t]`` knots, in order -- states it without loss and in a handful of
    entries: linearly interpolate between the bracketing pair.  The knots sit at
    the edges of every compressed gap (the only places the slope changes), plus
    the origin and the last event.  Inside a gap the interpolation spreads the
    real elapsed time uniformly across the shortened stretch, which is the same
    thing the on-screen marker claims; no event is fed to the emulator there, so
    nothing observable depends on the choice.

    Times are seconds.  ``video_t * fps`` is the frame the moment lands on, and
    frames past ``total_video_seconds`` are the held tail, not recorded time.
    """
    knots: list[tuple[float, float]] = [(0.0, 0.0)]
    for gap in gaps:
        knots.append((gap.cast_start, gap.start))
        knots.append((gap.cast_end, gap.end))
    knots.append((total_cast_seconds, total_video_seconds - tail))

    # A gap that opens at t=0 duplicates the origin knot; a strictly increasing
    # list is what makes interpolation on the consumer side unambiguous.
    breakpoints: list[list[float]] = []
    for cast_t, video_t in knots:
        if breakpoints and cast_t <= breakpoints[-1][0]:
            continue
        breakpoints.append([round(cast_t, 6), round(video_t, 6)])
    if breakpoints[0][0] > 0.0:
        breakpoints.insert(0, [0.0, 0.0])

    doc = {
        "schema": "speedlm-session-timing/1",
        "cast": str(cast),
        "video": str(out),
        "fps": fps,
        "max_gap": max_gap,
        "speed": speed,
        "tail_seconds": tail,
        "total_cast_seconds": round(total_cast_seconds, 6),
        "total_video_seconds": round(total_video_seconds, 6),
        "total_frames": n_frames,
        "breakpoints": breakpoints,
        "gaps": [
            {
                "cast_start": round(g.cast_start, 6),
                "cast_end": round(g.cast_end, 6),
                "video_start": round(g.start, 6),
                "video_end": round(g.end, 6),
                "real_seconds": round(g.real, 6),
            }
            for g in gaps
        ],
    }
    return doc


def snapshot(screen: pyte.Screen) -> tuple[tuple, tuple[int, int] | None]:
    """Freeze the screen into a hashable form: per-cell attributes + cursor.

    Hashability is the point.  A terminal spends most of its frames unchanged,
    and comparing this tuple against the previous one lets the encoder reuse the
    last drawn image instead of redrawing thousands of identical grids.
    """
    rows = []
    for y in range(screen.lines):
        line = screen.buffer[y]
        rows.append(
            tuple(
                (c.data, c.fg, c.bg, c.bold, c.reverse, c.underscore)
                for c in (line[x] for x in range(screen.columns))
            )
        )
    cursor = None if screen.cursor.hidden else (screen.cursor.x, screen.cursor.y)
    return tuple(rows), cursor


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


# A codepoint in the Private Use Area, which no general-purpose font assigns.
# Whatever the font draws for it *is* its .notdef, so it makes a reference
# bitmap to compare candidate glyphs against.
_MISSING_PROBE = ""


def font_has_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    """Report whether ``font`` actually has a glyph for ``ch``.

    FreeType never errors on a missing character -- it silently substitutes
    .notdef, the hollow box -- so asking is the only way to know, and there is
    no cmap reader here to ask with (fontTools is not a dependency).  Rendering
    the character and a codepoint known to be unmapped and comparing the two
    bitmaps answers it exactly: identical output means both landed on .notdef.
    """
    try:
        candidate = font.getmask(ch, mode="L")
        notdef = font.getmask(_MISSING_PROBE, mode="L")
    except Exception:  # noqa: BLE001 - a font that cannot render it has not got it
        return False
    return not (candidate.size == notdef.size and bytes(candidate) == bytes(notdef))


class Renderer:
    def __init__(self, cols: int, rows: int, font_size: int, canvas: tuple[int, int]) -> None:
        self.cols, self.rows = cols, rows
        for path in (FONT_REGULAR, FONT_BOLD):
            if not path.exists():
                raise SystemExit(f"missing font {path}; install fonts-dejavu-core")
        self.regular = ImageFont.truetype(str(FONT_REGULAR), font_size)
        self.bold = ImageFont.truetype(str(FONT_BOLD), font_size)
        # The font is monospace, so any glyph's advance is the cell width; "M"
        # is just a conventional choice.
        self.cell_w = round(self.regular.getlength("M"))
        self.cell_h = round(font_size * 1.22)
        grid_w, grid_h = self.cols * self.cell_w, self.rows * self.cell_h
        # H.264 with yuv420p subsamples chroma by two, so both dimensions have
        # to be even or the encoder rejects the stream outright.
        self.width = canvas[0] + (canvas[0] % 2)
        self.height = canvas[1] + (canvas[1] % 2)
        if grid_w + 2 * MARGIN > self.width or grid_h + 2 * MARGIN > self.height:
            raise SystemExit(
                f"grid {grid_w}x{grid_h} plus margins does not fit canvas "
                f"{self.width}x{self.height}; lower --font-size"
            )
        self.ox = (self.width - grid_w) // 2
        self.oy = (self.height - grid_h) // 2
        # Nudge glyphs down inside the cell so they sit on a sensible baseline
        # instead of hugging the top of their row.
        self.glyph_dy = max(0, (self.cell_h - font_size) // 2 - 2)
        self.colors_seen: set[str] = set()
        # The marker is a caption about the video, not part of the terminal, so
        # it is drawn smaller than the grid's own type to read as an annotation.
        self.marker_font = ImageFont.truetype(str(FONT_REGULAR), max(12, round(font_size * 0.62)))
        self.marker_glyph = (
            MARKER_GLYPH
            if font_has_glyph(self.marker_font, MARKER_GLYPH)
            else MARKER_GLYPH_ASCII
        )

    def marker_text(self, real_seconds: float) -> str:
        """The exact words the compressed-gap overlay shows."""
        return f"{self.marker_glyph} {format_duration(real_seconds)} elapsed"

    def with_marker(self, base: Image.Image, text: str) -> Image.Image:
        """Return a copy of ``base`` with the compressed-gap marker on it.

        The base frame is left untouched so the frame cache upstream keeps
        working: the same unchanged screen can be reused both with and without
        the overlay, which matters because a compressed gap is by definition a
        stretch where the screen is not changing at all.
        """
        img = base.copy()
        d = ImageDraw.Draw(img)
        left, top, right, bottom = d.textbbox((0, 0), text, font=self.marker_font)
        w, h = right - left, bottom - top
        # Bottom-right of the character grid.  During a compressed gap nothing is
        # being printed, and the last rows of a scrolling terminal are the ones
        # least likely to be holding something the viewer still needs to read.
        x1 = self.ox + self.cols * self.cell_w
        y1 = self.oy + self.rows * self.cell_h
        x0 = x1 - w - 2 * MARKER_PAD
        y0 = y1 - h - 2 * MARKER_PAD
        d.rectangle([x0, y0, x1, y1], fill=MARKER_BG, outline=MARKER_BORDER, width=1)
        d.text((x0 + MARKER_PAD - left, y0 + MARKER_PAD - top), text,
               font=self.marker_font, fill=MARKER_FG)
        return img

    def draw(self, state: tuple, cursor: tuple[int, int] | None) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), DEFAULT_BG)
        d = ImageDraw.Draw(img)
        for y, row in enumerate(state):
            py = self.oy + y * self.cell_h
            # Coalesce runs of identically styled cells so a full-width line of
            # ordinary text costs one text() call rather than one per column.
            # Each run is positioned from its column index, never from the
            # accumulated width of the text drawn so far, so a double-width cell
            # can only disturb its own run and never shifts the rest of the row.
            run_start = 0
            while run_start < len(row):
                _, fg, bg, bold, reverse, underscore = row[run_start]
                run_end = run_start + 1
                while run_end < len(row) and row[run_end][1:] == row[run_start][1:]:
                    run_end += 1
                text = "".join(c[0] for c in row[run_start:run_end])
                fg_rgb = resolve_color(fg, bold=bold, default=DEFAULT_FG)
                bg_rgb = resolve_color(bg, bold=False, default=DEFAULT_BG)
                if reverse:
                    fg_rgb, bg_rgb = bg_rgb, fg_rgb
                for token in (fg, bg):
                    if token not in (None, "default"):
                        self.colors_seen.add(token)
                px = self.ox + run_start * self.cell_w
                run_w = (run_end - run_start) * self.cell_w
                if bg_rgb != DEFAULT_BG:
                    d.rectangle([px, py, px + run_w - 1, py + self.cell_h - 1], fill=bg_rgb)
                if text.strip():
                    d.text(
                        (px, py + self.glyph_dy),
                        text,
                        font=self.bold if bold else self.regular,
                        fill=fg_rgb,
                    )
                if underscore:
                    uy = py + self.cell_h - 3
                    d.line([px, uy, px + run_w - 1, uy], fill=fg_rgb)
                run_start = run_end
        if cursor is not None:
            cx, cy = cursor
            if 0 <= cx < self.cols and 0 <= cy < self.rows:
                px = self.ox + cx * self.cell_w
                py = self.oy + cy * self.cell_h
                d.rectangle(
                    [px, py, px + self.cell_w - 1, py + self.cell_h - 1],
                    outline=CURSOR_COLOR,
                    width=2,
                )
        return img


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an asciicast v2 recording to mp4.")
    ap.add_argument("cast", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--tail", type=float, default=1.5, help="seconds to hold the final frame")
    ap.add_argument(
        "--max-gap",
        type=float,
        default=2.0,
        help="shrink any stretch of no output longer than this to this many "
             "seconds, marking it on screen with the real time elapsed; "
             "0 disables compression and plays the recording at its real pace",
    )
    ap.add_argument(
        "--timing-out",
        type=Path,
        default=None,
        help="also write a JSON sidecar mapping recording time to video time, "
             "for overlays that must land on the frame where a line appeared",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="uniform playback scaling applied after gap compression (2.0 = twice as fast)",
    )
    args = ap.parse_args()

    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    if args.max_gap < 0:
        raise SystemExit("--max-gap must not be negative (use 0 to disable compression)")

    header, events = load_events(args.cast)
    cols, rows = header["width"], header["height"]
    if not events:
        raise SystemExit("recording contains no output events")

    # Measured before the remap, since the remap is what destroys the evidence.
    real_duration = events[-1][0]
    events, gaps = compress_timeline(events, args.max_gap, args.speed)
    removed = sum(g.real - args.max_gap for g in gaps)

    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    renderer = Renderer(cols, rows, args.font_size, (args.width, args.height))

    duration = events[-1][0] + args.tail
    n_frames = max(1, int(duration * args.fps))
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{renderer.width}x{renderer.height}", "-r", str(args.fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        # yuv420p plus faststart is the combination that plays in browsers and
        # in QuickTime; libx264's default yuv444p plays in neither.
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    idx = 0
    gap_idx = 0
    prev_key = None
    base_img: Image.Image | None = None
    plain_bytes = b""
    warned_alt_screen = False
    try:
        for f in range(n_frames):
            now = f / args.fps
            while idx < len(events) and events[idx][0] <= now:
                data = events[idx][1]
                if not warned_alt_screen and any(m in data for m in ALT_SCREEN_MARKERS):
                    print(
                        "warning: recording switches to the alternate screen; pyte ignores "
                        "that, so TUI output will mix into the scrollback",
                        file=sys.stderr,
                    )
                    warned_alt_screen = True
                # A single malformed or unsupported sequence must not take the
                # whole render down: the frames already encoded are still real,
                # and dropping one chunk is a smaller lie than no video at all.
                try:
                    stream.feed(data.encode("utf-8"))
                except Exception as exc:  # noqa: BLE001 - emulator robustness
                    print(f"warning: emulator rejected a chunk at t={events[idx][0]:.2f}s: "
                          f"{exc}", file=sys.stderr)
                idx += 1
            key = snapshot(screen)
            if key != prev_key or base_img is None:
                base_img = renderer.draw(*key)
                plain_bytes = base_img.tobytes()
                prev_key = key
            # Gaps are disjoint and in order, so a single advancing cursor finds
            # the one covering this frame without rescanning the list.
            while gap_idx < len(gaps) and now >= gaps[gap_idx].end:
                gap_idx += 1
            in_gap = gap_idx < len(gaps) and gaps[gap_idx].start <= now < gaps[gap_idx].end
            if in_gap:
                frame_bytes = renderer.with_marker(
                    base_img, renderer.marker_text(gaps[gap_idx].real)
                ).tobytes()
            else:
                frame_bytes = plain_bytes
            proc.stdin.write(frame_bytes)
            if f % (args.fps * 5) == 0:
                print(f"  frame {f}/{n_frames}", file=sys.stderr)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")

    if args.timing_out is not None:
        doc = timing_sidecar(
            cast=args.cast,
            out=args.out,
            gaps=gaps,
            fps=args.fps,
            max_gap=args.max_gap,
            speed=args.speed,
            tail=args.tail,
            total_cast_seconds=real_duration,
            total_video_seconds=duration,
            n_frames=n_frames,
        )
        args.timing_out.parent.mkdir(parents=True, exist_ok=True)
        args.timing_out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.timing_out}: {len(doc['breakpoints'])} breakpoints, "
              f"{len(doc['gaps'])} gap(s)")

    named = sorted(c for c in renderer.colors_seen if c in PALETTE)
    hexed = sorted(c for c in renderer.colors_seen if c not in PALETTE)
    print(
        f"wrote {args.out}: {n_frames} frames @ {args.fps}fps, "
        f"{renderer.width}x{renderer.height}, grid {cols}x{rows}, "
        f"cell {renderer.cell_w}x{renderer.cell_h}px"
    )
    # State the edit plainly.  Anyone handed this video should be able to see,
    # from the command that made it, exactly how much waiting was taken out.
    print(
        f"timeline: {format_duration(real_duration)} real -> "
        f"{format_duration(duration)} rendered; "
        f"{len(gaps)} gap(s) compressed at --max-gap {args.max_gap:g}s, "
        f"{format_duration(removed)} of wall time removed; --speed {args.speed:g}"
    )
    if gaps:
        print(f"gap marker drawn as: {renderer.marker_text(gaps[0].real)!r}")
    print(f"named ANSI colours rendered: {named}")
    print(f"256/truecolour values rendered: {hexed}")
    # A recording of a real coloured terminal that came out entirely in the
    # default foreground means the colour path is broken, and a silently
    # monochrome video is exactly the kind of thing nobody notices in review.
    if not renderer.colors_seen:
        raise SystemExit("no non-default colour was ever rendered -- ANSI colour is being lost")


if __name__ == "__main__":
    main()
