#!/usr/bin/env python3
"""Flip through the captured agent corpus, fast, one real instance per card.

This is the "here is the traffic we serve" beat of the demo.  Each card is
one real captured agent trajectory.  Cards are keyed on the opening task
instruction, which is generated per-seed by ``phrasing.py`` from a typed
``Brief`` — 200 seeds × 12 families gives 2,400 distinct prompts, so the
montage shows genuine variety at the prompt level.

Each card carries:

  * its ``instance_id`` and the run it came from — unique by construction;
  * the opening task instruction, shown verbatim and prominently (the star
    of the card);
  * its ``metadata`` (the planted bug module, the traced answer, the window
    size, the log length, the host count) — the generator's own per-instance
    parameters, printed verbatim;
  * its real tool-call trace, name plus arguments plus the first line of the
    result the sandbox returned, taken from ``turns[*].tool_calls``;
  * its ``submitted_summary`` when the agent submitted one;
  * and, streaming on the right, one real assistant turn, revealed over that
    turn's own recorded ``latency_seconds``.

Nothing is paraphrased and nothing is composed: every string on screen is read
out of a trajectory JSON.  A ``signature`` guard in ``main()`` asserts that no
two selected cards render identical text, and a separate ``prompt`` guard
asserts that each selected card carries a distinct opening instruction — the
property that phrasing.py was fixed to provide.

The tallies are the point of the segment.  They count what the cards represent
(instances, request records, generated tokens, recorded generation seconds) and
they sit next to the corpus totals so the sample is never mistaken for the
whole.  The coverage strip at the bottom draws one cell per captured instance,
all in the corpus, and lights the ones the montage actually plays.

``--speed`` scales only the replay clock, so the recorded seconds stay legible
as recorded seconds while the video runs many times faster; the badge is on
every frame for that reason.

Usage::

    python demo/montage.py \\
        --corpus /data/.../bigcorpus-run1/traffic/trajectories \\
        --corpus /data/.../agentenv-qwen8b-run5/traffic/trajectories \\
        --traces /data/.../bigcorpus-run1/speedlm_home/traces/traces.jsonl \\
        --out    /data/.../demo-versions/traffic-montage-v3.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
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
    encode,
    font,
    wrap_tail,
)

# Cards per family.  Twelve families, so this is the montage's breadth knob.
CARDS_PER_FAMILY = 3
# Where the replay should land, in seconds of video, before the held cards.
REPLAY_SECONDS = 11.0
# Replay speed is derived from REPLAY_SECONDS and clamped into this band, so a
# corpus with unusually long or short turns still produces a watchable clip.
SPEED_MIN, SPEED_MAX = 8.0, 20.0
# Static holds.
INTRO_SECONDS = 2.6
SUMMARY_SECONDS = 2.8
TAIL_SECONDS = 0.6

# All twelve task families, original six + v2 six.  Order determines the
# interleaving of cards in the replay and the row layout in the coverage strip.
FAMILY_ORDER = [
    "bugfix-localize",
    "call-chain-trace",
    "feature-implement",
    "log-triage",
    "refactor-rename",
    "schema-migrate",
    "api-contract-drift",
    "config-precedence-bug",
    "dep-version-conflict",
    "error-swallow-audit",
    "flaky-test-quarantine",
    "perf-hotspot",
]

# Family metadata keys that are the generator's per-instance parameters.  The
# seed is dropped from the card line because it names the draw, not the task.
FACT_KEYS = {
    "planted_bug_module": "planted bug in",
    "answer": "true answer",
    "stages": "stages",
    "window": "window",
    "log_lines": "log lines",
    "hosts": "hosts",
}

# Rows reserved at the top of the body pane for the task instruction.
PROMPT_ROWS = 5


# -- reading the corpus ----------------------------------------------------


@dataclass
class Instance:
    """One captured agent trajectory, reduced to what a card can show."""

    run: str
    instance_id: str
    family: str
    prompt: str
    facts: list[str]
    solved: bool
    stop_condition: str
    turn_count: int
    tokens: int
    wall_clock: float
    tool_lines: list[tuple[str, str]]
    summary: str
    response: str
    latency: float
    prompt_tokens: int

    # Filled in when the instance is selected for the replay.
    start: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.latency

    @property
    def signature(self) -> str:
        """What the card actually shows, hashed, so duplicates can be refused."""
        body = "\n".join(
            [
                self.instance_id,
                self.prompt,
                *self.facts,
                self.summary,
                *(a + b for a, b in self.tool_lines),
            ]
        )
        return hashlib.md5(body.encode()).hexdigest()


def compact_args(name: str, raw: str) -> str:
    """Shorten a recorded ``arguments_json`` to the part worth reading."""
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return raw[:60]
    if not isinstance(args, dict):
        return str(args)[:60]
    parts: list[str] = []
    for key in ("path", "pattern", "find", "summary", "content"):
        if key in args and isinstance(args[key], str):
            value = " ".join(args[key].split())
            if key in ("find", "content", "summary"):
                value = f"{key}={value}"
            parts.append(value)
    if not parts:
        parts = [f"{k}={v}" for k, v in args.items()]
    return "  ".join(parts)


def clip(text: str, columns: int) -> str:
    """Cut a line to the pane and say so, rather than ending mid-word."""
    if len(text) <= columns:
        return text
    return text[: max(0, columns - 2)] + " …"


def first_line(text: str) -> str:
    for line in (text or "").split("\n"):
        if line.strip():
            return " ".join(line.split())
    return ""


def read_instance(path: Path, run: str) -> Instance | None:
    data = json.loads(path.read_text())
    turns = data.get("turns") or []
    if not turns:
        return None

    # First user message — the task instruction the card shows as its star.
    prompt = ""
    for message in data.get("messages", []):
        if message.get("role") == "user":
            prompt = " ".join((message.get("content") or "").split())
            break

    facts = []
    for key, label in FACT_KEYS.items():
        if key in (data.get("metadata") or {}):
            facts.append(f"{label} {data['metadata'][key]}")

    tool_lines: list[tuple[str, str]] = []
    for turn in turns:
        for call in turn.get("tool_calls") or []:
            head = f"{call['name']}  {compact_args(call['name'], call.get('arguments_json', ''))}"
            mark = "->" if call.get("ok") else "!!"
            tool_lines.append((head, f"{mark} {first_line(call.get('result_text', ''))}"))

    # The turn the card streams: the longest recorded generation of the
    # trajectory.  Picked for legibility -- it is still one real turn, played
    # over its own real latency -- and stated as such on the card.
    turn = max(turns, key=lambda t: t.get("latency_seconds", 0.0))
    if not (turn.get("content") or "").strip():
        return None

    return Instance(
        run=run,
        instance_id=data["instance_id"],
        family=data["family"],
        prompt=prompt,
        facts=facts,
        solved=bool((data.get("grade") or {}).get("solved")),
        stop_condition=data.get("stop_condition", ""),
        turn_count=len(turns),
        tokens=sum(t.get("completion_tokens", 0) for t in turns),
        wall_clock=float(data.get("wall_clock_seconds", 0.0)),
        tool_lines=tool_lines,
        summary=" ".join((data.get("submitted_summary") or "").split()),
        response=turn["content"],
        latency=float(turn.get("latency_seconds", 0.0)),
        prompt_tokens=int(turn.get("prompt_tokens", 0)),
    )


@dataclass
class Corpus:
    instances: list[Instance] = field(default_factory=list)
    records: int = 0

    @property
    def tokens(self) -> int:
        return sum(i.tokens for i in self.instances)

    @property
    def wall_clock(self) -> float:
        return sum(i.wall_clock for i in self.instances)

    def by_family(self) -> dict[str, list[Instance]]:
        out: dict[str, list[Instance]] = {}
        for instance in self.instances:
            out.setdefault(instance.family, []).append(instance)
        return out


def load_corpus(roots: Sequence[Path]) -> Corpus:
    corpus = Corpus()
    for root in roots:
        run = root.parent.parent.name
        for path in sorted(root.glob("*.json")):
            instance = read_instance(path, run)
            if instance is not None:
                corpus.instances.append(instance)
                corpus.records += instance.turn_count
    return corpus


def select(corpus: Corpus, per_family: int) -> list[Instance]:
    """Pick cards: spread across families and runs, every card visibly distinct.

    Instances are ordered by how close their streamed turn is to the corpus
    median generation length, so no single card eats a third of the replay,
    then filtered so that no two selected cards render the same text.

    Interleaving spreads across all families present (up to 12) and across
    capture runs before taking a second card from any one run.
    """
    latencies = sorted(i.latency for i in corpus.instances)
    target = latencies[len(latencies) // 2] if latencies else 0.0

    # Use the canonical order, restricted to families that actually appear.
    present = set(corpus.by_family().keys())
    families = [f for f in FAMILY_ORDER if f in present]

    chosen: list[Instance] = []
    seen: set[str] = set()
    for family in families:
        pool = sorted(
            corpus.by_family().get(family, []),
            key=lambda i: (abs(i.latency - target), i.instance_id),
        )
        taken = 0
        runs_used: list[str] = []
        for instance in pool:
            if taken >= per_family:
                break
            if instance.signature in seen or not instance.tool_lines:
                continue
            # Prefer covering both capture runs before taking a second card
            # from a run we already have.
            if (
                taken
                and instance.run in runs_used
                and len({i.run for i in pool}) > 1
                and any(p.run not in runs_used and p.signature not in seen for p in pool)
            ):
                continue
            seen.add(instance.signature)
            runs_used.append(instance.run)
            chosen.append(instance)
            taken += 1

    # Interleave families so consecutive cards never repeat a family.
    buckets = {f: [i for i in chosen if i.family == f] for f in families}
    ordered: list[Instance] = []
    for rank in range(per_family):
        for family in families:
            if rank < len(buckets[family]):
                ordered.append(buckets[family][rank])

    clock = 0.0
    for instance in ordered:
        instance.start = clock
        clock += instance.latency
    return ordered


# -- the renderer ----------------------------------------------------------


class Montage:
    def __init__(
        self,
        corpus: Corpus,
        cards: list[Instance],
        speed: float,
        model: str,
        traces: int,
    ) -> None:
        self.corpus = corpus
        self.cards = cards
        self.speed = speed
        self.model = model
        self.traces = traces
        self.duration = sum(c.latency for c in cards)

        self.f_title = font(38, bold=True)
        self.f_sub = font(19)
        self.f_tally = font(52, bold=True)
        self.f_tally_label = font(16)
        self.f_head = font(20, bold=True)
        self.f_meta = font(16)
        self.f_body = font(17)
        self.f_mono = font(15)
        self.f_stat = font(21, bold=True)
        self.f_big = font(66, bold=True)
        self.f_card = font(25)

        self.tally_top = 148
        self.pane_top = 288
        self.pane_bottom = 856
        self.left_x, self.left_w = 40, 880
        self.right_x = self.left_x + self.left_w + 28
        self.right_w = WIDTH - 40 - self.right_x

        self.body_top = self.pane_top + 128
        self.line_h = 21
        self.rows = max(1, (self.pane_bottom - 20 - self.body_top) // self.line_h)
        # DejaVu Sans Mono advance is 0.602 em.
        self.left_cols = int((self.left_w - 44) / (15 * 0.602))
        self.right_cols = int((self.right_w - 44) / (17 * 0.602))

        self.grid_top = 878
        self.grid_index = {(i.run, i.instance_id): n for n, i in enumerate(corpus.instances)}

    # -- chrome ------------------------------------------------------------

    def frame(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        return image, ImageDraw.Draw(image)

    def header(self, draw: ImageDraw.ImageDraw, subtitle: str) -> None:
        draw.text((40, 36), "SpeedLM  ·  the traffic we serve", font=self.f_title, fill=TEXT)
        draw.text((42, 90), subtitle, font=self.f_sub, fill=MUTED)
        draw.line((40, 128, WIDTH - 40, 128), fill=RULE, width=1)

    def footer(self, draw: ImageDraw.ImageDraw) -> None:
        draw.line((40, HEIGHT - 74, WIDTH - 40, HEIGHT - 74), fill=RULE, width=1)
        draw.text((42, HEIGHT - 58), self.provenance(), font=self.f_body, fill=DIM)

    def provenance(self) -> str:
        return (
            f"{self.model}  ·  {len(self.corpus.instances)} agent trajectories  ·  "
            f"{self.traces:,} request records in traces.jsonl  ·  "
            "every string on screen is read from a recorded trajectory"
        )

    def speed_badge(self, draw: ImageDraw.ImageDraw) -> None:
        """Stamp the replay clock's scale on every frame.

        The tallies are the recording's own seconds and token counts.  Without
        this a 200s timer emptying in 13s of video reads as a bug.
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
        shown: int,
        records: int,
        tokens: int,
        elapsed: float,
        fraction: float,
    ) -> None:
        top = self.tally_top
        draw.rectangle((40, top, WIDTH - 40, top + 116), fill=PANEL)
        cells = [
            ("instances shown", f"{shown}", f"of {len(self.corpus.instances)}", TEXT),
            ("records represented", f"{records:,}", f"of {self.traces:,}", TEXT),
            ("tokens generated", f"{tokens:,}", f"of {self.corpus.tokens:,}", TUNED_ACCENT),
            (
                "generation time (real)",
                f"{elapsed:.1f}s",
                f"of {self.corpus.wall_clock / 3600:.1f}h",
                TEXT,
            ),
        ]
        x = 72
        for label, value, note, colour in cells:
            draw.text((x, top + 14), label, font=self.f_tally_label, fill=DIM)
            draw.text((x, top + 34), value, font=self.f_tally, fill=colour)
            draw.text((x, top + 92), note, font=self.f_tally_label, fill=DIM)
            x += 450
        bar_y = top + 108
        draw.rectangle((40, bar_y, WIDTH - 40, bar_y + 5), fill=RULE)
        if fraction > 0:
            draw.rectangle(
                (40, bar_y, 40 + int((WIDTH - 80) * fraction), bar_y + 5),
                fill=TUNED_ACCENT,
            )

    # -- the two columns ---------------------------------------------------

    def instance_pane(self, draw: ImageDraw.ImageDraw, card: Instance) -> None:
        x, w = self.left_x, self.left_w
        draw.rectangle((x, self.pane_top, x + w, self.pane_bottom), fill=PANEL)
        draw.rectangle((x, self.pane_top, x + w, self.pane_top + 4), fill=STOCK_ACCENT)

        draw.text(
            (x + 22, self.pane_top + 20),
            card.instance_id.upper(),
            font=self.f_head,
            fill=STOCK_ACCENT,
        )
        draw.text(
            (x + 22, self.pane_top + 48),
            f"{card.family} · {card.run} · "
            f"{card.turn_count} requests · {card.tokens:,} tok · "
            f"{card.wall_clock:.0f}s wall",
            font=self.f_meta,
            fill=DIM,
        )
        facts = "   ·   ".join(card.facts) if card.facts else "no generator parameters recorded"
        verdict = "graded solved" if card.solved else f"graded unsolved ({card.stop_condition})"
        draw.text(
            (x + 22, self.pane_top + 74),
            f"{facts}   ·   {verdict}",
            font=self.f_meta,
            fill=MUTED if card.solved else STOCK_ACCENT,
        )

        # ── Prompt section — the star ──────────────────────────────────────
        draw.text(
            (x + 22, self.body_top - 26),
            "TASK INSTRUCTION  ·  verbatim first user message",
            font=self.f_meta,
            fill=MUTED,
        )
        prompt_lines = textwrap.wrap(card.prompt, width=self.left_cols)
        shown_prompt = prompt_lines[:PROMPT_ROWS]
        y = self.body_top
        for line in shown_prompt:
            draw.text((x + 22, y), clip(line, self.left_cols), font=self.f_body, fill=TEXT)
            y += self.line_h
        # Pad to a fixed block so tool-call section starts at a stable y.
        y += (PROMPT_ROWS - len(shown_prompt)) * self.line_h

        # ── Tool call trace — supporting detail ───────────────────────────
        y += 4  # small breathing room between the two sections
        draw.text(
            (x + 22, y),
            "TOOL CALLS  ·  recorded trace and sandbox replies",
            font=self.f_meta,
            fill=MUTED,
        )
        y += self.line_h + 2

        summary_rows = 0
        summary_lines: list[str] = []
        if card.summary:
            summary_lines = textwrap.wrap("submitted: " + card.summary, width=self.left_cols)[:3]
            summary_rows = len(summary_lines) + 1

        # Pixel budget remaining for tool calls.
        tool_budget_px = self.pane_bottom - 20 - y - summary_rows * self.line_h
        tool_budget = max(0, tool_budget_px // self.line_h)

        tool_y = y
        drawn = 0
        for head, result in card.tool_lines:
            if drawn >= tool_budget - 2:
                left = len(card.tool_lines) - (drawn // 2)
                draw.text(
                    (x + 22, tool_y),
                    f"... {left} more recorded tool calls",
                    font=self.f_mono,
                    fill=DIM,
                )
                drawn += 1
                break
            draw.text((x + 22, tool_y), clip(head, self.left_cols), font=self.f_mono, fill=TEXT)
            tool_y += self.line_h
            draw.text(
                (x + 40, tool_y),
                clip(result, self.left_cols - 2),
                font=self.f_mono,
                fill=TUNED_ACCENT if result.startswith("->") else STOCK_ACCENT,
            )
            tool_y += self.line_h
            drawn += 2

        if summary_lines:
            sum_y = self.pane_bottom - 20 - summary_rows * self.line_h
            for line in summary_lines:
                draw.text((x + 22, sum_y), line, font=self.f_mono, fill=MUTED)
                sum_y += self.line_h

    def response_pane(self, draw: ImageDraw.ImageDraw, card: Instance, revealed: str) -> None:
        x, w = self.right_x, self.right_w
        draw.rectangle((x, self.pane_top, x + w, self.pane_bottom), fill=PANEL)
        draw.rectangle((x, self.pane_top, x + w, self.pane_top + 4), fill=TUNED_ACCENT)

        draw.text(
            (x + 22, self.pane_top + 20), "MODEL RESPONSE", font=self.f_head, fill=TUNED_ACCENT
        )
        draw.text(
            (x + 22, self.pane_top + 48),
            f"longest recorded turn  ·  {card.prompt_tokens:,} tok in  ·  "
            f"{card.latency:.2f}s on the wall",
            font=self.f_meta,
            fill=DIM,
        )
        draw.text(
            (x + 22, self.body_top - 26),
            "STREAMING AT ITS RECORDED LATENCY",
            font=self.f_meta,
            fill=MUTED,
        )

        y = self.body_top
        for line in wrap_tail(revealed, self.right_cols, self.rows):
            draw.text((x + 22, y), line, font=self.f_body, fill=TEXT)
            y += self.line_h

    # -- coverage strip ----------------------------------------------------

    def coverage(self, draw: ImageDraw.ImageDraw, lit: set[int]) -> None:
        top = self.grid_top
        by_family = self.corpus.by_family()
        draw.text(
            (40, top - 22),
            f"CAPTURED CORPUS  ·  one cell per trajectory  ·  "
            f"{len(self.corpus.instances)} instances, "
            f"{len(self.cards)} replayed here",
            font=self.f_meta,
            fill=MUTED,
        )
        widest = max((len(v) for v in by_family.values()), default=1)
        label_w = 172
        cell = max(4, min(14, (WIDTH - 80 - label_w) // widest - 2))
        gap = 2
        row_h = 15
        y = top
        for family in FAMILY_ORDER:
            members = by_family.get(family, [])
            if not members:
                continue
            draw.text((40, y - 1), family, font=self.f_tally_label, fill=DIM)
            x = 40 + label_w
            for instance in members:
                index = self.grid_index[(instance.run, instance.instance_id)]
                on = index in lit
                draw.rectangle((x, y, x + cell, y + row_h - 5), fill=TUNED_ACCENT if on else RULE)
                x += cell + gap
            y += row_h

    # -- cards -------------------------------------------------------------

    def intro(self) -> Image.Image:
        """Intro card: corpus scale and family breakdown."""
        image, draw = self.frame()
        by_family = self.corpus.by_family()
        families_present = [f for f in FAMILY_ORDER if f in by_family]
        subtitle = (
            f"{len(self.corpus.instances)} instances  ·  "
            f"{len(families_present)} task families  ·  "
            "each instance carries a distinct instruction"
        )
        self.header(draw, subtitle)

        y = 172
        draw.text(
            (40, y),
            f"{len(self.corpus.instances)} captured agent trajectories across "
            f"{len(families_present)} task families.",
            font=self.f_card,
            fill=TEXT,
        )
        draw.text(
            (40, y + 40),
            "Instructions are generated per-seed by phrasing.py — every instance "
            "opens with a different prompt.",
            font=self.f_card,
            fill=MUTED,
        )
        y += 100
        draw.line((40, y, WIDTH - 40, y), fill=RULE, width=1)
        y += 16

        row_h = 38
        for family in families_present:
            count = len(by_family[family])
            draw.text((40, y), family, font=self.f_head, fill=STOCK_ACCENT)
            draw.text((340, y), f"{count} instances", font=self.f_body, fill=TEXT)
            # Show a fragment of the first prompt in this family to hint at variety.
            sample = by_family[family][0].prompt
            sample_short = clip(" ".join(sample.split()), self.right_cols + 20)
            draw.text((540, y), sample_short, font=self.f_mono, fill=DIM)
            y += row_h
            if y > self.pane_bottom:
                break

        self.footer(draw)
        self.speed_badge(draw)
        return image

    def summary(self) -> Image.Image:
        image, draw = self.frame()
        self.header(draw, "what that traffic adds up to")

        shown = len(self.cards)
        by_family = self.corpus.by_family()
        families_present = [f for f in FAMILY_ORDER if f in by_family]
        cells = [
            (120, f"{len(self.corpus.instances)}", "agent trajectories", TEXT),
            (560, f"{self.traces:,}", "request records", TEXT),
            (1020, f"{self.corpus.tokens:,}", "tokens generated", TUNED_ACCENT),
            (1500, f"{self.corpus.wall_clock / 3600:.1f}h", "of agent wall clock", TEXT),
        ]
        for x, value, label, colour in cells:
            draw.text((x, 216), value, font=self.f_big, fill=colour)
            draw.text((x, 300), label, font=self.f_card, fill=MUTED)

        y = 410
        draw.line((120, y - 24, WIDTH - 120, y - 24), fill=RULE, width=1)
        counts = {f: len(v) for f, v in by_family.items()}
        family_line = "   ·   ".join(f"{f} {counts.get(f, 0)}" for f in families_present)
        for line in textwrap.wrap(family_line, width=98) or [""]:
            draw.text((120, y), line, font=self.f_card, fill=MUTED)
            y += 44
        y += 16
        for text, colour in (
            (f"{shown} of those instances played above, one card each, none repeated.", TEXT),
            (
                "Every instance opened with a distinct instruction generated from phrasing.py.",
                TEXT,
            ),
            ("Variety comes from the prompt and the workspace, not just the tool calls.", MUTED),
            ("", TEXT),
            ("All of it is recorded as traces. The idle cycle trains the draft head", MUTED),
            ("on this traffic, and the gate replays a session-disjoint slice of it.", TUNED_ACCENT),
        ):
            draw.text((120, y), text, font=self.f_card, fill=colour)
            y += 44

        self.footer(draw)
        self.speed_badge(draw)
        return image

    # -- the replay --------------------------------------------------------

    def replay_frames(self) -> Iterator[Image.Image]:
        subtitle = (
            f"{len(self.cards)} instances drawn from {len(self.corpus.instances)} "
            f"captured trajectories  ·  {self.model}  ·  greedy"
        )
        frames = int(self.duration / self.speed * FPS)
        lit: set[int] = set()
        index = 0
        for i in range(frames + int(TAIL_SECONDS * FPS)):
            t = min(i / FPS * self.speed, self.duration)
            while index + 1 < len(self.cards) and t >= self.cards[index].end:
                index += 1
            card = self.cards[index]
            lit.add(self.grid_index[(card.run, card.instance_id)])

            progress = 1.0
            if card.latency > 0:
                progress = min(1.0, max(0.0, (t - card.start) / card.latency))
            revealed = card.response[: max(1, int(len(card.response) * progress))]

            done = self.cards[: index + 1]
            image, draw = self.frame()
            self.header(draw, subtitle)
            self.tally(
                draw,
                shown=len(done),
                records=sum(c.turn_count for c in done),
                tokens=sum(c.tokens for c in done),
                elapsed=t,
                fraction=t / self.duration if self.duration else 0.0,
            )
            self.instance_pane(draw, card)
            self.response_pane(draw, card, revealed)
            self.coverage(draw, lit)
            self.footer(draw)
            self.speed_badge(draw)
            yield image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        required=True,
        help="a traffic/trajectories directory; repeatable",
    )
    parser.add_argument(
        "--traces", type=Path, default=None, help="traces.jsonl, counted for the record tally"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--per-family", type=int, default=CARDS_PER_FAMILY)
    parser.add_argument("--replay-seconds", type=float, default=REPLAY_SECONDS)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="replay clock scale; 0 derives it from --replay-seconds",
    )
    parser.add_argument(
        "--frames-only",
        type=int,
        default=0,
        help="render every Nth replay frame as a PNG instead of a video",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    if not corpus.instances:
        raise SystemExit(f"no trajectories under {args.corpus}")

    records = corpus.records
    if args.traces:
        records = sum(1 for line in args.traces.read_text().splitlines() if line.strip())

    cards = select(corpus, args.per_family)

    families_present = sorted(set(corpus.by_family().keys()))
    wanted = args.per_family * len(families_present)
    if len(cards) < wanted:
        raise SystemExit(
            f"only {len(cards)} distinct cards available, wanted "
            f"{wanted} ({args.per_family} × {len(families_present)} families)"
        )

    # Safety net 1: no two cards may render identical overall content.
    signatures = {c.signature for c in cards}
    if len(signatures) != len(cards):
        raise SystemExit("selected cards are not all distinct (signature collision)")

    # Safety net 2: every card must carry a distinct opening prompt.  This is
    # the property that phrasing.py was fixed to provide; it should pass
    # trivially against a v2 corpus and FAIL against a v1 corpus (where all
    # instances of a family share a byte-identical first user message).
    selected_prompts = [c.prompt for c in cards]
    if len(set(selected_prompts)) != len(cards):
        duplicates = len(cards) - len(set(selected_prompts))
        raise SystemExit(
            f"{duplicates} selected card(s) share a prompt with another selected card — "
            f"corpus may predate phrasing.py variant instructions "
            f"({len(set(selected_prompts))} unique prompts out of {len(cards)} cards)"
        )

    duration = sum(c.latency for c in cards)
    speed = args.speed or duration / args.replay_seconds
    speed = min(SPEED_MAX, max(SPEED_MIN, round(speed, 1)))

    montage = Montage(corpus, cards, speed, args.model, records)
    video = INTRO_SECONDS + duration / speed + TAIL_SECONDS + SUMMARY_SECONDS
    print(
        f"{len(corpus.instances)} trajectories  {records} records  "
        f"{corpus.tokens:,} tokens  {corpus.wall_clock / 3600:.2f}h wall clock\n"
        f"{len(cards)} cards  {duration:.1f}s recorded generation  "
        f"at {speed:g}x  ->  {video:.1f}s of video"
    )
    for card in cards:
        print(
            f"  {card.run:24s} {card.instance_id:26s} {card.latency:5.1f}s "
            f"{len(card.tool_lines):2d} tool calls"
        )

    if args.frames_only:
        out_dir = args.out.parent / "montage-frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        montage.intro().save(out_dir / "intro.png")
        montage.summary().save(out_dir / "summary.png")
        for i, image in enumerate(montage.replay_frames()):
            if i % args.frames_only == 0:
                image.save(out_dir / f"montage-{i:05d}.png")
        print(f"wrote sample frames to {out_dir}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encode(
        itertools.chain(
            itertools.repeat(montage.intro(), int(INTRO_SECONDS * FPS)),
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
