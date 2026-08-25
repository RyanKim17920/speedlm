#!/usr/bin/env python3
"""Replay a real run log as an asciinema cast, at the pace the run produced it.

Nothing is added. Every byte written to the cast is a line that the running
system actually printed. Timing comes from the log's own timestamps, so the
playback rate is the rate the events really happened at, not a rate chosen
here. Long idle gaps are capped (``--max-gap``) because a server waiting for
the idle threshold emits nothing for minutes, and a video of nothing is not
useful footage -- the cap is the one liberty taken, and it is reported.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# vLLM lines look like: "INFO 08-23 14:39:23 [api_utils.py:339] ..."
STAMP = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")


def _seconds(line: str) -> float | None:
    m = STAMP.search(line)
    if not m:
        return None
    _mo, day, hh, mm, ss = (int(g) for g in m.groups())
    return day * 86400.0 + hh * 3600.0 + mm * 60.0 + ss


def build(
    src: Path,
    out: Path,
    *,
    cols: int,
    rows: int,
    max_gap: float,
    line_interval: float = 0.08,
) -> dict:
    lines = src.read_text(errors="replace").splitlines()
    stamps: list[float | None] = [_seconds(ln) for ln in lines]

    # Carry the last known stamp forward so unstamped continuation lines land
    # with the line they belong to rather than at time zero.
    carried: list[float] = []
    last = None
    for s in stamps:
        if s is not None:
            last = s
        carried.append(last if last is not None else 0.0)

    has_stamps = any(v is not None for v in stamps)
    base = carried[0] if carried else 0.0
    events: list[list] = []
    clock = 0.0
    prev = base
    capped = 0
    for ln, stamp in zip(lines, carried, strict=True):
        # No timestamps anywhere: fall back to an even, declared pace.
        delta = line_interval if not has_stamps else stamp - prev
        if delta < 0:
            delta = 0.0
        if delta > max_gap:
            capped += 1
            delta = max_gap
        clock += delta
        prev = stamp
        events.append([round(clock, 6), "o", ln + "\r\n"])

    header = {
        "version": 2,
        "width": cols,
        "height": rows,
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }
    with out.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return {
        "lines": len(lines),
        "seconds": round(clock, 2),
        "gaps_capped": capped,
        "timing": "log timestamps" if has_stamps else f"fallback {line_interval}s/line",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=34)
    ap.add_argument("--max-gap", type=float, default=1.5)
    ap.add_argument(
        "--line-interval",
        type=float,
        default=0.08,
        help=(
            "Fallback seconds per line, used ONLY when the log carries no "
            "timestamps at all. This is a chosen pace, not the run's own, and "
            "the tool reports when it was used."
        ),
    )
    args = ap.parse_args()
    stats = build(
        args.log,
        args.out,
        cols=args.cols,
        rows=args.rows,
        max_gap=args.max_gap,
        line_interval=args.line_interval,
    )
    print(
        f"{args.out}: {stats['lines']} lines, {stats['seconds']}s, "
        f"{stats['gaps_capped']} gaps capped, timing={stats['timing']}"
    )


if __name__ == "__main__":
    main()
