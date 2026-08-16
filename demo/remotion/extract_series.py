#!/usr/bin/env python3
"""Extract the real SpeedLM training/gate series that the Remotion demo charts.

Reads only real artifacts produced by a cycle run:

  <run_root>/speedlm_home/runs/*/training-logs/stdout.log   -> loss + per-position accuracy
  --decision <path to a gate decision.json>                 -> throughput / accepted length

and writes ``src/data.json`` next to this script.

The two are deliberately decoupled. The loss curve belongs to the run that did
the training (bigcycle-run1, 8 epochs over the 3758-record corpus); the gate
numbers belong to the run that measured that head honestly on held-out traffic
(regate-big-run2). See DEFAULT_DECISION.

Nothing is synthesised. If an artifact is missing the script fails loudly rather
than emitting placeholder numbers.

Each datum also carries the frame of the composited video at which the terminal
on the left actually printed it, so the chart on the right can reveal it at that
moment rather than at a guessed fraction of the runtime.  That anchor is read
off two more real artifacts:

  --cast    <recording>.cast       -> when the line hit the terminal
  --timing  <sidecar>.json         -> where that moment landed in the video

The cast is the authority for *when*, not the training log: the log records when
the trainer wrote a line, while the video shows when the terminal displayed it,
and in this recording those differ by minutes -- the session tails a log that is
already complete, so twenty-four steps' worth of output arrives in one burst.
Anchoring to the log would put points on screen before the terminal has said
anything about them.  The sidecar is required because the renderer compresses
dead air, so recording seconds and video seconds are not proportional; see
``demo/session_render.py --timing-out``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

DEFAULT_RUN_ROOTS = [
    # The fast cut reads bigcycle-run1's completed training log off disk: 3758
    # captured records, a 1024-record lease, 652 rendered rows, 8 epochs / 922
    # optimizer steps. Longer and more informative than the earlier 3-epoch runs.
    Path("/data/ryan.kim/speedlm-runs/bigcycle-run1"),
    Path("/data/ryan.kim/speedlm-runs/agentenv-qwen8b-run5"),
    Path("/data/ryan.kim/speedlm-runs/demo-cycle-run10"),
    Path("/data/ryan.kim/speedlm-runs/demo-cycle-run4"),
]

# The gate numbers do NOT come from the training run's own decision.json.
# bigcycle-run1's in-run gate was VETOED on a non-stationary throughput delta and
# the cycle rolled back, so charting it would caption a rollback as a promotion.
# regate-big-run2 is the DEFINITIVE measurement of that head: the same 287-context,
# session-disjoint held-out suite, but 8 scored repeats after 3 warmups on a
# less-contended node. It supersedes regate-big-run1 (5 repeats, heavily contended).
#
#   accepted length  2.3051 -> 2.6507   +0.3457 (SE 0.0029) = +15.0% per verifier step
#   acceptance rate                     +11.52pp (SE 0.10)
#   decode tok/s     124.59 -> 144.71   central +16.15%, per-repeat range +12.9%..+20.7%
#   final_verdict    reject, vetoed=true, reason throughput_not_stationary
#
# The veto is real and the video shows it. Its cause is the BASELINE, not the head:
# the candidate arm was flat from repeat 0 (+0.051%/repeat, 143-147 tok/s throughout)
# while stock drifted -0.821%/repeat (127 -> 121 tok/s) and only settled at repeat 5.
# So the throughput figure is presented as a caused RANGE, never as a settled point,
# and no promote verdict is attached to it anywhere in the cut.
#
# Figures that deliberately appear NOWHERE: run5's session-overlapping +0.653 /
# +32.25% (superseded by ~2.2x leakage inflation), bigcycle-run1's own vetoed in-run
# +14.76%, the earlier clean +0.2989 / +9.94% gate, and regate-big-run1's +19.91%.
DEFAULT_DECISION = Path("/data/ryan.kim/speedlm-runs/regate-big-run2/decision.json")

# Independent gates of the SAME head, in the order they were run. The strongest
# property of the headline is that it REPRODUCED: three separate measurements on
# the 287-context session-disjoint suite land within 0.006 tokens/step of each
# other. The chart says "reproduced Nx" off the length of this list and prints
# each delta, so the claim is read out of real records rather than typed in --
# if a record goes missing the count drops instead of the claim standing.
#
# Note what is deliberately NOT carried across from these earlier records: their
# throughput deltas (+14.76%, +19.91%) are superseded and appear nowhere.
_RUNS = Path("/data/ryan.kim/speedlm-runs")
DEFAULT_CORROBORATING_DECISIONS: list[tuple[str, Path]] = [
    (
        "bigcycle-run1",
        _RUNS
        / "bigcycle-run1/speedlm_home/runs"
        / "e7004c4c0c7548fba65b05a924aa57ea/decision.json",
    ),
    ("regate-big-run1", _RUNS / "regate-big-run1/decision.json"),
    ("regate-big-run2", DEFAULT_DECISION),
]

# The recording the composition composites, and the renderer's timing sidecar.
DEFAULT_CAST = Path("/data/ryan.kim/speedlm-runs/demo-fast/session_fast.cast")
DEFAULT_TIMING = Path("/data/ryan.kim/speedlm-runs/demo-fast/timing.json")

# The trainer logs through `rich`, which hard-wraps a single logical record over
# several physical lines and prefixes them with a timestamp / logger gutter. So we
# rebuild logical records first, then pull key=value pairs out of the joined text.
_RECORD_START = re.compile(r"(?:^|\s)(?:INFO|WARNING|ERROR)\s{2,}")
_GUTTER = re.compile(r"\s+\w+\.py:\d+\s*$")
_KV = re.compile(r"([A-Za-z][\w/]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _logical_records(text: str) -> list[str]:
    """Join rich's wrapped continuation lines back into one record per log call."""
    records: list[str] = []
    current: list[str] | None = None
    for raw in text.splitlines():
        line = _GUTTER.sub("", raw.rstrip())
        if not line.strip():
            continue
        match = _RECORD_START.search(line)
        if match:
            if current is not None:
                records.append(" ".join(current))
            current = [line[match.end() :].strip()]
        elif current is not None:
            # A progress-bar repaint is not a continuation of the previous record.
            if "━" in line or line.lstrip().startswith("Epoch "):
                records.append(" ".join(current))
                current = None
                continue
            current.append(line.strip())
    if current is not None:
        records.append(" ".join(current))
    return records


def _kv(record: str) -> dict[str, float]:
    return {k: float(v) for k, v in _KV.findall(record)}


def parse_training_log(path: Path) -> dict:
    records = _logical_records(path.read_text(errors="replace"))

    train: list[dict] = []
    val: list[dict] = []
    for record in records:
        if "train/loss=" in record:
            f = _kv(record)
            train.append(
                {
                    "step": int(f["global_step"]),
                    "epoch": int(f["epoch"]),
                    "loss": f["train/loss"],
                    "lr": f.get("lr"),
                    "full_acc": [f[f"train/full_acc_{i}"] for i in range(3)],
                    "cond_acc": [f[f"train/cond_acc_{i}"] for i in range(3)],
                }
            )
        elif "val/loss_epoch=" in record:
            f = _kv(record)
            val.append(
                {
                    "epoch": int(f["epoch"]),
                    "loss": f["val/loss_epoch"],
                    "full_acc": [f[f"val/full_acc_{i}_epoch"] for i in range(3)],
                    "cond_acc": [f[f"val/cond_acc_{i}_epoch"] for i in range(3)],
                }
            )

    if not train:
        raise SystemExit(f"no train/loss records parsed from {path}")
    if not val:
        raise SystemExit(f"no val/loss_epoch records parsed from {path}")

    # Attach the epoch's validation step index so the chart can place val points on
    # the same x axis as the training steps.
    steps_by_epoch: dict[int, int] = {}
    for point in train:
        steps_by_epoch[point["epoch"]] = max(steps_by_epoch.get(point["epoch"], 0), point["step"])
    for point in val:
        point["step"] = steps_by_epoch[point["epoch"]]

    return {"train": train, "val": val}


#: ``final_reason`` a vetoed record carries.  Mirrors
#: ``speedlm.gate.decide.VETO_REASON_NON_STATIONARY``.
NON_STATIONARY_REASON = "throughput_not_stationary"


def _vetoed(d: dict) -> bool:
    """Whether a post-decision veto overrode this record's promotion.

    Prefers the persisted answer.  Falls back to re-deriving it from the
    stationarity block, so records written before ``final_verdict`` existed --
    bigcycle-run1 among them -- still read as the rollbacks they were.
    """
    if "vetoed" in d:
        return bool(d["vetoed"])
    stationarity = d.get("throughput_stationarity")
    if not isinstance(stationarity, dict):
        return False
    return (
        d.get("verdict") == "promote"
        and bool(stationarity.get("required_for_promotion"))
        and stationarity.get("status") == "non_stationary"
    )


def _final_verdict(d: dict) -> str:
    return d.get("final_verdict") or ("reject" if _vetoed(d) else d["verdict"])


def _final_reason(d: dict) -> str | None:
    if d.get("final_reason"):
        return str(d["final_reason"])
    return NON_STATIONARY_REASON if _vetoed(d) else d.get("reason")


def parse_reproductions(entries: list[tuple[str, Path]]) -> list[dict]:
    """Read each corroborating gate's accepted-length delta off its own record.

    Records that are absent, or that never measured the delta, are skipped rather
    than guessed at -- a missing file must lower the reproduction count, never
    invent one.  Only same-suite measurements count, so a record scored on a
    different number of held-out contexts is dropped with a warning.
    """
    out: list[dict] = []
    for label, path in entries:
        if not path.is_file():
            print(f"warning: corroborating gate missing: {path}", file=sys.stderr)
            continue
        d = json.loads(path.read_text())
        delta = d.get("accepted_length_delta")
        if delta is None:
            print(f"warning: {label} records no accepted_length_delta", file=sys.stderr)
            continue
        out.append(
            {
                "run": label,
                "delta": delta,
                "standard_error": d.get("accepted_length_delta_standard_error"),
                "num_contexts": d.get("num_contexts"),
                "num_repeats": d.get("num_repeats"),
            }
        )
    suites = {r["num_contexts"] for r in out}
    if len(suites) > 1:
        raise SystemExit(
            "corroborating gates disagree on suite size "
            f"({sorted(suites)}); they are not reproductions of one measurement"
        )
    return out


def parse_decision(path: Path) -> dict:
    d = json.loads(path.read_text())
    required = [
        "stock_avg_tok_per_sec",
        "candidate_avg_tok_per_sec",
        "stock_avg_accepted_length",
        "candidate_avg_accepted_length",
        "verdict",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        raise SystemExit(f"{path} is missing required keys: {missing}")

    return {
        # The OUTCOME, not the threshold comparison.  ``verdict`` is what the
        # numbers said; a promotion the gate vetoed on a non-stationary
        # throughput delta still records ``verdict: promote`` while the cycle
        # rolled back, so charting that field captions a rollback as a
        # promotion.  Derived for archived records written before the gate
        # persisted the outcome; see ``_final_verdict``.
        "verdict": _final_verdict(d),
        "reason": _final_reason(d),
        "threshold_verdict": d["verdict"],
        "threshold_reason": d.get("reason"),
        "throughput": {
            "stock": d["stock_avg_tok_per_sec"],
            "tuned": d["candidate_avg_tok_per_sec"],
            "delta_pct": d.get("throughput_delta_pct"),
            "delta_standard_error_pct": d.get("throughput_delta_standard_error_pct"),
            # Whether the delta held still across repeats. On regate-big-run2 it
            # did NOT -- and the trend fields below say WHOSE fault that is: the
            # candidate arm is flat from repeat 0 while the stock baseline drifts
            # downward on a shared node. That is why the chart draws a caused
            # range rather than a point estimate -- see ThroughputChart.
            "stationary": (d.get("throughput_stationarity") or {}).get("stationary"),
            "stationarity_status": (d.get("throughput_stationarity") or {}).get("status"),
            "stock_trend_pct_per_repeat": d.get("stock_throughput_trend_pct_per_repeat"),
            "tuned_trend_pct_per_repeat": d.get("candidate_throughput_trend_pct_per_repeat"),
            "stock_flat_from_repeat": d.get("stock_throughput_flat_from_repeat"),
            "tuned_flat_from_repeat": d.get("candidate_throughput_flat_from_repeat"),
        },
        "accepted_length": {
            "stock": d["stock_avg_accepted_length"],
            "tuned": d["candidate_avg_accepted_length"],
            "delta": d.get("accepted_length_delta"),
            "delta_standard_error": d.get("accepted_length_delta_standard_error"),
        },
        "acceptance_rate": {
            "stock": d.get("stock_avg_acceptance"),
            "tuned": d.get("candidate_avg_acceptance"),
            "delta_pp": d.get("acceptance_delta_pp"),
            "delta_standard_error_pp": d.get("acceptance_delta_standard_error_pp"),
        },
        "per_repeat": [
            {
                "repeat": r["repeat_index"],
                "stock_tok_per_sec": r["stock_tok_per_sec"],
                "tuned_tok_per_sec": r["candidate_tok_per_sec"],
                "stock_accepted_length": r["stock_accepted_length"],
                "tuned_accepted_length": r["candidate_accepted_length"],
            }
            for r in d.get("per_repeat", [])
        ],
        "num_contexts": d.get("num_contexts"),
        "num_repeats": d.get("num_repeats"),
        "stock_draft": d.get("stock_draft"),
    }


# ---------------------------------------------------------------------------
# Anchoring: cast time -> video time -> frame
# ---------------------------------------------------------------------------

# CSI/OSC escapes carry colour and cursor movement, neither of which is text the
# viewer reads, so they are stripped before matching. Carriage returns are
# treated as newlines: a progress-bar repaint that overwrites a line cannot
# fabricate a match for a record that was never printed, and treating it as a
# break stops one repaint from gluing two lines into a false match.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
# How much already-seen text to keep in front of each new chunk when matching.
# The recording splits typed input a character at a time, so a pattern can
# straddle events; every pattern here is a single short line, so a few kilobytes
# of overlap is far more than enough to catch one.
_MATCH_WINDOW = 8192


def scan_cast(cast: Path, patterns: dict[str, re.Pattern[str]]) -> dict[str, float]:
    """Find, for each pattern, the recording time at which its text first appeared.

    Returns cast-clock seconds keyed the same way as ``patterns``; a pattern that
    never appears is simply absent, which the caller has to handle rather than
    paper over -- a missing anchor means the terminal never showed that number,
    and that is worth knowing.
    """
    pending = dict(patterns)
    found: dict[str, float] = {}
    buffer = ""
    with cast.open(encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        if header.get("version") != 2:
            raise SystemExit(f"{cast}: not an asciicast v2 file")
        for line in fh:
            line = line.strip()
            if not line or not pending:
                continue
            t, kind, data = json.loads(line)
            if kind != "o":
                continue
            buffer += _ANSI.sub("", data).replace("\r\n", "\n").replace("\r", "\n")
            buffer = buffer[-_MATCH_WINDOW:]
            for key, pattern in list(pending.items()):
                if pattern.search(buffer):
                    found[key] = float(t)
                    del pending[key]
    return found


def video_seconds(timing: dict, cast_t: float) -> float:
    """Map a recording timestamp onto the rendered video's clock.

    The sidecar states the mapping as piecewise-linear breakpoints; between two
    of them the clock runs at a constant rate, so straight-line interpolation
    reproduces the renderer's remap exactly.
    """
    bps = timing["breakpoints"]
    if cast_t <= bps[0][0]:
        return bps[0][1]
    for (c0, v0), (c1, v1) in zip(bps, bps[1:], strict=False):
        if cast_t <= c1:
            span = c1 - c0
            return v0 if span <= 0 else v0 + (v1 - v0) * (cast_t - c0) / span
    return bps[-1][1]


def frame_for(timing: dict, cast_t: float) -> int:
    """The first frame on which the moment is actually on screen.

    Ceiling, not rounding, because that is the renderer's own rule: it feeds the
    emulator every event whose time has arrived by ``frame / fps``, so a line
    landing at 2018.66 frames is first drawn on frame 2019. Rounding would put
    the overlay up to half a frame early -- i.e. sometimes a whole frame before
    the terminal has printed the thing being overlaid, which is the exact error
    this anchoring exists to remove.
    """
    frame = math.ceil(video_seconds(timing, cast_t) * timing["fps"])
    return max(0, min(int(timing["total_frames"]) - 1, frame))


def _num(value: float) -> str:
    """The literal text the session's awk printed for a metric (3 decimals)."""
    return re.escape(f"{value:.3f}")


def anchor_series(training: dict, gate: dict, cast: Path, timing: dict) -> dict:
    """Stamp every datum with the frame at which the terminal printed it.

    Two honesty details drive the shape of this:

    * The session prints only every third optimizer step, so most training
      points are never named on screen. Those inherit the anchor of the next
      point that *is* printed -- the first moment the terminal has caught up to
      them -- and are labelled ``inherited`` so the distinction survives into
      the chart's data rather than being lost here.
    * Trailing points past the last printed step inherit the final validation
      line, which is the terminal declaring the epoch that contains them done.
    """
    train, val = training["train"], training["val"]

    patterns: dict[str, re.Pattern[str]] = {}
    for i, p in enumerate(train):
        patterns[f"train:{i}"] = re.compile(
            rf"step\s+{p['step']}\s+train/loss\s*=\s*{_num(p['loss'])}"
        )
    for i, p in enumerate(val):
        # The cycle session printed "EPOCH 1 done val/loss_epoch = ..."; the fast
        # session (demo/session_fast.json) drops the "done". Make it optional so
        # one anchor pass works against either recording -- without this the val
        # diamonds silently fall back to inherited anchors on the fast cut.
        patterns[f"val:{i}"] = re.compile(
            rf"EPOCH\s+{p['epoch'] + 1}\s+(?:done\s+)?val/loss_epoch\s*=\s*{_num(p['loss'])}"
        )
    # Case-insensitive: the fast session prints the outcome as a lowercase
    # `gate verdict   reject` row inside the result block rather than as a
    # shouted `VERDICT REJECT` banner, because the verdict is no longer the
    # headline. The anchor still keys off the same field either way.
    patterns["gate"] = re.compile(rf"verdict\s+{re.escape(gate['verdict'])}", re.IGNORECASE)

    found = scan_cast(cast, patterns)

    # Reading order on screen: a validation line follows the training step that
    # closed its epoch, so ties break train-before-val.
    order = [("train", i, p["step"], 0) for i, p in enumerate(train)]
    order += [("val", i, p["step"], 1) for i, p in enumerate(val)]
    order.sort(key=lambda r: (r[2], r[3]))

    # Backwards fill: each unprinted point takes the next printed point's time.
    next_t: float | None = None
    resolved: dict[tuple[str, int], tuple[float, str]] = {}
    for kind, i, _, _ in reversed(order):
        exact = found.get(f"{kind}:{i}")
        if exact is not None:
            next_t = exact
            resolved[(kind, i)] = (exact, "terminal")
        elif next_t is not None:
            resolved[(kind, i)] = (next_t, "inherited")
    # Anything still unresolved sits past the last printed line; the last thing
    # the terminal said about training is the closest honest anchor it has.
    last_t = max((t for t, _ in resolved.values()), default=None)
    stats = {"terminal": 0, "inherited": 0, "unanchored": 0}
    for kind, i, _, _ in order:
        point = (train if kind == "train" else val)[i]
        hit = resolved.get((kind, i))
        if hit is None and last_t is not None:
            hit = (last_t, "inherited")
        if hit is None:
            stats["unanchored"] += 1
            continue
        cast_t, source = hit
        point["cast_t"] = round(cast_t, 3)
        point["frame"] = frame_for(timing, cast_t)
        point["anchor"] = source
        stats[source] += 1

    gate_t = found.get("gate")
    if gate_t is not None:
        gate["cast_t"] = round(gate_t, 3)
        gate["frame"] = frame_for(timing, gate_t)
        gate["anchor"] = "terminal"

    return {
        "cast": str(cast),
        "fps": timing["fps"],
        "total_frames": timing["total_frames"],
        "total_cast_seconds": timing["total_cast_seconds"],
        "anchored_terminal": stats["terminal"],
        "anchored_inherited": stats["inherited"],
        "unanchored": stats["unanchored"],
        "gate_frame": gate.get("frame"),
    }


def find_run(run_roots: list[Path]) -> tuple[Path, Path, Path]:
    for root in run_roots:
        runs_dir = root / "speedlm_home" / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            log = run_dir / "training-logs" / "stdout.log"
            decision = run_dir / "decision.json"
            if log.is_file() and decision.is_file():
                return root, log, decision
    raise SystemExit(
        "no run with both training-logs/stdout.log and decision.json under: "
        + ", ".join(str(p) for p in run_roots)
    )


def find_training_log(run_roots: list[Path]) -> tuple[Path, Path]:
    """The run root and training log to chart, ignoring its own decision.json.

    Kept separate from the gate artifact on purpose: the run that produced the
    loss curve is not necessarily the run that produced the trustworthy gate
    numbers, and on this demo it is not (see DEFAULT_DECISION).
    """
    for root in run_roots:
        runs_dir = root / "speedlm_home" / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            log = run_dir / "training-logs" / "stdout.log"
            if log.is_file():
                return root, log
    raise SystemExit(
        "no run with training-logs/stdout.log under: " + ", ".join(str(p) for p in run_roots)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="cycle run root to search (repeatable); defaults to run10 then run4",
    )
    ap.add_argument(
        "--decision",
        type=Path,
        help=(
            "gate decision.json to chart; defaults to the honest re-gate "
            f"({DEFAULT_DECISION}) rather than the training run's own, "
            "session-overlapping one"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "src" / "data.json")
    ap.add_argument("--cast", type=Path, help="recording to read event times from")
    ap.add_argument("--timing", type=Path, help="sidecar from session_render.py --timing-out")
    ap.add_argument(
        "--corroborating",
        metavar="NAME=PATH",
        action="append",
        dest="corroborating",
        default=[],
        help=(
            "additional gate decision.json for the 'reproduced N times' panel; "
            "repeatable; format: name=path/to/decision.json; "
            "when absent the built-in three-run default list is used"
        ),
    )
    args = ap.parse_args()

    root, log_path = find_training_log(args.run_root or DEFAULT_RUN_ROOTS)
    decision_path = args.decision or DEFAULT_DECISION
    if not decision_path.is_file():
        raise SystemExit(f"gate decision not found: {decision_path}")
    training = parse_training_log(log_path)
    gate = parse_decision(decision_path)

    # Parse --corroborating NAME=PATH entries if supplied; fall back to the
    # built-in default list only when the flag is entirely absent so existing
    # behaviour is preserved for callers that do not pass the flag.
    if args.corroborating:
        corroborating_entries: list[tuple[str, Path]] = []
        for spec in args.corroborating:
            if "=" not in spec:
                raise SystemExit(f"--corroborating must be NAME=PATH, got: {spec!r}")
            name, _, path_str = spec.partition("=")
            corroborating_entries.append((name.strip(), Path(path_str.strip())))
    else:
        corroborating_entries = DEFAULT_CORROBORATING_DECISIONS
    gate["reproductions"] = parse_reproductions(corroborating_entries)

    cast = args.cast or DEFAULT_CAST
    timing_path = args.timing or DEFAULT_TIMING
    anchors = None
    if cast.is_file() and timing_path.is_file():
        anchors = anchor_series(training, gate, cast, json.loads(timing_path.read_text()))
        anchors["timing"] = str(timing_path)
    else:
        # Not fatal: the numbers are still real, only their placement in the
        # video is unknown, and the composition falls back to presentational
        # timing. Say so loudly, because a silently unanchored chart looks
        # exactly like an anchored one until you compare it to the terminal.
        print(
            f"warning: no frame anchors ({cast} / {timing_path} missing); "
            "charts will fall back to presentational reveal timing",
            file=sys.stderr,
        )

    data = {
        "source": {
            "run_root": str(root),
            "training_log": str(log_path),
            "decision": str(decision_path),
        },
        "anchors": anchors,
        "training": training,
        "gate": gate,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    train, val = training["train"], training["val"]
    losses = [p["loss"] for p in train]
    print(f"run root      : {root}", file=sys.stderr)
    print(
        f"train steps   : {len(train)} (global_step {train[0]['step']}..{train[-1]['step']})",
        file=sys.stderr,
    )
    print(f"train/loss    : {max(losses):.3f} -> {min(losses):.3f}", file=sys.stderr)
    print(
        f"val epochs    : {len(val)} loss {val[0]['loss']:.3f} -> {val[-1]['loss']:.3f}",
        file=sys.stderr,
    )
    for i in range(3):
        a = [p["full_acc"][i] for p in train]
        print(f"full_acc_{i}    : {a[0]:.3f} -> {a[-1]:.3f} (max {max(a):.3f})", file=sys.stderr)
    t = gate["throughput"]
    print(
        f"throughput    : stock {t['stock']:.2f} vs tuned {t['tuned']:.2f} tok/s "
        f"({t['delta_pct']:+.2f}% +/-{t['delta_standard_error_pct']:.2f}pp SE, "
        f"stationary={t['stationary']} {t['stationarity_status']})",
        file=sys.stderr,
    )
    al = gate["accepted_length"]
    print(
        f"accepted len  : stock {al['stock']:.4f} vs tuned {al['tuned']:.4f} "
        f"({al['delta']:+.4f} SE {al['delta_standard_error']:.4f})",
        file=sys.stderr,
    )
    ar = gate["acceptance_rate"]
    print(
        f"acceptance    : stock {ar['stock']:.3f} vs tuned {ar['tuned']:.3f} "
        f"({ar['delta_pp']:+.2f}pp SE {ar['delta_standard_error_pp']:.2f})",
        file=sys.stderr,
    )
    reps = gate["reproductions"]
    print(
        "reproduced    : "
        + ", ".join(f"{r['run']} {r['delta']:+.4f}" for r in reps)
        + f"  ({len(reps)}x)",
        file=sys.stderr,
    )
    print(f"verdict       : {gate['verdict']}", file=sys.stderr)
    if anchors:
        print(
            f"anchors       : {anchors['anchored_terminal']} from the terminal, "
            f"{anchors['anchored_inherited']} inherited, "
            f"{anchors['unanchored']} unanchored",
            file=sys.stderr,
        )
        frames = [p["frame"] for p in train + val if "frame" in p]
        print(
            f"frames        : train/val {min(frames)}..{max(frames)}, "
            f"gate {anchors['gate_frame']} (of {anchors['total_frames']})",
            file=sys.stderr,
        )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
