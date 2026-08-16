#!/usr/bin/env python3
"""Build the SpeedLM demo video end-to-end from run artifacts.

One command replaces the remembered sequence of six:

    python demo/build_video.py \\
        --corpus /data/.../bigcorpus-run1/traffic/trajectories \\
        --capture-dir /data/.../demo-video-run2 \\
        --decision /data/.../regate-big-run2/decision.json \\
        --corroborating /data/.../regate-big-run1/decision.json \\
        --training-run /data/.../bigcycle-run1 \\
        --out /data/.../demo-versions/speedlm-rebuilt.mp4 \\
        --work-dir /data/.../demo-build

Stages (in order)
-----------------
  montage    demo/montage.py          -> <work-dir>/traffic-montage.mp4   (CPU)
  terminal   session_record.py        -> <work-dir>/session_fast.cast     (CPU)
             session_render.py        -> <work-dir>/terminal.mp4 + timing.json
  remotion   extract_series.py        -> demo/remotion/src/data.json      (CPU)
             npx remotion render      -> <work-dir>/versionA-split.mp4
  race       demo/render.py --speed 8 -> <work-dir>/race-8x.mp4           (CPU)
  assemble   ffmpeg filter_complex    -> --out

Use --skip <stage> (repeatable) to reuse an existing intermediate when that
stage's inputs are unchanged.

Assembly details
----------------
The three clips are joined in one filter_complex re-encode pass that also
converts clip [1]'s yuvj420p (Remotion output) to yuv420p.

  [0] traffic-montage.mp4   whole clip, no trim
  [1] versionA-split.mp4    whole clip, no trim
  [2] race-8x.mp4           trim 1.5 s head, 26.9 s tail  (25.40 s kept)

libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart, no audio.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"
REMOTION_DIR = DEMO_DIR / "remotion"
ROOT_TSX = REMOTION_DIR / "src" / "Root.tsx"
SESSION_SCRIPT = DEMO_DIR / "session_fast.json"

FPS = 30
# Trim applied to the race clip in the final assembly — must match README.
RACE_TRIM_START = 1.5
RACE_TRIM_END = 26.9
RACE_TRIM_DURATION = RACE_TRIM_END - RACE_TRIM_START  # 25.40 s

STAGES = ["montage", "terminal", "remotion", "race", "assemble"]

PYTHON = sys.executable


class BuildError(Exception):
    """Raised to abort the build with a clean error message (no traceback)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hr(label: str) -> None:
    bar = "─" * max(0, 72 - len(label) - 3)
    print(f"\n── {label} {bar}", flush=True)


def run(cmd: list, *, cwd: Path | None = None) -> None:
    """Print the exact subprocess command line, then run it; abort on failure."""
    str_cmd = [str(c) for c in cmd]
    print(f"  $ {' '.join(str_cmd)}", flush=True)
    result = subprocess.run(str_cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise BuildError(f"command exited {result.returncode}: {str_cmd[0]}")


def ffmpeg_exe() -> str:
    r = subprocess.run(
        [PYTHON, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def video_duration(path: Path) -> float:
    """Return video duration in seconds.

    Uses ffprobe when it is on PATH.  Falls back to parsing the ``Duration:``
    line that ffmpeg -i prints to stderr — imageio_ffmpeg bundles only ffmpeg,
    not ffprobe, so the fallback is the common case on this machine.
    """
    # Try ffprobe first (accurate, machine-readable).
    fprobe = shutil.which("ffprobe")
    if not fprobe:
        # Look for ffprobe next to the bundled ffmpeg binary.
        ff = ffmpeg_exe()
        candidate = str(Path(ff).parent / "ffprobe")
        if Path(candidate).exists():
            fprobe = candidate
    if fprobe:
        r = subprocess.run(
            [
                fprobe,
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        text = r.stdout.strip()
        if not text or text == "N/A":
            r2 = subprocess.run(
                [
                    fprobe,
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            text = r2.stdout.strip()
        return float(text)

    # ffprobe absent — parse "Duration: HH:MM:SS.ss" from ffmpeg -i stderr.
    ff = ffmpeg_exe()
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + s
    raise BuildError(
        f"Could not determine duration of {path}.\n"
        "  Install ffprobe (system ffmpeg package) and retry."
    )


def file_report(path: Path) -> str:
    size_mb = path.stat().st_size / 1e6
    try:
        dur = video_duration(path)
        return f"{path.name}  {size_mb:.1f} MB  {dur:.2f}s"
    except Exception:
        return f"{path.name}  {size_mb:.1f} MB"


def parse_duration_in_frames(tsx: Path) -> int | None:
    m = re.search(
        r"export\s+const\s+DURATION_IN_FRAMES\s*=\s*(\d+)",
        tsx.read_text(),
    )
    return int(m.group(1)) if m else None


def check_skip(stage: str, artifact: Path) -> None:
    """Confirm the skip-stage artifact exists; abort if it does not."""
    if not artifact.exists():
        raise BuildError(
            f"--skip {stage}: expected output artifact does not exist: {artifact}\n"
            f"  Remove --skip {stage} to rebuild it, or supply the file."
        )
    print(f"  [skip] using existing: {file_report(artifact)}", flush=True)


# ---------------------------------------------------------------------------
# Input validation  (always runs up-front, before any stage executes)
# ---------------------------------------------------------------------------


def validate_inputs(args: argparse.Namespace, skip: set[str]) -> None:
    """Check every required input.  Abort immediately if anything is missing."""
    errors: list[str] = []

    def need(path: Path | str, label: str) -> None:
        if not Path(path).exists():
            errors.append(f"  missing {label}: {path}")

    # montage
    if "montage" not in skip:
        if not args.corpus:
            errors.append("  --corpus is required (no trajectory dirs supplied)")
        for d in args.corpus:
            need(d, "--corpus trajectory dir")

    # terminal
    if "terminal" not in skip:
        need(SESSION_SCRIPT, "session script (demo/session_fast.json)")

    # remotion
    if "remotion" not in skip:
        need(ROOT_TSX, "demo/remotion/src/Root.tsx")
        need(REMOTION_DIR / "package.json", "demo/remotion/package.json")
        need(args.training_run, "--training-run dir")
        need(args.decision, "--decision")

    # race
    if "race" not in skip:
        cap = Path(args.capture_dir)
        need(cap, "--capture-dir")
        need(cap / "timeline-stock.jsonl", "--capture-dir/timeline-stock.jsonl")
        need(cap / "timeline-candidate.jsonl", "--capture-dir/timeline-candidate.jsonl")
        need(cap / "capture_manifest.json", "--capture-dir/capture_manifest.json")
        need(args.decision, "--decision")
        for c in args.corroborating:
            need(c, "--corroborating")

    if errors:
        raise BuildError("Input validation failed — missing paths:\n" + "\n".join(errors))

    print("  All inputs present.", flush=True)


# ---------------------------------------------------------------------------
# Stage: montage
# ---------------------------------------------------------------------------


def stage_montage(args: argparse.Namespace, work_dir: Path, skip: set[str]) -> Path:
    out = work_dir / "traffic-montage.mp4"
    hr("stage 1/5: montage")

    if "montage" in skip:
        check_skip("montage", out)
        return out

    training_run = Path(args.training_run)
    traces = training_run / "speedlm_home" / "traces" / "traces.jsonl"

    cmd: list = [PYTHON, DEMO_DIR / "montage.py", "--out", out]
    for c in args.corpus:
        cmd += ["--corpus", c]
    if traces.exists():
        cmd += ["--traces", traces]
    else:
        print(
            f"  note: traces.jsonl not found at {traces}; "
            "--traces omitted (token tally will be absent from the totals card)",
            flush=True,
        )

    run(cmd, cwd=REPO_ROOT)
    print(f"\n  output: {file_report(out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage: terminal
# ---------------------------------------------------------------------------


def stage_terminal(
    args: argparse.Namespace,
    work_dir: Path,
    skip: set[str],
    remotion_will_run: bool,
) -> tuple[Path, Path, Path]:
    """Returns (cast_path, terminal_mp4, timing_json)."""
    cast = work_dir / "session_fast.cast"
    terminal_mp4 = work_dir / "terminal.mp4"
    timing_json = work_dir / "timing.json"

    hr("stage 2/5: terminal")

    if "terminal" in skip:
        check_skip("terminal", terminal_mp4)
        if remotion_will_run:
            for p, label in [(cast, "session_fast.cast"), (timing_json, "timing.json")]:
                if not p.exists():
                    raise BuildError(
                        f"--skip terminal: {label} is missing at {p} but the remotion"
                        " stage needs it.\n"
                        "  Either remove --skip terminal or also add --skip remotion."
                    )
        return cast, terminal_mp4, timing_json

    print("  step 1/2: record PTY session", flush=True)
    run(
        [
            PYTHON,
            DEMO_DIR / "session_record.py",
            "--script",
            SESSION_SCRIPT,
            "--out",
            cast,
            "--cwd",
            REPO_ROOT,
            "--cols",
            "100",
            "--rows",
            "30",
        ],
        cwd=REPO_ROOT,
    )

    print("\n  step 2/2: render cast to video", flush=True)
    run(
        [
            PYTHON,
            DEMO_DIR / "session_render.py",
            cast,
            terminal_mp4,
            "--timing-out",
            timing_json,
        ],
        cwd=REPO_ROOT,
    )

    print(f"\n  output: {file_report(terminal_mp4)}", flush=True)
    return cast, terminal_mp4, timing_json


# ---------------------------------------------------------------------------
# Stage: remotion
# ---------------------------------------------------------------------------


def stage_remotion(
    args: argparse.Namespace,
    work_dir: Path,
    skip: set[str],
    cast: Path,
    terminal_mp4: Path,
    timing_json: Path,
) -> Path:
    out = work_dir / "versionA-split.mp4"
    hr("stage 3/5: remotion")

    if "remotion" in skip:
        check_skip("remotion", out)
        return out

    # --- guard: npx available? -------------------------------------------
    npx = shutil.which("npx")
    if not npx:
        raise BuildError(
            "npx not found in PATH.  Install Node.js to run the remotion stage.\n"
            "  Without it the terminal+chart split-screen clip cannot be rendered.\n"
            "  Add --skip remotion if you have an existing versionA-split.mp4 to reuse."
        )

    # --- guard: node_modules installed? ------------------------------------
    node_modules = REMOTION_DIR / "node_modules"
    if not node_modules.is_dir():
        raise BuildError(
            f"Remotion node_modules not found: {node_modules}\n"
            f"  Run:  cd {REMOTION_DIR} && npm install"
        )
    print("  npx: present.  node_modules: present.", flush=True)

    # --- guard: DURATION_IN_FRAMES must match the new terminal video ------
    current_frames = parse_duration_in_frames(ROOT_TSX)
    if current_frames is None:
        raise BuildError(
            f"Could not parse DURATION_IN_FRAMES from {ROOT_TSX}.\n"
            "  The file must contain a line like:\n"
            "    export const DURATION_IN_FRAMES = 1344;"
        )

    term_dur = video_duration(terminal_mp4)
    actual_frames = round(term_dur * FPS)
    if actual_frames != current_frames:
        raise BuildError(
            f"Root.tsx has DURATION_IN_FRAMES = {current_frames} but the newly recorded\n"
            f"  terminal.mp4 is {actual_frames} frames ({term_dur:.2f}s at {FPS} fps).\n"
            f"\n"
            f"  Update {ROOT_TSX}\n"
            f"  (currently line 14) to:\n"
            f"\n"
            f"    export const DURATION_IN_FRAMES = {actual_frames};\n"
            f"\n"
            f"  Then re-run.  Proceeding with the wrong value would render a clip that\n"
            f"  is truncated or padded relative to the actual terminal recording."
        )
    print(
        f"  DURATION_IN_FRAMES check: {current_frames} frames = {term_dur:.2f}s  OK",
        flush=True,
    )

    # --- copy terminal.mp4 into Remotion's public/ -----------------------
    public_terminal = REMOTION_DIR / "public" / "terminal.mp4"
    public_terminal.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ cp {terminal_mp4} {public_terminal}", flush=True)
    import shutil as _shutil

    _shutil.copy2(terminal_mp4, public_terminal)

    # --- extract_series.py: build src/data.json -------------------------
    print("\n  step 1/2: extract_series.py", flush=True)
    run(
        [
            PYTHON,
            REMOTION_DIR / "extract_series.py",
            "--run-root",
            args.training_run,
            "--decision",
            args.decision,
            "--cast",
            cast,
            "--timing",
            timing_json,
        ],
        cwd=REPO_ROOT,
    )

    # --- npx remotion render -------------------------------------------
    print("\n  step 2/2: npx remotion render", flush=True)
    run(
        [
            "npx",
            "remotion",
            "render",
            "SpeedLMDemo",
            str(out.resolve()),
            "--codec=h264",
            "--crf=20",
        ],
        cwd=REMOTION_DIR,
    )

    print(f"\n  output: {file_report(out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage: race
# ---------------------------------------------------------------------------


def stage_race(args: argparse.Namespace, work_dir: Path, skip: set[str]) -> Path:
    out = work_dir / "race-8x.mp4"
    hr("stage 4/5: race")

    if "race" in skip:
        check_skip("race", out)
        return out

    cmd: list = [
        PYTHON,
        DEMO_DIR / "render.py",
        "--capture-dir",
        args.capture_dir,
        "--decision",
        args.decision,
        "--out",
        out,
        "--speed",
        "8",
    ]
    for c in args.corroborating:
        cmd += ["--corroborating", str(c)]

    run(cmd, cwd=REPO_ROOT)
    dur = video_duration(out)
    print(f"\n  output: {file_report(out)}", flush=True)

    # Sanity: the race clip must be long enough for the trim
    if dur < RACE_TRIM_END:
        raise BuildError(
            f"Race clip is {dur:.2f}s but assembly needs to trim to {RACE_TRIM_END}s.\n"
            f"  The clip is too short.  Check render.py --speed 8 output."
        )
    print(
        f"  Trim check: clip {dur:.2f}s >= trim end {RACE_TRIM_END}s  OK",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Stage: assemble
# ---------------------------------------------------------------------------


def stage_assemble(
    args: argparse.Namespace,
    work_dir: Path,
    skip: set[str],
    montage_mp4: Path,
    split_mp4: Path,
    race_mp4: Path,
) -> Path:
    out = Path(args.out)
    hr("stage 5/5: assemble")

    if "assemble" in skip:
        check_skip("assemble", out)
        return out

    for clip, label in [
        (montage_mp4, "montage clip"),
        (split_mp4, "remotion clip"),
        (race_mp4, "race clip"),
    ]:
        if not clip.exists():
            raise BuildError(f"assemble: {label} not found: {clip}")

    montage_dur = video_duration(montage_mp4)
    split_dur = video_duration(split_mp4)
    race_dur = video_duration(race_mp4)

    print("  source clips:", flush=True)
    print(f"    [0] montage  {montage_dur:.2f}s  {montage_mp4}", flush=True)
    print(f"    [1] split    {split_dur:.2f}s  {split_mp4}", flush=True)
    print(f"    [2] race     {race_dur:.2f}s  {race_mp4}", flush=True)

    if race_dur < RACE_TRIM_END:
        raise BuildError(
            f"Race clip is {race_dur:.2f}s but the trim end is {RACE_TRIM_END}s.\n"
            "  The clip is too short for the required trim.  Re-run the race stage."
        )

    # Build the filter_complex command that matches the authoritative README assembly.
    #   [0] montage: whole clip, format -> yuv420p, fps = 30
    #   [1] split: whole clip, format -> yuv420p (Remotion emits yuvj420p), fps = 30
    #   [2] race: trim 1.5..26.9, setpts, format -> yuv420p, fps = 30
    filter_complex = (
        "[0:v]format=yuv420p,fps=30[v0];"
        "[1:v]format=yuv420p,fps=30[v1];"
        f"[2:v]trim=start={RACE_TRIM_START}:end={RACE_TRIM_END},"
        "setpts=PTS-STARTPTS,format=yuv420p,fps=30[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[out]"
    )

    ff = ffmpeg_exe()
    out.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            ff,
            "-y",
            "-i",
            montage_mp4,
            "-i",
            split_mp4,
            "-i",
            race_mp4,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out,
        ],
        cwd=REPO_ROOT,
    )

    final_dur = video_duration(out)
    expected_dur = montage_dur + split_dur + RACE_TRIM_DURATION
    discrepancy = abs(final_dur - expected_dur)

    print(f"\n  final duration: {final_dur:.2f}s  (expected ~{expected_dur:.2f}s)", flush=True)
    print(
        f"  arithmetic:  montage {montage_dur:.2f}s"
        f" + split {split_dur:.2f}s"
        f" + race trim {RACE_TRIM_DURATION:.2f}s"
        f" = {expected_dur:.2f}s",
        flush=True,
    )
    if discrepancy > 0.5:
        print(
            f"\n  WARNING: duration discrepancy {discrepancy:.3f}s exceeds 0.5s threshold.\n"
            f"  A source clip's duration differed from what the trim math assumed.\n"
            f"  Inspect the clips above and verify the trim constants "
            f"(RACE_TRIM_START={RACE_TRIM_START}, RACE_TRIM_END={RACE_TRIM_END}).",
            flush=True,
        )
    else:
        print(f"  Duration check: OK (discrepancy {discrepancy:.3f}s)", flush=True)

    size_mb = out.stat().st_size / 1e6
    print(f"\n  output: {out}  {size_mb:.1f} MB  {final_dur:.2f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--corpus",
        metavar="DIR",
        action="append",
        default=[],
        help=(
            "traffic/trajectories directory for the montage stage; "
            "repeatable — pass one flag per corpus dir"
        ),
    )
    ap.add_argument(
        "--capture-dir",
        metavar="DIR",
        required=True,
        help=(
            "directory produced by capture.py, containing "
            "timeline-stock.jsonl, timeline-candidate.jsonl, and capture_manifest.json"
        ),
    )
    ap.add_argument(
        "--decision",
        metavar="PATH",
        type=Path,
        required=True,
        help=(
            "definitive gate decision.json; "
            "every on-screen gate figure is derived from this file at render time"
        ),
    )
    ap.add_argument(
        "--corroborating",
        metavar="PATH",
        action="append",
        default=[],
        type=Path,
        help=(
            "additional gate decision.json files for the 'reproduced N times' line "
            "on the race outro card; repeatable; the line is suppressed if none are given"
        ),
    )
    ap.add_argument(
        "--training-run",
        metavar="DIR",
        required=True,
        help=(
            "cycle run root (e.g. bigcycle-run1) that holds "
            "speedlm_home/runs/*/training-logs/stdout.log for the chart data, "
            "and speedlm_home/traces/traces.jsonl for the montage token tally"
        ),
    )
    ap.add_argument(
        "--out",
        metavar="PATH",
        required=True,
        help="final assembled mp4",
    )
    ap.add_argument(
        "--work-dir",
        metavar="DIR",
        required=True,
        help="directory for intermediate clips; created if absent",
    )
    ap.add_argument(
        "--skip",
        metavar="STAGE",
        action="append",
        default=[],
        choices=STAGES,
        help=(
            f"skip a stage and reuse its existing output; repeatable; choices: {', '.join(STAGES)}"
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    skip = set(args.skip)

    invalid = skip - set(STAGES)
    if invalid:
        print(f"ERROR: unknown stage(s) in --skip: {', '.join(sorted(invalid))}", file=sys.stderr)
        return 1

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    hr("SpeedLM demo video builder")
    print(f"  stages : {', '.join(STAGES)}", flush=True)
    print(f"  skip   : {', '.join(s for s in STAGES if s in skip) or '(none)'}", flush=True)
    print(f"  work   : {work_dir}", flush=True)
    print(f"  out    : {args.out}", flush=True)

    try:
        hr("validate inputs")
        validate_inputs(args, skip)

        montage_mp4 = stage_montage(args, work_dir, skip)
        remotion_will_run = "remotion" not in skip
        cast, terminal_mp4, timing_json = stage_terminal(args, work_dir, skip, remotion_will_run)
        split_mp4 = stage_remotion(args, work_dir, skip, cast, terminal_mp4, timing_json)
        race_mp4 = stage_race(args, work_dir, skip)
        final = stage_assemble(args, work_dir, skip, montage_mp4, split_mp4, race_mp4)

        hr("done")
        print(f"  {final}", flush=True)
        return 0

    except BuildError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
