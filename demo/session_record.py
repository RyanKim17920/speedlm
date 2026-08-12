#!/usr/bin/env python3
"""Drive a real shell on a real pseudo-terminal and record what it prints.

The honesty property this script exists for: every byte in the recording came
out of a real ``bash`` running real commands on a real PTY.  Nothing here
fabricates output, replays a canned transcript, or re-flows text that a program
did not actually emit.  ``pty.fork`` hands the shell a genuine terminal, so
programs take the same branch they take for a human -- git colourises, ``ls``
colourises, progress bars redraw with carriage returns -- and this script only
writes keystrokes in and reads bytes out.  If a command fails, the failure is
what gets recorded.

The commands come from a JSON script file rather than from this source, so the
demo content can change without anyone touching the recorder.  The schema is
deliberately small::

    {
      "typing_delay": 0.028,      // optional, seconds between keystrokes
      "settle": 1.4,              // optional, default pause_after for a step
      "prompt": "...",            // optional, PS1 for the recorded shell
      "steps": [
        {
          "comment": "narrate the next command",   // typed as a "# ..." line
          "command": "git status",
          "pause_before": 0.4,
          "pause_after": 1.6
        }
      ]
    }

The top level may also be a bare list of steps, and a step may be a bare string
(meaning: just that command, with the default pause).  ``comment`` is typed into
the shell as a real ``# ...`` line, which is real shell input -- the shell echoes
it and discards it -- so narration appears in the recording without any
post-hoc caption layer.  A step may carry a comment with no command.

Output is asciicast v2: one JSON header line, then one JSON array per event,
``[elapsed_seconds, "o", "decoded output chunk"]``.  Timestamps are real wall
clock, so the pauses in the script are what the viewer will actually wait
through.  ``demo/session_render.py`` turns the .cast into an mp4.

Known limitations, both of which live in the renderer rather than here (the
recording itself is faithful; it is the rasterisation that is ours):

  * No alternate-screen support.  A full-screen TUI (``vim``, ``htop``, ``less``)
    switches to the alternate buffer, which the renderer's emulator ignores, so
    its output bleeds into the scrollback instead of taking over the screen.
    Keep TUIs out of the script.
  * CJK and emoji are double-width cells.  The renderer draws one glyph from a
    monospace font into that pair of cells, so such text can bleed into its
    neighbour.

Usage:

    python demo/session_record.py --script demo_script.json --out session.cast \\
        [--cwd DIR] [--cols 100] [--rows 30]
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import pty
import select
import signal
import struct
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Fast enough not to bore the viewer, slow enough to read as a person typing.
DEFAULT_TYPING_DELAY = 0.028

# How long to hold after Enter before typing the next thing.  This is the knob
# that decides whether the video is legible, so it is per-step overridable.
DEFAULT_SETTLE = 1.4

# A short, coloured prompt.  It is set through PS1 in the child's environment
# rather than through a dotfile, because the shell is started with --norc so
# that whatever is in the operator's ~/.bashrc cannot leak into the recording.
DEFAULT_PROMPT = r"\[\033[1;34m\]speedlm\[\033[0m\]:\[\033[36m\]\w\[\033[0m\]$ "

# Time given to the shell to draw its first prompt before typing starts, and to
# flush the last of its output after "exit" is sent.
STARTUP_SETTLE = 1.0
SHUTDOWN_SETTLE = 1.0

READ_CHUNK = 65536


class ScriptError(ValueError):
    """Raised when the JSON session script cannot be understood."""


# ---------------------------------------------------------------------------
# Session script
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One thing to type into the live shell.

    ``comment`` and ``command`` are both optional individually but at least one
    must be present, because a step that types nothing would only be a pause and
    pauses belong on the step that earned them.
    """

    comment: str | None
    command: str | None
    pause_before: float
    pause_after: float


@dataclass(frozen=True)
class SessionScript:
    steps: tuple[Step, ...]
    typing_delay: float
    prompt: str


def _as_float(value: Any, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScriptError(f"{field} must be a number, got {value!r}")
    if value < 0:
        raise ScriptError(f"{field} must not be negative, got {value!r}")
    return float(value)


def _parse_step(raw: Any, index: int, default_settle: float) -> Step:
    if isinstance(raw, str):
        raw = {"command": raw}
    if not isinstance(raw, dict):
        raise ScriptError(f"step {index} must be a string or an object, got {type(raw).__name__}")

    unknown = set(raw) - {"comment", "command", "pause_before", "pause_after"}
    # Reject unknown keys rather than ignoring them: a typo'd "pause" that
    # silently did nothing would show up as a video that is subtly too fast, and
    # nobody would think to look at the script for the cause.
    if unknown:
        raise ScriptError(f"step {index} has unknown keys: {sorted(unknown)}")

    comment = raw.get("comment")
    command = raw.get("command")
    for name, value in (("comment", comment), ("command", command)):
        if value is not None and not isinstance(value, str):
            raise ScriptError(f"step {index} {name} must be a string, got {type(value).__name__}")
    if not comment and not command:
        raise ScriptError(f"step {index} has neither a comment nor a command")

    return Step(
        comment=comment or None,
        command=command or None,
        pause_before=_as_float(raw.get("pause_before"), f"step {index} pause_before", 0.0),
        pause_after=_as_float(raw.get("pause_after"), f"step {index} pause_after", default_settle),
    )


def load_script(path: Path) -> SessionScript:
    """Read and validate a session script, failing loudly on anything odd."""
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    if isinstance(doc, list):
        doc = {"steps": doc}
    if not isinstance(doc, dict):
        raise ScriptError("script must be a JSON object or a JSON list of steps")

    raw_steps = doc.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ScriptError("script must contain a non-empty 'steps' list")

    settle = _as_float(doc.get("settle"), "settle", DEFAULT_SETTLE)
    typing_delay = _as_float(doc.get("typing_delay"), "typing_delay", DEFAULT_TYPING_DELAY)
    prompt = doc.get("prompt", DEFAULT_PROMPT)
    if not isinstance(prompt, str):
        raise ScriptError("prompt must be a string")

    steps = tuple(_parse_step(raw, i, settle) for i, raw in enumerate(raw_steps))
    return SessionScript(steps=steps, typing_delay=typing_delay, prompt=prompt)


# ---------------------------------------------------------------------------
# The live terminal
# ---------------------------------------------------------------------------


def set_winsize(fd: int, rows: int, cols: int) -> None:
    """Tell the PTY how big it is.

    Without this the kernel reports 0x0 and every program that lays output out
    by terminal width -- git's graph, ls's columns, any progress bar -- either
    guesses 80 or refuses to colourise, and the recording stops matching what
    the video is going to be rendered at.
    """
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _child_env(cols: int, rows: int, prompt: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        TERM="xterm-256color",
        COLUMNS=str(cols),
        LINES=str(rows),
        PS1=prompt,
        # A pager would take over the screen and wait for a keypress that is
        # never coming, so anything that would page is told to just print.
        GIT_PAGER="cat",
        PAGER="cat",
        # Some tools disable colour merely because NO_COLOR is *set*, empty or
        # not, and the whole point of recording a real terminal is the colour.
        BASH_SILENCE_DEPRECATION_WARNING="1",
    )
    env.pop("NO_COLOR", None)
    # PROMPT_COMMAND can print anything it likes before every prompt, including
    # a window title escape or a git status line from the operator's setup.
    env.pop("PROMPT_COMMAND", None)
    return env


def _spawn_shell(cwd: Path, env: dict[str, str]) -> tuple[int, int]:
    """Fork a real bash onto a real PTY and return (pid, master_fd)."""
    pid, master = pty.fork()
    if pid == 0:  # child: become the shell
        try:
            os.chdir(cwd)
            # --norc/--noprofile keeps the operator's dotfiles out of the
            # recording; -i is what makes bash print a prompt and enable job
            # control, which is what a viewer expects to see.
            os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"], env)
        except OSError:
            pass
        os._exit(127)
    return pid, master


class Recorder:
    """Collects timestamped output chunks from the PTY master."""

    def __init__(self, master: int) -> None:
        self.master = master
        self.start = time.time()
        self.events: list[tuple[float, str, str]] = []
        self._tail = b""
        self._eof = False

    def pump(self, until: float) -> None:
        """Drain the PTY until the absolute time ``until``, recording chunks.

        This doubles as the sleep: waiting is done inside select() so output
        that arrives during a pause is still captured at the moment it arrived,
        which is what makes the timing in the .cast real rather than assigned.
        """
        while True:
            remaining = until - time.time()
            if remaining <= 0 or self._eof:
                return
            try:
                ready, _, _ = select.select([self.master], [], [], min(remaining, 0.05))
            except (OSError, InterruptedError):
                self._eof = True
                return
            if not ready:
                continue
            try:
                chunk = os.read(self.master, READ_CHUNK)
            except OSError as exc:
                # EIO on a PTY master means the child closed the slave, i.e. the
                # shell exited.  That is a normal end of session, not a failure.
                if exc.errno in (errno.EIO, errno.EBADF):
                    self._eof = True
                    return
                raise
            if not chunk:
                self._eof = True
                return
            # A read can land in the middle of a multi-byte UTF-8 sequence, so
            # the undecodable tail is carried into the next read rather than
            # being replaced -- otherwise a box-drawing character split across
            # two reads would become two replacement glyphs in the video.
            buf = self._tail + chunk
            try:
                text = buf.decode("utf-8")
                self._tail = b""
            except UnicodeDecodeError as exc:
                text = buf[: exc.start].decode("utf-8")
                self._tail = buf[exc.start :]
            if text:
                self.events.append((time.time() - self.start, "o", text))

    def type_line(self, text: str, delay: float) -> None:
        """Type ``text`` a character at a time, then press Enter.

        Character-at-a-time is not decoration: the shell echoes each keystroke
        as it arrives, so the recording contains the line appearing gradually
        exactly as it would for a person at the keyboard.  Writing the whole
        line at once would make it pop into existence in a single frame.
        """
        for ch in text:
            # A keyboard sends CR for Enter; the line discipline maps it to NL.
            # Sending NL directly would work here but would not be what a real
            # keystroke looks like, and multi-line commands rely on the shell's
            # own PS2 continuation handling to look right.
            os.write(self.master, b"\r" if ch == "\n" else ch.encode("utf-8"))
            self.pump(time.time() + delay)
        os.write(self.master, b"\r")


def record(script: SessionScript, out_path: Path, cwd: Path, cols: int, rows: int) -> None:
    pid, master = _spawn_shell(cwd, _child_env(cols, rows, script.prompt))
    set_winsize(master, rows, cols)

    rec = Recorder(master)
    rec.pump(time.time() + STARTUP_SETTLE)  # let the shell draw its first prompt

    for step in script.steps:
        if step.pause_before:
            rec.pump(time.time() + step.pause_before)
        if step.comment:
            # Typed as a real shell comment, so it is genuine input the shell
            # echoes and then ignores.  No caption is drawn over the video.
            comment = step.comment if step.comment.startswith("#") else f"# {step.comment}"
            rec.type_line(comment, script.typing_delay)
            rec.pump(time.time() + 0.35)
        if step.command:
            rec.type_line(step.command, script.typing_delay)
        rec.pump(time.time() + step.pause_after)

    os.write(master, b"exit\r")
    rec.pump(time.time() + SHUTDOWN_SETTLE)

    # SIGHUP is what a real terminal sends when its window closes, so a shell
    # that somehow did not take "exit" still goes away rather than being left
    # behind holding the PTY.
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGHUP)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)
    os.close(master)

    if not rec.events:
        raise SystemExit("the shell produced no output at all -- refusing to write an empty cast")

    header = {
        "version": 2,
        "width": cols,
        "height": rows,
        "timestamp": int(rec.start),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.writelines(
            json.dumps([round(t, 6), kind, data]) + "\n" for t, kind, data in rec.events
        )

    total = rec.events[-1][0]
    print(f"wrote {out_path}: {len(rec.events)} events, {total:.2f}s, {cols}x{rows}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a real PTY session to asciicast v2.")
    ap.add_argument("--script", required=True, type=Path, help="JSON session script")
    ap.add_argument("--out", required=True, type=Path, help="output .cast path")
    ap.add_argument("--cwd", type=Path, default=Path.cwd(), help="directory to run the shell in")
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=30)
    args = ap.parse_args()

    if args.cols < 20 or args.rows < 5:
        raise SystemExit("--cols/--rows are unreasonably small")
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise SystemExit(f"--cwd is not a directory: {cwd}")

    try:
        script = load_script(args.script)
    except (OSError, json.JSONDecodeError, ScriptError) as exc:
        raise SystemExit(f"bad session script {args.script}: {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record(script, args.out, cwd, args.cols, args.rows)


if __name__ == "__main__":
    main()
