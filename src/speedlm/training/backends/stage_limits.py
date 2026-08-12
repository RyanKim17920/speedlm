"""The two checkpoints every long-running pipeline stage polls.

A stage that copies gigabytes, walks a checkpoint tree or streams a corpus has
to be interruptible partway through, on two independent grounds: an incoming
request has arrived and idle work must yield, or the stage has simply run past
the time it was given.  Both answers are a raise, and both are spelled the same
way at every checkpoint in the pipeline.

They live in their own module rather than in
:mod:`speedlm.training.backends.eagle3` so that
:mod:`speedlm.training.backends.speculators_corpus` -- which polls both while
streaming a snapshot -- does not have to import the backend that calls it.
Nothing here knows about EAGLE-3, Speculators, scratch quotas or subprocesses;
a stage supplies the abort check it was handed and the clock it started on, and
these say whether it may keep going.
"""

from __future__ import annotations

import time

from speedlm.tuner.eagle3 import AbortCheck, StageTimeoutError
from speedlm.tuner.idle import TuningPreempted

__all__ = ["check_abort", "check_deadline"]


def check_abort(guard: AbortCheck, stage: str) -> None:
    """Yield *stage* to an incoming request when *guard* says one has arrived."""
    if guard():
        raise TuningPreempted(f"incoming request preempted {stage}")


def check_deadline(started: float, timeout: float, stage: str) -> None:
    """Stop *stage* once it has run *timeout* seconds past *started*."""
    elapsed = time.monotonic() - started
    if elapsed > timeout:
        raise StageTimeoutError(
            f"{stage} exceeded {timeout:.3f}s timeout (elapsed {elapsed:.3f}s)"
        )
