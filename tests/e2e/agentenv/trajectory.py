"""Self-play attestation, re-exported from the production module.

The implementation moved to :mod:`speedlm.training.provenance` once the training
pipeline needed it: the check decides whether captured rows are legal supervision
under ``MaskPolicy.ALL_ASSISTANT_TURNS``, so it has to live where production can
import it.

This module is a re-export and deliberately holds no logic of its own.  Two
copies of a safety check is worse than none -- they drift, and the one you are
reading is never the one that ran.  ``Trajectory`` stays here because it
describes how *this test package* groups rows into sessions, which is a harness
concern rather than a training one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from speedlm.training.provenance import (
    SelfPlayAttestation,
    assistant_fingerprint,
    iter_jsonl,
    self_play_attestation,
)

__all__ = [
    "SelfPlayAttestation",
    "Trajectory",
    "assistant_fingerprint",
    "iter_jsonl",
    "self_play_attestation",
]


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The captured rows belonging to one agent session, in request order."""

    instance_id: str
    rows: tuple[Mapping[str, object], ...]
