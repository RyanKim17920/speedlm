"""Real prompts, turned into the trace records the production suite builder eats.

The simulation replays a *real* prompt-length distribution rather than
``"prompt 1"``/``"prompt 2"``, so suite hashing, held-out splitting and trace
normalisation are exercised on the same shape of input the live system sees.

On the corpus loader duplication
--------------------------------
``tests/e2e/test_live_idle_tuning.py`` owns a corpus loader, and
``tests/test_prompt_corpus.py`` carries a copy-pasted mirror of it (its own
module docstring says so).  This module deliberately does **not** add a third
copy of *that* helper: it does a different job -- it produces
:class:`~speedlm.traces.store.TraceRecord` objects, not a ``list[str]`` -- and
it reuses the e2e loader when that module can be imported, falling back to a
minimal reader only when it cannot.  The pre-existing duplication between the
e2e module and ``tests/test_prompt_corpus.py`` is untouched; consolidating it
means editing a file this change does not own.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from speedlm.traces.store import TraceRecord

#: Where the prepared UltraChat prompt corpus lives on this host.  22,362 real
#: user turns, produced by ``scripts/prepare_ultrachat_corpus.py``.
DEFAULT_CORPUS_PATH = Path("/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl")

#: Environment override, sharing the name the e2e harness already uses so an
#: operator only has to set one variable.
CORPUS_ENV_VAR = "SPEEDLM_E2E_PROMPT_CORPUS"


def corpus_path() -> Path | None:
    """The corpus to read, or ``None`` when no readable corpus is present."""
    override = os.environ.get(CORPUS_ENV_VAR)
    candidate = Path(override).expanduser() if override else DEFAULT_CORPUS_PATH
    return candidate if candidate.is_file() else None


def load_prompts(*, limit: int | None = None) -> list[str]:
    """Read user prompts from the corpus, newest-first order preserved.

    Reads incrementally and stops at *limit*: the corpus is ~17 MB and a
    simulation needs tens of prompts, so slurping the file would dominate the
    suite's runtime for no gain.

    Raises:
        FileNotFoundError: If no corpus is available.  Callers that must run
            without one should check :func:`corpus_path` first.
    """
    path = corpus_path()
    if path is None:
        raise FileNotFoundError(
            f"no prompt corpus at {DEFAULT_CORPUS_PATH} and {CORPUS_ENV_VAR} is unset"
        )
    prompts: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                continue
            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "user":
                continue
            content = first.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            prompts.append(content)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise FileNotFoundError(f"prompt corpus {path} yielded no usable prompts")
    return prompts


def synthetic_prompts(count: int) -> list[str]:
    """Stand-in prompts, used only when the real corpus is absent.

    Deliberately varied in length: a fixed-length placeholder would make the
    held-out split degenerate in a way the real corpus never is.
    """
    return [
        "Explain, in {n} sentences, {topic}.".format(
            n=(index % 5) + 1,
            topic="how speculative decoding accepts draft tokens" * ((index % 3) + 1),
        )
        for index in range(count)
    ]


def prompts_for_simulation(count: int) -> list[str]:
    """*count* real prompts when the corpus exists, synthetic ones otherwise."""
    if corpus_path() is None:
        return synthetic_prompts(count)
    return load_prompts(limit=count)


def trace_records(
    prompts: Sequence[str],
    *,
    model: str = "sim/verifier-8b",
    seed: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[TraceRecord, ...]:
    """Turn prompts into captured-looking traces.

    Each record carries the provider-authored assistant turn tagged
    ``provenance_tag="generated"``, which is what
    :meth:`speedlm.gate.suite.FrozenContext.from_trace` strips back off to
    recover the replayable input and its reference output.  Building the
    records any other way would skip that path entirely.
    """
    records: list[TraceRecord] = []
    for index, prompt in enumerate(prompts):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        records.append(
            TraceRecord(
                id=f"sim-{index:04d}-{digest}",
                timestamp=1_700_000_000.0 + index,
                model=model,
                messages=(
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": f"Reference answer {index}.",
                        "provenance_tag": "generated",
                    },
                ),
                tool_calls=(),
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=8,
            )
        )
    return tuple(records)


def simulation_traces(count: int) -> tuple[TraceRecord, ...]:
    """*count* trace records built from real prompts where available."""
    return trace_records(prompts_for_simulation(count))
