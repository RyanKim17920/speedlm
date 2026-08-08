"""Literal GPU E2E for the production idle-tuning lifecycle.

The test is intentionally model- and vendor-agnostic.  The operator supplies a
complete SpeedLM config (including an EAGLE-3 profile and Speculators paths);
the test derives the served model, alias, trace threshold, and sampling values
from that config.

Required environment:

* ``SPEEDLM_E2E_IDLE_TUNING=1``
* ``SPEEDLM_E2E_TUNING_CONFIG=/absolute/path/to/config.json``
* ``SPEEDLM_E2E_ARTIFACT_DIR=/durable/artifact/root``

Optional environment:

* ``SPEEDLM_E2E_TUNING_PROFILE=/path/to/custom-profile.json``
* ``SPEEDLM_E2E_VLLM_ARGS='["--max-model-len", "4096"]'``
* ``SPEEDLM_E2E_READY_TIMEOUT=900``
* ``SPEEDLM_E2E_TUNING_TIMEOUT=7200``
* ``SPEEDLM_E2E_REQUEST_TIMEOUT=1200``
* ``SPEEDLM_E2E_SEED_REQUESTS=256``
* ``SPEEDLM_E2E_PROMPT_CORPUS=/path/to/prompts.jsonl``
* ``SPEEDLM_E2E_WORKLOAD=agentic-mixed-outcome``

``SPEEDLM_E2E_WORKLOAD`` names a manifest under
``tests/e2e/harness/workload_specs``.  When it is set to anything other than the
launcher's default (``generic-chat``), the workload's *verified* records file
becomes the seed corpus and ``SPEEDLM_E2E_PROMPT_CORPUS`` must not also be set --
two corpora would mean the run measures neither of them.  The default keeps the
historical path exactly: ``SPEEDLM_E2E_PROMPT_CORPUS`` if given, the synthetic
template otherwise.  Three archived runs and every number in
``docs/benchmark-evidence.md`` were taken on that path, so it does not move.

One invocation creates an isolated ``SPEEDLM_HOME`` under the artifact root,
so old traces, active artifacts, and scheduler state cannot satisfy assertions.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from speedlm.config import SpeedLMConfig, load_config
from speedlm.gate.replay import (
    NATURAL_STOP_FINISH_REASONS,
    TRUNCATED_FINISH_REASONS,
    is_natural_stop_finish_reason,
    is_truncated_finish_reason,
)
from speedlm.profiles import (
    ModelProfile,
    drafter_declared_speculative_tokens,
    resolve_profile,
)
from speedlm.tuner.composition import declared_draft_depth
from tests.e2e.harness import workloads

# ── prompt corpus helpers ───────────────────────────────────────────────────

#: The launcher's ``--workload`` default (make_snapshot_run.sh:218).  It is the
#: value that means "the operator selected nothing", so it routes to the legacy
#: corpus path rather than through the workload registry.
DEFAULT_WORKLOAD: Final = "generic-chat"

#: Roles an OpenAI-compatible ``/v1/chat/completions`` accepts on input.  A
#: record carrying anything else is a corpus defect, not traffic to replay: the
#: server would reject it mid-run, after the engine had already been paid for.
ACCEPTED_MESSAGE_ROLES: Final[frozenset[str]] = frozenset(
    {"system", "developer", "user", "assistant", "tool", "function"}
)


@dataclass(frozen=True, slots=True)
class SeedRequest:
    """One seed request, carried whole.

    The corpus loader used to return bare strings taken from the first message,
    which made every corpus single-turn and silently deleted system prompts and
    tool schemas -- i.e. exactly the parts that make agentic traffic agentic.
    ``messages`` is the record's array verbatim; ``tools`` is its ``tools`` array
    when it declared one.
    """

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()


def _user_request(text: str) -> SeedRequest:
    """A single-user-turn request, the shape the synthetic fallback produces."""
    return SeedRequest(messages=({"role": "user", "content": text},))


def _truthy_tool_calls(raw: object) -> bool:
    """Whether *raw* is a tool-call list with at least one entry in it.

    Separate from :func:`_check_tool_calls` because the two answer different
    questions: this one decides whether the message has anything to send at
    all, that one decides whether what it has is well formed.  The old code
    used bare truthiness, so the string ``"yes"`` satisfied "this message
    dispatches a tool".
    """
    return isinstance(raw, list) and bool(raw)


def _check_message_content(content: object, *, where: str) -> None:
    """Reject a ``content`` value an OpenAI-compatible server would not read.

    A string, or a non-empty list of content parts each naming its ``type``.
    Numbers, bare objects and empty lists are rejected: the server 400s on
    them, and it does so mid-run.  ``None`` never reaches here -- the caller
    treats it as absent content, which is legal only alongside a tool call.
    """
    if isinstance(content, str):
        return
    if not isinstance(content, list) or not content:
        raise AssertionError(
            f"{where} has 'content' of type {type(content).__name__}; an "
            "OpenAI-compatible server accepts a string or a non-empty list of "
            "content parts"
        )
    for index, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            raise AssertionError(
                f"{where} has content[{index}] that is not a content part "
                "object with a string 'type'"
            )


def _check_tool_calls(raw: object, *, where: str) -> None:
    """Reject a ``tool_calls`` value the server could not replay.

    Held to the same standard as :func:`speedlm.gate.replay._extract_tool_calls`
    applies on the way back: a list of objects, each naming a ``function`` with
    a non-empty ``name``.  Assistant turns replayed into a request must also
    carry the ``id`` the following ``tool`` turn refers to -- without it the
    server cannot pair the two, which is a 400 on a record that used to load.
    """
    if not isinstance(raw, list):
        raise AssertionError(
            f"{where} has 'tool_calls' of type {type(raw).__name__}, not a list"
        )
    for index, call in enumerate(raw):
        if not isinstance(call, dict):
            raise AssertionError(f"{where} has tool_calls[{index}] that is not an object")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(
            function.get("name"), str
        ):
            raise AssertionError(
                f"{where} has tool_calls[{index}] with no 'function' object "
                "naming a function, so it dispatches nothing"
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise AssertionError(
                f"{where} has tool_calls[{index}] with no non-empty string "
                "'id'; the tool result that answers it could not be paired"
            )


def _build_seed_request(obj: object, *, context: str) -> SeedRequest:
    """Validate one corpus record and carry it whole.

    Strict about the right thing.  The predecessor asserted that the FIRST
    message was a user turn, which is not a property of chat traffic at all --
    it is a property of the one single-turn corpus this test had ever seen, and
    it hard-fails on record 1 of every agentic workload (whose first message is
    the system prompt).  What must hold instead: the messages are a non-empty
    list of objects, every role is one the server accepts, every message carries
    something the server can actually read, and the record actually asks the
    model for something.

    "Something to send" used to mean the *key* ``content`` was present, or
    ``tool_calls`` was truthy.  That is presence, not validity, and everything
    it waved through reaches the HTTP payload verbatim: ``content`` could be a
    number, a bare object or ``null``; ``tool_calls`` could be the string
    ``"yes"``; a ``tool`` turn needed no ``tool_call_id`` to answer; and a tool
    schema needed only to be a dict, so ``{}`` passed.  Each of those is a 400
    from the server *after* the engine has been paid for and the run is
    underway -- which is the whole reason this validation runs up front.  The
    predecessor was narrower but stricter; widening the shape is not a licence
    to stop checking it.

    What is deliberately NOT required: that ``content`` be a ``str``.  Real
    agentic content is frequently a list of content parts
    (``[{"type": "text", "text": ...}]``), which ``workloads.content_text``
    exists to flatten, so demanding a string would reject exactly the traffic
    this loader was widened to carry.
    """
    if not isinstance(obj, dict):
        raise AssertionError(f"{context}: record is not a JSON object")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        raise AssertionError(f"{context}: record has no non-empty 'messages' list")
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise AssertionError(f"{context}: messages[{position}] is not an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in ACCEPTED_MESSAGE_ROLES:
            raise AssertionError(
                f"{context}: messages[{position}] has role {role!r}; an "
                "OpenAI-compatible server accepts only "
                f"{', '.join(sorted(ACCEPTED_MESSAGE_ROLES))}"
            )
        where = f"{context}: messages[{position}] (role {role!r})"
        content = message.get("content")
        has_content = "content" in message and content is not None
        if has_content:
            _check_message_content(content, where=where)
        has_tool_calls = "tool_calls" in message
        if has_tool_calls:
            _check_tool_calls(message.get("tool_calls"), where=where)
        if not has_content and not _truthy_tool_calls(message.get("tool_calls")):
            raise AssertionError(
                f"{where} carries neither 'content' nor a tool call, so there "
                "is nothing to send"
            )
        # An OpenAI-compatible server matches a tool result back to the call it
        # answers by id, and rejects the turn outright without one.  The old
        # check let such a record through on the strength of its content alone.
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise AssertionError(
                    f"{where} has no non-empty string 'tool_call_id'; a tool "
                    "result cannot be matched to the call it answers without one"
                )
    if not any(
        message.get("role") == "user"
        and workloads.content_text(message.get("content")).strip()
        for message in messages
    ):
        raise AssertionError(f"{context}: record carries no user message with content")

    raw_tools = obj.get("tools")
    tools: tuple[dict[str, Any], ...] = ()
    if raw_tools is not None:
        if not isinstance(raw_tools, list) or not raw_tools:
            raise AssertionError(
                f"{context}: 'tools' is present but is not a non-empty list"
            )
        for position, tool in enumerate(raw_tools):
            if not isinstance(tool, dict):
                raise AssertionError(f"{context}: tools[{position}] is not an object")
            # ``{}`` is an object and used to pass.  A tool schema the model
            # cannot name is not a tool: the server rejects the request, and if
            # it did not, the model could never emit a matching call.  Both
            # real workloads store exactly ``{"type": "function", "function":
            # {...}}``, so this is the shape on disk, not an invented one.
            function = tool.get("function")
            if tool.get("type") != "function" or not isinstance(function, dict):
                raise AssertionError(
                    f"{context}: tools[{position}] is not a "
                    "{'type': 'function', 'function': {...}} schema"
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise AssertionError(
                    f"{context}: tools[{position}].function has no non-empty "
                    "string 'name', so nothing could ever call it"
                )
        tools = tuple(dict(tool) for tool in raw_tools)
    return SeedRequest(
        messages=tuple(dict(message) for message in messages), tools=tools
    )


def _load_prompt_corpus() -> list[SeedRequest] | None:
    """Load real requests from SPEEDLM_E2E_PROMPT_CORPUS if set.

    Expects a JSONL file where each line is ``{"messages": [...]}`` and may also
    carry ``"tools": [...]``.  Returns ``None`` when the env var is not set.
    """
    corpus_path = os.environ.get("SPEEDLM_E2E_PROMPT_CORPUS")
    if corpus_path is None:
        return None
    path = Path(corpus_path).expanduser().resolve()
    assert path.is_file(), f"SPEEDLM_E2E_PROMPT_CORPUS is not a file: {path}"
    requests: list[SeedRequest] = []
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        requests.append(
            _build_seed_request(json.loads(stripped), context=f"{path}:{number}")
        )
    return requests


def _selected_workload() -> str | None:
    """The workload the operator asked for, or ``None`` for "not selected"."""
    raw = (os.environ.get("SPEEDLM_E2E_WORKLOAD") or "").strip()
    if not raw or raw == DEFAULT_WORKLOAD:
        return None
    return raw


def _declared_max_model_len() -> int:
    """The context window the engine will actually be started with.

    Read from ``SPEEDLM_E2E_VLLM_ARGS`` -- the argv this test hands vLLM -- and
    not from the launcher's ``--max-model-len``, which for this flavor reaches
    nothing.  An unbounded engine cannot be checked against a workload's
    requirement, so it refuses instead of guessing.
    """
    args = _vllm_args()
    for index, item in enumerate(args):
        if item == "--max-model-len" and index + 1 < len(args):
            raw = args[index + 1]
            break
        if item.startswith("--max-model-len="):
            raw = item.split("=", 1)[1]
            break
    else:
        raise AssertionError(
            "SPEEDLM_E2E_VLLM_ARGS declares no --max-model-len, so the workload's "
            "context requirement cannot be enforced; declare the window explicitly "
            "rather than let long prompts be truncated silently"
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise AssertionError(f"--max-model-len is not an integer: {raw!r}") from exc


def _workload_corpus(name: str) -> list[SeedRequest]:
    """Resolve ``SPEEDLM_E2E_WORKLOAD`` to a verified list of seed requests.

    The manifest's ``requirements.min_max_model_len`` is enforced against the
    engine argv through :func:`workloads.preflight_refusals` -- the same function
    the launch preflight uses -- so a run that would truncate the workload dies
    here, before the GPU is paid for, instead of producing numbers about a
    different corpus.
    """
    available = workloads.available_workloads()
    assert name in available, (
        f"SPEEDLM_E2E_WORKLOAD={name!r} is not a declared workload; "
        f"available: {', '.join(available)}"
    )
    assert not os.environ.get("SPEEDLM_E2E_PROMPT_CORPUS"), (
        f"SPEEDLM_E2E_WORKLOAD={name!r} and SPEEDLM_E2E_PROMPT_CORPUS are both set; "
        "the run would measure one of them and report the other"
    )
    spec = workloads.load_spec(name)
    refusals = workloads.preflight_refusals(
        spec,
        max_model_len=_declared_max_model_len(),
        # This test now posts the record's `tools` array verbatim, so tool
        # support is a property of the request it sends, not a promise.
        tool_support=True,
    )
    assert not refusals, "workload cannot be replayed as configured:\n  - " + (
        "\n  - ".join(refusals)
    )
    records = workloads.verify_workload(spec)
    return [
        _build_seed_request(
            {
                "messages": [dict(message) for message in record.messages],
                **({"tools": [dict(tool) for tool in record.tools]} if record.tools else {}),
            },
            context=f"workload {spec.name!r} record {record.id!r}",
        )
        for record in records
    ]


def _load_seed_corpus() -> list[SeedRequest] | None:
    """The seed corpus: the selected workload, else the legacy corpus, else none."""
    name = _selected_workload()
    if name is not None:
        return _workload_corpus(name)
    return _load_prompt_corpus()


def _select_requests(
    corpus: list[SeedRequest] | None, *, seed_count: int
) -> list[SeedRequest]:
    """Return ``seed_count`` requests, deterministically.

    When *corpus* is ``None``, fall back to the original synthetic template.
    When the corpus is too small, raise AssertionError with a clear message.
    """
    if corpus is None:
        return [
            _user_request(
                f"This is idle-tuning seed request {i + 1}/{seed_count}. "
                f"Reply with one short sentence."
            )
            for i in range(seed_count)
        ]
    if len(corpus) < seed_count:
        raise AssertionError(
            f"prompt corpus has {len(corpus)} prompts but "
            f"{seed_count} are needed; set SPEEDLM_E2E_SEED_REQUESTS <= {len(corpus)} "
            f"or use a larger corpus"
        )
    # Use a fixed-seed sample to draw a deterministic, well-spread subset across
    # the entire corpus.  A plain prefix (corpus[:seed_count]) is biased: the
    # ultrachat corpus can be topically clustered by index, so a prefix silently
    # narrows the distribution.  Random.sample with a dedicated instance (seed=42)
    # is O(corpus), requires no extra state, and yields the same subset every run
    # for a given seed_count — critical for reproducible comparison of results.
    return random.Random(42).sample(corpus, seed_count)


pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PREEMPTIBLE_STATES = frozenset(
    {
        "EXTRACTING",
        "TRAINING",
        "CANDIDATE_STARTING",
        "BENCHMARKING",
    }
)
# Outcomes that mean the gate measured both arms and reached a verdict.  A
# benchmark that timed out or was aborted produced no measurement at all, so it
# is deliberately absent: accepting it here is what let job 368959's exhausted
# 1800s deadline read as a terminal "rejected" result.
COMPLETE_OUTCOMES = frozenset({"promoted", "rejected"})
# Outcomes that must fail the run loudly rather than be waited out.
# ``benchmark_timed_out`` is in here because a deadline too small for the suite
# is a harness defect that will recur on every cycle; silently retrying it
# burns the whole e2e budget and then reports a timeout on the wrong thing.
FAILED_OUTCOMES = frozenset(
    {
        "failed",
        "final_assistant_mask_error",
        "benchmark_timed_out",
    }
)

# ── named failure diagnoses ─────────────────────────────────────────────────
#
# `CycleOutcome` has no member for a no-op training run: the orchestrator's
# catch-all `except Exception` maps `StockIdenticalDraftError` onto the generic
# `failed` outcome (src/speedlm/tuner/orchestrator.py:734-741), so the run
# reports as an opaque harness failure indistinguishable from an OOM or a dead
# subprocess. The distinguishing evidence *is* carried through, just not in the
# outcome: `CycleResult.error` is `_combine_error(exc, ...)`, i.e. `str(exc)`
# with the cleanup errors appended, and it reaches `scheduler.json` verbatim as
# `last_result["error"]` (src/speedlm/tuner/service.py:856). So the test can
# recover the distinct name from the message even though it may not edit src/.
#
# The marker is the tail of `StockIdenticalDraftError.__init__`'s message
# (src/speedlm/tuner/eagle3.py:188-191) and is unique in the tree.
STOCK_IDENTICAL_DRAFT_MARKER: Final = "training was a no-op"
STOCK_IDENTICAL_DRAFT_DIAGNOSIS: Final = "stock_identical_draft"

#: `(message marker, diagnosis name)` for failures the orchestrator can only
#: report as `failed`. Ordered; first match wins.
NAMED_FAILURE_DIAGNOSES: Final[tuple[tuple[str, str], ...]] = (
    (STOCK_IDENTICAL_DRAFT_MARKER, STOCK_IDENTICAL_DRAFT_DIAGNOSIS),
)


def _failure_diagnosis(result: Mapping[str, object]) -> str:
    """Name the failure behind a cycle result as precisely as the record allows.

    Falls back to the bare outcome when no marker matches, so an unrecognised
    failure is never silently renamed into a recognised one.
    """
    error = result.get("error")
    if isinstance(error, str):
        for marker, diagnosis in NAMED_FAILURE_DIAGNOSES:
            if marker in error:
                return diagnosis
    outcome = result.get("outcome")
    return outcome if isinstance(outcome, str) else repr(outcome)

# Verdicts reached after computing both deltas, i.e. the gate ran the whole
# comparison and judged it against the thresholds.  These are the only reasons
# for which `acceptance_delta_pp`, `accepted_length_delta`, and
# `throughput_delta_pct` are populated.  The gate now decides on
# `accepted_length_delta` against `min_accepted_length_delta`; the rate
# delta is still recorded but no longer gates.
DELTA_REASONS = frozenset(
    {
        "both_thresholds_met",
        "acceptance_below_threshold",
        "throughput_below_threshold",
    }
)

# Verdicts reached by measuring the arms and then short-circuiting *before* the
# deltas are computed.  `output_mismatch` is one: `decide_promotion` returns at
# the divergence check, which sits above the `acceptance_delta_pp` assignment,
# so both delta fields are null by design on this path.  That is not missing
# data -- the gate has evidence, it is just evidence of a different kind -- so
# these reasons are measured, and the evidence they *do* carry is asserted
# below instead of the deltas.
SHORT_CIRCUIT_MEASURED_REASONS = frozenset({"output_mismatch", "truncation_saturated"})

# `truncation_saturated` is the second.  It returns at the truncation check,
# which likewise sits above the delta computation, and it means something
# specific: both arms replayed the whole suite, and `benchmark_max_tokens` --
# not the model -- ended every single generation in one of them, so the run
# observed nothing about where this model stops and its throughput figure
# describes the cap rather than the workload.  See `TruncationRegime` in
# `src/speedlm/gate/decide.py`.  That is emphatically a measured outcome, and
# it must not be reported here as "the gate never compared the arms": the
# diagnosis a realistic-workload run most needs is "raise the cap", and the
# unmeasured-gate error would send the reader looking for a broken scrape.
# The evidence this reason carries is truncation counts, not divergences, so
# it is asserted on its own branch below.
TRUNCATION_REASON = "truncation_saturated"

# Deliberately NOT in either set above.  `truncation_unmeasured` is the gate
# saying no replayed response in an arm reported a finish reason at all, so it
# is exactly the "the gate returned a verdict without measuring" failure this
# file exists to make loud -- the opposite of `truncation_saturated`, which is a
# measured finding.  Leaving it out of MEASURED_REASONS means a live run that
# hits it fails here unless SPEEDLM_E2E_ALLOW_UNMEASURED_GATE=1, which is the
# intended behaviour: on a real vLLM endpoint every completed response carries a
# finish reason, so seeing this reason means the harness lost the field.
UNMEASURED_TRUNCATION_REASON = "truncation_unmeasured"
assert UNMEASURED_TRUNCATION_REASON not in SHORT_CIRCUIT_MEASURED_REASONS

# Verdicts reached by actually comparing the two arms.  Anything else means the
# gate rejected for want of data, which is a legitimate runtime outcome but not
# a passing end-to-end run: it proves the lifecycle turned over without proving
# the gate can measure.
MEASURED_REASONS = DELTA_REASONS | SHORT_CIRCUIT_MEASURED_REASONS
# Escape hatch for deliberately exercising an unmeasurable gate (e.g. a build
# with speculative decoding switched off).  Off by default so the silent
# zero-sample rejection that shipped cannot pass again.
ALLOW_UNMEASURED_GATE = os.environ.get("SPEEDLM_E2E_ALLOW_UNMEASURED_GATE") == "1"

JsonObject = dict[str, Any]


def _require_environment() -> tuple[Path, Path, Path | None]:
    if os.environ.get("SPEEDLM_E2E_IDLE_TUNING") != "1":
        pytest.skip("set SPEEDLM_E2E_IDLE_TUNING=1 in an accelerator job")

    config = _required_file("SPEEDLM_E2E_TUNING_CONFIG")
    artifact_root = _required_path("SPEEDLM_E2E_ARTIFACT_DIR")
    profile_raw = os.environ.get("SPEEDLM_E2E_TUNING_PROFILE")
    profile = None if profile_raw is None else Path(profile_raw).expanduser().resolve()
    if profile is not None:
        assert profile.is_file(), f"SPEEDLM_E2E_TUNING_PROFILE is not a file: {profile}"

    assert shutil.which("vllm"), "vllm must be available on PATH"
    return config, artifact_root, profile


def _required_file(name: str) -> Path:
    path = _required_path(name)
    assert path.is_file(), f"{name} is not a file: {path}"
    return path


def _required_path(name: str) -> Path:
    raw = os.environ.get(name)
    assert raw, f"{name} is required"
    return Path(raw).expanduser().resolve()


def _timeout(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be numeric, got {raw!r}") from exc
    assert value > 0, f"{name} must be positive, got {raw!r}"
    return value


def _seed_requests(config) -> int:
    raw = os.environ.get("SPEEDLM_E2E_SEED_REQUESTS")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as exc:
            raise AssertionError(
                f"SPEEDLM_E2E_SEED_REQUESTS must be an integer, got {raw!r}"
            ) from exc
        assert value > 0, f"SPEEDLM_E2E_SEED_REQUESTS must be positive, got {raw!r}"
        return value
    return max(config.tuning.min_trace_records, config.tuning.min_corpus_records)


def _vllm_args() -> list[str]:
    raw = os.environ.get("SPEEDLM_E2E_VLLM_ARGS", "[]")
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError("SPEEDLM_E2E_VLLM_ARGS must be a JSON array") from exc
    assert isinstance(value, list) and all(isinstance(item, str) for item in value), (
        "SPEEDLM_E2E_VLLM_ARGS must be a JSON array of strings"
    )
    return value


def _unique_artifact_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1_000):
        suffix = "" if attempt == 1 else f"-run{attempt}"
        candidate = root / f"live-idle-tuning{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError(f"could not allocate an artifact directory under {root}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_object(path: Path) -> JsonObject | None:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # The production writer uses atomic replacement.  Keep this tolerance
        # for network filesystems whose client cache may briefly expose stale
        # metadata during replacement.
        return None
    assert isinstance(value, dict), f"{path} must contain a JSON object"
    return value


def _wait_until(
    description: str,
    predicate: Callable[[], object | None],
    *,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise AssertionError(
                f"speedlm exited with code {returncode} while waiting for {description}"
            )
        result = predicate()
        if result is not None and result is not False:
            return result
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:g}s waiting for {description}")


def _wait_for_gateway(
    url: str,
    *,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> None:
    last_error = "not attempted"

    def ready() -> bool:
        nonlocal last_error
        try:
            response = httpx.get(f"{url}/health", timeout=2.0, trust_env=False)
        except httpx.HTTPError as exc:
            last_error = repr(exc)
            return False
        if 200 <= response.status_code < 300:
            return True
        last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        return False

    try:
        _wait_until("gateway readiness", ready, process=process, timeout=timeout)
    except AssertionError as exc:
        raise AssertionError(f"{exc}; last error: {last_error}") from exc


def _post_chat(
    gateway_url: str,
    config: SpeedLMConfig,
    request: SeedRequest,
    *,
    timeout: float,
) -> tuple[JsonObject, float]:
    payload: dict[str, Any] = {
        "model": config.alias,
        # The record's own turns, verbatim.  Synthesizing a single user message
        # here is what made every seed request single-turn no matter what the
        # corpus said.
        "messages": [dict(message) for message in request.messages],
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "seed": config.sampling.seed,
        # Reasoning models spend their opening tokens on the analysis channel, so
        # a tight cap truncates every reply mid-thought and never reaches a final
        # answer. Budget enough room to stop naturally while staying well inside
        # the tuner's training sequence length.
        "max_tokens": 512,
    }
    if request.tools:
        payload["tools"] = [dict(tool) for tool in request.tools]
    started = time.monotonic()
    response = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        trust_env=False,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    body: object = response.json()
    assert isinstance(body, dict), body
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, body
    assert isinstance(choices[0], dict), body
    # Accept any natural termination, or token-cap truncation.  A truncated
    # finish ("length") means the response hit max_tokens — this is normal
    # serving behaviour for longer prompts (e.g. ultrachat p95 ~2751 chars) and
    # produces valid training traces.  Reject genuinely bad terminal states.
    #
    # The hardcoded ``("stop", "length")`` this replaces was a self-inflicted
    # abort.  The payload above now ships the record's own tool schemas, so an
    # agentic seed can and does make the server answer with
    # ``finish_reason: "tool_calls"`` — a *complete* generation, the model
    # choosing to hand off.  The first seed that picked a tool therefore killed
    # the whole GPU run on an AssertionError, and the fix that added the tools
    # is what created the value the assertion rejected.
    #
    # The vocabulary is imported from ``speedlm.gate.replay`` rather than spelled
    # again here.  That module already owns both halves of this question, and
    # there are three deliberately-mirrored copies of the truncated set already;
    # ``traces.store`` documents its copy as restated "because the trace store
    # must not depend on a training backend", which is a layering constraint
    # this file does not have — it is a test, and importing the production
    # vocabulary is exactly how it stays honest about what the gate accepts.
    # So: no fourth copy, and no drift test needed for one.
    finish = choices[0].get("finish_reason")
    assert is_natural_stop_finish_reason(finish) or is_truncated_finish_reason(
        finish
    ), (
        f"unexpected finish_reason: {finish!r}; a usable response ends either "
        f"naturally ({', '.join(sorted(NATURAL_STOP_FINISH_REASONS))}) or at "
        f"the output cap ({', '.join(sorted(TRUNCATED_FINISH_REASONS))}); "
        f"got body: {body}"
    )
    usage = body.get("usage")
    assert isinstance(usage, dict), body
    assert isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
    assert (
        isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
    )
    return body, elapsed


def _trace_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    except FileNotFoundError:
        return 0


def _trace_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()
    ids: set[str] = set()
    for line in lines:
        if not line:
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            # Trace append is fsynced but a lock-free polling reader can observe
            # the final line between buffered writes. Retry on the next poll.
            return set()
        assert isinstance(value, dict), f"{path} must contain JSON objects"
        trace_id = value.get("id")
        assert isinstance(trace_id, str) and trace_id, f"trace has no valid id: {value}"
        ids.add(trace_id)
    return ids


def _scheduler_result_after(
    path: Path,
    *,
    after: float,
    accepted: frozenset[str],
) -> JsonObject | None:
    scheduler = _read_object(path)
    if scheduler is None:
        return None
    error = scheduler.get("last_error")
    result = scheduler.get("last_result")
    result_at = scheduler.get("last_result_at")
    if result is None:
        if error is not None:
            raise AssertionError(f"idle tuner failed outside a cycle: {error}")
        return None
    assert isinstance(result, dict), scheduler
    outcome = result.get("outcome")
    if outcome in FAILED_OUTCOMES:
        diagnosis = _failure_diagnosis(result)
        if diagnosis == STOCK_IDENTICAL_DRAFT_DIAGNOSIS:
            raise AssertionError(
                f"idle tuning cycle failed [{diagnosis}]: training was a no-op -- "
                "the materialized draft is byte-identical to the head it "
                "warm-started from, so the cycle learned nothing and the gate "
                "would only have been asked to distinguish a head from itself. "
                "Check the trainer actually stepped (lr, epochs, data volume) "
                f"before spending another GPU cycle. result={result}"
            )
        raise AssertionError(f"idle tuning cycle failed [{diagnosis}]: {result}")
    if (
        outcome in accepted
        and isinstance(result_at, (int, float))
        and not isinstance(result_at, bool)
        and float(result_at) > after
    ):
        return result
    return None


def _assert_gate_measured_something(decision: JsonObject) -> None:
    """Fail unless the gate reached its verdict by comparing real samples.

    A terminal `rejected` outcome is not on its own evidence that the gate
    works: the gate can reject because it collected nothing.  That is what
    shipped -- `num_repeats: 0`, `per_repeat: []`, both arms at 0.0 tok/s,
    `reason: acceptance_unavailable` -- while the lifecycle test passed.
    """
    reason = decision.get("reason")
    num_repeats = decision.get("num_repeats")
    per_repeat = decision.get("per_repeat")

    assert isinstance(per_repeat, list), decision
    assert isinstance(num_repeats, int) and not isinstance(num_repeats, bool), decision
    assert num_repeats == len(per_repeat), decision

    if reason not in MEASURED_REASONS:
        if ALLOW_UNMEASURED_GATE:
            return
        raise AssertionError(
            "the gate returned a verdict without comparing the two arms "
            f"(reason={reason!r}, num_repeats={num_repeats}); set "
            "SPEEDLM_E2E_ALLOW_UNMEASURED_GATE=1 only when that is expected. "
            f"decision={decision}"
        )

    assert num_repeats > 0, decision
    if reason in DELTA_REASONS:
        assert decision.get("acceptance_delta_pp") is not None, decision
        assert decision.get("accepted_length_delta") is not None, decision
        assert decision.get("throughput_delta_pct") is not None, decision
        assert decision.get("acceptance_criterion") == "mean_accepted_length_delta", decision
    elif reason == TRUNCATION_REASON:
        # Short-circuited at the truncation check.  Same contract as the
        # divergence branch below -- require the evidence that justified the
        # short circuit -- but the evidence is the finish-reason counts.
        #
        # An arm qualifies only by reporting finish reasons and having no
        # natural stop among them, so demand exactly that: some arm reported,
        # and some arm truncated everything it reported.  Asserting it here is
        # what stops `truncation_saturated` from becoming a way for a gate that
        # collected nothing to pass under a measured-looking reason -- a run
        # that reported no finish reason at all classifies as `untestable` and
        # can never reach this branch.
        rates = (
            decision.get("stock_truncation_rate"),
            decision.get("candidate_truncation_rate"),
        )
        regimes = (
            decision.get("stock_truncation_regime"),
            decision.get("candidate_truncation_regime"),
        )
        assert "saturated" in regimes, decision
        assert any(rate == 1.0 for rate in rates), decision
        # The counts the rates were derived from, so the record is reconcilable
        # by hand rather than only by trusting the derived field.
        assert any(
            row.get("stock_finish_reasons", 0) > 0
            or row.get("candidate_finish_reasons", 0) > 0
            for row in per_repeat
            if isinstance(row, dict)
        ), decision
    else:
        # Short-circuited above the delta computation.  Requiring the deltas
        # here would be unsatisfiable, so require the evidence that actually
        # justified the short circuit -- otherwise this branch would let a gate
        # that measured nothing through under a measured-looking reason.
        assert decision.get("output_early_divergences", 0) > 0, decision
        divergences = decision.get("output_divergences")
        assert isinstance(divergences, list) and divergences, decision
        assert any(d.get("early") for d in divergences), decision
        # The correctness pass is a different replay from the scored repeats.
        # Pin that the record says how many passes it made, so a reader cannot
        # mistake the zeros in the `per_repeat` `output_mismatches` column for
        # clean correctness passes that never ran.
        correctness_repeats = decision.get("correctness_repeats")
        assert (
            isinstance(correctness_repeats, int)
            and not isinstance(correctness_repeats, bool)
            and correctness_repeats > 0
        ), decision
        assert all(
            row.get("output_mismatches", 0) == 0
            for row in per_repeat[correctness_repeats:]
            if isinstance(row, dict)
        ), decision
    # Throughput is measured whenever the arms ran at all, so require it on
    # both.  Acceptance may legitimately be a measured zero for one arm (a head
    # whose every draft is rejected), but not for both -- that is the all-zero
    # signature of a scrape that matched nothing.
    for arm in ("stock_avg_tok_per_sec", "candidate_avg_tok_per_sec"):
        value = decision.get(arm)
        assert isinstance(value, (int, float)) and value > 0.0, decision
    acceptances = [
        decision.get("stock_avg_acceptance"),
        decision.get("candidate_avg_acceptance"),
    ]
    assert all(isinstance(v, (int, float)) for v in acceptances), decision
    assert any(v > 0.0 for v in acceptances if isinstance(v, (int, float))), decision
    # The gating criterion is mean accepted length, not acceptance rate.
    # Assert that its per-arm means are populated and > 1.0 (the floor).
    for field in ("stock_avg_accepted_length", "candidate_avg_accepted_length"):
        value = decision.get(field)
        assert isinstance(value, (int, float)) and value > 1.0, decision


#: What `Eagle3Converter._build_eagle3_speculator_config` hardcodes into every
#: converted stock EAGLE-3 drafter's
#: `speculators_config.proposal_methods[0].speculative_tokens`
#: (speculators `src/speculators/convert/eagle/eagle3_converter.py:118`,
#: `speculative_tokens=3`). Used only as the fallback expectation when the
#: warm-start drafter is not resolvable in the local Hub cache.
CONVERTER_INHERITED_SPECULATIVE_TOKENS: Final = 3


def _resolved_profile(config: SpeedLMConfig, home: Path) -> ModelProfile:
    """Resolve the profile the run actually served, the way the launcher does.

    Mirrors `build_tuning_launch_plan` (src/speedlm/tuner/composition.py:378)
    exactly -- same argument shape, same `home` -- so a custom profile copied
    into `<home>/profiles` by `_copy_profile` is resolved here too. Re-deriving
    the depth from anything else (a hardcoded 5, the built-in registry) would
    assert against a different profile than the one under test.
    """
    return resolve_profile(
        {"model": config.model, "profile": config.profile},
        served_model=config.model,
        home=home,
    )


def _manifest_training_params(manifest_path: Path) -> JsonObject:
    """Return the published manifest's `training_params`, or fail loudly."""
    manifest = _read_object(manifest_path)
    assert manifest is not None, f"artifact manifest is missing or unreadable: {manifest_path}"
    params = manifest.get("training_params")
    assert isinstance(params, dict), (
        f"artifact manifest has no training_params object: {manifest_path} -> {manifest}"
    )
    return params


def _assert_trained_at_serving_depth(
    params: Mapping[str, object], *, serving_depth: int
) -> int:
    """Fail unless the cycle trained the chain depth the profile serves.

    This is the assertion the depth-alignment commit exists to protect: before
    it, the trainer hard-required 3 TTT steps while the profile served 5, so
    positions 4-5 were pure extrapolation and nothing recorded the disagreement.
    `Eagle3Adapter.describe` now stamps the trained depth into the manifest
    (src/speedlm/tuner/eagle3.py:739), which is what makes the comparison
    possible at all.
    """
    trained = params.get("num_speculative_steps")
    assert isinstance(trained, int) and not isinstance(trained, bool), (
        "the artifact manifest records no integer num_speculative_steps, so "
        "nothing states the depth this cycle trained at; training_params="
        f"{dict(params)}"
    )
    assert trained == serving_depth, (
        f"training depth {trained} does not match the profile's serving depth "
        f"{serving_depth}: the head was fitted for a {trained}-deep chain and "
        f"will be rolled out {serving_depth} deep, so positions "
        f"{trained + 1}..{serving_depth} are extrapolation. "
        f"training_params={dict(params)}"
    )
    return trained


def _assert_training_changed_the_weights(
    params: Mapping[str, object],
) -> tuple[str, str]:
    """Fail unless the published head differs from its warm-start head.

    Identical fingerprints mean the cycle was a no-op: every tensor came out
    exactly as it went in, and the gate is being asked to distinguish a head
    from itself. `Eagle3Adapter._record_draft_weights` raises
    `StockIdenticalDraftError` on that case in production, but nothing read the
    two fingerprints back out of the manifest, so a *null* baseline -- which
    means the comparison was never made at all -- was indistinguishable from a
    comparison that passed.
    """
    fingerprint = params.get("draft_weight_fingerprint")
    baseline = params.get("draft_weight_baseline_fingerprint")
    assert isinstance(fingerprint, str) and fingerprint, (
        "the artifact manifest carries no draft_weight_fingerprint, so nothing "
        "ties the published weights to the weights that were trained; "
        f"training_params={dict(params)}"
    )
    assert isinstance(baseline, str) and baseline, (
        "the artifact manifest carries a null draft_weight_baseline_fingerprint: "
        "the warm-start head could not be resolved to local weights, so the "
        "no-op check was never made. That is not a passing comparison, it is an "
        f"absent one; training_params={dict(params)}"
    )
    assert fingerprint != baseline, (
        "the published head is byte-identical to the head it warm-started from "
        f"(fingerprint {fingerprint}): training was a no-op; "
        f"training_params={dict(params)}"
    )
    return fingerprint, baseline


def _artifact_declared_depth(published_dir: Path) -> tuple[JsonObject, int | None]:
    """Read the chain depth the published artifact's own config.json declares.

    `ArtifactRegistry.publish` copytrees the materialized draft directory into
    `runs/artifacts/<id>/`, so the draft's `config.json` sits at that
    directory's top level beside `manifest.json` (verified against the real
    published artifacts under `log_artifacts/`).
    """
    config_path = published_dir / "config.json"
    assert config_path.is_file(), (
        f"the published artifact has no config.json: {config_path}"
    )
    raw = _read_object(config_path)
    assert raw is not None, f"the published artifact's config.json is unreadable: {config_path}"
    return raw, drafter_declared_speculative_tokens(raw)


def _assert_artifact_declares_expected_depth(
    declared: int | None,
    *,
    serving_depth: int,
    stock_declared: int | None,
) -> str:
    """Check the depth read back out of the artifact the trainer produced.

    KNOWN-STALE FIELD -- read this before changing the assertion.

    Speculators does not rewrite `speculators_config.proposal_methods[*]
    .speculative_tokens` on the `--from-pretrained` path. `scripts/train.py`
    takes `model_class.from_pretrained(args.from_pretrained, ...)` (line 345-347)
    which loads the warm-start checkpoint's config verbatim; only the
    from-scratch branch, `Eagle3DraftModel.from_training_args`, writes
    `speculative_tokens=kwargs["ttt_steps"]`
    (`src/speculators/models/eagle3/core.py:317`). `save_pretrained` then writes
    that inherited config back out, and SpeedLM never post-processes it. The
    stock drafters were produced by `Eagle3Converter`, which hardcodes
    `speculative_tokens=3` (`convert/eagle/eagle3_converter.py:118`).

    Confirmed empirically on this host: the published artifact
    `log_artifacts/live-idle-validate5-20260730T232443Z/.../runs/artifacts/
    00c3fd3c.../config.json` declares `speculative_tokens: 3` under the
    `gpt-oss-20b-eagle3` profile, whose `num_speculative_tokens` is 5.

    So asserting `declared == serving_depth` would be an assertion that fails
    for a *correct* pipeline, and asserting `declared == 3` would be an
    assertion that passes for the wrong reason once Speculators is fixed.
    Instead this pins the field to exactly one of two truthful readings, and
    fails on any third:

    * `declared == serving_depth` -- the field became truthful. Reported as
      ``"truthful"``; nothing to do but delete the stale branch.
    * `declared == stock_declared` -- the documented staleness: the field is
      the warm start's inherited declaration, unchanged. Reported as
      ``"known_stale"``.

    Any other value means the depth in the artifact came from a third,
    unaccounted-for place, which is exactly the silent drift this check exists
    to catch. Depth *alignment* is still measured -- by
    `_assert_trained_at_serving_depth` against the manifest, which SpeedLM does
    write from the resolved depth.

    NOT DISCRIMINATING when `serving_depth == stock_declared` (e.g. the
    `qwen3-8b-eagle3` profile, which serves 3 and warm-starts from a drafter
    declaring 3): both branches expect the same number, so a ``"truthful"``
    verdict there is no evidence the field was rewritten. The recorded
    observation carries a `discriminating` flag so a reader cannot mistake that
    for proof. Only a profile whose serving depth differs from its warm start's
    declaration -- `gpt-oss-20b-eagle3`, 5 against 3 -- tests this at all.
    """
    assert declared is not None, (
        "the published artifact's config.json declares no usable "
        "speculators_config.proposal_methods[*].speculative_tokens, so the "
        "artifact states no chain depth at all (missing, non-integer, or "
        "several proposal methods disagreeing)"
    )
    if declared == serving_depth:
        return "truthful"
    expected_stale = (
        stock_declared
        if stock_declared is not None
        else CONVERTER_INHERITED_SPECULATIVE_TOKENS
    )
    assert declared == expected_stale, (
        f"the published artifact declares speculative_tokens={declared}, which is "
        f"neither the profile's serving depth ({serving_depth}) nor the depth "
        f"inherited unchanged from the warm-start drafter ({expected_stale}). "
        "A third value means the depth stamped into the artifact came from "
        "somewhere unaccounted for -- neither the profile nor the warm start -- "
        "and the served chain depth can no longer be reasoned about from the "
        "artifact"
    )
    return "known_stale"


def _collect_gate_metrics(run_dir: Path, artifact_dir: Path) -> None:
    """Copy the gate's raw Prometheus bodies out with the run's artifacts.

    Without these the reported acceptance and throughput can only be taken on
    trust: they are deltas, and the counters behind them live nowhere else once
    the child vLLM exits.
    """
    source = run_dir / "gate-metrics"
    assert source.is_dir(), f"gate left no raw metrics under {source}"
    bodies = sorted(source.glob("*.prom.gz"))
    assert bodies, f"gate metrics directory {source} is empty"
    destination = artifact_dir / "gate-metrics"
    destination.mkdir(parents=True, exist_ok=True)
    for body in bodies:
        shutil.copy2(body, destination / body.name)


def _copy_profile(profile: Path | None, home: Path) -> None:
    if profile is None:
        return
    profiles = home / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile, profiles / profile.name)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descendant_pids(root_pid: int) -> set[int]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    children: dict[int, set[int]] = {}
    for line in completed.stdout.splitlines():
        raw_pid, raw_parent = line.split()
        children.setdefault(int(raw_parent), set()).add(int(raw_pid))
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), set()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _assert_processes_gone(pids: set[int], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = {pid for pid in pids if Path(f"/proc/{pid}").exists()}
        if not alive:
            return
        time.sleep(0.1)
    pytest.fail(f"orphaned child processes remain: {sorted(alive)}")


def test_live_idle_tuning_preempts_then_completes() -> None:
    config_path, artifact_root, profile = _require_environment()
    config = load_config(config_path)
    artifact_dir = _unique_artifact_dir(artifact_root)
    home = artifact_dir / "speedlm_home"
    home.mkdir()
    _copy_profile(profile, home)

    ready_timeout = _timeout("SPEEDLM_E2E_READY_TIMEOUT", 900.0)
    tuning_timeout = _timeout("SPEEDLM_E2E_TUNING_TIMEOUT", 7_200.0)
    request_timeout = _timeout("SPEEDLM_E2E_REQUEST_TIMEOUT", 1_200.0)
    seed_count = _seed_requests(config)
    port = _free_port()
    gateway_url = f"http://127.0.0.1:{port}"
    gateway_log = artifact_dir / "gateway-and-vllm.log"
    command = [
        sys.executable,
        "-m",
        "speedlm.cli",
        "--home",
        str(home),
        "vllm",
        "serve",
        config.model,
        "--config",
        str(config_path),
        "--enable-idle-tuning",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *_vllm_args(),
    ]
    _write_json(
        artifact_dir / "invocation.json",
        {
            "argv": command,
            "config": str(config_path),
            "custom_profile": str(profile) if profile is not None else None,
            "model": config.model,
            "model_alias": config.alias,
            "min_trace_records": config.tuning.min_trace_records,
            "hostname": socket.gethostname(),
        },
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SPEEDLM_HOME": str(home),
        }
    )
    log_handle = gateway_log.open("wb")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    observed_pids: set[int] = set()
    try:
        _wait_for_gateway(gateway_url, process=process, timeout=ready_timeout)
        observed_pids.update(_descendant_pids(process.pid))
        assert observed_pids, "speedlm did not launch a vLLM child"

        traces_path = home / "traces" / "traces.jsonl"
        seed_corpus = _select_requests(
            _load_seed_corpus(), seed_count=seed_count
        )
        assert len(seed_corpus) == seed_count
        for index, seed_request in enumerate(seed_corpus):
            body, _ = _post_chat(
                gateway_url,
                config,
                seed_request,
                timeout=request_timeout,
            )
            _write_json(artifact_dir / f"seed-response-{index + 1:04d}.json", body)

        _wait_until(
            "all seed traces to be captured",
            lambda: (
                _trace_count(traces_path)
                if _trace_count(traces_path) >= seed_count
                else None
            ),
            process=process,
            timeout=60.0,
        )

        state_path = home / "runs" / "state.json"
        active_state = _wait_until(
            "vLLM to sleep and the first cycle to become preemptible",
            lambda: (
                state
                if (state := _read_object(state_path)) is not None
                and state.get("state") in ACTIVE_PREEMPTIBLE_STATES
                else None
            ),
            process=process,
            timeout=tuning_timeout,
        )
        assert isinstance(active_state, dict)
        preempt_started_at = time.time()
        queued_body, queued_seconds = _post_chat(
            gateway_url,
            config,
            _user_request(
                "Preempt the sleeping idle tuner, restore serving, and reply with READY."
            ),
            timeout=request_timeout,
        )
        queued_id = queued_body.get("id")
        assert isinstance(queued_id, str) and queued_id, queued_body
        _write_json(artifact_dir / "queued-response.json", queued_body)
        _write_json(
            artifact_dir / "preemption-observation.json",
            {
                "state_when_submitted": active_state.get("state"),
                "queued_request_seconds": queued_seconds,
            },
        )

        scheduler_path = home / "runs" / "scheduler.json"
        preempted = _wait_until(
            "the first cycle to report transparent preemption",
            lambda: _scheduler_result_after(
                scheduler_path,
                after=preempt_started_at,
                accepted=frozenset({"preempted"}),
            ),
            process=process,
            timeout=request_timeout,
        )
        assert isinstance(preempted, dict)

        _wait_until(
            "the queued request to be captured",
            lambda: (
                queued_id if queued_id in _trace_ids(traces_path) else None
            ),
            process=process,
            timeout=60.0,
        )

        terminal_after = time.time()
        terminal = _wait_until(
            "the next trace watermark to complete training and its held-out gate",
            lambda: _scheduler_result_after(
                scheduler_path,
                after=terminal_after,
                accepted=COMPLETE_OUTCOMES,
            ),
            process=process,
            timeout=tuning_timeout,
        )
        assert isinstance(terminal, dict)
        artifact_id = terminal.get("artifact_id")
        decision_path = terminal.get("decision_path")
        assert isinstance(artifact_id, str) and artifact_id, terminal
        assert isinstance(decision_path, str) and Path(decision_path).is_file(), terminal
        published_dir = home / "runs" / "artifacts" / artifact_id
        manifest_path = published_dir / "manifest.json"
        assert manifest_path.is_file()

        # ── depth alignment and weight provenance ───────────────────────────
        # The profile the run actually used, resolved from the same config and
        # the same SPEEDLM_HOME the gateway was launched with, so a custom
        # SPEEDLM_E2E_TUNING_PROFILE is honoured here as well.
        profile = _resolved_profile(config, home)
        serving_depth = profile.num_speculative_tokens
        training_params = _manifest_training_params(manifest_path)
        _assert_trained_at_serving_depth(training_params, serving_depth=serving_depth)
        _assert_training_changed_the_weights(training_params)

        raw_draft_config, declared_depth = _artifact_declared_depth(published_dir)
        stock_declared = declared_draft_depth(profile.draft_model)
        # Recorded before the assertion so the observed value survives into the
        # run artifacts even when the assertion below fails.
        _write_json(
            artifact_dir / "artifact-depth-observation.json",
            {
                "artifact_id": artifact_id,
                "profile": profile.name,
                "profile_num_speculative_tokens": serving_depth,
                "manifest_num_speculative_steps": training_params.get(
                    "num_speculative_steps"
                ),
                "artifact_declared_speculative_tokens": declared_depth,
                "stock_drafter": profile.draft_model,
                "stock_declared_speculative_tokens": stock_declared,
                # False when the serving depth and the warm start's inherited
                # declaration are the same number: the artifact-declared depth
                # then cannot distinguish "Speculators rewrote the field" from
                # "Speculators left it alone", so a truthful-looking reading
                # proves nothing.
                "discriminating": (
                    stock_declared is not None and stock_declared != serving_depth
                ),
                "draft_weight_fingerprint": training_params.get(
                    "draft_weight_fingerprint"
                ),
                "draft_weight_baseline": training_params.get("draft_weight_baseline"),
                "draft_weight_baseline_fingerprint": training_params.get(
                    "draft_weight_baseline_fingerprint"
                ),
                "speculators_config": raw_draft_config.get("speculators_config"),
            },
        )
        _assert_artifact_declares_expected_depth(
            declared_depth,
            serving_depth=serving_depth,
            stock_declared=stock_declared,
        )

        decision = _read_object(Path(decision_path))
        assert decision is not None, decision_path
        _write_json(artifact_dir / "terminal-decision.json", decision)
        _assert_gate_measured_something(decision)
        _collect_gate_metrics(Path(decision_path).parent, artifact_dir)
        if terminal["outcome"] == "promoted":
            active = _read_object(home / "runs" / "active.json")
            assert active is not None and active.get("artifact_id") == artifact_id

        scheduler = _read_object(scheduler_path)
        state = _read_object(state_path)
        assert scheduler is not None
        assert state is not None and state.get("state") == "READY", state
        _write_json(artifact_dir / "terminal-scheduler.json", scheduler)
        _write_json(artifact_dir / "terminal-state.json", state)
        events = home / "runs" / "events.jsonl"
        assert events.is_file() and events.stat().st_size > 0
    finally:
        if process.poll() is None:
            observed_pids.update(_descendant_pids(process.pid))
            os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=90.0)
        except subprocess.TimeoutExpired:
            observed_pids.update(_descendant_pids(process.pid))
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=30.0)
        log_handle.close()
        _write_json(
            artifact_dir / "shutdown.json",
            {
                "gateway_pid": process.pid,
                "observed_descendant_pids": sorted(observed_pids),
                "gateway_returncode": returncode,
            },
        )
        _assert_processes_gone(observed_pids)
