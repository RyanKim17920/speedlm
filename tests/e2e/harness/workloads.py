"""Declarative, verifiable benchmark workloads.

WHY THIS EXISTS
---------------
Every measurement this project has published so far was taken on ONE corpus:
``/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl`` -- 22,362 single-turn,
user-only chat prompts with no system prompts, no tool calls and no assistant
turns.  Worse, ``test_inference_configuration_matrix._prompts_by_context`` built
its "short" band from the eight *shortest* prompts in that corpus (mean ~16
tokens) and synthesized its "long" band by repetition-padding a single prompt out
to exactly 6000 characters.  Both moves are measurement errors, not shortcuts:

* the extremes of a distribution are not a sample of it -- a band taken from the
  tail measures the tail, and the eight shortest prompts of anything are
  degenerate;
* repetition-padded text is maximally predictable, so it flatters *any* drafter.
  A speculative decoding benchmark whose long band is a repeated string is
  measuring the repetition, not the drafter.

This module replaces that with workloads that are declared in JSON, materialized
by ``scripts/build_workload.py``, banded by percentile *window*, and -- the part
that actually matters -- **verified against the bytes on disk**.

THE ANTI-VACUITY RULE
---------------------
:func:`verify_workload` recomputes every declared characteristic from the corpus
file and fails when any of them disagrees beyond the manifest's declared
tolerance, and fails when the file digest does not match.  A corpus that was
swapped, truncated, re-shuffled or regenerated makes the workload go red.  This
is deliberately the most paranoid function in the package: the repository's
recurring defect is a green test that measures nothing, and a workload manifest
whose numbers were hand-typed is exactly that defect in data form.

Verification needs no tokenizer and no GPU.  Per-record token counts are baked
into the materialized records by the builder (which *does* use a real tokenizer)
and are covered by the file digest, so CI recomputes percentiles from stored
integers rather than trusting a summary somebody typed.

BAND SEMANTICS
--------------
A band is a percentile *window* over the workload's own prompt-length
distribution, e.g. ``short = [0.05, 0.25]``.  Records are never truncated to fit
a band and never repetition-padded.  If a band cannot supply the requested number
of *distinct* records, :class:`BandExhaustedError` is raised with both counts --
silently returning fewer would turn a sampling bug into a quiet loss of
statistical power.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "RECORD_FORMAT",
    "PERCENTILE_KEYS",
    "WorkloadError",
    "WorkloadVerificationError",
    "BandExhaustedError",
    "Band",
    "WorkloadRecord",
    "WorkloadSpec",
    "spec_directory",
    "available_workloads",
    "load_spec",
    "load_spec_file",
    "load_records",
    "flatten_prompt",
    "content_text",
    "percentile",
    "recompute_characteristics",
    "verify_workload",
    "band_records",
    "band_prompts",
    "band_messages",
    "prompts_by_context",
    "messages_by_context",
    "preflight_refusals",
    "file_sha256",
    "stable_json",
]

#: Manifest schema version.  Bumped when a manifest field changes meaning.
SCHEMA_VERSION = 1

#: Format tag of the materialized records file a manifest may point at.
RECORD_FORMAT = "speedlm.workload.records.v1"

#: Percentiles declared for every length distribution.  Fixed rather than
#: free-form so two manifests are always comparable, and so a manifest cannot
#: quietly drop the percentile that would have disagreed.
PERCENTILE_KEYS: tuple[str, ...] = (
    "p0",
    "p5",
    "p25",
    "p40",
    "p50",
    "p60",
    "p75",
    "p90",
    "p95",
    "p99",
    "p100",
)

_FRACTION_KEYS: tuple[str, ...] = (
    "tool_call_fraction",
    "tools_declared_fraction",
    "system_prompt_fraction",
    "multi_turn_fraction",
    "assistant_completion_fraction",
    "distinct_prompt_fraction",
)

_COUNT_KEYS: tuple[str, ...] = (
    "record_count",
    "unique_system_prompts",
    "unique_first_user_messages",
)


class WorkloadError(Exception):
    """Base class for every failure this module raises."""


class WorkloadVerificationError(WorkloadError):
    """A manifest's declared characteristics disagree with the corpus on disk.

    Carries *every* disagreement, not just the first.  A single mismatch usually
    means several -- a truncated file moves the record count and every
    percentile at once -- and reporting one at a time turns one investigation
    into five.
    """

    def __init__(self, workload: str, failures: Sequence[str]) -> None:
        self.workload = workload
        self.failures = tuple(failures)
        joined = "\n  - ".join(self.failures)
        super().__init__(f"workload {workload!r} failed verification:\n  - {joined}")


class BandExhaustedError(WorkloadError):
    """A band was asked for more distinct records than its window contains."""

    def __init__(self, workload: str, band: str, requested: int, available: int) -> None:
        self.workload = workload
        self.band = band
        self.requested = requested
        self.available = available
        super().__init__(
            f"workload {workload!r} band {band!r}: requested {requested} distinct "
            f"records but the percentile window holds only {available}. "
            "Widen the band, use a larger workload, or ask for fewer -- the band "
            "will not be topped up with duplicates, truncated records or padding."
        )


@dataclass(frozen=True, slots=True)
class Band:
    """A percentile window over the workload's prompt-length distribution."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise WorkloadError(
                f"band {self.name!r} bounds must satisfy 0 <= lower < upper <= 1; "
                f"got [{self.lower}, {self.upper}]"
            )


@dataclass(frozen=True, slots=True)
class WorkloadRecord:
    """One request in a workload.

    ``messages`` is the full OpenAI-format prompt prefix (system + history +
    the final user/tool turn).  ``completion`` is the reference next assistant
    turn where the source corpus recorded one; it is what makes a decode-length
    distribution recoverable rather than invented.
    """

    id: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    prompt_chars: int
    prompt_tokens: int
    completion: dict[str, Any] | None = None
    timestamps: tuple[str, ...] | None = None
    ground_truth: dict[str, Any] | None = None

    @property
    def prompt_text(self) -> str:
        """Deterministic flat rendering of the prompt prefix."""
        return flatten_prompt(self.messages, self.tools)

    @property
    def final_user_text(self) -> str:
        """The last user-authored message, or the flat prompt when there is none.

        The drop-in :func:`prompts_by_context` interface hands the config matrix
        one string per request.  For a single-turn corpus that string is the user
        turn.  For agentic traffic there is no honest single string, which is why
        :func:`messages_by_context` exists; this property is the lossy fallback
        and is documented as such in every agentic manifest.
        """
        for message in reversed(self.messages):
            if message.get("role") == "user":
                text = content_text(message.get("content"))
                if text:
                    return text
        return self.prompt_text


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """A parsed workload manifest.

    The raw manifest is kept whole in :attr:`manifest` so a consumer can read a
    field this class does not model yet without the manifest having to be
    re-parsed somewhere else with different rules.
    """

    name: str
    version: str
    description: str
    domain: str
    manifest_path: Path
    manifest: Mapping[str, Any]

    # -- source -----------------------------------------------------------
    @property
    def source_path(self) -> Path:
        return Path(self.manifest["source"]["path"])

    @property
    def source_sha256(self) -> str:
        return str(self.manifest["source"]["sha256"])

    @property
    def source_size_bytes(self) -> int:
        return int(self.manifest["source"]["size_bytes"])

    @property
    def source_format(self) -> str:
        return str(self.manifest["source"]["format"])

    # -- declared content -------------------------------------------------
    @property
    def characteristics(self) -> Mapping[str, Any]:
        return self.manifest["characteristics"]

    @property
    def tolerance(self) -> Mapping[str, Any]:
        return self.manifest["tolerance"]

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.manifest["provenance"]

    @property
    def requirements(self) -> Mapping[str, Any]:
        return self.manifest["requirements"]

    @property
    def ground_truth(self) -> Mapping[str, Any] | None:
        value = self.manifest.get("ground_truth")
        return value if isinstance(value, Mapping) else None

    @property
    def band_metric(self) -> str:
        return str(self.manifest.get("band_metric", "prompt_chars"))

    @property
    def seed(self) -> int:
        return int(self.manifest.get("sampling", {}).get("seed", 0))

    @property
    def bands(self) -> dict[str, Band]:
        return {
            name: Band(name, float(bounds[0]), float(bounds[1]))
            for name, bounds in self.manifest["bands"].items()
        }

    def band(self, name: str) -> Band:
        try:
            bounds = self.manifest["bands"][name]
        except KeyError:
            known = ", ".join(sorted(self.manifest["bands"])) or "(none)"
            raise WorkloadError(
                f"workload {self.name!r} has no band {name!r}; declared bands: {known}"
            ) from None
        return Band(name, float(bounds[0]), float(bounds[1]))


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def stable_json(value: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace, unicode preserved."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def content_text(content: Any) -> str:
    """Render OpenAI message content, string form or content-part list form.

    The list form is not decorative: ``agentic-mixed-outcome`` stores every user
    turn as ``[{"type": "text", "text": ...}]``, and an early version of this
    module rendered that as the empty string -- silently deleting the user's
    request from the prompt and understating its length.  Unknown part types are
    serialized rather than dropped, because a part that costs tokens must cost
    characters here too.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif item is not None:
            parts.append(stable_json(item))
    return "\n".join(parts)


def flatten_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Render a prompt prefix to one deterministic string.

    Tool schemas are included.  They are sent on the wire and they cost tokens;
    a character count that ignores them understates agentic prompts by a wide
    margin (measured 2.5 chars/token on the agentic corpora against 4.0 for
    generic chat, almost entirely because of tool JSON).

    Angle brackets are deliberately avoided in the separators so the rendering
    can be printed into any transcript without colliding with chat control
    tokens.
    """
    parts: list[str] = []
    if tools:
        parts.append("tools: " + stable_json(list(tools)))
    for message in messages:
        role = str(message.get("role", ""))
        segment = [f"{role}: {content_text(message.get('content'))}"]
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            segment.append("thinking: " + thinking)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            segment.append("tool_calls: " + stable_json(tool_calls))
        name = message.get("name")
        if isinstance(name, str) and name:
            segment.append("name: " + name)
        parts.append("\n".join(segment))
    return "\n\n".join(parts)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile on an already-collected sample.

    Nearest-rank rather than interpolated so the result is always a value that
    actually occurs in the corpus.  An interpolated p50 of an integer length
    distribution is a length no record has, which then cannot be asserted
    against anything real.
    """
    if not values:
        raise WorkloadError("percentile of an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise WorkloadError(f"quantile must be in [0, 1]; got {quantile}")
    ordered = sorted(values)
    index = int(round(quantile * (len(ordered) - 1)))
    return ordered[index]


def _percentile_block(values: Sequence[float]) -> dict[str, float]:
    return {key: percentile(values, int(key[1:]) / 100.0) for key in PERCENTILE_KEYS}


# ---------------------------------------------------------------------------
# Manifest and record loading
# ---------------------------------------------------------------------------


def spec_directory() -> Path:
    return Path(__file__).resolve().parent / "workload_specs"


def available_workloads(*, directory: Path | None = None) -> tuple[str, ...]:
    root = directory or spec_directory()
    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob("*.json")))


def load_spec_file(path: Path) -> WorkloadSpec:
    """Parse one manifest file.  Structural problems fail here, not at use."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WorkloadError(f"no workload manifest at {path}") from None
    except json.JSONDecodeError as error:
        raise WorkloadError(f"workload manifest {path} is not valid JSON: {error}") from None
    if not isinstance(raw, dict):
        raise WorkloadError(f"workload manifest {path} is not a JSON object")

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise WorkloadError(
            f"workload manifest {path} declares schema_version {schema_version!r}; "
            f"this module understands {SCHEMA_VERSION}"
        )
    required = (
        "name",
        "version",
        "description",
        "domain",
        "source",
        "provenance",
        "characteristics",
        "tolerance",
        "requirements",
        "bands",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise WorkloadError(f"workload manifest {path} is missing keys: {', '.join(missing)}")
    if not raw["bands"]:
        raise WorkloadError(f"workload manifest {path} declares no bands")

    spec = WorkloadSpec(
        name=str(raw["name"]),
        version=str(raw["version"]),
        description=str(raw["description"]),
        domain=str(raw["domain"]),
        manifest_path=Path(path),
        manifest=raw,
    )
    # Force band validation now so malformed bounds cannot survive to sampling.
    _ = spec.bands
    return spec


def load_spec(name: str, *, directory: Path | None = None) -> WorkloadSpec:
    """Load a workload by name.  Adding a workload is adding a JSON file."""
    root = directory or spec_directory()
    path = root / f"{name}.json"
    if not path.is_file():
        known = ", ".join(available_workloads(directory=root)) or "(none)"
        raise WorkloadError(f"unknown workload {name!r}; available: {known}")
    spec = load_spec_file(path)
    if spec.name != name:
        raise WorkloadError(
            f"workload manifest {path} declares name {spec.name!r} but is filed as {name!r}"
        )
    return spec


def _coerce_message(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("role"), str):
        raise WorkloadError(f"{context}: message is not an object with a string role")
    return dict(value)


def load_records(spec: WorkloadSpec) -> tuple[WorkloadRecord, ...]:
    """Read the materialized records file a manifest points at.

    Does *not* verify the digest -- :func:`verify_workload` owns that, and the
    two are separate so a verification failure can report what it actually
    found rather than refusing to read.
    """
    if spec.source_format != RECORD_FORMAT:
        raise WorkloadError(
            f"workload {spec.name!r} declares source format {spec.source_format!r}; "
            f"this module reads {RECORD_FORMAT!r}"
        )
    path = spec.source_path
    if not path.is_file():
        raise WorkloadError(f"workload {spec.name!r} corpus is missing: {path}")

    records: list[WorkloadRecord] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            context = f"{path}:{number}"
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise WorkloadError(f"{context}: not valid JSON: {error}") from None
            if not isinstance(raw, dict):
                raise WorkloadError(f"{context}: record is not a JSON object")
            messages = raw.get("messages")
            if not isinstance(messages, list) or not messages:
                raise WorkloadError(f"{context}: record has no messages")
            tools = raw.get("tools") or []
            if not isinstance(tools, list):
                raise WorkloadError(f"{context}: tools is not a list")
            for key in ("id", "prompt_chars", "prompt_tokens"):
                if key not in raw:
                    raise WorkloadError(f"{context}: record is missing {key!r}")
            timestamps = raw.get("timestamps")
            records.append(
                WorkloadRecord(
                    id=str(raw["id"]),
                    messages=tuple(_coerce_message(m, context=context) for m in messages),
                    tools=tuple(dict(t) for t in tools),
                    prompt_chars=int(raw["prompt_chars"]),
                    prompt_tokens=int(raw["prompt_tokens"]),
                    completion=raw.get("completion"),
                    timestamps=tuple(str(t) for t in timestamps) if timestamps else None,
                    ground_truth=raw.get("ground_truth"),
                )
            )
    if not records:
        raise WorkloadError(f"workload {spec.name!r} corpus {path} contains no records")
    return tuple(records)


# ---------------------------------------------------------------------------
# Characteristics and verification
# ---------------------------------------------------------------------------


def recompute_characteristics(records: Sequence[WorkloadRecord]) -> dict[str, Any]:
    """Derive every declared characteristic from records.

    The builder calls this to *write* a manifest and the verifier calls it to
    *check* one, so a declared statistic can never be hand-typed and wrong: both
    sides run the same code over the same definitions.
    """
    if not records:
        raise WorkloadError("cannot compute characteristics of zero records")

    chars = [record.prompt_chars for record in records]
    tokens = [record.prompt_tokens for record in records]
    total = len(records)

    def has_tool_activity(record: WorkloadRecord) -> bool:
        return any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in record.messages
        )

    system_prompts = {
        content_text(message.get("content"))
        for record in records
        for message in record.messages
        if message.get("role") == "system" and content_text(message.get("content"))
    }
    first_user: set[str] = set()
    for record in records:
        for message in record.messages:
            if message.get("role") == "user":
                text = content_text(message.get("content"))
                if text:
                    first_user.add(text)
                    break
    distinct_prompts = {
        hashlib.sha256(record.prompt_text.encode("utf-8")).hexdigest() for record in records
    }
    completion_lengths = [
        len(flatten_prompt([record.completion])) for record in records if record.completion
    ]

    characteristics: dict[str, Any] = {
        "record_count": total,
        "prompt_chars": _percentile_block(chars),
        "prompt_tokens": _percentile_block(tokens),
        "chars_per_token": round(sum(chars) / sum(tokens), 4) if sum(tokens) else None,
        "tool_call_fraction": sum(has_tool_activity(r) for r in records) / total,
        "tools_declared_fraction": sum(bool(r.tools) for r in records) / total,
        "system_prompt_fraction": sum(
            any(m.get("role") == "system" for m in r.messages) for r in records
        )
        / total,
        "multi_turn_fraction": sum(len(r.messages) > 1 for r in records) / total,
        "assistant_completion_fraction": sum(r.completion is not None for r in records) / total,
        "distinct_prompt_fraction": len(distinct_prompts) / total,
        "unique_system_prompts": len(system_prompts),
        "unique_first_user_messages": len(first_user),
        "completion_chars": _percentile_block(completion_lengths) if completion_lengths else None,
    }
    for key in _FRACTION_KEYS:
        characteristics[key] = round(float(characteristics[key]), 6)
    return characteristics


def _compare_number(
    label: str,
    declared: Any,
    recomputed: Any,
    *,
    absolute: float,
    relative: float,
) -> str | None:
    if declared is None and recomputed is None:
        return None
    if declared is None or recomputed is None:
        return f"{label}: declared {declared!r}, recomputed {recomputed!r}"
    delta = abs(float(declared) - float(recomputed))
    allowed = max(absolute, relative * abs(float(recomputed)))
    if delta > allowed:
        return (
            f"{label}: declared {declared}, recomputed {recomputed} "
            f"(delta {delta:.6g} exceeds tolerance {allowed:.6g})"
        )
    return None


def verify_workload(
    spec: WorkloadSpec,
    *,
    check_digest: bool = True,
    records: Sequence[WorkloadRecord] | None = None,
) -> tuple[WorkloadRecord, ...]:
    """Recompute the manifest from the corpus on disk and fail on disagreement.

    Checks, in order:

    1. the corpus file exists, and its size and sha256 match the manifest;
    2. every record's stored ``prompt_chars`` equals the length of its own
       rendered prompt (so an edited message body cannot hide behind a stale
       cached count);
    3. every declared characteristic -- record count, both percentile blocks,
       every fraction, every unique-value count -- matches the recomputation
       within the manifest's declared tolerance;
    4. the declared ``requirements.min_max_model_len`` is genuinely large enough
       for the longest prompt in the corpus plus the declared output reserve.

    Returns the loaded records so a caller that verifies does not pay to read
    the file twice.  Raises :class:`WorkloadVerificationError` listing every
    disagreement found.
    """
    failures: list[str] = []
    path = spec.source_path

    if records is None:
        if not path.is_file():
            raise WorkloadVerificationError(spec.name, [f"corpus file is missing: {path}"])
        if check_digest:
            size = path.stat().st_size
            if size != spec.source_size_bytes:
                failures.append(
                    f"source.size_bytes: declared {spec.source_size_bytes}, on disk {size}"
                )
            digest = file_sha256(path)
            if digest != spec.source_sha256:
                failures.append(
                    f"source.sha256: declared {spec.source_sha256}, on disk {digest} "
                    "-- the corpus file is not the one this manifest describes"
                )
        try:
            records = load_records(spec)
        except WorkloadError as error:
            failures.append(str(error))
            raise WorkloadVerificationError(spec.name, failures) from None

    # (2) per-record integrity.  Report at most a few; one is enough to act on
    # and a wholesale regeneration would otherwise produce thousands of lines.
    mismatched = [
        (record.id, record.prompt_chars, len(record.prompt_text))
        for record in records
        if record.prompt_chars != len(record.prompt_text)
    ]
    for record_id, declared, actual in mismatched[:5]:
        failures.append(
            f"record {record_id!r}: prompt_chars declared {declared} but the rendered "
            f"prompt is {actual} characters"
        )
    if len(mismatched) > 5:
        failures.append(f"... and {len(mismatched) - 5} further prompt_chars mismatches")

    tolerance = spec.tolerance
    relative = float(tolerance.get("percentile_relative", 0.0))
    fraction_absolute = float(tolerance.get("fraction_absolute", 0.0))
    count_absolute = float(tolerance.get("count_absolute", 0.0))

    declared_chars = spec.characteristics
    recomputed = recompute_characteristics(records)

    for key in _COUNT_KEYS:
        failure = _compare_number(
            f"characteristics.{key}",
            declared_chars.get(key),
            recomputed.get(key),
            absolute=count_absolute,
            relative=0.0,
        )
        if failure:
            failures.append(failure)

    for block in ("prompt_chars", "prompt_tokens", "completion_chars"):
        declared_block = declared_chars.get(block)
        recomputed_block = recomputed.get(block)
        if declared_block is None and recomputed_block is None:
            continue
        if not isinstance(declared_block, Mapping) or not isinstance(recomputed_block, Mapping):
            failures.append(
                f"characteristics.{block}: declared {declared_block!r}, "
                f"recomputed {recomputed_block!r}"
            )
            continue
        for key in PERCENTILE_KEYS:
            failure = _compare_number(
                f"characteristics.{block}.{key}",
                declared_block.get(key),
                recomputed_block.get(key),
                absolute=0.0,
                relative=relative,
            )
            if failure:
                failures.append(failure)

    for key in (*_FRACTION_KEYS, "chars_per_token"):
        failure = _compare_number(
            f"characteristics.{key}",
            declared_chars.get(key),
            recomputed.get(key),
            absolute=fraction_absolute if key in _FRACTION_KEYS else 0.0,
            relative=0.0 if key in _FRACTION_KEYS else relative,
        )
        if failure:
            failures.append(failure)

    # (4) the requirement must be honest about the corpus it describes.
    declared_window = spec.requirements.get("min_max_model_len")
    reserve = int(spec.requirements.get("output_reserve_tokens", 0))
    longest = max(record.prompt_tokens for record in records)
    if declared_window is None:
        failures.append("requirements.min_max_model_len is not declared")
    elif int(declared_window) < longest + reserve:
        failures.append(
            f"requirements.min_max_model_len: declared {declared_window} but the longest "
            f"prompt is {longest} tokens and the declared output reserve is {reserve}; "
            "a launch at the declared window would truncate the workload"
        )

    if failures:
        raise WorkloadVerificationError(spec.name, failures)
    return tuple(records)


# ---------------------------------------------------------------------------
# Band sampling
# ---------------------------------------------------------------------------


def _band_metric_value(record: WorkloadRecord, metric: str) -> int:
    if metric == "prompt_chars":
        return record.prompt_chars
    if metric == "prompt_tokens":
        return record.prompt_tokens
    raise WorkloadError(f"unknown band_metric {metric!r}")


def _seeded_random(*material: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(str(part) for part in material).encode("utf-8"))
    return random.Random(int.from_bytes(digest.digest()[:8], "big"))


def band_window(
    spec: WorkloadSpec,
    band_name: str,
    records: Sequence[WorkloadRecord],
) -> tuple[WorkloadRecord, ...]:
    """The distinct records whose prompt length falls inside a band's window.

    The window is an index range over the length-sorted corpus, so it is a true
    percentile slice of the workload's own distribution: never the extremes,
    never a synthesized length.  Records with byte-identical prompts collapse to
    one, because two copies of a prompt are one observation dressed as two.
    """
    band = spec.band(band_name)
    metric = spec.band_metric
    ordered = sorted(records, key=lambda record: (_band_metric_value(record, metric), record.id))
    total = len(ordered)
    start = min(total - 1, max(0, math.floor(band.lower * total)))
    stop = min(total, max(start + 1, math.ceil(band.upper * total)))

    seen: set[str] = set()
    window: list[WorkloadRecord] = []
    for record in ordered[start:stop]:
        digest = hashlib.sha256(record.prompt_text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        window.append(record)
    return tuple(window)


def band_records(
    spec: WorkloadSpec,
    band_name: str,
    count: int,
    *,
    records: Sequence[WorkloadRecord] | None = None,
    seed: int | None = None,
) -> tuple[WorkloadRecord, ...]:
    """Deterministically sample ``count`` distinct records from a band.

    Raises :class:`BandExhaustedError` rather than returning fewer.
    """
    if count <= 0:
        raise WorkloadError(f"band {band_name!r}: count must be positive, got {count}")
    if records is None:
        records = load_records(spec)
    window = band_window(spec, band_name, records)
    if len(window) < count:
        raise BandExhaustedError(spec.name, band_name, count, len(window))
    rng = _seeded_random(
        spec.name,
        spec.version,
        band_name,
        count,
        spec.seed if seed is None else seed,
    )
    chosen = rng.sample(list(window), count)
    return tuple(sorted(chosen, key=lambda record: (record.prompt_chars, record.id)))


def band_prompts(
    spec: WorkloadSpec,
    band_name: str,
    count: int,
    *,
    records: Sequence[WorkloadRecord] | None = None,
    seed: int | None = None,
) -> tuple[str, ...]:
    """Band sample rendered as one prompt string per request (lossy for agents)."""
    chosen = band_records(spec, band_name, count, records=records, seed=seed)
    return tuple(record.final_user_text for record in chosen)


def band_messages(
    spec: WorkloadSpec,
    band_name: str,
    count: int,
    *,
    records: Sequence[WorkloadRecord] | None = None,
    seed: int | None = None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Band sample as full OpenAI-format message lists (system + history + user)."""
    chosen = band_records(spec, band_name, count, records=records, seed=seed)
    return tuple(record.messages for record in chosen)


# ---------------------------------------------------------------------------
# Drop-in interface for the configuration matrix
# ---------------------------------------------------------------------------


def _resolve(
    workload: str | WorkloadSpec,
    *,
    directory: Path | None,
    verify: bool,
) -> tuple[WorkloadSpec, tuple[WorkloadRecord, ...]]:
    spec = (
        workload
        if isinstance(workload, WorkloadSpec)
        else load_spec(workload, directory=directory)
    )
    records = verify_workload(spec) if verify else load_records(spec)
    return spec, records


def prompts_by_context(
    workload: str | WorkloadSpec,
    counts: Mapping[str, int] | int,
    *,
    directory: Path | None = None,
    verify: bool = True,
    seed: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Drop-in replacement for ``_prompts_by_context``.

    Returns ``{band name: tuple of prompt strings}`` -- the same shape the
    configuration matrix already consumes.  ``counts`` is either one integer
    applied to every declared band or a per-band mapping.

    For agentic workloads this throws away the system prompt, the tool schemas
    and the whole history, which is precisely what makes agentic traffic
    expensive; use :func:`messages_by_context` there.
    """
    spec, records = _resolve(workload, directory=directory, verify=verify)
    wanted = (
        {name: int(counts) for name in spec.bands}
        if isinstance(counts, int)
        else {str(k): int(v) for k, v in counts.items()}
    )
    return {
        name: band_prompts(spec, name, size, records=records, seed=seed)
        for name, size in wanted.items()
    }


def messages_by_context(
    workload: str | WorkloadSpec,
    counts: Mapping[str, int] | int,
    *,
    directory: Path | None = None,
    verify: bool = True,
    seed: int | None = None,
) -> dict[str, tuple[tuple[dict[str, Any], ...], ...]]:
    """Richer accessor: ``{band name: tuple of OpenAI message lists}``."""
    spec, records = _resolve(workload, directory=directory, verify=verify)
    wanted = (
        {name: int(counts) for name in spec.bands}
        if isinstance(counts, int)
        else {str(k): int(v) for k, v in counts.items()}
    )
    return {
        name: band_messages(spec, name, size, records=records, seed=seed)
        for name, size in wanted.items()
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_refusals(
    spec: WorkloadSpec,
    *,
    max_model_len: int,
    tool_support: bool = False,
) -> tuple[str, ...]:
    """Reasons a launch with this configuration would misrepresent the workload.

    Empty means the launch is safe to make.  The configuration matrix currently
    serves with ``--max-model-len 4096``; three of the four shipped workloads
    exceed that, and running them anyway would silently truncate prompts -- which
    would produce numbers, just not numbers about this workload.
    """
    reasons: list[str] = []
    required = spec.requirements.get("min_max_model_len")
    if required is not None and int(max_model_len) < int(required):
        reasons.append(
            f"workload {spec.name!r} needs --max-model-len >= {int(required)} "
            f"(longest prompt plus a {spec.requirements.get('output_reserve_tokens', 0)}-token "
            f"output reserve) but the launch declares {int(max_model_len)}; "
            "prompts would be truncated"
        )
    if spec.requirements.get("needs_tool_support") and not tool_support:
        reasons.append(
            f"workload {spec.name!r} declares tool schemas on every request but the "
            "launch does not enable tool support"
        )
    return tuple(reasons)


def describe(spec: WorkloadSpec) -> str:  # pragma: no cover - convenience for humans
    """One-screen human summary of a manifest."""
    lines = [
        f"{spec.name} v{spec.version}  [{spec.domain}]",
        f"  {spec.description}",
        f"  source: {spec.source_path}",
        f"  records: {spec.characteristics.get('record_count')}",
        f"  prompt tokens p50/p90/p100: "
        f"{spec.characteristics['prompt_tokens']['p50']}/"
        f"{spec.characteristics['prompt_tokens']['p90']}/"
        f"{spec.characteristics['prompt_tokens']['p100']}",
        f"  min max-model-len: {spec.requirements.get('min_max_model_len')}",
        f"  bands: {stable_json(spec.manifest['bands'])}",
    ]
    return "\n".join(lines)


def iter_specs(*, directory: Path | None = None) -> Iterable[WorkloadSpec]:
    for name in available_workloads(directory=directory):
        yield load_spec(name, directory=directory)
