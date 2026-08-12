"""Immutable, content-addressed held-out benchmark suites."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.storage import (
    atomic_write_json,
    atomic_write_text,
)
from speedlm.traces.store import TraceRecord

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SuiteError(ValueError):
    """Raised when suite construction or validation fails."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def _context_hash(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the request a replay will actually send.

    Tools participate in the digest, but only when the request had any. Two
    captured requests carrying the same messages and *different* tool schemas
    are not the same benchmark context -- the schemas are rendered into the
    prompt by the chat template, so they change the prompt length, the
    engine's decoding, and whether a tool call is even reachable. Collapsing
    them onto one hash would let the split treat them as duplicates and let
    one silently stand in for the other.

    The tool-free branch is kept byte-identical to the pre-tools digest (the
    bare canonical JSON of ``messages``) rather than always hashing a wrapper
    object. Plain chat traffic is the overwhelming majority of what is
    captured, and rehashing it would invalidate every persisted suite
    manifest, every stored ``context_hash``, and every train-set hash a
    leakage check compares against.
    """
    payload: Any = (
        list(messages)
        if not tools
        else {"messages": list(messages), "tools": list(tools)}
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenContext:
    """A single benchmark context with a deterministic hash.

    Attributes:
        context_hash: SHA-256 hex digest of the canonical JSON of messages.
        messages: The conversation messages (tuple of dicts).
        seed: The random seed to use for this context.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p.
        expected_response: The reference response text for correctness checks.
        tools: The tool schemas the original request offered the model, in
            capture order. Empty for plain chat traffic.
    """

    context_hash: str
    messages: tuple[dict[str, Any], ...]
    seed: int
    temperature: float
    top_p: float
    expected_response: str = ""
    #: Tool schemas carried through from the captured request.
    #:
    #: A request that offered tools and one that did not are different
    #: requests: the tool schemas sit in the prompt the engine actually
    #: templates, and dropping them makes the gate benchmark a prompt
    #: production never served. They therefore ride into ``context_hash``
    #: whenever they are present -- but *only* when present, so chat traffic
    #: keeps byte-identical hashes and manifests from before tools existed.
    tools: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_trace(
        cls,
        record: TraceRecord,
        expected_response: str | None = None,
    ) -> FrozenContext:
        """Build a request context and reference output from a captured trace.

        Captured traces append the provider-authored assistant message to the
        original request. Replaying that whole list would benchmark a
        continuation *after* the answer instead of the captured input. The
        final generated assistant is therefore removed and used as the
        reference output.
        """
        messages, captured_response = cls._input_messages(record)
        tools = [dict(tool) for tool in record.tools]
        return cls(
            context_hash=_context_hash(messages, tools),
            messages=tuple(messages),
            seed=record.seed,
            temperature=record.temperature,
            top_p=record.top_p,
            expected_response=(
                captured_response if expected_response is None else expected_response
            ),
            tools=tuple(tools),
        )

    @staticmethod
    def _input_messages(record: TraceRecord) -> tuple[list[dict[str, Any]], str]:
        messages = [dict(message) for message in record.messages]
        output_index: int | None = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                message.get("role") == "assistant"
                and message.get("provenance_tag") == "generated"
            ):
                output_index = index
                break
        if output_index is None:
            inputs = messages
            expected = ""
        else:
            output = messages[output_index].get("content")
            expected = output if isinstance(output, str) else ""
            inputs = messages[:output_index]
        for message in inputs:
            message.pop("provenance_tag", None)
        if not inputs:
            raise SuiteError(f"trace {record.id!r} has no replayable input messages")
        return inputs, expected

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "context_hash": self.context_hash,
            "messages": [dict(m) for m in self.messages],
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "expected_response": self.expected_response,
        }
        # Emitted only when non-empty so a chat-only suite serialises to the
        # exact bytes it did before tools were carried at all.
        if self.tools:
            payload["tools"] = [dict(tool) for tool in self.tools]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrozenContext:
        required = {
            "context_hash", "messages", "seed", "temperature", "top_p"
        }
        missing = required - set(data.keys())
        if missing:
            raise SuiteError(
                f"Missing required key in frozen context: {sorted(missing)[0]}"
            )
        return cls(
            context_hash=data["context_hash"],
            messages=tuple(dict(m) for m in data["messages"]),
            seed=data["seed"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            expected_response=data.get("expected_response", ""),
            tools=tuple(dict(tool) for tool in data.get("tools", ())),
        )


def session_key(record: TraceRecord) -> str:
    """Identify the agent session a record was captured from.

    Multi-turn agentic traffic is *nested*, not independent: turn N's
    ``messages`` list contains turn N-1's messages verbatim, so splitting such
    records individually puts a gate context's own continuation into the
    training corpus.  Grouping needs a session identity, but ``TraceRecord``
    carries none -- so it is derived from the transcript itself.

    A session is identified by its *opening exchange*: the first three
    messages (system, first user, first assistant), compared on role, content
    and tool calls only.  Bookkeeping keys such as ``provenance_tag`` are
    ignored because they are per-capture, not per-session.  Every later turn of
    a session re-sends that same opening verbatim, so every record of a session
    hashes to the same key and records of different sessions do not.

    Records with fewer than three messages cannot carry an opening exchange and
    become their own session, keyed by their context hash.  That is the safe
    direction: over-grouping only withholds more rows from training, while
    under-grouping is the leak.  It also keeps single-turn corpora *exactly* as
    they were -- one record per session, ranked by the same key as before.
    """
    messages = record.messages
    if len(messages) < 3:
        inputs, _ = FrozenContext._input_messages(record)
        return _context_hash(inputs, [dict(tool) for tool in record.tools])
    opening = [
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
        for message in messages[:3]
    ]
    canonical = json.dumps(
        opening,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "session:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _session_components(
    records: Sequence[TraceRecord],
    record_hashes: Sequence[str],
) -> dict[str, frozenset[str]]:
    """Group context hashes into session-closed connected components.

    Contexts and sessions form a *bipartite* graph: each record contributes one
    edge joining its context hash to the session it was captured from.  A
    context may carry several such edges, and for agentic corpora it routinely
    does -- every trajectory in a rollout family opens with the same system
    prompt and the same first user message, so the turn-1 requests of sibling
    trajectories are byte-identical and hash alike, while ``session_key``
    (which hashes ``messages[:3]``, including the model's first reply) tells
    the trajectories apart.

    Collapsing such a context onto a single session -- the ``min`` this
    replaced -- makes reservation session-*atomic* but not session-*closed*:
    reserving the shared opening pulls in one trajectory and leaves its
    siblings, whose later turns nest that very opening, in the training set.
    The straddle guard in :mod:`speedlm.training.split` then correctly refuses
    the whole cycle.

    Reserving the entire connected component is the closure that fixes it: a
    component is a maximal set of sessions and contexts reachable from one
    another, so taking any member takes every session that could nest it.

    The result is a pure function of the *set* of edges -- union is resolved
    toward the lexicographically smaller root, so a component's representative
    is its minimum node no matter what order the records arrive in.  Records
    below the three-message floor make ``session_key`` return the record's own
    context hash, i.e. a self-edge, so single-turn corpora stay singleton
    components and their split remains byte-identical to the per-record one.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for record, context_hash in zip(records, record_hashes, strict=True):
        union(context_hash, session_key(record))

    grouped: dict[str, set[str]] = {}
    for context_hash in record_hashes:
        grouped.setdefault(find(context_hash), set()).add(context_hash)
    return {
        context_hash: frozenset(grouped[find(context_hash)])
        for context_hash in record_hashes
    }


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """An immutable, content-addressed benchmark suite.

    Attributes:
        suite_hash: SHA-256 over all context hashes (sorted for determinism).
        contexts: Ordered list of frozen contexts.
    """

    suite_hash: str
    contexts: tuple[FrozenContext, ...]

    @classmethod
    def build(
        cls,
        records: Sequence[TraceRecord],
        *,
        held_out_fraction: float = 0.2,
        seed: int = 42,
    ) -> BenchmarkSuite:
        """Build a suite from trace records, reserving held_out_fraction.

        Ranks unique context hashes with ``seed`` and reserves
        ``ceil(unique_contexts * held_out_fraction)`` of them. The same input
        always yields the same suite, and duplicate contexts cannot straddle
        the split.

        Reservation is *session-closed*.  Records of one multi-turn agent
        session are nested rather than independent (see :func:`session_key`),
        so reserving a single turn while training on its siblings hands the
        draft head the very continuation it is scored on predicting.  And a
        single context can belong to several sessions at once, because sibling
        trajectories of a rollout family share their opening exchange
        verbatim -- so reserving one session is not enough either.  The
        ranking is unchanged; what changed is that reserving a context also
        reserves its whole connected component in the context/session graph
        (see :func:`_session_components`), and the count is then satisfied.
        The reserved set can therefore overshoot ``held_out_fraction`` to land
        on a component boundary.  For single-turn corpora every record is its
        own singleton component and the selection is byte-identical to the
        per-record split.

        Args:
            records: Ordered trace records.
            held_out_fraction: Fraction of unique contexts to include in suite.
            seed: Seed for the hash-based split (defaults to 42).

        Returns:
            A new immutable :class:`BenchmarkSuite`.

        Raises:
            SuiteError: If fewer than 1 record survives filtering.
        """
        if (
            isinstance(held_out_fraction, bool)
            or not isinstance(held_out_fraction, (int, float))
            or not math.isfinite(held_out_fraction)
            or held_out_fraction < 0
            or held_out_fraction > 1
        ):
            raise SuiteError(
                f"held_out_fraction must be in [0, 1], got {held_out_fraction}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SuiteError(f"seed must be an integer, got {seed!r}")

        if not records:
            raise SuiteError("Cannot build suite from empty record list")

        record_hashes = tuple(cls._record_hash(rec) for rec in records)
        unique_hashes = set(record_hashes)
        held_out_count = math.ceil(len(unique_hashes) * held_out_fraction)
        ranked_hashes = sorted(
            unique_hashes,
            key=lambda context_hash: (
                hashlib.sha256(f"{seed}:{context_hash}".encode("ascii")).digest(),
                context_hash,
            ),
        )
        # Reserve by connected component of the context/session graph, not by
        # a single session per context: one context can belong to several
        # sessions at once (see :func:`_session_components`), and reserving
        # only one of them leaves the siblings that nest it in training.
        components = _session_components(records, record_hashes)

        held_out_hashes: set[str] = set()
        for context_hash in ranked_hashes:
            if len(held_out_hashes) >= held_out_count:
                break
            held_out_hashes |= components[context_hash]
        contexts = tuple(
            FrozenContext.from_trace(rec)
            for rec, context_hash in zip(records, record_hashes, strict=True)
            if context_hash in held_out_hashes
        )
        if not contexts:
            raise SuiteError(
                "No held-out records selected; held_out_fraction must reserve "
                "at least one context"
            )

        suite_hash = cls._compute_hash(contexts)

        return cls(suite_hash=suite_hash, contexts=contexts)

    @classmethod
    def build_with_split(
        cls,
        train_records: Sequence[TraceRecord],
        all_records: Sequence[TraceRecord],
        *,
        held_out_fraction: float = 0.2,
    ) -> BenchmarkSuite:
        """Build suite ensuring train/held-out disjointness.

        Args:
            train_records: Records used for training the candidate.
            all_records: All available records to split from.

        Returns:
            A suite with only records NOT in train.

        Raises:
            SuiteError: If any context appears in both train and held-out
                (leakage), or if no held-out records remain.
        """
        # Build a set of train context hashes
        train_hashes = {cls._record_hash(record) for record in train_records}

        # Filter out train records from all records
        held_out = [
            rec for rec in all_records
            if cls._record_hash(rec) not in train_hashes
        ]

        if not held_out:
            raise SuiteError(
                "No held-out records remain after removing train set"
            )

        return cls.build(held_out, held_out_fraction=held_out_fraction)

    @staticmethod
    def _record_hash(rec: TraceRecord) -> str:
        """Hash a record exactly as ``FrozenContext.from_trace`` would.

        This must stay in lockstep with :attr:`FrozenContext.context_hash`:
        the split selects on this value while ``check_leakage`` compares the
        frozen contexts' own hashes against it. Both therefore go through
        :func:`_context_hash` rather than re-deriving the canonical form.
        """
        messages, _ = FrozenContext._input_messages(rec)
        return _context_hash(messages, [dict(tool) for tool in rec.tools])

    @staticmethod
    def _compute_hash(contexts: tuple[FrozenContext, ...]) -> str:
        """Compute deterministic suite hash from sorted context hashes."""
        sorted_hashes = sorted(ctx.context_hash for ctx in contexts)
        combined = "|".join(sorted_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def check_leakage(self, train_hashes: set[str]) -> list[str]:
        """Check for overlap between suite contexts and train hashes.

        Returns:
            List of overlapping hashes (empty if no leakage).
        """
        suite_hashes = {ctx.context_hash for ctx in self.contexts}
        return sorted(suite_hashes & train_hashes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_hash": self.suite_hash,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkSuite:
        if "suite_hash" not in data:
            raise SuiteError("Missing 'suite_hash' in suite manifest")
        if "contexts" not in data:
            raise SuiteError("Missing 'contexts' in suite manifest")
        contexts = tuple(
            FrozenContext.from_dict(c) for c in data["contexts"]
        )
        # Recompute hash and verify
        computed = cls._compute_hash(contexts)
        if computed != data["suite_hash"]:
            raise SuiteError(
                f"Suite hash mismatch: stored={data['suite_hash']}, "
                f"computed={computed}"
            )
        return cls(suite_hash=data["suite_hash"], contexts=contexts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_suite(
    suite: BenchmarkSuite,
    run_dir: Path,
) -> tuple[Path, Path]:
    """Write suite to disk as JSONL contexts + JSON manifest.

    Returns:
        (contexts_jsonl_path, manifest_json_path)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "suite_contexts.jsonl"
    manifest_path = run_dir / "suite_manifest.json"

    # Write contexts as JSONL
    atomic_write_text(
        jsonl_path,
        "\n".join(json.dumps(ctx.to_dict()) for ctx in suite.contexts) + "\n"
        if suite.contexts else "",
    )

    # Write manifest
    atomic_write_json(manifest_path, suite.to_dict())

    return jsonl_path, manifest_path


def load_suite(run_dir: Path) -> BenchmarkSuite:
    """Load suite from a run directory.

    Args:
        run_dir: Directory containing suite_manifest.json.

    Returns:
        The deserialized :class:`BenchmarkSuite`.

    Raises:
        SuiteError: If manifest is missing or corrupted.
    """
    manifest_path = run_dir / "suite_manifest.json"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SuiteError(f"Suite manifest not found: {manifest_path}") from exc
    except OSError as exc:
        raise SuiteError(f"Cannot read suite manifest: {manifest_path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuiteError(f"Invalid JSON in manifest: {exc}") from exc

    if not isinstance(data, dict):
        raise SuiteError("Suite manifest must be a JSON object")

    return BenchmarkSuite.from_dict(data)


def build_suite(
    records: Sequence[TraceRecord],
    *,
    held_out_fraction: float = 0.2,
) -> BenchmarkSuite:
    """Convenience: build a suite from trace records.

    Alias for ``BenchmarkSuite.build``.
    """
    return BenchmarkSuite.build(records, held_out_fraction=held_out_fraction)
