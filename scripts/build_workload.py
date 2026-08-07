#!/usr/bin/env python
"""Materialize benchmark workloads and emit their manifests.

Run this, not a text editor, to add or refresh a workload.  The builder writes
two things:

* ``/data/ryan.kim/speedlm-workloads/<name>/records.jsonl`` -- the normalized
  corpus, one JSON object per request, in the single record format
  ``tests/e2e/harness/workloads.py`` knows how to read;
* ``tests/e2e/harness/workload_specs/<name>.json`` -- the manifest, whose
  declared statistics are computed by
  ``workloads.recompute_characteristics`` over the file just written.

The manifest statistics are therefore *never* hand-typed.  The verifier calls
the same function over the same file, so "declared" and "recomputed" are the same
code path applied at two different times; the only way they can disagree is if
the file changed, which is exactly the condition the verifier exists to catch.

TOKEN COUNTS
------------
Prompt token counts are real tokenizer output, not ``len(text) / 4``.  The
agentic corpora measure ~2.5 characters per token, not 4, because tool JSON
schemas are token-dense; a chars/4 estimate understates them by about 60%, which
is the difference between "fits in a 16k window" and "does not".  The tokenizer
identity is recorded in the manifest.  Counts are baked per record so the
verifier can recompute percentiles in CI without transformers installed.

    /admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python \
        scripts/build_workload.py all --tokenizer openai/gpt-oss-20b

THE TRACE EXPORTER (long-context-sessions)
------------------------------------------
The ``long-context-sessions`` builder is a PORT of
``/admin/home/ryan.kim/speedlm/qwen-trace-pilot/src/qwen_trace_pilot/core.py``,
not an import -- that package lives in a different tree and is shaped for
building drafter training data, not for benchmarking.  Four things were fixed in
the port, each a real defect for this purpose:

1. ``ABS_PATH_PATTERN`` there is ``(?<![\\w.-])/(?:home|Users)/...``.  On this
   cluster homes are ``/admin/home/<user>/...``, so the lookbehind sees ``n`` and
   refuses to match: a colleague's absolute paths pass through UNREDACTED.  The
   port accepts an optional leading path segment.  ``tests/e2e/
   test_harness_workloads.py::test_redaction_covers_cluster_home_layout`` proves
   it, and proves the original pattern fails.
2. ``minimum_assistant_chars`` defaults to 800 there, which deletes the modal
   short tool-call turn and inverts the decode-length distribution -- precisely
   the distribution a speculative-decoding benchmark is measuring.  Here it is a
   parameter defaulting to 1.
3. Per-turn timestamps are discarded there.  Here every turn's timestamp is
   carried, so arrival shape stays recoverable.
4. (Not in the brief, found while porting.)  ``build_traces`` hard-filters on
   ``message["model"] == "Qwen/Qwen3.6-27B"``.  Claude Code sessions carry
   ``claude-*`` model ids, so the unmodified exporter yields ZERO traces from
   this source.  The port accepts any real model id and records the observed
   model histogram in provenance instead of silently emptying the corpus.

Memory: the original loads every record of every file into one dict.  At 3.01 GB
of session data that does not fit.  The port processes one session file at a
time, which also bounds the ancestor chain to its own session -- the same thing
in practice, since ``parentUuid`` links do not cross files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e.harness.workloads import (  # noqa: E402
    RECORD_FORMAT,
    SCHEMA_VERSION,
    WorkloadRecord,
    content_text,
    file_sha256,
    flatten_prompt,
    recompute_characteristics,
    stable_json,
)

DEFAULT_OUTPUT_ROOT = Path("/data/ryan.kim/speedlm-workloads")
SPEC_DIR = REPO_ROOT / "tests" / "e2e" / "harness" / "workload_specs"

ULTRACHAT = Path("/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl")
TOOL_LOOP = Path(
    "/admin/home/ryan.kim/speedlm/mtp-hashline-demo/training/artifacts/"
    "gpt-oss-100k-v3-first-finalized-v6/success-only/corpus.eagle3.v1.jsonl"
)
MIXED_OUTCOME = Path(
    "/admin/home/ryan.kim/speedlm/mtp-hashline-demo/training/artifacts/"
    "iteration2-all-session-safe-train/normalized.jsonl"
)
SESSION_ROOT = Path("/admin/home/ryan.kim/.claude/projects")

#: Percentile tolerance for the verifier.  Zero would be defensible -- the
#: computation is deterministic -- but a hair of slack keeps a manifest usable
#: across a float-formatting change without weakening it: 0.5% of a percentile
#: cannot hide a truncated or swapped corpus, which moves percentiles by orders
#: of magnitude.
DEFAULT_TOLERANCE = {
    "percentile_relative": 0.005,
    "fraction_absolute": 0.001,
    "count_absolute": 0,
}

DEFAULT_BANDS = {
    "short": [0.05, 0.25],
    "medium": [0.40, 0.60],
    "long": [0.75, 0.95],
}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class Tokenizer:
    """Real tokenizer if one is available, otherwise an explicit refusal.

    A chars/N fallback is offered but must be asked for by name, and it is
    recorded in the manifest as an estimate.  Silently falling back would put a
    number in ``requirements.min_max_model_len`` that looks measured and is not.
    """

    def __init__(self, name: str, chars_per_token: float | None = None) -> None:
        self.name = name
        self.chars_per_token = chars_per_token
        self._impl = None
        self._basis = ""
        if chars_per_token is not None:
            self._basis = f"estimate:{chars_per_token}-chars-per-token"
            return
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._impl = AutoTokenizer.from_pretrained(name)
        self._basis = f"tokenizer:{name}"

    @property
    def basis(self) -> str:
        return self._basis

    def count(self, text: str) -> int:
        if self._impl is None:
            assert self.chars_per_token is not None
            return max(1, int(round(len(text) / self.chars_per_token)))
        return len(self._impl(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# Normalized record construction
# ---------------------------------------------------------------------------


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields that go on the wire (plus thinking, which costs tokens)."""
    kept: dict[str, Any] = {"role": str(message.get("role", ""))}
    # Content-part lists are flattened to text here so every materialized record
    # has string content.  agentic-mixed-outcome stores its user turns as
    # [{"type": "text", ...}]; leaving that shape in place is how the user's
    # actual request got silently rendered as the empty string in an earlier
    # version of this builder.
    kept["content"] = content_text(message.get("content"))
    for key in ("tool_calls", "tool_call_id", "name", "thinking"):
        value = message.get(key)
        if value not in (None, "", [], {}):
            kept[key] = value
    return kept


def make_record(
    record_id: str,
    messages: Sequence[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    tools: Sequence[dict[str, Any]] = (),
    completion: dict[str, Any] | None = None,
    timestamps: Sequence[str] | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = [_clean_message(m) for m in messages]
    tool_list = [dict(t) for t in tools]
    text = flatten_prompt(cleaned, tool_list)
    payload: dict[str, Any] = {
        "id": record_id,
        "messages": cleaned,
        "prompt_chars": len(text),
        "prompt_tokens": tokenizer.count(text),
    }
    if tool_list:
        payload["tools"] = tool_list
    if completion is not None:
        payload["completion"] = _clean_message(completion)
    if timestamps:
        payload["timestamps"] = list(timestamps)
    if ground_truth:
        payload["ground_truth"] = ground_truth
    return payload


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def upstream_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# (a) generic-chat
# ---------------------------------------------------------------------------

GENERIC_CHAT_DESCRIPTION = (
    "CONTROL WORKLOAD, NOT REPRESENTATIVE OF AGENTIC SERVING. 22,362 single-turn, "
    "user-only chat prompts from UltraChat, hard-truncated upstream at 4095 characters. "
    "There are no system prompts, no tool schemas, no tool results, no assistant "
    "history and no reference completions in this corpus: 0% of records have any of "
    "them. Every measurement this project published before the workload system existed "
    "was taken here, which is why it is kept -- it is the comparability anchor and a "
    "throughput-saturation set of generic English text. It is NOT evidence about "
    "agentic traffic, whose prompts are two orders of magnitude longer and whose token "
    "density is ~1.6x higher because of tool JSON. Do not generalize a speedup measured "
    "here to a coding agent."
)


def build_generic_chat(tokenizer: Tokenizer, limit: int | None) -> tuple[list[dict], dict]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(read_jsonl(ULTRACHAT)):
        if limit is not None and index >= limit:
            break
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        first = messages[0]
        if first.get("role") != "user" or not isinstance(first.get("content"), str):
            continue
        if not first["content"].strip():
            continue
        records.append(make_record(f"ultrachat-{index:06d}", [first], tokenizer))
    provenance = {
        "upstream": [upstream_entry(ULTRACHAT)],
        "method": (
            "one record per upstream line; the single user message is kept verbatim. "
            "No truncation and no padding are applied here -- the 4095-character ceiling "
            "is upstream, in prepare_ultrachat_corpus.py."
        ),
        "bounds": {"max_records": limit},
    }
    return records, provenance


# ---------------------------------------------------------------------------
# (b) agentic-tool-loop
# ---------------------------------------------------------------------------

TOOL_LOOP_DESCRIPTION = (
    "Per-TURN records from a synthetic coding-agent tool loop against gpt-oss-20b, with "
    "five real function schemas (read, grep, edit, bash, write) declared on every "
    "request. This is realistic in SHAPE -- huge tool-heavy prefix, tiny generated turn "
    "-- and narrow in CONTENT, and the narrowness is load-bearing, so it is stated here "
    "rather than in a footnote: the 626 records come from only ~43 session instances "
    "over 16 bug templates on ONE synthetic e-commerce application; there are just 129 "
    "distinct system prompts and 44 distinct user messages in the whole corpus; the "
    "system prompt is a near-constant ~27,000 characters, so records share an enormous "
    "common prefix and any prefix-cache-sensitive measurement taken here will look far "
    "better than production; and the corpus is 'success-only', i.e. survivorship-biased "
    "-- every trajectory that failed was dropped, so the failure modes that dominate "
    "real agent traffic are absent by construction. Prefer agentic-mixed-outcome, which "
    "keeps its failures. Use this one to study the tool-loop prefix/decode ratio."
)


def build_tool_loop(tokenizer: Tokenizer, limit: int | None) -> tuple[list[dict], dict]:
    records: list[dict[str, Any]] = []
    instances: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    for index, raw in enumerate(read_jsonl(TOOL_LOOP)):
        if limit is not None and index >= limit:
            break
        conversations = raw.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        prefix, final = conversations[:-1], conversations[-1]
        record_id = str(raw.get("id", f"tool-loop-{index:05d}"))
        run_id = record_id.split(":", 1)[0]
        instances[run_id] += 1
        # run id shape: shoplens-<split>-<index>-<template words...>-s<seed>-<hash>
        parts = run_id.split("-")
        templates["-".join(parts[3:-2]) if len(parts) > 5 else run_id] += 1
        records.append(
            make_record(
                record_id,
                prefix,
                tokenizer,
                tools=raw.get("tools") or (),
                completion=final,
                ground_truth={
                    "outcome": "success",
                    "outcome_note": "corpus is success-only; failures were dropped upstream",
                    "verifier": raw.get("verifier"),
                },
            )
        )
    provenance = {
        "upstream": [upstream_entry(TOOL_LOOP)],
        "method": (
            "one record per upstream line. messages = conversations[:-1] (the request "
            "prefix actually sent), completion = conversations[-1] (the reference "
            "generated turn), tools = the five declared function schemas."
        ),
        "bounds": {"max_records": limit},
        "diversity": {
            "session_instances": len(instances),
            "bug_templates": len(templates),
            "records_per_instance_max": max(instances.values()) if instances else 0,
            "note": (
                "instance and template counts are derived from the upstream run id; they "
                "are the honest denominator for any claim made on this corpus."
            ),
        },
    }
    return records, provenance


# ---------------------------------------------------------------------------
# (c) agentic-mixed-outcome
# ---------------------------------------------------------------------------

MIXED_DESCRIPTION = (
    "The best agentic set here, because it keeps its failures: 314 successful and 303 "
    "safe_failed_session trajectories, so it is not survivorship-biased the way "
    "agentic-tool-loop is. Per-turn records from a coding-agent session against "
    "gpt-oss-20b with tool schemas on every request. Two things make it uniquely useful. "
    "First, the upstream corpus carries its OWN token count per record "
    "(estimated_tokens), which is what this builder cross-checks its tokenizer against. "
    "Second, every record carries REAL MEASURED speculative-decoding metrics "
    "(metadata.metrics_delta: acceptance_fraction, draft/accepted token counts, "
    "mean_acceptance_length) captured at collection time; those are lifted into the "
    "manifest's ground_truth block and onto each record, so a harness result measured on "
    "this workload can be checked against an independently measured acceptance rate "
    "instead of being believed on its own. Caveat: 9 bug families / 9 templates on one "
    "synthetic application, so content diversity is still low even though outcome "
    "diversity is real."
)


def build_mixed_outcome(tokenizer: Tokenizer, limit: int | None) -> tuple[list[dict], dict]:
    records: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    acceptances: list[float] = []
    declared_tokens: list[int] = []
    measured_tokens: list[int] = []
    families: Counter[str] = Counter()
    for index, raw in enumerate(read_jsonl(MIXED_OUTCOME)):
        if limit is not None and index >= limit:
            break
        messages = raw.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        metadata = raw.get("metadata") or {}
        metrics = metadata.get("metrics_delta") or {}
        request = metadata.get("request") or {}
        tools = request.get("tools") or []
        status = str(raw.get("status", "unknown"))
        statuses[status] += 1
        families[str(raw.get("family_id", "unknown"))] += 1
        acceptance = metrics.get("acceptance_fraction")
        if isinstance(acceptance, int | float):
            acceptances.append(float(acceptance))
        record = make_record(
            str(raw.get("trajectory_id", f"mixed-{index:05d}")),
            messages[:-1],
            tokenizer,
            tools=tools,
            completion=messages[-1],
            ground_truth={
                "status": status,
                "measured_metrics": metrics,
                "measured_metrics_semantics": metadata.get("metrics_semantics"),
                "upstream_estimated_tokens": raw.get("estimated_tokens"),
                "checks": raw.get("checks"),
            },
        )
        records.append(record)
        if isinstance(raw.get("estimated_tokens"), int):
            declared_tokens.append(int(raw["estimated_tokens"]))
            measured_tokens.append(int(record["prompt_tokens"]))

    ratio = None
    if declared_tokens:
        pairs = sorted(m / d for m, d in zip(measured_tokens, declared_tokens, strict=True))
        ratio = round(pairs[len(pairs) // 2], 4)
    provenance = {
        "upstream": [upstream_entry(MIXED_OUTCOME)],
        "method": (
            "one record per upstream line. messages = messages[:-1], completion = "
            "messages[-1], tools = metadata.request.tools. metadata.metrics_delta is "
            "carried verbatim onto each record's ground_truth and aggregated into the "
            "manifest ground_truth block."
        ),
        "bounds": {"max_records": limit},
        "status_counts": dict(statuses),
        "family_count": len(families),
        "token_cross_check": {
            "median_ratio_measured_over_upstream_estimate": ratio,
            "note": (
                "upstream estimated_tokens counts the request the collector actually "
                "sent (chat template applied); this builder counts the template-free "
                "flattened prompt including tool schemas. A ratio near 1 means the two "
                "independent counts agree; it is recorded, not asserted."
            ),
        },
    }
    return records, provenance


def mixed_ground_truth(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(r["ground_truth"]["measured_metrics"]["acceptance_fraction"])
        for r in records
        if r.get("ground_truth", {}).get("measured_metrics", {}).get("acceptance_fraction")
        is not None
    )
    if not values:
        return {}

    def q(fraction: float) -> float:
        return round(values[int(round(fraction * (len(values) - 1)))], 6)

    successes = [r for r in records if r["ground_truth"]["status"] == "success"]
    return {
        "source": (
            "measured at collection time by the upstream harness against gpt-oss-20b "
            "with a 3-token eagle3 speculator (RedHatAI/gpt-oss-20b-speculator.eagle3), "
            "greedy sampling. NOT recomputed here."
        ),
        "metric": "acceptance_fraction",
        "semantics": "accepted speculative tokens / proposed draft tokens",
        "n": len(values),
        "p5": q(0.05),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "mean": round(sum(values) / len(values), 6),
        "success_record_count": len(successes),
        "usage": (
            "A harness run on this workload with the same verifier/drafter pair and "
            "greedy sampling should land near p50; landing far outside p5..p95 means the "
            "harness, not the model, changed."
        ),
    }


# ---------------------------------------------------------------------------
# (d) long-context-sessions -- ported trace exporter
# ---------------------------------------------------------------------------

HOME = str(Path.home())

SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|rk|pk|ghp|github_pat)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^\s'\"`]{8,}", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b", re.I),
    re.compile(r"\b(?:AWS|OPENAI|ANTHROPIC|HF)_[A-Z0-9_]+\s*=\s*[^\s]+", re.I),
)
BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){128,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")

#: The upstream pattern is ``(?<![\w.-])/(?:home|Users)/[^\s'"`]+``.  This
#: cluster's homes are ``/admin/home/<user>``, where the lookbehind sees the ``n``
#: of ``admin`` and refuses the match, so colleague paths survive redaction.  An
#: optional leading segment fixes it while still refusing to fire mid-identifier
#: (``a/home/b`` must not match, or every relative path in a diff gets mangled).
ABS_PATH_PATTERN = re.compile(r"(?<![\w.-])(?:/[A-Za-z0-9_.-]+)?/(?:home|Users)/[^\s'\"`]+")

#: Kept for the regression test that proves the port fixed a real defect.
UPSTREAM_ABS_PATH_PATTERN = re.compile(r"(?<![\w.-])/(?:home|Users)/[^\s'\"`]+")


def redact(text: str) -> str:
    result = text.replace(HOME, "HOME_PATH")
    result = ABS_PATH_PATTERN.sub("ABS_PATH", result)
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("REDACTED_SECRET", result)
    return result


def text_from_content(content: Any) -> str:
    """Flatten Anthropic block content to text.

    Tool calls and tool results are rendered with plain bracket-free tags: the
    upstream renderer used angle-bracket tags, which cannot be safely printed
    into every transcript this project writes.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif kind == "thinking" and isinstance(item.get("thinking"), str):
            parts.append("[thinking]\n" + item["thinking"] + "\n[/thinking]")
        elif kind == "tool_use":
            payload = {"name": item.get("name", "unknown"), "arguments": item.get("input", {})}
            parts.append("[tool_call]\n" + stable_json(payload) + "\n[/tool_call]")
        elif kind == "tool_result":
            payload = {"tool_use_id": item.get("tool_use_id"), "content": item.get("content", "")}
            parts.append("[tool_result]\n" + stable_json(payload) + "\n[/tool_result]")
    return "\n".join(part for part in parts if part).strip()


def is_safe_text(text: str) -> bool:
    if not text.strip() or len(text) > 1_500_000:
        return False
    return all(len(set(match.group(0))) < 12 for match in BASE64_PATTERN.finditer(text))


@dataclass(frozen=True)
class SessionTrace:
    trace_id: str
    session_id: str
    messages: tuple[dict[str, str], ...]
    timestamps: tuple[str, ...]
    completion: dict[str, str]
    source_file: str


def _ancestor_chain(
    record: dict[str, Any], by_uuid: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: dict[str, Any] | None = record
    while current is not None:
        uuid = current.get("uuid")
        if not isinstance(uuid, str) or uuid in seen:
            break
        seen.add(uuid)
        chain.append(current)
        parent = current.get("parentUuid")
        current = by_uuid.get(parent) if isinstance(parent, str) else None
    return list(reversed(chain))


def _real_model(message: Any) -> str | None:
    """The upstream exporter demanded one exact model id and would yield nothing here."""
    if not isinstance(message, dict):
        return None
    model = message.get("model")
    if not isinstance(model, str) or not model or model.startswith("<"):
        return None
    return model


def traces_from_session_file(
    path: Path,
    *,
    minimum_assistant_chars: int,
    minimum_prompt_chars: int,
    per_file_limit: int,
    model_counter: Counter[str],
) -> list[SessionTrace]:
    by_uuid: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return []
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("uuid"), str):
                continue
            by_uuid[record["uuid"]] = record
            order.append(record["uuid"])

    traces: list[SessionTrace] = []
    # Deepest turns last: taking from the end is what makes this a LONG-context
    # workload rather than a sample of session openings.
    for uuid in reversed(order):
        if len(traces) >= per_file_limit:
            break
        record = by_uuid[uuid]
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        model = _real_model(message)
        if model is None:
            continue
        target = redact(text_from_content(message.get("content")))
        if not is_safe_text(target) or len(target) < minimum_assistant_chars:
            continue
        chain = _ancestor_chain(record, by_uuid)[:-1]
        messages: list[dict[str, str]] = []
        timestamps: list[str] = []
        for entry in chain:
            entry_message = entry.get("message")
            if not isinstance(entry_message, dict):
                continue
            role = entry_message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = redact(text_from_content(entry_message.get("content")))
            if not content or not is_safe_text(content):
                continue
            messages.append({"role": str(role), "content": content})
            timestamps.append(str(entry.get("timestamp", "")))
        if not messages or messages[-1]["role"] != "user":
            continue
        prompt_chars = sum(len(m["content"]) for m in messages)
        if prompt_chars < minimum_prompt_chars:
            continue
        model_counter[model] += 1
        traces.append(
            SessionTrace(
                trace_id=f"{path.stem}:{uuid[:12]}",
                session_id=str(record.get("sessionId", "")),
                messages=tuple(messages),
                timestamps=tuple(timestamps),
                completion={"role": "assistant", "content": target},
                source_file=str(path),
            )
        )
    return traces


def _round_robin_by_project(files: Sequence[Path]) -> list[Path]:
    """Interleave session files across project directories before bounding the scan.

    Taking the first N paths in sorted order is not a bounded sample of the
    tree, it is a sample of whatever sorts first.  Measured here: the first 1400
    of 13,531 files were 92% ``subagents/`` sidechains belonging to two project
    directories and ~34 parent sessions, so the "diverse real sessions" workload
    was mostly one project's subagent transcripts.  Round-robin over the
    top-level project directory spends the same budget across the whole tree.
    """
    buckets: dict[str, list[Path]] = {}
    for path in files:
        try:
            key = path.relative_to(SESSION_ROOT).parts[0]
        except ValueError:  # pragma: no cover - defensive
            key = path.parent.name
        buckets.setdefault(key, []).append(path)
    ordered: list[Path] = []
    queues = [iter(bucket) for _, bucket in sorted(buckets.items())]
    while queues:
        survivors = []
        for queue in queues:
            nxt = next(queue, None)
            if nxt is not None:
                ordered.append(nxt)
                survivors.append(queue)
        queues = survivors
    return ordered


LONG_CONTEXT_DESCRIPTION = (
    "Real Claude Code sessions from this cluster, exported and redacted. This is the "
    "only workload here that is not synthetic: the prefixes are genuine multi-hour "
    "agent conversations with real tool calls, real file contents, real failures and "
    "real recoveries, so its length distribution and its prefix/decode ratio are "
    "measurements rather than constructions. It is also the longest by a wide margin "
    "and the one most likely to be refused by a preflight against a small serving "
    "window -- that refusal is the point. Caveats: the sessions are one operator's on "
    "one project family, so topic diversity is low even though structural diversity is "
    "high; content is redacted (home paths, absolute paths, secret-shaped strings), "
    "which perturbs a small fraction of characters; and one trace is taken per session "
    "file, deepest turn first, so the workload is deliberately biased toward long "
    "context and is NOT a uniform sample of turns."
)


def build_long_context(
    tokenizer: Tokenizer,
    limit: int | None,
    *,
    max_files: int,
    minimum_assistant_chars: int,
    minimum_prompt_chars: int,
    per_file_limit: int,
) -> tuple[list[dict], dict]:
    files = sorted(SESSION_ROOT.rglob("*.jsonl"))
    total_files = len(files)
    scanned = _round_robin_by_project(files)[:max_files]
    model_counter: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    sessions: set[str] = set()
    started = time.monotonic()
    for path in scanned:
        if limit is not None and len(records) >= limit:
            break
        for trace in traces_from_session_file(
            path,
            minimum_assistant_chars=minimum_assistant_chars,
            minimum_prompt_chars=minimum_prompt_chars,
            per_file_limit=per_file_limit,
            model_counter=model_counter,
        ):
            sessions.add(trace.session_id or trace.source_file)
            records.append(
                make_record(
                    trace.trace_id,
                    list(trace.messages),
                    tokenizer,
                    completion=trace.completion,
                    timestamps=trace.timestamps,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    provenance = {
        "upstream": [
            {
                "path": str(SESSION_ROOT),
                "sha256": None,
                "size_bytes": None,
                "note": (
                    "a live directory tree, not a file: it cannot be digested, so this "
                    "workload's reproducibility rests on the materialized records file "
                    "digest below, not on the upstream."
                ),
            }
        ],
        "method": (
            "ported from qwen-trace-pilot core.py. Per session file, walk records newest "
            "first, take up to per_file_limit assistant turns, reconstruct the ancestor "
            "chain as messages, keep the target assistant turn as completion, carry every "
            "turn's timestamp. Fixes over upstream: absolute-path redaction now covers "
            "/admin/home layouts; the 800-char assistant floor is configurable and set "
            "low; per-turn timestamps are retained; the hard Qwen-only model filter is "
            "replaced by 'any real model id' (upstream would export zero records here)."
        ),
        "bounds": {
            "session_files_available": total_files,
            "session_files_scanned": len(scanned),
            "scan_order": (
                "round-robin across top-level project directories, not sorted path "
                "order: sorted order put 92% of the first 1400 files in one project's "
                "subagents/ directory"
            ),
            "max_files": max_files,
            "max_records": limit,
            "per_file_limit": per_file_limit,
            "minimum_assistant_chars": minimum_assistant_chars,
            "minimum_prompt_chars": minimum_prompt_chars,
            "scan_seconds": round(time.monotonic() - started, 1),
        },
        "diversity": {
            "distinct_sessions": len(sessions),
            "assistant_models": dict(model_counter.most_common()),
        },
        "redaction": {
            "home_path": True,
            "absolute_paths": ABS_PATH_PATTERN.pattern,
            "secret_patterns": len(SECRET_PATTERNS),
            "base64_blob_rejection": True,
        },
    }
    return records, provenance


# ---------------------------------------------------------------------------
# Manifest emission
# ---------------------------------------------------------------------------


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record) + "\n")


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def emit_manifest(
    *,
    name: str,
    version: str,
    description: str,
    domain: str,
    records_path: Path,
    records: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    tokenizer: Tokenizer,
    output_reserve: int,
    needs_tool_support: bool,
    ground_truth: dict[str, Any] | None,
    spec_dir: Path,
) -> Path:
    typed = tuple(
        WorkloadRecord(
            id=r["id"],
            messages=tuple(r["messages"]),
            tools=tuple(r.get("tools") or ()),
            prompt_chars=r["prompt_chars"],
            prompt_tokens=r["prompt_tokens"],
            completion=r.get("completion"),
            timestamps=tuple(r.get("timestamps") or ()) or None,
            ground_truth=r.get("ground_truth"),
        )
        for r in records
    )
    characteristics = recompute_characteristics(typed)
    longest = max(record.prompt_tokens for record in typed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "version": version,
        "description": description,
        "domain": domain,
        "source": {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
            "size_bytes": records_path.stat().st_size,
            "format": RECORD_FORMAT,
        },
        "provenance": {
            "built_by": "scripts/build_workload.py",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "token_basis": tokenizer.basis,
            **provenance,
        },
        "characteristics": characteristics,
        "tolerance": dict(DEFAULT_TOLERANCE),
        "requirements": {
            "min_max_model_len": round_up(longest + output_reserve, 512),
            "output_reserve_tokens": output_reserve,
            "longest_prompt_tokens": longest,
            "needs_tool_support": needs_tool_support,
            "basis": (
                "longest prompt in the corpus plus an output reserve, rounded up to a "
                "multiple of 512. A launch below this truncates the workload; a preflight "
                "must refuse rather than measure a truncated corpus."
            ),
        },
        "bands": dict(DEFAULT_BANDS),
        "band_metric": "prompt_chars",
        "sampling": {"seed": 20260807},
    }
    if ground_truth:
        manifest["ground_truth"] = ground_truth
    spec_dir.mkdir(parents=True, exist_ok=True)
    target = spec_dir / f"{name}.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

WORKLOADS = ("generic-chat", "agentic-tool-loop", "agentic-mixed-outcome", "long-context-sessions")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workloads", nargs="+", help=f"one or more of {', '.join(WORKLOADS)}, or 'all'"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=SPEC_DIR)
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=None,
        help="skip the real tokenizer and estimate. Recorded in the manifest as an estimate.",
    )
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--limit", type=int, default=None, help="cap records (smoke runs)")
    parser.add_argument("--output-reserve", type=int, default=2048)
    parser.add_argument("--max-files", type=int, default=1400, help="long-context-sessions only")
    parser.add_argument("--minimum-assistant-chars", type=int, default=1)
    parser.add_argument("--minimum-prompt-chars", type=int, default=2000)
    parser.add_argument("--per-file-limit", type=int, default=1)
    args = parser.parse_args(argv)

    names = list(WORKLOADS) if "all" in args.workloads else list(args.workloads)
    unknown = [name for name in names if name not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s): {', '.join(unknown)}")

    os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = Tokenizer(args.tokenizer, args.chars_per_token)
    print(f"token basis: {tokenizer.basis}")

    for name in names:
        started = time.monotonic()
        ground_truth: dict[str, Any] | None = None
        if name == "generic-chat":
            records, provenance = build_generic_chat(tokenizer, args.limit)
            description, domain, tools_needed = GENERIC_CHAT_DESCRIPTION, "generic-chat", False
        elif name == "agentic-tool-loop":
            records, provenance = build_tool_loop(tokenizer, args.limit)
            description, domain, tools_needed = TOOL_LOOP_DESCRIPTION, "agentic-coding", True
        elif name == "agentic-mixed-outcome":
            records, provenance = build_mixed_outcome(tokenizer, args.limit)
            description, domain, tools_needed = MIXED_DESCRIPTION, "agentic-coding", True
            ground_truth = mixed_ground_truth(records)
        else:
            records, provenance = build_long_context(
                tokenizer,
                args.limit,
                max_files=args.max_files,
                minimum_assistant_chars=args.minimum_assistant_chars,
                minimum_prompt_chars=args.minimum_prompt_chars,
                per_file_limit=args.per_file_limit,
            )
            description, domain, tools_needed = LONG_CONTEXT_DESCRIPTION, "agentic-coding", False

        if not records:
            print(f"{name}: produced NO records; refusing to write a manifest", file=sys.stderr)
            return 1
        records_path = args.output_root / name / "records.jsonl"
        write_records(records_path, records)
        manifest_path = emit_manifest(
            name=name,
            version=args.version,
            description=description,
            domain=domain,
            records_path=records_path,
            records=records,
            provenance=provenance,
            tokenizer=tokenizer,
            output_reserve=args.output_reserve,
            needs_tool_support=tools_needed,
            ground_truth=ground_truth,
            spec_dir=args.spec_dir,
        )
        elapsed = time.monotonic() - started
        print(
            f"{name}: {len(records)} records -> {records_path} "
            f"({records_path.stat().st_size / 1e6:.1f} MB), manifest {manifest_path} "
            f"[{elapsed:.1f}s]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
