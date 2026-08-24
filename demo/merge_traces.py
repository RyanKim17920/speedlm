#!/usr/bin/env python
"""Merge sharded traffic capture stores into one trace store.

demo/traffic.sbatch runs N shards concurrently, each writing its own
``$SPEEDLM_HOME/traces/traces.jsonl``. This folds them, plus any earlier
single-node runs, into one store.

    demo/merge_traces.py \
        --out /data/ryan.kim/speedlm-runs/bigcorpus-merged/traces.jsonl \
        --store /data/ryan.kim/speedlm-runs/bigcorpus-run1/speedlm_home/traces/traces.jsonl \
        --glob '/data/ryan.kim/speedlm-runs/bigcorpus-shard*/speedlm_home/traces/traces.jsonl'

THE DE-DUPLICATION KEY IS ``id``, AND IT IS NOT INVENTED
--------------------------------------------------------
``TraceRecord.id`` (src/speedlm/traces/store.py) is the upstream completion id
-- ``chatcmpl-<16 hex>`` as vLLM issued it -- falling back to
``"tr-" + sha256(canonical_json)[:16]`` when a source carried none
(traces/normalize.py ``_generate_id``). Either way it is already the store's
own identity for a record, so merging on it is exactly the store's notion of
"the same record", not a new one imposed from outside.

It matters here because the shards do NOT partition the input space perfectly
by construction: bigcorpus-run1 was itself seeded with run5's 1065 records, so
naming both run1 and run5 on the command line double-counts every one of them
unless the merge is keyed. Passing overlapping sources is therefore SAFE.

``--drop-duplicate-content`` additionally collapses records that carry
different ids but identical (model, messages, tool_calls). That is OFF by
default on purpose: traffic is sampled at temperature 0.7, so two genuinely
distinct requests can legitimately share a prompt while differing in the
response -- and the response lives in ``messages``, so exact content collisions
are real duplicates rather than coincidences. The count is always REPORTED so
you can see whether the shards' seed windows actually stayed disjoint; a large
content-duplicate count means two shards ran the same seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _content_key(record: dict[str, Any]) -> str:
    """Hash of the parts that make a record's *content*, ignoring identity."""
    payload = json.dumps(
        {
            "model": record.get("model"),
            "messages": record.get("messages"),
            "tool_calls": record.get("tool_calls"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, type=Path, help="Merged store path.")
    parser.add_argument(
        "--store",
        action="append",
        default=[],
        type=Path,
        help="An input traces.jsonl. Repeatable. Overlapping inputs are safe.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Absolute glob matching input stores. Repeatable.",
    )
    parser.add_argument(
        "--drop-duplicate-content",
        action="store_true",
        help="Also collapse distinct ids whose (model, messages, tool_calls) match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not write --out.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip inputs that do not exist yet (for merging while shards still run).",
    )
    return parser.parse_args(argv)


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.glob:
        # Glob from the filesystem root so callers can pass absolute patterns.
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        if not matches and not args.allow_missing:
            raise SystemExit(f"glob matched nothing: {pattern}")
        paths.extend(matches)
    for store in args.store:
        if not store.exists():
            if args.allow_missing:
                print(f"  skip (missing): {store}")
                continue
            raise SystemExit(f"no such store: {store}")
        paths.append(store)
    # Same file named twice (e.g. by --store and --glob) is deduped by path
    # first; identical content reached by two different paths still dedupes on
    # the record id below.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inputs = _resolve_inputs(args)
    if not inputs:
        raise SystemExit("no input stores resolved")

    by_id: dict[str, dict[str, Any]] = {}
    per_source: list[tuple[Path, int, int]] = []
    id_dupes = 0
    malformed = 0

    # Ingestion ordinal per record: (source index, line number). Capture
    # timestamps have one-second resolution, so ~20% of rows tie. Breaking a
    # tie by record id -- a hash -- shuffles rows that share a second into
    # arbitrary order, which can place a response BEFORE the request whose
    # prefix cites it. Self-play attestation then reports the row's own
    # generator as "generated later" and the cycle aborts. Measured on the
    # 6,176-record capture: 1,209 rows shared a timestamp and 49 prefix turns
    # were rejected purely from tie ordering.
    order: dict[str, tuple[int, int]] = {}
    for source_index, path in enumerate(inputs):
        read = 0
        added = 0
        with path.open() as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                rid = record.get("id")
                if not isinstance(rid, str) or not rid:
                    malformed += 1
                    continue
                if rid in by_id:
                    id_dupes += 1
                    continue
                by_id[rid] = record
                order[rid] = (source_index, line_number)
                added += 1
        per_source.append((path, read, added))

    records = sorted(
        by_id.values(),
        key=lambda r: ((r.get("timestamp") or 0.0), *order[r["id"]]),
    )

    # Content-duplicate census, always computed, dropped only on request.
    content_counts: Counter[str] = Counter(_content_key(r) for r in records)
    content_dupes = sum(count - 1 for count in content_counts.values() if count > 1)
    if args.drop_duplicate_content:
        kept: list[dict[str, Any]] = []
        seen_content: set[str] = set()
        for record in records:
            key = _content_key(record)
            if key in seen_content:
                continue
            seen_content.add(key)
            kept.append(record)
        records = kept

    total_prompt = 0
    total_completion = 0
    untokenized = 0
    for record in records:
        prompt = record.get("prompt_tokens")
        completion = record.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            total_prompt += prompt
            total_completion += completion
        else:
            untokenized += 1

    payload = "".join(json.dumps(r, sort_keys=False) + "\n" for r in records)
    size_bytes = len(payload.encode("utf-8"))

    print("=== sources ===")
    for path, read, added in per_source:
        print(f"  {read:>7} read  {added:>7} new   {path}")
    print()
    print("=== merge ===")
    print(f"  records in            : {sum(r for _, r, _ in per_source)}")
    print(f"  dropped (duplicate id): {id_dupes}")
    print(f"  dropped (malformed)   : {malformed}")
    print(
        f"  duplicate content sets: {content_dupes} "
        f"({'dropped' if args.drop_duplicate_content else 'kept'})"
    )
    print(f"  records out           : {len(records)}")
    print()
    print("=== corpus ===")
    print(
        f"  total tokens : {total_prompt + total_completion} "
        f"(prompt {total_prompt}, completion {total_completion})"
    )
    if records:
        print(f"  mean tokens  : {(total_prompt + total_completion) / len(records):.1f} per record")
        print(f"  mean bytes   : {size_bytes / len(records):.0f} per record")
    print(f"  records without token counts: {untokenized}")
    print(f"  size         : {size_bytes} bytes ({_human(size_bytes)})")

    if args.dry_run:
        print("\n(dry run; nothing written)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
