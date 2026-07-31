"""Extract first-user-turn prompts from the cached ultrachat_200k test split.

Reads the cached parquet file (no network needed) and emits a JSONL corpus
where each line is a bare-conversation record:

    {"messages": [{"role": "user", "content": "..."}]}

This shape is recognised by `speedlm.traces.normalize._detect_shape` as
"bare-conversation", so the existing normaliser handles it without changes.

Filters applied (documented here for traceability):
- Drop rows whose first user message is empty or whitespace-only.
- Drop first-user messages longer than 4096 characters.  The e2e harness caps
  ``max_tokens`` at 512, so a prompt exceeding 4096 chars is almost certainly
  going to hit context limits or produce truncated, low-quality traces.  The
  p99 of this corpus lands around 1-2k chars, so 4096 is a generous ceiling
  that catches only clear outliers (pasted source code, book chapters, etc.).
- Exact-duplicate deduplication on the final prompt string (order-preserving;
  first occurrence kept).

Usage:
    python scripts/prepare_ultrachat_corpus.py \
        --input /data/ryan.kim/hf-cache/.../test_sft-....parquet \
        --output /data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to ultrachat parquet")
    parser.add_argument("--output", required=True, help="Destination JSONL path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    table = pq.read_table(input_path)
    messages_col = table.column("messages")

    filtered: list[str] = []
    dropped_empty = 0
    dropped_long = 0
    dropped_dupe = 0
    seen: set[str] = set()

    for i in range(table.num_rows):
        messages = list(messages_col[i])
        if not messages:
            dropped_empty += 1
            continue
        first = dict(messages[0])
        content = str(first.get("content", ""))
        if not content.strip():
            dropped_empty += 1
            continue
        if len(content) > 4096:
            dropped_long += 1
            continue
        if content in seen:
            dropped_dupe += 1
            continue
        seen.add(content)
        filtered.append(content)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for prompt in filtered:
            fh.write(
                json.dumps({"messages": [{"role": "user", "content": prompt}]})
                + "\n"
            )

    # Stats
    lengths = [len(p) for p in filtered]
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    print(f"Input rows:          {table.num_rows}")
    print(f"Dropped (empty):     {dropped_empty}")
    print(f"Dropped (>4096 ch):  {dropped_long}")
    print(f"Dropped (dupes):     {dropped_dupe}")
    print(f"Emitted prompts:     {n}")
    print()
    print("Length distribution (chars):")
    print(f"  min:    {sorted_lengths[0]}")
    print(f"  median: {sorted_lengths[n // 2]}")
    p95_idx = int(n * 0.95)
    print(f"  p95:    {sorted_lengths[p95_idx]}")
    print(f"  max:    {sorted_lengths[-1]}")
    print()
    print("Samples (truncated to 120 chars):")
    for i in (0, n // 3, 2 * n // 3):
        print(f"  [{i}]: {filtered[i][:120]}...")


if __name__ == "__main__":
    main()