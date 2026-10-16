#!/usr/bin/env python3
"""Extract the on-screen traffic examples verbatim from a SpeedLM trace store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NEEDLES = (
    "heads up -- `pipeline/rolling.py`",
    "we need to hit `2` to pass",
    "`needs_review` is where things go wrong",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_store", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    totals = {"records": 0, "prompt_tokens": 0, "completion_tokens": 0}
    with args.trace_store.open(encoding="utf-8") as handle:
        for raw in handle:
            totals["records"] += 1
            record = json.loads(raw)
            totals["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(record.get("completion_tokens") or 0)
            messages = record.get("messages") or []
            users = [item for item in messages if item.get("role") == "user"]
            assistants = [item for item in messages if item.get("role") == "assistant"]
            if not users or not assistants:
                continue
            user = str(users[-1].get("content") or "")
            for index, needle in enumerate(NEEDLES):
                if needle not in user or any(row["index"] == index for row in rows):
                    continue
                rows.append(
                    {
                        "index": index,
                        "trace_id": record.get("id"),
                        "exchange_id": record.get("exchange_id"),
                        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "prompt_tokens": record.get("prompt_tokens"),
                        "completion_tokens": record.get("completion_tokens"),
                        "input": user,
                        "output": str(assistants[-1].get("content") or ""),
                    }
                )

    if len(rows) != len(NEEDLES):
        raise SystemExit(f"found {len(rows)} of {len(NEEDLES)} required examples")
    rows.sort(key=lambda row: int(row["index"]))
    document = {
        "source": str(args.trace_store),
        "source_sha256": hashlib.sha256(args.trace_store.read_bytes()).hexdigest(),
        "summary": totals,
        "examples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
