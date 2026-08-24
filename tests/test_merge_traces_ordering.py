"""Merging shard stores must preserve the order each shard recorded.

Capture timestamps have one-second resolution, so many rows tie. Breaking a
tie by record id -- a hash -- shuffles same-second rows arbitrarily and can
place a response BEFORE the request whose prefix cites it. Self-play
attestation then reports a row's own generator as "generated later" and
refuses the whole corpus. Job 391535 died that way: 49 of 37,489 prefix turns
rejected, then a retry loop until the 6-hour wall.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MERGE = REPO / "demo" / "merge_traces.py"

# Same timestamp, and the id that must come SECOND sorts FIRST as a string.
# A hash tie-break therefore inverts them; a positional tie-break does not.
GENERATOR_ID = "chatcmpl-zzzz-generated-first"
CITER_ID = "chatcmpl-aaaa-cites-it-second"
SHARED_TIMESTAMP = 1787496300.0


def _row(rid: str, *, prefix_content: str | None) -> dict:
    messages: list[dict] = [{"role": "user", "content": "do the thing"}]
    if prefix_content is not None:
        messages.append(
            {"role": "assistant", "content": prefix_content, "provenance_tag": "client_supplied"}
        )
    messages.append({"role": "assistant", "content": "the answer", "provenance_tag": "generated"})
    return {"id": rid, "timestamp": SHARED_TIMESTAMP, "messages": messages}


def _merge(tmp_path: Path, rows: list[dict]) -> list[dict]:
    shard = tmp_path / "shard0" / "speedlm_home" / "traces"
    shard.mkdir(parents=True)
    store = shard / "traces.jsonl"
    store.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "merged.jsonl"
    proc = subprocess.run(
        [sys.executable, str(MERGE), "--out", str(out), "--store", str(store)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def test_same_timestamp_rows_keep_their_recorded_order(tmp_path: Path) -> None:
    """Two rows sharing a timestamp must come out in the order the shard wrote them."""
    merged = _merge(
        tmp_path,
        [_row(GENERATOR_ID, prefix_content=None), _row(CITER_ID, prefix_content="the answer")],
    )
    ids = [r["id"] for r in merged]
    assert ids == [GENERATOR_ID, CITER_ID], (
        f"same-timestamp rows were reordered: {ids}. Sorting by id would give "
        f"{sorted(ids)}, which places the citing row before its own generator."
    )


def test_merged_corpus_still_attests_when_timestamps_tie(tmp_path: Path) -> None:
    """The ordering must be good enough for self-play attestation to pass."""
    from speedlm.training.provenance import self_play_attestation

    merged = _merge(
        tmp_path,
        [_row(GENERATOR_ID, prefix_content=None), _row(CITER_ID, prefix_content="the answer")],
    )
    att = self_play_attestation(merged, reference_rows=merged)
    assert att.attested, f"attestation failed on tied timestamps: {att.detail}"
    assert not att.unmatched
