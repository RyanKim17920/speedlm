from __future__ import annotations

import json
from pathlib import Path

from conftest import assert_clean_cli_result, run_cli


def test_bootstrap_mixed_jsonl_reports_partial_success_and_token_provenance(
    speedlm_home: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-openai.jsonl"
    records: list[object] = [
        {
            "id": "measured",
            "model": "journey-model",
            "created": 1700000000,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Measured answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
        {
            "model": "journey-model",
            "messages": [{"role": "user", "content": "Estimate these tokens"}],
        },
    ]
    lines = [
        json.dumps(records[0]),
        '{"model": "broken",',
        json.dumps(records[1]),
        json.dumps({"unexpected": True}),
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    imported = run_cli(speedlm_home, "traces", "import", str(source))
    assert_clean_cli_result(imported)
    assert "imported 2 record(s)" in imported.stdout
    assert "openai-response: 1" in imported.stdout
    assert "bare-conversation: 1" in imported.stdout
    assert "rejected: 2" in imported.stdout
    assert "line 2: malformed JSON" in imported.stdout
    assert "line 4:" in imported.stdout
    assert "cannot recover a conversation" in imported.stdout
    assert "2 record(s) rejected during import" in imported.stderr

    stats = run_cli(speedlm_home, "traces", "stats")
    assert_clean_cli_result(stats)
    assert "count    : 2" in stats.stdout
    assert "measured : 10" in stats.stdout
    estimated_line = next(
        line for line in stats.stdout.splitlines() if line.startswith("estimated:")
    )
    assert int(estimated_line.partition(":")[2]) > 0
