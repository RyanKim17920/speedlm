from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import CliResult, run_cli


def assert_actionable_rejection(result: CliResult) -> None:
    assert result.returncode != 0
    assert result.output.strip()
    assert "Traceback (most recent call last)" not in result.output
    lowered = result.output.lower()
    assert any(
        clue in lowered
        for clue in ("error", "rejected", "malformed", "not a json object", "bom")
    )


@pytest.mark.parametrize("kind", ["csv", "json-array", "directory", "missing", "bom"])
def test_wrong_file_is_rejected_cleanly(
    kind: str,
    speedlm_home: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / kind
    if kind == "csv":
        target.write_text("prompt,response\nhello,world\n", encoding="utf-8")
    elif kind == "json-array":
        target.write_text(
            json.dumps([{"model": "m", "messages": [{"role": "user", "content": "x"}]}]),
            encoding="utf-8",
        )
    elif kind == "directory":
        target.mkdir()
    elif kind == "bom":
        target.write_text(
            '\ufeff{"model":"m","messages":[{"role":"user","content":"x"}]}\n',
            encoding="utf-8",
        )

    result = run_cli(speedlm_home, "traces", "import", str(target))
    assert_actionable_rejection(result)

    if kind == "json-array":
        assert "not a JSON object" in result.stdout
    elif kind == "directory":
        assert "directory" in result.stderr.lower()
    elif kind == "missing":
        assert "file not found" in result.stderr.lower()
    elif kind == "bom":
        assert "BOM" in result.stdout


@pytest.mark.parametrize("label", ["empty-file", "zero-byte-file"])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/cli.py:377-393 prints only 'imported 0 record(s) []' when the "
        "input has no records; it gives no error or next action"
    ),
)
def test_empty_file_explains_that_jsonl_records_are_required(
    label: str,
    speedlm_home: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / label
    target.touch()

    result = run_cli(speedlm_home, "traces", "import", str(target))
    assert_actionable_rejection(result)
    assert "empty" in result.output.lower() or "no jsonl records" in result.output.lower()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/traces/normalize.py:648-649 lets UnicodeDecodeError escape, and "
        "src/speedlm/cli.py:396-400 does not catch it, so binary import shows a traceback"
    ),
)
def test_binary_file_never_exposes_a_traceback(
    speedlm_home: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "binary.jsonl"
    target.write_bytes(b"\xff\xfe\x00\x01")

    result = run_cli(speedlm_home, "traces", "import", str(target))
    assert_actionable_rejection(result)
    assert "utf-8" in result.output.lower()
    assert "text" in result.output.lower() or "jsonl" in result.output.lower()
