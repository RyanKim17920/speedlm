from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from speedlm.cli import main


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


GOOD_RECORD: dict = {
    "messages": [{"role": "user", "content": "hello"}],
    "model": "test-model",
    "timestamp": 1700000000.0,
    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
}


def test_traces_import_and_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "input.jsonl"
    _write_jsonl(jsonl, [GOOD_RECORD, GOOD_RECORD])

    code = main(["traces", "import", str(jsonl)])
    assert code == 0
    out = capsys.readouterr()
    assert "accepted: 2" in out.out

    code = main(["traces", "stats"])
    assert code == 0
    out = capsys.readouterr()
    assert "count    : 2" in out.out
    assert "tokens   : 60" in out.out


def test_traces_import_mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "mixed.jsonl"
    _write_jsonl(jsonl, [GOOD_RECORD, {"bad": True}, GOOD_RECORD])

    code = main(["traces", "import", str(jsonl)])
    assert code == 0
    out = capsys.readouterr()
    assert "accepted: 2" in out.out
    assert "rejected: 1" in out.out
    assert "line 2:" in out.out


def test_traces_import_all_bad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "bad.jsonl"
    _write_jsonl(jsonl, [{"nope": 1}, {"also": "no"}])

    code = main(["traces", "import", str(jsonl)])
    assert code == 1


def test_traces_import_missing_file(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code = main(["traces", "import", "/nonexistent/path/file.jsonl"])
    assert code == 1
    out = capsys.readouterr()
    assert "not found" in out.err


def test_traces_stats_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    code = main(["traces", "stats"])
    assert code == 0
    out = capsys.readouterr()
    assert "count    : 0" in out.out


def test_vllm_serve_passthrough(capsys) -> None:
    code = main([
        "vllm", "serve", "my-model",
        "--tensor-parallel-size", "2",
        "--enable-prefix-caching",
    ])
    assert code == 2
    out = capsys.readouterr()
    assert "--tensor-parallel-size" in out.err
    assert "2" in out.err
    assert "--enable-prefix-caching" in out.err
    assert "my-model" in out.err


def test_status_returns_2(capsys) -> None:
    code = main(["status"])
    assert code == 2
    out = capsys.readouterr()
    assert "not yet implemented" in out.err


def test_gain_returns_2(capsys) -> None:
    code = main(["gain"])
    assert code == 2
    out = capsys.readouterr()
    assert "not yet implemented" in out.err


def test_doctor_returns_2(capsys) -> None:
    code = main(["doctor"])
    assert code == 2
    out = capsys.readouterr()
    assert "not yet implemented" in out.err


def test_no_args_returns_2(capsys) -> None:
    code = main([])
    assert code == 2


def test_version_returns_0(capsys) -> None:
    code = main(["--version"])
    assert code == 0