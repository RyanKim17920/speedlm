from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from speedlm.storage import (
    StorageError,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    ensure_layout,
    new_run_dir,
    read_jsonl,
    resolve_layout,
)

# ---------------------------------------------------------------------------
# resolve_layout / ensure_layout
# ---------------------------------------------------------------------------


def test_resolve_layout_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    layout = resolve_layout()
    assert layout.root == tmp_path.resolve()
    assert layout.traces_dir == tmp_path / "traces"
    assert layout.profiles_dir == tmp_path / "profiles"
    assert layout.runs_dir == tmp_path / "runs"
    # Should NOT create directories
    assert not (tmp_path / "traces").exists()


def test_resolve_layout_explicit(tmp_path: Path) -> None:
    layout = resolve_layout(tmp_path)
    assert layout.root == tmp_path


def test_ensure_layout_creates_dirs(tmp_path: Path) -> None:
    layout = ensure_layout(tmp_path)
    assert layout.root.is_dir()
    assert layout.traces_dir.is_dir()
    assert layout.profiles_dir.is_dir()
    assert layout.runs_dir.is_dir()


def test_ensure_layout_idempotent(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    ensure_layout(tmp_path)
    layout = ensure_layout(tmp_path)
    assert layout.root.is_dir()


# ---------------------------------------------------------------------------
# new_run_dir
# ---------------------------------------------------------------------------


def test_new_run_dir(tmp_path: Path) -> None:
    layout = resolve_layout(tmp_path)
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2025, 6, 15, 10, 30, 0)
    d = new_run_dir(layout, now=now)
    assert d.name == "run-20250615-103000"
    assert d.is_dir()


def test_new_run_dir_collision_suffix(tmp_path: Path) -> None:
    layout = resolve_layout(tmp_path)
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2025, 1, 1, 0, 0, 1)
    d1 = new_run_dir(layout, now=now)
    d2 = new_run_dir(layout, now=now)
    d3 = new_run_dir(layout, now=now)
    assert d1.name == "run-20250101-000001"
    assert d2.name == "run-20250101-000001-1"
    assert d3.name == "run-20250101-000001-2"


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_text(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    atomic_write_text(path, "hello world")
    assert path.read_text() == "hello world"
    # No .tmp files left behind
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert tmp_files == []


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    atomic_write_json(path, {"key": "value", "num": 42})
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    obj = json.loads(content)
    assert obj == {"key": "value", "num": 42}
    # No .tmp files left behind
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def test_append_and_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    for i in range(3):
        append_jsonl(path, {"idx": i})
    results = list(read_jsonl(path))
    assert len(results) == 3
    assert [r["idx"] for r in results] == [0, 1, 2]


def test_read_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        list(read_jsonl(tmp_path / "missing.jsonl"))


def test_read_jsonl_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nNOT JSON\n{"also": true}\n', encoding="utf-8")
    with pytest.raises(StorageError, match="line 2"):
        list(read_jsonl(path))


def test_read_jsonl_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "non_obj.jsonl"
    path.write_text('{"ok": true}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(StorageError, match="line 2"):
        list(read_jsonl(path))


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
    results = list(read_jsonl(path))
    assert len(results) == 2
    assert results[0] == {"a": 1}
    assert results[1] == {"b": 2}