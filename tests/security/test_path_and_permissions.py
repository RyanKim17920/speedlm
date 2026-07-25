"""Proofs for path containment and local confidentiality gaps."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from speedlm.profiles import GPT_OSS_EAGLE3_PROFILE, ProfileError, load_profiles
from speedlm.storage import atomic_write_text, ensure_layout, new_run_dir, resolve_layout
from speedlm.traces.store import TraceRecord, TraceStore


@contextmanager
def _umask(value: int) -> Iterator[None]:
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.xfail(
    strict=True,
    reason="profile discovery follows *.json symlinks outside the profile directory",
)
def test_profile_loader_must_reject_symlink_escape(tmp_path: Path) -> None:
    home = tmp_path / "home"
    profiles = home / "profiles"
    profiles.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(GPT_OSS_EAGLE3_PROFILE.to_dict()),
        encoding="utf-8",
    )
    (profiles / "linked.json").symlink_to(outside)

    with pytest.raises(ProfileError, match="symlink"):
        load_profiles(home)


@pytest.mark.xfail(
    strict=True,
    reason="new_run_dir interpolates prefix without containment validation",
)
def test_run_prefix_must_not_escape_runs_directory(tmp_path: Path) -> None:
    layout = resolve_layout(tmp_path / "home")
    layout.runs_dir.mkdir(parents=True)

    created = new_run_dir(
        layout,
        prefix="../escaped",
        now=datetime(2026, 7, 25, 12, 0, 0),
    )

    assert created.resolve().is_relative_to(layout.runs_dir.resolve())


@pytest.mark.xfail(
    strict=True,
    reason="layout directories inherit the process umask instead of forcing 0700",
)
def test_speedlm_directories_must_be_owner_only(tmp_path: Path) -> None:
    with _umask(0o022):
        layout = ensure_layout(tmp_path / "home")

    for path in (layout.root, layout.traces_dir, layout.profiles_dir, layout.runs_dir):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


@pytest.mark.xfail(
    strict=True,
    reason="trace and lock files are created with mode 0644",
)
def test_trace_files_must_be_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "home" / "traces" / "traces.jsonl"
    store = TraceStore(path)
    record = TraceRecord(
        id="permissions",
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": "private prompt"},
            {"role": "assistant", "content": "private answer"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=3,
        completion_tokens=3,
    )

    with _umask(0o022):
        assert store.append(record) is not None

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((path.parent / ".traces.jsonl.lock").stat().st_mode) == 0o600


@pytest.mark.xfail(
    strict=True,
    reason="atomic files are created with mode 0644",
)
def test_atomic_metadata_files_must_be_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    with _umask(0o022):
        atomic_write_text(path, "sensitive metadata")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
