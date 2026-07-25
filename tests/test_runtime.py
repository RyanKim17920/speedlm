"""Tests for the gateway runtime record — no GPU, no network, no real vLLM.

These prove the *writer* half of the ``speedlm status`` contract: what
:mod:`speedlm.runtime` writes is exactly what :func:`speedlm.report
.read_gateway_status` reads back, including the stale-record case.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from speedlm.report import GatewayState, build_status_report, read_gateway_status
from speedlm.runtime import (
    GATEWAY_FILE_NAME,
    GatewayRecord,
    GatewayRuntimeRecord,
    RuntimeRecordError,
    gateway_record_path,
    gateway_runtime_record,
)
from speedlm.storage import resolve_layout


def _dead_pid() -> int:
    """Return the pid of a process that has certainly exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=60)
    return proc.pid


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------


def test_record_path_is_the_path_report_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    assert gateway_record_path() == tmp_path / GATEWAY_FILE_NAME
    assert gateway_record_path() == resolve_layout(None).root / GATEWAY_FILE_NAME


# ---------------------------------------------------------------------------
# write on start / remove on clean shutdown
# ---------------------------------------------------------------------------


def test_record_is_written_on_start_and_reported_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))

    with gateway_runtime_record(
        host="127.0.0.1",
        port=8100,
        model="openai/gpt-oss-20b",
        child_pid=os.getpid(),
    ) as runtime:
        assert runtime.path == tmp_path / GATEWAY_FILE_NAME
        record = json.loads(runtime.path.read_text(encoding="utf-8"))
        assert record["pid"] == os.getpid()
        assert record["child_pid"] == os.getpid()
        assert record["host"] == "127.0.0.1"
        assert record["port"] == 8100
        assert record["model"] == "openai/gpt-oss-20b"
        assert record["started_at"] > 0

        status = read_gateway_status(resolve_layout(tmp_path))
        assert status.state is GatewayState.RUNNING
        assert status.pid == os.getpid()
        assert status.port == 8100
        assert status.model == "openai/gpt-oss-20b"
        assert "8100" in status.detail


def test_record_is_removed_on_clean_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))

    with gateway_runtime_record(host="127.0.0.1", port=8100, model="m") as runtime:
        path = runtime.path
        assert path.exists()

    assert not path.exists()
    status = read_gateway_status(resolve_layout(tmp_path))
    assert status.state is GatewayState.STOPPED
    assert "no gateway is running" in status.detail


def test_record_is_removed_when_serve_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    path = gateway_record_path()

    with (
        pytest.raises(RuntimeError, match="vllm exploded"),
        gateway_runtime_record(host="127.0.0.1", port=8100, model="m"),
    ):
        assert path.exists()
        raise RuntimeError("vllm exploded")

    assert not path.exists()


def test_write_is_atomic_and_leaves_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    with gateway_runtime_record(host="127.0.0.1", port=8100, model="m"):
        names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [GATEWAY_FILE_NAME]


def test_home_directory_is_created_if_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "fresh" / "home"
    monkeypatch.setenv("SPEEDLM_HOME", str(home))
    with gateway_runtime_record(host="127.0.0.1", port=8100, model="m") as runtime:
        assert runtime.path.is_file()


# ---------------------------------------------------------------------------
# SIGTERM / SIGINT
# ---------------------------------------------------------------------------


class _Terminated(Exception):
    """Stand-in for the process death a default SIGTERM would cause."""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_record_is_removed_on_terminating_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signum: signal.Signals
) -> None:
    """The guard must delete the record before the signal kills the process.

    The previously-installed handler is replaced with one that raises, so the
    re-raise the guard performs is observable instead of terminating pytest.
    """
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    path = gateway_record_path()

    def outer_handler(_signum: int, _frame: object) -> None:
        raise _Terminated

    previous = signal.signal(signum, outer_handler)
    try:
        with (
            pytest.raises(_Terminated),
            gateway_runtime_record(host="127.0.0.1", port=8100, model="m"),
        ):
            assert path.exists()
            signal.raise_signal(signum)
    finally:
        signal.signal(signum, previous)

    assert not path.exists()


def test_guard_restores_the_previous_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))

    def outer_handler(_signum: int, _frame: object) -> None:  # pragma: no cover
        raise _Terminated

    previous = signal.signal(signal.SIGTERM, outer_handler)
    try:
        with gateway_runtime_record(host="127.0.0.1", port=8100, model="m"):
            assert signal.getsignal(signal.SIGTERM) is not outer_handler
        assert signal.getsignal(signal.SIGTERM) is outer_handler
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_signal_guard_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    before = signal.getsignal(signal.SIGTERM)
    with gateway_runtime_record(
        host="127.0.0.1", port=8100, model="m", guard_signals=False
    ):
        assert signal.getsignal(signal.SIGTERM) is before


# ---------------------------------------------------------------------------
# stale records
# ---------------------------------------------------------------------------


def test_record_left_by_a_killed_process_is_reported_stale(tmp_path: Path) -> None:
    """A crash cannot remove the record; it must read back as stale, not running."""
    path = tmp_path / GATEWAY_FILE_NAME
    dead = _dead_pid()
    GatewayRuntimeRecord(
        path,
        GatewayRecord(pid=dead, host="127.0.0.1", port=8100, model="m", child_pid=dead),
    ).write()

    status = read_gateway_status(resolve_layout(tmp_path))
    assert status.state is GatewayState.STALE
    assert status.pid == dead
    assert "stale record" in status.detail


def test_status_report_surfaces_a_stale_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    dead = _dead_pid()
    GatewayRuntimeRecord(
        gateway_record_path(),
        GatewayRecord(pid=dead, host="127.0.0.1", port=8100, model="m"),
    ).write()

    report = build_status_report()
    assert report.gateway.state is GatewayState.STALE
    assert "stale" in report.render_text()


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------


def test_remove_does_not_delete_a_successors_record(tmp_path: Path) -> None:
    path = tmp_path / GATEWAY_FILE_NAME
    mine = GatewayRuntimeRecord(
        path, GatewayRecord(pid=os.getpid(), host="127.0.0.1", port=8100, model="m")
    )
    mine.write()

    successor = GatewayRuntimeRecord(
        path,
        GatewayRecord(pid=os.getpid() + 1, host="127.0.0.1", port=8101, model="m"),
    )
    successor.write()

    assert mine.remove() is False
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["port"] == 8101


def test_remove_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / GATEWAY_FILE_NAME
    runtime = GatewayRuntimeRecord(
        path, GatewayRecord(pid=os.getpid(), host="127.0.0.1", port=8100, model="m")
    )
    runtime.write()
    assert runtime.remove() is True
    assert runtime.remove() is False


def test_remove_leaves_an_unreadable_record_alone(tmp_path: Path) -> None:
    path = tmp_path / GATEWAY_FILE_NAME
    runtime = GatewayRuntimeRecord(
        path, GatewayRecord(pid=os.getpid(), host="127.0.0.1", port=8100, model="m")
    )
    runtime.write()
    path.write_text("{not json", encoding="utf-8")

    assert runtime.remove() is False
    assert path.exists()
    assert read_gateway_status(resolve_layout(tmp_path)).state is GatewayState.UNREADABLE


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pid": 0, "host": "127.0.0.1", "port": 8100, "model": "m"},
        {"pid": os.getpid(), "host": "", "port": 8100, "model": "m"},
        {"pid": os.getpid(), "host": "127.0.0.1", "port": "8100", "model": "m"},
        {"pid": os.getpid(), "host": "127.0.0.1", "port": 8100, "model": ""},
        {
            "pid": os.getpid(),
            "host": "127.0.0.1",
            "port": 8100,
            "model": "m",
            "child_pid": "x",
        },
    ],
)
def test_invalid_records_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(RuntimeRecordError):
        GatewayRecord(**kwargs)  # type: ignore[arg-type]
