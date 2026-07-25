from __future__ import annotations

import os
from pathlib import Path

import pytest

from speedlm import report as report_module
from speedlm.report import GatewayState, build_status_report
from speedlm.runtime import gateway_runtime_record


def test_runtime_writer_is_seen_running_then_stale_after_pid_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))

    with gateway_runtime_record(
        host="127.0.0.1",
        port=8100,
        model="integration-model",
        pid=os.getpid(),
        guard_signals=False,
    ) as runtime:
        running = build_status_report()
        assert running.gateway.state is GatewayState.RUNNING
        assert running.gateway.record_path == runtime.path
        assert running.gateway.pid == os.getpid()

        def dead_pid_probe(pid: int, signal_number: int) -> None:
            assert pid == os.getpid()
            assert signal_number == 0
            raise ProcessLookupError

        monkeypatch.setattr(report_module.os, "kill", dead_pid_probe)
        stale = build_status_report()
        assert stale.gateway.state is GatewayState.STALE
        assert stale.gateway.pid == os.getpid()
        assert "stale record" in stale.gateway.detail
