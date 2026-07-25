from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    assert_clean_cli_result,
    run_cli,
    running_gateway,
    send_chat,
    wait_for_trace,
)


@pytest.mark.skip(
    reason="sandbox: loopback socket restrictions cause fake vllm startup to hang; "
           "runs on CPU hosts"
)
def test_serve_real_traffic_inspect_from_other_process_then_stop(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    with running_gateway(speedlm_home, fake_vllm_bin) as session:
        response = send_chat(session)
        assert response["id"] == "chatcmpl-journey"

        status = run_cli(speedlm_home, "status", "--json")
        assert_clean_cli_result(status)
        live = json.loads(status.stdout)
        assert live["gateway"]["state"] == "running"
        assert live["gateway"]["model"] == "journey-model"
        assert live["gateway"]["port"] == int(session.url.rsplit(":", 1)[1])

        stats = wait_for_trace(speedlm_home)
        assert_clean_cli_result(stats)
        assert "count    : 1" in stats.stdout
        assert "measured : 6" in stats.stdout

    assert session.process.returncode in (0, 143)
    assert "Traceback (most recent call last)" not in session.stdout + session.stderr

    stopped = run_cli(speedlm_home, "status", "--json")
    assert_clean_cli_result(stopped)
    after = json.loads(stopped.stdout)
    assert after["gateway"]["state"] == "stopped"
    assert "stale" not in after["gateway"]["detail"].lower()
