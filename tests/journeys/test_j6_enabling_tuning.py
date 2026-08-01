from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conftest import (
    REPOSITORY_ROOT,
    SPEEDLM,
    assert_clean_cli_result,
    cli_environment,
    reserve_port,
    run_cli,
    running_gateway,
    send_chat,
)


def _serve_attempt(
    home: Path,
    fake_bin: Path,
    *extra_args: str,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Try to start the gateway and report how it exited.

    Used for the refusal path, where ``running_gateway`` is the wrong tool: it
    waits for readiness, and the whole point here is that readiness never comes.
    """
    return subprocess.run(
        [
            str(SPEEDLM),
            "vllm",
            "serve",
            "journey-model",
            "--host",
            "127.0.0.1",
            "--port",
            str(reserve_port()),
            *extra_args,
        ],
        cwd=REPOSITORY_ROOT,
        env=cli_environment(home, path_prefix=fake_bin),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def test_enabling_tuning_without_a_training_config_refuses_before_it_serves(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    """Asking for tuning that cannot work is refused up front, by name.

    This is fail-closed on purpose: idle tuning needs a Speculators checkout and
    a training interpreter, and starting the gateway anyway would leave an
    operator serving happily while the feature they switched on silently never
    runs.  The refusal has to name both missing keys, because "tuning is
    misconfigured" is not an actionable message.
    """
    attempt = _serve_attempt(speedlm_home, fake_vllm_bin, "--enable-idle-tuning")

    assert attempt.returncode == 1, attempt.stdout + attempt.stderr
    output = attempt.stdout + attempt.stderr
    assert "Traceback (most recent call last)" not in output
    # Both missing keys are named, so the next action is obvious from the message.
    assert "tuning.speculators_repo" in output
    assert "tuning.training_python" in output
    # It refused *before* serving: nothing is left running or half-configured.
    status = run_cli(speedlm_home, "status", "--json")
    assert_clean_cli_result(status)
    assert json.loads(status.stdout)["gateway"]["state"] == "stopped"


def test_a_refused_tuning_request_does_not_prevent_plain_serving(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    """The refusal is recoverable: drop the flag and serving still works.

    The failure mode this guards against is a refusal that leaves persisted
    state behind -- a half-written config, a stale PID file, a claimed port --
    so that the obvious next thing the operator tries also fails.
    """
    refused = _serve_attempt(speedlm_home, fake_vllm_bin, "--enable-idle-tuning")
    assert refused.returncode == 1

    with running_gateway(speedlm_home, fake_vllm_bin) as session:
        response = send_chat(session)
        assert response["id"] == "chatcmpl-journey"

        status = run_cli(speedlm_home, "status", "--json")
        assert_clean_cli_result(status)
        assert json.loads(status.stdout)["gateway"]["state"] == "running"

    assert session.process.returncode in (0, 143)
    assert "Traceback (most recent call last)" not in session.stdout + session.stderr
