from __future__ import annotations

from pathlib import Path

import pytest
from conftest import running_gateway, send_chat


def test_no_gpu_tuning_refusal_does_not_prevent_serving(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    with running_gateway(
        speedlm_home,
        fake_vllm_bin,
        enable_idle_tuning=True,
    ) as session:
        response = send_chat(session)
        assert response["choices"]

    assert session.process.returncode in (0, 143)
    assert "Traceback (most recent call last)" not in session.stdout + session.stderr


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src/speedlm/cli.py:64-69 logs the no-GPU tuner refusal at INFO without "
        "configuring CLI logging, so the user sees no refusal explanation"
    ),
)
def test_no_gpu_tuning_refusal_is_explained_to_the_user(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    with running_gateway(
        speedlm_home,
        fake_vllm_bin,
        enable_idle_tuning=True,
    ) as session:
        send_chat(session)

    output = (session.stdout + session.stderr).lower()
    assert "idle tuner refused" in output
    assert "no usable nvidia gpu" in output or "execution mode unavailable" in output
    assert "serving will continue" in output
