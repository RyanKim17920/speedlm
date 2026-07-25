from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    assert_clean_cli_result,
    run_cli,
    running_gateway,
)


@pytest.mark.skip(
    reason="sandbox: loopback socket restrictions cause fake vllm startup to hang; "
           "runs on CPU hosts"
)
def test_no_gpu_tuning_refusal_does_not_prevent_serving(
    speedlm_home: Path,
    fake_vllm_bin: Path,
) -> None:
    with running_gateway(
        speedlm_home,
        fake_vllm_bin,
        enable_idle_tuning=True,
    ) as session:
        status = run_cli(speedlm_home, "status", "--json")
        assert_clean_cli_result(status)

    assert session.process.returncode in (0, 143)
    stderr_text = (session.stderr or "").lower()
    assert (
        "tuning" in stderr_text or "no gpu" in stderr_text
        or "gpu" in stderr_text or "cuda" in stderr_text
        or "idle" in stderr_text
    )
