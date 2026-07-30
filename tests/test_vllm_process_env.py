from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from speedlm.gateway.process import VLLMProcess


def test_environment_overrides_reach_the_real_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "child-env.json"
    monkeypatch.setenv("SPEEDLM_PROCESS_ENV_TEST", "parent")
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    overrides = {
        "SPEEDLM_PROCESS_ENV_TEST": "child",
        "SPEEDLM_PROCESS_ENV_OUTPUT": str(output_path),
        "VLLM_SERVER_DEV_MODE": "1",
    }
    child = VLLMProcess(
        [
            sys.executable,
            "-c",
            (
                "import json, os, pathlib; "
                "pathlib.Path(os.environ['SPEEDLM_PROCESS_ENV_OUTPUT']).write_text("
                "json.dumps({"
                "'override': os.environ['SPEEDLM_PROCESS_ENV_TEST'], "
                "'dev_mode': os.environ['VLLM_SERVER_DEV_MODE'], "
                "'unbuffered': os.environ['PYTHONUNBUFFERED']"
                "}), encoding='utf-8')"
            ),
        ],
        health_url="http://127.0.0.1:1/health",
        env_overrides=overrides,
    )
    # Construction takes a defensive copy; later caller mutation must not alter
    # the environment eventually handed to the child.
    overrides["VLLM_SERVER_DEV_MODE"] = "0"

    async def scenario() -> None:
        await child.start()
        async with asyncio.timeout(5.0):
            while child.returncode is None:
                await asyncio.sleep(0.01)
        assert await child.shutdown() == 0

    asyncio.run(scenario())

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "override": "child",
        "dev_mode": "1",
        "unbuffered": "1",
    }
    assert os.environ["SPEEDLM_PROCESS_ENV_TEST"] == "parent"


@pytest.mark.parametrize(
    "overrides",
    [
        {"": "value"},
        {"BAD=NAME": "value"},
        {"BAD\0NAME": "value"},
        {"NAME": "bad\0value"},
        {1: "value"},
        {"NAME": 1},
    ],
)
def test_environment_overrides_reject_invalid_os_environ_entries(
    overrides: dict[Any, Any],
) -> None:
    with pytest.raises(ValueError, match="valid string names and values"):
        VLLMProcess(
            ["vllm"],
            health_url="http://127.0.0.1:1/health",
            env_overrides=overrides,  # type: ignore[arg-type]
        )
