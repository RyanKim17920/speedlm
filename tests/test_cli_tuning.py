from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import speedlm.cli as cli
from speedlm.config import IdleTuningConfig


@pytest.mark.parametrize(
    ("configured", "flag", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (False, "--enable-idle-tuning", True),
        (True, "--no-enable-idle-tuning", False),
    ],
)
def test_cli_flag_overrides_config_and_omission_inherits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
    flag: str | None,
    expected: bool,
) -> None:
    config_path = tmp_path / "serve-config.json"
    config_path.write_text(
        json.dumps(
            cli.SpeedLMConfig(
                model="openai/gpt-oss-20b",
                tuning_enabled=configured,
                tuning=IdleTuningConfig(
                    speculators_repo="/portable/speculators",
                    training_python="/portable/training-python",
                ),
            ).to_dict()
        ),
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    async def fake_run(*_args, **kwargs) -> int:
        received.update(kwargs)
        return 0

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "_run_vllm_gateway", fake_run)
    argv = [
        "vllm",
        "serve",
        "openai/gpt-oss-20b",
        "--config",
        str(config_path),
    ]
    if flag is not None:
        argv.append(flag)

    assert cli.main(argv) == 0

    assert received["enable_tuning"] is expected
    tuning_config = received.get("tuning_config")
    if expected:
        assert isinstance(tuning_config, cli.SpeedLMConfig)
        assert tuning_config.tuning_enabled
    else:
        assert tuning_config is None


def test_missing_portable_training_config_fails_before_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config_path = tmp_path / "serve-config.json"
    config_path.write_text(
        json.dumps(
            cli.SpeedLMConfig(
                model="openai/gpt-oss-20b",
                tuning_enabled=True,
            ).to_dict()
        ),
        encoding="utf-8",
    )
    supervisor_instances: list[object] = []

    class NeverStartedSupervisor:
        def __init__(self, *_args, **_kwargs) -> None:
            supervisor_instances.append(self)

        async def start(self, _draft: object) -> None:
            raise AssertionError("child launch must not be attempted")

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "VLLMSupervisor", NeverStartedSupervisor)

    code = cli.main(
        [
            "vllm",
            "serve",
            "openai/gpt-oss-20b",
            "--config",
            str(config_path),
        ]
    )

    assert code == 1
    assert supervisor_instances == []
    error = capsys.readouterr().err
    assert "requires tuning.speculators_repo and tuning.training_python" in error


def test_tuned_gateway_starts_service_and_awaits_stop_before_child_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    capture = object()

    class FakeAdmission:
        def __init__(self, _activity: object) -> None:
            events.append("admission.created")

        def stop_admitting(self) -> None:
            events.append("admission.stopped")

    class FakeSupervisor:
        def __init__(
            self,
            *,
            on_pid_changed,
            **_kwargs,
        ) -> None:
            self._on_pid_changed = on_pid_changed

        async def start(self, draft: object) -> None:
            assert draft == "base-draft"
            events.append("supervisor.started")
            self._on_pid_changed(4321)

        async def wait_ready(self, *, timeout: float | None = None) -> None:
            assert timeout == 7.0
            events.append("supervisor.ready")

        async def wait(self) -> int:
            events.append("supervisor.waiting")
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def restart(
            self,
            _draft: object,
            *,
            timeout_seconds: float,
        ) -> None:
            del timeout_seconds

        async def shutdown(self) -> int:
            events.append("supervisor.shutdown")
            self._on_pid_changed(None)
            return 0

    class FakeService:
        def start(self) -> None:
            events.append("service.started")

        def stop(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 30.0
            events.append("service.stopped")

    class FakeHTTP:
        def __init__(self, url: str, *, model: str) -> None:
            assert url == "http://127.0.0.1:8123"
            assert model == "openai/gpt-oss-20b"
            events.append("http.created")

        def wait_ready(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 7.0
            events.append("http.ready")

        def close(self) -> None:
            events.append("http.closed")

    class FakeServer:
        started = True
        should_exit = False

        def __init__(self, _config: object) -> None:
            events.append("server.created")

        async def serve(self) -> None:
            events.append("server.served")
            await asyncio.sleep(0)

    def fake_create_app(*_args, **kwargs):
        assert isinstance(kwargs["admission"], FakeAdmission)
        events.append("app.created")
        return SimpleNamespace(state=SimpleNamespace(capture=capture))

    def fake_create_tuner(*_args, **kwargs) -> FakeService:
        assert kwargs["capture"] is capture
        assert isinstance(kwargs["admission"], FakeAdmission)
        events.append("service.created")
        return FakeService()

    async def fake_to_thread(function, *args, **kwargs):
        is_service_stop = function.__name__ == "stop"
        if is_service_stop:
            events.append("service.stop_scheduled")
        await asyncio.sleep(0)
        result = function(*args, **kwargs)
        if is_service_stop:
            events.append("service.stop_awaited")
        return result

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "reserve_loopback_port", lambda: 8123)
    monkeypatch.setattr(
        cli,
        "build_tuning_launch_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            profile=object(),
            active_draft="base-draft",
            argv_factory=lambda _draft: ["vllm"],
        ),
    )
    monkeypatch.setattr(cli, "AdmissionGate", FakeAdmission)
    monkeypatch.setattr(cli, "VLLMSupervisor", FakeSupervisor)
    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli, "VLLMControlClient", FakeHTTP)
    monkeypatch.setattr(cli, "create_production_tuner", fake_create_tuner)
    monkeypatch.setattr(cli, "_GatewayServer", FakeServer)
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "_shutdown_signals",
        lambda _callback: contextlib.nullcontext(),
    )
    monkeypatch.setattr(cli.asyncio, "to_thread", fake_to_thread)
    config = cli.SpeedLMConfig(
        model="openai/gpt-oss-20b",
        startup_timeout_seconds=7.0,
        tuning_enabled=True,
    )

    result = asyncio.run(
        cli._run_tuned_vllm_gateway(
            config.model,
            wrapper=config.wrapper,
            passthrough=[],
            store=cli.TraceStore(tmp_path / "traces" / "traces.jsonl"),
            exchange_ledger=None,
            config=config,
            activity=None,
        )
    )

    assert result == 0
    for before, after in (
        ("supervisor.ready", "service.created"),
        ("http.ready", "service.created"),
        ("service.created", "service.started"),
        ("service.started", "server.served"),
        ("admission.stopped", "service.stop_scheduled"),
        ("service.stop_scheduled", "service.stopped"),
        ("service.stopped", "service.stop_awaited"),
        ("service.stop_awaited", "http.closed"),
        ("http.closed", "supervisor.shutdown"),
    ):
        assert events.index(before) < events.index(after)
