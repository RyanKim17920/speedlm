from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn

from speedlm import __version__
from speedlm.config import (
    ConfigError,
    SamplingConfig,
    SpeedLMConfig,
    WrapperConfig,
    load_config,
)
from speedlm.doctor import CheckStatus, run_doctor
from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.app import create_app
from speedlm.gateway.control import AdmissionGate
from speedlm.gateway.exchange import ExchangeLedger
from speedlm.gateway.process import (
    LOOPBACK_HOST,
    ProcessError,
    VLLMProcess,
    build_vllm_argv,
    forwarded_signals,
    reserve_loopback_port,
)
from speedlm.gateway.supervisor import ThreadsafeProcessControl, VLLMSupervisor
from speedlm.gateway.vllm_http import VLLMControlClient
from speedlm.profiles import (
    ModelProfile,
    ProfileError,
    canonical_verifier_reference,
    resolve_model_parsers,
    resolve_profile,
)
from speedlm.report import ReportError, build_gain_report, build_status_report
from speedlm.runtime import gateway_runtime_record
from speedlm.storage import StorageError, ensure_layout, resolve_layout
from speedlm.traces.normalize import NormalizeError, normalize_file
from speedlm.traces.store import DROP_REASONS, TraceError, TraceStore
from speedlm.tuner.artifacts import ArtifactError
from speedlm.tuner.composition import (
    ProductionTuningError,
    build_tuning_launch_plan,
    create_production_tuner,
)
from speedlm.tuner.service import TunerServiceStartupError, TunerServiceStopError

logger = logging.getLogger("speedlm.cli")

_COMMANDS = frozenset({"vllm", "traces", "status", "gain", "doctor"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speedlm",
        description="SpeedLM — speculative decoding for vLLM",
    )
    parser.add_argument("--version", action="store_true", default=False, help="Show version")
    parser.add_argument("--home", type=str, default=None, help="Override SPEEDLM_HOME")

    subparsers = parser.add_subparsers(dest="command")

    # ---- vllm serve (nested) ----
    vllm_parser = subparsers.add_parser("vllm", help="vLLM proxy commands")
    vllm_sub = vllm_parser.add_subparsers(dest="vllm_command")

    serve_parser = vllm_sub.add_parser(
        "serve",
        help="Launch vLLM with SpeedLM proxy",
        usage="speedlm vllm serve MODEL [SPEEDLM_OPTIONS] [VLLM_ARGS...]",
        description=(
            "Launch a real vLLM server behind the SpeedLM streaming and capture gateway."
        ),
        epilog="Arguments SpeedLM does not recognize are forwarded unchanged to vLLM.",
    )
    serve_parser.add_argument("model", help="Model name or path")
    serve_parser.add_argument("--host", default=None, help="Wrapper listen host")
    serve_parser.add_argument("--port", type=int, default=None, help="Wrapper listen port")
    serve_parser.add_argument("--config", default=None, help="SpeedLM JSON config")
    serve_parser.add_argument("--profile", default=None, help="Model profile override")
    serve_parser.add_argument(
        "--enable-idle-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start the idle auto-tuner in the background",
    )

    # ---- traces import ----
    traces_parser = subparsers.add_parser("traces", help="Trace management")
    traces_sub = traces_parser.add_subparsers(dest="traces_command")

    import_parser = traces_sub.add_parser(
        "import",
        help="import traces from existing logs (auto-detects format)",
        description=(
            "Import traces from existing logs (auto-detects format). "
            "Traces are captured automatically by 'speedlm vllm serve'; "
            "import is only for bootstrapping."
        ),
    )
    import_parser.add_argument("path", help="Path to JSONL file")
    import_parser.add_argument("--model", default=None, help="Default model name")
    import_parser.add_argument("--store", default=None, help="Override trace store path")

    # ---- traces stats ----
    stats_parser = traces_sub.add_parser("stats", help="Show trace store statistics")
    stats_parser.add_argument("--store", default=None, help="Override trace store path")

    # ---- status ----
    status_parser = subparsers.add_parser("status", help="Show SpeedLM status")
    status_parser.add_argument(
        "--json", action="store_true", default=False, help="Emit JSON instead of text"
    )

    # ---- gain ----
    gain_parser = subparsers.add_parser("gain", help="Show measured draft gain")
    gain_parser.add_argument(
        "--json", action="store_true", default=False, help="Emit JSON instead of text"
    )

    # ---- doctor ----
    doctor_parser = subparsers.add_parser("doctor", help="Diagnose environment issues")
    doctor_parser.add_argument(
        "--json", action="store_true", default=False, help="Emit JSON instead of text"
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_vllm_serve(
    model: str,
    host: str | None,
    port: int | None,
    passthrough: list[str],
    enable_tuning: bool | None = None,
    config_path: str | None = None,
    profile: str | None = None,
) -> int:
    try:
        layout = ensure_layout()
        candidate_config_path = (
            Path(config_path) if config_path is not None else layout.root / "config.json"
        )
        config = (
            load_config(candidate_config_path)
            if config_path is not None or candidate_config_path.exists()
            else SpeedLMConfig(model=model)
        )
        if config.model != model:
            raise ConfigError(
                f"serve model {model!r} does not match config model {config.model!r}"
            )
        if profile is not None:
            config = replace(config, profile=profile)
        effective_tuning = (
            config.tuning_enabled if enable_tuning is None else enable_tuning
        )
        if effective_tuning and (
            config.tuning.speculators_repo is None
            or config.tuning.training_python is None
        ):
            raise ConfigError(
                "idle tuning requires tuning.speculators_repo and "
                "tuning.training_python in the SpeedLM config"
            )
        config = replace(
            config,
            tuning_enabled=effective_tuning,
            wrapper=WrapperConfig(
                host=config.wrapper.host if host is None else host,
                port=config.wrapper.port if port is None else port,
            ),
        )
        store = TraceStore.from_config(
            layout.traces_dir / "traces.jsonl",
            config.buffer,
            redaction=config.redaction,
        )
        exchange_ledger = ExchangeLedger(layout.exchanges_dir)
        gateway = (
            _run_vllm_gateway(
                model,
                wrapper=config.wrapper,
                passthrough=passthrough,
                store=store,
                exchange_ledger=exchange_ledger,
                enable_tuning=True,
                tuning_config=config,
                serve_config=config,
            )
            if effective_tuning
            else _run_vllm_gateway(
                model,
                wrapper=config.wrapper,
                passthrough=passthrough,
                store=store,
                exchange_ledger=exchange_ledger,
                enable_tuning=False,
                serve_config=config,
            )
        )
        return asyncio.run(gateway)
    except (
        ConfigError,
        ProcessError,
        ProfileError,
        ProductionTuningError,
        ArtifactError,
        OSError,
        RuntimeError,
    ) as exc:
        sys.stderr.write(f"[speedlm] error: {exc}\n")
        return 1


class _GatewayServer(uvicorn.Server):
    """Let SpeedLM own signal forwarding instead of uvicorn."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _profiled_vllm_passthrough(
    model: str,
    passthrough: Sequence[str],
    *,
    profile: ModelProfile | None = None,
    home: Path | None = None,
) -> list[str]:
    if profile is not None and canonical_verifier_reference(
        profile.verifier_model
    ) != canonical_verifier_reference(model):
        raise ProfileError(
            f"profile {profile.name!r} verifier {profile.verifier_model!r} does not "
            f"match served model {model!r}"
        )
    effective = list(passthrough)
    resolution = resolve_model_parsers(
        model,
        passthrough,
        profile=profile,
        home=home,
        allow_remote_config=True,
    )

    if resolution.tool_call_parser is not None and not any(
        argument == "--enable-auto-tool-choice"
        or argument.startswith("--enable-auto-tool-choice=")
        for argument in passthrough
    ):
        effective.append("--enable-auto-tool-choice")

    for option, value, source in (
        (
            "--tool-call-parser",
            resolution.tool_call_parser,
            resolution.tool_call_parser_source,
        ),
        (
            "--reasoning-parser",
            resolution.reasoning_parser,
            resolution.reasoning_parser_source,
        ),
    ):
        if value is not None and source != "user-supplied":
            effective.extend((option, value))
    return effective


def _configured_model_alias(
    config: SpeedLMConfig,
    passthrough: Sequence[str],
) -> list[str]:
    effective = list(passthrough)
    supplied, value = _option_value(passthrough, "--served-model-name")
    if supplied:
        if value != config.alias:
            raise ConfigError(
                f"--served-model-name must match configured model alias "
                f"{config.alias!r}, got {value!r}"
            )
        return effective
    if config.model_alias:
        effective.extend(("--served-model-name", config.model_alias))
    return effective


def _option_value(
    arguments: Sequence[str],
    option: str,
) -> tuple[bool, str | None]:
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            return True, argument.partition("=")[2] or None
        if argument == option:
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                return True, arguments[index + 1]
            return True, None
    return False, None


async def _run_vllm_gateway(
    model: str,
    *,
    wrapper: WrapperConfig,
    passthrough: Sequence[str],
    store: TraceStore,
    exchange_ledger: ExchangeLedger | None = None,
    enable_tuning: bool = False,
    activity: ActivityTracker | None = None,
    tuning_config: SpeedLMConfig | None = None,
    serve_config: SpeedLMConfig | None = None,
) -> int:
    if enable_tuning:
        if tuning_config is None:
            raise ProductionTuningError(
                "idle tuning requires a validated SpeedLM config"
            )
        return await _run_tuned_vllm_gateway(
            model,
            wrapper=wrapper,
            passthrough=passthrough,
            store=store,
            exchange_ledger=exchange_ledger,
            config=tuning_config,
            activity=activity,
        )

    tracker = activity or ActivityTracker()
    config = serve_config or SpeedLMConfig(model=model, wrapper=wrapper)
    layout = ensure_layout()
    explicit_profile = (
        resolve_profile(
            {"model": model, "profile": config.profile},
            served_model=model,
            home=layout.root,
        )
        if config.profile is not None
        else None
    )
    configured_passthrough = _configured_model_alias(config, passthrough)

    child_port = reserve_loopback_port()
    child_url = f"http://{LOOPBACK_HOST}:{child_port}"
    child = VLLMProcess(
        build_vllm_argv(
            model,
            _profiled_vllm_passthrough(
                model,
                configured_passthrough,
                profile=explicit_profile,
                home=layout.root,
            ),
            host=LOOPBACK_HOST,
            port=child_port,
        ),
        health_url=f"{child_url}/health",
        startup_timeout=config.startup_timeout_seconds,
        startup_stall_timeout=config.startup_stall_seconds,
    )
    sys.stderr.write(
        f"[speedlm] launching vLLM on {LOOPBACK_HOST}:{child_port}; "
        f"gateway listening on {wrapper.host}:{wrapper.port}\n"
    )
    received_signal: int | None = None
    server: _GatewayServer | None = None
    child_started = False

    def on_signal(signum: int) -> None:
        nonlocal received_signal
        received_signal = signum
        if server is not None:
            if server.should_exit:
                server.force_exit = True
            server.should_exit = True

    # The runtime record is what makes this gateway visible to `speedlm status`.
    # It lands in SPEEDLM_HOME (the same home `_cmd_vllm_serve` just ensured) and
    # is entered on a stack so it is removed *after* the child is reaped, and so
    # a SIGTERM arriving before uvicorn owns the signals still cleans it up.
    record_stack = contextlib.ExitStack()
    runtime_record = record_stack.enter_context(
        gateway_runtime_record(
            host=wrapper.host,
            port=wrapper.port,
            model=model,
            child_pid=None,
        )
    )
    try:
        with forwarded_signals(child, on_signal):
            await child.start()
            child_started = True
            runtime_record.update_child_pid(child.pid)
            if received_signal is not None:
                return 128 + received_signal
            try:
                await child.wait_ready()
            except ProcessError:
                if received_signal is not None:
                    return 128 + received_signal
                raise
            if received_signal is not None:
                return 128 + received_signal

            app = create_app(
                child_url,
                trace_store=store,
                sampling=config.sampling,
                exchange_ledger=exchange_ledger,
                activity=tracker,
            )
            server = _GatewayServer(
                uvicorn.Config(
                    app,
                    host=wrapper.host,
                    port=wrapper.port,
                    log_level="info",
                    lifespan="on",
                )
            )

            async def serve() -> int:
                if server is None:
                    raise RuntimeError("gateway server was not initialized")
                try:
                    await server.serve()
                except SystemExit as exc:
                    return exc.code if isinstance(exc.code, int) else 1
                return 0 if server.started else 1

            server_task = asyncio.create_task(serve())
            child_task = asyncio.create_task(child.wait())
            runtime_done, _ = await asyncio.wait(
                {server_task, child_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if child_task in runtime_done:
                child_code = child_task.result()
                server.should_exit = True
                await server_task
                if received_signal is not None:
                    return 128 + received_signal
                return _shell_exit_code(child_code)

            server_code = server_task.result()
            child_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child_task
            if received_signal is not None:
                return 128 + received_signal
            return server_code
    finally:
        try:
            if child_started:
                await child.shutdown()
        finally:
            record_stack.close()


async def _run_tuned_vllm_gateway(
    model: str,
    *,
    wrapper: WrapperConfig,
    passthrough: Sequence[str],
    store: TraceStore,
    exchange_ledger: ExchangeLedger | None,
    config: SpeedLMConfig,
    activity: ActivityTracker | None,
) -> int:
    tracker = activity or ActivityTracker()
    admission = AdmissionGate(tracker)
    admission.hold()
    child_port = reserve_loopback_port()
    child_url = f"http://{LOOPBACK_HOST}:{child_port}"
    layout = ensure_layout()
    explicit_profile = resolve_profile(
        {"model": model, "profile": config.profile},
        served_model=model,
        home=layout.root,
    )
    configured_passthrough = _configured_model_alias(config, passthrough)
    effective_passthrough = _profiled_vllm_passthrough(
        model,
        configured_passthrough,
        profile=explicit_profile,
        home=layout.root,
    )
    launch = build_tuning_launch_plan(
        config,
        passthrough=effective_passthrough,
        child_port=child_port,
        home=layout.root,
    )
    record_stack = contextlib.ExitStack()
    runtime_record = record_stack.enter_context(
        gateway_runtime_record(
            host=wrapper.host,
            port=wrapper.port,
            model=model,
            child_pid=None,
        )
    )

    def process_factory(argv: Sequence[str], *, health_url: str) -> VLLMProcess:
        return VLLMProcess(
            argv,
            health_url=health_url,
            startup_timeout=config.startup_timeout_seconds,
            startup_stall_timeout=config.startup_stall_seconds,
            env_overrides={"VLLM_SERVER_DEV_MODE": "1"},
        )

    supervisor = VLLMSupervisor(
        argv_factory=launch.argv_factory,
        health_url=f"{child_url}/health",
        process_factory=process_factory,
        on_pid_changed=runtime_record.update_child_pid,
    )
    server: _GatewayServer | None = None
    service = None
    http: VLLMControlClient | None = None
    received_signal: int | None = None
    graceful_shutdown_task: asyncio.Task[None] | None = None
    service_stop_task: asyncio.Task[None] | None = None
    server_task: asyncio.Task[int] | None = None
    child_task: asyncio.Task[int] | None = None
    shutdown_requested = asyncio.Event()

    def on_signal(signum: int) -> None:
        nonlocal received_signal, graceful_shutdown_task
        received_signal = signum
        shutdown_requested.set()
        if graceful_shutdown_task is None:
            graceful_shutdown_task = asyncio.create_task(
                stop_tuner_then_server(),
                name="speedlm-tuned-graceful-shutdown",
            )
        elif server is not None:
            server.force_exit = True
            server.should_exit = True

    async def stop_tuner() -> None:
        nonlocal service_stop_task
        if service_stop_task is None:
            service_stop_task = asyncio.create_task(
                stop_tuner_effects(),
                name="speedlm-tuner-stop",
            )
        await asyncio.shield(service_stop_task)

    async def stop_tuner_effects() -> None:
        admission.close()
        if service is not None:
            try:
                await asyncio.to_thread(
                    service.stop,
                    timeout_seconds=config.tuning.shutdown_timeout_seconds,
                )
            except TunerServiceStopError:
                logger.warning(
                    "idle tuner exceeded the graceful shutdown deadline; "
                    "waiting for its serving-recovery contract to finish"
                )
                await asyncio.to_thread(service.stop, timeout_seconds=None)

    async def stop_tuner_then_server() -> None:
        await stop_tuner()
        if server is not None:
            server.should_exit = True

    try:
        with _shutdown_signals(on_signal):
            await supervisor.start(launch.active_draft)
            if shutdown_requested.is_set():
                return 128 + (received_signal or signal.SIGTERM)
            readiness_task = asyncio.create_task(
                supervisor.wait_ready(timeout=config.startup_timeout_seconds)
            )
            startup_signal_task = asyncio.create_task(shutdown_requested.wait())
            done, _ = await asyncio.wait(
                {readiness_task, startup_signal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if startup_signal_task in done:
                readiness_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await readiness_task
                return 128 + (received_signal or signal.SIGTERM)
            startup_signal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup_signal_task
            await readiness_task
            if shutdown_requested.is_set():
                return 128 + (received_signal or signal.SIGTERM)
            http = VLLMControlClient(child_url, model=config.alias)
            try:
                await asyncio.to_thread(
                    http.wait_ready,
                    timeout_seconds=config.startup_timeout_seconds,
                    should_abort=shutdown_requested.is_set,
                )
            except Exception:
                if shutdown_requested.is_set():
                    return 128 + (received_signal or signal.SIGTERM)
                raise
            if shutdown_requested.is_set():
                return 128 + (received_signal or signal.SIGTERM)
            app = create_app(
                child_url,
                trace_store=store,
                sampling=config.sampling,
                exchange_ledger=exchange_ledger,
                activity=tracker,
                admission=admission,
                before_shutdown=stop_tuner,
            )
            capture = app.state.capture
            if capture is None:
                raise ProductionTuningError("semantic capture is required for idle tuning")
            loop = asyncio.get_running_loop()
            service = create_production_tuner(
                config,
                profile=launch.profile,
                active_draft=launch.active_draft,
                # Read off the same argv this process handed the supervisor
                # above, so the gate's decision records the regime it measured.
                engine_execution=launch.engine_execution,
                activity=tracker,
                admission=admission,
                traces=store,
                capture=capture,
                process=ThreadsafeProcessControl(supervisor, loop),
                http=http,
                child_url=child_url,
                loop=loop,
                home=layout.root,
            )
            server = _GatewayServer(
                uvicorn.Config(
                    app,
                    host=wrapper.host,
                    port=wrapper.port,
                    log_level="info",
                    lifespan="on",
                )
            )

            async def serve() -> int:
                if server is None:
                    raise RuntimeError("gateway server was not initialized")
                try:
                    await server.serve()
                except SystemExit as exc:
                    return exc.code if isinstance(exc.code, int) else 1
                return 0 if server.started else 1

            server_task = asyncio.create_task(serve())
            while not server.started:
                if server_task.done():
                    return server_task.result()
                if shutdown_requested.is_set():
                    await stop_tuner_then_server()
                    return 128 + (received_signal or signal.SIGTERM)
                await asyncio.sleep(0.01)

            service.start(paused=True)
            try:
                await asyncio.to_thread(
                    service.wait_started,
                    timeout_seconds=config.startup_timeout_seconds,
                )
            except TunerServiceStartupError:
                if shutdown_requested.is_set():
                    return 128 + (received_signal or signal.SIGTERM)
                raise
            if shutdown_requested.is_set():
                return 128 + (received_signal or signal.SIGTERM)
            admission.release()
            service.activate()
            child_task = asyncio.create_task(supervisor.wait())
            runtime_done, _ = await asyncio.wait(
                {server_task, child_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if child_task in runtime_done:
                child_code = child_task.result()
                await stop_tuner_then_server()
                await server_task
                if received_signal is not None:
                    return 128 + received_signal
                return _shell_exit_code(child_code)
            server_code = server_task.result()
            child_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child_task
            if received_signal is not None:
                return 128 + received_signal
            return server_code
    finally:
        admission.close()
        if graceful_shutdown_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await graceful_shutdown_task
        await stop_tuner_then_server()
        if server_task is not None and not server_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
        if child_task is not None and not child_task.done():
            child_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child_task
        if http is not None:
            http.close()
        try:
            await supervisor.shutdown()
        finally:
            record_stack.close()


@contextlib.contextmanager
def _shutdown_signals(
    on_signal: Callable[[int], None],
) -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: object) -> None:
        on_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for restored_signal, old_handler in previous.items():
            signal.signal(restored_signal, old_handler)


def _shell_exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def _cmd_traces_import(path_str: str, model: str | None, store: str | None) -> int:
    try:
        path = Path(path_str)
        if not path.exists():
            sys.stderr.write(f"[speedlm] error: file not found: {path}\n")
            return 1

        defaults = SamplingConfig(temperature=0.0, top_p=1.0, seed=0)
        result = normalize_file(path, defaults=defaults, default_model=model)

        layout = ensure_layout()
        store_path = Path(store) if store is not None else layout.traces_dir / "traces.jsonl"

        store_obj = TraceStore(path=store_path)
        for record in result.accepted:
            store_obj.append(record)

        accepted = result.accepted_count
        rejected = result.rejected_count

        shape_summary = ", ".join(
            f"{shape}: {count}" for shape, count in result.shape_counts.items()
        )
        out_lines = [
            f"imported {accepted} record(s) [{shape_summary}]"
        ]
        if rejected > 0:
            out_lines.append(f"rejected: {rejected}")
            for rej in result.rejected[:10]:
                out_lines.append(f"  line {rej.line}: {rej.reason}")

        sys.stdout.write("\n".join(out_lines) + "\n")

        if rejected > 0:
            sys.stderr.write(
                f"[speedlm] warning: {rejected} record(s) rejected during import\n"
            )

        if accepted == 0:
            return 1
        return 0

    except NormalizeError as e:
        sys.stderr.write(f"[speedlm] error: {e}\n")
        return 1
    except (ConfigError, StorageError, TraceError, OSError) as e:
        sys.stderr.write(f"[speedlm] error: {e}\n")
        return 1


def _cmd_traces_stats(store: str | None) -> int:
    try:
        layout = ensure_layout()
        store_path = Path(store) if store is not None else layout.traces_dir / "traces.jsonl"

        store_obj = TraceStore(path=store_path)
        stats = store_obj.stats()

        oldest_str = (
            datetime.fromtimestamp(stats.oldest, tz=UTC).isoformat()
            if stats.oldest is not None
            else "-"
        )
        newest_str = (
            datetime.fromtimestamp(stats.newest, tz=UTC).isoformat()
            if stats.newest is not None
            else "-"
        )

        lines = [
            f"count    : {stats.count}",
            f"tokens   : {stats.tokens}",
            f"measured : {stats.measured_tokens}",
            f"estimated: {stats.estimated_tokens}",
            f"dropped  : {stats.total_dropped}",
            *(
                f"  {reason}: {stats.drops_by_reason.get(reason, 0)}"
                for reason in DROP_REASONS
            ),
            "truncated_at_line: "
            f"{stats.truncated_at_line if stats.truncated_at_line is not None else '-'}",
            f"oldest   : {oldest_str}",
            f"newest   : {newest_str}",
            f"store    : {store_path}",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    except (ConfigError, StorageError, TraceError, OSError) as e:
        sys.stderr.write(f"[speedlm] error: {e}\n")
        return 1


def _cmd_status(as_json: bool) -> int:
    try:
        report = build_status_report()
    except (ConfigError, StorageError, TraceError, ReportError, OSError) as exc:
        sys.stderr.write(f"[speedlm] error: {exc}\n")
        return 1
    rendered = report.to_json() if as_json else report.render_text()
    sys.stdout.write(rendered + "\n")
    return 0


def _cmd_gain(as_json: bool) -> int:
    try:
        report = build_gain_report()
    except (ConfigError, StorageError, ReportError, OSError) as exc:
        sys.stderr.write(f"[speedlm] error: {exc}\n")
        return 1
    rendered = report.to_json() if as_json else report.render_text()
    sys.stdout.write(rendered + "\n")
    return 0


def _cmd_doctor(as_json: bool) -> int:
    try:
        layout = resolve_layout()
        config_path = layout.root / "config.json"
        config = load_config(config_path) if config_path.exists() else None
        report = run_doctor(config, home=layout.root)
    except (ConfigError, StorageError, OSError) as exc:
        sys.stderr.write(f"[speedlm] error: {exc}\n")
        return 1
    rendered = report.to_json() if as_json else report.render_text()
    sys.stdout.write(rendered + "\n")
    return 1 if report.overall_status is CheckStatus.FAIL else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _scan_global_options(argv: Sequence[str]) -> tuple[str | None, bool]:
    """Read SpeedLM globals without inspecting wrapped-command arguments."""
    home: str | None = None
    show_version = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in _COMMANDS:
            break
        if argument == "--home":
            if index + 1 < len(argv):
                home = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--home="):
            home = argument.partition("=")[2]
        elif argument == "--version":
            show_version = True
        index += 1
    return home, show_version


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="[speedlm] %(levelname)s: %(message)s",
        force=True,
    )
    # WARNING is the right default for the libraries we sit on top of -- vLLM
    # alone is thousands of INFO lines -- but it silently discarded our own.
    # The whole package emits six INFO records, and they are the only account
    # of what a background tuning cycle did: which revision it pinned, how
    # many records it leased out of how many exist, how a cycle ended, and
    # whether retention ran.  None of that reached the run artifacts, so
    # landed features could not be confirmed from a run.
    logging.getLogger("speedlm").setLevel(logging.INFO)
    parser = _build_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Stop at the subcommand: later options belong to the wrapped command.
    home, show_version = _scan_global_options(argv)
    if home is not None:
        os.environ["SPEEDLM_HOME"] = home

    # Only a global --version is ours; a later one is a vLLM passthrough option.
    if show_version:
        sys.stdout.write(f"speedlm {__version__}\n")
        return 0

    args, unknown = parser.parse_known_args(argv)

    command = args.command
    if command is None:
        parser.print_help(sys.stderr)
        return 2

    # ---- vllm ----
    if command == "vllm":
        vllm_cmd = getattr(args, "vllm_command", None)
        if vllm_cmd is None:
            sys.stderr.write("[speedlm] error: 'vllm' requires a subcommand (serve)\n")
            return 2
        if vllm_cmd == "serve":
            # Re-parse serve args to capture passthrough
            serve_argv = argv[argv.index("vllm") + 2:]
            sub = argparse.ArgumentParser(prog="speedlm vllm serve")
            sub.add_argument("model")
            sub.add_argument("--host", default=None)
            sub.add_argument("--port", type=int, default=None)
            sub.add_argument("--config", default=None)
            sub.add_argument("--profile", default=None)
            sub.add_argument(
                "--enable-idle-tuning",
                action=argparse.BooleanOptionalAction,
                default=None,
            )
            s_args, s_unknown = sub.parse_known_args(serve_argv)
            return _cmd_vllm_serve(
                s_args.model, s_args.host, s_args.port, s_unknown,
                enable_tuning=getattr(s_args, "enable_idle_tuning", False),
                config_path=s_args.config,
                profile=s_args.profile,
            )

    # ---- traces ----
    if command == "traces":
        traces_cmd = getattr(args, "traces_command", None)
        if traces_cmd is None:
            sys.stderr.write("[speedlm] error: 'traces' requires a subcommand (import, stats)\n")
            return 2
        if traces_cmd == "import":
            return _cmd_traces_import(args.path, args.model, args.store)
        if traces_cmd == "stats":
            return _cmd_traces_stats(args.store)

    # ---- status / gain ----
    if command == "status":
        return _cmd_status(bool(args.json))
    if command == "gain":
        return _cmd_gain(bool(args.json))

    # ---- doctor ----
    if command == "doctor":
        return _cmd_doctor(bool(args.json))

    # Fallback (should not be reached)
    sys.stderr.write(f"[speedlm] error: unknown command: {command}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
