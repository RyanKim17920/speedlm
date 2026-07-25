from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from speedlm import __version__
from speedlm.config import ConfigError, SamplingConfig, WrapperConfig
from speedlm.gateway.app import create_app
from speedlm.gateway.process import (
    LOOPBACK_HOST,
    ProcessError,
    VLLMProcess,
    build_vllm_argv,
    forwarded_signals,
    reserve_loopback_port,
)
from speedlm.report import ReportError, build_gain_report, build_status_report
from speedlm.storage import StorageError, ensure_layout
from speedlm.traces.normalize import NormalizeError, normalize_file
from speedlm.traces.store import TraceError, TraceStore


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

    serve_parser = vllm_sub.add_parser("serve", help="Launch vLLM with SpeedLM proxy")
    serve_parser.add_argument("model", help="Model name or path")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Wrapper listen host")
    serve_parser.add_argument("--port", type=int, default=8100, help="Wrapper listen port")

    # ---- traces import ----
    traces_parser = subparsers.add_parser("traces", help="Trace management")
    traces_sub = traces_parser.add_subparsers(dest="traces_command")

    import_parser = traces_sub.add_parser("import", help="Import OpenAI-format JSONL traces")
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

    # ---- stub subcommands ----
    subparsers.add_parser("doctor", help="Diagnose environment issues")

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_vllm_serve(
    model: str, host: str, port: int, passthrough: list[str]
) -> int:
    try:
        wrapper = WrapperConfig(host=host, port=port)
        layout = ensure_layout()
        store = TraceStore(layout.traces_dir / "traces.jsonl")
        return asyncio.run(
            _run_vllm_gateway(
                model,
                wrapper=wrapper,
                passthrough=passthrough,
                store=store,
            )
        )
    except (ConfigError, ProcessError, OSError, RuntimeError) as exc:
        sys.stderr.write(f"[speedlm] error: {exc}\n")
        return 1


class _GatewayServer(uvicorn.Server):
    """Let SpeedLM own signal forwarding instead of uvicorn."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def _run_vllm_gateway(
    model: str,
    *,
    wrapper: WrapperConfig,
    passthrough: Sequence[str],
    store: TraceStore,
) -> int:
    child_port = reserve_loopback_port()
    child_url = f"http://{LOOPBACK_HOST}:{child_port}"
    child = VLLMProcess(
        build_vllm_argv(
            model,
            passthrough,
            host=LOOPBACK_HOST,
            port=child_port,
        ),
        health_url=f"{child_url}/health",
    )
    sys.stderr.write(
        f"[speedlm] launching vLLM on {LOOPBACK_HOST}:{child_port}; "
        f"gateway listening on {wrapper.host}:{wrapper.port}\n"
    )
    received_signal: int | None = None
    server: _GatewayServer | None = None

    def on_signal(signum: int) -> None:
        nonlocal received_signal
        received_signal = signum
        if server is not None:
            if server.should_exit:
                server.force_exit = True
            server.should_exit = True

    await child.start()
    try:
        with forwarded_signals(child, on_signal):
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
                sampling=SamplingConfig(),
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
            done, _ = await asyncio.wait(
                {server_task, child_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if child_task in done:
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
        await child.shutdown()


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

        out_lines: list[str] = []
        out_lines.append(f"accepted: {accepted}")
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


def _cmd_stub(name: str, description: str) -> int:
    sys.stderr.write(
        f"[speedlm] {name}: not yet implemented (will {description})\n"
    )
    return 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Pre-scan for --home to set env before layout resolution
    home_idx = None
    for i, arg in enumerate(argv):
        if arg == "--home":
            home_idx = i
            break
    if home_idx is not None and home_idx + 1 < len(argv):
        os.environ["SPEEDLM_HOME"] = argv[home_idx + 1]

    # Handle --version before argparse
    if "--version" in argv:
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
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8100)
            s_args, s_unknown = sub.parse_known_args(serve_argv)
            return _cmd_vllm_serve(s_args.model, s_args.host, s_args.port, s_unknown)

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

    # ---- stubs ----
    if command == "doctor":
        return _cmd_stub("doctor", "diagnose GPU, disk, and config issues")

    # Fallback (should not be reached)
    sys.stderr.write(f"[speedlm] error: unknown command: {command}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
