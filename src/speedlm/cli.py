from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from speedlm import __version__
from speedlm.config import ConfigError, SamplingConfig
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

    # ---- stub subcommands ----
    subparsers.add_parser("status", help="Show SpeedLM status")
    subparsers.add_parser("gain", help="Show token gain analytics")
    subparsers.add_parser("doctor", help="Diagnose environment issues")

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_vllm_serve(
    model: str, host: str, port: int, passthrough: list[str]
) -> int:
    addr = f"{host}:{port}"
    passthrough_str = " ".join(f"{a!s}" for a in passthrough) if passthrough else "(none)"
    msg = (
        f"[speedlm] vllm serve (NOT YET IMPLEMENTED)\n"
        f"  model : {model}\n"
        f"  listen: {addr}\n"
        f"  vllm argv: {passthrough_str}"
    )
    sys.stderr.write(msg + "\n")
    return 2


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

    # ---- stubs ----
    if command == "status":
        return _cmd_stub("status", "show cluster and proxy health")
    if command == "gain":
        return _cmd_stub("gain", "display token savings analytics")
    if command == "doctor":
        return _cmd_stub("doctor", "diagnose GPU, disk, and config issues")

    # Fallback (should not be reached)
    sys.stderr.write(f"[speedlm] error: unknown command: {command}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())