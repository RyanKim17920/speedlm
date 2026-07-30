from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from collections.abc import Sequence
from pathlib import Path

import pytest

import speedlm.cli as cli
import speedlm.doctor as doctor
import speedlm.profiles as profiles
from speedlm.cli import main
from speedlm.profiles import ParserRegistry
from speedlm.traces.store import TraceStore


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


GOOD_RECORD: dict = {
    "messages": [{"role": "user", "content": "hello"}],
    "model": "test-model",
    "timestamp": 1700000000.0,
    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
}


def test_traces_import_and_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "input.jsonl"
    _write_jsonl(jsonl, [GOOD_RECORD, GOOD_RECORD])

    code = main(["traces", "import", str(jsonl)])
    assert code == 0
    out = capsys.readouterr()
    assert "imported 2 record(s) [internal: 2]" in out.out

    code = main(["traces", "stats"])
    assert code == 0
    out = capsys.readouterr()
    assert "count    : 2" in out.out
    assert "tokens   : 60" in out.out


def test_traces_import_mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "mixed.jsonl"
    _write_jsonl(jsonl, [GOOD_RECORD, {"bad": True}, GOOD_RECORD])

    code = main(["traces", "import", str(jsonl)])
    assert code == 0
    out = capsys.readouterr()
    assert "imported 2 record(s) [internal: 2]" in out.out
    assert "rejected: 1" in out.out
    assert "line 2:" in out.out


def test_traces_import_all_bad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "bad.jsonl"
    _write_jsonl(jsonl, [{"nope": 1}, {"also": "no"}])

    code = main(["traces", "import", str(jsonl)])
    assert code == 1


def test_traces_import_missing_file(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code = main(["traces", "import", "/nonexistent/path/file.jsonl"])
    assert code == 1
    out = capsys.readouterr()
    assert "not found" in out.err


def test_traces_import_help_explains_bootstrapping(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["traces", "import", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr()
    assert "auto-detects format" in out.out
    assert "captured" in out.out
    assert "automatically by 'speedlm vllm serve'" in out.out
    assert "import is only for bootstrapping" in out.out


def test_traces_help_uses_generic_import_description(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["traces", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr()
    assert "import traces from existing logs (auto-detects format)" in out.out


def test_traces_stats_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    code = main(["traces", "stats"])
    assert code == 0
    out = capsys.readouterr()
    assert "count    : 0" in out.out
    assert "dropped  : 0" in out.out
    assert "truncated_at_line: -" in out.out


def test_traces_stats_renders_drop_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    store = TraceStore(tmp_path / "traces" / "traces.jsonl")
    assert store.record_drop("lock_timeout")

    assert main(["traces", "stats"]) == 0

    out = capsys.readouterr()
    assert "dropped  : 1" in out.out
    assert "  lock_timeout: 1" in out.out
    assert "  capture_error: 0" in out.out


def test_vllm_serve_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    async def fake_run(
        model,
        *,
        wrapper,
        passthrough,
        store,
        exchange_ledger=None,
        enable_tuning=False,
        activity=None,
        serve_config=None,
    ) -> int:
        received.update(
            model=model,
            wrapper=wrapper,
            passthrough=list(passthrough),
            store=store,
            exchange_ledger=exchange_ledger,
        )
        return 0

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr("speedlm.cli._run_vllm_gateway", fake_run)
    code = main([
        "vllm", "serve", "my-model",
        "--tensor-parallel-size", "2",
        "--enable-prefix-caching",
    ])
    assert code == 0
    assert received["model"] == "my-model"
    assert received["passthrough"] == [
        "--tensor-parallel-size",
        "2",
        "--enable-prefix-caching",
    ]


def test_gpt_oss_profile_injects_parser_flags() -> None:
    passthrough = cli._profiled_vllm_passthrough("openai/gpt-oss-20b", [])

    assert passthrough == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        "--reasoning-parser",
        "openai_gptoss",
    ]


@pytest.mark.parametrize(
    "user_flag",
    [
        ["--tool-call-parser", "custom"],
        ["--tool-call-parser=custom"],
    ],
)
def test_user_supplied_tool_call_parser_wins(user_flag: list[str]) -> None:
    passthrough = cli._profiled_vllm_passthrough(
        "openai/gpt-oss-20b",
        user_flag,
    )

    assert passthrough[: len(user_flag)] == user_flag
    assert "openai" not in passthrough
    assert passthrough[-2:] == ["--reasoning-parser", "openai_gptoss"]


def test_unprofiled_model_injects_no_parser_flags() -> None:
    passthrough = ["--tensor-parallel-size", "2"]

    assert cli._profiled_vllm_passthrough("acme/unprofiled", passthrough) == passthrough


def test_explicit_profile_for_different_model_is_rejected() -> None:
    profile = profiles.ModelProfile(
        name="other-model",
        verifier_model="acme/other-model",
        draft_model=None,
        speculative_method="mtp",
        num_speculative_tokens=1,
        target_layer_ids=None,
        chat_template_kind="auto",
        max_seq_len=1024,
    )

    with pytest.raises(cli.ProfileError, match="does not match served model"):
        cli._profiled_vllm_passthrough(
            "acme/served-model",
            [],
            profile=profile,
        )


def test_qwen_profile_injects_tool_call_parser() -> None:
    passthrough = cli._profiled_vllm_passthrough("Qwen/Qwen3.5-9B", [])

    assert passthrough == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]


def test_user_supplied_auto_tool_choice_is_not_duplicated() -> None:
    passthrough = cli._profiled_vllm_passthrough(
        "Qwen/Qwen3.5-9B",
        ["--enable-auto-tool-choice"],
    )

    assert passthrough.count("--enable-auto-tool-choice") == 1


def test_user_tool_parser_on_unprofiled_model_gets_mandatory_flag() -> None:
    passthrough = cli._profiled_vllm_passthrough(
        "acme/unprofiled",
        ["--tool-call-parser", "custom"],
    )

    assert passthrough == [
        "--tool-call-parser",
        "custom",
        "--enable-auto-tool-choice",
    ]


def test_non_builtin_local_model_gets_auto_detected_parser_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "custom-qwen"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        profiles,
        "discover_vllm_parser_registry",
        lambda: ParserRegistry(
            tool_parsers=("hermes", "qwen3_coder", "qwen3_xml"),
            reasoning_parsers=("qwen3",),
        ),
    )

    passthrough = cli._profiled_vllm_passthrough(str(model), [])

    assert passthrough == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
    ]


def test_vllm_readiness_error_is_preserved_and_runtime_record_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = []

    class FailingVLLMProcess:
        pid = 4321

        def __init__(self, *_args, **_kwargs) -> None:
            self.shutdown_called = False
            instances.append(self)

        async def start(self) -> None:
            return None

        async def wait_ready(self) -> None:
            assert (tmp_path / "gateway.json").exists()
            raise cli.ProcessError("vLLM readiness failed")

        async def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "VLLMProcess", FailingVLLMProcess)
    monkeypatch.setattr(cli, "reserve_loopback_port", lambda: 8123)
    monkeypatch.setattr(
        cli,
        "forwarded_signals",
        lambda *_args: contextlib.nullcontext(),
    )

    with pytest.raises(cli.ProcessError, match="vLLM readiness failed"):
        asyncio.run(
            cli._run_vllm_gateway(
                "my-model",
                wrapper=cli.WrapperConfig(host="127.0.0.1", port=8100),
                passthrough=[],
                store=cli.TraceStore(tmp_path / "traces" / "traces.jsonl"),
            )
        )

    assert instances[0].shutdown_called
    assert not (tmp_path / "gateway.json").exists()


def test_vllm_signal_handlers_cover_child_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = False
    callback = None
    instances = []

    class StartingVLLMProcess:
        pid = 4321

        def __init__(self, *_args, **_kwargs) -> None:
            self.shutdown_called = False
            instances.append(self)

        async def start(self) -> None:
            assert installed
            assert callback is not None
            callback(signal.SIGTERM)

        async def shutdown(self) -> None:
            self.shutdown_called = True

    @contextlib.contextmanager
    def fake_forwarded(_child, on_signal):
        nonlocal installed, callback
        installed = True
        callback = on_signal
        try:
            yield
        finally:
            installed = False

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "VLLMProcess", StartingVLLMProcess)
    monkeypatch.setattr(cli, "reserve_loopback_port", lambda: 8123)
    monkeypatch.setattr(cli, "forwarded_signals", fake_forwarded)

    result = asyncio.run(
        cli._run_vllm_gateway(
            "my-model",
            wrapper=cli.WrapperConfig(host="127.0.0.1", port=8100),
            passthrough=[],
            store=cli.TraceStore(tmp_path / "traces" / "traces.jsonl"),
        )
    )

    assert result == 128 + signal.SIGTERM
    assert instances[0].shutdown_called
    assert not (tmp_path / "gateway.json").exists()


def test_vllm_version_is_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    received: dict[str, object] = {}

    async def fake_run(
        model,
        *,
        wrapper,
        passthrough,
        store,
        exchange_ledger=None,
        enable_tuning=False,
        activity=None,
        serve_config=None,
    ) -> int:
        received["passthrough"] = list(passthrough)
        return 0

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_run_vllm_gateway", fake_run)

    assert main(["vllm", "serve", "my-model", "--version"]) == 0
    assert received["passthrough"] == ["--version"]
    assert "speedlm " not in capsys.readouterr().out


def test_vllm_home_is_passthrough_without_changing_speedlm_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speedlm_home = tmp_path / "speedlm"
    received: dict[str, object] = {}

    async def fake_run(
        model,
        *,
        wrapper,
        passthrough,
        store,
        exchange_ledger=None,
        enable_tuning=False,
        activity=None,
        serve_config=None,
    ) -> int:
        received["passthrough"] = list(passthrough)
        return 0

    monkeypatch.setenv("SPEEDLM_HOME", str(speedlm_home))
    monkeypatch.setattr(cli, "_run_vllm_gateway", fake_run)

    assert main(["vllm", "serve", "my-model", "--home", "/vllm/home"]) == 0
    assert received["passthrough"] == ["--home", "/vllm/home"]
    assert speedlm_home.is_dir()


def test_status_fresh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    home = tmp_path / "fresh"
    monkeypatch.setenv("SPEEDLM_HOME", str(home))
    code = main(["status"])
    assert code == 0
    out = capsys.readouterr()
    assert "SpeedLM status" in out.out
    assert "no active draft" in out.out
    assert "0 record(s), 0 token(s)" in out.out
    assert not home.exists()


def test_status_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "fresh"))
    code = main(["status", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gateway"]["state"] == "stopped"
    assert payload["traces"]["count"] == 0


def test_status_after_trace_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "input.jsonl"
    _write_jsonl(jsonl, [GOOD_RECORD, GOOD_RECORD])
    assert main(["traces", "import", str(jsonl)]) == 0
    capsys.readouterr()

    assert main(["status"]) == 0
    out = capsys.readouterr()
    assert "2 record(s), 60 token(s)" in out.out


def test_gain_no_gate_ever_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "fresh"))
    code = main(["gain"])
    assert code == 0
    out = capsys.readouterr()
    assert "No gate has ever run" in out.out
    assert "tok/s" not in out.out


def test_gain_json_no_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "fresh"))
    code = main(["gain", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_gate_run"
    assert payload["measurement"] is None


def test_gain_reports_persisted_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "decision.json").write_text(
        json.dumps(
            {
                "verdict": "reject",
                "reason": "counter_reset",
                "acceptance_delta_pp": 0.0,
                "throughput_delta_pct": 0.0,
                "min_acceptance_delta_pp": 1.0,
                "min_throughput_delta_pct": 2.0,
                "num_repeats": 0,
                "per_repeat": [],
                "stock_avg_acceptance": 0.0,
                "candidate_avg_acceptance": 0.0,
                "stock_avg_tok_per_sec": 0.0,
                "candidate_avg_tok_per_sec": 0.0,
            }
        ),
        encoding="utf-8",
    )
    code = main(["gain"])
    assert code == 0
    out = capsys.readouterr()
    assert "counter_reset" in out.out
    assert "speedup           : not measured" in out.out
    assert "tok/s" not in out.out


@pytest.mark.parametrize("joined", [False, True])
def test_status_home_flag_forms_override_default_without_touching_it(
    joined: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    default = tmp_path / "user-home" / ".speedlm"
    override = tmp_path / "override"
    monkeypatch.delenv("SPEEDLM_HOME", raising=False)
    monkeypatch.setenv("HOME", str(default.parent))
    home_args = [f"--home={override}"] if joined else ["--home", str(override)]

    code = main([*home_args, "status"])

    assert code == 0
    out = capsys.readouterr()
    assert str(override) in out.out
    assert not default.exists()


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (doctor.CheckStatus.PASS, 0),
        (doctor.CheckStatus.WARN, 0),
        (doctor.CheckStatus.FAIL, 1),
    ],
)
def test_doctor_exit_code_follows_overall_status(
    status: doctor.CheckStatus,
    expected_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    report = doctor.DoctorReport(
        checks=(doctor.Check("probe", status, "result"),),
        plan=doctor.ExecutionPlan(doctor.ExecutionMode.UNAVAILABLE, "test plan"),
    )
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "run_doctor", lambda _config, *, home: report)

    assert main(["doctor"]) == expected_code
    out = capsys.readouterr()
    assert f"Overall: {status.value}" in out.out
    assert out.err == ""


def test_doctor_json_output_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    report = doctor.DoctorReport(
        checks=(doctor.Check("tuning", doctor.CheckStatus.WARN, "unavailable"),),
        plan=doctor.ExecutionPlan(doctor.ExecutionMode.IDLE, "safe mode"),
    )
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "run_doctor", lambda _config, *, home: report)

    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"status", "execution_mode", "plan", "checks"}
    assert payload["status"] == "WARN"
    assert payload["execution_mode"] == "idle"
    assert payload["checks"] == [
        {"name": "tuning", "status": "WARN", "detail": "unavailable"}
    ]


def test_doctor_driver_unreachable_is_clean_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    error = "NVIDIA-SMI has failed because it couldn't communicate with the driver."

    def unreachable(command, **_kwargs):
        return doctor.subprocess.CompletedProcess(command, 9, "", error)

    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    monkeypatch.setattr(doctor.subprocess, "run", unreachable)

    assert main(["doctor"]) == 1
    out = capsys.readouterr()
    assert "[FAIL] gpu: nvidia-smi is present but cannot communicate" in out.out
    assert "[SKIP] cuda: CUDA detection skipped" in out.out
    assert "Execution mode: unavailable" in out.out
    assert out.err == ""


def test_no_args_returns_2(capsys) -> None:
    code = main([])
    assert code == 2


def test_version_returns_0(capsys) -> None:
    code = main(["--version"])
    assert code == 0


def test_traces_import_iso_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Defect 1 regression: ISO-8601 timestamps must be accepted end-to-end."""
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "iso.jsonl"
    _write_jsonl(jsonl, [{
        "id": "t1",
        "timestamp": "2026-07-25T04:00:00Z",
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "prompt_tokens": 5,
        "completion_tokens": 3,
    }])
    code = main(["traces", "import", str(jsonl)])
    assert code == 0
    out = capsys.readouterr()
    assert "imported 1 record(s) [internal: 1]" in out.out

    code = main(["traces", "stats"])
    assert code == 0
    out = capsys.readouterr()
    assert "count    : 1" in out.out
    assert "tokens   : 8" in out.out


def test_traces_import_openai_response_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "openai.jsonl"
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
    }]
    _write_jsonl(jsonl, [{
        "id": "chatcmpl-1",
        "model": "gpt-4",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "created": 1750000000,
    }])

    assert main(["traces", "import", str(jsonl)]) == 0
    out = capsys.readouterr()
    assert "imported 1 record(s) [openai-response: 1]" in out.out

    stored = [
        json.loads(line)
        for line in (tmp_path / "traces" / "traces.jsonl").read_text().splitlines()
    ]
    assert stored[0]["id"] == "chatcmpl-1"
    assert stored[0]["timestamp"] == 1750000000.0
    assert stored[0]["messages"][0]["tool_calls"] == tool_calls
    assert stored[0]["tool_calls"] == tool_calls

    assert main(["traces", "stats"]) == 0
    out = capsys.readouterr()
    assert "count    : 1" in out.out
    assert "tokens   : 8" in out.out


def test_traces_import_bare_conversation_marks_estimated_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "bare.jsonl"
    _write_jsonl(jsonl, [{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
    }])

    assert main(["traces", "import", str(jsonl)]) == 0
    out = capsys.readouterr()
    assert "imported 1 record(s) [bare-conversation: 1]" in out.out

    stored = json.loads(
        (tmp_path / "traces" / "traces.jsonl").read_text().splitlines()[0]
    )
    assert stored["prompt_tokens"] > 0
    assert stored["completion_tokens"] >= 0
    assert stored["token_count_source"] == "estimated"

    assert main(["traces", "stats"]) == 0
    out = capsys.readouterr().out
    assert "measured : 0" in out
    assert "estimated:" in out
    assert "estimated: 0" not in out


def test_traces_import_mixed_token_sources_and_proxy_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path))
    jsonl = tmp_path / "mixed-shapes.jsonl"
    proxy = {
        "capture": {
            "request_payload": {
                "model": "proxy-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            "response_payload": {
                "choices": [{
                    "message": {"role": "assistant", "content": "hi"},
                }],
            },
        },
    }
    _write_jsonl(jsonl, [GOOD_RECORD, proxy])

    assert main(["traces", "import", str(jsonl)]) == 0
    out = capsys.readouterr()
    assert "internal: 1" in out.out
    assert "proxy-capture: 1" in out.out

    assert main(["traces", "stats"]) == 0
    out = capsys.readouterr().out
    assert "measured : 30" in out
    estimated_line = next(
        line for line in out.splitlines() if line.startswith("estimated:")
    )
    assert int(estimated_line.split(":", 1)[1]) > 0
