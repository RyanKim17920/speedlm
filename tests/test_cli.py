from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import speedlm.cli as cli
import speedlm.doctor as doctor
from speedlm.cli import main


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
        enable_tuning=False,
        activity=None,
    ) -> int:
        received.update(
            model=model,
            wrapper=wrapper,
            passthrough=list(passthrough),
            store=store,
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


def test_status_home_flag_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("SPEEDLM_HOME", str(tmp_path / "ignored"))
    override = tmp_path / "override"
    code = main(["--home", str(override), "status"])
    assert code == 0
    out = capsys.readouterr()
    assert str(override) in out.out


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
