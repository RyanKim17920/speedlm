from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from speedlm.training.backends.eagle3 import (
    Eagle3Backend,
    EmptySpeculatorsDatasetError,
    FinalAssistantMaskError,
    ScratchQuotaExceeded,
    SpeculatorsPipelineConfig,
    TrainingError,
)
from speedlm.training.backends.speculators_runner import (
    ProcessResult,
    RunningProcess,
)
from speedlm.training.check_prepared_dataset import _column
from speedlm.tuner.idle import TuningPreempted


@dataclass
class _FakeProcess:
    argv: tuple[str, ...]
    timeout_seconds: float
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    terminated: int = 0


RunEffect = Callable[[tuple[str, ...], Callable[[], bool]], ProcessResult]


class _FakeRunner:
    def __init__(self) -> None:
        self.run_calls: list[tuple[str, ...]] = []
        self.start_calls: list[tuple[str, ...]] = []
        self.effects: list[RunEffect] = []
        self.started: list[_FakeProcess] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> ProcessResult:
        del cwd, env, timeout_seconds
        command = tuple(argv)
        self.run_calls.append(command)
        if self.effects:
            return self.effects.pop(0)(command, should_abort)
        if should_abort():
            raise TuningPreempted("fake subprocess observed abort")
        self._create_expected_output(command)
        return ProcessResult(command, 0, "", "")

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
    ) -> RunningProcess:
        del cwd, env
        process = _FakeProcess(tuple(argv), timeout_seconds)
        self.start_calls.append(process.argv)
        self.started.append(process)
        return process

    def check_running(
        self,
        process: RunningProcess,
        *,
        should_abort: Callable[[], bool],
    ) -> int | None:
        if should_abort():
            raise TuningPreempted("fake vLLM process observed abort")
        assert isinstance(process, _FakeProcess)
        return process.returncode

    def terminate(
        self,
        process: RunningProcess,
        *,
        grace_seconds: float,
    ) -> ProcessResult:
        del grace_seconds
        assert isinstance(process, _FakeProcess)
        process.terminated += 1
        process.returncode = process.returncode if process.returncode is not None else -15
        return ProcessResult(
            process.argv,
            process.returncode,
            process.stdout,
            process.stderr,
        )

    @staticmethod
    def _create_expected_output(command: tuple[str, ...]) -> None:
        script = Path(command[1]).name if len(command) > 1 else ""
        if script in {"prepare_data.py", "data_generation_offline.py"}:
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
        if script == "train.py":
            output = Path(command[command.index("--save-path") + 1]) / "checkpoint_best"
            output.mkdir(parents=True)
            (output / "config.json").write_text("{}\n", encoding="utf-8")
            (output / "model.safetensors").write_bytes(b"weights")
            (output / "optimizer_state_dict.pt").write_bytes(b"transient")


@pytest.fixture
def pipeline(tmp_path: Path) -> SpeculatorsPipelineConfig:
    return SpeculatorsPipelineConfig(
        prepared_validator_script=tmp_path / "check_prepared_dataset.py",
        row_count=2,
        speculators_repo=tmp_path / "speculators",
        training_python=tmp_path / "venv" / "bin" / "python",
        vllm_python=tmp_path / "vllm" / "bin" / "python",
        verifier_model="/models/verifier",
        warm_start_model="/models/warm-start",
        target_layer_ids=(3, 9, 15),
        sequence_length=4096,
        learning_rate=1e-5,
        epochs=2,
        seed=7,
        port=8123,
        concurrency=4,
    )


def _backend(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
    runner: _FakeRunner,
    *,
    should_be_healthy: bool = True,
    records: Sequence[Mapping[str, object]] | None = None,
) -> tuple[Eagle3Backend, Path]:
    traces = tmp_path / "traces.jsonl"
    if records is None:
        records = [{"messages": [{"role": "assistant", "content": "ok"}]}]
    traces.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    backend = Eagle3Backend.from_speculators(
        pipeline,
        trace_source=traces,
        runner=runner,
        health_check=lambda _url, _timeout: should_be_healthy,
        sleeper=lambda _seconds: None,
    )
    return backend, tmp_path / "work"


def test_pipeline_uses_exact_stage_argv_and_separate_draft(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)

    prepared = backend.prepare(work, should_abort=lambda: False)
    hidden = backend.extract(prepared, work, should_abort=lambda: False)
    trained = backend.train(hidden, work, should_abort=lambda: False)
    draft = backend.materialize(trained, work, should_abort=lambda: False)
    backend.validate(draft, should_abort=lambda: False)

    python = str(pipeline.training_python)
    repo = pipeline.speculators_repo
    assert runner.run_calls[:4] == [
        (
            python,
            str(repo / "scripts" / "prepare_data.py"),
            "--model",
            pipeline.verifier_model,
            "--data",
            str(work / "speculators-conversations.jsonl"),
            "--output",
            str(work / "training-rows"),
            "--seq-length",
            "4096",
            "--seed",
            "7",
            "--num-preprocessing-workers",
            "0",
            "--overwrite",
        ),
        (
            python,
            str(pipeline.prepared_validator_script),
            str(work / "training-rows"),
            "2",
            "--require-nonzero-loss-mask",
            "--max-seq-len",
            "4096",
        ),
        (
            python,
            str(repo / "scripts" / "data_generation_offline.py"),
            "--endpoint",
            "http://127.0.0.1:8123/v1",
            "--preprocessed-data",
            str(work / "training-rows"),
            "--output",
            str(work / "hidden-states"),
            "--max-samples",
            "2",
            "--concurrency",
            "4",
            "--validate-outputs",
            "--fail-on-error",
        ),
        (
            python,
            str(repo / "scripts" / "train.py"),
            "--verifier-name-or-path",
            pipeline.verifier_model,
            "--from-pretrained",
            pipeline.warm_start_model,
            "--data-path",
            str(work / "training-rows"),
            "--hidden-states-path",
            str(work / "hidden-states"),
            "--on-missing",
            "raise",
            "--save-path",
            str(work / "speculators-training"),
            "--speculator-type",
            "eagle3",
            "--target-layer-ids",
            "3",
            "9",
            "15",
            "--seed",
            "7",
            "--epochs",
            "2",
            "--lr",
            "1e-05",
            "--total-seq-len",
            "4096",
            "--no-resume-from-checkpoint",
            "--save-best",
        ),
    ]
    assert runner.start_calls == [
        (
            str(pipeline.vllm_python),
            str(repo / "scripts" / "launch_vllm.py"),
            pipeline.verifier_model,
            "--hidden-states-path",
            str(work / "hidden-states"),
            "--target-layer-ids",
            "3",
            "9",
            "15",
            "--",
            "--port",
            "8123",
            "--max-num-seqs",
            "1",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.80",
        )
    ]
    validate_argv = runner.run_calls[4]
    assert validate_argv[:3] == (python, "-c", validate_argv[2])
    assert validate_argv[3:] == (
        str(work / "draft-model"),
        pipeline.verifier_model,
        "3",
        "9",
        "15",
    )
    assert "--from-pretrained" in runner.run_calls[3]
    assert draft == work / "draft-model"
    assert draft != trained.checkpoint_best
    assert not (draft / "optimizer_state_dict.pt").exists()
    assert runner.started[0].terminated == 1


def test_failing_stage_raises_typed_error_with_actual_stderr(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    runner.effects.append(
        lambda argv, _abort: ProcessResult(argv, 17, "ignored", "CUDA allocator exploded")
    )
    backend, work = _backend(tmp_path, pipeline, runner)

    with pytest.raises(TrainingError) as raised:
        backend.prepare(work, should_abort=lambda: False)

    assert raised.value.stderr == "CUDA allocator exploded"
    assert "CUDA allocator exploded" in str(raised.value)


def test_abort_between_stages_starts_no_next_subprocess(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)
    calls = len(runner.run_calls)

    with pytest.raises(TuningPreempted):
        backend.extract(prepared, work, should_abort=lambda: True)

    assert len(runner.run_calls) == calls
    assert not runner.start_calls


def test_abort_during_subprocess_cleans_transient_output(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    abort_requested = False

    def abort_inside(argv: tuple[str, ...], should_abort: Callable[[], bool]) -> ProcessResult:
        nonlocal abort_requested
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "partial.arrow").write_bytes(b"partial")
        abort_requested = True
        assert should_abort()
        raise TuningPreempted("serving preempted fake prepare")

    runner.effects.append(abort_inside)
    backend, work = _backend(tmp_path, pipeline, runner)

    with pytest.raises(TuningPreempted):
        backend.prepare(work, should_abort=lambda: abort_requested)

    assert not (work / "training-rows").exists()


def test_scratch_quota_exceeded_during_stage_cleans_output(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    limited = SpeculatorsPipelineConfig(
        **{
            field: getattr(pipeline, field)
            for field in pipeline.__dataclass_fields__
            if field != "scratch_quota_bytes"
        },
        scratch_quota_bytes=256,
    )
    runner = _FakeRunner()

    def exceed_quota(
        argv: tuple[str, ...], should_abort: Callable[[], bool]
    ) -> ProcessResult:
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "partial.arrow").write_bytes(b"x" * 1024)
        should_abort()
        raise AssertionError("quota guard did not raise")

    runner.effects.append(exceed_quota)
    backend, work = _backend(tmp_path, limited, runner)

    with pytest.raises(ScratchQuotaExceeded) as raised:
        backend.prepare(work, should_abort=lambda: False)

    assert raised.value.used_bytes > 256
    assert not (work / "training-rows").exists()


@pytest.mark.parametrize("mode", ["success", "failure", "abort"])
def test_extract_always_tears_down_vllm(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
    mode: str,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)

    if mode == "failure":
        runner.effects.append(
            lambda argv, _abort: ProcessResult(argv, 9, "", "offline generation failed")
        )
    elif mode == "abort":

        def abort(
            _argv: tuple[str, ...], should_abort: Callable[[], bool]
        ) -> ProcessResult:
            assert should_abort()
            raise TuningPreempted("serving preempted extraction")

        runner.effects.append(abort)

    checks = iter((False, True)) if mode == "abort" else None
    should_abort = (
        (lambda: next(checks, True)) if checks is not None else (lambda: False)
    )
    if mode == "success":
        backend.extract(prepared, work, should_abort=should_abort)
    elif mode == "failure":
        with pytest.raises(TrainingError) as raised:
            backend.extract(prepared, work, should_abort=should_abort)
        assert raised.value.stderr == "offline generation failed"
    else:
        with pytest.raises(TuningPreempted):
            backend.extract(prepared, work, should_abort=should_abort)

    assert runner.started[-1].terminated == 1


def test_all_zero_mask_raises_named_error_with_row(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    runner.effects.extend(
        [
            lambda argv, _abort: (
                _FakeRunner._create_expected_output(argv)
                or ProcessResult(argv, 0, "", "")
            ),
            lambda argv, _abort: ProcessResult(
                argv,
                1,
                "",
                "prepared dataset contains all-zero loss mask at row trace-17",
            ),
        ]
    )
    backend, work = _backend(tmp_path, pipeline, runner)

    with pytest.raises(FinalAssistantMaskError) as raised:
        backend.prepare(work, should_abort=lambda: False)

    assert raised.value.row_id == "trace-17"
    assert "trace-17" in str(raised.value)


def test_learning_rate_above_safe_value_is_rejected(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    values = {
        field: getattr(pipeline, field)
        for field in pipeline.__dataclass_fields__
        if field != "learning_rate"
    }
    with pytest.raises(ValueError, match="safe 1e-5"):
        SpeculatorsPipelineConfig(**values, learning_rate=1e-4)


def test_model_revisions_are_resolved_and_used_by_prepare_and_train(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    revised = replace(
        pipeline,
        verifier_model="org/verifier",
        verifier_revision="verifier-sha",
        warm_start_model="org/draft",
        warm_start_revision="draft-sha",
    )
    runner = _FakeRunner()

    def resolved(path: str) -> RunEffect:
        return lambda argv, _abort: ProcessResult(argv, 0, f"{path}\n", "")

    runner.effects.extend(
        [
            resolved("/snapshots/verifier"),
            lambda argv, _abort: (
                _FakeRunner._create_expected_output(argv)
                or ProcessResult(argv, 0, "", "")
            ),
            lambda argv, _abort: ProcessResult(argv, 0, "", ""),
        ]
    )
    backend, work = _backend(tmp_path, revised, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)
    hidden = backend.extract(prepared, work, should_abort=lambda: False)
    runner.effects.extend(
        [
            resolved("/snapshots/draft"),
            lambda argv, _abort: (
                _FakeRunner._create_expected_output(argv)
                or ProcessResult(argv, 0, "", "")
            ),
        ]
    )
    backend.train(hidden, work, should_abort=lambda: False)

    assert runner.run_calls[0][-2:] == ("org/verifier", "verifier-sha")
    assert "--model" in runner.run_calls[1]
    assert runner.run_calls[1][runner.run_calls[1].index("--model") + 1] == (
        "/snapshots/verifier"
    )
    assert runner.run_calls[4][-2:] == ("org/draft", "draft-sha")
    train = runner.run_calls[5]
    assert train[train.index("--verifier-name-or-path") + 1] == "/snapshots/verifier"
    assert train[train.index("--from-pretrained") + 1] == "/snapshots/draft"


def test_local_warm_start_is_pinned_with_resolved_verifier(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    verifier = tmp_path / "verifier-snapshot"
    verifier.mkdir()
    warm_start = tmp_path / "draft-snapshot"
    warm_start.mkdir()
    (warm_start / "config.json").write_text(
        json.dumps(
            {
                "speculators_config": {
                    "verifier": {"name_or_path": "unresolved/verifier"}
                }
            }
        ),
        encoding="utf-8",
    )
    (warm_start / "model.safetensors").write_bytes(b"weights")
    configured = replace(
        pipeline,
        verifier_model=str(verifier),
        warm_start_model=str(warm_start),
    )
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, configured, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)
    hidden = backend.extract(prepared, work, should_abort=lambda: False)
    backend.train(hidden, work, should_abort=lambda: False)

    train = runner.run_calls[3]
    pinned = work / "warm-start-pinned"
    assert train[train.index("--from-pretrained") + 1] == str(pinned)
    value = json.loads((pinned / "config.json").read_text(encoding="utf-8"))
    assert value["speculators_config"]["verifier"]["name_or_path"] == str(verifier)
    assert (pinned / "model.safetensors").is_symlink()


def _rendered(work: Path) -> list[dict[str, object]]:
    text = (work / "speculators-conversations.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_prepare_renders_top_level_conversations_instead_of_messages(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(
        tmp_path,
        pipeline,
        runner,
        records=[
            {
                "id": "trace-1",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
        ],
    )

    backend.prepare(work, should_abort=lambda: False)

    rendered = _rendered(work)
    assert [record["id"] for record in rendered] == ["trace-1"]
    assert "messages" not in rendered[0]
    assert rendered[0]["conversations"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_prepare_preserves_tools_tool_calls_and_reasoning(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look something up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query": "x"}'},
        }
    ]
    runner = _FakeRunner()
    backend, work = _backend(
        tmp_path,
        pipeline,
        runner,
        records=[
            {
                "id": "trace-2",
                "tools": tools,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "look it up"}]},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "consider the tool",
                        "tool_calls": tool_calls,
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "found"},
                    {"role": "assistant", "content": "done"},
                ],
            }
        ],
    )

    backend.prepare(work, should_abort=lambda: False)

    record = _rendered(work)[0]
    assert record["tools"] == tools
    turns = record["conversations"]
    assert turns[0]["content"] == "look it up"
    assert turns[1]["tool_calls"] == tool_calls
    assert turns[1]["thinking"] == "consider the tool"
    assert turns[1]["reasoning_content"] == "consider the tool"
    assert turns[2] == {"role": "tool", "content": "found", "tool_call_id": "call-1"}


def test_prepare_without_convertible_records_raises_named_error(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(
        tmp_path,
        pipeline,
        runner,
        records=[
            {"id": "trace-3", "messages": [{"role": "user", "content": "hi"}]},
            {"id": "trace-4", "messages": []},
        ],
    )

    with pytest.raises(EmptySpeculatorsDatasetError) as raised:
        backend.prepare(work, should_abort=lambda: False)

    assert raised.value.source.name == "traces.jsonl"
    assert "Speculators conversation" in str(raised.value)
    assert not (work / "speculators-conversations.jsonl").exists()
    assert not runner.run_calls


class _FakeTensor:
    """Stand in for the torch tensors ``load_from_disk`` returns."""

    def __init__(self, values: Sequence[object]) -> None:
        self._values = list(values)

    def tolist(self) -> list[object]:
        return list(self._values)


def test_prepared_columns_accept_torch_formatted_sequences() -> None:
    assert _column(_FakeTensor([1, 2, 3])) == [1, 2, 3]
    assert _column([1, 2, 3]) == [1, 2, 3]
    assert _column((1, 2, 3)) == [1, 2, 3]


def test_prepared_columns_reject_missing_and_scalar_values() -> None:
    assert _column(None) is None
    assert _column("101") is None
    assert _column(7) is None
