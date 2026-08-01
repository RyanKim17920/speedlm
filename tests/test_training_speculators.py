from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from speedlm.training.backends.eagle3 import (
    DRAFT_COPY_CHUNK_BYTES,
    MAX_SCRATCH_BYTES,
    Eagle3Backend,
    EmptySpeculatorsDatasetError,
    FinalAssistantMaskError,
    ScratchQuotaExceeded,
    SpeculatorsDraftMaterializer,
    SpeculatorsPipelineConfig,
    TrainingError,
    persist_training_output,
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


def _write_hidden_state_shard(path: Path, layers: int, *, tokens: int = 4) -> None:
    """Write a minimal safetensors shard shaped [tokens, layers, hidden]."""
    hidden = 8
    payload = b"\0" * (tokens * layers * hidden * 2)
    header = json.dumps(
        {
            "hidden_states": {
                "dtype": "BF16",
                "shape": [tokens, layers, hidden],
                "data_offsets": [0, len(payload)],
            },
            "token_ids": {
                "dtype": "I64",
                "shape": [tokens],
                "data_offsets": [len(payload), len(payload) + tokens * 8],
            },
        }
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(payload)
        handle.write(b"\0" * (tokens * 8))


class _FakeRunner:
    def __init__(self) -> None:
        self.run_calls: list[tuple[str, ...]] = []
        self.start_calls: list[tuple[str, ...]] = []
        self.effects: list[RunEffect] = []
        self.started: list[_FakeProcess] = []
        # target_layer_ids=(3, 9, 15) plus the verifier's appended final layer.
        self.hidden_state_layers = 4

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
        self._create_expected_output(command, self.hidden_state_layers)
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
    def _create_expected_output(command: tuple[str, ...], layers: int = 4) -> None:
        script = Path(command[1]).name if len(command) > 1 else ""
        if script in {"prepare_data.py", "data_generation_offline.py"}:
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            if script == "data_generation_offline.py":
                _write_hidden_state_shard(output / "hs_0.safetensors", layers)
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
            "--max-model-len",
            "4096",
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


def test_hidden_state_server_caps_context_at_configured_sequence_length(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    pipeline = replace(pipeline, sequence_length=2048)
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)

    prepared = backend.prepare(work, should_abort=lambda: False)
    backend.extract(prepared, work, should_abort=lambda: False)

    argv = runner.start_calls[0]
    assert argv[argv.index("--max-model-len") + 1] == "2048"


def test_hidden_state_server_requests_the_configured_aux_layers_verbatim(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    # launch_vllm.py appends the verifier's final layer on top of these, which
    # training slices back off as the regression target. Passing
    # --no-include-last-layer would starve the draft's input_norm of one layer.
    pipeline = replace(pipeline, target_layer_ids=(1, 4, 8))
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)

    prepared = backend.prepare(work, should_abort=lambda: False)
    backend.extract(prepared, work, should_abort=lambda: False)

    argv = runner.start_calls[0]
    separator = argv.index("--")
    start = argv.index("--target-layer-ids")
    assert argv[start + 1 : separator] == ("1", "4", "8")
    assert "--no-include-last-layer" not in argv


def test_wrong_hidden_state_layer_count_fails_at_extraction(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    # A short shard used to survive extraction and only blow up ~100s later
    # inside Dynamo as an opaque broadcast error over the flattened hidden size.
    runner = _FakeRunner()
    runner.hidden_state_layers = 3
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)

    with pytest.raises(TrainingError) as raised:
        backend.extract(prepared, work, should_abort=lambda: False)

    assert "hidden-state layer count breaks the EAGLE-3 contract" in str(raised.value)
    assert "expected [sequence_length, 4, hidden_size]" in str(raised.value)


def test_correct_hidden_state_layer_count_passes_extraction(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)

    destination = backend.extract(prepared, work, should_abort=lambda: False)

    assert sorted(path.name for path in destination.glob("hs_*.safetensors")) == [
        "hs_0.safetensors"
    ]


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

    assert runner.run_calls[0][3:5] == ("org/verifier", "verifier-sha")
    assert "--model" in runner.run_calls[1]
    assert runner.run_calls[1][runner.run_calls[1].index("--model") + 1] == (
        "/snapshots/verifier"
    )
    assert runner.run_calls[4][3:5] == ("org/draft", "draft-sha")
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


def test_missing_input_ids_raises_named_error_with_row(
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
                "row 0 has no input_ids sequence\n",
            ),
        ]
    )
    backend, work = _backend(tmp_path, pipeline, runner)

    with pytest.raises(FinalAssistantMaskError) as raised:
        backend.prepare(work, should_abort=lambda: False)

    assert raised.value.row_id == "0"


def test_training_output_is_captured_to_the_run_directory(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    """Training is ~22% of a cycle and left no evidence behind at all."""
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)
    hidden = backend.extract(prepared, work, should_abort=lambda: False)

    def train_effect(argv: tuple[str, ...], _abort: object) -> ProcessResult:
        _FakeRunner._create_expected_output(argv)
        return ProcessResult(argv, 0, "step 1 loss 0.5\n", "startup: loading verifier\n")

    runner.effects.append(train_effect)
    backend.train(hidden, work, should_abort=lambda: False)

    logs = work / "training-logs"
    assert (logs / "stdout.log").read_text(encoding="utf-8") == "step 1 loss 0.5\n"
    assert (logs / "stderr.log").read_text(encoding="utf-8") == (
        "startup: loading verifier\n"
    )


def test_training_output_is_captured_even_when_training_fails(
    tmp_path: Path,
    pipeline: SpeculatorsPipelineConfig,
) -> None:
    runner = _FakeRunner()
    backend, work = _backend(tmp_path, pipeline, runner)
    prepared = backend.prepare(work, should_abort=lambda: False)
    hidden = backend.extract(prepared, work, should_abort=lambda: False)
    runner.effects.append(
        lambda argv, _abort: ProcessResult(argv, 9, "partial progress\n", "OOM\n")
    )

    with pytest.raises(TrainingError):
        backend.train(hidden, work, should_abort=lambda: False)

    logs = work / "training-logs"
    assert (logs / "stdout.log").read_text(encoding="utf-8") == "partial progress\n"
    assert (logs / "stderr.log").read_text(encoding="utf-8") == "OOM\n"


def test_a_huge_training_log_is_bounded_and_keeps_both_ends() -> None:
    result = ProcessResult(("train.py",), 0, "A" * 5_000 + "Z" * 5_000, "")
    directory = Path(tempfile.mkdtemp())

    persist_training_output(directory, result, max_bytes=1_000)

    written = (directory / "training-logs" / "stdout.log").read_text(encoding="utf-8")
    assert len(written.encode("utf-8")) < 1_200
    assert written.startswith("A" * 500)
    assert written.endswith("Z" * 500)
    assert "9000 bytes elided" in written


# --- draft materialization copy -------------------------------------------


def _materializer(quota: int = MAX_SCRATCH_BYTES) -> SpeculatorsDraftMaterializer:
    return SpeculatorsDraftMaterializer(scratch_quota_bytes=quota)


def test_materialize_copies_files_larger_than_one_chunk_byte_for_byte(
    tmp_path: Path,
) -> None:
    """The copy is chunked for interruptibility, not for correctness.

    A ~2 GB draft used to move 1 MiB at a time with a scratch-quota check on
    both sides of every write, and that check walks the whole scratch tree --
    thousands of full-tree walks on the cycle's critical path with serving
    stopped.  Widening the chunk must not change what lands on disk, including
    for a file that is not a whole number of chunks.
    """
    source = tmp_path / "checkpoint_best"
    (source / "nested").mkdir(parents=True)
    payload = bytes(range(256)) * (DRAFT_COPY_CHUNK_BYTES // 256) + b"tail"
    (source / "model.safetensors").write_bytes(payload)
    (source / "nested" / "config.json").write_bytes(b"{}")
    # Still skipped: transient training state is not part of a draft.
    (source / "optimizer_state_dict.pt").write_bytes(b"transient")

    destination = _materializer().materialize(
        source,
        tmp_path / "draft",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert len(payload) > DRAFT_COPY_CHUNK_BYTES
    assert (destination / "model.safetensors").read_bytes() == payload
    assert (destination / "nested" / "config.json").read_bytes() == b"{}"
    assert not (destination / "optimizer_state_dict.pt").exists()


def test_materialize_still_honours_abort_and_removes_the_partial_draft(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoint_best"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"x" * (DRAFT_COPY_CHUNK_BYTES + 1))
    destination = tmp_path / "draft"

    with pytest.raises(TuningPreempted):
        _materializer().materialize(
            source,
            destination,
            timeout_seconds=60.0,
            should_abort=lambda: True,
        )

    assert not destination.exists()


def test_materialize_still_enforces_the_scratch_quota_mid_copy(
    tmp_path: Path,
) -> None:
    """A quota breach caused by the copy itself must still be caught inside it.

    The checkpoint moved from "twice per 1 MiB" to "once per chunk, plus once
    after the last write", so this pins that the *inside* of a multi-chunk copy
    is still guarded rather than deferred to the caller.
    """
    source = tmp_path / "checkpoint_best"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"x" * (3 * DRAFT_COPY_CHUNK_BYTES))
    destination = tmp_path / "draft"

    with pytest.raises(ScratchQuotaExceeded) as raised:
        _materializer(quota=DRAFT_COPY_CHUNK_BYTES).materialize(
            source,
            destination,
            timeout_seconds=60.0,
            should_abort=lambda: False,
        )

    assert raised.value.used_bytes > DRAFT_COPY_CHUNK_BYTES
    assert not destination.exists()
