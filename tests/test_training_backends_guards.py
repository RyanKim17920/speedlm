"""Guards added to the EAGLE-3 backend: warm-start alignment, the acceptance
gate, and the truncated-row filter.

Every test here is written to be *capable of failing*.  The fixtures are real
artifacts, not stand-ins: the trainer stdout is a verbatim excerpt of a
production run's ``training-logs/stdout.log`` including Rich's 80-column
wrapping, and the safetensors files are genuine containers whose headers are
actually parsed for a shape.  The recurring defect in this repo is a green test
that cannot observe the thing it claims to check -- an existing example writes
an empty ``model.safetensors`` and stubs the reader to return a hardcoded key
set, so it is structurally incapable of seeing a tensor shape.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from speedlm.training.backends.eagle3 import (
    ACCEPTANCE_METRIC,
    TRUNCATED_FINISH_REASONS,
    AccuracyRegressionError,
    ClientSupervisedCorpusError,
    Eagle3Error,
    EmptySpeculatorsDatasetError,
    RenderedRowCounts,
    SpeculatorsPipelineConfig,
    SpeculatorsTrainingProcess,
    TruncatedRowPolicy,
    TruncationFilteredCorpusError,
    UnattributedCorpusShortfallError,
    WarmStartLayerMismatchError,
    _check_warm_start_alignment,
    _checkpoint_aux_count,
    _render_speculators_dataset,
    _Resolver,
    _speculators_default_aux_ids,
    _speculators_record,
    _State,
    parse_val_accuracy_epochs,
    summarize_val_accuracy,
)
from speedlm.training.backends.speculators_runner import ProcessResult, RunningProcess
from speedlm.tuner.eagle3 import TraceSnapshot

# ---------------------------------------------------------------------------
# Fixture: a verbatim excerpt of a real trainer stdout log.
#
# Source: /data/ryan.kim/speedlm-runs/qwen-cross-20260731T235000Z/results/
#   live-idle-tuning/speedlm_home/runs/6b150b5c424f483c83f0bc49c79bc4c5/
#   training-logs/stdout.log
#
# One train-step record (which the parser must ignore) followed by all three
# validation-epoch records.  The 20-space continuation indent and the
# right-aligned ``trainer.py:NNN`` source tags are Rich's, and the record
# boundaries fall exactly where they fell in the run -- which is the hazard: a
# parser that scans line by line reads only whichever TTT step happened to land
# on the line it looked at.  (Rich's trailing pad to column 80 is the one thing
# not reproduced, because a formatter would strip it; the parser was separately
# run against all twelve unmodified logs under
# /data/ryan.kim/speedlm-runs/*/**/training-logs/stdout.log and recovered three
# epochs from each.)
# ---------------------------------------------------------------------------
REAL_TRAINER_STDOUT = """\
[23:47:39] INFO     train/loss_0=1.141, train/full_acc_0=0.636,   trainer.py:233
                    train/cond_acc_0=0.636, train/loss_1=4.125,
                    train/full_acc_1=0.367,
                    train/cond_acc_1=0.576, train/loss_2=5.625,
                    train/full_acc_2=0.171,
                    train/cond_acc_2=0.466, train/loss=10.891,
           INFO     Validation epoch 1/3 started                  trainer.py:359
Epoch 0 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/7  [ 0:00:11 < 0:00:00 , 39 it/s ]
[23:47:55] INFO     val/loss_0_epoch=1.087,                       trainer.py:290
                    val/full_acc_0_epoch=0.648,
                    val/cond_acc_0_epoch=0.648,
                    val/loss_1_epoch=3.469,
                    val/full_acc_1_epoch=0.374,
                    val/cond_acc_1_epoch=0.577,
                    val/loss_2_epoch=4.746,
                    val/full_acc_2_epoch=0.171,
                    val/cond_acc_2_epoch=0.457,
                    val/loss_epoch=9.301, epoch=0
           INFO     Validation epoch 1/3 completed                trainer.py:361
[23:48:05] INFO     Updated checkpoint_best -> 0                  trainer.py:332
           INFO     Validation epoch 2/3 started                  trainer.py:359
Epoch 1 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/7  [ 0:00:00 < 0:00:00 , 20 it/s ]
[23:48:12] INFO     val/loss_0_epoch=1.093,                       trainer.py:290
                    val/full_acc_0_epoch=0.647,
                    val/cond_acc_0_epoch=0.647,
                    val/loss_1_epoch=3.379,
                    val/full_acc_1_epoch=0.372,
                    val/cond_acc_1_epoch=0.574,
                    val/loss_2_epoch=4.680,
                    val/full_acc_2_epoch=0.169,
                    val/cond_acc_2_epoch=0.455,
                    val/loss_epoch=9.152, epoch=1
           INFO     Validation epoch 2/3 completed                trainer.py:361
[23:48:22] INFO     Updated checkpoint_best -> 1                  trainer.py:332
           INFO     Validation epoch 3/3 started                  trainer.py:359
Epoch 2 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/7  [ 0:00:00 < 0:00:00 , 20 it/s ]
[23:48:29] INFO     val/loss_0_epoch=1.091,                       trainer.py:290
                    val/full_acc_0_epoch=0.648,
                    val/cond_acc_0_epoch=0.648,
                    val/loss_1_epoch=3.371,
                    val/full_acc_1_epoch=0.372,
                    val/cond_acc_1_epoch=0.574,
                    val/loss_2_epoch=4.664,
                    val/full_acc_2_epoch=0.169,
                    val/cond_acc_2_epoch=0.454,
                    val/loss_epoch=9.125, epoch=2
           INFO     Validation epoch 3/3 completed                trainer.py:361
"""


def _write_safetensors(path: Path, shapes: Mapping[str, Sequence[int]]) -> None:
    """Write a genuine safetensors container with the given tensor shapes.

    Real bytes, real header, real offsets -- so a reader that does not actually
    parse the header cannot pass a test that uses this.
    """
    header: dict[str, object] = {}
    payload = bytearray()
    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        start = len(payload)
        payload.extend(struct.pack(f"<{count}f", *([0.0] * count)))
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        handle.write(payload)


def _checkpoint(
    directory: Path,
    *,
    hidden_size: int = 4,
    aux_ids: Sequence[int] | None = None,
    declare_aux: bool = True,
    aux_count: int = 3,
    norm_before_fc: bool = False,
) -> Path:
    """Materialize a warm-start checkpoint directory shaped like a real one."""
    directory.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "speculators_model_type": "eagle3",
        "norm_before_fc": norm_before_fc,
        "embed_requires_grad": False,
        "norm_before_residual": True,
        "transformer_layer_config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": 1,
        },
        "speculators_config": {"verifier": {"name_or_path": "org/verifier"}},
    }
    if declare_aux:
        config["eagle_aux_hidden_state_layer_ids"] = (
            list(aux_ids) if aux_ids is not None else None
        )
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_safetensors(
        directory / "model.safetensors",
        {"fc.weight": [hidden_size, aux_count * hidden_size]},
    )
    return directory


def _verifier(directory: Path, num_hidden_layers: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"num_hidden_layers": num_hidden_layers}), encoding="utf-8"
    )
    return directory


# ===========================================================================
# FIX 1 -- warm-start aux layer alignment
# ===========================================================================


def test_aux_count_is_read_from_the_real_safetensors_header(tmp_path: Path) -> None:
    """The aux count must come from the weights, not from a config field.

    Shapes here are distinct per case, so a reader that returned a constant
    fails.
    """
    for aux_count in (2, 3, 5):
        directory = _checkpoint(
            tmp_path / f"ckpt-{aux_count}", hidden_size=8, aux_count=aux_count
        )
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert _checkpoint_aux_count(directory, config) == aux_count


def test_aux_count_is_none_when_the_container_is_empty(tmp_path: Path) -> None:
    """An empty ``model.safetensors`` must read as unavailable, never as a match.

    This is the exact fixture shape that made an existing test in this suite
    unable to observe anything; here it is asserted to be unobservable.
    """
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "model.safetensors").write_bytes(b"")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))

    assert _checkpoint_aux_count(directory, config) is None


def test_declared_ids_that_differ_from_extraction_fail_by_name(tmp_path: Path) -> None:
    """The whole point of the guard: divergence must not be silent."""
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=[2, 18, 33])

    with pytest.raises(WarmStartLayerMismatchError) as caught:
        _check_warm_start_alignment(str(directory), "org/verifier", (3, 9, 15))

    message = str(caught.value)
    assert "[2, 18, 33]" in message, message
    assert "[3, 9, 15]" in message, message
    assert "silently use the checkpoint's" in message, message


def test_declared_ids_that_match_pass_and_record_the_evidence(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=[3, 9, 15], norm_before_fc=True)

    record = _check_warm_start_alignment(str(directory), "org/verifier", (3, 9, 15))

    assert record["verdict"] == "checkpoint_declared"
    assert record["checkpoint_layer_ids"] == [3, 9, 15]
    assert record["extracted_layer_ids"] == [3, 9, 15]
    assert record["checkpoint_aux_count"] == 3
    # Dropped by scripts/train.py on this path but not requested by us; recorded
    # so a cross-model manifest can say which architecture actually trained.
    assert record["norm_before_fc"] is True
    assert record["embed_requires_grad"] is False
    assert record["num_layers"] == 1


def test_undeclared_ids_are_derived_from_the_verifier_and_checked(
    tmp_path: Path,
) -> None:
    """A checkpoint declaring nothing is the *default* stock warm start.

    ``resolve_target_layer_ids`` then substitutes ``[2, n//2, n-3]`` from the
    verifier with only a ``warnings.warn``, and ``--target-layer-ids`` is
    dropped -- so the check has to reproduce that substitution to see it.
    """
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=None)
    verifier = _verifier(tmp_path / "verifier", 36)

    with pytest.raises(WarmStartLayerMismatchError) as caught:
        _check_warm_start_alignment(str(directory), str(verifier), (3, 9, 15))

    message = str(caught.value)
    # [2, 36 // 2, 36 - 3]
    assert "[2, 18, 33]" in message, message
    assert "[3, 9, 15]" in message, message
    assert "silently use the checkpoint's" in message, message


def test_undeclared_ids_matching_the_substitution_pass(tmp_path: Path) -> None:
    """The arithmetic coincidence the production path currently survives on."""
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=None)
    verifier = _verifier(tmp_path / "verifier", 36)

    record = _check_warm_start_alignment(str(directory), str(verifier), (2, 18, 33))

    assert record["verdict"] == "derived_from_verifier"
    assert record["derived_layer_ids"] == [2, 18, 33]
    assert record["verifier_num_hidden_layers"] == 36
    assert record["checkpoint_layer_ids"] is None


def test_substitution_formula_matches_speculators(tmp_path: Path) -> None:
    """Pin the reproduced formula so a vendored change surfaces here."""
    assert _speculators_default_aux_ids(36) == (2, 18, 33)
    assert _speculators_default_aux_ids(24) == (2, 12, 21)
    assert _speculators_default_aux_ids(11) == (2, 5, 8)


def test_aux_arity_mismatch_fails_even_when_ids_agree(tmp_path: Path) -> None:
    """The fc's input width is baked into the weights and cannot be renegotiated."""
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=[1, 2, 3, 4], aux_count=3)

    with pytest.raises(WarmStartLayerMismatchError, match="auxiliary hidden states"):
        _check_warm_start_alignment(str(directory), "org/verifier", (1, 2, 3, 4))


def test_a_bare_repo_id_is_reported_unverified_not_silently_passed(
    tmp_path: Path,
) -> None:
    record = _check_warm_start_alignment("Org/stock-drafter", "org/verifier", (2, 3, 4))

    assert record["verdict"] == "unverified_no_local_checkpoint"


def test_undeclared_ids_without_a_readable_verifier_are_reported_unverified(
    tmp_path: Path,
) -> None:
    directory = _checkpoint(tmp_path / "ckpt", aux_ids=None)

    record = _check_warm_start_alignment(str(directory), "org/verifier", (2, 18, 33))

    assert record["verdict"] == "unverified_no_verifier_config"


def test_malformed_declared_ids_are_rejected(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "ckpt")
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["eagle_aux_hidden_state_layer_ids"] = ["two", "eighteen"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WarmStartLayerMismatchError, match="malformed"):
        _check_warm_start_alignment(str(directory), "org/verifier", (2, 18, 33))


class _RecordingRunner:
    """Records every subprocess launch and never actually runs one."""

    def __init__(self, stdout: str = "") -> None:
        self.run_calls: list[tuple[str, ...]] = []
        self.stdout = stdout

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> ProcessResult:
        del cwd, env, timeout_seconds, should_abort
        command = tuple(argv)
        self.run_calls.append(command)
        if Path(command[1]).name == "train.py":
            checkpoint = Path(command[command.index("--save-path") + 1])
            (checkpoint / "checkpoint_best").mkdir(parents=True, exist_ok=True)
        return ProcessResult(command, 0, self.stdout, "")

    # The stages under test never start a long-running process; these exist so
    # the fake really satisfies ProcessRunner rather than being cast to it.
    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_seconds: float,
    ) -> RunningProcess:
        raise AssertionError(f"unexpected long-running process: {list(argv)}")

    def check_running(
        self, process: RunningProcess, *, should_abort: Callable[[], bool]
    ) -> int | None:
        raise AssertionError("unexpected check_running")

    def terminate(
        self, process: RunningProcess, *, grace_seconds: float
    ) -> ProcessResult:
        raise AssertionError("unexpected terminate")

    @property
    def scripts(self) -> list[str]:
        return [Path(call[1]).name for call in self.run_calls if len(call) > 1]


def _training_process(
    tmp_path: Path,
    runner: _RecordingRunner,
    warm_start: Path,
    verifier: Path,
    *,
    layers: tuple[int, ...] = (3, 9, 15),
    require_accuracy_improvement: bool = True,
) -> tuple[SpeculatorsTrainingProcess, _State]:
    config = SpeculatorsPipelineConfig(
        prepared_validator_script=tmp_path / "check.py",
        speculators_repo=tmp_path / "speculators",
        training_python=tmp_path / "python",
        verifier_model=str(verifier),
        warm_start_model=str(warm_start),
        target_layer_ids=layers,
        require_accuracy_improvement=require_accuracy_improvement,
    )
    state = _State()
    state.prepared = tmp_path / "prepared"
    state.verifier = str(verifier)
    state.warm_start = str(warm_start)
    state.warm_start_source = str(warm_start)
    resolver = _Resolver(config, runner, state)
    return SpeculatorsTrainingProcess(config, runner, resolver, state), state


def test_the_guard_runs_before_the_gpu_hours(tmp_path: Path) -> None:
    """A misaligned warm start must stop the cycle, not merely annotate it."""
    warm_start = _checkpoint(tmp_path / "warm", aux_ids=[2, 18, 33])
    verifier = _verifier(tmp_path / "verifier", 36)
    runner = _RecordingRunner()
    process, _ = _training_process(tmp_path, runner, warm_start, verifier)

    with pytest.raises(WarmStartLayerMismatchError):
        process.train(
            tmp_path / "hidden",
            tmp_path / "out",
            from_pretrained=str(warm_start),
            training_params={"target_layer_ids": (3, 9, 15)},
            timeout_seconds=60.0,
            should_abort=lambda: False,
        )

    assert "train.py" not in runner.scripts


# ===========================================================================
# FIX 2 -- gate on the acceptance metric, not the loss
# ===========================================================================


def test_parses_every_epoch_out_of_a_real_wrapped_log() -> None:
    epochs = parse_val_accuracy_epochs(REAL_TRAINER_STDOUT)

    assert [entry["epoch"] for entry in epochs] == [0.0, 1.0, 2.0]
    # Every TTT step of every epoch, i.e. the wrap was actually reassembled.
    assert [entry["full_acc_0"] for entry in epochs] == [0.648, 0.647, 0.648]
    assert [entry["full_acc_1"] for entry in epochs] == [0.374, 0.372, 0.372]
    assert [entry["full_acc_2"] for entry in epochs] == [0.171, 0.169, 0.169]
    assert [entry["cond_acc_1"] for entry in epochs] == [0.577, 0.574, 0.574]
    assert [entry["loss"] for entry in epochs] == [9.301, 9.152, 9.125]
    # train/ records must not be mistaken for validation records.
    assert all("epoch" in entry for entry in epochs)
    assert 0.636 not in [entry["full_acc_0"] for entry in epochs]


def test_the_loss_falls_while_the_acceptance_proxy_does_not() -> None:
    """The premise of the whole fix, asserted on the real numbers."""
    epochs = parse_val_accuracy_epochs(REAL_TRAINER_STDOUT)

    assert epochs[-1]["loss"] < epochs[0]["loss"]
    assert epochs[-1]["full_acc_0"] <= epochs[0]["full_acc_0"]


def test_cond_acc_is_the_conditional_series_and_full_acc_is_not() -> None:
    """Which series to key on is an arithmetic question, so answer it here.

    ``cond_acc_k == full_acc_k / full_acc_{k-1}``: the two share a numerator
    and differ only in denominator.  Keying the gate on the conditional series
    would measure survival given survival, not acceptance.
    """
    for entry in parse_val_accuracy_epochs(REAL_TRAINER_STDOUT):
        assert entry["cond_acc_0"] == entry["full_acc_0"]
        assert entry["cond_acc_1"] == pytest.approx(
            entry["full_acc_1"] / entry["full_acc_0"], abs=0.002
        )
        assert entry["cond_acc_2"] == pytest.approx(
            entry["full_acc_2"] / entry["full_acc_1"], abs=0.002
        )
    assert ACCEPTANCE_METRIC == "full_acc_0_epoch"


def test_a_record_spanning_the_log_elision_is_skipped() -> None:
    """Oversized logs get their middle removed; do not parse across the hole."""
    corrupt = REAL_TRAINER_STDOUT.replace(
        "                    val/cond_acc_0_epoch=0.647,",
        "\n...[918273 bytes elided]...\n",
        1,
    )

    epochs = parse_val_accuracy_epochs(corrupt)

    assert [entry["epoch"] for entry in epochs] == [0.0, 2.0]


def test_a_flat_final_epoch_fails_the_cycle(tmp_path: Path) -> None:
    warm_start = _checkpoint(tmp_path / "warm", aux_ids=[3, 9, 15])
    verifier = _verifier(tmp_path / "verifier", 36)
    runner = _RecordingRunner(stdout=REAL_TRAINER_STDOUT)
    process, state = _training_process(tmp_path, runner, warm_start, verifier)

    with pytest.raises(AccuracyRegressionError) as caught:
        process.train(
            tmp_path / "hidden",
            tmp_path / "out",
            from_pretrained=str(warm_start),
            training_params={"target_layer_ids": (3, 9, 15)},
            timeout_seconds=60.0,
            should_abort=lambda: False,
        )

    assert "full_acc_0_epoch" in str(caught.value)
    assert "train.py" in runner.scripts
    assert state.val_accuracy is not None
    assert state.val_accuracy["verdict"] == "not_improved"
    assert state.val_accuracy["series"] == [0.648, 0.647, 0.648]


def test_an_improving_run_is_allowed_through(tmp_path: Path) -> None:
    # Lift only the final epoch's step-0 accuracy; everything else is untouched.
    tail = REAL_TRAINER_STDOUT.rindex("val/full_acc_0_epoch=0.648,")
    improving = (
        REAL_TRAINER_STDOUT[:tail]
        + "val/full_acc_0_epoch=0.712,"
        + REAL_TRAINER_STDOUT[tail + len("val/full_acc_0_epoch=0.648,") :]
    )
    assert improving != REAL_TRAINER_STDOUT
    warm_start = _checkpoint(tmp_path / "warm", aux_ids=[3, 9, 15])
    verifier = _verifier(tmp_path / "verifier", 36)
    runner = _RecordingRunner(stdout=improving)
    process, state = _training_process(tmp_path, runner, warm_start, verifier)

    process.train(
        tmp_path / "hidden",
        tmp_path / "out",
        from_pretrained=str(warm_start),
        training_params={"target_layer_ids": (3, 9, 15)},
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.val_accuracy is not None
    assert state.val_accuracy["verdict"] == "improved"
    assert state.val_accuracy["series"] == [0.648, 0.647, 0.712]


def test_the_gate_is_configurable_off(tmp_path: Path) -> None:
    warm_start = _checkpoint(tmp_path / "warm", aux_ids=[3, 9, 15])
    verifier = _verifier(tmp_path / "verifier", 36)
    runner = _RecordingRunner(stdout=REAL_TRAINER_STDOUT)
    process, state = _training_process(
        tmp_path, runner, warm_start, verifier, require_accuracy_improvement=False
    )

    process.train(
        tmp_path / "hidden",
        tmp_path / "out",
        from_pretrained=str(warm_start),
        training_params={"target_layer_ids": (3, 9, 15)},
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    # Off means "do not fail", never "do not look".
    assert state.val_accuracy is not None
    assert state.val_accuracy["verdict"] == "not_improved"


def test_one_epoch_is_not_evaluated_rather_than_passed(tmp_path: Path) -> None:
    """Nothing to compare against; inventing a verdict is the untestable kind."""
    single = REAL_TRAINER_STDOUT[: REAL_TRAINER_STDOUT.index("Validation epoch 2/3")]
    summary = summarize_val_accuracy(parse_val_accuracy_epochs(single))

    assert summary["series"] == [0.648]

    warm_start = _checkpoint(tmp_path / "warm", aux_ids=[3, 9, 15])
    verifier = _verifier(tmp_path / "verifier", 36)
    runner = _RecordingRunner(stdout=single)
    process, state = _training_process(tmp_path, runner, warm_start, verifier)

    process.train(
        tmp_path / "hidden",
        tmp_path / "out",
        from_pretrained=str(warm_start),
        training_params={"target_layer_ids": (3, 9, 15)},
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.val_accuracy is not None
    assert state.val_accuracy["verdict"] == "not_evaluated"


# ===========================================================================
# FIX 3 -- carry finish_reason and filter truncated rows
# ===========================================================================


def _snapshot(path: Path, records: Sequence[Mapping[str, object]]) -> TraceSnapshot:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return TraceSnapshot(path, "hash")


def _row(index: int, finish_reason: object = ...) -> dict[str, object]:
    record: dict[str, object] = {
        "id": f"row-{index}",
        "messages": [
            # Tagged as the gateway tags them: the request body is
            # ``client_supplied``, the assembled response is ``generated``.
            {"role": "user", "content": "q", "provenance_tag": "client_supplied"},
            {"role": "assistant", "content": "a", "provenance_tag": "generated"},
        ],
    }
    if finish_reason is not ...:
        record["finish_reason"] = finish_reason
    return record


def _render(
    tmp_path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    policy: TruncatedRowPolicy = TruncatedRowPolicy.KEEP,
    minimum_rows: int = 1,
    name: str = "out.jsonl",
    trust_untagged: bool = False,
) -> tuple[RenderedRowCounts, list[dict[str, object]]]:
    snapshot = _snapshot(tmp_path / f"snap-{name}", records)
    destination = tmp_path / name
    counts = _render_speculators_dataset(
        snapshot,
        destination,
        guard=lambda: False,
        started=time.monotonic(),
        timeout=60.0,
        policy=policy,
        minimum_rows=minimum_rows,
        trust_untagged_assistant_messages=trust_untagged,
    )
    written = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return counts, written


def test_finish_reason_reaches_the_rendered_row() -> None:
    """It is captured, persisted and leased; it was dropped only here."""
    converted = _speculators_record(_row(0, "length"))

    assert converted is not None
    assert converted["finish_reason"] == "length"


def test_a_row_without_a_finish_reason_does_not_invent_one() -> None:
    converted = _speculators_record(_row(0))

    assert converted is not None
    assert "finish_reason" not in converted


def test_truncated_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    records = [
        _row(0, "stop"),
        _row(1, "length"),
        _row(2, "tool_calls"),
        _row(3, "incomplete"),
        _row(4),
    ]

    counts, written = _render(tmp_path, records, policy=TruncatedRowPolicy.DROP)

    assert [row["id"] for row in written] == ["row-0", "row-2", "row-4"]
    assert counts.read == 5
    assert counts.written == 3
    assert counts.dropped_truncated == 2
    assert counts.truncated_seen == 2
    assert counts.dropped_untrainable == 0
    assert counts.to_dict()["truncated_row_policy"] == "drop"


def test_keeping_truncated_rows_still_counts_them(tmp_path: Path) -> None:
    """The policy changes the corpus; it must not change the evidence."""
    records = [_row(0, "stop"), _row(1, "length")]

    counts, written = _render(tmp_path, records, policy=TruncatedRowPolicy.KEEP)

    assert [row["id"] for row in written] == ["row-0", "row-1"]
    assert counts.truncated_seen == 1
    assert counts.dropped_truncated == 0


def test_unknown_finish_reasons_are_kept_not_guessed(tmp_path: Path) -> None:
    """``None``/``""`` mean unknown; archived corpora predate the field."""
    records = [_row(0, None), _row(1, ""), _row(2), _row(3, "completed")]

    counts, written = _render(tmp_path, records, policy=TruncatedRowPolicy.DROP)

    assert len(written) == 4
    assert counts.dropped_truncated == 0
    assert "length" in TRUNCATED_FINISH_REASONS
    assert "stop" not in TRUNCATED_FINISH_REASONS


def test_the_filter_emptying_the_corpus_is_a_loud_named_failure(
    tmp_path: Path,
) -> None:
    """A silently tiny corpus produces a checkpoint that looks like any other."""
    records = [_row(index, "length") for index in range(9)] + [_row(9, "stop")]

    with pytest.raises(TruncationFilteredCorpusError) as caught:
        _render(
            tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4
        )

    message = str(caught.value)
    assert "1 trainable rows" in message
    assert "below the floor of 4" in message
    assert "9 of 10 rows were truncated" in message
    assert "truncated_row_policy" in message


def test_the_floor_does_not_fire_on_a_corpus_that_is_merely_small(
    tmp_path: Path,
) -> None:
    """The floor bounds this filter's damage, not the corpus's own size."""
    counts, written = _render(
        tmp_path,
        [_row(0, "stop")],
        policy=TruncatedRowPolicy.DROP,
        minimum_rows=100,
    )

    assert counts.written == 1
    assert len(written) == 1


def test_an_entirely_truncated_corpus_names_truncation_not_conversion(
    tmp_path: Path,
) -> None:
    """``EmptySpeculatorsDatasetError`` would name the wrong cause."""
    with pytest.raises(TruncationFilteredCorpusError):
        _render(
            tmp_path,
            [_row(0, "length"), _row(1, "length")],
            policy=TruncatedRowPolicy.DROP,
        )


def test_a_corpus_with_no_trainable_turn_still_raises_the_empty_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(EmptySpeculatorsDatasetError):
        _render(
            tmp_path,
            [{"id": "x", "messages": [{"role": "user", "content": "q"}]}],
            policy=TruncatedRowPolicy.DROP,
        )


def test_the_renderer_publishes_its_counts_through_the_real_stage(
    tmp_path: Path,
) -> None:
    """Through ``render_rows``, not by assigning state in the test.

    Otherwise nothing proves the stage that owns the filter actually reports
    what it did, and the manifest's counts could be permanently absent.
    """
    from speedlm.training.backends.eagle3 import SpeculatorsTrainingRowRenderer

    config = SpeculatorsPipelineConfig(
        prepared_validator_script=tmp_path / "check.py",
        speculators_repo=tmp_path / "speculators",
        training_python=tmp_path / "python",
        verifier_model=str(_verifier(tmp_path / "verifier", 36)),
        warm_start_model=str(tmp_path / "warm"),
        truncated_row_policy=TruncatedRowPolicy.DROP,
        min_rendered_rows=1,
    )
    runner = _RecordingRunner()
    state = _State()
    state.verifier = config.verifier_model
    renderer = SpeculatorsTrainingRowRenderer(
        config, runner, _Resolver(config, runner, state), state
    )
    work = tmp_path / "work"
    work.mkdir()
    snapshot = _snapshot(
        work / "snap.jsonl",
        [_row(0, "stop"), _row(1, "length"), _row(2, "stop")],
    )

    renderer.render_rows(
        snapshot,
        work / "training-rows",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.rendered_rows is not None
    assert state.rendered_rows["written"] == 2
    assert state.rendered_rows["dropped_truncated"] == 1
    # The row count handed to the prepared-dataset validator must be the
    # post-filter count, or the validator asserts a size the corpus no longer has.
    assert state.row_count == 2


def test_the_policy_must_be_a_policy() -> None:
    with pytest.raises(ValueError, match="truncated_row_policy"):
        SpeculatorsPipelineConfig(
            prepared_validator_script=Path("check.py"),
            speculators_repo=Path("speculators"),
            training_python=Path("python"),
            verifier_model="v",
            warm_start_model="w",
            truncated_row_policy="drop",  # type: ignore[arg-type]
        )


def test_config_rejects_a_non_boolean_accuracy_gate() -> None:
    with pytest.raises(ValueError, match="require_accuracy_improvement"):
        SpeculatorsPipelineConfig(
            prepared_validator_script=Path("check.py"),
            speculators_repo=Path("speculators"),
            training_python=Path("python"),
            verifier_model="v",
            warm_start_model="w",
            require_accuracy_improvement="yes",  # type: ignore[arg-type]
        )


# ===========================================================================
# FIX 4 -- only turns this verifier produced may be supervised
#
# The gateway tags every inbound message ``client_supplied`` and only the
# response it assembled ``generated`` (``gateway/capture.py``).  The Speculators
# loader masks assistant spans by matching the *rendered* text, so it has no
# per-turn input: a client-supplied assistant turn cannot be rendered as
# context while being withheld from the loss.  These tests pin the only
# representable answer -- drop the row -- and pin that the tag reaches the
# decision at all, which it previously did not.
# ===========================================================================


def _turn(role: str, content: str, **extra: object) -> dict[str, object]:
    return {"role": role, "content": content, **extra}


def _conversation_row(
    index: int,
    turns: Sequence[Mapping[str, object]],
    finish_reason: object = ...,
) -> dict[str, object]:
    record: dict[str, object] = {"id": f"row-{index}", "messages": list(turns)}
    if finish_reason is not ...:
        record["finish_reason"] = finish_reason
    return record


def _generated(index: int, **extra: object) -> dict[str, object]:
    """A single-turn exchange exactly as the gateway writes it."""
    return _conversation_row(
        index,
        [
            _turn("user", "q", provenance_tag="client_supplied"),
            _turn("assistant", "a", provenance_tag="generated", **extra),
        ],
    )


def _replayed(index: int) -> dict[str, object]:
    """A multi-turn request: the client replayed an assistant turn as history.

    Only the trailing turn is this verifier's; the middle one arrived in the
    request body and may have come from any model at all.
    """
    return _conversation_row(
        index,
        [
            _turn("user", "q1", provenance_tag="client_supplied"),
            _turn("assistant", "history", provenance_tag="client_supplied"),
            _turn("user", "q2", provenance_tag="client_supplied"),
            _turn("assistant", "a", provenance_tag="generated"),
        ],
    )


def test_the_provenance_tag_reaches_the_rendered_row() -> None:
    """It is captured, validated and leased; it was dropped only here."""
    converted = _speculators_record(_generated(0))

    assert converted is not None
    turns = converted["conversations"]
    assert [turn.get("provenance_tag") for turn in turns] == [  # type: ignore[union-attr]
        "client_supplied",
        "generated",
    ]


def test_a_client_supplied_assistant_turn_is_never_supervised(
    tmp_path: Path,
) -> None:
    """The draft head must not be taught to predict another model's outputs."""
    records = [_generated(0), _replayed(1), _generated(2)]

    counts, written = _render(tmp_path, records)

    assert [row["id"] for row in written] == ["row-0", "row-2"]
    assert counts.dropped_client_supplied == 1
    assert counts.client_supplied_turns_seen == 1
    assert counts.written == 2
    assert counts.to_dict()["dropped_client_supplied"] == 1


def test_every_client_supplied_turn_in_a_row_is_counted(tmp_path: Path) -> None:
    """One row, two foreign turns -- the evidence must say two, not one."""
    records = [
        _conversation_row(
            0,
            [
                _turn("user", "q1", provenance_tag="client_supplied"),
                _turn("assistant", "h1", provenance_tag="client_supplied"),
                _turn("assistant", "h2", provenance_tag="client_supplied"),
                _turn("assistant", "a", provenance_tag="generated"),
            ],
        ),
        _generated(1),
    ]

    counts, _ = _render(tmp_path, records)

    assert counts.dropped_client_supplied == 1
    assert counts.client_supplied_turns_seen == 2


def test_truncation_seen_counts_rows_an_earlier_filter_drops(tmp_path: Path) -> None:
    """Evidence counters may overlap even though drop buckets are exclusive.

    RED before the fix: authorship ran first and continued, so a replayed row
    with ``finish_reason=length`` incremented ``dropped_client_supplied`` but
    disappeared from ``truncated_seen``.  The returned accounting reconciled
    while its truncation diagnostic understated the corpus.
    """
    replayed = _replayed(0)
    replayed["finish_reason"] = "length"

    counts, _ = _render(
        tmp_path,
        [replayed, _generated(1)],
        policy=TruncatedRowPolicy.DROP,
    )

    assert counts.read == 2
    assert counts.written == 1
    assert counts.dropped_client_supplied == 1
    assert counts.dropped_truncated == 0
    assert counts.truncated_seen == 1


def test_an_untagged_assistant_turn_fails_closed(tmp_path: Path) -> None:
    """An absent tag means "corpus predates tagging", not "we wrote it"."""
    records = [_conversation_row(0, [_turn("user", "q"), _turn("assistant", "a")])]

    with pytest.raises(ClientSupervisedCorpusError):
        _render(tmp_path, records)


def test_the_named_opt_in_admits_a_trusted_untagged_corpus(tmp_path: Path) -> None:
    """Fail-closed must leave a way back, or the failure is unrecoverable."""
    records = [_conversation_row(0, [_turn("user", "q"), _turn("assistant", "a")])]

    counts, written = _render(tmp_path, records, trust_untagged=True)

    assert counts.written == 1
    assert counts.dropped_client_supplied == 0
    assert [row["id"] for row in written] == ["row-0"]


def test_the_opt_in_does_not_admit_an_explicitly_foreign_turn(
    tmp_path: Path,
) -> None:
    """It relabels *untagged* turns only; a known-foreign tag still fails."""
    with pytest.raises(ClientSupervisedCorpusError):
        _render(tmp_path, [_replayed(0)], trust_untagged=True)


def test_a_prefilled_assistant_turn_is_not_wholly_this_verifiers(
    tmp_path: Path,
) -> None:
    """A prefill continuation is one turn tagged ``generated`` whose *prefix*
    the client wrote; that prefix renders inside the masked assistant span."""
    records = [
        _generated(0, prefill_prefix_chars=7),
        _generated(1, prefill_prefix_chars=0),
    ]

    counts, written = _render(tmp_path, records)

    assert [row["id"] for row in written] == ["row-1"]
    assert counts.dropped_client_supplied == 1


def test_an_unmeasurable_prefill_prefix_is_not_assumed_empty(
    tmp_path: Path,
) -> None:
    """``normalize.py`` writes ``None`` when it could not measure the split."""
    with pytest.raises(ClientSupervisedCorpusError):
        _render(tmp_path, [_generated(0, prefill_prefix_chars=None)])


def test_the_authorship_filter_emptying_the_corpus_is_a_loud_named_failure(
    tmp_path: Path,
) -> None:
    """A silently tiny corpus produces a checkpoint that looks like any other."""
    records = [_replayed(index) for index in range(9)] + [_generated(9)]

    with pytest.raises(ClientSupervisedCorpusError) as caught:
        _render(tmp_path, records, minimum_rows=4)

    message = str(caught.value)
    assert "1 trainable rows" in message
    assert "below the floor of 4" in message
    assert "9 of 10 rows" in message
    assert "trust_untagged_assistant_messages" in message


def test_the_authorship_floor_does_not_fire_on_a_merely_small_corpus(
    tmp_path: Path,
) -> None:
    """The floor bounds this filter's damage, not the corpus's own size."""
    counts, written = _render(tmp_path, [_generated(0)], minimum_rows=100)

    assert counts.written == 1
    assert len(written) == 1


def test_the_two_row_filters_report_their_own_causes(tmp_path: Path) -> None:
    """Each filter names itself; neither is attributed to the other."""
    records = [_generated(0, ), _replayed(1), _generated(2)]
    records[0]["finish_reason"] = "length"
    records[2]["finish_reason"] = "stop"

    counts, written = _render(tmp_path, records, policy=TruncatedRowPolicy.DROP)

    assert [row["id"] for row in written] == ["row-2"]
    assert counts.dropped_truncated == 1
    assert counts.dropped_client_supplied == 1


def test_the_filter_that_dropped_the_most_rows_owns_the_failure(
    tmp_path: Path,
) -> None:
    """Job 375414's shape: both filters fired, the authorship one did the damage.

    409 rows read, none written, 36 dropped as truncated and 373 as
    client-supplied -- and the failure raised was the truncation one, purely
    because it is checked first.  The remedy it printed (raise the serving
    ``max_tokens`` cap) would have left the corpus exactly as empty.
    """
    records: list[Mapping[str, object]] = [_replayed(index) for index in range(8)]
    truncated = _generated(8)
    truncated["finish_reason"] = "length"
    records.append(truncated)

    with pytest.raises(ClientSupervisedCorpusError) as caught:
        _render(
            tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4
        )

    message = str(caught.value)
    assert caught.value.counts.dropped_client_supplied == 8
    assert caught.value.counts.dropped_truncated == 1
    assert "trust_untagged_assistant_messages" in message
    # The subordinate cause is reported, not silently folded into the dominant
    # one: an operator who fixes only the named cause must be able to see what
    # is left.
    assert "1 were truncated" in message


def test_truncation_still_owns_a_shortfall_it_dominates(tmp_path: Path) -> None:
    """The tiebreak must not simply invert the old fixed order.

    Its companion above passes for an implementation that always blames the
    authorship filter; this one fails for it.
    """
    records: list[Mapping[str, object]] = []
    for index in range(8):
        row = _generated(index)
        row["finish_reason"] = "length"
        records.append(row)
    records.append(_replayed(8))

    with pytest.raises(TruncationFilteredCorpusError) as caught:
        _render(
            tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4
        )

    message = str(caught.value)
    assert caught.value.counts.dropped_truncated == 8
    assert caught.value.counts.dropped_client_supplied == 1
    # Truncation dominates, so the truncation remedy leads -- but the row the
    # other filter took is named, with the knob that would return it.
    assert "raise the serving max_tokens cap" in message
    assert "trust_untagged_assistant_messages" in message


def _untrainable(index: int) -> dict[str, object]:
    """A row with no assistant turn at all -- nothing to supervise."""
    return {
        "id": f"untrainable-{index}",
        "messages": [{"role": "user", "content": "q"}],
    }


def test_the_third_bucket_can_win_the_attribution(tmp_path: Path) -> None:
    """``dropped_untrainable`` is reconciled, so it must be attributable too.

    The two-way comparison could not see it.  90 untrainable, 6 truncated, 4
    client-supplied and nothing written reported *truncation* and prescribed
    raising the token cap -- a remedy for 6 of the 100 missing rows, and the
    smallest of the three causes at that.  Job 375414 was this same shape one
    bucket over; the fix for it left this one open.

    Untrainability has no knob, so the honest answer names no remedy and prints
    the whole accounting instead.
    """
    records: list[Mapping[str, object]] = [_untrainable(i) for i in range(90)]
    for index in range(90, 96):
        row = _generated(index)
        row["finish_reason"] = "length"
        records.append(row)
    records.extend(_replayed(index) for index in range(96, 100))

    with pytest.raises(Eagle3Error) as caught:
        _render(tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4)

    assert not isinstance(caught.value, TruncationFilteredCorpusError), (
        "the smallest cause was reported as the dominant one"
    )
    message = str(caught.value)
    assert "90 carried no trainable assistant turn" in message
    assert "6 were truncated" in message
    assert "4 carried an assistant turn this verifier did not produce" in message
    assert "raise the serving max_tokens cap" not in message, (
        "a remedy was prescribed for a cause it cannot fix"
    )


def test_a_shortfall_no_bucket_dominates_names_no_single_cause(
    tmp_path: Path,
) -> None:
    """An exact tie is not a diagnosis, and picking one was a coin flip.

    The predecessor's comment said "ties keep the historical answer", i.e. on
    5 truncated against 5 client-supplied it blamed truncation -- for no reason
    beyond which branch was written first.  An operator raising the token cap
    on that advice recovers half the loss and learns nothing about the rest.
    """
    records: list[Mapping[str, object]] = []
    for index in range(5):
        row = _generated(index)
        row["finish_reason"] = "length"
        records.append(row)
    records.extend(_replayed(index) for index in range(5, 10))

    with pytest.raises(Eagle3Error) as caught:
        _render(tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4)

    assert not isinstance(
        caught.value, (TruncationFilteredCorpusError, ClientSupervisedCorpusError)
    ), "a tie was resolved in favour of one filter anyway"
    message = str(caught.value)
    assert "5 were truncated" in message
    assert "5 carried an assistant turn this verifier did not produce" in message


def test_a_partial_exact_tie_reaches_the_unattributed_failure(
    tmp_path: Path,
) -> None:
    """The tie branch must be exercised below the floor with rows remaining.

    An all-dropped tie correctly takes ``EmptySpeculatorsDatasetError`` because
    "nothing converted" is its truest headline.  That means the companion test
    above never reaches ``UnattributedCorpusShortfallError``'s ``dominant is
    None`` message.  With one row written, the same tie reaches that branch --
    and both tied filters have settings that can recover rows, so it must not
    claim that only replacing the corpus can help.
    """
    records: list[Mapping[str, object]] = []
    for index in range(3):
        row = _generated(index)
        row["finish_reason"] = "length"
        records.append(row)
    records.extend(_replayed(index) for index in range(3, 6))
    records.append(_generated(6))

    with pytest.raises(UnattributedCorpusShortfallError) as caught:
        _render(tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=4)

    message = str(caught.value)
    assert "1 trainable rows, below the floor of 4" in message
    assert "no single filter dominates" in message
    assert "filter settings can recover rows" in message
    assert "not a change to the filters" not in message


def test_a_partial_render_no_bucket_can_fix_is_reported_as_such(
    tmp_path: Path,
) -> None:
    """Below the floor but not empty, and the dominant cause has no remedy.

    The one case that is neither of the two dedicated failures nor "nothing
    converted", so it is what earns ``UnattributedCorpusShortfallError`` its
    own class: there is a real, quantified shortfall and deliberately no knob
    to recommend.
    """
    records: list[Mapping[str, object]] = [_untrainable(i) for i in range(20)]
    for index in range(20, 23):
        row = _generated(index)
        row["finish_reason"] = "length"
        records.append(row)
    records.extend(_generated(index) for index in range(23, 25))

    with pytest.raises(UnattributedCorpusShortfallError) as caught:
        _render(tmp_path, records, policy=TruncatedRowPolicy.DROP, minimum_rows=10)

    assert caught.value.counts.written == 2
    message = str(caught.value)
    assert "2 trainable rows, below the floor of 10" in message
    assert "20 carried no trainable assistant turn" in message
    assert "no setting on this pipeline can recover" in message


@pytest.mark.parametrize(
    "error_type",
    [TruncationFilteredCorpusError, ClientSupervisedCorpusError],
)
def test_every_drop_bucket_is_named_in_the_failure(
    error_type: type[Eagle3Error],
) -> None:
    """The numbers in the message must reconcile against ``read``.

    Job 375414's message accounted for 36 of 409 missing rows and named a
    remedy for those 36 only.  Reconciliation is what makes that unwritable:
    a drop bucket added to ``RenderedRowCounts`` without a clause in
    ``_DROP_BUCKETS`` leaves the message short of ``read`` and fails here.
    """
    import dataclasses

    from speedlm.training.backends.eagle3 import _DROP_BUCKETS

    counts = RenderedRowCounts(
        read=409,
        written=0,
        dropped_untrainable=0,
        dropped_truncated=36,
        truncated_seen=36,
        policy=TruncatedRowPolicy.DROP,
        dropped_client_supplied=373,
        client_supplied_turns_seen=1_193,
    )
    buckets = [
        field.name
        for field in dataclasses.fields(RenderedRowCounts)
        if field.name.startswith("dropped_")
    ]
    assert set(buckets) == set(_DROP_BUCKETS), (
        "a drop bucket without a clause in _DROP_BUCKETS cannot appear in the "
        "failure message"
    )
    assert counts.read == counts.written + sum(
        getattr(counts, name) for name in buckets
    )

    message = str(error_type(Path("/corpus.jsonl"), counts, 32))

    for name in buckets:
        assert f"{getattr(counts, name)} {_DROP_BUCKETS[name]}" in message, (
            f"{name} is not accounted for in {error_type.__name__}"
        )
    assert "409 rows read" in message
    assert "trust_untagged_assistant_messages" in message


def _authorship_renderer(
    tmp_path: Path,
    **config_kwargs: object,
) -> tuple[object, _RecordingRunner, _State]:
    from speedlm.training.backends.eagle3 import SpeculatorsTrainingRowRenderer

    config = SpeculatorsPipelineConfig(
        prepared_validator_script=tmp_path / "check.py",
        speculators_repo=tmp_path / "speculators",
        training_python=tmp_path / "python",
        verifier_model=str(_verifier(tmp_path / "verifier", 36)),
        warm_start_model=str(tmp_path / "warm"),
        min_rendered_rows=1,
        **config_kwargs,  # type: ignore[arg-type]
    )
    runner = _RecordingRunner()
    state = _State()
    state.verifier = config.verifier_model
    renderer = SpeculatorsTrainingRowRenderer(
        config, runner, _Resolver(config, runner, state), state
    )
    return renderer, runner, state


def test_enabled_loss_mask_dilation_runs_between_prepare_and_validation(
    tmp_path: Path,
) -> None:
    """The post-pass must use the training venv and precede validation."""
    renderer, runner, _ = _authorship_renderer(
        tmp_path, dilate_loss_mask_span_starts=True
    )
    work = tmp_path / "work"
    work.mkdir()
    snapshot = _snapshot(work / "snap.jsonl", [_generated(0)])

    renderer.render_rows(  # type: ignore[attr-defined]
        snapshot,
        work / "training-rows",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    stage_calls = [
        call
        for call in runner.run_calls
        if Path(call[1]).name
        in {"prepare_data.py", "dilate_prepared_loss_mask.py", "check.py"}
    ]
    assert [Path(call[1]).name for call in stage_calls] == [
        "prepare_data.py",
        "dilate_prepared_loss_mask.py",
        "check.py",
    ]
    assert stage_calls[1] == (
        str(tmp_path / "python"),
        str(
            Path(__file__).parents[1]
            / "src"
            / "speedlm"
            / "training"
            / "dilate_prepared_loss_mask.py"
        ),
        str(work / "training-rows"),
    )


def test_the_authorship_filter_runs_in_the_real_render_stage(
    tmp_path: Path,
) -> None:
    """Through ``render_rows`` and the real config, not by calling the helper.

    Otherwise nothing proves the stage that owns the filter passes the operator
    setting through, and the filter could be permanently inert in production.
    """
    renderer, runner, state = _authorship_renderer(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    snapshot = _snapshot(
        work / "snap.jsonl", [_generated(0), _replayed(1), _generated(2)]
    )

    renderer.render_rows(  # type: ignore[attr-defined]
        snapshot,
        work / "training-rows",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.rendered_rows is not None
    assert state.rendered_rows["dropped_client_supplied"] == 1
    assert state.rendered_rows["written"] == 2
    # The row count handed to the prepared-dataset validator must be the
    # post-filter count, or the validator asserts a size the corpus no longer has.
    assert state.row_count == 2
    prepare = next(
        call for call in runner.run_calls if "prepare_data.py" in " ".join(call)
    )
    # Every assistant span the loader finds receives loss, because nothing here
    # narrows it to the final one.  That is exactly why the filter above must
    # guarantee that every span it leaves behind is this verifier's own output.
    assert "--final-assistant-only-loss-mask" not in prepare


def test_the_render_stage_refuses_untagged_turns_by_default(tmp_path: Path) -> None:
    """The stage must carry the *configured* setting, not a hardcoded one.

    Its companion below proves the opt-in reaches the filter; this proves the
    default does too, which a stage that always opted in would pass.
    """
    renderer, _, state = _authorship_renderer(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    snapshot = _snapshot(
        work / "snap.jsonl",
        [
            _conversation_row(0, [_turn("user", "q"), _turn("assistant", "a")]),
            _generated(1),
        ],
    )

    renderer.render_rows(  # type: ignore[attr-defined]
        snapshot,
        work / "training-rows",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.rendered_rows is not None
    assert state.rendered_rows["dropped_client_supplied"] == 1
    assert state.rendered_rows["written"] == 1


def test_the_render_stage_honours_the_untagged_opt_in(tmp_path: Path) -> None:
    """The knob is useless if the stage that owns the filter never reads it."""
    renderer, _, state = _authorship_renderer(
        tmp_path, trust_untagged_assistant_messages=True
    )
    work = tmp_path / "work"
    work.mkdir()
    snapshot = _snapshot(
        work / "snap.jsonl",
        [_conversation_row(0, [_turn("user", "q"), _turn("assistant", "a")])],
    )

    renderer.render_rows(  # type: ignore[attr-defined]
        snapshot,
        work / "training-rows",
        timeout_seconds=60.0,
        should_abort=lambda: False,
    )

    assert state.rendered_rows is not None
    assert state.rendered_rows["written"] == 1


def test_the_untagged_opt_in_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="trust_untagged_assistant_messages"):
        SpeculatorsPipelineConfig(
            prepared_validator_script=Path("check.py"),
            speculators_repo=Path("speculators"),
            training_python=Path("python"),
            verifier_model="v",
            warm_start_model="w",
            trust_untagged_assistant_messages="yes",  # type: ignore[arg-type]
        )


def test_the_authorship_failure_is_catchable_as_an_eagle3_error() -> None:
    """Orchestration catches Eagle3Error; a new failure outside it escapes."""
    assert issubclass(ClientSupervisedCorpusError, Eagle3Error)


# ===========================================================================
# All three land in the manifest
# ===========================================================================


def test_every_guard_publishes_its_verdict_into_the_training_params(
    tmp_path: Path,
) -> None:
    """A record written only on one outcome cannot be compared against one that
    says the opposite, so all three are written whenever their stage ran."""
    from speedlm.training.backends.eagle3 import Eagle3Backend, Eagle3Config

    backend = object.__new__(Eagle3Backend)
    backend.config = Eagle3Config(
        verifier_model="org/verifier",
        draft_model="org/draft",
        from_pretrained="org/draft",
        training_params={"epochs": 3},
    )
    state = _State()
    state.rendered_rows = RenderedRowCounts(
        read=10,
        written=7,
        dropped_untrainable=1,
        dropped_truncated=2,
        truncated_seen=2,
        policy=TruncatedRowPolicy.DROP,
    ).to_dict()
    state.warm_start_aux = {"verdict": "checkpoint_declared"}
    state.val_accuracy = {"verdict": "improved", "final": 0.71}
    backend._state = state

    params = backend.describe().training_params

    assert params["rendered_rows"]["dropped_truncated"] == 2  # type: ignore[index]
    assert params["rendered_rows"]["truncated_row_policy"] == "drop"  # type: ignore[index]
    assert params["warm_start_aux"]["verdict"] == "checkpoint_declared"  # type: ignore[index]
    assert params["val_accuracy"]["final"] == 0.71  # type: ignore[index]
    # JSON-serializable, because this becomes manifest.json.
    json.dumps(dict(params))


def test_describe_is_unchanged_when_no_stage_ran() -> None:
    from speedlm.training.backends.eagle3 import Eagle3Backend, Eagle3Config

    backend = object.__new__(Eagle3Backend)
    backend.config = Eagle3Config(
        verifier_model="org/verifier",
        draft_model="org/draft",
        from_pretrained="org/draft",
        training_params={"epochs": 3},
    )
    backend._state = _State()

    params = backend.describe().training_params

    assert "rendered_rows" not in params
    assert "val_accuracy" not in params
    assert "warm_start_aux" not in params


def test_eagle3_error_is_the_base_of_the_new_failures() -> None:
    """Orchestration catches Eagle3Error; a new failure outside it escapes."""
    assert issubclass(TruncationFilteredCorpusError, Eagle3Error)
    assert issubclass(WarmStartLayerMismatchError, Eagle3Error)
    assert issubclass(AccuracyRegressionError, Eagle3Error)
