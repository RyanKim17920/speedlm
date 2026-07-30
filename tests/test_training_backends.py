from __future__ import annotations

import contextlib
import io
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from speedlm.training.backends.eagle3 import (
    _RESOLVE_MODEL,
    Eagle3Backend,
    Eagle3Config,
    SpeculatorsPipelineConfig,
    _Resolver,
    _State,
)
from speedlm.training.backends.speculators_runner import ProcessResult
from speedlm.training.base import SpeculatorBackend
from speedlm.training.masking import (
    MaskPolicy,
    require_trainable_window,
    summarize_training_window,
)


def test_shifted_label_and_truncation_accounting() -> None:
    rows = [
        {
            "id": "long",
            "input_ids": [1] * 10,
            "loss_mask": [False] * 7 + [True] * 3,
            "seq_len": 10,
        },
        {
            "id": "short",
            "input_ids": [1] * 5,
            "loss_mask": [True, False, False, True, True],
            "seq_len": 5,
        },
    ]

    summary = summarize_training_window(rows, total_seq_len=8)

    assert summary.total_supervised_tokens == 5
    assert summary.retained_supervised_tokens == 4
    assert summary.truncated_supervised_tokens == 1
    assert summary.rows_with_truncated_supervision == 1
    assert summary.rows_with_all_supervision_truncated == 0
    assert require_trainable_window(rows, 8) == summary


def test_shift_drops_position_zero_and_reports_row_id() -> None:
    rows = [
        {
            "id": "position-zero-only",
            "input_ids": [1, 2],
            "loss_mask": [True, False],
            "seq_len": 2,
        }
    ]

    try:
        require_trainable_window(rows, 8)
    except ValueError as error:
        assert "position-zero-only" in str(error)
        assert "post-shift" in str(error)
    else:
        raise AssertionError("expected shifted all-zero supervision to fail")


def test_eagle3_backend_satisfies_protocol_and_declares_distillation_contract() -> None:
    backend = object.__new__(Eagle3Backend)
    config = Eagle3Config(mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS)

    assert isinstance(backend, SpeculatorBackend)
    assert config.from_pretrained == "RedHatAI/gpt-oss-20b-speculator.eagle3"
    assert config.effective_training_params["distillation_loss"] == "soft_kl"
    assert config.effective_training_params["draft_vocabulary"] == "reduced_d2t_t2d"
    assert config.effective_training_params["num_speculative_steps"] == 3
    assert config.effective_training_params["ttt_loss_reduction"] == "sum"


def test_describe_carries_the_pinned_verifier_revision_into_the_manifest() -> None:
    """The manifest must say *which* verifier, not merely which repo."""
    backend = object.__new__(Eagle3Backend)
    backend.config = Eagle3Config(
        mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS,
        verifier_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        training_params={"epochs": 1},
    )

    info = backend.describe()

    assert info.training_params["verifier_revision"] == (
        "6cee5e81ee83917806bbde320786a8fb61efebee"
    )
    assert info.training_params["epochs"] == 1


def test_describe_records_an_unresolved_revision_as_null() -> None:
    """An absent key cannot be told apart from an unpinned cycle; null can."""
    backend = object.__new__(Eagle3Backend)
    backend.config = Eagle3Config(mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS)

    params = backend.describe().training_params

    assert "verifier_revision" in params
    assert params["verifier_revision"] is None


def _pipeline(tmp_path: Path) -> SpeculatorsPipelineConfig:
    script = tmp_path / "check_prepared.py"
    script.write_text("", encoding="utf-8")
    return SpeculatorsPipelineConfig(
        prepared_validator_script=script,
        speculators_repo=tmp_path,
        training_python=tmp_path / "python",
        verifier_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
    )


class _CapturingRunner:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] = ()
        self.stdout = "/data/hf-cache/hub/models--openai--gpt-oss-20b/snapshots/abc\n"

    def run(self, argv: Sequence[str], **_kwargs: object) -> ProcessResult:
        self.argv = tuple(argv)
        return ProcessResult(
            argv=tuple(argv),
            returncode=0,
            stdout=self.stdout,
            stderr="",
        )


def test_a_pinned_revision_narrows_completeness_to_the_files_training_reads(
    tmp_path: Path,
) -> None:
    """The pin must not demand licences and model cards a minimal cache omits."""
    pipeline = _pipeline(tmp_path)
    runner = _CapturingRunner()
    resolver = _Resolver(pipeline, runner, _State())

    resolver.verifier(lambda: False, tmp_path)

    assert pipeline.model_resolve_allow_patterns == ("*.json", "*.safetensors", "*.jinja")
    assert runner.argv[-3:] == pipeline.model_resolve_allow_patterns
    assert runner.argv[3] == "openai/gpt-oss-20b"
    assert runner.argv[4] == "6cee5e81ee83917806bbde320786a8fb61efebee"


def _run_resolve_script(
    argv: list[str],
    *,
    incomplete_for_revision: bool,
) -> tuple[str, list[dict[str, object]]]:
    """Execute the real _RESOLVE_MODEL source against a stub huggingface_hub."""
    calls: list[dict[str, object]] = []

    class IncompleteSnapshotError(Exception):
        pass

    def snapshot_download(
        *,
        repo_id: str,
        revision: str | None = None,
        allow_patterns: object = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": allow_patterns,
            }
        )
        if revision is not None and incomplete_for_revision:
            raise IncompleteSnapshotError(
                f"The cached snapshot for {repo_id!r} is incomplete: "
                "8 file(s) are missing (.gitattributes, LICENSE, README.md, ...)"
            )
        return "/data/hf-cache/hub/snapshots/resolved"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    errors = types.ModuleType("huggingface_hub.errors")
    errors.IncompleteSnapshotError = IncompleteSnapshotError  # type: ignore[attr-defined]

    stdout = io.StringIO()
    with mock.patch.dict(
        sys.modules,
        {"huggingface_hub": hub, "huggingface_hub.errors": errors},
    ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
        stdout
    ), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(_RESOLVE_MODEL, "<resolve>", "exec"), {"__name__": "__main__"})
    return stdout.getvalue().strip(), calls


def test_a_pinned_revision_resolves_the_partial_offline_cache(tmp_path: Path) -> None:
    """A complete-enough minimal cache resolves under the pin, no fallback."""
    path, calls = _run_resolve_script(
        ["-", "openai/gpt-oss-20b", "6cee5e81", "*.json", "*.safetensors"],
        incomplete_for_revision=False,
    )

    assert path == "/data/hf-cache/hub/snapshots/resolved"
    assert calls == [
        {
            "repo_id": "openai/gpt-oss-20b",
            "revision": "6cee5e81",
            "allow_patterns": ["*.json", "*.safetensors"],
        }
    ]


def test_an_unsatisfiable_pin_reports_unresolved_rather_than_redownloading() -> None:
    """Reproduces job 368719: revision=None is not the unpinned path.

    The unpinned path never called snapshot_download at all, so retrying with
    revision=None does not reproduce it -- the completeness check still runs,
    against main.  Resolution must instead say it could not satisfy the pin.
    """
    path, calls = _run_resolve_script(
        ["-", "openai/gpt-oss-20b", "6cee5e81", "*.json"],
        incomplete_for_revision=True,
    )

    assert path == "SPEEDLM_UNRESOLVED"
    assert [call["revision"] for call in calls] == ["6cee5e81"]


def test_an_unsatisfiable_pin_falls_back_to_the_bare_repo_id(tmp_path: Path) -> None:
    """A pin must not stop a cycle the unpinned path would have run."""
    pipeline = _pipeline(tmp_path)
    runner = _CapturingRunner()
    runner.stdout = "SPEEDLM_UNRESOLVED\n"
    resolver = _Resolver(pipeline, runner, _State())

    assert resolver.verifier(lambda: False, tmp_path) == "openai/gpt-oss-20b"
