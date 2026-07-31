from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import pytest

from speedlm.training.backends.eagle3 import (
    _RESOLVE_MODEL,
    _VALIDATE_DRAFT,
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


def test_eagle3_config_requires_explicit_model() -> None:
    """Constructing Eagle3Config without a model must raise."""
    with pytest.raises(TypeError, match="missing"):
        Eagle3Config()  # type: ignore[call-arg]


def test_eagle3_backend_satisfies_protocol_and_declares_distillation_contract() -> None:
    backend = object.__new__(Eagle3Backend)
    config = Eagle3Config(
        verifier_model="openai/gpt-oss-20b",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        from_pretrained="RedHatAI/gpt-oss-20b-speculator.eagle3",
        mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS,
    )

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
        verifier_model="openai/gpt-oss-20b",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        from_pretrained="RedHatAI/gpt-oss-20b-speculator.eagle3",
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
    backend.config = Eagle3Config(
        verifier_model="openai/gpt-oss-20b",
        draft_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
        from_pretrained="RedHatAI/gpt-oss-20b-speculator.eagle3",
        mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS,
    )

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
        verifier_model="openai/gpt-oss-20b",
        warm_start_model="RedHatAI/gpt-oss-20b-speculator.eagle3",
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


_QWEN_SNAPSHOT = (
    "/data/ryan.kim/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d"
)
#: Distinguishes a config that omits the key from one that spells it ``null``.
_ABSENT = object()


def _write_draft(
    root: Path,
    *,
    verifier: str,
    layer_ids: object,
    nest_layer_ids: bool = False,
) -> Path:
    """Materialize the on-disk shape of a published EAGLE-3 drafter."""
    draft = root / "draft"
    draft.mkdir()
    speculators: dict[str, object] = {
        "algorithm": "eagle3",
        "verifier": {"architectures": ["Qwen3ForCausalLM"], "name_or_path": verifier},
    }
    config: dict[str, object] = {
        "speculators_model_type": "eagle3",
        "speculators_config": speculators,
    }
    if layer_ids is not _ABSENT:
        if nest_layer_ids:
            speculators["eagle_aux_hidden_state_layer_ids"] = layer_ids
        else:
            config["eagle_aux_hidden_state_layer_ids"] = layer_ids
    (draft / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (draft / "model.safetensors").write_text("", encoding="utf-8")
    return draft


def _run_validate_script(draft: Path, verifier: str, layers: Sequence[int]) -> None:
    """Execute the real _VALIDATE_DRAFT source against a stub safetensors.

    The repo venv carries no safetensors, and the weight check is not what is
    under test here, so the reader is stubbed to report the vocab mappings a
    healthy draft has.  Everything before it is the real snippet.
    """

    class _Handle:
        def __enter__(self) -> _Handle:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def keys(self) -> set[str]:
            return {"d2t", "t2d"}

    module = types.ModuleType("safetensors")
    module.safe_open = lambda *_a, **_k: _Handle()  # type: ignore[attr-defined]

    argv = ["-", str(draft), verifier, *(str(layer) for layer in layers)]
    with mock.patch.dict(sys.modules, {"safetensors": module}), mock.patch.object(
        sys, "argv", argv
    ):
        exec(compile(_VALIDATE_DRAFT, "<validate>", "exec"), {"__name__": "__main__"})


def test_a_snapshot_path_validates_against_the_repo_id_it_resolves_to(
    tmp_path: Path,
) -> None:
    """Reproduces job 368962: the same model named two ways is not a mismatch."""
    draft = _write_draft(tmp_path, verifier="Qwen/Qwen3-8B", layer_ids=None)

    _run_validate_script(draft, _QWEN_SNAPSHOT + "/", (2, 18, 33))


def test_a_genuinely_different_verifier_still_fails_loudly(tmp_path: Path) -> None:
    """Canonicalisation must not blur two distinct models together."""
    draft = _write_draft(tmp_path, verifier="openai/gpt-oss-20b", layer_ids=None)

    with pytest.raises(SystemExit, match="draft verifier mismatch"):
        _run_validate_script(draft, _QWEN_SNAPSHOT, (2, 18, 33))


@pytest.mark.parametrize("layer_ids", [None, _ABSENT])
def test_unpinned_layer_ids_do_not_contradict_the_contract(
    tmp_path: Path, layer_ids: object
) -> None:
    """Both published drafters ship null here; null pins nothing."""
    draft = _write_draft(tmp_path, verifier="Qwen/Qwen3-8B", layer_ids=layer_ids)

    _run_validate_script(draft, _QWEN_SNAPSHOT, (2, 18, 33))


def test_layer_ids_compare_as_ints_not_as_json_versus_tuple(tmp_path: Path) -> None:
    """A JSON list and the contract's tuple are the same pinned layers."""
    draft = _write_draft(tmp_path, verifier="Qwen/Qwen3-8B", layer_ids=[2, 18, 33])

    _run_validate_script(draft, _QWEN_SNAPSHOT, (2, 18, 33))


@pytest.mark.parametrize("nested", [False, True])
def test_pinned_layer_ids_that_disagree_with_the_contract_fail(
    tmp_path: Path, nested: bool
) -> None:
    """A present list is a claim, and a false claim must stop the run."""
    draft = _write_draft(
        tmp_path,
        verifier="Qwen/Qwen3-8B",
        layer_ids=[1, 2, 3],
        nest_layer_ids=nested,
    )

    with pytest.raises(SystemExit, match="draft target layer ids"):
        _run_validate_script(draft, _QWEN_SNAPSHOT, (2, 18, 33))
