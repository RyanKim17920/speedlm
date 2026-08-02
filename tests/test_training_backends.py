from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import sys
import types
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from speedlm.training.backends.eagle3 import (
    _RESOLVE_MODEL,
    _VALIDATE_DRAFT,
    Eagle3Adapter,
    Eagle3Backend,
    Eagle3Config,
    Eagle3Error,
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
    assert pipeline.model_resolve_ignore_patterns == ("*/*",)
    assert runner.argv[3] == "openai/gpt-oss-20b"
    assert runner.argv[4] == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert runner.argv[5:] == (
        "--allow",
        *pipeline.model_resolve_allow_patterns,
        "--ignore",
        *pipeline.model_resolve_ignore_patterns,
    )


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
        ignore_patterns: object = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "ignore_patterns": ignore_patterns,
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
        [
            "-",
            "openai/gpt-oss-20b",
            "6cee5e81",
            "--allow",
            "*.json",
            "*.safetensors",
            "--ignore",
            "*/*",
        ],
        incomplete_for_revision=False,
    )

    assert path == "/data/hf-cache/hub/snapshots/resolved"
    assert calls == [
        {
            "repo_id": "openai/gpt-oss-20b",
            "revision": "6cee5e81",
            "allow_patterns": ["*.json", "*.safetensors"],
            "ignore_patterns": ["*/*"],
        }
    ]


def test_an_unsatisfiable_pin_reports_unresolved_rather_than_redownloading() -> None:
    """Reproduces job 368719: revision=None is not the unpinned path.

    The unpinned path never called snapshot_download at all, so retrying with
    revision=None does not reproduce it -- the completeness check still runs,
    against main.  Resolution must instead say it could not satisfy the pin.
    """
    path, calls = _run_resolve_script(
        ["-", "openai/gpt-oss-20b", "6cee5e81", "--allow", "*.json", "--ignore", "*/*"],
        incomplete_for_revision=True,
    )

    assert path == "SPEEDLM_UNRESOLVED"
    assert [call["revision"] for call in calls] == ["6cee5e81"]


#: The file listing of ``openai/gpt-oss-20b`` at the pinned revision, taken
#: from the cached tree listing this host resolves against.  ``original/`` and
#: ``metal/`` are auxiliary formats a minimal cache deliberately never pulls.
_GPT_OSS_REPO_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "USAGE_POLICY",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "metal/model.bin",
    "model-00000-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "original/config.json",
    "original/dtypes.json",
    "original/model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
#: What a minimal cache of that repo holds on disk: the top level only.
_GPT_OSS_MINIMAL_CACHE = tuple(
    name for name in _GPT_OSS_REPO_FILES if "/" not in name
)


def _expected_files(
    repo_files: Sequence[str],
    allow_patterns: Sequence[str],
    ignore_patterns: Sequence[str] = (),
) -> list[str]:
    """Mirror ``huggingface_hub.utils.filter_repo_objects``.

    That helper is what ``_raise_if_incomplete_snapshot`` runs the repo's tree
    listing through before demanding every survivor exist on disk: ``fnmatch``
    against the *full path*, allow first, then ignore.  It is reproduced here
    rather than imported because huggingface_hub is a dependency of the
    training venv, not of this repo, and the point of the test is the pattern
    semantics, which are ``fnmatch``'s.
    """
    return [
        name
        for name in repo_files
        if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns)
        and not any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns)
    ]


def test_unanchored_allow_patterns_demand_files_a_minimal_cache_never_pulled(
    tmp_path: Path,
) -> None:
    """Pins the defect: ``*`` crosses ``/``, so ``*.json`` claims ``original/``.

    This is what made job 369373 record ``verifier_revision_satisfied: false``
    on a cache that held every file training reads.
    """
    pipeline = _pipeline(tmp_path)

    unanchored = _expected_files(
        _GPT_OSS_REPO_FILES, pipeline.model_resolve_allow_patterns
    )

    assert [name for name in unanchored if "/" in name] == [
        "original/config.json",
        "original/dtypes.json",
        "original/model.safetensors",
    ]
    assert [name for name in unanchored if name not in _GPT_OSS_MINIMAL_CACHE]


def test_the_shipped_patterns_match_the_top_level_files_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The anchored pair expects exactly what a minimal cache holds."""
    pipeline = _pipeline(tmp_path)

    expected = _expected_files(
        _GPT_OSS_REPO_FILES,
        pipeline.model_resolve_allow_patterns,
        pipeline.model_resolve_ignore_patterns,
    )

    assert expected == [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert not [name for name in expected if name not in _GPT_OSS_MINIMAL_CACHE]


def test_the_anchored_patterns_leave_a_flat_repo_untouched(tmp_path: Path) -> None:
    """Qwen3-8B publishes no nested paths, so anchoring must change nothing."""
    pipeline = _pipeline(tmp_path)
    qwen_files = (
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    anchored = _expected_files(
        qwen_files,
        pipeline.model_resolve_allow_patterns,
        pipeline.model_resolve_ignore_patterns,
    )

    assert anchored == _expected_files(
        qwen_files, pipeline.model_resolve_allow_patterns
    )
    assert "config.json" in anchored
    assert "model-00005-of-00005.safetensors" in anchored
    assert "vocab.json" in anchored
    assert "README.md" not in anchored


def test_a_pattern_colliding_with_an_argv_marker_is_rejected(tmp_path: Path) -> None:
    """The two lists are split on markers, so a pattern may not be one."""
    with pytest.raises(ValueError, match="model_resolve_allow_patterns"):
        replace(_pipeline(tmp_path), model_resolve_allow_patterns=("--ignore",))
    with pytest.raises(ValueError, match="model_resolve_ignore_patterns"):
        replace(_pipeline(tmp_path), model_resolve_ignore_patterns=("",))


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


# ---------------------------------------------------------------------------
# Per-cycle warm start
# ---------------------------------------------------------------------------


class _TrackingRunner:
    """Records every resolution the resolver actually delegated."""

    def __init__(self) -> None:
        self.models: list[tuple[str, str]] = []

    def run(self, argv: Sequence[str], **_kwargs: object) -> ProcessResult:
        self.models.append((argv[3], argv[4]))
        return ProcessResult(
            argv=tuple(argv),
            returncode=0,
            stdout=f"/snapshots/{argv[3].replace('/', '--')}\n",
            stderr="",
        )


def test_the_warm_start_memo_re_resolves_when_the_requested_base_moves(
    tmp_path: Path,
) -> None:
    """The memo outlives a cycle, so it has to be keyed on what it memoized.

    ``_State`` is built once per process, not once per cycle.  An unkeyed memo
    returns the first cycle's resolution forever, which would reintroduce the
    frozen warm start one layer below the resolver that exists to remove it.
    """
    pipeline = replace(_pipeline(tmp_path), warm_start_revision="draft-sha")
    runner = _TrackingRunner()
    resolver = _Resolver(pipeline, runner, _State())

    first = resolver.warm_start(pipeline.warm_start_model, lambda: False, tmp_path)
    again = resolver.warm_start(pipeline.warm_start_model, lambda: False, tmp_path)
    assert first == again == "/snapshots/RedHatAI--gpt-oss-20b-speculator.eagle3"
    # Memoized: the same base resolves once, not once per call.
    assert runner.models == [(pipeline.warm_start_model, "draft-sha")]

    promoted = tmp_path / "artifacts" / "promoted"
    promoted.mkdir(parents=True)
    assert resolver.warm_start(str(promoted), lambda: False, tmp_path) == str(promoted)
    # A promoted artifact is a directory of materialized weights.  It must not
    # be handed to Hub snapshot resolution at all, and the stock drafter's
    # revision pin must not follow it: two independent guards, both required.
    assert runner.models == [(pipeline.warm_start_model, "draft-sha")]

    # And moving back is a re-resolution, not a stale hit on the directory.
    assert resolver.warm_start(pipeline.warm_start_model, lambda: False, tmp_path) == (
        "/snapshots/RedHatAI--gpt-oss-20b-speculator.eagle3"
    )
    assert runner.models == [(pipeline.warm_start_model, "draft-sha")] * 2


def _bare_adapter(resolver: object) -> Eagle3Adapter:
    """An adapter whose only reachable stage is the warm-start resolution."""
    return Eagle3Adapter(
        Eagle3Config(
            verifier_model="acme/verifier",
            draft_model="acme/stock",
            from_pretrained="acme/stock",
            mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS,
        ),
        leaser=object(),  # type: ignore[arg-type]
        renderer=object(),  # type: ignore[arg-type]
        extractor=object(),  # type: ignore[arg-type]
        trainer=object(),  # type: ignore[arg-type]
        materializer=object(),  # type: ignore[arg-type]
        validator=object(),  # type: ignore[arg-type]
        warm_start_resolver=resolver,  # type: ignore[arg-type]
    )


def test_describe_reports_the_configured_base_until_a_cycle_resolves_one(
    tmp_path: Path,
) -> None:
    """Which is what ``TunerOrchestrator._active_draft`` needs from it.

    That caller reads ``from_pretrained`` only when the registry holds no
    active artifact, so it must never be handed an artifact directory that the
    registry no longer names.  Before any training there is nothing resolved to
    hand it, and a resolved value can only ever *be* an artifact directory
    because the registry had one.
    """
    adapter = _bare_adapter(lambda: str(tmp_path / "promoted"))
    assert adapter.describe().from_pretrained == "acme/stock"


def test_a_warm_start_resolver_that_names_nothing_fails_closed(
    tmp_path: Path,
) -> None:
    """Silently substituting stock would restart the chain invisibly.

    The artifacts would then be indistinguishable from a chain that had never
    been compounding at all, which is exactly the ambiguity the resolved
    ``base_draft`` exists to remove.
    """
    adapter = _bare_adapter(lambda: "")
    with pytest.raises(Eagle3Error, match="named no checkpoint"):
        adapter.train(tmp_path / "hidden", tmp_path / "work", should_abort=lambda: False)
