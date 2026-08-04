"""The draft validator must prove the trained weights landed.

Before these, the embedded validator checked only that safetensors existed and
that ``d2t``/``t2d`` were among their keys.  It never required ``fc.weight``,
never compared tensor bytes, and so could not tell "the trained checkpoint was
published" apart from "something else was" -- the checkpoint_best ->
materialize -> publish path was correct, but only by inference, never by
proof.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from draft_weights import draft_payloads, safetensors_bytes, write_draft_weights

from speedlm.training.masking import MaskPolicy
from speedlm.tuner.artifacts import ArtifactRegistry, ArtifactSpec, hash_directory
from speedlm.tuner.eagle3 import (
    REQUIRED_DRAFT_TENSORS,
    DeclaredDepthError,
    DraftWeightsError,
    Eagle3Adapter,
    Eagle3Config,
    StockIdenticalDraftError,
    declare_speculative_tokens,
    declared_speculative_tokens,
    draft_tensor_keys,
    weight_fingerprint,
)


def _adapter(
    tmp_path: Path,
    *,
    draft_model: str,
    warm_start: str | None = None,
    depth: int = 3,
) -> Eagle3Adapter:
    adapter = object.__new__(Eagle3Adapter)
    adapter.config = Eagle3Config(
        verifier_model="acme/verifier",
        draft_model=draft_model,
        from_pretrained=draft_model,
        num_speculative_steps=depth,
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    if warm_start is not None:
        adapter._resolved_warm_start = warm_start
    del tmp_path
    return adapter


def _speculators_config(depth: int, *, methods: int = 1) -> dict[str, object]:
    """A stock-shaped Speculators EAGLE-3 draft config declaring *depth*."""
    return {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 8,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": depth,
                    "verifier_accept_k": 1,
                }
                for _ in range(methods)
            ],
            "verifier": {"architectures": [], "name_or_path": "acme/verifier"},
        },
    }


def _write_config(directory: Path, payload: object) -> Path:
    path = directory / "config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --- the fingerprint itself -------------------------------------------------


def test_the_fingerprint_covers_tensor_bytes(tmp_path: Path) -> None:
    same = write_draft_weights(tmp_path / "a", seed=1).parent
    other = write_draft_weights(tmp_path / "b", seed=2).parent
    copy = write_draft_weights(tmp_path / "c", seed=1).parent

    assert weight_fingerprint(same) == weight_fingerprint(copy)
    assert weight_fingerprint(same) != weight_fingerprint(other)


def test_the_fingerprint_ignores_shard_layout_and_sidecar_files(tmp_path: Path) -> None:
    """Two directories holding the same weights must fingerprint alike.

    Otherwise the comparison against the warm-start head would report a
    difference every time repackaging moved a tensor between shards, and the
    no-op check would be useless.
    """
    payloads = draft_payloads(seed=3)
    single = tmp_path / "single"
    single.mkdir()
    (single / "model.safetensors").write_bytes(safetensors_bytes(payloads))
    (single / "config.json").write_text("{}", encoding="utf-8")

    names = sorted(payloads)
    split = tmp_path / "split"
    split.mkdir()
    (split / "model-00001.safetensors").write_bytes(
        safetensors_bytes({name: payloads[name] for name in names[:4]})
    )
    (split / "model-00002.safetensors").write_bytes(
        safetensors_bytes({name: payloads[name] for name in names[4:]})
    )
    (split / "config.json").write_text(
        json.dumps({"speculators_model_type": "eagle3"}), encoding="utf-8"
    )

    assert weight_fingerprint(single) == weight_fingerprint(split)


def test_a_tensor_duplicated_across_shards_is_ambiguous_not_averaged(
    tmp_path: Path,
) -> None:
    payloads = draft_payloads(seed=4)
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "a.safetensors").write_bytes(safetensors_bytes(payloads))
    (draft / "b.safetensors").write_bytes(
        safetensors_bytes({"fc.weight": payloads["fc.weight"]})
    )

    with pytest.raises(DraftWeightsError, match="ambiguous"):
        weight_fingerprint(draft)


def test_a_truncated_container_is_a_loud_failure(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "model.safetensors").write_bytes(b"")

    with pytest.raises(DraftWeightsError, match="truncated"):
        weight_fingerprint(draft)


def test_a_directory_with_no_weights_cannot_be_fingerprinted(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DraftWeightsError, match="no safetensors"):
        weight_fingerprint(draft)


# --- the required tensor set ------------------------------------------------


def test_the_required_set_is_what_the_cached_stock_drafters_publish() -> None:
    """Read off the checkpoints, not assumed from the paper.

    ``input_norm.weight`` is deliberately absent: gpt-oss-20b carries it
    because its config sets ``norm_before_fc``, Qwen3-8B does not, and
    requiring it would reject a valid Qwen3 head.
    """
    assert "fc.weight" in REQUIRED_DRAFT_TENSORS
    assert {"d2t", "t2d"} <= REQUIRED_DRAFT_TENSORS
    assert "input_norm.weight" not in REQUIRED_DRAFT_TENSORS
    assert len(REQUIRED_DRAFT_TENSORS) == 16


def test_vocab_maps_alone_are_not_a_trained_head(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    write_draft_weights(draft, names=("d2t", "t2d"))
    adapter = _adapter(tmp_path, draft_model="acme/stock")

    assert draft_tensor_keys(draft) == frozenset({"d2t", "t2d"})
    with pytest.raises(DraftWeightsError, match="fc.weight"):
        adapter._record_draft_weights(draft)


# --- the no-op cycle --------------------------------------------------------


def test_a_candidate_identical_to_stock_is_detected_not_benchmarked(
    tmp_path: Path,
) -> None:
    """A byte-identical candidate means training changed nothing.

    Publishing it would spend a whole gate cycle establishing that a head is
    as good as itself, and the result would look like an ordinary rejection
    rather than a wasted cycle.
    """
    stock = tmp_path / "stock"
    write_draft_weights(stock, seed=7)
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=7)
    adapter = _adapter(tmp_path, draft_model=str(stock))

    with pytest.raises(StockIdenticalDraftError) as raised:
        adapter._record_draft_weights(draft)

    assert raised.value.baseline == str(stock)
    assert raised.value.fingerprint == weight_fingerprint(stock)


def test_a_candidate_identical_to_the_incumbent_is_detected_too(
    tmp_path: Path,
) -> None:
    """Compounding moves the baseline, so the check has to move with it."""
    stock = tmp_path / "stock"
    write_draft_weights(stock, seed=1)
    incumbent = tmp_path / "incumbent"
    write_draft_weights(incumbent, seed=2)
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=2)
    adapter = _adapter(tmp_path, draft_model=str(stock), warm_start=str(incumbent))

    with pytest.raises(StockIdenticalDraftError, match="incumbent"):
        adapter._record_draft_weights(draft)


def test_a_genuinely_trained_candidate_records_both_fingerprints(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock"
    write_draft_weights(stock, seed=1)
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=2)
    adapter = _adapter(tmp_path, draft_model=str(stock))

    adapter._record_draft_weights(draft)
    params = adapter.describe().training_params

    assert params["draft_weight_fingerprint"] == weight_fingerprint(draft)
    assert params["draft_weight_baseline"] == str(stock)
    assert params["draft_weight_baseline_fingerprint"] == weight_fingerprint(stock)


def test_an_unresolvable_baseline_records_nulls_rather_than_assuming(
    tmp_path: Path,
) -> None:
    """The manifest must be able to say the no-op check did not run.

    Same reasoning as ``verifier_revision_satisfied``: a field that is merely
    absent cannot be told apart from one that was never consulted.
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=5)
    adapter = _adapter(tmp_path, draft_model="acme/not-in-the-cache")

    adapter._record_draft_weights(draft)
    params = adapter.describe().training_params

    assert params["draft_weight_fingerprint"] == weight_fingerprint(draft)
    assert params["draft_weight_baseline"] == "acme/not-in-the-cache"
    assert params["draft_weight_baseline_fingerprint"] is None


# --- publication ------------------------------------------------------------


def test_publishing_a_different_tree_than_was_trained_fails(tmp_path: Path) -> None:
    stock = tmp_path / "stock"
    write_draft_weights(stock, seed=1)
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=2)
    _write_config(draft, _speculators_config(3))
    published = tmp_path / "published"
    write_draft_weights(published, seed=3)
    _write_config(published, _speculators_config(3))
    adapter = _adapter(tmp_path, draft_model=str(stock))
    adapter._record_draft_weights(draft)

    adapter.assert_published_weights(draft)
    with pytest.raises(DraftWeightsError, match="do not match"):
        adapter.assert_published_weights(published)


def test_publishing_without_a_materialized_fingerprint_fails_closed(
    tmp_path: Path,
) -> None:
    published = tmp_path / "published"
    write_draft_weights(published, seed=3)
    adapter = _adapter(tmp_path, draft_model="acme/stock")

    with pytest.raises(DraftWeightsError, match="no materialized draft fingerprint"):
        adapter.assert_published_weights(published)


# --- the declared depth -----------------------------------------------------
#
# Speculators never rewrites
# ``speculators_config.proposal_methods[*].speculative_tokens`` on the
# ``--from-pretrained`` path, so a head warm-started from a vendor drafter
# inherits that drafter's declaration.  Run d993eee-gptoss-idle published a
# head trained 5-deep (manifest ``num_speculative_steps: 5``) whose own
# config.json declared 3, inherited from
# ``RedHatAI/gpt-oss-20b-speculator.eagle3``.
#
# vLLM does not read that field for a drafter passed as
# ``--speculative-config.model`` -- the only reader,
# ``SpeculatorsConfig.build_vllm_speculative_config``, is reached solely from
# ``maybe_override_with_speculators``, which runs against the top-level
# ``--model`` (SpeedLM's verifier, which has no ``speculators_config``).  So
# this is a provenance defect, not the tail-collapse cause; it is repaired
# because SpeedLM's own ``drafter_declared_speculative_tokens`` reads it, and
# so does every external consumer.


def test_a_stale_declaration_is_rewritten_to_the_trained_depth(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, _speculators_config(3))

    assert declare_speculative_tokens(draft, 5) is True
    assert declared_speculative_tokens(draft) == 5


def test_the_rewrite_preserves_every_other_config_field(tmp_path: Path) -> None:
    """A depth repair must not become a config rewrite.

    ``transformer_layer_config`` and ``eagle_aux_hidden_state_layer_ids`` are
    what vLLM actually builds the draft model from; losing either would turn a
    provenance fix into a load failure.
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    payload = _speculators_config(3)
    payload["transformer_layer_config"] = {"model_type": "qwen3", "hidden_size": 8}
    payload["eagle_aux_hidden_state_layer_ids"] = [1, 2, 3]
    _write_config(draft, payload)

    declare_speculative_tokens(draft, 5)
    rewritten = json.loads((draft / "config.json").read_text(encoding="utf-8"))

    assert declared_speculative_tokens(draft) == 5
    assert rewritten["transformer_layer_config"] == payload["transformer_layer_config"]
    assert rewritten["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3]
    assert rewritten["speculators_config"]["verifier"]["name_or_path"] == "acme/verifier"
    assert rewritten["speculators_config"]["proposal_methods"][0]["proposal_type"] == "greedy"


def test_every_proposal_method_is_rewritten_not_only_the_first(tmp_path: Path) -> None:
    """``drafter_declared_speculative_tokens`` returns None on disagreement.

    Rewriting only ``proposal_methods[0]`` -- which is the one field vLLM
    would read -- would leave the config *ambiguous*, which SpeedLM's own
    reader treats as no declaration at all.
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, _speculators_config(3, methods=2))

    declare_speculative_tokens(draft, 5)

    assert declared_speculative_tokens(draft) == 5


def test_an_already_correct_declaration_is_left_untouched(tmp_path: Path) -> None:
    """A no-change rewrite must not perturb the tree.

    ``ArtifactRegistry`` is content-addressed, so an unconditional rewrite
    that reordered or reformatted the config would change the artifact id of
    an otherwise identical publication.
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, _speculators_config(5))
    before = (draft / "config.json").read_bytes()

    assert declare_speculative_tokens(draft, 5) is False
    assert (draft / "config.json").read_bytes() == before


def test_a_draft_that_declares_nothing_is_a_loud_failure(tmp_path: Path) -> None:
    """Silence is how the stale value survived; it must not be a no-op."""
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, {"speculators_model_type": "eagle3"})

    with pytest.raises(DeclaredDepthError, match="no speculators_config"):
        declare_speculative_tokens(draft, 5)


def test_a_draft_with_no_proposal_methods_is_a_loud_failure(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, {"speculators_config": {"proposal_methods": []}})

    with pytest.raises(DeclaredDepthError, match="no proposal_methods"):
        declare_speculative_tokens(draft, 5)


def test_the_rewrite_reseals_the_read_only_draft_tree(tmp_path: Path) -> None:
    """The materializer hands over a 0o444/0o555 tree.

    The rewrite has to unseal it and put the seal back; a draft left writable
    would be published writable.
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    config = _write_config(draft, _speculators_config(3))
    config.chmod(0o444)
    draft.chmod(0o555)

    assert declare_speculative_tokens(draft, 5) is True

    assert declared_speculative_tokens(draft) == 5
    assert config.stat().st_mode & 0o777 == 0o444
    assert draft.stat().st_mode & 0o777 == 0o555


def test_the_rewrite_moves_the_artifact_id_but_not_the_weight_fingerprint(
    tmp_path: Path,
) -> None:
    """The two digests answer different questions, and must keep doing so.

    ``weight_fingerprint`` covers tensors only -- explicitly not config or
    tokenizer files -- so the publish-time weight assertion still compares
    like with like across the rewrite.  ``hash_directory`` covers every byte,
    so the artifact id *does* move, which is why the rewrite has to happen
    before ``ArtifactRegistry.publish`` takes its first hash rather than
    inside the publish, where it would read as "artifact changed while
    publishing".
    """
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=1)
    _write_config(draft, _speculators_config(3))
    fingerprint_before = weight_fingerprint(draft)
    tree_before = hash_directory(draft)

    declare_speculative_tokens(draft, 5)

    assert weight_fingerprint(draft) == fingerprint_before
    assert hash_directory(draft) != tree_before


def test_materialize_makes_the_draft_declare_the_depth_it_trained_at(
    tmp_path: Path,
) -> None:
    """The wiring, not just the function.

    Every prior defect on this path was of the form "the step exists and runs
    on something else", so the rewrite is asserted through the real
    ``materialize`` rather than only at its own call signature.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    checkpoint = tmp_path / "checkpoint_best"
    write_draft_weights(checkpoint, seed=9)
    _write_config(checkpoint, _speculators_config(3))

    class _Materializer:
        def materialize(
            self,
            source: Path,
            destination: Path,
            *,
            timeout_seconds: float,
            should_abort: object,
        ) -> Path:
            shutil.copytree(source, destination)
            return destination

    adapter = _adapter(tmp_path, draft_model="acme/stock", depth=5)
    adapter._materializer = _Materializer()
    adapter._clock = lambda: 0.0

    draft = adapter.materialize(
        SimpleNamespace(checkpoint_best=checkpoint),
        work_dir,
        should_abort=lambda: False,
    )

    assert declared_speculative_tokens(draft) == 5
    assert adapter.describe().training_params["num_speculative_steps"] == 5


def test_publishing_an_artifact_that_misdeclares_its_depth_fails(
    tmp_path: Path,
) -> None:
    """The regression guard: a 5-deep head declaring 3 must not publish."""
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=2)
    _write_config(draft, _speculators_config(3))
    adapter = _adapter(tmp_path, draft_model="acme/stock", depth=5)
    adapter._record_draft_weights(draft)

    with pytest.raises(DeclaredDepthError, match="declares speculative_tokens=3"):
        adapter.assert_published_weights(draft)


def test_publishing_an_artifact_whose_config_vanished_fails(tmp_path: Path) -> None:
    """An unreadable declaration is not a passing one."""
    draft = tmp_path / "draft"
    write_draft_weights(draft, seed=2)
    adapter = _adapter(tmp_path, draft_model="acme/stock", depth=3)
    adapter._record_draft_weights(draft)

    with pytest.raises(DeclaredDepthError, match="declares speculative_tokens=None"):
        adapter.assert_published_weights(draft)


def test_the_registry_publishes_a_self_describing_artifact_end_to_end(
    tmp_path: Path,
) -> None:
    """Both publish-time checks have to hold at once.

    The weight assertion and the depth assertion run from the same
    ``before_publish`` hook, against the staged tree, after the registry has
    already hashed the source and re-hashed the copy.  This proves the depth
    rewrite does not trip that equality check -- because it happened upstream,
    at materialization -- and that the guard passes on a correct artifact.
    """
    source = tmp_path / "source"
    write_draft_weights(source, seed=4)
    _write_config(source, _speculators_config(3))
    adapter = _adapter(tmp_path, draft_model="acme/stock", depth=5)
    adapter._record_draft_weights(source)
    declare_speculative_tokens(source, 5)

    registry = ArtifactRegistry(
        tmp_path / "registry",
        clock=lambda: 1.0,
        before_publish=adapter.assert_published_weights,
    )
    artifact = registry.publish(
        source,
        ArtifactSpec(
            verifier_model="acme/verifier",
            draft_model="acme/stock",
            base_draft="acme/stock",
            trace_hash="t0",
            training_params={},
        ),
    )

    assert declared_speculative_tokens(artifact.path) == 5
    assert weight_fingerprint(artifact.path) == adapter._draft_weight_fingerprint
