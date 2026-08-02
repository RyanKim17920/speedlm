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
from pathlib import Path

import pytest
from draft_weights import draft_payloads, safetensors_bytes, write_draft_weights

from speedlm.training.masking import MaskPolicy
from speedlm.tuner.eagle3 import (
    REQUIRED_DRAFT_TENSORS,
    DraftWeightsError,
    Eagle3Adapter,
    Eagle3Config,
    StockIdenticalDraftError,
    draft_tensor_keys,
    weight_fingerprint,
)


def _adapter(tmp_path: Path, *, draft_model: str, warm_start: str | None = None) -> Eagle3Adapter:
    adapter = object.__new__(Eagle3Adapter)
    adapter.config = Eagle3Config(
        verifier_model="acme/verifier",
        draft_model=draft_model,
        from_pretrained=draft_model,
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    if warm_start is not None:
        adapter._resolved_warm_start = warm_start
    del tmp_path
    return adapter


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
    published = tmp_path / "published"
    write_draft_weights(published, seed=3)
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
