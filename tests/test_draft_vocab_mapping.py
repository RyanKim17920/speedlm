"""The reduced-vocabulary index maps must be provably applicable.

EAGLE-3 draft heads here run ~64k of the verifier's ~201k tokens and carry two
index maps: ``d2t`` (a per-draft-id *delta* to the target id) and ``t2d`` (a
bool mask over the verifier's vocabulary).  Every check this repo shipped for
them was presence-only -- ``"d2t" in tensor_keys`` -- and zeroing every entry
of ``d2t`` left the whole suite green.

That matters because vLLM will not catch it either.  It applies the map as
``base + d2t`` with no clamp and no assert (``llama_eagle3.py:347-357``), and
constructs ``draft_id_to_target_id`` as an all-zeros parameter that stays
all-zeros when the checkpoint carries no ``d2t`` (``:304-307``, ``:419-420``).
So the two worst states -- a map that points outside the verifier's vocabulary
and a map that is the identity -- both serve silently.

These tests pin the contents, not the presence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from draft_weights import (
    FIXTURE_DRAFT_VOCAB,
    FIXTURE_VERIFIER_VOCAB,
    d2t_bytes,
    fixture_d2t,
    fixture_t2d,
    t2d_bytes,
    write_draft_config,
    write_typed_draft,
    write_verifier_config,
)

from speedlm.tuner.eagle3 import (
    DraftVocabMappingError,
    DraftWeightsError,
    assert_draft_vocab_mapping,
    check_vocab_mapping,
    verifier_vocab_size,
)


def _write_maps(
    directory: Path,
    *,
    d2t: list[int] | None = None,
    t2d: list[int] | None = None,
    include_d2t: bool = True,
    include_t2d: bool = True,
) -> Path:
    """Write a draft directory carrying only the two index maps."""
    d2t = fixture_d2t() if d2t is None else d2t
    t2d = fixture_t2d() if t2d is None else t2d
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    if include_d2t:
        tensors["d2t"] = ("I64", (len(d2t),), d2t_bytes(d2t))
    if include_t2d:
        tensors["t2d"] = ("BOOL", (len(t2d),), t2d_bytes(t2d))
    write_typed_draft(directory, tensors)
    write_draft_config(directory)
    return directory


def _verifier(tmp_path: Path, *, vocab_size: int = FIXTURE_VERIFIER_VOCAB) -> str:
    directory = tmp_path / "verifier"
    write_verifier_config(directory, vocab_size=vocab_size)
    return str(directory)


# ---------------------------------------------------------------------------
# The rule itself, over plain integers
# ---------------------------------------------------------------------------


class TestCheckVocabMapping:
    def test_a_well_formed_reduction_passes_and_reports_what_it_read(self) -> None:
        report = check_vocab_mapping(
            fixture_d2t(),
            fixture_t2d(),
            verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
            source="fixture",
        )
        assert report == {
            "draft_vocab_size": FIXTURE_DRAFT_VOCAB,
            "verifier_vocab_size": FIXTURE_VERIFIER_VOCAB,
            "min_target": 0,
            "max_target": (FIXTURE_DRAFT_VOCAB - 1) * 2,
        }

    def test_a_negative_target_is_rejected(self) -> None:
        """The silent case: PyTorch advanced indexing wraps a negative index.

        Verified directly against torch: with ``vocab_size=32`` and a target
        of ``-7``, ``logits_new[:, targets] = logits`` writes column 25 and
        raises nothing.  Nothing downstream of this check would notice.
        """
        d2t = fixture_d2t()
        d2t[3] = -10  # draft id 3 -> target -7
        with pytest.raises(DraftVocabMappingError, match=r"d2t\[3\]=-10 -> -7"):
            check_vocab_mapping(
                d2t,
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_a_target_past_the_end_of_the_verifier_vocab_is_rejected(self) -> None:
        d2t = fixture_d2t()
        d2t[5] = FIXTURE_VERIFIER_VOCAB  # draft id 5 -> well past the end
        with pytest.raises(DraftVocabMappingError, match="map outside"):
            check_vocab_mapping(
                d2t,
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_the_message_counts_every_offender_and_quotes_only_a_few(self) -> None:
        """A whole-map shift must not produce a 64,000-entry exception."""
        d2t = [-(index + 1) for index in range(FIXTURE_DRAFT_VOCAB)]
        with pytest.raises(DraftVocabMappingError) as caught:
            check_vocab_mapping(
                d2t,
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )
        message = str(caught.value)
        assert f"{FIXTURE_DRAFT_VOCAB} of {FIXTURE_DRAFT_VOCAB} d2t entries" in message
        assert message.count(" -> ") == 5
        assert ", ..." in message

    def test_an_all_zeros_identity_map_on_a_reduced_vocab_is_rejected(self) -> None:
        """The state vLLM lands in when ``d2t`` is missing from the weights.

        It is in range, self-consistent and collision-free -- every structural
        check passes.  It is still a corruption: draft id ``i`` is not target
        id ``i`` on a head whose head rows were fitted to a 64k selection.
        """
        with pytest.raises(DraftVocabMappingError, match="identity map"):
            check_vocab_mapping(
                [0] * FIXTURE_DRAFT_VOCAB,
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_an_identity_map_on_a_full_vocab_head_is_legal(self) -> None:
        """Not every all-zeros ``d2t`` is a defect -- only a reduced one is.

        A drafter sharing the verifier's vocabulary has nothing to translate,
        and vLLM's own default (``draft_vocab_size`` falls back to
        ``vocab_size``, ``llama_eagle3.py:278-280``) produces exactly this.
        Rejecting it would be a false positive, so the reduction is what the
        identity rule is conditioned on.
        """
        report = check_vocab_mapping(
            [0] * FIXTURE_VERIFIER_VOCAB,
            None,
            verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
            source="fixture",
        )
        assert report["draft_vocab_size"] == FIXTURE_VERIFIER_VOCAB

    def test_two_draft_ids_colliding_on_one_target_are_rejected(self) -> None:
        """``logits_new[:, targets] = logits`` keeps only the last write."""
        d2t = fixture_d2t()
        d2t[4] = 2  # 4 + 2 == 6, the target draft id 3 already claims
        with pytest.raises(DraftVocabMappingError, match="collide on a target id"):
            check_vocab_mapping(
                d2t,
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_a_draft_vocab_larger_than_the_verifiers_is_rejected(self) -> None:
        with pytest.raises(DraftVocabMappingError, match="larger than the verifier"):
            check_vocab_mapping(
                fixture_d2t(draft_vocab=FIXTURE_VERIFIER_VOCAB + 1, stride=1),
                None,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_an_empty_d2t_is_rejected(self) -> None:
        with pytest.raises(DraftVocabMappingError, match="d2t is empty"):
            check_vocab_mapping(
                [], None, verifier_vocab_size=FIXTURE_VERIFIER_VOCAB, source="fixture"
            )


# ---------------------------------------------------------------------------
# t2d, the independent witness
# ---------------------------------------------------------------------------


class TestT2dCrossCheck:
    def test_an_all_false_t2d_is_rejected(self) -> None:
        """No draftable token at all -- the head would have no vocabulary."""
        with pytest.raises(DraftVocabMappingError, match="no target token"):
            check_vocab_mapping(
                fixture_d2t(),
                [0] * FIXTURE_VERIFIER_VOCAB,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_a_t2d_of_the_wrong_length_is_rejected(self) -> None:
        with pytest.raises(DraftVocabMappingError, match="cannot be a mask over it"):
            check_vocab_mapping(
                fixture_d2t(),
                fixture_t2d()[:-1],
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_a_popcount_disagreeing_with_d2t_is_rejected(self) -> None:
        """Both maps are built from one selected-id list; a popcount mismatch
        means one of them was regenerated without the other."""
        t2d = fixture_t2d()
        t2d[1] = 1  # one extra draftable id that d2t does not reach
        with pytest.raises(DraftVocabMappingError, match="not built from"):
            check_vocab_mapping(
                fixture_d2t(),
                t2d,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )

    def test_a_target_d2t_reaches_but_t2d_denies_is_rejected(self) -> None:
        """Popcount-preserving disagreement -- the case a count cannot catch."""
        t2d = fixture_t2d()
        t2d[0] = 0  # d2t[0] -> 0 is no longer marked draftable
        t2d[1] = 1  # ...but the popcount is unchanged
        with pytest.raises(DraftVocabMappingError, match="not\n?\\s*marked draftable"):
            check_vocab_mapping(
                fixture_d2t(),
                t2d,
                verifier_vocab_size=FIXTURE_VERIFIER_VOCAB,
                source="fixture",
            )


# ---------------------------------------------------------------------------
# Reading the verifier's vocabulary from a runtime fact
# ---------------------------------------------------------------------------


class TestVerifierVocabSize:
    def test_it_is_read_from_the_verifiers_own_config(self, tmp_path: Path) -> None:
        assert verifier_vocab_size(_verifier(tmp_path, vocab_size=201_088)) == 201_088

    def test_a_multimodal_verifier_is_unwrapped_to_its_text_config(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "verifier"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"vocab_size": 7, "text_config": {"vocab_size": 151_936}}),
            encoding="utf-8",
        )
        assert verifier_vocab_size(str(directory)) == 151_936

    def test_an_unresolvable_verifier_is_a_failure_not_a_skip(self) -> None:
        """The check exists *because* the verifier can change underneath a map
        that is never rebuilt; not knowing which verifier is the alarm."""
        with pytest.raises(DraftVocabMappingError, match="no local snapshot"):
            verifier_vocab_size("acme/nowhere")

    def test_a_config_without_a_vocab_size_is_a_failure(self, tmp_path: Path) -> None:
        directory = tmp_path / "verifier"
        directory.mkdir()
        (directory / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(DraftVocabMappingError, match="not a\\s*vocabulary size"):
            verifier_vocab_size(str(directory))


# ---------------------------------------------------------------------------
# The publish-side entry point, over real safetensors
# ---------------------------------------------------------------------------


class TestAssertDraftVocabMapping:
    def test_a_well_formed_artifact_passes(self, tmp_path: Path) -> None:
        draft = _write_maps(tmp_path / "draft")
        report = assert_draft_vocab_mapping(draft, _verifier(tmp_path))
        assert report["draft_vocab_size"] == FIXTURE_DRAFT_VOCAB
        assert report["verifier_vocab_size"] == FIXTURE_VERIFIER_VOCAB

    def test_an_out_of_range_map_on_disk_is_rejected(self, tmp_path: Path) -> None:
        d2t = fixture_d2t()
        d2t[2] = -5
        draft = _write_maps(tmp_path / "draft", d2t=d2t, include_t2d=False)
        with pytest.raises(DraftVocabMappingError, match="map outside"):
            assert_draft_vocab_mapping(draft, _verifier(tmp_path))

    def test_an_all_zeros_map_on_disk_is_rejected(self, tmp_path: Path) -> None:
        """Zeroing every ``d2t`` entry used to leave the whole suite green."""
        draft = _write_maps(
            tmp_path / "draft", d2t=[0] * FIXTURE_DRAFT_VOCAB, include_t2d=False
        )
        with pytest.raises(DraftVocabMappingError, match="identity map"):
            assert_draft_vocab_mapping(draft, _verifier(tmp_path))

    def test_a_missing_d2t_is_rejected(self, tmp_path: Path) -> None:
        """vLLM would fall back to the all-zeros identity with no error."""
        draft = _write_maps(tmp_path / "draft", include_d2t=False)
        with pytest.raises(DraftVocabMappingError, match="publishes no d2t"):
            assert_draft_vocab_mapping(draft, _verifier(tmp_path))

    def test_a_float_d2t_is_rejected_rather_than_reinterpreted(
        self, tmp_path: Path
    ) -> None:
        """Decoding an ``F32`` payload as ``I64`` would invent index values."""
        write_typed_draft(
            tmp_path / "draft",
            {"d2t": ("F32", (2,), b"\0" * 8)},
        )
        write_draft_config(tmp_path / "draft")
        with pytest.raises(DraftVocabMappingError, match="not an\\s*integer index map"):
            assert_draft_vocab_mapping(tmp_path / "draft", _verifier(tmp_path))

    def test_a_config_declaring_a_different_draft_vocab_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """The artifact must not misdescribe its own vocabulary."""
        draft = _write_maps(
            tmp_path / "draft",
            d2t=fixture_d2t(draft_vocab=4),
            t2d=fixture_t2d(draft_vocab=4),
        )
        # ``speculators_config_payload`` declares draft_vocab_size=8.
        with pytest.raises(DraftVocabMappingError, match="misdescribes its own"):
            assert_draft_vocab_mapping(draft, _verifier(tmp_path))

    def test_the_error_is_a_draft_weights_error(self, tmp_path: Path) -> None:
        """So existing ``except DraftWeightsError`` handlers keep working."""
        assert issubclass(DraftVocabMappingError, DraftWeightsError)
        draft = _write_maps(
            tmp_path / "draft", d2t=[0] * FIXTURE_DRAFT_VOCAB, include_t2d=False
        )
        with pytest.raises(DraftWeightsError):
            assert_draft_vocab_mapping(draft, _verifier(tmp_path))


# ---------------------------------------------------------------------------
# The publish hook
# ---------------------------------------------------------------------------


class TestPublishHook:
    def test_publishing_an_identity_mapped_draft_fails(self, tmp_path: Path) -> None:
        """End to end through ``assert_published_weights``.

        This is the gate that matters: the fingerprint check above it would
        happily certify a corrupted map, because the map's *bytes* are exactly
        the bytes that were trained.
        """
        from draft_weights import draft_payloads, safetensors_bytes

        from speedlm.tuner.eagle3 import weight_fingerprint

        published = tmp_path / "published"
        published.mkdir()
        payloads = draft_payloads()
        payloads["d2t"] = d2t_bytes([0] * FIXTURE_DRAFT_VOCAB)
        (published / "model.safetensors").write_bytes(safetensors_bytes(payloads))
        write_draft_config(published)

        adapter = _publish_adapter(tmp_path, fingerprint=weight_fingerprint(published))
        with pytest.raises(DraftVocabMappingError, match="identity map"):
            adapter.assert_published_weights(published)

    def test_a_valid_publish_records_what_the_check_read(self, tmp_path: Path) -> None:
        """A guard whose only output is an exception cannot be told apart from
        a guard that never ran, so the passing path leaves evidence too."""
        from draft_weights import draft_payloads, safetensors_bytes

        from speedlm.tuner.eagle3 import weight_fingerprint

        published = tmp_path / "published"
        published.mkdir()
        (published / "model.safetensors").write_bytes(
            safetensors_bytes(draft_payloads())
        )
        write_draft_config(published)

        adapter = _publish_adapter(tmp_path, fingerprint=weight_fingerprint(published))
        adapter.assert_published_weights(published)
        assert adapter._draft_vocab_mapping == {
            "draft_vocab_size": FIXTURE_DRAFT_VOCAB,
            "verifier_vocab_size": FIXTURE_VERIFIER_VOCAB,
            "min_target": 0,
            "max_target": (FIXTURE_DRAFT_VOCAB - 1) * 2,
        }


def _publish_adapter(tmp_path: Path, *, fingerprint: str):
    from speedlm.training.masking import MaskPolicy
    from speedlm.tuner.eagle3 import Eagle3Adapter, Eagle3Config

    adapter = object.__new__(Eagle3Adapter)
    adapter.config = Eagle3Config(
        verifier_model=_verifier(tmp_path),
        draft_model="acme/stock",
        from_pretrained="acme/stock",
        num_speculative_steps=3,
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    adapter._draft_weight_fingerprint = fingerprint
    return adapter
