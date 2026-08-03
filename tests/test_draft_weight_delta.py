"""Byte-identity was never enough: a re-rounded head is still a no-op.

Three real cycles (``gptoss-pinned-2``, ``full-gptoss-head``, ``qwen-pinned-2``)
produced draft heads whose every tensor differed from the warm-start snapshot
in the *bytes* -- so ``StockIdenticalDraftError`` stayed silent and the
fingerprints differed -- while differing in *value* by at most one bf16 ULP.
Measured per tensor as ``||cand - base||_F / ||base||_F``, the largest movement
across all three runs was 0.006090, inside the ``2^-7 = 0.0078125`` envelope
that one-ULP re-rounding cannot exceed, with ~8 % of elements changed and every
norm-type tensor at 0 or ~1e-5.

So the fixtures here are built as real bf16 bit patterns and mutated by
*incrementing the pattern* -- which is precisely "one ULP" -- rather than by
perturbing Python floats, because the thing under test is whether the guard can
tell dither from learning at exactly that granularity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import pytest
from draft_weights import DTYPE_ITEMSIZE, write_typed_draft

from speedlm.training.masking import MaskPolicy
from speedlm.tuner.eagle3 import (
    BF16_RELATIVE_ULP,
    MIN_TRAINED_RELATIVE_DELTA,
    NORM_MOVED_EPSILON,
    REQUIRED_DRAFT_TENSORS,
    AuxLayerCountMismatch,
    DraftWeightsError,
    Eagle3Adapter,
    Eagle3Config,
    NoOpTrainingDeltaError,
    _bf16_to_floats,
    drafter_aux_count,
    is_noop_training_delta,
    weight_delta_report,
)

#: Hidden size of the synthetic head.  Small enough to stay a millisecond
#: fixture, large enough that ``fc.weight`` is genuinely ``(h, aux * h)``.
_HIDDEN = 32
_AUX = 3

#: Norm vectors are deliberately long.  ``qwen-pinned-2`` moved exactly *one*
#: element of ``layers.0.input_layernorm.weight`` (4,096 wide) for a relative
#: delta of 1.0e-5, and reproducing "a single element moved" is the whole
#: reason :data:`NORM_MOVED_EPSILON` is an epsilon rather than an equality.
_NORM_ELEMENTS = 8192

#: bf16 bit patterns for the binade ``[1.0, 2.0)``: sign 0, exponent 127,
#: mantissa 0..127.  One ULP here is ``2^-7`` absolute, so incrementing a
#: pattern by one is exactly the dither the failing runs exhibited.
_BF16_ONE = 0x3F80

_NORM_NAMES = tuple(sorted(name for name in REQUIRED_DRAFT_TENSORS if "norm" in name))
_INDEX_NAMES = ("d2t", "t2d")
_PROJECTION_NAMES = tuple(
    sorted(set(REQUIRED_DRAFT_TENSORS) - set(_NORM_NAMES) - set(_INDEX_NAMES))
)

#: name -> (dtype, shape, values), where values are raw dtype-native integers.
Head = dict[str, tuple[str, tuple[int, ...], list[int]]]


def _patterns(count: int, seed: int) -> list[int]:
    """Deterministic bf16 patterns in ``[1.0, 2.0)``."""
    state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    values: list[int] = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        values.append(_BF16_ONE + ((state >> 16) % 128))
    return values


def _baseline_head(*, hidden: int = _HIDDEN, aux: int = _AUX, seed: int = 11) -> Head:
    head: Head = {}
    for index, name in enumerate(_PROJECTION_NAMES):
        shape = (hidden, aux * hidden) if name == "fc.weight" else (hidden, hidden)
        head[name] = ("BF16", shape, _patterns(shape[0] * shape[1], seed + index))
    for index, name in enumerate(_NORM_NAMES):
        head[name] = ("BF16", (_NORM_ELEMENTS,), _patterns(_NORM_ELEMENTS, seed + 90 + index))
    for index, name in enumerate(_INDEX_NAMES):
        head[name] = ("I64", (8,), [index * 100 + position for position in range(8)])
    return head


def _dither(values: list[int]) -> list[int]:
    """Move ~8 % of *values* by one to three ULPs: the observed signature."""
    return [
        value + 1 + (position % 3) if (position * 2654435761) % 100 < 8 else value
        for position, value in enumerate(values)
    ]


def _shift(values: list[int], ulps: int) -> list[int]:
    return [value + ulps for value in values]


def _map_head(head: Head, names: tuple[str, ...], transform) -> Head:  # type: ignore[no-untyped-def]
    return {
        name: (dtype, shape, transform(values) if name in names else list(values))
        for name, (dtype, shape, values) in head.items()
    }


def _encode(head: Head) -> Mapping[str, tuple[str, tuple[int, ...], bytes]]:
    return {
        name: (
            dtype,
            shape,
            b"".join(
                int(value).to_bytes(DTYPE_ITEMSIZE[dtype], "little") for value in values
            ),
        )
        for name, (dtype, shape, values) in head.items()
    }


def _write(directory: Path, head: Head) -> Path:
    write_typed_draft(directory, _encode(head))
    return directory


def _adapter(
    *, draft_model: str, target_layer_ids: tuple[int, ...] | None = None
) -> Eagle3Adapter:
    adapter = object.__new__(Eagle3Adapter)
    adapter.config = Eagle3Config(
        verifier_model="acme/verifier",
        draft_model=draft_model,
        from_pretrained=draft_model,
        target_layer_ids=target_layer_ids,
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    return adapter


# --- the byte-level math ----------------------------------------------------


def test_bf16_widening_matches_known_bit_patterns() -> None:
    """Pin the widening itself, not just the guard built on it.

    A bf16 pattern is the top 16 bits of the float32 of equal value, so these
    are exact identities and not approximations.
    """
    raw = b"".join(
        pattern.to_bytes(2, "little")
        for pattern in (0x3F80, 0x4000, 0xBF80, 0x0000, 0x3F00, 0x4049)
    )

    assert _bf16_to_floats(raw) == (1.0, 2.0, -1.0, 0.0, 0.5, 3.140625)


def test_one_bf16_ulp_is_one_increment_of_the_bit_pattern() -> None:
    """The fixtures' notion of "one ULP" must be the format's."""
    (one,) = _bf16_to_floats((0x3F80).to_bytes(2, "little"))
    (next_up,) = _bf16_to_floats((0x3F81).to_bytes(2, "little"))

    assert next_up - one == BF16_RELATIVE_ULP == 2.0**-7


def test_the_floor_sits_above_every_measured_dither() -> None:
    """The threshold's derivation, asserted rather than left in a comment.

    Largest per-tensor delta actually observed on a no-op cycle: 0.006090
    (``qwen-pinned-2``, ``layers.0.self_attn.k_proj.weight``).  One-ULP
    re-rounding cannot exceed ``2^-7`` whatever fraction of elements moves.
    """
    assert MIN_TRAINED_RELATIVE_DELTA == 2.0 * BF16_RELATIVE_ULP == 0.015625
    assert MIN_TRAINED_RELATIVE_DELTA > BF16_RELATIVE_ULP > 0.006090
    # Every measured non-zero norm delta was 1.0e-5 .. 3.0e-5.
    assert NORM_MOVED_EPSILON > 3.0e-5


def test_the_same_values_in_f32_and_bf16_give_the_same_delta(tmp_path: Path) -> None:
    """The dtype decode is a re-encoding, not a different measurement."""
    base = _baseline_head(seed=3)
    candidate = _map_head(base, _PROJECTION_NAMES, _dither)

    _write(tmp_path / "bf16-base", base)
    _write(tmp_path / "bf16-cand", candidate)

    def widen(head: Head) -> Head:
        return {
            name: (
                ("F32", shape, [value << 16 for value in values])
                if dtype == "BF16"
                else (dtype, shape, list(values))
            )
            for name, (dtype, shape, values) in head.items()
        }

    _write(tmp_path / "f32-base", widen(base))
    _write(tmp_path / "f32-cand", widen(candidate))

    narrow = weight_delta_report(tmp_path / "bf16-cand", tmp_path / "bf16-base")
    wide = weight_delta_report(tmp_path / "f32-cand", tmp_path / "f32-base")

    assert narrow.deltas.keys() == wide.deltas.keys()
    for name, value in narrow.deltas.items():
        assert wide.deltas[name] == pytest.approx(value, rel=1e-12)


# --- the report -------------------------------------------------------------


def test_the_report_excludes_index_maps_and_says_so(tmp_path: Path) -> None:
    """``d2t``/``t2d`` are token indices; a Frobenius norm over them is noise."""
    base = _write(tmp_path / "base", _baseline_head())
    candidate = _write(tmp_path / "cand", _map_head(_baseline_head(), _PROJECTION_NAMES, _dither))

    report = weight_delta_report(candidate, base)

    assert set(report.deltas) == set(_PROJECTION_NAMES) | set(_NORM_NAMES)
    assert set(report.skipped) == set(_INDEX_NAMES)
    assert all("float" in reason for reason in report.skipped.values())


def test_tensors_that_changed_shape_or_dtype_are_reported_not_dropped(
    tmp_path: Path,
) -> None:
    base_head = _baseline_head()
    candidate_head = _baseline_head()
    candidate_head["norm.weight"] = ("BF16", (16,), _patterns(16, 5))
    candidate_head["lm_head.weight"] = (
        "F32",
        (_HIDDEN, _HIDDEN),
        [value << 16 for value in base_head["lm_head.weight"][2]],
    )
    del candidate_head["layers.0.mlp.up_proj.weight"]

    report = weight_delta_report(
        _write(tmp_path / "cand", candidate_head), _write(tmp_path / "base", base_head)
    )

    assert "shape" in report.skipped["norm.weight"]
    assert "dtype" in report.skipped["lm_head.weight"]
    assert report.skipped["layers.0.mlp.up_proj.weight"] == "absent from the candidate"
    assert "norm.weight" not in report.deltas


def test_the_dither_fixture_reproduces_the_measured_signature(tmp_path: Path) -> None:
    """Guard the guard: if the fixture stops being dither, so does the test.

    Real runs: ~8 % of elements moved, max relative delta 0.005818 / 0.005810 /
    0.006090, i.e. inside the ``2^-7`` envelope, with norms at 0 or ~1e-5.
    """
    report = weight_delta_report(
        _write(tmp_path / "cand", _map_head(_baseline_head(), _PROJECTION_NAMES, _dither)),
        _write(tmp_path / "base", _baseline_head()),
    )

    assert 0.0 < report.max_delta < BF16_RELATIVE_ULP
    assert all(report.deltas[name] > 0.0 for name in _PROJECTION_NAMES)
    assert all(report.deltas[name] == 0.0 for name in _NORM_NAMES)


# --- the guard --------------------------------------------------------------


def _noop_candidate() -> Head:
    """Projections dithered by 1-3 ULPs; one norm element moved by one ULP."""
    head = _map_head(_baseline_head(), _PROJECTION_NAMES, _dither)
    dtype, shape, values = head["layers.0.input_layernorm.weight"]
    moved = list(values)
    moved[0] += 1
    head["layers.0.input_layernorm.weight"] = (dtype, shape, moved)
    return head


def test_a_one_ulp_re_rounding_is_caught_as_a_no_op(tmp_path: Path) -> None:
    base = _write(tmp_path / "stock", _baseline_head())
    candidate = _write(tmp_path / "draft", _noop_candidate())
    adapter = _adapter(draft_model=str(base))

    with pytest.raises(NoOpTrainingDeltaError) as raised:
        adapter._record_draft_weights(candidate)

    error = raised.value
    assert error.baseline == str(base)
    assert error.threshold == MIN_TRAINED_RELATIVE_DELTA
    assert error.max_delta < BF16_RELATIVE_ULP
    # The deltas have to survive on the exception, or diagnosing a failed
    # cycle means re-running the cycle.
    assert set(error.deltas) == set(_PROJECTION_NAMES) | set(_NORM_NAMES)
    assert error.deltas["fc.weight"] > 0.0
    for name in _PROJECTION_NAMES:
        assert f"{name}={error.deltas[name]:.6f}" in str(error)


def test_a_single_moved_norm_element_does_not_excuse_the_no_op(
    tmp_path: Path,
) -> None:
    """``qwen-pinned-2`` moved exactly one element of an input layernorm.

    A strict "norms are exactly zero" test would have let that run through,
    which is why the comparison is against an epsilon.
    """
    report = weight_delta_report(
        _write(tmp_path / "cand", _noop_candidate()),
        _write(tmp_path / "base", _baseline_head()),
    )

    moved = report.deltas["layers.0.input_layernorm.weight"]
    assert 0.0 < moved <= NORM_MOVED_EPSILON
    assert is_noop_training_delta(report)


def test_a_genuinely_trained_head_is_not_condemned(tmp_path: Path) -> None:
    base = _write(tmp_path / "stock", _baseline_head())
    trained = _map_head(_baseline_head(), _PROJECTION_NAMES + _NORM_NAMES, lambda v: _shift(v, 12))
    candidate = _write(tmp_path / "draft", trained)
    adapter = _adapter(draft_model=str(base))

    adapter._record_draft_weights(candidate)

    params = adapter.describe().training_params
    assert params["draft_weight_max_relative_delta"] > MIN_TRAINED_RELATIVE_DELTA


def test_a_moved_head_whose_norms_stayed_put_is_still_trained(tmp_path: Path) -> None:
    """The two signals are a conjunction, and this is why.

    Norm gains that did not move are an ordinary outcome; on their own they
    say nothing.  Only "norms still AND nothing anywhere reached the floor" is
    the re-rounding signature.  Were this an ``or``, this head -- projections
    moved ~6 %, three orders of magnitude past one ULP -- would be rejected.
    """
    base = _write(tmp_path / "stock", _baseline_head())
    moved = _map_head(_baseline_head(), _PROJECTION_NAMES, lambda v: _shift(v, 12))
    candidate = _write(tmp_path / "draft", moved)

    report = weight_delta_report(candidate, base)
    assert all(report.deltas[name] <= NORM_MOVED_EPSILON for name in _NORM_NAMES)
    assert report.max_delta > MIN_TRAINED_RELATIVE_DELTA
    assert not is_noop_training_delta(report)

    _adapter(draft_model=str(base))._record_draft_weights(candidate)


def test_without_a_resolvable_baseline_the_check_cannot_fire(tmp_path: Path) -> None:
    candidate = _write(tmp_path / "draft", _noop_candidate())
    adapter = _adapter(draft_model="acme/not-in-the-cache")

    adapter._record_draft_weights(candidate)

    params = adapter.describe().training_params
    assert params["draft_weight_relative_deltas"] is None
    assert params["draft_weight_max_relative_delta"] is None
    assert params["draft_weight_baseline_fingerprint"] is None


def test_a_passing_cycle_records_its_deltas_in_the_manifest(tmp_path: Path) -> None:
    """A pass has to be as diagnosable as a failure.

    The floor is calibrated only from below -- no genuinely trained candidate
    existed to measure -- so the recorded deltas of passing cycles are the
    only evidence that will ever refine it.
    """
    base = _write(tmp_path / "stock", _baseline_head())
    trained = _map_head(_baseline_head(), _PROJECTION_NAMES + _NORM_NAMES, lambda v: _shift(v, 12))
    adapter = _adapter(draft_model=str(base))

    adapter._record_draft_weights(_write(tmp_path / "draft", trained))
    params = adapter.describe().training_params

    deltas = params["draft_weight_relative_deltas"]
    assert isinstance(deltas, dict)
    assert set(deltas) == set(_PROJECTION_NAMES) | set(_NORM_NAMES)
    assert all(isinstance(value, float) and math.isfinite(value) for value in deltas.values())
    assert params["draft_weight_max_relative_delta"] == max(deltas.values())


# --- aux-layer arity --------------------------------------------------------


def test_fc_weight_states_the_aux_count(tmp_path: Path) -> None:
    """Both stock drafters imply 3: (2880, 8640) and (4096, 12288)."""
    head = _baseline_head()
    head["fc.weight"] = ("BF16", (4, 12), _patterns(48, 1))

    assert drafter_aux_count(_write(tmp_path / "three", head)) == 3

    head["fc.weight"] = ("BF16", (4, 16), _patterns(64, 1))
    assert drafter_aux_count(_write(tmp_path / "four", head)) == 4


def test_three_configured_layers_match_a_three_aux_head(tmp_path: Path) -> None:
    """Production today: profiles pin three target layers and fc implies 3."""
    base = _write(tmp_path / "stock", _baseline_head())
    candidate = _write(
        tmp_path / "draft",
        _map_head(_baseline_head(), _PROJECTION_NAMES + _NORM_NAMES, lambda v: _shift(v, 12)),
    )
    adapter = _adapter(draft_model=str(base), target_layer_ids=(2, 12, 21))

    adapter._record_draft_weights(candidate)

    assert drafter_aux_count(candidate) == 3 == len(adapter.config.target_layer_ids or ())


def test_four_configured_layers_against_a_three_aux_head_is_a_mismatch(
    tmp_path: Path,
) -> None:
    base = _write(tmp_path / "stock", _baseline_head())
    candidate = _write(
        tmp_path / "draft",
        _map_head(_baseline_head(), _PROJECTION_NAMES + _NORM_NAMES, lambda v: _shift(v, 12)),
    )
    adapter = _adapter(draft_model=str(base), target_layer_ids=(2, 8, 14, 21))

    with pytest.raises(AuxLayerCountMismatch) as raised:
        adapter._record_draft_weights(candidate)

    # expected = the drafter's claim, actual = what the cycle supplied.
    assert raised.value.expected == 3
    assert raised.value.actual == 4
    assert "expects 3 aux layers" in str(raised.value)
    assert "but 4 were provided" in str(raised.value)


def test_a_malformed_fc_weight_is_not_an_aux_count(tmp_path: Path) -> None:
    head = _baseline_head()
    head["fc.weight"] = ("BF16", (_HIDDEN,), _patterns(_HIDDEN, 2))
    with pytest.raises(DraftWeightsError, match="2-D"):
        drafter_aux_count(_write(tmp_path / "flat", head))

    head["fc.weight"] = ("BF16", (5, 12), _patterns(60, 2))
    with pytest.raises(DraftWeightsError, match="malformed"):
        drafter_aux_count(_write(tmp_path / "ragged", head))

    head = _baseline_head()
    del head["fc.weight"]
    with pytest.raises(DraftWeightsError, match="no fc.weight"):
        drafter_aux_count(_write(tmp_path / "headless", head))
