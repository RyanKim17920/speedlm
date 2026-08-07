"""Unit tests for src/speedlm/training/masking.py.

All tests are torch-free. Every test has mutation evidence confirming it can fail.
See the accompanying mutation evidence table in the PR description.
"""
from __future__ import annotations

import pytest

from speedlm.training.masking import (
    FinalAssistantMaskError,
    MaskPolicy,
    loss_mask_from_offsets,
    require_nonzero_loss_mask,
    select_spans,
)
from speedlm.training.templates.base import AssistantSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def span(start: int, end: int, turn: int = 0, channel: str = "final") -> AssistantSpan:
    return AssistantSpan(start=start, end=end, turn=turn, channel=channel)


# ---------------------------------------------------------------------------
# FinalAssistantMaskError — construction paths
# ---------------------------------------------------------------------------

class TestFinalAssistantMaskError:
    """Two distinct construction paths: backward-compat (1 arg) and full form."""

    def test_single_arg_backward_compat_message(self):
        """Single-arg form must preserve the string as the exception message."""
        err = FinalAssistantMaskError("legacy message")
        assert str(err) == "legacy message"

    def test_single_arg_backward_compat_row_id(self):
        """Single-arg form must set row_id to '<unknown>'."""
        err = FinalAssistantMaskError("legacy message")
        assert err.row_id == "<unknown>"

    def test_single_arg_backward_compat_policy(self):
        """Single-arg form must default policy to FINAL_SPAN."""
        err = FinalAssistantMaskError("legacy message")
        assert err.policy is MaskPolicy.FINAL_SPAN

    def test_full_form_row_id(self):
        """Full three-arg form must expose the supplied row_id."""
        err = FinalAssistantMaskError("row-99", MaskPolicy.ALL_ASSISTANT_TURNS, "oops")
        assert err.row_id == "row-99"

    def test_full_form_policy(self):
        """Full form must expose the supplied policy."""
        err = FinalAssistantMaskError("row-99", MaskPolicy.ALL_ASSISTANT_TURNS, "oops")
        assert err.policy is MaskPolicy.ALL_ASSISTANT_TURNS

    def test_full_form_message_contains_row_id(self):
        """Full-form message must mention the row id."""
        err = FinalAssistantMaskError("row-99", MaskPolicy.ALL_ASSISTANT_TURNS, "oops")
        assert "row-99" in str(err)

    def test_full_form_message_contains_policy_value(self):
        """Full-form message must mention the policy string value."""
        err = FinalAssistantMaskError("row-99", MaskPolicy.ALL_ASSISTANT_TURNS, "oops")
        assert MaskPolicy.ALL_ASSISTANT_TURNS.value in str(err)

    def test_full_form_message_contains_detail(self):
        """Full-form message must include the detail suffix."""
        err = FinalAssistantMaskError("r1", MaskPolicy.FINAL_SPAN, "some detail")
        assert "some detail" in str(err)

    def test_full_form_no_detail_no_colon_suffix(self):
        """Full-form with no detail must not add a trailing colon."""
        err = FinalAssistantMaskError("r1", MaskPolicy.FINAL_SPAN)
        assert not str(err).endswith(":")

    def test_is_value_error(self):
        """FinalAssistantMaskError must be a ValueError subclass."""
        assert isinstance(FinalAssistantMaskError("x"), ValueError)


# ---------------------------------------------------------------------------
# select_spans — FINAL_SPAN
# ---------------------------------------------------------------------------

class TestSelectSpansFinalSpan:
    policy = MaskPolicy.FINAL_SPAN

    def test_empty_spans(self):
        assert select_spans([], self.policy) == ()

    def test_single_final_channel(self):
        s = span(0, 10, turn=0, channel="final")
        assert select_spans([s], self.policy) == (s,)

    def test_single_non_final_channel_falls_back_to_last(self):
        """When no 'final' channel exists, must return the last span."""
        s = span(0, 10, turn=0, channel="draft")
        assert select_spans([s], self.policy) == (s,)

    def test_prefers_final_channel_over_non_final(self):
        """Must return the last 'final'-channel span even if it precedes non-final ones."""
        draft = span(100, 200, turn=1, channel="draft")
        final = span(0, 50, turn=0, channel="final")
        # 'draft' appears later in the list; policy must still prefer 'final'
        result = select_spans([final, draft], self.policy)
        assert result == (final,)

    def test_final_channel_last_among_finals(self):
        """When multiple 'final' spans exist, must return the last one."""
        f1 = span(0, 10, turn=0, channel="final")
        f2 = span(20, 30, turn=1, channel="final")
        assert select_spans([f1, f2], self.policy) == (f2,)

    def test_no_final_channel_returns_last_span(self):
        """With no 'final' channel, must fall back to the very last span regardless of turn."""
        s1 = span(0, 10, turn=0, channel="draft")
        s2 = span(20, 30, turn=1, channel="think")
        assert select_spans([s1, s2], self.policy) == (s2,)

    def test_returns_single_element_tuple(self):
        """Policy must always return at most one span."""
        spans = [span(i * 10, i * 10 + 5, turn=i, channel="final") for i in range(5)]
        result = select_spans(spans, self.policy)
        assert len(result) == 1

    def test_multi_turn_multi_channel_prefers_final(self):
        """Multi-turn conversation: 'final' channel wins even on an earlier turn."""
        think_t1 = span(50, 80, turn=1, channel="think")
        final_t0 = span(0, 40, turn=0, channel="final")
        result = select_spans([final_t0, think_t1], self.policy)
        assert result == (final_t0,)


# ---------------------------------------------------------------------------
# select_spans — FINAL_TURN_ALL_CHANNELS
# ---------------------------------------------------------------------------

class TestSelectSpansFinalTurnAllChannels:
    policy = MaskPolicy.FINAL_TURN_ALL_CHANNELS

    def test_empty_spans(self):
        assert select_spans([], self.policy) == ()

    def test_single_span(self):
        s = span(0, 10, turn=0)
        assert select_spans([s], self.policy) == (s,)

    def test_picks_max_turn_only(self):
        """Must return only spans whose turn equals the maximum turn."""
        t0 = span(0, 10, turn=0, channel="final")
        t1a = span(20, 30, turn=1, channel="final")
        t1b = span(40, 50, turn=1, channel="think")
        result = select_spans([t0, t1a, t1b], self.policy)
        assert set(result) == {t1a, t1b}
        assert t0 not in result

    def test_excludes_earlier_turns(self):
        """Turns below the maximum must be excluded entirely."""
        t0 = span(0, 5, turn=0, channel="final")
        t1 = span(10, 15, turn=1, channel="final")
        t2 = span(20, 25, turn=2, channel="final")
        result = select_spans([t0, t1, t2], self.policy)
        assert result == (t2,)

    def test_all_channels_on_final_turn(self):
        """All channels of the maximum turn must be included."""
        a = span(0, 10, turn=2, channel="final")
        b = span(11, 20, turn=2, channel="think")
        c = span(21, 30, turn=2, channel="draft")
        earlier = span(50, 60, turn=0, channel="final")
        result = select_spans([earlier, a, b, c], self.policy)
        assert set(result) == {a, b, c}

    def test_single_turn_all_channels(self):
        """When all spans share turn=0, all must be selected."""
        spans_list = [
            span(0, 5, turn=0, channel="final"),
            span(10, 15, turn=0, channel="think"),
        ]
        assert set(select_spans(spans_list, self.policy)) == set(spans_list)


# ---------------------------------------------------------------------------
# select_spans — ALL_ASSISTANT_TURNS
# ---------------------------------------------------------------------------

class TestSelectSpansAllAssistantTurns:
    policy = MaskPolicy.ALL_ASSISTANT_TURNS

    def test_empty_spans(self):
        assert select_spans([], self.policy) == ()

    def test_single_span(self):
        s = span(0, 10, turn=0)
        assert select_spans([s], self.policy) == (s,)

    def test_returns_all_spans(self):
        """Must return every span unchanged."""
        spans_list = [span(i * 10, i * 10 + 5, turn=i) for i in range(4)]
        assert select_spans(spans_list, self.policy) == tuple(spans_list)

    def test_preserves_order(self):
        """Span order must be preserved."""
        s0 = span(0, 5, turn=0, channel="final")
        s1 = span(10, 15, turn=1, channel="think")
        s2 = span(20, 25, turn=2, channel="draft")
        assert select_spans([s0, s1, s2], self.policy) == (s0, s1, s2)

    def test_multi_turn_multi_channel_all_included(self):
        """Multi-channel agentic traffic: every span across every turn must be selected."""
        spans_list = [
            span(0, 10, turn=0, channel="final"),
            span(15, 25, turn=0, channel="think"),
            span(30, 40, turn=1, channel="final"),
            span(45, 55, turn=1, channel="draft"),
            span(60, 70, turn=2, channel="final"),
        ]
        assert select_spans(spans_list, self.policy) == tuple(spans_list)

    def test_does_not_filter_by_channel(self):
        """Non-'final' channels must not be filtered out."""
        think_span = span(0, 10, turn=0, channel="think")
        result = select_spans([think_span], self.policy)
        assert think_span in result

    def test_does_not_filter_by_turn(self):
        """Older turns must not be filtered out."""
        old = span(0, 5, turn=0)
        new = span(10, 15, turn=5)
        result = select_spans([old, new], self.policy)
        assert old in result and new in result

    def test_result_length_matches_input(self):
        """Result length must equal input length."""
        spans_list = [span(i * 10, i * 10 + 3, turn=i % 3) for i in range(6)]
        assert len(select_spans(spans_list, self.policy)) == 6


# ---------------------------------------------------------------------------
# loss_mask_from_offsets
# ---------------------------------------------------------------------------

class TestLossMaskFromOffsets:
    """Character offsets -> token boolean mask."""

    def _call(self, offsets, spans_list, *, policy=MaskPolicy.FINAL_SPAN, row_id="r0"):
        return loss_mask_from_offsets(offsets, spans_list, policy=policy, row_id=row_id)

    # --- fail-closed empty-selection path ---

    def test_empty_spans_returns_all_false_no_error(self):
        """Empty spans list -> all-False mask, no exception (fail-closed)."""
        offsets = [(0, 5), (5, 10), (10, 15)]
        result = self._call(offsets, [])
        assert result == (False, False, False)

    def test_empty_spans_empty_offsets_no_error(self):
        """Empty spans + empty offsets -> empty mask, no exception."""
        result = self._call([], [])
        assert result == ()

    # --- normal overlap ---

    def test_overlapping_token_is_true(self):
        """A token that overlaps the selected span must be masked True."""
        # span covers chars 5-15; token (7, 12) is inside
        s = span(5, 15, turn=0, channel="final")
        offsets = [(7, 12)]
        result = self._call(offsets, [s])
        assert result == (True,)

    def test_non_overlapping_token_is_false(self):
        """A token entirely outside the span must be masked False; the overlapping
        neighbor prevents the all-zero raise so we can inspect the False entry."""
        s = span(5, 15, turn=0, channel="final")
        # (0, 4) is outside; (7, 12) overlaps -> prevents raise
        result = self._call([(0, 4), (7, 12)], [s])
        assert result == (False, True)

    def test_adjacent_but_not_overlapping_is_false(self):
        """Token ending exactly at span start -> no overlap (half-open intervals).
        An overlapping neighbour is included to avoid the all-zero raise."""
        s = span(10, 20, turn=0, channel="final")
        # (5, 10): token_end == span.start -> NOT overlapping
        # (10, 15): overlaps -> keeps mask non-zero
        result = self._call([(5, 10), (10, 15)], [s])
        assert result[0] is False
        assert result[1] is True

    def test_adjacent_span_end_at_token_start_is_false(self):
        """Token starting exactly at span end -> no overlap.
        An overlapping neighbour prevents the all-zero raise."""
        s = span(0, 10, turn=0, channel="final")
        # (0, 5) overlaps; (10, 15) starts at span end -> no overlap
        result = self._call([(0, 5), (10, 15)], [s])
        assert result[0] is True
        assert result[1] is False

    def test_partial_overlap_produces_correct_mask(self):
        """Multiple tokens: only overlapping ones should be True."""
        s = span(10, 20, turn=0, channel="final")
        offsets = [(0, 5), (8, 15), (18, 25), (25, 30)]
        result = self._call(offsets, [s])
        # (0,5) no overlap; (8,15) overlaps [10,20); (18,25) overlaps; (25,30) no overlap
        assert result == (False, True, True, False)

    def test_degenerate_token_zero_width_is_false(self):
        """A zero-width token (start == end) is never supervised even inside span;
        a real token neighbour keeps the mask non-zero to avoid the raise."""
        s = span(0, 20, turn=0, channel="final")
        # (5, 5) is zero-width -> False; (1, 3) overlaps -> True
        result = self._call([(5, 5), (1, 3)], [s])
        assert result[0] is False
        assert result[1] is True

    def test_multiple_spans_union(self):
        """Token overlapping any selected span must be True (union semantics)."""
        s1 = span(0, 10, turn=0, channel="final")
        s2 = span(20, 30, turn=1, channel="final")
        offsets = [(5, 8), (15, 18), (25, 28)]
        result = self._call(offsets, [s1, s2], policy=MaskPolicy.ALL_ASSISTANT_TURNS)
        assert result == (True, False, True)

    def test_returns_tuple_of_bools(self):
        """Return type must be a tuple of booleans."""
        s = span(0, 10)
        result = self._call([(0, 5)], [s])
        assert isinstance(result, tuple)
        assert all(isinstance(v, bool) for v in result)

    # --- FinalAssistantMaskError when no overlap ---

    def test_raises_when_no_overlap_with_span(self):
        """Non-empty selection producing zero overlapping tokens must raise."""
        s = span(100, 200, turn=0, channel="final")
        offsets = [(0, 5), (6, 10)]  # entirely before the span
        with pytest.raises(FinalAssistantMaskError):
            self._call(offsets, [s])

    def test_raises_preserves_row_id(self):
        """The raised error must carry the supplied row_id."""
        s = span(100, 200)
        with pytest.raises(FinalAssistantMaskError) as exc_info:
            self._call([(0, 5)], [s], row_id="my-row-42")
        assert exc_info.value.row_id == "my-row-42"

    def test_raises_preserves_policy(self):
        """The raised error must carry the supplied policy."""
        s = span(100, 200)
        with pytest.raises(FinalAssistantMaskError) as exc_info:
            self._call([(0, 5)], [s], policy=MaskPolicy.ALL_ASSISTANT_TURNS)
        assert exc_info.value.policy is MaskPolicy.ALL_ASSISTANT_TURNS

    def test_empty_offsets_with_nonempty_span_raises(self):
        """Non-empty selection + zero tokens -> no overlap -> must raise."""
        s = span(0, 10)
        with pytest.raises(FinalAssistantMaskError):
            self._call([], [s])


# ---------------------------------------------------------------------------
# require_nonzero_loss_mask
# ---------------------------------------------------------------------------

class TestRequireNonzeroLossMask:

    def test_all_false_raises(self):
        """All-False mask must raise FinalAssistantMaskError."""
        with pytest.raises(FinalAssistantMaskError):
            require_nonzero_loss_mask(
                [False, False, False], row_id="r1", policy=MaskPolicy.FINAL_SPAN
            )

    def test_all_zero_int_raises(self):
        """Integer zeros are falsy and must also trigger the error."""
        with pytest.raises(FinalAssistantMaskError):
            require_nonzero_loss_mask([0, 0, 0], row_id="r1", policy=MaskPolicy.FINAL_SPAN)

    def test_empty_mask_raises(self):
        """An empty mask has no True values and must raise."""
        with pytest.raises(FinalAssistantMaskError):
            require_nonzero_loss_mask([], row_id="r1", policy=MaskPolicy.FINAL_SPAN)

    def test_one_true_does_not_raise(self):
        """A mask with at least one True must not raise."""
        require_nonzero_loss_mask([False, True, False], row_id="r1", policy=MaskPolicy.FINAL_SPAN)

    def test_all_true_does_not_raise(self):
        """An all-True mask must not raise."""
        require_nonzero_loss_mask([True, True, True], row_id="r1", policy=MaskPolicy.FINAL_SPAN)

    def test_error_carries_row_id(self):
        """Raised error must expose the supplied row_id."""
        with pytest.raises(FinalAssistantMaskError) as exc_info:
            require_nonzero_loss_mask([False], row_id="bad-row", policy=MaskPolicy.FINAL_SPAN)
        assert exc_info.value.row_id == "bad-row"

    def test_error_carries_policy(self):
        """Raised error must expose the supplied policy."""
        with pytest.raises(FinalAssistantMaskError) as exc_info:
            require_nonzero_loss_mask([0], row_id="r1", policy=MaskPolicy.ALL_ASSISTANT_TURNS)
        assert exc_info.value.policy is MaskPolicy.ALL_ASSISTANT_TURNS

    def test_truthy_nonbool_does_not_raise(self):
        """Truthy non-bool values (e.g. 1) must count as nonzero."""
        require_nonzero_loss_mask([0, 1, 0], row_id="r1", policy=MaskPolicy.FINAL_SPAN)
