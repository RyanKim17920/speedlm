"""Proofs for authorship ambiguity and per-row loss domination."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from speedlm.config import SamplingConfig
from speedlm.gateway.capture import _build_raw_record
from speedlm.gateway.sse import AssembledResponse
from speedlm.traces.normalize import normalize_record
from speedlm.training.masking import MaskPolicy
from speedlm.training.rows import TrainingRow, prepare_training_row, training_row_from_trace
from speedlm.training.templates.chatml import ChatMLTemplate


class _CharacterTokenizer:
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        limit = kwargs.get("max_length", len(text))
        assert isinstance(limit, int)
        length = min(len(text), limit)
        return {
            "input_ids": list(range(length)),
            "offset_mapping": [(index, index + 1) for index in range(length)],
        }


@pytest.mark.xfail(
    strict=True,
    reason="training rows do not yet exclude client_supplied assistant messages",
)
def test_client_supplied_assistant_turn_must_not_be_supervised() -> None:
    marker = "CLIENT_AUTHORED_ASSISTANT_TEXT"
    raw = _build_raw_record(
        {
            "model": "test-model",
            "messages": [{"role": "assistant", "content": marker}],
        },
        AssembledResponse(
            id="response-1",
            model="test-model",
            created=1_700_000_000.0,
            content="provider response",
            tool_calls=(),
            prompt_tokens=5,
            completion_tokens=2,
        ),
        endpoint="/v1/chat/completions",
        timestamp=1_700_000_000.0,
    )
    trace = normalize_record(raw, defaults=SamplingConfig())
    row = training_row_from_trace(trace)

    prepared = prepare_training_row(
        row,
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    supervised = "".join(
        char
        for char, selected in zip(prepared.rendered, prepared.loss_mask, strict=True)
        if selected
    )

    assert marker not in supervised


def test_capture_assigns_trustworthy_per_message_provenance() -> None:
    raw = _build_raw_record(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "client history",
                    "provenance_tag": "generated",
                }
            ],
        },
        AssembledResponse(
            id="response-1",
            model="test-model",
            created=1_700_000_000.0,
            content="provider response",
            tool_calls=(),
            prompt_tokens=5,
            completion_tokens=2,
        ),
        endpoint="/v1/chat/completions",
        timestamp=1_700_000_000.0,
    )

    trace = normalize_record(raw, defaults=SamplingConfig())

    assert trace.messages[0]["provenance_tag"] == "client_supplied"
    assert trace.messages[-1]["provenance_tag"] == "generated"


def test_one_large_assistant_message_can_dominate_sum_reduced_loss() -> None:
    tokenizer = _CharacterTokenizer()

    def supervised_tokens(answer: str) -> int:
        prepared = prepare_training_row(
            TrainingRow(
                id=f"row-{len(answer)}",
                conversation=({"role": "assistant", "content": answer},),
            ),
            template=ChatMLTemplate(),
            tokenizer=tokenizer,
            mask_policy=MaskPolicy.FINAL_SPAN,
            max_seq_length=8_192,
        )
        return sum(prepared.loss_mask)

    short = supervised_tokens("A")
    large = supervised_tokens("A" * 8_000)

    assert short == 1
    assert large >= 7_900
    assert large > short * 1_000
