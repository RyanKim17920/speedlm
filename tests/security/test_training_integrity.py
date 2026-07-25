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


@pytest.mark.parametrize("mask_policy", list(MaskPolicy))
def test_client_supplied_assistant_turn_must_not_be_supervised(
    mask_policy: MaskPolicy,
) -> None:
    marker = "CLIENT_AUTHORED_ASSISTANT_TEXT"
    row = TrainingRow(
        id="client-only",
        conversation=(
            {
                "role": "assistant",
                "content": marker,
                "provenance_tag": "client_supplied",
            },
        ),
    )

    prepared = prepare_training_row(
        row,
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=mask_policy,
    )
    supervised = "".join(
        char
        for char, selected in zip(prepared.rendered, prepared.loss_mask, strict=True)
        if selected
    )

    assert marker not in supervised
    assert sum(prepared.loss_mask) == 0
    assert marker in prepared.rendered


@pytest.mark.parametrize("mask_policy", list(MaskPolicy))
def test_generated_assistant_turn_is_supervised(mask_policy: MaskPolicy) -> None:
    marker = "GENERATED_ASSISTANT_TEXT"
    prepared = prepare_training_row(
        TrainingRow(
            id="generated",
            conversation=(
                {
                    "role": "assistant",
                    "content": marker,
                    "provenance_tag": "generated",
                },
            ),
        ),
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=mask_policy,
    )

    assert sum(prepared.loss_mask) == len(marker)


@pytest.mark.parametrize("provenance_tag", [None, "unexpected"])
def test_missing_or_unrecognized_provenance_is_not_supervised(
    provenance_tag: str | None,
) -> None:
    message = {"role": "assistant", "content": "UNKNOWN_AUTHOR"}
    if provenance_tag is not None:
        message["provenance_tag"] = provenance_tag
    prepared = prepare_training_row(
        TrainingRow(id="unknown", conversation=(message,)),
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )

    assert sum(prepared.loss_mask) == 0


def test_trusted_offline_import_can_opt_in_untagged_assistant_messages() -> None:
    raw = {
        "id": "trusted-offline",
        "messages": [{"role": "assistant", "content": "legacy answer"}],
    }
    default_row = training_row_from_trace(raw)
    default_prepared = prepare_training_row(
        default_row,
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    row = training_row_from_trace(
        raw,
        trust_untagged_assistant_messages=True,
    )
    prepared = prepare_training_row(
        row,
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )

    assert sum(default_prepared.loss_mask) == 0
    assert row.conversation[0]["provenance_tag"] == "generated"
    assert sum(prepared.loss_mask) == len("legacy answer")


def test_mixed_provenance_reports_only_generated_supervised_tokens() -> None:
    client_text = "CLIENT_CONTEXT"
    generated_text = "GENERATED_TARGET"
    prepared = prepare_training_row(
        TrainingRow(
            id="mixed",
            conversation=(
                {
                    "role": "assistant",
                    "content": client_text,
                    "provenance_tag": "client_supplied",
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": generated_text,
                    "provenance_tag": "generated",
                },
            ),
        ),
        template=ChatMLTemplate(),
        tokenizer=_CharacterTokenizer(),
        mask_policy=MaskPolicy.ALL_ASSISTANT_TURNS,
    )
    supervised = "".join(
        char
        for char, selected in zip(prepared.rendered, prepared.loss_mask, strict=True)
        if selected
    )

    assert client_text in prepared.rendered
    assert supervised == generated_text
    assert sum(prepared.loss_mask) == len(generated_text)


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
                conversation=(
                    {
                        "role": "assistant",
                        "content": answer,
                        "provenance_tag": "generated",
                    },
                ),
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
