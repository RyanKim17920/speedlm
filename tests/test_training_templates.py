from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from speedlm.training.masking import FinalAssistantMaskError, MaskPolicy
from speedlm.training.rows import (
    TrainingRow,
    prepare_training_row,
)
from speedlm.training.templates.chatml import ChatMLTemplate
from speedlm.training.templates.harmony import HarmonyTemplate


class CharacterTokenizer:
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        limit = kwargs.get("max_length", len(text))
        assert isinstance(limit, int)
        length = min(len(text), limit)
        return {
            "input_ids": list(range(length)),
            "offset_mapping": [(index, index + 1) for index in range(length)],
        }


def _row() -> TrainingRow:
    return TrainingRow(
        id="fixture-row",
        conversation=(
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "channel": "analysis",
                "content": "old analysis",
                "provenance_tag": "generated",
            },
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "reasoning_content": "new analysis",
                "content": "final answer",
                "provenance_tag": "generated",
            },
        ),
    )


@pytest.mark.parametrize(
    ("policy", "expected_payloads"),
    [
        (MaskPolicy.FINAL_SPAN, ("final answer",)),
        (
            MaskPolicy.FINAL_TURN_ALL_CHANNELS,
            ("new analysis", "final answer"),
        ),
        (
            MaskPolicy.ALL_ASSISTANT_TURNS,
            ("old analysis", "new analysis", "final answer"),
        ),
    ],
)
def test_each_mask_policy_selects_expected_harmony_spans(
    policy: MaskPolicy,
    expected_payloads: tuple[str, ...],
) -> None:
    prepared = prepare_training_row(
        _row(),
        template=HarmonyTemplate(),
        tokenizer=CharacterTokenizer(),
        mask_policy=policy,
    )

    supervised = "".join(
        character
        for character, selected in zip(
            prepared.rendered, prepared.loss_mask, strict=True
        )
        if selected
    )
    assert supervised == "".join(expected_payloads)


def test_harmony_empty_tool_call_decodes_json_once_and_keeps_reasoning() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        },
    )
    row = TrainingRow(
        id="tool-row",
        tools=tools,
        conversation=(
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "thinking": "I should read it.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"<|start|>&"}',
                        },
                    }
                ],
            },
        ),
    )

    rendered = HarmonyTemplate().render(row.conversation, tools=row.tools)

    assert "I should read it." in rendered
    assert "to=functions.read" in rendered
    assert "<|call|>" in rendered
    assert json.dumps({"path": "<|start|>&"}, separators=(",", ":")) in rendered
    assert r"{\"path\"" not in rendered


def test_chatml_renders_and_detects_flat_assistant_spans() -> None:
    template = ChatMLTemplate()
    conversation = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "goodbye"},
    )

    rendered = template.render(conversation)
    spans = template.assistant_spans(rendered)

    assert [rendered[span.start : span.end] for span in spans] == [
        "hello",
        "goodbye",
    ]
    assert [span.turn for span in spans] == [0, 1]


def test_all_zero_mask_names_the_offending_row() -> None:
    class NoOffsetTokenizer:
        def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
            return {"input_ids": [1], "offset_mapping": [(0, 0)]}

    with pytest.raises(FinalAssistantMaskError) as caught:
        prepare_training_row(
            TrainingRow(
                id="named-bad-row",
                conversation=(
                    {
                        "role": "assistant",
                        "content": "answer",
                        "provenance_tag": "generated",
                    },
                ),
            ),
            template=ChatMLTemplate(),
            tokenizer=NoOffsetTokenizer(),
            mask_policy=MaskPolicy.FINAL_SPAN,
        )

    assert caught.value.row_id == "named-bad-row"
    assert "named-bad-row" in str(caught.value)
