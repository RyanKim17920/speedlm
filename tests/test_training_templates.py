from __future__ import annotations

import json

import pytest

from speedlm.training.rows import TrainingRow
from speedlm.training.templates.chatml import ChatMLStructureError, ChatMLTemplate
from speedlm.training.templates.harmony import HarmonyTemplate


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

# ===========================================================================
# ChatMLTemplate's marker assumption is narrow and must be validated
#
# ``assistant_spans`` keys off one family's ChatML control markers.  Handed the
# output of an arbitrary Hugging Face chat template it used to find no marker
# and return an empty tuple -- indistinguishable, to every caller, from a
# legitimate conversation with no assistant content.  Wrong spans dressed as an
# answer is the worst of the three possible outcomes; these pin the refusal.
#
# The foreign renderings below use the Gemma-style turn markers deliberately:
# they carry no pipe characters, so nothing here can be mistaken for a ChatML
# control token by a reader or by a tool.
# ===========================================================================


def _chatml() -> ChatMLTemplate:
    return ChatMLTemplate()


def test_chatml_refuses_a_foreign_template_instead_of_reporting_no_spans() -> None:
    """An empty tuple here reads as "supervises nothing", which is a lie."""
    foreign = (
        "<start_of_turn>user\nhi<end_of_turn>\n"
        "<start_of_turn>model\nhello<end_of_turn>\n"
    )

    with pytest.raises(ChatMLStructureError) as caught:
        _chatml().assistant_spans(foreign)

    message = str(caught.value)
    assert "no ChatML block" in message
    assert "chat template" in message


def test_chatml_refuses_text_wedged_between_two_chatml_blocks() -> None:
    """Half-ChatML is not ChatML; the spans either side would be untrustworthy."""
    template = _chatml()
    head = template.render(({"role": "user", "content": "hi"},))
    tail = template.render(({"role": "assistant", "content": "hello"},))

    with pytest.raises(ChatMLStructureError) as caught:
        template.assistant_spans(f"{head}<start_of_turn>model\nstray<end_of_turn>\n{tail}")

    assert "between ChatML blocks" in str(caught.value)


def test_chatml_refuses_trailing_text_after_the_final_block() -> None:
    """Trailing content means the assumed structure did not describe the text."""
    template = _chatml()
    rendered = template.render(
        (
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        )
    )

    with pytest.raises(ChatMLStructureError) as caught:
        template.assistant_spans(f"{rendered}<start_of_turn>model\ntrailing")

    assert "trailing text" in str(caught.value)


def test_chatml_structure_refusal_stays_a_value_error() -> None:
    """Callers already catch the renderer's structural ValueErrors."""
    assert issubclass(ChatMLStructureError, ValueError)


def test_chatml_counts_an_empty_assistant_turn_without_supervising_it() -> None:
    """An empty turn yields no span but still consumes a turn number."""
    template = _chatml()
    rendered = template.render(
        (
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        )
    )

    spans = template.assistant_spans(rendered)

    assert [rendered[span.start : span.end] for span in spans] == ["hello"]
    assert [span.turn for span in spans] == [1]
