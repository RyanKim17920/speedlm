from __future__ import annotations

import json

from speedlm.training.rows import TrainingRow
from speedlm.training.templates.chatml import ChatMLTemplate
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