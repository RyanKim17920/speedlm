"""Server-level unit tests for the OpenAI-compatible executable agent loop."""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.agentenv.loop import (
    AgentLoopResult,
    LoopLimits,
    ToolParserMissingError,
    run_agent_loop,
)
from tests.e2e.agentenv.tasks import AGENT_TOOLS, Grade, TaskInstance, ToolSpec, Workspace
from tests.e2e.agentenv.workspace import WorkspaceSandbox


@dataclass
class _CannedEndpoint:
    url: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    transport: httpx.BaseTransport | None = None


@contextmanager
def _serve_chat_completions(
    responses: Sequence[Mapping[str, Any]],
) -> Iterator[_CannedEndpoint]:
    """Serve scripted responses over loopback, with a sandbox-only fallback.

    A real ``ThreadingHTTPServer`` is always attempted first.  This execution
    sandbox forbids even loopback socket creation, so ``PermissionError`` alone
    falls back to ``httpx.MockTransport``.  The server-level version MUST be
    rerun outside that sandbox; the fallback covers loop semantics, not TCP.
    """
    endpoint = _CannedEndpoint(url="")

    def scripted_response(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        endpoint.requests.append(payload)
        index = len(endpoint.requests) - 1
        if request.url.path != "/v1/chat/completions" or index >= len(responses):
            return httpx.Response(500, json={"error": "unexpected scripted request"})
        return httpx.Response(200, json=responses[index])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            endpoint.requests.append(request)
            index = len(endpoint.requests) - 1
            if self.path != "/v1/chat/completions" or index >= len(responses):
                self.send_error(500, "unexpected scripted request")
                return
            body = json.dumps(responses[index]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        endpoint.url = "http://agentenv-loop.test"
        endpoint.transport = httpx.MockTransport(scripted_response)
        yield endpoint
        return
    server.daemon_threads = True
    host, port = server.server_address[:2]
    endpoint.url = f"http://{host}:{port}"
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _completion(
    message: Mapping[str, Any], *, finish_reason: str = "tool_calls"
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": dict(message), "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 5},
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _loop_subject(tmp_path: Path) -> tuple[TaskInstance, WorkspaceSandbox]:
    instance = TaskInstance(
        id="loop-fixture-0000",
        family="loop-fixture",
        instruction="Read note.txt, then submit what it says.",
        workspace=Workspace(files={"note.txt": "hello from the workspace\n"}),
        grader=lambda _sandbox: Grade(False, "the loop tests do not invoke grading"),
    )
    return instance, instance.materialize(tmp_path / "workspace")


def _run(
    endpoint: _CannedEndpoint,
    instance: TaskInstance,
    sandbox: WorkspaceSandbox,
    *,
    limits: LoopLimits,
    tools: Sequence[ToolSpec] = AGENT_TOOLS,
) -> AgentLoopResult:
    with httpx.Client(transport=endpoint.transport, trust_env=False) as client:
        return run_agent_loop(
            instance,
            sandbox,
            base_url=endpoint.url,
            model="fixture-model",
            client=client,
            limits=limits,
            tools=tools,
        )


def test_two_turn_session_executes_a_tool_then_ends_on_submit(tmp_path: Path) -> None:
    """A submit on turn two must end a real request/tool/request exchange.

    Returning after the first tool call makes the request-count and submitted
    assertions red; trusting submit prose without dispatch makes the summary red.
    """
    instance, sandbox = _loop_subject(tmp_path)
    responses = [
        _completion(
            {
                "role": "assistant",
                "content": "I will read the note.",
                "tool_calls": [_tool_call("read-1", "read_file", '{"path":"note.txt"}')],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "content": "The note was read.",
                "tool_calls": [
                    _tool_call("submit-1", "submit", '{"summary":"Read the note."}')
                ],
            }
        ),
    ]

    with _serve_chat_completions(responses) as endpoint:
        result = _run(endpoint, instance, sandbox, limits=LoopLimits(max_turns=3))

    assert result.stop_condition == "submitted"
    assert result.submitted_summary == "Read the note."
    assert len(result.turns) == len(endpoint.requests) == 2
    second_messages = endpoint.requests[1]["messages"]
    assert any(
        message.get("role") == "tool" and message.get("tool_call_id") == "read-1"
        for message in second_messages
    )


def test_every_structured_tool_call_gets_exactly_one_answer(tmp_path: Path) -> None:
    """Success, malformed arguments, tool failure, and submit must all be paired.

    Any exception-only dispatch path or duplicate append makes the per-id counts
    differ from one.  The malformed and missing-file calls force real failures.
    """
    instance, sandbox = _loop_subject(tmp_path)
    calls = [
        _tool_call("ok", "read_file", '{"path":"note.txt"}'),
        _tool_call("bad-json", "read_file", "{"),
        _tool_call("missing", "read_file", '{"path":"absent.txt"}'),
        _tool_call("done", "submit", '{"summary":"paired every call"}'),
    ]
    responses = [
        _completion({"role": "assistant", "content": "", "tool_calls": calls})
    ]

    with _serve_chat_completions(responses) as endpoint:
        result = _run(
            endpoint,
            instance,
            sandbox,
            limits=LoopLimits(max_turns=1, max_tool_calls_per_turn=len(calls)),
        )

    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    counts = Counter(message["tool_call_id"] for message in tool_messages)
    assert counts == Counter({call["id"]: 1 for call in calls})
    assert {record.ok for record in result.turns[0].tool_calls} == {True, False}


def test_tool_call_overflow_still_answers_every_call_id(tmp_path: Path) -> None:
    """The per-turn execution budget must not leave an invalid chat prefix.

    Only the first call may execute, but all three IDs must receive one tool
    message.  The historical "slice and forget" implementation goes red here.
    """
    instance, sandbox = _loop_subject(tmp_path)
    calls = [
        _tool_call("first", "list_dir", "{}"),
        _tool_call("overflow-1", "read_file", '{"path":"note.txt"}'),
        _tool_call("overflow-2", "search", '{"pattern":"hello"}'),
    ]
    responses = [
        _completion({"role": "assistant", "content": "", "tool_calls": calls})
    ]

    with _serve_chat_completions(responses) as endpoint:
        result = _run(
            endpoint,
            instance,
            sandbox,
            limits=LoopLimits(max_turns=1, max_tool_calls_per_turn=1),
        )

    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    counts = Counter(message["tool_call_id"] for message in tool_messages)
    assert counts == Counter({call["id"]: 1 for call in calls})
    overflow = {message["tool_call_id"]: message["content"] for message in tool_messages[1:]}
    assert all("not executed" in content for content in overflow.values())
    assert len(result.turns[0].tool_calls) == 1


def test_unparsed_tool_syntax_is_refused_instead_of_ending_cleanly(tmp_path: Path) -> None:
    """Textual tool syntax with no structured calls proves the parser is absent.

    Removing the parser guard turns this response into a healthy-looking
    ``no_tool_call`` result, so the negative assertion directly goes red.
    """
    instance, sandbox = _loop_subject(tmp_path)
    # Build the control-token-shaped probe from fragments.  A literal token in
    # a source file can prematurely stop agents that inspect this test.
    probe = chr(60) + "tool_" + "call" + chr(62) + '{"name":"read_file"}'
    responses = [
        _completion(
            {"role": "assistant", "content": probe, "tool_calls": []},
            finish_reason="stop",
        )
    ]

    with (
        _serve_chat_completions(responses) as endpoint,
        pytest.raises(ToolParserMissingError, match="tool-call parser"),
    ):
        _run(endpoint, instance, sandbox, limits=LoopLimits(max_turns=1))


def test_a_tool_the_task_never_declared_is_refused_before_the_sandbox_runs_it(
    tmp_path: Path,
) -> None:
    """A call to an undeclared tool must be answered with an error, not executed.

    The task declares only ``read_file`` and ``submit``, but the model calls
    ``write_file`` -- a tool the sandbox itself implements perfectly well.  That
    is what makes this test able to fail: without the ``name not in declared``
    guard the call falls straight through to the sandbox, the file appears, and
    the record comes back ``ok=True``.  A weaker version that used a name no
    sandbox tool implements would pass either way, because the sandbox would
    raise and the loop would report ``ok=False`` regardless -- green, and
    measuring nothing.
    """
    instance, sandbox = _loop_subject(tmp_path)
    declared = tuple(tool for tool in AGENT_TOOLS if tool.name in {"read_file", "submit"})
    assert len(declared) == 2
    responses = [
        _completion(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "undeclared",
                        "write_file",
                        '{"path":"planted.txt","content":"the guard did not hold"}',
                    ),
                    _tool_call("done", "submit", '{"summary":"tried an undeclared tool"}'),
                ],
            }
        )
    ]

    with _serve_chat_completions(responses) as endpoint:
        result = _run(
            endpoint,
            instance,
            sandbox,
            limits=LoopLimits(max_turns=1, max_tool_calls_per_turn=2),
            tools=declared,
        )

    refused = result.turns[0].tool_calls[0]
    assert refused.name == "write_file"
    assert refused.ok is False
    assert "no tool named 'write_file' is available" in refused.result_text
    assert not (sandbox.root / "planted.txt").exists()
    answers = [
        message
        for message in result.messages
        if message["role"] == "tool" and message["tool_call_id"] == "undeclared"
    ]
    assert len(answers) == 1
    assert "no tool named" in answers[0]["content"]
