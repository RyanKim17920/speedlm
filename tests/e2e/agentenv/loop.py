"""The agent loop: model -> tool call -> real execution -> model.

This is the part that makes the traffic agentic.  Each iteration sends the whole
conversation so far plus the task's tool schemas, reads the assistant turn back,
executes every tool call it made against the live workspace, appends the real
results, and goes round again.  Nothing is scripted; the conversation is as long
as the model needs it to be, and it ends when the model calls ``submit``, stops
calling tools, or hits a declared budget.

WHAT THIS FILE REFUSES TO PAPER OVER
------------------------------------
A server without a tool-call parser returns the model's tool syntax as ordinary
``content`` and an empty ``tool_calls``.  The loop then sees "the model answered
without calling a tool", ends the trajectory at turn one, and reports a short,
clean, completely fictitious agent session.  That is the exact defect class this
repository keeps rediscovering, so :func:`run_agent_loop` looks for tool syntax
in the content of any turn that made no structured call and raises
:class:`ToolParserMissingError` naming the flag to pass.  The alternative --
parsing the text ourselves -- would work and would be wrong: it would measure a
tool loop the server is not actually capable of serving.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from tests.e2e.agentenv.tasks import AGENT_TOOLS, SYSTEM_PROMPT, TaskInstance, ToolSpec
from tests.e2e.agentenv.workspace import ToolError, ToolResult, WorkspaceSandbox

__all__ = [
    "AgentLoopResult",
    "LoopLimits",
    "ToolCallRecord",
    "ToolParserMissingError",
    "Turn",
    "run_agent_loop",
]

#: Substrings that mean "this content contains a tool call the server did not
#: parse".  Each is a control-token or wrapper shape emitted by one of the
#: families served here, written as a regex over ordinary characters so this
#: file never contains a literal chat control token (a subagent reading it would
#: stop generating; see the handoff's environment notes).
_UNPARSED_CALL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"<\s*tool_call\s*>"),
    re.compile(r"<\s*function_call\s*>"),
    re.compile(r"\|\s*channel\s*\|.{0,40}\bcommentary\b", re.DOTALL),
    re.compile(r"\bfunctions\.(list_dir|read_file|search|write_file|replace_in_file|run_tests|submit)\b"),
)


class ToolParserMissingError(AssertionError):
    """The server returned tool syntax as text instead of structured calls."""


@dataclass(frozen=True, slots=True)
class LoopLimits:
    """Budgets for one trajectory.

    ``max_turns`` is the interesting one.  Real coding-agent sessions run to
    dozens of turns and the long tail is where the prompt gets big, so this is
    generous; the wall clock and the context window are the real constraints and
    they are enforced separately so a run that ended early says *which* budget
    it hit rather than just "budget".
    """

    max_turns: int = 30
    max_tool_calls_per_turn: int = 4
    max_output_tokens: int = 1024
    request_timeout_seconds: float = 600.0
    wall_clock_seconds: float = 1800.0
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_tool_calls_per_turn", "max_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("request_timeout_seconds", "wall_clock_seconds"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One dispatched tool call and the real result it produced."""

    call_id: str
    name: str
    arguments_json: str
    ok: bool
    result_text: str
    elapsed_seconds: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments_json": self.arguments_json,
            "ok": self.ok,
            "result_text": self.result_text,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class Turn:
    """One assistant turn and everything measured about it.

    ``prompt_tokens`` and ``completion_tokens`` come from the server's own usage
    block rather than from a local tokenizer, so a turn's cost is what the engine
    charged rather than what a re-tokenization guesses.  ``finish_reason`` is
    carried because the benchmark gate reads it: a trajectory whose turns all
    ended at the output cap measured the cap, not the workload.
    """

    index: int
    content: str
    tool_calls: tuple[ToolCallRecord, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
        }


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """A finished trajectory.

    ``messages`` is the full OpenAI-format conversation, which is both what the
    next request would have sent and what the capture layer recorded, so the two
    can be compared rather than assumed equal.
    """

    instance_id: str
    family: str
    messages: tuple[dict[str, Any], ...]
    turns: tuple[Turn, ...]
    stop_condition: str
    submitted_summary: str | None
    wall_clock_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tool_call_count(self) -> int:
        return sum(len(turn.tool_calls) for turn in self.turns)

    @property
    def failed_tool_call_count(self) -> int:
        return sum(1 for turn in self.turns for call in turn.tool_calls if not call.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "family": self.family,
            "messages": [dict(message) for message in self.messages],
            "turns": [turn.to_dict() for turn in self.turns],
            "stop_condition": self.stop_condition,
            "submitted_summary": self.submitted_summary,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "tool_call_count": self.tool_call_count,
            "failed_tool_call_count": self.failed_tool_call_count,
            "metadata": dict(self.metadata),
        }


def run_agent_loop(
    instance: TaskInstance,
    sandbox: WorkspaceSandbox,
    *,
    base_url: str,
    model: str,
    client: httpx.Client,
    limits: LoopLimits | None = None,
    tools: Sequence[ToolSpec] = AGENT_TOOLS,
    system_prompt: str = SYSTEM_PROMPT,
    seed: int | None = None,
) -> AgentLoopResult:
    """Drive one task to completion or to a budget, and return the trajectory."""
    budgets = limits or LoopLimits()
    tool_schemas = [tool.schema() for tool in tools]
    declared = {tool.name for tool in tools}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance.instruction},
    ]
    turns: list[Turn] = []
    started = time.monotonic()
    stop_condition = "max_turns"
    submitted: str | None = None

    for index in range(budgets.max_turns):
        if time.monotonic() - started > budgets.wall_clock_seconds:
            stop_condition = "wall_clock"
            break

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "max_tokens": budgets.max_output_tokens,
            "temperature": budgets.temperature,
            "top_p": budgets.top_p,
        }
        if seed is not None:
            payload["seed"] = seed

        request_started = time.monotonic()
        response = client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=budgets.request_timeout_seconds,
        )
        latency = time.monotonic() - request_started
        if response.status_code != 200:
            raise AssertionError(
                f"turn {index} of {instance.id!r} was rejected with HTTP "
                f"{response.status_code}: {response.text[:2000]}"
            )
        body = response.json()
        choice, usage = _single_choice(body, instance_id=instance.id, turn=index)
        assistant_message = choice.get("message")
        if not isinstance(assistant_message, Mapping):
            raise AssertionError(f"turn {index} of {instance.id!r} returned no message object")

        content = assistant_message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        raw_calls = assistant_message.get("tool_calls")
        calls = list(raw_calls) if isinstance(raw_calls, list) else []

        if not calls:
            _refuse_unparsed_tool_syntax(content, instance_id=instance.id, turn=index)

        # The assistant turn is appended exactly as the server returned it,
        # minus keys the server will not accept back on the next request. Any
        # rewriting here would put the next request's prefix out of step with
        # what the capture layer recorded, and the two are compared later.
        echoed: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            echoed["tool_calls"] = [dict(call) for call in calls]
        messages.append(echoed)

        records: list[ToolCallRecord] = []
        finished_by_submit = False
        for call in calls[: budgets.max_tool_calls_per_turn]:
            record, tool_message, summary = _dispatch(
                call,
                sandbox=sandbox,
                declared=declared,
            )
            records.append(record)
            messages.append(tool_message)
            if summary is not None:
                submitted = summary
                finished_by_submit = True

        dropped = len(calls) - len(records)
        if dropped > 0:
            # Answering only some of the calls would leave unpaired ids in the
            # prefix, which the server rejects on the next request. Say so and
            # stop rather than sending a conversation that cannot be continued.
            for call in calls[budgets.max_tool_calls_per_turn :]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id", "")),
                        "name": _call_name(call),
                        "content": (
                            f"not executed: this turn made {len(calls)} tool calls and the "
                            f"limit is {budgets.max_tool_calls_per_turn} per turn"
                        ),
                    }
                )

        turns.append(
            Turn(
                index=index,
                content=content,
                tool_calls=tuple(records),
                finish_reason=_finish_reason(choice),
                prompt_tokens=_usage_int(usage, "prompt_tokens"),
                completion_tokens=_usage_int(usage, "completion_tokens"),
                latency_seconds=latency,
            )
        )

        if finished_by_submit:
            stop_condition = "submitted"
            break
        if not calls:
            stop_condition = "no_tool_call"
            break
    else:
        stop_condition = "max_turns"

    return AgentLoopResult(
        instance_id=instance.id,
        family=instance.family,
        messages=tuple(messages),
        turns=tuple(turns),
        stop_condition=stop_condition,
        submitted_summary=submitted,
        wall_clock_seconds=time.monotonic() - started,
        metadata=dict(instance.metadata),
    )


def _single_choice(
    body: Mapping[str, Any], *, instance_id: str, turn: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError(f"turn {turn} of {instance_id!r} returned no choices: {body!r}")
    if len(choices) != 1:
        raise AssertionError(
            f"turn {turn} of {instance_id!r} returned {len(choices)} choices; the loop "
            "sends n=1 and continuing an ambiguous branch would silently pick one"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise AssertionError(f"turn {turn} of {instance_id!r} returned a non-object choice")
    usage = body.get("usage")
    return choice, usage if isinstance(usage, Mapping) else {}


def _finish_reason(choice: Mapping[str, Any]) -> str | None:
    value = choice.get("finish_reason")
    return value if isinstance(value, str) else None


def _usage_int(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _call_name(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def _refuse_unparsed_tool_syntax(content: str, *, instance_id: str, turn: int) -> None:
    for pattern in _UNPARSED_CALL_PATTERNS:
        if pattern.search(content):
            raise ToolParserMissingError(
                f"turn {turn} of {instance_id!r} returned tool syntax inside the "
                f"message content and an empty tool_calls array, matching "
                f"{pattern.pattern!r}. The server is serving without a tool-call "
                "parser, so no tool would ever be dispatched and the trajectory "
                "would end at turn one looking healthy. Serve with "
                "--tool-call-parser (and --enable-auto-tool-choice) for this "
                f"model family. First 400 characters: {content[:400]!r}"
            )


def _dispatch(
    call: Mapping[str, Any],
    *,
    sandbox: WorkspaceSandbox,
    declared: set[str],
) -> tuple[ToolCallRecord, dict[str, Any], str | None]:
    """Execute one tool call and build the tool message that answers it.

    Returns ``(record, tool_message, submit_summary)``.  Every failure path
    still produces a tool message: an unanswered ``tool_call_id`` makes the
    *next* request invalid, so a harness that raised here would convert a model
    mistake into a harness crash.
    """
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise AssertionError(f"server returned a tool call with no id: {call!r}")
    name = _call_name(call)
    function = call.get("function")
    raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
    arguments_json = raw_arguments if isinstance(raw_arguments, str) else ""

    def answer(text: str, *, ok: bool, elapsed: float = 0.0, truncated: bool = False) -> tuple[
        ToolCallRecord, dict[str, Any], None
    ]:
        return (
            ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments_json=arguments_json,
                ok=ok,
                result_text=text,
                elapsed_seconds=elapsed,
                truncated=truncated,
            ),
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": text},
            None,
        )

    if name not in declared:
        return answer(
            f"error: no tool named {name!r} is available on this task", ok=False
        )
    try:
        parsed = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as error:
        return answer(
            f"error: arguments were not valid JSON ({error}); send a JSON object", ok=False
        )
    if not isinstance(parsed, Mapping):
        return answer("error: arguments must be a JSON object", ok=False)

    if name == "submit":
        summary = parsed.get("summary")
        text = summary if isinstance(summary, str) else ""
        record, message, _ = answer("submitted", ok=True)
        return record, message, text

    try:
        result: ToolResult = sandbox.execute(name, parsed)
    except ToolError as error:
        return answer(f"error: {error}", ok=False)
    return answer(
        result.text,
        ok=True,
        elapsed=result.elapsed_seconds,
        truncated=result.truncated,
    )
