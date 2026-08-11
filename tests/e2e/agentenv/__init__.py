"""Executable agentic environments: real tools, real workspaces, real grading.

WHY THIS EXISTS
---------------
Everything this project has measured on "agentic" traffic so far was a *replay*.
``tests/e2e/harness/workload_specs/agentic-*.json`` are recordings of some other
model's trajectory (``openai/gpt-oss-20b``, or Claude), and replaying them asks
the served model to continue a conversation it did not write.  That is a fine
way to get a realistic *prompt length distribution* and a hopeless way to get
realistic *supervision*:

* the assistant turns in the prefix are another model's tokens, so training a
  draft head on them teaches it to predict a distribution the verifier does not
  have (this is precisely the finding recorded in the truncation-gate handoff
  §5, where the attempted unblock was reverted);
* nothing in a replay can go wrong.  A recorded trajectory never mistypes a
  path, never gets an error back, never has to recover.  Error-recovery turns
  are a large fraction of real agent traffic and they are exactly the turns a
  drafter finds hard.

``tests/e2e/test_agent_harness.py`` does drive a live loop, but against one toy
task (read ``input.json``, write ``result.json``) whose success condition is the
model emitting a fixed marker string.

This package is the missing middle: a small library of **environments** that
materialize a genuine workspace on disk, expose tools that really execute
against it, run the served model in a real tool loop, and grade the outcome by
re-checking the world rather than by trusting anything the model said.

THE THREE RULES
---------------
1. **No canned tool results.**  Every tool call is executed.  When the model
   calls ``read_file`` on a path that does not exist it gets the real error and
   has to recover, because that recovery turn is the traffic we are here to
   measure.
2. **Grading inspects the world, never the transcript.**  A task is solved when
   its checker -- a subprocess running the workspace's own tests, or a direct
   read of the resulting file -- says so.  A marker string in the final message
   is evidence of nothing; the model can emit it without doing the work.  See
   :class:`~tests.e2e.agentenv.tasks.Grade`.
3. **Every assistant turn in a trajectory came from the model under test.**
   That is what makes these trajectories legal training supervision under
   ``MaskPolicy.ALL_ASSISTANT_TURNS`` when a replay is not, and it is asserted
   rather than assumed -- see
   :func:`~tests.e2e.agentenv.trajectory.self_play_attestation`.
"""

from __future__ import annotations

from tests.e2e.agentenv.loop import (
    AgentLoopResult,
    LoopLimits,
    ToolCallRecord,
    Turn,
    run_agent_loop,
)
from tests.e2e.agentenv.tasks import (
    Grade,
    Task,
    TaskInstance,
    ToolSpec,
    Workspace,
)
from tests.e2e.agentenv.trajectory import (
    SelfPlayAttestation,
    Trajectory,
    self_play_attestation,
)
from tests.e2e.agentenv.workspace import (
    ToolError,
    ToolResult,
    WorkspaceSandbox,
)

__all__ = [
    "AgentLoopResult",
    "Grade",
    "LoopLimits",
    "SelfPlayAttestation",
    "Task",
    "TaskInstance",
    "ToolCallRecord",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "Turn",
    "Workspace",
    "WorkspaceSandbox",
    "run_agent_loop",
    "self_play_attestation",
]
