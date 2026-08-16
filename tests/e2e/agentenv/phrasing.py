"""Prompt phrasing variants for the agent environment catalog.

Given a ``Brief`` (goal, facts, constraints) and a seed, this module renders
free-form instruction text that reads like something a human teammate wrote.
The same ``Brief`` + ``seed`` + ``salt`` always produces the same bytes, but
different seeds should yield *different* surface text so a captured trajectory
corpus doesn't collapse to six identical prompts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Brief", "FactKind", "Fact", "render_instruction"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class FactKind(StrEnum):
    """Classifier for a required fact, used to pick grammatically fitting prose."""

    PATH = "path"  # a file or directory
    SYMBOL = "symbol"  # function/class/variable name
    COMMAND = "command"  # something you run
    NUMBER = "number"  # genuinely numeric: thresholds, counts, ports, limits
    TOKEN = "token"  # opaque strings: currency codes, service names, error codes, ids
    VALUE = "value"  # DEPRECATED alias for TOKEN — still accepted


@dataclass(frozen=True, slots=True)
class Fact:
    """A typed required-fact entry.

    Attributes:
        text: The literal string that must appear verbatim in the render.
        kind: Classifier used to pick a fitting sentence frame.
    """

    text: str
    kind: FactKind


@dataclass(frozen=True, slots=True)
class Brief:
    """A structured task specification that can be rendered as free-form prose.

    Attributes:
        goal: One clause describing what must end up true.
        required_facts: Literal strings or :class:`Fact` instances that
            **must** appear verbatim in the rendered output.  Plain ``str``
            entries default to :attr:`FactKind.TOKEN`.
        context: Optional situational sentence the renderer may reposition.
        constraints: Guardrails (e.g., "don't change the tests").
        artifact: Optional pasted blob (log excerpt, traceback, diff).
        has_test_suite: When False, suppresses all test-suite language
            ("re-run the tests", "builds red", etc.).
        creates: Fact texts that do NOT exist yet; the agent must produce
            them.  These facts only receive output frames ("write X",
            "produce X"), never locative/diagnostic frames.
        blame_path: The single path that may receive "where the bug lives"
            framing.  All other paths get neutral framing only.
    """

    goal: str
    required_facts: tuple[str | Fact, ...] = ()
    context: str = ""
    constraints: tuple[str, ...] = ()
    artifact: str = ""
    has_test_suite: bool = True
    creates: tuple[str, ...] = ()
    blame_path: str = ""


# ---------------------------------------------------------------------------
# Banned-substring constants (for frame classification & test assertions)
# ---------------------------------------------------------------------------

QUANTITATIVE_WORDS: tuple[str, ...] = (
    "threshold",
    "on about a third",
    "the config has",
    "we need to hit",
)

LOCATIVE_WORDS: tuple[str, ...] = (
    "the failure's in",
    "is where the bug lives",
    "the issue traces back to",
    "start at",
    "is the hot spot",
    "traces back to",
)

TEST_SUITE_WORDS: tuple[str, ...] = (
    "test failure",
    "failing suite",
    "the suite",
    "failing test",
    "test suite",
    "failing tests",
    "tests are failing",
    "tests are broken",
    "tests broke",
    "the tests should",
    "the ci suite",
    "test names",
    "re-run the suite",
    "tests pass",
    "passing the suite",
    "test suite that",
    "test suite regression",
    "ci is failing",
    "ci just went red",
    "failing ci",
)


# ---------------------------------------------------------------------------
# Voice pools — independent sampling axes
# ---------------------------------------------------------------------------

# Each index corresponds to a register.


def _is_test_suite(text: str) -> bool:
    """Check if a text mentions test-suite concepts."""
    lower = text.lower()
    return any(w in lower for w in TEST_SUITE_WORDS)


_OPENERS: tuple[tuple[str, ...], ...] = (
    # 0: terse ticket
    (
        "We have a failing suite that needs attention.",
        "There's a test failure blocking the pipeline.",
        "A build is red and we need a fix.",
        "The CI just failed on main.",
        "Tests are failing -- we need to ship a fix.",
        "I'm filing this so we don't lose track of the broken build.",
        "There's a regression I need fixed before the release.",
        "A test is failing and it's not obvious what changed.",
    ),
    # 1: on-call page
    (
        "I just got paged -- the nightly build is failing.",
        "Someone pinged me on the incident channel about this.",
        "The on-call alert just fired for the CI suite.",
        "I was woken up by this alert, so could you take a look?",
        "PagerDuty just hit me with this build failure.",
        "I'm still groggy from the alert -- need a fresh pair of eyes.",
        "The on-call rotation flagged this as urgent.",
        "I got the build failure page and I'd rather you look at it.",
    ),
    # 2: conversational
    (
        "Hey, could you look into something for me?",
        "I noticed something weird and was wondering if you could check it out.",
        "Can you help me with a problem I've been stuck on?",
        "I've been going back and forth on this and could use your perspective.",
        "Do you mind taking a look at something?",
        "I'm not sure where to go with this one.",
        "Could you help me track down an issue?",
        "I've been going in circles on this -- can you help?",
    ),
    # 3: formal bug report
    (
        "QA flagged this on the nightly run.",
        "Customer escalation came in this morning.",
        "This was found during release branch review.",
        "Automated regression detection caught this.",
        "The staging environment started failing.",
        "Post-deployment smoke tests are failing.",
        "This was reported by the QA team.",
        "The test suite regression was caught in staging.",
    ),
    # 4: code review comment
    (
        "I noticed this while reviewing the PR.",
        "Found this during code review -- needs a fix.",
        "This came up in the last code review cycle.",
        "Reviewing the diff, I noticed this issue.",
        "While going through the changes, this stood out.",
        "I flagged this in the review comments.",
        "Code review caught a defect in this area.",
        "Looking through the code, there's something off here.",
    ),
    # 5: Slack-style
    (
        "hey the build just broke again",
        "hmm the tests are failing on main rn",
        "yo can someone fix the failing suite",
        "builds red again anyone on it",
        "the ci is failing and i dont know why",
        "so the tests are broken now and need help fixing",
        "tests broke and im not sure what did it",
        "ci just went red can you check",
    ),
    # 6: handoff note
    (
        "Before I head out, here's something that needs fixing.",
        "I'm handing this off since I'm moving to another project.",
        "Leaving a note on this before I sign off for the day.",
        "Quick handoff -- there's a test suite that needs fixing.",
        "Transferring this to you as I wrap up my last sprint.",
        "Documenting this before my last day on the team.",
        "Passing this along since it fell through the cracks.",
        "I'm stepping away from this codebase, but this needs attention.",
    ),
)

_CLOSINGS: tuple[tuple[str, ...], ...] = (
    # 0: terse
    (
        "",
        "",
        "Please fix.",
        "ETA appreciated.",
        "Fix ASAP.",
        "Let me know when done.",
        "PR when ready.",
        "Reply with the file and line you change.",
    ),
    # 1: on-call
    (
        "Let me know what you find.",
        "Acknowledge when you start looking.",
        "Reply with your findings.",
        "Ping me when you have something.",
        "I'll stand by until someone takes this.",
        "Update the channel once you have a fix.",
        "",
        "Let me know the root cause.",
    ),
    # 2: conversational
    (
        "Thanks in advance!",
        "Let me know what you find.",
        "Appreciate the help!",
        "No rush, but when you get a chance.",
        "Thanks!",
        "Let me know if you need more context.",
        "Give me a shout when you're done.",
        "",
    ),
    # 3: formal bug
    (
        "",
        "Please attach your analysis to this ticket.",
        "Attach a diff when done.",
        "Please update the ticket status once resolved.",
        "CC the team once merged.",
        "File a follow-up if you find additional issues.",
        "",
        "Reply with the root cause.",
    ),
    # 4: code review
    (
        "Please address this before merge.",
        "Fix and push an update.",
        "Reply with the changed file and line.",
        "Please re-run the suite after your fix.",
        "Address this and squash.",
        "Fix this and the review should pass.",
        "",
        "Let me know if the tests pass after your change.",
    ),
    # 5: Slack
    (
        "",
        "thanks!",
        "let me know what u find",
        "ty in advance",
        "no rush btw",
        "thanks a lot",
        "",
        "appreciate it",
    ),
    # 6: handoff
    (
        "Sorry I couldn't get to it myself.",
        "Happy to walk you through it if needed.",
        "The tests should guide you to the right file.",
        "All the context you need should be in the code.",
        "If you need background, check the README.",
        "The failing test names the symptom -- start there.",
        "",
        "I've left some notes in the comments.",
    ),
)

_NOISE_HEAD: tuple[str, ...] = (
    "",
    "",
    "fyi: ",
    "",
    "heads up -- ",
    "",
    "",
    "quick note: ",
    "",
    "",
)

_NOISE_TAIL: tuple[str, ...] = (
    " btw",
    "",
    "",
    " -- thanks",
    "",
    "",
    " (cc: team)",
    "",
    "",
    " pinging here for visibility",
    "",
    "",
)

# ---------------------------------------------------------------------------
# Template pools for weaving individual components
# ---------------------------------------------------------------------------

_GOAL_TEMPLATES: tuple[str, ...] = (
    "{goal}",
    "Here's the ask: {goal}.",
    "The goal here is to {goal}.",
    "What we need: {goal}.",
    "Basically, {goal}.",
    "So the task is {goal}.",
    "What needs to happen: {goal}.",
    "TL;DR -- {goal}.",
    "End goal: {goal}.",
    "This is what needs to be done: {goal}.",
)

_CONTEXT_TEMPLATES: tuple[str, ...] = (
    "{ctx}.",
    "Context: {ctx}.",
    "Background: {ctx}.",
    "For context, {ctx}.",
    "To give you some background: {ctx}.",
    "Just so you know -- {ctx}.",
    "Here's the situation: {ctx}.",
    "A bit of context: {ctx}.",
)

_CONSTRAINT_TEMPLATES_INLINE: tuple[str, ...] = (
    "Also, {constr}.",
    "Please note: {constr}.",
    "One more thing: {constr}.",
    "Keep in mind that {constr}.",
    "Important: {constr}.",
    "Make sure to {constr}.",
    "Remember to {constr}.",
    "Just make sure {constr}.",
)

_CONSTRAINT_TEMPLATES_BULLETED: tuple[str, ...] = (
    "A few things to keep in mind:\n- {constr}",
    "Important constraints:\n- {constr}",
    "Please keep these in mind:\n- {constr}",
    "Just a few ground rules:\n- {constr}",
)

_CONSTRAINT_TEMPLATES_TRAILING: tuple[str, ...] = (
    "Oh, and {constr}.",
    "By the way, {constr}.",
    "Constraints: {constr}.",
    "Make sure {constr}.",
)

_ARTIFACT_FENCED_TEMPLATES: tuple[str, ...] = (
    "Here's the relevant output:\n```\n{artifact}\n```",
    "The output I'm seeing:\n```\n{artifact}\n```",
    "Relevant excerpt:\n```\n{artifact}\n```",
    "Here's what the output looks like:\n```\n{artifact}\n```",
    "Log excerpt:\n```\n{artifact}\n```",
    "Here's what I see:\n```\n{artifact}\n```",
)

_ARTIFACT_INLINE_TEMPLATES: tuple[str, ...] = (
    ' (excerpt: "{artifact}")',
    ' -- here\'s a sample: "{artifact}"',
    ' (sample output: "{artifact}")',
    ' -- I see: "{artifact}"',
)

_ARTIFACT_INTRO_TEMPLATES: tuple[str, ...] = (
    "Here's what I see: {artifact}.",
    "This is the output I'm getting: {artifact}.",
    "The relevant line is: {artifact}.",
    "Here's the key part: {artifact}.",
    "What I'm seeing is this: {artifact}.",
    "I'm seeing the following: {artifact}.",
)

_CONSTRAINT_STYLES = ("inline", "bulleted", "trailing")
_ARTIFACT_STYLES = ("fenced", "inline", "intro", "skip")


# ---------------------------------------------------------------------------
# Per-fact-kind sentence-frame pools  (>= 6 each)
# ---------------------------------------------------------------------------

_PATH_NEUTRAL_FRAMES: tuple[str, ...] = (
    "`{fact}` is in scope",
    "it touches `{fact}`",
    "relevant: `{fact}`",
    "`{fact}` is part of the picture",
    "you'll need `{fact}`",
    "`{fact}` is involved here",
    "the relevant code includes `{fact}`",
    "`{fact}` is one of the pieces",
)

_PATH_BLAME_FRAMES: tuple[str, ...] = (
    "the failure's in `{fact}`",
    "`{fact}` is where the bug lives",
    "the issue traces back to `{fact}`",
    "check `{fact}` -- that's the hot spot",
    "everything you need is under `{fact}`",
    "start at `{fact}`",
    "you'll want to look at `{fact}` first",
)

_SYMBOL_FRAMES: tuple[str, ...] = (
    "`{fact}` is the culprit",
    "look at `{fact}`",
    "the problem starts with `{fact}`",
    "you'll want to trace `{fact}`",
    "`{fact}` is where things go wrong",
    "start by looking at `{fact}`",
    "the issue involves `{fact}`",
    "`{fact}` seems to be at the heart of it",
    "`{fact}` is the one to watch",
    "pay attention to `{fact}`",
)

_COMMAND_FRAMES: tuple[str, ...] = (
    "`{fact}` goes red",
    "just run `{fact}` and you'll see it",
    "`{fact}` was clean yesterday",
    "run `{fact}` to reproduce",
    "`{fact}` is the failing command",
    "you can reproduce with `{fact}`",
    "`{fact}` catches the regression",
    "the smoke test is `{fact}`",
)

_NUMBER_FRAMES: tuple[str, ...] = (
    "the threshold's set to `{fact}`",
    "`{fact}` is the value we're working with",
    "keep `{fact}` in mind as the limit",
    "we need to hit `{fact}` to pass",
    "`{fact}` is what's expected",
    "the target is `{fact}`",
    "the limit is `{fact}`",
    "`{fact}` is the number to watch",
)

_TOKEN_FRAMES: tuple[str, ...] = (
    "we're seeing `{fact}`",
    "`{fact}` is the one in play",
    "it mentions `{fact}`",
    "`{fact}` comes up in the output",
    "the relevant value is `{fact}`",
    "`{fact}` shows up in the trace",
    "you'll see `{fact}` if you look",
    "there's a reference to `{fact}`",
)

_CREATES_FRAMES: tuple[str, ...] = (
    "write your answer to `{fact}`",
    "produce `{fact}` with your findings",
    "`{fact}` is what you need to create",
    "the output should go into `{fact}`",
    "generate `{fact}` as the deliverable",
    "`{fact}` is the file you should produce",
    "you'll need to create `{fact}`",
    "save your results to `{fact}`",
)

_FACT_KIND_FRAMES: dict[FactKind, tuple[str, ...]] = {
    FactKind.PATH: _PATH_NEUTRAL_FRAMES,
    FactKind.SYMBOL: _SYMBOL_FRAMES,
    FactKind.COMMAND: _COMMAND_FRAMES,
    FactKind.NUMBER: _NUMBER_FRAMES,
    FactKind.TOKEN: _TOKEN_FRAMES,
    FactKind.VALUE: _TOKEN_FRAMES,  # DEPRECATED alias for TOKEN
}


# ---------------------------------------------------------------------------
# Component weavers
# ---------------------------------------------------------------------------


def _render_goal(goal: str, rng: random.Random) -> str:
    """Rephrase the goal text using a random template."""
    template = rng.choice(_GOAL_TEMPLATES)
    if template == "{goal}":
        return goal
    g = goal.rstrip(".")
    if g:
        g = g[0].lower() + g[1:]
    return template.format(goal=g)


def _render_context(context: str, rng: random.Random) -> str:
    """Weave the optional context sentence."""
    if not context:
        return ""
    t = rng.choice(_CONTEXT_TEMPLATES)
    ctx = context.rstrip(".")
    return t.format(ctx=ctx)


def _render_constraints(constraints: tuple[str, ...], rng: random.Random) -> str:
    """Weave constraints into prose (inline, bulleted, or trailing)."""
    if not constraints:
        return ""
    style = rng.choice(_CONSTRAINT_STYLES)
    if style == "bulleted":
        t = rng.choice(_CONSTRAINT_TEMPLATES_BULLETED)
        bullets = "\n- ".join(c for c in constraints)
        return t.format(constr=bullets)
    if style == "trailing":
        t = rng.choice(_CONSTRAINT_TEMPLATES_TRAILING)
        bullets = "\n- ".join(c for c in constraints)
        return t.format(constr=bullets)
    # inline
    t = rng.choice(_CONSTRAINT_TEMPLATES_INLINE)
    if len(constraints) == 1:
        joined = constraints[0]
    else:
        joined = ", ".join(constraints[:-1]) + ", and " + constraints[-1]
    return t.format(constr=joined)


def _render_artifact(artifact: str, rng: random.Random) -> str:
    """Embed an artifact (log excerpt, traceback, diff)."""
    if not artifact:
        return ""
    style = rng.choice(_ARTIFACT_STYLES)
    if style == "skip":
        return ""
    if style == "fenced":
        t = rng.choice(_ARTIFACT_FENCED_TEMPLATES)
        return t.format(artifact=artifact)
    if style == "inline":
        short = artifact.replace("\n", " | ")
        t = rng.choice(_ARTIFACT_INLINE_TEMPLATES)
        return t.format(artifact=short)
    # intro
    t = rng.choice(_ARTIFACT_INTRO_TEMPLATES)
    return t.format(artifact=artifact)


def _select_frame(
    fact: Fact,
    blame_path: str,
    creates: tuple[str, ...],
    rng: random.Random,
) -> str:
    """Pick a sentence frame for a single fact, respecting role constraints."""
    if fact.text in creates:
        return rng.choice(_CREATES_FRAMES)
    if fact.kind == FactKind.PATH and fact.text == blame_path:
        return rng.choice(_PATH_BLAME_FRAMES)
    frame_pool = _FACT_KIND_FRAMES[fact.kind]
    return rng.choice(frame_pool)


# ---------------------------------------------------------------------------
# Ordering functions — decide how to arrange the components
# ---------------------------------------------------------------------------

_ORDERING_STANDARD = 0  # opener, context, goal, constraints, artifact, closing
_ORDERING_GOAL_EARLY = 1  # opener, goal, context, constraints, artifact, closing
_ORDERING_CONTEXT_FIRST = 2  # context, opener, goal, constraints, artifact, closing
_ORDERING_CONSTRAINTS_MID = 3  # opener, goal, constraints, context, artifact, closing
_ORDERING_ARTIFACT_EARLY = 4  # opener, artifact, goal, context, constraints, closing
_ORDERING_FACTS_EARLY = 5  # opener, goal, context, constraints, artifact, closing
# (facts distributed, same backbone as GOAL_EARLY)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_instruction(brief: Brief, *, seed: int, salt: int) -> str:
    """Render *brief* as free-form instruction prose.

    The output is a pure function of ``(brief, seed, salt)`` -- the same inputs
    always produce the same bytes, but different seeds produce different
    surface text.  All ``required_facts`` are guaranteed to appear verbatim.
    """
    rng = random.Random(seed ^ salt ^ 0x9E3779B9)

    # 1. Pick register
    register = rng.randint(0, len(_OPENERS) - 1)

    # 2. Pick opener from the chosen register (filter test-suite refs if needed)
    opener_pool = _OPENERS[register]
    if not brief.has_test_suite:
        filtered = [o for o in opener_pool if not _is_test_suite(o)]
        if filtered:
            opener_pool = tuple(filtered)
    opener = rng.choice(opener_pool)

    # 3. Pick closing from the chosen register (filter test-suite refs if needed)
    closing_pool = _CLOSINGS[register]
    if not brief.has_test_suite:
        filtered = [c for c in closing_pool if not _is_test_suite(c)]
        if filtered:
            closing_pool = tuple(filtered)
    closing = rng.choice(closing_pool)

    # 4. Normalize facts
    normalized: list[Fact] = []
    for f in brief.required_facts:
        if isinstance(f, Fact):
            normalized.append(f)
        else:
            normalized.append(Fact(text=f, kind=FactKind.TOKEN))

    # 5. Weave components
    goal_text = _render_goal(brief.goal, rng)

    # Shuffle facts to randomize order across seeds
    rng.shuffle(normalized)
    fact_sentences: list[str] = []
    for fact_obj in normalized:
        frame = _select_frame(fact_obj, brief.blame_path, brief.creates, rng)
        sentence = frame.format(fact=fact_obj.text) + "."
        fact_sentences.append(sentence)

    context_text = _render_context(brief.context, rng)
    constraints_text = _render_constraints(brief.constraints, rng)
    artifact_text = _render_artifact(brief.artifact, rng)

    # 6. Pick ordering
    ordering = rng.randint(0, 5)

    # 7. Surface noise
    head_noise = rng.choice(_NOISE_HEAD)
    tail_noise = rng.choice(_NOISE_TAIL)

    # 8. Assemble parts according to ordering (facts are NOT in parts yet)
    if ordering == _ORDERING_STANDARD:
        parts = [
            opener,
            context_text,
            goal_text,
            constraints_text,
            artifact_text,
            closing,
        ]
    elif ordering == _ORDERING_GOAL_EARLY:
        parts = [
            opener,
            goal_text,
            context_text,
            constraints_text,
            artifact_text,
            closing,
        ]
    elif ordering == _ORDERING_CONTEXT_FIRST:
        parts = [
            context_text,
            opener,
            goal_text,
            constraints_text,
            artifact_text,
            closing,
        ]
    elif ordering == _ORDERING_CONSTRAINTS_MID:
        parts = [
            opener,
            goal_text,
            constraints_text,
            context_text,
            artifact_text,
            closing,
        ]
    elif ordering == _ORDERING_ARTIFACT_EARLY:
        parts = [
            opener,
            artifact_text,
            goal_text,
            context_text,
            constraints_text,
            closing,
        ]
    else:  # FACTS_EARLY (same backbone as GOAL_EARLY; facts distributed)
        parts = [
            opener,
            goal_text,
            context_text,
            constraints_text,
            artifact_text,
            closing,
        ]

    # 9. Filter empty parts
    parts = [p for p in parts if p]

    # 10. Distribute fact sentences into parts at evenly-spaced positions
    if fact_sentences:
        chunk = max(1, len(parts) // (len(fact_sentences) + 1))
        pos = 0
        for sentence in fact_sentences:
            pos = min(pos, len(parts))
            parts.insert(pos, sentence)
            pos += chunk + 1

    # 11. Join and add noise
    result = " ".join(parts)
    if head_noise:
        result = head_noise + result
    if tail_noise:
        result = result + tail_noise

    # 12. Clean up whitespace
    result = " ".join(result.split())

    # 13. For Slack register, lowercase while protecting required facts
    if register == 5:
        placeholders: list[tuple[str, str]] = []
        for i, raw in enumerate(brief.required_facts):
            fact_text = raw.text if isinstance(raw, Fact) else raw
            if fact_text in result:
                placeholder = f"\x00FACT{i}\x00"
                result = result.replace(fact_text, placeholder, 1)
                placeholders.append((placeholder, fact_text))
        result = result.lower()
        for placeholder, fact_text in placeholders:
            result = result.replace(placeholder.lower(), fact_text)

    return result
