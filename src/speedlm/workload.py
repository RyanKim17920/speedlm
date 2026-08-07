"""Compare the traffic SpeedLM actually captured against what it is configured for.

SpeedLM tunes a draft head on live traffic, so every tuning knob is implicitly
a claim about the *shape* of that traffic.  Those claims were never checked.
``tuning.sequence_length`` assumes prompts fit; ``tuning.benchmark_max_tokens``
assumes answers fit; the Speculators renderer assumes every assistant turn is
this deployment's own output.  On the single-turn, tool-free chat corpus every
default was chosen against, all three hold.  On agentic tool dispatch, on
retrieval traffic, or on a client that replays its own conversation history,
they stop holding -- and nothing says so.  Training still runs, the gate still
measures something, and the failure reads as "the drafter is not very good".

This module turns each of those assumptions into a named finding that says
what was measured, which knob it contradicts, and what to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from speedlm.config import WorkloadConfig
from speedlm.traces.store import TraceStats

#: The fraction of a corpus at which a traffic shape stops being an outlier
#: and starts being part of the workload.
#:
#: One number, used by every fraction check in this module, because a set of
#: individually plausible per-check thresholds is not a policy -- it is five
#: unexplained numbers that happen to be in the same file.
#:
#: 5% is chosen against the cycle the findings are about.  A tuning cycle
#: trains on a window of ``tuning.training_window_records`` (default 256)
#: records, so 5% is ~13 records: enough that it cannot be a handful of stray
#: probe requests, and small enough to fire well before the affected traffic
#: dominates.  It is also the point where the consequence stops being
#: theoretical: 5% of requests mis-served, or 5% of the training corpus
#: silently discarded, is a regression an operator would want named rather
#: than absorbed.
MATERIAL_FRACTION = 0.05

#: How close to ``tuning.benchmark_max_tokens`` a typical completion may sit
#: before the gate is assumed to be measuring truncated answers.
#:
#: Not 1.0: a completion cap does not have to be *reached* to distort the
#: measurement.  Once typical answers run to within 10% of the cap, the
#: distribution is being clipped -- the gate compares two arms over a length
#: neither arm chose -- and acceptance measured there does not transfer to
#: serving, where the answer continues.
COMPLETION_CAP_HEADROOM_FRACTION = 0.9

#: Severity of a finding.  Advisory by construction: see
#: :func:`evaluate_workload` for why nothing here blocks a cycle.
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass(frozen=True, slots=True)
class WorkloadFinding:
    """One named, actionable mismatch between measured traffic and config.

    A finding is useless unless it survives being read six weeks later in a
    run artifact by someone who did not run the cycle, so all four parts are
    mandatory: a stable *code* to grep for, *measured* (what the corpus
    actually looks like), *contradicts* (the configuration this puts in
    doubt), and *remedy* (what to change).
    """

    code: str
    severity: str
    measured: str
    contradicts: str
    remedy: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "measured": self.measured,
            "contradicts": self.contradicts,
            "remedy": self.remedy,
        }

    def render(self) -> str:
        return (
            f"[{self.severity}] {self.code}: {self.measured} "
            f"This contradicts {self.contradicts} {self.remedy}"
        )


@dataclass(frozen=True, slots=True)
class WorkloadShape:
    """The measured traffic shape of a trace corpus.

    A projection of :class:`~speedlm.traces.store.TraceStats` onto the fields
    that describe *what kind* of traffic was captured, so that the checks
    below -- and the cycle record they are written into -- do not depend on
    the buffer's size, redaction or drop accounting.
    """

    records: int
    multi_turn_records: int
    tool_schema_records: int
    tool_call_records: int
    system_prompt_records: int
    assistant_history_records: int
    client_supplied_assistant_records: int
    truncated_completion_records: int
    training_dropped_records: int
    prompt_tokens_p50: int | None
    prompt_tokens_p95: int | None
    prompt_tokens_max: int | None
    completion_tokens_p50: int | None
    completion_tokens_p95: int | None
    completion_tokens_max: int | None

    @classmethod
    def from_stats(cls, stats: TraceStats) -> WorkloadShape:
        return cls(
            records=stats.count,
            multi_turn_records=stats.multi_turn_records,
            tool_schema_records=stats.tool_schema_records,
            tool_call_records=stats.tool_call_records,
            system_prompt_records=stats.system_prompt_records,
            assistant_history_records=stats.assistant_history_records,
            client_supplied_assistant_records=stats.client_supplied_assistant_records,
            truncated_completion_records=stats.truncated_completion_records,
            training_dropped_records=stats.training_dropped_records,
            prompt_tokens_p50=stats.prompt_tokens_p50,
            prompt_tokens_p95=stats.prompt_tokens_p95,
            prompt_tokens_max=stats.prompt_tokens_max,
            completion_tokens_p50=stats.completion_tokens_p50,
            completion_tokens_p95=stats.completion_tokens_p95,
            completion_tokens_max=stats.completion_tokens_max,
        )

    def fraction(self, count: int) -> float:
        """Return *count* as a fraction of the corpus; 0.0 for an empty one."""
        if self.records <= 0:
            return 0.0
        return count / self.records

    def is_material(self, count: int) -> bool:
        """Whether *count* records are enough of the corpus to act on."""
        return count > 0 and self.fraction(count) >= MATERIAL_FRACTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "multi_turn_records": self.multi_turn_records,
            "tool_schema_records": self.tool_schema_records,
            "tool_call_records": self.tool_call_records,
            "system_prompt_records": self.system_prompt_records,
            "assistant_history_records": self.assistant_history_records,
            "client_supplied_assistant_records": (
                self.client_supplied_assistant_records
            ),
            "truncated_completion_records": self.truncated_completion_records,
            "training_dropped_records": self.training_dropped_records,
            "prompt_tokens_p50": self.prompt_tokens_p50,
            "prompt_tokens_p95": self.prompt_tokens_p95,
            "prompt_tokens_max": self.prompt_tokens_max,
            "completion_tokens_p50": self.completion_tokens_p50,
            "completion_tokens_p95": self.completion_tokens_p95,
            "completion_tokens_max": self.completion_tokens_max,
        }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def evaluate_workload(
    shape: WorkloadShape,
    *,
    workload: WorkloadConfig,
    sequence_length: int,
    benchmark_max_tokens: int,
) -> tuple[WorkloadFinding, ...]:
    """Report every way the measured corpus contradicts the configuration.

    Nothing here blocks a tuning cycle, and that is a deliberate decision
    rather than an omission.  A finding means "this cycle is training on
    traffic the configuration did not anticipate", not "this cycle's result is
    meaningless": the benchmark gate still measures the candidate against the
    incumbent on held-out contexts drawn from *the same* corpus, so a
    candidate that does not help on this traffic is still rejected on
    evidence.  Blocking would instead stop tuning outright on every deployment
    whose traffic is merely different from the chat corpus the defaults were
    chosen against -- which is the failure this module exists to prevent, not
    to cause.

    Two findings come close to the "result is meaningless" bar and are worth
    naming for whoever revisits this:
    ``prompt_tokens_exceed_sequence_length`` (training silently truncates) and
    ``completions_near_benchmark_cap`` (the gate scores clipped answers).
    Both degrade the *quality* of a measurement that still happens, and both
    are recoverable by raising a bound and re-running, so they are reported
    loudly and recorded into the cycle rather than allowed to halt tuning.
    """
    findings: list[WorkloadFinding] = []
    records = shape.records

    if records <= 0:
        return ()

    # ── Shape the training and gate paths must handle ───────────────────────

    if shape.is_material(shape.tool_schema_records):
        findings.append(
            WorkloadFinding(
                code="tool_schemas_present",
                severity=SEVERITY_WARNING,
                measured=(
                    f"{shape.tool_schema_records} of {records} captured requests "
                    f"({_pct(shape.fraction(shape.tool_schema_records))}) offer tool "
                    f"schemas, and {shape.tool_call_records} produced tool calls."
                ),
                contradicts=(
                    "the tuning defaults, which were calibrated on a tool-free "
                    "chat corpus: tool schemas sit in the prompt that training "
                    "renders and the gate replays, so they change both the "
                    "prompt length and the answer distribution being scored."
                ),
                remedy=(
                    "Declare workload.domain and workload.expect_tools=true so "
                    "the assumption is recorded, and confirm "
                    "tuning.sequence_length and tuning.benchmark_max_tokens were "
                    "sized against tool-carrying prompts rather than bare chat."
                ),
            )
        )

    if shape.is_material(shape.client_supplied_assistant_records):
        findings.append(
            WorkloadFinding(
                code="client_supplied_assistant_turns_dropped",
                severity=SEVERITY_WARNING,
                measured=(
                    f"{shape.client_supplied_assistant_records} of {records} "
                    f"captured requests "
                    f"({_pct(shape.fraction(shape.client_supplied_assistant_records))})"
                    " carry an assistant turn this deployment did not generate, "
                    f"and {shape.multi_turn_records} are multi-turn; "
                    f"{shape.training_dropped_records} records in total are "
                    "untrainable."
                ),
                contradicts=(
                    "the training path, which drops any row containing an "
                    "assistant turn it cannot attribute to this verifier "
                    "(eagle3 trust_untagged_assistant_messages defaults to "
                    "false). The cycle trains on the remainder while the "
                    "accumulation gate counts the whole corpus."
                ),
                remedy=(
                    "Declare workload.expect_client_supplied_assistant_turns=true "
                    "to record that the loss is expected. If the corpus predates "
                    "provenance tagging and every assistant turn really is this "
                    "verifier's own output, set the backend's "
                    "trust_untagged_assistant_messages instead; otherwise expect "
                    "the effective corpus to be smaller than min_corpus_records "
                    "suggests."
                ),
            )
        )

    # ── Bounds the corpus does not fit inside ───────────────────────────────

    p95_prompt = shape.prompt_tokens_p95
    if p95_prompt is not None and p95_prompt > sequence_length:
        findings.append(
            WorkloadFinding(
                code="prompt_tokens_exceed_sequence_length",
                severity=SEVERITY_WARNING,
                measured=(
                    f"p95 prompt length is {p95_prompt} tokens "
                    f"(p50 {shape.prompt_tokens_p50}, max {shape.prompt_tokens_max}) "
                    f"over {records} captured requests."
                ),
                contradicts=(
                    f"tuning.sequence_length={sequence_length}: at least 5% of the "
                    "corpus is longer than the training window, so those rows are "
                    "truncated during data preparation and the draft head is "
                    "trained on prompts the deployment never sends."
                ),
                remedy=(
                    "Raise tuning.sequence_length above the p95 prompt length, or "
                    "accept the truncation deliberately and record it in "
                    "workload.domain."
                ),
            )
        )

    p50_completion = shape.completion_tokens_p50
    cap = benchmark_max_tokens * COMPLETION_CAP_HEADROOM_FRACTION
    if p50_completion is not None and p50_completion >= cap:
        findings.append(
            WorkloadFinding(
                code="completions_near_benchmark_cap",
                severity=SEVERITY_WARNING,
                measured=(
                    f"typical completions run to {p50_completion} tokens "
                    f"(p95 {shape.completion_tokens_p95}, max "
                    f"{shape.completion_tokens_max}), and "
                    f"{shape.truncated_completion_records} of {records} captured "
                    "completions were already cut off upstream."
                ),
                contradicts=(
                    f"tuning.benchmark_max_tokens={benchmark_max_tokens}: the gate "
                    "stops both arms at that cap, so it measures acceptance over a "
                    "truncated answer rather than the answer serving produces."
                ),
                remedy=(
                    "Raise tuning.benchmark_max_tokens above the typical completion "
                    "length so the gate scores whole answers; otherwise the "
                    "promotion decision is made on a prefix."
                ),
            )
        )

    # ── Declared expectations the measurement contradicts ───────────────────

    findings.extend(_declaration_findings(shape, workload))

    if not workload.domain:
        findings.append(
            WorkloadFinding(
                code="workload_domain_undeclared",
                severity=SEVERITY_INFO,
                measured=(
                    f"{records} captured requests: "
                    f"{shape.multi_turn_records} multi-turn, "
                    f"{shape.tool_schema_records} with tool schemas, "
                    f"{shape.system_prompt_records} with a system prompt, "
                    f"{shape.assistant_history_records} with conversation history, "
                    f"p50/p95 prompt tokens {shape.prompt_tokens_p50}/"
                    f"{shape.prompt_tokens_p95}."
                ),
                contradicts=(
                    "workload.domain, which is unset: the traffic this deployment "
                    "trained on is recorded only as these numbers, with no operator "
                    "statement of what it is supposed to be."
                ),
                remedy=(
                    "Set workload.domain to a label for this traffic so future runs "
                    "can be compared against a stated intent rather than against an "
                    "assumption."
                ),
            )
        )

    return tuple(findings)


#: ``(config field, measured count, singular description)`` for every
#: declaration that can be checked directly against a corpus counter.
_DECLARATIONS: tuple[tuple[str, str, str], ...] = (
    ("expect_tools", "tool_schema_records", "requests offering tool schemas"),
    ("expect_multi_turn", "multi_turn_records", "multi-turn requests"),
    ("expect_system_prompts", "system_prompt_records", "requests with a system prompt"),
    (
        "expect_client_supplied_assistant_turns",
        "client_supplied_assistant_records",
        "requests with a client-supplied assistant turn",
    ),
)


def _declaration_findings(
    shape: WorkloadShape,
    workload: WorkloadConfig,
) -> list[WorkloadFinding]:
    """Report declared expectations the corpus does not bear out.

    Both directions are reported.  "Declared absent, measured present" is the
    silent-degradation case -- the deployment is serving traffic it told the
    tuner to ignore.  "Declared present, measured absent" matters too: it
    usually means the capture path is not seeing what the operator thinks it
    is (tools stripped before the gateway, history reconstructed downstream),
    and a tuner that trained happily on the wrong corpus would report nothing.
    """
    findings: list[WorkloadFinding] = []
    for field_name, count_name, description in _DECLARATIONS:
        declared = getattr(workload, field_name)
        if declared is None:
            continue
        count = getattr(shape, count_name)
        present = shape.is_material(count)
        if declared and count == 0:
            findings.append(
                WorkloadFinding(
                    code="declared_shape_absent",
                    severity=SEVERITY_WARNING,
                    measured=(
                        f"none of the {shape.records} captured requests are "
                        f"{description}."
                    ),
                    contradicts=(
                        f"workload.{field_name}=true, which declares that this "
                        "deployment's traffic has that shape."
                    ),
                    remedy=(
                        "Either the capture path is not seeing the traffic you "
                        f"think it is, or workload.{field_name} should be false. "
                        "Check the gateway before trusting a cycle trained on this "
                        "corpus."
                    ),
                )
            )
        elif not declared and present:
            findings.append(
                WorkloadFinding(
                    code="declared_shape_present",
                    severity=SEVERITY_WARNING,
                    measured=(
                        f"{count} of {shape.records} captured requests "
                        f"({_pct(shape.fraction(count))}) are {description}."
                    ),
                    contradicts=(
                        f"workload.{field_name}=false, which declares that this "
                        "deployment's traffic does not have that shape. Every "
                        "tuning bound sized under that declaration is now sized "
                        "for traffic that is not what arrives."
                    ),
                    remedy=(
                        f"Set workload.{field_name}=true and re-check "
                        "tuning.sequence_length and tuning.benchmark_max_tokens "
                        "against the measured distribution."
                    ),
                )
            )
    return findings


def format_workload_findings(findings: tuple[WorkloadFinding, ...]) -> str:
    """Render findings as one operator-readable block."""
    if not findings:
        return "workload: no mismatch between measured traffic and configuration"
    lines = [f"workload: {len(findings)} finding(s)"]
    lines.extend(f"  - {finding.render()}" for finding in findings)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class WorkloadAssessment:
    """The measured shape plus every finding, as recorded into a cycle.

    Kept as one object because the two halves are worthless apart: findings
    without the shape cannot be re-derived or disputed, and a shape without
    findings is the silent number the tuner already had.
    """

    domain: str
    shape: WorkloadShape
    findings: tuple[WorkloadFinding, ...]

    @classmethod
    def measure(
        cls,
        stats: TraceStats,
        *,
        workload: WorkloadConfig,
        sequence_length: int,
        benchmark_max_tokens: int,
    ) -> WorkloadAssessment:
        shape = WorkloadShape.from_stats(stats)
        return cls(
            domain=workload.domain,
            shape=shape,
            findings=evaluate_workload(
                shape,
                workload=workload,
                sequence_length=sequence_length,
                benchmark_max_tokens=benchmark_max_tokens,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            # Recorded even when empty: "" is the durable evidence that the
            # cycle ran against an undeclared domain, which is the condition
            # this whole module exists to stop being invisible.
            "domain": self.domain,
            "shape": self.shape.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }
