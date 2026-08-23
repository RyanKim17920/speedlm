from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------
from speedlm.storage import atomic_write_json

#: ``speedlm_home`` is defined in :mod:`speedlm.storage`, which owns the on-disk
#: layout rooted at it; nothing in this module uses it.  It is re-exported here
#: only because ``speedlm.config.speedlm_home`` is the import path callers
#: already use.  Importing it the other way round -- storage reaching back into
#: config for it -- is what put these two modules in an import cycle.  The
#: redundant alias is the explicit re-export form, so linters do not read this
#: as an unused import.
from speedlm.storage import speedlm_home as speedlm_home

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
STARTUP_TIMEOUT_ENV_VAR = "SPEEDLM_STARTUP_TIMEOUT_SECONDS"
DEFAULT_STARTUP_STALL_SECONDS = 600.0
STARTUP_STALL_ENV_VAR = "SPEEDLM_STARTUP_STALL_SECONDS"

#: Learning rate the Speculators reference recipe trains a draft head at
#: (``speculators/scripts/train.py``: ``--lr`` default ``1e-4``).  This is the
#: value the upstream project has actually validated EAGLE-3 heads with, so it
#: is what this project defaults to as well.
REFERENCE_LEARNING_RATE = 1e-4
#: Upper bound on ``tuning.learning_rate``.
#:
#: The bound exists as a fat-finger guard -- a config that says ``1`` or
#: ``0.1`` where it meant ``1e-4`` would wreck a draft head, and there is no
#: cheap way to notice that except at parse time.  It is *not* a claim about
#: where the safe/unsafe boundary lies, and it must not be read as one: the
#: previous bound of ``1e-5`` was introduced with no recorded rationale (commit
#: ee54b35, "feat(config): define portable idle tuning settings" -- no comment,
#: no docstring, no message body justifying it) and it silently made training a
#: no-op.  At ``1e-5`` over the ~202 optimizer steps a real cycle runs, the
#: AdamW update to a norm weight of magnitude ~1.0-1.5 never reaches half a
#: bf16 ULP and is truncated away at checkpoint write.  Measured on the
#: published artifacts: every RMSNorm tensor moved EXACTLY 0.000000 and every
#: Linear moved 1-5 bf16 ULPs (max relative delta 0.0058-0.0061 against a bf16
#: relative ULP of 2^-7 = 0.0078), while the trainer's own step-0 accuracy
#: stayed flat across epochs (gpt-oss 0.706/0.705/0.705).
#:
#: ``1e-3`` is a deliberate decade of headroom above the reference ``1e-4``:
#: high enough that the reference recipe and any ordinary 2-5x sweep around it
#: are comfortably inside the bound rather than sitting on its edge, low enough
#: that the mistakes worth catching -- a dropped exponent (``1e-4`` -> ``1``),
#: a shifted one (``1e-4`` -> ``1e-1``), a stray percent -- are still rejected.
#: Raise it if a sweep has evidence for a larger rate; do not raise it to make
#: one config load.
MAX_LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _validate_host(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}.host must be a non-empty string, got {value!r}")
    return value


def _validate_port(value: Any, name: str) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(f"{name}.port must be an int, got {type(value).__name__!r}")
    if not (1 <= value <= 65535):
        raise ConfigError(f"{name}.port must be in 1..65535, got {value}")
    return value


def _validate_float_gte(value: Any, name: str, minimum: float) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {type(value).__name__!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def _validate_float(value: Any, name: str) -> float:
    """Validate a finite numeric value with no bound on either side."""
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {type(value).__name__!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {value}")
    return float(value)


def _validate_int_gte(value: Any, name: str, minimum: int) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an int, got {type(value).__name__!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def startup_timeout_seconds() -> float:
    """Return the configured vLLM startup hard ceiling.

    ``SPEEDLM_STARTUP_TIMEOUT_SECONDS`` overrides the default for process
    launches that do not load a model config.
    """
    raw = os.environ.get(STARTUP_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_STARTUP_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{STARTUP_TIMEOUT_ENV_VAR} must be numeric, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{STARTUP_TIMEOUT_ENV_VAR} must be > 0, got {raw!r}")
    return value


def startup_stall_seconds() -> float:
    """Return the configured vLLM startup liveness window.

    ``SPEEDLM_STARTUP_STALL_SECONDS`` overrides the default for process
    launches that do not load a model config.
    """
    raw = os.environ.get(STARTUP_STALL_ENV_VAR)
    if raw is None:
        return DEFAULT_STARTUP_STALL_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{STARTUP_STALL_ENV_VAR} must be numeric, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{STARTUP_STALL_ENV_VAR} must be > 0, got {raw!r}")
    return value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if _is_bool(self.temperature) or not isinstance(self.temperature, (int, float)):
            raise ConfigError(
                f"temperature must be numeric, got {type(self.temperature).__name__!r}"
            )
        if self.temperature < 0:
            raise ConfigError(f"temperature must be >= 0, got {self.temperature}")
        if _is_bool(self.top_p) or not isinstance(self.top_p, (int, float)):
            raise ConfigError(f"top_p must be numeric, got {type(self.top_p).__name__!r}")
        if not (0 < self.top_p <= 1):
            raise ConfigError(f"top_p must be in (0, 1], got {self.top_p}")
        if _is_bool(self.seed) or not isinstance(self.seed, int):
            raise ConfigError(f"seed must be an int, got {type(self.seed).__name__!r}")
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """What kind of traffic this deployment expects to capture and tune on.

    SpeedLM's premise is that it learns from *your* traffic, but every default
    in the tuning section was chosen against one corpus: single-turn,
    user-only, tool-free chat with a median prompt around 117 tokens.  Nothing
    recorded that.  A deployment serving agentic tool dispatch or long-context
    retrieval inherits those defaults silently and degrades in ways that read
    as drafter quality rather than as a corpus mismatch.

    This section makes the assumption explicit.  ``domain`` is a free-text
    label that is recorded and never interpreted -- it exists so a run
    artifact says what it trained on.  The ``expect_*`` fields are operator
    *claims* about the traffic; ``None`` means "not declared, infer from the
    measurement" and is the default, because a wrong default claim is worse
    than no claim.  A declared claim that the measured corpus contradicts is
    reported by :mod:`speedlm.workload`.
    """

    #: Free-text traffic label, recorded into every cycle. Never interpreted.
    domain: str = ""
    #: Whether requests are expected to offer tool schemas.
    expect_tools: bool | None = None
    #: Whether requests are expected to carry conversation history.
    expect_multi_turn: bool | None = None
    #: Whether requests are expected to carry a system prompt.
    expect_system_prompts: bool | None = None
    #: Whether requests are expected to carry assistant turns this deployment
    #: did not generate. These cannot be supervised, so the training path
    #: drops the whole row; declaring them says the loss is expected.
    expect_client_supplied_assistant_turns: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str):
            raise ConfigError(
                f"workload.domain must be a string, got {type(self.domain).__name__!r}"
            )
        for name in (
            "expect_tools",
            "expect_multi_turn",
            "expect_system_prompts",
            "expect_client_supplied_assistant_turns",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ConfigError(
                    f"workload.{name} must be a bool or null, got "
                    f"{type(value).__name__!r}"
                )


@dataclass(frozen=True, slots=True)
class TargetConfig:
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        _validate_host(self.host, "target")
        _validate_port(self.port, "target")


@dataclass(frozen=True, slots=True)
class WrapperConfig:
    host: str = "127.0.0.1"
    port: int = 8100

    def __post_init__(self) -> None:
        _validate_host(self.host, "wrapper")
        _validate_port(self.port, "wrapper")


@dataclass(frozen=True, slots=True)
class TraceBufferConfig:
    max_tokens: int = 8_000_000
    max_age_days: float = 14.0

    def __post_init__(self) -> None:
        if _is_bool(self.max_tokens) or not isinstance(self.max_tokens, int):
            raise ConfigError(f"max_tokens must be an int, got {type(self.max_tokens).__name__!r}")
        if self.max_tokens <= 0:
            raise ConfigError(f"max_tokens must be > 0, got {self.max_tokens}")
        if _is_bool(self.max_age_days) or not isinstance(self.max_age_days, (int, float)):
            raise ConfigError(
                f"max_age_days must be numeric, got {type(self.max_age_days).__name__!r}"
            )
        if self.max_age_days <= 0:
            raise ConfigError(f"max_age_days must be > 0, got {self.max_age_days}")


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    """Privacy controls for newly persisted traces."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError(
                f"redaction.enabled must be a bool, got {type(self.enabled).__name__!r}"
            )


class DivergenceCriterion(Enum):
    """Whether the position criterion can discriminate at all.

    ``promotion.min_divergence_token_index`` only means something *relative to*
    ``tuning.correctness_max_tokens``: the threshold divides an observable
    range that the correctness pass itself bounds.  Push the threshold to
    either end of that range and the criterion stops discriminating, without
    any single field looking wrong on its own -- which is exactly how job
    369373 was read as an ordinary rejection.

    * ``CALIBRATED`` -- ``0 < threshold < cap``.  Some offsets classify early
      and some late, which is the regime the threshold was calibrated for.
    * ``SATURATED`` -- ``threshold >= cap``.  Every divergence at every
      observable offset classifies *early* by construction, so with the
      default ``max_output_mismatches = 0`` the gate demands a bitwise
      identical generation over the whole correctness pass.  On hardware that
      is not bitwise reproducible that is a near-certain reject, and the
      rejection says nothing about the drafter.  Job 369373 set both to 128
      and rejected on 48 divergences at offsets 24-127 -- the same benign
      distribution earlier jobs recorded as *late* against a threshold of 16.
    * ``DISABLED`` -- ``threshold == 0``.  No divergence can ever be early, so
      the position criterion is off and any divergence is accepted.  Recorded
      because this is the failure mode in the *other* direction: the gate is
      the only safeguard and there is no post-promotion rollback.

    The two degenerate states are legitimate to configure -- 369373 ran
    ``SATURATED`` deliberately, to exercise the early branch -- so they warn
    rather than raise.  What must not happen is a reader of ``decision.json``
    mistaking a constructed rejection for a drafter-quality one, which is why
    the state is carried into the decision record as well.
    """

    CALIBRATED = "calibrated"
    SATURATED = "saturated"
    DISABLED = "disabled"


def classify_divergence_criterion(
    min_divergence_token_index: int,
    correctness_max_tokens: int | None,
) -> DivergenceCriterion:
    """Classify the promotion/tuning threshold relationship.

    *correctness_max_tokens* is optional because an archived decision written
    before the field existed reads back as ``None``; without the cap there is
    no range to saturate, so only the ``DISABLED`` end is decidable.
    """
    if min_divergence_token_index <= 0:
        return DivergenceCriterion.DISABLED
    if correctness_max_tokens is not None and (
        min_divergence_token_index >= correctness_max_tokens
    ):
        return DivergenceCriterion.SATURATED
    return DivergenceCriterion.CALIBRATED


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Bars a candidate draft head must clear before it is promoted.

    The two knobs play *different* roles, and the defaults are derived from the
    measured dispersion of the live gate, not chosen for symmetry.

    ``min_acceptance_delta_pp`` is the **promotion criterion**.  Improving draft
    acceptance is the entire point of idle tuning, so a candidate that does not
    measurably raise acceptance is not worth shipping however good its clock
    looks.  Acceptance is read from vLLM's ``spec_decode`` counters over a
    deterministic, greedy replay of a fixed held-out suite, so the same suite
    replayed against two arms has *no* timing component and no measured noise:
    on job 368670 the two arms produced byte-identical draft/accept counters
    (1155 drafted, 730 accepted, delta 0.0 pp).  The noise floor is therefore
    the counter quantum itself -- one accepted token out of ~1155 drafted, or
    0.087 pp.  ``1.0`` pp is ~12 accepted tokens, a ~2% relative lift in
    acceptance: comfortably resolvable, and small enough that a genuinely
    better head clears it.

    ``min_throughput_delta_pct`` is a **regression guard**, deliberately
    negative.  It is applied to the *replay* statistic
    (``replay_per_repeat_mean``, see ``GATING_THROUGHPUT_STATISTIC``), so every
    figure quoted below is replay-derived unless explicitly labelled Prometheus;
    the Prometheus decode rate is recorded alongside it for diagnosis but is not
    what the gate compares.  Throughput *is* timing, and it is noisy: across the
    three scored repeats of job 368670 the within-arm **replay** standard
    deviation was 1.80 tok/s (stock) and 0.58 tok/s (candidate), pooled
    1.34 tok/s on a ~76.7 tok/s mean, giving a standard error on the arm-to-arm
    delta of 1.43% at three repeats.  The one-sided 95% minimum detectable
    effect is ~3.0%, so any *positive* threshold below that cannot be earned on
    merit -- it is cleared by whichever way the timing noise happened to fall.
    Job 368670 is the worked example: its **replay** delta was +0.96%
    (t=0.67, p~0.28) -- the Prometheus delta happened to agree at +0.96% -- and
    it would have promoted under the old ``2.0`` bar roughly one run in seven by
    chance alone; drop its first scored repeat and the same data reads -0.78%.
    Requiring throughput to *prove* an improvement is therefore not achievable
    at this sample size, so the gate instead requires only that throughput not
    visibly regress.  ``-2.0`` sits ~1.8 standard errors below zero at five
    repeats (SE 1.10%), so ordinary jitter does not trip it, while the real
    regression this gate has already caught -- the un-warmed candidate arm of
    job 368648, at **-17.5% replay** (-19.2% on the Prometheus decode rate) --
    is still ~16 standard errors past it.

    That 1.10% figure is now superseded, in the *reassuring* direction.  It was
    extrapolated from job 368670, which predates ``benchmark_max_tokens`` and
    ``benchmark_concurrency``; measured directly on the two capped, concurrent
    runs, the standard error on the arm-to-arm replay delta over five repeats is
    0.607% (job 369161, Qwen3-8B) and 0.640% (job 369162, gpt-oss-20b).  ``-2.0``
    therefore sits 3.3 and 3.1 standard errors below zero, not 1.8.  Do not read
    that as slack to spend: the same two runs show most of the surviving
    dispersion is *drift*, not noise (see ``IdleTuningConfig.benchmark_repeats``),
    and ``sd/sqrt(n)`` overstates how much a shorter measurement would keep.

    A caveat on ``min_acceptance_delta_pp``'s derivation: the "no measured
    noise" claim above rests on job 368670's two arms reporting byte-identical
    counters, and on every archived run reporting acceptance as a *single*
    pooled window per arm.  That was n=1 per arm, so it could not have shown
    dispersion even if dispersion existed -- job 369005's five per-repeat rows
    all carry ``0.40302518489174166``, which is
    ``(539777-88070)/(1343865-223074)`` stamped five times, not five
    measurements.  The gate now samples acceptance once per repeat and
    publishes the vector's standard deviation
    (``Decision.stock_acceptance_stdev``), so the threshold becomes checkable
    for the first time.  It is deliberately *not* being retuned here, because
    no artifact in the archive contains the number that would justify a change.
    What the archive does bound is the floor: job 369005 drafted ~224k tokens
    per repeat, so pure counting noise on a 0.403 rate is 0.104 pp per repeat,
    putting the standard error on the arm-to-arm delta at 0.066 pp over five
    repeats and 1.0 pp at ~15 standard errors.  Drafted tokens are correlated
    within a request, so the real dispersion will exceed that floor; 1.0 pp
    remains a meaningful bar (>= 3 standard errors) for any per-repeat standard
    deviation up to ~0.52 pp, and stops being one (< 1.6 standard errors) above
    ~1.0 pp.  Recalibrate from the first run whose ``per_repeat`` acceptance
    column actually varies, not from this comment.

    Those runs have now happened, and they close the question rather than
    answering it.  Job 369161 (Qwen3-8B) is the first record whose acceptance
    column varies at all: per-repeat deltas of -0.199, -0.151, -0.171, -0.199,
    -0.226 pp, sample sd 0.0287 pp, standard error on the delta of means
    0.0174 pp -- so 1.0 pp sits at ~57 standard errors, far inside the
    "<= 0.52 pp sd" band the paragraph above predicted, and the ">= 3 SE" claim
    held with two orders of magnitude to spare.  Job 369162 (gpt-oss-20b) does
    not vary at all: all five repeats read -0.5674 pp, both
    ``*_acceptance_stdev`` fields are exactly 0.0, and the standard error is
    undefined.

    The right conclusion is *not* that the acceptance measurement is superb.
    Reading the raw ``spec_decode`` counter deltas out of
    ``gate-metrics/*-after-repeat-*.prom.gz`` shows why: on job 369162 every
    repeat of the candidate arm drafted 86830 tokens and accepted 30400, and
    every repeat of the stock arm drafted 86325 and accepted 30713 -- the same
    integers five times.  Job 369161's candidate alternates between exactly two
    states (73530/27362 and 73680/27398), a 150-token wobble in 73.6k that comes
    from batch composition at ``benchmark_concurrency``, not from the head.
    Greedy replay of a frozen suite against a frozen draft is deterministic by
    construction, so repeats do not *sample* acceptance; a zero standard
    deviation here is evidence of no measurement, not of a tight one.  The gate
    now says so in the record -- see
    :class:`speedlm.gate.decide.DispersionBasis` -- and
    ``IdleTuningConfig.benchmark_repeats`` is justified from throughput
    dispersion alone, because throughput is the only thing repeats measure.

    Both values remain fully configurable via ``promotion`` in ``config.json``;
    these are defaults, not policy.  Note that setting them to ``0.0``/``0.0``
    reduces the gate to "not measurably worse", which promotes on noise
    indefinitely -- see docs/DEMO.md on why lowering the gate is the failure mode
    this system exists to prevent.
    """

    #: **No longer the promotion criterion.  Reported, not enforced.**
    #:
    #: ``acceptance_rate`` is ``accepted_tokens / drafted_tokens``, and
    #: ``drafted_tokens == num_drafts * k`` for a draft chain of depth ``k``, so
    #: the statistic is algebraically ``(mean_accepted_length - 1) / k``.  The
    #: ``k`` in the denominator makes it **incomparable across draft depths**,
    #: and the archive shows the gate paying for that: Qwen3-8B at ``k=3``
    #: scores 0.380 while gpt-oss-20b at ``k=5`` scores 0.356 -- yet gpt-oss
    #: emits 2.82 tokens per verifier step against Qwen's 2.14, i.e. it is 28%
    #: better at the thing speculative decoding is for, and on a like-for-like
    #: ``k=3`` basis it would score 0.486.
    #:
    #: The failure that forced this change is not the mis-ranking but its
    #: direction.  gpt-oss's per-position conditional acceptance is flat to
    #: *rising* at the last drafted position (0.689 -> 0.715 at position 5),
    #: which says ``k=5`` is too shallow and throughput is being left unclaimed.
    #: Raising ``k`` would raise tokens/s while *lowering* ``acceptance_rate``
    #: to roughly 0.31 -- a -4.6 pp delta, a hard reject under this bar.  The
    #: primary promotion criterion would have vetoed the single largest speedup
    #: available.  See :attr:`min_accepted_length_delta` for what gates now.
    #:
    #: It is still recorded on every decision, and every archived run in
    #: ``/data/ryan.kim/speedlm-runs`` is described by it, so it keeps its name,
    #: its units (percentage points of acceptance rate) and its default.  What
    #: it no longer does is decide.  ``Decision.acceptance_criterion`` names the
    #: statistic that actually gated, so a reader never has to infer it.
    min_acceptance_delta_pp: float = 1.0
    #: The promotion criterion: required lift in **mean accepted length**, in
    #: tokens emitted per verifier forward pass.
    #:
    #: ``mean_accepted_length = 1 + accepted_tokens / num_drafts`` -- tokens per
    #: *step*, not per *drafted position*.  It carries no ``k`` in its
    #: denominator, so two runs at different draft depths are directly
    #: comparable, and it moves in the same direction as the speedup: at a fixed
    #: cost per verifier step, throughput is proportional to it.  vLLM already
    #: computes it (``speedlm.gate.metrics.MetricsDelta.mean_accepted_length``)
    #: once per scored repeat, so nothing new is measured to gate on it.
    #:
    #: Why not gate on tokens/s directly, given that it is the quantity we
    #: actually want?  Because the gate already thresholds throughput, and the
    #: two criteria buy different things.  ``min_throughput_delta_pct`` is a
    #: *regression guard* (see below) precisely because a positive throughput
    #: bar cannot be earned on merit at this sample size: the one-sided 95%
    #: minimum detectable effect is ~3.0%, and the measured standard error on
    #: the arm-to-arm replay delta is 0.607-0.640% over five repeats.  Promoting
    #: on tokens/s would mean re-deriving a positive bar this file has already
    #: shown to be unreachable, and would collapse both criteria onto one noisy,
    #: timing-confounded statistic.  Acceptance-side statistics have no timing
    #: component at all -- greedy replay of a frozen suite against a frozen
    #: draft is deterministic by construction -- so they isolate the head's
    #: quality from batch composition, co-tenancy and drift.  Keeping the pair
    #: asks two independent questions: "does the head emit more tokens per
    #: verifier step" (this bar) and "did that not cost wall-clock"
    #: (``min_throughput_delta_pct``).  That pair is also what keeps a deeper
    #: draft chain honest: a larger ``k`` mechanically raises mean accepted
    #: length, and the throughput guard is what catches the case where the extra
    #: depth costs more per step than it buys.
    #:
    #: **Calibration.**  ``0.05`` tokens/step is the old ``1.0`` pp bar
    #: re-expressed at ``k=5`` (``delta_mal = k * delta_acc``), i.e. the
    #: strictest reading of the outgoing bar across the two depths in the
    #: archive -- at ``k=3`` the old bar was only 0.03 tokens/step, so this is
    #: strictly harder for Qwen and identical for gpt-oss.  No archived
    #: rejection can therefore flip to a promotion.  Replayed over every
    #: ``terminal-decision.json`` in ``/data/ryan.kim/speedlm-runs``:
    #:
    #: ===============================  ===  ============  =========  =========================
    #: run                                k  acc delta pp  delta MAL  verdict (old -> new)
    #: ===============================  ===  ============  =========  =========================
    #: multicycle-20260731T190000Z        5       +6.8286    +0.3414  promote -> promote
    #: full-gptoss-head (run2)            5       -1.2028    -0.0601  reject(acceptance) -> same
    #: gptoss-pinned-2                    5       -0.5674    -0.0284  reject(acceptance) -> same
    #: qwen-cross-20260801T014500Z        3       -0.5540    -0.0166  reject(acceptance) -> same
    #: qwen-pinned-2                      3       -0.1890    -0.0057  reject(acceptance) -> same
    #: early-divergence-3                 5      (-1.2714)  (-0.0636) reject(output_mismatch)
    #: d993eee-qwen-idle                  3       (+0.8274)  (+0.0248) reject(output_mismatch)
    #: qwen-cross-20260731T235000Z        3      (-0.6423)  (-0.0193) reject(output_mismatch)
    #: full-qwen-head (run2)              3      (-0.1714)  (-0.0051) reject(output_mismatch)
    #: qwen-pinned-1                      3      (-0.2380)  (-0.0071) reject(high_invalid_rate)
    #: ===============================  ===  ============  =========  =========================
    #:
    #: Parenthesised deltas are recomputed from the ``per_repeat`` column: those
    #: five runs short-circuit *above* the delta computation and are untouched
    #: by this threshold either way.  The five runs that do reach it split
    #: +0.341 against -0.060 and below -- a gap of a factor of five to the bar
    #: in the promoting direction and of nearly one in the rejecting direction,
    #: so the archive does not constrain the bar tightly and 0.05 is chosen for
    #: continuity with the outgoing one rather than to thread that gap.
    #:
    #: Against noise: job 369161 is the only archived run whose acceptance
    #: column varies, at a per-repeat sample sd of 0.0287 pp and a standard
    #: error on the delta of means of 0.0174 pp; at ``k=3`` those are 0.00086
    #: and 0.00052 tokens/step, putting 0.05 at ~96 standard errors.  The
    #: counting-noise floor from job 369005 (~224k drafted tokens per repeat,
    #: SE 0.066 pp over five repeats) is 0.00198 tokens/step at ``k=3``, i.e.
    #: ~25 standard errors.  In relative terms 0.05 asks for a 2.4% lift on
    #: Qwen's 2.12 tokens/step and a 1.8% lift on gpt-oss's 2.78 -- the same
    #: order the outgoing bar asked for, but now in a unit that does not move
    #: when ``k`` does.  The counterfactual that motivated the change clears it
    #: easily: gpt-oss at ``k=7`` would land near 3.17 tokens/step, ``+0.39``.
    #:
    #: Recalibrate from the first run whose ``per_repeat`` accepted-length
    #: column actually varies at a *different* ``k``, not from this comment.
    min_accepted_length_delta: float = 0.05
    #: Negative by design: a floor on regression, not a required speedup.
    min_throughput_delta_pct: float = -2.0
    #: How far into a generation the candidate must stay token-identical to
    #: stock before a divergence is treated as behaviourally harmless.
    #:
    #: The gate used to compare whole response strings for exact equality with
    #: zero tolerance.  That check is unsound on realistic generations, and job
    #: 369005 is the proof: 82-87 of 103 held-out contexts "mismatched" on a
    #: candidate whose acceptance was within 0.65 pp of stock, at an average of
    #: ~1602 generated tokens per request.  Rejection sampling at temperature 0
    #: is *mathematically* lossless, not *bitwise* lossless -- vLLM's target
    #: forward pass is not bitwise reproducible when batch composition varies,
    #: so a per-token divergence hazard ``p`` turns into a whole-string failure
    #: probability of ``1 - (1 - p) ** L``.  At L~1602 the observed 80%
    #: mismatch rate implies ``p ~ 1.0e-3`` at replay concurrency 8; the same
    #: measurement at concurrency 1 (commit cbaff80) put ``p`` roughly 12x
    #: lower, ~8.5e-5.  Truncation and prefix-cache asymmetry were both ruled
    #: out on that run: 18-19 length-finishes against 82-87 mismatches, and a
    #: 71.1% prefix hit rate on *both* arms.
    #:
    #: So the question a correctness check can actually answer is not "are the
    #: two strings equal" but "how soon do they part".  A drafter that is
    #: genuinely broken -- wrong vocabulary mapping, mis-loaded weights, a head
    #: trained against a different verifier -- makes the verifier reject
    #: immediately and the sequences part within a handful of tokens.  Float
    #: non-determinism parts them, on average, around token ``1/p``.
    #:
    #: 16 is chosen against the concurrency-1 hazard the correctness pass
    #: actually runs at.  A benign context parts within its first 16 tokens
    #: with probability ``1 - (1 - 8.5e-5) ** 16 = 1.4e-3``, so across a
    #: 103-context suite the expected number of benign early divergences is
    #: 0.14 -- i.e. roughly one gate run in eight rejects a good candidate.
    #: That is the price of the fail-closed posture, and it is the right price
    #: here: a rejection costs one tuning cycle, while a false promotion is
    #: permanent because there is no post-promotion rollback.  Raising this to
    #: 32 doubles the false-reject rate to ~25%; lowering it to 8 halves it to
    #: ~7% but starts to admit heads that only survive the formulaic opening of
    #: an answer.  Set to 0 to disable the position criterion entirely and
    #: accept any divergence, which is *not* recommended.
    min_divergence_token_index: int = 16

    def __post_init__(self) -> None:
        _validate_float_gte(self.min_acceptance_delta_pp, "min_acceptance_delta_pp", 0)
        _validate_float_gte(
            self.min_accepted_length_delta, "min_accepted_length_delta", 0
        )
        # No lower bound of zero here: a negative value is the intended
        # regression-guard form ("reject anything more than N% slower").
        _validate_float(self.min_throughput_delta_pct, "min_throughput_delta_pct")
        _validate_int_gte(self.min_divergence_token_index, "min_divergence_token_index", 0)


@dataclass(frozen=True, slots=True)
class ValLossPreFilterConfig:
    """Cheap pre-filter to skip expensive benchmarks on non-improving candidates.

    The Speculators trainer computes validation loss on an internal 10% split
    of the training data.  This metric is free (no engine restart required) and
    serves as an early indicator that a candidate is worth benchmarking.

    IMPORTANT: this is a COST FILTER, not a promotion criterion.  The acceptance
    gate remains the sole authority on whether a candidate gets promoted.  A
    candidate that passes this pre-filter still must clear the acceptance gate.

    The validation split is the trainer's INTERNAL 10% holdout — it is
    independent of speedlm's held-out benchmark split.  Those benchmark
    contexts must NOT be used for validation: src/speedlm/training/split.py
    raises Eagle3Error on train/benchmark overlap.  DO NOT change this to use
    the benchmark split.
    """

    enabled: bool = True
    #: Minimum improvement in validation loss to justify a benchmark.
    #: A positive value means the candidate must be strictly better (lower loss)
    #: than the incumbent by at least this delta.  The incumbent's val_loss is
    #: read from its artifact manifest; if neither side has a val_loss, the
    #: pre-filter falls through to the benchmark (fail open).
    min_improvement: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError(
                f"tuning.val_loss_prefilter.enabled must be a bool, "
                f"got {type(self.enabled).__name__!r}"
            )
        _validate_float_gte(self.min_improvement, "tuning.val_loss_prefilter.min_improvement", 0)


@dataclass(frozen=True, slots=True)
class IdleTuningConfig:
    """Production composition settings for the opt-in idle tuner."""

    min_trace_records: int = 32
    #: Accumulation threshold: the corpus must reach this size before a tuning
    #: cycle is allowed to fire.  Matches ``training_window_records`` (256) by
    #: default so the cycle trains on the full window rather than a partial one.
    #: ``min_trace_records`` (32) remains as the lower-bound sanity floor; the
    #: accumulation gate is a separate, higher bar that prevents training on a
    #: corpus too small to produce a meaningful gradient step.
    min_corpus_records: int = 256
    poll_interval_seconds: float = 1.0
    #: Consecutive idle polls required before a cycle may arm.
    #:
    #: ``IdleDetector.should_tune`` is an instantaneous predicate over one
    #: reading of the activity watermark, and arming commits the gateway to a
    #: quiesce/sleep/train/benchmark sequence that costs tens of minutes of
    #: closed-gate time.  Job 369040 armed on a single sample and was preempted
    #: 0.1s into extraction, paying a full rollback restart for a cycle that had
    #: done no work.  Requiring the reading to repeat turns one sample into a
    #: short observation window, at a cost of
    #: ``(idle_confirmations - 1) * poll_interval_seconds`` of extra latency
    #: before a genuinely idle system starts tuning.
    #:
    #: Three, against the default 1.0s poll, is two extra seconds -- negligible
    #: beside the cycle it guards, and enough to reject a gap between two bursts
    #: of the same conversation.  Set to 1 to restore single-sample arming.
    idle_confirmations: int = 3
    #: Quiet period enforced after a cycle that was preempted or failed.
    #:
    #: The only thing stopping back-to-back attempts was the watermark dedupe in
    #: :class:`speedlm.tuner.service.TunerService`, and a preempting request
    #: writes its own trace record -- which advances the watermark and so
    #: defeats the dedupe by construction.  Job 369040 re-armed 9.2s after a
    #: preempted cycle and paid a second full rollback.  A cycle that has just
    #: proved the system is not actually idle (or not actually healthy) must
    #: wait before spending another engine restart on the same conclusion.
    #:
    #: 600s is well short of a cycle's own duration, so a genuinely idle machine
    #: loses at most one attempt window; it applies only to PREEMPTED/FAILED
    #: outcomes, so a normal promote or reject is never delayed.
    retry_cooldown_seconds: float = 600.0
    #: Minimum gap between re-attempts of a restore that has already failed.
    #:
    #: Deliberately *not* the retry cooldown above, and deliberately much
    #: shorter.  The cooldown exists to stop a healthy system from re-paying an
    #: engine restart to reach a conclusion it just reached; this exists for the
    #: opposite situation, where the engine is serving a draft the durable
    #: active pointer does not name and every second of delay is a second of
    #: unvalidated throughput.  ``PREEMPTED`` arms the 600s cooldown, so a
    #: preemption whose rollback could not respawn would otherwise wait out the
    #: whole quiet period before anything looked at it again --
    #: :meth:`speedlm.tuner.service.TunerService._recover_unrestored_serving`
    #: therefore runs *ahead* of the cooldown check and is paced by this knob
    #: instead.
    #:
    #: 30s is a floor, not a schedule.  A real re-attempt is a vLLM restart at
    #: ~100-105s, so the interval is only reachable at all when the restore
    #: fails fast -- an engine that is refusing immediately, which is precisely
    #: the case that would otherwise spin at the 1s poll interval and fill the
    #: log.  Set to 0 to retry on every poll.
    serving_recovery_interval_seconds: float = 30.0
    held_out_fraction: float = 0.2
    #: Scored suite passes per arm.  Five, not three, because the gate's
    #: throughput regression guard is only as trustworthy as its standard
    #: error: job 368670's pooled within-arm replay dispersion of 1.34 tok/s on a
    #: ~76.7 tok/s mean puts the arm-to-arm standard error at 1.43% over three
    #: repeats but 1.10% over five, which moves the -2.0% guard from 1.4 to 1.8
    #: standard errors clear of zero.  The cost is four extra suite passes per
    #: tuning cycle.
    #:
    #: A previous revision of this comment claimed "Engine restarts, not
    #: repeats, dominate that phase", extrapolating from job 368670's ~8s suite
    #: pass.  That is false for the gpt-oss profile: on job 368959 a *single*
    #: 103-context pass ran for over 1720s while two engine restarts cost ~90s
    #: each, so repeats outweighed restarts by more than an order of magnitude
    #: and the fixed 1800s benchmark deadline expired inside the stock arm's
    #: warmup.  Suite-pass cost scales with suite size, output length and
    #: replay concurrency, so it is not a constant that can be assumed small;
    #: ``benchmark_concurrency`` is what makes it affordable, and
    #: :func:`speedlm.tuner.orchestrator.derive_benchmark_timeout` is what
    #: keeps the deadline sized to it.
    #:
    #: **Five was re-examined against measured per-repeat data and kept.**  The
    #: case for cutting it is strong on cost -- benchmarking is the dominant
    #: remaining downtime (job 369161: 1043.4s of a 1546.3s cycle; job 369162:
    #: 1808.5s of 2386.6s), each repeat is one of the six suite passes an arm
    #: runs, and 5 -> 3 would return roughly a third of that -- about 300s/cycle
    #: on Qwen3-8B and 500s+ on gpt-oss-20b.  It is nonetheless the wrong trade,
    #: for two reasons, in increasing order of importance.
    #:
    #: First, repeats buy nothing for *acceptance* and never did.  Greedy replay
    #: of a frozen suite is deterministic by construction: job 369162's five
    #: repeats produced bit-identical ``spec_decode`` counter deltas in both
    #: arms, and job 369161's candidate alternated between exactly two states.
    #: See ``PromotionConfig``'s docstring for the counters.  So the repeat count
    #: has to be justified from throughput dispersion alone -- there is no second
    #: quantity it is also serving.
    #:
    #: Second, and decisively: truncating the repeat vector is not merely
    #: noisier, it is *biased*.  The candidate arm runs first
    #: (``benchmark_candidate_arm_first``) and is still warming when its
    #: measurement window opens.  Its per-repeat throughput trends +0.63%/repeat
    #: on job 369161 and +0.80%/repeat on job 369162 -- 83% and 94% of that arm's
    #: per-repeat variance is that monotone drift, not noise -- while the stock
    #: arm shows no such structure (-0.42%/repeat and +0.13%/repeat).  Replaying
    #: the recorded arrays over a shortened window moves the *gated* delta
    #: toward the reject bar every time: on job 369161 from +0.160% (n=5) to
    #: -1.068% (n=3), on job 369162 from -1.243% to -1.715%, and at n=2 job
    #: 369162 reads -2.485%, past ``min_throughput_delta_pct`` entirely.  Losing
    #: 1.2 points of a 2.0-point guard to a truncation artefact is a far larger
    #: error than the ~0.2 points of extra standard error it would have saved,
    #: and it errs toward rejecting good heads -- which, with no post-promotion
    #: rollback, is the direction the gate is allowed to fail in but not the
    #: direction that makes idle tuning worth running.
    #:
    #: The measured standard errors are the same story quantitatively: 0.607%
    #: (369161) and 0.640% (369162) over five repeats, putting the -2.0% guard
    #: 3.3 and 3.1 standard errors clear of zero.  Scaling by ``sqrt(5/3)`` puts
    #: n=3 at 2.5 and 2.4 -- and that scaling is itself optimistic, because it
    #: treats the drift above as exchangeable noise.
    #:
    #: Adaptive stopping ("halt once the standard error is tight enough") was
    #: considered and rejected on the same evidence.  A monotonically warming
    #: arm produces its *smallest* consecutive-repeat spread while it is still
    #: cold, so an SE-triggered stop fires early and preferentially, locking in
    #: exactly the bias described above.  It would add a control loop that makes
    #: the measurement worse.
    #:
    #: Where the saving actually is: not here.  These arms are under-warmed at
    #: ``warmup_repeats=1``, so the honest direction for this knob is *up*, not
    #: down.  Retuning warmup needs a run that scores enough repeats to see
    #: where the candidate trend flattens -- the two runs above only show that
    #: it has not flattened by repeat five -- so it is deliberately not being
    #: changed from a rearrangement of the same two arrays that motivated it.
    #: ``Decision.candidate_throughput_trend_pct_per_repeat`` is now published on
    #: every decision so that measurement accumulates by itself.
    benchmark_repeats: int = 5
    #: Unscored suite passes each arm runs after activation, before its
    #: measurement window opens.
    #:
    #: This existed only as a :class:`speedlm.gate.runner.BenchmarkGateRunner`
    #: constructor default of 1, which production never passed, so the one knob
    #: that ``benchmark_repeats``' own analysis names as "the honest direction
    #: is *up*" was the one knob an operator could not turn.  The default here
    #: is 1 precisely so that surfacing it changes nothing that runs today.
    #:
    #: A warmup pass and a scored repeat are *the same suite pass at the same
    #: cost*: :meth:`~speedlm.gate.runner.BenchmarkGateRunner._warmup` calls the
    #: same replay under the same ``benchmark_max_tokens``, and
    #: :func:`speedlm.tuner.orchestrator.derive_benchmark_timeout` charges them
    #: identically as ``arms x (warmup_repeats + repeats) x num_contexts``.
    #: That equality is what makes the trade arithmetic, not a judgement call:
    #: on jobs 369161/369162 one pass-index costs ~174s (Qwen3-8B) and ~301s
    #: (gpt-oss-20b) of the 1043.4s and 1808.5s benchmark phases, so moving a
    #: pass from ``benchmark_repeats`` to here is *exactly* a wash and only
    #: ``warmup_repeats + benchmark_repeats < 6`` saves wall clock at all.
    #:
    #: Raising this is therefore not an efficiency lever.  It is a *bias* lever:
    #: it buys an unbiased delta, at the price of the extra passes, and only
    #: once the warming curve is known to flatten inside the window it opens.
    #: See ``Decision.candidate_throughput_flat_from_repeat``, which is what
    #: measures that, and raise ``benchmark_repeats`` -- already unbounded above
    #: -- to characterise the curve rather than raising this one blind.
    warmup_repeats: int = 1
    #: Held-out requests kept in flight per arm during a gate replay.
    #:
    #: Before this existed the replay was strictly serial and the served engine
    #: never saw more than one request at a time.  Eight matches
    #: ``concurrency`` below, which is the degree this codebase already drives
    #: a vLLM engine at; the served engine sets no ``--max-num-seqs``, so it
    #: runs vLLM's default scheduler width and can absorb it.  Set to 1 to
    #: restore single-stream measurement.
    benchmark_concurrency: int = 8
    #: Output cap, in tokens, for the gate's separate output-correctness pass.
    #:
    #: The gate runs two different replays for two different jobs.  The
    #: throughput/acceptance pass runs at ``benchmark_concurrency`` under
    #: ``benchmark_max_tokens``, because that is what steady-state serving looks
    #: like and the throughput threshold is calibrated against its dispersion.
    #: The correctness pass runs at concurrency 1 with this cap, because bitwise
    #: agreement is a property of a *single-stream, bounded* generation: job
    #: 369005 replayed unbounded (averaging ~1602 tokens against the 4096
    #: model-len cap) at concurrency 8, which is precisely the regime where
    #: float non-determinism accumulates fastest -- see
    #: ``PromotionConfig.min_divergence_token_index``.
    #:
    #: 128 is eight times that divergence threshold: long enough that a head
    #: cannot pass by reproducing only the formulaic opening of an answer, and
    #: short enough that the correctness pass costs ~1/12th of an unbounded
    #: pass per context even though it gives up all batching.
    correctness_max_tokens: int = 128
    #: Output cap, in tokens, for the gate's throughput/acceptance pass.
    #:
    #: This pass used to send no ``max_tokens`` at all, which was never actually
    #: "uncapped": it was bounded at ``max_model_len`` minus the prompt, i.e. by
    #: an accident of the served model.  Job 369005 measured gpt-oss-20b
    #: averaging 2091 completion tokens per request with 25.6% of requests
    #: stopping at ``finish_reason=length``, against Qwen3-8B's 1602 tokens and
    #: 3.5%.  A model-dependent implicit truncation is a worse statistic than an
    #: explicit uniform one, so this replaces it rather than introducing it.
    #:
    #: 512 is the cap the live harness puts on production traffic
    #: (``tests/e2e/test_live_idle_tuning.py``), and 93.6% of seed responses hit
    #: it, so the gate had been replaying roughly 4.4x longer than the system it
    #: gates ever serves.  Measured effect of the cap on the benchmark phase:
    #: gpt-oss 4764s -> 1385s, Qwen 2318s -> 924s.
    #:
    #: Honesty about the bias this introduces.  *Acceptance* is biased, mildly:
    #: late-sequence tokens are modestly harder to draft, measured at the repeat
    #: tails as -3.6 pp for the stock arm and -5.6 pp for the candidate, so
    #: truncating the tail removes a penalty that differs by arm and shifts the
    #: reported acceptance delta by roughly +0.1 pp -- about 10% of
    #: ``PromotionConfig.min_acceptance_delta_pp``.  The gate is therefore very
    #: slightly more permissive on acceptance than an unbounded pass would be.
    #: *Throughput* is not biased: shortening decode raises the prefill share of
    #: every request, but it raises it identically in both arms (the suite,
    #: prompts and cap are shared), and the gating statistic is an arm-to-arm
    #: ratio, so a common-mode inflation cancels out of the threshold.
    #:
    #: The +0.1 pp figure above understates it, and the first capped run says so.
    #: Reading the gate-metrics counters directly: SLURM 369147 (capped) drew
    #: 504.2 completion tokens per request against qwen-cross-20260801T014500Z's
    #: 1588.9 uncapped, and its stock arm accepted 0.3752 against the uncapped
    #: run's 0.3995 -- 2.4 pp *lower*, not higher, so late tokens in a reasoning
    #: model's ``<think>`` block are easier to draft than the answer that
    #: follows, and truncation removes the easy part.  On the gated *delta* the
    #: two runs read -0.238 pp (capped) against -0.554 pp (uncapped): the cap
    #: shifts the delta by about +0.32 pp, roughly a third of
    #: ``PromotionConfig.min_acceptance_delta_pp``, in the permissive direction.
    #: That is a cross-run comparison of two different candidates, so it bounds
    #: the effect rather than measuring it, but it is three times the earlier
    #: estimate and should be treated as the working number.
    #:
    #: 512 is nevertheless kept, for a reason stronger than "unbiased": it is
    #: the cap live traffic runs under, so a capped gate measures the regime the
    #: system actually serves, and raising the gate's cap above the serving cap
    #: would measure a regime that never occurs in production.  It is a property
    #: of the harness, not of the model, so it needs no per-model override --
    #: which matters, because Qwen3-8B and gpt-oss-20b differ in almost every
    #: other respect here.  Reverting to an uncapped pass costs the benchmark
    #: phase 969.5s -> 2713s per cycle on Qwen (2.8x); if the acceptance bias
    #: above ever needs eliminating rather than bounding, raise the *serving*
    #: cap first and let this follow it.
    benchmark_max_tokens: int = 512
    #: Whether the gate measures the candidate arm before the stock arm.
    #:
    #: The benchmark's first act used to be an activation onto the *stock*
    #: draft, which threw away the engine ``CANDIDATE_STARTING`` had just spent
    #: ~110s building and left the benchmark ending on the candidate -- so a
    #: rejecting cycle then paid a third restart to roll back.  Running the
    #: candidate arm first reuses the engine that is already up and ends the
    #: benchmark on stock, which is exactly the draft a rejection wants left
    #: serving: two of the four per-cycle vLLM startups disappear (~213s).
    #:
    #: The tradeoff is order bias: whichever arm runs first meets a colder
    #: machine.  Each arm already restarts the engine and runs its own warmup
    #: pass before its measurement window opens, which is what the existing
    #: stock-first order relied on, so the mitigation is unchanged -- only which
    #: arm it protects moves.  The residual bias is in the *conservative*
    #: direction: any cold-start cost now lands on the candidate, so it can only
    #: make the gate reject a good draft, never promote a bad one.  Set to false
    #: to restore the historical stock-first order at the cost of two restarts.
    benchmark_candidate_arm_first: bool = True
    #: How many of the newest trace records one cycle may train on.
    #:
    #: Trace selection is a sliding window, not a full rescan: without a bound
    #: every cycle re-extracted and re-trained on the entire corpus, which is
    #: why cycle 1 and cycle 2 of the archived runs produced byte-identical
    #: trace snapshots while the watermark advanced by a single record.  The
    #: window is what makes a cycle cost O(recent traffic) instead of
    #: O(everything ever captured).
    #:
    #: 256 is chosen against ``min_trace_records`` (32): eight arming
    #: thresholds of history is enough that the per-cycle training
    #: distribution is a stable sample of recent traffic rather than a
    #: high-variance snapshot, while still bounding hidden-state extraction --
    #: the stage the window actually pays for -- to a fixed ceiling.  Set to
    #: null to restore the unbounded full-corpus scan.
    training_window_records: int | None = 256
    #: Whether each cycle warm-starts from the artifact currently serving.
    #:
    #: Off, every cycle re-runs the same one-shot fine-tune of the profile's
    #: stock speculator and discards the previous cycle's promoted head, so
    #: learning cannot accumulate across cycles at all.  That was the behaviour
    #: before this knob existed, and it was not a decision -- the warm start was
    #: simply captured at composition time and never consulted the registry
    #: again, the same defect ``benchmark_candidate_arm_first``'s neighbour
    #: ``stock_draft`` carried on the gate side.
    #:
    #: **On by default**, and the argument is the measured record rather than
    #: the premise.  Four realistic-data runs across two models produced a
    #: candidate *worse* than stock every time -- acceptance deltas of -0.189,
    #: -0.554, -0.567 and -1.27 pp -- and each was a single ~409-record
    #: fine-tune of a professionally trained speculator from a standing start.
    #: A one-shot pass over a few hundred records is not enough signal to beat
    #: that starting point, and re-taking it every cycle cannot become enough,
    #: because the cycles do not add up.  Compounding is the only configuration
    #: in which the product's stated premise is even testable, and it has never
    #: been exercised.
    #:
    #: The safety argument for defaulting it on is that the gate is unaffected.
    #: Promotion still requires beating the *current incumbent* on a held-out
    #: suite, so a worse head is rejected whether it was trained from stock or
    #: from the incumbent; what changes is only where training starts.  What the
    #: gate does *not* cover is drift -- see ``warm_start_max_chain_depth``.
    #:
    #: Set to false to restore the historical from-stock-every-cycle behaviour,
    #: which is also the configuration to reproduce an archived run in.
    compounding_warm_start: bool = True
    #: Promotions deep the warm-start chain may get before a forced re-baseline.
    #:
    #: There is no post-promotion rollback in this system: the gate is the only
    #: safeguard, and it is a *marginal* test -- each candidate beats the head
    #: before it, on a suite split from the same captured traffic.  Marginal
    #: gains can therefore accumulate into a head that is excellent on this
    #: deployment's recent traffic and quietly worse in general, and no arm of
    #: the gate can see that, because both arms are scored on traffic drawn
    #: from the same distribution the drift is toward.  Speculative decoding is
    #: lossless, so the exposure is throughput on unlike traffic, never wrong
    #: answers -- but it is unmeasured and it compounds.
    #:
    #: Setting this to *N* makes a cycle whose incumbent is already *N*
    #: artifacts deep train from the stock speculator instead.  That costs one
    #: cycle, and it buys the only drift signal available: if a from-stock head
    #: *beats* an N-deep incumbent on the gate's own suite, the chain has
    #: regressed on the very traffic it was specialising for.
    #:
    #: **Null (unbounded) by default, deliberately.**  No compounding cycle has
    #: ever run, so any *N* would be invented rather than measured, and a bound
    #: pointed at a hazard nobody has yet observed would spend a cycle in
    #: ``max_chain_depth`` to buy nothing.  Operators running long unattended
    #: chains should set it -- 5 to 10 is a reasonable place to start -- and the
    #: honest way to pick it is to watch what the re-baseline cycle reports.
    warm_start_max_chain_depth: int | None = None
    #: Immutable commit SHA of the verifier the cycle trains and benchmarks
    #: against.  Left null it is resolved from the Hub once per composition and
    #: pinned for the process; set it explicitly to reproduce an archived run.
    verifier_revision: str | None = None
    speculators_repo: str | None = None
    training_python: str | None = None
    vllm_python: str | None = None
    prepared_validator_script: str | None = None
    sequence_length: int = 16_384
    #: Optimizer learning rate for draft-head training, bounded by
    #: :data:`MAX_LEARNING_RATE`.  Defaults to the Speculators reference recipe
    #: (:data:`REFERENCE_LEARNING_RATE`); the previous default of ``1e-5`` was
    #: 10x below it and produced checkpoints bit-identical to their
    #: initialisation in every norm tensor.
    learning_rate: float = REFERENCE_LEARNING_RATE
    epochs: int = 1
    #: Requests kept in flight by the Speculators *offline hidden-state
    #: extraction* step (``data_generation_offline.py --concurrency``).
    #:
    #: Named ``concurrency`` until it was found to be a trap: run configs set
    #: ``tuning.concurrency`` believing it controlled the gate's replay degree,
    #: while the gate reads ``benchmark_concurrency`` and never consults this
    #: field at all.  Job 369006's config recorded ``concurrency: 4`` and the
    #: gate replayed at 8, so the archived config described traffic that never
    #: happened.  The two knobs drive different processes -- this one the
    #: training-side extraction engine, ``benchmark_concurrency`` the served
    #: engine under gate replay -- so the fix is the name, not a rewiring; see
    #: ``from_dict`` for the migration error the old key now raises.
    extraction_concurrency: int = 8
    training_port: int = 8_131
    #: Hard byte limit on the per-cycle scratch directory.
    #:
    #: Derived, not chosen: scratch is sized by the hidden-state shards, one
    #: per leased training row, and the lease is bounded by
    #: ``training_window_records``.  This default is
    #: ``speedlm.tuner.eagle3.derive_scratch_quota_bytes(256)`` -- the default
    #: window at that function's default geometry -- written out as a literal
    #: because importing that module at class-definition time would invert the
    #: existing lazy import in ``__post_init__``::
    #:
    #:     256 rows x 128 MiB/row + 1 GiB headroom = 33 GiB = 35,433,480,192
    #:
    #: The per-row term is the widest builtin verifier geometry (hidden_size
    #: 4096, three aux layers plus the target, 4,096 tokens/row at bf16), not
    #: the flat 32 MiB this default used to carry.  That constant was fitted on
    #: gpt-oss-20b and under-charged Qwen3-8B by 2.3x, and because the quota is
    #: fail-closed the shortfall surfaced as a cycle dying mid-extraction rather
    #: than as a config error.  A model-specific run should pass its real
    #: geometry to ``derive_scratch_quota_bytes`` and set the tighter number.
    #:
    #: The previous default was a flat 5 GiB, and job 369325 died on it: every
    #: archived real run had overridden it to 20 GiB, so the one run that
    #: accepted the default aborted at 5,384,048,233 bytes -- a 0.29 %
    #: overshoot that disguised a ~21 % under-provision.  A default that no
    #: real workload can use is not a safe default.
    #:
    #: A run that raises ``training_window_records`` must raise this with it;
    #: ``derive_scratch_quota_bytes`` is the function that says by how much.
    scratch_quota_bytes: int = 33 * 1024 * 1024 * 1024
    shutdown_timeout_seconds: float = 30.0
    #: Budget for ``restore``'s wake-instead-of-respawn fast path.
    #:
    #: The fast path is a wake, one readiness wait and one bounded canary.  It
    #: is bounded separately from the restore deadline it runs inside so that a
    #: fast path which hangs cannot eat the budget the full restart behind it
    #: still needs.  120s is roughly twice a cold engine's launch-to-ready time
    #: -- deliberately generous for something that normally completes in well
    #: under a second -- and was previously unreachable from configuration, so
    #: an operator on a slower host had no way to move it.  See
    #: :data:`speedlm.gateway.control.DEFAULT_RESTORE_FAST_PATH_TIMEOUT_SECONDS`.
    restore_fast_path_timeout_seconds: float = 120.0
    val_loss_prefilter: ValLossPreFilterConfig = field(default_factory=ValLossPreFilterConfig)
    #: In-place draft weight hot-swap.  When enabled, the controller attempts
    #: to swap the drafter's weights via collective-RPC instead of restarting
    #: the entire vLLM process for each candidate.  Requires VLLM_SERVER_DEV_MODE=1
    #: and that the new draft has identical architecture, shapes, and quantization.
    #: Defaults to DISABLED pending GPU validation of cudagraph-pointer stability.
    draft_hot_swap_enabled: bool = False
    #: Accept assistant turns that carry no ``provenance_tag`` as this
    #: verifier's own output.
    #:
    #: The render stage drops any row carrying an assistant turn it cannot
    #: attribute to this verifier, because supervising another model's tokens
    #: teaches the draft head the wrong distribution.  A corpus replayed from
    #: an offline capture has assistant turns authored elsewhere and therefore
    #: carries no tag at all, so that filter empties it -- which is what
    #: :mod:`speedlm.workload` tells the operator to fix with this flag.  It
    #: was named in that advice, in the backend and in
    #: :func:`speedlm.training.rows.training_row_from_trace`, and reachable
    #: from none of them: ``SpeedLMConfig.from_dict`` rejected the key, so the
    #: advice could not be followed.
    #:
    #: Off by default because "predates tagging" is not evidence of
    #: authorship; see the field of the same name on
    #: :class:`speedlm.training.backends.eagle3.SpeculatorsPipelineConfig`.
    trust_untagged_assistant_messages: bool = False
    #: Trust that is EARNED rather than asserted.  When set, the training
    #: pipeline attests the whole snapshot as self-play traffic -- every
    #: client-supplied assistant turn must reproduce a turn this server generated
    #: in an earlier row -- and only then relabels those turns as trainable.  The
    #: cycle fails loudly if the attestation does not hold, so unlike its
    #: neighbour above this flag cannot quietly launder another model's tokens
    #: into supervision.
    trust_self_play_assistant_turns: bool = False

    def __post_init__(self) -> None:
        _validate_int_gte(self.min_trace_records, "tuning.min_trace_records", 2)
        _validate_float_gte(
            self.poll_interval_seconds,
            "tuning.poll_interval_seconds",
            0.001,
        )
        if (
            isinstance(self.held_out_fraction, bool)
            or not isinstance(self.held_out_fraction, (int, float))
            or not 0 < self.held_out_fraction < 1
        ):
            raise ConfigError("tuning.held_out_fraction must be in (0, 1)")
        _validate_int_gte(self.idle_confirmations, "tuning.idle_confirmations", 1)
        _validate_float_gte(
            self.retry_cooldown_seconds,
            "tuning.retry_cooldown_seconds",
            0,
        )
        _validate_float_gte(
            self.serving_recovery_interval_seconds,
            "tuning.serving_recovery_interval_seconds",
            0,
        )
        _validate_int_gte(self.benchmark_repeats, "tuning.benchmark_repeats", 3)
        # Zero is legal: it restores the pre-warmup behaviour, which is a
        # measurement an operator may legitimately want back for comparison.
        _validate_int_gte(self.warmup_repeats, "tuning.warmup_repeats", 0)
        if not isinstance(self.compounding_warm_start, bool):
            raise ConfigError(
                "tuning.compounding_warm_start must be a bool, "
                f"got {type(self.compounding_warm_start).__name__!r}"
            )
        if self.warm_start_max_chain_depth is not None:
            # One is the smallest bound that means anything: it re-baselines
            # every cycle whose incumbent is itself trained, i.e. it is exactly
            # ``compounding_warm_start = false`` expressed the long way.  Zero
            # or negative would be a bound on a depth no artifact can have.
            _validate_int_gte(
                self.warm_start_max_chain_depth,
                "tuning.warm_start_max_chain_depth",
                1,
            )
        if not isinstance(self.benchmark_candidate_arm_first, bool):
            raise ConfigError(
                "tuning.benchmark_candidate_arm_first must be a bool, "
                f"got {type(self.benchmark_candidate_arm_first).__name__!r}"
            )
        _validate_int_gte(
            self.benchmark_concurrency,
            "tuning.benchmark_concurrency",
            1,
        )
        _validate_int_gte(
            self.correctness_max_tokens,
            "tuning.correctness_max_tokens",
            1,
        )
        _validate_int_gte(
            self.benchmark_max_tokens,
            "tuning.benchmark_max_tokens",
            1,
        )
        if self.training_window_records is not None:
            _validate_int_gte(
                self.training_window_records,
                "tuning.training_window_records",
                self.min_trace_records,
            )
        _validate_int_gte(self.min_corpus_records, "tuning.min_corpus_records", 2)
        if (
            self.training_window_records is not None
            and self.training_window_records < self.min_corpus_records
        ):
            raise ConfigError(
                "tuning.training_window_records must be >= min_corpus_records "
                f"({self.training_window_records} < {self.min_corpus_records})"
            )
        for name, value in (
            ("verifier_revision", self.verifier_revision),
            ("speculators_repo", self.speculators_repo),
            ("training_python", self.training_python),
            ("vllm_python", self.vllm_python),
            ("prepared_validator_script", self.prepared_validator_script),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ConfigError(f"tuning.{name} must be a non-empty string or null")
        _validate_int_gte(self.sequence_length, "tuning.sequence_length", 1)
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not 0 < self.learning_rate <= MAX_LEARNING_RATE
        ):
            raise ConfigError(
                f"tuning.learning_rate must be in (0, {MAX_LEARNING_RATE:g}]"
            )
        _validate_int_gte(self.epochs, "tuning.epochs", 1)
        _validate_int_gte(
            self.extraction_concurrency,
            "tuning.extraction_concurrency",
            1,
        )
        _validate_port(self.training_port, "tuning")
        from speedlm.tuner.eagle3 import MAX_SCRATCH_BYTES  # noqa: PLC0415

        if not 1 <= self.scratch_quota_bytes <= MAX_SCRATCH_BYTES:
            raise ConfigError(
                f"tuning.scratch_quota_bytes must be in 1..{MAX_SCRATCH_BYTES} bytes "
                f"(1..96 GiB), got {self.scratch_quota_bytes}"
            )
        _validate_float_gte(
            self.shutdown_timeout_seconds,
            "tuning.shutdown_timeout_seconds",
            0.001,
        )
        _validate_float_gte(
            self.restore_fast_path_timeout_seconds,
            "tuning.restore_fast_path_timeout_seconds",
            0.001,
        )
        if not isinstance(self.draft_hot_swap_enabled, bool):
            raise ConfigError(
                "tuning.draft_hot_swap_enabled must be a bool, "
                f"got {type(self.draft_hot_swap_enabled).__name__!r}"
            )
        if not isinstance(self.trust_untagged_assistant_messages, bool):
            raise ConfigError(
                "tuning.trust_untagged_assistant_messages must be a bool, "
                f"got {type(self.trust_untagged_assistant_messages).__name__!r}"
            )
        if not isinstance(self.trust_self_play_assistant_turns, bool):
            raise ConfigError(
                "tuning.trust_self_play_assistant_turns must be a bool, "
                f"got {type(self.trust_self_play_assistant_turns).__name__!r}"
            )


@dataclass(frozen=True, slots=True)
class SpeedLMConfig:
    model: str
    model_alias: str = ""
    profile: str | None = None
    target: TargetConfig = field(default_factory=TargetConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    buffer: TraceBufferConfig = field(default_factory=TraceBufferConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    tuning: IdleTuningConfig = field(default_factory=IdleTuningConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    idle_threshold_seconds: float = 300.0
    startup_timeout_seconds: float = field(default_factory=startup_timeout_seconds)
    startup_stall_seconds: float = field(default_factory=startup_stall_seconds)
    tuning_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ConfigError("model must be a non-empty string")
        if not isinstance(self.model_alias, str):
            raise ConfigError(
                f"model_alias must be a string, got {type(self.model_alias).__name__!r}"
            )
        if self.profile is not None and (
            not isinstance(self.profile, str) or not self.profile
        ):
            raise ConfigError(
                "profile must be a non-empty string or null, got "
                f"{self.profile!r}"
            )
        if _is_bool(self.idle_threshold_seconds) or not isinstance(
            self.idle_threshold_seconds, (int, float)
        ):
            raise ConfigError(
                "idle_threshold_seconds must be numeric, got "
                f"{type(self.idle_threshold_seconds).__name__!r}"
            )
        if self.idle_threshold_seconds <= 0:
            raise ConfigError(
                f"idle_threshold_seconds must be > 0, got {self.idle_threshold_seconds}"
            )
        if _is_bool(self.startup_timeout_seconds) or not isinstance(
            self.startup_timeout_seconds, (int, float)
        ):
            raise ConfigError(
                "startup_timeout_seconds must be numeric, got "
                f"{type(self.startup_timeout_seconds).__name__!r}"
            )
        if (
            not math.isfinite(self.startup_timeout_seconds)
            or self.startup_timeout_seconds <= 0
        ):
            raise ConfigError(
                f"startup_timeout_seconds must be > 0, got {self.startup_timeout_seconds}"
            )
        if _is_bool(self.startup_stall_seconds) or not isinstance(
            self.startup_stall_seconds, (int, float)
        ):
            raise ConfigError(
                "startup_stall_seconds must be numeric, got "
                f"{type(self.startup_stall_seconds).__name__!r}"
            )
        if (
            not math.isfinite(self.startup_stall_seconds)
            or self.startup_stall_seconds <= 0
        ):
            raise ConfigError(
                f"startup_stall_seconds must be > 0, got {self.startup_stall_seconds}"
            )
        if not isinstance(self.tuning_enabled, bool):
            raise ConfigError(
                f"tuning_enabled must be a bool, got {type(self.tuning_enabled).__name__!r}"
            )
        self._warn_on_degenerate_divergence_criterion()

    def _warn_on_degenerate_divergence_criterion(self) -> None:
        """Say so at load when the position criterion cannot discriminate.

        ``promotion.min_divergence_token_index`` and
        ``tuning.correctness_max_tokens`` are validated independently and each
        looks fine on its own; it is their *relationship* that decides whether
        the criterion has a range to work in -- see
        :class:`DivergenceCriterion`.  Both degenerate ends are deliberately
        configurable, so this warns rather than raising: the point is that an
        operator who set one accidentally finds out here, before a GPU cycle,
        rather than from a rejection that reads like drafter quality.
        """
        criterion = classify_divergence_criterion(
            self.promotion.min_divergence_token_index,
            self.tuning.correctness_max_tokens,
        )
        if criterion is DivergenceCriterion.SATURATED:
            logger.warning(
                "promotion.min_divergence_token_index (%d) is at or above "
                "tuning.correctness_max_tokens (%d): every divergence at every "
                "observable offset will classify as early, so the gate demands a "
                "bitwise identical generation and will reject almost any "
                "candidate. Set the threshold below the correctness cap unless "
                "this is deliberate.",
                self.promotion.min_divergence_token_index,
                self.tuning.correctness_max_tokens,
            )
        elif criterion is DivergenceCriterion.DISABLED:
            logger.warning(
                "promotion.min_divergence_token_index is 0: the divergence "
                "position criterion is disabled and any divergence, however "
                "early, will be accepted. The gate is the only safeguard and "
                "promotion cannot be rolled back."
            )

    @property
    def alias(self) -> str:
        return self.model_alias if self.model_alias else self.model

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "model_alias": self.model_alias,
            "profile": self.profile,
            "idle_threshold_seconds": self.idle_threshold_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "startup_stall_seconds": self.startup_stall_seconds,
            "tuning_enabled": self.tuning_enabled,
        }
        result["target"] = {
            "host": self.target.host,
            "port": self.target.port,
        }
        result["wrapper"] = {
            "host": self.wrapper.host,
            "port": self.wrapper.port,
        }
        result["buffer"] = {
            "max_tokens": self.buffer.max_tokens,
            "max_age_days": self.buffer.max_age_days,
        }
        result["redaction"] = {
            "enabled": self.redaction.enabled,
        }
        result["promotion"] = {
            "min_acceptance_delta_pp": self.promotion.min_acceptance_delta_pp,
            "min_accepted_length_delta": self.promotion.min_accepted_length_delta,
            "min_throughput_delta_pct": self.promotion.min_throughput_delta_pct,
            "min_divergence_token_index": self.promotion.min_divergence_token_index,
        }
        result["tuning"] = {
            "min_trace_records": self.tuning.min_trace_records,
            "min_corpus_records": self.tuning.min_corpus_records,
            "poll_interval_seconds": self.tuning.poll_interval_seconds,
            "idle_confirmations": self.tuning.idle_confirmations,
            "retry_cooldown_seconds": self.tuning.retry_cooldown_seconds,
            "serving_recovery_interval_seconds": (
                self.tuning.serving_recovery_interval_seconds
            ),
            "held_out_fraction": self.tuning.held_out_fraction,
            "benchmark_repeats": self.tuning.benchmark_repeats,
            "warmup_repeats": self.tuning.warmup_repeats,
            "benchmark_candidate_arm_first": (
                self.tuning.benchmark_candidate_arm_first
            ),
            "benchmark_concurrency": self.tuning.benchmark_concurrency,
            "correctness_max_tokens": self.tuning.correctness_max_tokens,
            "benchmark_max_tokens": self.tuning.benchmark_max_tokens,
            "training_window_records": self.tuning.training_window_records,
            "compounding_warm_start": self.tuning.compounding_warm_start,
            "warm_start_max_chain_depth": self.tuning.warm_start_max_chain_depth,
            "verifier_revision": self.tuning.verifier_revision,
            "speculators_repo": self.tuning.speculators_repo,
            "training_python": self.tuning.training_python,
            "vllm_python": self.tuning.vllm_python,
            "prepared_validator_script": self.tuning.prepared_validator_script,
            "sequence_length": self.tuning.sequence_length,
            "learning_rate": self.tuning.learning_rate,
            "epochs": self.tuning.epochs,
            "extraction_concurrency": self.tuning.extraction_concurrency,
            "training_port": self.tuning.training_port,
            "scratch_quota_bytes": self.tuning.scratch_quota_bytes,
            "shutdown_timeout_seconds": self.tuning.shutdown_timeout_seconds,
            "restore_fast_path_timeout_seconds": (
                self.tuning.restore_fast_path_timeout_seconds
            ),
            "draft_hot_swap_enabled": self.tuning.draft_hot_swap_enabled,
            "trust_untagged_assistant_messages": (
                self.tuning.trust_untagged_assistant_messages
            ),
            "trust_self_play_assistant_turns": (
                self.tuning.trust_self_play_assistant_turns
            ),
            "val_loss_prefilter": {
                "enabled": self.tuning.val_loss_prefilter.enabled,
                "min_improvement": self.tuning.val_loss_prefilter.min_improvement,
            },
        }
        result["sampling"] = {
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "seed": self.sampling.seed,
        }
        result["workload"] = {
            "domain": self.workload.domain,
            "expect_tools": self.workload.expect_tools,
            "expect_multi_turn": self.workload.expect_multi_turn,
            "expect_system_prompts": self.workload.expect_system_prompts,
            "expect_client_supplied_assistant_turns": (
                self.workload.expect_client_supplied_assistant_turns
            ),
        }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpeedLMConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("config data must be a mapping")
        if "model" not in data:
            raise ConfigError("'model' is required")

        known_keys = {
            "model",
            "model_alias",
            "profile",
            "target",
            "wrapper",
            "buffer",
            "redaction",
            "promotion",
            "tuning",
            "sampling",
            "workload",
            "idle_threshold_seconds",
            "startup_timeout_seconds",
            "startup_stall_seconds",
            "tuning_enabled",
        }
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise ConfigError(f"unknown top-level keys: {', '.join(sorted(unknown))}")

        def _nested(mapping: Mapping[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
            if section in mapping:
                val = mapping[section]
                if not isinstance(val, Mapping):
                    raise ConfigError(f"{section} must be a mapping, got {type(val).__name__!r}")
                sub_unknown = set(val.keys()) - allowed
                if sub_unknown:
                    raise ConfigError(
                        f"unknown keys in {section}: {', '.join(sorted(sub_unknown))}"
                    )
                return cast(dict[str, Any], val)
            return {}

        target_data = _nested(data, "target", {"host", "port"})
        wrapper_data = _nested(data, "wrapper", {"host", "port"})
        buffer_data = _nested(data, "buffer", {"max_tokens", "max_age_days"})
        redaction_data = _nested(data, "redaction", {"enabled"})
        promotion_data = _nested(
            data,
            "promotion",
            {
                "min_acceptance_delta_pp",
                "min_accepted_length_delta",
                "min_throughput_delta_pct",
                "min_divergence_token_index",
            },
        )
        # ``tuning.concurrency`` used to exist and never reached the gate: it is
        # the Speculators extraction knob, while gate replay is driven by
        # ``benchmark_concurrency``.  Job 369006 set it to 4 and the gate ran at
        # 8, so the archived config asserted a degree of parallelism that never
        # ran.  Falling through to the generic "unknown keys in tuning" error
        # would be loud but not informative -- an operator reading it would most
        # likely re-add the key under some other spelling -- so the ambiguity is
        # named explicitly here and the caller is forced to choose a side.
        _legacy_tuning = data.get("tuning")
        if isinstance(_legacy_tuning, Mapping) and "concurrency" in _legacy_tuning:
            raise ConfigError(
                "tuning.concurrency was renamed because it never reached the "
                "benchmark gate: use tuning.extraction_concurrency for the "
                "Speculators hidden-state extraction degree, or "
                "tuning.benchmark_concurrency for the gate's replay degree"
            )
        tuning_data = _nested(
            data,
            "tuning",
            {
                "min_trace_records",
                "min_corpus_records",
                "poll_interval_seconds",
                "idle_confirmations",
                "retry_cooldown_seconds",
                "serving_recovery_interval_seconds",
                "held_out_fraction",
                "benchmark_repeats",
                "warmup_repeats",
                "benchmark_candidate_arm_first",
                "benchmark_concurrency",
                "correctness_max_tokens",
                "benchmark_max_tokens",
                "training_window_records",
                "compounding_warm_start",
                "warm_start_max_chain_depth",
                "verifier_revision",
                "speculators_repo",
                "training_python",
                "vllm_python",
                "prepared_validator_script",
                "sequence_length",
                "learning_rate",
                "epochs",
                "extraction_concurrency",
                "training_port",
                "scratch_quota_bytes",
                "shutdown_timeout_seconds",
                "restore_fast_path_timeout_seconds",
                "draft_hot_swap_enabled",
                "trust_untagged_assistant_messages",
                "trust_self_play_assistant_turns",
                "val_loss_prefilter",
            },
        )
        sampling_data = _nested(data, "sampling", {"temperature", "top_p", "seed"})
        workload_data = _nested(
            data,
            "workload",
            {
                "domain",
                "expect_tools",
                "expect_multi_turn",
                "expect_system_prompts",
                "expect_client_supplied_assistant_turns",
            },
        )

        # Handle nested val_loss_prefilter config.  Shallow-copy to avoid
        # mutating the caller's dict — the _nested() helper returns a reference
        # to the original mapping, and we replace the dict with a dataclass.
        tuning_data = dict(tuning_data)
        if "val_loss_prefilter" in tuning_data:
            vlp_data = tuning_data["val_loss_prefilter"]
            if isinstance(vlp_data, dict):
                tuning_data["val_loss_prefilter"] = ValLossPreFilterConfig(**vlp_data)

        return cls(
            model=data["model"],
            model_alias=data.get("model_alias", ""),
            profile=data.get("profile"),
            target=TargetConfig(**target_data),
            wrapper=WrapperConfig(**wrapper_data),
            buffer=TraceBufferConfig(**buffer_data),
            redaction=RedactionConfig(**redaction_data),
            promotion=PromotionConfig(**promotion_data),
            tuning=IdleTuningConfig(**tuning_data),
            sampling=SamplingConfig(**sampling_data),
            workload=WorkloadConfig(**workload_data),
            idle_threshold_seconds=data.get("idle_threshold_seconds", 300.0),
            startup_timeout_seconds=data.get(
                "startup_timeout_seconds", startup_timeout_seconds()
            ),
            startup_stall_seconds=data.get(
                "startup_stall_seconds", startup_stall_seconds()
            ),
            tuning_enabled=data.get("tuning_enabled", False),
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_config(path: Path) -> SpeedLMConfig:
    """Load a ``SpeedLMConfig`` from a JSON file.

    Raises ``ConfigError`` for missing files, invalid JSON, or non-object
    top-level values.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {path}") from exc

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(obj, dict):
        raise ConfigError(f"top-level JSON value must be an object, got {type(obj).__name__}")

    return SpeedLMConfig.from_dict(obj)


def save_config(config: SpeedLMConfig, path: Path) -> None:
    """Write ``config`` to *path* as JSON using an atomic rename."""
    atomic_write_json(path, config.to_dict())
