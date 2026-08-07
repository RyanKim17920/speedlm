"""Read-only ``status`` and ``gain`` diagnostics for SpeedLM.

Both reports are strictly read-only: they never create the storage layout,
never start a process, and never touch the network.  They are expected to run
cleanly against a completely fresh (or entirely absent) ``SPEEDLM_HOME``.

The reporting contract mirrors :mod:`speedlm.doctor` — every report exposes
``to_dict()``, ``to_json()`` and ``render_text()`` so the CLI can render either
form from the same data.

Honesty rule for :func:`build_gain_report`: a number is only ever printed when
it was actually measured.  When acceptance data was unavailable, when a
Prometheus counter reset was detected, or when no benchmark ever completed, the
report says so plainly and omits the deltas entirely rather than presenting a
fabricated or zeroed speedup.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from speedlm.config import ConfigError, SpeedLMConfig, load_config
from speedlm.doctor import PRIMARY_VERIFIER
from speedlm.gate.decide import (
    DIVERGENCE_ALPHA,
    DIVERGENCE_STATISTICS,
    LEGACY_ACCEPTANCE_CRITERION,
    LEGACY_ACCEPTANCE_STATISTIC,
    LEGACY_THROUGHPUT_STATISTIC,
    ContextDivergence,
    Decision,
    DispersionBasis,
    MeasurementBlock,
    Reason,
    RepeatSummary,
    ThroughputStationarity,
    Verdict,
)
from speedlm.profiles import ModelProfile, ProfileError, resolve_profile
from speedlm.storage import Layout, resolve_layout
from speedlm.traces.store import TraceStore

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ReportError(RuntimeError):
    """Raised when an on-disk report input exists but cannot be interpreted."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_FILE_NAME: Final = "config.json"
GATEWAY_FILE_NAME: Final = "gateway.json"
ACTIVE_FILE_NAME: Final = "active.json"
STATE_FILE_NAME: Final = "state.json"
SCHEDULER_FILE_NAME: Final = "scheduler.json"
EVENTS_FILE_NAME: Final = "events.jsonl"
DECISION_FILE_NAME: Final = "decision.json"
#: Durable marker written by ``TunerOrchestrator._mark_serving_unrestored`` when
#: a rollback could not return the engine to the active draft.  It lives beside
#: ``scheduler.json`` and is removed by the first restore that succeeds.
SERVING_UNRESTORED_FILE_NAME: Final = "serving-unrestored.json"

#: Reasons for which the gate aborted before it could measure comparable deltas.
#: Legacy decisions may carry zeroes for these reasons; new decisions use nulls.
UNMEASURED_REASONS: Final[frozenset[Reason]] = frozenset(
    {
        Reason.COUNTER_RESET,
        Reason.ACCEPTANCE_UNAVAILABLE,
        Reason.THROUGHPUT_UNAVAILABLE,
        Reason.TOO_FEW_REPEATS,
        Reason.HIGH_INVALID_RATE,
        Reason.OUTPUT_MISMATCH,
        # Returns above the delta computation, exactly as OUTPUT_MISMATCH does,
        # so its record carries no `acceptance_delta_pp` to require.  Without
        # this entry `parse_decision` raised "'acceptance_delta_pp' must be
        # numeric" on every saturated run -- the gate could write a verdict its
        # own reader could not load back.
        Reason.TRUNCATION_SATURATED,
        # Returns from the same block, above the delta computation, for the same
        # reason: its record carries no `acceptance_delta_pp` to require.
        Reason.TRUNCATION_UNMEASURED,
    }
)

_REASON_EXPLANATIONS: Final[Mapping[Reason, str]] = {
    Reason.BOTH_THRESHOLDS_MET: "both promotion thresholds were met",
    Reason.ACCEPTANCE_BELOW_THRESHOLD: "the acceptance gain missed its threshold",
    Reason.THROUGHPUT_BELOW_THRESHOLD: "the throughput gain missed its threshold",
    Reason.COUNTER_RESET: (
        "a vLLM Prometheus counter reset mid-benchmark, so the measurement window "
        "is invalid"
    ),
    Reason.ACCEPTANCE_UNAVAILABLE: (
        "vLLM did not expose speculative-decoding acceptance counters, so "
        "acceptance could not be measured"
    ),
    Reason.THROUGHPUT_UNAVAILABLE: (
        "the stock benchmark produced no measurable throughput, so a throughput "
        "gain could not be computed"
    ),
    Reason.HIGH_INVALID_RATE: "too many replayed requests failed to produce valid output",
    Reason.TOO_FEW_REPEATS: "the benchmark did not complete enough repeats to be meaningful",
    Reason.OUTPUT_MISMATCH: "candidate output diverged from stock output",
    Reason.TRUNCATION_SATURATED: (
        "the output cap ended every generation in an arm, so the benchmark "
        "measured fixed-length decode rather than the workload"
    ),
    Reason.TRUNCATION_UNMEASURED: (
        "no replayed response in an arm reported a finish reason, so whether the "
        "output cap ended the generations could not be measured"
    ),
    Reason.UNCERTAIN: "the gate could not reach a confident conclusion",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read *path* as a JSON object, raising :class:`ReportError` on any problem."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{path} must contain a JSON object, got {type(value).__name__}")
    return value


def _format_age(seconds: float) -> str:
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def _format_timestamp(value: float, *, now: float) -> str:
    return f"{value:.0f} ({_format_age(now - value)})"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# status: gateway
# ---------------------------------------------------------------------------


class GatewayState(StrEnum):
    """Whether a SpeedLM gateway (and its child vLLM) appears to be running."""

    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class GatewayStatus:
    """Liveness of the gateway process as recorded on disk.

    SpeedLM records a gateway runtime file when it serves; its absence means no
    gateway was started from this home.  Liveness is confirmed locally with a
    signal-0 probe — never by touching the network.
    """

    state: GatewayState
    detail: str
    record_path: Path
    pid: int | None = None
    child_pid: int | None = None
    host: str | None = None
    port: int | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "detail": self.detail,
            "record_path": str(self.record_path),
            "pid": self.pid,
            "child_pid": self.child_pid,
            "host": self.host,
            "port": self.port,
            "model": self.model,
        }


def read_gateway_status(layout: Layout) -> GatewayStatus:
    """Inspect ``<home>/gateway.json`` and report gateway liveness."""
    path = layout.root / GATEWAY_FILE_NAME
    if not path.exists():
        return GatewayStatus(
            state=GatewayState.STOPPED,
            detail=f"no gateway is running (no runtime record at {path})",
            record_path=path,
        )
    try:
        record = _read_json_object(path)
    except ReportError as exc:
        return GatewayStatus(
            state=GatewayState.UNREADABLE,
            detail=f"gateway runtime record is unreadable: {exc}",
            record_path=path,
        )

    pid = _optional_int(record.get("pid"))
    child_pid = _optional_int(record.get("child_pid"))
    host = _optional_str(record.get("host"))
    port = _optional_int(record.get("port"))
    model = _optional_str(record.get("model"))

    if pid is None:
        return GatewayStatus(
            state=GatewayState.UNREADABLE,
            detail=f"gateway runtime record at {path} has no usable 'pid'",
            record_path=path,
            child_pid=child_pid,
            host=host,
            port=port,
            model=model,
        )

    endpoint = f"{host}:{port}" if host is not None and port is not None else "unknown endpoint"
    if not _pid_alive(pid):
        return GatewayStatus(
            state=GatewayState.STALE,
            detail=(
                f"gateway process {pid} is gone but {path} still exists "
                "(stale record from a crashed or killed run)"
            ),
            record_path=path,
            pid=pid,
            child_pid=child_pid,
            host=host,
            port=port,
            model=model,
        )

    if child_pid is not None and not _pid_alive(child_pid):
        return GatewayStatus(
            state=GatewayState.RUNNING,
            detail=(
                f"gateway pid {pid} is alive on {endpoint} but child vLLM pid "
                f"{child_pid} is gone"
            ),
            record_path=path,
            pid=pid,
            child_pid=child_pid,
            host=host,
            port=port,
            model=model,
        )

    child_note = f", child vLLM pid {child_pid}" if child_pid is not None else ""
    return GatewayStatus(
        state=GatewayState.RUNNING,
        detail=f"gateway pid {pid} is alive on {endpoint}{child_note}",
        record_path=path,
        pid=pid,
        child_pid=child_pid,
        host=host,
        port=port,
        model=model,
    )


# ---------------------------------------------------------------------------
# status: active draft
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveDraftStatus:
    """The currently published draft artifact, if any."""

    present: bool
    detail: str
    artifact_id: str | None = None
    history: tuple[str, ...] = ()
    updated_at: float | None = None
    source_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "detail": self.detail,
            "artifact_id": self.artifact_id,
            "history": list(self.history),
            "updated_at": self.updated_at,
            "source_path": str(self.source_path) if self.source_path is not None else None,
        }


def _active_pointer_candidates(layout: Layout) -> tuple[Path, ...]:
    return (
        layout.root / ACTIVE_FILE_NAME,
        layout.profiles_dir / ACTIVE_FILE_NAME,
        layout.runs_dir / ACTIVE_FILE_NAME,
    )


def read_active_draft(layout: Layout) -> ActiveDraftStatus:
    """Read the artifact registry's ``active.json`` pointer if one exists."""
    for path in _active_pointer_candidates(layout):
        if not path.exists():
            continue
        try:
            record = _read_json_object(path)
        except ReportError as exc:
            return ActiveDraftStatus(
                present=False,
                detail=f"active draft pointer is unreadable: {exc}",
                source_path=path,
            )
        artifact_id = _optional_str(record.get("artifact_id"))
        if artifact_id is None:
            return ActiveDraftStatus(
                present=False,
                detail=f"active draft pointer at {path} has no usable 'artifact_id'",
                source_path=path,
            )
        raw_history = record.get("history")
        history = (
            tuple(item for item in raw_history if isinstance(item, str))
            if isinstance(raw_history, list)
            else ()
        )
        return ActiveDraftStatus(
            present=True,
            detail=f"active draft artifact {artifact_id}",
            artifact_id=artifact_id,
            history=history,
            updated_at=_optional_float(record.get("updated_at")),
            source_path=path,
        )

    return ActiveDraftStatus(
        present=False,
        detail="no active draft (nothing has been promoted yet)",
    )


# ---------------------------------------------------------------------------
# status: traces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceStatus:
    """Trace buffer occupancy."""

    path: Path
    count: int
    tokens: int
    oldest: float | None
    newest: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "count": self.count,
            "tokens": self.tokens,
            "oldest": self.oldest,
            "newest": self.newest,
        }


def read_trace_status(layout: Layout) -> TraceStatus:
    """Summarise the trace buffer; a missing buffer reports zeroes, not an error."""
    path = layout.traces_dir / "traces.jsonl"
    stats = TraceStore(path).stats()
    return TraceStatus(
        path=path,
        count=stats.count,
        tokens=stats.tokens,
        oldest=stats.oldest,
        newest=stats.newest,
    )


# ---------------------------------------------------------------------------
# status: tuner state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TunerStatus:
    """Durable tuner state machine snapshot, if one has been written."""

    present: bool
    detail: str
    state: str | None = None
    sequence: int | None = None
    updated_at: float | None = None
    reason: str | None = None
    source_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "detail": self.detail,
            "state": self.state,
            "sequence": self.sequence,
            "updated_at": self.updated_at,
            "reason": self.reason,
            "source_path": str(self.source_path) if self.source_path is not None else None,
        }


def read_tuner_status(layout: Layout) -> TunerStatus:
    """Read ``<runs>/state.json`` written by the tuner state machine."""
    path = layout.runs_dir / STATE_FILE_NAME
    if not path.exists():
        return TunerStatus(
            present=False,
            detail=f"no tuner state (the tuner has not run; {path} does not exist)",
        )
    try:
        record = _read_json_object(path)
    except ReportError as exc:
        return TunerStatus(
            present=False,
            detail=f"tuner state is unreadable: {exc}",
            source_path=path,
        )
    state = _optional_str(record.get("state"))
    if state is None:
        return TunerStatus(
            present=False,
            detail=f"tuner state at {path} has no usable 'state'",
            source_path=path,
        )
    reason = record.get("reason")
    return TunerStatus(
        present=True,
        detail=state,
        state=state,
        sequence=_optional_int(record.get("sequence")),
        updated_at=_optional_float(record.get("updated_at")),
        reason=reason if isinstance(reason, str) and reason else None,
        source_path=path,
    )


# ---------------------------------------------------------------------------
# status: scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchedulerStatus:
    """Durable lifecycle and latest-cycle snapshot for the tuner scheduler."""

    present: bool
    detail: str
    enabled: bool | None = None
    lifecycle: str | None = None
    #: Whether the engine is serving a draft the active pointer does not name.
    #: ``None`` when the record predates the field -- which is *not* the same as
    #: ``False``: an older scheduler never reported the condition at all, so the
    #: honest reading is "unknown", never "fine".
    serving_unrestored: bool | None = None
    last_watermark: Mapping[str, object] | None = None
    last_result: Mapping[str, object] | None = None
    last_error: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    lifecycle_changed_at: float | None = None
    last_attempt_at: float | None = None
    last_result_at: float | None = None
    last_error_at: float | None = None
    source_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "detail": self.detail,
            "enabled": self.enabled,
            "lifecycle": self.lifecycle,
            "serving_unrestored": self.serving_unrestored,
            "last_watermark": (
                dict(self.last_watermark) if self.last_watermark is not None else None
            ),
            "last_result": (
                dict(self.last_result) if self.last_result is not None else None
            ),
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "lifecycle_changed_at": self.lifecycle_changed_at,
            "last_attempt_at": self.last_attempt_at,
            "last_result_at": self.last_result_at,
            "last_error_at": self.last_error_at,
            "source_path": str(self.source_path) if self.source_path is not None else None,
        }


def read_scheduler_status(layout: Layout) -> SchedulerStatus:
    """Read ``<runs>/scheduler.json`` written by :class:`TunerService`."""
    path = layout.runs_dir / SCHEDULER_FILE_NAME
    if not path.exists():
        return SchedulerStatus(
            present=False,
            detail=f"no scheduler status ({path} does not exist)",
        )
    try:
        record = _read_json_object(path)
    except ReportError as exc:
        return SchedulerStatus(
            present=False,
            detail=f"scheduler status is unreadable: {exc}",
            source_path=path,
        )
    enabled = record.get("enabled")
    lifecycle = _optional_str(record.get("lifecycle"))
    if not isinstance(enabled, bool) or lifecycle is None:
        return SchedulerStatus(
            present=False,
            detail=f"scheduler status at {path} has no usable lifecycle",
            source_path=path,
        )
    watermark = record.get("last_watermark")
    result = record.get("last_result")
    raw_unrestored = record.get("serving_unrestored")
    return SchedulerStatus(
        present=True,
        detail=f"{lifecycle} ({'enabled' if enabled else 'disabled'})",
        enabled=enabled,
        lifecycle=lifecycle,
        serving_unrestored=(
            raw_unrestored if isinstance(raw_unrestored, bool) else None
        ),
        last_watermark=watermark if isinstance(watermark, dict) else None,
        last_result=result if isinstance(result, dict) else None,
        last_error=_optional_str(record.get("last_error")),
        created_at=_optional_float(record.get("created_at")),
        updated_at=_optional_float(record.get("updated_at")),
        lifecycle_changed_at=_optional_float(record.get("lifecycle_changed_at")),
        last_attempt_at=_optional_float(record.get("last_attempt_at")),
        last_result_at=_optional_float(record.get("last_result_at")),
        last_error_at=_optional_float(record.get("last_error_at")),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# status: model pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPairStatus:
    """The configured verifier/draft pair and where it came from."""

    verifier: str
    draft: str
    configured: bool
    detail: str
    profile: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "verifier": self.verifier,
            "draft": self.draft,
            "configured": self.configured,
            "detail": self.detail,
            "profile": dict(self.profile),
        }


def _profile_report(profile: ModelProfile) -> dict[str, object]:
    report = profile.to_dict()
    detail = f"matched profile {profile.name!r}"
    if not profile.trainable:
        detail += (
            f"; tuning is unavailable for {profile.speculative_method!r}"
        )
    report.update(
        {
            "status": "profiled",
            "tuning_available": profile.trainable,
            "detail": detail,
        }
    )
    return report


def _unprofiled_report(served_model: str, reason: str) -> dict[str, object]:
    return {
        "status": "unprofiled",
        "name": "unprofiled",
        "verifier_model": served_model,
        "draft_model": None,
        "speculative_method": "unknown",
        "num_speculative_tokens": None,
        "target_layer_ids": None,
        "chat_template_kind": "unknown",
        "max_seq_len": None,
        "trainable": False,
        "tuning_available": False,
        "detail": f"no profile matched {served_model!r}; tuning is unavailable ({reason})",
    }


def read_model_pair(
    layout: Layout,
    *,
    served_model: str | None = None,
) -> ModelPairStatus:
    """Resolve the configured/served model to a profile without guessing a draft."""
    path = layout.root / CONFIG_FILE_NAME
    config: SpeedLMConfig | None = None
    configured = path.exists()
    source_detail: str
    if not path.exists():
        source_detail = f"built-in default pair (no {path})"
    else:
        try:
            config = load_config(path)
        except ConfigError as exc:
            configured = False
            source_detail = f"built-in default pair ({path} is unusable: {exc})"
        else:
            source_detail = f"configured in {path}"

    candidate = served_model
    if candidate is None and config is not None:
        candidate = config.model
    if candidate is None:
        candidate = PRIMARY_VERIFIER

    profile_config = (
        None
        if config is None
        else {"model": config.model, "profile": config.profile}
    )
    try:
        profile = resolve_profile(
            profile_config,
            served_model=candidate,
            home=layout.root,
        )
    except ProfileError as exc:
        return ModelPairStatus(
            verifier=candidate,
            draft="unknown",
            configured=configured,
            detail=(
                f"no profile matched served model {candidate!r}; "
                "tuning is unavailable"
            ),
            profile=_unprofiled_report(candidate, str(exc)),
        )

    draft = profile.draft_model if profile.draft_model is not None else "native"
    return ModelPairStatus(
        verifier=profile.verifier_model,
        draft=draft,
        configured=configured,
        detail=source_detail,
        profile=_profile_report(profile),
    )


# ---------------------------------------------------------------------------
# status report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatusReport:
    """Aggregate read-only view of the current SpeedLM install."""

    home: Path
    home_exists: bool
    gateway: GatewayStatus
    active_draft: ActiveDraftStatus
    traces: TraceStatus
    tuner: TunerStatus
    scheduler: SchedulerStatus
    models: ModelPairStatus
    generated_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "home": str(self.home),
            "home_exists": self.home_exists,
            "generated_at": self.generated_at,
            "gateway": self.gateway.to_dict(),
            "active_draft": self.active_draft.to_dict(),
            "traces": self.traces.to_dict(),
            "tuner": self.tuner.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "models": self.models.to_dict(),
            "profile": dict(self.models.profile),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        now = self.generated_at
        lines = ["SpeedLM status"]
        home_note = "" if self.home_exists else " (not created yet)"
        lines.append(f"home         : {self.home}{home_note}")
        lines.append(f"gateway      : {self.gateway.detail}")
        lines.append(f"active draft : {self.active_draft.detail}")
        if self.active_draft.present:
            if self.active_draft.updated_at is not None:
                lines.append(
                    f"  updated    : {_format_timestamp(self.active_draft.updated_at, now=now)}"
                )
            if self.active_draft.history:
                lines.append(f"  rollbacks  : {len(self.active_draft.history)} prior artifact(s)")

        lines.append(f"traces       : {self.traces.count} record(s), {self.traces.tokens} token(s)")
        if self.traces.oldest is not None and self.traces.newest is not None:
            lines.append(f"  oldest     : {_format_timestamp(self.traces.oldest, now=now)}")
            lines.append(f"  newest     : {_format_timestamp(self.traces.newest, now=now)}")
        else:
            lines.append("  age        : n/a (trace buffer is empty)")

        lines.append(f"tuner        : {self.tuner.detail}")
        if self.tuner.present:
            if self.tuner.sequence is not None:
                lines.append(f"  sequence   : {self.tuner.sequence}")
            if self.tuner.updated_at is not None:
                lines.append(f"  updated    : {_format_timestamp(self.tuner.updated_at, now=now)}")
            if self.tuner.reason is not None:
                lines.append(f"  reason     : {self.tuner.reason}")

        lines.append(f"scheduler    : {self.scheduler.detail}")
        if self.scheduler.present:
            if self.scheduler.last_result is not None:
                outcome = _optional_str(self.scheduler.last_result.get("outcome"))
                if outcome is not None:
                    lines.append(f"  last cycle : {outcome}")
            # An incident, not a status: say so where nobody can skim past it.
            if self.scheduler.serving_unrestored:
                lines.append(
                    "  SERVING    : NOT RESTORED -- the engine is serving a draft "
                    "the active pointer does not name"
                )
                lines.append(
                    "               (answers are unaffected; throughput is "
                    "unvalidated until the tuner recovers)"
                )
            if self.scheduler.last_error is not None:
                lines.append(f"  last error : {self.scheduler.last_error}")
            if self.scheduler.updated_at is not None:
                lines.append(
                    f"  updated    : "
                    f"{_format_timestamp(self.scheduler.updated_at, now=now)}"
                )

        lines.append(f"verifier     : {self.models.verifier}")
        lines.append(f"draft        : {self.models.draft}")
        lines.append(f"  source     : {self.models.detail}")
        return "\n".join(lines)


def build_status_report(
    *,
    home: Path | None = None,
    now: float | None = None,
) -> StatusReport:
    """Collect every ``status`` input without creating or mutating anything."""
    layout = resolve_layout(home)
    gateway = read_gateway_status(layout)
    return StatusReport(
        home=layout.root,
        home_exists=layout.root.is_dir(),
        gateway=gateway,
        active_draft=read_active_draft(layout),
        traces=read_trace_status(layout),
        tuner=read_tuner_status(layout),
        scheduler=read_scheduler_status(layout),
        models=read_model_pair(layout, served_model=gateway.model),
        generated_at=time.time() if now is None else now,
    )


# ---------------------------------------------------------------------------
# gain: decision loading
# ---------------------------------------------------------------------------


def _decision_for_run_id(layout: Layout, run_id: object) -> Path | None:
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        return None
    path = layout.runs_dir / run_id / DECISION_FILE_NAME
    return path if path.is_file() else None


def _journal_decision(layout: Layout) -> Path | None:
    """Bind a decision mtime to the latest benchmark interval in the tuner journal."""
    state_path = layout.runs_dir / STATE_FILE_NAME
    events_path = layout.runs_dir / EVENTS_FILE_NAME
    if not state_path.is_file() or not events_path.is_file():
        return None
    try:
        state = _read_json_object(state_path)
        state_sequence = _optional_int(state.get("sequence"))
        state_updated_at = _optional_float(state.get("updated_at"))
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, ReportError):
        return None
    if state_sequence is None or state_updated_at is None:
        return None

    events: list[tuple[int, float, str | None]] = []
    try:
        for line in lines:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                return None
            sequence = _optional_int(event.get("sequence"))
            event_time = _optional_float(event.get("timestamp"))
            target = event.get("to")
            if (
                sequence is None
                or event_time is None
                or target is not None
                and not isinstance(target, str)
            ):
                return None
            if sequence <= state_sequence:
                events.append((sequence, event_time, target))
    except json.JSONDecodeError:
        return None

    benchmark_events = [event for event in events if event[2] == "BENCHMARKING"]
    if not benchmark_events:
        return None
    benchmark_sequence, benchmark_started_at, _ = max(benchmark_events)
    later_times = [
        event_time
        for sequence, event_time, _ in events
        if sequence > benchmark_sequence
    ]
    benchmark_finished_at = (
        min(later_times) if later_times else state_updated_at
    )
    if benchmark_finished_at < benchmark_started_at:
        return None

    matches: list[Path] = []
    for path in layout.runs_dir.glob(f"*/{DECISION_FILE_NAME}"):
        if not path.is_file():
            continue
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        if benchmark_started_at <= modified_at <= benchmark_finished_at:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def find_latest_decision(layout: Layout) -> Path | None:
    """Return the decision belonging to a run known by durable tuner provenance.

    Explicit ``run_id`` links take precedence. Current tuner journals do not
    persist that link yet, so their latest ``BENCHMARKING`` interval is used to
    bind exactly one direct-child decision by write time. A lone direct-child
    decision remains readable as a compatibility path for pre-journal homes;
    recursive filesystem discovery is deliberately forbidden.
    """
    runs_dir = layout.runs_dir
    if not runs_dir.is_dir():
        return None

    provenance_present = False
    for provenance_path in (
        runs_dir / STATE_FILE_NAME,
        *_active_pointer_candidates(layout),
    ):
        if not provenance_path.is_file():
            continue
        provenance_present = True
        try:
            provenance = _read_json_object(provenance_path)
        except ReportError:
            continue
        if "run_id" in provenance:
            return _decision_for_run_id(layout, provenance["run_id"])

    events_path = runs_dir / EVENTS_FILE_NAME
    if events_path.is_file():
        provenance_present = True
        decision = _journal_decision(layout)
        if decision is not None:
            return decision
    if provenance_present:
        return None

    legacy_candidates = [
        path
        for path in runs_dir.glob(f"*/{DECISION_FILE_NAME}")
        if path.is_file()
    ]
    return legacy_candidates[0] if len(legacy_candidates) == 1 else None


def _require_float(record: Mapping[str, Any], key: str, source: Path) -> float:
    value = _optional_float(record.get(key))
    if value is None:
        raise ReportError(f"{source}: '{key}' must be numeric")
    return value


def _require_str(record: Mapping[str, Any], key: str, source: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ReportError(f"{source}: '{key}' must be a string")
    return value


def _require_int(record: Mapping[str, Any], key: str, source: Path) -> int:
    value = _optional_int(record.get(key))
    if value is None:
        raise ReportError(f"{source}: '{key}' must be an integer")
    return value


def _parse_delta(
    record: Mapping[str, Any],
    key: str,
    source: Path,
    *,
    measured: bool,
) -> float | None:
    value = record.get(key)
    if value is None and not measured:
        return None
    parsed = _optional_float(value)
    if parsed is None:
        raise ReportError(f"{source}: '{key}' must be numeric")
    return parsed


def _parse_repeat(record: Mapping[str, Any], source: Path) -> RepeatSummary:
    return RepeatSummary(
        repeat_index=_require_int(record, "repeat_index", source),
        stock_tok_per_sec=_require_float(record, "stock_tok_per_sec", source),
        candidate_tok_per_sec=_require_float(record, "candidate_tok_per_sec", source),
        stock_acceptance_rate=_require_float(record, "stock_acceptance_rate", source),
        candidate_acceptance_rate=_require_float(record, "candidate_acceptance_rate", source),
        invalid_rate=_require_float(record, "invalid_rate", source),
        output_mismatches=_require_int(record, "output_mismatches", source),
        # Optional: every decision written before the gate's promotion
        # criterion moved to mean accepted length lacks these two columns, and
        # 0.0 is how "never recorded" reads back -- it is not a reachable
        # measurement, whose floor is 1.0.
        stock_accepted_length=(
            _optional_float(record.get("stock_accepted_length")) or 0.0
        ),
        candidate_accepted_length=(
            _optional_float(record.get("candidate_accepted_length")) or 0.0
        ),
        # Optional: every archived decision predates the finish-reason counts
        # and defaults them to 0. That makes the derived truncation regime
        # classify as UNTESTABLE rather than being silently relabelled BOUNDED,
        # which is a claim that truncation was measured and was low.
        stock_finish_reasons=(
            _optional_int(record.get("stock_finish_reasons")) or 0
        ),
        candidate_finish_reasons=(
            _optional_int(record.get("candidate_finish_reasons")) or 0
        ),
        stock_truncated=(
            _optional_int(record.get("stock_truncated")) or 0
        ),
        candidate_truncated=(
            _optional_int(record.get("candidate_truncated")) or 0
        ),
    )


def parse_decision(record: Mapping[str, Any], *, source: Path) -> Decision:
    """Rebuild a :class:`Decision` from its ``to_dict()`` form.

    Raises:
        ReportError: If any required field is missing or of the wrong type.
    """
    try:
        verdict = Verdict(record["verdict"])
        reason = Reason(record["reason"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportError(f"{source}: unusable 'verdict'/'reason'") from exc
    measured = reason not in UNMEASURED_REASONS

    raw_repeats = record.get("per_repeat", [])
    if not isinstance(raw_repeats, list):
        raise ReportError(f"{source}: 'per_repeat' must be a list")
    per_repeat: list[RepeatSummary] = []
    for item in raw_repeats:
        if not isinstance(item, Mapping):
            raise ReportError(f"{source}: each 'per_repeat' entry must be an object")
        per_repeat.append(_parse_repeat(item, source))

    return Decision(
        verdict=verdict,
        reason=reason,
        acceptance_delta_pp=_parse_delta(
            record, "acceptance_delta_pp", source, measured=measured
        ),
        throughput_delta_pct=_parse_delta(
            record, "throughput_delta_pct", source, measured=measured
        ),
        min_acceptance_delta_pp=_require_float(record, "min_acceptance_delta_pp", source),
        min_throughput_delta_pct=_require_float(record, "min_throughput_delta_pct", source),
        num_repeats=_require_int(record, "num_repeats", source),
        # Optional: decisions written before the gate warmed its arms have no
        # warmup field, and zero is the truth for those runs.
        warmup_repeats=(
            _require_int(record, "warmup_repeats", source)
            if "warmup_repeats" in record
            else 0
        ),
        per_repeat=tuple(per_repeat),
        stock_avg_acceptance=_require_float(record, "stock_avg_acceptance", source),
        candidate_avg_acceptance=_require_float(record, "candidate_avg_acceptance", source),
        stock_avg_tok_per_sec=_require_float(record, "stock_avg_tok_per_sec", source),
        candidate_avg_tok_per_sec=_require_float(record, "candidate_avg_tok_per_sec", source),
        # All optional: decisions written before acceptance was measured per
        # repeat carry none of these.  Zero is the truth for those runs -- a
        # single pooled sample really does have no dispersion to report -- and
        # an empty divergence array really is all the divergence detail they
        # kept.
        stock_acceptance_stdev=_optional_float(record.get("stock_acceptance_stdev")) or 0.0,
        candidate_acceptance_stdev=(
            _optional_float(record.get("candidate_acceptance_stdev")) or 0.0
        ),
        min_divergence_token_index=(
            _require_int(record, "min_divergence_token_index", source)
            if "min_divergence_token_index" in record
            else 0
        ),
        output_divergences=_parse_divergences(record, source),
        # The divergence noise floor.  All optional: a record written before
        # the gate measured its own engine nondeterminism carries none of these
        # keys, and each absence reads back as "no control was run" rather than
        # as a measurement.  ``divergence_trials``/``control_trials`` at zero
        # make ``divergence_rate``/``control_divergence_rate`` return ``None``
        # -- never a rate of 0.0 -- and ``divergence_control_available`` stays
        # False, which is what stops the criterion's *assumed* zero floor being
        # read back as a measured one.
        divergence_trials=_optional_int(record.get("divergence_trials")) or 0,
        control_trials=_optional_int(record.get("control_trials")) or 0,
        control_divergences=_parse_divergences(
            record, source, key="control_divergences"
        ),
        divergence_control_available=bool(
            record.get("divergence_control_available", False)
        ),
        divergence_total_p_value=_optional_float(
            record.get("divergence_total_p_value")
        ),
        divergence_early_p_value=_optional_float(
            record.get("divergence_early_p_value")
        ),
        # Absent means the record predates the test entirely; it is then
        # rendered only alongside a p-value, and there are none.
        divergence_alpha=(
            _require_float(record, "divergence_alpha", source)
            if "divergence_alpha" in record
            else DIVERGENCE_ALPHA / DIVERGENCE_STATISTICS
        ),
        acceptance_statistic=(
            _require_str(record, "acceptance_statistic", source)
            if "acceptance_statistic" in record
            else LEGACY_ACCEPTANCE_STATISTIC
        ),
        # The promotion criterion.  All optional: a record written before the
        # criterion moved off the acceptance rate carries none of them, and is
        # labelled with what it was actually gated on rather than relabelled
        # with the current criterion.
        acceptance_criterion=(
            _require_str(record, "acceptance_criterion", source)
            if "acceptance_criterion" in record
            else LEGACY_ACCEPTANCE_CRITERION
        ),
        accepted_length_delta=_optional_float(record.get("accepted_length_delta")),
        min_accepted_length_delta=_optional_float(
            record.get("min_accepted_length_delta")
        ),
        stock_avg_accepted_length=(
            _optional_float(record.get("stock_avg_accepted_length")) or 0.0
        ),
        candidate_avg_accepted_length=(
            _optional_float(record.get("candidate_avg_accepted_length")) or 0.0
        ),
        stock_accepted_length_stdev=(
            _optional_float(record.get("stock_accepted_length_stdev")) or 0.0
        ),
        candidate_accepted_length_stdev=(
            _optional_float(record.get("candidate_accepted_length_stdev")) or 0.0
        ),
        # Measurement context.  Optional in both directions: absent means the
        # record predates the field, and ``None`` is exactly how that reads
        # back -- never zero, which would claim an unbatched, uncapped run.
        benchmark_max_tokens=_optional_int(record.get("benchmark_max_tokens")),
        replay_concurrency=_optional_int(record.get("replay_concurrency")),
        correctness_max_tokens=_optional_int(record.get("correctness_max_tokens")),
        correctness_repeats=_optional_int(record.get("correctness_repeats")),
        suite_hash=_optional_str(record.get("suite_hash")),
        num_contexts=_optional_int(record.get("num_contexts")),
        stock_draft=_optional_str(record.get("stock_draft")),
        # How the measurement was taken.  Optional in both directions: a record
        # written before the schedule was persisted has no blocks and no
        # stationarity verdict, and reads back as "not recorded" -- never as
        # "one block" or "stationary", either of which would invent evidence.
        arm_blocks=_optional_int(record.get("arm_blocks")),
        block_schedule=_parse_block_schedule(record, source),
        throughput_stationarity=_parse_stationarity(record, source),
        **_parse_throughput_statistic(record, source, measured=measured),
    )


def _parse_block_schedule(
    record: Mapping[str, Any], source: Path
) -> tuple[MeasurementBlock, ...]:
    """Recover the realized block order from a persisted decision."""
    raw = record.get("block_schedule", [])
    if not isinstance(raw, list):
        raise ReportError(f"{source}: 'block_schedule' must be a list")
    blocks: list[MeasurementBlock] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReportError(f"{source}: each 'block_schedule' entry must be an object")
        blocks.append(
            MeasurementBlock(
                arm=_require_str(item, "arm", source),
                repeats=_require_int(item, "repeats", source),
                # A block whose record does not say it restarted did not: the
                # absent key is the un-restarted case, which is the one worth
                # noticing.
                restarted=bool(item.get("restarted", False)),
            )
        )
    return tuple(blocks)


def _parse_stationarity(
    record: Mapping[str, Any], source: Path
) -> ThroughputStationarity | None:
    """Recover the stationarity verdict, or ``None`` when none was recorded."""
    raw = record.get("throughput_stationarity")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ReportError(f"{source}: 'throughput_stationarity' must be an object")
    return ThroughputStationarity(
        testable=bool(raw.get("testable", False)),
        stationary=bool(raw.get("stationary", False)),
        required_for_promotion=bool(raw.get("required_for_promotion", False)),
        min_repeats=_require_int(raw, "min_repeats", source),
        delta_shift_pct=_optional_float(raw.get("delta_shift_pct")),
        delta_shift_t_statistic=_optional_float(raw.get("delta_shift_t_statistic")),
        min_shift_t_statistic=_require_float(raw, "min_shift_t_statistic", source),
        materiality_pct=_require_float(raw, "materiality_pct", source),
        stock_flat_from_repeat=_optional_int(raw.get("stock_flat_from_repeat")),
        candidate_flat_from_repeat=_optional_int(raw.get("candidate_flat_from_repeat")),
        stock_trend_pct_per_repeat=_optional_float(
            raw.get("stock_trend_pct_per_repeat")
        ),
        candidate_trend_pct_per_repeat=_optional_float(
            raw.get("candidate_trend_pct_per_repeat")
        ),
    )


def _parse_divergences(
    record: Mapping[str, Any],
    source: Path,
    *,
    key: str = "output_divergences",
) -> tuple[ContextDivergence, ...]:
    """Parse one divergence array -- the candidate's, or the control's.

    Both are persisted in the same shape by ``Decision.to_dict``, and the
    control array is the noise floor's own evidence, so it gets the same
    per-entry validation rather than being trusted wholesale.
    """
    raw = record.get(key, [])
    if not isinstance(raw, list):
        raise ReportError(f"{source}: '{key}' must be a list")
    parsed: list[ContextDivergence] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReportError(f"{source}: each '{key}' entry must be an object")
        parsed.append(
            ContextDivergence(
                context_hash=_require_str(item, "context_hash", source),
                repeat_index=_require_int(item, "repeat_index", source),
                first_divergence_index=_require_int(item, "first_divergence_index", source),
                basis=_require_str(item, "basis", source),
                stock_length=_require_int(item, "stock_length", source),
                candidate_length=_require_int(item, "candidate_length", source),
                early=bool(item.get("early", False)),
            )
        )
    return tuple(parsed)


def _parse_throughput_statistic(
    record: Mapping[str, Any],
    source: Path,
    *,
    measured: bool,
) -> dict[str, Any]:
    """Recover which statistic a persisted decision was actually gated on.

    Records written before the gate pinned its statistic carry no
    ``throughput_statistic`` key.  Their ``throughput_delta_pct`` came from the
    Prometheus decode window, and their ``*_avg_tok_per_sec`` pair came from it
    too -- which is why averaging their ``per_repeat`` column does not
    reproduce it.  Labelling them :data:`LEGACY_THROUGHPUT_STATISTIC` and
    mirroring those values into the Prometheus fields keeps an old record
    honest instead of silently reattributing it to the current statistic.
    """
    if "throughput_statistic" not in record:
        prom_delta = _parse_delta(
            record, "throughput_delta_pct", source, measured=measured
        )
        return {
            "throughput_statistic": LEGACY_THROUGHPUT_STATISTIC,
            "stock_prometheus_decode_tok_per_sec": _require_float(
                record, "stock_avg_tok_per_sec", source
            ),
            "candidate_prometheus_decode_tok_per_sec": _require_float(
                record, "candidate_avg_tok_per_sec", source
            ),
            "prometheus_throughput_delta_pct": prom_delta,
        }

    statistic = record["throughput_statistic"]
    if not isinstance(statistic, str) or not statistic:
        raise ReportError(f"{source}: 'throughput_statistic' must be a non-empty string")
    raw_delta = record.get("prometheus_throughput_delta_pct")
    if raw_delta is not None and not isinstance(raw_delta, (int, float)):
        raise ReportError(
            f"{source}: 'prometheus_throughput_delta_pct' must be a number or null"
        )
    return {
        "throughput_statistic": statistic,
        "stock_prometheus_decode_tok_per_sec": _require_float(
            record, "stock_prometheus_decode_tok_per_sec", source
        ),
        "candidate_prometheus_decode_tok_per_sec": _require_float(
            record, "candidate_prometheus_decode_tok_per_sec", source
        ),
        "prometheus_throughput_delta_pct": (
            None if raw_delta is None else float(raw_delta)
        ),
    }


def load_decision(path: Path) -> Decision:
    """Load and validate a persisted gate decision."""
    return parse_decision(_read_json_object(path), source=path)


# ---------------------------------------------------------------------------
# serving-unrestored marker
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServingUnrestoredMarker:
    """A rollback left the engine on a draft the active pointer does not name.

    Read from ``<runs>/serving-unrestored.json``.  Its mere *presence* is the
    signal; every payload field is optional so that a truncated or partially
    written marker still reports the incident rather than being discarded.  The
    condition outlives the cycle that caused it -- the state machine ends a
    preempted cycle at ``READY`` and has nowhere to carry "the cycle is over
    *and* serving is wrong" -- which is why it is a file and not a state.
    """

    source_path: Path
    detected_at: float | None = None
    expected_active_draft: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "detected_at": self.detected_at,
            "expected_active_draft": self.expected_active_draft,
            "error": self.error,
        }


def read_serving_unrestored(layout: Layout) -> ServingUnrestoredMarker | None:
    """Return the serving-unrestored marker, or ``None`` when serving is sound.

    An unreadable marker is still a marker: it is reported with empty detail
    rather than swallowed, because the file only ever exists while the incident
    holds and losing it would turn a live warning into silence.
    """
    path = layout.runs_dir / SERVING_UNRESTORED_FILE_NAME
    if not path.is_file():
        return None
    try:
        record = _read_json_object(path)
    except ReportError:
        return ServingUnrestoredMarker(source_path=path)
    return ServingUnrestoredMarker(
        source_path=path,
        detected_at=_optional_float(record.get("detected_at")),
        expected_active_draft=_optional_str(record.get("expected_active_draft")),
        error=_optional_str(record.get("error")),
    )


# ---------------------------------------------------------------------------
# gain report
# ---------------------------------------------------------------------------


class GainStatus(StrEnum):
    """Whether a real, trustworthy measurement exists.

    Deliberately *not* extended with a ``serving_unrestored`` member.  This enum
    answers one question -- "is there a number, and was it measured" -- and the
    four members are mutually exclusive answers to it.  Whether the engine is
    currently delivering that number is an orthogonal fact: a serving-unrestored
    incident can coexist with any of these four, so folding it in would force a
    choice between reporting the incident and reporting the measurement, and
    would silently reclassify a perfectly good ``MEASURED`` decision.  It is
    carried alongside, on :attr:`GainReport.serving_unrestored`.
    """

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    NO_GATE_RUN = "no_gate_run"
    UNREADABLE = "unreadable"


def _dispersion_fragments(
    basis: DispersionBasis,
    standard_error: float | None,
    *,
    repeats: int,
    unit: str,
) -> tuple[str, str]:
    """Render a delta's dispersion as ``(suffix, note)`` text fragments.

    ``suffix`` follows the delta itself and is empty unless a standard error
    genuinely exists; ``note`` always says something, so the degenerate case can
    never render as a bare number that reads like a tight measurement.
    """
    if basis is DispersionBasis.DEGENERATE:
        return "", f"no variance observed across {repeats} identical repeats"
    if basis is DispersionBasis.UNSAMPLED:
        return "", f"no dispersion: {repeats} repeat(s)"
    if standard_error is None:
        return "", f"n={repeats}, standard error unavailable"
    return f" +/- {standard_error:.3f}{unit}", f"n={repeats}"


@dataclass(frozen=True, slots=True)
class GainReport:
    """Measured benefit of the current candidate draft, or an honest denial."""

    status: GainStatus
    detail: str
    source_path: Path | None = None
    source_mtime: float | None = None
    decision: Decision | None = None
    #: Set when ``<runs>/serving-unrestored.json`` exists, i.e. the gain below
    #: was measured for a draft the engine is not currently serving.  Reported
    #: beside the measurement rather than as a :class:`GainStatus`; see there.
    serving_unrestored: ServingUnrestoredMarker | None = None
    generated_at: float = 0.0

    # -- derived honesty predicates ----------------------------------------

    @property
    def acceptance_available(self) -> bool:
        if self.decision is None:
            return False
        return self.decision.reason is not Reason.ACCEPTANCE_UNAVAILABLE

    @property
    def counter_reset(self) -> bool:
        return self.decision is not None and self.decision.reason is Reason.COUNTER_RESET

    @property
    def deltas_measured(self) -> bool:
        """True only when the gate actually produced comparable numbers.

        *Either* acceptance statistic counts, not both.  The gate's promotion
        criterion moved from the acceptance rate to the mean accepted length,
        and every decision written before that move carries
        ``acceptance_delta_pp`` and no ``accepted_length_delta`` -- which is
        every archived run on disk.  Demanding both would relabel that entire
        archive "not measured" even though the numbers in it were measured, and
        the reader would lose the only readout those runs have.  Demanding
        either keeps a legacy record printing exactly what it printed before
        while letting a record that carries only the new statistic print that.

        ``throughput_delta_pct`` stays mandatory: the throughput line has no
        second statistic to fall back on, so without it there is nothing to
        report.
        """
        decision = self.decision
        return (
            decision is not None
            and decision.reason not in UNMEASURED_REASONS
            and (
                decision.acceptance_delta_pp is not None
                or decision.accepted_length_delta is not None
            )
            and decision.throughput_delta_pct is not None
        )

    # -- rendering ---------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status.value,
            "detail": self.detail,
            "generated_at": self.generated_at,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "source_mtime": self.source_mtime,
            "acceptance_available": self.acceptance_available,
            "counter_reset": self.counter_reset,
            "deltas_measured": self.deltas_measured,
            "serving_unrestored": (
                None
                if self.serving_unrestored is None
                else self.serving_unrestored.to_dict()
            ),
            "verdict": None,
            "reason": None,
            "reason_detail": None,
            "measurement": None,
            "per_repeat": [],
        }
        decision = self.decision
        if decision is None:
            return result

        result["verdict"] = decision.verdict.value
        result["reason"] = decision.reason.value
        result["reason_detail"] = _REASON_EXPLANATIONS[decision.reason]
        result["num_repeats"] = decision.num_repeats
        result["thresholds"] = {
            "min_acceptance_delta_pp": decision.min_acceptance_delta_pp,
            "min_accepted_length_delta": decision.min_accepted_length_delta,
            "min_throughput_delta_pct": decision.min_throughput_delta_pct,
        }
        # What the deltas were measured *over*.  Emitted even for an aborted
        # decision: it describes the attempt, not the result, and it is what
        # makes two runs comparable at all.  Every entry is ``None`` on records
        # written before the gate persisted its measurement context.
        result["measurement_context"] = {
            "suite_hash": decision.suite_hash,
            "num_contexts": decision.num_contexts,
            "stock_draft": decision.stock_draft,
            "benchmark_max_tokens": decision.benchmark_max_tokens,
            "replay_concurrency": decision.replay_concurrency,
            "correctness_max_tokens": decision.correctness_max_tokens,
            # Passes the correctness replay made.  Without it the
            # ``output_mismatches`` column in ``per_repeat`` cannot be read:
            # rows at or beyond this index were never checked.
            "correctness_repeats": decision.correctness_repeats,
        }
        # The divergence evidence and the floor it was judged against.  Emitted
        # outside the ``deltas_measured`` branch on purpose: an
        # ``output_mismatch`` rejection has no speed deltas at all and is
        # entirely a statement about divergence, so this is the one verdict for
        # which suppressing it would hide the whole basis of the decision.
        #
        # Every control field is ``None`` -- never ``0`` -- when no control ran.
        # A floor that was never measured is not a floor of zero; the criterion
        # only *assumed* that value, and ``control_available`` is what says so.
        control_measured = decision.divergence_control_available
        result["divergence"] = {
            "min_divergence_token_index": decision.min_divergence_token_index,
            "candidate_early": decision.output_early_divergences,
            "candidate_total": decision.output_total_divergences,
            "candidate_trials": decision.divergence_trials or None,
            "candidate_rate": decision.divergence_rate,
            "control_available": control_measured,
            "control_early": (
                decision.control_early_divergences if control_measured else None
            ),
            "control_total": (
                decision.control_total_divergences if control_measured else None
            ),
            "control_trials": decision.control_trials if control_measured else None,
            "control_rate": decision.control_divergence_rate,
            "control_divergences": [
                d.to_dict() for d in decision.control_divergences
            ],
            "total_p_value": decision.divergence_total_p_value,
            "early_p_value": decision.divergence_early_p_value,
            "alpha": decision.divergence_alpha,
        }
        if self.deltas_measured:
            result["measurement"] = {
                "acceptance_criterion": decision.acceptance_criterion,
                "stock_acceptance": decision.stock_avg_acceptance,
                "candidate_acceptance": decision.candidate_avg_acceptance,
                "acceptance_delta_pp": decision.acceptance_delta_pp,
                # A zero standard deviation and a small one are different
                # facts; the basis says which this is, and the standard error
                # is ``null`` -- never ``0.0`` -- when there is no measurement
                # behind it.  See ``gate.decide.DispersionBasis``.
                "acceptance_dispersion": decision.acceptance_dispersion.value,
                "acceptance_delta_standard_error_pp": (
                    decision.acceptance_delta_standard_error_pp
                ),
                "stock_accepted_length": decision.stock_avg_accepted_length,
                "candidate_accepted_length": decision.candidate_avg_accepted_length,
                "accepted_length_delta": decision.accepted_length_delta,
                "accepted_length_dispersion": decision.accepted_length_dispersion.value,
                "accepted_length_delta_standard_error": (
                    decision.accepted_length_delta_standard_error
                ),
                "stock_tok_per_sec": decision.stock_avg_tok_per_sec,
                "candidate_tok_per_sec": decision.candidate_avg_tok_per_sec,
                "throughput_delta_pct": decision.throughput_delta_pct,
                "throughput_dispersion": decision.throughput_dispersion.value,
                "throughput_delta_standard_error_pct": (
                    decision.throughput_delta_standard_error_pct
                ),
            }
            result["per_repeat"] = [
                {
                    "repeat_index": r.repeat_index,
                    "stock_tok_per_sec": r.stock_tok_per_sec,
                    "candidate_tok_per_sec": r.candidate_tok_per_sec,
                    "stock_acceptance_rate": r.stock_acceptance_rate,
                    "candidate_acceptance_rate": r.candidate_acceptance_rate,
                    "invalid_rate": r.invalid_rate,
                    "output_mismatches": r.output_mismatches,
                    "stock_accepted_length": r.stock_accepted_length,
                    "candidate_accepted_length": r.candidate_accepted_length,
                    "stock_finish_reasons": r.stock_finish_reasons,
                    "candidate_finish_reasons": r.candidate_finish_reasons,
                    "stock_truncated": r.stock_truncated,
                    "candidate_truncated": r.candidate_truncated,
                }
                for r in decision.per_repeat
            ]
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def _serving_unrestored_banner(self) -> list[str]:
        """Say up front when the quoted gain is not what is being delivered."""
        marker = self.serving_unrestored
        if marker is None:
            return []
        expected = marker.expected_active_draft or "unknown"
        lines = [
            "!! SERVING NOT RESTORED -- the gain below is NOT being delivered.",
            f"!! The engine is answering live traffic with whichever draft an "
            f"abandoned cycle left loaded; the active pointer names {expected}.",
            "!! Speculative decoding is lossless, so answers are unaffected, but "
            "throughput is unvalidated until the tuner recovers.",
        ]
        if marker.error is not None:
            lines.append(f"!! Restore failed with: {marker.error}")
        lines.append(f"!! Marker: {marker.source_path}")
        return lines

    @staticmethod
    def _measurement_context_lines(decision: Decision) -> list[str]:
        """Render only the context an operator needs to compare two runs.

        Deliberately a subset of what the gate persists.  ``suite_hash`` plus
        ``num_contexts`` identify the held-out suite -- two deltas measured over
        different suites are different measurements; ``stock_draft`` names the
        baseline the delta is *against*, which changes on every promotion; and
        the replay pair fixes the throughput operating point, so a tok/s figure
        from concurrency 8 is not comparable to one from concurrency 1.
        ``correctness_max_tokens`` bounds only the separate correctness replay
        and moves none of the quoted numbers, so it stays in ``to_dict()`` for
        tooling rather than costing the reader a line here.
        """
        lines: list[str] = []
        if decision.suite_hash is not None:
            contexts = (
                "" if decision.num_contexts is None else f" ({decision.num_contexts} contexts)"
            )
            lines.append(f"suite             : {decision.suite_hash}{contexts}")
        if decision.stock_draft is not None:
            lines.append(f"baseline draft    : {decision.stock_draft}")
        if (
            decision.replay_concurrency is not None
            or decision.benchmark_max_tokens is not None
        ):
            concurrency = (
                "unknown"
                if decision.replay_concurrency is None
                else str(decision.replay_concurrency)
            )
            max_tokens = (
                "unknown"
                if decision.benchmark_max_tokens is None
                else str(decision.benchmark_max_tokens)
            )
            lines.append(
                f"replay            : concurrency {concurrency}, "
                f"max {max_tokens} tokens"
            )
        # Truncation regime and rates: how the output cap affected the measurement.
        # Always rendered so the reader can see whether the cap was the binding
        # constraint on generation lengths — and therefore whether throughput is
        # attributable to the workload or to benchmark_max_tokens.
        stock_regime = decision.stock_truncation_regime.value
        cand_regime = decision.candidate_truncation_regime.value
        stock_rate = decision.stock_truncation_rate
        cand_rate = decision.candidate_truncation_rate
        # When the rate is None, no finish reasons were reported — the measurement
        # was never taken. Rendering it as 0% would be fabricated precision: the
        # rate is unknown, not absent.
        stock_rate_str = (
            f"{stock_rate * 100:.1f}%" if stock_rate is not None else "unmeasured"
        )
        cand_rate_str = (
            f"{cand_rate * 100:.1f}%" if cand_rate is not None else "unmeasured"
        )
        lines.append(
            f"truncation        : stock {stock_regime} ({stock_rate_str}), "
            f"candidate {cand_regime} ({cand_rate_str})"
        )
        return lines

    @staticmethod
    def _divergence_lines(decision: Decision) -> list[str]:
        """Render the divergence evidence *and the floor it was judged against*.

        A raw divergence count is not readable on its own.  The gate's own
        criterion compares the candidate-versus-stock count against a
        stock-versus-stock control replayed in the same run, because a target
        forward pass is not bitwise reproducible and a MoE target diverges from
        itself on a majority of contexts.  Printing "230 contexts parted" with
        no floor beside it invites exactly the false-rejection reading the
        criterion was changed to stop, so the two rates are rendered together or
        the absence of the floor is stated outright.
        """
        lines: list[str] = []
        if decision.output_divergences:
            offsets = sorted(
                d.first_divergence_index for d in decision.output_divergences
            )
            lines.append(
                f"divergences       : {len(offsets)} contexts parted "
                f"({decision.output_early_divergences} before token "
                f"{decision.min_divergence_token_index}); "
                f"first-divergence offsets min {offsets[0]}, "
                f"median {offsets[len(offsets) // 2]}, max {offsets[-1]}"
            )
        candidate_rate = decision.divergence_rate
        if candidate_rate is not None:
            lines.append(
                f"divergence rate   : candidate {decision.output_total_divergences}"
                f"/{decision.divergence_trials} ({candidate_rate * 100:.2f}% of "
                f"context comparisons)"
            )
        control_rate = decision.control_divergence_rate
        if control_rate is not None:
            lines.append(
                f"engine noise floor: stock vs stock "
                f"{decision.control_total_divergences}/{decision.control_trials} "
                f"({control_rate * 100:.2f}%), "
                f"{decision.control_early_divergences} before token "
                f"{decision.min_divergence_token_index}"
            )
        elif candidate_rate is not None:
            # Never zeros: a floor that was not measured is not a floor of
            # zero, and the criterion only *assumed* that value.
            #
            # Gated on the candidate rate rather than on the divergence list,
            # so it appears only for records the divergence criterion actually
            # judged.  A record written before the criterion existed has no
            # trials and gets no line: it was gated by the old
            # any-early-divergence rule, against no floor at all, and saying
            # "the criterion assumed a zero floor" would describe a test that
            # never ran on it.
            # Worded as the missing *control*, not as "not measured": in this
            # report "not measured" is the phrase for a statistic with no
            # numbers behind it at all, and the divergence counts above are
            # measured -- it is only the floor they should be read against that
            # is absent.
            lines.append(
                "engine noise floor: no stock-vs-stock control ran (the "
                "criterion assumed a zero-divergence floor)"
            )
        p_total = decision.divergence_total_p_value
        p_early = decision.divergence_early_p_value
        if p_total is not None or p_early is not None:
            total_str = "n/a" if p_total is None else f"{p_total:.4f}"
            early_str = "n/a" if p_early is None else f"{p_early:.4f}"
            lines.append(
                f"divergence test   : total p {total_str}, early p {early_str} "
                f"(one-sided Fisher exact; rejects below alpha "
                f"{decision.divergence_alpha:.4f})"
            )
        return lines

    def render_text(self) -> str:
        lines = ["SpeedLM gain"]
        # First, before any number: whatever follows may not be what is being
        # served.  A banner under the figures would be read after them.
        lines.extend(self._serving_unrestored_banner())
        if self.source_path is not None:
            lines.append(f"source            : {self.source_path}")
            if self.source_mtime is not None:
                mtime_str = datetime.datetime.fromtimestamp(
                    self.source_mtime, tz=datetime.UTC
                ).isoformat()
                lines.append(f"source mtime      : {mtime_str}")

        decision = self.decision
        if decision is None:
            lines.append(f"measurement       : {self.status.value}")
            lines.append(self.detail)
            return "\n".join(lines)

        lines.append(f"verdict           : {decision.verdict.value}")
        lines.append(
            f"reason            : {decision.reason.value} "
            f"({_REASON_EXPLANATIONS[decision.reason]})"
        )
        lines.append(f"repeats           : {decision.num_repeats}")
        lines.extend(self._measurement_context_lines(decision))

        if not self.deltas_measured:
            lines.append("acceptance        : not measured")
            lines.append("throughput        : not measured")
            lines.append("speedup           : not measured")
            # An ``output_mismatch`` rejection is unmeasured on the speed
            # statistics and yet is *entirely* about divergence -- so this is
            # precisely the verdict whose evidence, and whose noise floor, the
            # reader most needs.  Withholding it here would leave the operator
            # with a bare "rejected" and nothing to audit.
            lines.extend(self._divergence_lines(decision))
            lines.append(self.detail)
            return "\n".join(lines)

        acceptance_delta_pp = decision.acceptance_delta_pp
        throughput_delta_pct = decision.throughput_delta_pct
        assert throughput_delta_pct is not None
        repeats = len(decision.per_repeat)
        # Each acceptance statistic renders only when the record carries it.
        # ``deltas_measured`` admits a record holding either one, so neither
        # block may assume the other's presence: a legacy record has only the
        # rate delta below, a record written under the current criterion has
        # both, and a record with only the length delta prints only that.
        if acceptance_delta_pp is not None:
            lines.append(
                f"acceptance stock  : {decision.stock_avg_acceptance * 100:.2f}%"
            )
            lines.append(
                f"acceptance cand   : {decision.candidate_avg_acceptance * 100:.2f}%"
            )
            acceptance_suffix, acceptance_note = _dispersion_fragments(
                decision.acceptance_dispersion,
                decision.acceptance_delta_standard_error_pp,
                repeats=repeats,
                unit=" pp",
            )
            # The rate delta is reported but no longer decides -- it divides by
            # the draft depth, so it is not comparable across depths.  Saying
            # "threshold >= 1.00 pp" here would name a bar the gate does not
            # apply.
            lines.append(
                f"acceptance delta  : {acceptance_delta_pp:+.2f} pp{acceptance_suffix} "
                f"({acceptance_note}; recorded, not gated)"
            )
        accepted_length_delta = decision.accepted_length_delta
        if accepted_length_delta is not None:
            lines.append(
                f"accepted len stock: "
                f"{decision.stock_avg_accepted_length:.3f} tok/step"
            )
            lines.append(
                f"accepted len cand : "
                f"{decision.candidate_avg_accepted_length:.3f} tok/step"
            )
            length_suffix, length_note = _dispersion_fragments(
                decision.accepted_length_dispersion,
                decision.accepted_length_delta_standard_error,
                repeats=repeats,
                unit=" tok/step",
            )
            threshold = (
                f">= {decision.min_accepted_length_delta:.3f} tok/step"
                if decision.min_accepted_length_delta is not None
                else "not recorded"
            )
            lines.append(
                f"accepted len delta: {accepted_length_delta:+.3f} tok/step"
                f"{length_suffix} ({length_note}; threshold {threshold})"
            )
        lines.append(
            f"throughput stock  : {decision.stock_avg_tok_per_sec:.2f} tok/s"
        )
        lines.append(
            f"throughput cand   : {decision.candidate_avg_tok_per_sec:.2f} tok/s"
        )
        throughput_suffix, throughput_note = _dispersion_fragments(
            decision.throughput_dispersion,
            decision.throughput_delta_standard_error_pct,
            repeats=repeats,
            unit="%",
        )
        lines.append(
            f"throughput delta  : {throughput_delta_pct:+.2f}%{throughput_suffix} "
            f"({throughput_note}; threshold >= "
            f"{decision.min_throughput_delta_pct:.2f}%)"
        )
        if decision.per_repeat:
            lines.append("per-repeat:")
            for r in decision.per_repeat:
                # Truncation rates are computed per-arm, per-repeat from the raw
                # counts so the reader can see where the cap bit.  When the
                # denominator is zero (no finish reasons reported) we render
                # "n/a" rather than 0.00% — a zero denominator means the
                # measurement was never taken, not that nothing was truncated.
                stock_trunc = (
                    f"{r.stock_truncated / r.stock_finish_reasons * 100:.1f}%"
                    if r.stock_finish_reasons > 0
                    else "n/a"
                )
                cand_trunc = (
                    f"{r.candidate_truncated / r.candidate_finish_reasons * 100:.1f}%"
                    if r.candidate_finish_reasons > 0
                    else "n/a"
                )
                lines.append(
                    f"  [{r.repeat_index}] stock {r.stock_tok_per_sec:.2f} tok/s, "
                    f"candidate {r.candidate_tok_per_sec:.2f} tok/s, "
                    f"acceptance {r.stock_acceptance_rate * 100:.2f}% -> "
                    f"{r.candidate_acceptance_rate * 100:.2f}%, "
                    f"invalid {r.invalid_rate * 100:.2f}%, "
                    f"mismatches {r.output_mismatches}, "
                    f"truncation stock {stock_trunc}, candidate {cand_trunc}"
                )
        lines.extend(self._divergence_lines(decision))
        lines.append(self.detail)
        return "\n".join(lines)


def _gain_detail(decision: Decision) -> str:
    reason = decision.reason
    explanation = _REASON_EXPLANATIONS[reason]
    if reason is Reason.COUNTER_RESET:
        return (
            "No gain can be reported: a vLLM counter reset invalidated the benchmark "
            "window. Re-run the gate against a stable server."
        )
    if reason is Reason.ACCEPTANCE_UNAVAILABLE:
        return (
            "Acceptance is UNAVAILABLE: vLLM exposed no speculative-decoding counters, "
            "so neither acceptance nor a speedup was measured."
        )
    if reason in UNMEASURED_REASONS:
        return f"No gain can be reported: {explanation}."
    if decision.verdict is Verdict.PROMOTE:
        return "Verdict: the candidate draft was promoted on measured gains."
    return f"Verdict: the candidate draft was rejected because {explanation}."


def build_gain_report(
    *,
    home: Path | None = None,
    now: float | None = None,
) -> GainReport:
    """Load the most recent gate decision and report it without embellishment."""
    layout = resolve_layout(home)
    timestamp = time.time() if now is None else now
    # Read before the decision, and attached to every outcome below: whether a
    # gain is being *delivered* is independent of whether one was *measured*,
    # and the incident is worth reporting even when there is no number at all.
    unrestored = read_serving_unrestored(layout)

    path = find_latest_decision(layout)
    if path is None:
        return GainReport(
            status=GainStatus.NO_GATE_RUN,
            detail=(
                "No gate has ever run: there is no completed benchmark in "
                f"{layout.runs_dir}, so there is no measured gain to report."
            ),
            serving_unrestored=unrestored,
            generated_at=timestamp,
        )

    try:
        decision = load_decision(path)
    except ReportError as exc:
        try:
            mtime_val = path.stat().st_mtime
        except OSError:
            mtime_val = None
        return GainReport(
            status=GainStatus.UNREADABLE,
            detail=(
                f"The most recent gate decision could not be read ({exc}); "
                "no gain is being reported."
            ),
            source_path=path,
            source_mtime=mtime_val,
            serving_unrestored=unrestored,
            generated_at=timestamp,
        )

    # Provenance: structural consistency check
    # When the gate completes repeats it writes per_repeat entries.  If per_repeat
    # is non-empty but num_repeats disagrees, the file is untrusted.
    # (Aborted decisions may legitimately have num_repeats > 0 with empty per_repeat.)
    if decision.per_repeat and decision.num_repeats != len(decision.per_repeat):
        try:
            mtime_val = path.stat().st_mtime
        except OSError:
            mtime_val = None
        return GainReport(
            status=GainStatus.UNREADABLE,
            detail=(
                f"The gate decision at {path} has inconsistent provenance "
                f"(num_repeats={decision.num_repeats} but "
                f"len(per_repeat)={len(decision.per_repeat)}); no gain is being reported."
            ),
            source_path=path,
            source_mtime=mtime_val,
            serving_unrestored=unrestored,
            generated_at=timestamp,
        )

    # Capture file mtime
    try:
        mtime_val = path.stat().st_mtime
    except OSError:
        mtime_val = None

    measured = decision.reason not in UNMEASURED_REASONS
    return GainReport(
        status=GainStatus.MEASURED if measured else GainStatus.NOT_MEASURED,
        detail=_gain_detail(decision),
        source_path=path,
        source_mtime=mtime_val,
        decision=decision,
        serving_unrestored=unrestored,
        generated_at=timestamp,
    )
