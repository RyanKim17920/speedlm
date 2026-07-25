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
from speedlm.doctor import PRIMARY_DRAFT, PRIMARY_VERIFIER, SUPPORTED_MODEL_PAIRS
from speedlm.gate.decide import Decision, Reason, RepeatSummary, Verdict
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
DECISION_FILE_NAME: Final = "decision.json"

#: Reasons whose ``Decision`` carries zeroed deltas because the gate aborted
#: before it could measure anything.  Their numbers must never be reported.
UNMEASURED_REASONS: Final[frozenset[Reason]] = frozenset(
    {
        Reason.COUNTER_RESET,
        Reason.ACCEPTANCE_UNAVAILABLE,
        Reason.TOO_FEW_REPEATS,
        Reason.HIGH_INVALID_RATE,
        Reason.OUTPUT_MISMATCH,
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
    Reason.HIGH_INVALID_RATE: "too many replayed requests failed to produce valid output",
    Reason.TOO_FEW_REPEATS: "the benchmark did not complete enough repeats to be meaningful",
    Reason.OUTPUT_MISMATCH: "candidate output diverged from stock output",
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
# status: model pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPairStatus:
    """The configured verifier/draft pair and where it came from."""

    verifier: str
    draft: str
    configured: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "verifier": self.verifier,
            "draft": self.draft,
            "configured": self.configured,
            "detail": self.detail,
        }


def read_model_pair(layout: Layout) -> ModelPairStatus:
    """Resolve the model pair from ``<home>/config.json``, else the built-in default."""
    path = layout.root / CONFIG_FILE_NAME
    if not path.exists():
        return ModelPairStatus(
            verifier=PRIMARY_VERIFIER,
            draft=PRIMARY_DRAFT,
            configured=False,
            detail=f"built-in default pair (no {path})",
        )
    try:
        config: SpeedLMConfig = load_config(path)
    except ConfigError as exc:
        return ModelPairStatus(
            verifier=PRIMARY_VERIFIER,
            draft=PRIMARY_DRAFT,
            configured=False,
            detail=f"built-in default pair ({path} is unusable: {exc})",
        )

    draft = SUPPORTED_MODEL_PAIRS.get(config.model)
    if draft is None:
        return ModelPairStatus(
            verifier=config.model,
            draft="unknown",
            configured=True,
            detail=(
                f"configured verifier {config.model!r} has no supported EAGLE-3 draft; "
                "run 'speedlm doctor' for details"
            ),
        )
    return ModelPairStatus(
        verifier=config.model,
        draft=draft,
        configured=True,
        detail=f"configured in {path}",
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
            "models": self.models.to_dict(),
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
    return StatusReport(
        home=layout.root,
        home_exists=layout.root.is_dir(),
        gateway=read_gateway_status(layout),
        active_draft=read_active_draft(layout),
        traces=read_trace_status(layout),
        tuner=read_tuner_status(layout),
        models=read_model_pair(layout),
        generated_at=time.time() if now is None else now,
    )


# ---------------------------------------------------------------------------
# gain: decision loading
# ---------------------------------------------------------------------------


def find_latest_decision(layout: Layout) -> Path | None:
    """Return the most recently modified ``decision.json`` under the runs dir."""
    runs_dir = layout.runs_dir
    if not runs_dir.is_dir():
        return None
    candidates: list[Path] = []
    direct = runs_dir / DECISION_FILE_NAME
    if direct.is_file():
        candidates.append(direct)
    for pattern in (f"*/{DECISION_FILE_NAME}", f"*/*/{DECISION_FILE_NAME}"):
        candidates.extend(path for path in runs_dir.glob(pattern) if path.is_file())
    if not candidates:
        return None

    def _mtime(path: Path) -> tuple[float, str]:
        try:
            return (path.stat().st_mtime, str(path))
        except OSError:
            return (0.0, str(path))

    return max(candidates, key=_mtime)


def _require_float(record: Mapping[str, Any], key: str, source: Path) -> float:
    value = _optional_float(record.get(key))
    if value is None:
        raise ReportError(f"{source}: '{key}' must be numeric")
    return value


def _require_int(record: Mapping[str, Any], key: str, source: Path) -> int:
    value = _optional_int(record.get(key))
    if value is None:
        raise ReportError(f"{source}: '{key}' must be an integer")
    return value


def _parse_repeat(record: Mapping[str, Any], source: Path) -> RepeatSummary:
    return RepeatSummary(
        repeat_index=_require_int(record, "repeat_index", source),
        stock_tok_per_sec=_require_float(record, "stock_tok_per_sec", source),
        candidate_tok_per_sec=_require_float(record, "candidate_tok_per_sec", source),
        stock_acceptance_rate=_require_float(record, "stock_acceptance_rate", source),
        candidate_acceptance_rate=_require_float(record, "candidate_acceptance_rate", source),
        invalid_rate=_require_float(record, "invalid_rate", source),
        output_mismatches=_require_int(record, "output_mismatches", source),
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
        acceptance_delta_pp=_require_float(record, "acceptance_delta_pp", source),
        throughput_delta_pct=_require_float(record, "throughput_delta_pct", source),
        min_acceptance_delta_pp=_require_float(record, "min_acceptance_delta_pp", source),
        min_throughput_delta_pct=_require_float(record, "min_throughput_delta_pct", source),
        num_repeats=_require_int(record, "num_repeats", source),
        per_repeat=tuple(per_repeat),
        stock_avg_acceptance=_require_float(record, "stock_avg_acceptance", source),
        candidate_avg_acceptance=_require_float(record, "candidate_avg_acceptance", source),
        stock_avg_tok_per_sec=_require_float(record, "stock_avg_tok_per_sec", source),
        candidate_avg_tok_per_sec=_require_float(record, "candidate_avg_tok_per_sec", source),
    )


def load_decision(path: Path) -> Decision:
    """Load and validate a persisted gate decision."""
    return parse_decision(_read_json_object(path), source=path)


# ---------------------------------------------------------------------------
# gain report
# ---------------------------------------------------------------------------


class GainStatus(StrEnum):
    """Whether a real, trustworthy measurement exists."""

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    NO_GATE_RUN = "no_gate_run"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class GainReport:
    """Measured benefit of the current candidate draft, or an honest denial."""

    status: GainStatus
    detail: str
    source_path: Path | None = None
    source_mtime: float | None = None
    decision: Decision | None = None
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
        """True only when the gate actually produced comparable numbers."""
        return self.decision is not None and self.decision.reason not in UNMEASURED_REASONS

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
            "min_throughput_delta_pct": decision.min_throughput_delta_pct,
        }
        if self.deltas_measured:
            result["measurement"] = {
                "stock_acceptance": decision.stock_avg_acceptance,
                "candidate_acceptance": decision.candidate_avg_acceptance,
                "acceptance_delta_pp": decision.acceptance_delta_pp,
                "stock_tok_per_sec": decision.stock_avg_tok_per_sec,
                "candidate_tok_per_sec": decision.candidate_avg_tok_per_sec,
                "throughput_delta_pct": decision.throughput_delta_pct,
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
                }
                for r in decision.per_repeat
            ]
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        lines = ["SpeedLM gain"]
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

        if not self.deltas_measured:
            lines.append("acceptance        : not measured")
            lines.append("throughput        : not measured")
            lines.append("speedup           : not measured")
            lines.append(self.detail)
            return "\n".join(lines)

        lines.append(
            f"acceptance stock  : {decision.stock_avg_acceptance * 100:.2f}%"
        )
        lines.append(
            f"acceptance cand   : {decision.candidate_avg_acceptance * 100:.2f}%"
        )
        lines.append(
            f"acceptance delta  : {decision.acceptance_delta_pp:+.2f} pp "
            f"(threshold >= {decision.min_acceptance_delta_pp:.2f} pp)"
        )
        lines.append(
            f"throughput stock  : {decision.stock_avg_tok_per_sec:.2f} tok/s"
        )
        lines.append(
            f"throughput cand   : {decision.candidate_avg_tok_per_sec:.2f} tok/s"
        )
        lines.append(
            f"throughput delta  : {decision.throughput_delta_pct:+.2f}% "
            f"(threshold >= {decision.min_throughput_delta_pct:.2f}%)"
        )
        if decision.per_repeat:
            lines.append("per-repeat:")
            for r in decision.per_repeat:
                lines.append(
                    f"  [{r.repeat_index}] stock {r.stock_tok_per_sec:.2f} tok/s, "
                    f"candidate {r.candidate_tok_per_sec:.2f} tok/s, "
                    f"acceptance {r.stock_acceptance_rate * 100:.2f}% -> "
                    f"{r.candidate_acceptance_rate * 100:.2f}%, "
                    f"invalid {r.invalid_rate * 100:.2f}%, "
                    f"mismatches {r.output_mismatches}"
                )
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

    path = find_latest_decision(layout)
    if path is None:
        return GainReport(
            status=GainStatus.NO_GATE_RUN,
            detail=(
                "No gate has ever run: there is no completed benchmark in "
                f"{layout.runs_dir}, so there is no measured gain to report."
            ),
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
        generated_at=timestamp,
    )
