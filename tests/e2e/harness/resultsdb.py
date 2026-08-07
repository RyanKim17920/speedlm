"""Index the historical benchmark runs under ``/data/ryan.kim/speedlm-runs``.

Sixty-five runs accumulated between 2026-07-31 and 2026-08-07 in eleven launcher
flavors, written by code that changed underneath them.  Nothing aggregated them,
so every conclusion the project reached was reconstructed by hand out of JSON.
This module turns that tree into :class:`~model.RunRecord` values the comparator
can diff.

Three properties are load-bearing and are the reason the code is shaped the way
it is:

* **Individual observations survive.**  Wherever an artifact preserved the
  per-repeat list, :attr:`~model.MetricSeries.values` *is* that list.
  ``reduced=True`` is set only where the artifact genuinely kept nothing but a
  summary, because a mean cannot be given a standard error afterwards and this
  project has published deltas smaller than one standard error.
* **Pass indices survive.**  Every extractor that has an artifact-recorded
  notion of "which pass this observation came from" carries it into
  :attr:`~model.MetricSeries.pass_indices`: the per-arm repeat ordinal for the
  ABBA config matrix, the ``(cycle, half, prompt)`` cell for the capture
  overhead bench, the shared ``repeat_index`` for the idle-tuning gate, the
  request seed for the proxy bench.  Where the artifact does not record one, or
  where the recorded one does not account for the observations present, the
  field is ``None`` and a warning says so -- the comparator then falls back to
  positional pairing and reports that it did.
* **Nothing is dropped in silence.**  Files under ``results/`` land in exactly
  one of three places: consumed by an extractor, matched by a *named* policy in
  :data:`NON_MEASUREMENT_PATTERNS` and reported as one aggregated line in
  :attr:`~model.RunRecord.unparsed`, or reported individually in ``unparsed``
  with the reason it could not be read.  An index that skips what it cannot
  parse reports a clean history it has not got.

Extractors are registered in :data:`EXTRACTORS`, a priority-ordered table.  Each
entry declares the basenames it requires inside one directory; adding an
artifact shape is a new registration, not a new branch.

Metric direction is a *declaration*, never an inference: :data:`METRICS` maps
every indexed metric name to its unit and :class:`~model.Direction`.  A numeric
field that is not in that table is not indexed at all -- it is reported in the
cell's warnings.  Guessing "latency" from a name is how a comparator eventually
reports a throughput regression as an improvement.

Prometheus text is parsed by :func:`speedlm.gate.metrics.parse_metrics` and
differenced by :func:`speedlm.gate.metrics.compute_delta`.  The repository
already contains three Prometheus parsers; this is not the fourth.
"""

from __future__ import annotations

import fnmatch
import gzip
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from speedlm.gate.metrics import (
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)

from .model import CellRecord, Direction, MetricSeries, RunRecord

__all__ = [
    "EXTRACTORS",
    "METRICS",
    "Extractor",
    "FileStamp",
    "MetricSpec",
    "ResultsIndex",
    "build_index",
    "index_runs",
    "load_run",
]


# ---------------------------------------------------------------------------
# Metric declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """The unit and the direction of one indexed metric.

    ``direction`` is declared here and nowhere else.  It drives the
    comparator's regression verdicts, and the cost of getting it wrong is a
    published finding with its sign inverted.
    """

    unit: str
    direction: Direction


_HIGHER = Direction.HIGHER_IS_BETTER
_LOWER = Direction.LOWER_IS_BETTER
_NEUTRAL = Direction.NEUTRAL

#: Every metric this module will emit.  A numeric field absent from this table
#: is deliberately *not* indexed: see the module docstring.
METRICS: Final[dict[str, MetricSpec]] = {
    # --- throughput -------------------------------------------------------
    "tokens_per_second": MetricSpec("tok/s", _HIGHER),
    "batch_tokens_per_second": MetricSpec("tok/s", _HIGHER),
    "throughput_tok_per_sec": MetricSpec("tok/s", _HIGHER),
    "tok_per_sec": MetricSpec("tok/s", _HIGHER),
    "output_tok_per_sec": MetricSpec("tok/s", _HIGHER),
    "prometheus_decode_tok_per_sec": MetricSpec("tok/s", _HIGHER),
    # --- acceptance -------------------------------------------------------
    "mean_accepted_length": MetricSpec("tokens", _HIGHER),
    "accepted_length": MetricSpec("tokens", _HIGHER),
    "acceptance_rate": MetricSpec("ratio", _HIGHER),
    # --- latency ----------------------------------------------------------
    "elapsed_seconds": MetricSpec("s", _LOWER),
    "e2e_seconds": MetricSpec("s", _LOWER),
    "ttft_seconds": MetricSpec("s", _LOWER),
    "first_chunk_seconds": MetricSpec("s", _LOWER),
    "last_chunk_seconds": MetricSpec("s", _LOWER),
    "median_inter_token_seconds": MetricSpec("s", _LOWER),
    "inter_token_seconds": MetricSpec("s", _LOWER),
    "tpot_ms": MetricSpec("ms", _LOWER),
    "exchange_seconds": MetricSpec("s", _LOWER),
    "queued_request_seconds": MetricSpec("s", _LOWER),
    # --- error rates ------------------------------------------------------
    "invalid_rate": MetricSpec("ratio", _LOWER),
    "divergence_rate": MetricSpec("ratio", _LOWER),
    # --- descriptive counts ----------------------------------------------
    "completion_tokens": MetricSpec("tokens", _NEUTRAL),
    "prompt_tokens": MetricSpec("tokens", _NEUTRAL),
    "total_completion_tokens": MetricSpec("tokens", _NEUTRAL),
    "generated_tokens": MetricSpec("tokens", _NEUTRAL),
    "drafted_tokens": MetricSpec("tokens", _NEUTRAL),
    "accepted_tokens": MetricSpec("tokens", _NEUTRAL),
    "token_event_count": MetricSpec("events", _NEUTRAL),
    "event_count": MetricSpec("events", _NEUTRAL),
    "output_mismatches": MetricSpec("count", _NEUTRAL),
    "checks_passed": MetricSpec("count", _NEUTRAL),
    "checks_failed": MetricSpec("count", _NEUTRAL),
    "cases_passed": MetricSpec("count", _NEUTRAL),
    "cases_failed": MetricSpec("count", _NEUTRAL),
    "injected_rpc_calls": MetricSpec("count", _NEUTRAL),
    "exchange_count": MetricSpec("count", _NEUTRAL),
    "captured_rows_per_layer": MetricSpec("rows", _NEUTRAL),
    "captured_layer_count": MetricSpec("layers", _NEUTRAL),
    "prefix_cache_hit_tokens": MetricSpec("tokens", _NEUTRAL),
    "prompt_rows_missing": MetricSpec("rows", _NEUTRAL),
    "prompt_token_count": MetricSpec("tokens", _NEUTRAL),
    # --- draft-head training validation -----------------------------------
    "val_loss": MetricSpec("nats", _LOWER),
    "val_full_accuracy": MetricSpec("ratio", _HIGHER),
    "val_cond_accuracy": MetricSpec("ratio", _HIGHER),
}

#: Reasons for files that reach the end of the extractor table unclaimed and
#: are *recognised*: configuration, state, provenance, or half of a pair whose
#: other half the run never wrote.  They are still reported in ``unparsed`` --
#: this is a statement about the artifact, not an admission that the index
#: failed to read something it should have.
CONFIGURATION_ARTIFACTS: Final[dict[str, str]] = {
    "metrics-before.prom": (
        "baseline Prometheus scrape with no metrics-after.prom beside it; "
        "no delta is computable from one scrape"
    ),
    "config.json": "run configuration, not a measurement",
    "config.py": "run configuration, not a measurement",
    "invocation.json": "process invocation record, not a measurement",
    "shutdown.json": "process teardown record, not a measurement",
    "command.json": "engine command line, not a measurement",
    "state.json": "tuner state snapshot, not a measurement",
    "scheduler.json": "tuner scheduler state, not a measurement",
    "terminal-state.json": "tuner state snapshot, not a measurement",
    "terminal-scheduler.json": "tuner scheduler state, not a measurement",
    "suite_manifest.json": "held-out suite manifest, not a measurement",
    "engine-regime.json": "engine configuration, not a measurement",
    "expected-profile.json": "expected profile declaration, not a measurement",
    "speedlm-doctor.json": "CLI diagnostic output, not a measurement",
    "speedlm-status.json": "CLI status output, not a measurement",
    "artifact-depth-observation.json": "draft artifact fingerprints, not a measurement",
    "traces.json": "captured trace index, not a measurement",
}


# ---------------------------------------------------------------------------
# Files that are payload, not measurement
# ---------------------------------------------------------------------------

#: Basename globs for files under ``results/`` that are request/response
#: payloads, engine logs, model weights or dataset shards rather than
#: measurements.  Matches are still *reported* -- one aggregated line per run in
#: :attr:`~model.RunRecord.unparsed` -- but they are not individually itemised,
#: because a single idle-tuning run contributes several thousand of them.  Each
#: entry carries the reason it is here.
NON_MEASUREMENT_PATTERNS: Final[dict[str, str]] = {
    "seed-response-*.json": "captured seeding response body",
    "queued-response.json": "captured response body",
    "nonstream-response.*": "captured response body",
    "stream-response.*": "captured response body",
    "request.body": "captured request body",
    "response.body": "captured response body",
    "query.bin": "captured request query string",
    "*.log": "engine/gateway log",
    "*.txt": "human-readable transcript of a JSON sibling",
    "*.jsonl": "append-only event/trace stream",
    "*.lock": "file lock",
    "*.safetensors": "tensor payload",
    "*.meta.json": "tensor payload sidecar",
    "*.arrow": "dataset shard",
    "*.pt": "tensor payload",
}

#: Directory names under ``results/`` whose entire subtree is capture payload.
NON_MEASUREMENT_DIRS: Final[frozenset[str]] = frozenset(
    {"exchanges", "captured", "offline", "offline_ds", "offline_hs", "offline_prepared"}
)


# ---------------------------------------------------------------------------
# Flavor recovery
# ---------------------------------------------------------------------------

#: ``#SBATCH --job-name=speedlm-snap-<suffix>`` suffix -> launcher flavor.  The
#: suffixes are ``job_suffix`` in ``scripts/make_snapshot_run.sh``.
_JOB_SUFFIX_TO_FLAVOR: Final[dict[str, str]] = {
    "idle": "idle-tuning",
    "capture": "activation-capture",
    "capoverhead": "capture-overhead",
    "hotswap": "hot-swap",
    "livevllm": "live-vllm",
    "overhead": "proxy-overhead",
    "fidelity": "token-fidelity",
    "matrix": "model-matrix",
    "capmatrix": "capture-matrix",
    "agent": "agent-harness",
    "configmatrix": "config-matrix",
}

#: Test module -> launcher flavor, for the hand-written sbatch files that
#: predate ``make_snapshot_run.sh`` and carry no job-name suffix.
_TEST_TO_FLAVOR: Final[dict[str, str]] = {
    "test_live_idle_tuning": "idle-tuning",
    "test_serving_activation_capture": "activation-capture",
    "test_serving_activation_capture_overhead": "capture-overhead",
    "test_serving_draft_hot_swap": "hot-swap",
    "test_live_vllm": "live-vllm",
    "test_proxy_overhead": "proxy-overhead",
    "test_token_fidelity": "token-fidelity",
    "test_model_matrix": "model-matrix",
    "test_capture_harness_matrix": "capture-matrix",
    "test_agent_harness": "agent-harness",
    "test_inference_configuration_matrix": "config-matrix",
}

_JOB_NAME_RE: Final[re.Pattern[str]] = re.compile(r"--job-name=speedlm-snap-(\S+)")
_TEST_PATH_RE: Final[re.Pattern[str]] = re.compile(r"tests/e2e/(test_[A-Za-z0-9_]+)\.py")
_SBATCH_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^commit=([0-9a-f]{40})\s*$", re.M)
_SBATCH_SNAPSHOT_RE: Final[re.Pattern[str]] = re.compile(r"^snapshot=(\S+)\s*$", re.M)
_SBATCH_GENERATED_RE: Final[re.Pattern[str]] = re.compile(
    r"^# GENERATED by .* on (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", re.M
)
_SLURM_OUT_RE: Final[re.Pattern[str]] = re.compile(r"^slurm-(\d+)\.out$")
_SLURM_STARTED_RE: Final[re.Pattern[str]] = re.compile(
    r"^started_at=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", re.M
)
_DIRNAME_STAMP_RE: Final[re.Pattern[str]] = re.compile(r"(\d{8}T\d{6}Z)")


# ---------------------------------------------------------------------------
# Extractor plumbing
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What one extractor produced from one directory."""

    cells: list[CellRecord] = field(default_factory=list)
    #: Files the extractor is responsible for; they will not be reported unread.
    consumed: set[Path] = field(default_factory=set)
    #: Files the extractor found but could not use: path -> reason.
    unparsed: dict[Path, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Extractor:
    """One artifact shape.

    ``required`` and ``optional`` are basenames within a single directory;
    ``glob`` is an optional basename pattern used by shapes that address a
    variable family of files (the gate's per-repeat Prometheus scrapes).  The
    extractor is offered the directory when every ``required`` basename is
    present, in :data:`EXTRACTORS` order.
    """

    name: str
    required: tuple[str, ...]
    extract: Callable[[Path, dict[str, Path]], Outcome]
    optional: tuple[str, ...] = ()
    glob: str | None = None

    def claims(self, files: dict[str, Path]) -> bool:
        if self.required and not all(n in files for n in self.required):
            return False
        if self.glob is not None:
            return any(fnmatch.fnmatch(n, self.glob) for n in files)
        return bool(self.required)


def _spec(name: str) -> MetricSpec | None:
    return METRICS.get(name)


def _series(
    name: str,
    values: Sequence[float],
    *,
    pass_indices: Sequence[int] | None = None,
    reduced: bool = False,
    warnings: list[str],
) -> MetricSeries | None:
    """Build a series, or record why it was not built.

    Returns ``None`` -- and appends to *warnings* -- when the metric has no
    declared direction, or when there is nothing to record.  Refusing to index
    an undeclared metric is deliberate: see the module docstring.
    """
    spec = _spec(name)
    if spec is None:
        warnings.append(f"metric {name!r} has no declared direction; not indexed")
        return None
    numeric: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"metric {name!r} has a non-numeric observation; not indexed")
            return None
        numeric.append(float(value))
    if not numeric:
        return None
    if reduced and len(numeric) != 1:
        warnings.append(f"metric {name!r} marked reduced with {len(numeric)} values; not indexed")
        return None
    indices: tuple[int, ...] | None = None
    if pass_indices is not None:
        if len(pass_indices) != len(numeric):
            warnings.append(
                f"metric {name!r}: {len(pass_indices)} pass indices for "
                f"{len(numeric)} values; falling back to positional pairing"
            )
        else:
            indices = tuple(int(i) for i in pass_indices)
    return MetricSeries(
        name=name,
        unit=spec.unit,
        direction=spec.direction,
        values=tuple(numeric),
        pass_indices=indices,
        reduced=reduced,
    )


def _collect(
    samples: Sequence[dict[str, Any]],
    fields: Iterable[str],
    *,
    pass_indices: Sequence[int] | None,
    warnings: list[str],
) -> dict[str, MetricSeries]:
    """Turn a list of per-observation dicts into one series per declared field.

    Observations missing a field are skipped for that field only, and the
    corresponding pass indices are dropped with them, so a series never carries
    a pass index belonging to a different observation.
    """
    out: dict[str, MetricSeries] = {}
    for name in fields:
        values: list[float] = []
        kept: list[int] = []
        for position, sample in enumerate(samples):
            if name not in sample:
                continue
            raw = sample[name]
            if isinstance(raw, list):
                # A per-token list (proxy stream inter-token gaps).  The median
                # is the artifact's own summary statistic for it; recording it
                # as one observation of the request keeps requests, not tokens,
                # as the unit of repetition.
                numbers = [v for v in raw if isinstance(v, (int, float))]
                if not numbers:
                    continue
                numbers.sort()
                mid = len(numbers) // 2
                raw = (
                    numbers[mid]
                    if len(numbers) % 2
                    else (numbers[mid - 1] + numbers[mid]) / 2.0
                )
            values.append(raw)
            if pass_indices is not None and position < len(pass_indices):
                kept.append(pass_indices[position])
        if not values:
            continue
        series = _series(
            name,
            values,
            pass_indices=kept if pass_indices is not None and len(kept) == len(values) else None,
            warnings=warnings,
        )
        if series is not None:
            out[name] = series
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def _numeric_fields(samples: Sequence[dict[str, Any]]) -> list[str]:
    """Every field name that appears with a numeric (or list) value."""
    seen: dict[str, None] = {}
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, list)):
                seen.setdefault(key, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Extractor: config matrix cell (results/<cell>/result.json)
# ---------------------------------------------------------------------------


def _config_matrix_pass_indices(
    payload: dict[str, Any], arm: str, samples: Sequence[dict[str, Any]], warnings: list[str]
) -> list[int] | None:
    """Per-arm repeat ordinals, cross-checked against the ABBA block schedule.

    The samples carry their own ``repeat`` ordinal.  That ordinal is the pairing
    slot: ``scripts``' mirrored-round schedule issues the same block sizes to
    both arms in each round, so stock repeat *k* and candidate repeat *k* are
    the two arms' observations of the same round.  The schedule is used to
    *verify* that -- if the blocks recorded for this arm do not account for
    exactly the samples present, the run is not the run the schedule describes
    and the ordinals are refused rather than trusted.
    """
    ordinals: list[int] = []
    for sample in samples:
        value = sample.get("repeat")
        if not isinstance(value, int) or isinstance(value, bool):
            warnings.append(f"arm {arm!r}: a sample has no integer 'repeat'; no pass indices")
            return None
        ordinals.append(value)

    schedule = payload.get("schedule")
    blocks = schedule.get("blocks") if isinstance(schedule, dict) else None
    if not isinstance(blocks, list):
        warnings.append(f"arm {arm!r}: no block schedule recorded; pass indices unverified")
        return None
    scheduled = 0
    for block in blocks:
        if not isinstance(block, dict) or block.get("arm") != arm:
            continue
        count = block.get("scored_repeats")
        if not isinstance(count, int) or isinstance(count, bool):
            warnings.append(f"arm {arm!r}: block has no integer scored_repeats; no pass indices")
            return None
        scheduled += count
    if scheduled != len(ordinals):
        warnings.append(
            f"arm {arm!r}: schedule scores {scheduled} repeats but {len(ordinals)} "
            "samples are present; pass indices refused"
        )
        return None
    if sorted(ordinals) != list(range(len(ordinals))):
        warnings.append(f"arm {arm!r}: repeat ordinals are not a permutation of 0..n-1")
        return None
    return ordinals


def _extract_config_matrix_cell(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["result.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "samples" not in payload or "cell" not in payload:
        out.unparsed[path] = "not a configuration-matrix cell result (no cell/samples)"
        return out

    cell_config = payload.get("cell")
    if not isinstance(cell_config, dict):
        out.unparsed[path] = "configuration-matrix cell result has a non-object 'cell'"
        return out
    name = cell_config.get("name")
    if not isinstance(name, str) or not name:
        out.unparsed[path] = "configuration-matrix cell result has no cell name"
        return out

    warnings: list[str] = []
    samples = payload.get("samples")
    arms: dict[str, dict[str, MetricSeries]] = {}
    if isinstance(samples, dict):
        for arm, arm_samples in samples.items():
            if not isinstance(arm_samples, list) or not arm_samples:
                warnings.append(f"arm {arm!r} has no samples")
                continue
            rows = [s for s in arm_samples if isinstance(s, dict)]
            if len(rows) != len(arm_samples):
                warnings.append(f"arm {arm!r}: {len(arm_samples) - len(rows)} non-object samples")
            indices = _config_matrix_pass_indices(payload, arm, rows, warnings)
            series = _collect(rows, _numeric_fields(rows), pass_indices=indices, warnings=warnings)
            if series:
                arms[arm] = series
    else:
        warnings.append("no per-repeat samples recorded")

    # The summary is a reduction of the samples above; it is recorded only for
    # arms whose individual observations did NOT survive, so that the index
    # never presents a mean where the raw list exists.
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_arms = summary.get("arms")
        if isinstance(summary_arms, dict):
            for arm, stats in summary_arms.items():
                if arm in arms or not isinstance(stats, dict):
                    continue
                reduced: dict[str, MetricSeries] = {}
                for metric, stat in stats.items():
                    if not isinstance(stat, dict) or "mean" not in stat:
                        continue
                    series = _series(
                        metric, [stat["mean"]], reduced=True, warnings=warnings
                    )
                    if series is not None:
                        reduced[metric] = series
                if reduced:
                    warnings.append(
                        f"arm {arm!r}: only the summary survived; series marked reduced"
                    )
                    arms[arm] = reduced

    config: dict[str, Any] = dict(cell_config)
    config["passed"] = payload.get("passed")
    if isinstance(summary, dict):
        fault = summary.get("fault_injection")
        if isinstance(fault, dict):
            config["fault_injection"] = fault
        loss = summary.get("paired_candidate_loss")
        if isinstance(loss, dict):
            # A paired statistic, not an arm measurement.  Kept as configuration
            # so it is queryable without being mistaken for a series.
            config["paired_candidate_loss"] = loss
    schedule = payload.get("schedule")
    if isinstance(schedule, dict):
        config["schedule_design"] = schedule.get("design")
        config["arm_blocks"] = schedule.get("arm_blocks")
        config["repeats_per_arm"] = schedule.get("repeats_per_arm")
        blocks = schedule.get("blocks")
        if isinstance(blocks, list) and blocks:
            first = blocks[0]
            if isinstance(first, dict) and isinstance(first.get("engine_regime"), dict):
                config["engine_regime"] = first["engine_regime"]
    failures = payload.get("regression_failures")
    if failures:
        warnings.append(f"cell recorded {len(failures)} regression failure(s)")

    out.cells.append(
        CellRecord(name=name, arms=arms, config=config, warnings=tuple(warnings))
    )
    out.consumed.add(path)
    return out


# ---------------------------------------------------------------------------
# Extractor: config matrix roll-up (results/matrix-result.json + manifest.json)
# ---------------------------------------------------------------------------


def _extract_config_matrix_rollup(directory: Path, files: dict[str, Path]) -> Outcome:
    """The roll-up.  Its cells duplicate the per-cell ``result.json`` files.

    Only the cells and failures are taken here; :func:`load_run` drops any cell
    whose name a per-cell extractor already produced, so the raw samples always
    win over the copy.
    """
    out = Outcome()
    path = files["matrix-result.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "cells" not in payload:
        out.unparsed[path] = "not a configuration-matrix roll-up (no 'cells')"
        return out
    out.consumed.add(path)

    cells = payload.get("cells")
    if isinstance(cells, list):
        for entry in cells:
            if not isinstance(entry, dict):
                continue
            sub = _extract_config_matrix_cell_payload(entry)
            if sub is not None:
                out.cells.append(sub)
    failures = payload.get("failures")
    if isinstance(failures, list):
        for failure in failures:
            out.unparsed[path.with_name(f"{path.name}#failure")] = (
                f"matrix recorded a cell failure: {json.dumps(failure)[:200]}"
            )
    return out


def _extract_config_matrix_cell_payload(payload: dict[str, Any]) -> CellRecord | None:
    """Reuse the per-cell extractor on an in-memory cell object."""
    fake = Outcome()
    cell_config = payload.get("cell")
    if not isinstance(cell_config, dict) or not isinstance(cell_config.get("name"), str):
        return None
    # Route through the same code path by writing nothing: the per-cell
    # extractor only needs the payload, so factor the body out via a shim.
    holder: dict[str, Any] = payload
    warnings: list[str] = []
    samples = holder.get("samples")
    arms: dict[str, dict[str, MetricSeries]] = {}
    if isinstance(samples, dict):
        for arm, arm_samples in samples.items():
            rows = [s for s in arm_samples or [] if isinstance(s, dict)]
            if not rows:
                continue
            indices = _config_matrix_pass_indices(holder, arm, rows, warnings)
            series = _collect(rows, _numeric_fields(rows), pass_indices=indices, warnings=warnings)
            if series:
                arms[arm] = series
    config = dict(cell_config)
    config["passed"] = holder.get("passed")
    del fake
    return CellRecord(
        name=cell_config["name"], arms=arms, config=config, warnings=tuple(warnings)
    )


def _extract_manifest(directory: Path, files: dict[str, Path]) -> Outcome:
    """The matrix manifest is run-level configuration, not a measurement."""
    out = Outcome()
    path = files["manifest.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "measurement" not in payload:
        if isinstance(payload, dict) and "artifact_id" in payload:
            # The tuner's own draft-artifact manifest, which happens to share
            # the basename.  Provenance for a trained draft head, not a run.
            out.unparsed[path] = "draft artifact manifest, not a measurement"
        else:
            out.unparsed[path] = "not a configuration-matrix manifest (no 'measurement')"
        return out
    out.consumed.add(path)
    # Carried as a zero-arm cell so the configuration is queryable and the file
    # is demonstrably read, without inventing observations it does not contain.
    out.cells.append(
        CellRecord(
            name="manifest",
            arms={},
            config=payload,
            warnings=("run manifest: configuration only, no observations",),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: capture overhead (results/<stamp>/capture_overhead.json)
# ---------------------------------------------------------------------------


def _capture_overhead_pass_indices(
    samples_by_arm: dict[str, list[dict[str, Any]]], warnings: list[str]
) -> dict[str, list[int]] | None:
    """Rank each observation by its ``(cycle, half, prompt)`` identity.

    That triple is what the bench holds fixed between the ON and OFF arms; two
    observations are comparable exactly when it matches.  If the arms do not
    present the same set of triples the pairing is refused.
    """
    keys: dict[str, list[tuple[Any, ...]]] = {}
    for arm, rows in samples_by_arm.items():
        arm_keys: list[tuple[Any, ...]] = []
        for row in rows:
            if not all(k in row for k in ("cycle", "half", "prompt")):
                warnings.append(f"arm {arm!r}: a sample lacks cycle/half/prompt; no pass indices")
                return None
            arm_keys.append((row["cycle"], row["half"], row["prompt"]))
        if len(set(arm_keys)) != len(arm_keys):
            warnings.append(f"arm {arm!r}: duplicate (cycle, half, prompt); no pass indices")
            return None
        keys[arm] = arm_keys
    distinct = {frozenset(v) for v in keys.values()}
    if len(distinct) != 1:
        warnings.append("arms do not share a (cycle, half, prompt) set; pass indices refused")
        return None
    ordering = {key: i for i, key in enumerate(sorted(next(iter(keys.values()))))}
    return {arm: [ordering[k] for k in arm_keys] for arm, arm_keys in keys.items()}


def _extract_capture_overhead(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["capture_overhead.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), dict):
        out.unparsed[path] = "not a capture-overhead result (no 'samples' object)"
        return out
    out.consumed.add(path)

    warnings: list[str] = []
    samples_by_arm = {
        arm: [r for r in rows or [] if isinstance(r, dict)]
        for arm, rows in payload["samples"].items()
    }
    samples_by_arm = {a: r for a, r in samples_by_arm.items() if r}
    indices = _capture_overhead_pass_indices(samples_by_arm, warnings) or {}
    arms: dict[str, dict[str, MetricSeries]] = {}
    for arm, rows in samples_by_arm.items():
        series = _collect(
            rows, _numeric_fields(rows), pass_indices=indices.get(arm), warnings=warnings
        )
        # ``cycle`` and ``half`` are the pass identity, not measurements.
        for identity in ("cycle", "half"):
            series.pop(identity, None)
        if series:
            arms[arm] = series

    config = {k: v for k, v in payload.items() if k not in ("samples", "summary")}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        config["summary"] = summary
    out.cells.append(
        CellRecord(
            name="capture-overhead", arms=arms, config=config, warnings=tuple(warnings)
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: proxy overhead
# ---------------------------------------------------------------------------


def _proxy_pass_indices(
    rows_by_arm: dict[str, list[dict[str, Any]]], key: str, warnings: list[str]
) -> dict[str, list[int]] | None:
    values: dict[str, list[Any]] = {}
    for arm, rows in rows_by_arm.items():
        if not all(key in r for r in rows):
            warnings.append(f"arm {arm!r}: a sample lacks {key!r}; no pass indices")
            return None
        arm_values = [r[key] for r in rows]
        if len(set(arm_values)) != len(arm_values):
            warnings.append(f"arm {arm!r}: duplicate {key!r}; no pass indices")
            return None
        values[arm] = arm_values
    if len({frozenset(v) for v in values.values()}) != 1:
        warnings.append(f"arms do not share a {key!r} set; pass indices refused")
        return None
    ordering = {v: i for i, v in enumerate(sorted(next(iter(values.values()))))}
    return {arm: [ordering[v] for v in arm_values] for arm, arm_values in values.items()}


def _proxy_cell(
    name: str, rows_by_arm: dict[str, list[dict[str, Any]]], pair_key: str, config: dict[str, Any]
) -> CellRecord | None:
    warnings: list[str] = []
    rows_by_arm = {a: r for a, r in rows_by_arm.items() if r}
    if not rows_by_arm:
        return None
    indices = _proxy_pass_indices(rows_by_arm, pair_key, warnings) or {}
    arms: dict[str, dict[str, MetricSeries]] = {}
    for arm, rows in rows_by_arm.items():
        series = _collect(
            rows, _numeric_fields(rows), pass_indices=indices.get(arm), warnings=warnings
        )
        for identity in ("seed", "repeat", "concurrency", "seeds"):
            series.pop(identity, None)
        if series:
            arms[arm] = series
    if not arms:
        return None
    return CellRecord(name=name, arms=arms, config=config, warnings=tuple(warnings))


def _extract_proxy_overhead(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["proxy_overhead.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "metadata" not in payload:
        out.unparsed[path] = "not a proxy-overhead result (no 'metadata')"
        return out
    out.consumed.add(path)

    raw = payload.get("raw_samples")
    raw_path = files.get("proxy_overhead_raw_samples.json")
    if raw_path is not None:
        try:
            raw = _load_json(raw_path)
        except (OSError, ValueError) as exc:
            out.unparsed[raw_path] = f"unreadable JSON: {exc}"
        else:
            out.consumed.add(raw_path)
    if not isinstance(raw, dict):
        out.unparsed[path] = "proxy-overhead result carries no per-request raw samples"
        return out

    metadata = payload.get("metadata")
    base = dict(metadata) if isinstance(metadata, dict) else {}

    for kind in ("nonstream", "stream"):
        rows = {
            arm: [r for r in (raw.get(arm) or {}).get(kind, []) or [] if isinstance(r, dict)]
            for arm in raw
        }
        cell = _proxy_cell(f"proxy-overhead-{kind}", rows, "seed", {**base, "path_kind": kind})
        if cell is not None:
            out.cells.append(cell)

    levels: set[str] = set()
    for arm in raw:
        concurrent = (raw.get(arm) or {}).get("concurrent")
        if isinstance(concurrent, dict):
            levels.update(concurrent)
    for level in sorted(levels, key=lambda s: (len(s), s)):
        rows = {
            arm: [
                r
                for r in ((raw.get(arm) or {}).get("concurrent") or {}).get(level, []) or []
                if isinstance(r, dict)
            ]
            for arm in raw
        }
        cell = _proxy_cell(
            f"proxy-overhead-concurrent-c{level}",
            rows,
            "repeat",
            {**base, "path_kind": "concurrent", "request_concurrency": level},
        )
        if cell is not None:
            out.cells.append(cell)
    return out


# ---------------------------------------------------------------------------
# Extractor: idle-tuning gate decision
# ---------------------------------------------------------------------------

#: ``terminal-decision.json`` names its per-repeat fields ``<arm>_<metric>``.
_DECISION_ARM_PREFIXES: Final[tuple[str, ...]] = ("stock", "candidate")
#: Per-repeat fields with no arm prefix belong to the comparison itself.
_DECISION_SHARED_FIELDS: Final[tuple[str, ...]] = ("invalid_rate", "output_mismatches")


def _extract_idle_tuning_decision(directory: Path, files: dict[str, Path]) -> Outcome:
    return _extract_decision(directory, files["terminal-decision.json"], "idle-tuning")


def _extract_cycle_decision(directory: Path, files: dict[str, Path]) -> Outcome:
    """A single tuning cycle's gate decision, under ``speedlm_home/runs/<id>/``.

    Same shape as the terminal decision, one per cycle.  Multi-cycle runs kept
    *only* these -- indexing the terminal decision alone would report the last
    cycle of a run and drop every earlier one.
    """
    return _extract_decision(
        directory, files["decision.json"], f"idle-tuning-cycle-{directory.name[:8]}"
    )


def _extract_decision(directory: Path, path: Path, cell_name: str) -> Outcome:
    out = Outcome()
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "verdict" not in payload:
        out.unparsed[path] = "not an idle-tuning gate decision (no 'verdict')"
        return out
    out.consumed.add(path)

    warnings: list[str] = []
    per_repeat = payload.get("per_repeat")
    arms: dict[str, dict[str, MetricSeries]] = {}
    if isinstance(per_repeat, list) and per_repeat:
        rows = [r for r in per_repeat if isinstance(r, dict)]
        indices: list[int] | None = []
        for row in rows:
            value = row.get("repeat_index")
            if not isinstance(value, int) or isinstance(value, bool):
                warnings.append("a per-repeat record has no integer repeat_index")
                indices = None
                break
            indices.append(value)
        for arm in _DECISION_ARM_PREFIXES:
            prefix = f"{arm}_"
            renamed = [
                {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}
                for row in rows
            ]
            series = _collect(
                renamed, _numeric_fields(renamed), pass_indices=indices, warnings=warnings
            )
            if series:
                arms[arm] = series
        shared = [
            {k: v for k, v in row.items() if k in _DECISION_SHARED_FIELDS} for row in rows
        ]
        shared_series = _collect(
            shared, _DECISION_SHARED_FIELDS, pass_indices=indices, warnings=warnings
        )
        if shared_series:
            arms["comparison"] = shared_series
    else:
        warnings.append("gate decision kept no per-repeat record")
        for arm in _DECISION_ARM_PREFIXES:
            reduced: dict[str, MetricSeries] = {}
            for suffix, metric in (
                ("avg_tok_per_sec", "tok_per_sec"),
                ("avg_accepted_length", "accepted_length"),
                ("avg_acceptance", "acceptance_rate"),
            ):
                value = payload.get(f"{arm}_{suffix}")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    series = _series(metric, [value], reduced=True, warnings=warnings)
                    if series is not None:
                        reduced[metric] = series
            if reduced:
                arms[arm] = reduced

    config = {
        k: v
        for k, v in payload.items()
        if k not in ("per_repeat", "output_divergences", "control_divergences")
    }
    out.cells.append(
        CellRecord(name=cell_name, arms=arms, config=config, warnings=tuple(warnings))
    )
    return out


def _extract_preemption(directory: Path, files: dict[str, Path]) -> Outcome:
    """How long a request sat queued while the tuner held the GPU."""
    out = Outcome()
    path = files["preemption-observation.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "queued_request_seconds" not in payload:
        out.unparsed[path] = "not a preemption observation (no 'queued_request_seconds')"
        return out
    out.consumed.add(path)
    warnings: list[str] = []
    series = _collect(
        [payload], ("queued_request_seconds",), pass_indices=None, warnings=warnings
    )
    if not series:
        return out
    out.cells.append(
        CellRecord(
            name="idle-tuning-preemption",
            arms={"single": series},
            config={"state_when_submitted": payload.get("state_when_submitted")},
            warnings=tuple(warnings),
        )
    )
    return out


_VAL_METRIC_RE: Final[re.Pattern[str]] = re.compile(
    r"^(loss|full_acc|cond_acc)_(\d+)_epoch$"
)
_VAL_METRIC_NAMES: Final[dict[str, str]] = {
    "loss": "val_loss",
    "full_acc": "val_full_accuracy",
    "cond_acc": "val_cond_accuracy",
}


def _extract_val_metrics(directory: Path, files: dict[str, Path]) -> Outcome:
    """Draft-head validation, one arm per speculative position.

    Position is *not* a pass index -- it is a different axis entirely -- so it
    becomes the arm name.  Recording it as a pass index would let the
    comparator pair position 0 of one run against position 0 of another and
    call that a repeat.
    """
    out = Outcome()
    path = files["val_metrics.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "loss_epoch" not in payload:
        out.unparsed[path] = "not a draft-head validation record (no 'loss_epoch')"
        return out
    out.consumed.add(path)

    warnings: list[str] = []
    by_position: dict[str, dict[str, float]] = {}
    for key, value in payload.items():
        match = _VAL_METRIC_RE.match(key)
        if match and isinstance(value, (int, float)) and not isinstance(value, bool):
            by_position.setdefault(f"position-{match.group(2)}", {})[
                _VAL_METRIC_NAMES[match.group(1)]
            ] = float(value)
    arms: dict[str, dict[str, MetricSeries]] = {}
    for arm, values in sorted(by_position.items()):
        series = _collect([values], values, pass_indices=None, warnings=warnings)
        if series:
            arms[arm] = series
    epoch = payload.get("loss_epoch")
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        series = _collect(
            [{"val_loss": float(epoch)}], ("val_loss",), pass_indices=None, warnings=warnings
        )
        if series:
            arms["epoch"] = series
    if not arms:
        return out
    out.cells.append(
        CellRecord(
            name="draft-head-validation", arms=arms, config={}, warnings=tuple(warnings)
        )
    )
    return out


_ARM_PROM_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<arm>[a-z]+)-after\.prom\.gz$")


def _extract_gate_window_prom(directory: Path, files: dict[str, Path]) -> Outcome:
    """Whole-arm Prometheus windows, for gates that kept no per-repeat scrape.

    One observation per arm covering the arm's entire measurement window.  It
    is a single measurement, not a reduction of several, so it is *not* marked
    ``reduced`` -- but with n=1 the comparator cannot resolve a standard error
    from it, which is exactly what it should report.
    """
    out = Outcome()
    warnings: list[str] = []
    arms: dict[str, dict[str, MetricSeries]] = {}
    for name, after_path in sorted(files.items()):
        match = _ARM_PROM_RE.match(name)
        if not match:
            continue
        arm = match["arm"]
        # The baseline scrape is read but NOT consumed: the per-repeat extractor
        # needs the same file to bracket its first repeat.
        before_path = after_path.with_name(f"{arm}-before.prom.gz")
        if not before_path.is_file():
            out.unparsed[after_path] = f"arm {arm!r}: closing scrape with no baseline scrape"
            continue
        try:
            delta = compute_delta(_read_prom(before_path), _read_prom(after_path))
        except (OSError, ValueError, CounterResetError) as exc:
            out.unparsed[after_path] = f"arm {arm!r}: {exc}"
            continue
        rows = [{n: getattr(delta, n) for n in _PROM_DELTA_FIELDS}]
        series = _collect(rows, _PROM_DELTA_FIELDS, pass_indices=None, warnings=warnings)
        if series:
            arms[arm] = series
        # This extractor runs after the per-repeat one, so the baseline scrape
        # has already been offered where it was usable; claiming it here keeps
        # it from being reported as an artifact nothing read.
        out.consumed.update({after_path, before_path})
    if not arms:
        return out
    out.cells.append(
        CellRecord(
            name="idle-tuning-prometheus-window",
            arms=arms,
            config={"source": "whole-arm Prometheus scrape deltas"},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: gate per-repeat Prometheus scrapes
# ---------------------------------------------------------------------------

_REPEAT_PROM_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<arm>[a-z]+)-after-repeat-(?P<repeat>\d+)\.prom\.gz$"
)


def _read_prom(path: Path) -> MetricsSnapshot:
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return parse_metrics(text)


_PROM_DELTA_FIELDS: Final[tuple[str, ...]] = (
    "acceptance_rate",
    "mean_accepted_length",
    "tpot_ms",
    "output_tok_per_sec",
    "drafted_tokens",
    "accepted_tokens",
)


def _extract_gate_metrics(directory: Path, files: dict[str, Path]) -> Outcome:
    """Recover per-repeat throughput/acceptance from the gate's raw scrapes.

    The gate writes one cumulative scrape before an arm starts and one after
    each scored repeat.  Consecutive scrapes therefore bracket exactly one
    repeat, so differencing them recovers the individual observations rather
    than the whole-arm mean the ``*-after.prom.gz`` file would give.
    """
    out = Outcome()
    warnings: list[str] = []
    by_arm: dict[str, dict[int, Path]] = {}
    for name, path in files.items():
        match = _REPEAT_PROM_RE.match(name)
        if match:
            by_arm.setdefault(match["arm"], {})[int(match["repeat"])] = path

    arms: dict[str, dict[str, MetricSeries]] = {}
    for arm, repeats in sorted(by_arm.items()):
        before = files.get(f"{arm}-before.prom.gz")
        ordered = sorted(repeats)

        def refuse(reason: str, paths: dict[int, Path] = repeats) -> None:
            for path in paths.values():
                out.unparsed[path] = reason

        if before is None:
            refuse(f"arm {arm!r}: no baseline scrape; per-repeat deltas refused")
            continue
        if ordered != list(range(len(ordered))):
            refuse(f"arm {arm!r}: repeat scrapes are not contiguous from 0")
            continue
        try:
            snapshots = [_read_prom(before)] + [_read_prom(repeats[i]) for i in ordered]
        except (OSError, ValueError) as exc:
            refuse(f"arm {arm!r}: unreadable scrape: {exc}")
            continue
        rows: list[dict[str, Any]] = []
        reset: str | None = None
        for index in range(len(ordered)):
            try:
                delta = compute_delta(snapshots[index], snapshots[index + 1])
            except CounterResetError as exc:
                reset = f"arm {arm!r} repeat {index}: {exc}"
                rows = []
                break
            rows.append({name: getattr(delta, name) for name in _PROM_DELTA_FIELDS})
        if not rows:
            refuse(reset or f"arm {arm!r}: no per-repeat deltas recovered")
            continue
        series = _collect(
            rows, _PROM_DELTA_FIELDS, pass_indices=list(range(len(rows))), warnings=warnings
        )
        if series:
            arms[arm] = series
        out.consumed.add(before)
        out.consumed.update(repeats[i] for i in ordered)

    if not arms:
        return out
    out.cells.append(
        CellRecord(
            name="idle-tuning-prometheus",
            arms=arms,
            config={"source": "per-repeat Prometheus scrape deltas"},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: Prometheus before/after pair
# ---------------------------------------------------------------------------


_BLOCK_DIR_RE: Final[re.Pattern[str]] = re.compile(r"^block-(?P<index>\d+)-(?P<arm>[a-z]+)$")
_BLOCK_REPEAT_PROM_RE: Final[re.Pattern[str]] = re.compile(
    r"^metrics-after-repeat-(?P<repeat>\d+)\.prom$"
)


def _extract_block_prom(directory: Path, files: dict[str, Path]) -> Outcome:
    """Per-repeat scrapes inside one configuration-matrix block directory.

    Blocks live at ``results/<cell>/block-NN-<arm>/`` and each holds one
    baseline scrape plus one scrape per scored repeat.  The cell is named for
    the block rather than the arm because a block's repeat ordinals are local
    to it: ``block-00-stock`` repeat 0 and ``block-03-stock`` repeat 0 are
    different passes, and merging them here would invent a pairing the
    directory layout does not support.
    """
    out = Outcome()
    match = _BLOCK_DIR_RE.match(directory.name)
    if match is None:
        return out
    before = files.get("metrics-before.prom")
    repeats: dict[int, Path] = {}
    for name, path in files.items():
        found = _BLOCK_REPEAT_PROM_RE.match(name)
        if found:
            repeats[int(found["repeat"])] = path
    if before is None or not repeats:
        return out
    ordered = sorted(repeats)
    warnings: list[str] = []
    if ordered != list(range(len(ordered))):
        for path in repeats.values():
            out.unparsed[path] = "repeat scrapes are not contiguous from 0"
        return out
    try:
        snapshots = [_read_prom(before)] + [_read_prom(repeats[i]) for i in ordered]
    except (OSError, ValueError) as exc:
        for path in repeats.values():
            out.unparsed[path] = f"unreadable scrape: {exc}"
        return out
    rows: list[dict[str, Any]] = []
    for index in range(len(ordered)):
        try:
            delta = compute_delta(snapshots[index], snapshots[index + 1])
        except CounterResetError as exc:
            for path in repeats.values():
                out.unparsed[path] = f"repeat {index}: {exc}"
            return out
        rows.append({name: getattr(delta, name) for name in _PROM_DELTA_FIELDS})
    series = _collect(
        rows, _PROM_DELTA_FIELDS, pass_indices=list(range(len(rows))), warnings=warnings
    )
    if not series:
        return out
    out.consumed.add(before)
    out.consumed.update(repeats[i] for i in ordered)
    out.cells.append(
        CellRecord(
            name=f"{directory.parent.name}-{directory.name}-prometheus",
            arms={match["arm"]: series},
            config={
                "block_index": int(match["index"]),
                "source": "per-repeat Prometheus scrape deltas",
            },
            warnings=tuple(warnings),
        )
    )
    return out


def _extract_prom_pair(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    before_path = files["metrics-before.prom"]
    after_path = files["metrics-after.prom"]
    warnings: list[str] = []
    try:
        before = _read_prom(before_path)
        after = _read_prom(after_path)
    except (OSError, ValueError) as exc:
        out.unparsed[before_path] = f"unreadable Prometheus exposition: {exc}"
        return out
    out.consumed.update({before_path, after_path})
    try:
        delta = compute_delta(before, after)
    except CounterResetError as exc:
        out.unparsed[after_path] = f"counter reset between scrapes: {exc}"
        return out
    if not delta.acceptance_available:
        warnings.append("no speculative-decoding counters in the window")
    rows = [{name: getattr(delta, name) for name in _PROM_DELTA_FIELDS}]
    series = _collect(rows, _PROM_DELTA_FIELDS, pass_indices=None, warnings=warnings)
    if not series:
        return out
    out.cells.append(
        CellRecord(
            name=f"{directory.name}-prometheus",
            arms={"single": series},
            config={"source": "metrics-before.prom / metrics-after.prom"},
            warnings=tuple(warnings),
        )
    )
    return out


def _extract_acceptance_metrics(directory: Path, files: dict[str, Path]) -> Outcome:
    """``acceptance-metrics.json`` holds two counter maps, not exposition text.

    They are rendered back into exposition lines and handed to the same
    :func:`~speedlm.gate.metrics.parse_metrics` the ``.prom`` files use, rather
    than growing a second interpretation of the same counter names.
    """
    out = Outcome()
    path = files["acceptance-metrics.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or not isinstance(payload.get("before"), dict):
        out.unparsed[path] = "not an acceptance-metrics record (no 'before' object)"
        return out
    out.consumed.add(path)

    def render(counters: dict[str, Any]) -> str:
        return "\n".join(
            f"{name} {value}"
            for name, value in counters.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )

    warnings: list[str] = []
    try:
        delta = compute_delta(
            parse_metrics(render(payload["before"])), parse_metrics(render(payload["after"]))
        )
    except (CounterResetError, KeyError, TypeError) as exc:
        out.unparsed[path] = f"counters do not form a usable pair: {exc}"
        return out
    if not delta.acceptance_available:
        warnings.append("no speculative-decoding counters in the window")
    rows = [{name: getattr(delta, name) for name in _PROM_DELTA_FIELDS}]
    series = _collect(rows, _PROM_DELTA_FIELDS, pass_indices=None, warnings=warnings)
    if not series:
        return out
    out.cells.append(
        CellRecord(
            name=f"{directory.name}-acceptance",
            arms={"single": series},
            config={"source": "acceptance-metrics.json", "matching_counter_count": payload.get(
                "matching_counter_count"
            )},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: model matrix verification
# ---------------------------------------------------------------------------


def _extract_verification(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["verification.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or not isinstance(payload.get("checks"), dict):
        out.unparsed[path] = "not a model-matrix verification (no 'checks' object)"
        return out
    out.consumed.add(path)

    warnings: list[str] = []
    checks = payload["checks"]
    passed = sum(1 for c in checks.values() if isinstance(c, dict) and c.get("passed") is True)
    failed = len(checks) - passed
    rows = [{"checks_passed": passed, "checks_failed": failed}]
    series = _collect(
        rows, ("checks_passed", "checks_failed"), pass_indices=None, warnings=warnings
    )
    name = payload.get("cell") if isinstance(payload.get("cell"), str) else directory.name
    config = {
        "expected_profile": payload.get("expected_profile"),
        "passed": payload.get("passed"),
        "ready": payload.get("ready"),
        "failures": payload.get("failures"),
    }
    out.cells.append(
        CellRecord(name=str(name), arms={"single": series}, config=config, warnings=tuple(warnings))
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: activation capture
# ---------------------------------------------------------------------------


def _extract_activation_capture(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["result.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        out.unparsed[path] = "not an activation-capture result (no 'cases' list)"
        return out
    out.consumed.add(path)

    warnings: list[str] = []
    cases = [c for c in payload["cases"] if isinstance(c, dict)]
    passed = 0
    for case in cases:
        result = case.get("result")
        if isinstance(result, dict) and result.get("verdict") == "PASS":
            passed += 1
    rows = [{"cases_passed": passed, "cases_failed": len(cases) - passed}]
    series = _collect(rows, ("cases_passed", "cases_failed"), pass_indices=None, warnings=warnings)
    out.cells.append(
        CellRecord(
            name="activation-capture",
            arms={"single": series},
            config={
                "model": payload.get("model"),
                "runner": payload.get("runner"),
                "verdict": payload.get("verdict"),
                "case_labels": [c.get("label") for c in cases],
            },
            warnings=tuple(warnings),
        )
    )
    return out


def _extract_prefix_cache(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["prefix_cache_result.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "cache_hit" not in payload:
        out.unparsed[path] = "not a prefix-cache result (no 'cache_hit')"
        return out
    out.consumed.add(path)
    warnings: list[str] = []
    series = _collect([payload], _numeric_fields([payload]), pass_indices=None, warnings=warnings)
    out.cells.append(
        CellRecord(
            name="prefix-cache",
            arms={"single": series} if series else {},
            config={k: v for k, v in payload.items() if k not in series},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: hot swap
# ---------------------------------------------------------------------------


def _extract_hot_swap(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["result.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "injected_rpc_calls" not in payload:
        out.unparsed[path] = "not a draft-hot-swap result (no 'injected_rpc_calls')"
        return out
    out.consumed.add(path)
    warnings: list[str] = []
    calls = payload.get("injected_rpc_calls")
    rows = [{"injected_rpc_calls": len(calls) if isinstance(calls, list) else 0}]
    series = _collect(rows, ("injected_rpc_calls",), pass_indices=None, warnings=warnings)
    out.cells.append(
        CellRecord(
            name="draft-hot-swap",
            arms={"single": series},
            config={k: v for k, v in payload.items() if k != "injected_rpc_calls"},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: live-vllm stream summary
# ---------------------------------------------------------------------------


def _extract_stream_summary(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["stream-summary.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, dict) or "first_chunk_seconds" not in payload:
        out.unparsed[path] = "not a stream summary (no 'first_chunk_seconds')"
        return out
    out.consumed.add(path)
    warnings: list[str] = []
    row = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    usage = payload.get("usage")
    if isinstance(usage, dict):
        row.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
    # One streamed request: an individual observation with n=1, not a reduction.
    series = _collect([row], _numeric_fields([row]), pass_indices=None, warnings=warnings)
    if not series:
        return out
    out.cells.append(
        CellRecord(
            name=f"{directory.name}-stream",
            arms={"single": series},
            config={"model": payload.get("model"), "id": payload.get("id")},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Extractor: capture matrix exchange manifests
# ---------------------------------------------------------------------------


def _extract_capture_manifests(directory: Path, files: dict[str, Path]) -> Outcome:
    out = Outcome()
    path = files["manifests.json"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        out.unparsed[path] = f"unreadable JSON: {exc}"
        return out
    if not isinstance(payload, list):
        out.unparsed[path] = "not a capture-matrix manifest list"
        return out
    out.consumed.add(path)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        started, completed = entry.get("started_at"), entry.get("completed_at")
        if isinstance(started, (int, float)) and isinstance(completed, (int, float)):
            rows.append({"exchange_seconds": float(completed) - float(started)})
    if not rows:
        out.unparsed[path] = "manifest list carries no timed exchanges"
        return out
    series = _collect(
        rows, ("exchange_seconds",), pass_indices=list(range(len(rows))), warnings=warnings
    )
    count = _series("exchange_count", [len(rows)], warnings=warnings)
    if count is not None:
        series["exchange_count"] = count
    out.cells.append(
        CellRecord(
            name=f"{directory.name}-exchanges",
            arms={"single": series},
            config={"source": "manifests.json"},
            warnings=tuple(warnings),
        )
    )
    return out


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

#: Priority-ordered.  ``result.json`` is claimed by three different shapes, so
#: the discriminating extractors (which reject on a missing key) come first.
EXTRACTORS: Final[tuple[Extractor, ...]] = (
    Extractor("config-matrix-cell", ("result.json",), _extract_config_matrix_cell),
    Extractor("activation-capture", ("result.json",), _extract_activation_capture),
    Extractor("draft-hot-swap", ("result.json",), _extract_hot_swap),
    Extractor("config-matrix-rollup", ("matrix-result.json",), _extract_config_matrix_rollup),
    Extractor("config-matrix-manifest", ("manifest.json",), _extract_manifest),
    Extractor("capture-overhead", ("capture_overhead.json",), _extract_capture_overhead),
    Extractor(
        "proxy-overhead",
        ("proxy_overhead.json",),
        _extract_proxy_overhead,
        optional=("proxy_overhead_raw_samples.json",),
    ),
    Extractor("idle-tuning-decision", ("terminal-decision.json",), _extract_idle_tuning_decision),
    Extractor("idle-tuning-cycle-decision", ("decision.json",), _extract_cycle_decision),
    Extractor("idle-tuning-preemption", ("preemption-observation.json",), _extract_preemption),
    Extractor("draft-head-validation", ("val_metrics.json",), _extract_val_metrics),
    Extractor(
        "gate-metrics-per-repeat", (), _extract_gate_metrics, glob="*-after-repeat-*.prom.gz"
    ),
    Extractor("gate-metrics-window", (), _extract_gate_window_prom, glob="*-after.prom.gz"),
    Extractor(
        "block-prometheus-per-repeat",
        ("metrics-before.prom",),
        _extract_block_prom,
        glob="metrics-after-repeat-*.prom",
    ),
    Extractor(
        "prometheus-pair",
        ("metrics-before.prom", "metrics-after.prom"),
        _extract_prom_pair,
    ),
    Extractor("acceptance-metrics", ("acceptance-metrics.json",), _extract_acceptance_metrics),
    Extractor("model-matrix-verification", ("verification.json",), _extract_verification),
    Extractor("prefix-cache", ("prefix_cache_result.json",), _extract_prefix_cache),
    Extractor("stream-summary", ("stream-summary.json",), _extract_stream_summary),
    Extractor("capture-matrix-manifests", ("manifests.json",), _extract_capture_manifests),
)


# ---------------------------------------------------------------------------
# Provenance recovery
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _recover_provenance(run_dir: Path) -> tuple[str | None, str | None, str | None, str]:
    """Return ``(commit, created_at, slurm_job_id, flavor)``.

    The commit is taken from ``snapshot-provenance.txt`` first and from the
    generated sbatch's ``commit=<sha>`` assignment second -- both are written
    when the immutable snapshot is materialised, so both pin the code that ran.
    Hand-written sbatch files instead *echo* ``git rev-parse HEAD`` at job
    start, which ``scripts/make_snapshot_run.sh`` documents as a label rather
    than a guarantee; those runs get ``commit=None``.
    """
    commit: str | None = None
    created_at: str | None = None

    provenance = _read_text(run_dir / "snapshot-provenance.txt")
    if provenance:
        match = re.search(r"^snapshot_commit=([0-9a-f]{40})\s*$", provenance, re.M)
        if match:
            commit = match.group(1)
        stamp = re.search(r"^generated_at=(\S+)\s*$", provenance, re.M)
        if stamp:
            created_at = stamp.group(1)

    flavor = "unknown"
    sbatch = _read_text(run_dir / "job.sbatch")
    if sbatch:
        job_name = _JOB_NAME_RE.search(sbatch)
        if job_name:
            flavor = _JOB_SUFFIX_TO_FLAVOR.get(job_name.group(1), f"unknown:{job_name.group(1)}")
        else:
            test = _TEST_PATH_RE.search(sbatch)
            if test:
                flavor = _TEST_TO_FLAVOR.get(test.group(1), f"unknown:{test.group(1)}")
        if commit is None and _SBATCH_SNAPSHOT_RE.search(sbatch):
            match = _SBATCH_COMMIT_RE.search(sbatch)
            if match:
                commit = match.group(1)
        if created_at is None:
            generated = _SBATCH_GENERATED_RE.search(sbatch)
            if generated:
                created_at = generated.group(1)

    slurm_job_id: str | None = None
    outs = sorted(run_dir.glob("slurm-*.out"))
    for candidate in outs:
        match = _SLURM_OUT_RE.match(candidate.name)
        if match:
            slurm_job_id = match.group(1)
            break
    if created_at is None and outs:
        text = _read_text(outs[0])
        if text:
            started = _SLURM_STARTED_RE.search(text)
            if started:
                created_at = started.group(1)
    if created_at is None:
        stamp = _DIRNAME_STAMP_RE.search(run_dir.name)
        if stamp:
            raw = stamp.group(1)
            created_at = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"
    return commit, created_at, slurm_job_id, flavor


def _recover_workload(run_dir: Path) -> str | None:
    """The workload name, if the run declared one.

    Every run in the historical tree predates the workload system, so this is
    ``None`` for all of them; it reads the manifest rather than hard-coding
    ``None`` so a future run that does declare one is picked up.
    """
    for candidate in (
        run_dir / "results" / "manifest.json",
        run_dir / "results" / "workload.json",
    ):
        if not candidate.is_file():
            continue
        try:
            payload = _load_json(candidate)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            name = payload.get("workload")
            if isinstance(name, str) and name:
                return name
    return None


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------


def _is_non_measurement(name: str) -> str | None:
    for pattern, reason in NON_MEASUREMENT_PATTERNS.items():
        if fnmatch.fnmatch(name, pattern):
            return reason
    return None


def _walk_results(results: Path) -> Iterator[tuple[Path, dict[str, Path], dict[str, str]]]:
    """Yield ``(directory, measurement_files, skipped)`` for every subtree.

    ``skipped`` maps a glob-shaped label to the reason its matches were not
    itemised, so a run reports the payload it carried without listing every
    seed response.
    """
    stack = [results]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        files: dict[str, Path] = {}
        skipped: dict[str, str] = {}
        for entry in entries:
            if entry.is_dir():
                if entry.name in NON_MEASUREMENT_DIRS:
                    skipped[f"{entry.name}/"] = "capture payload subtree"
                    continue
                stack.append(entry)
                continue
            reason = _is_non_measurement(entry.name)
            if reason is not None:
                skipped.setdefault(entry.name, reason)
                continue
            files[entry.name] = entry
        yield directory, files, skipped


def _condense(skipped: dict[str, str]) -> dict[str, str]:
    """Fold ``seed-response-0001.json`` and friends into one line per pattern."""
    groups: dict[tuple[str, str], int] = {}
    for name, reason in skipped.items():
        label = re.sub(r"\d+", "N", name)
        groups[(label, reason)] = groups.get((label, reason), 0) + 1
    return {
        label: (f"{reason} ({count} files)" if count > 1 else reason)
        for (label, reason), count in groups.items()
    }


# ---------------------------------------------------------------------------
# Loading one run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileStamp:
    """Size and mtime of one file the index read, for staleness detection."""

    path: str
    size: int
    mtime_ns: int


def load_run(run_dir: Path) -> RunRecord:
    """Parse one run directory into a :class:`~model.RunRecord`."""
    record, _ = _load_run_stamped(run_dir)
    return record


def _load_run_stamped(run_dir: Path) -> tuple[RunRecord, tuple[FileStamp, ...]]:
    commit, created_at, slurm_job_id, flavor = _recover_provenance(run_dir)
    cells: dict[str, CellRecord] = {}
    unparsed: dict[str, str] = {}
    read: list[Path] = []

    for name in ("snapshot-provenance.txt", "job.sbatch"):
        candidate = run_dir / name
        if candidate.is_file():
            read.append(candidate)

    results = run_dir / "results"
    if not results.is_dir():
        unparsed["results/"] = "run produced no results directory"
        return (
            RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir,
                flavor=flavor,
                commit=commit,
                created_at=created_at,
                slurm_job_id=slurm_job_id,
                workload=None,
                cells={},
                unparsed=unparsed,
            ),
            tuple(_stamp(run_dir, p) for p in read),
        )

    for directory, files, skipped in _walk_results(results):
        for label, reason in _condense(skipped).items():
            key = str((directory / label).relative_to(run_dir))
            unparsed.setdefault(key, reason)
        if not files:
            continue
        consumed: set[Path] = set()
        # Why an extractor declined a file, kept aside until the whole table has
        # run: three shapes share the basename ``result.json``, so "this is not
        # a configuration-matrix cell" is only a finding if nothing else claimed
        # the file afterwards.
        declined: dict[Path, str] = {}
        for extractor in EXTRACTORS:
            offer = {n: p for n, p in files.items() if p not in consumed}
            if not extractor.claims(offer):
                continue
            outcome = extractor.extract(directory, offer)
            if not outcome.cells and not outcome.consumed and not outcome.unparsed:
                continue
            if not outcome.cells and outcome.unparsed and not outcome.consumed:
                # The shape did not match; leave the files for the next
                # extractor and remember why this one declined.
                for path, reason in outcome.unparsed.items():
                    declined.setdefault(path, f"[{extractor.name}] {reason}")
                continue
            for cell in outcome.cells:
                # Raw per-repeat cells win over the roll-up's copy of them.
                existing = cells.get(cell.name)
                if existing is not None and _observation_count(existing) >= _observation_count(
                    cell
                ):
                    continue
                cells[cell.name] = cell
            consumed |= outcome.consumed
            read.extend(outcome.consumed)
            for path, reason in outcome.unparsed.items():
                unparsed[str(path.relative_to(run_dir))] = f"[{extractor.name}] {reason}"
        for name, path in files.items():
            if path in consumed:
                continue
            key = str(path.relative_to(run_dir))
            reason = declined.get(path) or CONFIGURATION_ARTIFACTS.get(name)
            unparsed.setdefault(key, reason or f"no registered extractor claims {name!r}")

    return (
        RunRecord(
            run_id=run_dir.name,
            run_dir=run_dir,
            flavor=flavor,
            commit=commit,
            created_at=created_at,
            slurm_job_id=slurm_job_id,
            workload=_recover_workload(run_dir),
            cells=cells,
            unparsed=unparsed,
        ),
        tuple(_stamp(run_dir, p) for p in read),
    )


def _observation_count(cell: CellRecord) -> int:
    return sum(len(s.values) for arm in cell.arms.values() for s in arm.values())


def _stamp(run_dir: Path, path: Path) -> FileStamp:
    try:
        stat = path.stat()
    except OSError:
        return FileStamp(path=str(path.relative_to(run_dir)), size=-1, mtime_ns=-1)
    return FileStamp(
        path=str(path.relative_to(run_dir)), size=stat.st_size, mtime_ns=stat.st_mtime_ns
    )


def index_runs(root: Path) -> list[RunRecord]:
    """Parse every run directory under *root*, in name order."""
    return [record for record, _ in _index_runs_stamped(root)]


def _index_runs_stamped(root: Path) -> list[tuple[RunRecord, tuple[FileStamp, ...]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    return [
        _load_run_stamped(entry)
        for entry in sorted(root.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]


# ---------------------------------------------------------------------------
# The persisted index
# ---------------------------------------------------------------------------

INDEX_SCHEMA_VERSION: Final[int] = 1


def _series_to_json(series: MetricSeries) -> dict[str, Any]:
    return {
        "name": series.name,
        "unit": series.unit,
        "direction": series.direction.value,
        "values": list(series.values),
        "pass_indices": list(series.pass_indices) if series.pass_indices is not None else None,
        "reduced": series.reduced,
    }


def _series_from_json(payload: dict[str, Any]) -> MetricSeries:
    indices = payload.get("pass_indices")
    return MetricSeries(
        name=payload["name"],
        unit=payload["unit"],
        direction=Direction(payload["direction"]),
        values=tuple(float(v) for v in payload["values"]),
        pass_indices=tuple(int(i) for i in indices) if indices is not None else None,
        reduced=bool(payload.get("reduced", False)),
    )


def _record_to_json(record: RunRecord, stamps: tuple[FileStamp, ...]) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "flavor": record.flavor,
        "commit": record.commit,
        "created_at": record.created_at,
        "slurm_job_id": record.slurm_job_id,
        "workload": record.workload,
        "cells": {
            name: {
                "name": cell.name,
                "arms": {
                    arm: {m: _series_to_json(s) for m, s in series.items()}
                    for arm, series in cell.arms.items()
                },
                "config": cell.config,
                "warnings": list(cell.warnings),
            }
            for name, cell in record.cells.items()
        },
        "unparsed": dict(record.unparsed),
        "stamps": [{"path": s.path, "size": s.size, "mtime_ns": s.mtime_ns} for s in stamps],
    }


def _record_from_json(payload: dict[str, Any]) -> tuple[RunRecord, tuple[FileStamp, ...]]:
    cells = {
        name: CellRecord(
            name=cell["name"],
            arms={
                arm: {m: _series_from_json(s) for m, s in series.items()}
                for arm, series in cell["arms"].items()
            },
            config=cell.get("config", {}),
            warnings=tuple(cell.get("warnings", ())),
        )
        for name, cell in payload.get("cells", {}).items()
    }
    record = RunRecord(
        run_id=payload["run_id"],
        run_dir=Path(payload["run_dir"]),
        flavor=payload["flavor"],
        commit=payload.get("commit"),
        created_at=payload.get("created_at"),
        slurm_job_id=payload.get("slurm_job_id"),
        workload=payload.get("workload"),
        cells=cells,
        unparsed=dict(payload.get("unparsed", {})),
    )
    stamps = tuple(
        FileStamp(path=s["path"], size=int(s["size"]), mtime_ns=int(s["mtime_ns"]))
        for s in payload.get("stamps", ())
    )
    return record, stamps


@dataclass(frozen=True, slots=True)
class ResultsIndex:
    """A parsed, queryable, persistable index of a run root."""

    root: Path
    runs: tuple[RunRecord, ...]
    #: ``run_id -> stamps of every file the extractors read for it``.
    stamps: dict[str, tuple[FileStamp, ...]]
    built_at: str

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "root": str(self.root),
            "built_at": self.built_at,
            "runs": [_record_to_json(r, self.stamps.get(r.run_id, ())) for r in self.runs],
        }
        path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ResultsIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"index schema version {version!r} != {INDEX_SCHEMA_VERSION}; rebuild the index"
            )
        runs: list[RunRecord] = []
        stamps: dict[str, tuple[FileStamp, ...]] = {}
        for entry in payload.get("runs", ()):
            record, run_stamps = _record_from_json(entry)
            runs.append(record)
            stamps[record.run_id] = run_stamps
        return cls(
            root=Path(payload["root"]),
            runs=tuple(runs),
            stamps=stamps,
            built_at=payload.get("built_at", ""),
        )

    # -- staleness ---------------------------------------------------------

    def stale_runs(self) -> tuple[str, ...]:
        """Run ids whose recorded files no longer match what is on disk.

        A run is stale when a file it read has changed size or mtime, has been
        removed, or when a run directory has appeared that the index has never
        seen.  Appearing runs are reported under their own id so a caller can
        tell "rebuild" from "nothing to do" without re-parsing the tree.
        """
        stale: list[str] = []
        for record in self.runs:
            for stamp in self.stamps.get(record.run_id, ()):
                candidate = record.run_dir / stamp.path
                try:
                    stat = candidate.stat()
                except OSError:
                    stale.append(record.run_id)
                    break
                if stat.st_size != stamp.size or stat.st_mtime_ns != stamp.mtime_ns:
                    stale.append(record.run_id)
                    break
        known = {r.run_id for r in self.runs}
        if self.root.is_dir():
            for entry in sorted(self.root.iterdir()):
                if entry.is_dir() and not entry.name.startswith(".") and entry.name not in known:
                    stale.append(entry.name)
        return tuple(stale)

    # -- queries -----------------------------------------------------------

    def select(
        self,
        *,
        flavor: str | Iterable[str] | None = None,
        commit: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        cell: str | None = None,
        metric: str | None = None,
        arm: str | None = None,
        non_empty: bool = False,
    ) -> list[RunRecord]:
        """Filter the index.

        All criteria are ANDed.  ``commit`` accepts a prefix, so a short sha
        selects the run it names.  ``since``/``until`` are ISO-8601 strings
        compared against :attr:`~model.RunRecord.created_at`; a run with no
        recoverable timestamp is excluded from a bounded query rather than
        silently included at an assumed date.  ``metric`` selects runs that
        have at least one cell carrying that metric (in ``arm`` if given).
        """
        flavors = _as_set(flavor)
        commits = _as_set(commit)
        lower = _parse_stamp(since) if since else None
        upper = _parse_stamp(until) if until else None

        out: list[RunRecord] = []
        for record in self.runs:
            if flavors is not None and record.flavor not in flavors:
                continue
            if commits is not None:
                if record.commit is None:
                    continue
                if not any(record.commit.startswith(c) for c in commits):
                    continue
            if lower is not None or upper is not None:
                if record.created_at is None:
                    continue
                stamp = _parse_stamp(record.created_at)
                if stamp is None:
                    continue
                if lower is not None and stamp < lower:
                    continue
                if upper is not None and stamp > upper:
                    continue
            if cell is not None and cell not in record.cells:
                continue
            if metric is not None and not _has_metric(record, metric, cell, arm):
                continue
            if non_empty and record.is_empty:
                continue
            out.append(record)
        return out

    def cells(self) -> dict[str, list[tuple[str, CellRecord]]]:
        """``cell name -> [(run id, cell)]`` across the whole index."""
        out: dict[str, list[tuple[str, CellRecord]]] = {}
        for record in self.runs:
            for name, cell in record.cells.items():
                out.setdefault(name, []).append((record.run_id, cell))
        return out


def _as_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def _parse_stamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_metric(record: RunRecord, metric: str, cell: str | None, arm: str | None) -> bool:
    for name, cell_record in record.cells.items():
        if cell is not None and name != cell:
            continue
        for arm_name, series in cell_record.arms.items():
            if arm is not None and arm_name != arm:
                continue
            if metric in series:
                return True
    return False


def build_index(root: Path) -> ResultsIndex:
    """Parse *root* into a :class:`ResultsIndex`."""
    parsed = _index_runs_stamped(root)
    return ResultsIndex(
        root=root,
        runs=tuple(record for record, _ in parsed),
        stamps={record.run_id: stamps for record, stamps in parsed},
        built_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
