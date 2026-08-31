"""Tests for the historical run index.

Every fixture below is a miniature of a shape that exists on disk under
``/data/ryan.kim/speedlm-runs``.  ``test_fixture_shapes_match_real_artifacts``
loads the real files and asserts the miniatures use the same keys, so a fixture
cannot drift into describing an artifact the harness never wrote.  The rest of
the suite runs entirely under ``tmp_path``: no GPU, no ``e2e`` marker, no
dependence on ``/data`` existing.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.harness.model import Direction
from tests.e2e.harness.resultsdb import (
    METRICS,
    ResultsIndex,
    build_index,
    index_runs,
    load_run,
)

REAL_ROOT = Path("/data/ryan.kim/speedlm-runs")
requires_real_runs = pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="historical run tree is not on this host"
)
requires_full_real_run_archive = pytest.mark.skipif(
    not REAL_ROOT.is_dir()
    or sum(1 for path in REAL_ROOT.iterdir() if path.is_dir()) < 60,
    reason="the complete historical run archive is not on this host",
)


# ---------------------------------------------------------------------------
# Fixture builders -- miniatures of the real artifact shapes
# ---------------------------------------------------------------------------

PROVENANCE = (
    "snapshot_commit=5c492ab63e793e8033bc53f1cac8b9136a01e526\n"
    "snapshot_source_ref=5c492ab\n"
    "source_repo=/admin/home/ryan.kim/speedlm-fr\n"
    "source_tree_dirty_at_generation=0\n"
    "generated_at=2026-08-06T23:37:24Z\n"
    "note=git archive contains the snapshot commit only\n"
)

SNAPSHOT_SBATCH = """#!/bin/bash
#SBATCH --job-name=speedlm-snap-configmatrix
#SBATCH --output=/runs/x/slurm-%j.out
snapshot=/data/ryan.kim/speedlm-snapshots/5c492ab63e793e8033bc53f1cac8b9136a01e526
commit=5c492ab63e793e8033bc53f1cac8b9136a01e526
"""

#: A hand-written sbatch: it names the test file but pins nothing.  Runs like
#: this must come back with ``commit=None``.
LEGACY_SBATCH = """#!/bin/bash
#SBATCH --job-name=speedlm-idle
cd /admin/home/ryan.kim/speedlm-fr
echo "commit=$(git rev-parse HEAD)"
.venv/bin/python -m pytest tests/e2e/test_live_idle_tuning.py
"""


def _sample(repeat: int, tps: float, mal: float) -> dict[str, Any]:
    return {
        "batch_tokens_per_second": tps * 6.7,
        "completion_tokens": 1024,
        "elapsed_seconds": 1024.0 / tps,
        "mean_accepted_length": mal,
        "prompt_tokens": [10, 15, 15, 14],
        "repeat": repeat,
        "tokens_per_second": tps,
    }


def config_matrix_cell_payload(
    *, stock_repeats: int = 4, candidate_repeats: int = 4, scored_per_block: int = 2
) -> dict[str, Any]:
    """A ``results/<cell>/result.json``, ABBA schedule and all."""
    return {
        "cell": {
            "context_length": "short",
            "execution_mode": "cuda_graphs",
            "name": "cuda_graphs-c8-short",
            "request_concurrency": 8,
            "resolved_num_speculative_tokens": 3,
        },
        "passed": True,
        "regression_failures": [],
        "samples": {
            "candidate": [
                _sample(i, 290.0 + i, 2.28 + i / 1000) for i in range(candidate_repeats)
            ],
            "stock": [_sample(i, 274.0 + i, 2.15 + i / 1000) for i in range(stock_repeats)],
        },
        "schedule": {
            "arm_blocks": 2,
            "blocks": [
                {
                    "arm": arm,
                    "block_index": index,
                    "engine_regime": {"execution_mode": "cuda_graphs", "max_model_len": 4096},
                    "fresh_process": True,
                    "scored_repeats": scored_per_block,
                    "warmup_repeats": 1,
                }
                for index, arm in enumerate(("stock", "candidate", "candidate", "stock"))
            ],
            "design": "mirrored rounds from speedlm.gate.runner._block_schedule",
            "repeats_per_arm": 4,
        },
        "summary": {
            "arms": {
                "candidate": {
                    "tokens_per_second": {"mean": 291.5, "n": 4, "standard_error": 1.09},
                    "mean_accepted_length": {"mean": 2.2815, "n": 4, "standard_error": 0.01},
                },
                "stock": {
                    "tokens_per_second": {"mean": 275.5, "n": 4, "standard_error": 0.5},
                    "mean_accepted_length": {"mean": 2.1515, "n": 4, "standard_error": 0.006},
                },
            },
            "fault_injection": {
                "applied_to": "decision input only; raw arm measurements are unchanged",
                "candidate_slowdown_percent": 0.0,
            },
            "paired_candidate_loss": {
                "mean_accepted_length": {"mean": -0.12, "n": 4, "standard_error": 0.0098},
                "tokens_per_second_percent": {"mean": -5.27, "n": 4, "standard_error": 0.36},
            },
        },
    }


CONFIG_MATRIX_MANIFEST: dict[str, Any] = {
    "candidate_draft": "/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/results/draft",
    "corpus": "/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl",
    "fault_injection": {"candidate_slowdown_percent": 0.0},
    "matrix": ["cuda_graphs-c8-short"],
    "matrix_dimensions": {
        "context_lengths": ["short", "long"],
        "execution_modes": ["eager", "cuda_graphs"],
        "request_concurrency": [1, 8, 32],
    },
    "measurement": {
        "accepted_length_statistic": "1 + accepted tokens / draft steps",
        "blocks_per_draft": 2,
        "max_tokens": 128,
        "repeats_per_draft": 4,
        "resolved_num_speculative_tokens": 3,
        "throughput_statistic": "Prometheus generation tokens / decode seconds",
        "warmup_repeats_per_block": 1,
    },
    "model": "Qwen/Qwen3-8B",
    "profile": "qwen3-8b-eagle3",
    "reference_draft": "RedHatAI/Qwen3-8B-speculator.eagle3",
}


def capture_overhead_payload(*, cycles: int = 2, prompts: tuple[str, ...] = ("short", "long")):
    """A ``results/<stamp>/capture_overhead.json``."""

    def rows(arm: str, bias: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cycle in range(cycles):
            for half in (0, 1):
                for prompt in prompts:
                    out.append(
                        {
                            "arm": arm,
                            "completion_tokens": 128,
                            "cycle": cycle,
                            "e2e_seconds": 0.47 + bias + cycle / 1000,
                            "half": half,
                            "median_inter_token_seconds": 0.0084 + bias / 100,
                            "prompt": prompt,
                            "throughput_tok_per_sec": 271.0 - bias * 10,
                            "token_event_count": 55,
                            "ttft_seconds": 0.0197 + bias,
                        }
                    )
        return out

    return {
        "armed_blocks": 20,
        "bounds": {"max_ttft_absolute_overhead_seconds": 0.1},
        "cycles": cycles,
        "disarmed_blocks": 20,
        "drafter": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "enforce_eager": False,
        "inject_delay_ms": 0.0,
        "max_tokens": 128,
        "per_block_warmup_passes": 1,
        "prefix_caching": False,
        "prompts": list(prompts),
        "reps_per_condition": 20,
        "samples": {"off": rows("off", 0.0), "on": rows("on", 0.004)},
        "shared_warmup_passes": 2,
        "summary": {"pairs": cycles * 2 * len(prompts)},
        "verifier": "Qwen/Qwen3-8B",
    }


def decision_payload(*, repeats: int = 3) -> dict[str, Any]:
    """A ``terminal-decision.json`` / per-cycle ``decision.json``."""
    return {
        "acceptance_criterion": "mean_accepted_length_delta",
        "arm_blocks": 2,
        "block_schedule": [
            {"arm": "candidate", "repeats": 2, "restarted": False},
            {"arm": "stock", "repeats": 2, "restarted": True},
        ],
        "candidate_avg_acceptance": 0.385,
        "candidate_avg_accepted_length": 2.156,
        "candidate_avg_tok_per_sec": 108.2,
        "engine_execution_mode": "eager",
        "per_repeat": [
            {
                "candidate_acceptance_rate": 0.386 + i / 10000,
                "candidate_accepted_length": 2.158 + i / 10000,
                "candidate_tok_per_sec": 108.4 - i,
                "invalid_rate": 0.0,
                "output_mismatches": 3 if i == 0 else 0,
                "repeat_index": i,
                "stock_acceptance_rate": 0.377 + i / 10000,
                "stock_accepted_length": 2.132 + i / 10000,
                "stock_tok_per_sec": 108.8 - i,
            }
            for i in range(repeats)
        ],
        "reason": "acceptance_below_threshold",
        "stock_avg_acceptance": 0.377,
        "stock_avg_accepted_length": 2.132,
        "stock_avg_tok_per_sec": 107.7,
        "verdict": "reject",
    }


def prom_text(*, generated: float, decode_seconds: float, drafted: float, accepted: float) -> str:
    """A vLLM ``/metrics`` scrape, trimmed to the counters the gate reads."""
    return (
        "# HELP vllm:generation_tokens_total Number of generation tokens\n"
        "# TYPE vllm:generation_tokens_total counter\n"
        f'vllm:generation_tokens_total{{model_name="Qwen/Qwen3-8B"}} {generated}\n'
        f'vllm:request_decode_time_seconds_sum{{model_name="Qwen/Qwen3-8B"}} {decode_seconds}\n'
        f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
        f"vllm:spec_decode_num_accepted_tokens_total {accepted}\n"
        "vllm:spec_decode_num_drafts_total 100.0\n"
    )


def write_run(
    root: Path,
    name: str,
    *,
    sbatch: str | None = SNAPSHOT_SBATCH,
    provenance: str | None = PROVENANCE,
    slurm_job: str | None = "371234",
    files: dict[str, Any] | None = None,
) -> Path:
    """Materialise one run directory.  ``files`` maps ``results``-relative paths
    to JSON payloads, raw strings, or ``bytes``."""
    run = root / name
    (run / "results").mkdir(parents=True)
    if sbatch is not None:
        (run / "job.sbatch").write_text(sbatch)
    if provenance is not None:
        (run / "snapshot-provenance.txt").write_text(provenance)
    if slurm_job is not None:
        (run / f"slurm-{slurm_job}.out").write_text("started_at=2026-08-06T23:40:00Z\n")
    for relative, payload in (files or {}).items():
        target = run / "results" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        elif isinstance(payload, str):
            target.write_text(payload)
        else:
            target.write_text(json.dumps(payload))
    return run


@pytest.fixture
def config_matrix_run(tmp_path: Path) -> Path:
    return write_run(
        tmp_path,
        "config-matrix-fixture",
        files={
            "manifest.json": CONFIG_MATRIX_MANIFEST,
            "cuda_graphs-c8-short/result.json": config_matrix_cell_payload(),
        },
    )


# ---------------------------------------------------------------------------
# Rule 1 -- individual observations survive
# ---------------------------------------------------------------------------


def test_config_matrix_keeps_every_repeat_not_a_mean(config_matrix_run: Path) -> None:
    cell = load_run(config_matrix_run).cells["cuda_graphs-c8-short"]
    series = cell.arms["stock"]["tokens_per_second"]
    assert series.values == (274.0, 275.0, 276.0, 277.0)
    assert series.reduced is False
    # The summary in the same artifact says 275.5.  Recording that instead
    # would destroy the dispersion the comparator needs.
    assert 275.5 not in series.values


def test_summary_only_arm_is_marked_reduced(tmp_path: Path) -> None:
    payload = config_matrix_cell_payload()
    payload["samples"] = {}
    run = write_run(
        tmp_path, "summary-only", files={"cuda_graphs-c8-short/result.json": payload}
    )
    cell = load_run(run).cells["cuda_graphs-c8-short"]
    series = cell.arms["stock"]["tokens_per_second"]
    assert series.reduced is True
    assert series.values == (275.5,)
    assert any("only the summary survived" in w for w in cell.warnings)


def test_idle_tuning_decision_keeps_per_repeat_observations(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "idle-fixture",
        files={"live-idle-tuning/terminal-decision.json": decision_payload(repeats=3)},
    )
    cell = load_run(run).cells["idle-tuning"]
    assert cell.arms["stock"]["tok_per_sec"].values == (108.8, 107.8, 106.8)
    assert cell.arms["candidate"]["tok_per_sec"].values == (108.4, 107.4, 106.4)
    assert all(
        not s.reduced for arm in cell.arms.values() for s in arm.values()
    )


def test_decision_without_per_repeat_falls_back_to_reduced(tmp_path: Path) -> None:
    payload = decision_payload()
    del payload["per_repeat"]
    run = write_run(
        tmp_path, "idle-noreps", files={"live-idle-tuning/terminal-decision.json": payload}
    )
    cell = load_run(run).cells["idle-tuning"]
    series = cell.arms["stock"]["tok_per_sec"]
    assert series.reduced is True
    assert series.values == (107.7,)
    assert any("kept no per-repeat record" in w for w in cell.warnings)


def test_gate_prometheus_scrapes_yield_per_repeat_deltas(tmp_path: Path) -> None:
    """Consecutive cumulative scrapes bracket one repeat each."""
    scrapes = {
        "live-idle-tuning/gate-metrics/stock-before.prom.gz": gzip.compress(
            prom_text(generated=0.0, decode_seconds=0.0, drafted=0.0, accepted=0.0).encode()
        ),
        "live-idle-tuning/gate-metrics/stock-after-repeat-0.prom.gz": gzip.compress(
            prom_text(generated=100.0, decode_seconds=1.0, drafted=90.0, accepted=45.0).encode()
        ),
        "live-idle-tuning/gate-metrics/stock-after-repeat-1.prom.gz": gzip.compress(
            prom_text(generated=300.0, decode_seconds=3.0, drafted=270.0, accepted=189.0).encode()
        ),
    }
    run = write_run(tmp_path, "gate-prom", files=scrapes)
    cell = load_run(run).cells["idle-tuning-prometheus"]
    rates = cell.arms["stock"]["acceptance_rate"]
    assert rates.values == (0.5, 0.8)
    assert rates.pass_indices == (0, 1)
    assert cell.arms["stock"]["output_tok_per_sec"].values == (100.0, 100.0)


# ---------------------------------------------------------------------------
# Rule 2 -- pass indices
# ---------------------------------------------------------------------------


def test_config_matrix_carries_pass_indices(config_matrix_run: Path) -> None:
    cell = load_run(config_matrix_run).cells["cuda_graphs-c8-short"]
    assert cell.arms["stock"]["tokens_per_second"].pass_indices == (0, 1, 2, 3)
    assert cell.arms["candidate"]["tokens_per_second"].pass_indices == (0, 1, 2, 3)


def test_pass_indices_refused_when_schedule_does_not_account_for_samples(
    tmp_path: Path,
) -> None:
    """A schedule that scores fewer repeats than are present is not this run's."""
    payload = config_matrix_cell_payload(scored_per_block=1)  # schedule says 2 per arm
    run = write_run(tmp_path, "bad-sched", files={"c/result.json": payload})
    cell = load_run(run).cells["cuda_graphs-c8-short"]
    assert cell.arms["stock"]["tokens_per_second"].pass_indices is None
    assert any("pass indices refused" in w for w in cell.warnings)


def test_capture_overhead_pairs_arms_by_cycle_half_prompt(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "capoverhead",
        files={"capture-overhead-20260806T234924Z/capture_overhead.json": (
            capture_overhead_payload()
        )},
    )
    cell = load_run(run).cells["capture-overhead"]
    off = cell.arms["off"]["ttft_seconds"]
    on = cell.arms["on"]["ttft_seconds"]
    assert off.pass_indices == on.pass_indices
    assert sorted(off.pass_indices or ()) == list(range(8))
    # ``cycle`` and ``half`` identify the pass; they are not measurements.
    assert "cycle" not in cell.arms["off"]
    assert "half" not in cell.arms["off"]


def test_capture_overhead_refuses_pairing_when_arms_differ(tmp_path: Path) -> None:
    payload = capture_overhead_payload()
    payload["samples"]["on"] = payload["samples"]["on"][:-1]
    run = write_run(tmp_path, "capskew", files={"co/capture_overhead.json": payload})
    cell = load_run(run).cells["capture-overhead"]
    assert cell.arms["off"]["ttft_seconds"].pass_indices is None
    assert any("do not share a (cycle, half, prompt) set" in w for w in cell.warnings)


def test_idle_tuning_pass_indices_come_from_repeat_index(tmp_path: Path) -> None:
    run = write_run(
        tmp_path, "idle-pi", files={"live-idle-tuning/terminal-decision.json": decision_payload()}
    )
    cell = load_run(run).cells["idle-tuning"]
    assert cell.arms["stock"]["tok_per_sec"].pass_indices == (0, 1, 2)
    assert cell.arms["candidate"]["tok_per_sec"].pass_indices == (0, 1, 2)


# ---------------------------------------------------------------------------
# Rule 3 -- nothing is dropped in silence
# ---------------------------------------------------------------------------


def test_malformed_artifact_lands_in_unparsed(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "broken",
        files={
            "cuda_graphs-c8-short/result.json": "{not json at all",
            "manifest.json": CONFIG_MATRIX_MANIFEST,
        },
    )
    record = load_run(run)
    key = "results/cuda_graphs-c8-short/result.json"
    assert key in record.unparsed, record.unparsed
    assert "unreadable JSON" in record.unparsed[key]
    assert "cuda_graphs-c8-short" not in record.cells


def test_unrecognised_artifact_is_reported_not_skipped(tmp_path: Path) -> None:
    run = write_run(tmp_path, "novel", files={"novel-benchmark.json": {"score": 1}})
    record = load_run(run)
    assert record.unparsed["results/novel-benchmark.json"] == (
        "no registered extractor claims 'novel-benchmark.json'"
    )


def test_run_without_results_directory_says_so(tmp_path: Path) -> None:
    run = tmp_path / "aborted"
    run.mkdir()
    (run / "job.sbatch").write_text(SNAPSHOT_SBATCH)
    record = load_run(run)
    assert record.is_empty
    assert record.unparsed["results/"] == "run produced no results directory"


def test_payload_files_are_reported_in_aggregate(tmp_path: Path) -> None:
    """Thousands of captured bodies must be reported, but not one line each."""
    files: dict[str, Any] = {
        f"live-idle-tuning/seed-response-{i:04d}.json": {"choices": []} for i in range(1, 26)
    }
    files["live-idle-tuning/terminal-decision.json"] = decision_payload()
    record = load_run(write_run(tmp_path, "payloads", files=files))
    seed_lines = [k for k in record.unparsed if "seed-response" in k]
    assert seed_lines == ["results/live-idle-tuning/seed-response-N.json"]
    assert "25 files" in record.unparsed[seed_lines[0]]


def test_declining_extractor_does_not_shadow_a_later_one(tmp_path: Path) -> None:
    """Three shapes share the basename ``result.json``.

    The configuration-matrix extractor is offered a hot-swap result first and
    declines it; that decline must not be reported once the hot-swap extractor
    has claimed the file.
    """
    payload = {
        "verifier": "Qwen/Qwen3-8B",
        "drafter": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "injected_rpc_calls": ["_apply_weights"] * 30,
    }
    record = load_run(
        write_run(tmp_path, "hotswap", files={"draft-hot-swap-20260801T070503Z/result.json": (
            payload
        )})
    )
    assert record.cells["draft-hot-swap"].arms["single"]["injected_rpc_calls"].values == (30.0,)
    assert not [k for k in record.unparsed if k.endswith("result.json")]


# ---------------------------------------------------------------------------
# Rule 4 -- provenance
# ---------------------------------------------------------------------------


def test_commit_recovered_from_snapshot_provenance(tmp_path: Path) -> None:
    """Provenance wins over the sbatch: it is written when the snapshot is."""
    disagreeing = SNAPSHOT_SBATCH.replace(
        "commit=5c492ab63e793e8033bc53f1cac8b9136a01e526\n",
        "commit=" + "b" * 40 + "\n",
    )
    record = load_run(write_run(tmp_path, "prov", sbatch=disagreeing, files={}))
    assert record.commit == "5c492ab63e793e8033bc53f1cac8b9136a01e526"
    assert record.created_at == "2026-08-06T23:37:24Z"


def test_commit_recovered_from_snapshot_sbatch_without_provenance(tmp_path: Path) -> None:
    run = write_run(tmp_path, "nosprov", provenance=None, files={})
    record = load_run(run)
    assert record.commit == "5c492ab63e793e8033bc53f1cac8b9136a01e526"


def test_legacy_run_gets_no_commit(tmp_path: Path) -> None:
    """A run whose sbatch only echoes ``git rev-parse HEAD`` pins nothing."""
    run = write_run(tmp_path, "legacy", sbatch=LEGACY_SBATCH, provenance=None, files={})
    record = load_run(run)
    assert record.commit is None
    assert record.flavor == "idle-tuning"


def test_slurm_job_id_comes_from_the_out_filename(tmp_path: Path) -> None:
    run = write_run(tmp_path, "slurmid", slurm_job="369146", files={})
    assert load_run(run).slurm_job_id == "369146"


def test_flavor_comes_from_the_sbatch_job_name(config_matrix_run: Path) -> None:
    assert load_run(config_matrix_run).flavor == "config-matrix"


# ---------------------------------------------------------------------------
# Rule 5 -- direction is declared, never inferred
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "direction"),
    [
        ("tokens_per_second", Direction.HIGHER_IS_BETTER),
        ("throughput_tok_per_sec", Direction.HIGHER_IS_BETTER),
        ("mean_accepted_length", Direction.HIGHER_IS_BETTER),
        ("acceptance_rate", Direction.HIGHER_IS_BETTER),
        ("ttft_seconds", Direction.LOWER_IS_BETTER),
        ("tpot_ms", Direction.LOWER_IS_BETTER),
        ("e2e_seconds", Direction.LOWER_IS_BETTER),
        ("elapsed_seconds", Direction.LOWER_IS_BETTER),
        ("val_loss", Direction.LOWER_IS_BETTER),
        ("completion_tokens", Direction.NEUTRAL),
        ("prompt_tokens", Direction.NEUTRAL),
        ("token_event_count", Direction.NEUTRAL),
    ],
)
def test_declared_directions(metric: str, direction: Direction) -> None:
    assert METRICS[metric].direction is direction


def test_directions_survive_into_the_series(config_matrix_run: Path) -> None:
    arms = load_run(config_matrix_run).cells["cuda_graphs-c8-short"].arms
    assert arms["stock"]["tokens_per_second"].direction is Direction.HIGHER_IS_BETTER
    assert arms["stock"]["elapsed_seconds"].direction is Direction.LOWER_IS_BETTER
    assert arms["stock"]["completion_tokens"].direction is Direction.NEUTRAL


def test_undeclared_metric_is_refused_and_warned(tmp_path: Path) -> None:
    payload = config_matrix_cell_payload()
    for sample in payload["samples"]["stock"]:
        sample["mystery_score"] = 1.0
    run = write_run(tmp_path, "mystery", files={"c/result.json": payload})
    cell = load_run(run).cells["cuda_graphs-c8-short"]
    assert "mystery_score" not in cell.arms["stock"]
    assert any("'mystery_score' has no declared direction" in w for w in cell.warnings)


# ---------------------------------------------------------------------------
# Prometheus reuse
# ---------------------------------------------------------------------------


def test_prometheus_pair_uses_the_gate_parser(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "prom",
        files={
            "gpt-oss-20b-eagle3/metrics-before.prom": prom_text(
                generated=1000.0, decode_seconds=10.0, drafted=500.0, accepted=200.0
            ),
            "gpt-oss-20b-eagle3/metrics-after.prom": prom_text(
                generated=1290.0, decode_seconds=12.0, drafted=790.0, accepted=292.0
            ),
        },
    )
    cell = load_run(run).cells["gpt-oss-20b-eagle3-prometheus"]
    single = cell.arms["single"]
    assert single["drafted_tokens"].values == (290.0,)
    assert single["accepted_tokens"].values == (92.0,)
    assert single["acceptance_rate"].values == pytest.approx((92.0 / 290.0,))
    assert single["output_tok_per_sec"].values == pytest.approx((145.0,))


def test_counter_reset_is_reported_not_averaged(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "reset",
        files={
            "cell/metrics-before.prom": prom_text(
                generated=1000.0, decode_seconds=10.0, drafted=500.0, accepted=200.0
            ),
            "cell/metrics-after.prom": prom_text(
                generated=10.0, decode_seconds=1.0, drafted=5.0, accepted=2.0
            ),
        },
    )
    record = load_run(run)
    assert "cell-prometheus" not in record.cells
    assert "counter reset between scrapes" in record.unparsed["results/cell/metrics-after.prom"]


# ---------------------------------------------------------------------------
# Index: persistence, staleness, queries
# ---------------------------------------------------------------------------


def test_index_roundtrips_through_json(tmp_path: Path, config_matrix_run: Path) -> None:
    index = build_index(config_matrix_run.parent)
    path = tmp_path / "out" / "index.json"
    index.save(path)
    back = ResultsIndex.load(path)

    original = index.runs[0].cells["cuda_graphs-c8-short"].arms["stock"]["tokens_per_second"]
    restored = back.runs[0].cells["cuda_graphs-c8-short"].arms["stock"]["tokens_per_second"]
    assert restored == original
    assert restored.direction is Direction.HIGHER_IS_BETTER
    assert restored.pass_indices == (0, 1, 2, 3)
    assert back.runs[0].commit == index.runs[0].commit
    assert back.runs[0].unparsed == index.runs[0].unparsed


def test_index_detects_a_changed_artifact(tmp_path: Path, config_matrix_run: Path) -> None:
    index = build_index(config_matrix_run.parent)
    assert index.stale_runs() == ()

    target = config_matrix_run / "results" / "cuda_graphs-c8-short" / "result.json"
    payload = json.loads(target.read_text())
    payload["samples"]["stock"].append(_sample(4, 999.0, 2.5))
    target.write_text(json.dumps(payload))

    assert index.stale_runs() == ("config-matrix-fixture",)


def test_index_detects_a_new_run(tmp_path: Path, config_matrix_run: Path) -> None:
    index = build_index(config_matrix_run.parent)
    write_run(config_matrix_run.parent, "later-run", files={})
    assert "later-run" in index.stale_runs()


def test_index_rejects_a_foreign_schema_version(tmp_path: Path, config_matrix_run: Path) -> None:
    path = tmp_path / "index.json"
    build_index(config_matrix_run.parent).save(path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rebuild the index"):
        ResultsIndex.load(path)


@pytest.fixture
def mixed_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    write_run(
        root,
        "config-matrix-fixture",
        files={"cuda_graphs-c8-short/result.json": config_matrix_cell_payload()},
    )
    write_run(
        root,
        "idle-fixture",
        sbatch=LEGACY_SBATCH,
        provenance=None,
        files={"live-idle-tuning/terminal-decision.json": decision_payload()},
    )
    write_run(root, "aborted-fixture", sbatch=LEGACY_SBATCH, provenance=None, files={})
    # No provenance, no generated sbatch, no slurm log, no stamp in the name:
    # nothing on disk says when this ran.
    write_run(
        root,
        "undated-fixture",
        sbatch=LEGACY_SBATCH,
        provenance=None,
        slurm_job=None,
        files={"live-idle-tuning/terminal-decision.json": decision_payload()},
    )
    return root


def test_select_by_flavor_commit_cell_and_metric(mixed_root: Path) -> None:
    index = build_index(mixed_root)
    assert [r.run_id for r in index.select(flavor="config-matrix")] == ["config-matrix-fixture"]
    assert [r.run_id for r in index.select(commit="5c492ab")] == ["config-matrix-fixture"]
    assert [r.run_id for r in index.select(cell="idle-tuning")] == [
        "idle-fixture",
        "undated-fixture",
    ]
    assert [r.run_id for r in index.select(metric="tokens_per_second")] == [
        "config-matrix-fixture"
    ]
    assert [r.run_id for r in index.select(metric="tok_per_sec", arm="candidate")] == [
        "idle-fixture",
        "undated-fixture",
    ]
    assert {r.run_id for r in index.select(non_empty=True)} == {
        "config-matrix-fixture",
        "idle-fixture",
        "undated-fixture",
    }


def test_select_by_date_range_excludes_undated_runs(mixed_root: Path) -> None:
    index = build_index(mixed_root)
    # The legacy runs have no provenance and no timestamped name; their only
    # date comes from the slurm log, so bound the window below it.
    # config-matrix-fixture is dated 23:37:24Z by its provenance file; the two
    # legacy runs have no provenance, so their date comes from the slurm log at
    # 23:40:00Z.
    assert index.select()[-1].run_id == "undated-fixture"
    assert index.select()[-1].created_at is None
    # undated-fixture is excluded from every bounded query rather than being
    # silently placed at an assumed date.
    assert [r.run_id for r in index.select(since="2026-08-06T23:39:00Z")] == [
        "aborted-fixture",
        "idle-fixture",
    ]
    assert index.select(since="2026-08-07T00:00:00Z") == []
    assert [r.run_id for r in index.select(until="2026-08-06T23:38:00Z")] == [
        "config-matrix-fixture"
    ]


def test_index_runs_returns_every_directory(mixed_root: Path) -> None:
    records = index_runs(mixed_root)
    assert [r.run_id for r in records] == [
        "aborted-fixture",
        "config-matrix-fixture",
        "idle-fixture",
        "undated-fixture",
    ]
    assert [r.is_empty for r in records] == [True, False, False, False]


def test_index_runs_refuses_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        index_runs(tmp_path / "nope")


# ---------------------------------------------------------------------------
# The fixtures describe the real artifacts
# ---------------------------------------------------------------------------


def _real(pattern: str) -> Path | None:
    return next(iter(sorted(REAL_ROOT.glob(pattern))), None)


def _real_all(pattern: str) -> list[Path]:
    return sorted(REAL_ROOT.glob(pattern))


@requires_real_runs
@pytest.mark.parametrize(
    ("pattern", "builder"),
    [
        (
            "config-matrix-*/results/*/result.json",
            lambda: config_matrix_cell_payload(),
        ),
        ("config-matrix-*/results/manifest.json", lambda: CONFIG_MATRIX_MANIFEST),
        (
            "capture-overhead-*/results/*/capture_overhead.json",
            lambda: capture_overhead_payload(),
        ),
        ("*/results/*/terminal-decision.json", lambda: decision_payload()),
    ],
)
def test_fixture_shapes_match_real_artifacts(pattern: str, builder: Any) -> None:
    """Every key a fixture uses must exist in the real artifact of that shape."""
    paths = _real_all(pattern)
    if not paths:
        pytest.skip(f"no real artifact matches {pattern}")
    # The shape drifted across the week: early gate decisions predate
    # ``arm_blocks``.  A fixture key must appear in *some* real artifact of the
    # shape, and the sub-key checks below run against the richest one.
    union: set[str] = set()
    for candidate in paths:
        payload = json.loads(candidate.read_text())
        if isinstance(payload, dict):
            union |= set(payload)
    real = max(
        (json.loads(c.read_text()) for c in paths),
        key=lambda p: len(p) if isinstance(p, dict) else 0,
    )
    fixture = builder()
    missing = set(fixture) - union
    assert not missing, f"{pattern}: fixture invents top-level keys {sorted(missing)}"
    for key, value in fixture.items():
        if key in ("samples", "arms", "summary", "schedule", "cell", "measurement") and (
            isinstance(value, dict) and isinstance(real.get(key), dict)
        ):
            sub_missing = set(value) - set(real[key])
            assert not sub_missing, f"{pattern}.{key}: invents {sorted(sub_missing)}"


@requires_real_runs
def test_real_sample_rows_use_the_fixture_field_names() -> None:
    path = _real("config-matrix-*/results/*/result.json")
    if path is None:
        pytest.skip("no real configuration-matrix cell on this host")
    real = json.loads(path.read_text())["samples"]["stock"][0]
    assert set(_sample(0, 1.0, 1.0)) == set(real)


@requires_full_real_run_archive
def test_the_real_tree_indexes_without_raising() -> None:
    records = index_runs(REAL_ROOT)
    assert len(records) >= 60
    non_empty = [r for r in records if not r.is_empty]
    assert len(non_empty) >= 40
    # Every recovered commit is a full sha, never a short ref or a HEAD guess.
    assert all(len(r.commit) == 40 for r in records if r.commit is not None)
    # Nothing reached the index under a name the direction table never declared.
    emitted = {
        name
        for record in records
        for cell in record.cells.values()
        for arm in cell.arms.values()
        for name in arm
    }
    assert emitted <= set(METRICS), sorted(emitted - set(METRICS))
    assert "tokens_per_second" in emitted and "ttft_seconds" in emitted

    # The configuration matrix is the shape the project's conclusions rest on:
    # its cells must come back with the per-repeat lists, not a single number.
    matrix_cells = [
        cell
        for record in records
        if record.flavor == "config-matrix"
        for name, cell in record.cells.items()
        if name != "manifest" and not name.endswith("prometheus")
    ]
    assert matrix_cells, "no configuration-matrix cells recovered"
    for cell in matrix_cells:
        assert set(cell.arms) >= {"stock", "candidate"}
        for arm in ("stock", "candidate"):
            series = cell.arms[arm]["tokens_per_second"]
            assert len(series.values) > 1
            assert series.pass_indices is not None
            assert not series.reduced
