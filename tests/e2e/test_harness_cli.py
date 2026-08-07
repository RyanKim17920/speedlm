"""Tests for ``scripts/speedbench``, the harness's single entry point.

CI-safe on purpose: no ``e2e`` marker, no GPU, no SLURM, no ``/data``.  Every
subcommand is driven in-process (the CLI is loaded by path because the script is
extensionless) against fixture directories under ``tmp_path``.

The CLI's contract is one sentence -- *every subcommand exits non-zero on the
failure it exists to detect* -- so these tests are organised around that claim
and each one asserts BOTH halves: the failure is caught AND the healthy case is
not.  A gate that refuses everything is worth exactly as little as one that
refuses nothing, and this repository has shipped both.

The launcher-drift test is the load-bearing one.  ``speedbench launch`` composes
an argv for ``scripts/make_snapshot_run.sh``; that argv is checked against the
option names parsed out of the launcher's own ``case`` block rather than against
a list restated here, so the CLI cannot grow an option the launcher would
reject.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.harness import workloads as W

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEEDBENCH = REPO_ROOT / "scripts" / "speedbench"
LAUNCHER = REPO_ROOT / "scripts" / "make_snapshot_run.sh"


def _load_cli() -> Any:
    """Import ``scripts/speedbench``.  It has no ``.py`` suffix, hence the loader."""
    loader = importlib.machinery.SourceFileLoader("speedbench_cli", str(SPEEDBENCH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because @dataclass resolves its own module out
    # of sys.modules while the class body is being processed.
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


cli = _load_cli()


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Invoke the CLI the way a shell would; return ``(exit code, stdout, stderr)``."""
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Run fixtures -- miniatures of a real ``terminal-decision.json`` run directory
# ---------------------------------------------------------------------------
def _provenance(commit: str) -> str:
    return (
        f"snapshot_commit={commit}\n"
        f"snapshot_source_ref={commit[:7]}\n"
        "source_repo=/admin/home/ryan.kim/speedlm-fr\n"
        "source_tree_dirty_at_generation=0\n"
        "generated_at=2026-08-06T23:37:24Z\n"
    )


def _decision(stock: list[float], candidate: list[float]) -> dict[str, Any]:
    """An idle-tuning ``terminal-decision.json`` carrying per-repeat observations."""
    return {
        "acceptance_criterion": "mean_accepted_length_delta",
        "arm_blocks": 2,
        "candidate_avg_acceptance": 0.385,
        "candidate_avg_accepted_length": 2.156,
        "candidate_avg_tok_per_sec": sum(candidate) / len(candidate),
        "engine_execution_mode": "eager",
        "per_repeat": [
            {
                "candidate_acceptance_rate": 0.386,
                "candidate_accepted_length": 2.158,
                "candidate_tok_per_sec": candidate[i],
                "invalid_rate": 0.0,
                "output_mismatches": 0,
                "repeat_index": i,
                "stock_acceptance_rate": 0.377,
                "stock_accepted_length": 2.132,
                "stock_tok_per_sec": stock[i],
            }
            for i in range(len(stock))
        ],
        "reason": "acceptance_below_threshold",
        "stock_avg_acceptance": 0.377,
        "stock_avg_accepted_length": 2.132,
        "stock_avg_tok_per_sec": sum(stock) / len(stock),
        "verdict": "reject",
    }


def write_run(
    root: Path,
    name: str,
    *,
    commit: str,
    stock: list[float] | None = None,
    candidate: list[float] | None = None,
) -> Path:
    """Materialise one run directory.  Omit the arms for a run with no measurements."""
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "snapshot-provenance.txt").write_text(_provenance(commit))
    (run / "job.sbatch").write_text(
        "#!/bin/bash\n#SBATCH --job-name=speedlm-snap-idle\n"
        f"commit={commit}\n"
    )
    (run / "slurm-371234.out").write_text("started_at=2026-08-06T23:40:00Z\n")
    if stock is not None and candidate is not None:
        target = run / "results" / "live-idle-tuning" / "terminal-decision.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_decision(stock, candidate)))
    return run


#: The same stock arm on both sides, so only the candidate arm moves.
STOCK = [100.0, 101.0, 99.0, 100.5]
#: Within the 2% materiality bar, measured tightly enough to prove flatness.
FLAT_CANDIDATE = [100.2, 100.8, 99.4, 100.3]
#: A 20% throughput collapse: material, and many standard errors wide.
REGRESSED_CANDIDATE = [80.0, 80.5, 79.6, 80.2]


@pytest.fixture
def flat_pair(tmp_path: Path) -> Path:
    root = tmp_path / "runs-flat"
    root.mkdir()
    write_run(root, "run-base", commit="a" * 40, stock=STOCK, candidate=FLAT_CANDIDATE)
    write_run(root, "run-cand", commit="b" * 40, stock=STOCK, candidate=FLAT_CANDIDATE)
    return root


@pytest.fixture
def regressed_pair(tmp_path: Path) -> Path:
    root = tmp_path / "runs-regressed"
    root.mkdir()
    write_run(root, "run-base", commit="a" * 40, stock=STOCK, candidate=FLAT_CANDIDATE)
    write_run(
        root, "run-cand", commit="b" * 40, stock=STOCK, candidate=REGRESSED_CANDIDATE
    )
    return root


# ---------------------------------------------------------------------------
# Workload fixtures -- a manifest and a corpus small enough to live in tmp_path
# ---------------------------------------------------------------------------
def _records(count: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index in range(count):
        messages = [
            {"role": "user", "content": f"question number {index} about alpha particles"}
        ]
        text = W.flatten_prompt(messages)
        out.append(
            {
                "id": f"rec-{index}",
                "messages": messages,
                "prompt_chars": len(text),
                # A stand-in for the builder's real tokenizer count; verification
                # never recomputes it, it only checks the digest covers it.
                "prompt_tokens": 10 + index,
            }
        )
    return out


@pytest.fixture
def workload_dir(tmp_path: Path) -> Path:
    """A spec directory whose single manifest genuinely describes its corpus.

    The characteristics are produced by ``recompute_characteristics`` -- the same
    function the verifier uses -- rather than typed out here, so the fixture
    starts honest and the only thing a test has to do to make it lie is change
    the bytes.
    """
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    corpus = tmp_path / "corpus" / "records.jsonl"
    corpus.parent.mkdir()
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()), encoding="utf-8"
    )

    manifest: dict[str, Any] = {
        "schema_version": W.SCHEMA_VERSION,
        "name": "fixture-chat",
        "version": "1",
        "description": "a corpus small enough to verify in milliseconds",
        "domain": "test",
        "source": {
            "format": W.RECORD_FORMAT,
            "path": str(corpus),
            "sha256": W.file_sha256(corpus),
            "size_bytes": corpus.stat().st_size,
        },
        "provenance": {"built_by": "tests/e2e/test_harness_cli.py"},
        "characteristics": {},
        "tolerance": {
            "percentile_relative": 0.0,
            "fraction_absolute": 0.0,
            "count_absolute": 0.0,
        },
        "requirements": {"min_max_model_len": 4096, "output_reserve_tokens": 128},
        "bands": {"short": [0.0, 0.5], "long": [0.5, 1.0]},
    }
    path = spec_dir / "fixture-chat.json"
    path.write_text(json.dumps(manifest))
    # Fill the declared characteristics from the corpus itself.
    manifest["characteristics"] = W.recompute_characteristics(
        W.load_records(W.load_spec_file(path))
    )
    path.write_text(json.dumps(manifest))
    return spec_dir


# ---------------------------------------------------------------------------
# 1 + 2 -- launch refuses a failing preflight, and the override is the only way past
# ---------------------------------------------------------------------------
#: config-matrix hard-requires --candidate-drafter and --pytest-k; omitting them
#: is a preflight ERROR and nothing else about this argv is wrong.
FAILING_LAUNCH = [
    "launch",
    "--dry-run",
    "--flavor",
    "config-matrix",
    "--skip-git-checks",
]
PASSING_LAUNCH = [
    "launch",
    "--dry-run",
    "--flavor",
    "config-matrix",
    "--candidate-drafter",
    "/data/drafts/candidate",
    "--pytest-k",
    "test_cell",
    "--skip-git-checks",
]


def test_launch_dry_run_refuses_a_config_that_fails_preflight(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_cli(FAILING_LAUNCH, capsys)
    assert code != 0
    assert "missing-required-option" in err
    assert "refusing to launch" in err
    # The important half: nothing that could be pasted into a shell was printed.
    assert out.strip() == ""
    assert "make_snapshot_run.sh" not in out


def test_launch_dry_run_prints_the_command_for_a_config_that_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_cli(PASSING_LAUNCH, capsys)
    assert code == 0, err
    assert "make_snapshot_run.sh" in out
    assert "--flavor config-matrix" in out
    assert "--candidate-drafter /data/drafts/candidate" in out
    assert "refusing to launch" not in err


def test_override_is_the_only_way_past_a_failing_preflight_and_it_shouts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    refused, refused_out, _ = run_cli(FAILING_LAUNCH, capsys)
    assert refused != 0
    assert refused_out.strip() == ""

    code, out, err = run_cli([*FAILING_LAUNCH, cli.OVERRIDE_FLAG], capsys)
    assert code == 0
    assert "make_snapshot_run.sh" in out
    assert "WARNING" in err
    assert "FAILED preflight" in err
    assert "overridden, not fixed" in err
    # The warning must reach stderr, never be buried in the machine-readable
    # stdout a caller might be piping into a launcher.
    assert "WARNING" not in out


# ---------------------------------------------------------------------------
# 3 -- the composed argv is one the real launcher would accept
# ---------------------------------------------------------------------------
def launcher_accepted_options() -> set[str]:
    """Option names parsed out of ``make_snapshot_run.sh``'s own argv ``case``.

    Read from the script rather than restated so the two cannot drift: an option
    the CLI emits and the launcher does not parse falls into the launcher's
    ``*) echo "unknown option"; exit 2`` arm.
    """
    text = LAUNCHER.read_text()
    start = text.index("while [[ $# -gt 0 ]]; do")
    block = text[start : text.index("\ndone", start)]
    names: set[str] = set()
    for arm in re.findall(r"^\s+(-[-a-zA-Z0-9|]+)\)", block, re.MULTILINE):
        names.update(arm.split("|"))
    return names


def test_launcher_option_parsing_finds_the_real_case_arms() -> None:
    """Guards the guard: an empty parse would make the next test vacuous."""
    accepted = launcher_accepted_options()
    assert {"--flavor", "--vllm-args", "--no-corpus", "--force", "--skip-preflight"} <= accepted


def test_launch_emits_only_options_the_launcher_parses() -> None:
    accepted = launcher_accepted_options()
    argv = ["launch", "--dry-run", "--skip-git-checks", "--flavor", "live-vllm"]
    for _dest, option in cli.LAUNCHER_OPTIONS:
        if option == "--flavor":
            continue
        argv.extend([option, "auto" if option == "--runner" else f"value-for{option}"])
    argv.append("--force")
    args = cli.build_parser().parse_args(argv)
    command = cli.launcher_command(args)

    emitted = {token for token in command if token.startswith("-")}
    assert len(emitted) >= len(cli.LAUNCHER_OPTIONS), emitted
    unknown = sorted(emitted - accepted)
    assert unknown == [], f"speedbench emits options make_snapshot_run.sh cannot parse: {unknown}"
    # speedbench's own flags must never be forwarded.
    assert "--skip-git-checks" not in command
    assert "--dry-run" not in command
    assert "--spec-dir" not in command
    # --workload and --max-model-len, by contrast, MUST be forwarded.  They
    # were speedbench-only until the launcher grew them; if they stop being
    # forwarded the job silently runs generic-chat at 4096 whatever preflight
    # was told, which is the exact class of "looks configured, is not" bug the
    # preflight gate exists to prevent.  The `unknown == []` assertion above
    # separately proves the launcher really parses both.
    assert "--workload" in command
    assert "--max-model-len" in command


def test_launch_forwards_no_corpus_which_the_launcher_also_parses() -> None:
    args = cli.build_parser().parse_args(
        ["launch", "--flavor", "live-vllm", "--no-corpus", "--skip-git-checks"]
    )
    command = cli.launcher_command(args)
    assert "--no-corpus" in command
    assert "--no-corpus" in launcher_accepted_options()


# ---------------------------------------------------------------------------
# 4 -- compare exits non-zero on a regression and zero on a proven-flat pair
# ---------------------------------------------------------------------------
def test_compare_exits_non_zero_on_an_injected_regression(
    regressed_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(
        ["compare", "run-base", "run-cand", "--root", str(regressed_pair)], capsys
    )
    assert code != 0, err
    assert "PROVEN REGRESSION(S) FOUND" in out
    assert "proven_regression" in out


def test_compare_exits_zero_on_a_proven_flat_pair(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(
        ["compare", "run-base", "run-cand", "--root", str(flat_pair)], capsys
    )
    assert code == 0, err + out
    assert "PROVEN REGRESSION(S) FOUND" not in out
    assert "proven_flat" in out


# ---------------------------------------------------------------------------
# 5 -- workloads verify catches a one-byte corpus edit
# ---------------------------------------------------------------------------
def test_workloads_verify_passes_on_the_corpus_its_manifest_describes(
    workload_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(
        ["workloads", "verify", "fixture-chat", "--spec-dir", str(workload_dir)], capsys
    )
    assert code == 0, out + err
    assert out.startswith("OK    fixture-chat")


def test_workloads_verify_fails_when_one_byte_of_the_corpus_changes(
    workload_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = json.loads((workload_dir / "fixture-chat.json").read_text())
    corpus = Path(manifest["source"]["path"])
    original = corpus.read_bytes()
    # One byte, same length: the size check still passes, so only the digest can
    # catch this.  A corpus that was quietly regenerated looks exactly like it.
    altered = original.replace(b"alpha", b"alphb", 1)
    assert altered != original and len(altered) == len(original)
    corpus.write_bytes(altered)
    assert hashlib.sha256(altered).hexdigest() != manifest["source"]["sha256"]

    code, out, err = run_cli(
        ["workloads", "verify", "fixture-chat", "--spec-dir", str(workload_dir)], capsys
    )
    assert code != 0, out + err
    assert "FAIL  fixture-chat" in out
    assert "source.sha256" in out


# ---------------------------------------------------------------------------
# 6 -- --json parses, and the standard-error field is never silently omitted
# ---------------------------------------------------------------------------
def test_json_comparison_carries_standard_error_resolved_and_unresolved(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(
        ["compare", "run-base", "run-cand", "--root", str(flat_pair), "--json"], capsys
    )
    assert code == 0, err
    payload = json.loads(out)
    comparisons = [c for pair in payload["pairs"] for c in pair["comparisons"]]
    assert comparisons

    for comparison in comparisons:
        assert "delta_standard_error" in comparison, comparison["metric"]
        assert "dispersion_basis" in comparison, comparison["metric"]

    resolved = [c for c in comparisons if c["delta_standard_error"] is not None]
    unresolved = [c for c in comparisons if c["delta_standard_error"] is None]
    # Both kinds must be present, or this test would be asserting about one case
    # and claiming to cover two.
    assert resolved, "fixture produced no comparison with a real standard error"
    assert unresolved, "fixture produced no comparison lacking a standard error"
    for comparison in unresolved:
        # Explicitly null, not missing: a consumer must be able to tell "no error
        # bar" from "field not implemented".
        assert comparison["delta_standard_error"] is None
        assert comparison["dispersion_basis"] in {"degenerate", "unsampled"}
    for comparison in resolved:
        assert isinstance(comparison["delta_standard_error"], float)
        assert comparison["dispersion_basis"] == "measured"


def test_json_show_reports_standard_error_or_an_explicit_null(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(
        ["show", "run-base", "--root", str(flat_pair), "--json"], capsys
    )
    assert code == 0, err
    payload = json.loads(out)
    metrics = [
        metric
        for cell in payload["cells"]
        for arm in cell["arms"].values()
        for metric in arm.values()
    ]
    assert metrics
    assert all("standard_error" in metric for metric in metrics)
    assert any(metric["standard_error"] is None for metric in metrics)
    assert any(metric["standard_error"] is not None for metric in metrics)


# ---------------------------------------------------------------------------
# 7 -- every reporting subcommand fails on the thing it exists to detect
# ---------------------------------------------------------------------------
def test_index_of_an_empty_root_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, err = run_cli(["index", "--root", str(empty)], capsys)
    assert code != 0
    assert "no runs found" in err
    assert "runs                    0" in out


def test_index_of_a_missing_root_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(["index", "--root", str(tmp_path / "nope")], capsys)
    assert code != 0
    assert "no such run root" in err


def test_index_of_a_populated_root_exits_zero(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(["index", "--root", str(flat_pair)], capsys)
    assert code == 0, err
    assert "runs                    2" in out
    assert "runs with measurements  2" in out


def test_show_of_a_run_with_no_measurements_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    write_run(root, "run-empty", commit="c" * 40)
    code, _out, err = run_cli(["show", "run-empty", "--root", str(root)], capsys)
    assert code != 0
    assert "recorded no measurements" in err


def test_show_of_a_run_with_measurements_exits_zero(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(["show", "run-base", "--root", str(flat_pair)], capsys)
    assert code == 0, err
    assert "tok_per_sec" in out
    assert "recorded no measurements" not in err


def test_show_of_an_unknown_run_exits_non_zero(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(["show", "no-such-run", "--root", str(flat_pair)], capsys)
    assert code != 0
    assert "no indexed run named" in err


def test_compare_with_an_unknown_selector_exits_non_zero(
    flat_pair: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(
        ["compare", "run-base", "nope", "--root", str(flat_pair)], capsys
    )
    assert code != 0
    assert "neither an indexed run id nor a commit" in err


def test_workloads_verify_of_a_missing_spec_dir_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(
        ["workloads", "verify", "--spec-dir", str(tmp_path / "gone")], capsys
    )
    assert code != 0
    assert "no workload manifests" in err


# ---------------------------------------------------------------------------
# The launcher must actually EXPORT what preflight validated.
#
# Everything above proves speedbench composes an argv the launcher accepts.
# None of it proves the launcher then puts the workload into the job.  That gap
# is not hypothetical: deleting the `export SPEEDLM_E2E_WORKLOAD` line from
# make_snapshot_run.sh left the whole suite green while every job silently ran
# generic-chat at 4096 no matter what was requested -- preflight would have
# validated a configuration the job does not run, which is precisely the
# "looks configured, is not" failure the gate exists to prevent.
#
# The launcher never submits; it writes job.sbatch and prints the sbatch line.
# So this runs on a login node with no GPU and no SLURM.
# ---------------------------------------------------------------------------


def _generate_sbatch(tmp_path: Path, *args: str) -> str:
    """Run the real launcher for one flavor and return the job.sbatch text."""
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")
    run_root = tmp_path / "runs"
    run_root.mkdir()
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "--commit", commit,
            "--run-root", str(run_root),
            "--run-name", "probe",
            "--corpus", str(corpus),
            # The working tree is dirty while this suite runs, which preflight
            # WARNs about; a warning does not block, so the gate stays live.
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    sbatch = run_root / "probe" / "job.sbatch"
    assert sbatch.is_file(), (
        f"launcher produced no job.sbatch (exit {completed.returncode})\n"
        f"stdout: {completed.stdout[-2000:]}\nstderr: {completed.stderr[-2000:]}"
    )
    return sbatch.read_text(encoding="utf-8")


@pytest.mark.skipif(not LAUNCHER.is_file(), reason="launcher script absent")
def test_launcher_exports_the_workload_and_window_it_was_given(tmp_path: Path) -> None:
    text = _generate_sbatch(
        tmp_path,
        "--flavor", "config-matrix",
        "--candidate-drafter", "some/drafter",
        "--pytest-k", "eager-c1-short",
        "--workload", "generic-chat",
        "--max-model-len", "8192",
    )
    assert "export SPEEDLM_E2E_WORKLOAD=generic-chat" in text, (
        "job.sbatch does not export the workload; the cell would fall back to "
        "the default whatever preflight was told"
    )
    assert "export SPEEDLM_CONFIG_MATRIX_MAX_MODEL_LEN=8192" in text, (
        "job.sbatch does not export the context window; the cell would run at "
        "its hardcoded default and preflight's capacity check would be vacuous"
    )


@pytest.mark.skipif(not LAUNCHER.is_file(), reason="launcher script absent")
def test_launcher_defaults_reproduce_the_pre_workload_behaviour(tmp_path: Path) -> None:
    """Omitting both options must keep the historical configuration exactly."""
    text = _generate_sbatch(
        tmp_path,
        "--flavor", "config-matrix",
        "--candidate-drafter", "some/drafter",
        "--pytest-k", "eager-c1-short",
    )
    assert "export SPEEDLM_E2E_WORKLOAD=generic-chat" in text
    assert "export SPEEDLM_CONFIG_MATRIX_MAX_MODEL_LEN=4096" in text


@pytest.mark.skipif(not LAUNCHER.is_file(), reason="launcher script absent")
def test_launcher_preflight_gate_refuses_and_writes_nothing(tmp_path: Path) -> None:
    """The original silent-OOM bug: --vllm-args to a flavor that never reads it.

    Both halves.  A gate that also blocked the good configuration would be
    indistinguishable from one that blocks everything.
    """
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    run_root = tmp_path / "runs"
    run_root.mkdir()

    def launch(name: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(LAUNCHER),
                "--flavor", "model-matrix",
                "--matrix-cell", "gpt-oss-20b-eagle3",
                "--commit", commit,
                "--run-root", str(run_root),
                "--run-name", name,
                *extra,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    refused = launch("bad", "--vllm-args", '["--max-model-len","4096"]')
    assert refused.returncode != 0, "preflight let the silent-OOM configuration through"
    # The findings render on stdout (that is where the operator reads them);
    # only the launcher's own abort line goes to stderr.  Assert against both
    # so the test does not depend on which stream carries which half.
    combined = refused.stdout + refused.stderr
    assert "vllm-args-ignored" in combined, combined[-2000:]
    assert "nothing was submitted" in refused.stderr
    assert not (run_root / "bad").exists(), (
        "a refused launch left a run directory behind; a later index would "
        "report a run that was never configured"
    )

    accepted = launch("good")
    assert accepted.returncode == 0, (
        f"preflight refused a valid launch\nstderr: {accepted.stderr[-2000:]}"
    )
    assert (run_root / "good" / "job.sbatch").is_file()
