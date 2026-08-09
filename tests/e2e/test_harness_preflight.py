"""Tests for the GPU-free launch preflight.

Deliberately CI-safe: no ``e2e`` marker, no GPU, no ``/data``, no network.  The
only file outside the package these tests touch is the launcher script itself,
which is read as text by the consistency test.

Every check gets a PAIR of assertions -- a bad config that must be rejected and
a good config that must pass.  A validator that rejects everything is exactly as
useless as one that rejects nothing, and only the pair distinguishes them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.harness.preflight import (
    DEFAULT_GATEWAY_VLLM_ARGS,
    FLAVORS,
    Finding,
    GitState,
    PreflightConfig,
    Severity,
    launcher_flavors,
    parse_slurm_time,
    parse_vllm_args,
    preflight,
    render_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "make_snapshot_run.sh"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def config_for(flavor: str, **overrides: object) -> PreflightConfig:
    """A config that is valid for ``flavor`` unless an override breaks it."""
    spec = FLAVORS[flavor]
    base: dict[str, object] = {
        "flavor": flavor,
        "env": {spec.gate_var: "1"},
        "options": {option: "supplied" for option in spec.required_options},
        "git": GitState(commit_exists=True, dirty_paths=()),
        "commit": "0123456789abcdef",
    }
    base.update(overrides)
    return PreflightConfig(**base)  # type: ignore[arg-type]


def checks(findings: tuple[Finding, ...], severity: Severity) -> list[str]:
    return [f.check for f in findings if f.severity is severity]


def errors(findings: tuple[Finding, ...]) -> list[str]:
    return checks(findings, Severity.ERROR)


def warnings(findings: tuple[Finding, ...]) -> list[str]:
    return checks(findings, Severity.WARNING)


@dataclass(frozen=True)
class FakeRequirements:
    """Stands in for ``workloads.WorkloadRequirements`` (not written yet)."""

    min_max_model_len: int | None = None


@dataclass(frozen=True)
class FakeWorkload:
    """Duck-typed stand-in for a ``tests.e2e.harness.workloads`` spec."""

    name: str
    requirements: FakeRequirements


BOUNDED_ARGS = ("--max-model-len", "4096", "--gpu-memory-utilization", "0.75")


# --------------------------------------------------------------------------
# Baseline: the good config really is good
# --------------------------------------------------------------------------
def test_baseline_configs_pass_for_every_flavor() -> None:
    """No flavor's default wiring is itself an error.

    Without this, every "bad config is rejected" test below could be passing
    for the wrong reason -- because preflight rejects everything.
    """
    for name in FLAVORS:
        findings, ok = preflight(config_for(name))
        assert ok, f"{name}: {render_findings(findings)}"
        assert errors(findings) == [], name


# --------------------------------------------------------------------------
# Check 1 -- --vllm-args supplied to a flavor that never reads it
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flavor",
    ["activation-capture", "capture-overhead", "hot-swap", "model-matrix", "config-matrix"],
)
def test_vllm_args_for_non_consuming_flavor_is_error(flavor: str) -> None:
    findings, ok = preflight(config_for(flavor, vllm_args=BOUNDED_ARGS))
    assert not ok
    assert "vllm-args-ignored" in errors(findings)
    message = render_findings(findings)
    assert FLAVORS[flavor].vllm_args_var in message


@pytest.mark.parametrize(
    "flavor",
    ["idle-tuning", "live-vllm", "proxy-overhead", "token-fidelity", "capture-matrix",
     "agent-harness"],
)
def test_vllm_args_for_consuming_flavor_is_accepted(flavor: str) -> None:
    findings, ok = preflight(config_for(flavor, vllm_args=BOUNDED_ARGS))
    assert ok, render_findings(findings)
    assert "vllm-args-ignored" not in errors(findings)


def test_every_flavor_records_whether_its_test_reads_the_vllm_args_variable() -> None:
    """The 'actually reads it' column, pinned to what grepping the tests found."""
    consuming = {name for name, spec in FLAVORS.items() if spec.consumes_vllm_args}
    assert consuming == {
        "idle-tuning",
        "live-vllm",
        "proxy-overhead",
        "token-fidelity",
        "capture-matrix",
        "agent-harness",
    }


# --------------------------------------------------------------------------
# Check 2 -- required per-flavor options
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("flavor", "option"),
    [
        ("idle-tuning", "--tuning-config"),
        ("model-matrix", "--matrix-cell"),
        ("config-matrix", "--candidate-drafter"),
        ("config-matrix", "--corpus"),
        ("config-matrix", "--pytest-k"),
    ],
)
def test_missing_required_option_is_error(flavor: str, option: str) -> None:
    spec = FLAVORS[flavor]
    supplied = {o: "supplied" for o in spec.required_options if o != option}
    findings, ok = preflight(config_for(flavor, options=supplied))
    assert not ok
    assert "missing-required-option" in errors(findings)
    assert option in render_findings(findings)


def test_all_required_options_supplied_passes() -> None:
    findings, ok = preflight(
        config_for(
            "config-matrix",
            options={
                "--candidate-drafter": "RedHatAI/x.eagle3",
                "--corpus": "/corpora/ultrachat-prompts.jsonl",
                "--pytest-k": "eager-c1-short",
            },
        )
    )
    assert ok, render_findings(findings)
    assert "missing-required-option" not in errors(findings)


def test_empty_string_option_counts_as_missing() -> None:
    """``--corpus ''`` is how ``--no-corpus`` reaches the config; it is not a value."""
    findings, ok = preflight(
        config_for(
            "config-matrix",
            options={
                "--candidate-drafter": "RedHatAI/x.eagle3",
                "--corpus": "",
                "--pytest-k": "eager-c1-short",
            },
        )
    )
    assert not ok
    assert "missing-required-option" in errors(findings)


@pytest.mark.parametrize(
    ("flavor", "option"),
    [
        ("token-fidelity", "--model"),
        ("hot-swap", "--prompt-set"),
        ("live-vllm", "--runner"),
        ("activation-capture", "--max-model-len"),
    ],
)
def test_supplied_option_with_no_flavor_consumer_is_an_error(
    flavor: str, option: str
) -> None:
    """An accepted spelling must not imply a configuration the job never uses.

    Each pair above reached either a variable the selected test never reads or
    no export at all.  RED before the fix: preflight had no representation of
    which options the operator explicitly supplied, so every pair passed this
    check silently.
    """
    findings, ok = preflight(
        config_for(
            flavor,
            options={
                **{required: "supplied" for required in FLAVORS[flavor].required_options},
                option: "operator-value",
            },
            supplied_options=frozenset({option}),
        )
    )

    assert not ok
    assert "option-ignored" in errors(findings)
    assert option in render_findings(findings)


@pytest.mark.parametrize(
    ("flavor", "option", "value"),
    [
        ("activation-capture", "--target-layer-ids", "2,18,33"),
        ("activation-capture", "--hf-reference", "maybe"),
        ("capture-overhead", "--inject-ms", "1.5"),
        ("config-matrix", "--inject-percent", "nan"),
        ("config-matrix", "--max-model-len", "0"),
        ("model-matrix", "--matrix-cell", "not-a-cell"),
    ],
)
def test_launcher_option_values_that_the_consumer_cannot_use_are_errors(
    flavor: str, option: str, value: str
) -> None:
    """Advertised options must fail cheaply instead of inside an allocated test."""
    findings, ok = preflight(
        config_for(
            flavor,
            options={
                **{required: "supplied" for required in FLAVORS[flavor].required_options},
                option: value,
            },
            supplied_options=frozenset({option}),
        )
    )
    assert not ok
    assert "invalid-option" in errors(findings)
    assert option in render_findings(findings)


# --------------------------------------------------------------------------
# Check 3 -- memory bounds
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--enforce-eager",),
        ("--max-model-len", "4096"),
        ("--gpu-memory-utilization", "0.75"),
    ],
)
def test_vllm_args_without_memory_bounds_are_error(args: tuple[str, ...]) -> None:
    findings, ok = preflight(config_for("capture-matrix", vllm_args=args))
    assert not ok
    assert "unbounded-engine" in errors(findings)


def test_vllm_args_with_memory_bounds_pass() -> None:
    findings, ok = preflight(config_for("capture-matrix", vllm_args=BOUNDED_ARGS))
    assert ok, render_findings(findings)
    assert "unbounded-engine" not in errors(findings)


def test_equals_spelling_of_memory_bounds_is_accepted() -> None:
    args = ("--max-model-len=4096", "--gpu-memory-utilization=0.75")
    findings, ok = preflight(config_for("capture-matrix", vllm_args=args))
    assert ok, render_findings(findings)


def test_flavor_with_no_bounds_from_anywhere_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the launcher ever stops defaulting a flavor's args, preflight must object.

    ``capture-matrix``'s test falls back to ``os.environ.get(..., "[]")``
    (test_capture_harness_matrix.py:81): with no launcher default and no
    ``--vllm-args`` the engine would size its KV cache for the whole card.
    """
    unbounded = dataclasses.replace(
        FLAVORS["capture-matrix"], launcher_default_vllm_args=None
    )
    monkeypatch.setitem(FLAVORS, "capture-matrix", unbounded)  # type: ignore[arg-type]
    findings, ok = preflight(config_for("capture-matrix"))
    assert not ok
    assert "unbounded-engine" in errors(findings)


def test_flavor_whose_test_supplies_bounded_defaults_passes_with_no_args() -> None:
    """agent-harness builds bounded per-model defaults at test_agent_harness.py:209."""
    assert FLAVORS["agent-harness"].launcher_default_vllm_args is None
    findings, ok = preflight(config_for("agent-harness"))
    assert ok, render_findings(findings)
    assert "unbounded-engine" not in errors(findings)


# --------------------------------------------------------------------------
# Check 4 -- GDN prefill backend
# --------------------------------------------------------------------------
def test_gdn_model_without_triton_backend_warns() -> None:
    """token-fidelity hardcodes Qwen/Qwen3.5-2B and gets the gateway default args,
    which carry the memory bounds but not --gdn-prefill-backend."""
    findings, ok = preflight(config_for("token-fidelity"))
    assert ok, "the JIT cost is expensive, not wrong -- it must not block"
    assert "gdn-jit" in warnings(findings)
    assert "10m33s" in render_findings(findings)


def test_gdn_model_with_triton_backend_does_not_warn() -> None:
    args = (*DEFAULT_GATEWAY_VLLM_ARGS, "--gdn-prefill-backend", "triton")
    findings, _ = preflight(config_for("token-fidelity", vllm_args=args))
    assert "gdn-jit" not in warnings(findings)


def test_non_gdn_model_without_triton_backend_does_not_warn() -> None:
    """Qwen3-8B is not a GDN model; warning on it would be noise."""
    findings, _ = preflight(config_for("live-vllm"))
    assert FLAVORS["live-vllm"].default_model == "Qwen/Qwen3-8B"
    assert "gdn-jit" not in warnings(findings)


def test_gdn_detection_follows_the_requested_model_not_the_flavor_default() -> None:
    findings, _ = preflight(config_for("live-vllm", model="Qwen/Qwen3.5-9B"))
    assert "gdn-jit" in warnings(findings)


# --------------------------------------------------------------------------
# Check 5 -- workload compatibility
#
# The carrier flavor is ``idle-tuning`` rather than ``live-vllm``: check 5b
# (workload-ignored) now refuses a --workload handed to a flavor whose test
# never reads SPEEDLM_E2E_WORKLOAD, and live-vllm is one of those.  Both take
# --vllm-args, so the capacity check still reads the window out of the argv
# exactly as before.
# --------------------------------------------------------------------------
def test_workload_needing_more_context_than_the_launch_is_error() -> None:
    workload = FakeWorkload("long-context", FakeRequirements(min_max_model_len=8192))
    findings, ok = preflight(config_for("idle-tuning", workload=workload))
    assert not ok
    assert "workload-incompatible" in errors(findings)
    assert "8192" in render_findings(findings)


def test_workload_within_the_launch_context_passes() -> None:
    workload = FakeWorkload("short-context", FakeRequirements(min_max_model_len=4096))
    findings, ok = preflight(config_for("idle-tuning", workload=workload))
    assert ok, render_findings(findings)
    assert "workload-incompatible" not in errors(findings)


def test_workload_without_a_context_requirement_is_not_checked() -> None:
    workload = FakeWorkload("unconstrained", FakeRequirements(min_max_model_len=None))
    findings, ok = preflight(config_for("idle-tuning", workload=workload))
    assert ok, render_findings(findings)
    assert "workload-incompatible" not in errors(findings)


def test_real_workload_spec_is_refused_below_its_window_and_accepted_above() -> None:
    """The regression: a real ``WorkloadSpec`` must not slip past this check.

    ``WorkloadSpec.requirements`` is a Mapping, but preflight used to read
    ``requirements.min_max_model_len`` as an attribute, so ``getattr`` returned
    ``None`` and the capacity check passed on every launch.  Both halves are
    asserted: refusal at a too-small window AND acceptance at a large enough one,
    because a check that refuses everything proves nothing either.
    """
    workloads = pytest.importorskip("tests.e2e.harness.workloads")
    if not (Path(workloads.spec_directory()) / "agentic-tool-loop.json").is_file():
        pytest.skip("workload spec directory is not available on this host")
    spec = workloads.load_spec("agentic-tool-loop")
    assert spec.requirements["min_max_model_len"] == 18432

    too_small = ("--max-model-len", "4096", "--gpu-memory-utilization", "0.75")
    findings, ok = preflight(
        config_for("idle-tuning", workload=spec, vllm_args=too_small)
    )
    assert not ok, render_findings(findings)
    assert "workload-incompatible" in errors(findings)
    assert "18432" in render_findings(findings)

    big_enough = ("--max-model-len", "20480", "--gpu-memory-utilization", "0.75")
    findings, ok = preflight(
        config_for("idle-tuning", workload=spec, vllm_args=big_enough)
    )
    assert ok, render_findings(findings)
    assert "workload-incompatible" not in errors(findings)
    assert "workload-unreadable" not in errors(findings)


def test_workload_whose_requirements_cannot_be_read_is_an_error() -> None:
    """"I could not determine the required context window" must block."""

    class NoRequirements:
        name = "opaque"
        requirements = object()

    findings, ok = preflight(config_for("idle-tuning", workload=NoRequirements()))
    assert not ok
    assert "workload-unreadable" in errors(findings)


def test_no_workload_declared_is_not_an_error() -> None:
    findings, ok = preflight(config_for("idle-tuning", workload=None))
    assert ok
    assert "workload-incompatible" not in errors(findings)


# --------------------------------------------------------------------------
# Check 5b -- a --workload the flavor never reads
#
# The defect: `speedbench preflight --flavor idle-tuning --workload
# agentic-mixed-outcome --max-model-len 24576` answered "OK -- no findings"
# while test_live_idle_tuning.py read SPEEDLM_E2E_PROMPT_CORPUS and nothing
# else.  The capacity check above was doing its job; nobody was checking that
# the job would serve the workload at all.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flavor",
    ["live-vllm", "activation-capture", "hot-swap", "model-matrix", "token-fidelity"],
)
def test_workload_for_a_flavor_that_never_reads_it_is_error(flavor: str) -> None:
    workload = FakeWorkload("agentic", FakeRequirements(min_max_model_len=None))
    findings, ok = preflight(config_for(flavor, workload=workload))
    assert not ok
    assert "workload-ignored" in errors(findings)
    assert "SPEEDLM_E2E_WORKLOAD" in render_findings(findings)


@pytest.mark.parametrize("flavor", ["idle-tuning", "config-matrix"])
def test_workload_for_a_flavor_that_reads_it_is_accepted(flavor: str) -> None:
    workload = FakeWorkload("agentic", FakeRequirements(min_max_model_len=None))
    findings, ok = preflight(config_for(flavor, workload=workload))
    assert ok, render_findings(findings)
    assert "workload-ignored" not in errors(findings)


def test_every_flavor_records_whether_its_test_reads_the_workload_variable() -> None:
    """The column, pinned to what grepping the tests actually finds.

    Read out of the test sources rather than restated, so a flavor that grows
    (or loses) a ``SPEEDLM_E2E_WORKLOAD`` consumer cannot leave the table saying
    the opposite -- which is the exact shape of the bug this check exists for.
    """
    for name, spec in FLAVORS.items():
        source = (REPO_ROOT / spec.test_path).read_text(encoding="utf-8")
        reads_it = "SPEEDLM_E2E_WORKLOAD" in source
        assert spec.consumes_workload == reads_it, (
            f"flavor {name!r} declares consumes_workload={spec.consumes_workload} "
            f"but {spec.test_path} {'does' if reads_it else 'does not'} mention "
            "SPEEDLM_E2E_WORKLOAD"
        )
    assert {name for name, spec in FLAVORS.items() if spec.consumes_workload} == {
        "idle-tuning",
        "config-matrix",
    }


# --------------------------------------------------------------------------
# Check 5b-ii -- a --workload and a --corpus that name different traffic
#
# The defect: `speedbench preflight --flavor idle-tuning --workload
# agentic-mixed-outcome` answered "OK -- no findings" even though the launcher
# defaults --corpus, so the sbatch exported both variables and
# test_live_idle_tuning.py:229 refused the pair -- 1m48s into an allocated GPU
# (job 375376).  The pair-half of this section is the config-matrix case: it
# also reads the corpus variable, but only to record it, and its launcher arm
# makes --corpus mandatory, so refusing it there would refuse every legal
# config-matrix launch.
# --------------------------------------------------------------------------
CORPUS = "/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl"


def test_workload_together_with_a_corpus_is_error_for_idle_tuning() -> None:
    workload = FakeWorkload("agentic", FakeRequirements(min_max_model_len=None))
    findings, ok = preflight(
        config_for(
            "idle-tuning",
            workload=workload,
            options={"--tuning-config": "supplied", "--corpus": CORPUS},
        )
    )
    assert not ok
    assert "workload-corpus-collision" in errors(findings)
    rendered = render_findings(findings)
    assert "--no-corpus" in rendered, (
        "the finding must name the flag that resolves it; the operator cannot "
        "act on 'these two collide' alone"
    )


def test_workload_without_a_corpus_is_accepted_for_idle_tuning() -> None:
    """The other half: dropping the corpus is what makes the launch legal."""
    workload = FakeWorkload("agentic", FakeRequirements(min_max_model_len=None))
    findings, ok = preflight(config_for("idle-tuning", workload=workload))
    assert ok, render_findings(findings)
    assert "workload-corpus-collision" not in errors(findings)


def test_a_corpus_without_a_workload_is_not_a_collision() -> None:
    """The historical idle-tuning launch -- corpus only -- must stay clean."""
    findings, ok = preflight(
        config_for(
            "idle-tuning",
            options={"--tuning-config": "supplied", "--corpus": CORPUS},
        )
    )
    assert ok, render_findings(findings)
    assert "workload-corpus-collision" not in errors(findings)


def test_workload_together_with_a_corpus_is_accepted_for_config_matrix() -> None:
    """config-matrix records the corpus and seeds from the workload regardless.

    ``config_for`` already supplies ``--corpus`` because the flavor requires it,
    which is the point: make_snapshot_run.sh:503 refuses ``--no-corpus`` here,
    so a check scoped on ``consumes_workload`` would make every config-matrix
    launch that named a workload unlaunchable by either gate.
    """
    workload = FakeWorkload("agentic", FakeRequirements(min_max_model_len=None))
    config = config_for("config-matrix", workload=workload)
    assert config.options.get("--corpus"), "the fixture stopped supplying --corpus"
    findings, ok = preflight(config)
    assert ok, render_findings(findings)
    assert "workload-corpus-collision" not in errors(findings)


def test_every_flavor_records_whether_its_test_refuses_a_workload_corpus_pair() -> None:
    """The column, pinned to the refusal the tests actually contain.

    Read out of the test sources for the same reason as the workload column:
    the whole failure mode is a table that says something the test does not.
    The predicate is the refusal itself, not a mention of the variable --
    config-matrix mentions it and does not refuse.
    """
    refusal = 'assert not os.environ.get("SPEEDLM_E2E_PROMPT_CORPUS")'
    for name, spec in FLAVORS.items():
        source = (REPO_ROOT / spec.test_path).read_text(encoding="utf-8")
        refuses = refusal in source
        assert spec.corpus_collides_with_workload == refuses, (
            f"flavor {name!r} declares corpus_collides_with_workload="
            f"{spec.corpus_collides_with_workload} but {spec.test_path} "
            f"{'does' if refuses else 'does not'} refuse a workload/corpus pair"
        )
    assert {
        name for name, spec in FLAVORS.items() if spec.corpus_collides_with_workload
    } == {"idle-tuning"}
    # A flavor cannot collide over a variable it never selects a workload from.
    for name, spec in FLAVORS.items():
        if spec.corpus_collides_with_workload:
            assert spec.consumes_workload, (
                f"flavor {name!r} claims a workload/corpus collision but does "
                "not consume a workload at all"
            )


# --------------------------------------------------------------------------
# Check 5c -- --max-model-len that contradicts the engine argv
# --------------------------------------------------------------------------
def test_max_model_len_that_contradicts_the_vllm_args_is_error() -> None:
    """The option is a stand-in for the argv, so it may not disagree with one.

    ``--flavor idle-tuning --max-model-len 24576`` with the launcher's default
    vLLM args means the capacity check validates 24576 and the engine starts at
    4096.
    """
    findings, ok = preflight(config_for("idle-tuning", max_model_len=24576))
    assert not ok
    assert "max-model-len-disagreement" in errors(findings)
    assert "4096" in render_findings(findings)


def test_max_model_len_agreeing_with_the_vllm_args_passes() -> None:
    findings, ok = preflight(
        config_for(
            "idle-tuning",
            max_model_len=24576,
            vllm_args=("--max-model-len", "24576", "--gpu-memory-utilization", "0.75"),
        )
    )
    assert ok, render_findings(findings)
    assert "max-model-len-disagreement" not in errors(findings)


def test_max_model_len_for_a_flavor_with_no_argv_is_not_second_guessed() -> None:
    """config-matrix builds its own engine argv; the option is the only source."""
    findings, ok = preflight(config_for("config-matrix", max_model_len=24576))
    assert ok, render_findings(findings)
    assert "max-model-len-disagreement" not in errors(findings)


# --------------------------------------------------------------------------
# Check 6 -- timeout sanity
# --------------------------------------------------------------------------
def test_timeout_far_shorter_than_wall_clock_is_error() -> None:
    """The 'killed a healthy job with eight hours left' failure."""
    findings, ok = preflight(
        config_for("idle-tuning", slurm_time="12:00:00", timeouts={"tuning": 14400})
    )
    assert not ok
    assert "timeout" in errors(findings)
    assert "28800s of paid wall clock left" in render_findings(findings)


def test_timeout_inside_the_launcher_margin_passes() -> None:
    """make_snapshot_run.sh:544-550 defaults the tuning timeout to wall minus 900s."""
    findings, ok = preflight(
        config_for("idle-tuning", slurm_time="12:00:00", timeouts={"tuning": 43200 - 900})
    )
    assert ok, render_findings(findings)
    assert "timeout" not in errors(findings)


def test_timeout_at_least_as_long_as_wall_clock_is_error() -> None:
    """The launcher rejects this outright; CLI preflight must not call it safe."""
    findings, ok = preflight(
        config_for("idle-tuning", slurm_time="02:00:00", timeouts={"tuning": 7200})
    )
    assert not ok
    assert "timeout" in errors(findings)
    assert "can never fire" in render_findings(findings)


def test_timeout_is_compared_against_the_flavor_default_time_when_time_is_omitted() -> None:
    """idle-tuning's default is 12:00:00; a 1h timeout abandons 11h of it."""
    findings, ok = preflight(config_for("idle-tuning", timeouts={"tuning": 3600}))
    assert not ok
    assert "timeout" in errors(findings)


def test_unparseable_slurm_time_is_error() -> None:
    findings, ok = preflight(config_for("live-vllm", slurm_time="90:00"))
    assert not ok
    assert "timeout" in errors(findings)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("00:15:00", 900), ("12:00:00", 43200), ("1-00:00:00", 86400), ("2-03:04:05", 183845)],
)
def test_parse_slurm_time_accepts_both_launcher_formats(value: str, seconds: int) -> None:
    assert parse_slurm_time(value) == seconds


@pytest.mark.parametrize("value", ["90:00", "01:60:00", "01:00:60", "", "abc"])
def test_parse_slurm_time_rejects_what_the_launcher_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        parse_slurm_time(value)


# --------------------------------------------------------------------------
# Check 7 -- silent skip / silent green
# --------------------------------------------------------------------------
def test_unset_gate_variable_is_error() -> None:
    findings, ok = preflight(config_for("live-vllm", env={}))
    assert not ok
    assert "silent-skip" in errors(findings)


@pytest.mark.parametrize("value", ["true", "yes", "0", "", "1 "])
def test_truthy_looking_gate_value_that_is_not_exactly_one_is_error(value: str) -> None:
    """The tests compare against the literal string '1'; everything else skips."""
    findings, ok = preflight(config_for("live-vllm", env={"SPEEDLM_E2E": value}))
    assert not ok
    assert "silent-skip" in errors(findings)


def test_gate_variable_set_to_one_passes() -> None:
    findings, ok = preflight(config_for("live-vllm", env={"SPEEDLM_E2E": "1"}))
    assert ok, render_findings(findings)
    assert "silent-skip" not in errors(findings)


def test_each_flavor_is_gated_on_its_own_variable() -> None:
    """Setting some other flavor's gate does not unlock this one."""
    findings, ok = preflight(config_for("idle-tuning", env={"SPEEDLM_E2E": "1"}))
    assert not ok
    assert "silent-skip" in errors(findings)
    assert "SPEEDLM_E2E_IDLE_TUNING" in render_findings(findings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEEDLM_E2E_ALLOW_UNMEASURED_GATE", "1"),
        ("SPEEDLM_E2E_STRICT_VERDICT", "0"),
        ("SPEEDLM_E2E_HF_REFERENCE", "0"),
    ],
)
def test_result_relaxing_flag_is_error(name: str, value: str) -> None:
    env = {"SPEEDLM_E2E": "1", name: value}
    findings, ok = preflight(config_for("live-vllm", env=env))
    assert not ok
    assert "silent-green" in errors(findings)
    assert name in render_findings(findings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEEDLM_E2E_ALLOW_UNMEASURED_GATE", "0"),
        ("SPEEDLM_E2E_STRICT_VERDICT", "1"),
        ("SPEEDLM_E2E_HF_REFERENCE", "1"),
    ],
)
def test_result_relaxing_flag_at_its_strict_value_passes(name: str, value: str) -> None:
    env = {"SPEEDLM_E2E": "1", name: value}
    findings, ok = preflight(config_for("live-vllm", env=env))
    assert ok, render_findings(findings)
    assert "silent-green" not in errors(findings)


# --------------------------------------------------------------------------
# Check 8 -- provenance
# --------------------------------------------------------------------------
def test_missing_commit_is_error() -> None:
    findings, ok = preflight(
        config_for("live-vllm", commit="nope", git=GitState(commit_exists=False))
    )
    assert not ok
    assert "provenance" in errors(findings)


def test_existing_commit_passes() -> None:
    findings, ok = preflight(config_for("live-vllm", git=GitState(commit_exists=True)))
    assert ok, render_findings(findings)
    assert "provenance" not in errors(findings)


@pytest.mark.parametrize(
    "path",
    [
        "src/speedlm/gate/decide.py",
        "tests/e2e/test_live_vllm.py",
        "tests/e2e/harness/preflight.py",
        "scripts/make_snapshot_run.sh",
    ],
)
def test_uncommitted_changes_to_code_the_run_executes_warn(path: str) -> None:
    findings, ok = preflight(
        config_for("live-vllm", git=GitState(commit_exists=True, dirty_paths=(path,)))
    )
    assert ok, "dirtiness does not invalidate the launch, it invalidates the attribution"
    assert "provenance" in warnings(findings)
    assert path in render_findings(findings)


@pytest.mark.parametrize(
    "path",
    ["docs/e2e-harness.md", "README.md", "tests/e2e/test_model_matrix.py"],
)
def test_uncommitted_changes_elsewhere_do_not_warn(path: str) -> None:
    """live-vllm does not execute another flavor's test file or the docs."""
    findings, _ = preflight(
        config_for("live-vllm", git=GitState(commit_exists=True, dirty_paths=(path,)))
    )
    assert "provenance" not in warnings(findings)


# --------------------------------------------------------------------------
# Flavor table vs. the launcher it describes
# --------------------------------------------------------------------------
def test_flavor_table_matches_the_launcher_case_block() -> None:
    """Parsed from the script, so the table cannot silently rot."""
    assert launcher_flavors(LAUNCHER) == frozenset(FLAVORS)


def test_launcher_parser_finds_a_plausible_number_of_flavors() -> None:
    """Guards the parser itself: a regex that matched nothing would make the
    consistency test above pass only when the table is also empty."""
    found = launcher_flavors(LAUNCHER)
    assert len(found) >= 11
    assert "idle-tuning" in found
    assert "*" not in found


def test_every_flavor_names_a_test_file_that_exists() -> None:
    for name, spec in FLAVORS.items():
        assert (REPO_ROOT / spec.test_path).is_file(), f"{name}: {spec.test_path}"


def test_flavor_keys_match_their_own_name_field() -> None:
    for key, spec in FLAVORS.items():
        assert key == spec.name


# --------------------------------------------------------------------------
# Reporting surface
# --------------------------------------------------------------------------
def test_render_findings_puts_blocking_findings_first() -> None:
    env = {"SPEEDLM_E2E": "1", "SPEEDLM_E2E_STRICT_VERDICT": "0"}
    findings, ok = preflight(config_for("token-fidelity", env=env))
    assert not ok
    rendered = render_findings(findings).splitlines()
    assert rendered[0].startswith("ERROR")
    assert any(line.startswith("WARNING") for line in rendered)
    error_index = max(i for i, line in enumerate(rendered) if line.startswith("ERROR"))
    warning_index = min(i for i, line in enumerate(rendered) if line.startswith("WARNING"))
    assert error_index < warning_index
    assert rendered[-1].endswith("DO NOT LAUNCH.")


def test_render_findings_on_a_clean_config_says_so() -> None:
    findings, ok = preflight(config_for("live-vllm"))
    assert ok
    assert render_findings(findings) == "preflight: OK -- no findings."


def test_warnings_alone_do_not_block() -> None:
    findings, ok = preflight(config_for("token-fidelity"))
    assert ok
    assert warnings(findings) == ["gdn-jit"]
    assert render_findings(findings).endswith("launch is OK.")


def test_unknown_flavor_is_error() -> None:
    findings, ok = preflight(PreflightConfig(flavor="not-a-flavor"))
    assert not ok
    assert errors(findings) == ["unknown-flavor"]


def test_ok_is_false_exactly_when_an_error_is_present() -> None:
    for cfg in (
        config_for("live-vllm"),
        config_for("token-fidelity"),
        config_for("live-vllm", env={}),
        config_for("model-matrix", vllm_args=BOUNDED_ARGS),
    ):
        findings, ok = preflight(cfg)
        assert ok == (not any(f.blocking for f in findings)), cfg.flavor


# --------------------------------------------------------------------------
# --vllm-args parsing
# --------------------------------------------------------------------------
def test_parse_vllm_args_accepts_a_json_string_array() -> None:
    assert parse_vllm_args('["--max-model-len", "4096"]') == ("--max-model-len", "4096")


@pytest.mark.parametrize("raw", ["not json", "{}", '"a"', "[1, 2]", '["a", 2]'])
def test_parse_vllm_args_rejects_what_the_tests_reject(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_vllm_args(raw)
