"""GPU-free preflight validation for ``scripts/make_snapshot_run.sh`` runs.

Every check here exists because a real allocation was already burned on the
misconfiguration it detects:

* a flavor that silently drops its ``--vllm-args`` (the launcher exports the
  variable unconditionally, but only some tests read it) OOMed four minutes in;
* a hardcoded pytest timeout would have killed a healthy job with eight hours of
  wall clock still paid for;
* a silently skipped test consumed a whole allocation while reporting success;
* ``--gdn-prefill-backend triton`` missing from a Qwen3.5 launch cost 10m33s of
  silent FlashInfer JIT compilation on each of four H100 runs.

The module is import-light and GPU-free on purpose: it runs on a login node,
before ``sbatch``, so the classes of error above fail on a laptop instead of on
an H100.

The flavor table below is the load-bearing part.  For each launcher flavor it
records which vLLM-args variable the launcher would export **and whether the
test on the other end actually reads it** -- those are different questions, and
the gap between them is the silent-miss bug.  ``tests/e2e/test_harness_preflight
.py`` asserts the table's flavor names match the launcher's ``case "$flavor"``
arms exactly, so the table cannot rot when the launcher changes.

Workload integration point
--------------------------
:func:`preflight` accepts an optional ``workload`` object and only ever touches
``workload.name`` and the ``min_max_model_len`` entry of
``workload.requirements``.  That requirements object may be a Mapping (which is
what ``workloads.WorkloadSpec.requirements`` actually is) or a plain attribute
holder; :func:`_declared_context_window` reads both.  It used to read only the
attribute form, so handing it a real ``WorkloadSpec`` made the capacity check
return ``None`` and pass silently on every launch -- a check that cannot fail is
this project's recurring defect, so a requirements object that cannot be read at
all is now an ERROR rather than a quiet success.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "COMMON_LAUNCHER_OPTIONS",
    "FLAVORS",
    "FLAVOR_OPTION_CONSUMERS",
    "Finding",
    "Flavor",
    "GitState",
    "PreflightConfig",
    "Severity",
    "launcher_flavors",
    "parse_slurm_time",
    "parse_vllm_args",
    "preflight",
    "render_findings",
]


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------
class Severity(Enum):
    """How bad a finding is.

    Only :attr:`ERROR` blocks a launch.  A :attr:`WARNING` still costs real
    money (the JIT warning is worth ~10 minutes of H100 time per run) but does
    not make the measurement wrong, so it must not be able to hide an error.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing preflight established about a proposed run."""

    severity: Severity
    check: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.ERROR


# --------------------------------------------------------------------------
# Flavor table
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Flavor:
    """What one ``--flavor`` of the launcher actually wires up.

    ``vllm_args_var`` is what the launcher exports; ``consumes_vllm_args`` is
    whether the test reads it.  Keeping them separate is the whole point: the
    launcher's ``if [[ -n "$vllm_args" ]]; then echo "export $vllm_args_var=..."``
    at make_snapshot_run.sh:863-865 runs for *every* flavor, including the five
    whose tests never look at the variable.
    """

    name: str
    test_path: str
    #: Whether the test reads :data:`WORKLOAD_VAR`.  Same separation as
    #: ``consumes_vllm_args`` and for the same reason: ``--workload`` was a
    #: launcher option and a preflight input long before any flavor but
    #: config-matrix looked at it, so an operator could ask for agentic traffic,
    #: get a clean preflight, and receive generic chat.
    consumes_workload: bool
    #: Where that was verified (or why the answer is "no").
    workload_evidence: str
    #: Whether a ``--workload`` and a ``--corpus`` in the same launch are a
    #: *contradiction* for this flavor -- i.e. the test treats the two variables
    #: as rival seed sources and refuses the pair -- rather than merely both
    #: being read.  Deliberately not "does the test read CORPUS_VAR": both
    #: workload-consuming flavors read it, and only one of them collides.
    corpus_collides_with_workload: bool
    #: Where that was verified (or why the answer is "no").
    corpus_collision_evidence: str
    #: Environment variable that gates the test; unset means "skip silently".
    gate_var: str
    #: Which venv runs it.  Not a free choice: the capture/hot-swap tests import
    #: torch and safetensors, which only exist in the vLLM venv (which in turn
    #: has no pytest, hence /data/ryan.kim/pylibs/pytest on PYTHONPATH).
    interpreter: str
    artifact_var: str
    #: Variable the launcher would export ``--vllm-args`` into.
    vllm_args_var: str
    #: Whether the test actually reads :attr:`vllm_args_var`.  Verified by
    #: grepping each test file, not inferred from the launcher.
    consumes_vllm_args: bool
    #: Where that was verified (or why the answer is "no").
    vllm_args_evidence: str
    #: vLLM args the launcher substitutes when ``--vllm-args`` is omitted.
    launcher_default_vllm_args: tuple[str, ...] | None
    #: Whether the test's own fallback argv carries memory bounds when the
    #: variable is absent.  ``False`` means "unbounded engine, whole-card KV".
    test_default_is_bounded: bool
    #: Launcher options the flavor hard-requires.
    required_options: tuple[str, ...]
    #: ``#SBATCH --time`` the launcher picks when ``--time`` is omitted.
    default_time: str
    #: Model the run serves when ``--model`` is not given; ``None`` when the
    #: model is not knowable from launcher arguments alone.
    default_model: str | None


#: make_snapshot_run.sh:69 -- the shared gateway default.  Note it carries the
#: memory bounds but NOT --gdn-prefill-backend, which is why check 4 fires on
#: token-fidelity (whose model is a hardcoded Qwen3.5) under stock settings.
DEFAULT_GATEWAY_VLLM_ARGS: tuple[str, ...] = (
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.75",
    "--enforce-eager",
)

_NO_LAUNCHER_ARGS = "test spawns its own engine with hardcoded argv"

#: The one variable the launcher exports ``--workload`` into.
WORKLOAD_VAR = "SPEEDLM_E2E_WORKLOAD"

#: The one the launcher exports ``--corpus`` into (make_snapshot_run.sh:808,913).
CORPUS_VAR = "SPEEDLM_E2E_PROMPT_CORPUS"

#: make_snapshot_run.sh:61 and the per-flavor `interpreter=` assignments.
_PROJECT_VENV_PYTHON = "/admin/home/ryan.kim/speedlm-fr/.venv/bin/python"
_VLLM_VENV_PYTHON = "/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python"

FLAVORS: Mapping[str, Flavor] = {
    "idle-tuning": Flavor(
        name="idle-tuning",
        test_path="tests/e2e/test_live_idle_tuning.py",
        consumes_workload=True,
        workload_evidence="test_live_idle_tuning.py::_selected_workload reads SPEEDLM_E2E_WORKLOAD",
        corpus_collides_with_workload=True,
        corpus_collision_evidence=(
            "test_live_idle_tuning.py:229 asserts not "
            "os.environ.get(CORPUS_VAR) once a workload is "
            "selected -- the two are rival seed sources and the pair is refused"
        ),
        gate_var="SPEEDLM_E2E_IDLE_TUNING",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_live_idle_tuning.py:274 reads SPEEDLM_E2E_VLLM_ARGS",
        launcher_default_vllm_args=DEFAULT_GATEWAY_VLLM_ARGS,
        test_default_is_bounded=False,  # os.environ.get(..., "[]")
        required_options=("--tuning-config",),
        default_time="12:00:00",
        default_model=None,  # comes from the tuning config, not the launcher
    ),
    "activation-capture": Flavor(
        name="activation-capture",
        test_path="tests/e2e/test_serving_activation_capture.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_serving_activation_capture.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E_ACTIVATION_CAPTURE",
        interpreter=_VLLM_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=False,
        vllm_args_evidence=(
            "no SPEEDLM_E2E_VLLM_ARGS read anywhere in "
            f"test_serving_activation_capture.py; {_NO_LAUNCHER_ARGS}"
        ),
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,
        required_options=(),
        default_time="03:00:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "capture-overhead": Flavor(
        name="capture-overhead",
        test_path="tests/e2e/test_serving_activation_capture_overhead.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_serving_activation_capture_overhead.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E_CAPTURE_OVERHEAD",
        interpreter=_VLLM_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=False,
        vllm_args_evidence=(
            "no SPEEDLM_E2E_VLLM_ARGS read anywhere in "
            f"test_serving_activation_capture_overhead.py; {_NO_LAUNCHER_ARGS} "
            "(make_snapshot_run.sh:344-347 says so explicitly)"
        ),
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,
        required_options=(),
        default_time="02:00:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "hot-swap": Flavor(
        name="hot-swap",
        test_path="tests/e2e/test_serving_draft_hot_swap.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_serving_draft_hot_swap.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E_DRAFT_HOT_SWAP",
        interpreter=_VLLM_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=False,
        vllm_args_evidence=(
            "no SPEEDLM_E2E_VLLM_ARGS read anywhere in "
            f"test_serving_draft_hot_swap.py; {_NO_LAUNCHER_ARGS}"
        ),
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,
        required_options=(),
        default_time="02:00:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "live-vllm": Flavor(
        name="live-vllm",
        test_path="tests/e2e/test_live_vllm.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_live_vllm.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_live_vllm.py:371 reads SPEEDLM_E2E_VLLM_ARGS",
        launcher_default_vllm_args=DEFAULT_GATEWAY_VLLM_ARGS,
        test_default_is_bounded=False,  # json.loads(os.environ.get(..., "[]"))
        required_options=(),
        default_time="01:30:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "proxy-overhead": Flavor(
        name="proxy-overhead",
        test_path="tests/e2e/test_proxy_overhead.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_proxy_overhead.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_proxy_overhead.py:607 reads SPEEDLM_E2E_VLLM_ARGS",
        launcher_default_vllm_args=DEFAULT_GATEWAY_VLLM_ARGS,
        test_default_is_bounded=False,  # json.loads(os.environ.get(..., "[]"))
        required_options=(),
        default_time="02:00:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "token-fidelity": Flavor(
        name="token-fidelity",
        test_path="tests/e2e/test_token_fidelity.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_token_fidelity.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_E2E_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_token_fidelity.py:245 reads SPEEDLM_E2E_VLLM_ARGS",
        launcher_default_vllm_args=DEFAULT_GATEWAY_VLLM_ARGS,
        test_default_is_bounded=True,  # :246-253 falls back to bounded argv
        required_options=(),
        # test_token_fidelity.py:56 hardcodes MODEL with no env override.
        default_time="01:30:00",
        default_model="Qwen/Qwen3.5-2B",
    ),
    "model-matrix": Flavor(
        name="model-matrix",
        test_path="tests/e2e/test_model_matrix.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_model_matrix.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_MATRIX_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=False,
        vllm_args_evidence=(
            "no SPEEDLM_E2E_VLLM_ARGS read anywhere in test_model_matrix.py; each "
            "MATRIX cell carries its own vllm_args (test_model_matrix.py:98-132)"
        ),
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,
        required_options=("--matrix-cell",),
        default_time="01:30:00",
        default_model=None,  # the cell pins the model; --model is not consulted
    ),
    "capture-matrix": Flavor(
        name="capture-matrix",
        test_path="tests/e2e/test_capture_harness_matrix.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_capture_harness_matrix.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_CAPTURE_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_CAPTURE_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_capture_harness_matrix.py:81 reads SPEEDLM_CAPTURE_VLLM_ARGS",
        launcher_default_vllm_args=DEFAULT_GATEWAY_VLLM_ARGS,
        test_default_is_bounded=False,  # os.environ.get(..., "[]") -> whole-card KV
        required_options=(),
        default_time="01:30:00",
        default_model="Qwen/Qwen3-8B",
    ),
    "agent-harness": Flavor(
        name="agent-harness",
        test_path="tests/e2e/test_agent_harness.py",
        consumes_workload=False,
        workload_evidence=(
            "no SPEEDLM_E2E_WORKLOAD read anywhere in "
            "test_agent_harness.py"
        ),
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "flavor never reads SPEEDLM_E2E_WORKLOAD, so no workload/corpus "
            "pair can arise; make_snapshot_run.sh blanks corpus for it"
        ),
        gate_var="SPEEDLM_E2E",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_AGENT_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_AGENT_VLLM_ARGS",
        consumes_vllm_args=True,
        vllm_args_evidence="test_agent_harness.py:223 reads SPEEDLM_AGENT_VLLM_ARGS",
        # make_snapshot_run.sh:840-848 deliberately does NOT default this one.
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,  # _default_vllm_args() at :209-219
        required_options=(),
        default_time="01:30:00",
        default_model="openai/gpt-oss-20b",
    ),
    "config-matrix": Flavor(
        name="config-matrix",
        test_path="tests/e2e/test_inference_configuration_matrix.py",
        consumes_workload=True,
        workload_evidence="test_inference_configuration_matrix.py:189 reads SPEEDLM_E2E_WORKLOAD",
        corpus_collides_with_workload=False,
        corpus_collision_evidence=(
            "test_inference_configuration_matrix.py:185-188 reads "
            "SPEEDLM_E2E_PROMPT_CORPUS only into "
            "LiveConfiguration.legacy_prompt_corpus, which is recorded in the "
            "manifest and never seeds anything; the workload manifest is the "
            "sole prompt source, and make_snapshot_run.sh:503 makes --corpus "
            "mandatory here, so the pair is the normal case rather than a "
            "contradiction"
        ),
        gate_var="SPEEDLM_E2E_CONFIG_MATRIX",
        interpreter=_PROJECT_VENV_PYTHON,
        artifact_var="SPEEDLM_CONFIG_MATRIX_ARTIFACT_DIR",
        vllm_args_var="SPEEDLM_E2E_VLLM_ARGS",
        consumes_vllm_args=False,
        vllm_args_evidence=(
            "no SPEEDLM_E2E_VLLM_ARGS read anywhere in "
            "test_inference_configuration_matrix.py; _engine_argv() at :269-295 "
            "builds each cell's argv itself"
        ),
        launcher_default_vllm_args=None,
        test_default_is_bounded=True,  # :251-259 asserts its own bounds
        required_options=("--candidate-drafter", "--corpus", "--pytest-k"),
        default_time="12:00:00",
        default_model="Qwen/Qwen3-8B",
    ),
}


#: Flavor-scoped launcher options and the flavors whose generated job or test
#: actually consumes the supplied value.  Infrastructure options such as
#: ``--commit``, ``--run-root``, ``--time`` and ``--hf-home`` are deliberately
#: absent: the launcher itself consumes them for every flavor.
#:
#: WHY a positive map rather than a growing list of known-bad pairs: an option
#: parser is shared by all eleven flavors, so every newly added flavor starts by
#: accepting every spelling.  The failure mode is therefore omission -- the new
#: arm forgets to export a value, or exports a variable its test never reads.
#: Listing consumers makes omission fail closed.  ``--vllm-args`` and
#: ``--workload`` retain their older, more detailed checks below, but live here
#: too so this table remains the exhaustive option-to-consumer audit.
FLAVOR_OPTION_CONSUMERS: Mapping[str, frozenset[str]] = {
    "--tuning-timeout": frozenset({"idle-tuning"}),
    "--tuning-config": frozenset({"idle-tuning"}),
    "--tuning-profile": frozenset({"idle-tuning"}),
    "--verifier": frozenset(
        {"activation-capture", "capture-overhead", "hot-swap"}
    ),
    "--drafter": frozenset(
        {"activation-capture", "capture-overhead", "hot-swap", "config-matrix"}
    ),
    "--candidate-drafter": frozenset({"config-matrix"}),
    "--drafter-dir": frozenset({"activation-capture", "hot-swap"}),
    "--inject-ms": frozenset({"capture-overhead"}),
    "--inject-percent": frozenset({"config-matrix"}),
    "--runner": frozenset({"activation-capture", "hot-swap"}),
    "--prompt-set": frozenset({"activation-capture"}),
    "--target-layer-ids": frozenset({"activation-capture"}),
    "--hf-reference": frozenset({"activation-capture"}),
    "--strict-verdict": frozenset({"activation-capture"}),
    "--corpus": frozenset({"idle-tuning", "config-matrix"}),
    "--no-corpus": frozenset({"idle-tuning"}),
    "--model": frozenset(
        {"live-vllm", "proxy-overhead", "capture-matrix", "agent-harness", "config-matrix"}
    ),
    "--matrix-cell": frozenset({"model-matrix"}),
    "--vllm-args": frozenset(
        {
            "idle-tuning",
            "live-vllm",
            "proxy-overhead",
            "token-fidelity",
            "capture-matrix",
            "agent-harness",
        }
    ),
    "--workload": frozenset({"idle-tuning", "config-matrix"}),
    "--max-model-len": frozenset({"idle-tuning", "config-matrix"}),
}

#: Options whose value the launcher infrastructure consumes for every flavor.
#: Together with :data:`FLAVOR_OPTION_CONSUMERS`, this partitions the launcher's
#: public option surface.  The completeness test parses the shell's real case
#: arms, so adding an option without deciding which side it belongs to goes red.
COMMON_LAUNCHER_OPTIONS: frozenset[str] = frozenset(
    {
        "--flavor",
        "--commit",
        "--run-name",
        "--run-root",
        "--time",
        "--partition",
        "--cpus",
        "--pytest-k",
        "--hf-home",
        "--force",
        "--skip-preflight",
    }
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@runtime_checkable
class WorkloadLike(Protocol):
    """The only shape of a workload spec this module depends on.

    ``tests.e2e.harness.workloads`` owns the real type; preflight deliberately
    reaches for nothing beyond ``name`` and the ``min_max_model_len`` entry of
    ``requirements``.  ``requirements`` is typed ``Any`` because both shapes are
    accepted: a Mapping (``WorkloadSpec.requirements``) and an attribute holder.
    """

    name: str
    requirements: Any


@dataclass(frozen=True, slots=True)
class GitState:
    """Facts about the repo, injected rather than shelled out for.

    ``dirty_paths`` are repo-relative paths with uncommitted changes -- the
    launcher ``git archive``s the COMMIT, so anything listed here is *absent*
    from the snapshot the job will actually execute.
    """

    commit_exists: bool = True
    dirty_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreflightConfig:
    """A proposed ``make_snapshot_run.sh`` invocation, as data."""

    flavor: str
    #: Parsed ``--vllm-args``; ``None`` means the option was not passed.
    vllm_args: tuple[str, ...] | None = None
    #: Launcher options that were supplied, e.g. ``{"--matrix-cell": "..."}``.
    options: Mapping[str, str] = field(default_factory=dict)
    #: Spellings the operator explicitly supplied.  ``options`` also contains
    #: modeled launcher defaults (notably ``--corpus``), and treating a default
    #: as an ignored user request would turn the consumer check into noise.
    supplied_options: frozenset[str] = frozenset()
    model: str | None = None
    #: ``--time``; ``None`` means the flavor's default is used.
    slurm_time: str | None = None
    #: Any in-test / pytest deadline, label -> seconds.
    timeouts: Mapping[str, int] = field(default_factory=dict)
    #: Environment the sbatch will export.
    env: Mapping[str, str] = field(default_factory=dict)
    workload: WorkloadLike | None = None
    #: Context window the run will be configured with, when it is not
    #: recoverable from ``--vllm-args``.  ``None`` means "read it from the argv".
    max_model_len: int | str | None = None
    commit: str | None = None
    git: GitState = field(default_factory=GitState)


# --------------------------------------------------------------------------
# Small parsers
# --------------------------------------------------------------------------
def parse_vllm_args(raw: str) -> tuple[str, ...]:
    """Parse a ``--vllm-args`` JSON array the same way the tests do."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--vllm-args is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--vllm-args must be a JSON array of strings")
    return tuple(parsed)


_SLURM_TIME_DHMS = re.compile(r"^(\d+)-(\d+):(\d{2}):(\d{2})$")
_SLURM_TIME_HMS = re.compile(r"^(\d+):(\d{2}):(\d{2})$")


def parse_slurm_time(value: str) -> int:
    """Normalize ``HH:MM:SS`` / ``D-HH:MM:SS`` to seconds (launcher:520-540)."""
    match = _SLURM_TIME_DHMS.match(value)
    if match:
        days, hours, minutes, seconds = (int(g) for g in match.groups())
    else:
        match = _SLURM_TIME_HMS.match(value)
        if not match:
            raise ValueError(f"invalid --time {value!r} (want HH:MM:SS or D-HH:MM:SS)")
        days = 0
        hours, minutes, seconds = (int(g) for g in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid --time {value!r} (minutes and seconds must be < 60)")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _option_value(args: Sequence[str], name: str) -> str | None:
    """Value of ``name`` in an argv, supporting both ``--k v`` and ``--k=v``."""
    for index, item in enumerate(args):
        if item == name:
            return args[index + 1] if index + 1 < len(args) else None
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def launcher_flavors(script_path: str | Path) -> frozenset[str]:
    """Flavor names the launcher's per-flavor ``case`` block actually handles.

    Parsed from the script rather than restated, so the table above cannot
    silently disagree with the thing it describes.
    """
    text = Path(script_path).read_text()
    start = text.index('case "$flavor" in')
    block = text[start:]
    end = block.index("\nesac")
    names: set[str] = set()
    for line in block[:end].splitlines():
        match = re.match(r"^ {4}([a-z0-9|.-]+)\)", line)
        if match:
            names.update(match.group(1).split("|"))
    return frozenset(names)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
MEMORY_BOUND_FLAGS: tuple[str, ...] = ("--max-model-len", "--gpu-memory-utilization")

_MODEL_MATRIX_CELLS: frozenset[str] = frozenset(
    {
        "gpt-oss-20b-eagle3",
        "qwen3.5-9b-mtp",
        "qwen3.5-2b-none",
        "qwen3.5-2b-ngram",
    }
)

#: Flags that, set to these values, turn a real measurement into a green
#: report about nothing.  Every one of them is a legitimate debugging aid and
#: an illegitimate thing to spend an H100 allocation on.
SILENT_GREEN_FLAGS: Mapping[str, str] = {
    "SPEEDLM_E2E_ALLOW_UNMEASURED_GATE": "1",
    "SPEEDLM_E2E_STRICT_VERDICT": "0",
    "SPEEDLM_E2E_HF_REFERENCE": "0",
}

#: A deadline this far below the allocation abandons paid wall clock.  The
#: launcher itself only ever shortens by 900s (make_snapshot_run.sh:544).
TIMEOUT_WASTE_ERROR_SECONDS = 1800

_GDN_MODEL = re.compile(r"qwen3\.5|gdn", re.IGNORECASE)


def _is_gdn_model(model: str) -> bool:
    """Whether the model uses GDN linear attention (FlashInfer JIT risk)."""
    return bool(_GDN_MODEL.search(model))


def _effective_vllm_args(
    flavor: Flavor, config: PreflightConfig
) -> tuple[str, ...] | None:
    """The argv the engine will actually be launched with, or ``None``.

    ``None`` means "no argv comes from the launcher" -- either nothing was
    supplied and no default applies, or the flavor builds its own.
    """
    if config.vllm_args is not None:
        return config.vllm_args
    return flavor.launcher_default_vllm_args


def _check_vllm_args_consumed(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    if config.vllm_args is None or flavor.consumes_vllm_args:
        return []
    return [
        Finding(
            Severity.ERROR,
            "vllm-args-ignored",
            f"--vllm-args was supplied for flavor {flavor.name!r}, but that flavor "
            f"never consumes it: the launcher exports {flavor.vllm_args_var} "
            f"(make_snapshot_run.sh:863-865) and {flavor.vllm_args_evidence}. "
            "The job would look configured and would not be.",
        )
    ]


def _check_required_options(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    findings = []
    for option in flavor.required_options:
        value = config.options.get(option)
        if value is None or value == "":
            findings.append(
                Finding(
                    Severity.ERROR,
                    "missing-required-option",
                    f"flavor {flavor.name!r} requires {option}; it was not supplied.",
                )
            )
    return findings


def _check_options_consumed(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    """Reject every explicitly supplied flavor option that reaches no consumer.

    ``--vllm-args`` and ``--workload`` are handled by their dedicated checks,
    which can name the exact environment variable and test evidence.  Keeping
    them in :data:`FLAVOR_OPTION_CONSUMERS` still makes the table exhaustive;
    avoiding a second finding keeps one defect to one diagnostic.
    """
    findings: list[Finding] = []
    for option in sorted(config.supplied_options):
        consumers = FLAVOR_OPTION_CONSUMERS.get(option)
        if option in COMMON_LAUNCHER_OPTIONS:
            continue
        if consumers is None:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "option-routing-unknown",
                    f"{option} was supplied, but preflight has no consumer entry for "
                    "it. Refusing until the launcher option is classified as common "
                    "or assigned to the flavors that actually consume it.",
                )
            )
            continue
        if flavor.name in consumers:
            continue
        if option in {"--vllm-args", "--workload"}:
            continue
        findings.append(
            Finding(
                Severity.ERROR,
                "option-ignored",
                f"{option} was supplied for flavor {flavor.name!r}, but neither "
                "that launcher's flavor arm nor its test consumes the value. The "
                "job would look configured and run unchanged. This option is "
                f"consumed only by: {', '.join(sorted(consumers))}.",
            )
        )
    return findings


def _check_option_values(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    """Validate values whose consumer otherwise discovers the error on a GPU.

    The shell already validates its scheduling syntax.  These are different:
    they are exported verbatim and parsed only by the selected Python test.  A
    malformed JSON layer list or unknown matrix cell therefore used to survive
    until after allocation, even though every fact needed to reject it was in
    the launcher argv.
    """
    findings: list[Finding] = []

    def invalid(option: str, detail: str) -> None:
        findings.append(
            Finding(
                Severity.ERROR,
                "invalid-option",
                f"{option} for flavor {flavor.name!r} {detail}",
            )
        )

    for option in ("--hf-reference", "--strict-verdict"):
        if option in config.supplied_options and config.options.get(option) not in {"0", "1"}:
            invalid(option, "must be exactly 0 or 1")

    if "--target-layer-ids" in config.supplied_options:
        raw = config.options.get("--target-layer-ids", "")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if (
            not isinstance(parsed, list)
            or not parsed
            or any(isinstance(value, bool) or not isinstance(value, int) for value in parsed)
        ):
            invalid(
                "--target-layer-ids",
                "must be a non-empty JSON array of integers, for example '[2,18,33]'",
            )

    if "--inject-ms" in config.supplied_options:
        raw = config.options.get("--inject-ms", "")
        if re.fullmatch(r"[0-9]+", raw) is None:
            invalid("--inject-ms", "must be a non-negative integer number of milliseconds")

    if "--inject-percent" in config.supplied_options:
        raw = config.options.get("--inject-percent", "")
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value < 0:
            invalid("--inject-percent", "must be a finite non-negative number")

    if "--max-model-len" in config.supplied_options:
        raw = config.options.get("--max-model-len", "")
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            invalid("--max-model-len", "must be a positive integer")

    if "--matrix-cell" in config.supplied_options:
        value = config.options.get("--matrix-cell", "")
        if value not in _MODEL_MATRIX_CELLS:
            invalid(
                "--matrix-cell",
                "must name one of " + ", ".join(sorted(_MODEL_MATRIX_CELLS)),
            )
    return findings


def _check_memory_bounds(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    if not flavor.consumes_vllm_args:
        # The engine argv is hardcoded in the test; there is nothing here to
        # bound, and _check_vllm_args_consumed already objected to any attempt.
        return []
    args = _effective_vllm_args(flavor, config)
    if args is None:
        if flavor.test_default_is_bounded:
            return []
        return [
            Finding(
                Severity.ERROR,
                "unbounded-engine",
                f"flavor {flavor.name!r} spawns an engine but no vLLM args reach it: "
                f"{flavor.vllm_args_var} would be unset and the test defaults to an "
                "empty list, so vLLM sizes its KV cache for the whole card. "
                f"Supply --vllm-args with {' and '.join(MEMORY_BOUND_FLAGS)}.",
            )
        ]
    missing = [flag for flag in MEMORY_BOUND_FLAGS if _option_value(args, flag) is None]
    if missing:
        return [
            Finding(
                Severity.ERROR,
                "unbounded-engine",
                f"vLLM args for flavor {flavor.name!r} omit {', '.join(missing)}; "
                "an engine without memory bounds sizes its KV cache for the whole "
                "card and OOMs alongside anything else on the GPU.",
            )
        ]
    return []


def _check_gdn_prefill_backend(
    flavor: Flavor, config: PreflightConfig
) -> list[Finding]:
    model = config.model or flavor.default_model
    if model is None or not _is_gdn_model(model):
        return []
    args = _effective_vllm_args(flavor, config)
    if args is None:
        return []
    if _option_value(args, "--gdn-prefill-backend") == "triton":
        return []
    return [
        Finding(
            Severity.WARNING,
            "gdn-jit",
            f"model {model!r} uses GDN linear attention but the vLLM args do not set "
            "--gdn-prefill-backend triton. vLLM then JIT-compiles the FlashInfer GDN "
            "prefill kernel on first use: a measured 10m33s of silent startup "
            "(05:05:48 to 05:16:21 in stage1-qwen/gateway-and-vllm.log), paid on each "
            "of four H100 runs, while weights load in ~2.2s.",
        )
    ]


class _Unreadable:
    """Sentinel: the workload's context requirement could not be determined.

    Distinct from ``None`` ("the workload declares no requirement", which is a
    legitimate answer and passes) because "I could not read this object" must
    block a launch.  Returning ``None`` for both is precisely the bug this
    sentinel exists to make impossible.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unreadable>"


_UNREADABLE = _Unreadable()


def _declared_context_window(workload: WorkloadLike) -> int | None | _Unreadable:
    """``min_max_model_len`` from a Mapping-style or attribute-style requirements.

    ``WorkloadSpec.requirements`` is a Mapping; the hand-rolled stand-ins in the
    tests are attribute holders.  Both are read here rather than in the caller so
    there is exactly one place that can get it wrong.  Anything that is neither
    -- a workload with no ``requirements`` at all, a ``None`` requirements block,
    an object exposing the field by neither route, or a value that is not an
    integer -- comes back as :data:`_UNREADABLE`.
    """
    requirements = getattr(workload, "requirements", _UNREADABLE)
    if isinstance(requirements, _Unreadable) or requirements is None:
        return _UNREADABLE
    if isinstance(requirements, Mapping):
        # A Mapping we could read that simply does not declare the key is an
        # honest "no requirement"; workloads.verify_workload is the check that
        # objects to a manifest omitting it.
        value = requirements.get("min_max_model_len")
    elif hasattr(requirements, "min_max_model_len"):
        value = requirements.min_max_model_len
    else:
        return _UNREADABLE
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return _UNREADABLE


def _check_workload_compatibility(
    flavor: Flavor, config: PreflightConfig
) -> list[Finding]:
    workload = config.workload
    if workload is None:
        return []
    name = getattr(workload, "name", "<unnamed>")
    required = _declared_context_window(workload)
    if isinstance(required, _Unreadable):
        return [
            Finding(
                Severity.ERROR,
                "workload-unreadable",
                f"could not determine the required context window of workload {name!r}: "
                f"its requirements are {type(getattr(workload, 'requirements', None)).__name__} "
                "and expose no readable 'min_max_model_len' (expected a Mapping entry or an "
                "attribute holding an integer). Refusing rather than passing the check "
                "silently -- an unreadable requirement is how this check went dead before.",
            )
        ]
    if required is None:
        return []
    # An explicit --max-model-len wins over anything recoverable from the argv.
    # config-matrix and model-matrix build their own engine argv and never
    # receive --vllm-args, so for those two the argv route yields nothing and
    # the check would go quiet exactly where the workload system is used.
    raw: str | None = None
    if config.max_model_len is not None:
        raw = str(config.max_model_len)
    else:
        args = _effective_vllm_args(flavor, config)
        raw = _option_value(args, "--max-model-len") if args is not None else None
    if raw is None:
        # An unbounded engine is already an error from _check_memory_bounds for
        # the flavors that take launcher args; for the rest we cannot see the
        # argv, so say nothing rather than guess.
        return []
    try:
        launched = int(raw)
    except ValueError:
        return [
            Finding(
                Severity.ERROR,
                "workload-incompatible",
                f"--max-model-len {raw!r} is not an integer, so workload {name!r} "
                f"(needs at least {required}) cannot be checked against it.",
            )
        ]
    if launched < required:
        return [
            Finding(
                Severity.ERROR,
                "workload-incompatible",
                f"workload {name!r} requires --max-model-len >= {required} but the "
                f"launch requests {launched}; its long prompts would be truncated or "
                "rejected and the run would measure a different workload.",
            )
        ]
    return []


def _check_workload_consumed(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    """A ``--workload`` a flavor never reads must refuse, not reassure.

    The capacity check above validates the workload against the window; it says
    nothing about whether the job will *serve* that workload.  For every flavor
    whose test does not read :data:`WORKLOAD_VAR` the answer is no, and a
    validated-but-unused workload is worse than no workload at all: the operator
    is told the configuration is fine, the run measures something else, and the
    results index records the name of traffic that was never sent.
    """
    if config.workload is None or flavor.consumes_workload:
        return []
    name = getattr(config.workload, "name", "<unnamed>")
    return [
        Finding(
            Severity.ERROR,
            "workload-ignored",
            f"--workload {name!r} was supplied for flavor {flavor.name!r}, but that "
            f"flavor never consumes it: {flavor.workload_evidence}. The job would "
            "seed from whatever that test selects on its own while this preflight "
            "blessed a configuration the run does not use. Drop --workload, or run "
            "a flavor that reads it "
            f"({', '.join(sorted(f.name for f in FLAVORS.values() if f.consumes_workload))}).",
        )
    ]


def _check_workload_does_not_collide_with_corpus(
    flavor: Flavor, config: PreflightConfig
) -> list[Finding]:
    """A workload and a prompt corpus together name two different traffics.

    The idle-tuning flavor defaults ``--corpus`` to the ultrachat file, so
    ``--workload agentic-mixed-outcome`` alone exports *both* variables and the
    run cannot say which one it measured.  ``test_live_idle_tuning`` refuses
    that pair at runtime -- correctly -- but it refuses it after the job has
    been queued, scheduled and given a GPU, which is precisely the failure
    preflight exists to move forward in time.  Job 375376 died 1m48s into an
    allocation on exactly this.

    Checked here rather than fixed by silently dropping the corpus: which
    traffic the operator meant is not something this tool should guess.

    Scoped to :attr:`Flavor.corpus_collides_with_workload`, NOT to
    ``consumes_workload``.  Both workload-consuming flavors read
    :data:`CORPUS_VAR`, and only idle-tuning treats it as a rival seed source.
    config-matrix reads it into ``legacy_prompt_corpus``, records it in the
    manifest and seeds entirely from the workload -- and its launcher arm makes
    ``--corpus`` mandatory, so scoping on ``consumes_workload`` would refuse
    every config-matrix launch that named a workload, including the ones the
    launcher itself requires.
    """
    if config.workload is None or not flavor.corpus_collides_with_workload:
        return []
    # ``options`` is the modelled launcher invocation, and the caller already
    # fills in the flavor's DEFAULT corpus there (and pops it for
    # ``--no-corpus``), so this sees the pair the sbatch will really export --
    # not merely the flags the operator happened to type.
    corpus = config.options.get("--corpus")
    if not corpus:
        return []
    name = getattr(config.workload, "name", "<unnamed>")
    return [
        Finding(
            Severity.ERROR,
            "workload-corpus-collision",
            f"--workload {name!r} and --corpus {corpus!r} are both in "
            f"effect for flavor {flavor.name!r}, which treats them as rival seed "
            "sources and would measure one while the record named the other "
            f"({flavor.corpus_collision_evidence}). Note this "
            "flavor supplies a DEFAULT corpus, so supplying only --workload "
            "still produces the pair. Pass --no-corpus to seed from the "
            "workload, or drop --workload to seed from the corpus.",
        )
    ]


def _check_max_model_len_agrees_with_argv(
    flavor: Flavor, config: PreflightConfig
) -> list[Finding]:
    """``--max-model-len`` must not contradict the engine argv it stands in for.

    ``--max-model-len`` exists for the flavors that build their own engine argv
    and therefore have no ``--vllm-args`` for the capacity check to read.  When
    a flavor *does* take ``--vllm-args``, the argv is what the engine gets, and
    the capacity check prefers the explicit option -- so a mismatch means the
    workload was validated against a window the run will not use.  That is how
    ``--flavor idle-tuning --workload agentic-mixed-outcome --max-model-len
    24576`` passed while the job would have served at the launcher's 4096.
    """
    if config.max_model_len is None:
        return []
    args = _effective_vllm_args(flavor, config)
    if args is None:
        return []
    argv_value = _option_value(args, "--max-model-len")
    if argv_value is None or argv_value == str(config.max_model_len):
        return []
    return [
        Finding(
            Severity.ERROR,
            "max-model-len-disagreement",
            f"--max-model-len {str(config.max_model_len)!r} disagrees with the "
            f"--max-model-len {argv_value!r} in the vLLM args flavor {flavor.name!r} "
            "will actually launch with. The engine follows the argv, so any check "
            "made against the option was made against a window the run does not "
            "have. Set both to the same value, or set only the vLLM args.",
        )
    ]


def _check_timeouts(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    raw_time = config.slurm_time or flavor.default_time
    try:
        wall = parse_slurm_time(raw_time)
    except ValueError as exc:
        return [Finding(Severity.ERROR, "timeout", str(exc))]
    findings = []
    for label, timeout in sorted(config.timeouts.items()):
        if timeout >= wall:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "timeout",
                    f"timeout {label!r} is {timeout}s but the allocation is only "
                    f"{wall}s ({raw_time}); it can never fire, so the job dies by "
                    "SLURM kill with no artifacts flushed.",
                )
            )
        elif wall - timeout > TIMEOUT_WASTE_ERROR_SECONDS:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "timeout",
                    f"timeout {label!r} is {timeout}s against a {wall}s allocation "
                    f"({raw_time}): it would kill a healthy job with {wall - timeout}s "
                    "of paid wall clock left.",
                )
            )
    return findings


def _check_silent_green(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    findings = []
    gate = config.env.get(flavor.gate_var)
    if gate != "1":
        findings.append(
            Finding(
                Severity.ERROR,
                "silent-skip",
                f"{flavor.gate_var} is {gate!r}, not '1': every test in "
                f"{flavor.test_path} would skip and the job would report success "
                "having measured nothing, after holding the allocation.",
            )
        )
    for name, bad in SILENT_GREEN_FLAGS.items():
        if config.env.get(name) == bad:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "silent-green",
                    f"{name}={bad} relaxes the run's own verdict: the job can finish "
                    "green while the property it exists to measure went unchecked. "
                    "Acceptable for local debugging, never for an allocation.",
                )
            )
    return findings


def _depends_on(flavor: Flavor, path: str) -> bool:
    """Whether a repo-relative path is code the flavor's test actually runs."""
    return (
        path == flavor.test_path
        or path.startswith("src/")
        or path.startswith("tests/e2e/harness/")
        or path == "scripts/make_snapshot_run.sh"
    )


def _check_provenance(flavor: Flavor, config: PreflightConfig) -> list[Finding]:
    findings = []
    if not config.git.commit_exists:
        findings.append(
            Finding(
                Severity.ERROR,
                "provenance",
                f"commit {config.commit!r} does not exist; the launcher resolves it "
                "with `git rev-parse --verify` before it can archive anything.",
            )
        )
    relevant = sorted(p for p in config.git.dirty_paths if _depends_on(flavor, p))
    if relevant:
        findings.append(
            Finding(
                Severity.WARNING,
                "provenance",
                "uncommitted changes to code this flavor executes: "
                f"{', '.join(relevant)}. The launcher `git archive`s the COMMIT, so "
                "these edits are NOT in the snapshot -- the run would measure "
                "different code than the working tree you are looking at.",
            )
        )
    return findings


_CHECKS = (
    _check_vllm_args_consumed,
    _check_required_options,
    _check_options_consumed,
    _check_option_values,
    _check_memory_bounds,
    _check_gdn_prefill_backend,
    _check_workload_compatibility,
    _check_workload_consumed,
    _check_workload_does_not_collide_with_corpus,
    _check_max_model_len_agrees_with_argv,
    _check_timeouts,
    _check_silent_green,
    _check_provenance,
)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def preflight(config: PreflightConfig) -> tuple[tuple[Finding, ...], bool]:
    """Validate a proposed run.  Returns ``(findings, ok)``.

    ``ok`` is ``False`` if and only if at least one finding is an ERROR, so a
    caller can gate ``sbatch`` on it without re-deriving severity.
    """
    flavor = FLAVORS.get(config.flavor)
    if flavor is None:
        known = ", ".join(sorted(FLAVORS))
        return (
            (
                Finding(
                    Severity.ERROR,
                    "unknown-flavor",
                    f"unknown flavor {config.flavor!r}; known flavors: {known}",
                ),
            ),
            False,
        )
    findings: list[Finding] = []
    for check in _CHECKS:
        findings.extend(check(flavor, config))
    ok = not any(finding.blocking for finding in findings)
    return tuple(findings), ok


def render_findings(findings: Iterable[Finding]) -> str:
    """Render findings for an operator, blocking ones first."""
    findings = list(findings)
    if not findings:
        return "preflight: OK -- no findings."
    errors = [f for f in findings if f.severity is Severity.ERROR]
    warnings = [f for f in findings if f.severity is Severity.WARNING]
    lines = []
    for finding in (*errors, *warnings):
        label = finding.severity.value.upper()
        lines.append(f"{label:<7} [{finding.check}] {finding.message}")
    verdict = "DO NOT LAUNCH" if errors else "launch is OK"
    lines.append(
        f"preflight: {len(errors)} error(s), {len(warnings)} warning(s) -- {verdict}."
    )
    return "\n".join(lines)
