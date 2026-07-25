from __future__ import annotations

import importlib.metadata as metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import speedlm.doctor as doctor
from speedlm.config import SpeedLMConfig
from speedlm.profiles import QWEN_35_9B_MTP_PROFILE, ModelProfile


def _completed(
    args: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _healthy_run(
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    assert kwargs["timeout"] > 0
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    if command[:2] == ["nvcc", "--version"]:
        return _completed(command, stdout="Cuda compilation tools, release 12.9, V12.9.41")
    if "--query-gpu=name,memory.total,memory.used,driver_version" in command:
        return _completed(command, stdout="NVIDIA H100, 81920 MiB, 2048 MiB, 575.57.08\n")
    if command == ["nvidia-smi"]:
        return _completed(command, stdout="NVIDIA-SMI 575.57.08  CUDA Version: 12.9")
    raise AssertionError(f"unexpected command: {command}")


def _patch_healthy_non_gpu_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "vllm": "0.25.1+cu129",
        "torch": "2.11.0+cu129",
        "speculators": "v0.6.0",
    }
    monkeypatch.setattr(doctor.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        doctor.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=20 * 1024**3, free=80 * 1024**3),
    )
    monkeypatch.setattr(
        doctor,
        "_linux_memory_bytes",
        lambda: (128 * 1024**3, 96 * 1024**3),
    )
    python_check = doctor.check_python((3, 12, 8))
    monkeypatch.setattr(doctor, "check_python", lambda: python_check)


def test_gpu_nvidia_smi_absent_is_clean_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == doctor.NVIDIA_SMI_TIMEOUT_SECONDS
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(doctor.subprocess, "run", absent)

    probe = doctor.probe_gpu()

    assert probe.check.status is doctor.CheckStatus.FAIL
    assert probe.check.data is not None
    assert probe.check.data["binary_present"] is False
    assert "not installed" in probe.check.detail
    assert doctor.plan_execution(probe).mode is doctor.ExecutionMode.UNAVAILABLE


def test_gpu_driver_unreachable_is_clean_actionable_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."

    def unreachable(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0
        return _completed(command, stderr=error, returncode=9)

    monkeypatch.setattr(doctor.subprocess, "run", unreachable)

    probe = doctor.probe_gpu()

    assert probe.check.status is doctor.CheckStatus.FAIL
    assert probe.check.data is not None
    assert probe.check.data["binary_present"] is True
    assert probe.check.data["driver_reachable"] is False
    assert "cannot communicate" in probe.check.detail
    assert error in probe.check.detail


def test_healthy_gpu_defaults_to_idle_and_report_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(doctor.subprocess, "run", _healthy_run)
    _patch_healthy_non_gpu_checks(monkeypatch)

    report = doctor.run_doctor(
        SpeedLMConfig(model=doctor.PRIMARY_VERIFIER),
        home=tmp_path,
    )

    assert report.execution_mode is doctor.ExecutionMode.IDLE
    assert report.plan.scratch_limit_mib == 5120
    assert report.plan.headroom_mib == 79872
    assert report.overall_status is doctor.CheckStatus.PASS
    payload = json.loads(report.to_json())
    assert set(payload) == {"status", "execution_mode", "plan", "checks"}
    assert payload["status"] == "PASS"
    assert payload["execution_mode"] == "idle"
    assert [check["name"] for check in payload["checks"]] == [
        "python",
        "gpu",
        "cuda",
        "packages",
        "disk",
        "memory",
        "model_pair",
    ]
    assert "Overall: PASS" in report.render_text()


def test_little_free_vram_falls_back_from_colocated_to_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def little_free(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0
        return _completed(
            command,
            stdout="NVIDIA A10, 23028 MiB, 22000 MiB, 575.57.08\n",
        )

    monkeypatch.setattr(doctor.subprocess, "run", little_free)
    probe = doctor.probe_gpu()

    plan = doctor.plan_execution(probe, prefer_colocated=True)

    assert plan.mode is doctor.ExecutionMode.IDLE
    assert plan.headroom_mib == 1028
    assert "lacks 5120 MiB" in plan.detail


def test_colocated_requires_measured_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.subprocess, "run", _healthy_run)
    probe = doctor.probe_gpu()

    plan = doctor.plan_execution(probe, prefer_colocated=True)

    assert plan.mode is doctor.ExecutionMode.COLOCATED
    assert plan.headroom_mib is not None
    assert plan.headroom_mib >= plan.scratch_limit_mib


def test_gpu_below_scratch_capacity_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def tiny(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout="NVIDIA tiny, 4096 MiB, 100 MiB, 550.1\n")

    monkeypatch.setattr(doctor.subprocess, "run", tiny)

    plan = doctor.plan_execution(doctor.probe_gpu())

    assert plan.mode is doctor.ExecutionMode.UNAVAILABLE
    assert plan.scratch_limit_mib == 5120


def test_cuda_13_is_known_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    device = doctor.GPUDevice("NVIDIA H100", 81920, 1024, "580.1")
    probe = doctor.GPUProbe(
        doctor.Check("gpu", doctor.CheckStatus.PASS, "ok"),
        (device,),
    )

    def cuda_13(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0
        if command == ["nvcc", "--version"]:
            return _completed(command, stdout="Cuda compilation tools, release 13.0, V13.0.1")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(doctor.subprocess, "run", cuda_13)

    result = doctor.check_cuda(probe)

    assert result.status is doctor.CheckStatus.FAIL
    assert "known-incompatible" in result.detail
    assert result.data is not None
    assert result.data["version"] == "13.0"


def test_cuda_mismatch_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    device = doctor.GPUDevice("NVIDIA A100", 40960, 1024, "570.1")
    probe = doctor.GPUProbe(
        doctor.Check("gpu", doctor.CheckStatus.PASS, "ok"),
        (device,),
    )

    def cuda_12_8(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout="Cuda compilation tools, release 12.8, V12.8.1")

    monkeypatch.setattr(doctor.subprocess, "run", cuda_12_8)

    assert doctor.check_cuda(probe).status is doctor.CheckStatus.WARN


def test_packages_missing_and_wrong_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    def package_version(name: str) -> str:
        if name == "vllm":
            raise metadata.PackageNotFoundError(name)
        if name == "torch":
            return "2.10.0"
        return "0.6.0"

    monkeypatch.setattr(doctor.metadata, "version", package_version)

    result = doctor.check_packages()

    assert result.status is doctor.CheckStatus.FAIL
    assert "vllm is missing" in result.detail
    assert "torch 2.10.0" in result.detail
    assert result.data is not None
    packages = result.data["packages"]
    assert isinstance(packages, dict)
    assert packages["vllm"]["installed"] is None


def test_packages_accept_expected_build_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "vllm": "0.25.1+cu129",
        "torch": "2.11.0+cu129",
        "speculators": "v0.6.0",
    }
    monkeypatch.setattr(doctor.metadata, "version", versions.__getitem__)

    assert doctor.check_packages().status is doctor.CheckStatus.PASS


def test_low_disk_is_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        doctor.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=90 * 1024**3, free=10 * 1024**3),
    )

    result = doctor.check_disk(tmp_path)

    assert result.status is doctor.CheckStatus.FAIL
    assert "at least 20 GiB" in result.detail
    assert result.data is not None
    assert result.data["free_gib"] == 10.0


@pytest.mark.parametrize(
    ("verifier", "draft", "pair_name"),
    [
        (doctor.PRIMARY_VERIFIER, doctor.PRIMARY_DRAFT, "primary"),
        (doctor.FALLBACK_VERIFIER, doctor.FALLBACK_DRAFT, "fallback"),
    ],
)
def test_supported_model_pairs(
    verifier: str,
    draft: str,
    pair_name: str,
) -> None:
    derived = doctor.check_model_pair(SpeedLMConfig(model=verifier))
    explicit = doctor.check_model_pair(
        SimpleNamespace(model=verifier, draft_model=draft)
    )

    assert derived.status is doctor.CheckStatus.PASS
    assert explicit.status is doctor.CheckStatus.PASS
    assert derived.data is not None
    assert derived.data["pair"] == pair_name
    assert derived.data["draft"] == draft
    assert derived.data["draft_derived"] is True


def test_incoherent_model_pair_fails() -> None:
    result = doctor.check_model_pair(
        SimpleNamespace(
            model=doctor.PRIMARY_VERIFIER,
            draft_model=doctor.FALLBACK_DRAFT,
        )
    )

    assert result.status is doctor.CheckStatus.FAIL
    assert "incoherent" in result.detail


def test_resolved_non_default_profile_is_validated() -> None:
    result = doctor.check_model_pair(
        SpeedLMConfig(model=QWEN_35_9B_MTP_PROFILE.verifier_model)
    )

    assert result.status is doctor.CheckStatus.PASS
    assert result.data is not None
    assert result.data["profile"] == QWEN_35_9B_MTP_PROFILE.name
    assert result.data["verifier"] == QWEN_35_9B_MTP_PROFILE.verifier_model
    assert result.data["draft"] is None
    assert result.data["method"] == "mtp"
    assert result.data["layout"] == "native"
    profile = result.data["resolved_profile"]
    assert isinstance(profile, dict)
    assert profile["speculative_method"] == "mtp"
    assert profile["chat_template_kind"] == "chatml"
    assert profile["tool_call_parser"] == "hermes"
    assert profile["reasoning_parser"] is None


def test_doctor_resolves_huggingface_cache_snapshot_path() -> None:
    served_model = (
        "/root/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-9B/snapshots/0123456789abcdef"
    )

    result = doctor.check_model_pair(SpeedLMConfig(model=served_model))

    assert result.status is doctor.CheckStatus.PASS
    assert result.data is not None
    assert result.data["profile"] == QWEN_35_9B_MTP_PROFILE.name
    assert result.data["verifier"] == QWEN_35_9B_MTP_PROFILE.verifier_model
    assert result.data["method"] == "mtp"


def test_unknown_model_reports_unprofiled_and_tuning_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(doctor.subprocess, "run", _healthy_run)
    _patch_healthy_non_gpu_checks(monkeypatch)

    report = doctor.run_doctor(
        SpeedLMConfig(model="acme/unknown-verifier"),
        home=tmp_path,
    )
    payload = json.loads(report.to_json())
    model_pair = next(
        check for check in payload["checks"] if check["name"] == "model_pair"
    )

    assert model_pair["status"] == "WARN"
    assert "no profile matched" in model_pair["detail"].lower()
    assert "tuning is unavailable" in model_pair["detail"]
    assert isinstance(model_pair["data"], dict)
    assert model_pair["data"]["tuning_available"] is False
    profile = model_pair["data"]["resolved_profile"]
    assert isinstance(profile, dict)
    assert profile["status"] == "unprofiled"
    assert profile["verifier_model"] == "acme/unknown-verifier"
    assert profile["tuning_available"] is False
    assert report.execution_mode is doctor.ExecutionMode.UNAVAILABLE
    assert report.overall_status is doctor.CheckStatus.WARN


def test_non_trainable_profile_warns_that_tuning_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ModelProfile(
        name="local-ngram",
        verifier_model="acme/local-verifier",
        draft_model=None,
        speculative_method="ngram",
        num_speculative_tokens=4,
        target_layer_ids=None,
        chat_template_kind="auto",
        max_seq_len=4096,
    )
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "local-ngram.json").write_text(
        json.dumps(profile.to_dict()),
        encoding="utf-8",
    )

    result = doctor.check_model_pair(
        SpeedLMConfig(model=profile.verifier_model, profile=profile.name),
        home=tmp_path,
    )

    assert result.status is doctor.CheckStatus.WARN
    assert "tuning is unavailable" in result.detail
    assert result.data is not None
    assert result.data["profile"] == profile.name
    assert result.data["method"] == "ngram"
    assert result.data["trainable"] is False
    assert result.data["tuning_available"] is False
    resolved_profile = result.data["resolved_profile"]
    assert isinstance(resolved_profile, dict)
    assert resolved_profile["status"] == "profiled"
    assert resolved_profile["tuning_available"] is False

    monkeypatch.setattr(doctor.subprocess, "run", _healthy_run)
    _patch_healthy_non_gpu_checks(monkeypatch)
    report = doctor.run_doctor(
        SpeedLMConfig(model=profile.verifier_model, profile=profile.name),
        home=tmp_path,
    )

    assert report.overall_status is doctor.CheckStatus.WARN
    assert report.execution_mode is doctor.ExecutionMode.UNAVAILABLE
    assert "tuning is unavailable" in report.plan.detail


def test_overall_status_is_worst_check() -> None:
    report = doctor.DoctorReport(
        checks=(
            doctor.Check("pass", doctor.CheckStatus.PASS, "ok"),
            doctor.Check("skip", doctor.CheckStatus.SKIP, "not applicable"),
            doctor.Check("warn", doctor.CheckStatus.WARN, "caution"),
            doctor.Check("fail", doctor.CheckStatus.FAIL, "broken"),
        ),
        plan=doctor.ExecutionPlan(doctor.ExecutionMode.UNAVAILABLE, "no GPU"),
    )

    assert report.overall_status is doctor.CheckStatus.FAIL
    assert report.to_dict()["status"] == "FAIL"


def test_no_gpu_full_report_skips_cuda_and_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def absent(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(doctor.subprocess, "run", absent)
    _patch_healthy_non_gpu_checks(monkeypatch)

    report = doctor.run_doctor(
        SpeedLMConfig(model=doctor.FALLBACK_VERIFIER),
        home=tmp_path,
    )
    checks = {check.name: check for check in report.checks}

    assert checks["gpu"].status is doctor.CheckStatus.FAIL
    assert checks["cuda"].status is doctor.CheckStatus.SKIP
    assert report.execution_mode is doctor.ExecutionMode.UNAVAILABLE
    assert report.overall_status is doctor.CheckStatus.FAIL
