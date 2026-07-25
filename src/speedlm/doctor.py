"""Read-only host diagnostics and execution planning for SpeedLM.

The doctor deliberately avoids importing CUDA-backed packages and never creates
the SpeedLM storage layout.  Every individual probe converts operational errors
into a :class:`Check` result so it is safe to run on non-GPU hosts.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from speedlm.storage import resolve_layout

NVIDIA_SMI_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_MIN_DISK_FREE_GB: Final = 20.0
SCRATCH_LIMIT_MIB: Final = 5 * 1024

PRIMARY_VERIFIER: Final = "openai/gpt-oss-20b"
PRIMARY_DRAFT: Final = "RedHatAI/gpt-oss-20b-speculator.eagle3"
FALLBACK_VERIFIER: Final = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_DRAFT: Final = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"

SUPPORTED_MODEL_PAIRS: Final[Mapping[str, str]] = {
    PRIMARY_VERIFIER: PRIMARY_DRAFT,
    FALLBACK_VERIFIER: FALLBACK_DRAFT,
}

EXPECTED_PACKAGES: Final[Mapping[str, str]] = {
    "vllm": "0.25.1+cu129",
    "torch": "2.11.0",
    "speculators": "0.6.0",
}

class CheckStatus(StrEnum):
    """Outcome of one doctor probe."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


_STATUS_RANK: Final[Mapping[CheckStatus, int]] = {
    CheckStatus.PASS: 0,
    CheckStatus.SKIP: 1,
    CheckStatus.WARN: 2,
    CheckStatus.FAIL: 3,
}


class ExecutionMode(StrEnum):
    """How the auto-tuning cycle can use the GPU."""

    IDLE = "idle"
    COLOCATED = "colocated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Check:
    """A stable, JSON-renderable result contract for one diagnostic."""

    name: str
    status: CheckStatus
    detail: str
    data: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
        }
        if self.data is not None:
            result["data"] = dict(self.data)
        return result


@dataclass(frozen=True, slots=True)
class GPUDevice:
    """Parsed fields for one GPU reported by ``nvidia-smi``."""

    name: str
    memory_total_mib: int
    memory_used_mib: int
    driver_version: str

    @property
    def memory_free_mib(self) -> int:
        return max(0, self.memory_total_mib - self.memory_used_mib)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True, slots=True)
class GPUProbe:
    """GPU check plus the parsed devices used by execution planning."""

    check: Check
    devices: tuple[GPUDevice, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Selected execution mode and its measured VRAM budget."""

    mode: ExecutionMode
    detail: str
    scratch_limit_mib: int = SCRATCH_LIMIT_MIB
    gpu_index: int | None = None
    headroom_mib: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "detail": self.detail,
            "scratch_limit_mib": self.scratch_limit_mib,
            "gpu_index": self.gpu_index,
            "headroom_mib": self.headroom_mib,
        }


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Aggregate doctor results and the resulting execution plan."""

    checks: tuple[Check, ...]
    plan: ExecutionPlan

    @property
    def overall_status(self) -> CheckStatus:
        if not self.checks:
            return CheckStatus.SKIP
        return max((check.status for check in self.checks), key=_STATUS_RANK.__getitem__)

    @property
    def status(self) -> CheckStatus:
        """Alias useful to callers treating a report like a check."""

        return self.overall_status

    @property
    def execution_mode(self) -> ExecutionMode:
        return self.plan.mode

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.overall_status.value,
            "execution_mode": self.execution_mode.value,
            "plan": self.plan.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_text(self) -> str:
        """Render a compact terminal-oriented report."""

        return self.render_text()

    def render_text(self) -> str:
        lines = [
            "SpeedLM doctor",
            *(f"[{check.status.value:4}] {check.name}: {check.detail}" for check in self.checks),
            f"Overall: {self.overall_status.value}",
            f"Execution mode: {self.execution_mode.value} — {self.plan.detail}",
        ]
        return "\n".join(lines)


def check_python(
    version_info: Sequence[int] | None = None,
) -> Check:
    """Check the pinned CPython minor version without raising."""

    try:
        version = tuple(version_info if version_info is not None else sys.version_info)
        major, minor, micro = version[:3]
        rendered = f"{major}.{minor}.{micro}"
        supported = major == 3 and minor == 12
        if supported:
            return Check(
                "python",
                CheckStatus.PASS,
                f"Python {rendered} is within >=3.12,<3.13",
                {"version": rendered},
            )
        return Check(
            "python",
            CheckStatus.FAIL,
            f"Python {rendered} is unsupported; install Python >=3.12,<3.13",
            {"version": rendered, "required": ">=3.12,<3.13"},
        )
    except Exception as exc:
        return Check(
            "python",
            CheckStatus.FAIL,
            f"Could not determine Python version: {exc}",
        )


def _parse_mib(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:MiB)?\s*", value)
    if match is None:
        raise ValueError(f"invalid MiB value {value!r}")
    return int(float(match.group(1)))


def _single_line(value: str) -> str:
    return " ".join(value.strip().split())


def probe_gpu(*, timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS) -> GPUProbe:
    """Probe NVIDIA GPUs, distinguishing binary and driver failures."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                "nvidia-smi is not installed or not on PATH; install the NVIDIA driver tools",
                {"binary_present": False, "device_count": 0},
            )
        )
    except subprocess.TimeoutExpired:
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                f"nvidia-smi timed out after {timeout_seconds:g}s; check the NVIDIA driver",
                {"binary_present": True, "device_count": 0},
            )
        )
    except OSError as exc:
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                f"Could not execute nvidia-smi: {exc}",
                {"binary_present": False, "device_count": 0},
            )
        )
    except Exception as exc:
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                f"Unexpected nvidia-smi failure: {exc}",
                {"binary_present": None, "device_count": 0},
            )
        )

    if completed.returncode != 0:
        reason = _single_line(completed.stderr or completed.stdout) or (
            f"nvidia-smi exited with status {completed.returncode}"
        )
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                f"nvidia-smi is present but cannot communicate with a usable driver: {reason}",
                {
                    "binary_present": True,
                    "driver_reachable": False,
                    "returncode": completed.returncode,
                    "device_count": 0,
                },
            )
        )

    try:
        devices: list[GPUDevice] = []
        for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"line {line_number} has {len(fields)} fields, expected 4")
            name, total, used, driver = fields
            device = GPUDevice(
                name=name,
                memory_total_mib=_parse_mib(total),
                memory_used_mib=_parse_mib(used),
                driver_version=driver,
            )
            if device.memory_used_mib > device.memory_total_mib:
                raise ValueError(f"line {line_number} reports used VRAM above total VRAM")
            devices.append(device)
        if not devices:
            return GPUProbe(
                Check(
                    "gpu",
                    CheckStatus.FAIL,
                    "nvidia-smi returned no GPUs; an NVIDIA GPU is required",
                    {
                        "binary_present": True,
                        "driver_reachable": True,
                        "device_count": 0,
                    },
                )
            )
    except Exception as exc:
        return GPUProbe(
            Check(
                "gpu",
                CheckStatus.FAIL,
                f"Could not parse nvidia-smi output: {exc}",
                {
                    "binary_present": True,
                    "driver_reachable": True,
                    "device_count": 0,
                },
            )
        )

    device_data = [device.to_dict() for device in devices]
    return GPUProbe(
        Check(
            "gpu",
            CheckStatus.PASS,
            f"Detected {len(devices)} usable NVIDIA GPU(s); "
            f"best measured headroom is {max(device.memory_free_mib for device in devices)} MiB",
            {
                "binary_present": True,
                "driver_reachable": True,
                "device_count": len(devices),
                "gpus": device_data,
            },
        ),
        tuple(devices),
    )


def check_gpu(*, timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS) -> Check:
    """Return just the public check contract for the GPU probe."""

    return probe_gpu(timeout_seconds=timeout_seconds).check


def _cuda_version_from_output(output: str) -> str | None:
    patterns = (
        r"\brelease\s+(\d+\.\d+)",
        r"\bCUDA Version:\s*(\d+\.\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1)
    return None


def _run_version_command(
    command: list[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def check_cuda(
    gpu_probe: GPUProbe,
    *,
    timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS,
) -> Check:
    """Detect the CUDA toolkit or driver-advertised CUDA compatibility."""

    try:
        drivers = sorted({device.driver_version for device in gpu_probe.devices})
        if gpu_probe.check.status is not CheckStatus.PASS:
            return Check(
                "cuda",
                CheckStatus.SKIP,
                "CUDA detection skipped because no usable NVIDIA driver was found",
                {"driver_versions": drivers},
            )

        detected: str | None = None
        source: str | None = None
        nvcc = _run_version_command(["nvcc", "--version"], timeout_seconds=timeout_seconds)
        if nvcc is not None and nvcc.returncode == 0:
            detected = _cuda_version_from_output(f"{nvcc.stdout}\n{nvcc.stderr}")
            if detected is not None:
                source = "nvcc"

        if detected is None:
            banner = _run_version_command(["nvidia-smi"], timeout_seconds=timeout_seconds)
            if banner is not None and banner.returncode == 0:
                detected = _cuda_version_from_output(f"{banner.stdout}\n{banner.stderr}")
                if detected is not None:
                    source = "nvidia-smi"

        data: dict[str, object] = {
            "version": detected,
            "expected": "12.9",
            "source": source,
            "driver_versions": drivers,
        }
        if detected is None:
            return Check(
                "cuda",
                CheckStatus.WARN,
                "NVIDIA driver is usable, but CUDA version could not be detected; "
                "SpeedLM expects CUDA 12.9",
                data,
            )
        major, minor = (int(part) for part in detected.split(".", maxsplit=1))
        if major == 13:
            return Check(
                "cuda",
                CheckStatus.FAIL,
                f"CUDA {detected} is known-incompatible; install the CUDA 12.9 stack",
                data,
            )
        if (major, minor) != (12, 9):
            return Check(
                "cuda",
                CheckStatus.WARN,
                f"CUDA {detected} does not match the pinned CUDA 12.9 stack",
                data,
            )
        return Check(
            "cuda",
            CheckStatus.PASS,
            f"CUDA {detected} matches the pinned stack; driver {', '.join(drivers)}",
            data,
        )
    except Exception as exc:
        return Check("cuda", CheckStatus.FAIL, f"Could not interpret CUDA version: {exc}")


def _package_version_matches(package: str, installed: str, expected: str) -> bool:
    normalized = installed.removeprefix("v")
    if package in {"torch", "speculators"}:
        return normalized.split("+", maxsplit=1)[0] == expected
    return normalized == expected


def check_packages() -> Check:
    """Check pinned package metadata without importing heavy modules."""

    package_data: dict[str, object] = {}
    problems: list[str] = []
    try:
        for package, expected in EXPECTED_PACKAGES.items():
            try:
                installed: str | None = metadata.version(package)
            except metadata.PackageNotFoundError:
                installed = None
            matches = (
                installed is not None
                and _package_version_matches(package, installed, expected)
            )
            package_data[package] = {
                "installed": installed,
                "expected": expected,
                "matches": matches,
            }
            if installed is None:
                problems.append(f"{package} is missing")
            elif not matches:
                problems.append(f"{package} {installed} (expected {expected})")
    except Exception as exc:
        return Check(
            "packages",
            CheckStatus.FAIL,
            f"Could not read installed package metadata: {exc}",
            {"packages": package_data},
        )

    if problems:
        return Check(
            "packages",
            CheckStatus.FAIL,
            "Pinned runtime package check failed: " + "; ".join(problems),
            {"packages": package_data},
        )
    return Check(
        "packages",
        CheckStatus.PASS,
        "Pinned runtime packages match: vllm, torch, speculators",
        {"packages": package_data},
    )


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_disk(
    home: Path,
    *,
    minimum_free_gb: float = DEFAULT_MIN_DISK_FREE_GB,
) -> Check:
    """Check free bytes on the filesystem that will contain SpeedLM home."""

    try:
        if minimum_free_gb < 0:
            raise ValueError("minimum_free_gb must be non-negative")
        probe_path = _nearest_existing_path(home)
        usage = shutil.disk_usage(probe_path)
        free_gib = usage.free / (1024**3)
        data: dict[str, object] = {
            "path": str(home),
            "filesystem_probe_path": str(probe_path),
            "free_bytes": usage.free,
            "free_gib": round(free_gib, 2),
            "minimum_free_gib": minimum_free_gb,
        }
        if free_gib < minimum_free_gb:
            return Check(
                "disk",
                CheckStatus.FAIL,
                f"{free_gib:.1f} GiB free at {home}; at least "
                f"{minimum_free_gb:g} GiB is required",
                data,
            )
        return Check(
            "disk",
            CheckStatus.PASS,
            f"{free_gib:.1f} GiB free at {home}",
            data,
        )
    except Exception as exc:
        return Check(
            "disk",
            CheckStatus.FAIL,
            f"Could not determine free space for {home}: {exc}",
            {"path": str(home), "minimum_free_gib": minimum_free_gb},
        )


def _linux_memory_bytes() -> tuple[int, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as meminfo:
        for line in meminfo:
            key, separator, value = line.partition(":")
            if separator and key in {"MemTotal", "MemAvailable"}:
                fields = value.split()
                if not fields:
                    raise ValueError(f"{key} has no value")
                values[key] = int(fields[0]) * 1024
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("/proc/meminfo lacks MemTotal or MemAvailable")
    return values["MemTotal"], values["MemAvailable"]


def check_memory() -> Check:
    """Report total and currently available system RAM."""

    try:
        total, available = _linux_memory_bytes()
        total_gib = total / (1024**3)
        available_gib = available / (1024**3)
        return Check(
            "memory",
            CheckStatus.PASS,
            f"{available_gib:.1f} GiB available of {total_gib:.1f} GiB system RAM",
            {
                "total_bytes": total,
                "available_bytes": available,
                "total_gib": round(total_gib, 2),
                "available_gib": round(available_gib, 2),
            },
        )
    except Exception as exc:
        return Check(
            "memory",
            CheckStatus.WARN,
            f"Could not determine total/available system RAM: {exc}",
        )


def _optional_string_attribute(config: object, name: str) -> str | None:
    value = getattr(config, name, None)
    return value if isinstance(value, str) and value else None


def _local_model_path(reference: str) -> Path | None:
    candidate = Path(reference).expanduser()
    explicitly_local = (
        candidate.is_absolute()
        or reference.startswith("./")
        or reference.startswith("../")
        or reference.startswith("~")
    )
    if explicitly_local or candidate.exists():
        return candidate
    return None


def check_model_pair(config: object) -> Check:
    """Validate a supported verifier/EAGLE-3 pair without fetching weights."""

    try:
        verifier = _optional_string_attribute(config, "verifier_model")
        if verifier is None:
            verifier = _optional_string_attribute(config, "model")
        explicit_draft = _optional_string_attribute(config, "draft_model")
        if verifier is None:
            return Check(
                "model_pair",
                CheckStatus.FAIL,
                "No verifier model is configured",
            )

        expected_draft = SUPPORTED_MODEL_PAIRS.get(verifier)
        if expected_draft is None:
            supported = ", ".join(SUPPORTED_MODEL_PAIRS)
            return Check(
                "model_pair",
                CheckStatus.FAIL,
                f"Unsupported verifier {verifier!r}; supported verifiers: {supported}",
                {"verifier": verifier, "draft": explicit_draft},
            )

        draft = explicit_draft or expected_draft
        if draft != expected_draft:
            return Check(
                "model_pair",
                CheckStatus.FAIL,
                f"Draft {draft!r} is incoherent with verifier {verifier!r}; "
                f"expected {expected_draft!r}",
                {
                    "verifier": verifier,
                    "draft": draft,
                    "expected_draft": expected_draft,
                },
            )

        verifier_path = _local_model_path(verifier)
        draft_path = _local_model_path(draft)
        for role, path in (("verifier", verifier_path), ("draft", draft_path)):
            if path is not None and (not path.exists() or not path.is_dir()):
                return Check(
                    "model_pair",
                    CheckStatus.FAIL,
                    f"Configured local {role} model directory does not exist: {path}",
                    {"verifier": verifier, "draft": draft},
                )
        if (
            verifier_path is not None
            and draft_path is not None
            and verifier_path.resolve() == draft_path.resolve()
        ):
            return Check(
                "model_pair",
                CheckStatus.FAIL,
                "Verifier and EAGLE-3 draft must use separate model directories",
                {"verifier": verifier, "draft": draft},
            )

        pair_name = "primary" if verifier == PRIMARY_VERIFIER else "fallback"
        return Check(
            "model_pair",
            CheckStatus.PASS,
            f"{pair_name.capitalize()} verifier/draft pair is coherent; "
            "EAGLE-3 draft uses a separate model reference",
            {
                "pair": pair_name,
                "verifier": verifier,
                "draft": draft,
                "method": "eagle3",
                "layout": "separate",
                "draft_derived": explicit_draft is None,
            },
        )
    except Exception as exc:
        return Check(
            "model_pair",
            CheckStatus.FAIL,
            f"Could not validate model pair: {exc}",
        )


def plan_execution(
    gpu_probe: GPUProbe,
    *,
    prefer_colocated: bool = False,
) -> ExecutionPlan:
    """Choose a safe mode while capping tuning scratch VRAM at 5 GiB."""

    if gpu_probe.check.status is not CheckStatus.PASS or not gpu_probe.devices:
        return ExecutionPlan(
            ExecutionMode.UNAVAILABLE,
            "No usable NVIDIA GPU is available for auto-tuning",
        )

    usable = [
        (index, device)
        for index, device in enumerate(gpu_probe.devices)
        if device.memory_total_mib >= SCRATCH_LIMIT_MIB
    ]
    if not usable:
        return ExecutionPlan(
            ExecutionMode.UNAVAILABLE,
            f"No GPU has the enforced {SCRATCH_LIMIT_MIB} MiB scratch capacity",
        )

    best_index, best = max(usable, key=lambda item: item[1].memory_free_mib)
    if prefer_colocated and best.memory_free_mib >= SCRATCH_LIMIT_MIB:
        return ExecutionPlan(
            ExecutionMode.COLOCATED,
            f"GPU {best_index} has {best.memory_free_mib} MiB measured headroom, "
            f"enough for the capped {SCRATCH_LIMIT_MIB} MiB scratch budget",
            gpu_index=best_index,
            headroom_mib=best.memory_free_mib,
        )

    if prefer_colocated:
        reason = (
            f"colocated mode lacks {SCRATCH_LIMIT_MIB} MiB measured headroom; "
            "using vLLM level-1 sleep before tuning"
        )
    else:
        reason = "default safe mode; vLLM level-1 sleep frees VRAM before tuning"
    return ExecutionPlan(
        ExecutionMode.IDLE,
        reason,
        gpu_index=best_index,
        headroom_mib=best.memory_free_mib,
    )


def run_doctor(
    config: object,
    *,
    home: Path | None = None,
    minimum_disk_free_gb: float = DEFAULT_MIN_DISK_FREE_GB,
    timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS,
    prefer_colocated: bool = False,
) -> DoctorReport:
    """Run all read-only checks and return a complete execution plan."""

    layout = resolve_layout(home)
    gpu_probe = probe_gpu(timeout_seconds=timeout_seconds)
    checks = (
        check_python(),
        gpu_probe.check,
        check_cuda(gpu_probe, timeout_seconds=timeout_seconds),
        check_packages(),
        check_disk(layout.root, minimum_free_gb=minimum_disk_free_gb),
        check_memory(),
        check_model_pair(config),
    )
    return DoctorReport(
        checks=checks,
        plan=plan_execution(gpu_probe, prefer_colocated=prefer_colocated),
    )
