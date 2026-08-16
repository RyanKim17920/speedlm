"""Second batch of task families for the agent environment catalog.

Six new families that widen the prompt distribution beyond the original six:

``flaky-test-quarantine``
    A test suite where one test is order-dependent due to in-place mutation of
    shared data. Agent must find and fix the mutation site.

``dep-version-conflict``
    A module whose version assertion pins an incompatible range; a shim fails
    to import. Agent must resolve the constraint.

``api-contract-drift``
    Client and server disagree on a field name after a rename. Agent must
    update the client to the new contract.

``perf-hotspot``
    A function with accidental O(n^2) from repeated list scans. Agent must
    make it linear. Graded by operation-count instrumentation.

``config-precedence-bug``
    Layered config where one layer is applied in the wrong order. Agent must
    fix the precedence chain.

``error-swallow-audit``
    A module with several ``except: pass`` sites, exactly one of which hides a
    real failure. Agent must make that one propagate.

Determinism
-----------
Every family uses its own ``random.Random(seed ^ SALT)`` with a distinct hex
salt.  Nothing samples from the global RNG and nothing reads the clock.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Final

from tests.e2e.agentenv.phrasing import Brief, Fact, FactKind, render_instruction
from tests.e2e.agentenv.tasks import Grade, Task, TaskInstance, Workspace
from tests.e2e.agentenv.workspace import WorkspaceSandbox

__all__ = ["CATALOG_V2", "TASKS_V2", "task_by_name"]


# ---------------------------------------------------------------------------
# 1. flaky-test-quarantine
# ---------------------------------------------------------------------------
_FLAKY_SALT: Final[int] = 0xF141  # F(L)AKY


def _flaky_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _FLAKY_SALT)
    pkg = rng.choice(["datakit", "ledgex", "streamz"])
    cls_name = rng.choice(["Processor", "Engine", "Transformer"])
    count = rng.choice([5, 7])
    values = list(range(100, 100 + count * 3))

    if bug:
        # BUG: shuffles DATA_POINTS in-place, so second call gets a permuted list
        process_body = (
            "        random.shuffle(DATA_POINTS)\n"
            "        return [p for p in DATA_POINTS if p >= 100][:N]\n"
        )
    else:
        # FIX: sort instead of shuffle — deterministic ordering
        process_body = (
            "        sorted_points = sorted(DATA_POINTS)\n"
            "        return [p for p in sorted_points if p >= 100][:N]\n"
        )

    service = (
        f'"""Data processing service for {pkg}."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"import random\n"
        f"\n"
        f"DATA_POINTS = {values}\n"
        f"N = {count}\n"
        f"\n"
        f"\n"
        f"class {cls_name}:\n"
        f'    """Process a subset of the data points."""\n'
        f"\n"
        f"    def process(self) -> list[int]:\n"
        f'        """Return a deterministic sample of DATA_POINTS."""\n'
        f"{process_body}"
        f"\n"
        f"    def count(self) -> int:\n"
        f"        return len(DATA_POINTS)\n"
    )

    tests = (
        f"import random\n"
        f"\n"
        f"from {pkg}.service import {cls_name}, DATA_POINTS\n"
        f"\n"
        f"\n"
        f"def test_process_output():\n"
        f"    engine = {cls_name}()\n"
        f"    result = engine.process()\n"
        f"    assert len(result) == {count}\n"
        f"\n"
        f"\n"
        f"def test_data_points_not_mutated():\n"
        f'    """DATA_POINTS must not be shuffled in-place."""\n'
        f"    original = DATA_POINTS.copy()\n"
        f"    engine = {cls_name}()\n"
        f"    engine.process()\n"
        f"    assert DATA_POINTS == original\n"
    )

    return {
        f"{pkg}/__init__.py": f'"""{pkg} package."""\n',
        f"{pkg}/service.py": service,
        "tests/test_service.py": tests,
        "README.md": f"# {pkg}\n\nRun the suite with `pytest`.\n",
    }


def _flaky_instance(seed: int) -> TaskInstance:
    files = _flaky_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _FLAKY_SALT)
    pkg = rng.choice(["datakit", "ledgex", "streamz"])
    cls_name = rng.choice(["Processor", "Engine", "Transformer"])

    # Advance rng by the same draws _flaky_files uses, then read count.
    count = rng.choice([5, 7])

    brief = Brief(
        goal="identify the flaky test and make the suite deterministic",
        required_facts=(
            Fact(text="tests/test_service.py", kind=FactKind.PATH),
            Fact(text=f"{pkg}/", kind=FactKind.PATH),
            Fact(text=cls_name, kind=FactKind.SYMBOL),
            Fact(text=str(count), kind=FactKind.NUMBER),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            f"The {pkg} package has a test suite that is flaky: one test depends "
            "on execution order because shared state is mutated in-place."
        ),
        constraints=(
            "Run `pytest` to check the suite.",
            f"Fix the source in {pkg}/service.py, not the tests.",
        ),
        has_test_suite=True,
        blame_path=f"{pkg}/service.py",
    )
    return TaskInstance(
        id=f"flaky-test-quarantine-{seed:04d}",
        family="flaky-test-quarantine",
        instruction=render_instruction(brief, seed=seed, salt=_FLAKY_SALT),
        workspace=Workspace(files=files),
        grader=_flaky_grader(pkg, cls_name, seed),
        metadata={"package": pkg, "class_name": cls_name, "seed": seed},
    )


def _flaky_grader(pkg: str, cls_name: str, seed: int) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        service_path = sandbox.root / pkg / "service.py"
        source = service_path.read_text(encoding="utf-8") if service_path.is_file() else ""
        no_shuffle_inplace = "random.shuffle(DATA_POINTS)" not in source
        passed, output = sandbox.tests_pass("tests/test_service.py")
        checks = {
            "pytest": passed,
            "no_shuffle_inplace": no_shuffle_inplace,
        }
        solved = passed and no_shuffle_inplace
        return Grade(
            solved=solved,
            detail="suite passes and mutation is fixed"
            if solved
            else f"checks={checks}; pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _flaky_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _flaky_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# 2. dep-version-conflict
# ---------------------------------------------------------------------------
_DEP_SALT: Final[int] = 0xD3F2  # (D)E(F)2


def _dep_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _DEP_SALT)
    service = rng.choice(["datacore", "eventhub", "configserver"])
    installed_ver = f"{rng.choice([1, 2])}.{rng.choice([0, 5])}.0"
    min_ver = f"{int(installed_ver.split('.')[0]) + 1}.0.0" if bug else "0.1.0"

    versions = (
        f'"""Version pinning for {service}."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f'MIN_VERSION = "{min_ver}"\n'
        f'INSTALLED_VERSION = "{installed_ver}"\n'
        f"\n"
        f"\n"
        f"def check_compatible() -> bool:\n"
        f'    """Check that the installed version meets the minimum."""\n'
        f"    def _tuple(v: str) -> tuple[int, ...]:\n"
        f'        return tuple(int(p) for p in v.split("."))\n'
        f"    return _tuple(INSTALLED_VERSION) >= _tuple(MIN_VERSION)\n"
    )

    shim = (
        f'"""Shim module that wraps {service} functionality."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from .versions import check_compatible\n"
        f"\n"
        f"if not check_compatible():\n"
        f"    raise ImportError(\n"
        f'        f"{{INSTALLED_VERSION!r}} does not meet minimum '
        f'{{MIN_VERSION!r}} for {service}"\n'
        f"    )\n"
        f"\n"
        f"\n"
        f"def connect() -> str:\n"
        f'    return "connected"\n'
    )

    tests = (
        f"from {service}.shim import connect\n"
        f"\n"
        f"\n"
        f"def test_shim_connects():\n"
        f'    assert connect() == "connected"\n'
    )

    return {
        f"{service}/__init__.py": f'"""{service} package."""\n',
        f"{service}/versions.py": versions,
        f"{service}/shim.py": shim,
        "tests/test_shim.py": tests,
        "README.md": f"# {service}\n\nRun the suite with `pytest`.\n",
    }


def _dep_instance(seed: int) -> TaskInstance:
    files = _dep_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _DEP_SALT)
    service = rng.choice(["datacore", "eventhub", "configserver"])
    installed_ver = f"{rng.choice([1, 2])}.{rng.choice([0, 5])}.0"

    brief = Brief(
        goal="resolve the dependency version conflict so the shim imports successfully",
        required_facts=(
            Fact(text=f"{service}/", kind=FactKind.PATH),
            Fact(text="tests/test_shim.py", kind=FactKind.PATH),
            Fact(text=service, kind=FactKind.TOKEN),
            Fact(text=installed_ver, kind=FactKind.TOKEN),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            f"The {service} package has a version constraint that rejects the "
            "installed version, causing the shim module to fail on import."
        ),
        constraints=("Run `pytest` to confirm the suite passes.",),
        has_test_suite=True,
    )
    return TaskInstance(
        id=f"dep-version-conflict-{seed:04d}",
        family="dep-version-conflict",
        instruction=render_instruction(brief, seed=seed, salt=_DEP_SALT),
        workspace=Workspace(files=files),
        grader=_dep_grader(service),
        metadata={"service": service, "seed": seed},
    )


def _dep_grader(service: str) -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass("tests/test_shim.py")
        checks = {"pytest": passed}
        return Grade(
            solved=passed,
            detail="shim imports and test passes"
            if passed
            else f"pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _dep_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _dep_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# 3. api-contract-drift
# ---------------------------------------------------------------------------
_API_SALT: Final[int] = 0xA2CD


def _api_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _API_SALT)
    field_old = rng.choice(["address", "host", "origin"])
    field_new = (
        "updated_address"
        if field_old == "address"
        else ("resolved_host" if field_old == "host" else "canonical_origin")
    )
    rng.randrange(3000, 9000)  # advance RNG state for variation
    client_field = field_old if bug else field_new

    server = (
        f'"""Server module — the source of truth for the API contract."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from dataclasses import dataclass\n"
        f"\n"
        f"\n"
        f"@dataclass(frozen=True, slots=True)\n"
        f"class Endpoint:\n"
        f'    """A single service endpoint."""\n'
        f"    {field_new}: str\n"
        f"    port: int\n"
    )

    client = (
        f'"""Client module — consumes the server Endpoint type."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from .server import Endpoint\n"
        f"\n"
        f"\n"
        f"def build_endpoint(host: str, port: int) -> Endpoint:\n"
        f'    """Create an Endpoint from connection parameters."""\n'
        f"    return Endpoint({client_field}=host, port=port)\n"
        f"\n"
        f"\n"
        f"def get_host(ep: Endpoint) -> str:\n"
        f'    """Extract the host string from an Endpoint."""\n'
        f"    return ep.{client_field}\n"
    )

    tests = (
        "from apicontract.client import build_endpoint, get_host\n"
        "from apicontract.server import Endpoint\n"
        "\n"
        "\n"
        "def test_build_and_extract():\n"
        '    ep = build_endpoint("localhost", 8080)\n'
        '    assert get_host(ep) == "localhost"\n'
    )

    return {
        "apicontract/__init__.py": '"""API contract package."""\n',
        "apicontract/server.py": server,
        "apicontract/client.py": client,
        "tests/test_contract.py": tests,
        "README.md": "# apicontract\n\nRun the suite with `pytest`.\n",
    }


def _api_instance(seed: int) -> TaskInstance:
    files = _api_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _API_SALT)
    field_old = rng.choice(["address", "host", "origin"])
    field_new = (
        "updated_address"
        if field_old == "address"
        else ("resolved_host" if field_old == "host" else "canonical_origin")
    )

    brief = Brief(
        goal=(
            f"update the client to use the new field name `{field_new}` "
            f"(the old name `{field_old}` was renamed in the server)"
        ),
        required_facts=(
            Fact(text="apicontract/", kind=FactKind.PATH),
            Fact(text="tests/test_contract.py", kind=FactKind.PATH),
            Fact(text=field_old, kind=FactKind.TOKEN),
            Fact(text=field_new, kind=FactKind.TOKEN),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            "The server module was updated with a field rename, but the client "
            "is still using the old field name, causing AttributeErrors at runtime."
        ),
        constraints=("Run `pytest` to confirm the round-trip tests pass.",),
        has_test_suite=True,
        creates=(field_new,),
    )
    return TaskInstance(
        id=f"api-contract-drift-{seed:04d}",
        family="api-contract-drift",
        instruction=render_instruction(brief, seed=seed, salt=_API_SALT),
        workspace=Workspace(files=files),
        grader=_api_grader(),
        metadata={"seed": seed},
    )


def _api_grader() -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass("tests/test_contract.py")
        client_path = sandbox.root / "apicontract" / "client.py"
        client_src = client_path.read_text(encoding="utf-8") if client_path.is_file() else ""
        uses_new = any(
            x in client_src for x in ("updated_address", "resolved_host", "canonical_origin")
        )
        checks = {
            "pytest": passed,
            "client_uses_new_field": uses_new,
        }
        solved = passed and uses_new
        return Grade(
            solved=solved,
            detail="client updated and tests pass"
            if solved
            else f"checks={checks}; pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _api_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _api_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# 4. perf-hotspot
# ---------------------------------------------------------------------------
_PERF_SALT: Final[int] = 0x512E  # (S)IZE -> perf


def _perf_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _PERF_SALT)
    cls_name = rng.choice(["Filter", "Gatekeeper", "Validator"])
    method_name = rng.choice(["process", "validate", "admit"])
    allowed_count = rng.randrange(10, 40)
    limit = allowed_count + 50

    if bug:
        core = (
            f"    def {method_name}(self, items, _on_check=None):\n"
            f"        results = []\n"
            f"        for item in items:\n"
            f"            for a in self.allowed:\n"
            f"                if _on_check is not None:\n"
            f"                    _on_check(item)\n"
            f"                if item == a:\n"
            f"                    results.append(item)\n"
            f"                    break\n"
            f"        return results\n"
        )
    else:
        core = (
            f"    def {method_name}(self, items, _on_check=None):\n"
            f"        allowed_set = set(self.allowed)\n"
            f"        results = []\n"
            f"        for item in items:\n"
            f"            if item in allowed_set:\n"
            f"                if _on_check is not None:\n"
            f"                    _on_check(item)\n"
            f"                results.append(item)\n"
            f"        return results\n"
        )

    service = (
        f'"""Performance-sensitive service."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"\n"
        f"class {cls_name}:\n"
        f'    """Filter items against an allowlist."""\n'
        f"\n"
        f"    def __init__(self, allowed: list[int]) -> None:\n"
        f"        self.allowed = allowed\n"
        f"\n"
        f"{core}"
    )

    tests = (
        f"from fastcheck.service import {cls_name}\n"
        f"\n"
        f"\n"
        f"def test_correctness():\n"
        f"    svc = {cls_name}([1, 2, 3])\n"
        f"    assert svc.{method_name}([0, 1, 2, 3, 4]) == [1, 2, 3]\n"
        f"\n"
        f"\n"
        f"def test_empty_input():\n"
        f"    svc = {cls_name}([1, 2, 3])\n"
        f"    assert svc.{method_name}([]) == []\n"
        f"\n"
        f"\n"
        f"def test_operation_count():\n"
        f'    """Instrument each _on_check callback to count valid lookups.\n'
        f"\n"
        f"    O(n) implementation: checks == number of matching items.\n"
        f"    O(n^2) implementation: checks >> number of matching items.\n"
        f'    """\n'
        f"    checks: list = []\n"
        f"    allowed = list(range(1, {allowed_count} + 1))\n"
        f"    svc = {cls_name}(allowed)\n"
        f"    result = svc.{method_name}(list(range(50)), _on_check=checks.append)\n"
        f"    assert set(result) == set(allowed)\n"
        f'    assert len(checks) < {limit}, f"too many checks: {{len(checks)}} > {limit}"\n'
    )

    return {
        "fastcheck/__init__.py": '"""Fast check package."""\n',
        "fastcheck/service.py": service,
        "tests/test_perf.py": tests,
        "README.md": (
            f"# fastcheck\n\n"
            f"`{cls_name}.{method_name}` filters items against an allowlist.\n\n"
            f"Run the suite with `pytest`.\n"
        ),
    }


def _perf_instance(seed: int) -> TaskInstance:
    files = _perf_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _PERF_SALT)
    cls_name = rng.choice(["Filter", "Gatekeeper", "Validator"])
    method_name = rng.choice(["process", "validate", "admit"])
    allowed_count = rng.randrange(10, 40)
    limit = allowed_count + 50

    brief = Brief(
        goal=(
            f"fix the O(n^2) performance issue in `{cls_name}.{method_name}` "
            "by using a set-based lookup instead of repeated list scans"
        ),
        required_facts=(
            Fact(text="fastcheck/", kind=FactKind.PATH),
            Fact(text="tests/test_perf.py", kind=FactKind.PATH),
            Fact(text=f"{cls_name}.{method_name}", kind=FactKind.SYMBOL),
            Fact(text=str(allowed_count), kind=FactKind.NUMBER),
            Fact(text=str(limit), kind=FactKind.NUMBER),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            f"The `{cls_name}.{method_name}` method uses nested loops over the "
            "allowlist and input, making it O(n*m) instead of O(n+m)."
        ),
        constraints=("Run `pytest` to confirm all tests pass including operation count.",),
        has_test_suite=True,
    )
    return TaskInstance(
        id=f"perf-hotspot-{seed:04d}",
        family="perf-hotspot",
        instruction=render_instruction(brief, seed=seed, salt=_PERF_SALT),
        workspace=Workspace(files=files),
        grader=_perf_grader(),
        metadata={"seed": seed},
    )


def _perf_grader() -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass("tests/test_perf.py")
        checks = {"pytest": passed}
        return Grade(
            solved=passed,
            detail="all tests pass including operation count"
            if passed
            else f"pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _perf_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _perf_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# 5. config-precedence-bug
# ---------------------------------------------------------------------------
_CFG_SALT: Final[int] = 0x4F17


def _cfg_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _CFG_SALT)
    setting = rng.choice(["timeout", "retries", "batch_size"])
    default_val = rng.choice([10, 30, 60])
    env_val = rng.choice([100, 300, 600])
    cli_val = rng.choice([200, 400, 800])

    if bug:
        merge_body = (
            "    config = dict(defaults)\n"
            "    config.update(env_overrides)\n"
            "    config.update(cli_overrides)\n"
            "    config.update(defaults)  # bug: re-applies defaults last\n"
            "    return config\n"
        )
    else:
        merge_body = (
            "    config = dict(defaults)\n"
            "    config.update(env_overrides)\n"
            "    config.update(cli_overrides)\n"
            "    return config\n"
        )

    config_py = (
        f'"""Layered configuration: defaults < env < cli."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from typing import Any\n"
        f"\n"
        f"\n"
        f"def merge_config(\n"
        f"    defaults: dict[str, Any],\n"
        f"    env_overrides: dict[str, Any],\n"
        f"    cli_overrides: dict[str, Any],\n"
        f") -> dict[str, Any]:\n"
        f'    """Merge configuration layers in precedence order.\n'
        f"\n"
        f"    Higher-precedence layers override lower ones:\n"
        f"    defaults < env_overrides < cli_overrides\n"
        f'    """\n'
        f"{merge_body}"
    )

    tests = (
        f"from appconfig.config import merge_config\n"
        f"\n"
        f"\n"
        f"SETTING = {setting!r}\n"
        f"DEFAULT = {{{setting!r}: {default_val}}}\n"
        f"ENV_VAL = {{{setting!r}: {env_val}}}\n"
        f"CLI_VAL = {{{setting!r}: {cli_val}}}\n"
        f"\n"
        f"\n"
        f"def test_cli_overrides_env():\n"
        f"    result = merge_config(DEFAULT, ENV_VAL, CLI_VAL)\n"
        f"    assert result[SETTING] == {cli_val}\n"
        f"\n"
        f"\n"
        f"def test_env_overrides_default():\n"
        f"    result = merge_config(DEFAULT, ENV_VAL, {{}})\n"
        f"    assert result[SETTING] == {env_val}\n"
        f"\n"
        f"\n"
        f"def test_default_when_no_overrides():\n"
        f"    result = merge_config(DEFAULT, {{}}, {{}})\n"
        f"    assert result[SETTING] == {default_val}\n"
        f"\n"
        f"\n"
        f"def test_cli_overrides_all():\n"
        f"    result = merge_config(DEFAULT, ENV_VAL, CLI_VAL)\n"
        f"    expected = {cli_val}\n"
        f"    assert result[SETTING] == expected, (\n"
        f'        f"expected cli value {{expected!r}}, got {{result[SETTING]!r}}"\n'
        f"    )\n"
    )

    return {
        "appconfig/__init__.py": '"""App config package."""\n',
        "appconfig/config.py": config_py,
        "tests/test_config.py": tests,
        "README.md": "# appconfig\n\nRun the suite with `pytest`.\n",
    }


def _cfg_instance(seed: int) -> TaskInstance:
    files = _cfg_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _CFG_SALT)
    setting = rng.choice(["timeout", "retries", "batch_size"])
    default_val = rng.choice([10, 30, 60])
    env_val = rng.choice([100, 300, 600])
    cli_val = rng.choice([200, 400, 800])

    brief = Brief(
        goal="fix the config precedence bug so that CLI overrides take priority over defaults",
        required_facts=(
            Fact(text="appconfig/", kind=FactKind.PATH),
            Fact(text="tests/test_config.py", kind=FactKind.PATH),
            Fact(text=setting, kind=FactKind.TOKEN),
            Fact(text=str(default_val), kind=FactKind.NUMBER),
            Fact(text=str(env_val), kind=FactKind.NUMBER),
            Fact(text=str(cli_val), kind=FactKind.NUMBER),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            f"The config merger applies the `{setting}` layer in the wrong order, "
            "causing defaults to override CLI arguments."
        ),
        constraints=("Run `pytest` to confirm all precedence tests pass.",),
        has_test_suite=True,
        blame_path="appconfig/config.py",
    )
    return TaskInstance(
        id=f"config-precedence-bug-{seed:04d}",
        family="config-precedence-bug",
        instruction=render_instruction(brief, seed=seed, salt=_CFG_SALT),
        workspace=Workspace(files=files),
        grader=_cfg_grader(),
        metadata={"seed": seed},
    )


def _cfg_grader() -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass("tests/test_config.py")
        config_path = sandbox.root / "appconfig" / "config.py"
        source = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
        no_reapply_defaults = source.count("config.update(defaults)") == 0
        checks = {
            "pytest": passed,
            "no_reapply_defaults": no_reapply_defaults,
        }
        solved = passed and no_reapply_defaults
        return Grade(
            solved=solved,
            detail="precedence is correct and tests pass"
            if solved
            else f"checks={checks}; pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _cfg_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _cfg_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# 6. error-swallow-audit
# ---------------------------------------------------------------------------
_ERR_SALT: Final[int] = 0xE355  # (E)RR(o)r -> E355


def _err_files(*, bug: bool, seed: int) -> dict[str, str]:
    rng = random.Random(seed ^ _ERR_SALT)
    err_type = ["ZeroDivisionError", "OverflowError", "TypeError"][seed % 3]
    err_msg = [
        "cannot divide by zero",
        "numeric result exceeds bounds",
        "incompatible operand types",
    ][seed % 3]
    svc_name = rng.choice(["Worker", "Handler", "Dispatcher"])

    if bug:
        # BUG: bare except catches ValueError which should propagate
        process_handler = (
            "    def process_data(self, data: dict) -> dict | None:\n"
            "        try:\n"
            "            result = {}\n"
            "            for k, v in data.items():\n"
            "                result[k] = v / data.get('denominator', 1)\n"
            "            if data.get('raise_error'):\n"
            "                raise ValueError('error_msg_PLACEHOLDER')\n"
            "            return result\n"
            "        except:\n"
            "            pass\n"
            "        return None\n"
        ).replace("error_msg_PLACEHOLDER", err_msg)
    else:
        # FIX: catch only the specific benign exception type
        process_handler = (
            (
                "    def process_data(self, data: dict) -> dict | None:\n"
                "        try:\n"
                "            result = {}\n"
                "            for k, v in data.items():\n"
                "                result[k] = v / data.get('denominator', 1)\n"
                "            if data.get('raise_error'):\n"
                "                raise ValueError('error_msg_PLACEHOLDER')\n"
                "            return result\n"
                "        except err_type_PLACEHOLDER:\n"
                "            pass\n"
                "        return None\n"
            )
            .replace("error_msg_PLACEHOLDER", err_msg)
            .replace("err_type_PLACEHOLDER", err_type)
        )

    service = (
        f'"""Service with error handling that needs auditing."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"import logging\n"
        f"\n"
        f"logger = logging.getLogger(__name__)\n"
        f"\n"
        f"\n"
        f"class {svc_name}:\n"
        f'    """Handle various operations with error recovery."""\n'
        f"\n"
        f"    def read_config(self, path: str) -> str | None:\n"
        f"        try:\n"
        f"            with open(path) as f:\n"
        f"                return f.read()\n"
        f"        except FileNotFoundError:\n"
        f'            logger.warning("config %s not found", path)\n'
        f"            return None\n"
        f"\n"
        f"    def connect_network(self, host: str) -> bool:\n"
        f"        try:\n"
        f'            return host.startswith("valid-")\n'
        f"        except Exception:\n"
        f'            logger.warning("network failure for %s", host)\n'
        f"            return False\n"
        f"\n"
        f"    def log_message(self, msg: str) -> None:\n"
        f"        try:\n"
        f"            logger.info(msg)\n"
        f"        except Exception:\n"
        f"            pass\n"
        f"\n"
        f"{process_handler}"
    )

    tests = (
        f"from errorsvc.service import {svc_name}\n"
        f"\n"
        f"import pytest\n"
        f"\n"
        f"\n"
        f"def test_read_missing_config_returns_none():\n"
        f"    svc = {svc_name}()\n"
        f'    assert svc.read_config("nonexistent_file.txt") is None\n'
        f"\n"
        f"\n"
        f"def test_connect_invalid_host():\n"
        f"    svc = {svc_name}()\n"
        f'    assert svc.connect_network("invalid-host") is False\n'
        f"\n"
        f"\n"
        f"def test_log_silent():\n"
        f"    svc = {svc_name}()\n"
        f'    svc.log_message("test message")  # should not raise\n'
        f"\n"
        f"\n"
        f"def test_process_data_raises_on_error():\n"
        f"    svc = {svc_name}()\n"
        f'    pytest.raises(ValueError, svc.process_data, {{"raise_error": True, "a": 1}})\n'
    )

    return {
        "errorsvc/__init__.py": '"""Error service package."""\n',
        "errorsvc/service.py": service,
        "tests/test_errors.py": tests,
        "README.md": "# errorsvc\n\nRun the suite with `pytest`.\n",
    }


def _err_instance(seed: int) -> TaskInstance:
    files = _err_files(bug=True, seed=seed)
    rng = random.Random(seed ^ _ERR_SALT)
    svc_name = rng.choice(["Worker", "Handler", "Dispatcher"])
    err_type = ["ZeroDivisionError", "OverflowError", "TypeError"][seed % 3]
    err_msg = [
        "cannot divide by zero",
        "numeric result exceeds bounds",
        "incompatible operand types",
    ][seed % 3]

    brief = Brief(
        goal=(
            "audit the error handling in the service module: find the one "
            "``except: pass`` that hides a real failure and make it propagate, "
            "while keeping the benign ones quiet"
        ),
        required_facts=(
            Fact(text="errorsvc/", kind=FactKind.PATH),
            Fact(text="tests/test_errors.py", kind=FactKind.PATH),
            Fact(text=svc_name, kind=FactKind.SYMBOL),
            Fact(text=err_type, kind=FactKind.TOKEN),
            Fact(text=err_msg, kind=FactKind.TOKEN),
            Fact(text="pytest", kind=FactKind.COMMAND),
        ),
        context=(
            f"The `{svc_name}` class has several bare ``except: pass`` handlers. "
            "Most are intentional graceful degradations, but one swallows an error "
            "that should propagate to callers."
        ),
        constraints=("Run `pytest` to confirm all tests pass.",),
        has_test_suite=True,
        creates=(err_type,),
    )
    return TaskInstance(
        id=f"error-swallow-audit-{seed:04d}",
        family="error-swallow-audit",
        instruction=render_instruction(brief, seed=seed, salt=_ERR_SALT),
        workspace=Workspace(files=files),
        grader=_err_grader(),
        metadata={"seed": seed},
    )


def _err_grader() -> Any:
    def grade(sandbox: WorkspaceSandbox) -> Grade:
        passed, output = sandbox.tests_pass("tests/test_errors.py")
        svc_path = sandbox.root / "errorsvc" / "service.py"
        source = svc_path.read_text(encoding="utf-8") if svc_path.is_file() else ""
        no_bare_except = "except:\n" not in source and "except :" not in source
        checks = {
            "pytest": passed,
            "no_bare_except": no_bare_except,
        }
        solved = passed and no_bare_except
        return Grade(
            solved=solved,
            detail="errors audited and tests pass"
            if solved
            else f"checks={checks}; pytest tail: {output.strip()[-800:]}",
            checks=checks,
        )

    return grade


def _err_solution(seed: int) -> Any:
    def solve(sandbox: WorkspaceSandbox) -> None:
        clean = _err_files(bug=False, seed=seed)
        for path, content in clean.items():
            (sandbox.root / path).write_text(content, encoding="utf-8")

    return solve


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TASKS_V2: Final[tuple[Task, ...]] = (
    Task(
        name="flaky-test-quarantine",
        summary=(
            "One test in a suite is order-dependent due to in-place mutation "
            "of shared data; agent must find and fix the mutation."
        ),
        build=_flaky_instance,
    ),
    Task(
        name="dep-version-conflict",
        summary=(
            "A version constraint pins an incompatible range, causing a shim "
            "module to fail on import; agent must resolve the constraint."
        ),
        build=_dep_instance,
    ),
    Task(
        name="api-contract-drift",
        summary=(
            "Client still uses an old field name after the server renamed it; "
            "agent must update the client to the new contract."
        ),
        build=_api_instance,
    ),
    Task(
        name="perf-hotspot",
        summary=(
            "A function uses O(n^2) list scans instead of set lookups; "
            "agent must optimize, graded by operation-count instrumentation."
        ),
        build=_perf_instance,
    ),
    Task(
        name="config-precedence-bug",
        summary=(
            "Layered config merger applies defaults in the wrong order, "
            "overriding higher-precedence layers."
        ),
        build=_cfg_instance,
    ),
    Task(
        name="error-swallow-audit",
        summary=(
            "Several bare ``except: pass`` handlers, exactly one hiding a real "
            "failure that should propagate."
        ),
        build=_err_instance,
    ),
)

#: Reference solutions, used only by the solvability pre-flight.
CATALOG_V2: Final[Mapping[str, Any]] = {
    "flaky-test-quarantine": _flaky_solution,
    "dep-version-conflict": _dep_solution,
    "api-contract-drift": _api_solution,
    "perf-hotspot": _perf_solution,
    "config-precedence-bug": _cfg_solution,
    "error-swallow-audit": _err_solution,
}


def task_by_name(name: str) -> Task:
    for task in TASKS_V2:
        if task.name == name:
            return task
    known = ", ".join(task.name for task in TASKS_V2)
    raise KeyError(f"unknown task {name!r}; available: {known}")
