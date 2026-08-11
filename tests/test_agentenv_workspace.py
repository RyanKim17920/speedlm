"""GPU-free contract tests for the executable agent workspace.

Every refusal below feeds the real filesystem tool the condition it promises to
reject.  That matters more than a green happy-path assertion here: containment,
caps, and edit uniqueness are safety boundaries only if malformed work can make
them go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.agentenv.workspace import (
    MAX_READ_BYTES,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_WRITE_BYTES,
    ToolError,
    WorkspaceSandbox,
)

PROJECT_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"


def _sandbox(tmp_path: Path) -> WorkspaceSandbox:
    root = tmp_path / "workspace"
    root.mkdir()
    return WorkspaceSandbox(root, python=PROJECT_PYTHON)


def test_parent_traversal_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """Resolving before containment must reject ``..`` paths.

    RED demonstration: removing ``relative_to(self.root)`` makes this resolve
    successfully to the sibling file created by the fixture.
    """
    sandbox = _sandbox(tmp_path)
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ToolError, match="outside the workspace"):
        sandbox.read_file({"path": "../outside.txt"})


def test_absolute_outside_path_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """An absolute path is legal only when its resolved target is contained.

    RED demonstration: accepting all absolute paths exposes this real sibling
    file even though no traversal segment appears in the supplied string.
    """
    sandbox = _sandbox(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ToolError, match="outside the workspace"):
        sandbox.read_file({"path": str(outside.resolve())})


def test_symlink_cannot_indirectly_escape_the_workspace(tmp_path: Path) -> None:
    """Containment must be checked after following an in-workspace symlink.

    RED demonstration: checking the raw ``escape.txt`` path instead of its
    resolved target lets the tool read the external file.
    """
    sandbox = _sandbox(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (sandbox.root / "escape.txt").symlink_to(outside)

    with pytest.raises(ToolError, match="outside the workspace"):
        sandbox.read_file({"path": "escape.txt"})


def test_read_file_refuses_a_file_above_the_byte_cap(tmp_path: Path) -> None:
    """A model must not pull an arbitrarily large file into the prompt.

    The file is one byte over the limit, so changing ``>`` to ``>=`` is caught
    by the complementary boundary while removing the cap makes this test red.
    """
    sandbox = _sandbox(tmp_path)
    (sandbox.root / "large.txt").write_bytes(b"x" * (MAX_READ_BYTES + 1))

    with pytest.raises(ToolError, match=rf"{MAX_READ_BYTES}-byte read limit"):
        sandbox.read_file({"path": "large.txt"})


def test_write_file_measures_the_cap_in_bytes_not_characters(tmp_path: Path) -> None:
    """UTF-8 expansion must count against the write limit.

    This string has fewer characters than the cap but more encoded bytes.  A
    character-count implementation therefore accepts it and makes the test red.
    """
    sandbox = _sandbox(tmp_path)
    content = "é" * (MAX_WRITE_BYTES // 2 + 1)
    assert len(content) < MAX_WRITE_BYTES
    assert len(content.encode("utf-8")) > MAX_WRITE_BYTES

    with pytest.raises(ToolError, match="write limit"):
        sandbox.write_file({"path": "large.txt", "content": content})
    assert not (sandbox.root / "large.txt").exists()


def test_replace_in_file_refuses_a_missing_anchor(tmp_path: Path) -> None:
    """A zero-match edit is a failed edit, never a successful no-op.

    RED demonstration: deleting the occurrence-count guard makes the call
    report success while leaving these real file bytes unchanged.
    """
    sandbox = _sandbox(tmp_path)
    path = sandbox.root / "module.py"
    original = "value = 1\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="does not occur"):
        sandbox.replace_in_file(
            {"path": "module.py", "find": "value = 2", "replace": "value = 3"}
        )
    assert path.read_text(encoding="utf-8") == original


def test_replace_in_file_refuses_a_non_unique_anchor(tmp_path: Path) -> None:
    """An ambiguous edit must not silently choose one occurrence.

    RED demonstration: replacing only the first match changes this file instead
    of raising and is detected by both assertions.
    """
    sandbox = _sandbox(tmp_path)
    path = sandbox.root / "module.py"
    original = "flag = False\nflag = False\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="occurs 2 times"):
        sandbox.replace_in_file(
            {"path": "module.py", "find": "flag = False", "replace": "flag = True"}
        )
    assert path.read_text(encoding="utf-8") == original


def test_run_tests_reports_a_real_workspace_failure(tmp_path: Path) -> None:
    """The model-facing test tool must preserve a nonzero subprocess result.

    A canned-success implementation goes red because this deliberately failing
    pytest file is executed by the project interpreter in the real temp tree.
    """
    sandbox = _sandbox(tmp_path)
    test_dir = sandbox.root / "tests"
    test_dir.mkdir()
    (test_dir / "test_failure.py").write_text(
        "def test_deliberate_failure():\n    assert 1 == 2\n", encoding="utf-8"
    )

    result = sandbox.run_tests({"path": "tests/test_failure.py"})

    assert result.metadata == {"returncode": 1}
    assert "-> FAILED" in result.text
    assert "test_deliberate_failure" in result.text


def test_tool_output_is_clipped_while_preserving_both_ends(tmp_path: Path) -> None:
    """Large output must not evict the task, but its diagnostic tail must survive.

    The source stays below the read byte cap and exceeds only the prompt-output
    cap.  Removing ``_clip`` makes ``truncated`` false and the result oversized.
    """
    sandbox = _sandbox(tmp_path)
    content = "START-" + "x" * (MAX_TOOL_OUTPUT_CHARS + 100) + "-END"
    (sandbox.root / "verbose.txt").write_text(content, encoding="utf-8")

    result = sandbox.read_file({"path": "verbose.txt"})

    assert result.truncated
    assert len(result.text) <= MAX_TOOL_OUTPUT_CHARS
    assert "START-" in result.text
    assert "-END" in result.text
    assert "characters omitted from the middle" in result.text
