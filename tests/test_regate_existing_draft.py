"""Tests for the re-gate driver's pure guards (scripts/regate_existing_draft.py).

Only the parts that need no GPU: the baseline it will measure against, the
training hashes its leakage proof rests on, and its refusal to invent a suite.
Each of these is a way the experiment could report a confident number about the
wrong thing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "regate_existing_draft.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regate_existing_draft", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regate = _load_module()


def test_stock_draft_is_read_from_the_original_decision(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps({"stock_draft": "RedHatAI/Qwen3-8B-speculator.eagle3"}))
    assert regate._read_stock_draft(path) == "RedHatAI/Qwen3-8B-speculator.eagle3"


@pytest.mark.parametrize("payload", [{}, {"stock_draft": ""}, {"stock_draft": None}])
def test_missing_stock_draft_is_refused_not_defaulted(tmp_path: Path, payload) -> None:
    """A guessed baseline still produces a number, just not a comparable one."""
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(regate.RegateError):
        regate._read_stock_draft(path)


def test_empty_training_hashes_are_refused(tmp_path: Path) -> None:
    """An empty set satisfies the gate's leakage check trivially.

    That is the "test that cannot fail" shape: the gate would report a proof it
    never actually ran against anything.
    """
    path = tmp_path / "hashes.json"
    path.write_text(json.dumps({"training_context_hashes": []}))
    with pytest.raises(regate.RegateError):
        regate._read_training_hashes(path)


def test_real_training_hashes_load(tmp_path: Path) -> None:
    path = tmp_path / "hashes.json"
    path.write_text(json.dumps({"training_context_hashes": ["a" * 64, "b" * 64]}))
    assert regate._read_training_hashes(path) == frozenset({"a" * 64, "b" * 64})


def test_trace_source_refuses_to_be_consulted() -> None:
    """Building a suite here would benchmark contexts nobody chose."""
    with pytest.raises(regate.RegateError):
        list(regate._RefusingTraceSource().iter_records())
