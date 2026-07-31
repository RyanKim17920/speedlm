"""Tests for the prompt-corpus helpers used in test_live_idle_tuning.

We inline the helpers rather than importing from the e2e module to avoid
e2e-level dependencies (httpx) and import-path issues with nested test dirs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ── inline helpers (mirror of tests/e2e/test_live_idle_tuning) ──────────────


def _load_prompt_corpus() -> list[str] | None:
    corpus_path = os.environ.get("SPEEDLM_E2E_PROMPT_CORPUS")
    if corpus_path is None:
        return None
    path = Path(corpus_path).expanduser().resolve()
    assert path.is_file(), f"SPEEDLM_E2E_PROMPT_CORPUS is not a file: {path}"
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        obj: object = json.loads(stripped)
        assert isinstance(obj, dict), f"corpus line is not a JSON object: {stripped[:80]}"
        messages = obj.get("messages")
        assert isinstance(messages, list) and messages, (
            f"corpus line missing 'messages': {stripped[:80]}"
        )
        first = messages[0]
        assert isinstance(first, dict) and first.get("role") == "user", (
            f"first message must be user role: {stripped[:80]}"
        )
        content = first.get("content")
        assert isinstance(content, str) and content.strip(), (
            f"empty user content: {stripped[:80]}"
        )
        prompts.append(content)
    return prompts


def _select_prompts(
    corpus: list[str] | None, *, seed_count: int
) -> list[str]:
    if corpus is None:
        return [
            f"This is idle-tuning seed request {i + 1}/{seed_count}. "
            f"Reply with one short sentence."
            for i in range(seed_count)
        ]
    if len(corpus) < seed_count:
        raise AssertionError(
            f"prompt corpus has {len(corpus)} prompts but "
            f"{seed_count} are needed; set SPEEDLM_E2E_SEED_REQUESTS "
            f"<= {len(corpus)} or use a larger corpus"
        )
    return corpus[:seed_count]


# ── tests ────────────────────────────────────────────────────────────────────


class TestLoadPromptCorpus:
    def test_returns_none_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
        assert _load_prompt_corpus() is None

    def test_parses_valid_jsonl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        corpus = tmp_path / "prompts.jsonl"
        lines = [
            {"messages": [{"role": "user", "content": "What is Python?"}]},
            {"messages": [{"role": "user", "content": "Explain gravity."}]},
        ]
        corpus.write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        result = _load_prompt_corpus()
        assert result is not None
        assert result == ["What is Python?", "Explain gravity."]

    def test_skips_blank_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        corpus = tmp_path / "prompts.jsonl"
        corpus.write_text(
            json.dumps({"messages": [{"role": "user", "content": "hello"}]})
            + "\n\n"
            + json.dumps({"messages": [{"role": "user", "content": "world"}]})
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        assert _load_prompt_corpus() == ["hello", "world"]

    def test_raises_on_missing_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", "/no/such/file.jsonl")
        with pytest.raises(AssertionError, match="not a file"):
            _load_prompt_corpus()

    def test_raises_on_non_json_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text('["not", "an", "object"]\n', encoding="utf-8")
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        with pytest.raises(AssertionError, match="not a JSON object"):
            _load_prompt_corpus()

    def test_raises_on_missing_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text('{"prompt": "hello"}\n', encoding="utf-8")
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        with pytest.raises(AssertionError, match="missing"):
            _load_prompt_corpus()

    def test_raises_on_non_user_role(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text(
            '{"messages": [{"role": "assistant", "content": "hi"}]}\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        with pytest.raises(AssertionError, match="user role"):
            _load_prompt_corpus()

    def test_raises_on_empty_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text(
            '{"messages": [{"role": "user", "content": "  "}]}\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
        with pytest.raises(AssertionError, match="empty"):
            _load_prompt_corpus()


class TestSelectPrompts:
    def test_synthetic_fallback_when_corpus_none(self) -> None:
        result = _select_prompts(None, seed_count=3)
        assert len(result) == 3
        assert result[0] == "This is idle-tuning seed request 1/3. Reply with one short sentence."
        assert result[2] == "This is idle-tuning seed request 3/3. Reply with one short sentence."

    def test_deterministic_prefix(self) -> None:
        corpus = ["alpha", "beta", "gamma", "delta"]
        result = _select_prompts(corpus, seed_count=2)
        assert result == ["alpha", "beta"]

    def test_same_subset_every_time(self) -> None:
        corpus = ["first", "second", "third", "fourth"]
        a = _select_prompts(corpus, seed_count=3)
        b = _select_prompts(corpus, seed_count=3)
        assert a == b == ["first", "second", "third"]

    def test_raises_when_corpus_too_small(self) -> None:
        corpus = ["only", "two"]
        with pytest.raises(AssertionError, match="corpus has 2"):
            _select_prompts(corpus, seed_count=5)