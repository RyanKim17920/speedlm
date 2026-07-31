"""Unit tests for the e2e seed-request resolution logic."""

from __future__ import annotations

import os

from speedlm.config import IdleTuningConfig, SpeedLMConfig


def _seed_requests(config: SpeedLMConfig) -> int:
    """Replicate the e2e harness's seed-count resolver for unit testing."""
    raw = os.environ.get("SPEEDLM_E2E_SEED_REQUESTS")
    if raw is not None:
        value = int(raw)
        assert value > 0, f"SPEEDLM_E2E_SEED_REQUESTS must be positive, got {raw!r}"
        return value
    return max(config.tuning.min_trace_records, config.tuning.min_corpus_records)


class TestSeedRequests:
    def test_default_uses_max_of_both_thresholds(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SPEEDLM_E2E_SEED_REQUESTS", raising=False)
        config = SpeedLMConfig(
            model="test/model",
            tuning=IdleTuningConfig(
                min_trace_records=8,
                min_corpus_records=256,
            ),
        )
        assert _seed_requests(config) == 256

    def test_default_falls_back_to_min_trace_records_when_higher(self) -> None:
        config = SpeedLMConfig(
            model="test/model",
            tuning=IdleTuningConfig(
                min_trace_records=512,
                min_corpus_records=256,
                training_window_records=512,
            ),
        )
        assert _seed_requests(config) == 512

    def test_env_override_takes_precedence(self, monkeypatch) -> None:
        config = SpeedLMConfig(
            model="test/model",
            tuning=IdleTuningConfig(
                min_trace_records=8,
                min_corpus_records=256,
            ),
        )
        monkeypatch.setenv("SPEEDLM_E2E_SEED_REQUESTS", "1024")
        assert _seed_requests(config) == 1024

    def test_env_override_ignores_config(self, monkeypatch) -> None:
        config = SpeedLMConfig(
            model="test/model",
            tuning=IdleTuningConfig(
                min_trace_records=100,
                min_corpus_records=200,
            ),
        )
        monkeypatch.setenv("SPEEDLM_E2E_SEED_REQUESTS", "50")
        assert _seed_requests(config) == 50