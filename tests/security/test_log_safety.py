"""Proof that subprocess diagnostic text is retained and later loggable verbatim."""

from __future__ import annotations

from speedlm.tuner.eagle3 import TrainingError


def test_training_error_must_not_expose_secret_stderr() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"

    error = TrainingError("training failed", stderr=f"debug credential={secret}")

    assert secret not in str(error)

