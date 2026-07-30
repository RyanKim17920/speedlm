"""Proofs and regression checks for upstream destination/path safety."""

from __future__ import annotations

import httpx
import pytest

from speedlm.gateway.proxy import _BLOCKED_V1_PATHS, _is_allowed_path


def test_dot_segment_cannot_reach_blocked_admin_endpoint() -> None:
    client_path = "/v1/harmless/../sleep"
    forwarded = httpx.URL("http://127.0.0.1:8000").copy_with(path=client_path)

    assert forwarded.path == "/v1/sleep"
    assert forwarded.path in _BLOCKED_V1_PATHS
    assert not _is_allowed_path(client_path)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/sleep",
        "/v1/sleep/",
        "/v1/sleep////",
        "/v1/load_lora_adapter/",
    ],
)
def test_blocked_paths_with_trailing_slashes_are_rejected(path: str) -> None:
    assert not _is_allowed_path(path)


def test_unrelated_paths_outside_v1_are_rejected() -> None:
    assert not _is_allowed_path("/metrics")
    assert not _is_allowed_path("//attacker.example/v1/chat/completions")
