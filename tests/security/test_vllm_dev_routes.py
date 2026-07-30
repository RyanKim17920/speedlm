from __future__ import annotations

import pytest

from speedlm.gateway.proxy import _BLOCKED_V1_PATHS, _is_allowed_path

_INSTALLED_VLLM_DEV_ROUTES = {
    "/collective_rpc",
    "/finish_weight_update",
    "/get_world_size",
    "/init_weight_transfer_engine",
    "/is_paused",
    "/is_sleeping",
    "/pause",
    "/reset_encoder_cache",
    "/reset_mm_cache",
    "/reset_prefix_cache",
    "/resume",
    "/server_info",
    "/sleep",
    "/start_weight_update",
    "/update_weights",
    "/wake_up",
}


def test_every_installed_vllm_dev_route_is_explicitly_denylisted() -> None:
    assert _INSTALLED_VLLM_DEV_ROUTES <= _BLOCKED_V1_PATHS


@pytest.mark.parametrize("path", sorted(_INSTALLED_VLLM_DEV_ROUTES))
def test_every_installed_vllm_dev_route_is_blocked(path: str) -> None:
    assert not _is_allowed_path(path)
    assert not _is_allowed_path(f"{path}/")
