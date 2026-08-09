"""Safety boundary for Speculators' final-assistant-only loss mask."""

from pathlib import Path

import pytest

from speedlm.training.backends.eagle3 import SpeculatorsPipelineConfig
from speedlm.training.masking import MaskPolicy


def test_final_turn_policy_is_rejected_until_vendor_proves_turn_containment() -> None:
    """A last rendered span is not necessarily inside the last assistant turn.

    In the pinned Speculators implementation, an empty final assistant turn can
    leave the previous assistant span as the last nonempty HF-mask span or the
    last regex match.  That earlier turn may be client-authored, so SpeedLM must
    keep this policy unavailable until the vendor mask is bounded by turn.
    """
    with pytest.raises(ValueError, match="ALL_ASSISTANT_TURNS"):
        SpeculatorsPipelineConfig(
            prepared_validator_script=Path("check.py"),
            speculators_repo=Path("speculators"),
            training_python=Path("python"),
            verifier_model="verifier",
            warm_start_model="warm-start",
            mask_policy=MaskPolicy.FINAL_TURN_ALL_CHANNELS,
        )
