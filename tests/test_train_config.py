from __future__ import annotations

import pytest

from scripts.train import _degradation_config


def _config(scale=4):
    return {
        "data": {
            "image_size_hr": [256, 256],
            "image_size_lr": [64, 64],
        },
        "degradation": {
            "downsample": {
                "method": "bicubic",
                "scale": scale,
            },
        },
    }


def test_degradation_config_sets_expected_scale_and_target_size():
    degradation = _degradation_config(_config())

    assert degradation["scale"] == 4
    assert degradation["mode"] == "bicubic"
    assert degradation["target_size"] == (64, 64)


def test_degradation_config_rejects_scale_mismatch():
    config = _config(scale=2)

    with pytest.raises(ValueError, match="degradation.downsample.scale"):
        _degradation_config(config)
