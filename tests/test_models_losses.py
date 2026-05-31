from __future__ import annotations

import torch

from src.losses import MultiScaleReconstructionLoss
from src.models import (
    ConditionEncoder,
    ProgressiveSRGenerator,
    ProgressiveUpsampleBlock,
)


def test_condition_encoder_returns_requested_scales_and_sizes():
    encoder = ConditionEncoder(
        in_channels=3, base_channels=8, out_channels=16, scales=(1, 2, 4)
    )
    lr = torch.randn(2, 3, 8, 8)

    features = encoder(lr, {1: (8, 8), 2: (16, 16), 4: (32, 32)})

    assert set(features) == {1, 2, 4}
    assert features[1].shape == (2, 16, 8, 8)
    assert features[2].shape == (2, 16, 16, 16)
    assert features[4].shape == (2, 16, 32, 32)


def test_progressive_upsample_block_returns_rgb_residual_at_requested_size():
    block = ProgressiveUpsampleBlock(
        channels=8,
        image_channels=3,
        condition_channels=16,
        condition_type="film",
        num_res_blocks=1,
    )

    residual, features = block(
        residual=torch.randn(2, 3, 4, 4),
        features=torch.randn(2, 8, 4, 4),
        condition=torch.randn(2, 16, 8, 8),
        size=(8, 8),
    )

    assert residual.shape == (2, 3, 8, 8)
    assert features.shape == (2, 8, 8, 8)


def test_progressive_sr_generator_returns_image_pyramid_and_features():
    generator = ProgressiveSRGenerator(
        base_channels=8,
        max_channels=8,
        num_res_blocks_per_stage=1,
        pyramid_scales=(1, 2, 4),
    )
    lr = torch.randn(2, 3, 8, 8)

    output = generator(lr, target_size=(32, 32), return_intermediates=True)

    assert output["image"].shape == (2, 3, 32, 32)
    assert set(output["pyramid"]) == {1, 2, 4}
    assert output["pyramid"][1].shape == (2, 3, 32, 32)
    assert output["pyramid"][2].shape == (2, 3, 16, 16)
    assert output["pyramid"][4].shape == (2, 3, 8, 8)
    assert output["features"]


def test_progressive_sr_generator_uses_bicubic_baselines_when_residual_is_zero():
    generator = ProgressiveSRGenerator(
        base_channels=8,
        max_channels=8,
        num_res_blocks_per_stage=1,
        pyramid_scales=(1, 2, 4),
    )
    torch.nn.init.zeros_(generator.block.to_rgb.weight)
    torch.nn.init.zeros_(generator.block.to_rgb.bias)
    lr = torch.randn(2, 3, 8, 8)

    output = generator(lr, target_size=(32, 32), return_intermediates=True)

    expected_image = torch.nn.functional.interpolate(
        lr, size=(32, 32), mode="bicubic", align_corners=False
    )
    assert output["image"].shape == (2, 3, 32, 32)
    assert torch.allclose(output["image"], expected_image)
    for scale, expected_size in {1: (32, 32), 2: (16, 16), 4: (8, 8)}.items():
        expected_pyramid = torch.nn.functional.interpolate(
            lr, size=expected_size, mode="bicubic", align_corners=False
        )
        assert output["pyramid"][scale].shape == (2, 3, *expected_size)
        assert torch.allclose(output["pyramid"][scale], expected_pyramid)


def test_multiscale_reconstruction_loss_matches_int_and_x_prefixed_keys():
    generated = {
        1: torch.zeros(1, 3, 8, 8),
        2: torch.zeros(1, 3, 16, 16),
        4: torch.zeros(1, 3, 32, 32),
    }
    hr = {
        "x1": torch.ones(1, 3, 8, 8),
        "x2": torch.ones(1, 3, 16, 16),
        "x4": torch.ones(1, 3, 32, 32),
    }

    loss = MultiScaleReconstructionLoss(weight=0.5)(generated, hr)

    assert torch.isclose(loss, torch.tensor(0.5))
