from __future__ import annotations

import torch
from torch import nn

import src.losses.perceptual as perceptual_module

from src.losses import (
    LOSS_KEYS,
    discriminator_loss,
    generator_loss,
    generator_total_loss,
    lr_consistency_loss,
    r1_regularization,
    r2_regularization,
    reconstruction_loss,
)


def _assert_scalar_tensor(loss: torch.Tensor) -> None:
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_r3gan_losses_return_scalar_tensors():
    real_scores = torch.randn(2, 1)
    fake_scores = torch.randn(2, 1)
    real_images = torch.randn(2, 3, 8, 8)
    fake_images = torch.randn(2, 3, 8, 8)

    _assert_scalar_tensor(
        discriminator_loss(real_scores, fake_scores, real_images, fake_images)
    )
    _assert_scalar_tensor(generator_loss(real_scores, fake_scores))


def test_r1_and_r2_regularization_support_autograd():
    real_images = torch.randn(2, 3, 4, 4, requires_grad=True)
    fake_images = torch.randn(2, 3, 4, 4, requires_grad=True)
    real_scores = real_images.square().mean(dim=(1, 2, 3), keepdim=True)
    fake_scores = fake_images.square().mean(dim=(1, 2, 3), keepdim=True)

    r1 = r1_regularization(real_scores, real_images)
    r2 = r2_regularization(fake_scores, fake_images)

    _assert_scalar_tensor(r1)
    _assert_scalar_tensor(r2)
    (r1 + r2).backward()
    assert real_images.grad is not None
    assert fake_images.grad is not None


def test_reconstruction_losses_return_scalar_tensors():
    prediction = torch.zeros(2, 3, 8, 8)
    target = torch.ones(2, 3, 8, 8)

    for loss_type in ("l1", "mse", "charbonnier"):
        _assert_scalar_tensor(reconstruction_loss(prediction, target, loss_type))


def test_lr_consistency_loss_downsamples_to_lr_and_returns_scalar():
    lr = torch.zeros(1, 3, 4, 4)
    sr = torch.ones(1, 3, 8, 8)

    loss = lr_consistency_loss(sr, lr)

    _assert_scalar_tensor(loss)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_generator_total_loss_returns_all_keys_with_optional_zero_losses():
    sr = torch.zeros(1, 3, 8, 8)
    hr = torch.ones(1, 3, 8, 8)
    fake_scores = torch.zeros(1, 1)

    losses = generator_total_loss(
        fake_scores=fake_scores,
        sr=sr,
        hr=hr,
        weights={
            "lambda_adv": 1.0,
            "lambda_pixel": 1.0,
            "lambda_multiscale": 0.0,
            "lambda_perceptual": 1.0,
            "lambda_consistency": 1.0,
            "lambda_diffusion": 0.0,
        },
    )

    assert set(losses) == set(LOSS_KEYS)
    for loss in losses.values():
        _assert_scalar_tensor(loss)
    assert losses["loss_perceptual"].item() == 0.0
    assert losses["loss_consistency"].item() == 0.0
    assert losses["loss_diffusion"].item() == 0.0


def test_vgg_perceptual_loss_prepares_minus_one_one_images_for_imagenet(monkeypatch):
    class TinyVGG:
        features = nn.Sequential(nn.Identity(), nn.Identity())

    monkeypatch.setattr(perceptual_module, "vgg16", lambda weights=None: TinyVGG())
    loss = perceptual_module.VGGPerceptualLoss(layers=("relu1_1",))

    images = torch.tensor(
        [[[[-1.0]], [[0.0]], [[1.0]]]],
        dtype=torch.float32,
    )
    prepared = loss._prepare_for_vgg(images)
    expected_unit_range = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
    expected = (expected_unit_range - loss.mean) / loss.std

    assert torch.allclose(prepared, expected)
