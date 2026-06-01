"""Aggregation helpers for generator losses."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from src.losses.consistency import lr_consistency_loss
from src.losses.r3gan import generator_loss as adversarial_generator_loss
from src.losses.reconstruction import (
    reconstruction_loss,
    multi_scale_reconstruction_loss,
)

LOSS_KEYS = (
    "loss_total",
    "loss_adv",
    "loss_pixel",
    "loss_multiscale",
    "loss_perceptual",
    "loss_consistency",
    "loss_diffusion",
)

LOSS_WEIGHT_ALIASES = {
    "adv": ("lambda_adv", "adv", 1.0),
    "pixel": ("lambda_pixel", "lambda_pix", "pixel", 1.0),
    "multiscale": ("lambda_multiscale", "lambda_ms", "multiscale", 0.0),
    "perceptual": ("lambda_perceptual", "lambda_perc", "perceptual", 0.0),
    "consistency": ("lambda_consistency", "lambda_cons", "consistency", 0.0),
    "diffusion": ("lambda_diffusion", "lambda_diff", "diffusion", 0.0),
}


def _weight(weights: Mapping[str, float], name: str) -> float:
    *aliases, default = LOSS_WEIGHT_ALIASES[name]
    for alias in aliases:
        if alias in weights:
            return float(weights[alias])
    return float(default)


def _loss_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    weights = weights or {}
    return {name: _weight(weights, name) for name in LOSS_WEIGHT_ALIASES}


def _first_tensor(*values: object) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            nested = _first_tensor(*value.values())
            if nested is not None:
                return nested
    return None


def _zero_like_reference(*values: object) -> torch.Tensor:
    reference = _first_tensor(*values)
    if reference is None:
        return torch.tensor(0.0)
    return reference.new_zeros(())


def _as_scalar_loss(
    value: torch.Tensor | Callable[[], torch.Tensor] | None,
    zero: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return zero
    if callable(value):
        value = value()
    if not isinstance(value, torch.Tensor):
        raise TypeError("loss values must be tensors or zero-argument callables")
    return value.mean() if value.ndim > 0 else value


def _weighted_optional_loss(
    *,
    weight: float,
    zero: torch.Tensor,
    missing: bool,
    loss: torch.Tensor | Callable[[], torch.Tensor] | None,
) -> torch.Tensor:
    if weight == 0.0 or missing:
        return zero
    return _as_scalar_loss(loss, zero) * weight


def generator_losses(
    *,
    fake_scores: torch.Tensor | None = None,
    real_scores: torch.Tensor | None = None,
    sr: torch.Tensor | None = None,
    hr: torch.Tensor | None = None,
    lr: torch.Tensor | None = None,
    generated_pyramid: Mapping[int | str, torch.Tensor] | None = None,
    hr_pyramid: Mapping[int | str, torch.Tensor] | None = None,
    reconstruction_prediction: torch.Tensor | None = None,
    reconstruction_target: torch.Tensor | None = None,
    perceptual_loss: torch.Tensor | Callable[[], torch.Tensor] | None = None,
    diffusion_loss: torch.Tensor | Callable[[], torch.Tensor] | None = None,
    weights: Mapping[str, float] | None = None,
    pixel_loss_type: str = "l1",
    consistency_mode: str = "bicubic",
) -> dict[str, torch.Tensor]:
    """Aggregate generator losses and always return all logging keys.

    Optional terms are represented by scalar zero tensors whenever their weight is
    zero or the inputs needed to compute them are not provided.
    """
    loss_weights = _loss_weights(weights)
    lambda_adv = loss_weights["adv"]
    lambda_pixel = loss_weights["pixel"]
    lambda_multiscale = loss_weights["multiscale"]
    lambda_perceptual = loss_weights["perceptual"]
    lambda_consistency = loss_weights["consistency"]
    lambda_diffusion = loss_weights["diffusion"]

    zero = _zero_like_reference(
        fake_scores,
        real_scores,
        sr,
        hr,
        lr,
        generated_pyramid,
        hr_pyramid,
        reconstruction_prediction,
        reconstruction_target,
        perceptual_loss if isinstance(perceptual_loss, torch.Tensor) else None,
        diffusion_loss if isinstance(diffusion_loss, torch.Tensor) else None,
    )

    loss_adv = _weighted_optional_loss(
        weight=lambda_adv,
        zero=zero,
        missing=fake_scores is None,
        loss=lambda: adversarial_generator_loss(real_scores, fake_scores),
    )

    pixel_prediction = (
        reconstruction_prediction if reconstruction_prediction is not None else sr
    )
    pixel_target = reconstruction_target if reconstruction_target is not None else hr
    loss_pixel = _weighted_optional_loss(
        weight=lambda_pixel,
        zero=zero,
        missing=pixel_prediction is None or pixel_target is None,
        loss=lambda: reconstruction_loss(
            pixel_prediction, pixel_target, loss_type=pixel_loss_type
        ),
    )

    loss_multiscale = _weighted_optional_loss(
        weight=lambda_multiscale,
        zero=zero,
        missing=(
            generated_pyramid is None
            or hr_pyramid is None
            or not generated_pyramid
            or not hr_pyramid
        ),
        loss=lambda: multi_scale_reconstruction_loss(
            dict(generated_pyramid),
            dict(hr_pyramid),
        ),
    )

    loss_perceptual = _weighted_optional_loss(
        weight=lambda_perceptual,
        zero=zero,
        missing=perceptual_loss is None,
        loss=perceptual_loss,
    )

    loss_consistency = _weighted_optional_loss(
        weight=lambda_consistency,
        zero=zero,
        missing=sr is None or lr is None,
        loss=lambda: lr_consistency_loss(sr, lr, mode=consistency_mode),
    )

    loss_diffusion = _weighted_optional_loss(
        weight=lambda_diffusion,
        zero=zero,
        missing=diffusion_loss is None,
        loss=diffusion_loss,
    )

    loss_total = (
        loss_adv
        + loss_pixel
        + loss_multiscale
        + loss_perceptual
        + loss_consistency
        + loss_diffusion
    )

    return {
        "loss_total": loss_total,
        "loss_adv": loss_adv,
        "loss_pixel": loss_pixel,
        "loss_multiscale": loss_multiscale,
        "loss_perceptual": loss_perceptual,
        "loss_consistency": loss_consistency,
        "loss_diffusion": loss_diffusion,
    }


compute_generator_loss = generator_losses
generator_total_loss = generator_losses
