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
    weights = weights or {}
    lambda_adv = float(weights.get("lambda_adv", weights.get("adv", 1.0)))
    lambda_pixel = float(
        weights.get(
            "lambda_pixel", weights.get("lambda_pix", weights.get("pixel", 1.0))
        )
    )
    lambda_multiscale = float(
        weights.get(
            "lambda_multiscale",
            weights.get("lambda_ms", weights.get("multiscale", 0.0)),
        )
    )
    lambda_perceptual = float(
        weights.get(
            "lambda_perceptual",
            weights.get("lambda_perc", weights.get("perceptual", 0.0)),
        )
    )
    lambda_consistency = float(
        weights.get(
            "lambda_consistency",
            weights.get("lambda_cons", weights.get("consistency", 0.0)),
        )
    )
    lambda_diffusion = float(
        weights.get(
            "lambda_diffusion",
            weights.get("lambda_diff", weights.get("diffusion", 0.0)),
        )
    )

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

    if lambda_adv == 0.0 or fake_scores is None:
        loss_adv = zero
    else:
        loss_adv = adversarial_generator_loss(real_scores, fake_scores) * lambda_adv

    pixel_prediction = (
        reconstruction_prediction if reconstruction_prediction is not None else sr
    )
    pixel_target = reconstruction_target if reconstruction_target is not None else hr
    if lambda_pixel == 0.0 or pixel_prediction is None or pixel_target is None:
        loss_pixel = zero
    else:
        loss_pixel = (
            reconstruction_loss(
                pixel_prediction, pixel_target, loss_type=pixel_loss_type
            )
            * lambda_pixel
        )

    if (
        lambda_multiscale == 0.0
        or generated_pyramid is None
        or hr_pyramid is None
        or not generated_pyramid
        or not hr_pyramid
    ):
        loss_multiscale = zero
    else:
        loss_multiscale = multi_scale_reconstruction_loss(
            dict(generated_pyramid),
            dict(hr_pyramid),
            weight=lambda_multiscale,
        )

    if lambda_perceptual == 0.0 or perceptual_loss is None:
        loss_perceptual = zero
    else:
        loss_perceptual = _as_scalar_loss(perceptual_loss, zero) * lambda_perceptual

    if lambda_consistency == 0.0 or sr is None or lr is None:
        loss_consistency = zero
    else:
        loss_consistency = (
            lr_consistency_loss(sr, lr, mode=consistency_mode) * lambda_consistency
        )

    if lambda_diffusion == 0.0 or diffusion_loss is None:
        loss_diffusion = zero
    else:
        loss_diffusion = _as_scalar_loss(diffusion_loss, zero) * lambda_diffusion

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
