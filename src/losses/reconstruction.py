"""Pixel and multi-scale reconstruction losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _canonical_key(key: int | str) -> int | str:
    if isinstance(key, str) and key.startswith("x"):
        suffix = key[1:]
        if suffix.isdigit():
            return int(suffix)
    return key


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a configurable pixel reconstruction loss.

    Supported ``loss_type`` values are ``"l1"``, ``"mse"`` and
    ``"charbonnier"``. The Charbonnier loss is a smooth L1-like loss defined
    as ``sqrt((prediction - target)^2 + eps^2)``.
    """
    normalized_type = loss_type.lower()
    if normalized_type in {"l1", "mae"}:
        return F.l1_loss(prediction, target, reduction=reduction)
    if normalized_type in {"mse", "l2"}:
        return F.mse_loss(prediction, target, reduction=reduction)
    if normalized_type in {"charbonnier", "charb"}:
        loss = torch.sqrt((prediction - target).square() + eps**2)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction}")
    raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")


class ReconstructionLoss(nn.Module):
    """Configurable pixel-space reconstruction loss module."""

    def __init__(
        self,
        loss_type: str = "l1",
        weight: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.weight = float(weight)
        self.reduction = reduction
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            reconstruction_loss(
                prediction,
                target,
                loss_type=self.loss_type,
                reduction=self.reduction,
                eps=self.eps,
            )
            * self.weight
        )


def reconstruction_loss_from_config(config: dict[str, object]) -> ReconstructionLoss:
    """Build :class:`ReconstructionLoss` from a lightweight config mapping."""
    return ReconstructionLoss(
        loss_type=str(config.get("type", config.get("loss_type", "l1"))),
        weight=float(config.get("weight", 1.0)),
        reduction=str(config.get("reduction", "mean")),
        eps=float(config.get("eps", 1e-6)),
    )


class MultiScaleReconstructionLoss(nn.Module):
    """L1 reconstruction loss over matching generated and HR pyramid scales."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = float(weight)
        self.reduction = reduction

    def forward(
        self,
        generated_pyramid: dict[int | str, torch.Tensor],
        hr_pyramid: dict[int | str, torch.Tensor],
    ) -> torch.Tensor:
        if not generated_pyramid:
            raise ValueError("generated_pyramid must not be empty")
        canonical_hr = {_canonical_key(key): value for key, value in hr_pyramid.items()}
        losses: list[torch.Tensor] = []

        for gen_key, generated in generated_pyramid.items():
            scale = _canonical_key(gen_key)
            if scale not in canonical_hr:
                continue
            target = canonical_hr[scale]
            if target.shape[-2:] != generated.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=generated.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(F.l1_loss(generated, target, reduction=self.reduction))

        if not losses:
            raise ValueError(
                "generated_pyramid and hr_pyramid do not share any scale keys"
            )
        return torch.stack(losses).mean() * self.weight


def multi_scale_reconstruction_loss(
    generated_pyramid: dict[int | str, torch.Tensor],
    hr_pyramid: dict[int | str, torch.Tensor],
    weight: float = 1.0,
) -> torch.Tensor:
    """Functional wrapper for :class:`MultiScaleReconstructionLoss`."""
    return MultiScaleReconstructionLoss(weight=weight)(generated_pyramid, hr_pyramid)
