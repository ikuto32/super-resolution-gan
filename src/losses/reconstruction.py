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
