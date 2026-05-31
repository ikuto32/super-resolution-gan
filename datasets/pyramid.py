"""Helpers for constructing multi-scale HR image pyramids."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _scale_key(scale: float | int) -> str:
    if float(scale).is_integer():
        return f"x{int(scale)}"
    return f"x{str(scale).replace('.', '_')}"


def build_image_pyramid(
    hr: torch.Tensor,
    scales: Iterable[float | int],
    min_size: int = 1,
    mode: str = "bicubic",
) -> dict[str, torch.Tensor]:
    """Build a dictionary of downsampled HR tensors keyed by scale.

    ``scales`` are interpreted as downsampling factors. A scale of ``1`` returns
    the original tensor. If a target side becomes ``1``, that dimension is
    produced by spatial averaging rather than interpolation.
    """
    if hr.ndim not in {3, 4}:
        raise ValueError(f"expected CHW or BCHW tensor, got shape {tuple(hr.shape)}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")

    batched = hr.ndim == 4
    source = hr if batched else hr.unsqueeze(0)
    height, width = source.shape[-2:]
    pyramid: dict[str, torch.Tensor] = {}

    for scale in scales:
        factor = float(scale)
        if factor <= 0:
            raise ValueError(f"scales must be positive, got {scale!r}")
        target_h = max(min_size, int(round(height / factor)))
        target_w = max(min_size, int(round(width / factor)))

        if target_h == height and target_w == width:
            resized = source.clone()
        elif target_h == 1 and target_w == 1:
            resized = source.mean(dim=(-2, -1), keepdim=True)
        elif target_h == 1:
            resized = F.interpolate(
                source.mean(dim=-2, keepdim=True),
                size=(1, target_w),
                mode=mode,
                align_corners=False,
            )
        elif target_w == 1:
            resized = F.interpolate(
                source.mean(dim=-1, keepdim=True),
                size=(target_h, 1),
                mode=mode,
                align_corners=False,
            )
        else:
            resized = F.interpolate(
                source, size=(target_h, target_w), mode=mode, align_corners=False
            )

        pyramid[_scale_key(scale)] = resized if batched else resized.squeeze(0)

    return pyramid
