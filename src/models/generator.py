"""Progressive super-resolution generator."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from src.models.condition_encoder import ConditionEncoder
from src.models.progressive_blocks import ProgressiveUpsampleBlock


class ProgressiveSRGenerator(nn.Module):
    """Generate SR images as bicubic LR baselines plus learned residuals."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        max_channels: int = 512,
        num_res_blocks_per_stage: int = 2,
        condition_type: str = "film",
        pyramid_scales: Iterable[int] = (1, 2, 4),
        return_intermediates: bool = True,
        condition_channels: int | None = None,
    ) -> None:
        super().__init__()
        channels = min(base_channels, max_channels)
        condition_channels = condition_channels or channels
        self.image_channels = image_channels
        self.channels = channels
        self.pyramid_scales = tuple(int(scale) for scale in pyramid_scales)
        self.return_intermediates = return_intermediates
        self.condition_encoder = ConditionEncoder(
            in_channels=image_channels,
            base_channels=min(base_channels, condition_channels),
            out_channels=condition_channels,
            scales=self.pyramid_scales,
        )
        self.input_proj = nn.Sequential(
            nn.Conv2d(image_channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block = ProgressiveUpsampleBlock(
            channels=channels,
            image_channels=image_channels,
            condition_channels=condition_channels,
            condition_type=condition_type,
            num_res_blocks=num_res_blocks_per_stage,
        )

    @staticmethod
    def _normalize_target_size(
        target_size: int | tuple[int, int] | list[int],
    ) -> tuple[int, int]:
        if isinstance(target_size, int):
            return (target_size, target_size)
        if len(target_size) != 2:
            raise ValueError(f"target_size must have two elements, got {target_size!r}")
        return (int(target_size[0]), int(target_size[1]))

    @staticmethod
    def _progressive_sizes(target_size: tuple[int, int]) -> list[tuple[int, int]]:
        target_h, target_w = target_size
        if target_h < 1 or target_w < 1:
            raise ValueError(f"target spatial size must be positive, got {target_size}")
        sizes: list[tuple[int, int]] = []
        height = width = 1
        while (height, width) != (target_h, target_w):
            height = min(target_h, height * 2)
            width = min(target_w, width * 2)
            sizes.append((height, width))
        return sizes

    def _pyramid_output_sizes(
        self, _lr: torch.Tensor, target_size: tuple[int, int]
    ) -> dict[int, tuple[int, int]]:
        return {
            scale: (
                max(1, int(round(target_size[0] / scale))),
                max(1, int(round(target_size[1] / scale))),
            )
            for scale in self.pyramid_scales
        }

    def forward(
        self,
        lr: torch.Tensor,
        target_size: int | tuple[int, int] | list[int],
        noise: torch.Tensor | None = None,
        return_intermediates: bool | None = None,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor] | list[torch.Tensor]]:
        if lr.ndim != 4:
            raise ValueError(f"expected lr to be BCHW, got shape {tuple(lr.shape)}")
        target_hw = self._normalize_target_size(target_size)
        should_return_intermediates = (
            self.return_intermediates
            if return_intermediates is None
            else return_intermediates
        )

        pyramid_sizes = self._pyramid_output_sizes(lr, target_hw)
        condition_features = self.condition_encoder(lr, pyramid_sizes)
        seed = lr.mean(dim=(-2, -1), keepdim=True)
        if noise is not None:
            seed = seed + noise.mean(dim=(-2, -1), keepdim=True)
        baseline = F.interpolate(
            lr, size=target_hw, mode="bicubic", align_corners=False
        )
        residual = torch.zeros_like(seed)
        features = self.input_proj(seed)
        residual_pyramid: dict[int, torch.Tensor] = (
            {1: residual} if pyramid_sizes.get(1) == (1, 1) else {}
        )
        pyramid: dict[int, torch.Tensor] = {}
        intermediate_features: list[torch.Tensor] = []

        for size in self._progressive_sizes(target_hw):
            condition = None
            for scale, scale_size in condition_features.items():
                if scale_size.shape[-2:] == size:
                    condition = scale_size
                    break
            if condition is None:
                condition = (
                    next(iter(condition_features.values()))
                    if condition_features
                    else None
                )
            residual, features = self.block(
                residual, features, condition=condition, size=size
            )
            if should_return_intermediates:
                intermediate_features.append(features)
            for scale, pyramid_size in pyramid_sizes.items():
                if size == pyramid_size:
                    residual_pyramid[scale] = residual

        image = baseline + residual

        for scale, size in pyramid_sizes.items():
            scale_baseline = F.interpolate(
                lr, size=size, mode="bicubic", align_corners=False
            )
            scale_residual = residual_pyramid.get(scale)
            if scale_residual is None:
                scale_residual = F.interpolate(
                    residual, size=size, mode="bilinear", align_corners=False
                )
            pyramid[scale] = scale_baseline + scale_residual

        return {
            "image": image,
            "pyramid": pyramid,
            "features": intermediate_features
            if should_return_intermediates
            else features,
        }
