"""Progressive upsampling blocks for super-resolution generation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A lightweight convolutional residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ConditionInjection(nn.Module):
    """Inject condition features using FiLM, additive, or concat conditioning."""

    def __init__(
        self, channels: int, condition_channels: int, condition_type: str = "film"
    ) -> None:
        super().__init__()
        self.condition_type = condition_type.lower()
        if self.condition_type == "film":
            self.proj = nn.Conv2d(condition_channels, channels * 2, kernel_size=1)
        elif self.condition_type == "add":
            self.proj = nn.Conv2d(condition_channels, channels, kernel_size=1)
        elif self.condition_type == "concat":
            self.proj = nn.Conv2d(
                channels + condition_channels, channels, kernel_size=1
            )
        elif self.condition_type in {"none", "identity"}:
            self.proj = nn.Identity()
        else:
            raise ValueError(f"unsupported condition_type: {condition_type!r}")

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None or self.condition_type in {"none", "identity"}:
            return x
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(
                condition, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.condition_type == "film":
            gamma, beta = self.proj(condition).chunk(2, dim=1)
            return x * (1.0 + gamma) + beta
        if self.condition_type == "add":
            return x + self.proj(condition)
        return self.proj(torch.cat([x, condition], dim=1))


class ProgressiveUpsampleBlock(nn.Module):
    """Upsample residual/features, inject condition, refine, and predict RGB residual."""

    def __init__(
        self,
        channels: int,
        image_channels: int = 3,
        condition_channels: int = 128,
        condition_type: str = "film",
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.condition = ConditionInjection(
            channels, condition_channels, condition_type
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        self.to_rgb = nn.Conv2d(channels, image_channels, kernel_size=1)

    def forward(
        self,
        residual: torch.Tensor,
        features: torch.Tensor,
        condition: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the progressive RGB residual and updated feature map."""
        if size is None:
            size = (features.shape[-2] * 2, features.shape[-1] * 2)
        upsampled_residual = F.interpolate(
            residual, size=size, mode="bilinear", align_corners=False
        )
        x = F.interpolate(features, size=size, mode="bilinear", align_corners=False)
        x = self.activation(self.conv(x))
        x = self.condition(x, condition)
        x = self.residual_blocks(x)
        residual_delta = self.to_rgb(x)
        residual = upsampled_residual + residual_delta
        return residual, x
