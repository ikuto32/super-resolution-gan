"""Resolution-agnostic conditional discriminator."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResolutionAgnosticDiscriminator(nn.Module):
    """Fully convolutional discriminator for real/generated SR images."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        conditional: bool = True,
        input_condition_mode: str = "concat",
        patch_output: bool = True,
        global_aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if input_condition_mode != "concat":
            raise ValueError("only concat condition mode is currently supported")
        if global_aggregation != "mean":
            raise ValueError("only mean global aggregation is currently supported")
        self.conditional = bool(conditional)
        self.patch_output = bool(patch_output)
        in_channels = image_channels * (2 if self.conditional else 1)
        channels = min(int(base_channels), int(max_channels))
        hidden = min(channels * 2, int(max_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.conditional:
            if condition is None:
                raise ValueError("condition is required for conditional discriminator")
            condition = F.interpolate(condition, size=image.shape[-2:], mode="bilinear", align_corners=False)
            image = torch.cat([image, condition], dim=1)
        patch_logits = self.net(image)
        score = patch_logits.mean(dim=(1, 2, 3))
        return {"score": score, "patch_logits": patch_logits}


__all__ = ["ResolutionAgnosticDiscriminator"]
