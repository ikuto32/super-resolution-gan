"""Condition encoders for progressive super-resolution generators."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """Encode an LR image into condition features at requested spatial sizes.

    The encoder first extracts a compact LR feature map, then resizes it to each
    requested stage size and adapts it with a small per-stage projection. The
    returned dictionary is keyed by the caller-provided scale identifiers (for
    example ``1``, ``2`` and ``4`` for LR, 2x LR and 4x LR supervision).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        out_channels: int = 128,
        scales: Iterable[int] = (1, 2, 4),
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.stem = nn.Sequential(*layers)
        self.scales = tuple(int(scale) for scale in scales)
        self.projections = nn.ModuleDict(
            {
                str(scale): nn.Sequential(
                    nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for scale in self.scales
            }
        )

    def forward(
        self,
        lr: torch.Tensor,
        output_sizes: dict[int, tuple[int, int]] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Return condition features for each configured/requested scale.

        Args:
            lr: Low-resolution condition image in ``BCHW`` format.
            output_sizes: Optional mapping from scale key to target ``(H, W)``.
                If omitted, keys are interpreted as factors relative to LR size.
        """
        if lr.ndim != 4:
            raise ValueError(f"expected lr to be BCHW, got shape {tuple(lr.shape)}")

        encoded = self.stem(lr)
        lr_h, lr_w = lr.shape[-2:]
        scale_keys = tuple(output_sizes) if output_sizes is not None else self.scales
        features: dict[int, torch.Tensor] = {}

        for scale in scale_keys:
            key = int(scale)
            if output_sizes is None:
                size = (max(1, lr_h * key), max(1, lr_w * key))
            else:
                size = output_sizes[key]
            resized = F.interpolate(
                encoded, size=size, mode="bilinear", align_corners=False
            )
            projection = (
                self.projections[str(key)]
                if str(key) in self.projections
                else next(iter(self.projections.values()))
            )
            features[key] = projection(resized)

        return features
