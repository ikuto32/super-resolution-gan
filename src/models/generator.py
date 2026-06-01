"""Progressive super-resolution generator."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from src.models.condition_encoder import ConditionEncoder
from src.models.progressive_blocks import ProgressiveUpsampleBlock
from src.utils.spatial import normalize_hw


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
    def _progressive_sizes(
        target_size: tuple[int, int],
        start_size: tuple[int, int] = (1, 1),
    ) -> list[tuple[int, int]]:
        target_h, target_w = target_size
        if target_h < 1 or target_w < 1:
            raise ValueError(f"target spatial size must be positive, got {target_size}")
        start_h, start_w = start_size
        if start_h < 1 or start_w < 1:
            raise ValueError(f"start spatial size must be positive, got {start_size}")
        sizes: list[tuple[int, int]] = []
        height, width = min(target_h, start_h), min(target_w, start_w)
        if (height, width) != start_size:
            sizes.append((height, width))
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

    @staticmethod
    def _bicubic(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode="bicubic", align_corners=False)

    @staticmethod
    def _bilinear(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def _initial_residual(
        self,
        lr: torch.Tensor,
        target_hw: tuple[int, int],
        noise: torch.Tensor | None,
        noisy_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if noisy_condition is None:
            residual = torch.zeros_like(lr.mean(dim=(-2, -1), keepdim=True))
            if noise is not None:
                residual = residual + noise.mean(dim=(-2, -1), keepdim=True)
            return residual

        noisy_image = noisy_condition
        if noisy_image.shape[-2] > target_hw[0] or noisy_image.shape[-1] > target_hw[1]:
            noisy_image = self._bilinear(noisy_image, target_hw)

        noisy_baseline = self._bicubic(lr, noisy_image.shape[-2:])
        return noisy_image - noisy_baseline

    @staticmethod
    def _condition_for_size(
        condition_features: dict[int, torch.Tensor], size: tuple[int, int]
    ) -> torch.Tensor | None:
        for scale_feature in condition_features.values():
            if scale_feature.shape[-2:] == size:
                return scale_feature
        return next(iter(condition_features.values())) if condition_features else None

    def _build_pyramid(
        self,
        lr: torch.Tensor,
        residual: torch.Tensor,
        residual_pyramid: dict[int, torch.Tensor],
        pyramid_sizes: dict[int, tuple[int, int]],
    ) -> dict[int, torch.Tensor]:
        pyramid: dict[int, torch.Tensor] = {}
        for scale, size in pyramid_sizes.items():
            scale_baseline = self._bicubic(lr, size)
            scale_residual = residual_pyramid.get(scale)
            if scale_residual is None:
                scale_residual = self._bilinear(residual, size)
            pyramid[scale] = scale_baseline + scale_residual
        return pyramid

    def forward(
        self,
        lr: torch.Tensor,
        target_size: int | tuple[int, int] | list[int],
        noise: torch.Tensor | None = None,
        return_intermediates: bool | None = None,
        diffusion_timestep: torch.Tensor | None = None,
        noisy_condition: torch.Tensor | None = None,
        return_diffusion: bool = False,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor] | list[torch.Tensor]]:
        if lr.ndim != 4:
            raise ValueError(f"expected lr to be BCHW, got shape {tuple(lr.shape)}")
        target_hw = normalize_hw(target_size, name="target_size")
        should_return_intermediates = (
            self.return_intermediates
            if return_intermediates is None
            else return_intermediates
        )

        condition_source = noisy_condition if noisy_condition is not None else lr
        if condition_source.ndim != 4:
            raise ValueError(
                f"expected condition source to be BCHW, got shape {tuple(condition_source.shape)}"
            )

        pyramid_sizes = self._pyramid_output_sizes(condition_source, target_hw)
        condition_features = self.condition_encoder(condition_source, pyramid_sizes)

        # Final SR baseline is always bicubic LR.
        baseline = self._bicubic(lr, target_hw)

        # The network predicts residuals over the bicubic baseline.
        residual = self._initial_residual(lr, target_hw, noise, noisy_condition)
        features = self.input_proj(residual)

        residual_pyramid: dict[int, torch.Tensor] = {
            scale: residual
            for scale, size in pyramid_sizes.items()
            if size == residual.shape[-2:]
        }
        intermediate_features: list[torch.Tensor] = []

        start_size = residual.shape[-2:]
        for size in self._progressive_sizes(target_hw, start_size=start_size):
            condition = self._condition_for_size(condition_features, size)

            residual, features = self.block(
                residual, features, condition=condition, size=size
            )

            if should_return_intermediates:
                intermediate_features.append(features)

            for scale, pyramid_size in pyramid_sizes.items():
                if size == pyramid_size:
                    residual_pyramid[scale] = residual

        image = baseline + residual
        pyramid = self._build_pyramid(lr, residual, residual_pyramid, pyramid_sizes)

        output: dict[
            str, torch.Tensor | dict[int, torch.Tensor] | list[torch.Tensor]
        ] = {
            "image": image,
            "baseline": baseline,
            "residual": residual,
            "pyramid": pyramid,
            "features": intermediate_features
            if should_return_intermediates
            else features,
        }
        if return_diffusion:
            output["diffusion"] = image
            if diffusion_timestep is not None:
                output["diffusion_timestep"] = diffusion_timestep
        return output
