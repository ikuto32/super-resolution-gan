"""Perceptual losses for super-resolution training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


VGG16_LAYER_INDICES: dict[str, int] = {
    "relu1_1": 2,
    "relu1_2": 4,
    "relu2_1": 7,
    "relu2_2": 9,
    "relu3_1": 12,
    "relu3_2": 14,
    "relu3_3": 16,
    "relu4_1": 19,
    "relu4_2": 21,
    "relu4_3": 23,
    "relu5_1": 26,
    "relu5_2": 28,
    "relu5_3": 30,
}


class VGGPerceptualLoss(nn.Module):
    """VGG feature reconstruction loss for tensors normalized to ``[-1, 1]``.

    Inputs are first converted from the training range ``[-1, 1]`` to ``[0, 1]``
    and then normalized with ImageNet channel statistics before feature
    extraction. VGG parameters are frozen; gradients flow only to ``prediction``.
    """

    def __init__(
        self,
        *,
        layers: Sequence[str] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
        layer_weights: Mapping[str, float] | None = None,
        pretrained: bool = False,
        resize_to: int | Sequence[int] | None = None,
        loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one VGG perceptual layer is required")
        unknown_layers = [layer for layer in layers if layer not in VGG16_LAYER_INDICES]
        if unknown_layers:
            valid = ", ".join(sorted(VGG16_LAYER_INDICES))
            raise ValueError(f"unknown VGG layer(s) {unknown_layers}; valid layers: {valid}")
        if loss_type not in {"l1", "mse"}:
            raise ValueError("loss_type must be 'l1' or 'mse'")

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16(weights=weights).features.eval()
        for parameter in features.parameters():
            parameter.requires_grad_(False)

        self.layers = tuple(layers)
        self.layer_weights = {
            layer: float(layer_weights.get(layer, 1.0)) if layer_weights else 1.0
            for layer in self.layers
        }
        self.loss_type = loss_type
        self.resize_to = resize_to
        self.feature_blocks = nn.ModuleList()

        previous_index = 0
        for layer in self.layers:
            end_index = VGG16_LAYER_INDICES[layer]
            self.feature_blocks.append(features[previous_index:end_index])
            previous_index = end_index

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _prepare_for_vgg(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("perceptual loss expects NCHW tensors")
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.shape[1] != 3:
            raise ValueError("perceptual loss expects 1- or 3-channel images")

        images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)
        if self.resize_to is not None:
            size = (
                (self.resize_to, self.resize_to)
                if isinstance(self.resize_to, int)
                else tuple(self.resize_to)
            )
            images = F.interpolate(
                images, size=size, mode="bilinear", align_corners=False
            )
        return (images - self.mean) / self.std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return weighted feature distance between prediction and target."""
        prediction_features = self._prepare_for_vgg(prediction)
        with torch.no_grad():
            target_features = self._prepare_for_vgg(target)

        loss = prediction.new_zeros(())
        for layer, block in zip(self.layers, self.feature_blocks, strict=True):
            prediction_features = block(prediction_features)
            with torch.no_grad():
                target_features = block(target_features)
            if self.loss_type == "l1":
                layer_loss = F.l1_loss(prediction_features, target_features)
            else:
                layer_loss = F.mse_loss(prediction_features, target_features)
            loss = loss + layer_loss * self.layer_weights[layer]
        return loss


def build_perceptual_loss(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a perceptual loss module from a loss config mapping."""
    config = config or {}
    perceptual_cfg = config.get("perceptual", {})
    if not isinstance(perceptual_cfg, Mapping):
        perceptual_cfg = {}
    loss_type_name = str(perceptual_cfg.get("type", "vgg")).lower()
    if loss_type_name != "vgg":
        raise ValueError("only VGG perceptual loss is currently implemented")

    layers = perceptual_cfg.get(
        "layers", ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
    )
    if isinstance(layers, str):
        layers = (layers,)
    layer_weights = perceptual_cfg.get("layer_weights")
    if layer_weights is not None and not isinstance(layer_weights, Mapping):
        raise TypeError("loss.perceptual.layer_weights must be a mapping")

    return VGGPerceptualLoss(
        layers=tuple(layers),
        layer_weights=layer_weights,
        pretrained=bool(perceptual_cfg.get("pretrained", False)),
        resize_to=perceptual_cfg.get("resize_to"),
        loss_type=str(perceptual_cfg.get("loss_type", "l1")).lower(),
    )
