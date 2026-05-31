"""Configurable image degradation pipeline for LR image generation."""

from __future__ import annotations

import io
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from datasets.transforms import pil_to_tensor, tensor_to_pil


@dataclass(frozen=True)
class DegradationPipeline:
    """Apply configured HR-to-LR degradations.

    Supported configuration keys:
    ``scale`` (downsample factor), ``target_size``, ``mode``, ``blur``, ``noise``,
    ``jpeg``, and ``clamp``. ``target_size`` takes precedence over ``scale``.
    Blur/noise/jpeg may be booleans, scalars, or
    dictionaries with an ``enabled`` flag plus operation-specific settings.
    """

    config: Mapping[str, Any] | None = None
    target_size: int | tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config or {}))
        if self.target_size is None:
            target_size = self.config.get("target_size")
        else:
            target_size = self.target_size
        object.__setattr__(self, "target_size", _normalize_target_size(target_size))

    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        rng: random.Random | None = None,
        torch_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        rng = rng or random.Random()
        tensor = (
            pil_to_tensor(image)
            if isinstance(image, Image.Image)
            else image.to(dtype=torch.float32)
        )
        if tensor.ndim != 3:
            raise ValueError(
                f"expected CHW tensor or PIL image, got shape {tuple(tensor.shape)}"
            )
        if tensor.min().item() < 0.0:
            tensor = tensor.add(1.0).div(2.0)

        tensor = self._downsample(tensor)
        tensor = self._blur(tensor, rng)
        tensor = self._noise(tensor, rng, torch_generator)
        tensor = self._jpeg(tensor, rng)

        if self.config.get("clamp", True):
            tensor = tensor.clamp(0.0, 1.0)
        return tensor

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.target_size is None:
            scale = float(self.config.get("scale", self.config.get("downsample", 1)))
            if scale <= 0:
                raise ValueError(f"scale must be positive, got {scale}")
            height, width = tensor.shape[-2:]
            target_h = max(1, int(round(height / scale)))
            target_w = max(1, int(round(width / scale)))
        else:
            target_h, target_w = self.target_size
        mode = str(self.config.get("mode", "bicubic"))
        align_corners = (
            False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        )
        kwargs = {"mode": mode}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        return F.interpolate(
            tensor.unsqueeze(0), size=(target_h, target_w), **kwargs
        ).squeeze(0)

    def _blur(self, tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
        spec = self.config.get("blur", self.config.get("gaussian_blur", None))
        if not self._enabled(spec):
            return tensor
        sigma = self._sample_value(spec, "sigma", rng, default=1.0)
        if sigma <= 0:
            return tensor
        return pil_to_tensor(
            tensor_to_pil(tensor).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        )

    def _noise(
        self,
        tensor: torch.Tensor,
        rng: random.Random,
        torch_generator: torch.Generator | None,
    ) -> torch.Tensor:
        spec = self.config.get("noise", self.config.get("gaussian_noise", None))
        if not self._enabled(spec):
            return tensor
        std = self._sample_value(spec, "std", rng, default=0.0)
        if std <= 0:
            return tensor
        mean = self._sample_value(spec, "mean", rng, default=0.0)
        noise = torch.randn(
            tensor.shape,
            generator=torch_generator,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor + noise.mul(float(std)).add(float(mean))

    def _jpeg(self, tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
        spec = self.config.get("jpeg", self.config.get("jpeg_compression", None))
        if not self._enabled(spec):
            return tensor
        quality = int(round(self._sample_value(spec, "quality", rng, default=95)))
        quality = max(1, min(100, quality))
        buffer = io.BytesIO()
        tensor_to_pil(tensor).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return pil_to_tensor(image.convert("RGB"))

    @staticmethod
    def _enabled(spec: Any) -> bool:
        if spec is None or spec is False:
            return False
        if isinstance(spec, Mapping):
            return bool(spec.get("enabled", True))
        return True

    @staticmethod
    def _sample_value(spec: Any, key: str, rng: random.Random, default: float) -> float:
        if isinstance(spec, Mapping):
            if key in spec:
                value = spec[key]
            elif f"{key}_min" in spec and f"{key}_max" in spec:
                return rng.uniform(
                    float(spec[f"{key}_min"]), float(spec[f"{key}_max"])
                )
            else:
                value = default
        elif isinstance(spec, (int, float)):
            value = spec
        else:
            value = default
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"range for {key!r} must have 2 values, got {value!r}")
            return rng.uniform(float(value[0]), float(value[1]))
        return float(value)


def _normalize_target_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = value
    else:
        raise ValueError(
            "target_size must be an int or a 2-item (height, width) sequence, "
            f"got {value!r}"
        )
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size dimensions must be positive, got {value!r}")
    return height, width
