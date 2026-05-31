"""Image transform utilities for super-resolution datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image

NumberPair: TypeAlias = tuple[int, int]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in ``[0, 1]`` with CHW layout."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL.Image.Image, got {type(image)!r}")

    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8).copy()
    data = torch.from_numpy(array)
    return data.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW/HWC tensor in ``[0, 1]`` or ``[-1, 1]`` to a PIL image."""
    if tensor.ndim != 3:
        raise ValueError(f"expected a 3D tensor, got shape {tuple(tensor.shape)}")

    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)

    if tensor.min().item() < 0.0:
        tensor = denormalize_minus_one_to_one(tensor)

    tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    if tensor.shape[-1] == 1:
        return Image.fromarray(tensor.squeeze(-1).numpy(), mode="L")
    return Image.fromarray(tensor.numpy())


def normalize_minus_one_to_one(tensor: torch.Tensor) -> torch.Tensor:
    """Map tensor values from ``[0, 1]`` to ``[-1, 1]``."""
    return tensor.mul(2.0).sub(1.0)


def denormalize_minus_one_to_one(tensor: torch.Tensor) -> torch.Tensor:
    """Map tensor values from ``[-1, 1]`` to ``[0, 1]``."""
    return tensor.add(1.0).div(2.0)


def _as_hw(size: int | Sequence[int]) -> NumberPair:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"size must contain 2 values, got {size!r}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"crop size must be positive, got {(height, width)!r}")
    return height, width


def _spatial_size(image: Image.Image | torch.Tensor) -> NumberPair:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if isinstance(image, torch.Tensor):
        if image.ndim < 2:
            raise ValueError(
                f"tensor must have spatial dimensions, got {tuple(image.shape)}"
            )
        return int(image.shape[-2]), int(image.shape[-1])
    raise TypeError(f"unsupported image type {type(image)!r}")


def _crop(
    image: Image.Image | torch.Tensor, top: int, left: int, height: int, width: int
):
    if isinstance(image, Image.Image):
        return image.crop((left, top, left + width, top + height))
    return image[..., top : top + height, left : left + width]


def random_crop(
    image: Image.Image | torch.Tensor,
    size: int | Sequence[int],
    rng: random.Random | None = None,
) -> Image.Image | torch.Tensor:
    """Crop a random spatial window from a PIL image or tensor."""
    crop_h, crop_w = _as_hw(size)
    image_h, image_w = _spatial_size(image)
    if crop_h > image_h or crop_w > image_w:
        raise ValueError(
            f"crop size {(crop_h, crop_w)!r} exceeds image size {(image_h, image_w)!r}"
        )

    rng = rng or random
    top = rng.randint(0, image_h - crop_h) if image_h > crop_h else 0
    left = rng.randint(0, image_w - crop_w) if image_w > crop_w else 0
    return _crop(image, top, left, crop_h, crop_w)


def center_crop(
    image: Image.Image | torch.Tensor, size: int | Sequence[int]
) -> Image.Image | torch.Tensor:
    """Crop the centered spatial window from a PIL image or tensor."""
    crop_h, crop_w = _as_hw(size)
    image_h, image_w = _spatial_size(image)
    if crop_h > image_h or crop_w > image_w:
        raise ValueError(
            f"crop size {(crop_h, crop_w)!r} exceeds image size {(image_h, image_w)!r}"
        )

    top = (image_h - crop_h) // 2
    left = (image_w - crop_w) // 2
    return _crop(image, top, left, crop_h, crop_w)
