"""Dataset returning normalized LR/HR super-resolution training pairs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from datasets.degradation import DegradationPipeline
from datasets.pyramid import build_image_pyramid
from datasets.transforms import (
    center_crop,
    normalize_minus_one_to_one,
    pil_to_tensor,
    random_crop,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImagePairDataset(Dataset):
    """Load HR images and generate or read aligned LR images.

    Items are returned as ``{"lr": ..., "hr": ..., "hr_pyramid": ..., "meta": ...}``.
    Training mode derives per-sample random state from the DataLoader worker seed
    and sample index, making stochastic crops/degradations controllable through
    the DataLoader seed. Validation mode derives all random state from ``seed``
    and the sample index so repeated access is stable.
    """

    def __init__(
        self,
        hr_root: str | Path | Sequence[str | Path],
        lr_root: str | Path | None = None,
        crop_size: int | Sequence[int] | None = None,
        degradation: DegradationPipeline | dict[str, Any] | None = None,
        pyramid_scales: Sequence[float | int] = (1, 2, 4),
        validation: bool = False,
        seed: int = 0,
        normalize: bool = True,
    ) -> None:
        self.hr_paths = self._collect_paths(hr_root)
        if not self.hr_paths:
            raise ValueError(f"no images found in {hr_root!r}")
        self.lr_root = Path(lr_root) if lr_root is not None else None
        self.crop_size = crop_size
        if isinstance(degradation, DegradationPipeline):
            self.degradation = degradation
        else:
            self.degradation = DegradationPipeline(degradation or {})
        self.pyramid_scales = tuple(pyramid_scales)
        self.validation = validation
        self.seed = int(seed)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        hr_path = self.hr_paths[index]
        with Image.open(hr_path) as image:
            hr_image = image.convert("RGB")

        item_seed = self._seed_for_index(index)
        rng = random.Random(item_seed)
        if self.crop_size is not None:
            crop = center_crop if self.validation else random_crop
            hr_image = (
                crop(hr_image, self.crop_size, rng)
                if crop is random_crop
                else crop(hr_image, self.crop_size)
            )

        hr = pil_to_tensor(hr_image)
        lr_path = self._lr_path_for(hr_path)
        if lr_path is not None and lr_path.exists():
            with Image.open(lr_path) as image:
                lr = pil_to_tensor(image.convert("RGB"))
        else:
            torch_generator = torch.Generator().manual_seed(item_seed)
            lr = self.degradation(hr, rng=rng, torch_generator=torch_generator)

        hr_pyramid = build_image_pyramid(hr, self.pyramid_scales)

        if self.normalize:
            hr = normalize_minus_one_to_one(hr)
            lr = normalize_minus_one_to_one(lr.clamp(0.0, 1.0))
            hr_pyramid = {
                key: normalize_minus_one_to_one(value.clamp(0.0, 1.0))
                for key, value in hr_pyramid.items()
            }

        return {
            "lr": lr,
            "hr": hr,
            "hr_pyramid": hr_pyramid,
            "meta": {
                "index": index,
                "hr_path": str(hr_path),
                "lr_path": str(lr_path) if lr_path is not None else "",
                "validation": self.validation,
            },
        }

    @staticmethod
    def _collect_paths(root_or_paths: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(root_or_paths, (str, Path)):
            root = Path(root_or_paths)
            if root.is_file():
                return [root]
            return sorted(
                path
                for path in root.rglob("*")
                if path.suffix.lower() in _IMAGE_EXTENSIONS
            )
        return sorted(Path(path) for path in root_or_paths)

    def _lr_path_for(self, hr_path: Path) -> Path | None:
        if self.lr_root is None:
            return None
        relative = Path(hr_path.name)
        try:
            common = Path(*hr_path.parts[:-1])
            relative = hr_path.relative_to(common)
        except ValueError:
            pass
        direct = self.lr_root / hr_path.name
        nested = self.lr_root / relative
        return nested if nested.exists() else direct

    def _seed_for_index(self, index: int) -> int:
        """Return a deterministic per-item seed for Python and PyTorch RNGs."""

        normalized_index = int(index)
        if self.validation:
            base_seed = self.seed
        else:
            worker_info = get_worker_info()
            if worker_info is None:
                base_seed = self.seed
            else:
                base_seed = torch.initial_seed() - worker_info.id
        return _mix_seed(base_seed, normalized_index)


def _mix_seed(base_seed: int, index: int) -> int:
    """Mix a base seed with an item index into a 64-bit seed value."""

    mask = (1 << 64) - 1
    mixed = (int(base_seed) + 0x9E3779B97F4A7C15) & mask
    mixed ^= (int(index) + 0xBF58476D1CE4E5B9) & mask
    mixed = (mixed * 0x94D049BB133111EB) & mask
    mixed ^= mixed >> 31
    return mixed
