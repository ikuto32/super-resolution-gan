"""Training CLI."""

from __future__ import annotations

import argparse
import inspect
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets import ImagePairDataset  # noqa: E402
from src.models import ProgressiveSRGenerator, ResolutionAgnosticDiscriminator  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.random import seed_everything  # noqa: E402


def _constructor_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _normalize_image_size(value: Any, key: str) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = value
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        height, width = value
    else:
        raise ValueError(f"{key} must be an int or a 2-item [height, width] sequence")

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{key} dimensions must be positive, got {value!r}")
    return height, width


def _expected_downsample_scale(data_cfg: dict[str, Any]) -> float:
    hr_h, hr_w = _normalize_image_size(data_cfg["image_size_hr"], "data.image_size_hr")
    lr_h, lr_w = _normalize_image_size(data_cfg["image_size_lr"], "data.image_size_lr")
    scale_h = hr_h / lr_h
    scale_w = hr_w / lr_w
    if not np.isclose(scale_h, scale_w):
        raise ValueError(
            "data.image_size_hr and data.image_size_lr must define a uniform "
            "downsample scale when degradation.downsample.scale is used: "
            f"height scale={scale_h:g}, width scale={scale_w:g}"
        )
    return float(scale_h)


def _degradation_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    image_size_lr = _normalize_image_size(
        data_cfg["image_size_lr"], "data.image_size_lr"
    )

    degradation = dict(config.get("degradation", {}))
    downsample = degradation.pop("downsample", {})
    if isinstance(downsample, dict):
        if "scale" in downsample:
            expected_scale = _expected_downsample_scale(data_cfg)
            configured_scale = float(downsample["scale"])
            if configured_scale <= 0:
                raise ValueError(
                    "degradation.downsample.scale must be positive, "
                    f"got {configured_scale:g}"
                )
            if not np.isclose(configured_scale, expected_scale):
                raise ValueError(
                    "degradation.downsample.scale does not match the scale implied by "
                    "data.image_size_hr and data.image_size_lr: "
                    f"configured={configured_scale:g}, expected={expected_scale:g}"
                )
            degradation["scale"] = configured_scale
        if "method" in downsample:
            degradation["mode"] = downsample["method"]
    degradation["target_size"] = image_size_lr
    return degradation


def _seed_worker(worker_id: int) -> None:
    """Seed per-worker Python and NumPy RNGs from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _data_loader_generator(seed: int) -> torch.Generator:
    """Create a DataLoader generator seeded from the training config."""

    return torch.Generator().manual_seed(int(seed))


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config["data"]
    seed = int(config.get("seed", 0))
    train_dataset = ImagePairDataset(
        data_cfg["train_dir"],
        crop_size=data_cfg.get("image_size_hr"),
        degradation=_degradation_config(config),
        validation=False,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=_data_loader_generator(seed),
    )

    val_loader = None
    val_dir = data_cfg.get("val_dir")
    if val_dir and Path(val_dir).exists():
        val_dataset = ImagePairDataset(
            val_dir,
            crop_size=data_cfg.get("image_size_hr"),
            degradation=_degradation_config(config),
            validation=True,
            seed=seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(data_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=_data_loader_generator(seed),
        )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the super-resolution GAN.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, default_path="configs/default.yaml")
    seed_everything(int(config.get("seed", 0)))

    generator = ProgressiveSRGenerator(**_constructor_kwargs(ProgressiveSRGenerator, config.get("model", {}).get("generator", {})))
    discriminator = ResolutionAgnosticDiscriminator(**_constructor_kwargs(ResolutionAgnosticDiscriminator, config.get("model", {}).get("discriminator", {})))
    train_loader, val_loader = build_dataloaders(config)
    trainer = Trainer(generator, discriminator, train_loader, val_loader, config=config)
    if args.resume:
        trainer.load(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
