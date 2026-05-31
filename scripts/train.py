"""Training CLI."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from datasets import ImagePairDataset  # noqa: E402
from src.models import ProgressiveSRGenerator, ResolutionAgnosticDiscriminator  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.random import seed_everything  # noqa: E402


def _constructor_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _degradation_config(config: dict[str, Any]) -> dict[str, Any]:
    degradation = dict(config.get("degradation", {}))
    downsample = degradation.pop("downsample", {})
    if isinstance(downsample, dict):
        if "scale" in downsample:
            degradation["scale"] = downsample["scale"]
        if "method" in downsample:
            degradation["mode"] = downsample["method"]
    return degradation


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config["data"]
    train_dataset = ImagePairDataset(
        data_cfg["train_dir"],
        crop_size=data_cfg.get("image_size_hr"),
        degradation=_degradation_config(config),
        validation=False,
        seed=int(config.get("seed", 0)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    val_loader = None
    val_dir = data_cfg.get("val_dir")
    if val_dir and Path(val_dir).exists():
        val_dataset = ImagePairDataset(
            val_dir,
            crop_size=data_cfg.get("image_size_hr"),
            degradation=_degradation_config(config),
            validation=True,
            seed=int(config.get("seed", 0)),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(data_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=True,
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
