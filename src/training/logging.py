"""Thin wrappers around TensorBoard, JSONL, and optional W&B logging."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import wandb

from datasets.transforms import tensor_to_pil

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - depends on optional tensorboard package.
    SummaryWriter = None  # type: ignore[assignment]


class TrainingLogger:
    """Write scalar/image logs to JSONL, TensorBoard, and optional Weights & Biases."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        enable_tensorboard: bool = True,
        wandb_config: Mapping[str, Any] | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self.writer = (
            SummaryWriter(str(self.log_dir))
            if enable_tensorboard and SummaryWriter is not None
            else None
        )
        self._wandb_run = self._init_wandb(wandb_config or {}, run_config or {})

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _init_wandb(
        self,
        wandb_config: Mapping[str, Any],
        run_config: Mapping[str, Any],
    ) -> Any | None:
        if not bool(wandb_config.get("enabled", False)):
            return None
        init_kwargs: dict[str, Any] = {
            "config": dict(run_config),
            "dir": str(self.log_dir.parent),
        }
        for key in (
            "project",
            "entity",
            "name",
            "group",
            "job_type",
            "mode",
            "id",
            "resume",
            "notes",
        ):
            value = wandb_config.get(key)
            if value is not None:
                init_kwargs[key] = value
        tags = wandb_config.get("tags")
        if tags is not None:
            init_kwargs["tags"] = (
                list(tags)
                if isinstance(tags, Sequence) and not isinstance(tags, str)
                else tags
            )
        return wandb.init(**init_kwargs)

    def log_scalars(
        self, scalars: Mapping[str, Any], step: int, prefix: str | None = None
    ) -> None:
        """Log a dictionary of scalar values."""
        normalized: dict[str, float] = {}
        for key, value in scalars.items():
            if value is None:
                continue
            name = f"{prefix}/{key}" if prefix else str(key)
            normalized[name] = self._to_float(value)
            if self.writer is not None:
                self.writer.add_scalar(name, normalized[name], step)
        self._jsonl.write(
            json.dumps({"step": int(step), **normalized}, sort_keys=True) + "\n"
        )
        self._jsonl.flush()
        if self._wandb_run is not None:
            wandb.log(normalized, step=int(step))

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """Log a BCHW or CHW image tensor to TensorBoard and W&B when available."""
        detached = images.detach().cpu()
        if self.writer is not None:
            if detached.ndim == 3:
                self.writer.add_image(tag, detached, step)
            else:
                self.writer.add_images(tag, detached, step)
        if self._wandb_run is not None:
            wandb_images = [
                wandb.Image(tensor_to_pil(image))
                for image in self._iter_images(detached)
            ]
            wandb.log({tag: wandb_images}, step=int(step))

    @staticmethod
    def _iter_images(images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim == 3:
            return [images]
        if images.ndim == 4:
            return [image for image in images]
        raise ValueError(
            f"expected a CHW or BCHW tensor, got shape {tuple(images.shape)}"
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self._wandb_run is not None:
            wandb.finish()
        self._jsonl.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["TrainingLogger"]
