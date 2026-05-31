"""Thin wrappers around TensorBoard and JSONL scalar logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - depends on optional tensorboard package.
    SummaryWriter = None  # type: ignore[assignment]


class TrainingLogger:
    """Write scalar/image logs to TensorBoard when available and JSONL always."""

    def __init__(self, log_dir: str | Path, *, enable_tensorboard: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self.writer = (
            SummaryWriter(str(self.log_dir))
            if enable_tensorboard and SummaryWriter is not None
            else None
        )

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def log_scalars(self, scalars: Mapping[str, Any], step: int, prefix: str | None = None) -> None:
        """Log a dictionary of scalar values."""
        normalized: dict[str, float] = {}
        for key, value in scalars.items():
            if value is None:
                continue
            name = f"{prefix}/{key}" if prefix else str(key)
            normalized[name] = self._to_float(value)
            if self.writer is not None:
                self.writer.add_scalar(name, normalized[name], step)
        self._jsonl.write(json.dumps({"step": int(step), **normalized}, sort_keys=True) + "\n")
        self._jsonl.flush()

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """Log a BCHW or CHW image tensor to TensorBoard when available."""
        if self.writer is not None:
            if images.ndim == 3:
                self.writer.add_image(tag, images.detach().cpu(), step)
            else:
                self.writer.add_images(tag, images.detach().cpu(), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self._jsonl.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["TrainingLogger"]
