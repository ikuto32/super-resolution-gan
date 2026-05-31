"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.training.ema import EMAGenerator
from src.utils.random import get_rng_state, set_rng_state

CHECKPOINT_KEYS = (
    "step",
    "next_epoch",
    "seen_images",
    "generator",
    "discriminator",
    "generator_ema",
    "optimizer_g",
    "optimizer_d",
    "scheduler_g",
    "scheduler_d",
    "grad_scaler",
    "config",
    "rng_state",
)


def _maybe_state_dict(obj: Any) -> Any:
    if obj is None:
        return None
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    next_epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    generator_ema: EMAGenerator | nn.Module | None = None,
    scheduler_g: Any | None = None,
    scheduler_d: Any | None = None,
    grad_scaler: Any | None = None,
    config: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    seen_images: int | None = None,
) -> dict[str, Any]:
    """Save a design-compliant training checkpoint and return its payload.

    ``next_epoch`` is the epoch index that ``Trainer.fit()`` should use as
    the start of ``range(next_epoch, epochs)`` when resuming.
    """
    checkpoint = {
        "step": int(step),
        "next_epoch": int(next_epoch),
        "seen_images": int(seen_images) if seen_images is not None else None,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_ema": _maybe_state_dict(generator_ema),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": _maybe_state_dict(scheduler_g),
        "scheduler_d": _maybe_state_dict(scheduler_d),
        "grad_scaler": _maybe_state_dict(grad_scaler),
        "config": dict(config or {}),
        "rng_state": rng_state if rng_state is not None else get_rng_state(),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module | None = None,
    discriminator: nn.Module | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    generator_ema: EMAGenerator | nn.Module | None = None,
    scheduler_g: Any | None = None,
    scheduler_d: Any | None = None,
    grad_scaler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore provided training objects."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)

    if generator is not None and checkpoint.get("generator") is not None:
        generator.load_state_dict(checkpoint["generator"], strict=strict)
    if discriminator is not None and checkpoint.get("discriminator") is not None:
        discriminator.load_state_dict(checkpoint["discriminator"], strict=strict)
    if optimizer_g is not None and checkpoint.get("optimizer_g") is not None:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_d is not None and checkpoint.get("optimizer_d") is not None:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if scheduler_g is not None and checkpoint.get("scheduler_g") is not None:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if scheduler_d is not None and checkpoint.get("scheduler_d") is not None:
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    if grad_scaler is not None and checkpoint.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    if generator_ema is not None and checkpoint.get("generator_ema") is not None:
        generator_ema.load_state_dict(checkpoint["generator_ema"], strict=strict)
    if restore_rng and checkpoint.get("rng_state") is not None:
        set_rng_state(checkpoint["rng_state"])

    return checkpoint


__all__ = ["CHECKPOINT_KEYS", "save_checkpoint", "load_checkpoint"]
