"""Optimizer construction helpers for training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _optimizer_config(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """Return the optimizer config for ``section`` from flexible config shapes."""
    if section in config and isinstance(config[section], Mapping):
        return config[section]
    optimizer = config.get("optimizer")
    if isinstance(optimizer, Mapping) and isinstance(optimizer.get(section), Mapping):
        return optimizer[section]
    return config


def build_adamw_optimizer(
    parameters,
    config: Mapping[str, Any] | None = None,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer from a training config mapping."""
    config = config or {}
    optimizer_type = str(config.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"unsupported optimizer type {optimizer_type!r}; expected 'adamw'")

    betas = config.get("betas", (0.0, 0.99))
    if len(betas) != 2:
        raise ValueError(f"AdamW betas must contain two values, got {betas!r}")

    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("lr", 2e-4)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        amsgrad=bool(config.get("amsgrad", False)),
    )


def build_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    config: Mapping[str, Any],
) -> tuple[torch.optim.AdamW, torch.optim.AdamW]:
    """Build separate AdamW optimizers for generator and discriminator."""
    optimizer_g = build_adamw_optimizer(
        generator.parameters(), _optimizer_config(config, "generator")
    )
    optimizer_d = build_adamw_optimizer(
        discriminator.parameters(), _optimizer_config(config, "discriminator")
    )
    return optimizer_g, optimizer_d


__all__ = ["build_adamw_optimizer", "build_optimizers"]
