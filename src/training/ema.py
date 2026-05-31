"""Exponential moving average helpers for generator weights."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from torch import nn


class EMAGenerator:
    """Maintain a detached EMA copy of a generator module."""

    def __init__(self, generator: nn.Module, decay: float = 0.999, device: torch.device | str | None = None) -> None:
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.module = deepcopy(generator)
        if device is not None:
            self.module.to(device)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, generator: nn.Module) -> None:
        """Update EMA parameters and copy non-floating buffers from ``generator``."""
        source_state = generator.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            source_value = source_state[name].detach().to(device=ema_value.device)
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpointable EMA state dictionary."""
        return {
            "decay": self.decay,
            "module": OrderedDict((k, v.detach().clone()) for k, v in self.module.state_dict().items()),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Restore EMA decay and generator weights from a checkpoint."""
        if "decay" in state_dict:
            self.decay = float(state_dict["decay"])
        module_state = state_dict.get("module", state_dict)
        self.module.load_state_dict(module_state, strict=strict)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)


__all__ = ["EMAGenerator"]
