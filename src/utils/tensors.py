"""Tensor utility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Move tensors in ``value`` to ``device`` while preserving non-tensors.

    Tensors are moved with ``Tensor.to(device)``. Mappings are traversed
    recursively so tensors nested at any mapping depth are moved, while
    non-tensor values are returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def batch_to_device(
    batch: Mapping[str, Any], device: torch.device | str
) -> dict[str, Any]:
    """Move tensor values in a batch mapping to ``device`` recursively."""
    return {key: move_to_device(value, device) for key, value in batch.items()}
