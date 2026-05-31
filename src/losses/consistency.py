"""Low-resolution consistency losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _interpolate_kwargs(mode: str) -> dict[str, bool | str]:
    kwargs: dict[str, bool | str] = {"mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return kwargs


def lr_consistency_loss(
    sr: torch.Tensor,
    lr: torch.Tensor,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Downsample ``sr`` to ``lr`` spatial size and compute L1 consistency."""
    if sr.ndim != lr.ndim:
        raise ValueError("sr and lr must have the same number of dimensions")

    downsampled = F.interpolate(
        sr,
        size=lr.shape[-2:],
        **_interpolate_kwargs(mode),
    )
    return F.l1_loss(downsampled, lr)
