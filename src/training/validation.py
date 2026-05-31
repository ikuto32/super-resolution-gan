"""Validation entry points for samples and metric aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from datasets.transforms import denormalize_minus_one_to_one, tensor_to_pil

MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            moved[key] = value
    return moved


def _default_metrics(sr: torch.Tensor, hr: torch.Tensor) -> dict[str, float]:
    """Compute default validation metrics on denormalized ``[0, 1]`` images.

    Training tensors are expected to use the project default ``[-1, 1]`` range,
    so convert SR/HR to image space before computing MSE and PSNR. Keeping PSNR
    on a ``[0, 1]`` data range makes the logged ``psnr`` value comparable to
    standard super-resolution reports.
    """
    sr_image = denormalize_minus_one_to_one(sr)
    hr_image = denormalize_minus_one_to_one(hr)
    mse = F.mse_loss(sr_image, hr_image).detach()
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {"mse": float(mse.cpu()), "psnr": float(psnr.cpu())}


def _save_sample_grid(
    lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: Path
) -> None:
    lr_up = F.interpolate(
        lr[:1], size=hr.shape[-2:], mode="bilinear", align_corners=False
    )
    grid = torch.cat([lr_up[0], sr[:1][0], hr[:1][0]], dim=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(grid).save(path)


@torch.no_grad()
def run_validation(
    generator: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str = "cpu",
    step: int = 0,
    output_dir: str | Path | None = None,
    metrics: Mapping[str, MetricFn] | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation, save sample triplets, and aggregate metrics."""
    device = torch.device(device)
    was_training = generator.training
    generator.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _batch_to_device(batch, device)
        lr = batch["lr"]
        hr = batch["hr"]
        output = generator(lr, target_size=hr.shape[-2:])
        sr = output["image"] if isinstance(output, Mapping) else output

        current = _default_metrics(sr, hr)
        if metrics:
            for name, metric in metrics.items():
                value = metric(sr, hr)
                current[name] = float(
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )
        for name, value in current.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1

        if output_dir is not None and batch_index == 0:
            _save_sample_grid(
                lr,
                sr,
                hr,
                Path(output_dir) / "validation" / f"step_{int(step):08d}.png",
            )

    if was_training:
        generator.train()
    if count == 0:
        return {}
    return {name: value / count for name, value in totals.items()}


__all__ = ["MetricFn", "run_validation"]
