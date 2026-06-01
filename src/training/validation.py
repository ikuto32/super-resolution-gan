"""Validation entry points for samples and metric aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from datasets.transforms import denormalize_minus_one_to_one, tensor_to_pil
from src.utils.tensors import batch_to_device

MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]


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


def _image_for_display(tensor: torch.Tensor) -> torch.Tensor:
    """Convert image tensors to display-space ``[0, 1]`` values."""
    tensor = tensor.detach().to(dtype=torch.float32)
    if tensor.min().item() < 0.0:
        tensor = denormalize_minus_one_to_one(tensor)
    return tensor.clamp(0.0, 1.0)


def _normalize_residual_for_display(
    residual: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Map residual tensors to ``[0, 1]`` with zero residual shown as mid-gray.

    Residual predictions and targets can be centered around zero and can exceed
    the normal image display range. Normalize each sample by its own maximum
    absolute residual so positive and negative errors remain visible without
    clipping.
    """
    residual = residual.detach().to(dtype=torch.float32)
    if residual.ndim < 2:
        raise ValueError(
            f"expected residual tensor with at least 2 dimensions, got {tuple(residual.shape)}"
        )
    reduce_dims = (
        tuple(range(1, residual.ndim))
        if residual.ndim > 3
        else tuple(range(residual.ndim))
    )
    max_abs = residual.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return residual.div(max_abs).mul(0.5).add(0.5).clamp(0.0, 1.0)


def _normalize_magnitude_for_display(
    magnitude: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize non-negative magnitude maps to display-space ``[0, 1]``."""
    magnitude = magnitude.detach().to(dtype=torch.float32).clamp_min(0.0)
    reduce_dims = (
        tuple(range(1, magnitude.ndim))
        if magnitude.ndim > 3
        else tuple(range(magnitude.ndim))
    )
    max_value = magnitude.amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return magnitude.div(max_value).clamp(0.0, 1.0)


def _make_sample_grid(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    pred_residual: torch.Tensor | None = None,
    target_residual: torch.Tensor | None = None,
    error_map: torch.Tensor | None = None,
    max_images: int = 1,
) -> torch.Tensor:
    """Build a validation/training sample grid in display-space."""
    max_images = max(1, min(max_images, int(hr.shape[0])))
    lr_up = F.interpolate(
        lr[:max_images], size=hr.shape[-2:], mode="bilinear", align_corners=False
    )
    has_residual_panels = all(
        tensor is not None
        for tensor in (baseline, pred_residual, target_residual, error_map)
    )

    if has_residual_panels:
        assert baseline is not None
        assert pred_residual is not None
        assert target_residual is not None
        assert error_map is not None
        panel_batches = [
            _image_for_display(lr_up),
            _image_for_display(baseline[:max_images]),
            _normalize_residual_for_display(pred_residual[:max_images]),
            _normalize_residual_for_display(target_residual[:max_images]),
            _image_for_display(sr[:max_images]),
            _image_for_display(hr[:max_images]),
            _normalize_magnitude_for_display(error_map[:max_images]),
        ]
    else:
        panel_batches = [
            _image_for_display(lr_up),
            _image_for_display(sr[:max_images]),
            _image_for_display(hr[:max_images]),
        ]

    rows = [
        torch.cat([panel_batch[index] for panel_batch in panel_batches], dim=-1)
        for index in range(max_images)
    ]
    return torch.cat(rows, dim=-2)


def _save_sample_grid(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    path: Path,
    *,
    baseline: torch.Tensor | None = None,
    pred_residual: torch.Tensor | None = None,
    target_residual: torch.Tensor | None = None,
    error_map: torch.Tensor | None = None,
) -> None:
    grid = _make_sample_grid(
        lr,
        sr,
        hr,
        baseline=baseline,
        pred_residual=pred_residual,
        target_residual=target_residual,
        error_map=error_map,
    )
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
        batch = batch_to_device(batch, device)
        lr = batch["lr"]
        hr = batch["hr"]
        output = generator(lr, target_size=hr.shape[-2:])
        sr = output["image"] if isinstance(output, Mapping) else output
        baseline = pred_residual = target_residual = error_map = None
        if (
            isinstance(output, Mapping)
            and "baseline" in output
            and "residual" in output
        ):
            baseline = output["baseline"]
            pred_residual = output["residual"]
            target_residual = hr - baseline
            error_map = (sr - hr).abs()

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
                baseline=baseline,
                pred_residual=pred_residual,
                target_residual=target_residual,
                error_map=error_map,
            )

    if was_training:
        generator.train()
    if count == 0:
        return {}
    return {name: value / count for name, value in totals.items()}


__all__ = ["MetricFn", "run_validation"]
