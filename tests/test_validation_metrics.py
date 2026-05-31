from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.training.validation import _default_metrics, run_validation


def test_default_metrics_exact_match_has_zero_mse_and_finite_psnr() -> None:
    hr = torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]])
    sr = hr.clone()

    metrics = _default_metrics(sr, hr)

    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == pytest.approx(120.0)


def test_default_metrics_denormalizes_before_known_mse_psnr() -> None:
    sr = torch.full((1, 1, 2, 2), -1.0)
    hr = torch.zeros((1, 1, 2, 2))

    metrics = _default_metrics(sr, hr)

    assert metrics["mse"] == pytest.approx(0.25)
    assert metrics["psnr"] == pytest.approx(6.020599913, rel=1e-6)


class ResidualToyGenerator(nn.Module):
    def forward(self, lr: torch.Tensor, target_size) -> dict[str, torch.Tensor]:
        baseline = torch.nn.functional.interpolate(
            lr, size=target_size, mode="bilinear", align_corners=False
        )
        residual = torch.full_like(baseline, 2.0)
        return {
            "image": baseline + residual,
            "baseline": baseline,
            "residual": residual,
        }


def test_validation_saves_residual_sample_grid(tmp_path: Path) -> None:
    lr = torch.zeros(1, 3, 4, 4)
    hr = torch.ones(1, 3, 8, 8)
    dataloader = [{"lr": lr, "hr": hr}]

    metrics = run_validation(
        ResidualToyGenerator(), dataloader, output_dir=tmp_path, max_batches=1
    )

    sample_path = tmp_path / "validation" / "step_00000000.png"
    assert sample_path.exists()
    with Image.open(sample_path) as image:
        assert image.size == (7 * 8, 8)
    assert set(metrics) == {"mse", "psnr"}
