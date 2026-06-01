from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.training.validation import _default_metrics, run_validation, sample_grid_extras


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


def test_sample_grid_extras_extracts_residual_panels() -> None:
    baseline = torch.zeros(1, 1, 2, 2)
    residual = torch.full_like(baseline, 0.25)
    hr = torch.ones_like(baseline)
    sr = baseline + residual

    extras = sample_grid_extras({"baseline": baseline, "residual": residual}, sr, hr)

    assert extras["baseline"] is baseline
    assert extras["pred_residual"] is residual
    assert torch.equal(extras["target_residual"], hr - baseline)
    assert torch.equal(extras["error_map"], (sr - hr).abs())


def test_sample_grid_extras_returns_none_panels_without_residual_output() -> None:
    hr = torch.ones(1, 1, 2, 2)
    sr = torch.zeros_like(hr)

    extras = sample_grid_extras({"image": sr}, sr, hr)

    assert extras == {
        "baseline": None,
        "pred_residual": None,
        "target_residual": None,
        "error_map": None,
    }


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
