from __future__ import annotations

import pytest
import torch
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


class ResidualValidationGenerator(nn.Module):
    def forward(self, lr: torch.Tensor, target_size) -> dict[str, object]:
        baseline = torch.nn.functional.interpolate(
            lr, size=target_size, mode="bilinear", align_corners=False
        )
        residual = torch.zeros_like(baseline)
        image = baseline + residual
        return {
            "baseline": baseline,
            "residual": residual,
            "image": image,
            "pyramid": {1: image},
        }


def test_run_validation_accepts_residual_generator_mapping(tmp_path) -> None:
    lr = torch.zeros(1, 3, 4, 4)
    hr = torch.zeros(1, 3, 8, 8)
    generator = ResidualValidationGenerator()
    generator.train()

    metrics = run_validation(
        generator,
        [{"lr": lr, "hr": hr, "hr_pyramid": {1: hr}}],
        device="cpu",
        step=7,
        output_dir=tmp_path,
    )

    assert generator.training
    assert set(metrics) == {"mse", "psnr"}
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["psnr"] == pytest.approx(120.0)
    assert (tmp_path / "validation" / "step_00000007.png").exists()
