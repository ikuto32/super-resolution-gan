from __future__ import annotations

import pytest
import torch

from src.training.validation import _default_metrics


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
