from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses import degraded_noisy_state, denoising_loss, generator_total_loss
from src.models import ProgressiveSRGenerator
from src.training import Trainer


def test_diffusion_denoising_loss_is_finite_for_non_power_of_two_sizes():
    generator = ProgressiveSRGenerator(
        base_channels=8,
        max_channels=8,
        num_res_blocks_per_stage=1,
        pyramid_scales=(1, 3),
    )
    hr = torch.randn(2, 3, 23, 37)
    timesteps = torch.tensor([0, 9])
    state = degraded_noisy_state(
        hr,
        timesteps,
        downscale=3,
        num_timesteps=10,
        noise_min=0.0,
        noise_max=0.05,
        degradation_min=0.0,
        degradation_max=0.25,
    )

    output = generator(
        state["x_t"],
        target_size=hr.shape[-2:],
        diffusion_timestep=state["timesteps"],
        noisy_condition=state["x_t"],
        return_diffusion=True,
        return_intermediates=False,
    )
    loss = denoising_loss(output["diffusion"], hr)

    assert output["diffusion"].shape == hr.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_generator_total_loss_keeps_diffusion_zero_when_lambda_is_zero():
    sr = torch.zeros(1, 3, 15, 17)
    hr = torch.ones(1, 3, 15, 17)
    diffusion_loss = torch.tensor(123.0)

    losses = generator_total_loss(
        fake_scores=torch.zeros(1, 1),
        sr=sr,
        hr=hr,
        diffusion_loss=diffusion_loss,
        weights={
            "lambda_adv": 0.0,
            "lambda_pixel": 1.0,
            "lambda_multiscale": 0.0,
            "lambda_perceptual": 0.0,
            "lambda_consistency": 0.0,
            "lambda_diffusion": 0.0,
        },
    )

    assert losses["loss_diffusion"].item() == 0.0
    assert torch.isclose(losses["loss_total"], losses["loss_pixel"])


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(3, 3, kernel_size=1)

    def forward(
        self,
        lr: torch.Tensor,
        target_size,
        *,
        noisy_condition: torch.Tensor | None = None,
        diffusion_timestep: torch.Tensor | None = None,
        return_diffusion: bool = False,
    ) -> dict[str, object]:
        condition = noisy_condition if noisy_condition is not None else lr
        sr = torch.nn.functional.interpolate(
            condition, size=target_size, mode="bilinear", align_corners=False
        )
        sr = self.net(sr)
        output: dict[str, object] = {"image": sr, "pyramid": {1: sr}}
        if return_diffusion:
            assert diffusion_timestep is not None
            output["diffusion"] = sr
        return output


class TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(6, 1, kernel_size=1)

    def forward(self, image: torch.Tensor, condition: torch.Tensor):
        condition = torch.nn.functional.interpolate(
            condition, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.net(torch.cat([image, condition], dim=1))
        return {"score": logits.mean(dim=(1, 2, 3))}


def _loader() -> DataLoader:
    lr = torch.randn(2, 3, 5, 7)
    hr = torch.randn(2, 3, 15, 21)
    dataset = TensorDataset(lr, hr)

    def collate(items):
        return {
            "lr": torch.stack([item[0] for item in items]),
            "hr": torch.stack([item[1] for item in items]),
        }

    return DataLoader(dataset, batch_size=2, collate_fn=collate)


def _config(tmp_path, lambda_diffusion: float) -> dict[str, object]:
    return {
        "project": {"output_dir": str(tmp_path)},
        "training": {
            "epochs": 1,
            "mixed_precision": False,
            "n_critic": 1,
            "log_every": 0,
            "validate_every": 0,
            "save_every": 0,
            "sample_every_kimg": 0,
            "ema": {"enabled": False},
        },
        "logging": {"tensorboard": False, "wandb": {"enabled": False}},
        "optimizer": {
            "generator": {"type": "adamw", "lr": 1e-3, "betas": [0.0, 0.99]},
            "discriminator": {"type": "adamw", "lr": 1e-3, "betas": [0.0, 0.99]},
        },
        "loss": {
            "lambda_adv": 0.0,
            "lambda_pixel": 1.0,
            "lambda_multiscale": 0.0,
            "lambda_perceptual": 0.0,
            "lambda_consistency": 0.0,
            "lambda_diffusion": lambda_diffusion,
            "lambda_r1": 0.0,
            "lambda_r2": 0.0,
            "diffusion": {
                "num_timesteps": 8,
                "schedule": "linear",
                "downscale": 3,
                "loss_type": "l1",
                "degradation": {"mode": "bicubic", "strength_min": 0.0, "strength_max": 0.2},
                "noise": {"std_min": 0.0, "std_max": 0.03},
            },
        },
    }


def test_trainer_adds_finite_diffusion_auxiliary_loss_when_enabled(tmp_path):
    trainer = Trainer(
        TinyGenerator(),
        TinyDiscriminator(),
        _loader(),
        config=_config(tmp_path, lambda_diffusion=0.5),
        device="cpu",
    )

    logs = trainer.train_step(next(iter(_loader())))

    assert torch.isfinite(torch.tensor(logs["loss_diffusion"]))
    assert logs["loss_diffusion"] > 0.0
