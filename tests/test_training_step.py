from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import EMAGenerator, Trainer, load_checkpoint, save_checkpoint


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, lr: torch.Tensor, target_size) -> dict[str, object]:
        sr = torch.nn.functional.interpolate(
            lr, size=target_size, mode="bilinear", align_corners=False
        )
        sr = self.net(sr)
        return {"image": sr, "pyramid": {1: sr}}


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class TypeErrorDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _ = condition
        raise TypeError("internal discriminator failure")


class TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(6, 1, kernel_size=1)

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        condition = torch.nn.functional.interpolate(
            condition, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.net(torch.cat([image, condition], dim=1))
        return {"score": logits.mean(dim=(1, 2, 3)), "patch_logits": logits}


def _loader() -> DataLoader:
    lr = torch.randn(2, 3, 4, 4)
    hr = torch.randn(2, 3, 8, 8)
    dataset = TensorDataset(lr, hr)

    def collate(items):
        lr_batch = torch.stack([item[0] for item in items])
        hr_batch = torch.stack([item[1] for item in items])
        return {"lr": lr_batch, "hr": hr_batch, "hr_pyramid": {1: hr_batch}}

    return DataLoader(dataset, batch_size=2, collate_fn=collate)


def _config(tmp_path):
    return {
        "project": {"output_dir": str(tmp_path)},
        "training": {
            "epochs": 1,
            "mixed_precision": False,
            "grad_clip_norm": 1.0,
            "n_critic": 1,
            "log_every": 1,
            "validate_every": 0,
            "save_every": 0,
            "sample_every_kimg": 0,
            "sample_max_images": 2,
            "ema": {"enabled": True, "decay": 0.5},
        },
        "logging": {"tensorboard": False, "wandb": {"enabled": False}},
        "optimizer": {
            "generator": {
                "type": "adamw",
                "lr": 1e-3,
                "betas": [0.0, 0.99],
                "weight_decay": 0.0,
            },
            "discriminator": {
                "type": "adamw",
                "lr": 1e-3,
                "betas": [0.0, 0.99],
                "weight_decay": 0.0,
            },
        },
        "loss": {
            "lambda_adv": 0.1,
            "lambda_pixel": 1.0,
            "lambda_multiscale": 0.0,
            "lambda_perceptual": 0.0,
            "lambda_consistency": 0.1,
            "lambda_diffusion": 0.0,
            "lambda_r1": 0.0,
            "lambda_r2": 0.0,
        },
    }


def test_cpu_tiny_training_step_checkpoint_and_no_nan(tmp_path):
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    trainer = Trainer(
        generator, discriminator, _loader(), config=_config(tmp_path), device="cpu"
    )
    assert trainer.scaler is None

    batch = next(iter(_loader()))
    logs = trainer.train_step(batch)

    assert trainer.step == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in logs.values())
    assert all(torch.isfinite(parameter).all() for parameter in generator.parameters())
    assert all(
        torch.isfinite(parameter).all() for parameter in discriminator.parameters()
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    saved = save_checkpoint(
        checkpoint_path,
        step=trainer.step,
        next_epoch=trainer.epoch,
        generator=trainer.generator,
        discriminator=trainer.discriminator,
        generator_ema=trainer.ema,
        optimizer_g=trainer.optimizer_g,
        optimizer_d=trainer.optimizer_d,
        config=trainer.config,
    )
    assert checkpoint_path.exists()
    assert {
        "step",
        "next_epoch",
        "seen_images",
        "generator",
        "discriminator",
        "generator_ema",
        "optimizer_g",
        "optimizer_d",
        "scheduler_g",
        "scheduler_d",
        "grad_scaler",
        "config",
        "rng_state",
    }.issubset(saved)
    assert "epoch" not in saved

    restored_g = TinyGenerator()
    restored_d = TinyDiscriminator()
    restored_ema = EMAGenerator(restored_g, decay=0.5)
    optimizer_g = torch.optim.AdamW(restored_g.parameters(), lr=1e-3)
    optimizer_d = torch.optim.AdamW(restored_d.parameters(), lr=1e-3)
    loaded = load_checkpoint(
        checkpoint_path,
        generator=restored_g,
        discriminator=restored_d,
        generator_ema=restored_ema,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        restore_rng=False,
    )

    assert loaded["step"] == 1
    for saved_parameter, restored_parameter in zip(
        trainer.generator.parameters(), restored_g.parameters(), strict=True
    ):
        assert torch.allclose(saved_parameter, restored_parameter)


def test_resume_after_completed_epoch_does_not_rerun_epoch(tmp_path):
    config = _config(tmp_path)
    trainer = Trainer(
        TinyGenerator(), TinyDiscriminator(), _loader(), config=config, device="cpu"
    )

    trainer.fit()

    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    completed = load_checkpoint(checkpoint_path, restore_rng=False)
    assert completed["next_epoch"] == 1
    assert completed["step"] == 1
    assert "epoch" not in completed

    resumed = Trainer(
        TinyGenerator(), TinyDiscriminator(), _loader(), config=config, device="cpu"
    )
    resumed.load(checkpoint_path, restore_rng=False)
    assert resumed.epoch == 1

    resumed.fit()

    reloaded = load_checkpoint(checkpoint_path, restore_rng=False)
    assert resumed.step == 1
    assert reloaded["next_epoch"] == 1
    assert reloaded["step"] == 1


def test_training_step_writes_samples_every_configured_kimg(tmp_path):
    config = _config(tmp_path)
    config["training"]["sample_every_kimg"] = 0.001
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    trainer = Trainer(generator, discriminator, _loader(), config=config, device="cpu")

    trainer.train_step(next(iter(_loader())))

    samples = list((tmp_path / "samples").glob("step_*.png"))
    assert len(samples) == 1
    assert trainer.seen_images == 2


def test_training_step_honors_n_critic(tmp_path):
    config = _config(tmp_path)
    config["training"]["n_critic"] = 3
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    optimizer_g = CountingSGD(generator.parameters(), lr=1e-3)
    optimizer_d = CountingSGD(discriminator.parameters(), lr=1e-3)
    trainer = Trainer(
        generator,
        discriminator,
        _loader(),
        config=config,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device="cpu",
    )

    logs = trainer.train_step(next(iter(_loader())))

    assert trainer.step == 1
    assert optimizer_d.step_calls == 3
    assert optimizer_g.step_calls == 1
    assert torch.isfinite(torch.tensor(logs["loss_d"]))


def test_discriminator_internal_type_error_is_not_swallowed(tmp_path):
    trainer = Trainer(
        TinyGenerator(),
        TypeErrorDiscriminator(),
        _loader(),
        config=_config(tmp_path),
        device="cpu",
    )

    with pytest.raises(TypeError, match="internal discriminator failure"):
        trainer.train_step(next(iter(_loader())))


def test_training_step_passes_perceptual_loss_when_weight_is_positive(
    tmp_path, monkeypatch
):
    class RecordingPerceptual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return (sr - hr).abs().mean()

    perceptual_module = RecordingPerceptual()
    monkeypatch.setattr(
        "src.training.trainer.build_perceptual_loss",
        lambda loss_config: perceptual_module,
    )
    config = _config(tmp_path)
    config["loss"]["lambda_perceptual"] = 0.5

    trainer = Trainer(
        TinyGenerator(), TinyDiscriminator(), _loader(), config=config, device="cpu"
    )
    logs = trainer.train_step(next(iter(_loader())))

    assert trainer.perceptual_loss is perceptual_module
    assert perceptual_module.calls == 1
    assert logs["loss_perceptual"] > 0.0


class ResidualGenerator(nn.Module):
    def __init__(self, residual_value: float = 0.25) -> None:
        super().__init__()
        self.residual = nn.Parameter(torch.full((1, 3, 1, 1), residual_value))

    def forward(self, lr: torch.Tensor, target_size) -> dict[str, object]:
        baseline = torch.nn.functional.interpolate(
            lr, size=target_size, mode="bicubic", align_corners=False
        )
        residual = self.residual.expand_as(baseline)
        image = baseline + residual
        return {
            "image": image,
            "baseline": baseline,
            "residual": residual,
            "pyramid": {1: image},
        }


def test_generator_forward_exposes_baseline_residual_and_image_identity(tmp_path):
    trainer = Trainer(
        ResidualGenerator(),
        TinyDiscriminator(),
        _loader(),
        config=_config(tmp_path),
        device="cpu",
    )
    batch = next(iter(_loader()))

    generated = trainer._generator_forward(batch["lr"], batch["hr"])

    assert set(generated) == {"image", "baseline", "residual", "pyramid"}
    assert torch.allclose(
        generated["image"], generated["baseline"] + generated["residual"]
    )


class ImageOnlyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, lr: torch.Tensor, target_size) -> dict[str, object]:
        image = torch.nn.functional.interpolate(
            lr, size=target_size, mode="bilinear", align_corners=False
        )
        return {"image": self.net(image)}


def test_generator_forward_backfills_baseline_residual_for_image_only_mapping(
    tmp_path,
):
    trainer = Trainer(
        ImageOnlyGenerator(),
        TinyDiscriminator(),
        _loader(),
        config=_config(tmp_path),
        device="cpu",
    )
    batch = next(iter(_loader()))

    generated = trainer._generator_forward(batch["lr"], batch["hr"])

    assert torch.allclose(
        generated["image"], generated["baseline"] + generated["residual"]
    )
    assert generated["pyramid"] == {}


def test_residual_loss_matches_image_reconstruction_without_clamp(tmp_path):
    config = _config(tmp_path)
    config["loss"]["prediction_target"] = "residual"
    config["loss"]["lambda_adv"] = 0.0
    config["loss"]["lambda_multiscale"] = 0.0
    config["loss"]["lambda_perceptual"] = 0.0
    config["loss"]["lambda_consistency"] = 0.0
    config["loss"]["lambda_diffusion"] = 0.0
    trainer = Trainer(
        ResidualGenerator(), TinyDiscriminator(), _loader(), config=config, device="cpu"
    )

    logs = trainer.train_step(next(iter(_loader())))

    assert torch.isclose(
        torch.tensor(logs["loss_residual"]), torch.tensor(logs["loss_pixel_image"])
    )
    assert torch.isclose(
        torch.tensor(logs["loss_pixel"]), torch.tensor(logs["loss_residual"])
    )
