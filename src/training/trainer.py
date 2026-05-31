"""End-to-end GAN training loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.losses.r3gan import discriminator_loss, r1_regularization, r2_regularization
from src.losses.total import generator_losses
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.ema import EMAGenerator
from src.training.logging import TrainingLogger
from src.training.optimizers import build_optimizers
from src.training.validation import run_validation


def _module_score(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "score" in output:
            return output["score"]
        if "patch_logits" in output:
            return output["patch_logits"].mean(dim=tuple(range(1, output["patch_logits"].ndim)))
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("discriminator output must be a tensor or mapping with score/patch_logits")


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            moved[key] = value
    return moved


class Trainer:
    """Coordinate D/G updates, AMP, clipping, EMA, logging, validation, and saves."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: Iterable[Mapping[str, Any]] | None = None,
        val_loader: Iterable[Mapping[str, Any]] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        optimizer_g: torch.optim.Optimizer | None = None,
        optimizer_d: torch.optim.Optimizer | None = None,
        scheduler_g: Any | None = None,
        scheduler_d: Any | None = None,
        device: torch.device | str | None = None,
        logger: TrainingLogger | None = None,
        generator_ema: EMAGenerator | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_g, self.optimizer_d = (
            (optimizer_g, optimizer_d)
            if optimizer_g is not None and optimizer_d is not None
            else build_optimizers(self.generator, self.discriminator, self.config)
        )
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

        training_cfg = self.config.get("training", {}) if isinstance(self.config.get("training", {}), Mapping) else {}
        ema_cfg = training_cfg.get("ema", {}) if isinstance(training_cfg.get("ema", {}), Mapping) else {}
        self.ema = generator_ema
        if self.ema is None and bool(ema_cfg.get("enabled", False)):
            self.ema = EMAGenerator(self.generator, decay=float(ema_cfg.get("decay", 0.999)), device=self.device)

        output_dir = Path(self.config.get("project", {}).get("output_dir", "runs/default") if isinstance(self.config.get("project", {}), Mapping) else "runs/default")
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.logger = logger or TrainingLogger(output_dir / "logs")
        self.mixed_precision = bool(training_cfg.get("mixed_precision", False)) and self.device.type == "cuda"
        self.grad_clip_norm = training_cfg.get("grad_clip_norm")
        self.log_every = int(training_cfg.get("log_every", 100))
        self.validate_every = int(training_cfg.get("validate_every", 1000))
        self.save_every = int(training_cfg.get("save_every", 5000))
        self.epochs = int(training_cfg.get("epochs", 1))
        self.loss_config = self.config.get("loss", {}) if isinstance(self.config.get("loss", {}), Mapping) else {}
        self.step = 0
        self.epoch = 0

    def _autocast(self):
        if self.mixed_precision:
            return torch.amp.autocast(device_type=self.device.type)
        return nullcontext()

    def _discriminate(self, images: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        try:
            output = self.discriminator(images, lr)
        except TypeError:
            output = self.discriminator(images)
        return _module_score(output)

    def _generator_forward(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, Mapping[Any, torch.Tensor]]:
        output = self.generator(lr, target_size=hr.shape[-2:])
        if isinstance(output, Mapping):
            return output["image"], output.get("pyramid", {})
        return output, {}

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Run one discriminator update and one generator update."""
        self.generator.train()
        self.discriminator.train()
        batch = _batch_to_device(batch, self.device)
        lr = batch["lr"]
        hr = batch["hr"]
        hr_pyramid = batch.get("hr_pyramid")

        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_detached, _ = self._generator_forward(lr, hr)
        real_for_d = hr.detach().requires_grad_(float(self.loss_config.get("lambda_r1", 0.0)) > 0.0)
        fake_for_d = fake_detached.detach().requires_grad_(float(self.loss_config.get("lambda_r2", 0.0)) > 0.0)
        with self._autocast():
            real_scores = self._discriminate(real_for_d, lr)
            fake_scores = self._discriminate(fake_for_d, lr)
            loss_d = discriminator_loss(real_scores, fake_scores, real_for_d, fake_for_d)
            if float(self.loss_config.get("lambda_r1", 0.0)) > 0.0:
                loss_d = loss_d + r1_regularization(real_scores, real_for_d) * float(self.loss_config.get("lambda_r1", 0.0))
            if float(self.loss_config.get("lambda_r2", 0.0)) > 0.0:
                loss_d = loss_d + r2_regularization(fake_scores, fake_for_d) * float(self.loss_config.get("lambda_r2", 0.0))
        loss_d.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.discriminator.parameters(), float(self.grad_clip_norm))
        self.optimizer_d.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

        self.optimizer_g.zero_grad(set_to_none=True)
        with self._autocast():
            fake, generated_pyramid = self._generator_forward(lr, hr)
            fake_scores_g = self._discriminate(fake, lr)
            g_losses = generator_losses(
                fake_scores=fake_scores_g,
                sr=fake,
                hr=hr,
                lr=lr,
                generated_pyramid=generated_pyramid,
                hr_pyramid=hr_pyramid,
                weights=self.loss_config,
            )
            loss_g = g_losses["loss_total"]
        loss_g.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.generator.parameters(), float(self.grad_clip_norm))
        self.optimizer_g.step()
        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.ema is not None:
            self.ema.update(self.generator)

        self.step += 1
        logs = {name: float(value.detach().cpu()) for name, value in g_losses.items()}
        logs["loss_d"] = float(loss_d.detach().cpu())
        logs["lr_g"] = float(self.optimizer_g.param_groups[0]["lr"])
        logs["lr_d"] = float(self.optimizer_d.param_groups[0]["lr"])
        if self.log_every > 0 and self.step % self.log_every == 0:
            self.logger.log_scalars(logs, self.step, prefix="train")
        if self.validate_every > 0 and self.val_loader is not None and self.step % self.validate_every == 0:
            metrics = self.validate()
            self.logger.log_scalars(metrics, self.step, prefix="val")
        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(self.checkpoint_dir / f"step_{self.step:08d}.pt")
            self.save(self.checkpoint_dir / "latest.pt")
        return logs

    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        module = self.ema.module if self.ema is not None else self.generator
        return run_validation(module, self.val_loader, device=self.device, step=self.step, output_dir=self.output_dir)

    def fit(self) -> None:
        if self.train_loader is None:
            raise ValueError("train_loader is required for fit()")
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            progress = tqdm(self.train_loader, desc=f"epoch={epoch}")
            for batch in progress:
                logs = self.train_step(batch)
                progress.set_postfix(loss_g=logs["loss_total"], loss_d=logs["loss_d"], lr=logs["lr_g"])
        self.save(self.checkpoint_dir / "latest.pt")
        self.logger.close()

    def save(self, path: str | Path) -> dict[str, Any]:
        return save_checkpoint(
            path,
            step=self.step,
            epoch=self.epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            generator_ema=self.ema,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
            config=dict(self.config),
        )

    def load(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        checkpoint = load_checkpoint(
            path,
            generator=self.generator,
            discriminator=self.discriminator,
            generator_ema=self.ema,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
            map_location=self.device,
            restore_rng=restore_rng,
        )
        self.step = int(checkpoint.get("step", 0))
        self.epoch = int(checkpoint.get("epoch", 0))
        return checkpoint


__all__ = ["Trainer"]
