"""End-to-end GAN training loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.losses.diffusion import degraded_noisy_state_from_config, denoising_loss
from src.losses.perceptual import build_perceptual_loss
from src.losses.reconstruction import reconstruction_loss
from src.losses.r3gan import discriminator_loss, r1_regularization, r2_regularization
from src.losses.total import generator_losses
from src.models.discriminator import ResolutionAgnosticDiscriminator
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.ema import EMAGenerator
from src.training.logging import TrainingLogger
from src.training.optimizers import build_optimizers
from src.training.validation import run_validation
from datasets.transforms import tensor_to_pil


def _module_score(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "score" in output:
            return output["score"]
        if "patch_logits" in output:
            return output["patch_logits"].mean(
                dim=tuple(range(1, output["patch_logits"].ndim))
            )
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(
        "discriminator output must be a tensor or mapping with score/patch_logits"
    )


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
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
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

        training_cfg = (
            self.config.get("training", {})
            if isinstance(self.config.get("training", {}), Mapping)
            else {}
        )
        ema_cfg = (
            training_cfg.get("ema", {})
            if isinstance(training_cfg.get("ema", {}), Mapping)
            else {}
        )
        self.ema = generator_ema
        if self.ema is None and bool(ema_cfg.get("enabled", False)):
            self.ema = EMAGenerator(
                self.generator,
                decay=float(ema_cfg.get("decay", 0.999)),
                device=self.device,
            )

        project_cfg = (
            self.config.get("project", {})
            if isinstance(self.config.get("project", {}), Mapping)
            else {}
        )
        logging_cfg = (
            self.config.get("logging", {})
            if isinstance(self.config.get("logging", {}), Mapping)
            else {}
        )
        output_dir = Path(project_cfg.get("output_dir", "runs/default"))
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.logger = logger or TrainingLogger(
            output_dir / "logs",
            enable_tensorboard=bool(logging_cfg.get("tensorboard", True)),
            wandb_config=logging_cfg.get("wandb", {})
            if isinstance(logging_cfg.get("wandb", {}), Mapping)
            else {},
            run_config=self.config,
        )
        self.mixed_precision = (
            bool(training_cfg.get("mixed_precision", False))
            and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None
        self.grad_clip_norm = training_cfg.get("grad_clip_norm")
        self.n_critic = max(1, int(training_cfg.get("n_critic", 1)))
        self.log_every = int(training_cfg.get("log_every", 100))
        self.validate_every = int(training_cfg.get("validate_every", 1000))
        self.save_every = int(training_cfg.get("save_every", 5000))
        self.epochs = int(training_cfg.get("epochs", 1))
        self.sample_every_kimg = float(
            training_cfg.get("sample_every_kimg", 0.0) or 0.0
        )
        self.sample_max_images = int(training_cfg.get("sample_max_images", 4))
        self.sample_dir = output_dir / str(training_cfg.get("sample_dir", "samples"))
        self.loss_config = (
            self.config.get("loss", {})
            if isinstance(self.config.get("loss", {}), Mapping)
            else {}
        )
        self.prediction_target = str(
            self.loss_config.get("prediction_target", "image")
        ).lower()
        if self.prediction_target not in {"image", "residual"}:
            raise ValueError(
                "loss.prediction_target must be either 'image' or 'residual'"
            )
        self.perceptual_loss = None
        if float(self.loss_config.get("lambda_perceptual", 0.0)) > 0.0:
            self.perceptual_loss = build_perceptual_loss(self.loss_config).to(
                self.device
            )
            self.perceptual_loss.eval()
        self.step = 0
        self.epoch = 0
        self.seen_images = 0
        self._next_sample_kimg = (
            self.sample_every_kimg if self.sample_every_kimg > 0 else 0.0
        )
        self._discriminator_accepts_condition: bool | None = None

    def _autocast(self):
        if self.mixed_precision:
            return torch.amp.autocast(device_type=self.device.type)
        return nullcontext()

    def _discriminator_uses_condition(self) -> bool:
        if isinstance(self.discriminator, ResolutionAgnosticDiscriminator):
            return bool(self.discriminator.conditional)
        if self._discriminator_accepts_condition is None:
            signature = inspect.signature(self.discriminator.forward)
            positional_parameters = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            self._discriminator_accepts_condition = (
                any(
                    parameter.kind is inspect.Parameter.VAR_POSITIONAL
                    for parameter in signature.parameters.values()
                )
                or len(positional_parameters) >= 2
            )
        return self._discriminator_accepts_condition

    def _discriminate(self, images: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        if self._discriminator_uses_condition():
            output = self.discriminator(images, lr)
        else:
            output = self.discriminator(images)
        return _module_score(output)

    def _generator_forward(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> dict[str, torch.Tensor | Mapping[Any, torch.Tensor]]:
        output = self.generator(lr, target_size=hr.shape[-2:])
        baseline = F.interpolate(
            lr, size=hr.shape[-2:], mode="bicubic", align_corners=False
        )
        if isinstance(output, Mapping):
            image = output.get("image")
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "generator output mapping must contain tensor key 'image'"
                )
            output_baseline = output.get("baseline", baseline)
            if not isinstance(output_baseline, torch.Tensor):
                raise TypeError("generator output key 'baseline' must be a tensor")
            residual = output.get("residual")
            if residual is None:
                residual = image - output_baseline
            if not isinstance(residual, torch.Tensor):
                raise TypeError("generator output key 'residual' must be a tensor")
            pyramid = output.get("pyramid", {})
            if not isinstance(pyramid, Mapping):
                raise TypeError("generator output key 'pyramid' must be a mapping")
            return {
                "image": image,
                "baseline": output_baseline,
                "residual": residual,
                "pyramid": pyramid,
            }
        if not isinstance(output, torch.Tensor):
            raise TypeError("generator output must be a tensor or mapping")
        return {
            "image": output,
            "baseline": baseline,
            "residual": output - baseline,
            "pyramid": {},
        }

    def _generator_supports_kwargs(self, *names: str) -> bool:
        signature = inspect.signature(self.generator.forward)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return True
        return all(name in signature.parameters for name in names)

    def _diffusion_generator_forward(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {}
        if self._generator_supports_kwargs("noisy_condition"):
            kwargs["noisy_condition"] = x_t
        else:
            lr = x_t
        if self._generator_supports_kwargs("diffusion_timestep"):
            kwargs["diffusion_timestep"] = timesteps
        if self._generator_supports_kwargs("return_diffusion"):
            kwargs["return_diffusion"] = True
        output = self.generator(lr, target_size=hr.shape[-2:], **kwargs)
        if isinstance(output, Mapping):
            diffusion = output.get("diffusion")
            if isinstance(diffusion, torch.Tensor):
                return diffusion
            return output["image"]
        return output

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Run ``n_critic`` discriminator updates and one generator update."""
        self.generator.train()
        self.discriminator.train()
        batch = _batch_to_device(batch, self.device)
        lr = batch["lr"]
        hr = batch["hr"]
        hr_pyramid = batch.get("hr_pyramid")

        loss_d_values: list[torch.Tensor] = []
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                generated_detached = self._generator_forward(lr, hr)
                fake_detached = generated_detached["image"]
            real_for_d = hr.detach().requires_grad_(
                float(self.loss_config.get("lambda_r1", 0.0)) > 0.0
            )
            fake_for_d = fake_detached.detach().requires_grad_(
                float(self.loss_config.get("lambda_r2", 0.0)) > 0.0
            )
            with self._autocast():
                real_scores = self._discriminate(real_for_d, lr)
                fake_scores = self._discriminate(fake_for_d, lr)
                loss_d = discriminator_loss(
                    real_scores, fake_scores, real_for_d, fake_for_d
                )
                if float(self.loss_config.get("lambda_r1", 0.0)) > 0.0:
                    loss_d = loss_d + r1_regularization(
                        real_scores, real_for_d
                    ) * float(self.loss_config.get("lambda_r1", 0.0))
                if float(self.loss_config.get("lambda_r2", 0.0)) > 0.0:
                    loss_d = loss_d + r2_regularization(
                        fake_scores, fake_for_d
                    ) * float(self.loss_config.get("lambda_r2", 0.0))
            if self.scaler is not None:
                self.scaler.scale(loss_d).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer_d)
                    clip_grad_norm_(
                        self.discriminator.parameters(), float(self.grad_clip_norm)
                    )
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
            else:
                loss_d.backward()
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(
                        self.discriminator.parameters(), float(self.grad_clip_norm)
                    )
                self.optimizer_d.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()
            loss_d_values.append(loss_d.detach())

        self.optimizer_g.zero_grad(set_to_none=True)
        with self._autocast():
            generated = self._generator_forward(lr, hr)
            fake = generated["image"]
            baseline = generated["baseline"]
            pred_residual = generated["residual"]
            generated_pyramid = generated["pyramid"]
            if not isinstance(fake, torch.Tensor):
                raise TypeError("normalized generator image must be a tensor")
            if not isinstance(baseline, torch.Tensor):
                raise TypeError("normalized generator baseline must be a tensor")
            if not isinstance(pred_residual, torch.Tensor):
                raise TypeError("normalized generator residual must be a tensor")
            if not isinstance(generated_pyramid, Mapping):
                raise TypeError("normalized generator pyramid must be a mapping")
            target_residual = hr - baseline
            pixel_loss_type = str(self.loss_config.get("pixel_loss_type", "l1"))
            loss_pixel_image = reconstruction_loss(fake, hr, loss_type=pixel_loss_type)
            loss_residual = reconstruction_loss(
                pred_residual, target_residual, loss_type=pixel_loss_type
            )
            pixel_prediction = fake
            pixel_target = hr
            if self.prediction_target == "residual":
                pixel_prediction = pred_residual
                pixel_target = target_residual
            fake_scores_g = self._discriminate(fake, lr)
            perceptual_loss = (
                self.perceptual_loss(fake, hr)
                if self.perceptual_loss is not None
                else None
            )
            diffusion_loss = None
            if float(self.loss_config.get("lambda_diffusion", 0.0)) > 0.0:
                diffusion_state = degraded_noisy_state_from_config(hr, self.loss_config)
                diffusion_prediction = self._diffusion_generator_forward(
                    lr,
                    hr,
                    diffusion_state["x_t"],
                    diffusion_state["timesteps"],
                )
                diffusion_cfg = self.loss_config.get("diffusion", {})
                if not isinstance(diffusion_cfg, Mapping):
                    diffusion_cfg = {}
                diffusion_loss = denoising_loss(
                    diffusion_prediction,
                    hr,
                    loss_type=str(diffusion_cfg.get("loss_type", "l1")),
                )
            g_losses = generator_losses(
                fake_scores=fake_scores_g,
                sr=fake,
                hr=hr,
                reconstruction_prediction=pixel_prediction,
                reconstruction_target=pixel_target,
                lr=lr,
                generated_pyramid=generated_pyramid,
                hr_pyramid=hr_pyramid,
                perceptual_loss=perceptual_loss,
                diffusion_loss=diffusion_loss,
                weights=self.loss_config,
                pixel_loss_type=pixel_loss_type,
            )
            lambda_pixel = float(
                self.loss_config.get(
                    "lambda_pixel",
                    self.loss_config.get(
                        "lambda_pix", self.loss_config.get("pixel", 1.0)
                    ),
                )
            )
            g_losses["loss_pixel_image"] = loss_pixel_image * lambda_pixel
            g_losses["loss_residual"] = loss_residual * lambda_pixel
            loss_g = g_losses["loss_total"]
        if self.scaler is not None:
            self.scaler.scale(loss_g).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer_g)
                clip_grad_norm_(self.generator.parameters(), float(self.grad_clip_norm))
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            loss_g.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.generator.parameters(), float(self.grad_clip_norm))
            self.optimizer_g.step()
        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.ema is not None:
            self.ema.update(self.generator)

        self.step += 1
        self.seen_images += int(hr.shape[0])
        logs = {name: float(value.detach().cpu()) for name, value in g_losses.items()}
        logs["loss_d"] = float(torch.stack(loss_d_values).mean().cpu())
        logs["lr_g"] = float(self.optimizer_g.param_groups[0]["lr"])
        logs["lr_d"] = float(self.optimizer_d.param_groups[0]["lr"])
        if self.log_every > 0 and self.step % self.log_every == 0:
            self.logger.log_scalars(logs, self.step, prefix="train")
        if self._should_write_sample():
            self._write_sample(lr, fake, hr)
        if (
            self.validate_every > 0
            and self.val_loader is not None
            and self.step % self.validate_every == 0
        ):
            metrics = self.validate()
            self.logger.log_scalars(metrics, self.step, prefix="val")
        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(self.checkpoint_dir / f"step_{self.step:08d}.pt")
            self.save(self.checkpoint_dir / "latest.pt")
        return logs

    def _should_write_sample(self) -> bool:
        if self.sample_every_kimg <= 0:
            return False
        current_kimg = self.seen_images / 1000.0
        if current_kimg + 1e-12 < self._next_sample_kimg:
            return False
        while self._next_sample_kimg <= current_kimg + 1e-12:
            self._next_sample_kimg += self.sample_every_kimg
        return True

    def _make_sample_grid(
        self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor
    ) -> torch.Tensor:
        max_images = max(1, min(self.sample_max_images, int(hr.shape[0])))
        lr_up = F.interpolate(
            lr[:max_images], size=hr.shape[-2:], mode="bilinear", align_corners=False
        )
        rows = [
            torch.cat(
                [lr_up[index], sr[:max_images][index], hr[:max_images][index]], dim=-1
            )
            for index in range(max_images)
        ]
        return torch.cat(rows, dim=-2)

    def _write_sample(
        self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor
    ) -> None:
        grid = self._make_sample_grid(lr.detach(), sr.detach(), hr.detach())
        sample_path = (
            self.sample_dir
            / f"step_{self.step:08d}_kimg_{self.seen_images / 1000.0:.3f}.png"
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(grid).save(sample_path)
        self.logger.log_images("samples/lr_sr_hr", grid, self.step)

    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        module = self.ema.module if self.ema is not None else self.generator
        return run_validation(
            module,
            self.val_loader,
            device=self.device,
            step=self.step,
            output_dir=self.output_dir,
        )

    def fit(self) -> None:
        if self.train_loader is None:
            raise ValueError("train_loader is required for fit()")
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            progress = tqdm(self.train_loader, desc=f"epoch={epoch}")
            for batch in progress:
                logs = self.train_step(batch)
                progress.set_postfix(
                    loss_g=logs["loss_total"], loss_d=logs["loss_d"], lr=logs["lr_g"]
                )
            self.epoch = epoch + 1
            self.save(self.checkpoint_dir / "latest.pt")
        self.logger.close()

    def save(self, path: str | Path) -> dict[str, Any]:
        return save_checkpoint(
            path,
            step=self.step,
            next_epoch=self.epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            generator_ema=self.ema,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
            grad_scaler=self.scaler,
            config=dict(self.config),
            seen_images=self.seen_images,
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
            grad_scaler=self.scaler,
            map_location=self.device,
            restore_rng=restore_rng,
        )
        self.step = int(checkpoint.get("step", 0))
        self.epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", 0)))
        loaded_seen_images = checkpoint.get("seen_images")
        if loaded_seen_images is not None:
            self.seen_images = int(loaded_seen_images)
        if self.sample_every_kimg > 0:
            self._next_sample_kimg = (
                int(self.seen_images / (self.sample_every_kimg * 1000.0)) + 1
            ) * self.sample_every_kimg
        return checkpoint


__all__ = ["Trainer"]
