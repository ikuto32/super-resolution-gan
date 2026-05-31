"""Diffusion-style degradation states and auxiliary denoising losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


_DEFAULT_SCHEDULE = "linear"


def _as_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value)


def _diffusion_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    diffusion = config.get("diffusion", config)
    return diffusion if isinstance(diffusion, Mapping) else {}


def sample_timesteps(
    batch_size: int,
    *,
    num_timesteps: int = 1000,
    device: torch.device | str | None = None,
    mode: str = "uniform",
) -> torch.Tensor:
    """Sample integer diffusion timesteps for a batch."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    mode = mode.lower()
    if mode != "uniform":
        raise ValueError(f"unsupported timestep sampling mode: {mode!r}")
    return torch.randint(0, int(num_timesteps), (batch_size,), device=device)


def timestep_weights(
    timesteps: torch.Tensor,
    *,
    num_timesteps: int = 1000,
    schedule: str = _DEFAULT_SCHEDULE,
    min_strength: float = 0.0,
    max_strength: float = 1.0,
) -> torch.Tensor:
    """Map integer timesteps to per-sample degradation/noise strengths."""
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    t = timesteps.to(dtype=torch.float32)
    if num_timesteps == 1:
        progress = torch.ones_like(t)
    else:
        progress = t.clamp(0, num_timesteps - 1) / float(num_timesteps - 1)
    schedule = schedule.lower()
    if schedule == "linear":
        weight = progress
    elif schedule == "cosine":
        weight = 1.0 - torch.cos(progress * torch.pi * 0.5)
    elif schedule == "quadratic":
        weight = progress.square()
    else:
        raise ValueError(f"unsupported diffusion schedule: {schedule!r}")
    return min_strength + (max_strength - min_strength) * weight


def _normalize_size(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must have two elements, got {size!r}")
    height, width = int(size[0]), int(size[1])
    if height < 1 or width < 1:
        raise ValueError(f"size values must be positive, got {size!r}")
    return height, width


def degraded_noisy_state(
    x_b: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    *,
    downscale: float | int | tuple[int, int] | list[int] = 4,
    num_timesteps: int = 1000,
    schedule: str = _DEFAULT_SCHEDULE,
    noise_min: float = 0.0,
    noise_max: float = 0.1,
    degradation_min: float = 1.0,
    degradation_max: float = 1.0,
    mode: str = "bicubic",
    noise: torch.Tensor | None = None,
    clamp: tuple[float, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Create a downscaled degraded/noisy state ``x_t`` from HR images ``x_b``.

    ``downscale`` may be an arbitrary scale factor (including non-powers of two) or
    an explicit spatial size. The returned ``x_t`` stays at the degraded spatial
    size so callers can condition a denoiser/SR generator on the lower-resolution
    state and ask it to restore ``x_b.shape[-2:]``.
    """
    if x_b.ndim != 4:
        raise ValueError(f"expected x_b to be BCHW, got shape {tuple(x_b.shape)}")
    batch, _channels, height, width = x_b.shape
    if timesteps is None:
        timesteps = sample_timesteps(batch, num_timesteps=num_timesteps, device=x_b.device)
    else:
        timesteps = timesteps.to(device=x_b.device)
    if timesteps.shape != (batch,):
        raise ValueError(
            f"timesteps must have shape ({batch},), got {tuple(timesteps.shape)}"
        )

    if isinstance(downscale, (tuple, list)):
        degraded_size = _normalize_size(downscale)
    else:
        scale = float(downscale)
        if scale <= 0:
            raise ValueError(f"downscale must be positive, got {downscale!r}")
        degraded_size = (max(1, round(height / scale)), max(1, round(width / scale)))

    clean_low = F.interpolate(x_b, size=degraded_size, mode=mode, align_corners=False)
    strength = timestep_weights(
        timesteps,
        num_timesteps=num_timesteps,
        schedule=schedule,
        min_strength=degradation_min,
        max_strength=degradation_max,
    ).to(device=x_b.device, dtype=x_b.dtype)
    sigma = timestep_weights(
        timesteps,
        num_timesteps=num_timesteps,
        schedule=schedule,
        min_strength=noise_min,
        max_strength=noise_max,
    ).to(device=x_b.device, dtype=x_b.dtype)

    if noise is None:
        noise = torch.randn_like(clean_low)
    else:
        noise = noise.to(device=x_b.device, dtype=x_b.dtype)
        if noise.shape != clean_low.shape:
            raise ValueError(
                f"noise must have shape {tuple(clean_low.shape)}, got {tuple(noise.shape)}"
            )

    view_shape = (batch,) + (1,) * (clean_low.ndim - 1)
    low_mean = clean_low.mean(dim=(-2, -1), keepdim=True)
    degraded = clean_low.lerp(low_mean, strength.view(view_shape).clamp(0.0, 1.0))
    x_t = degraded + noise * sigma.view(view_shape)
    if clamp is not None:
        x_t = x_t.clamp(float(clamp[0]), float(clamp[1]))

    return {
        "x_t": x_t,
        "target": x_b,
        "clean_low": clean_low,
        "noise": noise,
        "timesteps": timesteps,
        "sigma": sigma,
    }


def degraded_noisy_state_from_config(
    x_b: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    timesteps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create ``x_t`` using a config mapping with a ``diffusion`` section."""
    diffusion = _diffusion_section(config)
    sampling = diffusion.get("timestep_sampling", {})
    if not isinstance(sampling, Mapping):
        sampling = {}
    noise_cfg = diffusion.get("noise", {})
    if not isinstance(noise_cfg, Mapping):
        noise_cfg = {}
    degradation_cfg = diffusion.get("degradation", {})
    if not isinstance(degradation_cfg, Mapping):
        degradation_cfg = {}

    num_timesteps = int(diffusion.get("num_timesteps", 1000))
    if timesteps is None:
        timesteps = sample_timesteps(
            int(x_b.shape[0]),
            num_timesteps=num_timesteps,
            device=x_b.device,
            mode=str(sampling.get("mode", "uniform")),
        )

    return degraded_noisy_state(
        x_b,
        timesteps,
        downscale=diffusion.get("downscale", degradation_cfg.get("downscale", 4)),
        num_timesteps=num_timesteps,
        schedule=str(diffusion.get("schedule", _DEFAULT_SCHEDULE)),
        noise_min=_as_float(noise_cfg, "std_min", _as_float(diffusion, "noise_min", 0.0)),
        noise_max=_as_float(noise_cfg, "std_max", _as_float(diffusion, "noise_max", 0.1)),
        degradation_min=_as_float(degradation_cfg, "strength_min", 0.0),
        degradation_max=_as_float(degradation_cfg, "strength_max", 0.0),
        mode=str(degradation_cfg.get("mode", diffusion.get("mode", "bicubic"))),
        clamp=tuple(diffusion["clamp"]) if "clamp" in diffusion else None,
    )


def denoising_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compute scalar denoising loss against the clean HR target."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must be BCHW tensors")
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(
            prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    loss_type = loss_type.lower()
    if loss_type in {"l1", "mae"}:
        return F.l1_loss(prediction, target)
    if loss_type in {"mse", "l2"}:
        return F.mse_loss(prediction, target)
    if loss_type == "charbonnier":
        return torch.sqrt((prediction - target).square() + 1e-6).mean()
    raise ValueError(f"unsupported denoising loss_type: {loss_type!r}")


__all__ = [
    "degraded_noisy_state",
    "degraded_noisy_state_from_config",
    "denoising_loss",
    "sample_timesteps",
    "timestep_weights",
]
