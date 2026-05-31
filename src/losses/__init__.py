"""Loss functions for super-resolution training."""

from src.losses.consistency import lr_consistency_loss
from src.losses.r3gan import (
    discriminator_loss,
    generator_loss,
    r1_regularization,
    r2_regularization,
)
from src.losses.reconstruction import (
    MultiScaleReconstructionLoss,
    ReconstructionLoss,
    multi_scale_reconstruction_loss,
    reconstruction_loss,
    reconstruction_loss_from_config,
)
from src.losses.total import (
    LOSS_KEYS,
    compute_generator_loss,
    generator_losses,
    generator_total_loss,
)

__all__ = [
    "LOSS_KEYS",
    "MultiScaleReconstructionLoss",
    "ReconstructionLoss",
    "compute_generator_loss",
    "discriminator_loss",
    "generator_loss",
    "generator_losses",
    "generator_total_loss",
    "lr_consistency_loss",
    "multi_scale_reconstruction_loss",
    "r1_regularization",
    "r2_regularization",
    "reconstruction_loss",
    "reconstruction_loss_from_config",
]
