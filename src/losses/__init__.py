"""Loss functions for super-resolution training."""

from src.losses.reconstruction import (
    MultiScaleReconstructionLoss,
    multi_scale_reconstruction_loss,
)

__all__ = ["MultiScaleReconstructionLoss", "multi_scale_reconstruction_loss"]
