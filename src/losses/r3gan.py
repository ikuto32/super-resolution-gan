"""R3GAN-style adversarial and gradient regularization losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mean_score_loss(loss: torch.Tensor) -> torch.Tensor:
    """Reduce per-logit adversarial losses to a scalar tensor."""
    return loss.mean()


def discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    real_images: torch.Tensor | None = None,
    fake_images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the logistic discriminator adversarial loss.

    ``real_images`` and ``fake_images`` are accepted to keep the public API aligned
    with discriminator objectives that include image-dependent terms; gradient
    penalties are exposed separately through :func:`r1_regularization` and
    :func:`r2_regularization`.
    """
    del real_images, fake_images
    return _mean_score_loss(F.softplus(fake_scores) + F.softplus(-real_scores))


def generator_loss(
    real_scores: torch.Tensor | None,
    fake_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the non-saturating logistic generator adversarial loss.

    The preferred call signature is ``generator_loss(real_scores, fake_scores)``.
    Passing only ``fake_scores`` is also supported for convenience because the
    non-saturating generator term does not depend on real logits.
    """
    if fake_scores is None:
        fake_scores = real_scores
    if fake_scores is None:
        raise ValueError("fake_scores must be provided")
    return _mean_score_loss(F.softplus(-fake_scores))


def _gradient_regularization(
    scores: torch.Tensor,
    images: torch.Tensor,
) -> torch.Tensor:
    if not images.requires_grad:
        raise ValueError("images must require gradients for regularization")

    gradients = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients.square().flatten(start_dim=1).sum(dim=1).mean()


def r1_regularization(
    real_scores: torch.Tensor, real_images: torch.Tensor
) -> torch.Tensor:
    """Return the R1 gradient penalty on real images."""
    return _gradient_regularization(real_scores, real_images)


def r2_regularization(
    fake_scores: torch.Tensor, fake_images: torch.Tensor
) -> torch.Tensor:
    """Return the R2 gradient penalty on generated images."""
    return _gradient_regularization(fake_scores, fake_images)
