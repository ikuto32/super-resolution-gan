"""Randomness utilities for reproducible experiments."""

from __future__ import annotations

import random
from typing import Any, TypedDict

import numpy as np
import torch


class RNGState(TypedDict):
    """Serializable container for Python, NumPy, and PyTorch RNG states."""

    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python ``random``, NumPy, and PyTorch.

    Returns the normalized integer seed so callers can log the exact value used.
    If ``deterministic`` is true, PyTorch deterministic algorithm flags are
    enabled where available.
    """

    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return normalized_seed


def get_rng_state(*, include_cuda: bool = True) -> RNGState:
    """Capture current RNG states for Python, NumPy, and PyTorch."""

    cuda_state = None
    if include_cuda and torch.cuda.is_available():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_state,
    }


def set_rng_state(state: RNGState) -> None:
    """Restore RNG states captured by :func:`get_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            msg = "Cannot restore CUDA RNG state because CUDA is not available."
            raise RuntimeError(msg)
        torch.cuda.set_rng_state_all(cuda_state)


def save_rng_state(*, include_cuda: bool = True) -> RNGState:
    """Alias for :func:`get_rng_state` for checkpointing call sites."""

    return get_rng_state(include_cuda=include_cuda)


def restore_rng_state(state: RNGState) -> None:
    """Alias for :func:`set_rng_state` for checkpointing call sites."""

    set_rng_state(state)
