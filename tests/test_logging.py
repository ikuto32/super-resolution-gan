from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.training import logging as logging_module
from src.training.logging import TrainingLogger


class _FakeWandb:
    def __init__(self) -> None:
        self.images = []
        self.logged = []

    def Image(self, data):  # noqa: N802 - mirrors wandb.Image
        self.images.append(data)
        return data

    def log(self, data, step=None):
        self.logged.append((data, step))

    def finish(self):
        pass


def test_wandb_images_are_logged_as_uint8_pil_images(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(logging_module, "wandb", fake_wandb)

    logger = TrainingLogger(tmp_path, enable_tensorboard=False)
    logger._wandb_run = object()

    images = torch.tensor(
        [
            [
                [[-1.0, 0.0], [0.5, 2.0]],
                [[-0.5, 0.25], [0.75, 1.0]],
                [[0.0, 0.5], [1.0, 1.5]],
            ]
        ]
    )

    logger.log_images("samples", images, step=3)
    logger.close()

    assert len(fake_wandb.images) == 1
    wandb_image = fake_wandb.images[0]
    assert isinstance(wandb_image, Image.Image)
    assert wandb_image.mode == "RGB"
    assert wandb_image.size == (2, 2)
    pixels = np.asarray(wandb_image)
    assert pixels.min() >= 0
    assert pixels.max() <= 255
    assert fake_wandb.logged == [({"samples": [wandb_image]}, 3)]
