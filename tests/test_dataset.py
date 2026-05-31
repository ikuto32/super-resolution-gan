from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import (
    DegradationPipeline,
    ImagePairDataset,
    build_image_pyramid,
    center_crop,
    normalize_minus_one_to_one,
    pil_to_tensor,
)


def _write_image(path, size=(32, 32)):
    height, width = size
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    array = np.stack(
        [
            (x * 7 + y * 3) % 256,
            (x * 5 + 17) % 256,
            (y * 11 + 29) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_image_pair_dataset_shapes_range_and_pyramid_keys(tmp_path):
    _write_image(tmp_path / "sample.png", size=(32, 32))
    dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation={
            "scale": 4,
            "blur": {"sigma": 0.5},
            "noise": {"std": 0.0},
            "jpeg": {"quality": 95},
        },
        pyramid_scales=(1, 2, 4, 32),
        validation=True,
        seed=123,
    )

    item = dataset[0]

    assert item["hr"].shape == (3, 32, 32)
    assert item["lr"].shape == (3, 8, 8)
    assert set(item["hr_pyramid"]) == {"x1", "x2", "x4", "x32"}
    assert item["hr_pyramid"]["x32"].shape == (3, 1, 1)
    assert -1.0 <= item["hr"].min() <= item["hr"].max() <= 1.0
    assert -1.0 <= item["lr"].min() <= item["lr"].max() <= 1.0


def test_image_pair_dataset_default_collate_without_lr_root(tmp_path):
    _write_image(tmp_path / "sample_0.png", size=(32, 32))
    _write_image(tmp_path / "sample_1.png", size=(32, 32))
    dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation={"scale": 4, "noise": {"std": 0.0}},
        validation=True,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["lr"].shape == (2, 3, 8, 8)
    assert batch["hr"].shape == (2, 3, 32, 32)
    assert batch["meta"]["lr_path"] == ["", ""]


def test_degradation_noise_supports_min_max_keys():
    tensor = torch.full((3, 16, 16), 0.5)
    pipeline = DegradationPipeline(
        {
            "scale": 1,
            "noise": {"enabled": True, "std_min": 0.01, "std_max": 0.02},
            "clamp": False,
        }
    )
    generator = torch.Generator().manual_seed(123)

    degraded = pipeline(tensor, rng=random.Random(123), torch_generator=generator)

    assert degraded.shape == tensor.shape
    assert not torch.equal(degraded, tensor)
    assert torch.count_nonzero(degraded - tensor) > 0


def test_validation_mode_is_deterministic(tmp_path):
    _write_image(tmp_path / "sample.png", size=(40, 40))
    dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation={
            "scale": 2,
            "blur": {"sigma": [0.1, 1.2]},
            "noise": {"std": [0.001, 0.01]},
            "jpeg": {"quality": [45, 90]},
        },
        validation=True,
        seed=999,
    )

    first = dataset[0]
    second = dataset[0]

    assert torch.equal(first["hr"], second["hr"])
    assert torch.equal(first["lr"], second["lr"])
    for key in first["hr_pyramid"]:
        assert torch.equal(first["hr_pyramid"][key], second["hr_pyramid"][key])


def test_training_mode_rng_is_seeded_by_index(tmp_path):
    _write_image(tmp_path / "sample.png", size=(40, 40))
    degradation = {
        "scale": 2,
        "blur": {"sigma": [0.1, 1.2]},
        "noise": {"std": [0.001, 0.01]},
        "jpeg": {"quality": [45, 90]},
    }

    first_dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation=degradation,
        validation=False,
        seed=1234,
    )
    second_dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation=degradation,
        validation=False,
        seed=1234,
    )
    different_seed_dataset = ImagePairDataset(
        tmp_path,
        crop_size=32,
        degradation=degradation,
        validation=False,
        seed=4321,
    )

    first = first_dataset[0]
    second = second_dataset[0]
    different_seed = different_seed_dataset[0]

    assert torch.equal(first["hr"], second["hr"])
    assert torch.equal(first["lr"], second["lr"])
    assert not torch.equal(first["lr"], different_seed["lr"])


def test_one_by_one_pyramid_uses_spatial_mean():
    hr = torch.arange(12, dtype=torch.float32).view(3, 2, 2)

    pyramid = build_image_pyramid(hr, scales=(2,))

    assert torch.equal(pyramid["x2"], hr.mean(dim=(-2, -1), keepdim=True))


def test_transforms_crop_and_normalize():
    image = Image.new("RGB", (10, 8), color=(128, 64, 32))

    cropped = center_crop(image, (4, 6))
    tensor = normalize_minus_one_to_one(pil_to_tensor(cropped))

    assert cropped.size == (6, 4)
    assert tensor.shape == (3, 4, 6)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_degradation_pipeline_target_size_overrides_scale():
    tensor = torch.zeros((3, 32, 48), dtype=torch.float32)
    pipeline = DegradationPipeline({"scale": 4, "target_size": [10, 12]})

    degraded = pipeline(tensor)

    assert degraded.shape == (3, 10, 12)
