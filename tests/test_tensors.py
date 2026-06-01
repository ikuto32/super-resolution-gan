from __future__ import annotations

import torch

from src.utils.tensors import batch_to_device, move_to_device


def test_move_to_device_moves_tensors_in_nested_mappings() -> None:
    tensor = torch.ones(1)
    nested_tensor = torch.zeros(1)
    untouched = object()

    moved = move_to_device(
        {"tensor": tensor, "nested": {"tensor": nested_tensor, "value": untouched}},
        torch.device("cpu"),
    )

    assert moved["tensor"].device.type == "cpu"
    assert moved["nested"]["tensor"].device.type == "cpu"
    assert moved["nested"]["value"] is untouched


def test_batch_to_device_preserves_non_tensor_values() -> None:
    untouched = object()

    moved = batch_to_device({"name": "sample", "metadata": {"raw": untouched}}, "cpu")

    assert moved["name"] == "sample"
    assert moved["metadata"]["raw"] is untouched
