"""Spatial size normalization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload


@overload
def normalize_hw(
    value: int | Sequence[int], *, name: str = "size", allow_none: bool = False
) -> tuple[int, int]: ...


@overload
def normalize_hw(
    value: int | Sequence[int] | None, *, name: str = "size", allow_none: bool = True
) -> tuple[int, int] | None: ...


def normalize_hw(
    value: int | Sequence[int] | None,
    *,
    name: str = "size",
    allow_none: bool = False,
) -> tuple[int, int] | None:
    """Normalize an integer or 2-item sequence to a positive ``(height, width)``.

    Integers are interpreted as square spatial sizes. Sequences are interpreted
    as ``(height, width)`` pairs. Non-positive dimensions are rejected.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be an int or a 2-item (height, width) sequence")

    if isinstance(value, int):
        height = width = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                f"{name} must be an int or a 2-item (height, width) sequence, "
                f"got {value!r}"
            )
        height, width = value
    else:
        raise ValueError(
            f"{name} must be an int or a 2-item (height, width) sequence, got {value!r}"
        )

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {value!r}")
    return height, width
